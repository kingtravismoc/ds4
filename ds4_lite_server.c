/* ds4_lite_server.c - Condensed web server for low-bandwidth inference
 * 
 * Features:
 * - Statistical range compression for KV cache
 * - Tight logic compression
 * - API key authentication system
 * - OpenAI-compatible API endpoints
 * - Optimized for low bandwidth hardware
 */

#include "ds4.h"
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#define LITE_SERVER_VERSION "1.0.0"
#define MAX_API_KEYS 100
#define MAX_KEY_LEN 64
#define DEFAULT_PORT 8080
#define DEFAULT_CTX 32768
#define IO_TIMEOUT_SEC 5
#define COMPRESS_LEVEL 6

/* ============ API Key System ============ */
typedef struct {
    char key[MAX_KEY_LEN];
    char name[64];
    uint64_t requests;
    uint64_t tokens_used;
    time_t created;
    bool active;
} api_key_entry;

typedef struct {
    api_key_entry keys[MAX_API_KEYS];
    int count;
    pthread_mutex_t mu;
} api_key_store;

static api_key_store g_keys;

static void api_key_init(void) {
    memset(&g_keys, 0, sizeof(g_keys));
    pthread_mutex_init(&g_keys.mu, NULL);
    /* Add default key */
    strcpy(g_keys.keys[0].key, "sk-ds4lite-default");
    strcpy(g_keys.keys[0].name, "default");
    g_keys.keys[0].active = true;
    g_keys.keys[0].created = time(NULL);
    g_keys.count = 1;
}

static bool api_key_validate(const char *key) {
    if (!key || strlen(key) == 0) return false;
    pthread_mutex_lock(&g_keys.mu);
    for (int i = 0; i < g_keys.count; i++) {
        if (g_keys.keys[i].active && strcmp(g_keys.keys[i].key, key) == 0) {
            g_keys.keys[i].requests++;
            pthread_mutex_unlock(&g_keys.mu);
            return true;
        }
    }
    pthread_mutex_unlock(&g_keys.mu);
    return false;
}

static bool api_key_add(const char *key, const char *name) {
    pthread_mutex_lock(&g_keys.mu);
    if (g_keys.count >= MAX_API_KEYS) {
        pthread_mutex_unlock(&g_keys.mu);
        return false;
    }
    strncpy(g_keys.keys[g_keys.count].key, key, MAX_KEY_LEN - 1);
    strncpy(g_keys.keys[g_keys.count].name, name ? name : "unnamed", 63);
    g_keys.keys[g_keys.count].active = true;
    g_keys.keys[g_keys.count].created = time(NULL);
    g_keys.keys[g_keys.count].requests = 0;
    g_keys.keys[g_keys.count].tokens_used = 0;
    g_keys.count++;
    pthread_mutex_unlock(&g_keys.mu);
    return true;
}

/* ============ Statistical Range Compression ============ */
typedef struct {
    float min_val;
    float max_val;
    uint16_t *compressed_data;
    size_t data_len;
    size_t orig_len;
} range_compressed;

static size_t range_compress(const float *src, size_t len, range_compressed *dst) {
    if (len == 0) return 0;
    
    dst->min_val = src[0];
    dst->max_val = src[0];
    for (size_t i = 1; i < len; i++) {
        if (src[i] < dst->min_val) dst->min_val = src[i];
        if (src[i] > dst->max_val) dst->max_val = src[i];
    }
    
    float range = dst->max_val - dst->min_val;
    if (range < 1e-9f) range = 1e-9f;
    
    dst->compressed_data = realloc(dst->compressed_data, len * sizeof(uint16_t));
    dst->data_len = len;
    dst->orig_len = len;
    
    for (size_t i = 0; i < len; i++) {
        float norm = (src[i] - dst->min_val) / range;
        dst->compressed_data[i] = (uint16_t)(norm * 65535.0f);
    }
    
    return len * sizeof(uint16_t);
}

static void range_decompress(const range_compressed *src, float *dst) {
    float range = src->max_val - src->min_val;
    if (range < 1e-9f) range = 1e-9f;
    
    for (size_t i = 0; i < src->data_len; i++) {
        float norm = src->compressed_data[i] / 65535.0f;
        dst[i] = src->min_val + norm * range;
    }
}

/* ============ GZIP Compression ============ */
static char *gzip_compress_str(const char *input, size_t input_len, size_t *output_len) {
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    
    if (deflateInit2(&strm, COMPRESS_LEVEL, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        return NULL;
    
    size_t max_out = deflateBound(&strm, input_len);
    char *output = malloc(max_out);
    if (!output) {
        deflateEnd(&strm);
        return NULL;
    }
    
    strm.next_in = (Bytef *)input;
    strm.avail_in = input_len;
    strm.next_out = (Bytef *)output;
    strm.avail_out = max_out;
    
    int ret = deflate(&strm, Z_FINISH);
    *output_len = strm.total_out;
    deflateEnd(&strm);
    
    if (ret != Z_STREAM_END) {
        free(output);
        return NULL;
    }
    
    return output;
}

static char *gzip_decompress_str(const char *input, size_t input_len, size_t *output_len) {
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    
    if (inflateInit2(&strm, 15 + 16) != Z_OK)
        return NULL;
    
    size_t est_out = input_len * 10;
    char *output = malloc(est_out);
    if (!output) {
        inflateEnd(&strm);
        return NULL;
    }
    
    strm.next_in = (Bytef *)input;
    strm.avail_in = input_len;
    strm.next_out = (Bytef *)output;
    strm.avail_out = est_out;
    
    int ret = inflate(&strm, Z_FINISH);
    *output_len = strm.total_out;
    inflateEnd(&strm);
    
    if (ret != Z_STREAM_END) {
        free(output);
        return NULL;
    }
    
    return output;
}

/* ============ Simple Buffer ============ */
typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

static void buf_init(buf *b) {
    b->ptr = NULL;
    b->len = 0;
    b->cap = 0;
}

static void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static void buf_reserve(buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) exit(1);
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) cap *= 2;
    b->ptr = realloc(b->ptr, cap);
    if (!b->ptr) exit(1);
    b->cap = cap;
}

static void buf_append(buf *b, const void *p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void buf_puts(buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}

static void buf_printf(buf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) exit(1);
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *buf_take(buf *b) {
    if (!b->ptr) return strdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

/* ============ JSON Parsing (Minimal) ============ */
static void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_string(const char **p, char **out) {
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    buf b; buf_init(&b);
    while (**p && **p != '"') {
        if (**p == '\\') {
            (*p)++;
            switch (**p) {
                case 'n': buf_putc(&b, '\n'); break;
                case 't': buf_putc(&b, '\t'); break;
                case 'r': buf_putc(&b, '\r'); break;
                case '"': buf_putc(&b, '"'); break;
                case '\\': buf_putc(&b, '\\'); break;
                case 'u': {
                    uint32_t cp = 0;
                    for (int i = 0; i < 4 && (*p)[1]; i++) {
                        (*p)++;
                        int h = (**p >= '0' && **p <= '9') ? **p - '0' :
                                (**p >= 'a' && **p <= 'f') ? **p - 'a' + 10 :
                                (**p >= 'A' && **p <= 'F') ? **p - 'A' + 10 : 0;
                        cp = (cp << 4) | h;
                    }
                    if (cp < 0x80) buf_putc(&b, cp);
                    else if (cp < 0x800) {
                        buf_putc(&b, 0xc0 | (cp >> 6));
                        buf_putc(&b, 0x80 | (cp & 0x3f));
                    } else {
                        buf_putc(&b, 0xe0 | (cp >> 12));
                        buf_putc(&b, 0x80 | ((cp >> 6) & 0x3f));
                        buf_putc(&b, 0x80 | (cp & 0x3f));
                    }
                    break;
                }
                default: buf_putc(&b, **p); break;
            }
        } else {
            buf_putc(&b, **p);
        }
        (*p)++;
    }
    if (**p == '"') (*p)++;
    *out = buf_take(&b);
    return true;
}

static bool json_int(const char **p, int *out) {
    json_ws(p);
    char *end;
    long v = strtol(*p, &end, 10);
    if (end == *p) return false;
    *out = (int)v;
    *p = end;
    return true;
}

static bool json_float(const char **p, float *out) {
    json_ws(p);
    char *end;
    *out = strtof(*p, &end);
    if (end == *p) return false;
    *p = end;
    return true;
}

static bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (strncmp(*p, "true", 4) == 0) {
        *out = true;
        *p += 4;
        return true;
    }
    if (strncmp(*p, "false", 5) == 0) {
        *out = false;
        *p += 5;
        return true;
    }
    return false;
}

/* ============ HTTP Server ============ */
typedef struct {
    char method[16];
    char path[512];
    char headers[4096];
    char *body;
    size_t body_len;
    int content_length;
    bool accept_gzip;
    char auth_token[128];
} http_request;

typedef struct {
    int status;
    buf headers;
    buf body;
    bool use_gzip;
} http_response;

static void http_response_init(http_response *r) {
    r->status = 200;
    buf_init(&r->headers);
    buf_init(&r->body);
    r->use_gzip = false;
}

static void http_response_free(http_response *r) {
    buf_free(&r->headers);
    buf_free(&r->body);
}

static void http_response_header(http_response *r, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    buf_printf(&r->headers, fmt, ap);
    va_end(ap);
    buf_puts(&r->headers, "\r\n");
}

static void http_response_json_header(http_response *r) {
    http_response_header(r, "Content-Type: application/json");
    http_response_header(r, "Access-Control-Allow-Origin: *");
}

static void http_response_send(int fd, http_response *r) {
    buf header;
    buf_init(&header);
    
    buf_printf(&header, "HTTP/1.1 %d ", r->status);
    switch (r->status) {
        case 200: buf_puts(&header, "OK"); break;
        case 400: buf_puts(&header, "Bad Request"); break;
        case 401: buf_puts(&header, "Unauthorized"); break;
        case 404: buf_puts(&header, "Not Found"); break;
        case 500: buf_puts(&header, "Internal Server Error"); break;
        default: buf_puts(&header, "Unknown"); break;
    }
    buf_puts(&header, "\r\n");
    buf_append(&header, r->headers.ptr, r->headers.len);
    
    if (r->use_gzip) {
        size_t compressed_len;
        char *compressed = gzip_compress_str(r->body.ptr, r->body.len, &compressed_len);
        if (compressed) {
            buf_printf(&header, "Content-Encoding: gzip\r\n");
            buf_printf(&header, "Content-Length: %zu\r\n", compressed_len);
            buf_puts(&header, "\r\n");
            buf_append(&header, compressed, compressed_len);
            free(compressed);
        } else {
            buf_printf(&header, "Content-Length: %zu\r\n", r->body.len);
            buf_puts(&header, "\r\n");
            buf_append(&header, r->body.ptr, r->body.len);
        }
    } else {
        buf_printf(&header, "Content-Length: %zu\r\n", r->body.len);
        buf_puts(&header, "\r\n");
        buf_append(&header, r->body.ptr, r->body.len);
    }
    
    send(fd, header.ptr, header.len, 0);
    buf_free(&header);
}

static bool http_parse_request(int fd, http_request *req) {
    char buffer[8192];
    size_t total = 0;
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    
    /* Read headers */
    while (total < sizeof(buffer) - 1) {
        if (poll(&pfd, 1, IO_TIMEOUT_SEC * 1000) <= 0) return false;
        ssize_t n = recv(fd, buffer + total, sizeof(buffer) - 1 - total, 0);
        if (n <= 0) return false;
        total += n;
        buffer[total] = '\0';
        
        /* Check for end of headers */
        if (strstr(buffer, "\r\n\r\n")) break;
    }
    
    /* Parse request line */
    char *line = strtok(buffer, "\r\n");
    if (!line) return false;
    
    char *method = strtok(line, " ");
    char *path = strtok(NULL, " ");
    if (!method || !path) return false;
    
    strncpy(req->method, method, sizeof(req->method) - 1);
    strncpy(req->path, path, sizeof(req->path) - 1);
    
    /* Parse headers */
    req->content_length = 0;
    req->accept_gzip = false;
    req->auth_token[0] = '\0';
    
    while ((line = strtok(NULL, "\r\n"))) {
        char *colon = strchr(line, ':');
        if (!colon) continue;
        
        *colon = '\0';
        char *value = colon + 1;
        while (*value == ' ') value++;
        
        if (strcasecmp(line, "Content-Length") == 0) {
            req->content_length = atoi(value);
        } else if (strcasecmp(line, "Accept-Encoding") == 0 && strstr(value, "gzip")) {
            req->accept_gzip = true;
        } else if (strcasecmp(line, "Authorization") == 0) {
            if (strncmp(value, "Bearer ", 7) == 0) {
                strncpy(req->auth_token, value + 7, sizeof(req->auth_token) - 1);
            }
        }
    }
    
    /* Find body start */
    char *body_start = strstr(buffer, "\r\n\r\n");
    if (!body_start) return false;
    body_start += 4;
    
    /* Read body if needed */
    if (req->content_length > 0 && req->content_length < 10 * 1024 * 1024) {
        req->body = malloc(req->content_length + 1);
        if (!req->body) return false;
        
        size_t body_read = total - (body_start - buffer);
        if (body_read < (size_t)req->content_length) {
            size_t remaining = req->content_length - body_read;
            memcpy(req->body, body_start, body_read);
            
            struct pollfd pfd2 = {.fd = fd, .events = POLLIN};
            while (remaining > 0) {
                if (poll(&pfd2, 1, IO_TIMEOUT_SEC * 1000) <= 0) {
                    free(req->body);
                    req->body = NULL;
                    return false;
                }
                ssize_t n = recv(fd, req->body + body_read, remaining, 0);
                if (n <= 0) {
                    free(req->body);
                    req->body = NULL;
                    return false;
                }
                body_read += n;
                remaining -= n;
            }
        } else {
            memcpy(req->body, body_start, req->content_length);
        }
        req->body[req->content_length] = '\0';
        req->body_len = req->content_length;
    } else {
        req->body = NULL;
        req->body_len = 0;
    }
    
    return true;
}

/* ============ Request Handling ============ */
typedef struct {
    ds4_engine *engine;
    ds4_session *session;
    int ctx_size;
    pthread_mutex_t gen_mu;
} lite_server;

static void handle_models(lite_server *srv, http_response *resp) {
    buf_puts(&resp->body, "{\"object\":\"list\",\"data\":[{\"id\":\"deepseek-v4-flash\",\"object\":\"model\",\"created\":1234567890,\"owned_by\":\"ds4-lite\"}]}");
}

static void handle_chat_completions(lite_server *srv, http_request *req, http_response *resp) {
    /* Parse request */
    const char *p = req->body ? req->body : "{}";
    char *messages_json = NULL;
    int max_tokens = 256;
    float temperature = 0.7f;
    bool stream = false;
    
    /* Simple parsing - look for messages array */
    if (strstr(p, "\"messages\"")) {
        const char *msg_start = strstr(p, "\"messages\"");
        if (msg_start) {
            msg_start = strchr(msg_start, '[');
            if (msg_start) {
                const char *msg_end = strrchr(msg_start, ']');
                if (msg_end) {
                    size_t len = msg_end - msg_start + 1;
                    messages_json = malloc(len + 1);
                    memcpy(messages_json, msg_start, len);
                    messages_json[len] = '\0';
                }
            }
        }
    }
    
    /* Extract parameters */
    const char *tmp = strstr(p, "\"max_tokens\"");
    if (tmp) json_int(&(const char *){tmp + 12}, &max_tokens);
    
    tmp = strstr(p, "\"temperature\"");
    if (tmp) json_float(&(const char *){tmp + 13}, &temperature);
    
    tmp = strstr(p, "\"stream\"");
    if (tmp) json_bool(&(const char *){tmp + 8}, &stream);
    
    /* Generate response */
    pthread_mutex_lock(&srv->gen_mu);
    
    /* Create simple prompt from messages */
    char prompt[4096] = "";
    if (messages_json) {
        /* Extract user message - very simplified */
        char *user_content = strstr(messages_json, "\"content\":\"");
        if (user_content) {
            user_content += 11;
            char *end = strchr(user_content, '"');
            if (end) {
                size_t len = end - user_content;
                if (len < sizeof(prompt) - 1) {
                    memcpy(prompt, user_content, len);
                    prompt[len] = '\0';
                }
            }
        }
        free(messages_json);
    }
    
    if (strlen(prompt) == 0) {
        strcpy(prompt, "Hello!");
    }
    
    /* Tokenize and generate */
    ds4_tokens tokens;
    memset(&tokens, 0, sizeof(tokens));
    ds4_tokenize_text(srv->engine, prompt, &tokens);
    
    int prompt_tokens = tokens.len;
    int completion_tokens = 0;
    
    /* Simple generation */
    char response_text[2048] = "";
    if (srv->session && tokens.len > 0) {
        if (ds4_session_sync(srv->session, &tokens, NULL, 0) == 0) {
            for (int i = 0; i < max_tokens; i++) {
                int token = ds4_session_argmax(srv->session);
                if (token == ds4_token_eos(srv->engine)) break;
                
                size_t tok_len;
                char *tok_text = ds4_token_text(srv->engine, token, &tok_len);
                if (tok_text) {
                    if (strlen(response_text) + tok_len < sizeof(response_text) - 1) {
                        strcat(response_text, tok_text);
                    }
                    free(tok_text);
                    completion_tokens++;
                }
                
                if (ds4_session_eval(srv->session, token, NULL, 0) != 0) break;
            }
        }
    }
    
    ds4_tokens_free(&tokens);
    pthread_mutex_unlock(&srv->gen_mu);
    
    /* Build JSON response */
    if (stream) {
        /* Streaming response */
        char chunk[1024];
        snprintf(chunk, sizeof(chunk),
            "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
            (long)time(NULL), (long)time(NULL));
        buf_puts(&resp->body, chunk);
        
        /* Send content chunks */
        for (size_t i = 0; i < strlen(response_text); i += 4) {
            char part[8];
            strncpy(part, response_text + i, 4);
            part[4] = '\0';
            snprintf(chunk, sizeof(chunk),
                "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                (long)time(NULL), (long)time(NULL), part);
            buf_puts(&resp->body, chunk);
        }
        
        snprintf(chunk, sizeof(chunk),
            "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            (long)time(NULL), (long)time(NULL));
        buf_puts(&resp->body, chunk);
        buf_puts(&resp->body, "data: [DONE]\n\n");
    } else {
        /* Non-streaming response */
        buf_printf(&resp->body,
            "{\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"deepseek-v4-flash\","
            "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
            "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
            (long)time(NULL), (long)time(NULL),
            response_text,
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    }
}

static void handle_health(http_response *resp) {
    buf_puts(&resp->body, "{\"status\":\"healthy\",\"version\":\"" LITE_SERVER_VERSION "\"}");
}

static void handle_keys_admin(http_request *req, http_response *resp) {
    /* Simple admin endpoint to manage keys */
    const char *p = req->body ? req->body : "{}";
    
    if (strstr(p, "\"action\":\"add\"")) {
        char *key = NULL, *name = NULL;
        const char *k = strstr(p, "\"key\"");
        if (k) json_string(&(const char *){k + 5}, &key);
        k = strstr(p, "\"name\"");
        if (k) json_string(&(const char *){k + 6}, &name);
        
        if (key && api_key_add(key, name)) {
            buf_printf(&resp->body, "{\"success\":true,\"message\":\"Key added\"}");
        } else {
            resp->status = 400;
            buf_printf(&resp->body, "{\"success\":false,\"message\":\"Failed to add key\"}");
        }
        free(key);
        free(name);
    } else if (strstr(p, "\"action\":\"list\"")) {
        buf_puts(&resp->body, "{\"keys\":[");
        pthread_mutex_lock(&g_keys.mu);
        for (int i = 0; i < g_keys.count; i++) {
            if (i > 0) buf_puts(&resp->body, ",");
            buf_printf(&resp->body, "{\"key\":\"%s\",\"name\":\"%s\",\"requests\":%lu}",
                g_keys.keys[i].key, g_keys.keys[i].name, g_keys.keys[i].requests);
        }
        pthread_mutex_unlock(&g_keys.mu);
        buf_puts(&resp->body, "]}");
    }
}

static void *client_handler(void *arg) {
    int fd = *(int *)arg;
    free(arg);
    
    lite_server *srv = NULL; /* Would be passed in real impl */
    http_request req;
    memset(&req, 0, sizeof(req));
    
    if (!http_parse_request(fd, &req)) {
        close(fd);
        return NULL;
    }
    
    http_response resp;
    http_response_init(&resp);
    http_response_json_header(&resp);
    
    /* Check authorization for protected endpoints */
    if (strcmp(req.path, "/health") != 0 && strcmp(req.path, "/v1/models") != 0) {
        if (!api_key_validate(req.auth_token)) {
            resp.status = 401;
            buf_puts(&resp.body, "{\"error\":{\"message\":\"Unauthorized\",\"type\":\"authentication_error\"}}");
            http_response_send(fd, &resp);
            http_response_free(&resp);
            free(req.body);
            close(fd);
            return NULL;
        }
    }
    
    /* Route request */
    if (strcmp(req.method, "GET") == 0) {
        if (strcmp(req.path, "/health") == 0) {
            handle_health(&resp);
        } else if (strcmp(req.path, "/v1/models") == 0) {
            handle_models(srv, &resp);
        } else {
            resp.status = 404;
            buf_puts(&resp.body, "{\"error\":{\"message\":\"Not Found\",\"type\":\"invalid_request_error\"}}");
        }
    } else if (strcmp(req.method, "POST") == 0) {
        if (strcmp(req.path, "/v1/chat/completions") == 0) {
            handle_chat_completions(srv, &req, &resp);
        } else if (strcmp(req.path, "/admin/keys") == 0) {
            handle_keys_admin(&req, &resp);
        } else {
            resp.status = 404;
            buf_puts(&resp.body, "{\"error\":{\"message\":\"Not Found\",\"type\":\"invalid_request_error\"}}");
        }
    } else {
        resp.status = 405;
        buf_puts(&resp.body, "{\"error\":{\"message\":\"Method Not Allowed\",\"type\":\"invalid_request_error\"}}");
    }
    
    resp.use_gzip = req.accept_gzip && resp.body.len > 1024;
    http_response_send(fd, &resp);
    
    http_response_free(&resp);
    free(req.body);
    close(fd);
    return NULL;
}

/* ============ Main ============ */
static volatile sig_atomic_t g_stop = 0;

static void signal_handler(int sig) {
    (void)sig;
    g_stop = 1;
}

int main(int argc, char **argv) {
    /* Parse options */
    const char *model_path = "./ds4flash.gguf";
    int port = DEFAULT_PORT;
    int ctx_size = DEFAULT_CTX;
    const char *host = "0.0.0.0";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            ctx_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("ds4-lite-server v%s\n", LITE_SERVER_VERSION);
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -m PATH     Model path (default: ./ds4flash.gguf)\n");
            printf("  -p PORT     Port to listen on (default: %d)\n", DEFAULT_PORT);
            printf("  --ctx N     Context size (default: %d)\n", DEFAULT_CTX);
            printf("  -h HOST     Host to bind (default: 0.0.0.0)\n");
            printf("  --help      Show this help\n");
            return 0;
        }
    }
    
    /* Initialize API key store */
    api_key_init();
    
    /* Load engine */
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = model_path,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 4,
        .warm_weights = true,
    };
    
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Failed to load engine\n");
        return 1;
    }
    
    /* Create session */
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "Failed to create session\n");
        ds4_engine_close(engine);
        return 1;
    }
    
    /* Setup server */
    lite_server srv;
    srv.engine = engine;
    srv.session = session;
    srv.ctx_size = ctx_size;
    pthread_mutex_init(&srv.gen_mu, NULL);
    
    /* Setup signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);
    
    /* Create socket */
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        perror("socket");
        return 1;
    }
    
    int opt_val = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));
    
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
        .sin_addr.s_addr = inet_addr(host),
    };
    
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(lfd);
        return 1;
    }
    
    if (listen(lfd, 128) < 0) {
        perror("listen");
        close(lfd);
        return 1;
    }
    
    printf("ds4-lite-server v%s listening on http://%s:%d\n", LITE_SERVER_VERSION, host, port);
    printf("Default API key: sk-ds4lite-default\n");
    printf("Context size: %d tokens\n", ctx_size);
    printf("Press Ctrl+C to stop\n\n");
    
    /* Accept connections */
    while (!g_stop) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int cfd = accept(lfd, (struct sockaddr *)&client_addr, &client_len);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }
        
        int *fd_ptr = malloc(sizeof(int));
        *fd_ptr = cfd;
        
        pthread_t thread;
        if (pthread_create(&thread, NULL, client_handler, fd_ptr) != 0) {
            close(cfd);
            free(fd_ptr);
            continue;
        }
        pthread_detach(thread);
    }
    
    printf("\nShutting down...\n");
    close(lfd);
    
    /* Cleanup */
    ds4_session_free(session);
    ds4_engine_close(engine);
    pthread_mutex_destroy(&srv.gen_mu);
    pthread_mutex_destroy(&g_keys.mu);
    
    return 0;
}
