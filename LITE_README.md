# ds4-lite-server

Condensed web server for low-bandwidth DeepSeek V4 Flash inference with API key authentication and compression.

## Features

- **Statistical Range Compression**: 16-bit quantization of float ranges for KV cache transmission
- **GZIP Response Compression**: Automatic gzip compression for large responses
- **API Key Authentication**: GPT-compatible Bearer token system
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **Low Bandwidth Optimized**: Minimal HTTP overhead, compressed responses
- **Tight Logic**: ~900 lines vs ~10,000 lines in full server

## Building

```bash
make ds4-lite-server
```

Requires zlib (`-lz`).

## Usage

```bash
./ds4-lite-server -m ./ds4flash.gguf -p 8080 --ctx 32768
```

### Options

- `-m PATH`: Model path (default: ./ds4flash.gguf)
- `-p PORT`: Port to listen on (default: 8080)
- `--ctx N`: Context size (default: 32768)
- `-h HOST`: Host to bind (default: 0.0.0.0)

## API Endpoints

### GET /health
Health check endpoint (no auth required).

```bash
curl http://localhost:8080/health
```

Response:
```json
{"status":"healthy","version":"1.0.0"}
```

### GET /v1/models
List available models (no auth required).

```bash
curl http://localhost:8080/v1/models
```

### POST /v1/chat/completions
Chat completions endpoint (requires API key).

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ds4lite-default" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'
```

#### Streaming Mode

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ds4lite-default" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### POST /admin/keys
Manage API keys (requires valid API key).

#### Add new key:
```bash
curl http://localhost:8080/admin/keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ds4lite-default" \
  -d '{"action":"add","key":"sk-my-new-key","name":"my-app"}'
```

#### List all keys:
```bash
curl http://localhost:8080/admin/keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ds4lite-default" \
  -d '{"action":"list"}'
```

## API Key System

Default API key: `sk-ds4lite-default`

Keys are stored in memory. Add keys via the `/admin/keys` endpoint or modify the source to persist them.

Key format: `sk-*` (OpenAI compatible)

## Compression

### Statistical Range Compression
Float arrays are compressed to 16-bit integers using min-max normalization:
- Store min/max values (8 bytes total)
- Each float becomes 2 bytes
- Typical compression: 50% (32-bit float → 16-bit int)
- Lossless within 16-bit precision

### GZIP HTTP Compression
- Automatically enabled for responses > 1KB when client supports it
- Controlled by `Accept-Encoding: gzip` header
- Typical text compression: 70-90%

## Comparison with Full Server

| Feature | Full Server | Lite Server |
|---------|-------------|-------------|
| Lines of code | ~10,000 | ~900 |
| API key auth | No | Yes |
| Response compression | No | GZIP |
| KV cache compression | Disk-based | Range compression |
| Tool calls | Full DSML support | Basic |
| Concurrent requests | Single worker | Thread-per-client |
| Anthropic API | Yes | No |
| Speculative decoding | Yes | No |
| KV disk cache | Yes | No |
| Session reuse | Advanced | Basic |

## Use Cases

- **Edge deployment**: Run on low-bandwidth connections
- **Embedded systems**: Minimal memory footprint
- **Testing**: Quick API compatibility testing
- **Prototyping**: Fast iteration on API design
- **Learning**: Understand server structure

## Limitations

- Simplified JSON parsing (may not handle all edge cases)
- Basic session management (no advanced KV cache features)
- No speculative decoding support
- No tool call canonicalization
- Metal-only (no CPU fallback in lite version)
- Limited error handling

## License

Same as ds4.c project (MIT with GGML acknowledgements)
