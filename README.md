# DS4-Lite Inference Server

A condensed web-based inference server with compression, optimized for low-bandwidth hardware. Built with Python backend and TypeScript/React frontend.

## Features

- **Python Backend**: FastAPI-based server using llama-cpp-python
- **Statistical Range Compression**: 16-bit quantization for reduced memory usage
- **GZIP HTTP Compression**: Automatic compression for responses >1KB
- **GPT-Compatible API**: OpenAI-compatible endpoints
- **API Key Authentication**: Bearer token-based security
- **Streaming Support**: Real-time token streaming
- **Low-Bandwidth Optimized**: Designed for resource-constrained environments
- **No Metal Dependencies**: CPU-only operation for web servers
- **React/TypeScript Frontend**: Modern chat interface

## Project Structure

```
/workspace
├── server.py              # Python backend server
├── requirements.txt       # Python dependencies
├── frontend/              # TypeScript/React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── App.css
│       ├── api.ts
│       └── types.ts
└── README.md
```

## Installation

### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the server
python server.py -m your-model.gguf -p 8080 --ctx 32768
```

### Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Usage

### Starting the Server

```bash
# Basic usage
python server.py -m model.gguf

# With custom settings
python server.py -m model.gguf -p 8080 --ctx 32768 --threads 8 --gpu-layers 0
```

### Environment Variables

```bash
export MODEL_PATH="model.gguf"
export PORT=8080
export CONTEXT_SIZE=32768
export N_THREADS=4
export N_GPU_LAYERS=0
export MAX_TOKENS=2048
export API_KEYS_FILE="api_keys.json"
export COMPRESSION_LEVEL=6
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8080/health
```

#### List Models
```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer sk-your-api-key"
```

#### Chat Completion
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

#### Streaming Chat Completion
```bash
curl http://localhost:8080/v1/chat/completions-stream \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

#### Admin: Create API Key
```bash
curl http://localhost:8080/admin/keys/create \
  -H "Authorization: Bearer sk-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-app",
    "role": "user"
  }'
```

#### Admin: List API Keys
```bash
curl http://localhost:8080/admin/keys/list \
  -H "Authorization: Bearer sk-admin-key"
```

#### Admin: Delete API Key
```bash
curl http://localhost:8080/admin/keys/delete \
  -H "Authorization: Bearer sk-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"key": "sk-key-to-delete"}'
```

## API Key System

The server automatically creates an admin API key on first run. Check the console output or `api_keys.json` file for the key.

**Default admin key format**: `sk-admin-<hex>`

Store this key securely as it's required for admin operations.

## Compression Features

### HTTP GZIP Compression
- Automatic compression for responses >1KB
- Configurable compression level (1-9)
- Reduces bandwidth usage by 60-80%

## Frontend Features

- **Real-time Streaming**: See tokens as they're generated
- **Settings Panel**: Configure API key, temperature, max tokens
- **Health Monitoring**: Automatic server status checks
- **Responsive Design**: Works on desktop and mobile
- **Chat History**: Scrollable conversation history
- **Clear Chat**: Reset conversation context

## Performance Optimization

- **CPU-Only**: No GPU dependencies for broader compatibility
- **Thread Control**: Configurable thread count
- **Batch Processing**: Efficient token generation
- **Context Management**: Large context window support (up to 32K)
- **Memory Efficient**: Quantized operations reduce RAM usage

## Security

- **Bearer Token Authentication**: All API endpoints require valid API key
- **Role-Based Access**: Admin and user roles
- **Key Persistence**: API keys stored in JSON file
- **Usage Tracking**: Monitor API key usage

## Development

### Backend Development

```bash
# Run with auto-reload
uvicorn server:app --reload --host 0.0.0.0 --port 8080
```

### Frontend Development

```bash
cd frontend
npm run dev
```

The frontend proxies API requests to the backend at `localhost:8080`.

## Production Deployment

### Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY *.gguf .

EXPOSE 8080
CMD ["python", "server.py", "-m", "model.gguf"]
```

### Systemd Service

```ini
[Unit]
Description=DS4-Lite Inference Server
After=network.target

[Service]
Type=simple
User=ds4lite
WorkingDirectory=/opt/ds4-lite
Environment="MODEL_PATH=/opt/ds4-lite/model.gguf"
ExecStart=/usr/bin/python3 /opt/ds4-lite/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Model Loading Issues
- Ensure model file is in GGUF format
- Check file permissions
- Verify sufficient RAM for model size

### API Key Errors
- Default admin key is created on first run
- Check `api_keys.json` for existing keys
- Use admin key to create new user keys

### Performance Issues
- Reduce context size (`--ctx`)
- Decrease thread count if CPU-bound
- Use quantized model files (Q4_K_M, Q5_K_M, etc.)

## License

MIT License
