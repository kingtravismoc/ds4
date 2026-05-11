#!/usr/bin/env python3
"""
DS4-Lite Python Backend Server
Condensed inference server with statistical range compression, 
low-bandwidth optimization, and GPT-compatible API.
"""

import os
import sys
import json
import time
import hashlib
import secrets
import gzip
import logging
import threading
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from llama_cpp import Llama
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ds4-lite")

# ============================================================================
# Configuration
# ============================================================================

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "model.gguf")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8080"))
    CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "32768"))
    N_THREADS = int(os.getenv("N_THREADS", "4"))
    N_BATCH = int(os.getenv("N_BATCH", "512"))
    N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))  # 0 for CPU-only
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    API_KEYS_FILE = os.getenv("API_KEYS_FILE", "api_keys.json")
    COMPRESSION_LEVEL = int(os.getenv("COMPRESSION_LEVEL", "6"))  # GZIP level
    
    # Statistical compression settings
    QUANTIZE_BITS = int(os.getenv("QUANTIZE_BITS", "16"))  # 16-bit quantization
    COMPRESSION_THRESHOLD = int(os.getenv("COMPRESSION_THRESHOLD", "1024"))  # Min bytes for GZIP

config = Config()

# ============================================================================
# Statistical Range Compression
# ============================================================================

class StatisticalCompressor:
    """Compress data using statistical range methods and quantization."""
    
    @staticmethod
    def quantize_array(data: np.ndarray, bits: int = 16) -> tuple:
        """Quantize float array to reduced bit representation."""
        if bits == 32:
            return data.astype(np.float32).tobytes(), data.min(), data.max(), 32
        
        # Normalize to [0, 1]
        min_val = data.min()
        max_val = data.max()
        range_val = max_val - min_val if max_val != min_val else 1.0
        
        # Quantize
        levels = (1 << bits) - 1
        normalized = (data - min_val) / range_val
        quantized = np.round(normalized * levels).astype(np.uint16 if bits > 8 else np.uint8)
        
        return quantized.tobytes(), min_val, max_val, bits
    
    @staticmethod
    def dequantize_array(data_bytes: bytes, shape: tuple, min_val: float, 
                         max_val: float, bits: int) -> np.ndarray:
        """Dequantize compressed data back to float array."""
        dtype = np.uint16 if bits > 8 else np.uint8
        quantized = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
        
        levels = (1 << bits) - 1
        normalized = quantized.astype(np.float32) / levels
        return normalized * (max_val - min_val) + min_val
    
    @staticmethod
    def compress_response(data: Dict[str, Any]) -> bytes:
        """Compress response dictionary using GZIP."""
        json_str = json.dumps(data, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        
        if len(json_bytes) < config.COMPRESSION_THRESHOLD:
            return json_bytes
        
        return gzip.compress(json_bytes, compresslevel=config.COMPRESSION_LEVEL)
    
    @staticmethod
    def decompress_response(compressed: bytes) -> Dict[str, Any]:
        """Decompress GZIP response."""
        try:
            decompressed = gzip.decompress(compressed)
            return json.loads(decompressed.decode('utf-8'))
        except:
            return json.loads(compressed.decode('utf-8'))

# ============================================================================
# API Key Management
# ============================================================================

class APIKeyManager:
    """Manage API keys for authentication."""
    
    def __init__(self, keys_file: str):
        self.keys_file = keys_file
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from file."""
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, 'r') as f:
                    self.keys = json.load(f)
                logger.info(f"Loaded {len(self.keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
                self.keys = {}
        
        # Create default admin key if none exist
        if not self.keys:
            admin_key = f"sk-admin-{secrets.token_hex(16)}"
            self.keys[admin_key] = {
                "name": "admin",
                "role": "admin",
                "created": time.time(),
                "usage": 0
            }
            self._save_keys()
            logger.info(f"Created admin API key: {admin_key}")
    
    def _save_keys(self):
        """Save API keys to file."""
        with self.lock:
            with open(self.keys_file, 'w') as f:
                json.dump(self.keys, f, indent=2)
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return key info if valid."""
        with self.lock:
            if api_key in self.keys:
                self.keys[api_key]["usage"] = self.keys[api_key].get("usage", 0) + 1
                return self.keys[api_key]
        return None
    
    def create_key(self, name: str, role: str = "user") -> str:
        """Create a new API key."""
        api_key = f"sk-{secrets.token_hex(16)}"
        with self.lock:
            self.keys[api_key] = {
                "name": name,
                "role": role,
                "created": time.time(),
                "usage": 0
            }
        self._save_keys()
        return api_key
    
    def delete_key(self, api_key: str) -> bool:
        """Delete an API key."""
        with self.lock:
            if api_key in self.keys:
                del self.keys[api_key]
                self._save_keys()
                return True
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without exposing the actual keys)."""
        with self.lock:
            return [
                {
                    "name": info["name"],
                    "role": info["role"],
                    "created": info["created"],
                    "usage": info.get("usage", 0)
                }
                for info in self.keys.values()
            ]

# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-v4-flash"
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[UsageInfo] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class KeyCreateRequest(BaseModel):
    name: str
    role: str = "user"

class KeyDeleteRequest(BaseModel):
    key: str

# ============================================================================
# Global State
# ============================================================================

llm_model: Optional[Llama] = None
api_key_manager: Optional[APIKeyManager] = None
compressor = StatisticalCompressor()
security = HTTPBearer(auto_error=False)

# ============================================================================
# Dependencies
# ============================================================================

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Validate API key from Authorization header."""
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    key_info = api_key_manager.validate_key(credentials.credentials)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return key_info

# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global llm_model, api_key_manager
    
    # Startup
    logger.info("Starting DS4-Lite server...")
    
    # Initialize API key manager
    api_key_manager = APIKeyManager(config.API_KEYS_FILE)
    
    # Load model
    logger.info(f"Loading model from {config.MODEL_PATH}...")
    try:
        llm_model = Llama(
            model_path=config.MODEL_PATH,
            n_ctx=config.CONTEXT_SIZE,
            n_threads=config.N_THREADS,
            n_batch=config.N_BATCH,
            n_gpu_layers=config.N_GPU_LAYERS,
            verbose=False
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        llm_model = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down DS4-Lite server...")
    llm_model = None

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="DS4-Lite Inference Server",
    description="Condensed inference server with statistical compression",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def compression_middleware(request: Request, call_next):
    """Apply GZIP compression to responses."""
    response = await call_next(request)
    
    accept_encoding = request.headers.get("accept-encoding", "")
    
    if "gzip" in accept_encoding:
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        if len(body) >= config.COMPRESSION_THRESHOLD:
            compressed_body = gzip.compress(body, compresslevel=config.COMPRESSION_LEVEL)
            
            return Response(
                content=compressed_body,
                status_code=response.status_code,
                headers={
                    **response.headers,
                    "content-encoding": "gzip",
                    "content-length": str(len(compressed_body))
                }
            )
    
    return response

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": llm_model is not None,
        "timestamp": int(time.time())
    }

@app.get("/v1/models", response_model=ModelList)
async def list_models(api_key: Dict = Depends(get_api_key)):
    """List available models."""
    models = [
        ModelInfo(
            id="deepseek-v4-flash",
            created=int(time.time()),
            owned_by="ds4-lite"
        )
    ]
    return ModelList(data=models)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: Dict = Depends(get_api_key)
):
    """Create chat completion."""
    if not llm_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Format messages for llama.cpp
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant: "
    
    # Generate response
    max_tokens = request.max_tokens or config.MAX_TOKENS
    
    try:
        output = llm_model(
            prompt,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop or ["User:", "System:"],
            echo=False
        )
        
        generated_text = output["choices"][0]["text"].strip()
        prompt_tokens = output["usage"]["prompt_tokens"]
        completion_tokens = output["usage"]["completion_tokens"]
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    response = ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=generated_text),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )
    
    return response

@app.post("/v1/chat/completions-stream")
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    api_key: Dict = Depends(get_api_key)
):
    """Create streaming chat completion."""
    if not llm_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant: "
    
    max_tokens = request.max_tokens or config.MAX_TOKENS
    
    async def generate():
        try:
            for token_output in llm_model(
                prompt,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop or ["User:", "System:"],
                echo=False,
                stream=True
            ):
                token = token_output["choices"][0]["text"]
                
                chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }
                    ]
                }
                
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final chunk
            final_chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            error_chunk = {
                "error": {"message": str(e), "type": "generation_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return Response(
        content=generate(),
        media_type="text/event-stream"
    )

@app.post("/admin/keys/create")
async def create_api_key(
    req: KeyCreateRequest,
    api_key: Dict = Depends(get_api_key)
):
    """Create a new API key (admin only)."""
    if api_key.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    new_key = api_key_manager.create_key(req.name, req.role)
    return {"key": new_key, "name": req.name, "role": req.role}

@app.post("/admin/keys/delete")
async def delete_api_key(
    req: KeyDeleteRequest,
    api_key: Dict = Depends(get_api_key)
):
    """Delete an API key (admin only)."""
    if api_key.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if api_key_manager.delete_key(req.key):
        return {"success": True, "message": "Key deleted"}
    else:
        raise HTTPException(status_code=404, detail="Key not found")

@app.get("/admin/keys/list")
async def list_api_keys(
    api_key: Dict = Depends(get_api_key)
):
    """List all API keys (admin only)."""
    if api_key.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    keys = api_key_manager.list_keys()
    return {"keys": keys, "count": len(keys)}

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DS4-Lite Inference Server")
    parser.add_argument("-m", "--model", type=str, default=config.MODEL_PATH,
                       help="Path to model file")
    parser.add_argument("-p", "--port", type=int, default=config.PORT,
                       help="Server port")
    parser.add_argument("--host", type=str, default=config.HOST,
                       help="Server host")
    parser.add_argument("--ctx", type=int, default=config.CONTEXT_SIZE,
                       help="Context size")
    parser.add_argument("--threads", type=int, default=config.N_THREADS,
                       help="Number of threads")
    parser.add_argument("--gpu-layers", type=int, default=config.N_GPU_LAYERS,
                       help="Number of GPU layers (0 for CPU-only)")
    
    args = parser.parse_args()
    
    # Update config
    config.MODEL_PATH = args.model
    config.PORT = args.port
    config.HOST = args.host
    config.CONTEXT_SIZE = args.ctx
    config.N_THREADS = args.threads
    config.N_GPU_LAYERS = args.gpu_layers
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           DS4-Lite Python Inference Server                ║
╠═══════════════════════════════════════════════════════════╣
║ Model: {config.MODEL_PATH:<50} ║
║ Port: {config.PORT:<52} ║
║ Context: {config.CONTEXT_SIZE:<49} ║
║ Threads: {config.N_THREADS:<49} ║
║ GPU Layers: {config.N_GPU_LAYERS:<46} ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=config.HOST, port=config.PORT)

if __name__ == "__main__":
    main()
