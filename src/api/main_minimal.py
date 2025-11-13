"""
Minimal FastAPI server for Cloud Run deployment testing
This version focuses on startup reliability with lazy imports.
"""

import os
import json
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Create FastAPI app with minimal dependencies
app = FastAPI(
    title="RAG3 API - Minimal", 
    version="0.1.0",
    description="Minimal version for Cloud Run testing"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic models
class HealthResponse(BaseModel):
    status: str
    message: str
    port: int

class ModelsResponse(BaseModel):
    models: List[str]

# Health check endpoints
@app.get("/", response_model=HealthResponse)
def root():
    port = int(os.environ.get("PORT", 8080))
    return HealthResponse(
        status="ok", 
        message="RAG3 API is running", 
        port=port
    )

@app.get("/health", response_model=HealthResponse)
def health():
    port = int(os.environ.get("PORT", 8080))
    return HealthResponse(
        status="healthy", 
        message="Service is operational", 
        port=port
    )

@app.get("/test", response_model=HealthResponse)
def test():
    port = int(os.environ.get("PORT", 8080))
    return HealthResponse(
        status="success", 
        message="Test endpoint working", 
        port=port
    )

# Models endpoint - hardcoded for reliability
@app.get("/models", response_model=ModelsResponse)
def get_models():
    return ModelsResponse(models=[
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b"
        # All models confirmed working via direct Groq API testing 2025-11-02
    ])

@app.get("/models/available", response_model=ModelsResponse)
def get_available_models():
    return ModelsResponse(models=[
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b"
        # All models confirmed working via direct Groq API testing 2025-11-02
    ])

# Environment info endpoint for debugging
@app.get("/env-info")
def get_env_info():
    return {
        "port": os.environ.get("PORT", "8080"),
        "environment": os.environ.get("ENVIRONMENT", "development"), 
        "marker_max_memory": os.environ.get("MARKER_MAX_MEMORY_MB", "not set"),
        "gcs_bucket": os.environ.get("GCS_BUCKET_NAME", "not set"),
        "python_path": os.environ.get("PYTHONPATH", "not set"),
        "working_directory": os.getcwd(),
        "data_dir_exists": os.path.exists("/app/data"),
        "src_dir_exists": os.path.exists("/app/src")
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 RAG3 Minimal API starting up...")
    print(f"📝 Port: {os.environ.get('PORT', '8080')}")
    print(f"🌍 Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    print(f"📁 Working Directory: {os.getcwd()}")
    
    # Create basic directories
    os.makedirs("/app/data", exist_ok=True)
    os.makedirs("/app/data/uploads", exist_ok=True)
    os.makedirs("/app/data/markdown", exist_ok=True)
    
    print("✅ Basic directories created")
    print("✅ RAG3 Minimal API ready!")

if __name__ == "__main__":
    import uvicorn
    
    # Cloud Run requires listening on 0.0.0.0 with PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting RAG3 Minimal API server on 0.0.0.0:{port}")
    print(f"🔧 Working directory: {os.getcwd()}")
    print(f"🔧 Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )