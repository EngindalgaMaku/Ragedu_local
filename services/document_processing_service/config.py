"""
Configuration management for Document Processing Service
Handles environment variables and service settings
"""
import os
from typing import Optional

# Service port
PORT = int(os.getenv("PORT", "8080"))

# Model Inference Service URL configuration
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", os.getenv("MODEL_INFERENCE_URL", None))
if not MODEL_INFERENCER_URL:
    MODEL_INFERENCE_HOST = os.getenv("MODEL_INFERENCE_HOST", "model-inference-service")
    MODEL_INFERENCE_PORT = os.getenv("MODEL_INFERENCE_PORT", "8002")
    if MODEL_INFERENCE_HOST.startswith("http://") or MODEL_INFERENCE_HOST.startswith("https://"):
        MODEL_INFERENCER_URL = MODEL_INFERENCE_HOST
    else:
        MODEL_INFERENCER_URL = f"http://{MODEL_INFERENCE_HOST}:{MODEL_INFERENCE_PORT}"

# ChromaDB Service URL configuration
CHROMADB_URL = os.getenv("CHROMADB_URL", None)
if not CHROMADB_URL:
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "chromadb-service")
    CHROMADB_PORT = os.getenv("CHROMADB_PORT", "8000")
    if CHROMADB_HOST.startswith("http://") or CHROMADB_HOST.startswith("https://"):
        CHROMADB_URL = CHROMADB_HOST
    else:
        CHROMADB_URL = f"http://{CHROMADB_HOST}:{CHROMADB_PORT}"

CHROMA_SERVICE_URL = os.getenv("CHROMA_SERVICE_URL", CHROMADB_URL)

# Default settings
MIN_SIMILARITY_DEFAULT = float(os.getenv("MIN_SIMILARITY_DEFAULT", "0.5"))
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")

# Feature flags
UNIFIED_CHUNKING_AVAILABLE = False  # Will be set during import

def get_model_inferencer_url() -> str:
    """Get the model inference service URL"""
    return MODEL_INFERENCER_URL

def get_chroma_service_url() -> str:
    """Get the ChromaDB service URL"""
    return CHROMA_SERVICE_URL






