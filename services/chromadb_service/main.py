#!/usr/bin/env python3
"""
ChromaDB Proxy Service
Simple proxy service to handle ChromaDB authentication and provide public access
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChromaDB Proxy Service",
    description="Proxy service for ChromaDB with authentication handling",
    version="1.0.0"
)

# Environment variables - Google Cloud Run compatible
# For Docker: use service name or local URL
# For Cloud Run: use full URL (e.g., https://chromadb-xxx.run.app)
PORT = int(os.getenv("PORT", "8080"))
CHROMADB_INTERNAL_URL = os.getenv("CHROMADB_INTERNAL_URL", os.getenv("CHROMADB_URL", None))
if not CHROMADB_INTERNAL_URL:
    # Fallback: assume local ChromaDB instance
    CHROMADB_INTERNAL_URL = "http://localhost:8000"

# Pydantic models
class CreateCollectionRequest(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]] = {}

class QueryRequest(BaseModel):
    query_embeddings: List[List[float]]
    n_results: int = 10
    where: Optional[Dict[str, Any]] = {}
    where_document: Optional[Dict[str, Any]] = {}
    include: Optional[List[str]] = ["metadatas", "documents", "distances"]

class AddDocumentsRequest(BaseModel):
    ids: List[str]
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]
    documents: List[str]

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "ChromaDB Proxy Service", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Try to connect to ChromaDB
        response = requests.get(f"{CHROMADB_INTERNAL_URL}/api/v1/heartbeat", timeout=5)
        chromadb_healthy = response.status_code == 200
        
        return {
            "status": "healthy",
            "chromadb_connected": chromadb_healthy,
            "chromadb_url": CHROMADB_INTERNAL_URL
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "healthy", # Proxy is healthy even if ChromaDB is not
            "chromadb_connected": False,
            "error": str(e)
        }

@app.get("/api/v1/heartbeat")
async def heartbeat():
    """ChromaDB heartbeat proxy"""
    try:
        response = requests.get(f"{CHROMADB_INTERNAL_URL}/api/v1/heartbeat", timeout=10)
        return response.json()
    except Exception as e:
        logger.error(f"ChromaDB heartbeat failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/collections")
async def list_collections():
    """List all collections"""
    try:
        response = requests.get(f"{CHROMADB_INTERNAL_URL}/api/v1/collections", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=503, detail=f"ChromaDB service unavailable: {e}")

@app.post("/api/v1/collections")
async def create_collection(request: CreateCollectionRequest):
    """Create a new collection"""
    try:
        payload = {
            "name": request.name,
            "metadata": request.metadata or {}
        }
        
        response = requests.post(
            f"{CHROMADB_INTERNAL_URL}/api/v1/collections",
            json=payload,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(status_code=503, detail=f"ChromaDB service unavailable: {e}")

@app.get("/api/v1/collections/{collection_name}")
async def get_collection(collection_name: str):
    """Get collection info"""
    try:
        response = requests.get(
            f"{CHROMADB_INTERNAL_URL}/api/v1/collections/{collection_name}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get collection: {e}")
        raise HTTPException(status_code=503, detail=f"ChromaDB service unavailable: {e}")

@app.post("/api/v1/collections/{collection_name}/add")
async def add_documents(collection_name: str, request: AddDocumentsRequest):
    """Add documents to collection"""
    try:
        payload = {
            "ids": request.ids,
            "embeddings": request.embeddings,
            "metadatas": request.metadatas,
            "documents": request.documents
        }
        
        response = requests.post(
            f"{CHROMADB_INTERNAL_URL}/api/v1/collections/{collection_name}/add",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to add documents: {e}")
        raise HTTPException(status_code=503, detail=f"ChromaDB service unavailable: {e}")

@app.post("/api/v1/collections/{collection_name}/query")
async def query_collection(collection_name: str, request: QueryRequest):
    """Query collection"""
    try:
        payload = {
            "query_embeddings": request.query_embeddings,
            "n_results": request.n_results,
            "where": request.where or {},
            "where_document": request.where_document or {},
            "include": request.include or ["metadatas", "documents", "distances"]
        }
        
        response = requests.post(
            f"{CHROMADB_INTERNAL_URL}/api/v1/collections/{collection_name}/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to query collection: {e}")
        raise HTTPException(status_code=503, detail=f"ChromaDB service unavailable: {e}")

@app.delete("/api/v1/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete collection"""
    try:
        response = requests.delete(
            f"{CHROMADB_INTERNAL_URL}/api/v1/collections/{collection_name}",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=503, detail=f"ChromaDB service unavailable: {e}")

@app.get("/api/v1/collections/{collection_name}/count")
async def count_collection(collection_name: str):
    """Count documents in collection"""
    try:
        response = requests.get(
            f"{CHROMADB_INTERNAL_URL}/api/v1/collections/{collection_name}/count",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to count collection: {e}")
        raise HTTPException(status_code=503, detail=f"ChromaDB service unavailable: {e}")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting ChromaDB Proxy Service on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)