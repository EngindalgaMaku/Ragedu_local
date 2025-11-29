"""
Embedding generation service
Handles communication with model inference service for embeddings
"""
import requests
from typing import List
from fastapi import HTTPException
from utils.logger import logger
from config import MODEL_INFERENCER_URL


def get_embeddings_direct(texts: List[str], embedding_model: str = "text-embedding-v4") -> List[List[float]]:
    """
    Direct embedding function for use with unified chunking system.
    Get embeddings from local model inference service (Ollama)
    
    Args:
        texts: List of text strings to embed
        embedding_model: Model name to use for embeddings
        
    Returns:
        List of embedding vectors (each is a list of floats)
        
    Raises:
        HTTPException: If embedding generation fails
    """
    try:
        embed_url = f"{MODEL_INFERENCER_URL}/embed"
        
        logger.info(f"Getting embeddings for {len(texts)} texts using model: {embedding_model}")
        
        # Send all texts in a single request for efficiency
        response = requests.post(
            embed_url,
            json={"texts": texts, "model": embedding_model},
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes for multiple chunks with slow local embeddings
        )
        
        if response.status_code != 200:
            logger.error(f"Local embedding error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get local embeddings: {response.status_code}"
            )
        
        embedding_data = response.json()
        embeddings = embedding_data.get("embeddings", [])
        
        if len(embeddings) != len(texts):
            raise HTTPException(
                status_code=500,
                detail=f"Embedding count ({len(embeddings)}) doesn't match text count ({len(texts)})"
            )
        
        logger.info(f"Successfully retrieved {len(embeddings)} local embeddings")
        return embeddings
        
    except Exception as e:
        logger.warning(f"Embedding service error with {embedding_model}: {str(e)}")
        raise  # Re-raise to allow fallback mechanism






