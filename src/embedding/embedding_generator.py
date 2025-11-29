"""
Embedding generation module.

This module provides functions to generate text embeddings using the model-inference-service:
- HTTP requests to model-inference-service /embed endpoint
- Caching support for performance optimization
"""

from typing import List, Optional
import hashlib
import time
import numpy as np
import requests
import json
from .. import config
from ..config import is_cloud_environment, get_model_inference_url
from ..utils.helpers import setup_logging
from ..utils.cache import get_cache

logger = setup_logging()

# Global configuration
MODEL_INFERENCE_URL = get_model_inference_url()

def _truncate_text_for_embedding(text: str, max_length: int = 1500) -> str:
    """
    Truncate text to fit within embedding model context length.
    
    Args:
        text: Input text
        max_length: Maximum character length (approximate token limit)
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    last_sentence_end = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?')
    )
    
    if last_sentence_end > max_length * 0.7:  # If we found a good break point
        return truncated[:last_sentence_end + 1].strip()
    else:
        # Fallback to word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:
            return truncated[:last_space].strip()
        else:
            # Hard truncation as last resort
            return truncated.strip()

def _get_cache_key(text: str, model: str) -> str:
    """Generate cache key for text and model combination."""
    content = f"{model}:{text}"
    return f"embedding:{hashlib.md5(content.encode()).hexdigest()}"

def _clean_text_for_embedding(text: str) -> str:
    """
    Clean text for embedding generation - Turkish compatible.
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned text
    """
    # Basic cleaning
    clean_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    
    # Multiple spaces to single space
    import re
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Keep Turkish characters, clean special characters
    clean_text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?:;()-]', ' ', clean_text)
    
    # Clean multiple spaces again
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

def _generate_embeddings_via_http(texts: List[str], use_cache: bool = True, batch_size: int = 25) -> List[List[float]]:
    """
    Generate embeddings using HTTP requests to model-inference-service with batching support.
    
    Args:
        texts: List of texts to embed
        use_cache: Whether to use caching
        batch_size: Number of texts to process in each batch (default: 25)
    
    Returns:
        List of embeddings
    """
    if not texts:
        return []
    
    logger.info(f"Generating embeddings for {len(texts)} text(s) via model-inference-service with batch size {batch_size}")
    
    # Clean and preprocess texts
    cleaned_texts = []
    for text in texts:
        clean_text = _clean_text_for_embedding(text)
        clean_text = _truncate_text_for_embedding(clean_text)
        cleaned_texts.append(clean_text)
    
    # Initialize cache if enabled
    cache = get_cache(ttl=config.get_config().model_config.get('cache_ttl', 3600)) if use_cache else None
    
    # Prepare result list with placeholders
    embeddings = [None] * len(cleaned_texts)
    
    # Process in batches
    for batch_start in range(0, len(cleaned_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(cleaned_texts))
        batch_texts = cleaned_texts[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        
        # Check cache for this batch
        batch_to_fetch = []
        batch_to_fetch_indices = []
        
        for i, text in zip(batch_indices, batch_texts):
            if cache:
                cache_key = _get_cache_key(text, "nomic-embed-text")
                cached_embedding = cache.get(cache_key)
                if cached_embedding:
                    embeddings[i] = cached_embedding
                    logger.debug(f"Found cached embedding for text {i}")
                    continue
            
            batch_to_fetch.append(text)
            batch_to_fetch_indices.append(i)
        
        # Process uncached texts in this batch
        if not batch_to_fetch:
            continue
            
        logger.info(f"Processing batch of {len(batch_to_fetch)} texts (batch {batch_start//batch_size + 1}/{(len(cleaned_texts) + batch_size - 1)//batch_size})")
        
        try:
            # Prepare and send batch request
            payload = {"texts": batch_to_fetch}
            embed_url = f"{MODEL_INFERENCE_URL}/embed"
            
            response = requests.post(
                embed_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 minutes timeout for embedding generation
            )
            
            if response.status_code == 200:
                result = response.json()
                processed_embeddings = result.get("embeddings", [])
                
                if len(processed_embeddings) != len(batch_to_fetch):
                    logger.error(f"Mismatch in batch embedding count: expected {len(batch_to_fetch)}, got {len(processed_embeddings)}")
                    # Skip this batch to avoid index errors
                    continue
                
                # Cache and store results
                for i, (text, embedding) in enumerate(zip(batch_to_fetch, processed_embeddings)):
                    if embedding:  # Only cache valid embeddings
                        if cache:
                            cache_key = _get_cache_key(text, "nomic-embed-text")
                            cache.set(cache_key, embedding)
                        
                        # Store the embedding at the correct index
                        original_index = batch_to_fetch_indices[i]
                        embeddings[original_index] = embedding
            else:
                logger.error(f"HTTP request failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request error in batch {batch_start//batch_size + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during batch processing: {e}")
    
    # Handle any remaining None values (failed embeddings)
    final_embeddings = []
    for i, emb in enumerate(embeddings):
        if emb is not None and len(emb) > 0:
            final_embeddings.append(emb)
        else:
            logger.warning(f"Failed to generate embedding for text at index {i}, using fallback")
            # Use a simple fallback for failed embeddings
            fallback = _generate_simple_embeddings([texts[i]])[0] if i < len(texts) else [0.0] * 768
            final_embeddings.append(fallback)
    
    logger.info(f"Successfully generated {len(final_embeddings)} embeddings via model-inference-service")
    return final_embeddings

def _generate_simple_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate simple hash-based embeddings as a last resort fallback.
    This is NOT suitable for production but prevents complete failure.
    """
    logger.warning("Using simple hash-based embeddings - NOT recommended for production!")
    embeddings = []
    
    for text in texts:
        # Create a simple deterministic embedding based on text hash and content
        import hashlib
        
        # Generate multiple hash values to create a vector
        hash_md5 = hashlib.md5(text.encode()).hexdigest()
        hash_sha1 = hashlib.sha1(text.encode()).hexdigest()
        hash_sha256 = hashlib.sha256(text.encode()).hexdigest()
        
        # Convert hex to numbers and normalize
        embedding = []
        for i in range(0, min(len(hash_md5), 32), 2):
            val = int(hash_md5[i:i+2], 16) / 255.0  # Normalize to [0,1]
            embedding.append(val)
        
        for i in range(0, min(len(hash_sha1), 32), 2):
            val = int(hash_sha1[i:i+2], 16) / 255.0
            embedding.append(val)
            
        # Add some text-based features
        embedding.extend([
            len(text) / 1000.0,  # Text length feature
            text.count(' ') / 100.0,  # Word count feature
            text.count('.') / 10.0,   # Sentence count feature
            text.count('?') / 10.0,   # Question count feature
        ])
        
        # Pad or truncate to fixed size
        target_size = 384  # Common embedding size
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
            
        embeddings.append(embedding)
    
    logger.info(f"Generated {len(embeddings)} simple fallback embeddings")
    return embeddings

def generate_embeddings(texts: List[str], model: str = None, use_cache: bool = True, provider: str = None, batch_size: int = 25) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using model-inference-service.
    Supports caching for improved performance.

    Args:
        texts: A list of strings to be embedded.
        model: The embedding model to use (ignored, always uses nomic-embed-text via service).
        use_cache: Whether to use caching for embeddings.
        provider: Provider override (ignored, always uses model-inference-service).
        batch_size: Number of texts to process in each batch (default: 25)

    Returns:
        A list of embeddings, where each embedding is a list of floats.
        Returns an empty list if an error occurs or no provider is available.
    """
    if not texts:
        return []
    
    logger.info(f"Generating embeddings for {len(texts)} text(s) using model-inference-service with batch size {batch_size}")
    
    # Always use HTTP-based embedding generation via model-inference-service
    result = _generate_embeddings_via_http(texts, use_cache, batch_size)
    
    if not result:
        # Last resort: simple embeddings
        logger.warning("Model-inference-service embeddings failed, using simple fallback")
        result = _generate_simple_embeddings(texts)
    
    return result

if __name__ == '__main__':
    # Testing purposes
    print("--- Testing Embedding Generation via Model Inference Service ---")
    sample_texts = [
        "This is the first test sentence.",
        "Here is another sentence for embedding.",
        "RAG systems rely on high-quality embeddings."
    ]
    
    embeddings = generate_embeddings(sample_texts)
    
    if embeddings:
        print(f"Successfully generated {len(embeddings)} embeddings.")
        for i, emb in enumerate(embeddings):
            print(f"  - Embedding {i+1}: Dimension = {len(emb)}, First 5 values = {emb[:5]}")
        
        # Test single text embedding
        single_embedding = generate_embeddings(["A single piece of text."])
        if single_embedding:
            print("\nSuccessfully generated embedding for a single text.")
            print(f"  - Dimension = {len(single_embedding[0])}")
        else:
            print("\nFailed to generate embedding for a single text.")
    else:
        print("Failed to generate embeddings. Check your model-inference-service.")
