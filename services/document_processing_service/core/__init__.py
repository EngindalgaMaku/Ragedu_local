"""
Core components for Document Processing Service
"""
from .chromadb_client import get_chroma_client
from .embedding_service import get_embeddings_direct
from .turkish_utils import TURKISH_STOPWORDS, tokenize_turkish

__all__ = [
    'get_chroma_client',
    'get_embeddings_direct',
    'TURKISH_STOPWORDS',
    'tokenize_turkish'
]






