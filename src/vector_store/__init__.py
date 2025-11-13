# This file makes the 'vector_store' directory a Python package.

from .faiss_store import FaissVectorStore

__all__ = ["FaissVectorStore"]