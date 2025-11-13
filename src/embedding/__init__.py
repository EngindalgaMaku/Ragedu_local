# This file makes the 'embedding' directory a Python package.

try:
    from .embedding_generator import generate_embeddings
    __all__ = ["generate_embeddings"]
except ImportError as e:
    print(f"Warning: Could not import embedding_generator: {e}")
    def generate_embeddings(*args, **kwargs):
        return []
    __all__ = ["generate_embeddings"]