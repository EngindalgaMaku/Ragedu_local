"""
Re-ranking module for improving RAG system performance.

This module implements a Cross-Encoder based re-ranking system that improves
the quality of retrieved documents by providing more accurate relevance scores.
"""

import logging
from typing import List, Tuple, Dict, Any
from sentence_transformers import CrossEncoder
from ..utils.helpers import setup_logging

logger = setup_logging()


class ReRanker:
    """
    A re-ranking system that uses Cross-Encoder models to improve document retrieval quality.
    
    This class implements a two-stage retrieval approach:
    1. Initial retrieval gets more documents than needed
    2. Re-ranking filters and ranks these documents for better relevance
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the ReRanker with a Cross-Encoder model.
        
        Args:
            model_name: Name of the Cross-Encoder model to use for re-ranking.
                       Default is a more robust multilingual model for better Turkish support.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Cross-Encoder model."""
        try:
            logger.info(f"Loading Cross-Encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-Encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on their relevance to the query.
        
        Args:
            query: The user query string
            documents: List of document dictionaries, each with 'text', 'score', 'metadata'
            top_k: Number of top documents to return after re-ranking
            
        Returns:
            List of re-ranked document dictionaries with updated scores, limited to top_k results
        """
        if not self.model:
            logger.warning("Cross-Encoder model not available, returning original ranking")
            return documents[:top_k]
        
        if not documents:
            logger.warning("No documents provided for re-ranking")
            return []
        
        try:
            logger.info(f"Re-ranking {len(documents)} documents for query: '{query[:50]}...'")
            
            # Prepare query-document pairs for Cross-Encoder
            doc_texts = [doc.get('text', '') for doc in documents]
            query_doc_pairs = [[query, doc_text] for doc_text in doc_texts]
            
            # Get relevance scores from Cross-Encoder
            relevance_scores = self.model.predict(query_doc_pairs)
            
            # Add new scores to documents
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(relevance_scores[i])
            
            # Sort by new relevance scores (descending)
            reranked_docs = sorted(documents, key=lambda x: x.get('rerank_score', -1.0), reverse=True)
            
            # Return top_k results
            result = reranked_docs[:top_k]
            
            logger.info(f"Re-ranking completed. Returning top {len(result)} documents")
            logger.debug(f"Re-ranked scores: {[doc.get('rerank_score', 0.0) for doc in result[:3]]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            # Fallback to original ranking
            return documents[:top_k]
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "description": "Cross-Encoder model for document re-ranking",
            "purpose": "Improve retrieval quality by providing accurate relevance scores"
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the ReRanker
    reranker = ReRanker()
    
    # Sample documents (dictionaries)
    test_documents = [
        {"text": "The capital of France is Paris.", "score": 0.8, "metadata": {"source": "geography.txt"}},
        {"text": "Python is a programming language.", "score": 0.6, "metadata": {"source": "programming.txt"}},
        {"text": "Paris is known for the Eiffel Tower.", "score": 0.7, "metadata": {"source": "tourism.txt"}},
        {"text": "Machine learning uses algorithms.", "score": 0.5, "metadata": {"source": "ai.txt"}}
    ]
    
    test_query = "What is the capital of France?"
    
    print("Original documents:")
    for i, doc in enumerate(test_documents):
        print(f"{i+1}. {doc['text']} (score: {doc['score']:.2f})")
    
    print(f"\nRe-ranking for query: '{test_query}'")
    
    reranked = reranker.rerank(test_query, test_documents, top_k=3)
    
    print("\nRe-ranked documents:")
    for i, doc in enumerate(reranked):
        print(f"{i+1}. {doc['text']} (new score: {doc.get('rerank_score', 0.0):.4f})")
    
    print(f"\nModel info: {reranker.get_model_info()}")