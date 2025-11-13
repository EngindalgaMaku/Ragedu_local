"""
Retrieval Quality Evaluator for CRAG (Corrective RAG).

This module implements a lightweight evaluator that assesses the quality
of retrieved documents using a cross-encoder model. It helps filter out
irrelevant documents and reject queries that are not related to the
document corpus.

Based on: Corrective Retrieval Augmented Generation (CRAG)
Paper: https://arxiv.org/abs/2401.15884
"""

from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    CRAG-style retrieval quality evaluator.
    
    Uses a lightweight CrossEncoder to quickly assess the relevance of
    retrieved documents to a query. Provides three confidence levels:
    - correct: High confidence, accept results
    - incorrect: Low confidence, reject query
    - ambiguous: Mixed results, filter and refine
    
    Example:
        >>> evaluator = RetrievalEvaluator()
        >>> query = "What is machine learning?"
        >>> docs = [("ML is a branch of AI...", 0.9, {})]
        >>> result = evaluator.evaluate_retrieval(query, docs)
        >>> print(result['action'])  # 'accept', 'reject', or 'refine'
    """
    
    def __init__(
        self, 
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        correct_threshold: float = 0.7,
        incorrect_threshold: float = 0.3,
        filter_threshold: float = 0.5
    ):
        """
        Initialize the Retrieval Evaluator.
        
        Args:
            model_name: CrossEncoder model to use. Options:
                - 'cross-encoder/ms-marco-TinyBERT-L-2-v2' (fastest, 4MB)
                - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (balanced, 80MB) [RECOMMENDED]
                - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (most accurate, 130MB)
            
            correct_threshold: Avg score above this → 'correct' confidence
            incorrect_threshold: Max score below this → 'incorrect' confidence
            filter_threshold: Filter documents with score below this
        """
        self.model_name = model_name
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold
        self.filter_threshold = filter_threshold
        
        try:
            logger.info(f"Loading retrieval evaluator model: {model_name}")
            self.model = CrossEncoder(model_name)
            self.loaded = True
            logger.info("✅ Retrieval evaluator loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load evaluator model: {e}")
            logger.warning("Retrieval evaluation will be disabled")
            self.model = None
            self.loaded = False
    
    def evaluate_retrieval(
        self, 
        query: str, 
        retrieved_docs: List[Tuple[str, float, Dict]]
    ) -> Dict:
        """
        Evaluate the quality of retrieved documents for a query.
        
        This method scores each document using a cross-encoder and determines
        whether the retrieval quality is sufficient (correct), insufficient
        (incorrect), or mixed (ambiguous).
        
        Args:
            query: User's query string
            retrieved_docs: List of tuples (text, similarity_score, metadata)
                          from the vector store
        
        Returns:
            Dictionary with evaluation results:
            {
                'confidence': 'correct' | 'incorrect' | 'ambiguous',
                'action': 'accept' | 'reject' | 'refine',
                'scores': List[float],  # Cross-encoder scores
                'avg_score': float,     # Average score
                'max_score': float,     # Maximum score
                'min_score': float,     # Minimum score
                'message': str          # Human-readable message
            }
        
        Example:
            >>> result = evaluator.evaluate_retrieval(
            ...     "What is gradient descent?",
            ...     [("Gradient descent is...", 0.9, {})]
            ... )
            >>> if result['action'] == 'reject':
            ...     print("Query not related to documents")
        """
        # Handle evaluator not loaded
        if not self.loaded:
            logger.warning("Evaluator not loaded, accepting all results")
            return {
                'confidence': 'correct',
                'action': 'accept',
                'scores': [],
                'avg_score': 1.0,
                'max_score': 1.0,
                'min_score': 1.0,
                'message': 'Evaluator unavailable, bypassing quality check'
            }
        
        # Handle empty retrieval
        if not retrieved_docs:
            return {
                'confidence': 'incorrect',
                'action': 'reject',
                'scores': [],
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'message': 'No documents retrieved'
            }
        
        # Prepare query-document pairs for cross-encoder
        pairs = [(query, doc[0]) for doc in retrieved_docs]
        
        # Score all pairs
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Error during cross-encoder scoring: {e}")
            # Fallback to accepting if scoring fails
            return {
                'confidence': 'correct',
                'action': 'accept',
                'scores': [],
                'avg_score': 1.0,
                'max_score': 1.0,
                'min_score': 1.0,
                'message': f'Scoring failed: {e}, bypassing check'
            }
        
        # Calculate statistics
        avg_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))
        
        # Decision logic based on thresholds
        if max_score < self.incorrect_threshold:
            # Even the best document is not relevant
            confidence = 'incorrect'
            action = 'reject'
            message = (
                f'Soru ders materyalleriyle ilgili görünmüyor. '
                f'(max_score={max_score:.2f} < {self.incorrect_threshold})'
            )
        
        elif avg_score >= self.correct_threshold:
            # Most documents are relevant
            confidence = 'correct'
            action = 'accept'
            message = (
                f'Yüksek kaliteli sonuçlar bulundu '
                f'(avg_score={avg_score:.2f} ≥ {self.correct_threshold})'
            )
        
        else:
            # Mixed quality - some relevant, some not
            confidence = 'ambiguous'
            action = 'refine'
            message = (
                f'Sonuçlar belirsiz (avg_score={avg_score:.2f}), '
                f'filtreleme yapılıyor'
            )
        
        result = {
            'confidence': confidence,
            'action': action,
            'scores': scores.tolist(),
            'avg_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'message': message
        }
        
        logger.info(
            f"📊 Evaluation: {confidence.upper()} "
            f"(avg={avg_score:.3f}, max={max_score:.3f}) → {action.upper()}"
        )
        
        return result
    
    def filter_by_threshold(
        self,
        query: str,
        retrieved_docs: List[Tuple[str, float, Dict]],
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Filter retrieved documents by relevance threshold.
        
        Documents with cross-encoder score below the threshold are removed.
        This is useful for the 'ambiguous' case where we want to keep only
        the most relevant documents.
        
        Args:
            query: User's query string
            retrieved_docs: List of tuples (text, similarity_score, metadata)
            threshold: Minimum score to keep (None = use self.filter_threshold)
        
        Returns:
            Filtered list of documents (same format as input)
            Each document's metadata will include 'evaluator_score'
        
        Example:
            >>> filtered = evaluator.filter_by_threshold(
            ...     "What is ML?",
            ...     docs,
            ...     threshold=0.6
            ... )
            >>> print(f"Kept {len(filtered)}/{len(docs)} documents")
        """
        if not self.loaded or not retrieved_docs:
            return retrieved_docs
        
        threshold = threshold if threshold is not None else self.filter_threshold
        
        # Score all documents
        pairs = [(query, doc[0]) for doc in retrieved_docs]
        
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Error during filtering: {e}")
            return retrieved_docs  # Return all if scoring fails
        
        # Filter and add scores to metadata
        filtered = []
        for i, score in enumerate(scores):
            if score >= threshold:
                doc_text, orig_score, metadata = retrieved_docs[i]
                
                # Add evaluator score to metadata
                metadata = metadata.copy()  # Don't modify original
                metadata['evaluator_score'] = float(score)
                
                filtered.append((doc_text, orig_score, metadata))
        
        logger.info(
            f"🔍 Filtered {len(filtered)}/{len(retrieved_docs)} documents "
            f"(threshold={threshold:.2f})"
        )
        
        return filtered
    
    def get_stats(self) -> Dict:
        """
        Get evaluator statistics and configuration.
        
        Returns:
            Dictionary with evaluator info:
            {
                'loaded': bool,
                'model_name': str,
                'correct_threshold': float,
                'incorrect_threshold': float,
                'filter_threshold': float
            }
        """
        return {
            'loaded': self.loaded,
            'model_name': self.model_name,
            'correct_threshold': self.correct_threshold,
            'incorrect_threshold': self.incorrect_threshold,
            'filter_threshold': self.filter_threshold
        }


# Convenience function for quick testing
def test_evaluator():
    """
    Quick test function to verify evaluator is working.
    
    Usage:
        python -c "from src.rag.retrieval_evaluator import test_evaluator; test_evaluator()"
    """
    print("🧪 Testing Retrieval Evaluator...")
    
    evaluator = RetrievalEvaluator()
    
    if not evaluator.loaded:
        print("❌ Evaluator failed to load")
        return
    
    print("✅ Evaluator loaded successfully")
    
    # Test 1: Relevant query
    print("\n📝 Test 1: Relevant query")
    query1 = "What is machine learning?"
    docs1 = [
        ("Machine learning is a branch of artificial intelligence...", 0.9, {}),
        ("Deep learning is a subset of machine learning...", 0.85, {})
    ]
    result1 = evaluator.evaluate_retrieval(query1, docs1)
    print(f"   Result: {result1['action']} (score={result1['avg_score']:.2f})")
    
    # Test 2: Irrelevant query
    print("\n📝 Test 2: Irrelevant query")
    query2 = "What is the weather today?"
    docs2 = [
        ("Machine learning is a branch of artificial intelligence...", 0.9, {}),
        ("Deep learning is a subset of machine learning...", 0.85, {})
    ]
    result2 = evaluator.evaluate_retrieval(query2, docs2)
    print(f"   Result: {result2['action']} (score={result2['avg_score']:.2f})")
    
    print("\n✅ Tests completed")
    print(f"\n📊 Stats: {evaluator.get_stats()}")


if __name__ == "__main__":
    test_evaluator()
