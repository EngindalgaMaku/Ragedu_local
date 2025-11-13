"""
Unit tests for Retrieval Evaluator (CRAG).

Tests the functionality of the RetrievalEvaluator class including:
- Initialization and model loading
- Evaluation of relevant queries
- Evaluation of irrelevant queries
- Threshold-based filtering
- Edge cases and error handling
"""

import pytest
import numpy as np
from src.rag.retrieval_evaluator import RetrievalEvaluator


class TestRetrievalEvaluatorInitialization:
    """Test evaluator initialization and configuration."""
    
    def test_default_initialization(self):
        """Test evaluator can be initialized with default parameters."""
        evaluator = RetrievalEvaluator()
        
        assert evaluator.loaded is True
        assert evaluator.model is not None
        assert evaluator.correct_threshold == 0.7
        assert evaluator.incorrect_threshold == 0.3
        assert evaluator.filter_threshold == 0.5
    
    def test_custom_thresholds(self):
        """Test evaluator with custom threshold values."""
        evaluator = RetrievalEvaluator(
            correct_threshold=0.8,
            incorrect_threshold=0.2,
            filter_threshold=0.6
        )
        
        assert evaluator.correct_threshold == 0.8
        assert evaluator.incorrect_threshold == 0.2
        assert evaluator.filter_threshold == 0.6
    
    def test_different_models(self):
        """Test initialization with different cross-encoder models."""
        # TinyBERT (fastest)
        evaluator_tiny = RetrievalEvaluator(
            model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'
        )
        assert evaluator_tiny.loaded is True
        
        # MiniLM-L-6 (default, balanced)
        evaluator_mini = RetrievalEvaluator(
            model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        assert evaluator_mini.loaded is True
    
    def test_get_stats(self):
        """Test get_stats returns correct configuration."""
        evaluator = RetrievalEvaluator(
            correct_threshold=0.75,
            incorrect_threshold=0.25
        )
        
        stats = evaluator.get_stats()
        
        assert stats['loaded'] is True
        assert stats['correct_threshold'] == 0.75
        assert stats['incorrect_threshold'] == 0.25
        assert 'model_name' in stats


class TestRetrievalEvaluationRelevant:
    """Test evaluation with relevant queries."""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture providing a configured evaluator."""
        return RetrievalEvaluator()
    
    def test_relevant_ml_query(self, evaluator):
        """Test with a relevant machine learning query."""
        query = "What is machine learning?"
        docs = [
            (
                "Machine learning is a branch of artificial intelligence that "
                "enables computers to learn from data without being explicitly programmed.",
                0.95,
                {"source": "ml_textbook.pdf"}
            ),
            (
                "Deep learning is a subset of machine learning that uses neural networks "
                "with multiple layers to learn hierarchical representations.",
                0.88,
                {"source": "dl_guide.pdf"}
            ),
            (
                "Supervised learning is a type of machine learning where models are "
                "trained on labeled data to make predictions.",
                0.85,
                {"source": "ml_basics.pdf"}
            )
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        # Should be accepted
        assert result['action'] in ['accept', 'refine']
        assert result['confidence'] in ['correct', 'ambiguous']
        assert result['avg_score'] > 0.5
        assert result['max_score'] > 0.7
        assert len(result['scores']) == 3
    
    def test_relevant_gradient_descent_query(self, evaluator):
        """Test with a relevant gradient descent query."""
        query = "How does gradient descent work?"
        docs = [
            (
                "Gradient descent is an optimization algorithm used to minimize the cost "
                "function by iteratively moving in the direction of steepest descent.",
                0.92,
                {}
            ),
            (
                "The learning rate determines the step size in gradient descent, "
                "controlling how quickly the algorithm converges.",
                0.87,
                {}
            )
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        assert result['action'] != 'reject'
        assert result['avg_score'] > 0.6
        assert result['max_score'] > 0.7
    
    def test_high_quality_results(self, evaluator):
        """Test with very high quality, highly relevant documents."""
        query = "What is backpropagation?"
        docs = [
            (
                "Backpropagation is the algorithm used to calculate gradients in neural "
                "networks by propagating errors backward through the network layers.",
                0.98,
                {}
            )
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        assert result['action'] == 'accept'
        assert result['confidence'] == 'correct'
        assert result['avg_score'] >= evaluator.correct_threshold


class TestRetrievalEvaluationIrrelevant:
    """Test evaluation with irrelevant queries."""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture providing a configured evaluator."""
        return RetrievalEvaluator()
    
    def test_weather_query_with_ml_docs(self, evaluator):
        """Test weather query against ML documents (should reject)."""
        query = "What is the weather today?"
        docs = [
            ("Machine learning is a branch of AI...", 0.85, {}),
            ("Neural networks consist of layers of nodes...", 0.82, {}),
            ("Deep learning uses multiple hidden layers...", 0.80, {})
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        # Should be rejected due to low relevance
        assert result['action'] == 'reject'
        assert result['confidence'] == 'incorrect'
        assert result['max_score'] < evaluator.incorrect_threshold
    
    def test_cooking_query_with_ml_docs(self, evaluator):
        """Test cooking query against ML documents (should reject)."""
        query = "How do I bake a cake?"
        docs = [
            ("Gradient descent optimizes the loss function...", 0.90, {}),
            ("Convolutional neural networks process images...", 0.88, {})
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        assert result['action'] == 'reject'
        assert result['max_score'] < 0.4
    
    def test_sports_query_with_tech_docs(self, evaluator):
        """Test sports query against technical documents."""
        query = "Who won the football match?"
        docs = [
            ("Python is a programming language used for data science...", 0.75, {}),
            ("TensorFlow is a framework for machine learning...", 0.73, {})
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        assert result['action'] == 'reject'
        assert result['confidence'] == 'incorrect'


class TestRetrievalEvaluationAmbiguous:
    """Test evaluation with ambiguous/mixed quality results."""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture providing a configured evaluator."""
        return RetrievalEvaluator()
    
    def test_mixed_quality_results(self, evaluator):
        """Test with mix of relevant and irrelevant documents."""
        query = "What is neural network architecture?"
        docs = [
            (
                "Neural network architecture refers to the structure and organization "
                "of layers, nodes, and connections in a neural network.",
                0.92,
                {}
            ),
            (
                "The weather is nice today with sunny skies.",
                0.15,
                {}
            ),
            (
                "Convolutional layers extract spatial features from images.",
                0.78,
                {}
            )
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        # Should trigger refinement due to mixed scores
        if result['action'] == 'refine':
            assert result['confidence'] == 'ambiguous'
        
        # Max score should be good, but avg score pulled down
        assert result['max_score'] > 0.7
        assert result['avg_score'] < result['max_score']
    
    def test_borderline_relevance(self, evaluator):
        """Test with borderline relevance scores."""
        query = "Machine learning applications"
        docs = [
            ("Machine learning has many applications in industry.", 0.65, {}),
            ("Data preprocessing is important for model training.", 0.55, {}),
            ("Feature engineering improves model performance.", 0.60, {})
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        # Could be accept or refine depending on exact scores
        assert result['action'] in ['accept', 'refine']
        assert 0.4 < result['avg_score'] < 0.8


class TestThresholdFiltering:
    """Test threshold-based document filtering."""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture providing a configured evaluator."""
        return RetrievalEvaluator()
    
    def test_filter_removes_low_scores(self, evaluator):
        """Test that filtering removes low-scoring documents."""
        query = "What is supervised learning?"
        docs = [
            ("Supervised learning uses labeled training data.", 0.95, {}),
            ("The weather is cloudy today.", 0.20, {}),
            ("Classification is a type of supervised learning.", 0.88, {}),
            ("I like pizza.", 0.10, {})
        ]
        
        filtered = evaluator.filter_by_threshold(query, docs, threshold=0.5)
        
        # Should keep only 2 relevant docs
        assert len(filtered) < len(docs)
        assert len(filtered) >= 2  # At least the two relevant ones
        
        # Check evaluator_score added to metadata
        for doc_text, score, metadata in filtered:
            assert 'evaluator_score' in metadata
            assert metadata['evaluator_score'] >= 0.5
    
    def test_filter_with_custom_threshold(self, evaluator):
        """Test filtering with different threshold values."""
        query = "Neural networks"
        docs = [
            ("Neural networks are computational models...", 0.90, {}),
            ("Backpropagation trains neural networks...", 0.80, {}),
            ("Activation functions introduce non-linearity...", 0.70, {}),
            ("Random unrelated text here.", 0.30, {})
        ]
        
        # Low threshold - keep more
        filtered_low = evaluator.filter_by_threshold(query, docs, threshold=0.4)
        assert len(filtered_low) >= 3
        
        # High threshold - keep fewer
        filtered_high = evaluator.filter_by_threshold(query, docs, threshold=0.75)
        assert len(filtered_high) <= 2
        
        # filtered_high should be subset of filtered_low
        assert len(filtered_high) <= len(filtered_low)
    
    def test_filter_keeps_all_if_relevant(self, evaluator):
        """Test that all documents kept if all are relevant."""
        query = "What is deep learning?"
        docs = [
            ("Deep learning uses neural networks with many layers.", 0.95, {}),
            ("Deep learning excels at image recognition tasks.", 0.92, {}),
            ("Convolutional networks are used in deep learning.", 0.90, {})
        ]
        
        filtered = evaluator.filter_by_threshold(query, docs, threshold=0.7)
        
        # All should pass the threshold
        assert len(filtered) == len(docs)
    
    def test_filter_metadata_preservation(self, evaluator):
        """Test that original metadata is preserved during filtering."""
        query = "Machine learning"
        original_metadata = {"source": "textbook.pdf", "page": 42}
        docs = [
            ("Machine learning content...", 0.85, original_metadata.copy())
        ]
        
        filtered = evaluator.filter_by_threshold(query, docs, threshold=0.5)
        
        assert len(filtered) == 1
        _, _, metadata = filtered[0]
        assert metadata['source'] == "textbook.pdf"
        assert metadata['page'] == 42
        assert 'evaluator_score' in metadata


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture providing a configured evaluator."""
        return RetrievalEvaluator()
    
    def test_empty_document_list(self, evaluator):
        """Test with empty document list."""
        query = "What is AI?"
        docs = []
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        assert result['action'] == 'reject'
        assert result['avg_score'] == 0.0
        assert len(result['scores']) == 0
    
    def test_empty_query(self, evaluator):
        """Test with empty query string."""
        query = ""
        docs = [("Some content here", 0.8, {})]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        # Should still process without error
        assert 'action' in result
        assert 'scores' in result
    
    def test_single_document(self, evaluator):
        """Test with single document."""
        query = "What is reinforcement learning?"
        docs = [
            ("Reinforcement learning is learning through trial and error.", 0.90, {})
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        assert len(result['scores']) == 1
        assert result['avg_score'] == result['max_score'] == result['min_score']
    
    def test_very_long_document(self, evaluator):
        """Test with very long document text."""
        query = "What is machine learning?"
        long_text = "Machine learning " * 1000  # Very long document
        docs = [(long_text, 0.85, {})]
        
        # Should handle without error
        result = evaluator.evaluate_retrieval(query, docs)
        assert 'action' in result
    
    def test_special_characters_in_query(self, evaluator):
        """Test with special characters in query."""
        query = "What is C++? #Python @AI & ML!"
        docs = [("C++ is a programming language.", 0.80, {})]
        
        result = evaluator.evaluate_retrieval(query, docs)
        assert 'action' in result
    
    def test_unicode_characters(self, evaluator):
        """Test with unicode characters."""
        query = "Makine öğrenmesi nedir?"  # Turkish
        docs = [("Machine learning is AI...", 0.75, {})]
        
        result = evaluator.evaluate_retrieval(query, docs)
        assert 'action' in result


class TestPerformance:
    """Test performance and scalability."""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture providing a configured evaluator."""
        return RetrievalEvaluator()
    
    def test_many_documents(self, evaluator):
        """Test evaluation with many documents."""
        query = "What is machine learning?"
        
        # Create 50 documents
        docs = []
        for i in range(50):
            relevance = 0.5 + (i % 3) * 0.1  # Varying relevance
            docs.append(
                (f"Document {i} about machine learning concepts.", relevance, {})
            )
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        assert len(result['scores']) == 50
        assert result['avg_score'] > 0
    
    def test_batch_filtering_performance(self, evaluator):
        """Test filtering performance with large document set."""
        query = "Neural networks"
        
        docs = []
        for i in range(100):
            docs.append((f"Document {i} content", 0.5 + i/200, {}))
        
        filtered = evaluator.filter_by_threshold(query, docs, threshold=0.7)
        
        # Should filter out low-scoring docs
        assert len(filtered) < len(docs)


class TestIntegrationScenarios:
    """Test realistic end-to-end scenarios."""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture providing a configured evaluator."""
        return RetrievalEvaluator()
    
    def test_course_material_query_scenario(self, evaluator):
        """Test realistic course material query."""
        query = "Explain backpropagation algorithm in neural networks"
        docs = [
            (
                "Backpropagation is a supervised learning algorithm for training "
                "neural networks. It works by calculating gradients of the loss "
                "function with respect to weights using the chain rule.",
                0.95,
                {"source": "lecture_notes.pdf", "chapter": 5}
            ),
            (
                "The chain rule from calculus is essential for backpropagation, "
                "allowing us to compute derivatives of composed functions.",
                0.82,
                {"source": "math_review.pdf", "page": 12}
            ),
            (
                "Gradient descent uses the gradients computed by backpropagation "
                "to update neural network weights.",
                0.88,
                {"source": "lecture_notes.pdf", "chapter": 5}
            )
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        # Should accept these highly relevant documents
        assert result['action'] == 'accept'
        assert result['confidence'] == 'correct'
        assert result['avg_score'] > 0.8
    
    def test_off_topic_query_rejection(self, evaluator):
        """Test rejection of completely off-topic query."""
        query = "What time does the cafeteria open?"
        docs = [
            ("Neural networks consist of interconnected layers...", 0.80, {}),
            ("Machine learning algorithms learn from data...", 0.85, {}),
            ("Deep learning has revolutionized computer vision...", 0.82, {})
        ]
        
        result = evaluator.evaluate_retrieval(query, docs)
        
        # Should reject as completely irrelevant
        assert result['action'] == 'reject'
        assert result['confidence'] == 'incorrect'
        assert result['max_score'] < 0.4


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
