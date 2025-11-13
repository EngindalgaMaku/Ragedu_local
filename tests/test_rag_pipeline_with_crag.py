"""
Integration tests for RAGPipeline with CRAG Evaluator.

Tests the full integration between RAGPipeline and RetrievalEvaluator,
ensuring that irrelevant queries are rejected and relevant queries are processed.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.rag.rag_pipeline import RAGPipeline
from src.vector_store.chroma_store import ChromaVectorStore


class TestRAGPipelineWithCRAGIntegration:
    """Test RAGPipeline with CRAG evaluator integration."""
    
    @pytest.fixture
    def mock_chroma_store(self):
        """Create a mock ChromaDB store."""
        store = Mock(spec=ChromaVectorStore)
        store.chunks = []
        store._document_count = 0
        return store
    
    @pytest.fixture
    def config_with_crag_enabled(self):
        """Configuration with CRAG evaluator enabled."""
        return {
            'enable_cache': False,  # Disable cache for testing
            'enable_retrieval_evaluation': True,
            'retrieval_evaluator_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'retrieval_correct_threshold': 0.7,
            'retrieval_incorrect_threshold': 0.3,
            'retrieval_filter_threshold': 0.5,
            'enable_reranking': False,
            'ollama_embedding_model': 'nomic-embed-text'
        }
    
    @pytest.fixture
    def config_with_crag_disabled(self):
        """Configuration with CRAG evaluator disabled."""
        return {
            'enable_cache': False,
            'enable_retrieval_evaluation': False,
            'enable_reranking': False,
            'ollama_embedding_model': 'nomic-embed-text'
        }
    
    def test_pipeline_initialization_with_crag(self, mock_chroma_store, config_with_crag_enabled):
        """Test that pipeline initializes evaluator when enabled."""
        pipeline = RAGPipeline(config_with_crag_enabled, mock_chroma_store)
        
        # Evaluator should be initialized
        assert pipeline.retrieval_evaluator is not None
        assert pipeline.retrieval_evaluator.loaded is True
    
    def test_pipeline_initialization_without_crag(self, mock_chroma_store, config_with_crag_disabled):
        """Test that pipeline works without evaluator when disabled."""
        pipeline = RAGPipeline(config_with_crag_disabled, mock_chroma_store)
        
        # Evaluator should be None
        assert pipeline.retrieval_evaluator is None
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_relevant_query_acceptance(self, mock_embeddings, mock_chroma_store, config_with_crag_enabled):
        """Test that relevant queries are accepted by CRAG."""
        # Setup
        pipeline = RAGPipeline(config_with_crag_enabled, mock_chroma_store)
        
        # Mock embedding generation
        mock_embeddings.return_value = [[0.1] * 768]
        
        # Mock ChromaDB search - return ML-related documents
        mock_chroma_store.search.return_value = [
            (
                "Machine learning is a branch of artificial intelligence that enables "
                "computers to learn from data without being explicitly programmed.",
                0.95,
                {"source": "ml_textbook.pdf"}
            ),
            (
                "Deep learning uses neural networks with multiple layers to process data.",
                0.88,
                {"source": "dl_guide.pdf"}
            )
        ]
        
        # Execute
        query = "What is machine learning?"
        results = pipeline.retrieve(query, top_k=2)
        
        # Verify
        assert len(results) > 0, "Relevant query should return results"
        assert len(results) <= 2
        assert 'text' in results[0]
        assert 'machine learning' in results[0]['text'].lower()
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_irrelevant_query_rejection(self, mock_embeddings, mock_chroma_store, config_with_crag_enabled):
        """Test that irrelevant queries are rejected by CRAG."""
        # Setup
        pipeline = RAGPipeline(config_with_crag_enabled, mock_chroma_store)
        
        # Mock embedding generation
        mock_embeddings.return_value = [[0.1] * 768]
        
        # Mock ChromaDB search - return ML documents for weather query
        mock_chroma_store.search.return_value = [
            (
                "Machine learning is a branch of artificial intelligence.",
                0.85,
                {"source": "ml_textbook.pdf"}
            ),
            (
                "Neural networks consist of interconnected layers of nodes.",
                0.82,
                {"source": "nn_guide.pdf"}
            )
        ]
        
        # Execute - weather query against ML documents
        query = "What is the weather today?"
        results = pipeline.retrieve(query, top_k=2)
        
        # Verify - should reject and return empty
        assert len(results) == 0, "Irrelevant query should be rejected by CRAG"
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_ambiguous_query_filtering(self, mock_embeddings, mock_chroma_store, config_with_crag_enabled):
        """Test that ambiguous results are filtered by CRAG."""
        # Setup
        pipeline = RAGPipeline(config_with_crag_enabled, mock_chroma_store)
        
        # Mock embedding generation
        mock_embeddings.return_value = [[0.1] * 768]
        
        # Mock ChromaDB search - mix of relevant and irrelevant
        mock_chroma_store.search.return_value = [
            (
                "Neural network architecture refers to the structure of layers.",
                0.92,
                {"source": "ml_textbook.pdf"}
            ),
            (
                "The weather is nice today.",  # Irrelevant
                0.15,
                {"source": "random.txt"}
            ),
            (
                "Convolutional layers extract spatial features from images.",
                0.78,
                {"source": "cnn_guide.pdf"}
            ),
            (
                "I like pizza.",  # Irrelevant
                0.10,
                {"source": "food.txt"}
            )
        ]
        
        # Execute
        query = "What is neural network architecture?"
        results = pipeline.retrieve(query, top_k=2)
        
        # Verify - should filter out irrelevant docs
        assert len(results) > 0, "Should return some results"
        assert len(results) <= 2
        # Results should not contain weather or pizza text
        for result in results:
            text_lower = result['text'].lower()
            assert 'weather' not in text_lower
            assert 'pizza' not in text_lower
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_without_crag_no_filtering(self, mock_embeddings, mock_chroma_store, config_with_crag_disabled):
        """Test that without CRAG, no filtering occurs."""
        # Setup
        pipeline = RAGPipeline(config_with_crag_disabled, mock_chroma_store)
        
        # Mock embedding generation
        mock_embeddings.return_value = [[0.1] * 768]
        
        # Mock ChromaDB search
        mock_chroma_store.search.return_value = [
            ("Document 1", 0.9, {}),
            ("Document 2", 0.8, {})
        ]
        
        # Execute - even irrelevant query should get results
        query = "What is the weather?"
        results = pipeline.retrieve(query, top_k=2)
        
        # Verify - should return results without filtering
        assert len(results) == 2, "Without CRAG, all results should be returned"
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_empty_retrieval(self, mock_embeddings, mock_chroma_store, config_with_crag_enabled):
        """Test handling of empty retrieval results."""
        # Setup
        pipeline = RAGPipeline(config_with_crag_enabled, mock_chroma_store)
        
        # Mock embedding generation
        mock_embeddings.return_value = [[0.1] * 768]
        
        # Mock ChromaDB search - return empty
        mock_chroma_store.search.return_value = []
        
        # Execute
        query = "Some query"
        results = pipeline.retrieve(query, top_k=5)
        
        # Verify
        assert len(results) == 0
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_evaluator_stats_logging(self, mock_embeddings, mock_chroma_store, config_with_crag_enabled):
        """Test that evaluator stats are properly logged."""
        # Setup
        pipeline = RAGPipeline(config_with_crag_enabled, mock_chroma_store)
        
        # Check evaluator stats
        if pipeline.retrieval_evaluator:
            stats = pipeline.retrieval_evaluator.get_stats()
            
            assert stats['loaded'] is True
            assert stats['correct_threshold'] == 0.7
            assert stats['incorrect_threshold'] == 0.3
            assert stats['filter_threshold'] == 0.5
            assert 'cross-encoder' in stats['model_name']


class TestRAGPipelineRealisticScenarios:
    """Test realistic end-to-end scenarios."""
    
    @pytest.fixture
    def mock_chroma_store(self):
        """Create a mock ChromaDB store."""
        store = Mock(spec=ChromaVectorStore)
        store.chunks = []
        store._document_count = 0
        return store
    
    @pytest.fixture
    def config(self):
        """Standard configuration."""
        return {
            'enable_cache': False,
            'enable_retrieval_evaluation': True,
            'retrieval_evaluator_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'retrieval_correct_threshold': 0.7,
            'retrieval_incorrect_threshold': 0.3,
            'retrieval_filter_threshold': 0.5,
            'enable_reranking': False,
            'ollama_embedding_model': 'nomic-embed-text'
        }
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_course_material_query_scenario(self, mock_embeddings, mock_chroma_store, config):
        """Test realistic course material query scenario."""
        pipeline = RAGPipeline(config, mock_chroma_store)
        
        # Mock embedding
        mock_embeddings.return_value = [[0.1] * 768]
        
        # Mock course material documents
        mock_chroma_store.search.return_value = [
            (
                "Backpropagation is an algorithm used to train neural networks by "
                "calculating gradients using the chain rule.",
                0.95,
                {"source": "lecture_5.pdf", "chapter": "Neural Networks"}
            ),
            (
                "The gradient descent algorithm updates weights iteratively to minimize loss.",
                0.88,
                {"source": "lecture_5.pdf", "chapter": "Optimization"}
            ),
            (
                "Learning rate is a hyperparameter that controls the step size in gradient descent.",
                0.85,
                {"source": "lecture_6.pdf", "chapter": "Hyperparameters"}
            )
        ]
        
        # Execute - student asking about course material
        query = "Explain how backpropagation works in neural networks"
        results = pipeline.retrieve(query, top_k=3)
        
        # Verify
        assert len(results) >= 1, "Should return relevant course material"
        assert any('backpropagation' in r['text'].lower() for r in results)
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_off_topic_student_question(self, mock_embeddings, mock_chroma_store, config):
        """Test off-topic question from student."""
        pipeline = RAGPipeline(config, mock_chroma_store)
        
        # Mock embedding
        mock_embeddings.return_value = [[0.1] * 768]
        
        # Mock course material documents (ML/AI content)
        mock_chroma_store.search.return_value = [
            ("Neural networks are computational models...", 0.80, {}),
            ("Machine learning algorithms learn from data...", 0.85, {}),
            ("Deep learning has revolutionized AI...", 0.82, {})
        ]
        
        # Execute - completely off-topic question
        query = "What time does the cafeteria open?"
        results = pipeline.retrieve(query, top_k=3)
        
        # Verify - should be rejected
        assert len(results) == 0, "Off-topic question should be rejected"
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_threshold_tuning_impact(self, mock_embeddings, mock_chroma_store):
        """Test impact of different threshold settings."""
        # Mock embedding
        mock_embeddings.return_value = [[0.1] * 768]
        
        # Mock borderline relevance documents
        mock_chroma_store.search.return_value = [
            ("Machine learning applications in industry", 0.60, {}),
            ("Data preprocessing techniques", 0.55, {}),
            ("Feature engineering methods", 0.58, {})
        ]
        
        # Test with strict threshold
        strict_config = {
            'enable_cache': False,
            'enable_retrieval_evaluation': True,
            'retrieval_correct_threshold': 0.8,
            'retrieval_incorrect_threshold': 0.4,
            'retrieval_filter_threshold': 0.6,
            'enable_reranking': False
        }
        
        pipeline_strict = RAGPipeline(strict_config, mock_chroma_store)
        query = "Machine learning use cases"
        results_strict = pipeline_strict.retrieve(query, top_k=3)
        
        # Test with lenient threshold
        lenient_config = {
            'enable_cache': False,
            'enable_retrieval_evaluation': True,
            'retrieval_correct_threshold': 0.6,
            'retrieval_incorrect_threshold': 0.2,
            'retrieval_filter_threshold': 0.4,
            'enable_reranking': False
        }
        
        pipeline_lenient = RAGPipeline(lenient_config, mock_chroma_store)
        results_lenient = pipeline_lenient.retrieve(query, top_k=3)
        
        # Lenient should return more results than strict
        assert len(results_lenient) >= len(results_strict)


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
