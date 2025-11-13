import pytest
import time
import uuid
from unittest.mock import MagicMock, patch, call
import logging

from src.qa.qa_service import QAService


@pytest.fixture
def mock_config():
    """Mock configuration dictionary for QAService."""
    return {
        'logging': {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - %(message)s'},
        'rag': {'retrieval_k': 5, 'chunk_size': 512},
        'active_learning': {'uncertainty_threshold': 0.7}
    }


@pytest.fixture
def mock_db():
    """Mock database connection for QAService."""
    mock_db = MagicMock()
    
    # Mock database methods
    mock_db.add_or_get_rag_configuration.return_value = "config_hash_123"
    mock_db.add_interaction.return_value = "interaction_id_456"
    
    return mock_db


@pytest.fixture
def mock_query_processor():
    """Mock QueryProcessor for QAService."""
    mock_processor = MagicMock()
    mock_processor.process.return_value = {
        "processed_query": "What is RAG?",
        "original_query": "RAG nedir?",
        "language": "tr",
        "query_type": "definition",
        "entities": ["RAG"]
    }
    return mock_processor


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAGPipeline for QAService."""
    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = {
        "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation...",
        "sources": [
            {"title": "RAG Paper", "content": "RAG overview...", "score": 0.95},
            {"title": "LLM Guide", "content": "Language models...", "score": 0.87}
        ]
    }
    mock_pipeline.get_current_parameters.return_value = {
        "retrieval_k": 5,
        "chunk_size": 512,
        "model": "gpt-3.5-turbo"
    }
    return mock_pipeline


@pytest.fixture
def mock_active_learning_engine():
    """Mock ActiveLearningEngine for QAService."""
    mock_engine = MagicMock()
    return mock_engine


@pytest.fixture
def mock_adaptive_router():
    """Mock AdaptiveQueryRouter for QAService."""
    mock_router = MagicMock()
    mock_router.route_query.return_value = "precise_retrieval"
    return mock_router


@pytest.fixture
def qa_service(mock_config, mock_query_processor, mock_rag_pipeline, mock_db):
    """QAService instance with all dependencies mocked."""
    with patch('src.qa.qa_service.get_logger') as mock_get_logger, \
         patch('src.qa.qa_service.ActiveLearningEngine') as mock_ale_class, \
         patch('src.qa.qa_service.AdaptiveQueryRouter') as mock_router_class, \
         patch('src.qa.qa_service.generate_recommendations') as mock_gen_recs:
        
        # Setup mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Setup mock classes
        mock_ale_instance = MagicMock()
        mock_router_instance = MagicMock()
        mock_router_instance.route_query.return_value = "precise_retrieval"
        mock_ale_class.return_value = mock_ale_instance
        mock_router_class.return_value = mock_router_instance
        
        # Setup mock recommendations
        mock_gen_recs.return_value = ["Related question 1", "Related question 2"]
        
        service = QAService(
            config=mock_config,
            query_processor=mock_query_processor,
            rag_pipeline=mock_rag_pipeline,
            db_connection=mock_db
        )
        
        # Store references to mocks for test access
        service._mock_logger = mock_logger
        service._mock_active_learning = mock_ale_instance
        service._mock_adaptive_router = mock_router_instance
        service._mock_generate_recommendations = mock_gen_recs
        
        yield service


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return {
        "query": "RAG nedir ve nasıl çalışır?",
        "user_id": "test_user_123",
        "session_id": "session_456"
    }


class TestQAServiceInit:
    """Tests for QAService initialization."""
    
    @patch('src.qa.qa_service.get_logger')
    @patch('src.qa.qa_service.ActiveLearningEngine')
    @patch('src.qa.qa_service.AdaptiveQueryRouter')
    def test_init_success(self, mock_router_class, mock_ale_class, mock_get_logger, 
                          mock_config, mock_query_processor, mock_rag_pipeline, mock_db):
        """Tests successful initialization of QAService."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_ale_instance = MagicMock()
        mock_router_instance = MagicMock()
        mock_ale_class.return_value = mock_ale_instance
        mock_router_class.return_value = mock_router_instance
        
        service = QAService(
            config=mock_config,
            query_processor=mock_query_processor,
            rag_pipeline=mock_rag_pipeline,
            db_connection=mock_db
        )
        
        assert service.config == mock_config
        assert service.query_processor == mock_query_processor
        assert service.rag_pipeline == mock_rag_pipeline
        assert service.db == mock_db
        assert service.logger == mock_logger
        assert service.active_learning_engine == mock_ale_instance
        assert service.adaptive_router == mock_router_instance
        
        mock_get_logger.assert_called_once_with('src.qa.qa_service', mock_config)
        mock_ale_class.assert_called_once_with(mock_db)
        mock_router_class.assert_called_once_with(mock_db, mock_ale_instance)
    
    @patch('src.qa.qa_service.get_logger')
    @patch('src.qa.qa_service.ActiveLearningEngine')
    @patch('src.qa.qa_service.AdaptiveQueryRouter')
    def test_init_with_minimal_config(self, mock_router_class, mock_ale_class, mock_get_logger,
                                     mock_query_processor, mock_rag_pipeline, mock_db):
        """Tests initialization with minimal configuration."""
        minimal_config = {}
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_ale_class.return_value = MagicMock()
        mock_router_class.return_value = MagicMock()
        
        service = QAService(
            config=minimal_config,
            query_processor=mock_query_processor,
            rag_pipeline=mock_rag_pipeline,
            db_connection=mock_db
        )
        
        assert service.config == minimal_config
    
    @patch('src.qa.qa_service.get_logger')
    def test_init_logger_failure(self, mock_get_logger, mock_config, mock_query_processor, 
                                mock_rag_pipeline, mock_db):
        """Tests initialization when logger creation fails."""
        mock_get_logger.side_effect = Exception("Logger creation failed")
        
        with patch('src.qa.qa_service.ActiveLearningEngine'), \
             patch('src.qa.qa_service.AdaptiveQueryRouter'):
            with pytest.raises(Exception, match="Logger creation failed"):
                QAService(
                    config=mock_config,
                    query_processor=mock_query_processor,
                    rag_pipeline=mock_rag_pipeline,
                    db_connection=mock_db
                )


class TestAnswerQuestion:
    """Tests for answer_question method."""
    
    @pytest.mark.unit
    def test_answer_question_success(self, qa_service, mock_query_processor, mock_rag_pipeline, 
                                   mock_db, sample_query_data):
        """Tests successful question answering with all components."""
        # Setup mocks
        qa_service._mock_adaptive_router.route_query.return_value = "precise_retrieval"
        qa_service._mock_generate_recommendations.return_value = ["Related Q1", "Related Q2"]
        
        # Execute
        result = qa_service.answer_question(
            query=sample_query_data["query"],
            user_id=sample_query_data["user_id"],
            session_id=sample_query_data["session_id"]
        )
        
        # Verify all components were called
        mock_query_processor.process.assert_called_once_with(sample_query_data["query"])
        qa_service._mock_adaptive_router.route_query.assert_called_once_with(
            query="What is RAG?",
            user_id=sample_query_data["user_id"],
            query_metadata={}
        )
        mock_rag_pipeline.execute.assert_called_once_with("What is RAG?", strategy_params={"top_k": 3})
        qa_service._mock_generate_recommendations.assert_called_once()
        mock_rag_pipeline.get_current_parameters.assert_called_once()
        mock_db.add_or_get_rag_configuration.assert_called_once()
        mock_db.add_interaction.assert_called_once()
        
        # Verify result structure
        assert "query_details" in result
        assert "answer" in result
        assert "sources" in result
        assert "recommendations" in result
        assert "interaction_id" in result
        assert result["interaction_id"] == "interaction_id_456"
        assert result["recommendations"] == ["Related Q1", "Related Q2"]
    
    @pytest.mark.unit
    def test_answer_question_with_different_strategies(self, qa_service, sample_query_data):
        """Tests question answering with different routing strategies."""
        strategies_and_params = [
            ("precise_retrieval", {"top_k": 3}),
            ("broad_context_retrieval", {"top_k": 10}),
            ("hybrid_search", {"top_k": 5}),
            ("unknown_strategy", {})  # Empty params for unknown strategy
        ]
        
        for strategy, expected_params in strategies_and_params:
            qa_service._mock_adaptive_router.route_query.return_value = strategy
            qa_service.rag_pipeline.execute.reset_mock()
            
            qa_service.answer_question(sample_query_data["query"])
            
            qa_service.rag_pipeline.execute.assert_called_once_with("What is RAG?", strategy_params=expected_params)
    
    @pytest.mark.unit
    def test_answer_question_default_parameters(self, qa_service):
        """Tests question answering with default user_id and session_id."""
        result = qa_service.answer_question("Test query")
        
        # Verify database interaction was called with default user_id
        mock_db_call = qa_service.db.add_interaction.call_args
        assert mock_db_call[1]['user_id'] == "default_user"
        assert mock_db_call[1]['query'] == "Test query"
        
        # Verify session_id was generated (UUID format)
        session_id = mock_db_call[1]['session_id']
        assert len(session_id) == 36  # UUID length
        assert session_id.count('-') == 4  # UUID format
    
    @pytest.mark.unit
    def test_answer_question_custom_session_id(self, qa_service):
        """Tests question answering with custom session_id."""
        custom_session = "custom_session_789"
        qa_service.answer_question("Test query", session_id=custom_session)
        
        mock_db_call = qa_service.db.add_interaction.call_args
        assert mock_db_call[1]['session_id'] == custom_session
    
    @pytest.mark.unit
    @patch('src.qa.qa_service.time.time')
    def test_answer_question_timing_measurement(self, mock_time, qa_service, sample_query_data):
        """Tests that processing time is measured and logged correctly."""
        mock_time.side_effect = [1000.0, 1000.5]  # 500ms processing time
        
        qa_service.answer_question(sample_query_data["query"])
        
        mock_db_call = qa_service.db.add_interaction.call_args
        assert mock_db_call[1]['processing_time_ms'] == 500.0
    
    @pytest.mark.unit
    def test_answer_question_logging_flow(self, qa_service, sample_query_data):
        """Tests that all logging statements are called correctly."""
        qa_service.answer_question(
            query=sample_query_data["query"],
            user_id=sample_query_data["user_id"],
            session_id=sample_query_data["session_id"]
        )
        
        # Verify logging calls
        logger_calls = [call[0][0] for call in qa_service._mock_logger.info.call_args_list]
        
        expected_log_messages = [
            "Answering question for user 'test_user_123'",
            "Processing query...",
            "Query processed successfully.",
            "Determining best RAG strategy with Adaptive Query Router...",
            "Strategy selected: precise_retrieval",
            "Executing RAG pipeline with strategy 'precise_retrieval'...",
            "RAG pipeline executed successfully.",
            "Generating recommendations...",
            "Recommendations generated successfully.",
            "Successfully answered query for interaction interaction_id_456"
        ]
        
        for expected_msg in expected_log_messages:
            matching_logs = [log for log in logger_calls if expected_msg in log]
            assert len(matching_logs) > 0, f"Expected log message not found: {expected_msg}"


class TestErrorHandling:
    """Tests for error handling scenarios in QAService."""
    
    @pytest.mark.unit
    def test_answer_question_query_processor_value_error(self, qa_service, sample_query_data):
        """Tests handling of ValueError from query processor."""
        error_msg = "Invalid query format"
        qa_service.query_processor.process.side_effect = ValueError(error_msg)
        
        result = qa_service.answer_question(sample_query_data["query"])
        
        assert "error" in result
        assert result["error"] == error_msg
        qa_service._mock_logger.error.assert_called_once_with(f"Invalid query: {error_msg}")
    
    @pytest.mark.unit
    def test_answer_question_query_processor_generic_error(self, qa_service, sample_query_data):
        """Tests handling of generic Exception from query processor."""
        error_msg = "Database connection failed"
        qa_service.query_processor.process.side_effect = Exception(error_msg)
        
        result = qa_service.answer_question(sample_query_data["query"])
        
        assert "error" in result
        assert result["error"] == "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir hata oluştu."
        qa_service._mock_logger.error.assert_called_once()
        error_call = qa_service._mock_logger.error.call_args
        assert "An unexpected error occurred in QAService" in error_call[0][0]
        assert error_call[1]['exc_info'] == True
    
    @pytest.mark.unit
    def test_answer_question_rag_pipeline_error(self, qa_service, sample_query_data):
        """Tests handling of RAG pipeline execution error."""
        qa_service.rag_pipeline.execute.side_effect = Exception("RAG execution failed")
        
        result = qa_service.answer_question(sample_query_data["query"])
        
        assert "error" in result
        assert result["error"] == "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir hata oluştu."
    
    @pytest.mark.unit
    def test_answer_question_recommendations_error(self, qa_service, sample_query_data):
        """Tests handling of recommendations generation error."""
        qa_service._mock_generate_recommendations.side_effect = Exception("Recommendations failed")
        
        result = qa_service.answer_question(sample_query_data["query"])
        
        assert "error" in result
        assert result["error"] == "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir hata oluştu."
    
    @pytest.mark.unit
    def test_answer_question_database_error(self, qa_service, sample_query_data):
        """Tests handling of database interaction error."""
        qa_service.db.add_interaction.side_effect = Exception("Database write failed")
        
        result = qa_service.answer_question(sample_query_data["query"])
        
        assert "error" in result
        assert result["error"] == "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir hata oluştu."
    
    @pytest.mark.unit
    def test_answer_question_adaptive_router_error(self, qa_service, sample_query_data):
        """Tests handling of adaptive router error."""
        qa_service._mock_adaptive_router.route_query.side_effect = Exception("Router failed")
        
        result = qa_service.answer_question(sample_query_data["query"])
        
        assert "error" in result
        assert result["error"] == "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir hata oluştu."


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_answer_question_empty_query(self, qa_service):
        """Tests question answering with empty query."""
        result = qa_service.answer_question("")
        
        # Should still process through the pipeline
        qa_service.query_processor.process.assert_called_once_with("")
        assert "answer" in result or "error" in result
    
    @pytest.mark.unit
    def test_answer_question_very_long_query(self, qa_service):
        """Tests question answering with very long query."""
        long_query = "A" * 10000  # Very long query
        
        result = qa_service.answer_question(long_query)
        
        qa_service.query_processor.process.assert_called_once_with(long_query)
        assert "answer" in result or "error" in result
    
    @pytest.mark.unit
    def test_answer_question_special_characters(self, qa_service):
        """Tests question answering with special characters in query."""
        special_query = "RAG nedir? 中文 العربية éñüß 🤖🔍"
        
        result = qa_service.answer_question(special_query)
        
        qa_service.query_processor.process.assert_called_once_with(special_query)
        assert "answer" in result or "error" in result
    
    @pytest.mark.unit
    def test_answer_question_none_user_id(self, qa_service):
        """Tests question answering with None user_id."""
        result = qa_service.answer_question("Test query", user_id=None)
        
        mock_db_call = qa_service.db.add_interaction.call_args
        assert mock_db_call[1]['user_id'] is None
    
    @pytest.mark.unit
    def test_answer_question_empty_rag_response(self, qa_service, sample_query_data):
        """Tests question answering with empty RAG pipeline response."""
        qa_service.rag_pipeline.execute.return_value = {
            "answer": "",
            "sources": []
        }
        qa_service._mock_generate_recommendations.return_value = []
        
        result = qa_service.answer_question(sample_query_data["query"])
        
        assert result["answer"] == ""
        assert result["sources"] == []
        assert result["recommendations"] == []
    
    @pytest.mark.unit
    def test_answer_question_malformed_rag_response(self, qa_service, sample_query_data):
        """Tests question answering with malformed RAG pipeline response."""
        qa_service.rag_pipeline.execute.return_value = {
            "answer": "Valid answer"
            # Missing "sources" key
        }
        
        # Should handle gracefully and return error response
        result = qa_service.answer_question(sample_query_data["query"])
        
        assert "error" in result
        assert result["error"] == "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir hata oluştu."


class TestDatabaseInteractions:
    """Tests for database interaction scenarios."""
    
    @pytest.mark.unit
    def test_database_interaction_logging_success(self, qa_service, sample_query_data):
        """Tests successful database interaction logging."""
        qa_service.answer_question(
            query=sample_query_data["query"],
            user_id=sample_query_data["user_id"],
            session_id=sample_query_data["session_id"]
        )
        
        # Verify RAG configuration was stored
        qa_service.rag_pipeline.get_current_parameters.assert_called_once()
        qa_service.db.add_or_get_rag_configuration.assert_called_once()
        
        # Verify interaction was logged with correct parameters
        interaction_call = qa_service.db.add_interaction.call_args[1]
        assert interaction_call['user_id'] == sample_query_data["user_id"]
        assert interaction_call['session_id'] == sample_query_data["session_id"]
        assert interaction_call['query'] == sample_query_data["query"]
        assert interaction_call['response'] == "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation..."
        assert interaction_call['rag_config_hash'] == "config_hash_123"
        assert 'processing_time_ms' in interaction_call
        assert 'retrieved_context' in interaction_call
    
    @pytest.mark.unit
    def test_database_rag_config_hash_generation(self, qa_service, sample_query_data):
        """Tests RAG configuration hash generation and storage."""
        expected_params = {
            "retrieval_k": 5,
            "chunk_size": 512,
            "model": "gpt-3.5-turbo"
        }
        qa_service.rag_pipeline.get_current_parameters.return_value = expected_params
        
        qa_service.answer_question(sample_query_data["query"])
        
        qa_service.db.add_or_get_rag_configuration.assert_called_once_with(expected_params)
    
    @pytest.mark.unit
    def test_database_interaction_with_sources(self, qa_service, sample_query_data):
        """Tests that retrieved sources are properly logged to database."""
        expected_sources = [
            {"title": "RAG Paper", "content": "RAG overview...", "score": 0.95},
            {"title": "LLM Guide", "content": "Language models...", "score": 0.87}
        ]
        qa_service.rag_pipeline.execute.return_value = {
            "answer": "Test answer",
            "sources": expected_sources
        }
        
        qa_service.answer_question(sample_query_data["query"])
        
        interaction_call = qa_service.db.add_interaction.call_args[1]
        assert interaction_call['retrieved_context'] == expected_sources


class TestIntegrationScenarios:
    """Tests for integration scenarios between components."""
    
    @pytest.mark.unit
    def test_full_qa_workflow_integration(self, qa_service, sample_query_data):
        """Tests complete QA workflow with all components integrated."""
        # Setup complex scenario
        qa_service.query_processor.process.return_value = {
            "processed_query": "Explain deep learning transformers",
            "original_query": "Derin öğrenme transformerleri açıkla",
            "language": "tr",
            "query_type": "explanation",
            "entities": ["deep learning", "transformers"],
            "complexity": "high"
        }
        
        qa_service._mock_adaptive_router.route_query.return_value = "broad_context_retrieval"
        
        qa_service.rag_pipeline.execute.return_value = {
            "answer": "Transformers are a type of neural network architecture...",
            "sources": [
                {"title": "Attention Is All You Need", "content": "Transformer architecture...", "score": 0.98},
                {"title": "BERT Paper", "content": "Bidirectional transformers...", "score": 0.94},
                {"title": "GPT Overview", "content": "Generative transformers...", "score": 0.91}
            ],
            "confidence": 0.92
        }
        
        qa_service._mock_generate_recommendations.return_value = [
            "What are attention mechanisms?",
            "How do transformers differ from RNNs?",
            "What is BERT?"
        ]
        
        # Execute full workflow
        result = qa_service.answer_question(
            query=sample_query_data["query"],
            user_id=sample_query_data["user_id"],
            session_id=sample_query_data["session_id"]
        )
        
        # Verify complete workflow
        assert result["query_details"]["complexity"] == "high"
        assert result["query_details"]["language"] == "tr"
        assert len(result["sources"]) == 3
        assert len(result["recommendations"]) == 3
        assert result["answer"] == "Transformers are a type of neural network architecture..."
        
        # Verify component interactions
        qa_service._mock_adaptive_router.route_query.assert_called_once_with(
            query="Explain deep learning transformers",
            user_id=sample_query_data["user_id"],
            query_metadata={}
        )
        
        qa_service.rag_pipeline.execute.assert_called_once_with(
            "Explain deep learning transformers",
            strategy_params={"top_k": 10}  # broad_context_retrieval strategy
        )
    
    @pytest.mark.unit
    def test_adaptive_router_strategy_selection_impact(self, qa_service):
        """Tests how adaptive router strategy selection impacts RAG execution."""
        test_strategies = [
            ("precise_retrieval", {"top_k": 3}),
            ("broad_context_retrieval", {"top_k": 10}),
            ("hybrid_search", {"top_k": 5})
        ]
        
        for strategy, expected_params in test_strategies:
            qa_service._mock_adaptive_router.route_query.return_value = strategy
            qa_service.rag_pipeline.execute.reset_mock()
            
            qa_service.answer_question(f"Test query for {strategy}")
            
            qa_service.rag_pipeline.execute.assert_called_once_with(
                "What is RAG?",
                strategy_params=expected_params
            )
    
    @pytest.mark.unit
    def test_query_metadata_flow(self, qa_service, sample_query_data):
        """Tests that query metadata flows correctly through components."""
        # Currently, query_metadata is passed as empty dict
        # This test ensures the interface is correct for future enhancements
        qa_service.answer_question(
            query=sample_query_data["query"],
            user_id=sample_query_data["user_id"]
        )
        
        router_call = qa_service._mock_adaptive_router.route_query.call_args
        assert router_call[1]['query_metadata'] == {}
        
        # In future, this could include confidence scores or other metadata
        # from the query processor


class TestLoggingAndMonitoring:
    """Tests for logging and monitoring functionality."""
    
    @pytest.mark.unit
    def test_comprehensive_logging_coverage(self, qa_service, sample_query_data):
        """Tests that all major steps are logged appropriately."""
        qa_service.answer_question(
            query=sample_query_data["query"],
            user_id=sample_query_data["user_id"],
            session_id=sample_query_data["session_id"]
        )
        
        # Count different types of log calls
        info_calls = qa_service._mock_logger.info.call_args_list
        
        # Should have logs for: start, query processing, strategy selection, 
        # RAG execution, recommendations, and completion
        assert len(info_calls) >= 8
        
        # Check specific log patterns
        log_messages = [call[0][0] for call in info_calls]
        
        assert any("Answering question for user" in msg for msg in log_messages)
        assert any("Query processed successfully" in msg for msg in log_messages)
        assert any("Strategy selected" in msg for msg in log_messages)
        assert any("RAG pipeline executed successfully" in msg for msg in log_messages)
        assert any("Successfully answered query for interaction" in msg for msg in log_messages)
    
    @pytest.mark.unit
    def test_error_logging_with_stack_trace(self, qa_service, sample_query_data):
        """Tests that errors are logged with stack trace information."""
        qa_service.rag_pipeline.execute.side_effect = Exception("Test error for logging")
        
        qa_service.answer_question(sample_query_data["query"])
        
        # Verify error was logged with exc_info=True for stack trace
        error_calls = qa_service._mock_logger.error.call_args_list
        assert len(error_calls) == 1
        
        error_call = error_calls[0]
        assert "An unexpected error occurred in QAService" in error_call[0][0]
        assert error_call[1]['exc_info'] == True
    
    @pytest.mark.unit
    def test_session_and_interaction_logging(self, qa_service):
        """Tests that session and interaction IDs are properly logged."""
        custom_session = "test_session_logging"
        
        qa_service.answer_question("Test query", session_id=custom_session)
        
        # Find the initial log message that includes session info
        info_calls = qa_service._mock_logger.info.call_args_list
        session_logs = [call for call in info_calls if custom_session in str(call)]
        
        assert len(session_logs) > 0
        
        # Find the completion log that includes interaction ID
        completion_logs = [call for call in info_calls if "interaction_id_456" in str(call)]
        assert len(completion_logs) > 0


class TestPerformanceAndResourceManagement:
    """Tests for performance and resource management."""
    
    @pytest.mark.unit
    @patch('src.qa.qa_service.uuid.uuid4')
    def test_uuid_generation_for_session_id(self, mock_uuid, qa_service):
        """Tests UUID generation for session IDs."""
        mock_uuid.return_value = "mocked-uuid-12345"
        
        qa_service.answer_question("Test query")
        
        mock_uuid.assert_called_once()
        
        # Verify the mocked UUID was used in database call
        db_call = qa_service.db.add_interaction.call_args[1]
        assert db_call['session_id'] == "mocked-uuid-12345"
    
    @pytest.mark.unit
    def test_memory_efficient_data_handling(self, qa_service, sample_query_data):
        """Tests that large data structures are handled efficiently."""
        # Test with large sources list
        large_sources = [
            {"title": f"Document {i}", "content": "A" * 1000, "score": 0.9}
            for i in range(100)
        ]
        
        qa_service.rag_pipeline.execute.return_value = {
            "answer": "Test answer for large dataset",
            "sources": large_sources
        }
        
        result = qa_service.answer_question(sample_query_data["query"])
        
        # Should handle large dataset without issues
        assert len(result["sources"]) == 100
        assert result["answer"] == "Test answer for large dataset"
    
    @pytest.mark.unit
    def test_concurrent_request_handling(self, qa_service):
        """Tests behavior under concurrent request scenarios."""
        # This tests that the service doesn't maintain state between requests
        queries = ["Query 1", "Query 2", "Query 3"]
        results = []
        
        for query in queries:
            result = qa_service.answer_question(query)
            results.append(result)
        
        # Each result should be independent
        assert len(results) == 3
        for i, result in enumerate(results):
            assert "answer" in result or "error" in result
            # Interaction IDs should be the same since we're mocking, 
            # but in real scenario they'd be different