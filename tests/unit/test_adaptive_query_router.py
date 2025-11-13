import pytest
from unittest.mock import MagicMock, patch, call
import logging

from src.rag.adaptive_query_router import AdaptiveQueryRouter


@pytest.fixture
def mock_db():
    """Mock database connection for AdaptiveQueryRouter."""
    mock_db = MagicMock()
    
    # Default mock return values
    mock_db.get_user_profile.return_value = {
        "expertise_level": "intermediate", 
        "past_topics": ["RAG", "Python"]
    }
    mock_db.get_strategy_performance_by_query_type.return_value = {
        "best_strategy": "precise_retrieval",
        "avg_rating": 4.5
    }
    
    return mock_db


@pytest.fixture
def mock_active_learning_engine():
    """Mock ActiveLearningEngine for AdaptiveQueryRouter."""
    mock_ale = MagicMock()
    
    # Mock the sampler and its entropy method
    mock_sampler = MagicMock()
    mock_sampler.entropy.return_value = 0.5
    mock_ale.sampler = mock_sampler
    
    return mock_ale


@pytest.fixture
def adaptive_router(mock_db, mock_active_learning_engine):
    """AdaptiveQueryRouter instance with mocked dependencies."""
    return AdaptiveQueryRouter(
        db_connection=mock_db,
        active_learning_engine=mock_active_learning_engine
    )


@pytest.fixture
def sample_probabilities():
    """Sample probability distributions for testing."""
    return [0.6, 0.3, 0.1]


@pytest.fixture
def sample_query_metadata():
    """Sample query metadata for testing."""
    return {
        'probabilities': [0.6, 0.3, 0.1],
        'confidence': 0.8,
        'language': 'tr'
    }


class TestAdaptiveQueryRouterInit:
    """Tests for AdaptiveQueryRouter initialization."""
    
    def test_init_success(self, mock_db, mock_active_learning_engine):
        """Tests successful initialization of AdaptiveQueryRouter."""
        router = AdaptiveQueryRouter(mock_db, mock_active_learning_engine)
        
        assert router.db == mock_db
        assert router.ale == mock_active_learning_engine
        assert isinstance(router.rag_strategies, dict)
        assert len(router.rag_strategies) == 3
        assert "precise_retrieval" in router.rag_strategies
        assert "broad_context_retrieval" in router.rag_strategies
        assert "hybrid_search" in router.rag_strategies
    
    def test_init_with_none_database(self, mock_active_learning_engine):
        """Tests initialization with None database connection."""
        router = AdaptiveQueryRouter(None, mock_active_learning_engine)
        
        assert router.db is None
        assert router.ale == mock_active_learning_engine
        assert isinstance(router.rag_strategies, dict)
    
    def test_init_with_none_active_learning(self, mock_db):
        """Tests initialization with None active learning engine."""
        router = AdaptiveQueryRouter(mock_db, None)
        
        assert router.db == mock_db
        assert router.ale is None
        assert isinstance(router.rag_strategies, dict)
    
    def test_rag_strategies_content(self, adaptive_router):
        """Tests that RAG strategies contain expected descriptions."""
        strategies = adaptive_router.rag_strategies
        
        assert "Küçük chunk" in strategies["precise_retrieval"]
        assert "Geniş chunk" in strategies["broad_context_retrieval"]
        assert "Anahtar kelime ve anlamsal" in strategies["hybrid_search"]


class TestGetStudentProfile:
    """Tests for get_student_profile method."""
    
    def test_get_student_profile_success(self, adaptive_router, mock_db):
        """Tests successful retrieval of student profile."""
        user_id = "test_user_123"
        expected_profile = {
            "expertise_level": "advanced",
            "past_topics": ["Machine Learning", "Deep Learning"]
        }
        mock_db.get_user_profile.return_value = expected_profile
        
        result = adaptive_router.get_student_profile(user_id)
        
        mock_db.get_user_profile.assert_called_once_with(user_id)
        assert result == expected_profile
    
    def test_get_student_profile_no_profile_found(self, adaptive_router, mock_db):
        """Tests get_student_profile when no profile is found."""
        user_id = "nonexistent_user"
        mock_db.get_user_profile.return_value = None
        
        result = adaptive_router.get_student_profile(user_id)
        
        mock_db.get_user_profile.assert_called_once_with(user_id)
        assert result == {"expertise_level": "beginner", "past_topics": []}
    
    def test_get_student_profile_empty_profile(self, adaptive_router, mock_db):
        """Tests get_student_profile with empty profile data."""
        user_id = "empty_user"
        mock_db.get_user_profile.return_value = {}
        
        result = adaptive_router.get_student_profile(user_id)
        
        # Empty dict is treated the same as None in the implementation - returns default profile
        assert result == {"expertise_level": "beginner", "past_topics": []}
    
    def test_get_student_profile_with_special_characters(self, adaptive_router, mock_db):
        """Tests get_student_profile with special characters in user_id."""
        user_id = "user@domain.com_ñüéß"
        expected_profile = {"expertise_level": "beginner", "past_topics": ["Turkish"]}
        mock_db.get_user_profile.return_value = expected_profile
        
        result = adaptive_router.get_student_profile(user_id)
        
        mock_db.get_user_profile.assert_called_once_with(user_id)
        assert result == expected_profile
    
    def test_get_student_profile_database_error(self, adaptive_router, mock_db):
        """Tests get_student_profile when database error occurs."""
        user_id = "error_user"
        error_message = "Database connection failed"
        mock_db.get_user_profile.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            adaptive_router.get_student_profile(user_id)


class TestGetHistoricalPerformance:
    """Tests for get_historical_performance method."""
    
    def test_get_historical_performance_success(self, adaptive_router, mock_db):
        """Tests successful retrieval of historical performance."""
        query_type = "definition"
        expected_performance = {
            "best_strategy": "precise_retrieval",
            "avg_rating": 4.2,
            "total_queries": 15
        }
        mock_db.get_strategy_performance_by_query_type.return_value = expected_performance
        
        result = adaptive_router.get_historical_performance(query_type)
        
        mock_db.get_strategy_performance_by_query_type.assert_called_once_with(query_type)
        assert result == expected_performance
    
    def test_get_historical_performance_no_data(self, adaptive_router, mock_db):
        """Tests get_historical_performance when no historical data exists."""
        query_type = "unknown_type"
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        result = adaptive_router.get_historical_performance(query_type)
        
        mock_db.get_strategy_performance_by_query_type.assert_called_once_with(query_type)
        assert result is None
    
    def test_get_historical_performance_empty_data(self, adaptive_router, mock_db):
        """Tests get_historical_performance with empty performance data."""
        query_type = "explanation"
        mock_db.get_strategy_performance_by_query_type.return_value = {}
        
        result = adaptive_router.get_historical_performance(query_type)
        
        assert result == {}
    
    def test_get_historical_performance_all_query_types(self, adaptive_router, mock_db):
        """Tests get_historical_performance with different query types."""
        query_types = ["definition", "explanation", "comparison", "tutorial"]
        
        for query_type in query_types:
            mock_performance = {"best_strategy": "hybrid_search", "avg_rating": 3.5}
            mock_db.get_strategy_performance_by_query_type.return_value = mock_performance
            
            result = adaptive_router.get_historical_performance(query_type)
            
            assert result == mock_performance
        
        assert mock_db.get_strategy_performance_by_query_type.call_count == len(query_types)
    
    def test_get_historical_performance_database_error(self, adaptive_router, mock_db):
        """Tests get_historical_performance when database error occurs."""
        query_type = "definition"
        error_message = "Query execution timeout"
        mock_db.get_strategy_performance_by_query_type.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            adaptive_router.get_historical_performance(query_type)


class TestSelectStrategyDynamically:
    """Tests for select_strategy_dynamically method."""
    
    def test_select_strategy_with_historical_performance(self, adaptive_router, mock_db):
        """Tests strategy selection based on historical performance."""
        query = "RAG nedir?"
        user_id = "test_user"
        
        # Setup mocks
        mock_profile = {"expertise_level": "intermediate", "past_topics": ["AI"]}
        mock_performance = {"best_strategy": "broad_context_retrieval", "avg_rating": 4.8}
        
        mock_db.get_user_profile.return_value = mock_profile
        mock_db.get_strategy_performance_by_query_type.return_value = mock_performance
        
        with patch('builtins.print') as mock_print:
            result = adaptive_router.select_strategy_dynamically(query, user_id)
        
        assert result == "broad_context_retrieval"
        mock_print.assert_called_once_with("Geçmiş performansa göre en iyi strateji: broad_context_retrieval")
    
    def test_select_strategy_beginner_definition_query(self, adaptive_router, mock_db):
        """Tests strategy selection for beginner with definition query."""
        query = "Python nedir?"
        user_id = "beginner_user"
        
        # Setup mocks for no historical performance
        mock_profile = {"expertise_level": "beginner", "past_topics": []}
        mock_db.get_user_profile.return_value = mock_profile
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        result = adaptive_router.select_strategy_dynamically(query, user_id)
        
        assert result == "precise_retrieval"
    
    def test_select_strategy_no_historical_no_beginner_definition(self, adaptive_router, mock_db):
        """Tests strategy selection with no historical data and non-beginner or non-definition."""
        query = "How does Python work?"
        user_id = "advanced_user"
        
        mock_profile = {"expertise_level": "advanced", "past_topics": ["Python"]}
        mock_db.get_user_profile.return_value = mock_profile
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        result = adaptive_router.select_strategy_dynamically(query, user_id)
        
        assert result == "hybrid_search"
    
    def test_select_strategy_query_type_detection(self, adaptive_router, mock_db):
        """Tests query type detection logic."""
        test_cases = [
            ("RAG nedir?", "definition"),
            ("Python NEDİR ve nasıl kullanılır?", "explanation"),  # Unicode İ -> i̇ issue
            ("How does machine learning work?", "explanation"),
            ("Explain transformers", "explanation")
        ]
        
        mock_db.get_user_profile.return_value = {"expertise_level": "intermediate", "past_topics": []}
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        for query, expected_type in test_cases:
            # Reset mock to track individual calls
            mock_db.get_strategy_performance_by_query_type.reset_mock()
            
            adaptive_router.select_strategy_dynamically(query, "test_user")
            
            # Verify the correct query type was used for database lookup
            last_call = mock_db.get_strategy_performance_by_query_type.call_args[0][0]
            assert last_call == expected_type
    
    def test_select_strategy_empty_historical_performance(self, adaptive_router, mock_db):
        """Tests strategy selection when historical performance is empty dict."""
        query = "Test query"
        user_id = "test_user"
        
        mock_profile = {"expertise_level": "intermediate", "past_topics": []}
        mock_performance = {}  # Empty dict, no 'best_strategy' key
        
        mock_db.get_user_profile.return_value = mock_profile
        mock_db.get_strategy_performance_by_query_type.return_value = mock_performance
        
        result = adaptive_router.select_strategy_dynamically(query, user_id)
        
        # Should fall back to hybrid_search since conditions don't match beginner+definition
        assert result == "hybrid_search"
    
    def test_select_strategy_profile_based_fallback(self, adaptive_router, mock_db):
        """Tests various profile-based strategy selections."""
        test_cases = [
            # (profile, query, expected_strategy)
            # "What is AI?" -> explanation -> beginner + explanation = hybrid_search
            ({"expertise_level": "beginner", "past_topics": []}, "What is AI?", "hybrid_search"),
            # "Yapay zeka nedir?" -> definition -> beginner + definition = precise_retrieval
            ({"expertise_level": "beginner", "past_topics": []}, "Yapay zeka nedir?", "precise_retrieval"),
            # "AI nedir?" -> definition -> advanced + definition = hybrid_search (default)
            ({"expertise_level": "advanced", "past_topics": []}, "AI nedir?", "hybrid_search"),
            # "How does AI work?" -> explanation -> intermediate + explanation = hybrid_search
            ({"expertise_level": "intermediate", "past_topics": []}, "How does AI work?", "hybrid_search")
        ]
        
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        for profile, query, expected_strategy in test_cases:
            mock_db.get_user_profile.return_value = profile
            
            result = adaptive_router.select_strategy_dynamically(query, "test_user")
            
            assert result == expected_strategy


class TestRouteQuery:
    """Tests for route_query method."""
    
    @patch('builtins.print')
    def test_route_query_high_uncertainty_fallback(self, mock_print, adaptive_router, 
                                                 mock_active_learning_engine, sample_query_metadata):
        """Tests route_query with high uncertainty triggering fallback strategy."""
        query = "Complex ambiguous question"
        user_id = "test_user"
        
        # Setup high uncertainty score
        mock_active_learning_engine.sampler.entropy.return_value = 0.9
        high_uncertainty_metadata = {"probabilities": [0.25, 0.25, 0.25, 0.25]}
        
        result = adaptive_router.route_query(query, user_id, high_uncertainty_metadata)
        
        assert result == "broad_context_retrieval"
        mock_active_learning_engine.sampler.entropy.assert_called_once_with([0.25, 0.25, 0.25, 0.25])
        mock_print.assert_called_with("Yüksek belirsizlik (0.90) tespit edildi. Geniş bağlam stratejisine yönlendiriliyor.")
    
    @patch('builtins.print')
    def test_route_query_low_uncertainty_dynamic_selection(self, mock_print, adaptive_router, 
                                                         mock_active_learning_engine, mock_db):
        """Tests route_query with low uncertainty using dynamic strategy selection."""
        query = "RAG nedir?"
        user_id = "test_user"
        
        # Setup low uncertainty
        mock_active_learning_engine.sampler.entropy.return_value = 0.3
        metadata = {"probabilities": [0.8, 0.15, 0.05]}
        
        # Setup for dynamic selection
        mock_db.get_user_profile.return_value = {"expertise_level": "beginner", "past_topics": []}
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        result = adaptive_router.route_query(query, user_id, metadata)
        
        assert result == "precise_retrieval"  # beginner + definition query
        mock_active_learning_engine.sampler.entropy.assert_called_once_with([0.8, 0.15, 0.05])
        
        # Verify both print statements
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Dinamik olarak seçilen strateji: precise_retrieval" in msg for msg in print_calls)
    
    def test_route_query_empty_probabilities(self, adaptive_router, mock_active_learning_engine):
        """Tests route_query with empty probabilities in metadata."""
        query = "Test query"
        user_id = "test_user"
        metadata = {"probabilities": []}
        
        mock_active_learning_engine.sampler.entropy.return_value = 0.0
        
        with patch('builtins.print'):
            result = adaptive_router.route_query(query, user_id, metadata)
        
        mock_active_learning_engine.sampler.entropy.assert_called_once_with([])
        assert result in ["precise_retrieval", "hybrid_search"]  # Could be either based on dynamic selection
    
    def test_route_query_no_probabilities_in_metadata(self, adaptive_router, mock_active_learning_engine):
        """Tests route_query when probabilities key is missing from metadata."""
        query = "Test query"
        user_id = "test_user"
        metadata = {"language": "tr"}  # No probabilities key
        
        mock_active_learning_engine.sampler.entropy.return_value = 0.0
        
        with patch('builtins.print'):
            result = adaptive_router.route_query(query, user_id, metadata)
        
        mock_active_learning_engine.sampler.entropy.assert_called_once_with([])
        assert isinstance(result, str)
    
    def test_route_query_boundary_uncertainty_values(self, adaptive_router, mock_active_learning_engine):
        """Tests route_query with boundary uncertainty values."""
        query = "Boundary test query"
        user_id = "test_user"
        
        # Test exactly at threshold (0.8)
        mock_active_learning_engine.sampler.entropy.return_value = 0.8
        metadata = {"probabilities": [0.4, 0.3, 0.3]}
        
        with patch('builtins.print'):
            result = adaptive_router.route_query(query, user_id, metadata)
        
        # At exactly 0.8, should not trigger high uncertainty (> 0.8)
        assert result != "broad_context_retrieval"
        
        # Test just above threshold
        mock_active_learning_engine.sampler.entropy.return_value = 0.81
        
        with patch('builtins.print'):
            result = adaptive_router.route_query(query, user_id, metadata)
        
        # Above 0.8, should trigger high uncertainty fallback
        assert result == "broad_context_retrieval"
    
    @patch('builtins.print')
    def test_route_query_logging_verification(self, mock_print, adaptive_router, 
                                            mock_active_learning_engine, mock_db):
        """Tests that route_query logs appropriately in different scenarios."""
        query = "Test logging query"
        user_id = "test_user"
        
        # Test high uncertainty logging
        mock_active_learning_engine.sampler.entropy.return_value = 0.95
        metadata = {"probabilities": [0.3, 0.3, 0.2, 0.2]}
        
        adaptive_router.route_query(query, user_id, metadata)
        
        high_uncertainty_calls = [call for call in mock_print.call_args_list 
                                if len(call[0]) > 0 and "Yüksek belirsizlik" in call[0][0]]
        assert len(high_uncertainty_calls) == 1
        assert "(0.95)" in high_uncertainty_calls[0][0][0]
        
        # Reset mock and test dynamic selection logging
        mock_print.reset_mock()
        mock_active_learning_engine.sampler.entropy.return_value = 0.4
        mock_db.get_user_profile.return_value = {"expertise_level": "intermediate", "past_topics": []}
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        adaptive_router.route_query(query, user_id, {"probabilities": [0.7, 0.2, 0.1]})
        
        dynamic_calls = [call for call in mock_print.call_args_list 
                        if len(call[0]) > 0 and "Dinamik olarak seçilen strateji" in call[0][0]]
        assert len(dynamic_calls) == 1


class TestAdaptiveQueryRouterErrorHandling:
    """Tests for error handling scenarios in AdaptiveQueryRouter."""
    
    def test_route_query_active_learning_engine_error(self, adaptive_router, mock_active_learning_engine):
        """Tests route_query when active learning engine throws error."""
        query = "Error test query"
        user_id = "test_user"
        metadata = {"probabilities": [0.5, 0.3, 0.2]}
        
        error_message = "Entropy calculation failed"
        mock_active_learning_engine.sampler.entropy.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            adaptive_router.route_query(query, user_id, metadata)
    
    def test_get_student_profile_with_invalid_user_id(self, adaptive_router, mock_db):
        """Tests get_student_profile with None or invalid user_id."""
        # Test with None user_id
        mock_db.get_user_profile.return_value = None
        result = adaptive_router.get_student_profile(None)
        
        mock_db.get_user_profile.assert_called_once_with(None)
        assert result == {"expertise_level": "beginner", "past_topics": []}
        
        # Test with empty string user_id
        mock_db.get_user_profile.return_value = None
        result = adaptive_router.get_student_profile("")
        
        assert result == {"expertise_level": "beginner", "past_topics": []}
    
    def test_select_strategy_dynamically_database_errors(self, adaptive_router, mock_db):
        """Tests select_strategy_dynamically when database operations fail."""
        query = "Error test query"
        user_id = "error_user"
        
        # Test get_user_profile error
        mock_db.get_user_profile.side_effect = Exception("Profile query failed")
        
        with pytest.raises(Exception, match="Profile query failed"):
            adaptive_router.select_strategy_dynamically(query, user_id)
        
        # Test get_strategy_performance_by_query_type error
        mock_db.get_user_profile.side_effect = None
        mock_db.get_user_profile.return_value = {"expertise_level": "intermediate", "past_topics": []}
        mock_db.get_strategy_performance_by_query_type.side_effect = Exception("Performance query failed")
        
        with pytest.raises(Exception, match="Performance query failed"):
            adaptive_router.select_strategy_dynamically(query, user_id)
    
    def test_route_query_none_active_learning_engine(self, mock_db):
        """Tests route_query when active learning engine is None."""
        router = AdaptiveQueryRouter(mock_db, None)
        
        query = "Test query"
        user_id = "test_user"
        metadata = {"probabilities": [0.6, 0.3, 0.1]}
        
        with pytest.raises(AttributeError):
            router.route_query(query, user_id, metadata)


class TestAdaptiveQueryRouterEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_route_query_extreme_probability_distributions(self, adaptive_router, mock_active_learning_engine):
        """Tests route_query with extreme probability distributions."""
        query = "Extreme probability test"
        user_id = "test_user"
        
        extreme_cases = [
            [1.0, 0.0, 0.0],  # Certain prediction
            [0.0, 0.0, 1.0],  # Certain but different class
            [0.33, 0.33, 0.34],  # Nearly uniform
            [0.001, 0.998, 0.001],  # Very skewed
        ]
        
        for probabilities in extreme_cases:
            metadata = {"probabilities": probabilities}
            mock_active_learning_engine.sampler.entropy.return_value = 0.5  # Moderate uncertainty
            
            with patch('builtins.print'):
                result = adaptive_router.route_query(query, user_id, metadata)
            
            assert result in ["precise_retrieval", "broad_context_retrieval", "hybrid_search"]
    
    def test_get_student_profile_extreme_profile_data(self, adaptive_router, mock_db):
        """Tests get_student_profile with extreme profile data."""
        extreme_profiles = [
            {"expertise_level": "super_advanced", "past_topics": ["topic"] * 1000},
            {"expertise_level": "", "past_topics": []},
            {"expertise_level": "beginner" * 100, "past_topics": [""]},
            {"custom_field": "value", "unexpected_data": 12345}
        ]
        
        for profile in extreme_profiles:
            user_id = f"user_{hash(str(profile))}"
            mock_db.get_user_profile.return_value = profile
            
            result = adaptive_router.get_student_profile(user_id)
            
            assert result == profile
    
    def test_select_strategy_with_very_long_queries(self, adaptive_router, mock_db):
        """Tests select_strategy_dynamically with very long queries."""
        long_query = "nedir " * 1000 + "?"
        user_id = "test_user"
        
        mock_db.get_user_profile.return_value = {"expertise_level": "beginner", "past_topics": []}
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        result = adaptive_router.select_strategy_dynamically(long_query, user_id)
        
        # Should still detect "nedir" and classify as definition
        assert result == "precise_retrieval"
    
    def test_route_query_with_special_characters_in_query(self, adaptive_router, mock_active_learning_engine):
        """Tests route_query with special characters and Unicode in query."""
        special_queries = [
            "RAG nedir? 🤖",
            "Python'da lambda fonksiyonu nasıl yazılır? éñüß",
            "机器学习是什么？",
            "¿Qué es la inteligencia artificial?",
            "query with\nnewlines\tand\ttabs"
        ]
        
        mock_active_learning_engine.sampler.entropy.return_value = 0.4
        
        for query in special_queries:
            metadata = {"probabilities": [0.7, 0.2, 0.1]}
            
            with patch('builtins.print'):
                result = adaptive_router.route_query(query, "test_user", metadata)
            
            assert isinstance(result, str)
            assert result in ["precise_retrieval", "broad_context_retrieval", "hybrid_search"]


class TestAdaptiveQueryRouterIntegration:
    """Integration tests for AdaptiveQueryRouter workflows."""
    
    @patch('builtins.print')
    def test_full_routing_workflow_high_uncertainty(self, mock_print, adaptive_router, 
                                                  mock_active_learning_engine, mock_db):
        """Tests complete routing workflow with high uncertainty scenario."""
        query = "Very ambiguous and complex query"
        user_id = "integration_user"
        metadata = {"probabilities": [0.3, 0.3, 0.2, 0.2]}
        
        # Setup high uncertainty
        mock_active_learning_engine.sampler.entropy.return_value = 0.95
        
        result = adaptive_router.route_query(query, user_id, metadata)
        
        # Should bypass dynamic selection due to high uncertainty
        assert result == "broad_context_retrieval"
        
        # Verify entropy was calculated
        mock_active_learning_engine.sampler.entropy.assert_called_once_with([0.3, 0.3, 0.2, 0.2])
        
        # Verify database methods were NOT called due to early return
        mock_db.get_user_profile.assert_not_called()
        mock_db.get_strategy_performance_by_query_type.assert_not_called()
        
        # Verify logging
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Yüksek belirsizlik (0.95)" in msg for msg in print_calls)
    
    @patch('builtins.print')
    def test_full_routing_workflow_with_historical_data(self, mock_print, adaptive_router, 
                                                       mock_active_learning_engine, mock_db):
        """Tests complete routing workflow using historical performance data."""
        query = "Machine learning nedir?"
        user_id = "experienced_user"
        metadata = {"probabilities": [0.6, 0.3, 0.1]}
        
        # Setup moderate uncertainty
        mock_active_learning_engine.sampler.entropy.return_value = 0.4
        
        # Setup database responses
        mock_db.get_user_profile.return_value = {
            "expertise_level": "advanced",
            "past_topics": ["Machine Learning", "Deep Learning"]
        }
        mock_db.get_strategy_performance_by_query_type.return_value = {
            "best_strategy": "hybrid_search",
            "avg_rating": 4.7
        }
        
        result = adaptive_router.route_query(query, user_id, metadata)
        
        assert result == "hybrid_search"
        
        # Verify all database interactions
        mock_db.get_user_profile.assert_called_once_with(user_id)
        mock_db.get_strategy_performance_by_query_type.assert_called_once_with("definition")
        
        # Verify logging sequence
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Geçmiş performansa göre en iyi strateji: hybrid_search" in msg for msg in print_calls)
        assert any("Dinamik olarak seçilen strateji: hybrid_search" in msg for msg in print_calls)
    
    @patch('builtins.print')
    def test_full_routing_workflow_profile_based_fallback(self, mock_print, adaptive_router,
                                                         mock_active_learning_engine, mock_db):
        """Tests complete routing workflow falling back to profile-based strategy."""
        query = "What is artificial intelligence?"
        user_id = "new_user"
        metadata = {"probabilities": [0.8, 0.15, 0.05]}
        
        # Setup low uncertainty
        mock_active_learning_engine.sampler.entropy.return_value = 0.2
        
        # Setup database responses - no historical data
        mock_db.get_user_profile.return_value = {
            "expertise_level": "beginner",
            "past_topics": []
        }
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        result = adaptive_router.route_query(query, user_id, metadata)
        
        # Should use profile-based fallback (beginner + non-Turkish definition query)
        assert result == "hybrid_search"
        
        # Verify database interactions
        mock_db.get_user_profile.assert_called_once_with(user_id)
        mock_db.get_strategy_performance_by_query_type.assert_called_once_with("explanation")
        
        # Verify logging
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Dinamik olarak seçilen strateji: hybrid_search" in msg for msg in print_calls)
    
    def test_strategy_consistency_across_similar_queries(self, adaptive_router,
                                                       mock_active_learning_engine, mock_db):
        """Tests that similar queries receive consistent strategy routing."""
        # Use queries that all properly detect as "definition"
        similar_queries = [
            "Python nedir?",
            "Python programming dili nedir?",
            "RAG sistemi nedir?"
        ]
        
        user_id = "consistency_user"
        metadata = {"probabilities": [0.7, 0.2, 0.1]}
        
        # Setup consistent conditions
        mock_active_learning_engine.sampler.entropy.return_value = 0.3  # Low uncertainty
        mock_db.get_user_profile.return_value = {"expertise_level": "beginner", "past_topics": []}
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        results = []
        with patch('builtins.print'):
            for query in similar_queries:
                result = adaptive_router.route_query(query, user_id, metadata)
                results.append(result)
        
        # All similar "nedir" queries should get the same strategy for beginner
        # All contain "nedir" -> definition type -> beginner + definition = precise_retrieval
        assert all(result == "precise_retrieval" for result in results), f"Results were: {results}"


class TestAdaptiveQueryRouterPerformance:
    """Performance and resource management tests."""
    
    def test_route_query_with_large_probability_arrays(self, adaptive_router, mock_active_learning_engine):
        """Tests route_query performance with large probability arrays."""
        query = "Performance test query"
        user_id = "perf_user"
        
        # Large probability array (100 classes)
        large_probabilities = [0.01] * 100
        metadata = {"probabilities": large_probabilities}
        
        mock_active_learning_engine.sampler.entropy.return_value = 0.5
        
        with patch('builtins.print'):
            result = adaptive_router.route_query(query, user_id, metadata)
        
        # Should handle large arrays without issues
        mock_active_learning_engine.sampler.entropy.assert_called_once_with(large_probabilities)
        assert isinstance(result, str)
    
    def test_multiple_concurrent_routing_requests(self, adaptive_router, mock_active_learning_engine, mock_db):
        """Tests handling of multiple routing requests (state independence)."""
        queries = [f"Query {i}" for i in range(10)]
        user_ids = [f"user_{i}" for i in range(10)]
        
        mock_active_learning_engine.sampler.entropy.return_value = 0.4
        mock_db.get_user_profile.return_value = {"expertise_level": "intermediate", "past_topics": []}
        mock_db.get_strategy_performance_by_query_type.return_value = None
        
        results = []
        with patch('builtins.print'):
            for query, user_id in zip(queries, user_ids):
                metadata = {"probabilities": [0.6, 0.3, 0.1]}
                result = adaptive_router.route_query(query, user_id, metadata)
                results.append(result)
        
        # All requests should be processed independently
        assert len(results) == 10
        assert all(isinstance(result, str) for result in results)
        
        # Verify database was called for each request
        assert mock_db.get_user_profile.call_count == 10
        assert mock_db.get_strategy_performance_by_query_type.call_count == 10