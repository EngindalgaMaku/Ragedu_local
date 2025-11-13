import pytest
import json
import tempfile
import os
from unittest.mock import MagicMock, patch, call, mock_open
from typing import Dict, Any, List

from src.services.rag_optimizer import RAGParameterOptimizer


@pytest.fixture
def mock_db():
    """Mock database connection for RAGParameterOptimizer."""
    mock_db = MagicMock()
    
    # Default mock return values
    mock_db.get_performance_metrics.return_value = [
        {'query': 'Test query 1', 'rating': 3.5, 'retrieval_k': 5},
        {'query': 'Test query 2', 'rating': 4.0, 'retrieval_k': 5},
        {'query': 'Test query 3', 'rating': 2.8, 'retrieval_k': 5}
    ]
    mock_db.get_feedback_since.return_value = [
        {'query': 'Test feedback', 'rating': 3, 'topic': 'Test Topic'}
    ]
    
    return mock_db


@pytest.fixture
def default_config():
    """Default configuration for RAGParameterOptimizer."""
    return {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "retrieval_k": 5,
        "query_routing_rules": {
            "keyword_based": ["tanım", "nedir"],
            "semantic_based": ["nasıl", "neden", "karşılaştır"]
        }
    }


@pytest.fixture
def temp_config_file(default_config):
    """Temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(default_config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def rag_optimizer(mock_db):
    """RAGParameterOptimizer instance with mocked dependencies."""
    with patch('builtins.open', mock_open(read_data=json.dumps({
        "chunk_size": 512,
        "chunk_overlap": 50,
        "retrieval_k": 5,
        "query_routing_rules": {
            "keyword_based": ["tanım", "nedir"],
            "semantic_based": ["nasıl", "neden", "karşılaştır"]
        }
    }))):
        optimizer = RAGParameterOptimizer(db_connection=mock_db)
        yield optimizer


class TestRAGParameterOptimizerInit:
    """Tests for RAGParameterOptimizer initialization."""
    
    @patch('builtins.open', mock_open(read_data='{"test": "config"}'))
    def test_init_success(self, mock_db):
        """Tests successful initialization of RAGParameterOptimizer."""
        optimizer = RAGParameterOptimizer(db_connection=mock_db, config_path='test_config.py')
        
        assert optimizer.db == mock_db
        assert optimizer.config_path == 'test_config.py'
        assert optimizer.json_config_path == 'rag_config.json'
        assert optimizer.current_config == {"test": "config"}
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('src.services.rag_optimizer.RAGParameterOptimizer.update_config')
    def test_init_creates_default_config_when_file_not_found(self, mock_update_config, mock_open_func, mock_db):
        """Tests initialization creates default config when file doesn't exist."""
        optimizer = RAGParameterOptimizer(db_connection=mock_db)
        
        expected_default = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "retrieval_k": 5,
            "query_routing_rules": {
                "keyword_based": ["tanım", "nedir"],
                "semantic_based": ["nasıl", "neden", "karşılaştır"]
            }
        }
        
        mock_update_config.assert_called_once_with(expected_default)
        assert optimizer.current_config == expected_default
    
    def test_init_with_none_database(self):
        """Tests initialization with None database connection."""
        with patch('builtins.open', mock_open(read_data='{"test": "config"}')):
            optimizer = RAGParameterOptimizer(db_connection=None)
            assert optimizer.db is None
    
    def test_init_default_config_path(self, mock_db):
        """Tests initialization with default config path."""
        with patch('builtins.open', mock_open(read_data='{"test": "config"}')):
            optimizer = RAGParameterOptimizer(db_connection=mock_db)
            assert optimizer.config_path == 'src/config.py'


class TestLoadConfig:
    """Tests for load_config method."""
    
    @patch('builtins.open', mock_open(read_data='{"chunk_size": 256}'))
    def test_load_config_success(self, mock_db):
        """Tests successful configuration loading."""
        optimizer = RAGParameterOptimizer(db_connection=mock_db)
        result = optimizer.load_config()
        
        assert result == {"chunk_size": 256}
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('src.services.rag_optimizer.RAGParameterOptimizer.update_config')
    def test_load_config_file_not_found_creates_default(self, mock_update_config, mock_open_func, mock_db):
        """Tests load_config creates default config when file not found."""
        optimizer = RAGParameterOptimizer.__new__(RAGParameterOptimizer)
        optimizer.db = mock_db
        optimizer.config_path = 'test_config.py'
        optimizer.json_config_path = 'rag_config.json'
        
        result = optimizer.load_config()
        
        expected_default = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "retrieval_k": 5,
            "query_routing_rules": {
                "keyword_based": ["tanım", "nedir"],
                "semantic_based": ["nasıl", "neden", "karşılaştır"]
            }
        }
        
        assert result == expected_default
        mock_update_config.assert_called_once_with(expected_default)
    
    @patch('builtins.open', mock_open(read_data='invalid json'))
    def test_load_config_invalid_json(self, mock_db):
        """Tests load_config with invalid JSON content."""
        with pytest.raises(json.JSONDecodeError):
            RAGParameterOptimizer(db_connection=mock_db)


class TestUpdateConfig:
    """Tests for update_config method."""
    
    @patch('builtins.print')
    def test_update_config_success(self, mock_print, rag_optimizer):
        """Tests successful configuration update."""
        new_config = {"chunk_size": 1024, "retrieval_k": 8}
        
        with patch('builtins.open', mock_open()) as mock_file:
            rag_optimizer.update_config(new_config)
        
        # Verify file was written
        mock_file.assert_called_once_with(rag_optimizer.json_config_path, 'w')
        handle = mock_file()
        handle.write.assert_called()
        
        # Verify config was updated in memory
        assert rag_optimizer.current_config == new_config
        
        # Verify print message
        mock_print.assert_called_once_with(f"Yapılandırma güncellendi: {rag_optimizer.json_config_path}")
    
    @patch('builtins.print')
    def test_update_config_with_complex_data(self, mock_print, rag_optimizer):
        """Tests configuration update with complex nested data."""
        complex_config = {
            "chunk_size": 768,
            "retrieval_k": 10,
            "query_routing_rules": {
                "new_rule": ["test", "values"],
                "another_rule": {"nested": "structure"}
            },
            "performance_thresholds": [1.0, 2.5, 4.0]
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            rag_optimizer.update_config(complex_config)
        
        assert rag_optimizer.current_config == complex_config
        mock_print.assert_called_once()
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_update_config_permission_error(self, mock_open_func, rag_optimizer):
        """Tests update_config when file write fails due to permissions."""
        new_config = {"test": "config"}
        
        with pytest.raises(PermissionError, match="Permission denied"):
            rag_optimizer.update_config(new_config)


class TestAnalyzePerformanceData:
    """Tests for analyze_performance_data method."""
    
    def test_analyze_performance_data_success(self, rag_optimizer, mock_db):
        """Tests successful performance data analysis."""
        performance_data = [
            {'query': 'Q1', 'rating': 4.5, 'retrieval_k': 5},
            {'query': 'Q2', 'rating': 2.5, 'retrieval_k': 7},
            {'query': 'Q3', 'rating': 1.8, 'retrieval_k': 5},
            {'query': 'Q4', 'rating': 3.2, 'retrieval_k': 6}
        ]
        mock_db.get_performance_metrics.return_value = performance_data
        
        result = rag_optimizer.analyze_performance_data(time_window_days=14)
        
        mock_db.get_performance_metrics.assert_called_once_with(14)
        
        # Verify analysis results
        assert result['average_rating'] == (4.5 + 2.5 + 1.8 + 3.2) / 4
        assert result['low_rated_retrieval_k'] == [7, 5]  # ratings < 3
        assert result['high_rated_retrieval_k'] == [5, 6]  # ratings >= 3
    
    def test_analyze_performance_data_empty_data(self, rag_optimizer, mock_db):
        """Tests performance data analysis with empty data."""
        mock_db.get_performance_metrics.return_value = []
        
        result = rag_optimizer.analyze_performance_data()
        
        expected = {
            "average_rating": 0,
            "low_rated_retrieval_k": [],
            "high_rated_retrieval_k": []
        }
        assert result == expected
    
    def test_analyze_performance_data_all_low_ratings(self, rag_optimizer, mock_db):
        """Tests performance data analysis with all low ratings."""
        performance_data = [
            {'query': 'Q1', 'rating': 1.0, 'retrieval_k': 3},
            {'query': 'Q2', 'rating': 2.5, 'retrieval_k': 4},
        ]
        mock_db.get_performance_metrics.return_value = performance_data
        
        result = rag_optimizer.analyze_performance_data()
        
        assert result['average_rating'] == 1.75
        assert result['low_rated_retrieval_k'] == [3, 4]
        assert result['high_rated_retrieval_k'] == []
    
    def test_analyze_performance_data_all_high_ratings(self, rag_optimizer, mock_db):
        """Tests performance data analysis with all high ratings."""
        performance_data = [
            {'query': 'Q1', 'rating': 4.0, 'retrieval_k': 5},
            {'query': 'Q2', 'rating': 4.8, 'retrieval_k': 6},
        ]
        mock_db.get_performance_metrics.return_value = performance_data
        
        result = rag_optimizer.analyze_performance_data()
        
        assert result['average_rating'] == 4.4
        assert result['low_rated_retrieval_k'] == []
        assert result['high_rated_retrieval_k'] == [5, 6]
    
    def test_analyze_performance_data_default_time_window(self, rag_optimizer, mock_db):
        """Tests performance data analysis with default time window."""
        mock_db.get_performance_metrics.return_value = []
        
        rag_optimizer.analyze_performance_data()
        
        mock_db.get_performance_metrics.assert_called_once_with(7)
    
    def test_analyze_performance_data_database_error(self, rag_optimizer, mock_db):
        """Tests performance data analysis when database query fails."""
        error_message = "Database query timeout"
        mock_db.get_performance_metrics.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            rag_optimizer.analyze_performance_data()


class TestTuneRetrievalParameters:
    """Tests for tune_retrieval_parameters method."""
    
    @patch('builtins.print')
    def test_tune_retrieval_parameters_increase_k_for_low_rating(self, mock_print, rag_optimizer):
        """Tests increasing retrieval k for low average rating."""
        rag_optimizer.current_config['retrieval_k'] = 5
        performance_analysis = {'average_rating': 3.0}  # Below 3.5 threshold
        
        result = rag_optimizer.tune_retrieval_parameters(performance_analysis)
        
        assert result is True
        assert rag_optimizer.current_config['retrieval_k'] == 6
        mock_print.assert_called_once()
        print_call = mock_print.call_args[0][0]
        assert "düşük" in print_call and "6'ye yükseltiliyor" in print_call
    
    @patch('builtins.print')
    def test_tune_retrieval_parameters_decrease_k_for_high_rating(self, mock_print, rag_optimizer):
        """Tests decreasing retrieval k for high average rating."""
        rag_optimizer.current_config['retrieval_k'] = 8
        performance_analysis = {'average_rating': 4.8}  # Above 4.5 threshold
        
        result = rag_optimizer.tune_retrieval_parameters(performance_analysis)
        
        assert result is True
        assert rag_optimizer.current_config['retrieval_k'] == 7
        mock_print.assert_called_once()
        print_call = mock_print.call_args[0][0]
        assert "yüksek" in print_call and "7'ye düşürülüyor" in print_call
    
    def test_tune_retrieval_parameters_no_change_stable_rating(self, rag_optimizer):
        """Tests no parameter change for stable rating."""
        rag_optimizer.current_config['retrieval_k'] = 6
        performance_analysis = {'average_rating': 4.0}  # Between thresholds
        
        result = rag_optimizer.tune_retrieval_parameters(performance_analysis)
        
        assert result is False
        assert rag_optimizer.current_config['retrieval_k'] == 6
    
    def test_tune_retrieval_parameters_max_k_limit(self, rag_optimizer):
        """Tests that k doesn't increase beyond maximum limit."""
        rag_optimizer.current_config['retrieval_k'] = 10  # At max limit
        performance_analysis = {'average_rating': 2.0}  # Very low
        
        result = rag_optimizer.tune_retrieval_parameters(performance_analysis)
        
        assert result is False
        assert rag_optimizer.current_config['retrieval_k'] == 10
    
    def test_tune_retrieval_parameters_min_k_limit(self, rag_optimizer):
        """Tests that k doesn't decrease below minimum limit."""
        rag_optimizer.current_config['retrieval_k'] = 3  # At min limit
        performance_analysis = {'average_rating': 5.0}  # Very high
        
        result = rag_optimizer.tune_retrieval_parameters(performance_analysis)
        
        assert result is False
        assert rag_optimizer.current_config['retrieval_k'] == 3
    
    def test_tune_retrieval_parameters_missing_average_rating(self, rag_optimizer):
        """Tests parameter tuning with missing average rating."""
        rag_optimizer.current_config['retrieval_k'] = 5
        performance_analysis = {}  # Missing average_rating (defaults to 0)
        
        with patch('builtins.print'):
            result = rag_optimizer.tune_retrieval_parameters(performance_analysis)
        
        # Missing average_rating defaults to 0, which is < 3.5, so k gets increased
        assert result is True
        assert rag_optimizer.current_config['retrieval_k'] == 6
    
    def test_tune_retrieval_parameters_boundary_values(self, rag_optimizer):
        """Tests parameter tuning at exact boundary values."""
        rag_optimizer.current_config['retrieval_k'] = 5
        
        # Exactly at low threshold
        performance_analysis = {'average_rating': 3.5}
        result = rag_optimizer.tune_retrieval_parameters(performance_analysis)
        assert result is False  # Should not trigger increase
        
        # Exactly at high threshold
        performance_analysis = {'average_rating': 4.5}
        result = rag_optimizer.tune_retrieval_parameters(performance_analysis)
        assert result is False  # Should not trigger decrease


class TestOptimizeRoutingRules:
    """Tests for optimize_routing_rules method."""
    
    @patch('builtins.print')
    def test_optimize_routing_rules_not_implemented(self, mock_print, rag_optimizer):
        """Tests that routing rules optimization is not yet implemented."""
        feedback_data = [
            {'query': 'Test query', 'rating': 3, 'topic': 'Test'}
        ]
        
        rag_optimizer.optimize_routing_rules(feedback_data)
        
        mock_print.assert_called_once_with("Sorgu yönlendirme kuralları optimizasyonu henüz implemente edilmedi.")
    
    def test_optimize_routing_rules_empty_feedback(self, rag_optimizer):
        """Tests routing rules optimization with empty feedback."""
        with patch('builtins.print') as mock_print:
            rag_optimizer.optimize_routing_rules([])
            mock_print.assert_called_once()
    
    def test_optimize_routing_rules_none_feedback(self, rag_optimizer):
        """Tests routing rules optimization with None feedback."""
        with patch('builtins.print') as mock_print:
            rag_optimizer.optimize_routing_rules(None)
            mock_print.assert_called_once()


class TestRunOptimizationCycle:
    """Tests for run_optimization_cycle method."""
    
    @patch('builtins.print')
    @patch('src.services.rag_optimizer.RAGParameterOptimizer.update_config')
    def test_run_optimization_cycle_success(self, mock_update_config, mock_print, rag_optimizer, mock_db):
        """Tests successful optimization cycle execution."""
        # Setup mock data
        performance_data = [
            {'query': 'Q1', 'rating': 2.5, 'retrieval_k': 5},
            {'query': 'Q2', 'rating': 2.0, 'retrieval_k': 5}
        ]
        feedback_data = [{'query': 'F1', 'rating': 2, 'topic': 'Topic1'}]
        
        mock_db.get_performance_metrics.return_value = performance_data
        mock_db.get_feedback_since.return_value = feedback_data
        
        original_k = rag_optimizer.current_config['retrieval_k']
        
        result = rag_optimizer.run_optimization_cycle()
        
        # Verify database calls
        mock_db.get_performance_metrics.assert_called_once_with(7)
        mock_db.get_feedback_since.assert_called_once_with(days=7)
        
        # Verify config update was called (since performance was low)
        mock_update_config.assert_called_once_with(rag_optimizer.current_config)
        
        # Verify return value
        assert result == rag_optimizer.current_config
        
        # Verify print statements
        assert mock_print.call_count >= 3  # Start, analysis, completion messages
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        start_msgs = [msg for msg in print_calls if "Başlatılıyor" in msg]
        analysis_msgs = [msg for msg in print_calls if "Performans Analizi Tamamlandı" in msg]
        completion_msgs = [msg for msg in print_calls if "Tamamlandı" in msg]
        
        assert len(start_msgs) >= 1
        assert len(analysis_msgs) >= 1
        assert len(completion_msgs) >= 1
    
    @patch('builtins.print')
    @patch('src.services.rag_optimizer.RAGParameterOptimizer.update_config')
    def test_run_optimization_cycle_no_config_changes(self, mock_update_config, mock_print, rag_optimizer, mock_db):
        """Tests optimization cycle with no configuration changes needed."""
        # Setup data for stable performance
        performance_data = [
            {'query': 'Q1', 'rating': 4.0, 'retrieval_k': 5},
            {'query': 'Q2', 'rating': 4.2, 'retrieval_k': 5}
        ]
        feedback_data = []
        
        mock_db.get_performance_metrics.return_value = performance_data
        mock_db.get_feedback_since.return_value = feedback_data
        
        result = rag_optimizer.run_optimization_cycle()
        
        # Config should not be updated since no changes were needed
        mock_update_config.assert_not_called()
        
        # Should still return current config
        assert result == rag_optimizer.current_config
    
    @patch('builtins.print')
    def test_run_optimization_cycle_empty_performance_data(self, mock_print, rag_optimizer, mock_db):
        """Tests optimization cycle with empty performance data."""
        mock_db.get_performance_metrics.return_value = []
        mock_db.get_feedback_since.return_value = []
        
        result = rag_optimizer.run_optimization_cycle()
        
        assert result == rag_optimizer.current_config
        
        # Should still print analysis completion with 0.00 rating (not N/A)
        analysis_msgs = [call for call in mock_print.call_args_list
                        if len(call[0]) > 0 and "0.00" in call[0][0]]
        assert len(analysis_msgs) >= 1
    
    def test_run_optimization_cycle_database_error(self, rag_optimizer, mock_db):
        """Tests optimization cycle when database error occurs."""
        error_message = "Database connection lost"
        mock_db.get_performance_metrics.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            rag_optimizer.run_optimization_cycle()


class TestRAGParameterOptimizerIntegration:
    """Integration tests for RAGParameterOptimizer workflows."""
    
    @patch('builtins.print')
    def test_full_optimization_workflow_with_improvements(self, mock_print, rag_optimizer, mock_db):
        """Tests a complete optimization workflow that results in improvements."""
        # Setup declining performance scenario
        performance_data = [
            {'query': 'Q1', 'rating': 1.5, 'retrieval_k': 5},
            {'query': 'Q2', 'rating': 2.0, 'retrieval_k': 5},
            {'query': 'Q3', 'rating': 2.8, 'retrieval_k': 5}
        ]
        feedback_data = [
            {'query': 'F1', 'rating': 1, 'topic': 'Problem Topic'},
            {'query': 'F2', 'rating': 2, 'topic': 'Problem Topic'}
        ]
        
        mock_db.get_performance_metrics.return_value = performance_data
        mock_db.get_feedback_since.return_value = feedback_data
        
        original_config = rag_optimizer.current_config.copy()
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = rag_optimizer.run_optimization_cycle()
        
        # Verify performance analysis
        analysis = rag_optimizer.analyze_performance_data(7)
        expected_avg = (1.5 + 2.0 + 2.8) / 3
        assert abs(analysis['average_rating'] - expected_avg) < 0.01
        
        # Verify parameter tuning occurred
        assert rag_optimizer.current_config['retrieval_k'] == original_config['retrieval_k'] + 1
        
        # Verify file write occurred
        mock_file.assert_called_once()
        
        # Verify comprehensive logging
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Başlatılıyor" in msg for msg in print_calls)
        assert any("Performans Analizi" in msg for msg in print_calls)
        assert any("Tamamlandı" in msg for msg in print_calls)
    
    def test_optimization_cycle_stability(self, rag_optimizer, mock_db):
        """Tests multiple optimization cycles maintain system stability."""
        # Setup stable performance data
        stable_data = [
            {'query': f'Q{i}', 'rating': 4.0, 'retrieval_k': 5}
            for i in range(1, 6)
        ]
        mock_db.get_performance_metrics.return_value = stable_data
        mock_db.get_feedback_since.return_value = []
        
        original_config = rag_optimizer.current_config.copy()
        
        # Run multiple cycles
        for i in range(3):
            with patch('builtins.print'):
                result = rag_optimizer.run_optimization_cycle()
            
            # Configuration should remain stable
            assert rag_optimizer.current_config == original_config
            assert result == original_config


class TestRAGParameterOptimizerEdgeCases:
    """Edge cases and boundary condition tests."""
    
    def test_performance_data_with_extreme_values(self, rag_optimizer, mock_db):
        """Tests handling of extreme performance values."""
        extreme_data = [
            {'query': 'Q1', 'rating': 0.0, 'retrieval_k': 1},
            {'query': 'Q2', 'rating': 5.0, 'retrieval_k': 20},
            {'query': 'Q3', 'rating': -1.0, 'retrieval_k': 0}
        ]
        mock_db.get_performance_metrics.return_value = extreme_data
        
        result = rag_optimizer.analyze_performance_data()
        
        # Should handle extreme values gracefully
        assert 'average_rating' in result
        assert 'low_rated_retrieval_k' in result
        assert 'high_rated_retrieval_k' in result
        assert isinstance(result['average_rating'], (int, float))
    
    def test_config_with_missing_keys(self, mock_db):
        """Tests handling of configuration with missing keys."""
        incomplete_config = {"chunk_size": 256}  # Missing retrieval_k
        
        with patch('builtins.open', mock_open(read_data=json.dumps(incomplete_config))):
            optimizer = RAGParameterOptimizer(db_connection=mock_db)
        
        # Should handle missing keys gracefully
        performance_analysis = {'average_rating': 2.0}
        
        with pytest.raises(KeyError):
            optimizer.tune_retrieval_parameters(performance_analysis)
    
    def test_very_large_time_windows(self, rag_optimizer, mock_db):
        """Tests performance analysis with very large time windows."""
        mock_db.get_performance_metrics.return_value = []
        
        # Should handle large time windows without issues
        result = rag_optimizer.analyze_performance_data(time_window_days=36500)  # 100 years
        
        mock_db.get_performance_metrics.assert_called_once_with(36500)
        assert result['average_rating'] == 0
    
    def test_zero_time_window(self, rag_optimizer, mock_db):
        """Tests performance analysis with zero time window."""
        mock_db.get_performance_metrics.return_value = []
        
        result = rag_optimizer.analyze_performance_data(time_window_days=0)
        
        mock_db.get_performance_metrics.assert_called_once_with(0)
        assert result['average_rating'] == 0
    
    def test_negative_time_window(self, rag_optimizer, mock_db):
        """Tests performance analysis with negative time window."""
        mock_db.get_performance_metrics.return_value = []
        
        result = rag_optimizer.analyze_performance_data(time_window_days=-5)
        
        mock_db.get_performance_metrics.assert_called_once_with(-5)
        assert result['average_rating'] == 0


class TestRAGParameterOptimizerErrorHandling:
    """Error handling and exception management tests."""
    
    @patch('builtins.open', side_effect=IOError("Disk full"))
    def test_update_config_io_error(self, mock_open_func, rag_optimizer):
        """Tests update_config when disk write fails."""
        new_config = {"test": "config"}
        
        with pytest.raises(IOError, match="Disk full"):
            rag_optimizer.update_config(new_config)
    
    def test_analyze_performance_data_malformed_data(self, rag_optimizer, mock_db):
        """Tests performance analysis with malformed data."""
        malformed_data = [
            {'query': 'Q1'},  # Missing rating and retrieval_k
            {'rating': 'invalid'},  # Invalid rating type
            {'query': 'Q3', 'rating': 4.0, 'retrieval_k': 'invalid'},  # Invalid k type
        ]
        mock_db.get_performance_metrics.return_value = malformed_data
        
        # Should handle malformed data gracefully or raise appropriate error
        with pytest.raises((KeyError, TypeError)):
            rag_optimizer.analyze_performance_data()
    
    def test_tune_retrieval_parameters_invalid_analysis_data(self, rag_optimizer):
        """Tests parameter tuning with invalid analysis data."""
        # String comparison raises TypeError
        string_analysis = {'average_rating': 'invalid'}
        with pytest.raises(TypeError):
            rag_optimizer.tune_retrieval_parameters(string_analysis)
        
        # inf and nan are valid float values that the service handles
        with patch('builtins.print'):
            inf_analysis = {'average_rating': float('inf')}
            result_inf = rag_optimizer.tune_retrieval_parameters(inf_analysis)
            assert isinstance(result_inf, bool)
            
            nan_analysis = {'average_rating': float('nan')}
            result_nan = rag_optimizer.tune_retrieval_parameters(nan_analysis)
            assert isinstance(result_nan, bool)


class TestRAGParameterOptimizerLogging:
    """Logging and output verification tests."""
    
    @patch('builtins.print')
    def test_logging_during_optimization_cycle(self, mock_print, rag_optimizer, mock_db):
        """Tests comprehensive logging during optimization cycle."""
        mock_db.get_performance_metrics.return_value = [
            {'query': 'Q1', 'rating': 3.0, 'retrieval_k': 5}
        ]
        mock_db.get_feedback_since.return_value = []
        
        rag_optimizer.run_optimization_cycle()
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        # Verify specific log messages exist
        start_messages = [msg for msg in print_calls if "RAG Optimizasyon Döngüsü Başlatılıyor" in msg]
        analysis_messages = [msg for msg in print_calls if "Performans Analizi Tamamlandı" in msg]
        completion_messages = [msg for msg in print_calls if "Optimizasyon Döngüsü Tamamlandı" in msg]
        
        assert len(start_messages) == 1
        assert len(analysis_messages) == 1  
        assert len(completion_messages) == 1
    
    @patch('builtins.print')
    def test_logging_parameter_changes(self, mock_print, rag_optimizer):
        """Tests logging of parameter changes."""
        rag_optimizer.current_config['retrieval_k'] = 5
        
        # Test increase
        performance_analysis = {'average_rating': 2.0}
        rag_optimizer.tune_retrieval_parameters(performance_analysis)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        increase_messages = [msg for msg in print_calls if "yükseltiliyor" in msg]
        assert len(increase_messages) == 1
        
        mock_print.reset_mock()
        
        # Test decrease
        rag_optimizer.current_config['retrieval_k'] = 8
        performance_analysis = {'average_rating': 5.0}
        rag_optimizer.tune_retrieval_parameters(performance_analysis)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        decrease_messages = [msg for msg in print_calls if "düşürülüyor" in msg]
        assert len(decrease_messages) == 1
    
    @patch('builtins.print')
    def test_logging_config_updates(self, mock_print, rag_optimizer):
        """Tests logging of configuration updates."""
        new_config = {"test": "config"}
        
        with patch('builtins.open', mock_open()):
            rag_optimizer.update_config(new_config)
        
        mock_print.assert_called_once_with(f"Yapılandırma güncellendi: {rag_optimizer.json_config_path}")


class TestRAGParameterOptimizerPerformanceMetrics:
    """Performance metrics and calculation tests."""
    
    def test_performance_metrics_calculation_accuracy(self, rag_optimizer, mock_db):
        """Tests accuracy of performance metrics calculations."""
        test_data = [
            {'query': 'Q1', 'rating': 1.0, 'retrieval_k': 3},
            {'query': 'Q2', 'rating': 2.5, 'retrieval_k': 4}, 
            {'query': 'Q3', 'rating': 4.0, 'retrieval_k': 5},
            {'query': 'Q4', 'rating': 4.5, 'retrieval_k': 6},
        ]
        mock_db.get_performance_metrics.return_value = test_data
        
        result = rag_optimizer.analyze_performance_data()
        
        # Verify exact calculations
        expected_avg = (1.0 + 2.5 + 4.0 + 4.5) / 4
        assert abs(result['average_rating'] - expected_avg) < 1e-10
        
        # Low ratings (< 3): first two
        assert result['low_rated_retrieval_k'] == [3, 4]
        
        # High ratings (>= 3): last two  
        assert result['high_rated_retrieval_k'] == [5, 6]
    
    def test_performance_metrics_with_floating_point_precision(self, rag_optimizer, mock_db):
        """Tests handling of floating point precision in calculations."""
        precision_data = [
            {'query': 'Q1', 'rating': 3.333333333333333, 'retrieval_k': 5},
            {'query': 'Q2', 'rating': 3.666666666666667, 'retrieval_k': 5},
        ]
        mock_db.get_performance_metrics.return_value = precision_data
        
        result = rag_optimizer.analyze_performance_data()
        
        # Should handle floating point precision correctly
        expected_avg = (3.333333333333333 + 3.666666666666667) / 2
        assert abs(result['average_rating'] - expected_avg) < 1e-10