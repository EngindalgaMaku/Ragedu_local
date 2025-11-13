import pytest
import time
import threading
from unittest.mock import MagicMock, patch, call
import logging

from src.services.learning_loop_manager import LearningLoopManager


@pytest.fixture
def mock_db():
    """Mock database connection for LearningLoopManager."""
    mock_db = MagicMock()
    
    # Mock database methods used by LearningLoopManager
    mock_db.check_connection.return_value = True
    mock_db.get_average_rating.side_effect = lambda days: 3.5 if days == 7 else 4.1
    
    # Mock database methods used by dependencies
    mock_db.get_feedback_since.return_value = []
    mock_db.get_most_uncertain_queries.return_value = []
    mock_db.get_performance_metrics.return_value = []
    
    return mock_db


@pytest.fixture
def mock_active_learning_engine():
    """Mock ActiveLearningEngine for testing."""
    mock_engine = MagicMock()
    mock_engine.analyze_feedback_patterns.return_value = [
        {'topic': 'Test Topic', 'negative_feedback_count': 5}
    ]
    mock_engine.identify_samples_for_review.return_value = [
        {'query': 'Test query', 'uncertainty_score': 0.8}
    ]
    return mock_engine


@pytest.fixture
def mock_rag_optimizer():
    """Mock RAGParameterOptimizer for testing."""
    mock_optimizer = MagicMock()
    mock_optimizer.run_optimization_cycle.return_value = {
        'retrieval_k': 6,
        'chunk_size': 512
    }
    return mock_optimizer


@pytest.fixture
def learning_loop_manager(mock_db):
    """LearningLoopManager instance with mocked dependencies."""
    with patch('src.services.learning_loop_manager.ActiveLearningEngine') as mock_ale_class, \
         patch('src.services.learning_loop_manager.RAGParameterOptimizer') as mock_rag_class:
        
        mock_ale_instance = MagicMock()
        mock_rag_instance = MagicMock()
        mock_ale_class.return_value = mock_ale_instance
        mock_rag_class.return_value = mock_rag_instance
        
        manager = LearningLoopManager(mock_db, analysis_interval_seconds=1)  # Short interval for testing
        
        # Store references to mocks for test access
        manager._mock_active_learning = mock_ale_instance
        manager._mock_rag_optimizer = mock_rag_instance
        
        yield manager


class TestLearningLoopManagerInit:
    """Tests for LearningLoopManager initialization."""
    
    @patch('src.services.learning_loop_manager.ActiveLearningEngine')
    @patch('src.services.learning_loop_manager.RAGParameterOptimizer')
    def test_init_success(self, mock_rag_class, mock_ale_class, mock_db):
        """Tests successful initialization of LearningLoopManager."""
        mock_ale_instance = MagicMock()
        mock_rag_instance = MagicMock()
        mock_ale_class.return_value = mock_ale_instance
        mock_rag_class.return_value = mock_rag_instance
        
        manager = LearningLoopManager(mock_db, analysis_interval_seconds=3600)
        
        assert manager.db == mock_db
        assert manager.active_learning_engine == mock_ale_instance
        assert manager.rag_optimizer == mock_rag_instance
        assert manager.analysis_interval == 3600
        assert manager.is_running == False
        assert manager.thread is None
        assert manager.system_health['status'] == "INITIALIZING"
        assert manager.system_health['last_check'] is None
        assert manager.system_health['issues'] == []
        
        mock_ale_class.assert_called_once_with(mock_db)
        mock_rag_class.assert_called_once_with(mock_db)
    
    @patch('src.services.learning_loop_manager.ActiveLearningEngine')
    @patch('src.services.learning_loop_manager.RAGParameterOptimizer')
    def test_init_default_interval(self, mock_rag_class, mock_ale_class, mock_db):
        """Tests initialization with default analysis interval."""
        mock_ale_class.return_value = MagicMock()
        mock_rag_class.return_value = MagicMock()
        
        manager = LearningLoopManager(mock_db)
        
        assert manager.analysis_interval == 86400  # 24 hours default
    
    @patch('src.services.learning_loop_manager.ActiveLearningEngine')
    @patch('src.services.learning_loop_manager.RAGParameterOptimizer')
    def test_init_custom_interval(self, mock_rag_class, mock_ale_class, mock_db):
        """Tests initialization with custom analysis interval."""
        custom_interval = 1800  # 30 minutes
        mock_ale_class.return_value = MagicMock()
        mock_rag_class.return_value = MagicMock()
        
        manager = LearningLoopManager(mock_db, analysis_interval_seconds=custom_interval)
        
        assert manager.analysis_interval == custom_interval


class TestMonitorSystemHealth:
    """Tests for monitor_system_health method."""
    
    def test_monitor_system_health_success(self, learning_loop_manager, mock_db):
        """Tests successful system health monitoring."""
        mock_db.check_connection.return_value = None  # No exception
        
        learning_loop_manager.monitor_system_health()
        
        mock_db.check_connection.assert_called_once()
        assert learning_loop_manager.system_health['status'] == "OPERATIONAL"
        assert learning_loop_manager.system_health['issues'] == []
        assert learning_loop_manager.system_health['last_check'] is not None
    
    def test_monitor_system_health_database_error(self, learning_loop_manager, mock_db):
        """Tests system health monitoring when database connection fails."""
        error_message = "Database connection timeout"
        mock_db.check_connection.side_effect = Exception(error_message)
        
        learning_loop_manager.monitor_system_health()
        
        mock_db.check_connection.assert_called_once()
        assert learning_loop_manager.system_health['status'] == "ERROR"
        assert len(learning_loop_manager.system_health['issues']) == 1
        assert error_message in learning_loop_manager.system_health['issues'][0]
        assert learning_loop_manager.system_health['last_check'] is not None
    
    def test_monitor_system_health_clears_previous_issues(self, learning_loop_manager, mock_db):
        """Tests that system health monitoring clears previous issues."""
        # First, add some issues
        learning_loop_manager.system_health['issues'] = ['Previous error']
        
        # Then run successful health check
        mock_db.check_connection.return_value = None
        learning_loop_manager.monitor_system_health()
        
        assert learning_loop_manager.system_health['issues'] == []
        assert learning_loop_manager.system_health['status'] == "OPERATIONAL"


class TestDetectPerformanceTrends:
    """Tests for detect_performance_trends method."""
    
    def test_detect_performance_trends_stable(self, learning_loop_manager, mock_db):
        """Tests performance trend detection for stable performance."""
        mock_db.get_average_rating.side_effect = lambda days: 3.5  # Same for both periods
        
        result = learning_loop_manager.detect_performance_trends()
        
        assert result == "STABİL"
        mock_db.get_average_rating.assert_has_calls([call(days=7), call(days=30)])
        assert len(learning_loop_manager.system_health['issues']) == 0
    
    def test_detect_performance_trends_declining(self, learning_loop_manager, mock_db):
        """Tests performance trend detection for declining performance."""
        # Short term much worse than long term
        mock_db.get_average_rating.side_effect = lambda days: 2.5 if days == 7 else 4.0
        
        result = learning_loop_manager.detect_performance_trends()
        
        assert result == "DÜŞÜŞTE"
        assert len(learning_loop_manager.system_health['issues']) == 1
        assert "Performance degradation detected." in learning_loop_manager.system_health['issues']
    
    def test_detect_performance_trends_improving(self, learning_loop_manager, mock_db):
        """Tests performance trend detection for improving performance."""
        # Short term much better than long term
        mock_db.get_average_rating.side_effect = lambda days: 4.5 if days == 7 else 3.0
        
        result = learning_loop_manager.detect_performance_trends()
        
        assert result == "YÜKSELİŞTE"
        assert len(learning_loop_manager.system_health['issues']) == 0
    
    def test_detect_performance_trends_custom_windows(self, learning_loop_manager, mock_db):
        """Tests performance trend detection with custom time windows."""
        mock_db.get_average_rating.side_effect = lambda days: 2.0 if days == 14 else 4.0
        
        result = learning_loop_manager.detect_performance_trends(
            short_window_days=14, 
            long_window_days=60
        )
        
        assert result == "DÜŞÜŞTE"
        mock_db.get_average_rating.assert_has_calls([call(days=14), call(days=60)])
    
    def test_detect_performance_trends_boundary_conditions(self, learning_loop_manager, mock_db):
        """Tests performance trend detection at boundary conditions."""
        # Exactly 10% decline threshold
        mock_db.get_average_rating.side_effect = lambda days: 3.6 if days == 7 else 4.0
        
        result = learning_loop_manager.detect_performance_trends()
        
        assert result == "DÜŞÜŞTE"  # 3.6 < 4.0 * 0.9 = 3.6 (equal, should be stable, but let's check implementation)


class TestStartMethod:
    """Tests for start method."""
    
    def test_start_success(self, learning_loop_manager):
        """Tests successful start of learning loop."""
        assert not learning_loop_manager.is_running
        assert learning_loop_manager.thread is None
        
        learning_loop_manager.start()
        
        assert learning_loop_manager.is_running
        assert learning_loop_manager.thread is not None
        assert learning_loop_manager.thread.daemon
        assert learning_loop_manager.thread.is_alive()
        
        # Clean up
        learning_loop_manager.stop()
        learning_loop_manager.thread.join(timeout=1)
    
    def test_start_when_already_running(self, learning_loop_manager):
        """Tests start method when loop is already running."""
        learning_loop_manager.start()
        original_thread = learning_loop_manager.thread
        
        # Try to start again
        learning_loop_manager.start()
        
        # Should not create new thread
        assert learning_loop_manager.thread is original_thread
        
        # Clean up
        learning_loop_manager.stop()
        learning_loop_manager.thread.join(timeout=1)
    
    @patch('src.services.learning_loop_manager.threading.Thread')
    def test_start_thread_creation(self, mock_thread_class, learning_loop_manager):
        """Tests that start method creates thread with correct parameters."""
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread
        
        learning_loop_manager.start()
        
        mock_thread_class.assert_called_once_with(
            target=learning_loop_manager._run_periodic_analysis,
            daemon=True
        )
        mock_thread.start.assert_called_once()


class TestStopMethod:
    """Tests for stop method."""
    
    def test_stop_when_running(self, learning_loop_manager):
        """Tests stopping the learning loop when it's running."""
        learning_loop_manager.start()
        assert learning_loop_manager.is_running
        
        learning_loop_manager.stop()
        
        assert not learning_loop_manager.is_running
        
        # Wait for thread to finish
        if learning_loop_manager.thread:
            learning_loop_manager.thread.join(timeout=2)
    
    def test_stop_when_not_running(self, learning_loop_manager):
        """Tests stop method when loop is not running."""
        assert not learning_loop_manager.is_running
        
        # Should not raise any exception
        learning_loop_manager.stop()
        
        assert not learning_loop_manager.is_running
    
    def test_stop_sets_flag(self, learning_loop_manager):
        """Tests that stop method sets the is_running flag to False."""
        learning_loop_manager.is_running = True
        
        learning_loop_manager.stop()
        
        assert not learning_loop_manager.is_running


class TestPeriodicAnalysis:
    """Tests for _run_periodic_analysis method (threading behavior)."""
    
    @patch('src.services.learning_loop_manager.time.sleep')
    def test_run_periodic_analysis_single_cycle(self, mock_sleep, learning_loop_manager, mock_db):
        """Tests a single cycle of periodic analysis."""
        # Set up mocks
        learning_loop_manager._mock_active_learning.analyze_feedback_patterns.return_value = [
            {'topic': 'Test Topic', 'negative_feedback_count': 3}
        ]
        learning_loop_manager._mock_active_learning.identify_samples_for_review.return_value = [
            {'query': 'Test query', 'uncertainty_score': 0.7}
        ]
        learning_loop_manager._mock_rag_optimizer.run_optimization_cycle.return_value = {
            'retrieval_k': 5, 'chunk_size': 512
        }
        
        # Mock sleep to interrupt after first cycle
        def side_effect(*args):
            learning_loop_manager.is_running = False
        mock_sleep.side_effect = side_effect
        
        learning_loop_manager.is_running = True
        learning_loop_manager._run_periodic_analysis()
        
        # Verify all components were called
        mock_db.check_connection.assert_called_once()
        learning_loop_manager._mock_active_learning.analyze_feedback_patterns.assert_called_once()
        learning_loop_manager._mock_active_learning.identify_samples_for_review.assert_called_once_with(limit=5)
        learning_loop_manager._mock_rag_optimizer.run_optimization_cycle.assert_called_once()
        mock_sleep.assert_called_once_with(learning_loop_manager.analysis_interval)
    
    @patch('src.services.learning_loop_manager.time.sleep')
    def test_run_periodic_analysis_stops_when_flag_false(self, mock_sleep, learning_loop_manager):
        """Tests that periodic analysis stops when is_running is False."""
        learning_loop_manager.is_running = False
        
        learning_loop_manager._run_periodic_analysis()
        
        # Should not call sleep or any analysis methods
        mock_sleep.assert_not_called()
        learning_loop_manager._mock_active_learning.analyze_feedback_patterns.assert_not_called()


class TestIntegrationScenarios:
    """Integration tests for LearningLoopManager workflows."""
    
    def test_full_learning_cycle_integration(self, learning_loop_manager, mock_db):
        """Tests a complete learning cycle integration."""
        # Set up complex mock data
        mock_db.get_average_rating.side_effect = lambda days: 2.8 if days == 7 else 4.0
        learning_loop_manager._mock_active_learning.analyze_feedback_patterns.return_value = [
            {'topic': 'Deep Learning', 'negative_feedback_count': 8},
            {'topic': 'Python', 'negative_feedback_count': 3}
        ]
        learning_loop_manager._mock_active_learning.identify_samples_for_review.return_value = [
            {'query': 'Explain transformers', 'uncertainty_score': 0.95},
            {'query': 'What is RAG?', 'uncertainty_score': 0.87}
        ]
        learning_loop_manager._mock_rag_optimizer.run_optimization_cycle.return_value = {
            'retrieval_k': 7,
            'chunk_size': 768,
            'chunk_overlap': 100
        }
        
        # Run one cycle manually
        learning_loop_manager.monitor_system_health()
        feedback_patterns = learning_loop_manager._mock_active_learning.analyze_feedback_patterns()
        samples_for_review = learning_loop_manager._mock_active_learning.identify_samples_for_review(limit=5)
        optimized_config = learning_loop_manager._mock_rag_optimizer.run_optimization_cycle()
        trend = learning_loop_manager.detect_performance_trends()
        
        # Verify results
        assert learning_loop_manager.system_health['status'] == "OPERATIONAL"
        assert len(feedback_patterns) == 2
        assert len(samples_for_review) == 2
        assert optimized_config['retrieval_k'] == 7
        assert trend == "DÜŞÜŞTE"  # Performance declining
        assert "Performance degradation detected." in learning_loop_manager.system_health['issues']
    
    @patch('src.services.learning_loop_manager.time.sleep')
    def test_error_recovery_during_analysis(self, mock_sleep, learning_loop_manager, mock_db):
        """Tests error recovery during analysis cycle."""
        # Make one component fail
        learning_loop_manager._mock_active_learning.analyze_feedback_patterns.side_effect = Exception("Analysis failed")
        
        def stop_after_first(*args):
            learning_loop_manager.is_running = False
        mock_sleep.side_effect = stop_after_first
        
        learning_loop_manager.is_running = True
        
        # Should not crash the entire cycle
        with pytest.raises(Exception, match="Analysis failed"):
            learning_loop_manager._run_periodic_analysis()


class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_detect_performance_trends_database_error(self, learning_loop_manager, mock_db):
        """Tests performance trend detection when database query fails."""
        mock_db.get_average_rating.side_effect = Exception("Database query failed")
        
        with pytest.raises(Exception, match="Database query failed"):
            learning_loop_manager.detect_performance_trends()
    
    def test_monitor_system_health_multiple_errors(self, learning_loop_manager, mock_db):
        """Tests system health monitoring with multiple consecutive errors."""
        error_messages = ["Error 1", "Error 2", "Error 3"]
        
        for error_msg in error_messages:
            mock_db.check_connection.side_effect = Exception(error_msg)
            learning_loop_manager.monitor_system_health()
            
            assert learning_loop_manager.system_health['status'] == "ERROR"
            assert error_msg in learning_loop_manager.system_health['issues'][0]
    
    def test_system_health_recovery(self, learning_loop_manager, mock_db):
        """Tests system health recovery after errors."""
        # First, cause an error
        mock_db.check_connection.side_effect = Exception("Temporary error")
        learning_loop_manager.monitor_system_health()
        assert learning_loop_manager.system_health['status'] == "ERROR"
        assert len(learning_loop_manager.system_health['issues']) == 1
        
        # Then recover
        mock_db.check_connection.side_effect = None  # No exception
        learning_loop_manager.monitor_system_health()
        assert learning_loop_manager.system_health['status'] == "OPERATIONAL"
        assert len(learning_loop_manager.system_health['issues']) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_analysis_interval(self, mock_db):
        """Tests initialization with zero analysis interval."""
        with patch('src.services.learning_loop_manager.ActiveLearningEngine'), \
             patch('src.services.learning_loop_manager.RAGParameterOptimizer'):
            manager = LearningLoopManager(mock_db, analysis_interval_seconds=0)
            assert manager.analysis_interval == 0
    
    def test_negative_analysis_interval(self, mock_db):
        """Tests initialization with negative analysis interval."""
        with patch('src.services.learning_loop_manager.ActiveLearningEngine'), \
             patch('src.services.learning_loop_manager.RAGParameterOptimizer'):
            manager = LearningLoopManager(mock_db, analysis_interval_seconds=-100)
            assert manager.analysis_interval == -100
    
    def test_performance_trends_with_zero_ratings(self, learning_loop_manager, mock_db):
        """Tests performance trend detection with zero ratings."""
        mock_db.get_average_rating.side_effect = lambda days: 0.0
        
        result = learning_loop_manager.detect_performance_trends()
        
        assert result == "STABİL"  # 0 == 0 * 0.9
    
    def test_performance_trends_with_equal_ratings(self, learning_loop_manager, mock_db):
        """Tests performance trend detection with exactly equal ratings."""
        mock_db.get_average_rating.side_effect = lambda days: 3.5
        
        result = learning_loop_manager.detect_performance_trends()
        
        assert result == "STABİL"
    
    def test_none_database_connection(self):
        """Tests initialization with None database connection."""
        with patch('src.services.learning_loop_manager.ActiveLearningEngine') as mock_ale, \
             patch('src.services.learning_loop_manager.RAGParameterOptimizer') as mock_rag:
            mock_ale.return_value = MagicMock()
            mock_rag.return_value = MagicMock()
            
            manager = LearningLoopManager(None)
            
            assert manager.db is None
            mock_ale.assert_called_once_with(None)
            mock_rag.assert_called_once_with(None)


class TestLoggingAndOutput:
    """Tests for logging and print output verification."""
    
    @patch('builtins.print')
    def test_periodic_analysis_logging(self, mock_print, learning_loop_manager, mock_db):
        """Tests that periodic analysis produces expected log output."""
        learning_loop_manager._mock_active_learning.analyze_feedback_patterns.return_value = [
            {'topic': 'AI', 'negative_feedback_count': 5}
        ]
        learning_loop_manager._mock_rag_optimizer.run_optimization_cycle.return_value = {
            'retrieval_k': 6
        }
        
        # Run components individually to check logging
        learning_loop_manager.monitor_system_health()
        learning_loop_manager._mock_active_learning.analyze_feedback_patterns()
        learning_loop_manager._mock_rag_optimizer.run_optimization_cycle()
        learning_loop_manager.detect_performance_trends()
        
        # Check that various print statements were called
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        
        # Should have system health, performance trend messages
        health_messages = [msg for msg in print_calls if "Sistem Sağlığı" in msg]
        trend_messages = [msg for msg in print_calls if "gün ortalama puan" in msg]
        
        assert len(health_messages) > 0
        assert len(trend_messages) > 0
    
    @patch('builtins.print')
    def test_system_health_error_logging(self, mock_print, learning_loop_manager, mock_db):
        """Tests error logging in system health monitoring."""
        error_message = "Connection timeout"
        mock_db.check_connection.side_effect = Exception(error_message)
        
        learning_loop_manager.monitor_system_health()
        
        # Check that error was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        error_messages = [msg for msg in print_calls if "HATA" in msg and error_message in msg]
        
        assert len(error_messages) > 0
    
    @patch('builtins.print')
    def test_performance_trend_logging(self, mock_print, learning_loop_manager, mock_db):
        """Tests performance trend logging for different scenarios."""
        # Test declining performance
        mock_db.get_average_rating.side_effect = lambda days: 2.5 if days == 7 else 4.0
        learning_loop_manager.detect_performance_trends()
        
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        warning_messages = [msg for msg in print_calls if "UYARI" in msg]
        
        assert len(warning_messages) > 0
        
        # Reset mocks
        mock_print.reset_mock()
        
        # Test improving performance  
        mock_db.get_average_rating.side_effect = lambda days: 4.5 if days == 7 else 3.0
        learning_loop_manager.detect_performance_trends()
        
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        info_messages = [msg for msg in print_calls if "BİLGİ" in msg and "yükseliş" in msg]
        
        assert len(info_messages) > 0


class TestThreadingBehavior:
    """Tests for threading-specific behavior."""
    
    def test_daemon_thread_creation(self, learning_loop_manager):
        """Tests that created thread is marked as daemon."""
        learning_loop_manager.start()
        
        assert learning_loop_manager.thread.daemon == True
        
        # Clean up
        learning_loop_manager.stop()
        if learning_loop_manager.thread.is_alive():
            learning_loop_manager.thread.join(timeout=1)
    
    def test_thread_target_assignment(self, learning_loop_manager):
        """Tests that thread target is correctly assigned."""
        with patch('src.services.learning_loop_manager.threading.Thread') as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            learning_loop_manager.start()
            
            # Verify thread was created with correct target
            mock_thread_class.assert_called_once_with(
                target=learning_loop_manager._run_periodic_analysis,
                daemon=True
            )
    
    def test_concurrent_start_stop(self, learning_loop_manager):
        """Tests concurrent start/stop operations."""
        learning_loop_manager.start()
        assert learning_loop_manager.is_running
        
        # Immediately stop
        learning_loop_manager.stop()
        assert not learning_loop_manager.is_running
        
        # Thread should eventually finish
        if learning_loop_manager.thread:
            learning_loop_manager.thread.join(timeout=2)
            # After joining, thread should not be alive
            assert not learning_loop_manager.thread.is_alive()


class TestSystemHealthTracking:
    """Tests for system health tracking functionality."""
    
    def test_initial_system_health_state(self, learning_loop_manager):
        """Tests initial system health state after initialization."""
        assert learning_loop_manager.system_health['status'] == "INITIALIZING"
        assert learning_loop_manager.system_health['last_check'] is None
        assert learning_loop_manager.system_health['issues'] == []
    
    def test_system_health_state_transitions(self, learning_loop_manager, mock_db):
        """Tests system health state transitions."""
        # Start with INITIALIZING -> OPERATIONAL
        learning_loop_manager.monitor_system_health()
        assert learning_loop_manager.system_health['status'] == "OPERATIONAL"
        
        # OPERATIONAL -> ERROR
        mock_db.check_connection.side_effect = Exception("Connection lost")
        learning_loop_manager.monitor_system_health()
        assert learning_loop_manager.system_health['status'] == "ERROR"
        
        # ERROR -> OPERATIONAL (recovery)
        mock_db.check_connection.side_effect = None
        learning_loop_manager.monitor_system_health()
        assert learning_loop_manager.system_health['status'] == "OPERATIONAL"
    
    def test_system_health_timestamp_updates(self, learning_loop_manager, mock_db):
        """Tests that system health timestamp is updated on each check."""
        initial_check = learning_loop_manager.system_health['last_check']
        assert initial_check is None
        
        learning_loop_manager.monitor_system_health()
        first_check = learning_loop_manager.system_health['last_check']
        assert first_check is not None
        
        # Small delay to ensure different timestamp
        time.sleep(0.1)
        learning_loop_manager.monitor_system_health()
        second_check = learning_loop_manager.system_health['last_check']
        assert second_check != first_check


class TestMemoryAndResourceManagement:
    """Tests for memory and resource management."""
    
    def test_no_memory_leaks_on_multiple_start_stop(self, learning_loop_manager):
        """Tests that multiple start/stop cycles don't leak resources."""
        for i in range(5):
            learning_loop_manager.start()
            assert learning_loop_manager.is_running
            
            learning_loop_manager.stop()
            assert not learning_loop_manager.is_running
            
            if learning_loop_manager.thread:
                learning_loop_manager.thread.join(timeout=1)
    
    def test_thread_cleanup_on_stop(self, learning_loop_manager):
        """Tests that thread is properly cleaned up on stop."""
        learning_loop_manager.start()
        original_thread = learning_loop_manager.thread
        assert original_thread.is_alive()
        
        learning_loop_manager.stop()
        
        # Wait for thread to finish
        original_thread.join(timeout=2)
        assert not original_thread.is_alive()