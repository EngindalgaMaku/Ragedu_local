import pytest
import sqlite3
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import asdict
import threading
import time
from collections import defaultdict

from src.analytics.performance_tracker import (
    PerformanceMetric, 
    ChainPerformanceStats, 
    PerformanceTracker
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_performance.db"
    yield str(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Sample configuration for PerformanceTracker."""
    return {
        "analytics_db_path": None,  # Will be set in tests
        "input_token_cost": 0.0001,
        "output_token_cost": 0.0002,
        "stuff_compute_cost": 0.001,
        "refine_compute_cost": 0.003,
        "map_reduce_compute_cost": 0.005,
    }


@pytest.fixture
def performance_tracker(temp_db_path, sample_config):
    """Create a test PerformanceTracker instance."""
    sample_config["analytics_db_path"] = temp_db_path
    with patch('src.utils.logger.get_logger') as mock_logger:
        mock_logger.return_value = MagicMock()
        tracker = PerformanceTracker(sample_config)
        yield tracker


@pytest.fixture
def sample_metric():
    """Sample performance metric for testing."""
    return PerformanceMetric(
        timestamp=datetime.now().isoformat(),
        query_id="test-query-123",
        query_text="What is machine learning?",
        chain_type="stuff",
        query_type="definition",
        complexity_level="simple",
        execution_time=2.5,
        tokens_used=150,
        success=True,
        error_message=None,
        retrieval_time=1.0,
        generation_time=1.5,
        documents_retrieved=5,
        context_length=1024,
        estimated_cost=0.0,
        cost_per_token=0.0001
    )


class TestPerformanceMetric:
    """Tests for PerformanceMetric dataclass."""
    
    @pytest.mark.unit
    def test_performance_metric_creation(self, sample_metric):
        """Test successful creation of PerformanceMetric."""
        assert sample_metric.query_id == "test-query-123"
        assert sample_metric.chain_type == "stuff"
        assert sample_metric.execution_time == 2.5
        assert sample_metric.tokens_used == 150
        assert sample_metric.success is True
        assert sample_metric.error_message is None
    
    @pytest.mark.unit
    def test_performance_metric_with_error(self):
        """Test PerformanceMetric with error information."""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="error-query",
            query_text="Test query",
            chain_type="refine",
            query_type="complex",
            complexity_level="hard",
            execution_time=0.0,
            tokens_used=0,
            success=False,
            error_message="Connection timeout"
        )
        
        assert metric.success is False
        assert metric.error_message == "Connection timeout"
        assert metric.execution_time == 0.0
    
    @pytest.mark.unit
    def test_performance_metric_defaults(self):
        """Test PerformanceMetric with default values."""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="default-test",
            query_text="Default test",
            chain_type="stuff",
            query_type="test",
            complexity_level="medium",
            execution_time=1.0,
            tokens_used=100,
            success=True
        )
        
        # Check default values
        assert metric.error_message is None
        assert metric.retrieval_time == 0.0
        assert metric.generation_time == 0.0
        assert metric.documents_retrieved == 0
        assert metric.context_length == 0
        assert metric.estimated_cost == 0.0
        assert metric.cost_per_token == 0.0001
    
    @pytest.mark.unit
    def test_performance_metric_serialization(self, sample_metric):
        """Test PerformanceMetric serialization to dictionary."""
        metric_dict = asdict(sample_metric)
        
        expected_keys = {
            'timestamp', 'query_id', 'query_text', 'chain_type', 'query_type',
            'complexity_level', 'execution_time', 'tokens_used', 'success',
            'error_message', 'retrieval_time', 'generation_time',
            'documents_retrieved', 'context_length', 'estimated_cost', 'cost_per_token'
        }
        
        assert set(metric_dict.keys()) == expected_keys
        assert metric_dict['query_id'] == "test-query-123"
        assert metric_dict['execution_time'] == 2.5


class TestChainPerformanceStats:
    """Tests for ChainPerformanceStats dataclass."""
    
    @pytest.mark.unit
    def test_chain_performance_stats_creation(self):
        """Test successful creation of ChainPerformanceStats."""
        stats = ChainPerformanceStats(
            chain_type="stuff",
            total_queries=100,
            successful_queries=95,
            failed_queries=5,
            success_rate=0.95,
            avg_execution_time=2.5,
            avg_tokens_used=150.0,
            avg_cost=0.015,
            total_cost=1.5,
            min_execution_time=0.5,
            max_execution_time=10.0,
            percentile_95_time=8.0
        )
        
        assert stats.chain_type == "stuff"
        assert stats.total_queries == 100
        assert stats.success_rate == 0.95
        assert stats.avg_execution_time == 2.5
        assert stats.percentile_95_time == 8.0
    
    @pytest.mark.unit
    def test_chain_performance_stats_serialization(self):
        """Test ChainPerformanceStats serialization."""
        stats = ChainPerformanceStats(
            chain_type="refine",
            total_queries=50,
            successful_queries=48,
            failed_queries=2,
            success_rate=0.96,
            avg_execution_time=3.2,
            avg_tokens_used=200.0,
            avg_cost=0.025,
            total_cost=1.25,
            min_execution_time=1.0,
            max_execution_time=12.0,
            percentile_95_time=10.5
        )
        
        stats_dict = asdict(stats)
        
        expected_keys = {
            'chain_type', 'total_queries', 'successful_queries', 'failed_queries',
            'success_rate', 'avg_execution_time', 'avg_tokens_used', 'avg_cost',
            'total_cost', 'min_execution_time', 'max_execution_time', 'percentile_95_time'
        }
        
        assert set(stats_dict.keys()) == expected_keys
        assert stats_dict['chain_type'] == "refine"
        assert stats_dict['success_rate'] == 0.96


class TestPerformanceTrackerInit:
    """Tests for PerformanceTracker initialization."""
    
    @pytest.mark.unit
    def test_performance_tracker_init_success(self, temp_db_path, sample_config):
        """Test successful PerformanceTracker initialization."""
        sample_config["analytics_db_path"] = temp_db_path
        
        with patch('src.utils.logger.get_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            tracker = PerformanceTracker(sample_config)
            
            assert tracker.config == sample_config
            assert str(tracker.db_path) == temp_db_path
            assert Path(temp_db_path).exists()
            
            # Check cost configuration
            assert tracker.cost_config["token_costs"]["input"] == 0.0001
            assert tracker.cost_config["compute_costs"]["stuff"] == 0.001
    
    @pytest.mark.unit
    def test_performance_tracker_init_default_db_path(self, sample_config):
        """Test PerformanceTracker initialization with default database path."""
        with patch('src.utils.logger.get_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                with patch.object(PerformanceTracker, '_init_database'):
                    tracker = PerformanceTracker(sample_config)
                    
                    assert tracker.db_path == "data/analytics/performance.db"
                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @pytest.mark.unit
    def test_performance_tracker_init_real_time_metrics(self, performance_tracker):
        """Test that real-time metrics are properly initialized."""
        assert "current_session" in performance_tracker.real_time_metrics
        assert "chain_stats" in performance_tracker.real_time_metrics
        
        session = performance_tracker.real_time_metrics["current_session"]
        assert session["queries_processed"] == 0
        assert session["total_cost"] == 0.0
        assert session["avg_response_time"] == 0.0
        assert "start_time" in session
        
        assert isinstance(performance_tracker.real_time_metrics["chain_stats"], defaultdict)
    
    @pytest.mark.unit
    def test_performance_tracker_thread_lock(self, performance_tracker):
        """Test that thread lock is properly initialized."""
        assert isinstance(performance_tracker._db_lock, threading.Lock)
    
    @pytest.mark.unit
    def test_database_initialization(self, performance_tracker):
        """Test that database tables are properly created."""
        with sqlite3.connect(performance_tracker.db_path) as conn:
            cursor = conn.cursor()
            
            # Check performance_metrics table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_metrics'")
            assert cursor.fetchone() is not None
            
            # Check indexes exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'")
            assert cursor.fetchone() is not None
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chain_type'")
            assert cursor.fetchone() is not None


class TestPerformanceRecording:
    """Tests for performance recording functionality."""
    
    @pytest.mark.unit
    def test_record_performance_success(self, performance_tracker, sample_metric):
        """Test successful performance metric recording."""
        performance_tracker.record_performance(sample_metric)
        
        # Verify metric was stored in database
        with sqlite3.connect(performance_tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM performance_metrics WHERE query_id = ?", (sample_metric.query_id,))
            row = cursor.fetchone()
            
            assert row is not None
            # Check some key fields (SQLite column order may vary)
            columns = [desc for desc in cursor.description]
            row_dict = dict(zip(columns, row))
            
            assert row_dict['query_id'] == sample_metric.query_id
            assert row_dict['chain_type'] == sample_metric.chain_type
            assert row_dict['execution_time'] == sample_metric.execution_time
            assert row_dict['tokens_used'] == sample_metric.tokens_used
            assert row_dict['success'] == 1  # SQLite stores boolean as integer
    
    @pytest.mark.unit
    def test_record_performance_calculates_cost(self, performance_tracker, sample_metric):
        """Test that cost is calculated when recording performance."""
        original_cost = sample_metric.estimated_cost
        performance_tracker.record_performance(sample_metric)
        
        # Cost should have been calculated and updated
        assert sample_metric.estimated_cost != original_cost
        assert sample_metric.estimated_cost > 0
    
    @pytest.mark.unit
    def test_record_performance_updates_real_time_metrics(self, performance_tracker, sample_metric):
        """Test that real-time metrics are updated when recording."""
        initial_queries = performance_tracker.real_time_metrics["current_session"]["queries_processed"]
        initial_cost = performance_tracker.real_time_metrics["current_session"]["total_cost"]
        
        performance_tracker.record_performance(sample_metric)
        
        session = performance_tracker.real_time_metrics["current_session"]
        assert session["queries_processed"] == initial_queries + 1
        assert session["total_cost"] > initial_cost
        assert session["avg_response_time"] > 0
        
        # Check chain-specific stats
        chain_stats = performance_tracker.real_time_metrics["chain_stats"][sample_metric.chain_type]
        assert chain_stats["count"] == 1
        assert chain_stats["success_count"] == 1
        assert chain_stats["total_time"] == sample_metric.execution_time
    
    @pytest.mark.unit
    def test_record_performance_with_failed_metric(self, performance_tracker):
        """Test recording failed performance metric."""
        failed_metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="failed-query",
            query_text="Failed query",
            chain_type="stuff",
            query_type="test",
            complexity_level="simple",
            execution_time=0.0,
            tokens_used=0,
            success=False,
            error_message="Processing error"
        )
        
        performance_tracker.record_performance(failed_metric)
        
        # Check chain stats for failed query
        chain_stats = performance_tracker.real_time_metrics["chain_stats"]["stuff"]
        assert chain_stats["count"] == 1
        assert chain_stats["success_count"] == 0  # Failed query
    
    @pytest.mark.unit
    def test_record_performance_error_handling(self, performance_tracker, sample_metric):
        """Test error handling during performance recording."""
        # Mock database error
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")), \
             patch.object(performance_tracker, 'logger') as mock_logger:
            # Should not raise exception, should log error
            performance_tracker.record_performance(sample_metric)
            
            mock_logger.error.assert_called()


class TestCostCalculation:
    """Tests for cost calculation methods."""
    
    @pytest.mark.unit
    def test_calculate_cost_basic(self, performance_tracker, sample_metric):
        """Test basic cost calculation."""
        cost = performance_tracker._calculate_cost(sample_metric)
        
        assert isinstance(cost, float)
        assert cost > 0
        
        # Cost should include token cost and compute cost
        expected_token_cost = (
            sample_metric.tokens_used * 0.0001 +  # input tokens
            sample_metric.tokens_used * 0.3 * 0.0002  # output tokens (30% estimate)
        )
        expected_compute_cost = sample_metric.execution_time * 0.001  # stuff chain cost
        expected_total = expected_token_cost + expected_compute_cost
        
        assert abs(cost - expected_total) < 0.0001
    
    @pytest.mark.unit
    def test_calculate_cost_different_chain_types(self, performance_tracker):
        """Test cost calculation for different chain types."""
        base_metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="cost-test",
            query_text="Test",
            chain_type="stuff",  # Will be changed
            query_type="test",
            complexity_level="simple",
            execution_time=2.0,
            tokens_used=100,
            success=True
        )
        
        # Test different chain types
        chain_costs = {}
        for chain_type in ["stuff", "refine", "map_reduce"]:
            base_metric.chain_type = chain_type
            cost = performance_tracker._calculate_cost(base_metric)
            chain_costs[chain_type] = cost
        
        # More complex chains should cost more
        assert chain_costs["stuff"] < chain_costs["refine"] < chain_costs["map_reduce"]
    
    @pytest.mark.unit
    def test_calculate_cost_error_handling(self, performance_tracker, sample_metric):
        """Test cost calculation error handling."""
        # Mock error in calculation
        with patch.object(performance_tracker, 'cost_config', {"invalid": "config"}), \
             patch.object(performance_tracker, 'logger') as mock_logger:
            cost = performance_tracker._calculate_cost(sample_metric)
            
            # Should return default minimal cost
            assert cost == 0.001
            mock_logger.warning.assert_called()
    
    @pytest.mark.unit
    def test_calculate_cost_zero_tokens(self, performance_tracker):
        """Test cost calculation with zero tokens."""
        zero_token_metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="zero-tokens",
            query_text="Test",
            chain_type="stuff",
            query_type="test",
            complexity_level="simple",
            execution_time=1.0,
            tokens_used=0,
            success=True
        )
        
        cost = performance_tracker._calculate_cost(zero_token_metric)
        
        # Should still have compute cost
        expected_compute_cost = 1.0 * 0.001
        assert abs(cost - expected_compute_cost) < 0.0001


class TestChainPerformanceStats:
    """Tests for chain performance statistics."""
    
    @pytest.mark.unit
    def test_get_chain_performance_stats_empty(self, performance_tracker):
        """Test getting chain performance stats when no data exists."""
        stats = performance_tracker.get_chain_performance_stats()
        assert stats == []
    
    @pytest.mark.unit
    def test_get_chain_performance_stats_with_data(self, performance_tracker):
        """Test getting chain performance stats with existing data."""
        # Add multiple metrics
        for i in range(5):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"test-{i}",
                query_text=f"Test query {i}",
                chain_type="stuff",
                query_type="test",
                complexity_level="simple",
                execution_time=float(i + 1),
                tokens_used=100 + i * 10,
                success=i < 4,  # One failed query
                error_message="Error" if i == 4 else None
            )
            performance_tracker.record_performance(metric)
        
        stats = performance_tracker.get_chain_performance_stats()
        
        assert len(stats) == 1
        stuff_stats = stats
        
        assert stuff_stats.chain_type == "stuff"
        assert stuff_stats.total_queries == 5
        assert stuff_stats.successful_queries == 4
        assert stuff_stats.failed_queries == 1
        assert stuff_stats.success_rate == 0.8
        assert stuff_stats.avg_execution_time > 0
    
    @pytest.mark.unit
    def test_get_chain_performance_stats_specific_chain(self, performance_tracker):
        """Test getting stats for a specific chain type."""
        # Add metrics for different chains
        for chain_type in ["stuff", "refine"]:
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"test-{chain_type}",
                query_text="Test",
                chain_type=chain_type,
                query_type="test",
                complexity_level="simple",
                execution_time=2.0,
                tokens_used=100,
                success=True
            )
            performance_tracker.record_performance(metric)
        
        # Get stats for specific chain
        stats = performance_tracker.get_chain_performance_stats(chain_type="stuff")
        
        assert len(stats) == 1
        assert stats.chain_type == "stuff"
    
    @pytest.mark.unit
    def test_get_chain_performance_stats_time_range(self, performance_tracker):
        """Test getting stats with time range filter."""
        # Add old metric (beyond time range)
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        old_metric = PerformanceMetric(
            timestamp=old_time,
            query_id="old-test",
            query_text="Old test",
            chain_type="stuff",
            query_type="test",
            complexity_level="simple",
            execution_time=1.0,
            tokens_used=100,
            success=True
        )
        
        # Manually insert old metric (bypassing current timestamp)
        with sqlite3.connect(performance_tracker.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics 
                (timestamp, query_id, query_text, chain_type, query_type, 
                 complexity_level, execution_time, tokens_used, success, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (old_time, "old-test", "Old test", "stuff", "test", "simple", 1.0, 100, True, 0.001))
        
        # Add recent metric
        recent_metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="recent-test",
            query_text="Recent test",
            chain_type="stuff",
            query_type="test",
            complexity_level="simple",
            execution_time=2.0,
            tokens_used=200,
            success=True
        )
        performance_tracker.record_performance(recent_metric)
        
        # Get stats for last 24 hours
        stats = performance_tracker.get_chain_performance_stats(time_range_hours=24)
        
        assert len(stats) == 1
        assert stats.total_queries == 1  # Only recent metric
    
    @pytest.mark.unit
    def test_get_percentile_time(self, performance_tracker):
        """Test getting percentile execution times."""
        # Add multiple metrics with different execution times
        execution_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for i, exec_time in enumerate(execution_times):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"percentile-test-{i}",
                query_text="Test",
                chain_type="stuff",
                query_type="test",
                complexity_level="simple",
                execution_time=exec_time,
                tokens_used=100,
                success=True
            )
            performance_tracker.record_performance(metric)
        
        # Get stats to trigger percentile calculation
        stats = performance_tracker.get_chain_performance_stats()
        
        assert len(stats) == 1
        # 95th percentile of should be around 9.5
        assert 9.0 <= stats.percentile_95_time <= 10.0
    
    @pytest.mark.unit
    def test_get_chain_performance_stats_error_handling(self, performance_tracker):
        """Test error handling in getting chain performance stats."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")), \
             patch.object(performance_tracker, 'logger') as mock_logger:
            stats = performance_tracker.get_chain_performance_stats()
            
            assert stats == []
            mock_logger.error.assert_called()


class TestCostAnalysis:
    """Tests for cost analysis functionality."""
    
    @pytest.mark.unit
    def test_get_cost_analysis_empty(self, performance_tracker):
        """Test cost analysis when no data exists."""
        analysis = performance_tracker.get_cost_analysis()
        
        assert "analysis_period_hours" in analysis
        assert "overall_stats" in analysis
        assert "cost_by_chain" in analysis
        assert "cost_by_query_type" in analysis
        
        overall = analysis["overall_stats"]
        assert overall["total_queries"] == 0
        assert overall["total_cost"] == 0.0
    
    @pytest.mark.unit
    def test_get_cost_analysis_with_data(self, performance_tracker):
        """Test cost analysis with existing data."""
        # Add metrics with different costs
        metrics = [
            {"chain_type": "stuff", "tokens": 100, "time": 1.0, "query_type": "simple"},
            {"chain_type": "refine", "tokens": 200, "time": 2.0, "query_type": "complex"},
            {"chain_type": "stuff", "tokens": 150, "time": 1.5, "query_type": "simple"},
        ]
        
        for i, metric_data in enumerate(metrics):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"cost-test-{i}",
                query_text="Test",
                chain_type=metric_data["chain_type"],
                query_type=metric_data["query_type"],
                complexity_level="medium",
                execution_time=metric_data["time"],
                tokens_used=metric_data["tokens"],
                success=True
            )
            performance_tracker.record_performance(metric)
        
        analysis = performance_tracker.get_cost_analysis()
        
        # Check overall stats
        overall = analysis["overall_stats"]
        assert overall["total_queries"] == 3
        assert overall["total_cost"] > 0
        assert overall["projected_monthly_cost"] > 0
        
        # Check cost by chain
        chain_costs = analysis["cost_by_chain"]
        assert "stuff" in chain_costs
        assert "refine" in chain_costs
        assert chain_costs["stuff"]["queries"] == 2
        assert chain_costs["refine"]["queries"] == 1
        
        # Check cost by query type
        query_costs = analysis["cost_by_query_type"]
        assert "simple" in query_costs
        assert "complex" in query_costs
    
    @pytest.mark.unit
    def test_get_cost_analysis_time_range(self, performance_tracker):
        """Test cost analysis with different time ranges."""
        # Add metric
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="range-test",
            query_text="Test",
            chain_type="stuff",
            query_type="test",
            complexity_level="simple",
            execution_time=1.0,
            tokens_used=100,
            success=True
        )
        performance_tracker.record_performance(metric)
        
        # Test different time ranges
        analysis_24h = performance_tracker.get_cost_analysis(time_range_hours=24)
        analysis_1h = performance_tracker.get_cost_analysis(time_range_hours=1)
        
        assert analysis_24h["analysis_period_hours"] == 24
        assert analysis_1h["analysis_period_hours"] == 1
        
        # Both should have the same query since it's recent
        assert analysis_24h["overall_stats"]["total_queries"] == 1
        assert analysis_1h["overall_stats"]["total_queries"] == 1
    
    @pytest.mark.unit
    def test_generate_cost_recommendations(self, performance_tracker):
        """Test cost optimization recommendations generation."""
        # Create chain costs data
        chain_costs = {
            "expensive_chain": {
                "queries": 10,
                "total_cost": 1.0,
                "avg_cost": 0.1,
                "avg_time": 10.0,
                "cost_per_second": 0.01
            },
            "cheap_chain": {
                "queries": 10,
                "total_cost": 0.2,
                "avg_cost": 0.02,
                "avg_time": 2.0,
                "cost_per_second": 0.01
            }
        }
        
        recommendations = performance_tracker._generate_cost_recommendations(chain_costs)
        
        assert isinstance(recommendations, list)
        # Should recommend optimization due to cost difference
        assert len(recommendations) > 0
        assert any("expensive_chain" in rec for rec in recommendations)
    
    @pytest.mark.unit
    def test_get_cost_analysis_error_handling(self, performance_tracker):
        """Test cost analysis error handling."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")), \
             patch.object(performance_tracker, 'logger') as mock_logger:
            analysis = performance_tracker.get_cost_analysis()
            
            assert "error" in analysis
            mock_logger.error.assert_called()


class TestRealTimeMetrics:
    """Tests for real-time metrics functionality."""
    
    @pytest.mark.unit
    def test_get_real_time_metrics_initial(self, performance_tracker):
        """Test getting initial real-time metrics."""
        metrics = performance_tracker.get_real_time_metrics()
        
        assert "current_session" in metrics
        assert "chain_stats" in metrics
        assert "timestamp" in metrics
        
        session = metrics["current_session"]
        assert session["queries_processed"] == 0
        assert session["total_cost"] == 0.0
        assert session["avg_response_time"] == 0.0
    
    @pytest.mark.unit
    def test_get_real_time_metrics_after_recording(self, performance_tracker, sample_metric):
        """Test real-time metrics after recording performance."""
        performance_tracker.record_performance(sample_metric)
        
        metrics = performance_tracker.get_real_time_metrics()
        
        session = metrics["current_session"]
        assert session["queries_processed"] == 1
        assert session["total_cost"] > 0
        assert session["avg_response_time"] == sample_metric.execution_time
        
        chain_stats = metrics["chain_stats"]
        assert sample_metric.chain_type in chain_stats
        chain_stat = chain_stats[sample_metric.chain_type]
        assert chain_stat["count"] == 1
        assert chain_stat["success_count"] == 1
    
    @pytest.mark.unit
    def test_update_real_time_metrics_average_calculation(self, performance_tracker):
        """Test that average response time is calculated correctly."""
        # Add multiple metrics
        execution_times = [1.0, 2.0, 3.0]
        for i, exec_time in enumerate(execution_times):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"avg-test-{i}",
                query_text="Test",
                chain_type="stuff",
                query_type="test",
                complexity_level="simple",
                execution_time=exec_time,
                tokens_used=100,
                success=True
            )
            performance_tracker.record_performance(metric)
        
        metrics = performance_tracker.get_real_time_metrics()
        session = metrics["current_session"]
        
        expected_avg = sum(execution_times) / len(execution_times)
        assert abs(session["avg_response_time"] - expected_avg) < 0.001
    
    @pytest.mark.unit
    def test_update_real_time_metrics_error_handling(self, performance_tracker, sample_metric):
        """Test error handling in real-time metrics update."""
        with patch.object(performance_tracker.real_time_metrics, '__getitem__', side_effect=KeyError("Error")), \
             patch.object(performance_tracker, 'logger') as mock_logger:
            # Should not raise exception
            performance_tracker._update_real_time_metrics(sample_metric)
            
            mock_logger.warning.assert_called()


class TestOptimizationInsights:
    """Tests for optimization insights generation."""
    
    @pytest.mark.unit
    def test_get_optimization_insights_no_data(self, performance_tracker):
        """Test optimization insights when no data exists."""
        insights = performance_tracker.get_optimization_insights()
        
        assert "performance_insights" in insights
        assert "cost_insights" in insights
        assert "routing_insights" in insights
        assert "optimization_score" in insights
        assert "recommendations" in insights
        
        assert insights["optimization_score"] == 0.0
        assert "yeterli veri yok" in insights["recommendations"]
    
    @pytest.mark.unit
    def test_get_optimization_insights_with_data(self, performance_tracker):
        """Test optimization insights with performance data."""
        # Add metrics with mixed performance
        metrics_data = [
            {"success": True, "time": 1.0, "chain": "stuff", "tokens": 100},
            {"success": True, "time": 2.0, "chain": "stuff", "tokens": 150},
            {"success": False, "time": 0.0, "chain": "refine", "tokens": 0},  # Failed
            {"success": True, "time": 15.0, "chain": "refine", "tokens": 300},  # Slow
        ]
        
        for i, data in enumerate(metrics_data):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"insight-test-{i}",
                query_text="Test",
                chain_type=data["chain"],
                query_type="test",
                complexity_level="simple",
                execution_time=data["time"],
                tokens_used=data["tokens"],
                success=data["success"],
                error_message="Error" if not data["success"] else None
            )
            performance_tracker.record_performance(metric)
        
        insights = performance_tracker.get_optimization_insights()
        
        # Should have performance insights due to low success rate and slow response
        assert len(insights["performance_insights"]) > 0
        
        # Should have optimization score between 0 and 100
        assert 0 <= insights["optimization_score"] <= 100
        
        # Should have recommendations
        assert len(insights["recommendations"]) > 0
    
    @pytest.mark.unit
    def test_get_optimization_insights_excellent_performance(self, performance_tracker):
        """Test optimization insights with excellent performance."""
        # Add metrics with perfect performance
        for i in range(10):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"excellent-test-{i}",
                query_text="Test",
                chain_type="stuff",
                query_type="test",
                complexity_level="simple",
                execution_time=0.5,  # Fast
                tokens_used=50,  # Low cost
                success=True
            )
            performance_tracker.record_performance(metric)
        
        insights = performance_tracker.get_optimization_insights()
        
        # Should have high optimization score
        assert insights["optimization_score"] > 80
        
        # Should recommend system is optimal
        assert any("optimal" in rec for rec in insights["recommendations"])
    
    @pytest.mark.unit
    def test_get_optimization_insights_routing_analysis(self, performance_tracker):
        """Test routing insights generation."""
        # Add metrics with heavy stuff chain usage
        for i in range(10):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"routing-test-{i}",
                query_text="Test",
                chain_type="stuff",  # All stuff chains
                query_type="test",
                complexity_level="simple",
                execution_time=1.0,
                tokens_used=100,
                success=True
            )
            performance_tracker.record_performance(metric)
        
        insights = performance_tracker.get_optimization_insights()
        
        # Should have routing insights about high stuff usage
        assert len(insights["routing_insights"]) > 0
        assert any("yüksek" in insight for insight in insights["routing_insights"])
    
    @pytest.mark.unit
    def test_get_optimization_insights_error_handling(self, performance_tracker):
        """Test optimization insights error handling."""
        with patch.object(performance_tracker, 'get_chain_performance_stats', side_effect=Exception("Error")), \
             patch.object(performance_tracker, 'logger') as mock_logger:
            insights = performance_tracker.get_optimization_insights()
            
            assert "error" in insights
            mock_logger.error.assert_called()


class TestDataExport:
    """Tests for data export functionality."""
    
    @pytest.mark.unit
    def test_export_performance_data_json_empty(self, performance_tracker):
        """Test JSON export when no data exists."""
        export_data = performance_tracker.export_performance_data(format_type="json")
        
        assert isinstance(export_data, str)
        data = json.loads(export_data)
        
        assert "export_timestamp" in data
        assert "time_range_hours" in data
        assert "total_records" in data
        assert "data" in data
        
        assert data["total_records"] == 0
        assert data["data"] == []
    
    @pytest.mark.unit
    def test_export_performance_data_json_with_data(self, performance_tracker, sample_metric):
        """Test JSON export with existing data."""
        performance_tracker.record_performance(sample_metric)
        
        export_data = performance_tracker.export_performance_data(format_type="json")
        data = json.loads(export_data)
        
        assert data["total_records"] == 1
        assert len(data["data"]) == 1
        
        record = data["data"]
        assert record["query_id"] == sample_metric.query_id
        assert record["chain_type"] == sample_metric.chain_type
    
    @pytest.mark.unit
    def test_export_performance_data_csv(self, performance_tracker, sample_metric):
        """Test CSV export functionality."""
        performance_tracker.record_performance(sample_metric)
        
        export_data = performance_tracker.export_performance_data(format_type="csv")
        
        assert isinstance(export_data, str)
        lines = export_data.strip().split('\n')
        
        # Should have header and one data row
        assert len(lines) >= 2
        
        # Header should contain expected columns
        header = lines
        assert "query_id" in header
        assert "chain_type" in header
        assert "execution_time" in header
        
        # Data row should contain the metric data
        data_row = lines
        assert sample_metric.query_id in data_row
        assert sample_metric.chain_type in data_row
    
    @pytest.mark.unit
    def test_export_performance_data_unsupported_format(self, performance_tracker):
        """Test export with unsupported format."""
        export_data = performance_tracker.export_performance_data(format_type="xml")
        
        assert "Unsupported format" in export_data
    
    @pytest.mark.unit
    def test_export_performance_data_time_range(self, performance_tracker):
        """Test export with time range filter."""
        # Add metric
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="export-test",
            query_text="Test",
            chain_type="stuff",
            query_type="test",
            complexity_level="simple",
            execution_time=1.0,
            tokens_used=100,
            success=True
        )
        performance_tracker.record_performance(metric)
        
        # Export with different time ranges
        export_24h = performance_tracker.export_performance_data(time_range_hours=24)
        export_1h = performance_tracker.export_performance_data(time_range_hours=1)
        
        data_24h = json.loads(export_24h)
        data_1h = json.loads(export_1h)
        
        assert data_24h["time_range_hours"] == 24
        assert data_1h["time_range_hours"] == 1
        
        # Both should contain the recent metric
        assert data_24h["total_records"] == 1
        assert data_1h["total_records"] == 1
    
    @pytest.mark.unit
    def test_export_performance_data_error_handling(self, performance_tracker):
        """Test export error handling."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")), \
             patch.object(performance_tracker, 'logger') as mock_logger:
            export_data = performance_tracker.export_performance_data()
            
            assert "Export failed" in export_data
            mock_logger.error.assert_called()


class TestDataCleanup:
    """Tests for data cleanup functionality."""
    
    @pytest.mark.unit
    def test_reset_data_with_old_data(self, performance_tracker):
        """Test resetting old performance data."""
        # Add old metric by manually inserting with old timestamp
        old_timestamp = (datetime.now() - timedelta(days=35)).isoformat()
        
        with sqlite3.connect(performance_tracker.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics 
                (timestamp, query_id, query_text, chain_type, query_type, 
                 complexity_level, execution_time, tokens_used, success, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (old_timestamp, "old-data", "Old test", "stuff", "test", "simple", 1.0, 100, True, 0.001))
        
        # Add recent metric
        recent_metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="recent-data",
            query_text="Recent test",
            chain_type="stuff",
            query_type="test",
            complexity_level="simple",
            execution_time=1.0,
            tokens_used=100,
            success=True
        )
        performance_tracker.record_performance(recent_metric)
        
        # Reset data older than 30 days
        with patch.object(performance_tracker, 'logger') as mock_logger:
            performance_tracker.reset_data(older_than_days=30)
        
            # Verify old data was deleted, recent data remains
            with sqlite3.connect(performance_tracker.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM performance_metrics")
                total_count = cursor.fetchone()
                
                cursor.execute("SELECT COUNT(*) FROM performance_metrics WHERE query_id = ?", ("recent-data",))
                recent_count = cursor.fetchone()
                
                assert total_count == 1  # Only recent data remains
                assert recent_count == 1  # Recent data is still there
            
            mock_logger.info.assert_called()
    
    @pytest.mark.unit
    def test_reset_data_no_old_data(self, performance_tracker):
        """Test resetting data when no old data exists."""
        # Add only recent data
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="recent-only",
            query_text="Recent test",
            chain_type="stuff",
            query_type="test",
            complexity_level="simple",
            execution_time=1.0,
            tokens_used=100,
            success=True
        )
        performance_tracker.record_performance(metric)
        
        # Reset data older than 30 days (should delete nothing)
        with patch.object(performance_tracker, 'logger') as mock_logger:
            performance_tracker.reset_data(older_than_days=30)
        
            # Verify no data was deleted
            with sqlite3.connect(performance_tracker.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM performance_metrics")
                count = cursor.fetchone()
                
                assert count == 1  # Data still exists
            
            # Logger should indicate 0 records deleted
            mock_logger.info.assert_called()
    
    @pytest.mark.unit
    def test_reset_data_error_handling(self, performance_tracker):
        """Test reset data error handling."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")), \
             patch.object(performance_tracker, 'logger') as mock_logger:
            performance_tracker.reset_data()
            
            mock_logger.error.assert_called()


class TestThreadSafety:
    """Tests for thread safety of PerformanceTracker."""
    
    @pytest.mark.unit
    def test_concurrent_performance_recording(self, performance_tracker):
        """Test concurrent performance recording with threading."""
        def record_metric(tracker, thread_id):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"thread-{thread_id}",
                query_text=f"Thread test {thread_id}",
                chain_type="stuff",
                query_type="test",
                complexity_level="simple",
                execution_time=1.0,
                tokens_used=100,
                success=True
            )
            tracker.record_performance(metric)
        
        # Create and start multiple threads
        threads = []
        num_threads = 5
        for i in range(num_threads):
            thread = threading.Thread(target=record_metric, args=(performance_tracker, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all metrics were recorded
        with sqlite3.connect(performance_tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            count = cursor.fetchone()
            
            assert count == num_threads
        
        # Verify real-time metrics were updated correctly
        metrics = performance_tracker.get_real_time_metrics()
        assert metrics["current_session"]["queries_processed"] == num_threads
    
    @pytest.mark.unit
    def test_database_lock_usage(self, performance_tracker):
        """Test that database operations use the lock correctly."""
        # Mock the lock to verify it's being used
        with patch.object(performance_tracker, '_db_lock') as mock_lock:
            mock_lock.__enter__ = MagicMock(return_value=mock_lock)
            mock_lock.__exit__ = MagicMock(return_value=None)
            
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id="lock-test",
                query_text="Lock test",
                chain_type="stuff",
                query_type="test",
                complexity_level="simple",
                execution_time=1.0,
                tokens_used=100,
                success=True
            )
            
            performance_tracker.record_performance(metric)
            
            # Verify lock was used
            mock_lock.__enter__.assert_called()
            mock_lock.__exit__.assert_called()


class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases."""
    
    @pytest.mark.unit
    def test_performance_tracker_with_invalid_config(self):
        """Test PerformanceTracker with invalid configuration."""
        invalid_config = {}  # Missing required config
        
        with patch('src.utils.logger.get_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            
            # Should handle missing config gracefully
            tracker = PerformanceTracker(invalid_config)
            
            # Should use default values
            assert tracker.cost_config["token_costs"]["input"] == 0.0001
            assert tracker.cost_config["compute_costs"]["stuff"] == 0.001
    
    @pytest.mark.unit
    def test_performance_metric_with_extreme_values(self, performance_tracker):
        """Test performance metric with extreme values."""
        extreme_metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="extreme-test",
            query_text="A" * 10000,  # Very long query
            chain_type="stuff",
            query_type="extreme",
            complexity_level="maximum",
            execution_time=999999.99,  # Very long execution time
            tokens_used=999999,  # Very high token count
            success=True,
            retrieval_time=500000.0,
            generation_time=499999.99,
            documents_retrieved=1000,
            context_length=500000
        )
        
        # Should handle extreme values without error
        performance_tracker.record_performance(extreme_metric)
        
        # Verify metric was stored
        with sqlite3.connect(performance_tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM performance_metrics WHERE query_id = ?", (extreme_metric.query_id,))
            row = cursor.fetchone()
            
            assert row is not None
    
    @pytest.mark.unit
    def test_performance_metric_with_unicode_content(self, performance_tracker):
        """Test performance metric with Unicode content."""
        unicode_metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query_id="unicode-test-🚀",
            query_text="What is machine learning? 机器学习是什么？ Qu'est-ce que l'apprentissage automatique? 🤖",
            chain_type="stuff",
            query_type="unicode_test",
            complexity_level="🌟",
            execution_time=2.0,
            tokens_used=150,
            success=True,
            error_message=None
        )
        
        # Should handle Unicode content correctly
        performance_tracker.record_performance(unicode_metric)
        
        # Verify Unicode content was stored correctly
        with sqlite3.connect(performance_tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT query_text, complexity_level FROM performance_metrics WHERE query_id = ?",
                         (unicode_metric.query_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert "机器学习" in row  # Chinese characters
            assert "🤖" in row  # Emoji
            assert "🌟" in row  # Emoji in complexity level
    
    @pytest.mark.unit
    def test_get_chain_performance_stats_division_by_zero(self, performance_tracker):
        """Test chain performance stats with potential division by zero scenarios."""
        # Manually insert data that could cause division by zero
        with sqlite3.connect(performance_tracker.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics
                (timestamp, query_id, query_text, chain_type, query_type,
                 complexity_level, execution_time, tokens_used, success, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), "zero-test", "Test", "stuff", "test", "simple", 0.0, 0, True, 0.0))
        
        # Should handle zero values gracefully
        stats = performance_tracker.get_chain_performance_stats()
        
        assert len(stats) == 1
        assert stats.total_queries == 1
        assert stats.success_rate == 1.0  # 1/1 = 1.0
    
    @pytest.mark.unit
    def test_malformed_timestamp_handling(self, performance_tracker):
        """Test handling of malformed timestamps in database queries."""
        # Manually insert record with malformed timestamp
        with sqlite3.connect(performance_tracker.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics
                (timestamp, query_id, query_text, chain_type, query_type,
                 complexity_level, execution_time, tokens_used, success, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("invalid-timestamp", "malformed-test", "Test", "stuff", "test", "simple", 1.0, 100, True, 0.001))
        
        # Time range queries should handle malformed timestamps gracefully
        stats = performance_tracker.get_chain_performance_stats(time_range_hours=24)
        
        # Should still work, possibly including or excluding the malformed record
        assert isinstance(stats, list)


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""
    
    @pytest.mark.unit
    def test_realistic_performance_tracking_workflow(self, performance_tracker):
        """Test a realistic performance tracking workflow."""
        # Simulate a series of queries with different outcomes
        scenarios = [
            {"chain": "stuff", "time": 1.5, "tokens": 120, "success": True, "query_type": "simple"},
            {"chain": "stuff", "time": 2.1, "tokens": 180, "success": True, "query_type": "definition"},
            {"chain": "refine", "time": 4.5, "tokens": 300, "success": True, "query_type": "complex"},
            {"chain": "refine", "time": 0.0, "tokens": 0, "success": False, "query_type": "complex"},  # Failed
            {"chain": "map_reduce", "time": 8.2, "tokens": 500, "success": True, "query_type": "summary"},
            {"chain": "stuff", "time": 1.2, "tokens": 100, "success": True, "query_type": "simple"},
        ]
        
        # Record all scenarios
        for i, scenario in enumerate(scenarios):
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"workflow-{i}",
                query_text=f"Workflow test query {i}",
                chain_type=scenario["chain"],
                query_type=scenario["query_type"],
                complexity_level="medium",
                execution_time=scenario["time"],
                tokens_used=scenario["tokens"],
                success=scenario["success"],
                error_message="Timeout error" if not scenario["success"] else None
            )
            performance_tracker.record_performance(metric)
        
        # Analyze results
        stats = performance_tracker.get_chain_performance_stats()
        cost_analysis = performance_tracker.get_cost_analysis()
        insights = performance_tracker.get_optimization_insights()
        
        # Verify comprehensive analysis
        assert len(stats) == 3  # Three different chain types
        assert cost_analysis["overall_stats"]["total_queries"] == 6
        assert insights["optimization_score"] > 0
        
        # Export data
        export_data = performance_tracker.export_performance_data()
        exported = json.loads(export_data)
        assert exported["total_records"] == 6
        
        # Real-time metrics should reflect all queries
        real_time = performance_tracker.get_real_time_metrics()
        assert real_time["current_session"]["queries_processed"] == 6
        
        # Chain stats should show different performance for each chain
        chain_stats = real_time["chain_stats"]
        assert len(chain_stats) == 3
        assert chain_stats["stuff"]["success_count"] == 3
        assert chain_stats["refine"]["success_count"] == 1  # One failed
        assert chain_stats["map_reduce"]["success_count"] == 1
    
    @pytest.mark.unit
    def test_long_running_session_simulation(self, performance_tracker):
        """Test simulation of a long-running session with many queries."""
        # Simulate 100 queries with varying performance
        import random
        
        chain_types = ["stuff", "refine", "map_reduce"]
        query_types = ["simple", "definition", "complex", "summary"]
        
        for i in range(100):
            # Simulate realistic timing distributions
            base_time = random.uniform(0.5, 3.0)
            chain_type = random.choice(chain_types)
            
            # Adjust time based on chain complexity
            if chain_type == "refine":
                execution_time = base_time * 2
            elif chain_type == "map_reduce":
                execution_time = base_time * 3
            else:
                execution_time = base_time
            
            # Simulate occasional failures
            success = random.random() > 0.05  # 5% failure rate
            
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                query_id=f"session-{i}",
                query_text=f"Session query {i}",
                chain_type=chain_type,
                query_type=random.choice(query_types),
                complexity_level="medium",
                execution_time=execution_time if success else 0.0,
                tokens_used=random.randint(50, 400) if success else 0,
                success=success,
                error_message="Random failure" if not success else None
            )
            performance_tracker.record_performance(metric)
        
        # Analyze session performance
        stats = performance_tracker.get_chain_performance_stats()
        insights = performance_tracker.get_optimization_insights()
        real_time = performance_tracker.get_real_time_metrics()
        
        # Verify session tracking
        assert real_time["current_session"]["queries_processed"] == 100
        assert real_time["current_session"]["total_cost"] > 0
        assert real_time["current_session"]["avg_response_time"] > 0
        
        # Should have stats for all chain types used
        assert len(stats) >= 1
        
        # Should generate meaningful insights
        assert insights["optimization_score"] > 0
        assert len(insights["recommendations"]) > 0