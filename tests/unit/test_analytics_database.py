
import pytest
import sqlite3
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from src.analytics.database import ExperimentDatabase, get_experiment_db
from src.models.feedback import FeedbackCreate, FeedbackType


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_experiments.db"
    yield str(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def experiment_db(temp_db_path):
    """Create a test ExperimentDatabase instance."""
    with patch('src.utils.logger.get_logger') as mock_logger:
        mock_logger.return_value = MagicMock()
        db = ExperimentDatabase(temp_db_path)
        yield db


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        "name": "Test RAG Experiment",
        "description": "Testing RAG performance with different models",
        "created_by": "test_user",
        "metadata": {"model_type": "transformer", "version": "1.0"}
    }


@pytest.fixture
def sample_run_data():
    """Sample experiment run data for testing."""
    return {
        "experiment_id": 1,
        "query": "What is machine learning?",
        "generation_model": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-ada-002",
        "rag_answer": "Machine learning is a subset of artificial intelligence...",
        "direct_llm_answer": "Machine learning is a method of data analysis...",
        "quality_score": 4,
        "user_notes": "Good answer, relevant content",
        "rag_params": {"top_k": 5, "chunk_size": 512},
        "session_name": "test_session"
    }


@pytest.fixture
def sample_feedback_data():
    """Sample feedback data for testing."""
    return FeedbackCreate(
        interaction_id="test-interaction-123",
        student_id="test-student-456",
        feedback_type=FeedbackType.HELPFUL,
        comment="This answer was very helpful!"
    )


class TestExperimentDatabaseInit:
    """Tests for ExperimentDatabase initialization."""
    
    @pytest.mark.unit
    def test_database_init_creates_directory(self, temp_db_path):
        """Test that database initialization creates the directory structure."""
        db_path = Path(temp_db_path)
        # Remove the directory that the fixture created
        import shutil
        shutil.rmtree(db_path.parent, ignore_errors=True)
        assert not db_path.parent.exists()
        
        with patch('src.utils.logger.get_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            ExperimentDatabase(str(db_path))
        
        assert db_path.parent.exists()
        assert db_path.exists()
    
    @pytest.mark.unit
    def test_database_init_default_path(self):
        """Test database initialization with default path."""
        with patch('src.utils.logger.get_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                with patch.object(ExperimentDatabase, '_init_schema'):
                    db = ExperimentDatabase()
                    # Use os.path.normpath to handle path separator differences
                    import os
                    expected_path = os.path.normpath("data/analytics/experiments.db")
                    assert str(db.db_path) == expected_path
                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @pytest.mark.unit
    def test_database_schema_initialization(self, experiment_db):
        """Test that all required tables are created."""
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check experiments table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'")
            assert cursor.fetchone() is not None
            
            # Check experiment_runs table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_runs'")
            assert cursor.fetchone() is not None
            
            # Check performance_metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_metrics'")
            assert cursor.fetchone() is not None
            
            # Check retrieved_sources table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='retrieved_sources'")
            assert cursor.fetchone() is not None
            
            # Check active learning tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_configurations'")
            assert cursor.fetchone() is not None
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interactions'")
            assert cursor.fetchone() is not None
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
            assert cursor.fetchone() is not None
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_profiles'")
            assert cursor.fetchone() is not None
    
    @pytest.mark.unit
    def test_database_indexes_created(self, experiment_db):
        """Test that all required indexes are created."""
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            expected_indexes = [
                'idx_experiment_runs_exp_id',
                'idx_performance_metrics_run_id',
                'idx_retrieved_sources_run_id',
                'idx_experiments_timestamp',
                'idx_experiment_runs_timestamp',
                'idx_interactions_user_id',
                'idx_interactions_timestamp',
                'idx_feedback_interaction_id'
            ]
            
            for expected_index in expected_indexes:
                assert expected_index in indexes


class TestExperimentDatabaseContextManager:
    """Tests for database connection context manager."""
    
    @pytest.mark.unit
    def test_get_connection_success(self, experiment_db):
        """Test successful database connection."""
        with experiment_db.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            assert conn.row_factory == sqlite3.Row
    
    @pytest.mark.unit
    def test_get_connection_commits_on_success(self, experiment_db):
        """Test that connection commits on successful operation."""
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO experiments (name) VALUES (?)", ("Test Commit",))
        
        # Verify data was committed
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM experiments WHERE name = ?", ("Test Commit",))
            assert cursor.fetchone() is not None
    
    @pytest.mark.unit
    def test_get_connection_rollback_on_error(self, experiment_db):
        """Test that connection rolls back on error."""
        try:
            with experiment_db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO experiments (name) VALUES (?)", ("Test Rollback",))
                raise Exception("Simulated error")
        except Exception:
            pass
        
        # Verify data was not committed
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM experiments WHERE name = ?", ("Test Rollback",))
            assert cursor.fetchone() is None
    
    @pytest.mark.unit
    def test_get_connection_logs_error(self, experiment_db):
        """Test that connection manager logs errors."""
        # Replace the logger with a mock for this test
        mock_logger = MagicMock()
        experiment_db.logger = mock_logger
        
        try:
            with experiment_db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INVALID SQL")
        except Exception:
            pass
        
        mock_logger.error.assert_called_once()


class TestExperimentOperations:
    """Tests for experiment CRUD operations."""
    
    @pytest.mark.unit
    def test_create_experiment_success(self, experiment_db, sample_experiment_data):
        """Test successful experiment creation."""
        experiment_id = experiment_db.create_experiment(**sample_experiment_data)
        
        assert isinstance(experiment_id, int)
        assert experiment_id > 0
        
        # Verify experiment was created
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row['name'] == sample_experiment_data['name']
            assert row['description'] == sample_experiment_data['description']
            assert row['created_by'] == sample_experiment_data['created_by']
            
            metadata = json.loads(row['metadata'])
            assert metadata == sample_experiment_data['metadata']
    
    @pytest.mark.unit
    def test_create_experiment_minimal_data(self, experiment_db):
        """Test experiment creation with minimal required data."""
        experiment_id = experiment_db.create_experiment("Minimal Experiment")
        
        assert isinstance(experiment_id, int)
        
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()
            
            assert row['name'] == "Minimal Experiment"
            assert row['description'] == ""
            assert row['created_by'] == "system"
            assert json.loads(row['metadata']) == {}
    
    @pytest.mark.unit
    def test_create_experiment_none_metadata(self, experiment_db):
        """Test experiment creation with None metadata."""
        experiment_id = experiment_db.create_experiment("Test", metadata=None)
        
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metadata FROM experiments WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()
            
            assert json.loads(row['metadata']) == {}
    
    @pytest.mark.unit
    def test_get_experiments_empty(self, experiment_db):
        """Test getting experiments when none exist."""
        experiments = experiment_db.get_experiments()
        assert experiments == []
    
    @pytest.mark.unit
    def test_get_experiments_with_data(self, experiment_db, sample_experiment_data):
        """Test getting experiments with existing data."""
        experiment_id = experiment_db.create_experiment(**sample_experiment_data)
        
        experiments = experiment_db.get_experiments()
        
        assert len(experiments) == 1
        experiment = experiments[0]
        
        assert experiment['experiment_id'] == experiment_id
        assert experiment['name'] == sample_experiment_data['name']
        assert experiment['run_count'] == 0  # No runs yet
    
    @pytest.mark.unit
    def test_get_experiments_limit(self, experiment_db):
        """Test getting experiments with limit."""
        # Create multiple experiments
        for i in range(5):
            experiment_db.create_experiment(f"Experiment {i}")
        
        experiments = experiment_db.get_experiments(limit=3)
        assert len(experiments) == 3
    
    @pytest.mark.unit
    def test_get_experiments_ordering(self, experiment_db):
        """Test that experiments are returned in descending timestamp order."""
        # Create experiments with slight delay
        first_id = experiment_db.create_experiment("First Experiment")
        second_id = experiment_db.create_experiment("Second Experiment")
        
        experiments = experiment_db.get_experiments()
        
        assert experiments[0]['experiment_id'] == second_id  # Most recent first
        assert experiments[1]['experiment_id'] == first_id


class TestExperimentRunOperations:
    """Tests for experiment run operations."""
    
    @pytest.mark.unit
    def test_add_experiment_run_success(self, experiment_db, sample_run_data):
        """Test successful experiment run creation."""
        # First create an experiment
        experiment_id = experiment_db.create_experiment("Test Experiment")
        sample_run_data['experiment_id'] = experiment_id
        
        run_id = experiment_db.add_experiment_run(**sample_run_data)
        
        assert isinstance(run_id, int)
        assert run_id > 0
        
        # Verify run was created
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row['experiment_id'] == experiment_id
            assert row['query'] == sample_run_data['query']
            assert row['generation_model'] == sample_run_data['generation_model']
            assert row['quality_score'] == sample_run_data['quality_score']
            
            rag_params = json.loads(row['rag_params'])
            assert rag_params == sample_run_data['rag_params']
    
    @pytest.mark.unit
    def test_add_experiment_run_minimal_data(self, experiment_db):
        """Test experiment run creation with minimal required data."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        
        run_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Test query",
            generation_model="test-model",
            embedding_model="test-embedding",
            rag_answer="Test answer"
        )
        
        assert isinstance(run_id, int)
        
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            
            assert row['direct_llm_answer'] is None
            assert row['quality_score'] is None
            assert row['user_notes'] == ""
            assert json.loads(row['rag_params']) == {}
    
    @pytest.mark.unit
    def test_get_experiment_runs_empty(self, experiment_db):
        """Test getting runs for experiment with no runs."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        
        runs = experiment_db.get_experiment_runs(experiment_id)
        assert runs == []
    
    @pytest.mark.unit
    def test_get_experiment_runs_with_data(self, experiment_db, sample_run_data):
        """Test getting runs with existing data."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        sample_run_data['experiment_id'] = experiment_id
        
        run_id = experiment_db.add_experiment_run(**sample_run_data)
        
        runs = experiment_db.get_experiment_runs(experiment_id)
        
        assert len(runs) == 1
        run = runs[0]
        
        assert run['run_id'] == run_id
        assert run['query'] == sample_run_data['query']
    
    @pytest.mark.unit
    def test_get_experiment_runs_with_performance_metrics(self, experiment_db, sample_run_data):
        """Test getting runs with joined performance metrics."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        sample_run_data['experiment_id'] = experiment_id
        
        run_id = experiment_db.add_experiment_run(**sample_run_data)
        
        # Add performance metrics
        experiment_db.add_performance_metrics(
            run_id=run_id,
            retrieval_time_ms=100.5,
            generation_time_ms=200.3,
            total_response_time_ms=300.8
        )
        
        runs = experiment_db.get_experiment_runs(experiment_id)
        
        assert len(runs) == 1
        run = runs[0]
        
        assert run['retrieval_time_ms'] == 100.5
        assert run['generation_time_ms'] == 200.3
        assert run['total_response_time_ms'] == 300.8


class TestPerformanceMetrics:
    """Tests for performance metrics operations."""
    
    @pytest.mark.unit
    def test_add_performance_metrics_complete(self, experiment_db):
        """Test adding complete performance metrics."""
        # Create experiment and run first
        experiment_id = experiment_db.create_experiment("Test Experiment")
        run_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Test query",
            generation_model="test-model",
            embedding_model="test-embedding",
            rag_answer="Test answer"
        )
        
        experiment_db.add_performance_metrics(
            run_id=run_id,
            retrieval_time_ms=150.5,
            generation_time_ms=250.3,
            total_response_time_ms=400.8,
            gpu_utilization_percent=75.2,
            memory_usage_percent=60.1,
            cache_hit=True,
            num_retrieved_docs=5,
            context_length=1024
        )
        
        # Verify metrics were added
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM performance_metrics WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row['retrieval_time_ms'] == 150.5
            assert row['generation_time_ms'] == 250.3
            assert row['gpu_utilization_percent'] == 75.2
            assert row['cache_hit'] == 1  # SQLite stores boolean as integer
            assert row['num_retrieved_docs'] == 5
    
    @pytest.mark.unit
    def test_add_performance_metrics_partial(self, experiment_db):
        """Test adding partial performance metrics."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        run_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Test query",
            generation_model="test-model",
            embedding_model="test-embedding",
            rag_answer="Test answer"
        )
        
        experiment_db.add_performance_metrics(
            run_id=run_id,
            total_response_time_ms=500.0,
            cache_hit=False
        )
        
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM performance_metrics WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            
            assert row['total_response_time_ms'] == 500.0
            assert row['cache_hit'] == 0
            assert row['retrieval_time_ms'] is None
            assert row['generation_time_ms'] is None


class TestRetrievedSources:
    """Tests for retrieved sources operations."""
    
    @pytest.mark.unit
    def test_add_retrieved_sources_success(self, experiment_db):
        """Test adding retrieved sources successfully."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        run_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Test query",
            generation_model="test-model",
            embedding_model="test-embedding",
            rag_answer="Test answer"
        )
        
        sources = [
            {
                "text": "Source document 1 content",
                "score": 0.95,
                "metadata": {"document_id": "doc1", "page": 1}
            },
            {
                "text": "Source document 2 content",
                "score": 0.87,
                "metadata": {"document_id": "doc2", "page": 3}
            }
        ]
        
        experiment_db.add_retrieved_sources(run_id, sources)
        
        # Verify sources were added
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM retrieved_sources WHERE run_id = ? ORDER BY source_rank", 
                (run_id,)
            )
            rows = cursor.fetchall()
            
            assert len(rows) == 2
            
            # Check first source
            assert rows[0]['source_content'] == "Source document 1 content"
            assert rows[0]['relevance_score'] == 0.95
            assert rows[0]['source_rank'] == 1
            
            metadata1 = json.loads(rows[0]['metadata'])
            assert metadata1 == {"document_id": "doc1", "page": 1}
            
            # Check second source
            assert rows[1]['source_rank'] == 2
            assert rows[1]['relevance_score'] == 0.87
    
    @pytest.mark.unit
    def test_add_retrieved_sources_empty_list(self, experiment_db):
        """Test adding empty sources list."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        run_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Test query",
            generation_model="test-model",
            embedding_model="test-embedding",
            rag_answer="Test answer"
        )
        
        experiment_db.add_retrieved_sources(run_id, [])
        
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM retrieved_sources WHERE run_id = ?", (run_id,))
            count = cursor.fetchone()[0]
            
            assert count == 0
    
    @pytest.mark.unit
    def test_add_retrieved_sources_minimal_data(self, experiment_db):
        """Test adding sources with minimal data."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        run_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Test query",
            generation_model="test-model",
            embedding_model="test-embedding",
            rag_answer="Test answer"
        )
        
        sources = [{"text": "Minimal source content"}]
        
        experiment_db.add_retrieved_sources(run_id, sources)
        
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM retrieved_sources WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            
            assert row['source_content'] == "Minimal source content"
            assert row['relevance_score'] == 0.0  # Default value
            assert row['source_rank'] == 1
            assert json.loads(row['metadata']) == {}


class TestRunDetails:
    """Tests for getting detailed run information."""
    
    @pytest.mark.unit
    def test_get_run_details_nonexistent(self, experiment_db):
        """Test getting details for non-existent run."""
        result = experiment_db.get_run_details(999)
        assert result is None
    
    @pytest.mark.unit
    def test_get_run_details_complete(self, experiment_db, sample_run_data):
        """Test getting complete run details."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        sample_run_data['experiment_id'] = experiment_id
        
        run_id = experiment_db.add_experiment_run(**sample_run_data)
        
        # Add performance metrics
        experiment_db.add_performance_metrics(
            run_id=run_id,
            retrieval_time_ms=100.0,
            generation_time_ms=200.0
        )
        
        # Add sources
        sources = [{"text": "Test source", "score": 0.9}]
        experiment_db.add_retrieved_sources(run_id, sources)
        
        details = experiment_db.get_run_details(run_id)
        
        assert details is not None
        assert details['run_id'] == run_id
        assert details['query'] == sample_run_data['query']
        assert details['retrieval_time_ms'] == 100.0
        assert details['generation_time_ms'] == 200.0
        
        # Check parsed JSON fields
        assert isinstance(details['rag_params'], dict)
        assert details['rag_params'] == sample_run_data['rag_params']
        
        # Check sources
        assert len(details['sources']) == 1
        assert details['sources'][0]['source_content'] == "Test source"
        assert details['sources'][0]['relevance_score'] == 0.9


class TestStatistics:
    """Tests for experiment statistics."""
    
    @pytest.mark.unit
    def test_get_experiment_statistics_no_runs(self, experiment_db):
        """Test statistics for experiment with no runs."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        
        stats = experiment_db.get_experiment_statistics(experiment_id)
        
        assert stats['total_runs'] == 0
        assert stats['avg_quality_score'] is None
        assert stats['model_usage'] == []
    
    @pytest.mark.unit
    def test_get_experiment_statistics_with_runs(self, experiment_db):
        """Test statistics for experiment with runs."""
        experiment_id = experiment_db.create_experiment("Test Experiment")
        
        # Add multiple runs with different scores and models
        run1_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Query 1",
            generation_model="gpt-3.5-turbo",
            embedding_model="ada-002",
            rag_answer="Answer 1",
            quality_score=4
        )
        
        run2_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Query 2",
            generation_model="gpt-4",
            embedding_model="ada-002",
            rag_answer="Answer 2",
            quality_score=5
        )
        
        run3_id = experiment_db.add_experiment_run(
            experiment_id=experiment_id,
            query="Query 3",
            generation_model="gpt-3.5-turbo",
            embedding_model="ada-002",
            rag_answer="Answer 3",
            quality_score=3
        )
        
        # Add performance metrics
        experiment_db.add_performance_metrics(run1_id, total_response_time_ms=1000.0, cache_hit=True)
        experiment_db.add_performance_metrics(run2_id, total_response_time_ms=1500.0, cache_hit=False)
        experiment_db.add_performance_metrics(run3_id, total_response_time_ms=800.0, cache_hit=True)
        
        stats = experiment_db.get_experiment_statistics(experiment_id)
        
        assert stats['total_runs'] == 3
        assert stats['avg_quality_score'] == 4.0  # (4+5+3)/3
        assert abs(stats['avg_response_time'] - 1100.0) < 0.1  # (1000+1500+800)/3
        assert abs(stats['cache_hit_rate'] - 66.67) < 0.1  # 2/3 * 100
        
        # Check model usage
        assert len(stats['model_usage']) == 2
        model_usage = {item['generation_model']: item['count'] for item in stats['model_usage']}
        assert model_usage['gpt-3.5-turbo'] == 2
        assert model_usage['gpt-4'] == 1


class TestExportData:
    """Tests for data export functionality."""
    
    @pytest.mark.unit
    def test_export_experiment_data_empty(self, experiment_db):
        """Test exporting data when no experiments exist."""
        result = experiment_db.export_experiment_data(1)
        
        assert 'experiment' in result
        assert 'runs' in result
        assert 'statistics' in result
        assert 'export_timestamp' in result
        
        assert result['runs'] == []
    
    @pytest.mark.unit
    def test_export_experiment_data_with_data(self, experiment_db, sample_run_data):
        """Test exporting data with existing experiment."""
        experiment_id = experiment_db.create_experiment("Export Test")
        sample_run_data['experiment_id'] = experiment_id
        
        experiment_db.add_experiment_run(**sample_run_data)
        
        result = experiment_db.export_experiment_data(experiment_id)
        
        assert len(result['runs']) == 1
        assert result['runs'][0]['query'] == sample_run_data['query']
        assert result['statistics']['total_runs'] == 1
        assert result['export_timestamp'] is not None


class TestQualityUpdate:
    """Tests for updating run quality."""
    
    @pytest.mark.unit
    def test_update_experiment_run_quality(self, experiment_db, sample_run_data):
        """Test updating run quality score and notes."""
        experiment_id = experiment_db.create_experiment("Quality Test")
        sample_run_data['experiment_id'] = experiment_id
        
        run_id = experiment_db.add_experiment_run(**sample_run_data)
        
        # Update quality
        experiment_db.update_experiment_run_quality(
            run_id=run_id, 
            quality_score=5, 
            user_notes="Updated: Excellent answer"
        )
        
        # Verify update
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT quality_score, user_notes FROM experiment_runs WHERE run_id = ?", 
                (run_id,)
            )
            row = cursor.fetchone()
            
            assert row['quality_score'] == 5
            assert row['user_notes'] == "Updated: Excellent answer"


class TestActiveLearningMethods:
    """Tests for active learning helper methods."""
    
    @pytest.mark.unit
    def test_add_or_get_rag_configuration_new(self, experiment_db):
        """Test adding new RAG configuration."""
        rag_params = {"top_k": 5, "chunk_size": 512, "temperature": 0.7}
        
        config_hash = experiment_db.add_or_get_rag_configuration(rag_params)
        
        assert isinstance(config_hash, str)
        assert len(config_hash) == 64  # SHA256 hash length
        
        # Verify configuration was stored
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rag_configurations WHERE config_hash = ?", (config_hash,))
            row = cursor.fetchone()
            
            assert row is not None
            stored_params = json.loads(row['rag_params'])
            assert stored_params == rag_params
    
    @pytest.mark.unit
    def test_add_or_get_rag_configuration_existing(self, experiment_db):
        """Test getting existing RAG configuration."""
        rag_params = {"top_k": 3, "chunk_size": 256}
        
        # Add first time
        config_hash1 = experiment_db.add_or_get_rag_configuration(rag_params)
        
        # Add same config again
        config_hash2 = experiment_db.add_or_get_rag_configuration(rag_params)
        
        assert config_hash1 == config_hash2
        
        # Verify only one record exists
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rag_configurations WHERE config_hash = ?", (config_hash1,))
            count = cursor.fetchone()[0]
            
            assert count == 1
    
    @pytest.mark.unit
    def test_add_interaction_success(self, experiment_db):
        """Test adding user interaction."""
        rag_params = {"top_k": 5}
        config_hash = experiment_db.add_or_get_rag_configuration(rag_params)
        
        retrieved_context = [
            {"text": "Context 1", "score": 0.9},
            {"text": "Context 2", "score": 0.8}
        ]
        
        interaction_id = experiment_db.add_interaction(
            user_id="test_user",
            query="What is AI?",
            response="AI is artificial intelligence...",
            retrieved_context=retrieved_context,
            rag_config_hash=config_hash,
            session_id="session_123",
            uncertainty_score=0.15,
            feedback_requested=True,
            processing_time_ms=1500.0
        )
        
        assert isinstance(interaction_id, int)
        assert interaction_id > 0
        
        # Verify interaction was stored
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interactions WHERE interaction_id = ?", (interaction_id,))
            row = cursor.fetchone()
            
            assert row['user_id'] == "test_user"
            assert row['query'] == "What is AI?"
            assert row['uncertainty_score'] == 0.15
            assert row['feedback_requested'] == 1  # SQLite boolean as integer
            
            stored_context = json.loads(row['retrieved_context'])
            assert stored_context == retrieved_context
    
    @pytest.mark.unit
    def test_add_feedback_success(self, experiment_db, sample_feedback_data):
        """Test adding feedback for interaction."""
        # First create interaction
        rag_params = {"top_k": 5}
        config_hash = experiment_db.add_or_get_rag_configuration(rag_params)
        
        interaction_id = experiment_db.add_interaction(
            user_id=sample_feedback_data.student_id,
            query="Test query",
            response="Test response",
            retrieved_context=[],
            rag_config_hash=config_hash
        )
        
        # Update feedback data with correct interaction_id
        sample_feedback_data.interaction_id = str(interaction_id)
        
        feedback_id = experiment_db.add_feedback(sample_feedback_data)
        
        assert isinstance(feedback_id, int)
        assert feedback_id > 0
        
        # Verify feedback was stored
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,))
            row = cursor.fetchone()
            
            assert row['interaction_id'] == str(interaction_id)
            assert row['student_id'] == sample_feedback_data.student_id
            assert row['feedback_type'] == sample_feedback_data.feedback_type.value
            assert row['comment'] == sample_feedback_data.comment


class TestStudentProfiles:
    """Tests for student profile operations."""
    
    @pytest.mark.unit
    def test_get_student_profile_nonexistent(self, experiment_db):
        """Test getting profile for non-existent student."""
        profile = experiment_db.get_student_profile("nonexistent_student")
        assert profile is None
    
    @pytest.mark.unit
    def test_student_profile_updated_with_feedback(self, experiment_db, sample_feedback_data):
        """Test that student profile is updated when feedback is added."""
        # Create interaction and feedback
        rag_params = {"top_k": 5}
        config_hash = experiment_db.add_or_get_rag_configuration(rag_params)
        
        interaction_id = experiment_db.add_interaction(
            user_id=sample_feedback_data.student_id,
            query="Profile test query",
            response="Profile test response",
            retrieved_context=[],
            rag_config_hash=config_hash
        )
        
        sample_feedback_data.interaction_id = str(interaction_id)
        experiment_db.add_feedback(sample_feedback_data)
        
        # Check student profile was created/updated
        profile = experiment_db.get_student_profile(sample_feedback_data.student_id)
        
        assert profile is not None
        assert profile['user_id'] == sample_feedback_data.student_id
        assert profile['total_queries'] == 1
        assert profile['avg_feedback_score'] == 4.0  # HELPFUL maps to 4
        assert profile['last_updated'] is not None


class TestFeedbackQueries:
    """Tests for feedback query methods."""
    
    @pytest.mark.unit
    def test_get_feedbacks_by_interaction_id_empty(self, experiment_db):
        """Test getting feedbacks for interaction with no feedback."""
        feedbacks = experiment_db.get_feedbacks_by_interaction_id("nonexistent")
        assert feedbacks == []
    
    @pytest.mark.unit
    def test_get_feedback_statistics_empty(self, experiment_db):
        """Test getting feedback statistics when no feedback exists."""
        stats = experiment_db.get_feedback_statistics()
        
        assert stats['total_feedbacks'][0] == 0
        assert stats['positive_feedbacks'][0] == 0
        assert stats['negative_feedbacks'][0] == 0
        assert stats['by_type'] == {}
    
    @pytest.mark.unit
    def test_get_feedback_since_empty(self, experiment_db):
        """Test getting feedback since when none exists."""
        feedback = experiment_db.get_feedback_since(7)
        assert feedback == []
    
    @pytest.mark.unit
    def test_get_most_uncertain_queries_empty(self, experiment_db):
        """Test getting uncertain queries when none exist."""
        queries = experiment_db.get_most_uncertain_queries()
        assert queries == []
    
    @pytest.mark.unit
    def test_get_performance_metrics_empty(self, experiment_db):
        """Test getting performance metrics when none exist."""
        metrics = experiment_db.get_performance_metrics(7)
        assert metrics == []
    
    @pytest.mark.unit
    def test_get_average_rating_empty(self, experiment_db):
        """Test getting average rating when no feedback exists."""
        avg_rating = experiment_db.get_average_rating(7)
        assert avg_rating == 0.0


class TestConnectionCheck:
    """Tests for database connection checking."""
    
    @pytest.mark.unit
    def test_check_connection_success(self, experiment_db):
        """Test successful connection check."""
        # Replace the logger with a mock for this test
        mock_logger = MagicMock()
        experiment_db.logger = mock_logger
        
        experiment_db.check_connection()  # Should not raise exception
        mock_logger.debug.assert_called_with("Database connection check successful.")


class TestGlobalDatabaseInstance:
    """Tests for global database instance management."""
    
    @pytest.mark.unit
    def test_get_experiment_db_creates_instance(self):
        """Test that get_experiment_db creates database instance."""
        with patch('src.analytics.database.ExperimentDatabase') as mock_db_class:
            mock_instance = MagicMock()
            mock_db_class.return_value = mock_instance
            mock_instance.db_path = Path("test_path.db")
            
            db = get_experiment_db("test_path.db")
            
            mock_db_class.assert_called_once_with("test_path.db")
            assert db == mock_instance
    
    @pytest.mark.unit
    def test_get_experiment_db_reuses_instance(self):
        """Test that get_experiment_db reuses existing instance for same path."""
        with patch('src.analytics.database.ExperimentDatabase') as mock_db_class:
            mock_instance = MagicMock()
            mock_db_class.return_value = mock_instance
            mock_instance.db_path = Path("same_path.db")
            
            db1 = get_experiment_db("same_path.db")
            db2 = get_experiment_db("same_path.db")
            
            mock_db_class.assert_called_once()
            assert db1 == db2 == mock_instance
    
    @pytest.mark.unit
    def test_get_experiment_db_creates_new_for_different_path(self):
        """Test that get_experiment_db creates new instance for different path."""
        with patch('src.analytics.database.ExperimentDatabase') as mock_db_class:
            mock_instance1 = MagicMock()
            mock_instance2 = MagicMock()
            mock_instance1.db_path = Path("path1.db")
            mock_instance2.db_path = Path("path2.db")
            mock_db_class.side_effect = [mock_instance1, mock_instance2]
            
            db1 = get_experiment_db("path1.db")
            db2 = get_experiment_db("path2.db")
            
            assert mock_db_class.call_count == 2
            assert db1 == mock_instance1
            assert db2 == mock_instance2


class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    @pytest.mark.unit
    def test_database_error_handling(self, temp_db_path):
        """Test database error handling in operations."""
        mock_logger = MagicMock()
        with patch('src.utils.logger.get_logger') as mock_get_logger:
            mock_get_logger.return_value = mock_logger
            db = ExperimentDatabase(temp_db_path)
            # Replace the logger to ensure we can check it
            db.logger = mock_logger
            
            # Use an invalid SQL query that will definitely fail
            try:
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("INVALID SQL STATEMENT THAT WILL FAIL")
            except Exception:
                pass  # Expected to fail
            
            mock_logger.error.assert_called()
    
    @pytest.mark.unit
    def test_json_parse_error_handling(self, experiment_db):
        """Test handling of JSON parsing errors."""
        # Create experiment with invalid JSON manually
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO experiments (name, metadata) VALUES (?, ?)",
                ("Test", "invalid json")
            )
            experiment_id = cursor.lastrowid
        
        # This should handle JSON parsing gracefully
        experiments = experiment_db.get_experiments()
        assert len(experiments) == 1  # Should still return the experiment
    
    @pytest.mark.unit
    def test_invalid_feedback_type_mapping(self, experiment_db):
        """Test handling of invalid feedback types in rating mapping."""
        rag_params = {"top_k": 5}
        config_hash = experiment_db.add_or_get_rag_configuration(rag_params)
        
        # Create interaction with invalid feedback type stored directly
        interaction_id = experiment_db.add_interaction(
            user_id="test_user",
            query="Test query",
            response="Test response",
            retrieved_context=[],
            rag_config_hash=config_hash
        )
        
        # Manually insert feedback with invalid type
        with experiment_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (interaction_id, student_id, feedback_type) VALUES (?, ?, ?)",
                (interaction_id, "test_student", "invalid_feedback_type")
            )
        
        # The rating mapping should handle unknown types gracefully
        feedbacks = experiment_db.get_feedback_since(1)
        assert len(feedbacks) == 1
        assert feedbacks[0]['rating'] == 3  # Default rating for unknown types