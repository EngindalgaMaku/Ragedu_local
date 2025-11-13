import pytest
import json
from src.analytics.database import ExperimentDatabase
from src.models.feedback import FeedbackCreate, FeedbackType


@pytest.mark.unit
@pytest.mark.db
def test_database_initialization(db_instance: ExperimentDatabase):
    """Test if the database and its tables are created successfully."""
    assert db_instance is not None
    with db_instance.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row for row in cursor.fetchall()}
        expected_tables = {
            'experiments', 'experiment_runs', 'performance_metrics', 
            'retrieved_sources', 'rag_configurations', 'interactions', 
            'feedback', 'student_profiles'
        }
        assert expected_tables.issubset(tables)

@pytest.mark.unit
@pytest.mark.db
def test_create_experiment(db_instance: ExperimentDatabase):
    """Test creating a new experiment."""
    exp_id = db_instance.create_experiment(
        name="Test Experiment",
        description="A test experiment.",
        created_by="pytest"
    )
    assert exp_id > 0
    
    experiments = db_instance.get_experiments()
    assert len(experiments) == 1
    assert experiments['name'] == "Test Experiment"
    assert experiments['created_by'] == "pytest"

@pytest.mark.unit
@pytest.mark.db
def test_add_experiment_run(db_instance: ExperimentDatabase):
    """Test adding a run to an experiment."""
    exp_id = db_instance.create_experiment(name="Test Exp")
    run_id = db_instance.add_experiment_run(
        experiment_id=exp_id,
        query="What is RAG?",
        generation_model="test-gen-model",
        embedding_model="test-emb-model",
        rag_answer="This is a RAG answer.",
        rag_params={"top_k": 5}
    )
    assert run_id > 0
    
    runs = db_instance.get_experiment_runs(experiment_id=exp_id)
    assert len(runs) == 1
    assert runs['query'] == "What is RAG?"
    assert json.loads(runs['rag_params'])['top_k'] == 5

@pytest.mark.unit
@pytest.mark.db
def test_add_or_get_rag_configuration(db_instance: ExperimentDatabase):
    """Test adding and retrieving a RAG configuration."""
    rag_params = {"top_k": 7, "chunk_size": 512}
    
    # Add for the first time
    config_hash_1 = db_instance.add_or_get_rag_configuration(rag_params)
    assert isinstance(config_hash_1, str)
    
    # Retrieve the existing one
    config_hash_2 = db_instance.add_or_get_rag_configuration(rag_params)
    assert config_hash_1 == config_hash_2

    with db_instance.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rag_configurations")
        assert cursor.fetchone() == 1

@pytest.mark.unit
@pytest.mark.db
def test_add_interaction_and_feedback(db_instance: ExperimentDatabase):
    """Test logging an interaction and then adding feedback to it."""
    rag_params = {"top_k": 5}
    config_hash = db_instance.add_or_get_rag_configuration(rag_params)
    
    interaction_id = db_instance.add_interaction(
        user_id="student123",
        session_id="session_abc",
        query="What is active learning?",
        response="It's a machine learning strategy.",
        retrieved_context=[{"doc": "content"}],
        rag_config_hash=config_hash,
        uncertainty_score=0.85
    )
    assert interaction_id > 0

    feedback_data = FeedbackCreate(
        interaction_id=str(interaction_id),
        student_id="student123",
        feedback_type=FeedbackType.CORRECT,
        comment="Very clear answer!"
    )
    
    feedback_id = db_instance.add_feedback(feedback_data)
    assert feedback_id > 0
    
    feedbacks = db_instance.get_feedbacks_by_interaction_id(str(interaction_id))
    assert len(feedbacks) == 1
    assert feedbacks['comment'] == "Very clear answer!"
    assert feedbacks['feedback_type'] == "correct"

@pytest.mark.unit
@pytest.mark.db
def test_rag_config_performance_update(db_instance: ExperimentDatabase):
    """Test if RAG configuration performance is updated after feedback."""
    rag_params = {"top_k": 10}
    config_hash = db_instance.add_or_get_rag_configuration(rag_params)
    
    interaction_id = db_instance.add_interaction(
        user_id="student456", query="Test", response="Test", 
        retrieved_context=[], rag_config_hash=config_hash
    )
    
    # Feedback 1 (rating: 5)
    feedback1 = FeedbackCreate(interaction_id=str(interaction_id), student_id="student456", feedback_type=FeedbackType.CORRECT, comment="")
    db_instance.add_feedback(feedback1)
    
    with db_instance.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT avg_performance_score, usage_count FROM rag_configurations WHERE config_hash=?", (config_hash,))
        res1 = cursor.fetchone()
        assert res1['usage_count'] == 1
        assert res1['avg_performance_score'] == 5.0

    # Feedback 2 (rating: 1)
    interaction_id_2 = db_instance.add_interaction(
        user_id="student789", query="Test 2", response="Test 2",
        retrieved_context=[], rag_config_hash=config_hash
    )
    feedback2 = FeedbackCreate(interaction_id=str(interaction_id_2), student_id="student789", feedback_type=FeedbackType.INCORRECT, comment="")
    db_instance.add_feedback(feedback2)

    with db_instance.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT avg_performance_score, usage_count FROM rag_configurations WHERE config_hash=?", (config_hash,))
        res2 = cursor.fetchone()
        assert res2['usage_count'] == 2
        # (5*1 + 1*1) / 2 = 3.0
        assert res2['avg_performance_score'] == 3.0

@pytest.mark.unit
@pytest.mark.db
def test_get_best_performing_strategy(db_instance: ExperimentDatabase):
    """Test retrieving the best performing RAG strategy."""
    # Strategy 1 (should be best)
    params1 = {"top_k": 3} # precise_retrieval
    hash1 = db_instance.add_or_get_rag_configuration(params1)
    for i in range(6):
        interaction_id = db_instance.add_interaction("user1", f"q{i}", "r", [], hash1)
        db_instance.add_feedback(FeedbackCreate(interaction_id=str(interaction_id), student_id="user1", feedback_type=FeedbackType.CORRECT, comment="")) # rating 5

    # Strategy 2 (worse)
    params2 = {"top_k": 8} # broad_context_retrieval
    hash2 = db_instance.add_or_get_rag_configuration(params2)
    for i in range(6):
        interaction_id = db_instance.add_interaction("user2", f"q{i}", "r", [], hash2)
        db_instance.add_feedback(FeedbackCreate(interaction_id=str(interaction_id), student_id="user2", feedback_type=FeedbackType.UNCLEAR, comment="")) # rating 3

    best_strategy = db_instance.get_strategy_performance_by_query_type("any")
    
    assert best_strategy is not None
    assert best_strategy['best_strategy'] == "precise_retrieval"
    assert best_strategy['avg_rating'] == 5.0