"""
Database schema and management for academic experiment tracking.
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager
import hashlib
from src.utils.logger import get_logger

class ExperimentDatabase:
    """
    Manages SQLite database for academic experiment tracking.
    """
    
    def __init__(self, db_path: str = "data/analytics/experiments.db"):
        """
        Initialize the experiment database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__, {})
        
        # Initialize database schema
        self._init_schema()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_by TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT  -- JSON for additional experiment metadata
                )
            """)
            
            # Experiment runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query TEXT NOT NULL,
                    generation_model TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    rag_params TEXT,  -- JSON for RAG parameters
                    rag_answer TEXT,
                    direct_llm_answer TEXT,
                    quality_score INTEGER CHECK (quality_score >= 1 AND quality_score <= 5),
                    user_notes TEXT,
                    session_name TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    retrieval_time_ms REAL,
                    generation_time_ms REAL,
                    total_response_time_ms REAL,
                    gpu_utilization_percent REAL,
                    memory_usage_percent REAL,
                    cache_hit BOOLEAN,
                    num_retrieved_docs INTEGER,
                    context_length INTEGER,
                    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
                )
            """)
            
            # Retrieved sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrieved_sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    source_content TEXT NOT NULL,
                    relevance_score REAL,
                    source_rank INTEGER,  -- Position in retrieval results
                    metadata TEXT,  -- JSON for source metadata
                    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
                )
            """)
            
            # --- Active Learning & Feedback Loop Tables ---

            # RAG Configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_configurations (
                    config_hash TEXT PRIMARY KEY,
                    rag_params TEXT NOT NULL, -- JSON format
                    avg_performance_score REAL,
                    usage_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query TEXT NOT NULL,
                    response TEXT,
                    retrieved_context TEXT, -- JSON format
                    rag_config_hash TEXT,
                    uncertainty_score REAL,
                    feedback_requested BOOLEAN,
                    processing_time_ms REAL,
                    FOREIGN KEY (rag_config_hash) REFERENCES rag_configurations (config_hash)
                )
            """)

            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    comment TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES interactions (interaction_id)
                )
            """)

            # Student Profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_profiles (
                    user_id TEXT PRIMARY KEY,
                    total_queries INTEGER DEFAULT 0,
                    avg_feedback_score REAL,
                    common_low_score_topics TEXT, -- JSON format
                    last_updated DATETIME
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiment_runs_exp_id ON experiment_runs (experiment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_run_id ON performance_metrics (run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_retrieved_sources_run_id ON retrieved_sources (run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_timestamp ON experiments (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiment_runs_timestamp ON experiment_runs (timestamp)")
            
            # Indexes for new tables
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_interaction_id ON feedback (interaction_id)")
            
            # Safe migrations: add new columns to interactions if missing
            self._ensure_interactions_columns(conn)
            
            self.logger.info("Database schema initialized successfully")
    
    def _ensure_interactions_columns(self, conn: sqlite3.Connection) -> None:
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(interactions)")
            cols = {row[1] for row in cursor.fetchall()}
            def add_col(name: str, ddl: str):
                cursor.execute(f"ALTER TABLE interactions ADD COLUMN {name} {ddl}")
            if "success" not in cols:
                add_col("success", "BOOLEAN")
            if "error_message" not in cols:
                add_col("error_message", "TEXT")
            if "chain_type" not in cols:
                add_col("chain_type", "TEXT")
        except Exception as e:
            self.logger.warning(f"Failed to ensure interactions columns: {e}")
    
    def create_experiment(self, name: str, description: str = "", created_by: str = "system", 
                         metadata: Dict[str, Any] = None) -> int:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            created_by: Creator identifier
            metadata: Additional experiment metadata
            
        Returns:
            experiment_id: ID of created experiment
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (name, description, created_by, metadata)
                VALUES (?, ?, ?, ?)
            """, (name, description, created_by, json.dumps(metadata or {})))
            
            experiment_id = cursor.lastrowid
            self.logger.info(f"Created experiment '{name}' with ID {experiment_id}")
            return experiment_id
    
    def add_experiment_run(self, experiment_id: int, query: str, generation_model: str,
                          embedding_model: str, rag_answer: str, direct_llm_answer: str = None,
                          quality_score: int = None, user_notes: str = "", 
                          rag_params: Dict[str, Any] = None, session_name: str = "") -> int:
        """
        Add a new experiment run.
        
        Args:
            experiment_id: Parent experiment ID
            query: Query text
            generation_model: Model used for generation
            embedding_model: Model used for embeddings
            rag_answer: RAG system answer
            direct_llm_answer: Direct LLM answer (optional)
            quality_score: Quality score (1-5)
            user_notes: Additional notes
            rag_params: RAG parameters used
            session_name: Session/dataset name
            
        Returns:
            run_id: ID of created run
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiment_runs (
                    experiment_id, query, generation_model, embedding_model, 
                    rag_params, rag_answer, direct_llm_answer, quality_score, 
                    user_notes, session_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, query, generation_model, embedding_model,
                json.dumps(rag_params or {}), rag_answer, direct_llm_answer,
                quality_score, user_notes, session_name
            ))
            
            run_id = cursor.lastrowid
            self.logger.info(f"Added experiment run {run_id} for experiment {experiment_id}")
            return run_id
    
    def add_performance_metrics(self, run_id: int, retrieval_time_ms: float = None,
                              generation_time_ms: float = None, total_response_time_ms: float = None,
                              gpu_utilization_percent: float = None, memory_usage_percent: float = None,
                              cache_hit: bool = None, num_retrieved_docs: int = None,
                              context_length: int = None):
        """
        Add performance metrics for a run.
        
        Args:
            run_id: Experiment run ID
            retrieval_time_ms: Time for retrieval phase
            generation_time_ms: Time for generation phase
            total_response_time_ms: Total response time
            gpu_utilization_percent: GPU utilization
            memory_usage_percent: Memory usage
            cache_hit: Whether cache was hit
            num_retrieved_docs: Number of retrieved documents
            context_length: Length of context sent to LLM
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (
                    run_id, retrieval_time_ms, generation_time_ms, total_response_time_ms,
                    gpu_utilization_percent, memory_usage_percent, cache_hit,
                    num_retrieved_docs, context_length
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, retrieval_time_ms, generation_time_ms, total_response_time_ms,
                gpu_utilization_percent, memory_usage_percent, cache_hit,
                num_retrieved_docs, context_length
            ))
    
    def add_retrieved_sources(self, run_id: int, sources: List[Dict[str, Any]]):
        """
        Add retrieved sources for a run.
        
        Args:
            run_id: Experiment run ID
            sources: List of retrieved source documents with metadata
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for rank, source in enumerate(sources, 1):
                cursor.execute("""
                    INSERT INTO retrieved_sources (
                        run_id, source_content, relevance_score, source_rank, metadata
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    run_id,
                    source.get('text', ''),
                    source.get('score', 0.0),
                    rank,
                    json.dumps(source.get('metadata', {}))
                ))
    
    def get_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of experiments."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT experiment_id, timestamp, name, description, created_by, status,
                       (SELECT COUNT(*) FROM experiment_runs er WHERE er.experiment_id = e.experiment_id) as run_count
                FROM experiments e
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_experiment_runs(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Get runs for a specific experiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT er.*, pm.retrieval_time_ms, pm.generation_time_ms, 
                       pm.total_response_time_ms, pm.gpu_utilization_percent,
                       pm.memory_usage_percent, pm.cache_hit
                FROM experiment_runs er
                LEFT JOIN performance_metrics pm ON er.run_id = pm.run_id
                WHERE er.experiment_id = ?
                ORDER BY er.timestamp DESC
            """, (experiment_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_run_details(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get run info with performance metrics
            cursor.execute("""
                SELECT er.*, pm.*
                FROM experiment_runs er
                LEFT JOIN performance_metrics pm ON er.run_id = pm.run_id
                WHERE er.run_id = ?
            """, (run_id,))
            
            run_data = cursor.fetchone()
            if not run_data:
                return None
            
            run_dict = dict(run_data)
            
            # Get retrieved sources
            cursor.execute("""
                SELECT source_content, relevance_score, source_rank, metadata
                FROM retrieved_sources
                WHERE run_id = ?
                ORDER BY source_rank
            """, (run_id,))
            
            sources = []
            for source_row in cursor.fetchall():
                source_dict = dict(source_row)
                source_dict['metadata'] = json.loads(source_dict['metadata'] or '{}')
                sources.append(source_dict)
            
            run_dict['sources'] = sources
            
            # Parse JSON fields
            run_dict['rag_params'] = json.loads(run_dict['rag_params'] or '{}')
            
            return run_dict
    
    def get_experiment_statistics(self, experiment_id: int) -> Dict[str, Any]:
        """Get statistics for an experiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    AVG(quality_score) as avg_quality_score,
                    AVG(pm.total_response_time_ms) as avg_response_time,
                    AVG(pm.gpu_utilization_percent) as avg_gpu_usage,
                    COUNT(CASE WHEN pm.cache_hit = 1 THEN 1 END) * 100.0 / COUNT(*) as cache_hit_rate
                FROM experiment_runs er
                LEFT JOIN performance_metrics pm ON er.run_id = pm.run_id
                WHERE er.experiment_id = ?
            """, (experiment_id,))
            
            stats = dict(cursor.fetchone())
            
            # Model usage distribution
            cursor.execute("""
                SELECT generation_model, COUNT(*) as count
                FROM experiment_runs
                WHERE experiment_id = ?
                GROUP BY generation_model
                ORDER BY count DESC
            """, (experiment_id,))
            
            stats['model_usage'] = [dict(row) for row in cursor.fetchall()]
            
            return stats
    
    def export_experiment_data(self, experiment_id: int) -> Dict[str, Any]:
        """Export complete experiment data for analysis."""
        experiment = self.get_experiments()[0] if self.get_experiments() else {}
        runs = self.get_experiment_runs(experiment_id)
        stats = self.get_experiment_statistics(experiment_id)
        
        return {
            'experiment': experiment,
            'runs': runs,
            'statistics': stats,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def update_experiment_run_quality(self, run_id: int, quality_score: int, user_notes: str = ""):
        """Update quality score and notes for a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiment_runs 
                SET quality_score = ?, user_notes = ?
                WHERE run_id = ?
            """, (quality_score, user_notes, run_id))
            
            self.logger.info(f"Updated run {run_id} quality score to {quality_score}")

    # --- Active Learning Helper Methods ---

    def add_or_get_rag_configuration(self, rag_params: Dict[str, Any]) -> str:
        """
        Adds a new RAG configuration if it doesn't exist, or returns the hash of an existing one.
        The hash is a SHA256 of the sorted, JSON-dumped parameters.
        """
        params_str = json.dumps(rag_params, sort_keys=True)
        config_hash = hashlib.sha256(params_str.encode('utf-8')).hexdigest()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT config_hash FROM rag_configurations WHERE config_hash = ?", (config_hash,))
            if cursor.fetchone():
                self.logger.debug(f"RAG configuration with hash {config_hash[:10]}... already exists.")
                return config_hash
            
            cursor.execute("""
                INSERT INTO rag_configurations (config_hash, rag_params)
                VALUES (?, ?)
            """, (config_hash, params_str))
            self.logger.info(f"Added new RAG configuration with hash {config_hash[:10]}...")
            return config_hash

    def add_interaction(self, user_id: str, query: str, response: str, retrieved_context: List[Dict],
                        rag_config_hash: str, session_id: Optional[str] = None,
                        uncertainty_score: Optional[float] = None, feedback_requested: Optional[bool] = None,
                        processing_time_ms: Optional[float] = None,
                        success: Optional[bool] = True,
                        error_message: Optional[str] = None,
                        chain_type: Optional[str] = None) -> int:
        """Adds a new user interaction to the database."""
        with self.get_connection() as conn:
            # ensure columns exist (defensive)
            self._ensure_interactions_columns(conn)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (user_id, session_id, query, response, retrieved_context,
                                          rag_config_hash, uncertainty_score, feedback_requested, processing_time_ms,
                                          success, error_message, chain_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_id, query, response, json.dumps(retrieved_context or []),
                  rag_config_hash, uncertainty_score, feedback_requested, processing_time_ms,
                  success, error_message, chain_type))
            interaction_id = cursor.lastrowid
            self.logger.debug(f"Logged interaction {interaction_id} for user {user_id}.")
            return interaction_id

    def delete_interactions_by_session(self, session_id: str) -> int:
        """Delete interactions for a given session_id. Returns count deleted."""
        if not session_id:
            return 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM interactions WHERE session_id = ?", (session_id,))
            deleted = cursor.rowcount or 0
            self.logger.info(f"Deleted {deleted} interactions for session {session_id}")
            return deleted

    def add_feedback(self, feedback_data: 'FeedbackCreate') -> int:
        """Adds feedback for a specific interaction."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (interaction_id, student_id, feedback_type, comment)
                VALUES (?, ?, ?, ?)
            """, (feedback_data.interaction_id, feedback_data.student_id, feedback_data.feedback_type.value, feedback_data.comment))
            feedback_id = cursor.lastrowid
            self.logger.info(f"Received feedback {feedback_id} for interaction {feedback_data.interaction_id}.")

            # Map feedback type to a numeric rating
            rating_map = {
                'correct': 5, 'helpful': 4, 'unclear': 3,
                'not_helpful': 2, 'incorrect': 1, 'off_topic': 1
            }
            rating = rating_map.get(feedback_data.feedback_type.value, 3)

            # Update related performance metrics in the same transaction
            self._update_rag_config_performance(conn, feedback_data.interaction_id, rating)
            self._update_student_profile_with_feedback(conn, feedback_data.student_id, rating)

            return feedback_id

    def _update_rag_config_performance(self, conn: sqlite3.Connection, interaction_id: int, new_rating: int):
        """Internal method to update RAG config performance based on new feedback."""
        cursor = conn.cursor()
        cursor.execute("SELECT rag_config_hash FROM interactions WHERE interaction_id = ?", (interaction_id,))
        result = cursor.fetchone()
        if not result or not result['rag_config_hash']:
            return

        config_hash = result['rag_config_hash']
        
        cursor.execute("SELECT avg_performance_score, usage_count FROM rag_configurations WHERE config_hash = ?", (config_hash,))
        current = cursor.fetchone()
        current_score = current['avg_performance_score'] if current and current['avg_performance_score'] is not None else 0
        current_count = current['usage_count'] if current and current['usage_count'] is not None else 0

        new_count = current_count + 1
        new_avg_score = ((current_score * current_count) + new_rating) / new_count

        cursor.execute("""
            UPDATE rag_configurations
            SET usage_count = ?, avg_performance_score = ?
            WHERE config_hash = ?
        """, (new_count, new_avg_score, config_hash))
        self.logger.debug(f"Updated performance for RAG config {config_hash[:10]}...")

    def _update_student_profile_with_feedback(self, conn: sqlite3.Connection, user_id: str, new_rating: int):
        """Internal method to update a student's profile based on new feedback."""
        cursor = conn.cursor()
        
        cursor.execute("INSERT OR IGNORE INTO student_profiles (user_id, total_queries, avg_feedback_score) VALUES (?, 0, 0)", (user_id,))
        
        cursor.execute("SELECT total_queries, avg_feedback_score FROM student_profiles WHERE user_id = ?", (user_id,))
        current = cursor.fetchone()
        current_queries = current['total_queries'] if current else 0
        current_score = current['avg_feedback_score'] if current and current['avg_feedback_score'] is not None else 0

        new_total_queries = current_queries + 1
        new_avg_score = ((current_score * current_queries) + new_rating) / new_total_queries

        cursor.execute("""
            UPDATE student_profiles
            SET total_queries = ?, avg_feedback_score = ?, last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (new_total_queries, new_avg_score, user_id))
        self.logger.debug(f"Updated profile for student {user_id}.")

    def get_student_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the profile for a given student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student_profiles WHERE user_id = ?", (user_id,))
            profile = cursor.fetchone()
            if profile:
                profile_dict = dict(profile)
                profile_dict['common_low_score_topics'] = json.loads(profile_dict.get('common_low_score_topics') or '[]')
                return profile_dict
            return None

    def get_feedbacks_by_interaction_id(self, interaction_id: str) -> List[Dict[str, Any]]:
        """Retrieves all feedbacks for a given interaction ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback WHERE interaction_id = ?", (interaction_id,))
            feedbacks = [dict(row) for row in cursor.fetchall()]
            self.logger.debug(f"Found {len(feedbacks)} feedbacks for interaction {interaction_id}.")
            return feedbacks

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Calculates and returns feedback statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total feedbacks
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedbacks = cursor.fetchone()
            
            # Positive feedbacks (customize this logic as needed)
            positive_types = ('correct', 'helpful')
            placeholders = ','.join('?' for _ in positive_types)
            cursor.execute(f"SELECT COUNT(*) FROM feedback WHERE feedback_type IN ({placeholders})", positive_types)
            positive_feedbacks = cursor.fetchone()

            # Negative feedbacks (customize this logic as needed)
            negative_types = ('incorrect', 'not_helpful', 'unclear', 'off_topic')
            placeholders = ','.join('?' for _ in negative_types)
            cursor.execute(f"SELECT COUNT(*) FROM feedback WHERE feedback_type IN ({placeholders})", negative_types)
            negative_feedbacks = cursor.fetchone()
            
            # Count by type
            cursor.execute("SELECT feedback_type, COUNT(*) as count FROM feedback GROUP BY feedback_type")
            by_type = {row['feedback_type']: row['count'] for row in cursor.fetchall()}
            
            return {
                "total_feedbacks": total_feedbacks,
                "positive_feedbacks": positive_feedbacks,
                "negative_feedbacks": negative_feedbacks,
                "by_type": by_type
            }
    
    def check_connection(self):
        """Checks if the database connection is alive."""
        with self.get_connection() as conn:
            conn.cursor().execute("SELECT 1")
        self.logger.debug("Database connection check successful.")

    def get_feedback_since(self, days: int) -> List[Dict[str, Any]]:
        """
        Retrieves feedback records from the last N days with their associated query.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Map feedback type to a numeric rating for analysis
            cursor.execute("""
                SELECT
                    i.query,
                    f.feedback_type,
                    f.comment,
                    f.created_at,
                    CASE f.feedback_type
                        WHEN 'correct' THEN 5 WHEN 'helpful' THEN 4 WHEN 'unclear' THEN 3
                        WHEN 'not_helpful' THEN 2 WHEN 'incorrect' THEN 1 ELSE 3
                    END as rating
                FROM feedback f
                JOIN interactions i ON f.interaction_id = i.interaction_id
                WHERE f.created_at >= date('now', '-' || ? || ' days')
            """, (days,))
            return [dict(row) for row in cursor.fetchall()]

    def get_most_uncertain_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves the top N queries with the highest uncertainty scores."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query, uncertainty_score, timestamp
                FROM interactions
                WHERE uncertainty_score IS NOT NULL
                ORDER BY uncertainty_score DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_metrics(self, days: int) -> List[Dict[str, Any]]:
        """
        Retrieves performance-related data for the optimizer, linking RAG params to feedback ratings.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    i.interaction_id,
                    rc.rag_params,
                    CASE f.feedback_type
                        WHEN 'correct' THEN 5 WHEN 'helpful' THEN 4 WHEN 'unclear' THEN 3
                        WHEN 'not_helpful' THEN 2 WHEN 'incorrect' THEN 1 ELSE 3
                    END as rating
                FROM interactions i
                JOIN feedback f ON i.interaction_id = f.interaction_id
                JOIN rag_configurations rc ON i.rag_config_hash = rc.config_hash
                WHERE i.timestamp >= date('now', '-' || ? || ' days')
            """, (days,))
            
            results = []
            for row in cursor.fetchall():
                try:
                    params = json.loads(row['rag_params'])
                    results.append({
                        "rating": row['rating'],
                        "retrieval_k": params.get('top_k', 5)
                    })
                except (json.JSONDecodeError, AttributeError):
                    self.logger.warning(f"Could not parse rag_params for interaction {row['interaction_id']}")
            return results

    def get_strategy_performance_by_query_type(self, query_type: str) -> Optional[Dict[str, Any]]:
        """
        Finds the best performing RAG strategy.
        NOTE: This is a simplified implementation that does not use `query_type`.
        It returns the overall best-performing active configuration.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT rag_params, avg_performance_score
                FROM rag_configurations
                WHERE is_active = TRUE AND usage_count > 5
                ORDER BY avg_performance_score DESC
                LIMIT 1
            """)
            best_strategy = cursor.fetchone()
            if best_strategy and best_strategy['rag_params']:
                params = json.loads(best_strategy['rag_params'])
                top_k = params.get('top_k', 5)
                
                strategy_name = "hybrid_search" # Default
                if top_k <= 3:
                    strategy_name = "precise_retrieval"
                elif top_k >= 8:
                    strategy_name = "broad_context_retrieval"

                return {
                    "best_strategy": strategy_name,
                    "avg_rating": best_strategy['avg_performance_score']
                }
            return None

    def get_average_rating(self, days: int) -> float:
        """Calculates the average feedback rating over the last N days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    AVG(CASE f.feedback_type
                        WHEN 'correct' THEN 5 WHEN 'helpful' THEN 4 WHEN 'unclear' THEN 3
                        WHEN 'not_helpful' THEN 2 WHEN 'incorrect' THEN 1 ELSE 3
                    END) as avg_rating
                FROM feedback f
                WHERE f.created_at >= date('now', '-' || ? || ' days')
            """, (days,))
            result = cursor.fetchone()
            return result['avg_rating'] if result and result['avg_rating'] is not None else 0.0


# Global database instance
_db_instance: Optional['ExperimentDatabase'] = None

def get_experiment_db(db_path: str = "data/analytics/experiments.db") -> ExperimentDatabase:
    """Get the global experiment database instance."""
    global _db_instance
    if _db_instance is None or str(_db_instance.db_path) != db_path:
        _db_instance = ExperimentDatabase(db_path)
    return _db_instance