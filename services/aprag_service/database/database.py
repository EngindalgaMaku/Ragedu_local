"""
Database connection and management for APRAG Service
Uses the same database as auth_service for consistency
"""

import sqlite3
import os
import json
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for APRAG Service
    Uses the same SQLite database as auth_service
    """
    
    def __init__(self, db_path: str = "data/rag_assistant.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.ensure_database_directory()
        self.init_database()
    
    def ensure_database_directory(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """
        Get database connection with automatic close
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database and apply APRAG migrations"""
        try:
            with self.get_connection() as conn:
                # Check if APRAG tables exist
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='student_interactions'
                """)
                
                if not cursor.fetchone():
                    logger.info("APRAG tables not found. Applying migrations...")
                    self.apply_aprag_migrations(conn)
                    self.apply_topic_migrations(conn)
                    self.apply_knowledge_base_tables_migration(conn)
                    self.apply_session_settings_migration(conn)
                    self.apply_session_settings_fk_removal_migration(conn)
                    self.apply_document_global_scores_migration(conn)
                    self.ensure_feature_flags_table(conn)
                    conn.commit()  # Ensure commit after migration
                    logger.info("APRAG migrations applied successfully")
                else:
                    logger.info("APRAG tables already exist")
                    # Always try to apply migrations to ensure schema is up to date
                    # (migration file uses IF NOT EXISTS, so it's safe)
                    self.apply_aprag_migrations(conn)
                    self.apply_topic_migrations(conn)
                    self.apply_foreign_key_fix_migration(conn)
                    self.apply_topic_progress_fk_removal_migration(conn)
                    self.apply_qa_embeddings_migration(conn)
                    self.apply_satisfaction_fix_migration(conn)
                    self.apply_detailed_feedback_migration(conn)
                    self.apply_session_settings_migration(conn)
                    self.apply_session_settings_fk_removal_migration(conn)
                    self.apply_document_global_scores_migration(conn)
                    self.apply_avg_emoji_score_migration(conn)
                    self.apply_analytics_views(conn)
                    self.apply_knowledge_base_tables_migration(conn)
                    self.apply_ebars_migration(conn)
                    self.apply_completion_percentage_migration(conn)
                    self.ensure_feature_flags_table(conn)
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def apply_aprag_migrations(self, conn: sqlite3.Connection):
        """Apply APRAG database migrations"""
        try:
            # Read migration file
            # Try multiple possible paths
            possible_paths = [
                "/app/migrations/003_create_aprag_tables.sql",  # Docker volume mount path
                os.path.join(
                    os.path.dirname(__file__),
                    "../../auth_service/database/migrations/003_create_aprag_tables.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../services/auth_service/database/migrations/003_create_aprag_tables.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../../services/auth_service/database/migrations/003_create_aprag_tables.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration (split by semicolon for multiple statements)
                # SQLite doesn't support multiple statements in execute(), so we use executescript
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("APRAG migration applied successfully")
            else:
                logger.warning(f"APRAG migration file not found. Expected paths: {possible_paths}")
                logger.info("APRAG tables will be created by auth_service migration system")
                # Don't fail - auth_service will handle the migration
                
        except Exception as e:
            logger.warning(f"Failed to apply APRAG migrations (non-critical): {e}")
            # Don't raise - auth_service will handle migration
    
    def apply_topic_migrations(self, conn: sqlite3.Connection):
        """Apply Topic-Based Learning Path Tracking migrations"""
        try:
            # Check if topic tables already exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='course_topics'
            """)
            
            if cursor.fetchone():
                logger.info("Topic tables already exist")
                return
            
            logger.info("Applying Topic migrations...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/004_create_topic_tables.sql",  # Docker volume mount path
                os.path.join(
                    os.path.dirname(__file__),
                    "../../auth_service/database/migrations/004_create_topic_tables.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../services/auth_service/database/migrations/004_create_topic_tables.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../../services/auth_service/database/migrations/004_create_topic_tables.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("Topic migration applied successfully")
            else:
                logger.warning(f"Topic migration file not found. Expected paths: {possible_paths}")
                logger.info("Topic tables will be created by auth_service migration system")
                # Don't fail - auth_service will handle the migration
                
        except Exception as e:
            logger.warning(f"Failed to apply Topic migrations (non-critical): {e}")
            # Don't raise - auth_service will handle migration
    
    def apply_foreign_key_fix_migration(self, conn: sqlite3.Connection):
        """Apply Foreign Key Fix migration (005_fix_aprag_foreign_keys.sql)"""
        try:
            # Check if migration is already applied by checking user_id type
            cursor = conn.execute("PRAGMA table_info(student_interactions)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            # If user_id is INTEGER, migration is already applied
            if columns.get('user_id') == 'INTEGER':
                logger.info("Foreign key fix migration already applied")
                return
            
            logger.info("Applying Foreign Key Fix migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/005_fix_aprag_foreign_keys.sql",  # Docker volume mount path
                os.path.join(
                    os.path.dirname(__file__),
                    "../../auth_service/database/migrations/005_fix_aprag_foreign_keys.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../services/auth_service/database/migrations/005_fix_aprag_foreign_keys.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../../services/auth_service/database/migrations/005_fix_aprag_foreign_keys.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("Foreign Key Fix migration applied successfully")
            else:
                logger.warning(f"Foreign Key Fix migration file not found. Expected paths: {possible_paths}")
                logger.info("Foreign Key Fix will be handled by auth_service migration system")
                
        except Exception as e:
            logger.warning(f"Failed to apply Foreign Key Fix migration (non-critical): {e}")
    
    def apply_topic_progress_fk_removal_migration(self, conn: sqlite3.Connection):
        """Apply migration to remove foreign key constraint from topic_progress to users table and ensure correct schema"""
        try:
            # First check if topic_progress table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='topic_progress'
            """)
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                logger.info("topic_progress table does not exist, will be created by migration")
                # Continue to apply migration which will create the table
            else:
                # Check table schema
                cursor = conn.execute("PRAGMA table_info(topic_progress)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Check if migration is needed
                has_average_understanding = 'average_understanding' in columns
                has_average_satisfaction = 'average_satisfaction' in columns
                has_progress_id = 'progress_id' in columns
                
                # Check FK to users
                cursor = conn.execute("PRAGMA foreign_key_list(topic_progress)")
                fks = cursor.fetchall()
                has_users_fk = any(fk[2] == 'users' for fk in fks) if fks else False
                
                # If table has correct schema and no FK to users, migration already applied
                if has_average_understanding and has_average_satisfaction and has_progress_id and not has_users_fk:
                    logger.info("Topic progress FK removal migration already applied (correct schema, no FK to users)")
                    return
                
                logger.info(f"Topic progress schema check: average_understanding={has_average_understanding}, "
                          f"average_satisfaction={has_average_satisfaction}, progress_id={has_progress_id}, "
                          f"has_users_fk={has_users_fk}")
            
            logger.info("Applying Topic Progress FK Removal migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/010_remove_topic_progress_fk_to_users.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/010_remove_topic_progress_fk_to_users.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/010_remove_topic_progress_fk_to_users.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                # Check if table exists and has data before applying migration
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='topic_progress'
                """)
                table_exists = cursor.fetchone() is not None
                
                if table_exists:
                    # Check current schema to build safe migration
                    cursor = conn.execute("PRAGMA table_info(topic_progress)")
                    old_columns = {row[1]: row[2] for row in cursor.fetchall()}
                    
                    # Check if we need to migrate (if table has old schema)
                    has_old_schema = 'id' in old_columns or 'average_understanding' not in old_columns
                    
                    if has_old_schema:
                        logger.info("Old schema detected, applying migration with data copy...")
                        # Apply migration using Python (safer)
                        self._apply_topic_progress_migration_with_data_copy(conn, old_columns)
                    else:
                        logger.info("Schema already up to date, skipping migration")
                else:
                    # Table doesn't exist, just create it with correct schema
                    logger.info("Table doesn't exist, creating with correct schema...")
                    self._apply_topic_progress_migration_directly(conn)
            else:
                logger.warning(f"Topic Progress FK Removal migration file not found. Expected paths: {possible_paths}")
                # Apply migration directly if file not found
                logger.info("Applying Topic Progress FK Removal migration directly...")
                self._apply_topic_progress_migration_directly(conn)
                
        except Exception as e:
            logger.error(f"Failed to apply Topic Progress FK Removal migration: {e}", exc_info=True)
            # Try to apply directly as fallback
            try:
                logger.info("Attempting direct migration application as fallback...")
                self._apply_topic_progress_migration_directly(conn)
            except Exception as fallback_err:
                logger.error(f"Fallback migration also failed: {fallback_err}", exc_info=True)
    
    def _apply_topic_progress_migration_directly(self, conn: sqlite3.Connection):
        """Apply topic_progress migration directly (fallback method)"""
        try:
            # Check if table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='topic_progress'
            """)
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check current schema
                cursor = conn.execute("PRAGMA table_info(topic_progress)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Add ALL missing columns if needed
                # Required columns for topic_progress table
                required_columns = {
                    'average_understanding': 'DECIMAL(3,2)',
                    'average_satisfaction': 'DECIMAL(3,2)',
                    'mastery_score': 'DECIMAL(3,2)',
                    'mastery_level': 'VARCHAR(20)',
                    'is_ready_for_next': 'BOOLEAN DEFAULT FALSE',
                    'readiness_score': 'DECIMAL(3,2)',
                    'time_spent_minutes': 'INTEGER DEFAULT 0',
                    'first_question_timestamp': 'TIMESTAMP',
                    'last_question_timestamp': 'TIMESTAMP',
                    'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                }
                
                # Add progress_id if needed (handle both 'id' and 'progress_id')
                if 'id' in columns and 'progress_id' not in columns:
                    logger.info("Adding progress_id column...")
                    conn.execute("""
                        ALTER TABLE topic_progress 
                        ADD COLUMN progress_id INTEGER
                    """)
                    conn.execute("""
                        UPDATE topic_progress 
                        SET progress_id = id
                    """)
                
                # Add all missing columns
                for col_name, col_def in required_columns.items():
                    if col_name not in columns:
                        logger.info(f"Adding {col_name} column...")
                        try:
                            conn.execute(f"""
                                ALTER TABLE topic_progress 
                                ADD COLUMN {col_name} {col_def}
                            """)
                            logger.info(f"✅ Added {col_name} column")
                        except Exception as col_err:
                            logger.warning(f"⚠️ Failed to add {col_name} column: {col_err}")
                
                # Change user_id to VARCHAR if it's INTEGER (SQLite limitation - can't change type directly)
                # This will be handled in queries with CAST if needed
                if columns.get('user_id') == 'INTEGER':
                    logger.info("Note: user_id is INTEGER, will use CAST in queries if needed")
                
                conn.commit()
                logger.info("✅ Topic Progress migration applied directly - all required columns added")
            else:
                # Create table with correct schema
                logger.info("Creating topic_progress table with correct schema...")
                conn.execute("""
                    CREATE TABLE topic_progress (
                        progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id VARCHAR(255) NOT NULL,
                        session_id VARCHAR(255) NOT NULL,
                        topic_id INTEGER NOT NULL,
                        questions_asked INTEGER DEFAULT 0,
                        average_understanding DECIMAL(3,2),
                        average_satisfaction DECIMAL(3,2),
                        last_question_timestamp TIMESTAMP,
                        mastery_level VARCHAR(20),
                        mastery_score DECIMAL(3,2),
                        is_ready_for_next BOOLEAN DEFAULT FALSE,
                        readiness_score DECIMAL(3,2),
                        time_spent_minutes INTEGER DEFAULT 0,
                        first_question_timestamp TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, session_id, topic_id),
                        FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_topic_progress_user_topic 
                    ON topic_progress(user_id, topic_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_topic_progress_session 
                    ON topic_progress(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_topic_progress_topic 
                    ON topic_progress(topic_id)
                """)
                
                conn.commit()
                logger.info("Topic Progress table created with correct schema")
                
        except Exception as e:
            logger.error(f"Failed to apply topic_progress migration directly: {e}", exc_info=True)
            raise
    
    def _apply_topic_progress_migration_with_data_copy(self, conn: sqlite3.Connection, old_columns: dict):
        """Apply topic_progress migration with data copy (handles old schema)"""
        try:
            logger.info("Creating new topic_progress table with correct schema...")
            
            # Create new table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_progress_new (
                    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    topic_id INTEGER NOT NULL,
                    questions_asked INTEGER DEFAULT 0,
                    average_understanding DECIMAL(3,2),
                    average_satisfaction DECIMAL(3,2),
                    last_question_timestamp TIMESTAMP,
                    mastery_level VARCHAR(20),
                    mastery_score DECIMAL(3,2),
                    is_ready_for_next BOOLEAN DEFAULT FALSE,
                    readiness_score DECIMAL(3,2),
                    time_spent_minutes INTEGER DEFAULT 0,
                    first_question_timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, session_id, topic_id),
                    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
                )
            """)
            
            # Copy data with safe column mapping
            # Build SELECT statement based on what columns exist
            select_parts = []
            
            # progress_id (handle both 'id' and 'progress_id')
            if 'progress_id' in old_columns:
                select_parts.append("progress_id")
            elif 'id' in old_columns:
                select_parts.append("id as progress_id")
            else:
                select_parts.append("NULL as progress_id")
            
            # user_id
            select_parts.append("CAST(user_id AS VARCHAR(255)) as user_id")
            
            # session_id
            if 'session_id' in old_columns:
                select_parts.append("session_id")
            else:
                select_parts.append("'' as session_id")
            
            # topic_id
            if 'topic_id' in old_columns:
                select_parts.append("topic_id")
            else:
                select_parts.append("NULL as topic_id")
            
            # questions_asked
            if 'questions_asked' in old_columns:
                select_parts.append("COALESCE(questions_asked, 0) as questions_asked")
            else:
                select_parts.append("0 as questions_asked")
            
            # average_understanding
            if 'average_understanding' in old_columns:
                select_parts.append("CAST(COALESCE(average_understanding, 0.0) AS DECIMAL(3,2)) as average_understanding")
            else:
                select_parts.append("NULL as average_understanding")
            
            # average_satisfaction
            if 'average_satisfaction' in old_columns:
                select_parts.append("CAST(COALESCE(average_satisfaction, NULL) AS DECIMAL(3,2)) as average_satisfaction")
            else:
                select_parts.append("NULL as average_satisfaction")
            
            # last_question_timestamp
            if 'last_question_timestamp' in old_columns:
                select_parts.append("last_question_timestamp")
            elif 'first_interaction_timestamp' in old_columns:
                select_parts.append("first_interaction_timestamp as last_question_timestamp")
            else:
                select_parts.append("NULL as last_question_timestamp")
            
            # mastery_level
            if 'mastery_level' in old_columns:
                select_parts.append("mastery_level")
            else:
                select_parts.append("NULL as mastery_level")
            
            # mastery_score
            if 'mastery_score' in old_columns:
                select_parts.append("CAST(COALESCE(mastery_score, 0.0) AS DECIMAL(3,2)) as mastery_score")
            else:
                select_parts.append("NULL as mastery_score")
            
            # is_ready_for_next
            if 'is_ready_for_next' in old_columns:
                select_parts.append("COALESCE(is_ready_for_next, 0) as is_ready_for_next")
            else:
                select_parts.append("0 as is_ready_for_next")
            
            # readiness_score
            if 'readiness_score' in old_columns:
                select_parts.append("CAST(COALESCE(readiness_score, 0.0) AS DECIMAL(3,2)) as readiness_score")
            else:
                select_parts.append("NULL as readiness_score")
            
            # time_spent_minutes
            if 'time_spent_minutes' in old_columns:
                select_parts.append("COALESCE(time_spent_minutes, 0) as time_spent_minutes")
            else:
                select_parts.append("0 as time_spent_minutes")
            
            # first_question_timestamp
            if 'first_question_timestamp' in old_columns:
                select_parts.append("first_question_timestamp")
            elif 'first_interaction_timestamp' in old_columns:
                select_parts.append("first_interaction_timestamp as first_question_timestamp")
            else:
                select_parts.append("NULL as first_question_timestamp")
            
            # created_at
            if 'created_at' in old_columns:
                select_parts.append("COALESCE(created_at, CURRENT_TIMESTAMP) as created_at")
            else:
                select_parts.append("CURRENT_TIMESTAMP as created_at")
            
            # updated_at
            if 'updated_at' in old_columns:
                select_parts.append("COALESCE(updated_at, created_at, CURRENT_TIMESTAMP) as updated_at")
            elif 'created_at' in old_columns:
                select_parts.append("COALESCE(created_at, CURRENT_TIMESTAMP) as updated_at")
            else:
                select_parts.append("CURRENT_TIMESTAMP as updated_at")
            
            # Build and execute INSERT statement
            insert_sql = f"""
                INSERT INTO topic_progress_new 
                (progress_id, user_id, session_id, topic_id, questions_asked, average_understanding, 
                 average_satisfaction, last_question_timestamp, mastery_level, mastery_score, 
                 is_ready_for_next, readiness_score, time_spent_minutes, first_question_timestamp, 
                 created_at, updated_at)
                SELECT {', '.join(select_parts)}
                FROM topic_progress
            """
            
            logger.info("Copying data from old table to new table...")
            conn.execute(insert_sql)
            
            # Drop old table
            logger.info("Dropping old table...")
            conn.execute("DROP TABLE IF EXISTS topic_progress")
            
            # Rename new table
            logger.info("Renaming new table...")
            conn.execute("ALTER TABLE topic_progress_new RENAME TO topic_progress")
            
            # Create indexes
            logger.info("Creating indexes...")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_progress_user_topic 
                ON topic_progress(user_id, topic_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_progress_session 
                ON topic_progress(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_progress_topic 
                ON topic_progress(topic_id)
            """)
            
            conn.commit()
            logger.info("✅ Topic Progress migration with data copy applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply topic_progress migration with data copy: {e}", exc_info=True)
            conn.rollback()
            raise
    
    def apply_qa_embeddings_migration(self, conn: sqlite3.Connection):
        """Apply QA Question Embeddings migration (011_add_qa_question_embeddings.sql)"""
        try:
            # Check if migration is already applied by checking if question_embedding column exists
            cursor = conn.execute("PRAGMA table_info(topic_qa_pairs)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            if 'question_embedding' in columns:
                logger.info("QA embeddings migration already applied")
                return
            
            logger.info("Applying QA Question Embeddings migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/011_add_qa_question_embeddings.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/011_add_qa_question_embeddings.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/011_add_qa_question_embeddings.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("QA Question Embeddings migration applied successfully")
            else:
                logger.warning(f"QA Embeddings migration file not found. Expected paths: {possible_paths}")
                # Apply migration directly if file not found
                logger.info("Applying QA Embeddings migration directly...")
                
                try:
                    # Check if columns already exist
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA table_info(topic_qa_pairs)")
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    # Add columns if they don't exist
                    if 'question_embedding' not in columns:
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN question_embedding TEXT
                        """)
                        conn.commit()
                    
                    if 'embedding_model' not in columns:
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_model VARCHAR(100)
                        """)
                        conn.commit()
                    
                    if 'embedding_dim' not in columns:
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_dim INTEGER
                        """)
                        conn.commit()
                    
                    if 'embedding_updated_at' not in columns:
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_updated_at TIMESTAMP
                        """)
                        conn.commit()
                    
                    # Create indexes if they don't exist
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='index' AND name='idx_qa_pairs_topic_active'
                    """)
                    if not cursor.fetchone():
                        conn.execute("""
                            CREATE INDEX idx_qa_pairs_topic_active 
                            ON topic_qa_pairs(topic_id, is_active)
                        """)
                        conn.commit()
                    
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='index' AND name='idx_qa_pairs_embedding_model'
                    """)
                    if not cursor.fetchone():
                        conn.execute("""
                            CREATE INDEX idx_qa_pairs_embedding_model 
                            ON topic_qa_pairs(embedding_model)
                        """)
                        conn.commit()
                    
                    logger.info("QA Question Embeddings migration applied successfully")
                    
                except Exception as e:
                    logger.error(f"Error applying QA Embeddings migration: {e}")
                    raise
                
        except Exception as e:
            logger.warning(f"Failed to apply QA Embeddings migration (non-critical): {e}")
    
    def apply_satisfaction_fix_migration(self, conn: sqlite3.Connection):
        """Apply Satisfaction Values Fix migration (012_fix_satisfaction_values.sql)"""
        try:
            logger.info("Applying Satisfaction Values Fix migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/012_fix_satisfaction_values.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/012_fix_satisfaction_values.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/012_fix_satisfaction_values.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("Satisfaction Values Fix migration applied successfully")
            else:
                logger.warning(f"Satisfaction Fix migration file not found. Expected paths: {possible_paths}")
                # Apply migration directly if file not found
                logger.info("Applying Satisfaction Fix migration directly...")
                
                # Check if detailed_feedback_entries table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='detailed_feedback_entries'
                """)
                has_detailed_feedback = cursor.fetchone() is not None
                
                if has_detailed_feedback:
                    # Update profiles where satisfaction = understanding and no multi-dimensional feedback exists
                    conn.execute("""
                        UPDATE student_profiles
                        SET average_satisfaction = NULL
                        WHERE average_understanding IS NOT NULL
                          AND average_satisfaction IS NOT NULL
                          AND ABS(average_understanding - average_satisfaction) < 0.01
                          AND user_id || '|' || session_id NOT IN (
                            SELECT DISTINCT user_id || '|' || session_id
                            FROM detailed_feedback_entries
                            WHERE feedback_type = 'multi_dimensional'
                          )
                    """)
                else:
                    # If detailed_feedback_entries doesn't exist, just set satisfaction to NULL
                    # where it equals understanding (indicating it was incorrectly set)
                    conn.execute("""
                        UPDATE student_profiles
                        SET average_satisfaction = NULL
                        WHERE average_understanding IS NOT NULL
                          AND average_satisfaction IS NOT NULL
                          AND ABS(average_understanding - average_satisfaction) < 0.01
                    """)
                
                conn.commit()
                logger.info("Satisfaction Values Fix migration applied directly")
                
        except Exception as e:
            logger.warning(f"Failed to apply Satisfaction Fix migration (non-critical): {e}")
    
    def apply_detailed_feedback_migration(self, conn: sqlite3.Connection):
        """Apply Detailed Feedback Entries migration (014_create_detailed_feedback_entries.sql)"""
        try:
            # Check if detailed_feedback_entries table already exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='detailed_feedback_entries'
            """)
            
            if cursor.fetchone():
                logger.info("detailed_feedback_entries table already exists")
                return
            
            logger.info("Applying Detailed Feedback Entries migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/014_create_detailed_feedback_entries.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/014_create_detailed_feedback_entries.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/014_create_detailed_feedback_entries.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("Detailed Feedback Entries migration applied successfully")
            else:
                logger.warning(f"Detailed Feedback migration file not found. Expected paths: {possible_paths}")
                # Apply migration directly if file not found
                logger.info("Applying Detailed Feedback migration directly...")
                
                # Create detailed_feedback_entries table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS detailed_feedback_entries (
                        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interaction_id INTEGER NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        session_id VARCHAR(255) NOT NULL,
                        understanding_score INTEGER NOT NULL CHECK(understanding_score >= 1 AND understanding_score <= 5),
                        relevance_score INTEGER NOT NULL CHECK(relevance_score >= 1 AND relevance_score <= 5),
                        clarity_score INTEGER NOT NULL CHECK(clarity_score >= 1 AND clarity_score <= 5),
                        overall_score DECIMAL(3,2) NOT NULL,
                        emoji_feedback TEXT,
                        emoji_score DECIMAL(3,2),
                        comment TEXT,
                        feedback_type VARCHAR(50) DEFAULT 'multi_dimensional',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE CASCADE
                    )
                """)
                
                # Create multi_feedback_summary table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS multi_feedback_summary (
                        summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id VARCHAR(255) NOT NULL,
                        session_id VARCHAR(255) NOT NULL,
                        avg_understanding DECIMAL(3,2),
                        avg_relevance DECIMAL(3,2),
                        avg_clarity DECIMAL(3,2),
                        avg_overall DECIMAL(3,2),
                        total_feedback_count INTEGER DEFAULT 0,
                        dimension_feedback_count INTEGER DEFAULT 0,
                        emoji_only_count INTEGER DEFAULT 0,
                        understanding_distribution TEXT,
                        relevance_distribution TEXT,
                        clarity_distribution TEXT,
                        improvement_trend VARCHAR(50) DEFAULT 'insufficient_data',
                        weak_dimensions TEXT,
                        strong_dimensions TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, session_id)
                    )
                """)
                
                # Create indexes
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detailed_feedback_interaction 
                    ON detailed_feedback_entries(interaction_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detailed_feedback_user_session 
                    ON detailed_feedback_entries(user_id, session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_multi_feedback_summary_user_session 
                    ON multi_feedback_summary(user_id, session_id)
                """)
                
                conn.commit()
                logger.info("Detailed Feedback Entries migration applied directly")
                
        except Exception as e:
            logger.warning(f"Failed to apply Detailed Feedback migration (non-critical): {e}")
    
    def apply_analytics_views(self, conn: sqlite3.Connection):
        """Apply Topic Analytics Views"""
        try:
            logger.info("Applying Topic Analytics Views...")
            
            # Drop existing views first to ensure they're updated
            drop_views_sql = """
                DROP VIEW IF EXISTS topic_mastery_analytics;
                DROP VIEW IF EXISTS student_topic_progress_analytics;
                DROP VIEW IF EXISTS topic_difficulty_analysis;
                DROP VIEW IF EXISTS topic_recommendation_insights;
            """
            conn.executescript(drop_views_sql)
            
            # Read analytics views SQL file
            views_path = os.path.join(os.path.dirname(__file__), "topic_analytics_views.sql")
            
            if os.path.exists(views_path):
                with open(views_path, 'r', encoding='utf-8') as f:
                    views_sql = f.read()
                
                # Execute views creation
                conn.executescript(views_sql)
                conn.commit()
                logger.info("Topic Analytics Views applied successfully")
            else:
                logger.warning(f"Topic Analytics Views file not found at: {views_path}")
                
        except Exception as e:
            logger.warning(f"Failed to apply Topic Analytics Views (non-critical): {e}")
    
    def apply_session_settings_migration(self, conn: sqlite3.Connection):
        """Apply Session Settings migration (006_create_session_settings.sql)"""
        try:
            # Check if session_settings table already exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='session_settings'
            """)
            
            if cursor.fetchone():
                logger.info("Session settings table already exists")
                return
            
            logger.info("Applying Session Settings migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/006_create_session_settings.sql",  # Docker volume mount path
                os.path.join(
                    os.path.dirname(__file__),
                    "../../auth_service/database/migrations/006_create_session_settings.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../services/auth_service/database/migrations/006_create_session_settings.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../../services/auth_service/database/migrations/006_create_session_settings.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("Session Settings migration applied successfully")
            else:
                logger.warning(f"Session Settings migration file not found. Expected paths: {possible_paths}")
                logger.info("Session Settings will be handled by auth_service migration system")
                
        except Exception as e:
            logger.warning(f"Failed to apply Session Settings migration (non-critical): {e}")
    
    def apply_session_settings_fk_removal_migration(self, conn: sqlite3.Connection):
        """Apply Session Settings FK Removal migration (007_remove_session_settings_fk.sql)"""
        try:
            # Check if session_settings table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='session_settings'
            """)
            
            if not cursor.fetchone():
                logger.info("session_settings table does not exist, skipping FK removal migration")
                return
            
            # Check if migration already applied by checking table structure
            # If the table was recreated without FK, the migration was already applied
            cursor = conn.execute("PRAGMA foreign_key_list(session_settings)")
            fk_list = cursor.fetchall()
            
            # If no foreign keys exist, migration was already applied
            if not fk_list:
                logger.info("Session Settings FK removal migration already applied (no FK constraints found)")
                return
            
            logger.info("Applying Session Settings FK Removal migration...")
            
            # Drop views that depend on tables we're modifying (to avoid errors)
            try:
                conn.execute("DROP VIEW IF EXISTS student_topic_progress_analytics")
                conn.execute("DROP VIEW IF EXISTS topic_mastery_analytics")
                conn.execute("DROP VIEW IF EXISTS topic_difficulty_analysis")
                conn.execute("DROP VIEW IF EXISTS topic_recommendation_insights")
                conn.commit()
            except Exception as e:
                logger.warning(f"Could not drop views (may not exist): {e}")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/007_remove_session_settings_fk.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/007_remove_session_settings_fk.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/007_remove_session_settings_fk.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration with error handling for missing enable_ebars column
                try:
                    # Check if session_settings table exists and if enable_ebars column exists
                    table_exists = False
                    enable_ebars_exists = False
                    
                    try:
                        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_settings'")
                        table_exists = cursor.fetchone() is not None
                        
                        if table_exists:
                            # Check if enable_ebars column exists
                            cursor = conn.execute("PRAGMA table_info(session_settings)")
                            columns = [row[1] for row in cursor.fetchall()]
                            enable_ebars_exists = 'enable_ebars' in columns
                    except Exception:
                        pass
                    
                    # Split migration SQL: execute parts before Step 2, handle Step 2 manually, then execute rest
                    parts = migration_sql.split('-- Step 2:')
                    before_step2 = parts[0]
                    after_step2 = parts[1].split('-- Step 3:')[1] if '-- Step 3:' in parts[1] else ''
                    
                    # Execute parts before Step 2 (drop views, create new table)
                    conn.executescript(before_step2)
                    
                    # Handle Step 2: Copy data manually based on column existence
                    if table_exists:
                        if enable_ebars_exists:
                            # Column exists in old table, use it
                            conn.execute("""
                                INSERT INTO session_settings_new 
                                SELECT setting_id, session_id, user_id, 
                                       enable_progressive_assessment, enable_personalized_responses,
                                       enable_multi_dimensional_feedback, enable_topic_analytics,
                                       enable_cacs, enable_zpd, enable_bloom, enable_cognitive_load,
                                       enable_emoji_feedback, enable_ebars,
                                       created_at, updated_at
                                FROM session_settings
                            """)
                        else:
                            # Column doesn't exist in old table, use FALSE as default
                            conn.execute("""
                                INSERT INTO session_settings_new 
                                (setting_id, session_id, user_id, 
                                 enable_progressive_assessment, enable_personalized_responses,
                                 enable_multi_dimensional_feedback, enable_topic_analytics,
                                 enable_cacs, enable_zpd, enable_bloom, enable_cognitive_load,
                                 enable_emoji_feedback, enable_ebars,
                                 created_at, updated_at)
                                SELECT setting_id, session_id, user_id, 
                                       enable_progressive_assessment, enable_personalized_responses,
                                       enable_multi_dimensional_feedback, enable_topic_analytics,
                                       enable_cacs, enable_zpd, enable_bloom, enable_cognitive_load,
                                       enable_emoji_feedback, FALSE,
                                       created_at, updated_at
                                FROM session_settings
                            """)
                    
                    # Execute remaining parts (drop old table, rename, indexes, trigger)
                    if after_step2:
                        conn.executescript(after_step2)
                    
                    conn.commit()
                    logger.info("✅ Session Settings FK Removal migration applied successfully")
                except sqlite3.OperationalError as e:
                    error_msg = str(e).lower()
                    if "no such column" in error_msg or "has no column" in error_msg:
                        logger.info("Column mismatch detected, applying migration manually...")
                        # Apply migration manually
                        conn.execute("PRAGMA foreign_keys = OFF")
                        conn.execute("DROP VIEW IF EXISTS student_topic_progress_analytics")
                        conn.execute("DROP VIEW IF EXISTS topic_mastery_analytics")
                        conn.execute("DROP VIEW IF EXISTS topic_difficulty_analysis")
                        conn.execute("DROP VIEW IF EXISTS topic_recommendation_insights")
                        
                        # Create new table
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS session_settings_new (
                                setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id VARCHAR(255) NOT NULL,
                                user_id VARCHAR(255) NOT NULL,
                                enable_progressive_assessment BOOLEAN NOT NULL DEFAULT FALSE,
                                enable_personalized_responses BOOLEAN NOT NULL DEFAULT FALSE,
                                enable_multi_dimensional_feedback BOOLEAN NOT NULL DEFAULT FALSE,
                                enable_topic_analytics BOOLEAN NOT NULL DEFAULT TRUE,
                                enable_cacs BOOLEAN NOT NULL DEFAULT TRUE,
                                enable_zpd BOOLEAN NOT NULL DEFAULT TRUE,
                                enable_bloom BOOLEAN NOT NULL DEFAULT TRUE,
                                enable_cognitive_load BOOLEAN NOT NULL DEFAULT TRUE,
                                enable_emoji_feedback BOOLEAN NOT NULL DEFAULT TRUE,
                                enable_ebars BOOLEAN NOT NULL DEFAULT FALSE,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                UNIQUE(session_id)
                            )
                        """)
                        
                        # Copy data - check which columns exist
                        try:
                            cursor = conn.execute("PRAGMA table_info(session_settings)")
                            columns = [row[1] for row in cursor.fetchall()]
                            
                            if 'enable_ebars' in columns:
                                conn.execute("""
                                    INSERT INTO session_settings_new 
                                    SELECT setting_id, session_id, user_id, 
                                           enable_progressive_assessment, enable_personalized_responses,
                                           enable_multi_dimensional_feedback, enable_topic_analytics,
                                           enable_cacs, enable_zpd, enable_bloom, enable_cognitive_load,
                                           enable_emoji_feedback, enable_ebars,
                                           created_at, updated_at
                                    FROM session_settings
                                """)
                            else:
                                conn.execute("""
                                    INSERT INTO session_settings_new 
                                    SELECT setting_id, session_id, user_id, 
                                           enable_progressive_assessment, enable_personalized_responses,
                                           enable_multi_dimensional_feedback, enable_topic_analytics,
                                           enable_cacs, enable_zpd, enable_bloom, enable_cognitive_load,
                                           enable_emoji_feedback, FALSE,
                                           created_at, updated_at
                                    FROM session_settings
                                """)
                        except sqlite3.OperationalError as copy_error:
                            logger.warning(f"Could not copy data: {copy_error}")
                        
                        conn.execute("DROP TABLE IF EXISTS session_settings")
                        conn.execute("ALTER TABLE session_settings_new RENAME TO session_settings")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_settings_session_id ON session_settings(session_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_settings_user_id ON session_settings(user_id)")
                        
                        # Recreate trigger
                        conn.execute("""
                            DROP TRIGGER IF EXISTS update_session_settings_updated_at
                        """)
                        conn.execute("""
                            CREATE TRIGGER IF NOT EXISTS update_session_settings_updated_at
                                AFTER UPDATE ON session_settings
                                FOR EACH ROW
                                WHEN NEW.updated_at <= OLD.updated_at
                            BEGIN
                                UPDATE session_settings SET updated_at = CURRENT_TIMESTAMP WHERE setting_id = NEW.setting_id;
                            END
                        """)
                        
                        conn.execute("PRAGMA foreign_keys = ON")
                        conn.commit()
                        logger.info("✅ Session Settings FK Removal migration applied manually")
                    else:
                        raise
            else:
                logger.warning(f"Session Settings FK Removal migration file not found. Expected paths: {possible_paths}")
                logger.info("Applying FK removal migration directly...")
                
                # Apply migration directly
                try:
                    # Disable foreign keys temporarily
                    conn.execute("PRAGMA foreign_keys = OFF")
                    
                    # Create new table without FK
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS session_settings_new (
                            setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id VARCHAR(255) NOT NULL,
                            user_id VARCHAR(255) NOT NULL,
                            enable_progressive_assessment BOOLEAN NOT NULL DEFAULT FALSE,
                            enable_personalized_responses BOOLEAN NOT NULL DEFAULT FALSE,
                            enable_multi_dimensional_feedback BOOLEAN NOT NULL DEFAULT FALSE,
                            enable_topic_analytics BOOLEAN NOT NULL DEFAULT TRUE,
                            enable_cacs BOOLEAN NOT NULL DEFAULT TRUE,
                            enable_zpd BOOLEAN NOT NULL DEFAULT TRUE,
                            enable_bloom BOOLEAN NOT NULL DEFAULT TRUE,
                            enable_cognitive_load BOOLEAN NOT NULL DEFAULT TRUE,
                            enable_emoji_feedback BOOLEAN NOT NULL DEFAULT TRUE,
                            enable_ebars BOOLEAN NOT NULL DEFAULT FALSE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(session_id)
                        )
                    """)
                    
                    # Copy data
                    conn.execute("""
                        INSERT INTO session_settings_new 
                        SELECT setting_id, session_id, user_id, 
                               enable_progressive_assessment, enable_personalized_responses,
                               enable_multi_dimensional_feedback, enable_topic_analytics,
                               enable_cacs, enable_zpd, enable_bloom, enable_cognitive_load,
                               enable_emoji_feedback, 
                               COALESCE(enable_ebars, FALSE) as enable_ebars,
                               created_at, updated_at
                        FROM session_settings
                    """)
                    
                    # Drop old table
                    conn.execute("DROP TABLE IF EXISTS session_settings")
                    
                    # Rename new table
                    conn.execute("ALTER TABLE session_settings_new RENAME TO session_settings")
                    
                    # Recreate indexes
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_session_settings_session_id 
                        ON session_settings(session_id)
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_session_settings_user_id 
                        ON session_settings(user_id)
                    """)
                    
                    # Re-enable foreign keys
                    conn.execute("PRAGMA foreign_keys = ON")
                    
                    conn.commit()
                    logger.info("✅ Session Settings FK Removal migration applied directly")
                    
                except Exception as e:
                    conn.execute("PRAGMA foreign_keys = ON")  # Re-enable on error
                    raise
                
        except Exception as e:
            logger.error(f"Failed to apply Session Settings FK Removal migration: {e}", exc_info=True)
            # Don't raise - this is non-critical, but log the error
    
    def apply_document_global_scores_migration(self, conn: sqlite3.Connection):
        """Apply Document Global Scores migration (007_create_document_global_scores.sql)"""
        try:
            # Check if document_global_scores table already exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='document_global_scores'
            """)
            
            if cursor.fetchone():
                logger.info("document_global_scores table already exists")
                return
            
            logger.info("Applying Document Global Scores migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/007_create_document_global_scores.sql",  # Docker volume mount path
                os.path.join(
                    os.path.dirname(__file__),
                    "../../auth_service/database/migrations/007_create_document_global_scores.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../services/auth_service/database/migrations/007_create_document_global_scores.sql"
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../../services/auth_service/database/migrations/007_create_document_global_scores.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("Document Global Scores migration applied successfully")
            else:
                logger.warning(f"Document Global Scores migration file not found. Expected paths: {possible_paths}")
                logger.info("Document Global Scores will be handled by auth_service migration system")
                
        except Exception as e:
            logger.warning(f"Failed to apply Document Global Scores migration (non-critical): {e}")
    
    def apply_avg_emoji_score_migration(self, conn: sqlite3.Connection):
        """Apply avg_emoji_score column migration (013_add_avg_emoji_score_to_document_global_scores.sql)"""
        try:
            # Check if avg_emoji_score column already exists
            cursor = conn.execute("PRAGMA table_info(document_global_scores)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            if 'avg_emoji_score' in columns:
                logger.info("avg_emoji_score column already exists in document_global_scores")
                return
            
            logger.info("Applying avg_emoji_score migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/013_add_avg_emoji_score_to_document_global_scores.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/013_add_avg_emoji_score_to_document_global_scores.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/013_add_avg_emoji_score_to_document_global_scores.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("avg_emoji_score migration applied successfully")
            else:
                logger.warning(f"avg_emoji_score migration file not found. Expected paths: {possible_paths}")
                # Apply migration directly if file not found
                logger.info("Applying avg_emoji_score migration directly...")
                conn.execute("""
                    ALTER TABLE document_global_scores 
                    ADD COLUMN avg_emoji_score DECIMAL(5,3) DEFAULT 0.5
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_global_scores_avg_emoji_score 
                    ON document_global_scores(avg_emoji_score DESC)
                """)
                conn.commit()
                logger.info("avg_emoji_score migration applied directly")
                
        except Exception as e:
            logger.warning(f"Failed to apply avg_emoji_score migration (non-critical): {e}")
    
    def _create_aprag_tables_manual(self, conn: sqlite3.Connection):
        """Manually create APRAG tables if migration file is not available"""
        # This is a fallback - should use migration file in production
        logger.warning("Using manual table creation (fallback method)")
        # Tables will be created on first use if needed
        pass
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dicts
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries representing rows
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT query and return the last row ID
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Last inserted row ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def ensure_feature_flags_table(self, conn: sqlite3.Connection):
        """
        Ensure feature_flags table exists with correct schema
        This is a KALICI (permanent) fix - runs on every database init
        """
        try:
            # Check if table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='feature_flags'
            """)
            
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # Create table with feature_enabled column
                logger.info("Creating feature_flags table...")
                conn.execute("""
                    CREATE TABLE feature_flags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feature_name TEXT NOT NULL,
                        session_id TEXT,
                        feature_enabled BOOLEAN NOT NULL DEFAULT 1,
                        config_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(feature_name, session_id)
                    )
                """)
                logger.info("✅ feature_flags table created")
            else:
                # Table exists - check if feature_enabled column exists
                cursor = conn.execute("PRAGMA table_info(feature_flags)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'feature_enabled' not in columns:
                    # Add feature_enabled column if missing
                    logger.info("Adding feature_enabled column to feature_flags table...")
                    conn.execute("""
                        ALTER TABLE feature_flags 
                        ADD COLUMN feature_enabled BOOLEAN NOT NULL DEFAULT 1
                    """)
                    logger.info("✅ feature_enabled column added")
                
                # Also check for is_enabled (legacy column) and migrate if needed
                if 'is_enabled' in columns and 'feature_enabled' in columns:
                    # Migrate data from is_enabled to feature_enabled if feature_enabled is NULL
                    conn.execute("""
                        UPDATE feature_flags 
                        SET feature_enabled = is_enabled 
                        WHERE feature_enabled IS NULL
                    """)
                    logger.info("✅ Migrated is_enabled to feature_enabled")
                
        except Exception as e:
            logger.error(f"Error ensuring feature_flags table: {e}")
            # Don't raise - let the system continue
    
    def apply_knowledge_base_tables_migration(self, conn: sqlite3.Connection):
        """Apply Knowledge Base tables migration (topic_knowledge_base, topic_qa_pairs, etc.)"""
        try:
            # Check if topic_knowledge_base table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='topic_knowledge_base'
            """)
            
            if cursor.fetchone():
                logger.info("Knowledge base tables already exist")
                return
            
            logger.info("Applying Knowledge Base tables migration...")
            
            # Create topic_knowledge_base table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_knowledge_base (
                    knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL,
                    topic_summary TEXT NOT NULL,
                    key_concepts TEXT,
                    learning_objectives TEXT,
                    definitions TEXT,
                    formulas TEXT,
                    examples TEXT,
                    related_topics TEXT,
                    prerequisite_concepts TEXT,
                    real_world_applications TEXT,
                    common_misconceptions TEXT,
                    content_quality_score DECIMAL(3,2) DEFAULT 0.80,
                    extraction_model VARCHAR(100),
                    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_validated BOOLEAN DEFAULT FALSE,
                    validation_date TIMESTAMP,
                    validator_user_id VARCHAR(255),
                    view_count INTEGER DEFAULT 0,
                    usefulness_rating DECIMAL(3,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE,
                    UNIQUE(topic_id)
                )
            """)
            
            # Create topic_qa_pairs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_qa_pairs (
                    qa_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    explanation TEXT,
                    difficulty_level VARCHAR(20) NOT NULL,
                    question_type VARCHAR(50) NOT NULL,
                    bloom_taxonomy_level VARCHAR(50),
                    cognitive_complexity VARCHAR(20),
                    source_chunk_ids TEXT,
                    extraction_method VARCHAR(50) DEFAULT 'llm_generated',
                    extraction_model VARCHAR(100),
                    quality_score DECIMAL(3,2) DEFAULT 0.75,
                    is_validated BOOLEAN DEFAULT FALSE,
                    validator_user_id VARCHAR(255),
                    validation_notes TEXT,
                    times_asked INTEGER DEFAULT 0,
                    times_matched INTEGER DEFAULT 0,
                    average_student_rating DECIMAL(3,2),
                    success_rate DECIMAL(3,2),
                    average_followup_count INTEGER DEFAULT 0,
                    related_qa_ids TEXT,
                    related_concepts TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_featured BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_kb_topic_id ON topic_knowledge_base(topic_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_qa_topic_id ON topic_qa_pairs(topic_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_qa_active ON topic_qa_pairs(is_active)
            """)
            
            conn.commit()
            logger.info("✅ Knowledge Base tables migration applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply Knowledge Base tables migration: {e}", exc_info=True)
            # Don't raise - let the system continue, but log the error
    
    def apply_ebars_migration(self, conn: sqlite3.Connection):
        """Apply EBARS tables migration (015_create_ebars_tables.sql)"""
        try:
            # Check if student_comprehension_scores table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='student_comprehension_scores'
            """)
            
            if cursor.fetchone():
                logger.info("EBARS tables already exist")
                return
            
            logger.info("Applying EBARS tables migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/015_create_ebars_tables.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/015_create_ebars_tables.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/015_create_ebars_tables.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("✅ EBARS tables migration applied successfully")
            else:
                logger.warning(f"EBARS migration file not found. Expected paths: {possible_paths}")
                # Apply migration directly if file not found
                logger.info("Applying EBARS migration directly...")
                self._apply_ebars_migration_directly(conn)
                
        except Exception as e:
            logger.warning(f"Failed to apply EBARS migration (non-critical): {e}")
            # Don't raise - let the system continue
    
    def _apply_ebars_migration_directly(self, conn: sqlite3.Connection):
        """Apply EBARS migration directly (fallback if file not found)"""
        try:
            # Create student_comprehension_scores table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS student_comprehension_scores (
                    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    comprehension_score DECIMAL(5,2) NOT NULL DEFAULT 50.0,
                    current_difficulty_level VARCHAR(20) NOT NULL DEFAULT 'normal',
                    total_feedback_count INTEGER DEFAULT 0,
                    positive_feedback_count INTEGER DEFAULT 0,
                    negative_feedback_count INTEGER DEFAULT 0,
                    consecutive_positive_count INTEGER DEFAULT 0,
                    consecutive_negative_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_feedback_at TIMESTAMP,
                    UNIQUE(user_id, session_id)
                )
            """)
            
            # Create ebars_feedback_history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ebars_feedback_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    interaction_id INTEGER,
                    emoji_feedback TEXT NOT NULL,
                    previous_score DECIMAL(5,2) NOT NULL,
                    score_delta DECIMAL(5,2) NOT NULL,
                    new_score DECIMAL(5,2) NOT NULL,
                    previous_difficulty_level VARCHAR(20),
                    new_difficulty_level VARCHAR(20),
                    difficulty_changed BOOLEAN DEFAULT FALSE,
                    adjustment_type VARCHAR(50),
                    query_text TEXT,
                    response_preview TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE SET NULL
                )
            """)
            
            # Create ebars_prompt_cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ebars_prompt_cache (
                    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    difficulty_level VARCHAR(20) NOT NULL,
                    prompt_parameters TEXT NOT NULL,
                    full_prompt_text TEXT,
                    times_used INTEGER DEFAULT 0,
                    last_used_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, session_id, difficulty_level)
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_comprehension_scores_user_session 
                ON student_comprehension_scores(user_id, session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ebars_feedback_user_session 
                ON ebars_feedback_history(user_id, session_id)
            """)
            
            conn.commit()
            logger.info("✅ EBARS migration applied directly")
            
        except Exception as e:
            logger.error(f"Error applying EBARS migration directly: {e}")
    
    def apply_qa_embeddings_migration(self, conn: sqlite3.Connection):
        """Apply QA question embeddings migration (011_add_qa_question_embeddings.sql)"""
        try:
            # Check if topic_qa_pairs table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='topic_qa_pairs'
            """)
            
            if not cursor.fetchone():
                logger.info("topic_qa_pairs table does not exist, skipping QA embeddings migration")
                return
            
            # Check if question_embedding column already exists
            cursor = conn.execute("PRAGMA table_info(topic_qa_pairs)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            if 'question_embedding' in columns:
                logger.info("QA embeddings migration already applied (question_embedding column exists)")
                return
            
            logger.info("Applying QA embeddings migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/011_add_qa_question_embeddings.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/011_add_qa_question_embeddings.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/011_add_qa_question_embeddings.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("✅ QA embeddings migration applied successfully")
            else:
                logger.warning(f"QA embeddings migration file not found. Expected paths: {possible_paths}")
                # Apply migration directly if file not found
                logger.info("Applying QA embeddings migration directly...")
                conn.execute("""
                    ALTER TABLE topic_qa_pairs 
                    ADD COLUMN question_embedding TEXT
                """)
                conn.execute("""
                    ALTER TABLE topic_qa_pairs 
                    ADD COLUMN embedding_model VARCHAR(100)
                """)
                conn.execute("""
                    ALTER TABLE topic_qa_pairs 
                    ADD COLUMN embedding_dim INTEGER
                """)
                conn.execute("""
                    ALTER TABLE topic_qa_pairs 
                    ADD COLUMN embedding_updated_at TIMESTAMP
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_qa_pairs_topic_active 
                    ON topic_qa_pairs(topic_id, is_active)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_qa_pairs_embedding_model 
                    ON topic_qa_pairs(embedding_model)
                """)
                conn.commit()
                logger.info("✅ QA embeddings migration applied directly")
                
        except Exception as e:
            logger.warning(f"Failed to apply QA embeddings migration (non-critical): {e}")
            # Don't raise - let the system continue
    
    def apply_completion_percentage_migration(self, conn: sqlite3.Connection):
        """Apply completion_percentage column migration (016_add_completion_percentage_to_topic_progress.sql)"""
        try:
            # Check if topic_progress table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='topic_progress'
            """)
            
            if not cursor.fetchone():
                logger.info("topic_progress table does not exist, skipping completion_percentage migration")
                return
            
            # Check if completion_percentage column already exists
            cursor = conn.execute("PRAGMA table_info(topic_progress)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            if 'completion_percentage' in columns:
                logger.info("completion_percentage column already exists in topic_progress")
                return
            
            logger.info("Applying completion_percentage migration...")
            
            # Read migration file
            possible_paths = [
                "/app/migrations/016_add_completion_percentage_to_topic_progress.sql",  # Docker volume mount path
                os.path.join(os.path.dirname(__file__), "migrations/016_add_completion_percentage_to_topic_progress.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/migrations/016_add_completion_percentage_to_topic_progress.sql"
                ),
            ]
            
            migration_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("✅ completion_percentage migration applied successfully")
            else:
                logger.warning(f"completion_percentage migration file not found. Expected paths: {possible_paths}")
                # Apply migration directly if file not found
                logger.info("Applying completion_percentage migration directly...")
                try:
                    conn.execute("""
                        ALTER TABLE topic_progress 
                        ADD COLUMN completion_percentage DECIMAL(5,2) DEFAULT 0.0
                    """)
                    conn.execute("""
                        ALTER TABLE topic_progress 
                        ADD COLUMN last_interaction_date TIMESTAMP
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_topic_progress_completion 
                        ON topic_progress(completion_percentage DESC)
                    """)
                    # Update existing rows
                    conn.execute("""
                        UPDATE topic_progress 
                        SET completion_percentage = COALESCE(mastery_score * 100.0, 0.0)
                        WHERE completion_percentage IS NULL OR completion_percentage = 0.0
                    """)
                    conn.execute("""
                        UPDATE topic_progress 
                        SET last_interaction_date = last_question_timestamp
                        WHERE last_interaction_date IS NULL AND last_question_timestamp IS NOT NULL
                    """)
                    # Recreate view to use the new completion_percentage column
                    self._recreate_student_topic_progress_analytics_view(conn)
                    conn.commit()
                    logger.info("✅ completion_percentage migration applied directly")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        logger.info("completion_percentage column already exists (applied directly)")
                    else:
                        raise
                
        except Exception as e:
            logger.warning(f"Failed to apply completion_percentage migration (non-critical): {e}")
            # Don't raise - let the system continue
    
    def _recreate_student_topic_progress_analytics_view(self, conn: sqlite3.Connection):
        """Recreate student_topic_progress_analytics view with completion_percentage support"""
        try:
            # Read view definition from topic_analytics_views.sql
            possible_paths = [
                "/app/database/topic_analytics_views.sql",
                os.path.join(os.path.dirname(__file__), "topic_analytics_views.sql"),
                os.path.join(
                    os.path.dirname(__file__),
                    "../database/topic_analytics_views.sql"
                ),
            ]
            
            view_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    view_path = path
                    break
            
            if view_path and os.path.exists(view_path):
                with open(view_path, 'r', encoding='utf-8') as f:
                    view_sql = f.read()
                
                # Extract only the student_topic_progress_analytics view definition
                # Find the view creation statement
                import re
                pattern = r'CREATE VIEW IF NOT EXISTS student_topic_progress_analytics AS.*?;'
                match = re.search(pattern, view_sql, re.DOTALL | re.IGNORECASE)
                
                if match:
                    view_definition = match.group(0)
                    # Drop existing view first
                    conn.execute("DROP VIEW IF EXISTS student_topic_progress_analytics")
                    # Create new view
                    conn.execute(view_definition)
                    logger.info("✅ student_topic_progress_analytics view recreated")
                else:
                    logger.warning("Could not find student_topic_progress_analytics view definition in file")
            else:
                logger.warning(f"topic_analytics_views.sql file not found. Expected paths: {possible_paths}")
        except Exception as e:
            logger.warning(f"Failed to recreate student_topic_progress_analytics view (non-critical): {e}")

