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
                    self.apply_session_settings_migration(conn)
                    conn.commit()  # Ensure commit after migration
                    logger.info("APRAG migrations applied successfully")
                else:
                    logger.info("APRAG tables already exist")
                    # Always try to apply migrations to ensure schema is up to date
                    # (migration file uses IF NOT EXISTS, so it's safe)
                    self.apply_aprag_migrations(conn)
                    self.apply_topic_migrations(conn)
                    self.apply_foreign_key_fix_migration(conn)
                    self.apply_session_settings_migration(conn)
                    self.apply_analytics_views(conn)
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
    
    def apply_analytics_views(self, conn: sqlite3.Connection):
        """Apply Topic Analytics Views"""
        try:
            logger.info("Applying Topic Analytics Views...")
            
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

