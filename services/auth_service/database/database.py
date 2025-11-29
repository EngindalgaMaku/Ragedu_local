"""
Database connection and management utilities for RAG Education Assistant
Handles SQLite database connections, initialization, and migrations
"""

import sqlite3
import os
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import bcrypt
import logging

# Set up logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager for user authorization system
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
        """Initialize database with tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                # Check if tables exist
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='users'
                """)
                
                if not cursor.fetchone():
                    logger.info("Creating database tables...")
                    self.create_tables(conn)
                    self.insert_default_data(conn)
                    logger.info("Database initialized successfully")
                else:
                    logger.info("Database tables already exist")
                    # Apply migrations if needed
                    self.apply_migrations(conn)
                    
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def apply_migrations(self, conn: sqlite3.Connection):
        """Apply database migrations for existing databases"""
        try:
            # Check if user_sessions table exists before trying to modify it
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='user_sessions'
            """)
            
            if cursor.fetchone():
                # Check if last_activity column exists in user_sessions
                cursor = conn.execute("PRAGMA table_info(user_sessions)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'last_activity' not in columns:
                    logger.info("Adding last_activity column to user_sessions table")
                    # SQLite doesn't allow DEFAULT CURRENT_TIMESTAMP in ALTER TABLE
                    # So we add it as nullable first, then update existing rows
                    conn.execute("""
                        ALTER TABLE user_sessions
                        ADD COLUMN last_activity TIMESTAMP
                    """)
                    # Update existing rows with created_at value
                    conn.execute("""
                        UPDATE user_sessions
                        SET last_activity = created_at
                        WHERE last_activity IS NULL
                    """)
                    conn.commit()
                    logger.info("Migration: last_activity column added and populated")
            else:
                logger.info("user_sessions table not found, skipping last_activity migration")
            
            # Apply APRAG migrations (003_create_aprag_tables.sql)
            self.apply_aprag_migrations(conn)
            
            # Apply Topic migrations (004_create_topic_tables.sql)
            self.apply_topic_migrations(conn)
            
            # Apply Foreign Key Fix migration (005_fix_aprag_foreign_keys.sql)
            self.apply_foreign_key_fix_migration(conn)
            
            # Apply Module migrations (009_create_module_tables.sql)
            self.apply_module_migrations(conn)
                
        except Exception as e:
            logger.warning(f"Migration warning: {e}")
            # Even if other migrations fail, still try to apply module migrations
            try:
                logger.info("Attempting to apply module migrations despite earlier migration warnings...")
                self.apply_module_migrations(conn)
            except Exception as module_error:
                logger.warning(f"Module migration also failed: {module_error}")
    
    def create_aprag_tables(self, conn: sqlite3.Connection):
        """Create APRAG tables (for new databases)"""
        try:
            # Read migration file
            migration_paths = [
                os.path.join(os.path.dirname(__file__), "migrations/003_create_aprag_tables.sql"),
                os.path.join(os.path.dirname(__file__), "../database/migrations/003_create_aprag_tables.sql"),
            ]
            
            migration_path = None
            for path in migration_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration (IF NOT EXISTS will prevent errors if tables already exist)
                conn.executescript(migration_sql)
                logger.info("APRAG tables created successfully")
            else:
                logger.warning(f"APRAG migration file not found at expected paths: {migration_paths}")
                
        except Exception as e:
            logger.warning(f"APRAG table creation warning (non-critical): {e}")
    
    def create_topic_tables(self, conn: sqlite3.Connection):
        """Create Topic tables (for new databases)"""
        try:
            # Read migration file
            migration_paths = [
                os.path.join(os.path.dirname(__file__), "migrations/004_create_topic_tables.sql"),
                os.path.join(os.path.dirname(__file__), "../database/migrations/004_create_topic_tables.sql"),
            ]
            
            migration_path = None
            for path in migration_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration (IF NOT EXISTS will prevent errors if tables already exist)
                conn.executescript(migration_sql)
                logger.info("Topic tables created successfully")
            else:
                logger.warning(f"Topic migration file not found at expected paths: {migration_paths}")
                
        except Exception as e:
            logger.warning(f"Topic table creation warning (non-critical): {e}")
    
    def apply_aprag_migrations(self, conn: sqlite3.Connection):
        """Apply APRAG database migrations"""
        try:
            # Check if APRAG tables already exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='student_interactions'
            """)
            
            if cursor.fetchone():
                logger.info("APRAG tables already exist, skipping migration")
                return
            
            logger.info("Applying APRAG migrations...")
            
            # Read migration file
            migration_paths = [
                os.path.join(os.path.dirname(__file__), "migrations/003_create_aprag_tables.sql"),
                os.path.join(os.path.dirname(__file__), "../database/migrations/003_create_aprag_tables.sql"),
            ]
            
            migration_path = None
            for path in migration_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("APRAG migrations applied successfully")
            else:
                logger.warning(f"APRAG migration file not found at expected paths: {migration_paths}")
                
        except Exception as e:
            logger.warning(f"APRAG migration warning (non-critical): {e}")
    
    def apply_topic_migrations(self, conn: sqlite3.Connection):
        """Apply Topic-Based Learning Path Tracking migrations"""
        try:
            # Check if topic tables already exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='course_topics'
            """)
            
            if cursor.fetchone():
                logger.info("Topic tables already exist, skipping migration")
                return
            
            logger.info("Applying Topic migrations...")
            
            # Read migration file
            migration_paths = [
                os.path.join(os.path.dirname(__file__), "migrations/004_create_topic_tables.sql"),
                os.path.join(os.path.dirname(__file__), "../database/migrations/004_create_topic_tables.sql"),
            ]
            
            migration_path = None
            for path in migration_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration
                conn.executescript(migration_sql)
                conn.commit()
                logger.info("Topic migrations applied successfully")
            else:
                logger.warning(f"Topic migration file not found at expected paths: {migration_paths}")
                
        except Exception as e:
            logger.warning(f"Topic migration warning (non-critical): {e}")
    
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
            migration_paths = [
                os.path.join(os.path.dirname(__file__), "migrations/005_fix_aprag_foreign_keys.sql"),
                os.path.join(os.path.dirname(__file__), "../database/migrations/005_fix_aprag_foreign_keys.sql"),
            ]
            
            migration_path = None
            for path in migration_paths:
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
                logger.warning(f"Foreign Key Fix migration file not found at expected paths: {migration_paths}")
                
        except Exception as e:
            logger.warning(f"Foreign Key Fix migration warning (non-critical): {e}")
    
    def apply_module_migrations(self, conn: sqlite3.Connection):
        """Apply Module Extraction System migrations (009_create_module_tables.sql)"""
        try:
            # Check if ALL module tables exist (must check all 7, not just 'courses')
            required_tables = [
                'courses', 'course_modules', 'module_progress',
                'curriculum_standards', 'module_templates',
                'module_extraction_jobs', 'module_topic_relationships'
            ]
            
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ({})
            """.format(','.join('?' * len(required_tables))), required_tables)
            
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            # Only skip if ALL 7 tables exist
            if len(existing_tables) == len(required_tables):
                logger.info("All module tables already exist, skipping migration")
                return
            
            # Log which tables are missing for debugging
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                logger.info(f"Missing module tables detected: {missing_tables}")
            
            logger.info("Applying Module migrations...")
            
            # Read migration file
            migration_paths = [
                os.path.join(os.path.dirname(__file__), "migrations/009_create_module_tables.sql"),
                os.path.join(os.path.dirname(__file__), "../database/migrations/009_create_module_tables.sql"),
            ]
            
            migration_path = None
            for path in migration_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # CRITICAL FIX: Handle partial migration scenarios
                # If some tables exist but migration fails, create missing tables individually
                try:
                    # Try full migration first
                    conn.executescript(migration_sql)
                    conn.commit()
                    logger.info("Module migrations applied successfully")
                except Exception as migration_error:
                    logger.warning(f"Full migration failed: {migration_error}")
                    logger.info("Attempting to create missing tables individually...")
                    
                    # Create only the missing tables using individual CREATE statements
                    self._create_missing_module_tables(conn, missing_tables)
            else:
                logger.warning(f"Module migration file not found at expected paths: {migration_paths}")
                
        except Exception as e:
            logger.warning(f"Module migration warning (non-critical): {e}")
    
    def _create_missing_module_tables(self, conn: sqlite3.Connection, missing_tables):
        """Create missing module tables individually to handle partial migration scenarios"""
        try:
            # Define individual table creation statements for missing tables
            table_definitions = {
                'curriculum_standards': """
                    CREATE TABLE IF NOT EXISTS curriculum_standards (
                        standard_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        curriculum_type VARCHAR(100) NOT NULL,
                        subject_area VARCHAR(100) NOT NULL,
                        grade_level VARCHAR(50) NOT NULL,
                        standard_code VARCHAR(100) NOT NULL,
                        standard_title VARCHAR(500) NOT NULL,
                        standard_description TEXT,
                        parent_standard_id INTEGER,
                        standard_level INTEGER DEFAULT 1,
                        expected_outcomes TEXT,
                        assessment_criteria TEXT,
                        time_allocation_hours INTEGER,
                        official_source_url TEXT,
                        last_updated_official DATE,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (parent_standard_id) REFERENCES curriculum_standards(standard_id) ON DELETE SET NULL,
                        UNIQUE(curriculum_type, subject_area, grade_level, standard_code)
                    )
                """,
                'courses': """
                    CREATE TABLE IF NOT EXISTS courses (
                        course_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        course_code VARCHAR(50) NOT NULL,
                        course_name VARCHAR(255) NOT NULL,
                        course_description TEXT,
                        curriculum_standard VARCHAR(100) NOT NULL,
                        subject_area VARCHAR(100) NOT NULL,
                        grade_level VARCHAR(50) NOT NULL,
                        academic_year VARCHAR(20),
                        total_hours INTEGER,
                        difficulty_level VARCHAR(20) DEFAULT 'intermediate',
                        language VARCHAR(10) DEFAULT 'tr',
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(course_code)
                    )
                """,
                # Add other tables as needed...
            }
            
            # Create missing tables
            for table_name in missing_tables:
                if table_name in table_definitions:
                    logger.info(f"Creating missing table: {table_name}")
                    conn.execute(table_definitions[table_name])
            
            # Create indexes for curriculum_standards
            if 'curriculum_standards' in missing_tables:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_curriculum_standards_type ON curriculum_standards(curriculum_type, subject_area, grade_level)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_curriculum_standards_code ON curriculum_standards(standard_code)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_curriculum_standards_active ON curriculum_standards(is_active)")
            
            conn.commit()
            logger.info(f"Successfully created missing tables: {missing_tables}")
            
        except Exception as e:
            logger.error(f"Failed to create missing module tables: {e}")
    
    def create_module_tables(self, conn: sqlite3.Connection):
        """Create Module tables (for new databases)"""
        try:
            # Read migration file
            migration_paths = [
                os.path.join(os.path.dirname(__file__), "migrations/009_create_module_tables.sql"),
                os.path.join(os.path.dirname(__file__), "../database/migrations/009_create_module_tables.sql"),
            ]
            
            migration_path = None
            for path in migration_paths:
                if os.path.exists(path):
                    migration_path = path
                    break
            
            if migration_path and os.path.exists(migration_path):
                with open(migration_path, 'r', encoding='utf-8') as f:
                    migration_sql = f.read()
                
                # Execute migration (IF NOT EXISTS will prevent errors if tables already exist)
                conn.executescript(migration_sql)
                logger.info("Module tables created successfully")
            else:
                logger.warning(f"Module migration file not found at expected paths: {migration_paths}")
                
        except Exception as e:
            logger.warning(f"Module table creation warning (non-critical): {e}")
    
    def create_tables(self, conn: sqlite3.Connection):
        """Create all database tables"""
        
        # Create roles table
        conn.execute("""
            CREATE TABLE roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                permissions TEXT NOT NULL,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create users table
        conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role_id INTEGER NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                FOREIGN KEY (role_id) REFERENCES roles(id)
            )
        """)
        
        # Create user_sessions table
        conn.execute("""
            CREATE TABLE user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for performance
        conn.execute("CREATE INDEX idx_users_email ON users(email)")
        conn.execute("CREATE INDEX idx_users_username ON users(username)")
        conn.execute("CREATE INDEX idx_sessions_token ON user_sessions(token_hash)")
        conn.execute("CREATE INDEX idx_sessions_user ON user_sessions(user_id)")
        conn.execute("CREATE INDEX idx_sessions_expires ON user_sessions(expires_at)")
        
        # Create APRAG tables (for new databases)
        self.create_aprag_tables(conn)
        
        # Create Topic tables (for new databases)
        self.create_topic_tables(conn)
        
        # Create Module tables (for new databases)
        self.create_module_tables(conn)
        
        conn.commit()
        logger.info("Database tables created successfully")
    
    def insert_default_data(self, conn: sqlite3.Connection):
        """Insert default roles and admin user"""
        
        # Default role permissions
        admin_permissions = {
            "users": ["create", "read", "update", "delete"],
            "roles": ["create", "read", "update", "delete"],
            "sessions": ["create", "read", "update", "delete"],
            "documents": ["create", "read", "update", "delete"],
            "system": ["admin", "configure"]
        }
        
        teacher_permissions = {
            "users": ["read"],
            "sessions": ["create", "read", "update", "delete"],
            "documents": ["create", "read", "update", "delete"],
            "students": ["read"]
        }
        
        student_permissions = {
            "sessions": ["read"],
            "documents": ["read"]
        }
        
        # Insert default roles
        roles_data = [
            ("admin", "System Administrator", json.dumps(admin_permissions)),
            ("teacher", "Teacher", json.dumps(teacher_permissions)),
            ("student", "Student", json.dumps(student_permissions))
        ]
        
        conn.executemany("""
            INSERT INTO roles (name, description, permissions)
            VALUES (?, ?, ?)
        """, roles_data)
        
        # Create default admin user
        admin_password = "admin123"  # Should be changed on first login
        password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Get admin role ID
        admin_role = conn.execute("SELECT id FROM roles WHERE name = 'admin'").fetchone()
        
        conn.execute("""
            INSERT INTO users (username, email, password_hash, role_id, first_name, last_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("admin", "admin@rag-assistant.local", password_hash, admin_role['id'], "System", "Administrator"))
        
        conn.commit()
        logger.info("Default data inserted successfully")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def hash_token(self, token: str) -> str:
        """Hash session token using SHA-256"""
        return hashlib.sha256(token.encode('utf-8')).hexdigest()
    
    def execute_migration(self, migration_file: str):
        """Execute a migration file"""
        try:
            with open(migration_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            with self.get_connection() as conn:
                # Split and execute each statement
                statements = sql_content.split(';')
                for statement in statements:
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)
                conn.commit()
                
            logger.info(f"Migration {migration_file} executed successfully")
            
        except Exception as e:
            logger.error(f"Migration {migration_file} failed: {e}")
            raise
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from database"""
        try:
            with self.get_connection() as conn:
                result = conn.execute("""
                    DELETE FROM user_sessions 
                    WHERE expires_at < CURRENT_TIMESTAMP
                """)
                conn.commit()
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired sessions")
                    
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT u.*, r.name as role_name, r.permissions
                    FROM users u
                    JOIN roles r ON u.role_id = r.id
                    WHERE u.username = ? AND u.is_active = 1
                """, (username,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Failed to get user {username}: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT u.*, r.name as role_name, r.permissions
                    FROM users u
                    JOIN roles r ON u.role_id = r.id
                    WHERE u.email = ? AND u.is_active = 1
                """, (email,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Failed to get user {email}: {e}")
            return None
    
    def create_user_session(self, user_id: int, token_hash: str, 
                           expires_at: datetime, ip_address: str = None, 
                           user_agent: str = None) -> Optional[int]:
        """Create a new user session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO user_sessions (user_id, token_hash, expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, token_hash, expires_at, ip_address, user_agent))
                
                conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            return None
    
    def get_session_by_token(self, token_hash: str) -> Optional[Dict[str, Any]]:
        """Get session by token hash"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT s.*, u.username, u.email, u.role_id, r.name as role_name, r.permissions
                    FROM user_sessions s
                    JOIN users u ON s.user_id = u.id
                    JOIN roles r ON u.role_id = r.id
                    WHERE s.token_hash = ? AND s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
                """, (token_hash,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def delete_session(self, token_hash: str) -> bool:
        """Delete a session"""
        try:
            with self.get_connection() as conn:
                result = conn.execute("""
                    DELETE FROM user_sessions WHERE token_hash = ?
                """, (token_hash,))
                conn.commit()
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def update_user_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (user_id,))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update last login for user {user_id}: {e}")


# Global database manager instance
db_manager = None

def get_db_manager(db_path: str = "data/rag_assistant.db") -> DatabaseManager:
    """Get or create global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(db_path)
    return db_manager