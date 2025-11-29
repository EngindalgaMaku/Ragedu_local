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
        """Initialize database with tables if they don't exist.

        This keeps existing auth/session tables and also ensures that
        markdown category tables are present for document management.
        """
        try:
            with self.get_connection() as conn:
                # Check if core auth tables exist
                cursor = conn.execute(
                    """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='users'
                    """
                )

                if not cursor.fetchone():
                    logger.info("Creating database tables...")
                    self.create_tables(conn)
                    self.insert_default_data(conn)
                    logger.info("Database initialized successfully")
                else:
                    logger.info("Database tables already exist")

                # Always ensure markdown helper tables exist (idempotent)
                self.ensure_markdown_tables(conn)

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def create_tables(self, conn: sqlite3.Connection):
        """Create core auth/session tables.

        Markdown-related tables are managed separately by ensure_markdown_tables
        so they can be added safely to existing databases.
        """

        # Create roles table
        conn.execute(
            """
            CREATE TABLE roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                permissions TEXT NOT NULL,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Create users table
        conn.execute(
            """
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
            """
        )

        # Create user_sessions table
        conn.execute(
            """
            CREATE TABLE user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # Create indexes for performance
        conn.execute("CREATE INDEX idx_users_email ON users(email)")
        conn.execute("CREATE INDEX idx_users_username ON users(username)")
        conn.execute("CREATE INDEX idx_sessions_token ON user_sessions(token_hash)")
        conn.execute("CREATE INDEX idx_sessions_user ON user_sessions(user_id)")
        conn.execute("CREATE INDEX idx_sessions_expires ON user_sessions(expires_at)")

        # Also create markdown helper tables for a fully initialized DB
        self.ensure_markdown_tables(conn)

        conn.commit()
        logger.info("Database tables created successfully")

    def ensure_markdown_tables(self, conn: sqlite3.Connection):
        """Ensure markdown category tables exist (idempotent).

        - markdown_categories: stores category definitions
        - markdown_file_categories: maps markdown filenames to a single category
        """
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS markdown_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS markdown_file_categories (
                    filename TEXT PRIMARY KEY,
                    category_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (category_id) REFERENCES markdown_categories(id) ON DELETE SET NULL
                )
                """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_markdown_file_categories_category ON markdown_file_categories(category_id)"
            )

            conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure markdown tables: {e}")
            raise
    
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