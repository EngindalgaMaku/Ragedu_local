"""
Database Integration Tests
Tests database functionality, migrations, and data integrity
"""

import pytest
import os
import sys
import sqlite3
import json
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database import DatabaseManager, User, Role, UserSession


class TestDatabaseIntegration:
    """Test database integration and migrations"""
    
    @pytest.fixture(scope="function")
    def setup_test_db(self):
        """Setup clean test database for each test"""
        self.db_path = "test_database_integration.db"
        
        # Clean up any existing test database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        # Initialize database manager
        self.db_manager = DatabaseManager(self.db_path)
        self.user_model = User(self.db_manager)
        self.role_model = Role(self.db_manager)
        self.session_model = UserSession(self.db_manager)
        
        yield
        
        # Cleanup
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def test_database_initialization(self, setup_test_db):
        """Test database is properly initialized with correct schema"""
        # Check that all required tables exist
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['users', 'roles', 'user_sessions']
        for table in required_tables:
            assert table in tables, f"Required table {table} not found"
    
    def test_database_schema_validation(self, setup_test_db):
        """Test database schema has correct structure"""
        with self.db_manager.get_connection() as conn:
            # Test users table schema
            cursor = conn.execute("PRAGMA table_info(users)")
            users_columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            required_user_columns = {
                'id': 'INTEGER',
                'username': 'VARCHAR',
                'email': 'VARCHAR',
                'password_hash': 'VARCHAR',
                'first_name': 'VARCHAR',
                'last_name': 'VARCHAR',
                'is_active': 'BOOLEAN',
                'role_id': 'INTEGER',
                'created_at': 'DATETIME',
                'updated_at': 'DATETIME',
                'last_login': 'DATETIME'
            }
            
            for col_name, col_type in required_user_columns.items():
                assert col_name in users_columns, f"Missing column {col_name} in users table"
            
            # Test roles table schema
            cursor = conn.execute("PRAGMA table_info(roles)")
            roles_columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            required_role_columns = {
                'id': 'INTEGER',
                'name': 'VARCHAR',
                'description': 'TEXT',
                'permissions': 'TEXT',
                'created_at': 'DATETIME',
                'updated_at': 'DATETIME'
            }
            
            for col_name, col_type in required_role_columns.items():
                assert col_name in roles_columns, f"Missing column {col_name} in roles table"
            
            # Test user_sessions table schema
            cursor = conn.execute("PRAGMA table_info(user_sessions)")
            sessions_columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            required_session_columns = {
                'id': 'INTEGER',
                'user_id': 'INTEGER',
                'token_hash': 'VARCHAR',
                'expires_at': 'DATETIME',
                'created_at': 'DATETIME',
                'ip_address': 'VARCHAR',
                'user_agent': 'TEXT'
            }
            
            for col_name, col_type in required_session_columns.items():
                assert col_name in sessions_columns, f"Missing column {col_name} in user_sessions table"
    
    def test_role_creation_and_permissions(self, setup_test_db):
        """Test role creation and permission management"""
        # Create default roles
        self.role_model.create_default_roles()
        
        # Verify default roles exist
        admin_role = self.role_model.get_role_by_name("admin")
        teacher_role = self.role_model.get_role_by_name("teacher")
        student_role = self.role_model.get_role_by_name("student")
        
        assert admin_role is not None, "Admin role not created"
        assert teacher_role is not None, "Teacher role not created"
        assert student_role is not None, "Student role not created"
        
        # Test admin permissions
        admin_permissions = admin_role['permissions']
        assert 'users' in admin_permissions
        assert 'create' in admin_permissions['users']
        assert 'read' in admin_permissions['users']
        assert 'update' in admin_permissions['users']
        assert 'delete' in admin_permissions['users']
        
        # Test teacher permissions
        teacher_permissions = teacher_role['permissions']
        assert 'sessions' in teacher_permissions
        assert 'documents' in teacher_permissions
        assert 'create' in teacher_permissions['sessions']
        
        # Test student permissions (read-only)
        student_permissions = student_role['permissions']
        assert 'sessions' in student_permissions
        assert 'read' in student_permissions['sessions']
        assert 'create' not in student_permissions.get('users', [])
    
    def test_user_management_operations(self, setup_test_db):
        """Test complete user management CRUD operations"""
        # Create default roles first
        self.role_model.create_default_roles()
        admin_role = self.role_model.get_role_by_name("admin")
        
        # Test user creation
        user_id = self.user_model.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role_id=admin_role['id'],
            first_name="Test",
            last_name="User"
        )
        assert user_id is not None, "Failed to create user"
        
        # Test user retrieval
        user = self.user_model.get_user_by_id(user_id)
        assert user is not None
        assert user['username'] == 'testuser'
        assert user['email'] == 'test@example.com'
        assert user['is_active'] is True
        
        # Test user authentication
        auth_user = self.user_model.authenticate_user("testuser", "TestPassword123!")
        assert auth_user is not None
        assert auth_user['username'] == 'testuser'
        
        # Test wrong password
        wrong_auth = self.user_model.authenticate_user("testuser", "wrongpassword")
        assert wrong_auth is None
        
        # Test user update
        success = self.user_model.update_user(user_id, first_name="Updated")
        assert success is True
        
        updated_user = self.user_model.get_user_by_id(user_id)
        assert updated_user['first_name'] == 'Updated'
        
        # Test password change
        success = self.user_model.change_password(user_id, "TestPassword123!", "NewPassword123!")
        assert success is True
        
        # Verify new password works
        new_auth = self.user_model.authenticate_user("testuser", "NewPassword123!")
        assert new_auth is not None
        
        # Test user deactivation
        success = self.user_model.deactivate_user(user_id)
        assert success is True
        
        inactive_user = self.user_model.get_user_by_id(user_id)
        assert inactive_user['is_active'] is False
        
        # Test inactive user cannot authenticate
        inactive_auth = self.user_model.authenticate_user("testuser", "NewPassword123!")
        assert inactive_auth is None
    
    def test_session_management(self, setup_test_db):
        """Test session management functionality"""
        # Create test user
        self.role_model.create_default_roles()
        admin_role = self.role_model.get_role_by_name("admin")
        
        user_id = self.user_model.create_user(
            username="sessionuser",
            email="session@example.com",
            password="SessionTest123!",
            role_id=admin_role['id'],
            first_name="Session",
            last_name="User"
        )
        
        # Test session creation
        session_id = self.session_model.create_session(
            user_id=user_id,
            token="test_session_token",
            expires_in_hours=24,
            ip_address="127.0.0.1",
            user_agent="pytest-test-agent"
        )
        assert session_id is not None
        
        # Test session retrieval
        session = self.session_model.get_session_by_id(session_id)
        assert session is not None
        assert session['user_id'] == user_id
        assert session['ip_address'] == "127.0.0.1"
        
        # Test session validation
        valid_session = self.session_model.validate_session("test_session_token")
        assert valid_session is not None
        assert valid_session['user_id'] == user_id
        
        # Test session by token
        token_session = self.session_model.get_session_by_token("test_session_token")
        assert token_session is not None
        
        # Test multiple sessions for user
        session_id_2 = self.session_model.create_session(
            user_id=user_id,
            token="test_session_token_2",
            expires_in_hours=24
        )
        
        user_sessions = self.session_model.get_user_sessions(user_id)
        assert len(user_sessions) == 2
        
        # Test session deletion
        success = self.session_model.delete_session_by_token("test_session_token")
        assert success is True
        
        # Verify session is gone
        deleted_session = self.session_model.get_session_by_token("test_session_token")
        assert deleted_session is None
        
        # Test delete all user sessions
        deleted_count = self.session_model.delete_user_sessions(user_id)
        assert deleted_count == 1  # One remaining session
        
        user_sessions_after = self.session_model.get_user_sessions(user_id)
        assert len(user_sessions_after) == 0
    
    def test_data_integrity_constraints(self, setup_test_db):
        """Test database integrity constraints"""
        # Create default roles
        self.role_model.create_default_roles()
        admin_role = self.role_model.get_role_by_name("admin")
        
        # Test unique username constraint
        user1_id = self.user_model.create_user(
            username="uniquetest",
            email="unique1@example.com",
            password="Password123!",
            role_id=admin_role['id'],
            first_name="User1",
            last_name="Test"
        )
        assert user1_id is not None
        
        # Try to create user with same username
        user2_id = self.user_model.create_user(
            username="uniquetest",  # Same username
            email="unique2@example.com",  # Different email
            password="Password123!",
            role_id=admin_role['id'],
            first_name="User2",
            last_name="Test"
        )
        assert user2_id is None  # Should fail due to unique constraint
        
        # Test unique email constraint
        user3_id = self.user_model.create_user(
            username="uniquetest2",
            email="unique1@example.com",  # Same email as user1
            password="Password123!",
            role_id=admin_role['id'],
            first_name="User3",
            last_name="Test"
        )
        assert user3_id is None  # Should fail due to unique email constraint
        
        # Test foreign key constraint (role_id)
        invalid_user_id = self.user_model.create_user(
            username="invalidrole",
            email="invalid@example.com",
            password="Password123!",
            role_id=999,  # Non-existent role
            first_name="Invalid",
            last_name="User"
        )
        assert invalid_user_id is None  # Should fail due to foreign key constraint
    
    def test_database_migrations(self, setup_test_db):
        """Test database migration functionality"""
        # Verify migration files exist
        migrations_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'database', 'migrations')
        assert os.path.exists(migrations_dir), "Migrations directory not found"
        
        migration_files = os.listdir(migrations_dir)
        expected_migrations = [
            '001_create_tables.sql',
            '002_insert_default_data.sql'
        ]
        
        for migration_file in expected_migrations:
            assert migration_file in migration_files, f"Migration file {migration_file} not found"
        
        # Test that migrations create proper schema
        # This is implicitly tested by the database initialization working correctly
    
    def test_performance_and_indexing(self, setup_test_db):
        """Test database performance with indexes"""
        # Create default roles and test data
        self.role_model.create_default_roles()
        admin_role = self.role_model.get_role_by_name("admin")
        
        # Create multiple users for performance testing
        user_ids = []
        for i in range(100):
            user_id = self.user_model.create_user(
                username=f"perfuser{i}",
                email=f"perf{i}@example.com",
                password="PerfTest123!",
                role_id=admin_role['id'],
                first_name=f"Perf{i}",
                last_name="User"
            )
            user_ids.append(user_id)
        
        # Test query performance (should be fast with indexes)
        import time
        
        start_time = time.time()
        for i in range(10):
            user = self.user_model.get_user_by_username(f"perfuser{i}")
            assert user is not None
        end_time = time.time()
        
        query_time = end_time - start_time
        assert query_time < 1.0, f"Username queries too slow: {query_time}s"
        
        # Test user count functionality
        user_count = self.user_model.get_user_count()
        assert user_count == 100
        
        active_count = self.user_model.get_user_count(is_active=True)
        assert active_count == 100
    
    def test_database_backup_restore(self, setup_test_db):
        """Test database backup and restore functionality"""
        # Create test data
        self.role_model.create_default_roles()
        admin_role = self.role_model.get_role_by_name("admin")
        
        user_id = self.user_model.create_user(
            username="backuptest",
            email="backup@example.com",
            password="BackupTest123!",
            role_id=admin_role['id'],
            first_name="Backup",
            last_name="User"
        )
        
        # Create backup (simple file copy for SQLite)
        backup_path = "test_backup.db"
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        # Modify original database
        self.user_model.update_user(user_id, first_name="Modified")
        
        # Verify modification
        modified_user = self.user_model.get_user_by_id(user_id)
        assert modified_user['first_name'] == "Modified"
        
        # Restore from backup
        shutil.copy2(backup_path, self.db_path)
        
        # Re-initialize database connection after restore
        self.db_manager = DatabaseManager(self.db_path)
        self.user_model = User(self.db_manager)
        
        # Verify restoration
        restored_user = self.user_model.get_user_by_id(user_id)
        assert restored_user['first_name'] == "Backup"  # Should be original value
        
        # Cleanup
        if os.path.exists(backup_path):
            os.remove(backup_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])