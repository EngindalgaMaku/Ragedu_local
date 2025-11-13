#!/usr/bin/env python3
"""
Database Test Script for RAG Education Assistant
Tests database schema, models, and functionality
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.database import DatabaseManager
from src.database.models.user import User
from src.database.models.role import Role
from src.database.models.session import UserSession


class DatabaseTester:
    """
    Comprehensive database testing class
    """
    
    def __init__(self, use_temp_db=True):
        """Initialize tester with temporary or persistent database"""
        if use_temp_db:
            # Use temporary database for testing
            self.temp_dir = tempfile.mkdtemp()
            self.db_path = os.path.join(self.temp_dir, 'test_rag_assistant.db')
        else:
            # Use actual database path
            self.db_path = 'data/rag_assistant.db'
        
        self.db_manager = None
        self.user_model = None
        self.role_model = None
        self.session_model = None
        
        self.test_results = []
        
    def log_test(self, test_name, success, message=""):
        """Log test result"""
        status = "✓ PASS" if success else "✗ FAIL"
        result = f"{status} {test_name}"
        if message:
            result += f": {message}"
        
        self.test_results.append((test_name, success, message))
        print(result)
        return success
    
    def setup(self):
        """Initialize database and models"""
        try:
            self.db_manager = DatabaseManager(self.db_path)
            self.user_model = User(self.db_manager)
            self.role_model = Role(self.db_manager)
            self.session_model = UserSession(self.db_manager)
            
            return self.log_test("Database Setup", True, f"DB: {self.db_path}")
        except Exception as e:
            return self.log_test("Database Setup", False, str(e))
    
    def test_database_connection(self):
        """Test basic database connection"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT 1")
                result = cursor.fetchone()
                success = result[0] == 1
            
            return self.log_test("Database Connection", success)
        except Exception as e:
            return self.log_test("Database Connection", False, str(e))
    
    def test_table_creation(self):
        """Test if all required tables exist"""
        required_tables = ['users', 'roles', 'user_sessions']
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                
                existing_tables = [row[0] for row in cursor.fetchall()]
                missing_tables = [t for t in required_tables if t not in existing_tables]
                
                success = len(missing_tables) == 0
                message = f"Found: {existing_tables}"
                if missing_tables:
                    message += f", Missing: {missing_tables}"
                
            return self.log_test("Table Creation", success, message)
        except Exception as e:
            return self.log_test("Table Creation", False, str(e))
    
    def test_indexes_creation(self):
        """Test if indexes are created properly"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND name NOT LIKE 'sqlite_%'
                """)
                
                indexes = [row[0] for row in cursor.fetchall()]
                expected_indexes = [
                    'idx_users_email', 'idx_users_username', 'idx_sessions_token'
                ]
                
                missing_indexes = [idx for idx in expected_indexes if idx not in indexes]
                success = len(missing_indexes) == 0
                message = f"Found {len(indexes)} indexes"
                
            return self.log_test("Index Creation", success, message)
        except Exception as e:
            return self.log_test("Index Creation", False, str(e))
    
    def test_role_model(self):
        """Test role model functionality"""
        try:
            # Test default roles exist
            admin_role = self.role_model.get_role_by_name('admin')
            teacher_role = self.role_model.get_role_by_name('teacher')
            student_role = self.role_model.get_role_by_name('student')
            
            if not all([admin_role, teacher_role, student_role]):
                return self.log_test("Role Model - Default Roles", False, "Missing default roles")
            
            # Test role permissions
            admin_perms = admin_role['permissions']
            has_admin_perms = 'users' in admin_perms and 'create' in admin_perms['users']
            
            if not has_admin_perms:
                return self.log_test("Role Model - Permissions", False, "Admin permissions invalid")
            
            # Test create custom role
            test_role_id = self.role_model.create_role(
                name='test_role',
                description='Test role for validation',
                permissions={'test': ['read']}
            )
            
            if not test_role_id:
                return self.log_test("Role Model - Create Role", False, "Failed to create test role")
            
            # Test role retrieval
            test_role = self.role_model.get_role_by_id(test_role_id)
            if not test_role or test_role['name'] != 'test_role':
                return self.log_test("Role Model - Retrieve Role", False, "Failed to retrieve test role")
            
            # Test permission check
            has_test_perm = self.role_model.has_permission(test_role_id, 'test', 'read')
            if not has_test_perm:
                return self.log_test("Role Model - Permission Check", False, "Permission check failed")
            
            return self.log_test("Role Model", True, "All role operations successful")
            
        except Exception as e:
            return self.log_test("Role Model", False, str(e))
    
    def test_user_model(self):
        """Test user model functionality"""
        try:
            # Get admin role for test user
            admin_role = self.role_model.get_role_by_name('admin')
            if not admin_role:
                return self.log_test("User Model - Setup", False, "Admin role not found")
            
            # Test create user
            test_user_id = self.user_model.create_user(
                username='testuser',
                email='test@example.com',
                password='testpassword123',
                role_id=admin_role['id'],
                first_name='Test',
                last_name='User'
            )
            
            if not test_user_id:
                return self.log_test("User Model - Create User", False, "Failed to create test user")
            
            # Test user retrieval
            test_user = self.user_model.get_user_by_id(test_user_id)
            if not test_user or test_user['username'] != 'testuser':
                return self.log_test("User Model - Retrieve User", False, "Failed to retrieve test user")
            
            # Test authentication
            auth_result = self.user_model.authenticate_user('testuser', 'testpassword123')
            if not auth_result:
                return self.log_test("User Model - Authentication", False, "Authentication failed")
            
            # Test wrong password
            auth_fail = self.user_model.authenticate_user('testuser', 'wrongpassword')
            if auth_fail:
                return self.log_test("User Model - Auth Security", False, "Wrong password accepted")
            
            # Test password change
            password_changed = self.user_model.change_password(
                test_user_id, 'testpassword123', 'newpassword123'
            )
            if not password_changed:
                return self.log_test("User Model - Password Change", False, "Password change failed")
            
            # Test authentication with new password
            auth_new = self.user_model.authenticate_user('testuser', 'newpassword123')
            if not auth_new:
                return self.log_test("User Model - New Password Auth", False, "New password auth failed")
            
            return self.log_test("User Model", True, "All user operations successful")
            
        except Exception as e:
            return self.log_test("User Model", False, str(e))
    
    def test_session_model(self):
        """Test session model functionality"""
        try:
            # Get test user
            test_user = self.user_model.get_user_by_username('testuser')
            if not test_user:
                return self.log_test("Session Model - Setup", False, "Test user not found")
            
            # Generate test token
            test_token = self.session_model.generate_session_token()
            if len(test_token) < 20:
                return self.log_test("Session Model - Token Generation", False, "Token too short")
            
            # Create session
            session_id = self.session_model.create_session(
                user_id=test_user['id'],
                token=test_token,
                expires_in_hours=24,
                ip_address='127.0.0.1',
                user_agent='TestAgent/1.0'
            )
            
            if not session_id:
                return self.log_test("Session Model - Create Session", False, "Failed to create session")
            
            # Test session retrieval
            session = self.session_model.get_session_by_token(test_token)
            if not session or session['user_id'] != test_user['id']:
                return self.log_test("Session Model - Retrieve Session", False, "Failed to retrieve session")
            
            # Test session validation
            valid_session = self.session_model.validate_session(test_token)
            if not valid_session:
                return self.log_test("Session Model - Validate Session", False, "Session validation failed")
            
            # Test session refresh
            refresh_success = self.session_model.refresh_session(test_token, 48)
            if not refresh_success:
                return self.log_test("Session Model - Refresh Session", False, "Session refresh failed")
            
            # Test session deletion
            delete_success = self.session_model.delete_session_by_token(test_token)
            if not delete_success:
                return self.log_test("Session Model - Delete Session", False, "Session deletion failed")
            
            # Test expired session retrieval
            expired_session = self.session_model.get_session_by_token(test_token)
            if expired_session:
                return self.log_test("Session Model - Expired Cleanup", False, "Deleted session still found")
            
            return self.log_test("Session Model", True, "All session operations successful")
            
        except Exception as e:
            return self.log_test("Session Model", False, str(e))
    
    def test_foreign_key_constraints(self):
        """Test foreign key constraints"""
        try:
            with self.db_manager.get_connection() as conn:
                # Test invalid role_id in users table
                try:
                    conn.execute("""
                        INSERT INTO users (username, email, password_hash, role_id, first_name, last_name)
                        VALUES ('invalid', 'invalid@test.com', 'hash', 9999, 'Invalid', 'User')
                    """)
                    conn.commit()
                    return self.log_test("Foreign Key Constraints", False, "Invalid role_id accepted")
                except Exception:
                    # This should fail due to foreign key constraint
                    pass
                
                # Test cascade delete
                # Create test user and session
                admin_role = self.role_model.get_role_by_name('admin')
                test_user_id = self.user_model.create_user(
                    username='cascade_test',
                    email='cascade@test.com',
                    password='password123',
                    role_id=admin_role['id'],
                    first_name='Cascade',
                    last_name='Test'
                )
                
                test_token = self.session_model.generate_session_token()
                session_id = self.session_model.create_session(
                    user_id=test_user_id,
                    token=test_token,
                    expires_in_hours=1
                )
                
                # Delete user - should cascade delete session
                self.user_model.delete_user(test_user_id)
                
                # Check if session was deleted
                orphaned_session = self.session_model.get_session_by_id(session_id)
                if orphaned_session:
                    return self.log_test("Foreign Key Constraints", False, "Session not cascade deleted")
            
            return self.log_test("Foreign Key Constraints", True, "FK constraints working properly")
            
        except Exception as e:
            return self.log_test("Foreign Key Constraints", False, str(e))
    
    def test_performance(self):
        """Test basic performance with bulk operations"""
        try:
            import time
            
            admin_role = self.role_model.get_role_by_name('admin')
            
            # Test bulk user creation
            start_time = time.time()
            user_ids = []
            
            for i in range(10):
                user_id = self.user_model.create_user(
                    username=f'perftest{i}',
                    email=f'perftest{i}@test.com',
                    password='password123',
                    role_id=admin_role['id'],
                    first_name='Perf',
                    last_name=f'Test{i}'
                )
                if user_id:
                    user_ids.append(user_id)
            
            creation_time = time.time() - start_time
            
            # Test bulk authentication
            start_time = time.time()
            auth_success_count = 0
            
            for i in range(10):
                auth_result = self.user_model.authenticate_user(f'perftest{i}', 'password123')
                if auth_result:
                    auth_success_count += 1
            
            auth_time = time.time() - start_time
            
            # Clean up test users
            for user_id in user_ids:
                self.user_model.delete_user(user_id)
            
            success = (len(user_ids) == 10 and auth_success_count == 10 and 
                      creation_time < 5.0 and auth_time < 2.0)
            
            message = f"Created {len(user_ids)} users in {creation_time:.2f}s, {auth_success_count} auths in {auth_time:.2f}s"
            
            return self.log_test("Performance", success, message)
            
        except Exception as e:
            return self.log_test("Performance", False, str(e))
    
    def run_all_tests(self):
        """Run all database tests"""
        print("=" * 80)
        print("RAG Education Assistant - Database Test Suite")
        print("=" * 80)
        print(f"Database Path: {self.db_path}")
        print("-" * 80)
        
        # Run tests in order
        tests = [
            self.setup,
            self.test_database_connection,
            self.test_table_creation,
            self.test_indexes_creation,
            self.test_role_model,
            self.test_user_model,
            self.test_session_model,
            self.test_foreign_key_constraints,
            self.test_performance
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                result = test()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"✗ FAIL {test.__name__}: Unexpected error - {e}")
                failed += 1
        
        # Print summary
        print("-" * 80)
        print(f"Test Summary: {passed} passed, {failed} failed")
        print("=" * 80)
        
        if failed == 0:
            print("🎉 All tests passed! Database system is working correctly.")
            return True
        else:
            print(f"⚠️  {failed} test(s) failed. Please review the issues above.")
            return False
    
    def cleanup(self):
        """Clean up test resources"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RAG Education Assistant Database')
    parser.add_argument('--use-real-db', action='store_true', 
                       help='Use real database instead of temporary one')
    
    args = parser.parse_args()
    
    tester = DatabaseTester(use_temp_db=not args.use_real_db)
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        return 1
    finally:
        tester.cleanup()


if __name__ == "__main__":
    sys.exit(main())