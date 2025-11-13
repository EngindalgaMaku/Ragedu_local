"""
Integration Validation Script for RAG Education Assistant Auth Service
Validates integration with existing database models and system components
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import auth service components
from auth.auth_manager import AuthManager
from auth.schemas import LoginRequest, UserCreate
from src.database import DatabaseManager, User, Role, UserSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationValidator:
    """
    Comprehensive integration validator for auth service
    """
    
    def __init__(self, db_path: str = "data/rag_assistant.db"):
        """
        Initialize the validator
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path
        self.auth_manager = None
        self.test_results = {}
        
    async def run_all_validations(self):
        """Run all validation tests"""
        logger.info("Starting comprehensive integration validation...")
        
        validations = [
            self.validate_database_connection,
            self.validate_existing_models_compatibility,
            self.validate_auth_manager_initialization,
            self.validate_user_operations,
            self.validate_role_operations,
            self.validate_session_operations,
            self.validate_authentication_flow,
            self.validate_permission_system,
            self.validate_security_features,
            self.validate_api_integration
        ]
        
        for validation in validations:
            try:
                logger.info(f"Running {validation.__name__}...")
                await validation()
                self.test_results[validation.__name__] = "PASSED"
                logger.info(f"✅ {validation.__name__} PASSED")
            except Exception as e:
                logger.error(f"❌ {validation.__name__} FAILED: {e}")
                self.test_results[validation.__name__] = f"FAILED: {e}"
        
        self.print_summary()
    
    async def validate_database_connection(self):
        """Validate database connection and structure"""
        try:
            # Test direct database manager connection
            db_manager = DatabaseManager(self.db_path)
            
            # Test connection
            with db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row['name'] for row in cursor.fetchall()]
            
            # Verify required tables exist
            required_tables = ['users', 'roles', 'user_sessions']
            for table in required_tables:
                if table not in tables:
                    raise Exception(f"Required table '{table}' not found")
            
            logger.info(f"Database tables found: {tables}")
            
        except Exception as e:
            raise Exception(f"Database connection validation failed: {e}")
    
    async def validate_existing_models_compatibility(self):
        """Validate compatibility with existing database models"""
        try:
            db_manager = DatabaseManager(self.db_path)
            
            # Test User model
            user_model = User(db_manager)
            user_count = user_model.get_user_count()
            logger.info(f"Current user count: {user_count}")
            
            # Test Role model
            role_model = Role(db_manager)
            role_count = role_model.get_role_count()
            logger.info(f"Current role count: {role_count}")
            
            # Test UserSession model
            session_model = UserSession(db_manager)
            session_count = session_model.get_active_sessions_count()
            logger.info(f"Current active session count: {session_count}")
            
            # Test that default roles exist
            default_roles = ['admin', 'teacher', 'student']
            for role_name in default_roles:
                role = role_model.get_role_by_name(role_name)
                if not role:
                    raise Exception(f"Default role '{role_name}' not found")
            
        except Exception as e:
            raise Exception(f"Model compatibility validation failed: {e}")
    
    async def validate_auth_manager_initialization(self):
        """Validate AuthManager initialization"""
        try:
            self.auth_manager = AuthManager(
                secret_key="test-key-for-validation",
                db_path=self.db_path,
                access_token_expire_minutes=30,
                refresh_token_expire_days=7
            )
            
            # Verify components are initialized
            if not self.auth_manager.user_model:
                raise Exception("User model not initialized")
            if not self.auth_manager.role_model:
                raise Exception("Role model not initialized")
            if not self.auth_manager.session_model:
                raise Exception("Session model not initialized")
            
            # Test password hashing
            test_password = "TestPassword123!"
            hashed = self.auth_manager.user_model.db.hash_password(test_password)
            if not self.auth_manager.user_model.db.verify_password(test_password, hashed):
                raise Exception("Password hashing/verification failed")
            
        except Exception as e:
            raise Exception(f"AuthManager initialization failed: {e}")
    
    async def validate_user_operations(self):
        """Validate user CRUD operations"""
        if not self.auth_manager:
            raise Exception("AuthManager not initialized")
        
        try:
            # Test user creation
            test_username = f"test_user_validation_{os.getpid()}"
            test_email = f"validation_{os.getpid()}@example.com"
            
            # Get a role ID (use student role)
            student_role = self.auth_manager.role_model.get_role_by_name("student")
            if not student_role:
                raise Exception("Student role not found")
            
            user_id = self.auth_manager.user_model.create_user(
                username=test_username,
                email=test_email,
                password="ValidationTest123!",
                role_id=student_role['id'],
                first_name="Test",
                last_name="Validation",
                is_active=True
            )
            
            if not user_id:
                raise Exception("User creation failed")
            
            # Test user retrieval
            user = self.auth_manager.user_model.get_user_by_id(user_id)
            if not user or user['username'] != test_username:
                raise Exception("User retrieval failed")
            
            # Test user update
            success = self.auth_manager.user_model.update_user(
                user_id=user_id,
                first_name="Updated"
            )
            if not success:
                raise Exception("User update failed")
            
            # Test user authentication
            authenticated_user = self.auth_manager.user_model.authenticate_user(
                username=test_username,
                password="ValidationTest123!"
            )
            if not authenticated_user:
                raise Exception("User authentication failed")
            
            # Cleanup - delete test user
            self.auth_manager.user_model.delete_user(user_id)
            logger.info(f"Successfully validated user operations for user ID {user_id}")
            
        except Exception as e:
            raise Exception(f"User operations validation failed: {e}")
    
    async def validate_role_operations(self):
        """Validate role management operations"""
        if not self.auth_manager:
            raise Exception("AuthManager not initialized")
        
        try:
            # Test role creation
            test_role_name = f"test_role_validation_{os.getpid()}"
            test_permissions = {
                "test": ["read", "write"],
                "validation": ["execute"]
            }
            
            role_id = self.auth_manager.role_model.create_role(
                name=test_role_name,
                description="Test role for validation",
                permissions=test_permissions
            )
            
            if not role_id:
                raise Exception("Role creation failed")
            
            # Test role retrieval
            role = self.auth_manager.role_model.get_role_by_id(role_id)
            if not role or role['name'] != test_role_name:
                raise Exception("Role retrieval failed")
            
            # Test permission checking
            has_permission = self.auth_manager.role_model.has_permission(
                role_id, "test", "read"
            )
            if not has_permission:
                raise Exception("Permission checking failed")
            
            # Test permission addition
            success = self.auth_manager.role_model.add_permission(
                role_id, "new_resource", "new_action"
            )
            if not success:
                raise Exception("Permission addition failed")
            
            # Cleanup - delete test role
            self.auth_manager.role_model.delete_role(role_id)
            logger.info(f"Successfully validated role operations for role ID {role_id}")
            
        except Exception as e:
            raise Exception(f"Role operations validation failed: {e}")
    
    async def validate_session_operations(self):
        """Validate session management operations"""
        if not self.auth_manager:
            raise Exception("AuthManager not initialized")
        
        try:
            # Get admin user for session testing
            admin_user = self.auth_manager.user_model.get_user_by_username("admin")
            if not admin_user:
                raise Exception("Admin user not found")
            
            # Create a test session
            test_token = "test_session_token_validation"
            session_id = self.auth_manager.session_model.create_session(
                user_id=admin_user['id'],
                token=test_token,
                expires_in_hours=1,
                ip_address="127.0.0.1",
                user_agent="ValidationScript/1.0"
            )
            
            if not session_id:
                raise Exception("Session creation failed")
            
            # Test session validation
            session = self.auth_manager.session_model.validate_session(test_token)
            if not session:
                raise Exception("Session validation failed")
            
            # Test session refresh
            success = self.auth_manager.session_model.refresh_session(test_token, 2)
            if not success:
                raise Exception("Session refresh failed")
            
            # Test session cleanup
            deleted = self.auth_manager.session_model.delete_session_by_token(test_token)
            if not deleted:
                raise Exception("Session deletion failed")
            
            logger.info(f"Successfully validated session operations for session ID {session_id}")
            
        except Exception as e:
            raise Exception(f"Session operations validation failed: {e}")
    
    async def validate_authentication_flow(self):
        """Validate complete authentication flow"""
        if not self.auth_manager:
            raise Exception("AuthManager not initialized")
        
        try:
            # Test authentication with admin user
            username = "admin"
            password = "admin123"
            
            # Authenticate user
            user = self.auth_manager.authenticate_user(username, password, "127.0.0.1")
            if not user:
                raise Exception("User authentication failed")
            
            # Create JWT tokens
            token_data = {
                "sub": user['username'],
                "user_id": user['id'],
                "role": user['role_name'],
                "permissions": user['permissions']
            }
            
            access_token = self.auth_manager.create_access_token(token_data)
            refresh_token = self.auth_manager.create_refresh_token(token_data)
            
            if not access_token or not refresh_token:
                raise Exception("Token creation failed")
            
            # Verify tokens
            access_payload = self.auth_manager.verify_token(access_token, "access")
            refresh_payload = self.auth_manager.verify_token(refresh_token, "refresh")
            
            if not access_payload or not refresh_payload:
                raise Exception("Token verification failed")
            
            # Test token refresh
            refreshed = self.auth_manager.refresh_access_token(refresh_token)
            if not refreshed or 'access_token' not in refreshed:
                raise Exception("Token refresh failed")
            
            # Test getting current user from token
            current_user = self.auth_manager.get_current_user(access_token)
            if not current_user or current_user['id'] != user['id']:
                raise Exception("Get current user failed")
            
            logger.info("Successfully validated complete authentication flow")
            
        except Exception as e:
            raise Exception(f"Authentication flow validation failed: {e}")
    
    async def validate_permission_system(self):
        """Validate permission system"""
        if not self.auth_manager:
            raise Exception("AuthManager not initialized")
        
        try:
            # Get admin user
            admin_user = self.auth_manager.user_model.get_user_by_username("admin")
            if not admin_user:
                raise Exception("Admin user not found")
            
            # Test admin permissions
            has_users_create = self.auth_manager.validate_permission(admin_user, "users", "create")
            has_users_delete = self.auth_manager.validate_permission(admin_user, "users", "delete")
            has_system_admin = self.auth_manager.validate_permission(admin_user, "system", "admin")
            
            if not (has_users_create and has_users_delete and has_system_admin):
                raise Exception("Admin permission validation failed")
            
            # Test non-admin permissions (if student role exists)
            student_role = self.auth_manager.role_model.get_role_by_name("student")
            if student_role:
                # Create a temporary student user
                student_id = self.auth_manager.user_model.create_user(
                    username=f"temp_student_{os.getpid()}",
                    email=f"temp_{os.getpid()}@example.com",
                    password="TempStudent123!",
                    role_id=student_role['id'],
                    first_name="Temp",
                    last_name="Student",
                    is_active=True
                )
                
                if student_id:
                    student_user = self.auth_manager.user_model.get_user_by_id(student_id)
                    
                    # Student should NOT have admin permissions
                    has_admin_perm = self.auth_manager.validate_permission(student_user, "users", "create")
                    if has_admin_perm:
                        raise Exception("Student user has admin permissions (security issue)")
                    
                    # Student should have read permissions
                    has_read_perm = self.auth_manager.validate_permission(student_user, "documents", "read")
                    if not has_read_perm:
                        logger.warning("Student user doesn't have document read permissions")
                    
                    # Cleanup
                    self.auth_manager.user_model.delete_user(student_id)
            
            logger.info("Successfully validated permission system")
            
        except Exception as e:
            raise Exception(f"Permission system validation failed: {e}")
    
    async def validate_security_features(self):
        """Validate security features"""
        if not self.auth_manager:
            raise Exception("AuthManager not initialized")
        
        try:
            # Test password hashing
            test_password = "SecurityTest123!"
            hashed = self.auth_manager.user_model.db.hash_password(test_password)
            
            if test_password == hashed:
                raise Exception("Password not properly hashed")
            
            if not self.auth_manager.user_model.db.verify_password(test_password, hashed):
                raise Exception("Password verification failed")
            
            # Test token expiration (create a token with very short expiry)
            from datetime import datetime, timezone, timedelta
            import jwt
            
            expired_payload = {
                "sub": "test",
                "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
                "type": "access"
            }
            expired_token = jwt.encode(expired_payload, self.auth_manager.secret_key, algorithm=self.auth_manager.algorithm)
            
            # Verify token fails for expired token
            payload = self.auth_manager.verify_token(expired_token, "access")
            if payload is not None:
                raise Exception("Expired token validation failed")
            
            # Test rate limiting storage (basic test)
            if hasattr(self.auth_manager, '_failed_attempts'):
                # Record some failed attempts
                self.auth_manager._record_failed_attempt("test_ip")
                self.auth_manager._record_failed_attempt("test_ip")
                
                # Check if tracking works
                if "test_ip" not in self.auth_manager._failed_attempts:
                    raise Exception("Rate limiting tracking failed")
            
            logger.info("Successfully validated security features")
            
        except Exception as e:
            raise Exception(f"Security features validation failed: {e}")
    
    async def validate_api_integration(self):
        """Validate API integration readiness"""
        try:
            # Test that all required modules can be imported
            from main import app
            from api.auth import router as auth_router
            from api.users import router as users_router
            from api.roles import router as roles_router
            
            # Test that app is configured
            if not hasattr(app, 'dependency_overrides'):
                raise Exception("FastAPI app not properly configured")
            
            # Test that routers have routes
            if not auth_router.routes:
                raise Exception("Auth router has no routes")
            if not users_router.routes:
                raise Exception("Users router has no routes")
            if not roles_router.routes:
                raise Exception("Roles router has no routes")
            
            logger.info("Successfully validated API integration readiness")
            
        except Exception as e:
            raise Exception(f"API integration validation failed: {e}")
    
    def print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION VALIDATION SUMMARY")
        logger.info("="*60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result == "PASSED" else f"❌ FAILED"
            logger.info(f"{test_name:<40} {status}")
            
            if result == "PASSED":
                passed += 1
            else:
                failed += 1
                if result.startswith("FAILED:"):
                    logger.info(f"  Error: {result[8:]}")
        
        logger.info("-" * 60)
        logger.info(f"Total Tests: {passed + failed}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        
        if failed == 0:
            logger.info("🎉 ALL VALIDATIONS PASSED - Integration is ready!")
        else:
            logger.error(f"⚠️  {failed} validation(s) failed - Please review and fix issues")
        
        logger.info("="*60)


async def main():
    """Main validation function"""
    validator = IntegrationValidator()
    await validator.run_all_validations()


if __name__ == "__main__":
    asyncio.run(main())