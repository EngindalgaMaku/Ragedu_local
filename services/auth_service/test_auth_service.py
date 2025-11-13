"""
Comprehensive Tests for RAG Education Assistant Auth Service
Tests authentication, user management, role management, and security features
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app and components
from main import app, auth_manager
from auth.auth_manager import AuthManager
from auth.schemas import LoginRequest, UserCreate, RoleCreate


class TestConfig:
    """Test configuration"""
    TEST_DB_PATH = "test_auth.db"
    TEST_SECRET_KEY = "test-secret-key-for-testing-only"
    TEST_USERNAME = "testuser"
    TEST_EMAIL = "test@example.com"
    TEST_PASSWORD = "TestPass123!"


@pytest.fixture(scope="session")
def test_db():
    """Create a temporary database for testing"""
    # Create temporary directory for test database
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, TestConfig.TEST_DB_PATH)
    
    yield db_path
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_auth_manager(test_db):
    """Create AuthManager instance for testing"""
    return AuthManager(
        secret_key=TestConfig.TEST_SECRET_KEY,
        db_path=test_db,
        access_token_expire_minutes=5,  # Short expiry for testing
        refresh_token_expire_days=1
    )


@pytest.fixture(scope="session")
def client(test_auth_manager):
    """Create test client with test auth manager"""
    # Override the auth manager dependency
    app.dependency_overrides = {}
    from auth.dependencies import get_auth_manager
    app.dependency_overrides[get_auth_manager] = lambda: test_auth_manager
    
    # Set global auth manager for middleware
    global auth_manager
    auth_manager = test_auth_manager
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        "username": TestConfig.TEST_USERNAME,
        "email": TestConfig.TEST_EMAIL,
        "password": TestConfig.TEST_PASSWORD,
        "first_name": "Test",
        "last_name": "User",
        "role_id": 1,  # Admin role
        "is_active": True
    }


@pytest.fixture
def admin_token(client, test_auth_manager):
    """Get admin token for testing"""
    # Login as default admin user
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 200
    data = response.json()
    return data["access_token"]


class TestHealthAndInfo:
    """Test health check and info endpoints"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_info_endpoint(self, client):
        """Test service info endpoint"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "features" in data
        assert "endpoints" in data


class TestAuthentication:
    """Test authentication functionality"""
    
    def test_login_success(self, client):
        """Test successful login with default admin"""
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["username"] == "admin"
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        login_data = {
            "username": "admin",
            "password": "wrong_password"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
        assert "Invalid username or password" in response.json()["detail"]
    
    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent user"""
        login_data = {
            "username": "nonexistent",
            "password": "password123"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
    
    def test_refresh_token(self, client):
        """Test token refresh"""
        # First, login to get tokens
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        
        tokens = response.json()
        refresh_token = tokens["refresh_token"]
        
        # Use refresh token to get new access token
        refresh_data = {
            "refresh_token": refresh_token
        }
        response = client.post("/auth/refresh", json=refresh_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_refresh_invalid_token(self, client):
        """Test refresh with invalid token"""
        refresh_data = {
            "refresh_token": "invalid_token"
        }
        response = client.post("/auth/refresh", json=refresh_data)
        assert response.status_code == 401
    
    def test_get_current_user(self, client, admin_token):
        """Test getting current user info"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["username"] == "admin"
        assert data["role_name"] == "admin"
    
    def test_logout(self, client, admin_token):
        """Test user logout"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.post("/auth/logout", headers=headers)
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_change_password(self, client, admin_token):
        """Test password change"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        password_data = {
            "current_password": "admin123",
            "new_password": "NewPassword123!"
        }
        response = client.put("/auth/change-password", json=password_data, headers=headers)
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Change back to original password for other tests
        password_data = {
            "current_password": "NewPassword123!",
            "new_password": "admin123"
        }
        response = client.put("/auth/change-password", json=password_data, headers=headers)
        assert response.status_code == 200
    
    def test_unauthorized_access(self, client):
        """Test access without token"""
        response = client.get("/auth/me")
        assert response.status_code == 401
    
    def test_invalid_token_access(self, client):
        """Test access with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 401


class TestUserManagement:
    """Test user management functionality"""
    
    def test_create_user(self, client, admin_token, sample_user_data):
        """Test user creation"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.post("/users", json=sample_user_data, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["username"] == sample_user_data["username"]
        assert data["email"] == sample_user_data["email"]
        assert data["is_active"] is True
    
    def test_create_duplicate_user(self, client, admin_token, sample_user_data):
        """Test creating user with duplicate username"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # First creation should succeed
        response = client.post("/users", json=sample_user_data, headers=headers)
        if response.status_code != 200:
            # User might already exist from previous test
            pass
        
        # Second creation should fail
        response = client.post("/users", json=sample_user_data, headers=headers)
        assert response.status_code == 400
    
    def test_list_users(self, client, admin_token):
        """Test listing users"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/users", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert len(data["users"]) >= 1  # At least admin user
    
    def test_get_user_by_id(self, client, admin_token):
        """Test getting user by ID"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Get admin user (ID should be 1)
        response = client.get("/users/1", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["username"] == "admin"
    
    def test_get_nonexistent_user(self, client, admin_token):
        """Test getting nonexistent user"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/users/9999", headers=headers)
        assert response.status_code == 404
    
    def test_update_user(self, client, admin_token):
        """Test updating user"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Create a user first
        user_data = {
            "username": "updatetest",
            "email": "update@example.com",
            "password": "TestPass123!",
            "first_name": "Update",
            "last_name": "Test",
            "role_id": 3,  # Student role
            "is_active": True
        }
        create_response = client.post("/users", json=user_data, headers=headers)
        if create_response.status_code == 200:
            user_id = create_response.json()["id"]
            
            # Update the user
            update_data = {
                "first_name": "Updated",
                "last_name": "User"
            }
            response = client.put(f"/users/{user_id}", json=update_data, headers=headers)
            assert response.status_code == 200
            
            data = response.json()
            assert data["first_name"] == "Updated"
            assert data["last_name"] == "User"
    
    def test_activate_deactivate_user(self, client, admin_token):
        """Test user activation and deactivation"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Create a user first
        user_data = {
            "username": "activatetest",
            "email": "activate@example.com",
            "password": "TestPass123!",
            "first_name": "Activate",
            "last_name": "Test",
            "role_id": 3,
            "is_active": True
        }
        create_response = client.post("/users", json=user_data, headers=headers)
        if create_response.status_code == 200:
            user_id = create_response.json()["id"]
            
            # Deactivate user
            response = client.post(f"/users/{user_id}/deactivate", headers=headers)
            assert response.status_code == 200
            
            # Activate user
            response = client.post(f"/users/{user_id}/activate", headers=headers)
            assert response.status_code == 200
    
    def test_user_statistics(self, client, admin_token):
        """Test user statistics"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/users/stats/overview", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_users" in data
        assert "active_users" in data
        assert "users_by_role" in data


class TestRoleManagement:
    """Test role management functionality"""
    
    def test_list_roles(self, client, admin_token):
        """Test listing roles"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/roles", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "roles" in data
        assert "total" in data
        assert len(data["roles"]) >= 3  # At least admin, teacher, student
    
    def test_get_role_by_id(self, client, admin_token):
        """Test getting role by ID"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Get admin role (ID should be 1)
        response = client.get("/roles/1", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "admin"
        assert "permissions" in data
    
    def test_create_role(self, client, admin_token):
        """Test role creation"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        role_data = {
            "name": "test_role",
            "description": "Test role for testing",
            "permissions": {
                "documents": ["read"],
                "test": ["create", "read"]
            }
        }
        response = client.post("/roles", json=role_data, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "test_role"
        assert data["permissions"]["documents"] == ["read"]
    
    def test_update_role(self, client, admin_token):
        """Test role update"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Create a role first
        role_data = {
            "name": "update_test_role",
            "description": "Role for update testing",
            "permissions": {"test": ["read"]}
        }
        create_response = client.post("/roles", json=role_data, headers=headers)
        if create_response.status_code == 200:
            role_id = create_response.json()["id"]
            
            # Update the role
            update_data = {
                "description": "Updated description",
                "permissions": {"test": ["read", "write"]}
            }
            response = client.put(f"/roles/{role_id}", json=update_data, headers=headers)
            assert response.status_code == 200
            
            data = response.json()
            assert data["description"] == "Updated description"
            assert "write" in data["permissions"]["test"]
    
    def test_role_permissions(self, client, admin_token):
        """Test role permission management"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Create a role first
        role_data = {
            "name": "permission_test_role",
            "description": "Role for permission testing",
            "permissions": {}
        }
        create_response = client.post("/roles", json=role_data, headers=headers)
        if create_response.status_code == 200:
            role_id = create_response.json()["id"]
            
            # Add permission
            response = client.post(
                f"/roles/{role_id}/permissions?resource=test&action=read",
                headers=headers
            )
            assert response.status_code == 200
            
            # Check permission
            response = client.get(
                f"/roles/{role_id}/permissions/test?action=read",
                headers=headers
            )
            assert response.status_code == 200
            assert response.json()["has_permission"] is True
            
            # Remove permission
            response = client.delete(
                f"/roles/{role_id}/permissions?resource=test&action=read",
                headers=headers
            )
            assert response.status_code == 200


class TestPermissions:
    """Test permission system"""
    
    def test_permission_check(self, client, admin_token):
        """Test permission checking"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        permission_data = {
            "resource": "users",
            "action": "create"
        }
        response = client.post("/auth/check-permission", json=permission_data, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["has_permission"] is True  # Admin should have all permissions
        assert data["resource"] == "users"
        assert data["action"] == "create"
    
    def test_forbidden_access(self, client):
        """Test access to admin-only endpoint without permission"""
        # This test would require creating a non-admin user and token
        # For now, test that endpoints require authentication
        response = client.get("/users")
        assert response.status_code == 401


class TestSecurity:
    """Test security features"""
    
    def test_password_validation(self, client, admin_token):
        """Test password strength validation"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Test weak password
        weak_user_data = {
            "username": "weakpass",
            "email": "weak@example.com",
            "password": "123",  # Too weak
            "first_name": "Weak",
            "last_name": "Pass",
            "role_id": 3,
            "is_active": True
        }
        response = client.post("/users", json=weak_user_data, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_rate_limiting(self, client):
        """Test rate limiting on login endpoint"""
        # This is a basic test - in production you'd want more sophisticated testing
        login_data = {
            "username": "nonexistent",
            "password": "wrong"
        }
        
        # Make multiple rapid requests
        responses = []
        for _ in range(5):
            response = client.post("/auth/login", json=login_data)
            responses.append(response.status_code)
        
        # At least some should be 401 (failed login)
        assert 401 in responses
    
    def test_token_validation(self, client):
        """Test token validation endpoint"""
        # Test without token
        response = client.post("/auth/validate-token")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        
        # Test with valid token would require a token
        # This is covered in other tests


class TestSessionManagement:
    """Test session management"""
    
    def test_delete_session(self, client, admin_token):
        """Test session deletion"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # This would require knowing a session ID
        # For now, test the endpoint exists and handles errors properly
        response = client.delete("/auth/sessions/999", headers=headers)
        assert response.status_code in [404, 403]  # Session not found or forbidden
    
    def test_cleanup_sessions(self, client, admin_token):
        """Test session cleanup"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.post("/auth/cleanup-sessions", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data


class TestDatabaseIntegration:
    """Test database integration"""
    
    def test_database_initialization(self, test_auth_manager):
        """Test that database initializes correctly"""
        assert test_auth_manager is not None
        assert test_auth_manager.user_model is not None
        assert test_auth_manager.role_model is not None
        assert test_auth_manager.session_model is not None
    
    def test_default_roles_creation(self, test_auth_manager):
        """Test that default roles are created"""
        admin_role = test_auth_manager.role_model.get_role_by_name("admin")
        teacher_role = test_auth_manager.role_model.get_role_by_name("teacher")
        student_role = test_auth_manager.role_model.get_role_by_name("student")
        
        assert admin_role is not None
        assert teacher_role is not None
        assert student_role is not None
    
    def test_admin_user_creation(self, test_auth_manager):
        """Test that default admin user is created"""
        admin_user = test_auth_manager.user_model.get_user_by_username("admin")
        assert admin_user is not None
        assert admin_user["role_name"] == "admin"
        assert admin_user["is_active"] is True


# Performance and load testing (basic)
class TestPerformance:
    """Basic performance tests"""
    
    def test_login_performance(self, client):
        """Test login endpoint performance"""
        import time
        
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        
        start_time = time.time()
        response = client.post("/auth/login", json=login_data)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 2.0  # Should complete within 2 seconds
    
    def test_concurrent_requests(self, client):
        """Test handling concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/health").status_code
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(status == 200 for status in results)


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_user_workflow(self, client, admin_token):
        """Test complete user management workflow"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # 1. Create user
        user_data = {
            "username": "workflow_test",
            "email": "workflow@example.com",
            "password": "WorkflowTest123!",
            "first_name": "Workflow",
            "last_name": "Test",
            "role_id": 3,
            "is_active": True
        }
        create_response = client.post("/users", json=user_data, headers=headers)
        
        if create_response.status_code == 200:
            user_id = create_response.json()["id"]
            
            # 2. Get user
            get_response = client.get(f"/users/{user_id}", headers=headers)
            assert get_response.status_code == 200
            
            # 3. Update user
            update_data = {"first_name": "Updated"}
            update_response = client.put(f"/users/{user_id}", json=update_data, headers=headers)
            assert update_response.status_code == 200
            
            # 4. Deactivate user
            deactivate_response = client.post(f"/users/{user_id}/deactivate", headers=headers)
            assert deactivate_response.status_code == 200
            
            # 5. Verify user is deactivated
            final_response = client.get(f"/users/{user_id}", headers=headers)
            assert final_response.status_code == 200
            assert final_response.json()["is_active"] is False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])