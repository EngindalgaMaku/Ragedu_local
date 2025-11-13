"""
Comprehensive Authentication System Integration Tests
Tests the complete auth flow from database to API to frontend integration
"""

import pytest
import asyncio
import httpx
import json
import time
import subprocess
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database import DatabaseManager, User, Role, UserSession


class TestAuthSystemIntegration:
    """Comprehensive authentication system integration tests"""
    
    @pytest.fixture(scope="class")
    def setup_test_environment(self):
        """Setup test environment with clean database and auth service"""
        # Use test database
        self.db_path = "test_integration_auth.db"
        self.auth_service_url = "http://localhost:8002"
        self.api_gateway_url = "http://localhost:8000"
        
        # Clean up any existing test database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        # Initialize database
        self.db_manager = DatabaseManager(self.db_path)
        self.user_model = User(self.db_manager)
        self.role_model = Role(self.db_manager)
        self.session_model = UserSession(self.db_manager)
        
        # Create default roles
        self.role_model.create_default_roles()
        
        # Create test users
        self._create_test_users()
        
        yield
        
        # Cleanup
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def _create_test_users(self):
        """Create test users for different roles"""
        # Get role IDs
        admin_role = self.role_model.get_role_by_name("admin")
        teacher_role = self.role_model.get_role_by_name("teacher")
        student_role = self.role_model.get_role_by_name("student")
        
        # Create test users
        self.admin_user_id = self.user_model.create_user(
            username="admin_test",
            email="admin@test.com",
            password="AdminTest123!",
            role_id=admin_role['id'],
            first_name="Admin",
            last_name="User"
        )
        
        self.teacher_user_id = self.user_model.create_user(
            username="teacher_test",
            email="teacher@test.com", 
            password="TeacherTest123!",
            role_id=teacher_role['id'],
            first_name="Teacher",
            last_name="User"
        )
        
        self.student_user_id = self.user_model.create_user(
            username="student_test",
            email="student@test.com",
            password="StudentTest123!",
            role_id=student_role['id'],
            first_name="Student",
            last_name="User"
        )
    
    @pytest.mark.asyncio
    async def test_database_models_functionality(self, setup_test_environment):
        """Test all database models work correctly"""
        
        # Test User model
        user = self.user_model.get_user_by_username("admin_test")
        assert user is not None
        assert user['username'] == "admin_test"
        assert user['email'] == "admin@test.com"
        assert user['is_active'] is True
        
        # Test authentication
        auth_user = self.user_model.authenticate_user("admin_test", "AdminTest123!")
        assert auth_user is not None
        assert auth_user['username'] == "admin_test"
        
        # Test wrong password
        wrong_auth = self.user_model.authenticate_user("admin_test", "wrongpassword")
        assert wrong_auth is None
        
        # Test Role model
        admin_role = self.role_model.get_role_by_name("admin")
        assert admin_role is not None
        assert 'users' in admin_role['permissions']
        assert 'create' in admin_role['permissions']['users']
        
        # Test permission checking
        has_permission = self.role_model.has_permission(admin_role['id'], 'users', 'create')
        assert has_permission is True
        
        no_permission = self.role_model.has_permission(admin_role['id'], 'nonexistent', 'create')
        assert no_permission is False
        
        # Test Session model
        session_id = self.session_model.create_session(
            user_id=self.admin_user_id,
            token="test_token_123",
            expires_in_hours=24,
            ip_address="127.0.0.1",
            user_agent="pytest"
        )
        assert session_id is not None
        
        session = self.session_model.get_session_by_id(session_id)
        assert session is not None
        assert session['user_id'] == self.admin_user_id
        
    @pytest.mark.asyncio
    async def test_auth_service_health(self, setup_test_environment):
        """Test auth service health and availability"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.auth_service_url}/health")
                assert response.status_code == 200
                
                health_data = response.json()
                assert health_data['status'] == 'healthy'
                assert 'database_status' in health_data
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running - start with: python services/auth_service/main.py")
    
    @pytest.mark.asyncio
    async def test_auth_service_login_flow(self, setup_test_environment):
        """Test complete authentication flow through auth service"""
        async with httpx.AsyncClient() as client:
            try:
                # Test admin login
                login_data = {
                    "username": "admin_test",
                    "password": "AdminTest123!"
                }
                
                response = await client.post(f"{self.auth_service_url}/auth/login", json=login_data)
                assert response.status_code == 200
                
                login_result = response.json()
                assert 'access_token' in login_result
                assert 'refresh_token' in login_result
                assert login_result['token_type'] == 'bearer'
                assert login_result['user']['username'] == 'admin_test'
                assert login_result['user']['role_name'] == 'admin'
                
                # Store tokens for further tests
                access_token = login_result['access_token']
                refresh_token = login_result['refresh_token']
                
                # Test getting current user with access token
                headers = {"Authorization": f"Bearer {access_token}"}
                me_response = await client.get(f"{self.auth_service_url}/auth/me", headers=headers)
                assert me_response.status_code == 200
                
                user_data = me_response.json()
                assert user_data['username'] == 'admin_test'
                assert user_data['role_name'] == 'admin'
                
                # Test permission checking
                permission_data = {
                    "resource": "users",
                    "action": "create"
                }
                perm_response = await client.post(
                    f"{self.auth_service_url}/auth/check-permission",
                    json=permission_data,
                    headers=headers
                )
                assert perm_response.status_code == 200
                
                perm_result = perm_response.json()
                assert perm_result['has_permission'] is True
                
                # Test token refresh
                refresh_data = {"refresh_token": refresh_token}
                refresh_response = await client.post(
                    f"{self.auth_service_url}/auth/refresh",
                    json=refresh_data
                )
                assert refresh_response.status_code == 200
                
                refresh_result = refresh_response.json()
                assert 'access_token' in refresh_result
                
                # Test logout
                logout_data = {"refresh_token": refresh_token}
                logout_response = await client.post(
                    f"{self.auth_service_url}/auth/logout",
                    json=logout_data,
                    headers=headers
                )
                assert logout_response.status_code == 200
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio 
    async def test_role_based_access_control(self, setup_test_environment):
        """Test role-based access control across different user types"""
        async with httpx.AsyncClient() as client:
            try:
                # Test different user roles
                test_users = [
                    {"username": "admin_test", "password": "AdminTest123!", "role": "admin"},
                    {"username": "teacher_test", "password": "TeacherTest123!", "role": "teacher"},
                    {"username": "student_test", "password": "StudentTest123!", "role": "student"}
                ]
                
                for user_data in test_users:
                    # Login
                    login_response = await client.post(
                        f"{self.auth_service_url}/auth/login",
                        json={"username": user_data["username"], "password": user_data["password"]}
                    )
                    assert login_response.status_code == 200
                    
                    login_result = login_response.json()
                    access_token = login_result['access_token']
                    headers = {"Authorization": f"Bearer {access_token}"}
                    
                    # Verify user role
                    me_response = await client.get(f"{self.auth_service_url}/auth/me", headers=headers)
                    assert me_response.status_code == 200
                    assert me_response.json()['role_name'] == user_data["role"]
                    
                    # Test permissions based on role
                    permission_tests = [
                        {"resource": "users", "action": "create"},
                        {"resource": "users", "action": "read"},
                        {"resource": "sessions", "action": "create"},
                        {"resource": "documents", "action": "create"}
                    ]
                    
                    for perm_test in permission_tests:
                        perm_response = await client.post(
                            f"{self.auth_service_url}/auth/check-permission",
                            json=perm_test,
                            headers=headers
                        )
                        assert perm_response.status_code == 200
                        
                        perm_result = perm_response.json()
                        
                        # Verify permissions match role expectations
                        if user_data["role"] == "admin":
                            assert perm_result['has_permission'] is True  # Admin has all permissions
                        elif user_data["role"] == "teacher":
                            if perm_test["resource"] == "users" and perm_test["action"] == "create":
                                assert perm_result['has_permission'] is False  # Teachers can't create users
                            elif perm_test["resource"] in ["sessions", "documents"]:
                                assert perm_result['has_permission'] is True  # Teachers can manage sessions/docs
                        elif user_data["role"] == "student":
                            if perm_test["action"] in ["create", "update", "delete"]:
                                assert perm_result['has_permission'] is False  # Students have read-only access
                            else:
                                assert perm_result['has_permission'] is True  # Students can read
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_session_management(self, setup_test_environment):
        """Test session management functionality"""
        async with httpx.AsyncClient() as client:
            try:
                # Login to create session
                login_response = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "admin_test", "password": "AdminTest123!"}
                )
                assert login_response.status_code == 200
                
                login_result = login_response.json()
                refresh_token = login_result['refresh_token']
                headers = {"Authorization": f"Bearer {login_result['access_token']}"}
                
                # Test session cleanup (admin only)
                cleanup_response = await client.post(
                    f"{self.auth_service_url}/auth/cleanup-sessions",
                    headers=headers
                )
                assert cleanup_response.status_code == 200
                
                cleanup_result = cleanup_response.json()
                assert cleanup_result['success'] is True
                
                # Test logout from all sessions
                logout_all_response = await client.post(
                    f"{self.auth_service_url}/auth/logout",
                    json={"all_sessions": True},
                    headers=headers
                )
                assert logout_all_response.status_code == 200
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, setup_test_environment):
        """Test error handling and security measures"""
        async with httpx.AsyncClient() as client:
            try:
                # Test invalid login
                invalid_login = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "nonexistent", "password": "wrongpass"}
                )
                assert invalid_login.status_code == 401
                
                # Test invalid token
                invalid_headers = {"Authorization": "Bearer invalid_token"}
                me_response = await client.get(
                    f"{self.auth_service_url}/auth/me",
                    headers=invalid_headers
                )
                assert me_response.status_code == 401
                
                # Test malformed requests
                malformed_response = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "test"}  # Missing password
                )
                assert malformed_response.status_code == 422
                
                # Test token validation endpoint
                token_validation = await client.post(
                    f"{self.auth_service_url}/auth/validate-token",
                    headers=invalid_headers
                )
                assert token_validation.status_code == 200
                
                validation_result = token_validation.json()
                assert validation_result['success'] is False
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_api_gateway_integration(self, setup_test_environment):
        """Test integration with API gateway for protected endpoints"""
        async with httpx.AsyncClient() as client:
            try:
                # First login through auth service to get token
                login_response = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "admin_test", "password": "AdminTest123!"}
                )
                
                if login_response.status_code == 200:
                    login_result = login_response.json()
                    access_token = login_result['access_token']
                    headers = {"Authorization": f"Bearer {access_token}"}
                    
                    # Test API gateway endpoints that should require authentication
                    # Note: This assumes API gateway is configured to validate tokens with auth service
                    protected_endpoints = [
                        f"{self.api_gateway_url}/sessions",
                        f"{self.api_gateway_url}/documents", 
                        f"{self.api_gateway_url}/admin"
                    ]
                    
                    for endpoint in protected_endpoints:
                        try:
                            response = await client.get(endpoint, headers=headers)
                            # Should not get 401 Unauthorized with valid token
                            assert response.status_code != 401
                        except httpx.ConnectError:
                            pass  # Skip if endpoint not available
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_password_security(self, setup_test_environment):
        """Test password security features"""
        async with httpx.AsyncClient() as client:
            try:
                # Login first
                login_response = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "admin_test", "password": "AdminTest123!"}
                )
                assert login_response.status_code == 200
                
                login_result = login_response.json()
                headers = {"Authorization": f"Bearer {login_result['access_token']}"}
                
                # Test password change
                password_change = {
                    "current_password": "AdminTest123!",
                    "new_password": "NewAdminTest456!"
                }
                
                change_response = await client.put(
                    f"{self.auth_service_url}/auth/change-password",
                    json=password_change,
                    headers=headers
                )
                assert change_response.status_code == 200
                
                change_result = change_response.json()
                assert change_result['success'] is True
                
                # Test login with new password
                new_login_response = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "admin_test", "password": "NewAdminTest456!"}
                )
                assert new_login_response.status_code == 200
                
                # Test old password no longer works
                old_login_response = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "admin_test", "password": "AdminTest123!"}
                )
                assert old_login_response.status_code == 401
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])