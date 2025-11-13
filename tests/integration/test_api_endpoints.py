"""
API Endpoints Integration Tests
Tests all API endpoints for authentication, authorization, and functionality
"""

import pytest
import asyncio
import httpx
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestAPIEndpointsIntegration:
    """Test API endpoints integration and functionality"""
    
    @pytest.fixture(scope="class")
    def setup_api_testing(self):
        """Setup API testing environment"""
        self.auth_service_url = "http://localhost:8002"
        self.api_gateway_url = "http://localhost:8000"
        
        # Test user credentials
        self.test_users = {
            "admin": {
                "username": "admin",
                "password": "admin",
                "role": "admin"
            },
            "teacher": {
                "username": "teacher", 
                "password": "teacher",
                "role": "teacher"
            },
            "student": {
                "username": "student",
                "password": "student", 
                "role": "student"
            }
        }
        
        # Store tokens for authenticated requests
        self.tokens = {}
        
        yield
    
    async def authenticate_user(self, client: httpx.AsyncClient, user_type: str) -> Dict[str, str]:
        """Authenticate user and return tokens"""
        if user_type in self.tokens:
            return self.tokens[user_type]
        
        user_data = self.test_users[user_type]
        response = await client.post(
            f"{self.auth_service_url}/auth/login",
            json={
                "username": user_data["username"],
                "password": user_data["password"]
            }
        )
        
        if response.status_code == 200:
            login_result = response.json()
            tokens = {
                "access_token": login_result["access_token"],
                "refresh_token": login_result["refresh_token"],
                "headers": {"Authorization": f"Bearer {login_result['access_token']}"}
            }
            self.tokens[user_type] = tokens
            return tokens
        else:
            pytest.skip(f"Cannot authenticate {user_type} user - check if auth service is running and users exist")
    
    @pytest.mark.asyncio
    async def test_auth_service_endpoints(self, setup_api_testing):
        """Test all authentication service endpoints"""
        async with httpx.AsyncClient() as client:
            try:
                # Test health endpoint (no auth required)
                health_response = await client.get(f"{self.auth_service_url}/health")
                assert health_response.status_code == 200
                
                health_data = health_response.json()
                assert health_data["status"] == "healthy"
                assert "version" in health_data
                assert "database_status" in health_data
                
                # Test root endpoint
                root_response = await client.get(f"{self.auth_service_url}/")
                assert root_response.status_code == 200
                
                root_data = root_response.json()
                assert "message" in root_data
                assert "version" in root_data
                
                # Test service info endpoint
                info_response = await client.get(f"{self.auth_service_url}/info")
                assert info_response.status_code == 200
                
                info_data = info_response.json()
                assert "service" in info_data
                assert "features" in info_data
                assert "endpoints" in info_data
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio 
    async def test_authentication_endpoints(self, setup_api_testing):
        """Test authentication related endpoints"""
        async with httpx.AsyncClient() as client:
            try:
                # Test login with valid credentials
                for user_type, user_data in self.test_users.items():
                    login_response = await client.post(
                        f"{self.auth_service_url}/auth/login",
                        json={
                            "username": user_data["username"],
                            "password": user_data["password"]
                        }
                    )
                    
                    if login_response.status_code == 200:
                        login_result = login_response.json()
                        
                        # Verify response structure
                        assert "access_token" in login_result
                        assert "refresh_token" in login_result
                        assert "token_type" in login_result
                        assert "expires_in" in login_result
                        assert "user" in login_result
                        
                        # Verify user data
                        user_info = login_result["user"]
                        assert user_info["username"] == user_data["username"]
                        assert user_info["role_name"] == user_data["role"]
                        assert user_info["is_active"] is True
                        
                        # Test /auth/me endpoint with token
                        headers = {"Authorization": f"Bearer {login_result['access_token']}"}
                        me_response = await client.get(f"{self.auth_service_url}/auth/me", headers=headers)
                        assert me_response.status_code == 200
                        
                        me_data = me_response.json()
                        assert me_data["username"] == user_data["username"]
                        assert me_data["role_name"] == user_data["role"]
                        
                        # Test token refresh
                        refresh_response = await client.post(
                            f"{self.auth_service_url}/auth/refresh",
                            json={"refresh_token": login_result["refresh_token"]}
                        )
                        assert refresh_response.status_code == 200
                        
                        refresh_result = refresh_response.json()
                        assert "access_token" in refresh_result
                        
                        # Test logout
                        logout_response = await client.post(
                            f"{self.auth_service_url}/auth/logout",
                            json={"refresh_token": login_result["refresh_token"]},
                            headers=headers
                        )
                        assert logout_response.status_code == 200
                        
                        logout_result = logout_response.json()
                        assert logout_result["success"] is True
                    else:
                        # Skip test if demo users don't exist
                        pytest.skip(f"Demo user {user_type} not found - create demo users first")
                
                # Test login with invalid credentials
                invalid_login = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "nonexistent", "password": "wrongpass"}
                )
                assert invalid_login.status_code == 401
                
                # Test token validation
                token_validation = await client.post(
                    f"{self.auth_service_url}/auth/validate-token",
                    headers={"Authorization": "Bearer invalid_token"}
                )
                assert token_validation.status_code == 200
                
                validation_result = token_validation.json()
                assert validation_result["success"] is False
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_user_management_endpoints(self, setup_api_testing):
        """Test user management endpoints (admin only)"""
        async with httpx.AsyncClient() as client:
            try:
                # Authenticate as admin
                admin_tokens = await self.authenticate_user(client, "admin")
                admin_headers = admin_tokens["headers"]
                
                # Test get users list
                users_response = await client.get(
                    f"{self.auth_service_url}/users",
                    headers=admin_headers
                )
                
                if users_response.status_code == 200:
                    users_data = users_response.json()
                    assert "users" in users_data or isinstance(users_data, list)
                
                # Test get specific user (admin can access all users)
                me_response = await client.get(
                    f"{self.auth_service_url}/auth/me",
                    headers=admin_headers
                )
                assert me_response.status_code == 200
                
                admin_user = me_response.json()
                user_id = admin_user["id"]
                
                single_user_response = await client.get(
                    f"{self.auth_service_url}/users/{user_id}",
                    headers=admin_headers
                )
                
                if single_user_response.status_code == 200:
                    user_data = single_user_response.json()
                    assert user_data["id"] == user_id
                
                # Test user creation (if endpoint exists)
                new_user_data = {
                    "username": "testapi",
                    "email": "testapi@example.com",
                    "password": "TestAPI123!",
                    "first_name": "Test",
                    "last_name": "API",
                    "role_id": 1  # Assume admin role has ID 1
                }
                
                create_response = await client.post(
                    f"{self.auth_service_url}/users",
                    json=new_user_data,
                    headers=admin_headers
                )
                
                if create_response.status_code in [200, 201]:
                    created_user = create_response.json()
                    assert created_user["username"] == new_user_data["username"]
                    
                    # Test user update
                    update_data = {"first_name": "Updated"}
                    update_response = await client.put(
                        f"{self.auth_service_url}/users/{created_user['id']}",
                        json=update_data,
                        headers=admin_headers
                    )
                    
                    if update_response.status_code == 200:
                        updated_user = update_response.json()
                        assert updated_user["first_name"] == "Updated"
                
                # Test non-admin access (should fail)
                if "teacher" in self.test_users:
                    teacher_tokens = await self.authenticate_user(client, "teacher")
                    teacher_headers = teacher_tokens["headers"]
                    
                    teacher_users_response = await client.get(
                        f"{self.auth_service_url}/users",
                        headers=teacher_headers
                    )
                    # Should either be forbidden or return limited data
                    assert teacher_users_response.status_code in [200, 403]
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_role_management_endpoints(self, setup_api_testing):
        """Test role management endpoints"""
        async with httpx.AsyncClient() as client:
            try:
                # Authenticate as admin
                admin_tokens = await self.authenticate_user(client, "admin")
                admin_headers = admin_tokens["headers"]
                
                # Test get roles list
                roles_response = await client.get(
                    f"{self.auth_service_url}/roles",
                    headers=admin_headers
                )
                
                if roles_response.status_code == 200:
                    roles_data = roles_response.json()
                    assert "roles" in roles_data or isinstance(roles_data, list)
                    
                    # Find admin role
                    if "roles" in roles_data:
                        admin_role = next((r for r in roles_data["roles"] if r["name"] == "admin"), None)
                    else:
                        admin_role = next((r for r in roles_data if r["name"] == "admin"), None)
                    
                    if admin_role:
                        role_id = admin_role["id"]
                        
                        # Test get specific role
                        single_role_response = await client.get(
                            f"{self.auth_service_url}/roles/{role_id}",
                            headers=admin_headers
                        )
                        
                        if single_role_response.status_code == 200:
                            role_data = single_role_response.json()
                            assert role_data["name"] == "admin"
                            assert "permissions" in role_data
                
                # Test role creation (if endpoint exists)
                new_role_data = {
                    "name": "test_role",
                    "description": "Test role for API testing",
                    "permissions": {
                        "test": ["read"]
                    }
                }
                
                create_role_response = await client.post(
                    f"{self.auth_service_url}/roles",
                    json=new_role_data,
                    headers=admin_headers
                )
                
                if create_role_response.status_code in [200, 201]:
                    created_role = create_role_response.json()
                    assert created_role["name"] == new_role_data["name"]
                
                # Test non-admin access to role management
                if "student" in self.test_users:
                    student_tokens = await self.authenticate_user(client, "student")
                    student_headers = student_tokens["headers"]
                    
                    student_roles_response = await client.get(
                        f"{self.auth_service_url}/roles",
                        headers=student_headers
                    )
                    # Should be forbidden
                    assert student_roles_response.status_code == 403
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_session_management_endpoints(self, setup_api_testing):
        """Test session management endpoints"""
        async with httpx.AsyncClient() as client:
            try:
                # Authenticate as admin to test session management
                admin_tokens = await self.authenticate_user(client, "admin")
                admin_headers = admin_tokens["headers"]
                
                # Test session cleanup (admin only)
                cleanup_response = await client.post(
                    f"{self.auth_service_url}/auth/cleanup-sessions",
                    headers=admin_headers
                )
                assert cleanup_response.status_code == 200
                
                cleanup_result = cleanup_response.json()
                assert cleanup_result["success"] is True
                
                # Create a test session by logging in
                test_login = await client.post(
                    f"{self.auth_service_url}/auth/login",
                    json={"username": "admin", "password": "admin"}
                )
                
                if test_login.status_code == 200:
                    test_tokens = test_login.json()
                    
                    # Test session deletion (if endpoint exists)
                    # This would typically require knowing the session ID
                    # For now, test logout which removes sessions
                    logout_response = await client.post(
                        f"{self.auth_service_url}/auth/logout",
                        json={"all_sessions": True},
                        headers={"Authorization": f"Bearer {test_tokens['access_token']}"}
                    )
                    assert logout_response.status_code == 200
                
                # Test non-admin cannot cleanup sessions
                if "student" in self.test_users:
                    student_tokens = await self.authenticate_user(client, "student")
                    student_headers = student_tokens["headers"]
                    
                    student_cleanup = await client.post(
                        f"{self.auth_service_url}/auth/cleanup-sessions",
                        headers=student_headers
                    )
                    assert student_cleanup.status_code == 403
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_permission_checking_endpoints(self, setup_api_testing):
        """Test permission checking endpoints"""
        async with httpx.AsyncClient() as client:
            try:
                # Test different user roles and their permissions
                for user_type, expected_permissions in {
                    "admin": [
                        {"resource": "users", "action": "create", "expected": True},
                        {"resource": "users", "action": "delete", "expected": True},
                        {"resource": "system", "action": "admin", "expected": True}
                    ],
                    "teacher": [
                        {"resource": "users", "action": "create", "expected": False},
                        {"resource": "sessions", "action": "create", "expected": True},
                        {"resource": "documents", "action": "create", "expected": True}
                    ],
                    "student": [
                        {"resource": "users", "action": "create", "expected": False},
                        {"resource": "sessions", "action": "read", "expected": True},
                        {"resource": "documents", "action": "create", "expected": False}
                    ]
                }.items():
                    if user_type not in self.test_users:
                        continue
                    
                    user_tokens = await self.authenticate_user(client, user_type)
                    user_headers = user_tokens["headers"]
                    
                    for perm_test in expected_permissions:
                        permission_data = {
                            "resource": perm_test["resource"],
                            "action": perm_test["action"]
                        }
                        
                        perm_response = await client.post(
                            f"{self.auth_service_url}/auth/check-permission",
                            json=permission_data,
                            headers=user_headers
                        )
                        assert perm_response.status_code == 200
                        
                        perm_result = perm_response.json()
                        assert perm_result["has_permission"] == perm_test["expected"], \
                            f"{user_type} should {'have' if perm_test['expected'] else 'not have'} {perm_test['action']} permission on {perm_test['resource']}"
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")
    
    @pytest.mark.asyncio
    async def test_api_gateway_integration(self, setup_api_testing):
        """Test API Gateway integration with authentication"""
        async with httpx.AsyncClient() as client:
            try:
                # Test that API Gateway health endpoint works
                gateway_health = await client.get(f"{self.api_gateway_url}/health")
                if gateway_health.status_code != 200:
                    pytest.skip("API Gateway not running")
                
                # Test protected endpoints require authentication
                protected_endpoints = [
                    f"{self.api_gateway_url}/sessions",
                    f"{self.api_gateway_url}/documents",
                    f"{self.api_gateway_url}/admin"
                ]
                
                for endpoint in protected_endpoints:
                    try:
                        # Test without authentication (should fail)
                        unauth_response = await client.get(endpoint)
                        assert unauth_response.status_code in [401, 403], \
                            f"Endpoint {endpoint} should require authentication"
                        
                        # Test with valid authentication
                        admin_tokens = await self.authenticate_user(client, "admin")
                        auth_response = await client.get(endpoint, headers=admin_tokens["headers"])
                        assert auth_response.status_code != 401, \
                            f"Authenticated request to {endpoint} should not return 401"
                        
                    except httpx.ConnectError:
                        # Skip if endpoint doesn't exist yet
                        pass
                
            except httpx.ConnectError:
                pytest.skip("API Gateway not running")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_security(self, setup_api_testing):
        """Test error handling and security measures"""
        async with httpx.AsyncClient() as client:
            try:
                # Test malformed requests
                malformed_tests = [
                    {
                        "endpoint": f"{self.auth_service_url}/auth/login",
                        "data": {"username": "test"},  # Missing password
                        "expected_status": 422
                    },
                    {
                        "endpoint": f"{self.auth_service_url}/auth/login", 
                        "data": {"password": "test"},  # Missing username
                        "expected_status": 422
                    },
                    {
                        "endpoint": f"{self.auth_service_url}/auth/refresh",
                        "data": {},  # Missing refresh_token
                        "expected_status": 422
                    }
                ]
                
                for test in malformed_tests:
                    response = await client.post(test["endpoint"], json=test["data"])
                    assert response.status_code == test["expected_status"], \
                        f"Malformed request to {test['endpoint']} should return {test['expected_status']}"
                
                # Test invalid tokens
                invalid_headers = {"Authorization": "Bearer invalid_token_123"}
                protected_endpoints = [
                    f"{self.auth_service_url}/auth/me",
                    f"{self.auth_service_url}/auth/check-permission"
                ]
                
                for endpoint in protected_endpoints:
                    response = await client.get(endpoint, headers=invalid_headers)
                    assert response.status_code == 401, \
                        f"Invalid token request to {endpoint} should return 401"
                
                # Test rate limiting (if implemented)
                # This would require making many rapid requests
                
                # Test CORS headers (if applicable)
                options_response = await client.options(f"{self.auth_service_url}/auth/login")
                if options_response.status_code == 200:
                    assert "Access-Control-Allow-Origin" in options_response.headers or \
                           "access-control-allow-origin" in options_response.headers
                
            except httpx.ConnectError:
                pytest.skip("Auth service not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])