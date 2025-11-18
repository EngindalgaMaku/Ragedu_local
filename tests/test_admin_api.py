#!/usr/bin/env python3
"""
Real API Test Script for Admin Endpoints
"""
import requests
import json

AUTH_URL = "http://localhost:8006"

def test_login():
    """First login to get token"""
    print("=" * 50)
    print("1. Testing Login...")
    response = requests.post(
        f"{AUTH_URL}/auth/login",
        json={
            "username": "admin",
            "password": "admin123"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        token = data.get('access_token')
        print(f"✓ Login successful! Token: {token[:20]}...")
        return token
    else:
        print(f"✗ Login failed: {response.text}")
        return None

def test_get_users(token):
    """Test GET /admin/users"""
    print("=" * 50)
    print("2. Testing GET /admin/users...")
    response = requests.get(
        f"{AUTH_URL}/admin/users",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        users = response.json()
        print(f"✓ Got {len(users)} users")
        for user in users[:2]:
            print(f"  - {user.get('username')} ({user.get('email')})")
        return users
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_create_user(token):
    """Test POST /admin/users"""
    print("=" * 50)
    print("3. Testing POST /admin/users (Create User)...")
    
    import time
    username = f"test_{int(time.time())}"
    
    response = requests.post(
        f"{AUTH_URL}/admin/users",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "username": username,
            "email": f"{username}@example.com",
            "password": "Test123!",
            "first_name": "Test",
            "last_name": "User",
            "role_name": "user",
            "is_active": True
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        user = response.json()
        print(f"✓ User created: {user.get('username')} (ID: {user.get('id')})")
        return user
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_update_user(token, user_id):
    """Test PUT /admin/users/{user_id}"""
    print("=" * 50)
    print(f"4. Testing PUT /admin/users/{user_id} (Update User)...")
    
    response = requests.put(
        f"{AUTH_URL}/admin/users/{user_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "first_name": "Updated",
            "last_name": "Name"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        user = response.json()
        print(f"✓ User updated: {user.get('first_name')} {user.get('last_name')}")
        return user
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_delete_user(token, user_id):
    """Test DELETE /admin/users/{user_id}"""
    print("=" * 50)
    print(f"5. Testing DELETE /admin/users/{user_id} (Delete User)...")
    
    response = requests.delete(
        f"{AUTH_URL}/admin/users/{user_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"✓ User deleted successfully")
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False

if __name__ == "__main__":
    print("\n🔍 ADMIN API TEST SUITE")
    print("Testing all admin endpoints...\n")
    
    # 1. Login
    token = test_login()
    if not token:
        print("\n❌ Cannot continue without token!")
        exit(1)
    
    # 2. Get users
    users = test_get_users(token)
    
    # 3. Create user
    new_user = test_create_user(token)
    
    if new_user:
        user_id = new_user.get('id')
        
        # 4. Update user
        test_update_user(token, user_id)
        
        # 5. Delete user
        test_delete_user(token, user_id)
    
    print("\n" + "=" * 50)
    print("✅ TEST SUITE COMPLETED!")
    print("=" * 50)
