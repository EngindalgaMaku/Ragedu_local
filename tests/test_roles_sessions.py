#!/usr/bin/env python3
"""
Test Script for Roles and Sessions Endpoints
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

def test_get_roles(token):
    """Test GET /admin/roles"""
    print("=" * 50)
    print("2. Testing GET /admin/roles...")
    response = requests.get(
        f"{AUTH_URL}/admin/roles",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        roles = response.json()
        print(f"✓ Got {len(roles)} roles")
        for role in roles:
            print(f"  - {role.get('name')}: {role.get('description')}")
        return roles
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_get_sessions(token):
    """Test GET /admin/sessions"""
    print("=" * 50)
    print("3. Testing GET /admin/sessions...")
    response = requests.get(
        f"{AUTH_URL}/admin/sessions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        sessions = response.json()
        print(f"✓ Got {len(sessions)} sessions")
        for session in sessions[:3]:
            print(f"  - User: {session.get('username')} (ID: {session.get('id')})")
            print(f"    IP: {session.get('ip_address')}")
            print(f"    Active: {session.get('is_active')}")
        return sessions
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_terminate_session(token, session_id):
    """Test DELETE /admin/sessions/{session_id}"""
    print("=" * 50)
    print(f"4. Testing DELETE /admin/sessions/{session_id}...")
    response = requests.delete(
        f"{AUTH_URL}/admin/sessions/{session_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"✓ Session terminated successfully")
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False

def test_admin_stats(token):
    """Test GET /admin/stats"""
    print("=" * 50)
    print("5. Testing GET /admin/stats...")
    response = requests.get(
        f"{AUTH_URL}/admin/stats",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ Stats retrieved:")
        print(f"  - Total Users: {stats.get('totalUsers')}")
        print(f"  - Active Users: {stats.get('activeUsers')}")
        print(f"  - Active Sessions: {stats.get('activeSessions')}")
        print(f"  - Total Roles: {stats.get('totalRoles')}")
        return stats
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_activity_logs(token):
    """Test GET /admin/activity-logs"""
    print("=" * 50)
    print("6. Testing GET /admin/activity-logs...")
    response = requests.get(
        f"{AUTH_URL}/admin/activity-logs?limit=5",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        logs = response.json()
        print(f"✓ Got {len(logs)} activity logs")
        for log in logs[:3]:
            print(f"  - {log.get('type')}: {log.get('message')} {log.get('user')}")
        return logs
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_system_health(token):
    """Test GET /admin/system-health"""
    print("=" * 50)
    print("7. Testing GET /admin/system-health...")
    response = requests.get(
        f"{AUTH_URL}/admin/system-health",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        health = response.json()
        print(f"✓ System health retrieved:")
        print(f"  - Status: {health.get('status')}")
        print(f"  - Uptime: {health.get('uptime')}")
        print(f"  - Disk Usage: {health.get('diskUsage')}")
        print(f"  - Memory Usage: {health.get('memoryUsage')}")
        print(f"  - Services: {health.get('services')}")
        return health
    else:
        print(f"✗ Failed: {response.text}")
        return None

if __name__ == "__main__":
    print("\n🔍 ROLES & SESSIONS API TEST SUITE")
    print("Testing all admin endpoints...\n")
    
    # 1. Login
    token = test_login()
    if not token:
        print("\n❌ Cannot continue without token!")
        exit(1)
    
    # 2. Test Roles
    roles = test_get_roles(token)
    
    # 3. Test Sessions
    sessions = test_get_sessions(token)
    
    # 4. Test session termination (only if we have sessions)
    # Note: We won't terminate the current session
    if sessions and len(sessions) > 1:
        # Find a session that's not the current one
        for session in sessions:
            if not session.get('is_active'):
                test_terminate_session(token, session.get('id'))
                break
    
    # 5. Test Stats
    test_admin_stats(token)
    
    # 6. Test Activity Logs
    test_activity_logs(token)
    
    # 7. Test System Health
    test_system_health(token)
    
    print("\n" + "=" * 50)
    print("✅ ROLES & SESSIONS TEST SUITE COMPLETED!")
    print("=" * 50)
