#!/usr/bin/env python3
"""
Check sessions in database
"""
import requests

AUTH_URL = "http://localhost:8006"

# Login first
response = requests.post(
    f"{AUTH_URL}/auth/login",
    json={
        "username": "admin",
        "password": "admin123"
    }
)

if response.status_code == 200:
    data = response.json()
    token = data.get('access_token')
    print(f"✓ Logged in. Token: {token[:20]}...")
    
    # Now try to get sessions with the token
    response = requests.get(
        f"{AUTH_URL}/admin/sessions?limit=200",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    
    print(f"\nGET /admin/sessions")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        sessions = response.json()
        print(f"Sessions count: {len(sessions)}")
        
        if sessions:
            print("\nFirst few sessions:")
            for session in sessions[:5]:
                print(f"  - ID: {session.get('id')}")
                print(f"    User: {session.get('username')}")
                print(f"    Active: {session.get('is_active')}")
                print(f"    Created: {session.get('created_at')}")
                print()
        else:
            print("\n⚠️  No sessions found in database!")
            print("\nPossible reasons:")
            print("1. user_sessions table is empty")
            print("2. Query is failing silently")
            print("3. get_recent_sessions() is returning empty list")
    else:
        print(f"Error: {response.text}")
else:
    print(f"Login failed: {response.text}")
