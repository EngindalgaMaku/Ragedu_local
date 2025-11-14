#!/usr/bin/env python3
"""
Test panel üzerinden kullanıcı güncelleme
"""
import requests

AUTH_URL = "http://localhost:8006"

# 1. Admin login
print("1. Admin login...")
response = requests.post(
    f"{AUTH_URL}/auth/login",
    json={"username": "admin", "password": "admin123"}
)

if response.status_code != 200:
    print(f"❌ Login failed: {response.text}")
    exit(1)

token = response.json().get('access_token')
print(f"✓ Token: {token[:30]}...\n")

# 2. Get ogretmen user
print("2. Getting ogretmen user...")
response = requests.get(
    f"{AUTH_URL}/admin/users",
    headers={"Authorization": f"Bearer {token}"}
)

users = response.json()
ogretmen = next((u for u in users if u.get('username') == 'ogretmen'), None)

if not ogretmen:
    print("❌ ogretmen user not found!")
    exit(1)

print(f"✓ User ID: {ogretmen.get('id')}\n")

# 3. Test UPDATE user (without password)
print("3. Testing UPDATE /admin/users/{id} (without password)...")
response = requests.put(
    f"{AUTH_URL}/admin/users/{ogretmen.get('id')}",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={
        "id": ogretmen.get('id'),
        "username": "ogretmen",
        "email": "teacher@rag-assistant.local",
        "first_name": "Ogretmen",
        "last_name": "1",
        "role_name": "teacher",
        "is_active": True
    }
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("✓ User updated\n")
else:
    print(f"❌ Failed: {response.text}\n")

# 4. Test CHANGE PASSWORD
print("4. Testing PATCH /admin/users/{id}/password...")
response = requests.patch(
    f"{AUTH_URL}/admin/users/{ogretmen.get('id')}/password",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={"new_password": "ogretmen123"}
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("✓ Password changed\n")
else:
    print(f"❌ Failed: {response.text}\n")
    exit(1)

# 5. Test login with new password
print("5. Testing login with ogretmen / ogretmen123...")
response = requests.post(
    f"{AUTH_URL}/auth/login",
    json={"username": "ogretmen", "password": "ogretmen123"}
)

if response.status_code == 200:
    print("✅ LOGIN SUCCESSFUL!")
    print("\n" + "="*60)
    print("Backend is working correctly!")
    print("Password: ogretmen123")
    print("="*60)
else:
    print(f"❌ Login failed: {response.text}")
