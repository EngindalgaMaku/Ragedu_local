#!/usr/bin/env python3
"""
Test ogretmen login
"""
import requests

AUTH_URL = "http://localhost:8006"

print("🔐 Testing ogretmen login...")
print("Credentials: ogretmen / ogretmen123\n")

response = requests.post(
    f"{AUTH_URL}/auth/login",
    json={
        "username": "ogretmen",
        "password": "ogretmen123"
    }
)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print("\n✅ GİRİŞ BAŞARILI!")
    print("=" * 60)
    print(f"Token: {data.get('access_token')[:30]}...")
    print(f"Token Type: {data.get('token_type')}")
    print(f"User ID: {data.get('user_id')}")
    print(f"Username: {data.get('username')}")
    print(f"Role: {data.get('role')}")
    print("=" * 60)
else:
    print(f"\n❌ GİRİŞ BAŞARISIZ!")
    print(f"Hata: {response.text}")
