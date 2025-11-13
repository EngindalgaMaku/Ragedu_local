#!/usr/bin/env python3
"""
Check all users in database
"""
import requests

AUTH_URL = "http://localhost:8006"

# Login as admin first
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
    print(f"✓ Admin logged in\n")
    
    # Get all users
    response = requests.get(
        f"{AUTH_URL}/admin/users",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    
    if response.status_code == 200:
        users = response.json()
        print(f"📋 KULLANICILAR ({len(users)} adet):\n")
        print("-" * 80)
        
        for user in users:
            print(f"ID: {user.get('id')}")
            print(f"  Kullanıcı Adı: {user.get('username')}")
            print(f"  Email: {user.get('email')}")
            print(f"  İsim: {user.get('first_name')} {user.get('last_name')}")
            print(f"  Rol: {user.get('role_name')}")
            print(f"  Aktif: {user.get('is_active')}")
            print("-" * 80)
        
        print("\n💡 Giriş yapmak için bu kullanıcı adlarından birini kullanın:")
        for user in users:
            print(f"  - {user.get('username')}")
    else:
        print(f"Hata: {response.text}")
else:
    print(f"Admin girişi başarısız: {response.text}")
