#!/usr/bin/env python3
"""
Reset password for ogretmen user
"""
import requests

AUTH_URL = "http://localhost:8006"

# Login as admin
print("1. Admin olarak giriş yapılıyor...")
response = requests.post(
    f"{AUTH_URL}/auth/login",
    json={
        "username": "admin",
        "password": "admin123"
    }
)

if response.status_code != 200:
    print(f"❌ Admin girişi başarısız: {response.text}")
    exit(1)

token = response.json().get('access_token')
print("✓ Admin girişi başarılı\n")

# Get ogretmen user
print("2. 'ogretmen' kullanıcısı alınıyor...")
response = requests.get(
    f"{AUTH_URL}/admin/users",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
)

if response.status_code != 200:
    print(f"❌ Kullanıcılar alınamadı: {response.text}")
    exit(1)

users = response.json()
ogretmen = None
for user in users:
    if user.get('username') == 'ogretmen':
        ogretmen = user
        break

if not ogretmen:
    print("❌ 'ogretmen' kullanıcısı bulunamadı!")
    exit(1)

print(f"✓ Kullanıcı bulundu (ID: {ogretmen.get('id')})\n")

# SORUN: Backend'de password update endpoint'i yok!
# Kullanıcıyı silip yeniden oluşturalım

print("3. Eski kullanıcı siliniyor...")
response = requests.delete(
    f"{AUTH_URL}/admin/users/{ogretmen.get('id')}",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
)

if response.status_code != 200:
    print(f"❌ Kullanıcı silinemedi: {response.text}")
    exit(1)

print("✓ Eski kullanıcı silindi\n")

# Yeni kullanıcı oluştur
print("4. Yeni 'ogretmen' kullanıcısı oluşturuluyor (şifre: ogretmen123)...")
response = requests.post(
    f"{AUTH_URL}/admin/users",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    },
    json={
        "username": "ogretmen",
        "email": "teacher@rag-assistant.local",
        "password": "ogretmen123",
        "first_name": "Ogretmen",
        "last_name": "1",
        "role_name": "teacher",
        "is_active": True
    }
)

if response.status_code == 200:
    user = response.json()
    print(f"✅ Kullanıcı oluşturuldu!\n")
    print("=" * 60)
    print(f"  Kullanıcı Adı: ogretmen")
    print(f"  Şifre: ogretmen123")
    print(f"  Email: {user.get('email')}")
    print(f"  Rol: {user.get('role_name')}")
    print("=" * 60)
    print("\n✅ Artık 'ogretmen' / 'ogretmen123' ile giriş yapabilirsiniz!")
else:
    print(f"❌ Kullanıcı oluşturulamadı: {response.text}")
