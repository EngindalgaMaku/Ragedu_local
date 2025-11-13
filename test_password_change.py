#!/usr/bin/env python3
"""
Test password change endpoint
"""
import requests

AUTH_URL = "http://localhost:8006"

# Login as admin
print("1. Admin olarak giriş yapılıyor...")
response = requests.post(
    f"{AUTH_URL}/auth/login",
    json={"username": "admin", "password": "admin123"}
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
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
)

users = response.json()
ogretmen = next((u for u in users if u.get('username') == 'ogretmen'), None)

if not ogretmen:
    print("❌ 'ogretmen' kullanıcısı bulunamadı!")
    exit(1)

print(f"✓ Kullanıcı bulundu (ID: {ogretmen.get('id')})\n")

# Change password using new endpoint
print("3. Şifre değiştiriliyor (ogretmen123 → yeni_sifre_456)...")
response = requests.patch(
    f"{AUTH_URL}/admin/users/{ogretmen.get('id')}/password",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={"new_password": "yeni_sifre_456"}
)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    print("✓ Şifre başarıyla değiştirildi!\n")
    
    # Test with new password
    print("4. Yeni şifre ile giriş test ediliyor...")
    response = requests.post(
        f"{AUTH_URL}/auth/login",
        json={"username": "ogretmen", "password": "yeni_sifre_456"}
    )
    
    if response.status_code == 200:
        print("✅ GİRİŞ BAŞARILI! Yeni şifre çalışıyor!")
        print("\n" + "=" * 60)
        print("  Kullanıcı: ogretmen")
        print("  Yeni Şifre: yeni_sifre_456")
        print("=" * 60)
    else:
        print(f"❌ Giriş başarısız: {response.text}")
else:
    print(f"❌ Şifre değiştirilemedi: {response.text}")
