# RAG Eğitim Asistanı - Authentication Mikroservisi

Kapsamlı kimlik doğrulama ve yetkilendirme mikroservisi. JWT token yönetimi, rol tabanlı erişim kontrolü ve session yönetimi ile production-ready güvenlik özellikleri sunar.

## 🚀 Özellikler

### Kimlik Doğrulama

- **JWT Authentication**: Access ve refresh token ile güvenli kimlik doğrulama
- **Session Management**: Kapsamlı session yaşam döngüsü yönetimi
- **Password Security**: Güçlü şifre hashleme ve doğrulama
- **Rate Limiting**: Login denemeleri için hız sınırlandırması
- **Token Rotation**: Güvenlik için otomatik token yenileme

### Yetkilendirme

- **Role-Based Access Control (RBAC)**: Esnek izin sistemi
- **Granular Permissions**: Kaynak ve aksiyon tabanlı izinler
- **Default Roles**: Admin, Teacher, Student rolleri
- **Permission Inheritance**: Rol bazlı izin kalıtımı

### Kullanıcı Yönetimi

- **CRUD Operations**: Tam kullanıcı yaşam döngüsü yönetimi
- **User Activation/Deactivation**: Kullanıcı durumu kontrolü
- **Password Management**: Şifre değiştirme ve sıfırlama
- **Batch Operations**: Toplu kullanıcı işlemleri

### Güvenlik

- **Security Headers**: Kapsamlı güvenlik başlıkları
- **CORS Support**: Yapılandırılabilir CORS desteği
- **Input Validation**: Pydantic ile güçlü veri doğrulama
- **SQL Injection Protection**: Parametreli sorgularla korunma
- **Failed Login Tracking**: Başarısız giriş takibi

## 📁 Proje Yapısı

```
services/auth_service/
├── main.py                     # FastAPI uygulaması
├── requirements.txt            # Python bağımlılıkları
├── Dockerfile                  # Docker konteyner yapılandırması
├── test_auth_service.py        # Kapsamlı testler
├── validate_integration.py     # Entegrasyon doğrulama
├── README.md                   # Bu dosya
├── auth/                       # Kimlik doğrulama modülü
│   ├── __init__.py
│   ├── auth_manager.py         # JWT ve session yönetimi
│   ├── middleware.py           # Authentication middleware
│   ├── dependencies.py         # FastAPI dependencies
│   └── schemas.py              # Pydantic modelleri
└── api/                        # API endpoint'leri
    ├── __init__.py
    ├── auth.py                 # Kimlik doğrulama endpoint'leri
    ├── users.py                # Kullanıcı yönetimi
    └── roles.py                # Rol yönetimi
```

## 🔧 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.11+
- SQLite (veritabanı)
- Docker (opsiyonel)

### Lokal Kurulum

1. **Bağımlılıkları yükleyin:**

```bash
cd services/auth_service
pip install -r requirements.txt
```

2. **Uygulamayı çalıştırın:**

```bash
python main.py
```

3. **API dokümantasyonuna erişin:**

- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

### Docker ile Çalıştırma

1. **Docker image'ı oluşturun:**

```bash
docker build -t rag-auth-service .
```

2. **Konteyner'ı çalıştırın:**

```bash
docker run -p 8002:8002 -v ./data:/app/data rag-auth-service
```

## 🔑 API Endpoint'leri

### Kimlik Doğrulama (`/auth`)

| Method | Endpoint                      | Açıklama                   |
| ------ | ----------------------------- | -------------------------- |
| POST   | `/auth/login`                 | Kullanıcı girişi           |
| POST   | `/auth/logout`                | Çıkış işlemi               |
| POST   | `/auth/refresh`               | Token yenileme             |
| GET    | `/auth/me`                    | Mevcut kullanıcı bilgileri |
| PUT    | `/auth/change-password`       | Şifre değiştirme           |
| DELETE | `/auth/sessions/{session_id}` | Session silme              |
| POST   | `/auth/check-permission`      | İzin kontrolü              |
| GET    | `/auth/health`                | Sağlık kontrolü            |

### Kullanıcı Yönetimi (`/users`)

| Method | Endpoint                          | Açıklama                | Yetki         |
| ------ | --------------------------------- | ----------------------- | ------------- |
| POST   | `/users`                          | Yeni kullanıcı oluştur  | Admin         |
| GET    | `/users`                          | Kullanıcıları listele   | Admin/Teacher |
| GET    | `/users/{user_id}`                | Kullanıcı detayı        | Admin/Self    |
| PUT    | `/users/{user_id}`                | Kullanıcı güncelle      | Admin/Self    |
| DELETE | `/users/{user_id}`                | Kullanıcı sil           | Admin         |
| POST   | `/users/{user_id}/activate`       | Kullanıcı aktifleştir   | Admin         |
| POST   | `/users/{user_id}/deactivate`     | Kullanıcı deaktifleştir | Admin         |
| POST   | `/users/{user_id}/reset-password` | Şifre sıfırla           | Admin         |

### Rol Yönetimi (`/roles`)

| Method | Endpoint                       | Açıklama                  | Yetki         |
| ------ | ------------------------------ | ------------------------- | ------------- |
| GET    | `/roles`                       | Rolleri listele           | Admin/Teacher |
| GET    | `/roles/{role_id}`             | Rol detayı                | Admin/Teacher |
| POST   | `/roles`                       | Yeni rol oluştur          | Admin         |
| PUT    | `/roles/{role_id}`             | Rol güncelle              | Admin         |
| DELETE | `/roles/{role_id}`             | Rol sil                   | Admin         |
| GET    | `/roles/{role_id}/users`       | Role atanmış kullanıcılar | Admin         |
| POST   | `/roles/{role_id}/permissions` | Role izin ekle            | Admin         |
| DELETE | `/roles/{role_id}/permissions` | Rolden izin çıkar         | Admin         |

## 🔐 Varsayılan Roller ve İzinler

### Admin

```json
{
  "users": ["create", "read", "update", "delete"],
  "roles": ["create", "read", "update", "delete"],
  "sessions": ["create", "read", "update", "delete"],
  "documents": ["create", "read", "update", "delete"],
  "system": ["admin", "configure"]
}
```

### Teacher

```json
{
  "users": ["read"],
  "sessions": ["create", "read", "update", "delete"],
  "documents": ["create", "read", "update", "delete"],
  "students": ["read"]
}
```

### Student

```json
{
  "sessions": ["read"],
  "documents": ["read"]
}
```

## 🔧 Yapılandırma

Çevre değişkenleri ile yapılandırma:

```bash
# Server yapılandırması
HOST=0.0.0.0
PORT=8002
DEBUG=false

# JWT yapılandırması
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Veritabanı
DATABASE_PATH=data/rag_assistant.db

# CORS
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
CORS_CREDENTIALS=true

# Rate Limiting
RATE_LIMIT_RPM=60
RATE_LIMIT_BURST=10

# Güvenlik
REQUIRE_AUTH=true
```

## 🧪 Test Etme

### Unit Testler

```bash
# Tüm testleri çalıştır
python -m pytest test_auth_service.py -v

# Belirli test sınıfını çalıştır
python -m pytest test_auth_service.py::TestAuthentication -v

# Coverage ile çalıştır
python -m pytest test_auth_service.py --cov=auth --cov=api
```

### Entegrasyon Testleri

```bash
# Entegrasyon doğrulaması
python validate_integration.py
```

### Manuel Test

```bash
# Sağlık kontrolü
curl http://localhost:8002/health

# Login
curl -X POST "http://localhost:8002/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin123"}'

# Kullanıcı bilgisi
curl -X GET "http://localhost:8002/auth/me" \
     -H "Authorization: Bearer YOUR_TOKEN"
```

## 📊 İzleme ve Loglama

### Loglar

- Uygulama logları: stdout/stderr
- Log seviyesi: INFO (production), DEBUG (development)
- Structured logging desteği

### Metrikler

- Aktif session sayısı
- Kullanıcı istatistikleri
- Authentication başarı/başarısızlık oranları
- API endpoint kullanım metrikleri

### Sağlık Kontrolü

```json
GET /health
{
  "status": "healthy",
  "service": "RAG Education Assistant - Auth Service",
  "version": "1.0.0",
  "database": "connected",
  "active_sessions": 5,
  "environment": "production"
}
```

## 🔒 Güvenlik En İyi Uygulamaları

### Üretim Ortamı İçin

1. **Güçlü JWT Secret Key kullanın**
2. **HTTPS zorunlu hale getirin**
3. **Rate limiting'i etkinleştirin**
4. **Database şifrelemesi ekleyin**
5. **Security headers'ı doğrulayın**
6. **Düzenli güvenlik güncellemeleri yapın**

### İzin Yönetimi

1. **En az yetki prensibini uygulayın**
2. **Rol tabanlı erişimi doğru yapılandırın**
3. **Düzenli izin denetimi yapın**
4. **Session timeout'ları ayarlayın**

## 🚀 Deployment

### Docker Compose Örneği

```yaml
version: "3.8"
services:
  auth-service:
    build: .
    ports:
      - "8002:8002"
    environment:
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - DATABASE_PATH=/app/data/rag_assistant.db
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import requests; requests.get('http://localhost:8002/health')",
        ]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auth-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: auth-service
  template:
    metadata:
      labels:
        app: auth-service
    spec:
      containers:
        - name: auth-service
          image: rag-auth-service:latest
          ports:
            - containerPort: 8002
          env:
            - name: JWT_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: auth-secrets
                  key: jwt-secret-key
            - name: DATABASE_PATH
              value: "/app/data/rag_assistant.db"
          volumeMounts:
            - name: data-volume
              mountPath: /app/data
      volumes:
        - name: data-volume
          persistentVolumeClaim:
            claimName: auth-data-pvc
```

## 🤝 Entegrasyon

### Diğer Mikroservisler ile Entegrasyon

1. **API Gateway** üzerinden routing
2. **Shared JWT secret** ile token doğrulama
3. **Service-to-service** authentication
4. **Event-driven** kullanıcı değişiklikleri

### Frontend Entegrasyonu

```javascript
// Login örneği
const login = async (username, password) => {
  const response = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  const data = await response.json();
  if (data.access_token) {
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
  }
  return data;
};

// Authenticated request örneği
const makeAuthenticatedRequest = async (url, options = {}) => {
  const token = localStorage.getItem("access_token");
  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      Authorization: `Bearer ${token}`,
    },
  });
};
```

## 📚 API Dokümantasyonu

Detaylı API dokümantasyonu için:

- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc
- **OpenAPI JSON**: http://localhost:8002/openapi.json

## 🐛 Troubleshooting

### Yaygın Sorunlar

**1. Database bağlantı hatası**

```bash
# Veritabanı dosyası izinlerini kontrol edin
ls -la data/rag_assistant.db
chmod 664 data/rag_assistant.db
```

**2. JWT token hatası**

```bash
# Secret key'i kontrol edin
echo $JWT_SECRET_KEY
# Yeterince güçlü olduğundan emin olun (32+ karakter)
```

**3. Rate limiting hatası**

```bash
# Rate limiting ayarlarını kontrol edin
curl -H "X-Forwarded-For: 192.168.1.1" http://localhost:8002/auth/login
```

**4. CORS hatası**

```bash
# CORS ayarlarını kontrol edin
export CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"
```

## 🔄 Güncelleme ve Maintenance

### Düzenli Bakım

- **Session cleanup**: Expired sessionları temizleme
- **Log rotation**: Log dosyası yönetimi
- **Security updates**: Güvenlik güncellemeleri
- **Performance monitoring**: Performans izleme

### Backup

```bash
# Veritabanı yedekleme
sqlite3 data/rag_assistant.db ".backup data/rag_assistant_backup_$(date +%Y%m%d).db"

# Restore
cp data/rag_assistant_backup_20241101.db data/rag_assistant.db
```

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👥 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📞 Destek

Herhangi bir sorun veya soru için lütfen issue açın veya iletişime geçin.

---

**RAG Eğitim Asistanı Auth Service v1.0.0** - Production-ready authentication microservice
