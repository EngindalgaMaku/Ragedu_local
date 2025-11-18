# 🔧 Docker Connection Hatası - Çözüm Rehberi

## 🚨 Mevcut Sorunlar

1. **Docker Desktop çalışmıyor** - `ERR_CONNECTION_REFUSED` hatası
2. **OPENROUTER_API_KEY environment variable eksik**

## ✅ Çözüm Adımları

### 1. Docker Desktop'ı Başlat

**Windows:**

```bash
# Docker Desktop uygulamasını başlat
# Başlat menüsünden "Docker Desktop" arayın ve çalıştırın
# VEYA PowerShell'den:
& "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

**Docker Desktop başladığını kontrol edin:**

```bash
# Bu komut çalışmalı (hata vermemeli)
docker ps

# Eğer hala hata veriyorsa Docker Desktop'ı tamamen yeniden başlatın
```

### 2. .env Dosyası Oluştur

```bash
# .env.example'dan kopyala
cd rag3_for_local
cp .env.example .env

# .env dosyasını düzenleyin ve şu değerleri ekleyin:
# GROQ_API_KEY=your_groq_key_here
# HUGGINGFACE_API_KEY=your_huggingface_key_here
# OPENROUTER_API_KEY=your_openrouter_key_here
# DOCSTRANGE_API_KEY=your_docstrange_key_here
```

### 3. Docker Compose Başlat

Docker Desktop çalıştıktan ve .env dosyası oluşturulduktan sonra:

```bash
# Eski containerları temizle
docker-compose down

# Tüm servisleri başlat
docker-compose up -d

# Durumu kontrol et
docker-compose ps
```

## 🔍 Debug Komutları

### Docker Desktop Durumu Kontrol

```bash
# Docker daemon çalışıyor mu?
docker info

# Container durumları
docker ps -a

# Logları kontrol et
docker-compose logs api-gateway --tail=20
docker-compose logs model-inference-service --tail=20
```

### Port Kontrolü

```bash
# API Gateway (8000)
curl http://localhost:8000/health

# Model Inference (8002)
curl http://localhost:8002/health

# Auth Service (8006)
curl http://localhost:8006/health
```

## 🚀 Tam Restart Prosedürü

Eğer sorunlar devam ederse, full restart yapın:

```bash
# 1. Tüm containerları durdur
docker-compose down

# 2. Unused volumes ve networks temizle
docker system prune -f

# 3. .env dosyasını kontrol et
cat .env | grep -E "(GROQ|HUGGINGFACE|OPENROUTER)_API_KEY"

# 4. Servisleri yeniden build et ve başlat
docker-compose up --build -d

# 5. Durumu kontrol et
docker-compose ps
curl http://localhost:8000/health
```

## ⚡ Hızlı Test

Docker Desktop çalıştığında bu komut başarılı olmalı:

```bash
# Bu komut container listesini göstermeli (hata vermemeli)
docker ps

# Bu komut servis durumlarını göstermeli
docker-compose ps

# Frontend erişimi test et
curl http://localhost:3000 -I
```

## 🔧 Common Issues

### Issue 1: Docker Desktop Başlamıyor

**Çözüm:**

- Windows Services'den Docker Desktop Service'i restart edin
- Docker Desktop uygulamasını yönetici olarak çalıştırın
- Bilgisayarı restart edin

### Issue 2: Port Çakışması

**Çözüm:**

```bash
# Hangi process port kullanıyor kontrol et
netstat -ano | findstr :8000
netstat -ano | findstr :8002
netstat -ano | findstr :8006

# Process'i sonlandır (gerekirse)
taskkill /PID <process_id> /F
```

### Issue 3: .env Değişkenleri Yüklenmemiş

**Çözüm:**

```bash
# .env dosyası mevcut mu?
ls -la .env

# İçeriği kontrol et
cat .env

# Environment variable'lar yüklenmiş mi?
docker-compose config
```

## ✅ Başarılı Deployment Kontrolü

Tüm adımlar tamamlandığında:

- ✅ `docker ps` komutu container listesini gösterir
- ✅ `curl http://localhost:8000/health` başarılı response döner
- ✅ `curl http://localhost:8002/health` içinde `openrouter_available: true` olur
- ✅ Frontend `http://localhost:3000`'de açılır
- ✅ OpenRouter provider seçeneği UI'da görünür
