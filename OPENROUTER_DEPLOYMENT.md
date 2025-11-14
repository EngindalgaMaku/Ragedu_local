# 🚀 OpenRouter Docker Deployment Guide

## 📋 Gerekli Servis Restart'ları

OpenRouter integration'ı için şu servisler yeniden başlatılmalı:

### ✅ ZORUNLU Restart'lar

1. **model-inference-service** - OpenRouter API desteği eklendi
2. **frontend** - UI'da OpenRouter provider seçeneği eklendi
3. **api-gateway** - Model selector UI değişiklikleri için

### ⚠️ İsteğe Bağlı Restart'lar

- Diğer servisler değişmedi, restart gerekmez

---

## 🔧 Docker Deployment Adımları

### 1. Environment Variable Ayarı

`.env` dosyanızı oluşturun (.env.example'dan kopyalayın):

```bash
# .env.example'dan kopyalayın
cp .env.example .env

# Sonra .env dosyasını düzenleyin:
GROQ_API_KEY=your_groq_key
HUGGINGFACE_API_KEY=your_huggingface_key  # Bu eksikti!
OPENROUTER_API_KEY=your_openrouter_key    # Yeni eklendi!
DOCSTRANGE_API_KEY=your_docstrange_key
```

**⚠️ Önemli:** HuggingFace key daha önce çalışıyordu ama .env.example'da eksikti. Şimdi eklendi.

### 2. Servisleri Durdur

```bash
# Sadece değişen servisleri durdurun
docker-compose stop model-inference-service frontend api-gateway
```

### 3. İmajları Yeniden Build Et

```bash
# Model Inference Service'i rebuild et
docker-compose build model-inference-service

# Frontend'i rebuild et
docker-compose build frontend

# API Gateway'i rebuild et
docker-compose build api-gateway
```

### 4. Servisleri Başlat

```bash
# Servisleri sırayla başlat
docker-compose up -d model-inference-service
docker-compose up -d api-gateway
docker-compose up -d frontend
```

### 5. Durumu Kontrol Et

```bash
# Servislerin durumunu kontrol et
docker-compose ps

# Logları kontrol et
docker-compose logs model-inference-service -f
docker-compose logs frontend -f
```

---

## 🎯 Hızlı Restart Komutu

Tek komutla tüm gerekli servisleri restart edin:

```bash
# Hızlı restart - Tek komut
docker-compose down model-inference-service frontend api-gateway && \
docker-compose build model-inference-service frontend api-gateway && \
docker-compose up -d model-inference-service api-gateway frontend
```

---

## ✅ Test Etme

### 1. Model Inference Service Test

```bash
# Health check
curl http://localhost:8002/health

# Beklenen çıktı:
# {
#   "status": "ok",
#   "openrouter_available": true  <-- Bu true olmalı
# }
```

### 2. Available Models Test

```bash
# OpenRouter modellerini listele
curl http://localhost:8002/models/available

# Beklenen çıktı:
# {
#   "openrouter": [
#     "meta-llama/llama-3.1-8b-instruct:free",
#     "mistralai/mistral-7b-instruct:free",
#     ...
#   ]
# }
```

### 3. Frontend Test

1. Browser'da `http://your-server:3000` açın
2. Model seçimi sayfasına gidin
3. **"🚀 OpenRouter"** provider'ını görebilmelisiniz
4. OpenRouter seçince 5 free model görünmeli

---

## 🔍 Troubleshooting

### Problem: OpenRouter modelleri görünmüyor

**Çözüm:**

```bash
# Environment variable'ı kontrol et
docker exec -it model-inference-service env | grep OPENROUTER_API_KEY

# Eğer boşsa, .env dosyasını kontrol edin ve restart yapın
```

### Problem: API key hatası

**Çözüm:**

```bash
# API key'in doğru olup olmadığını test et
curl -H "Authorization: Bearer YOUR_KEY" \
     https://openrouter.ai/api/v1/models

# Başarılıysa modellistesi dönmeli
```

### Problem: Frontend'de provider görünmüyor

**Çözüm:**

```bash
# Frontend'i force rebuild et
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

---

## 📊 Production Ayarları

### Server IP Ayarları

Docker compose'da server IP'nizi güncelleyin:

```yaml
# docker-compose.yml'de
environment:
  - NEXT_PUBLIC_API_URL=http://YOUR_SERVER_IP:8000
  - CORS_ORIGINS=http://YOUR_SERVER_IP:3000,http://YOUR_SERVER_IP:8000
```

### Güvenlik

- OpenRouter API key'ini güvenli saklayın
- .env dosyasını Git'e commit etmeyin
- Production'da free modeller kullanarak maliyet kontrolü sağlayın

---

## 🎉 Başarılı Deployment Kontrolü

Deployment başarılı olduğunda:

✅ Model Inference Service `/health` endpoint'inde `openrouter_available: true`  
✅ `/models/available` endpoint'inde OpenRouter modellar listeleniyor  
✅ Frontend UI'da OpenRouter provider görünüyor  
✅ Free modeller seçilebiliyor ve çalışıyor

## 💰 Maliyet Kontrolü

- **Free modeller** kullanarak maliyet sıfır
- API key sadece rate limiting için gerekli
- Premium modeller kasıtlı olarak konfigürasyona dahil edilmedi
