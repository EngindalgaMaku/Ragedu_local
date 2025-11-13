# Konfigürasyon Rehberi

Bu doküman, projedeki tüm hardcoded değerlerin merkezi config dosyalarına nasıl taşındığını ve Google Cloud Run için nasıl hazırlandığını açıklar.

## 📋 İçindekiler

1. [Environment Variables](#environment-variables)
2. [Port Yapılandırması](#port-yapılandırması)
3. [Service URL'leri](#service-urlleri)
4. [Docker Compose Yapılandırması](#docker-compose-yapılandırması)
5. [Google Cloud Run Hazırlığı](#google-cloud-run-hazırlığı)

## 🔧 Environment Variables

Tüm hardcoded değerler environment variable'lara taşınmıştır. `.env.example` dosyasını `.env` olarak kopyalayıp değerleri doldurun.

### Ana Servis Portları

```bash
API_GATEWAY_PORT=8000
AUTH_SERVICE_PORT=8006
FRONTEND_PORT=3000
```

### Mikroservis Portları

```bash
DOCUMENT_PROCESSOR_PORT=8003
MODEL_INFERENCE_PORT=8002
CHROMADB_PORT=8004
MARKER_API_PORT=8090
```

### Service Host'ları (Docker içi iletişim)

```bash
API_GATEWAY_HOST=api-gateway
AUTH_SERVICE_HOST=auth-service
DOCUMENT_PROCESSOR_HOST=document-processing-service
MODEL_INFERENCE_HOST=model-inference-service
CHROMADB_HOST=chromadb-service
MARKER_API_HOST=marker-api
```

## 🌐 Port Yapılandırması

### Backend (`config/ports.py`)

Tüm portlar environment variable'lardan alınır:

```python
API_GATEWAY_PORT = int(os.getenv("API_GATEWAY_PORT", 8000))
AUTH_SERVICE_PORT = int(os.getenv("AUTH_SERVICE_PORT", 8006))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 3000))
```

### Frontend (`frontend/config/ports.ts`)

Tüm portlar environment variable'lardan alınır:

```typescript
export const PORTS = {
  API_GATEWAY: parseInt(process.env.NEXT_PUBLIC_API_GATEWAY_PORT || process.env.API_GATEWAY_PORT || "8000"),
  AUTH_SERVICE: parseInt(process.env.NEXT_PUBLIC_AUTH_SERVICE_PORT || process.env.AUTH_SERVICE_PORT || "8006"),
  FRONTEND: parseInt(process.env.NEXT_PUBLIC_FRONTEND_PORT || process.env.PORT || "3000"),
  // ...
}
```

## 🔗 Service URL'leri

### Backend (`src/api/main.py`)

Tüm service URL'leri environment variable'lardan oluşturulur:

```python
DOCUMENT_PROCESSOR_HOST = os.getenv('DOCUMENT_PROCESSOR_HOST', 'document-processing-service')
DOCUMENT_PROCESSOR_PORT = int(os.getenv('DOCUMENT_PROCESSOR_PORT', '8080'))
DOCUMENT_PROCESSOR_URL = os.getenv('DOCUMENT_PROCESSOR_URL', f'http://{DOCUMENT_PROCESSOR_HOST}:{DOCUMENT_PROCESSOR_PORT}')
```

### Frontend (`frontend/next.config.js`)

Next.js rewrites environment variable'lardan alınır:

```javascript
const apiGatewayHost = process.env.API_GATEWAY_HOST || (isDocker ? "api-gateway" : "localhost");
const apiGatewayPort = process.env.API_GATEWAY_PORT || process.env.API_GATEWAY_INTERNAL_PORT || "8000";
const apiUrl = `http://${apiGatewayHost}:${apiGatewayPort}`;
```

## 🐳 Docker Compose Yapılandırması

Tüm hardcoded değerler environment variable'lara taşınmıştır:

```yaml
api-gateway:
  ports:
    - "${API_GATEWAY_PORT:-8000}:${API_GATEWAY_PORT:-8000}"
  environment:
    - PORT=${API_GATEWAY_PORT:-8000}
    - API_GATEWAY_PORT=${API_GATEWAY_PORT:-8000}
    - DOCUMENT_PROCESSOR_HOST=${DOCUMENT_PROCESSOR_HOST:-document-processing-service}
    - DOCUMENT_PROCESSOR_PORT=${DOCUMENT_PROCESSOR_PORT:-8080}
    # ...
```

## ☁️ Google Cloud Run Hazırlığı

### PORT Environment Variable

Google Cloud Run otomatik olarak `PORT` environment variable'ını ayarlar. Tüm servisler bunu destekler:

**Backend (`src/api/main.py`):**
```python
port = int(os.environ.get("PORT", os.environ.get("API_GATEWAY_PORT", default_port)))
```

**Auth Service (`services/auth_service/main.py`):**
```python
PORT = int(os.getenv("PORT", AUTH_SERVICE_PORT))
```

**Frontend:**
- Next.js otomatik olarak `PORT` environment variable'ını kullanır

### HOST Environment Variable

Google Cloud Run için `HOST=0.0.0.0` ayarlanmalıdır (varsayılan olarak ayarlanmıştır).

### CORS Configuration

CORS origins environment variable'dan alınır:

```bash
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## 📝 Önemli Notlar

1. **Hardcoded Değer Yok**: Artık hiçbir yerde hardcoded host veya port yok
2. **Environment Variable Önceliği**: Tüm değerler environment variable'lardan alınır
3. **Fallback Değerler**: Her değer için mantıklı fallback'ler var
4. **Google Cloud Run Uyumlu**: Tüm servisler `PORT` environment variable'ını destekler
5. **Docker Compose Uyumlu**: Tüm değerler environment variable'lardan alınır

## 🚀 Kullanım

### Local Development

```bash
# .env dosyası oluşturun
cp .env.example .env

# Değerleri düzenleyin
# Docker Compose otomatik olarak .env dosyasını okur
docker compose up
```

### Google Cloud Run

Environment variable'ları Cloud Run'da ayarlayın:

```bash
gcloud run deploy api-gateway \
  --set-env-vars API_GATEWAY_PORT=8000,PORT=8000 \
  --port 8000
```

## ✅ Kontrol Listesi

- [x] Tüm hardcoded port'lar environment variable'lara taşındı
- [x] Tüm hardcoded host'lar environment variable'lara taşındı
- [x] Google Cloud Run için PORT desteği eklendi
- [x] Docker Compose environment variable'ları kullanıyor
- [x] Backend config merkezi hale getirildi
- [x] Frontend config merkezi hale getirildi
- [x] CORS configuration environment variable'dan alınıyor
- [x] .env.example dosyası oluşturuldu

