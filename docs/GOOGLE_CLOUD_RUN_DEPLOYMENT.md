# Google Cloud Run Deployment Guide

Bu doküman, sistemin Google Cloud Run'a deploy edilmesi için gerekli tüm environment variable'ları ve yapılandırmaları içerir.

## ✅ Hardcoded Değer Kontrolü - TAMAMLANDI

**Tüm sistem Google Cloud Run için hazırlandı!** 

✅ **Hardcoded Docker service adları kaldırıldı**
✅ **Hardcoded localhost adresleri kaldırıldı**  
✅ **Hardcoded Cloud Run URL'leri kaldırıldı**
✅ **Tüm servis URL'leri environment variable'lardan alınıyor**
✅ **Docker ve Cloud Run uyumlu fallback mekanizması eklendi**

### Yapılan Değişiklikler

1. **API Gateway** (`src/api/main.py`):
   - Tüm servis URL'leri environment variable'lardan alınıyor
   - Cloud Run URL formatı (https://xxx.run.app) destekleniyor
   - Docker service name formatı (http://service-name:port) fallback olarak kullanılıyor

2. **APRAG Service**:
   - `topics.py`: MODEL_INFERENCER_URL, CHROMA_SERVICE_URL, DOCUMENT_PROCESSING_URL
   - `personalization.py`: MODEL_INFERENCE_URL
   - `recommendations.py`: MODEL_INFERENCE_URL
   - `aprag_middleware.py`: APRAG_SERVICE_URL

3. **Document Processing Service**:
   - MODEL_INFERENCER_URL, CHROMADB_URL environment variable'lardan alınıyor
   - ChromaDB client Cloud Run URL'lerini destekliyor

4. **Model Inference Service**:
   - OLLAMA_HOST environment variable'dan alınıyor

5. **ChromaDB Service**:
   - CHROMADB_INTERNAL_URL hardcoded Cloud Run URL'si kaldırıldı

6. **Frontend**:
   - `next.config.js`: API Gateway URL environment variable'dan alınıyor
   - `ports.ts`: Cloud Run URL formatı destekleniyor

7. **Vector Store**:
   - `chroma_store.py`: CHROMADB_URL environment variable'dan alınıyor

## 🔄 URL Yapılandırma Mantığı

Sistem, environment variable'ları şu sırayla kontrol eder:

1. **Tam URL** (Cloud Run): Eğer `*_SERVICE_URL` environment variable'ı tam URL içeriyorsa (http:// veya https:// ile başlıyorsa), direkt kullanılır.
2. **Host + Port** (Docker): Eğer `*_SERVICE_HOST` ve `*_SERVICE_PORT` set edilmişse, bunlardan URL oluşturulur.
3. **Default** (Docker fallback): Hiçbiri yoksa, Docker service adları kullanılır (sadece local Docker için).

### Örnek Yapılandırma

**Docker (Local):**
```bash
APRAG_SERVICE_HOST=aprag-service
APRAG_SERVICE_PORT=8007
# Sonuç: http://aprag-service:8007
```

**Google Cloud Run:**
```bash
APRAG_SERVICE_URL=https://aprag-service-xxx.run.app
# Sonuç: https://aprag-service-xxx.run.app
```

## 🔧 Google Cloud Run Environment Variables

### API Gateway

```bash
# Port (Cloud Run otomatik ayarlar, override edilebilir)
PORT=8000
HOST=0.0.0.0

# Service URL'leri - Cloud Run'da tam URL olmalı
PDF_PROCESSOR_URL=https://pdf-processor-xxx.run.app
DOCUMENT_PROCESSOR_URL=https://document-processing-xxx.run.app
MODEL_INFERENCE_URL=https://model-inference-xxx.run.app
CHROMADB_URL=https://chromadb-xxx.run.app
MARKER_API_URL=https://marker-api-xxx.run.app
AUTH_SERVICE_URL=https://auth-service-xxx.run.app
APRAG_SERVICE_URL=https://aprag-service-xxx.run.app

# CORS Origins (Cloud Run domain'leri)
CORS_ORIGINS=https://your-frontend-domain.com,https://api-gateway-xxx.run.app

# Database
DATABASE_PATH=/app/data/rag_assistant.db

# JWT
JWT_SECRET_KEY=your-production-secret-key
```

### Frontend (Next.js)

```bash
# Port (Cloud Run otomatik ayarlar)
PORT=3000

# API Gateway URL (Cloud Run URL)
NEXT_PUBLIC_API_URL=https://api-gateway-xxx.run.app
NEXT_PUBLIC_AUTH_URL=https://auth-service-xxx.run.app

# Docker mode (Cloud Run için false veya unset)
DOCKER_ENV=false

# API Gateway Host (Cloud Run için unset veya Cloud Run URL)
API_GATEWAY_HOST=api-gateway-xxx.run.app
API_GATEWAY_PORT=443  # HTTPS için

# CORS
CORS_ORIGINS=https://your-frontend-domain.com
```

### APRAG Service

```bash
# Port
PORT=8007
HOST=0.0.0.0

# Service URL'leri (Cloud Run'da tam URL)
MODEL_INFERENCER_URL=https://model-inference-xxx.run.app
MODEL_INFERENCE_URL=https://model-inference-xxx.run.app
CHROMA_SERVICE_URL=https://chromadb-xxx.run.app
CHROMADB_URL=https://chromadb-xxx.run.app
DOCUMENT_PROCESSING_URL=https://document-processing-xxx.run.app

# Database
DATABASE_PATH=/app/data/rag_assistant.db
APRAG_DB_PATH=/app/data/rag_assistant.db

# CORS
CORS_ORIGINS=https://your-frontend-domain.com,https://api-gateway-xxx.run.app

# Feature Flags
APRAG_ENABLED=true
APRAG_FEEDBACK_COLLECTION=true
APRAG_PERSONALIZATION=true
APRAG_RECOMMENDATIONS=true
```

### Document Processing Service

```bash
# Port
PORT=8080
HOST=0.0.0.0

# Service URL'leri (Cloud Run'da tam URL)
MODEL_INFERENCER_URL=https://model-inference-xxx.run.app
CHROMADB_URL=https://chromadb-xxx.run.app
CHROMA_SERVICE_URL=https://chromadb-xxx.run.app
```

### Model Inference Service

```bash
# Port
PORT=8002
HOST=0.0.0.0
```

### ChromaDB Service

```bash
# Port
PORT=8004
HOST=0.0.0.0
```

### Auth Service

```bash
# Port
PORT=8006
HOST=0.0.0.0

# Database
DATABASE_PATH=/app/data/rag_assistant.db

# CORS
CORS_ORIGINS=https://your-frontend-domain.com,https://api-gateway-xxx.run.app

# Rate Limiting
RATE_LIMIT_RPM=300
RATE_LIMIT_BURST=50
```

## 🔄 URL Yapılandırma Mantığı

Sistem, environment variable'ları şu sırayla kontrol eder:

1. **Tam URL** (Cloud Run): Eğer `*_SERVICE_URL` environment variable'ı tam URL içeriyorsa (http:// veya https:// ile başlıyorsa), direkt kullanılır.
2. **Host + Port** (Docker): Eğer `*_SERVICE_HOST` ve `*_SERVICE_PORT` set edilmişse, bunlardan URL oluşturulur.
3. **Default** (Docker fallback): Hiçbiri yoksa, Docker service adları kullanılır (sadece local Docker için).

### Örnek Yapılandırma

**Docker (Local):**
```bash
APRAG_SERVICE_HOST=aprag-service
APRAG_SERVICE_PORT=8007
# Sonuç: http://aprag-service:8007
```

**Google Cloud Run:**
```bash
APRAG_SERVICE_URL=https://aprag-service-xxx.run.app
# Sonuç: https://aprag-service-xxx.run.app
```

## 📝 Deployment Checklist

### 1. Environment Variables Ayarlama

Her servis için Cloud Run'da environment variable'ları ayarlayın:

```bash
# API Gateway
gcloud run services update api-gateway \
  --set-env-vars="PDF_PROCESSOR_URL=https://pdf-processor-xxx.run.app,\
DOCUMENT_PROCESSOR_URL=https://document-processing-xxx.run.app,\
MODEL_INFERENCE_URL=https://model-inference-xxx.run.app,\
CHROMADB_URL=https://chromadb-xxx.run.app,\
AUTH_SERVICE_URL=https://auth-service-xxx.run.app,\
APRAG_SERVICE_URL=https://aprag-service-xxx.run.app,\
CORS_ORIGINS=https://your-frontend-domain.com"
```

### 2. CORS Ayarları

Frontend domain'inizi tüm backend servislerin `CORS_ORIGINS` environment variable'ına ekleyin.

### 3. Database Persistence

Cloud Run stateless olduğu için, database'i Cloud SQL veya başka bir persistent storage'a taşımanız gerekebilir. Şu an için SQLite kullanılıyor, bu production için uygun değildir.

### 4. Service Discovery

Cloud Run servisleri birbirlerini bulmak için tam URL'leri kullanmalıdır. Environment variable'larda tam URL'leri set edin.

## ⚠️ Önemli Notlar

1. **HTTPS Zorunlu**: Cloud Run tüm trafiği HTTPS üzerinden yönlendirir. Service URL'leri `https://` ile başlamalıdır.

2. **Port Değişkenliği**: Cloud Run `PORT` environment variable'ını otomatik ayarlar. Kodunuzda `PORT` variable'ını kullanın, hardcoded port kullanmayın.

3. **Internal Communication**: Cloud Run servisleri birbirleriyle iletişim kurarken public URL'leri kullanır. Internal network yoktur.

4. **Database**: SQLite dosya sistemi Cloud Run'da kalıcı değildir. Production için Cloud SQL veya başka bir managed database kullanın.

5. **CORS**: Frontend domain'inizi tüm backend servislerin CORS ayarlarına ekleyin.

## 🔍 Kontrol Komutları

Deployment sonrası kontrol için:

```bash
# API Gateway health check
curl https://api-gateway-xxx.run.app/health

# Service URL'lerini kontrol et
curl https://api-gateway-xxx.run.app/api/models

# CORS kontrolü
curl -H "Origin: https://your-frontend-domain.com" \
     -H "Access-Control-Request-Method: GET" \
     -X OPTIONS https://api-gateway-xxx.run.app/health
```

## 📚 İlgili Dokümanlar

- [ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md) - Tüm environment variable'ların detaylı listesi
- [CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md) - Genel yapılandırma rehberi
- [deployment_instructions.md](./deployment_instructions.md) - Deployment adımları

