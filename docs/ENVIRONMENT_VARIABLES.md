# Environment Variables Rehberi

Bu doküman, projedeki tüm environment variable'ları ve kullanımlarını açıklar.

## 📋 Ana Servis Portları

```bash
API_GATEWAY_PORT=8000          # API Gateway portu (Google Cloud Run için PORT kullanılır)
AUTH_SERVICE_PORT=8006         # Auth Service portu
FRONTEND_PORT=3000             # Frontend portu (Next.js için PORT kullanılır)
```

## 🔧 Mikroservis Portları

```bash
DOCUMENT_PROCESSOR_PORT=8003   # Document Processing Service portu
MODEL_INFERENCE_PORT=8002      # Model Inference Service portu
CHROMADB_PORT=8004             # ChromaDB Service portu
MARKER_API_PORT=8090           # Marker API portu
DOCSTRANGE_PORT=8005           # DocStrange Service portu
```

## 🌐 Service Host'ları (Docker içi iletişim)

```bash
API_GATEWAY_HOST=api-gateway
AUTH_SERVICE_HOST=auth-service
DOCUMENT_PROCESSOR_HOST=document-processing-service
MODEL_INFERENCE_HOST=model-inference-service
CHROMADB_HOST=chromadb-service
MARKER_API_HOST=marker-api
```

## 🔗 Service URL'leri

### External Service URL'leri (Google Cloud Run)

```bash
PDF_PROCESSOR_URL=https://pdf-processor-awe3elsvra-ew.a.run.app
MODEL_INFERENCE_URL=https://model-inferencer-awe3elsvra-ew.a.run.app
```

### Internal Service URL'leri (Docker içi)

Bu URL'ler otomatik olarak HOST ve PORT environment variable'larından oluşturulur:

```bash
DOCUMENT_PROCESSOR_URL=http://document-processing-service:8080
MODEL_INFERENCE_URL=http://model-inference-service:8002
CHROMADB_URL=http://chromadb-service:8000
MARKER_API_URL=http://marker-api:8090
AUTH_SERVICE_URL=http://auth-service:8006
```

## 🔐 CORS Configuration

```bash
# Virgülle ayrılmış origin listesi
CORS_ORIGINS=http://localhost:3000,http://localhost:8000,https://yourdomain.com
```

## 🔑 JWT Configuration

```bash
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

## 🗄️ Database Configuration

```bash
DATABASE_PATH=/app/data/rag_assistant.db
```

## 🎨 Frontend Environment Variables

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_AUTH_URL=http://localhost:8006
NEXT_PUBLIC_AUTH_ENABLED=true
NEXT_PUBLIC_DEMO_MODE=true
```

## 🐳 Docker Configuration

```bash
DOCKER_ENV=true
NODE_ENV=production
```

## ☁️ Google Cloud Run Specific

Google Cloud Run otomatik olarak `PORT` environment variable'ını ayarlar:

```bash
PORT=8000  # Cloud Run otomatik olarak ayarlar
HOST=0.0.0.0  # Cloud Run için gerekli
```

## 📝 Kullanım

### Local Development

`.env` dosyası oluşturun:

```bash
cp .env.example .env
```

Değerleri düzenleyin ve Docker Compose otomatik olarak okur.

### Google Cloud Run

Environment variable'ları Cloud Run'da ayarlayın:

```bash
gcloud run deploy api-gateway \
  --set-env-vars PORT=8000,API_GATEWAY_PORT=8000 \
  --port 8000
```

## ✅ Önemli Notlar

1. **PORT Environment Variable**: Google Cloud Run otomatik olarak `PORT` environment variable'ını ayarlar
2. **Fallback Değerler**: Tüm environment variable'lar için mantıklı fallback'ler var
3. **Merkezi Config**: Tüm değerler `config/ports.py` ve `frontend/config/ports.ts` dosyalarından alınır
4. **Hardcoded Yok**: Artık hiçbir yerde hardcoded host veya port yok

