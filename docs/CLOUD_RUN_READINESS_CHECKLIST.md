# Google Cloud Run Hazırlık Kontrol Listesi

## ✅ Tamamlanan Kontroller

### 1. Hardcoded URL'ler Kaldırıldı

#### ✅ API Gateway (`src/api/main.py`)
- [x] PDF_PROCESSOR_URL - Environment variable'dan alınıyor
- [x] DOCUMENT_PROCESSOR_URL - Environment variable'dan alınıyor
- [x] MODEL_INFERENCE_URL - Environment variable'dan alınıyor
- [x] CHROMADB_URL - Environment variable'dan alınıyor
- [x] MARKER_API_URL - Environment variable'dan alınıyor
- [x] AUTH_SERVICE_URL - Environment variable'dan alınıyor
- [x] APRAG_SERVICE_URL - Environment variable'dan alınıyor

#### ✅ APRAG Service
- [x] `topics.py`: MODEL_INFERENCER_URL, CHROMA_SERVICE_URL, DOCUMENT_PROCESSING_URL
- [x] `personalization.py`: MODEL_INFERENCE_URL
- [x] `recommendations.py`: MODEL_INFERENCE_URL
- [x] `aprag_middleware.py`: APRAG_SERVICE_URL

#### ✅ Document Processing Service
- [x] MODEL_INFERENCER_URL - Environment variable'dan alınıyor
- [x] CHROMADB_URL - Environment variable'dan alınıyor
- [x] ChromaDB client Cloud Run URL formatını destekliyor

#### ✅ Model Inference Service
- [x] OLLAMA_HOST - Environment variable'dan alınıyor

#### ✅ ChromaDB Service
- [x] CHROMADB_INTERNAL_URL - Hardcoded Cloud Run URL kaldırıldı

#### ✅ Frontend
- [x] `next.config.js`: API Gateway URL environment variable'dan alınıyor
- [x] `ports.ts`: Cloud Run URL formatı destekleniyor

#### ✅ Vector Store
- [x] `chroma_store.py`: CHROMADB_URL environment variable'dan alınıyor

### 2. URL Yapılandırma Mantığı

Tüm servisler şu mantıkla çalışıyor:

1. **Öncelik 1**: `*_SERVICE_URL` environment variable (tam URL - Cloud Run için)
2. **Öncelik 2**: `*_SERVICE_HOST` + `*_SERVICE_PORT` (Docker için)
3. **Fallback**: Docker service name formatı (sadece local Docker için)

### 3. Port Yapılandırması

- [x] Tüm servisler `PORT` environment variable'ını kullanıyor
- [x] Cloud Run otomatik olarak `PORT` ayarlar
- [x] Docker'da fallback port'lar var

### 4. CORS Yapılandırması

- [x] Tüm servisler `CORS_ORIGINS` environment variable'ından alıyor
- [x] Frontend domain'i environment variable ile set edilebilir

## 📋 Cloud Run Deployment İçin Gerekli Environment Variables

### Tüm Servisler İçin Ortak

```bash
# Her servis için
PORT=<Cloud Run otomatik ayarlar>
HOST=0.0.0.0

# CORS (tüm backend servisler için)
CORS_ORIGINS=https://your-frontend-domain.com,https://api-gateway-xxx.run.app
```

### API Gateway

```bash
PDF_PROCESSOR_URL=https://pdf-processor-xxx.run.app
DOCUMENT_PROCESSOR_URL=https://document-processing-xxx.run.app
MODEL_INFERENCE_URL=https://model-inference-xxx.run.app
CHROMADB_URL=https://chromadb-xxx.run.app
MARKER_API_URL=https://marker-api-xxx.run.app
AUTH_SERVICE_URL=https://auth-service-xxx.run.app
APRAG_SERVICE_URL=https://aprag-service-xxx.run.app
JWT_SECRET_KEY=your-production-secret-key
```

### Frontend

```bash
NEXT_PUBLIC_API_URL=https://api-gateway-xxx.run.app
NEXT_PUBLIC_AUTH_URL=https://auth-service-xxx.run.app
DOCKER_ENV=false  # Cloud Run için
```

### APRAG Service

```bash
MODEL_INFERENCER_URL=https://model-inference-xxx.run.app
MODEL_INFERENCE_URL=https://model-inference-xxx.run.app
CHROMA_SERVICE_URL=https://chromadb-xxx.run.app
CHROMADB_URL=https://chromadb-xxx.run.app
DOCUMENT_PROCESSING_URL=https://document-processing-xxx.run.app
```

### Document Processing Service

```bash
MODEL_INFERENCER_URL=https://model-inference-xxx.run.app
CHROMADB_URL=https://chromadb-xxx.run.app
CHROMA_SERVICE_URL=https://chromadb-xxx.run.app
```

## ⚠️ Önemli Notlar

1. **Database**: SQLite dosya sistemi Cloud Run'da kalıcı değildir. Production için Cloud SQL veya başka bir managed database kullanın.

2. **ChromaDB**: ChromaDB HttpClient HTTPS URL'leri için özel yapılandırma gerekebilir. Cloud Run'da ChromaDB servisi HTTPS destekliyorsa sorun yok.

3. **Service Discovery**: Cloud Run servisleri birbirlerini bulmak için public URL'leri kullanır. Internal network yoktur.

4. **Environment Variables**: Her servis için Cloud Run'da environment variable'ları set edin.

5. **CORS**: Frontend domain'inizi tüm backend servislerin CORS ayarlarına ekleyin.

## ✅ Sonuç

**Sistem Google Cloud Run'a deploy edilmeye hazır!**

- ✅ Hardcoded Docker service adları yok
- ✅ Hardcoded localhost adresleri yok
- ✅ Hardcoded Cloud Run URL'leri yok
- ✅ Tüm URL'ler environment variable'lardan alınıyor
- ✅ Docker ve Cloud Run uyumlu fallback mekanizması var

Deployment sırasında sadece environment variable'ları set etmeniz yeterli!



