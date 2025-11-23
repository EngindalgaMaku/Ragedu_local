# Docker Files Reference

Bu dosya, projede kullanılan tüm Dockerfile'ların güncel referans listesidir.

## 📋 Aktif Dockerfile'lar (2025-11-17)

Her servis için **sadece bir** Dockerfile bulunmaktadır. Karmaşayı önlemek için gereksiz/eski dosyalar temizlenmiştir.

### 1. API Gateway
- **Dosya:** `Dockerfile.gateway.local`
- **Konum:** `rag3_for_local/Dockerfile.gateway.local`
- **Servis:** api-gateway
- **Port:** 8000
- **Base Image:** python:3.11-slim

### 2. APRAG Service
- **Dosya:** `Dockerfile`
- **Konum:** `rag3_for_local/services/aprag_service/Dockerfile`
- **Servis:** aprag-service
- **Port:** 8007
- **Base Image:** python:3.11-slim

### 3. Auth Service
- **Dosya:** `Dockerfile`
- **Konum:** `rag3_for_local/services/auth_service/Dockerfile`
- **Servis:** auth-service
- **Port:** 8006
- **Base Image:** python:3.11-slim

### 4. Docstrange Service
- **Dosya:** `Dockerfile`
- **Konum:** `rag3_for_local/services/docstrange_service/Dockerfile`
- **Servis:** docstrange-service
- **Port:** 8005 (mapped to 80)
- **Base Image:** python:3.11-slim

### 5. Document Processing Service
- **Dosya:** `Dockerfile`
- **Konum:** `rag3_for_local/services/document_processing_service/Dockerfile`
- **Servis:** document-processing-service
- **Port:** 8003 (mapped to 8080)
- **Base Image:** python:3.11-slim

### 6. Model Inference Service
- **Dosya:** `Dockerfile.local`
- **Konum:** `rag3_for_local/services/model_inference_service/Dockerfile.local`
- **Servis:** model-inference-service
- **Port:** 8002
- **Base Image:** python:3.11-slim
- **Not:** `.local` suffix'i yerel development için Ollama kullanımını gösterir

### 7. Frontend
- **Dosya:** `Dockerfile.frontend`
- **Konum:** `rag3_for_local/frontend/Dockerfile.frontend`
- **Servis:** frontend
- **Port:** 3000
- **Base Image:** node:20-alpine

## 🗑️ Silinen Dosyalar (2025-11-17)

Aşağıdaki dosyalar karmaşayı önlemek için silindi:

1. ~~`rag3_for_local/Dockerfile`~~ → Eski API Gateway (artık `Dockerfile.gateway.local` kullanılıyor)
2. ~~`services/model_inference_service/Dockerfile`~~ → `.local` versiyonu kullanılıyor
3. ~~`services/chromadb_service/Dockerfile`~~ → Docker Hub image kullanılıyor (`chromadb/chroma:1.3.0`)
4. ~~`services/pdf_processing_service/Dockerfile`~~ → Service disabled
5. ~~`services/pdf_processing_service/Dockerfile.local`~~ → Service disabled

## 📦 Docker Hub Images

Bu servisler custom Dockerfile yerine direkt Docker Hub image kullanır:

### 1. ChromaDB Service
- **Image:** `chromadb/chroma:1.3.0`
- **Port:** 8004 (mapped to 8000)
- **Not:** Stable versiyonu kullanılıyor

### 2. Marker API
- **Image:** `wirawan/marker-api:latest`
- **Port:** 8090 (mapped to 8080)
- **Not:** PDF processing için kullanılıyor

## 🔍 Dockerfile Naming Conventions

- **Standard:** `Dockerfile` → Production/default build
- **Environment-specific:** `Dockerfile.local` → Local development
- **Service-specific:** `Dockerfile.gateway.local` → API Gateway özel konfig
- **Frontend:** `Dockerfile.frontend` → Next.js application

## 🚀 Build Komutları

Her servisi build etmek için:

```bash
# Tüm servisleri build et
docker-compose build

# Sadece bir servisi build et
docker-compose build [service-name]

# Örnek: API Gateway'i build et
docker-compose build api-gateway

# Örnek: Frontend'i build et
docker-compose build frontend
```

## 📝 Bakım Notları

- ✅ Her servis için **sadece bir** aktif Dockerfile var
- ✅ Tüm Dockerfile'lar `docker-compose.yml` ile senkronize
- ✅ Kullanılmayan dosyalar temizlendi
- ✅ Naming convention tutarlı
- ⚠️ Yeni Dockerfile eklerken bu dokümanı güncellemeyi unutmayın!

## 🔄 Son Güncelleme

**Tarih:** 2025-11-17  
**Güncelleme:** Docker dosyaları temizlendi ve standardize edildi  
**Temizleyen:** AI Assistant  
**Toplam Aktif Dockerfile:** 7















