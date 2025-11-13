# 🎓 RAG3 Eğitim Sistemi - Kapsamlı Sistem Mimarisi ve Deployment Rehberi

## 📋 İçindekiler

1. [Sistem Genel Bakış](#sistem-genel-bakış)
2. [Mimari Yapı](#mimari-yapı)
3. [Teknoloji Stack](#teknoloji-stack)
4. [Servis Detayları](#servis-detayları)
5. [Docker Deployment](#docker-deployment)
6. [Kurulum Rehberi](#kurulum-rehberi)
7. [API Dokümantasyonu](#api-dokümantasyonu)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Sistem Genel Bakış

**RAG3 Eğitim Sistemi**, öğretmenlerin ders materyallerini yükleyip öğrencilerinin bu materyaller üzerinde akıllı sorular sormasını sağlayan bir **Retrieval Augmented Generation (RAG)** tabanlı eğitim platformudur.

### 🌟 Ana Özellikler

- 📚 **Ders Oturumu Yönetimi**: Sınıf bazlı materyal organizasyonu
- 📄 **Çoklu Format Desteği**: PDF, DOCX, PPTX, XLSX, MD dosyaları
- 🤖 **AI Destekli Soru-Cevap**: GPT tabanlı akıllı asistan
- 🔍 **Kaynak Gösterimi**: Cevapların hangi belgelerden geldiğini gösterir
- 🎨 **Modern UI/UX**: Eğitim odaklı, kullanıcı dostu arayüz
- 🐳 **Docker Deployment**: Kolay kurulum ve yönetim

---

## 🏗️ Mimari Yapı

### 📊 Sistem Mimarisi Diyagramı

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │  Microservices  │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   Ecosystem     │
│                 │    │                 │    │                 │
│ • Teacher Panel │    │ • Route Mgmt    │    │ • Doc Processor │
│ • Student Chat  │    │ • Session Mgmt  │    │ • Model Inference│
│ • File Upload   │    │ • Auth Layer    │    │ • DocStrange    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Data Layer    │
                    │                 │
                    │ • ChromaDB      │
                    │ • SQLite        │
                    │ • File Storage  │
                    └─────────────────┘
```

### 🔄 Veri Akışı

1. **📤 Dosya Yükleme**: Öğretmen → Frontend → API Gateway → DocStrange/DocProcessor
2. **🔄 İşleme**: DocProcessor → Chunking → Embedding → ChromaDB
3. **❓ Soru Sorma**: Öğrenci → Frontend → API Gateway → Model Inference
4. **🔍 RAG İşlemi**: Model Inference → ChromaDB → LLM → Cevap
5. **📋 Kaynak Gösterimi**: Cevap + Kaynak Belgeler → Frontend

---

## 💻 Teknoloji Stack

### 🎨 Frontend
- **Framework**: Next.js 14 (React 18)
- **Styling**: Tailwind CSS
- **Language**: TypeScript
- **Build**: Standalone mode for Docker

### 🚀 Backend
- **API Gateway**: FastAPI (Python 3.11)
- **Session Management**: SQLite + Professional Session Manager
- **Authentication**: JWT-based (basit implementasyon)

### 🤖 AI & ML
- **LLM Provider**: Groq API (Llama 3.1, Mixtral, Gemma)
- **Local LLM**: Ollama (opsiyonel)
- **Embeddings**: mxbai-embed-large (Ollama)
- **Vector Database**: ChromaDB 1.3.0

### 📄 Document Processing
- **PDF Processing**: DocStrange API
- **Office Docs**: python-docx, python-pptx, openpyxl
- **Chunking**: Semantic chunking (1500 chars, 150 overlap)

### 🐳 Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Docker Compose
- **Storage**: Named volumes
- **Networking**: Bridge network

---

## 🔧 Servis Detayları

### 1. 🌐 Frontend Service (`rag3-frontend`)
```yaml
Port: 3000
Technology: Next.js 14
Environment:
  - NEXT_PUBLIC_API_URL=http://host.docker.internal:8000
  - NODE_ENV=development
```

**Özellikler**:
- 🎓 Eğitim odaklı modern UI
- 📱 Responsive tasarım
- 🔄 Real-time chat interface
- 📊 Dashboard ve analytics
- 📁 Dosya yükleme arayüzü

### 2. 🚪 API Gateway (`api-gateway`)
```yaml
Port: 8000 (internal: 8080)
Technology: FastAPI
Database: SQLite (sessions)
```

**Endpoints**:
- `POST /sessions` - Ders oturumu oluştur
- `GET /sessions` - Oturumları listele
- `POST /documents/convert` - Belge dönüştürme
- `POST /documents/upload-markdown` - MD dosyası yükle
- `POST /documents/process-and-store` - RAG işleme
- `POST /rag/query` - Soru-cevap

### 3. 📄 Document Processing Service (`document-processing-service`)
```yaml
Port: 8003 (internal: 8080)
Technology: FastAPI + Python
Dependencies: ChromaDB, Model Inference
```

**İşlevler**:
- 📝 Metin çıkarma ve temizleme
- ✂️ Semantic chunking
- 🔢 Embedding oluşturma
- 💾 ChromaDB'ye kaydetme

### 4. 🤖 Model Inference Service (`model-inference-service`)
```yaml
Port: 8002
Technology: FastAPI + Ollama
Models: llama3:8b, nomic-embed-text
```

**Özellikler**:
- 🌐 Groq API entegrasyonu
- 🏠 Local Ollama desteği
- 📊 Embedding generation
- 🔄 Model switching

### 5. 📋 DocStrange Service (`docstrange-service`)
```yaml
Port: 8005 (internal: 80)
Technology: FastAPI
API: DocStrange API
```

**Desteklenen Formatlar**:
- 📄 PDF → Markdown
- 📝 DOCX → Markdown
- 📊 PPTX → Markdown
- 📈 XLSX → Markdown

### 6. 🗃️ ChromaDB Service (`chromadb-service`)
```yaml
Port: 8004 (internal: 8000)
Technology: ChromaDB 1.3.0
Storage: Persistent volume
```

**Yapılandırma**:
- 💾 Persistent storage: `/data`
- 🔒 Authentication: Disabled
- 📊 Telemetry: Disabled
- 🔄 Reset capability: Enabled

---

## 🐳 Docker Deployment

### 📁 Dosya Yapısı
```
rag3_for_colab/
├── docker-compose.yml          # Ana orchestration
├── Dockerfile.gateway.local    # API Gateway
├── frontend/
│   └── Dockerfile.frontend     # Frontend
├── services/
│   ├── docstrange_service/
│   │   └── Dockerfile
│   ├── document_processing_service/
│   │   └── Dockerfile.local
│   └── model_inference_service/
│       └── Dockerfile.local
└── src/                        # API Gateway source
```

### 🚀 Deployment Komutları

#### 1. 🏗️ İlk Kurulum
```bash
# Repository'yi klonla
git clone <repository-url>
cd rag3_for_colab

# Environment dosyalarını kontrol et
# docker-compose.yml içindeki API key'leri güncelle

# Tüm servisleri build et
docker-compose build

# Servisleri başlat
docker-compose up -d
```

#### 2. 🔄 Güncelleme
```bash
# Servisleri durdur
docker-compose down

# Yeni kod değişikliklerini çek
git pull

# Değişen servisleri rebuild et
docker-compose build

# Servisleri yeniden başlat
docker-compose up -d
```

#### 3. 📊 Monitoring
```bash
# Servis durumlarını kontrol et
docker-compose ps

# Logları görüntüle
docker-compose logs -f [service-name]

# Kaynak kullanımını kontrol et
docker stats
```

### 💾 Volume Yönetimi

```yaml
volumes:
  ollama_cache:      # Ollama model cache
  chroma_data:       # ChromaDB persistent data
  session_data:      # Session database
  markdown_data:     # Uploaded markdown files
```

**Backup Komutu**:
```bash
# Volume backup
docker run --rm -v rag3_for_colab_chroma_data:/data -v $(pwd):/backup alpine tar czf /backup/chroma_backup.tar.gz -C /data .
```

---

## 🛠️ Kurulum Rehberi

### 📋 Gereksinimler

- 🐳 **Docker**: 20.10+
- 🐙 **Docker Compose**: 2.0+
- 💾 **RAM**: Minimum 8GB (Önerilen 16GB)
- 💿 **Disk**: Minimum 20GB boş alan
- 🌐 **Network**: İnternet bağlantısı (model indirme için)

### 🔧 Adım Adım Kurulum

#### 1. 📥 Repository Hazırlama
```bash
git clone <repository-url>
cd rag3_for_colab
```

#### 2. 🔑 API Key Yapılandırması
`docker-compose.yml` dosyasında:
```yaml
environment:
  - GROQ_API_KEY=your_groq_api_key_here
  - DOCSTRANGE_API_KEY=your_docstrange_key_here
```

#### 3. 🏗️ Build ve Deploy
```bash
# Tüm servisleri build et
docker-compose build --no-cache

# Servisleri başlat
docker-compose up -d

# Durumu kontrol et
docker-compose ps
```

#### 4. ✅ Doğrulama
- 🌐 Frontend: http://localhost:3000
- 🚪 API Gateway: http://localhost:8000/health
- 🗃️ ChromaDB: http://localhost:8004/api/v1/heartbeat

### 🔧 Geliştirme Ortamı

#### Hot Reload için:
```bash
# Frontend development
cd frontend
npm run dev

# Backend development (API Gateway)
cd src
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

---

## 📚 API Dokümantasyonu

### 🔐 Authentication
Basit token-based authentication:
```javascript
localStorage.setItem("isAuthenticated", "true");
```

### 📋 Session Management

#### Ders Oturumu Oluşturma
```http
POST /sessions
Content-Type: application/json

{
  "name": "Biyoloji 9. Sınıf",
  "description": "Hücre bölünmesi konusu",
  "category": "biology",
  "created_by": "teacher_id",
  "grade_level": "9",
  "subject_area": "Biyoloji"
}
```

#### Oturumları Listeleme
```http
GET /sessions?limit=10&category=biology
```

### 📄 Document Processing

#### Belge Dönüştürme
```http
POST /documents/convert
Content-Type: multipart/form-data

file: [PDF/DOCX/PPTX/XLSX file]
```

#### Markdown Yükleme
```http
POST /documents/upload-markdown
Content-Type: multipart/form-data

file: [.md file]
```

#### RAG İşleme
```http
POST /documents/process-and-store
Content-Type: multipart/form-data

session_id: "session-uuid"
markdown_files: ["file1.md", "file2.md"]
chunk_strategy: "semantic"
chunk_size: 1500
chunk_overlap: 150
embedding_model: "mxbai-embed-large"
```

### 🤖 RAG Query

#### Soru Sorma
```http
POST /rag/query
Content-Type: application/json

{
  "session_id": "session-uuid",
  "query": "Hücre bölünmesi nasıl gerçekleşir?",
  "top_k": 5,
  "use_rerank": true,
  "min_score": 0.1,
  "max_context_chars": 8000,
  "model": "llama-3.1-8b-instant"
}
```

**Response**:
```json
{
  "answer": "Hücre bölünmesi...",
  "sources": [
    {
      "content": "İlgili metin parçası...",
      "metadata": {
        "source_file": "biyoloji_ders_notu.md",
        "chunk_id": "chunk_123"
      },
      "score": 0.85
    }
  ]
}
```

---

## 🔧 Troubleshooting

### 🚨 Yaygın Sorunlar

#### 1. 🐳 Docker Build Hataları
```bash
# Cache temizleme
docker system prune -a

# Specific service rebuild
docker-compose build --no-cache [service-name]
```

#### 2. 🔌 Port Çakışması
```bash
# Port kullanımını kontrol et
netstat -tulpn | grep :3000

# Alternatif port kullan
docker-compose up -d --scale frontend=0
docker run -p 3001:3000 rag3_for_colab-frontend
```

#### 3. 🤖 Ollama Bağlantı Sorunu
```bash
# Container içinde Ollama durumunu kontrol et
docker exec model-inference-service curl http://127.0.0.1:11434/api/tags

# Model indirme
docker exec model-inference-service ollama pull llama3:8b
```

#### 4. 💾 ChromaDB Veri Kaybı
```bash
# Volume durumunu kontrol et
docker volume ls | grep chroma

# Backup'tan geri yükleme
docker run --rm -v rag3_for_colab_chroma_data:/data -v $(pwd):/backup alpine tar xzf /backup/chroma_backup.tar.gz -C /data
```

#### 5. 🌐 Frontend Build Hataları
```bash
# Node modules temizleme
docker-compose exec frontend rm -rf node_modules .next
docker-compose build frontend --no-cache
```

### 📊 Performance Tuning

#### Memory Optimization
```yaml
# docker-compose.yml
services:
  model-inference-service:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

#### ChromaDB Optimization
```yaml
chromadb-service:
  environment:
    - CHROMA_SERVER_CORS_ALLOW_ORIGINS=["*"]
    - CHROMA_SERVER_HTTP_PORT=8000
    - ANONYMIZED_TELEMETRY=false
```

---

## 📈 Monitoring ve Logs

### 📊 Health Checks
```bash
# API Gateway
curl http://localhost:8000/health

# ChromaDB
curl http://localhost:8004/api/v1/heartbeat

# Model Inference
curl http://localhost:8002/models/available
```

### 📝 Log Monitoring
```bash
# Tüm servislerin logları
docker-compose logs -f

# Specific service logs
docker-compose logs -f api-gateway
docker-compose logs -f frontend
docker-compose logs -f model-inference-service
```

### 🔍 Debug Mode
```yaml
# docker-compose.yml debug configuration
environment:
  - DEBUG=true
  - LOG_LEVEL=DEBUG
```

---

## 🚀 Production Deployment

### 🔒 Security Checklist
- [ ] API key'leri environment variables'a taşı
- [ ] HTTPS sertifikası ekle
- [ ] CORS ayarlarını güncelle
- [ ] Rate limiting ekle
- [ ] Authentication güçlendir

### 📊 Scaling Considerations
- 🔄 Load balancer ekle
- 📈 Horizontal scaling için Kubernetes
- 💾 External database (PostgreSQL)
- 🗃️ Distributed ChromaDB setup

---

## 📞 Destek ve Katkı

### 🐛 Bug Reporting
Issues açarken şunları ekleyin:
- 🐳 Docker version
- 💻 OS bilgisi
- 📝 Error logs
- 🔄 Reproduction steps

### 🤝 Contribution Guidelines
1. Fork repository
2. Feature branch oluştur
3. Changes commit et
4. Pull request aç
5. Review sürecini bekle

---

## 📄 Lisans

Bu proje **MIT License** altında lisanslanmıştır.

---

## 👨‍💻 Geliştirici

**Engin DALGA** - Burdur Mehmet Akif Ersoy Üniversitesi Yüksek Lisans Ödevi

📧 Contact: [email]
🔗 LinkedIn: [profile]
🐙 GitHub: [profile]

---

*Son güncelleme: Kasım 2024*
