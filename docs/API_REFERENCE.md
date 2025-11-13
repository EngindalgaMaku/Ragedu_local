# 📚 RAG3 Eğitim Sistemi - API Referans Dokümantasyonu

## 🔗 Base URL

```
http://localhost:8000
```

## 🔐 Authentication

Sistem basit token-based authentication kullanır:

```javascript
// Frontend localStorage
localStorage.setItem("isAuthenticated", "true");
```

---

## 📋 Session Management API

### 📚 Ders Oturumu Oluşturma

```http
POST /sessions
Content-Type: application/json
```

**Request Body:**

```json
{
  "name": "Biyoloji 9. Sınıf - Hücre",
  "description": "Hücre yapısı ve fonksiyonları",
  "category": "biology",
  "created_by": "teacher_001",
  "grade_level": "9",
  "subject_area": "Biyoloji",
  "learning_objectives": ["Hücre yapısını öğrenmek", "Organelleri tanımak"],
  "tags": ["hücre", "biyoloji", "9.sınıf"],
  "is_public": false
}
```

**Response:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Biyoloji 9. Sınıf - Hücre",
  "description": "Hücre yapısı ve fonksiyonları",
  "category": "biology",
  "status": "active",
  "created_by": "teacher_001",
  "created_at": "2024-11-01T12:00:00Z",
  "updated_at": "2024-11-01T12:00:00Z",
  "last_accessed": "2024-11-01T12:00:00Z",
  "grade_level": "9",
  "subject_area": "Biyoloji",
  "learning_objectives": ["Hücre yapısını öğrenmek", "Organelleri tanımak"],
  "tags": ["hücre", "biyoloji", "9.sınıf"],
  "document_count": 0,
  "total_chunks": 0,
  "query_count": 0,
  "user_rating": 0.0,
  "is_public": false,
  "backup_count": 0
}
```

### 📖 Oturumları Listeleme

```http
GET /sessions?created_by=teacher_001&category=biology&status=active&limit=10
```

**Query Parameters:**

- `created_by` (optional): Oluşturan kişi ID'si
- `category` (optional): Kategori filtresi
- `status` (optional): Durum filtresi (active, inactive, archived)
- `limit` (optional): Maksimum sonuç sayısı (default: 50)

**Response:**

```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Biyoloji 9. Sınıf - Hücre"
    // ... diğer session fields
  }
]
```

### 🔍 Tek Oturum Getirme

```http
GET /sessions/{session_id}
```

### 🗑️ Oturum Silme

```http
DELETE /sessions/{session_id}?create_backup=true&deleted_by=teacher_001
```

---

## 📄 Document Management API

### 📝 Belge Dönüştürme (PDF/DOCX/PPTX/XLSX → Markdown)

```http
POST /documents/convert
Content-Type: multipart/form-data
```

**Form Data:**

- `file`: Dosya (PDF, DOCX, PPTX, XLSX)

**Response:**

```json
{
  "success": true,
  "message": "Document converted successfully",
  "markdown_filename": "document_20241101_120000.md",
  "metadata": {
    "original_filename": "ders_notu.pdf",
    "file_size": 1024000,
    "page_count": 15,
    "conversion_time": 2.5
  }
}
```

### 📄 Markdown Dosyası Yükleme

```http
POST /documents/upload-markdown
Content-Type: multipart/form-data
```

**Form Data:**

- `file`: Markdown dosyası (.md)

**Response:**

```json
{
  "success": true,
  "message": "Markdown file uploaded successfully",
  "markdown_filename": "ders_notu.md"
}
```

### 📋 Markdown Dosyalarını Listeleme

```http
GET /documents/list-markdown
```

**Response:**

```json
{
  "markdown_files": [
    "biyoloji_ders_notu.md",
    "kimya_formulleri.md",
    "matematik_teoremler.md"
  ],
  "count": 3
}
```

### 📖 Markdown Dosyası İçeriği Getirme

```http
GET /documents/markdown/{filename}
```

**Response:**

```json
{
  "content": "# Hücre Biyolojisi\n\n## Giriş\n\nHücre, tüm canlıların temel yapı taşıdır..."
}
```

### 🔄 Belgeleri İşleme ve Saklama (RAG Pipeline)

```http
POST /documents/process-and-store
Content-Type: multipart/form-data
```

**Form Data:**

- `session_id`: Ders oturumu ID'si
- `markdown_files`: JSON array olarak dosya isimleri
- `chunk_strategy`: "semantic" (default)
- `chunk_size`: 1500 (default)
- `chunk_overlap`: 150 (default)
- `embedding_model`: "mxbai-embed-large" (default)

**Example:**

```bash
curl -X POST http://localhost:8000/documents/process-and-store \
  -F "session_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "markdown_files=[\"biyoloji_ders_notu.md\", \"hucre_yapisi.md\"]" \
  -F "chunk_strategy=semantic" \
  -F "chunk_size=1500" \
  -F "chunk_overlap=150" \
  -F "embedding_model=mxbai-embed-large"
```

**Response:**

```json
{
  "success": true,
  "processed_count": 2,
  "chunks_created": 45,
  "message": "Successfully processed 2 files",
  "successful_files": ["biyoloji_ders_notu.md", "hucre_yapisi.md"],
  "failed_files": null
}
```

---

## 🤖 RAG Query API

### ❓ Soru Sorma (RAG Query)

```http
POST /rag/query
Content-Type: application/json
```

**Request Body:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "Hücre zarının yapısı ve fonksiyonları nelerdir?",
  "top_k": 5,
  "use_rerank": true,
  "min_score": 0.1,
  "max_context_chars": 8000,
  "model": "llama-3.1-8b-instant"
}
```

**Parameters:**

- `session_id`: Ders oturumu ID'si (required)
- `query`: Soru metni (required)
- `top_k`: En alakalı kaç chunk getirileceği (default: 5)
- `use_rerank`: Sonuçları yeniden sıralama (default: true)
- `min_score`: Minimum benzerlik skoru (default: 0.1)
- `max_context_chars`: Maksimum context karakter sayısı (default: 8000)
- `model`: Kullanılacak LLM modeli (optional)

**Response:**

```json
{
  "answer": "Hücre zarı, hücrenin dış sınırını oluşturan ve hücre içi ile dışı arasındaki madde alışverişini kontrol eden önemli bir yapıdır. Temel olarak fosfolipid çift tabakasından oluşur ve seçici geçirgenlik özelliği gösterir...",
  "sources": [
    {
      "content": "Hücre zarı (plazma membranı), hücrenin en dış kısmında bulunan ve hücreyi çevreleyen ince bir tabakadır. Fosfolipidler, proteinler ve karbonhidratlardan oluşur...",
      "metadata": {
        "source_file": "biyoloji_ders_notu.md",
        "chunk_id": "chunk_001",
        "page_number": 3
      },
      "score": 0.89
    },
    {
      "content": "Seçici geçirgenlik, hücre zarının en önemli özelliklerinden biridir. Bu özellik sayesinde hücre, ihtiyaç duyduğu maddeleri içeri alır, zararlı maddeleri dışarıda bırakır...",
      "metadata": {
        "source_file": "hucre_yapisi.md",
        "chunk_id": "chunk_015"
      },
      "score": 0.76
    }
  ]
}
```

**Bağlam Dışı Soru Response:**

```json
{
  "answer": "⚠️ **DERS KAPSAMINDA DEĞİL**\n\nSorduğunuz soru ders dökümanlarında bulunamamıştır. Eğer sorunuzun ders içeriğiyle ilgili olduğunu düşünüyorsanız öğretmeninize bildiriniz.\n\n📚 *Lütfen ders materyalleri kapsamında sorular sorunuz.*",
  "sources": []
}
```

---

## 🤖 Model Management API

### 📋 Mevcut Modelleri Listeleme

```http
GET /models/available
```

**Response:**

```json
{
  "models": [
    {
      "id": "llama-3.1-8b-instant",
      "name": "Llama 3.1 8B (Instant)",
      "provider": "groq",
      "type": "cloud",
      "description": "Groq (Hızlı)"
    },
    {
      "id": "llama-3.3-70b-versatile",
      "name": "Llama 3.3 70B Versatile",
      "provider": "groq",
      "type": "cloud",
      "description": "Groq (Hızlı)"
    },
    {
      "id": "openai/gpt-oss-20b",
      "name": "OpenAI GPT OSS 20B",
      "provider": "groq",
      "type": "cloud",
      "description": "Groq (Hızlı)"
    },
    {
      "id": "qwen/qwen3-32b",
      "name": "Qwen 3 32B",
      "provider": "groq",
      "type": "cloud",
      "description": "Groq (Hızlı)"
    }
  ],
  "providers": {
    "groq": {
      "name": "Groq",
      "description": "Hızlı cloud modelleri",
      "models": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b"
      ]
    }
  }
}
```

---

## 🔧 System Health API

### ❤️ Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "ok",
  "service": "api-gateway",
  "timestamp": "2024-11-01T12:00:00Z",
  "version": "1.0.0"
}
```

### 🔍 Servis Durumları

```http
GET /health/services
```

**Response:**

```json
{
  "gateway": "ok",
  "services": {
    "document-processing": {
      "status": "ok",
      "url": "http://document-processing-service:8080",
      "response_time": 0.15
    },
    "model-inference": {
      "status": "ok",
      "url": "http://model-inference-service:8002",
      "response_time": 0.23
    },
    "chromadb": {
      "status": "ok",
      "url": "http://chromadb-service:8000",
      "response_time": 0.08
    }
  }
}
```

---

## 📊 Session Analytics API

### 📈 Oturum İstatistikleri

```http
GET /sessions/{session_id}/stats
```

**Response:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_count": 5,
  "total_chunks": 127,
  "query_count": 23,
  "avg_response_time": 1.2,
  "user_rating": 4.5,
  "last_accessed": "2024-11-01T12:00:00Z",
  "popular_queries": [
    "Hücre zarının yapısı nedir?",
    "Mitoz ve mayoz arasındaki fark nedir?",
    "DNA replikasyonu nasıl gerçekleşir?"
  ]
}
```

### 📋 Oturum Chunk'larını Listeleme

```http
GET /sessions/{session_id}/chunks?limit=10&offset=0
```

**Response:**

```json
{
  "chunks": [
    {
      "chunk_id": "chunk_001",
      "content": "Hücre zarı, hücrenin en dış kısmında bulunan...",
      "metadata": {
        "source_file": "biyoloji_ders_notu.md",
        "chunk_index": 1,
        "char_count": 1456
      },
      "embedding_model": "mxbai-embed-large",
      "created_at": "2024-11-01T12:00:00Z"
    }
  ],
  "total_count": 127,
  "limit": 10,
  "offset": 0
}
```

---

## ❌ Error Responses

### 🚨 Yaygın Hata Kodları

#### 400 Bad Request

```json
{
  "detail": "Invalid request format or missing required fields"
}
```

#### 404 Not Found

```json
{
  "detail": "Session not found"
}
```

#### 500 Internal Server Error

```json
{
  "detail": "Failed to process document: Connection timeout"
}
```

#### 503 Service Unavailable

```json
{
  "detail": "Ollama client is not available. Check connection to Ollama."
}
```

---

## 📝 Rate Limiting

- **Default**: 100 requests/minute per IP
- **RAG Queries**: 10 requests/minute per session
- **File Uploads**: 5 requests/minute per IP

---

## 🔒 Security Headers

Tüm API responses şu security header'ları içerir:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
```

---

## 📚 SDK Examples

### JavaScript/TypeScript

```typescript
// API Client örneği
class RAG3Client {
  private baseUrl: string;

  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async createSession(
    sessionData: CreateSessionRequest
  ): Promise<SessionResponse> {
    const response = await fetch(`${this.baseUrl}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sessionData),
    });
    return response.json();
  }

  async ragQuery(queryData: RAGQueryRequest): Promise<RAGQueryResponse> {
    const response = await fetch(`${this.baseUrl}/rag/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(queryData),
    });
    return response.json();
  }
}
```

### Python

```python
import requests
from typing import Dict, Any

class RAG3Client:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def create_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/sessions",
            json=session_data
        )
        return response.json()

    def rag_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/rag/query",
            json=query_data
        )
        return response.json()
```

---

## 🔄 Webhook Support (Future)

Gelecek versiyonlarda webhook desteği planlanmaktadır:

```http
POST /webhooks/register
Content-Type: application/json

{
  "url": "https://your-app.com/webhook",
  "events": ["session.created", "document.processed", "query.completed"],
  "secret": "your-webhook-secret"
}
```

---

_API Dokümantasyonu - Son güncelleme: Kasım 2024_
