# Document Processing Service - Modüler Mimari

## 🎯 Genel Bakış

2474 satırlık monolitik `main.py` dosyası, bakımı kolay ve test edilebilir modüler bir yapıya dönüştürüldü.

## 📁 Yeni Dizin Yapısı

```
document_processing_service/
├── main.py                          # ESKİ monolitik dosya (2474 satır) - YEDEK
├── main_new.py                      # YENİ modüler ana dosya (43 satır) ✨
├── config.py                        # Konfigürasyon yönetimi (48 satır)
├── requirements.txt                 # Değişmedi
├── Dockerfile                       # Değişmedi
│
├── core/                            # Temel bileşenler
│   ├── __init__.py
│   ├── chromadb_client.py          # ChromaDB bağlantı yönetimi
│   ├── embedding_service.py        # Embedding oluşturma
│   └── turkish_utils.py            # Türkçe NLP araçları
│
├── services/                        # İş mantığı servisleri
│   ├── __init__.py
│   ├── chunking_service.py         # Metin bölme işlemleri
│   ├── crag_evaluator.py           # CRAG değerlendirmesi
│   ├── hybrid_search.py            # Hibrit arama (Semantic + BM25)
│   └── chunk_improver.py           # LLM chunk iyileştirme
│
├── models/                          # Pydantic modelleri
│   ├── __init__.py
│   └── schemas.py                  # Tüm request/response modelleri
│
├── api/                             # API katmanı
│   ├── __init__.py
│   └── routes/                     # Endpoint grupları
│       ├── __init__.py             # Route kayıt sistemi
│       ├── health.py               # Sağlık kontrolü endpoints
│       ├── processing.py           # Metin işleme endpoints
│       ├── query.py                # RAG sorgu endpoints
│       ├── crag.py                 # CRAG değerlendirme endpoints
│       └── improvement.py          # Chunk iyileştirme endpoints
│
└── utils/                           # Yardımcı araçlar
    ├── __init__.py
    ├── logger.py                   # Logging konfigürasyonu
    └── helpers.py                  # Yardımcı fonksiyonlar
```

## 📊 Karşılaştırma

| Özellik | ESKİ (main.py) | YENİ (Modüler) |
|---------|----------------|----------------|
| **Toplam Dosya Sayısı** | 1 dosya | 18 dosya |
| **En Büyük Dosya** | 2474 satır | ~400 satır |
| **Kod Organizasyonu** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Test Edilebilirlik** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Bakım Kolaylığı** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Import Performansı** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Yeni Geliştirici Dostu** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🚀 Kullanım

### Yeni Modüler Yapıyı Çalıştırma

```bash
# main.py yerine main_new.py kullanın
python main_new.py

# veya uvicorn ile
uvicorn main_new:app --host 0.0.0.0 --port 8080
```

### Docker ile Çalıştırma

Dockerfile'ı güncelleyin:

```dockerfile
# Eski: COPY services/document_processing_service/main.py .
# Yeni:
COPY services/document_processing_service/main_new.py main.py
COPY services/document_processing_service/config.py .
COPY services/document_processing_service/core ./core
COPY services/document_processing_service/services ./services
COPY services/document_processing_service/models ./models
COPY services/document_processing_service/api ./api
COPY services/document_processing_service/utils ./utils
```

## 🔧 Modül Açıklamaları

### 1. Core Modülleri

#### `core/chromadb_client.py`
- ChromaDB bağlantı yönetimi
- Docker ve Cloud Run desteği
- HTTP/HTTPS otomatik algılama

```python
from core.chromadb_client import get_chroma_client

client = get_chroma_client()
collection = client.get_collection("my_collection")
```

#### `core/embedding_service.py`
- Model Inference Service entegrasyonu
- Batch embedding desteği
- Multi-model fallback

```python
from core.embedding_service import get_embeddings_direct

embeddings = get_embeddings_direct(
    texts=["text1", "text2"],
    embedding_model="nomic-embed-text"
)
```

#### `core/turkish_utils.py`
- Türkçe stopwords
- Tokenizasyon
- BM25 için metin işleme

```python
from core.turkish_utils import tokenize_turkish, TURKISH_STOPWORDS

tokens = tokenize_turkish("Merhaba dünya!", remove_stopwords=True)
```

### 2. Service Modülleri

#### `services/chunking_service.py`
- Unified chunking system entegrasyonu
- LLM post-processing desteği
- Chunk başlık çıkarma

```python
from services.chunking_service import chunk_text_with_strategy

chunks = chunk_text_with_strategy(
    text="Uzun metin...",
    chunk_size=1000,
    strategy="lightweight",
    use_llm_post_processing=True
)
```

#### `services/crag_evaluator.py`
- CRAG (Corrective RAG) değerlendirmesi
- Cross-encoder reranking
- Accept/Filter/Reject kararları

```python
from services.crag_evaluator import CRAGEvaluator

evaluator = CRAGEvaluator(model_inference_url="http://...")
result = evaluator.evaluate_retrieved_docs(query, docs)
```

#### `services/hybrid_search.py`
- Semantic + BM25 hibrit arama
- Reciprocal Rank Fusion (RRF)
- Türkçe optimizasyonu

```python
from services.hybrid_search import perform_hybrid_search

result = perform_hybrid_search(
    query="soru",
    documents=docs,
    distances=distances,
    bm25_weight=0.3
)
```

#### `services/chunk_improver.py`
- Tek chunk iyileştirme
- Toplu chunk iyileştirme
- ChromaDB güncelleme

```python
from services.chunk_improver import improve_single_chunk

result = improve_single_chunk(
    chunk_text="chunk içeriği",
    language="tr",
    model_name="llama-3.1-8b-instant"
)
```

### 3. Model Modülleri

#### `models/schemas.py`
- Tüm Pydantic modelleri tek yerde
- Request/Response şemaları
- Type safety

```python
from models.schemas import ProcessRequest, ProcessResponse

request = ProcessRequest(
    text="işlenecek metin",
    chunk_size=1000
)
```

### 4. API Route Modülleri

#### `api/routes/health.py`
- `/` - Basit health check
- `/health` - Detaylı health check

#### `api/routes/processing.py`
- `/process-and-store` - Metin işleme ve depolama

#### `api/routes/query.py`
- `/query` - RAG sorgu endpoint

#### `api/routes/crag.py`
- `/crag-evaluate` - CRAG değerlendirme
- `/crag/evaluate` - Alternatif endpoint

#### `api/routes/improvement.py`
- `/chunks/improve-single` - Tek chunk iyileştirme
- `/sessions/{session_id}/chunks/improve` - Session chunk iyileştirme
- `/sessions/{session_id}/chunks/improve-all` - Toplu iyileştirme

### 5. Utility Modülleri

#### `utils/logger.py`
- Merkezi logging konfigürasyonu
- Log level yönetimi

```python
from utils.logger import logger

logger.info("İşlem başarılı")
logger.error("Hata oluştu")
```

#### `utils/helpers.py`
- Metadata sanitizasyonu
- Collection name formatlama

```python
from utils.helpers import sanitize_metadata, format_collection_name

clean_metadata = sanitize_metadata(raw_metadata)
collection_name = format_collection_name(session_id, add_timestamp=True)
```

## 🧪 Test Etme

### Import Testleri

```python
# Tüm modüllerin düzgün import olup olmadığını test et
python -c "from core import *; from services import *; from models import *; from utils import *; print('✅ All imports successful')"
```

### API Testleri

```bash
# Health check
curl http://localhost:8080/health

# Process and store
curl -X POST http://localhost:8080/process-and-store \
  -H "Content-Type: application/json" \
  -d '{"text": "Test metni", "chunk_size": 500}'

# RAG query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session", "query": "test sorusu"}'
```

## 🔄 Geçiş Planı

### Aşama 1: Test (Şu An)
- [x] Modüler yapı oluşturuldu
- [x] Tüm dosyalar oluşturuldu
- [ ] Import hataları düzeltilecek
- [ ] API testleri yapılacak

### Aşama 2: Deployment
1. `main.py` → `main_old.py` (yedek)
2. `main_new.py` → `main.py` (aktif)
3. Dockerfile güncelleme
4. Docker build ve test
5. Production deployment

### Aşama 3: Cleanup
- Eski `main_old.py` dosyasını kaldır
- Documentation güncelle
- CI/CD pipeline güncelle

## 📝 Yapılacaklar (TODO)

### Yüksek Öncelik
- [ ] Import hatalarını düzelt
- [ ] main_new.py'yi test et
- [ ] Eksik endpoint'leri ekle (reprocess, delete-session)
- [ ] Unit testler yaz

### Orta Öncelik
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Integration testler
- [ ] Performance benchmarkları
- [ ] Error handling iyileştirmeleri

### Düşük Öncelik
- [ ] Type hints tamamlama
- [ ] Docstring'leri genişlet
- [ ] Code coverage artır
- [ ] Linting rules ekle

## 💡 Avantajlar

### 1. Bakım Kolaylığı
- Her modül tek bir sorumluluğa sahip
- Değişiklikler izole edilmiş
- Code review daha kolay

### 2. Test Edilebilirlik
- Her modül bağımsız test edilebilir
- Mock'lama kolay
- Unit test coverage artırılabilir

### 3. Yeniden Kullanılabilirlik
- Modüller başka projelerde kullanılabilir
- İyi tanımlanmış arayüzler
- Bağımlılıklar net

### 4. Takım Çalışması
- Farklı geliştiriciler farklı modüllerde çalışabilir
- Merge conflict'ler azalır
- Code ownership net

### 5. Performans
- Lazy loading
- Selective import
- Daha hızlı başlangıç

## ⚠️ Dikkat Edilmesi Gerekenler

1. **Import Path'ler**: Relative import'lar kullanıldı, doğru çalıştığından emin olun
2. **Config Yönetimi**: Environment variable'lar config.py'de merkezi
3. **Geriye Dönük Uyumluluk**: API endpoint'leri değişmedi
4. **Dependency Injection**: Gerektiğinde dependency injection kullanılabilir
5. **Circular Imports**: Dikkatli olunmalı, şu an yok ama eklerken dikkat

## 🚨 Bilinen Sorunlar

1. **Import Hatları**: Bazı import'ların düzeltilmesi gerekebilir
2. **Eksik Endpoint'ler**: `/reprocess`, `/delete-session` gibi endpoint'ler henüz eklenmedi
3. **Test Coverage**: Unit testler henüz yazılmadı
4. **Type Hints**: Bazı fonksiyonlarda eksik olabilir

## 📚 Daha Fazla Bilgi

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python Project Structure Best Practices](https://docs.python-guide.org/writing/structure/)
- [Clean Architecture in Python](https://www.cosmicpython.com/)

---

**Versiyon**: 2.0.0  
**Tarih**: 20 Kasım 2024  
**Durum**: 🚧 Development (Test aşamasında)






