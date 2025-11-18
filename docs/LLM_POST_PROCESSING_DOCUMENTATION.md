# LLM Post-Processing for Chunk Refinement

## Overview

The LLM Post-Processing feature provides optional chunk refinement using Large Language Models to improve the quality and semantic coherence of text chunks. This enterprise-grade enhancement maintains the lightweight architecture while offering advanced text processing capabilities.

## 🎯 Key Features

- **Optional Processing**: Can be enabled/disabled per request
- **Turkish Language Optimized**: Specialized prompts for Turkish content
- **Batch Processing**: Efficient parallel processing of multiple chunks
- **Performance Optimizations**: Caching, retry mechanisms, and timeout handling
- **Error Handling**: Graceful fallback to original chunks if LLM processing fails
- **Zero Breaking Changes**: Fully backward compatible with existing system

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway   │────┤ Document Service │────┤ Text Chunker    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌──────────────────┐             │
                       │ Model Inference  │◄────────────┘
                       │    Service       │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │ ChunkPostProcessor│
                       │  - Batch Processing│
                       │  - Turkish Prompts │
                       │  - Caching System  │
                       └──────────────────┘
```

## 📊 Performance Comparison

| Mode                  | Speed        | Quality      | Use Case                                 |
| --------------------- | ------------ | ------------ | ---------------------------------------- |
| **Lightweight Only**  | ⚡ Very Fast | ✅ Good      | Fast processing, high volume             |
| **Lightweight + LLM** | 🐌 Slower    | ⭐ Excellent | High-quality chunks, educational content |

## 🔧 API Integration

### Document Processing Endpoint

**URL:** `POST /documents/process-and-store`

#### Request Parameters

| Parameter                 | Type      | Default                                | Description                         |
| ------------------------- | --------- | -------------------------------------- | ----------------------------------- |
| `session_id`              | `string`  | **required**                           | Session identifier                  |
| `markdown_files`          | `string`  | **required**                           | JSON array of markdown files        |
| `chunk_strategy`          | `string`  | `"lightweight"`                        | Chunking strategy                   |
| `chunk_size`              | `integer` | `1000`                                 | Target chunk size in characters     |
| `chunk_overlap`           | `integer` | `100`                                  | Overlap between chunks              |
| `embedding_model`         | `string`  | `"mixedbread-ai/mxbai-embed-large-v1"` | Embedding model                     |
| `use_llm_post_processing` | `boolean` | `false`                                | **🆕 Enable LLM post-processing**   |
| `llm_model_name`          | `string`  | `"llama-3.1-8b-instant"`               | **🆕 LLM model for refinement**     |
| `model_inference_url`     | `string`  | `null`                                 | **🆕 Override model inference URL** |

#### Example Request

**Form Data:**

```bash
curl -X POST "http://localhost:8000/documents/process-and-store" \
  -F "session_id=my-session-123" \
  -F "markdown_files=[\"document1.md\", \"document2.md\"]" \
  -F "chunk_strategy=lightweight" \
  -F "chunk_size=800" \
  -F "chunk_overlap=80" \
  -F "embedding_model=nomic-embed-text" \
  -F "use_llm_post_processing=true" \
  -F "llm_model_name=llama-3.1-8b-instant" \
  -F "model_inference_url=http://localhost:8002"
```

**JavaScript:**

```javascript
const formData = new FormData();
formData.append("session_id", "my-session-123");
formData.append(
  "markdown_files",
  JSON.stringify(["document1.md", "document2.md"])
);
formData.append("chunk_strategy", "lightweight");
formData.append("chunk_size", "800");
formData.append("chunk_overlap", "80");
formData.append("embedding_model", "nomic-embed-text");
formData.append("use_llm_post_processing", "true");
formData.append("llm_model_name", "llama-3.1-8b-instant");
formData.append("model_inference_url", "http://localhost:8002");

const response = await fetch(
  "http://localhost:8000/documents/process-and-store",
  {
    method: "POST",
    body: formData,
  }
);
```

#### Response Format

```json
{
  "success": true,
  "message": "Successfully processed 2 files",
  "processed_count": 2,
  "total_chunks_added": 15,
  "processing_time": "45.2s"
}
```

## ⚙️ Configuration Options

### ChunkPostProcessor Configuration

```python
from src.text_processing.chunk_post_processor import PostProcessingConfig

config = PostProcessingConfig(
    model_name="llama-3.1-8b-instant",           # LLM model to use
    model_inference_url="http://localhost:8002",  # Model inference service URL
    batch_size=5,                                 # Chunks per batch
    max_workers=3,                                # Parallel threads
    enable_caching=True,                          # Enable result caching
    cache_ttl_seconds=3600,                       # Cache expiration (1 hour)
    request_timeout_seconds=30,                   # Request timeout
    max_retries=2,                                # Retry failed requests
    smart_filtering=True,                         # Only process chunks that need improvement
    preserve_formatting=True,                     # Keep markdown formatting
    language_hint="turkish"                       # Language for optimized prompts
)
```

### Environment Variables

```bash
# Model Inference Service
MODEL_INFERENCE_URL=http://localhost:8002
MODEL_INFERENCE_HOST=model-inference-service
MODEL_INFERENCE_PORT=8002

# LLM Post-Processing Defaults
DEFAULT_LLM_MODEL=llama-3.1-8b-instant
ENABLE_LLM_POST_PROCESSING=false
LLM_BATCH_SIZE=5
LLM_MAX_WORKERS=3
LLM_CACHE_TTL=3600
```

## 🚀 Usage Examples

### 1. Basic Usage (Programmatic)

```python
from src.text_processing.text_chunker import chunk_text

chunks = chunk_text(
    text="Türkiye coğrafyası hakkında detaylı bilgi...",
    chunk_size=500,
    chunk_overlap=50,
    strategy="lightweight",
    use_llm_post_processing=True,
    llm_model_name="llama-3.1-8b-instant",
    model_inference_url="http://localhost:8002"
)

print(f"Generated {len(chunks)} high-quality chunks")
```

### 2. Direct ChunkPostProcessor Usage

```python
from src.text_processing.chunk_post_processor import ChunkPostProcessor, PostProcessingConfig

# Configure the processor
config = PostProcessingConfig(
    model_name="llama-3.1-8b-instant",
    batch_size=3,
    enable_caching=True
)

processor = ChunkPostProcessor(config)

# Process chunks
original_chunks = [
    "Türkiye'de iklim çeşitliliği görülür...",
    "Akdeniz iklimi güney sahillerde etkili...",
    "Karasal iklim iç kesimlerde hakimdir..."
]

refined_chunks = processor.process_chunks(original_chunks)

for i, (original, refined) in enumerate(zip(original_chunks, refined_chunks)):
    print(f"Chunk {i+1}:")
    print(f"Original: {original[:50]}...")
    print(f"Refined:  {refined[:50]}...")
    print()
```

### 3. API Usage with Different Configurations

#### High-Quality Processing (Slower)

```bash
curl -X POST "http://localhost:8000/documents/process-and-store" \
  -F "session_id=quality-session" \
  -F "markdown_files=[\"educational-content.md\"]" \
  -F "use_llm_post_processing=true" \
  -F "llm_model_name=llama-3.3-70b-versatile" \
  -F "chunk_size=1200"
```

#### Fast Processing (No LLM)

```bash
curl -X POST "http://localhost:8000/documents/process-and-store" \
  -F "session_id=fast-session" \
  -F "markdown_files=[\"large-document.md\"]" \
  -F "use_llm_post_processing=false" \
  -F "chunk_size=800"
```

#### Balanced Processing

```bash
curl -X POST "http://localhost:8000/documents/process-and-store" \
  -F "session_id=balanced-session" \
  -F "markdown_files=[\"medium-content.md\"]" \
  -F "use_llm_post_processing=true" \
  -F "llm_model_name=llama-3.1-8b-instant" \
  -F "chunk_size=1000"
```

## 🎨 Turkish Language Optimization

The system uses specialized prompts optimized for Turkish educational content:

### Prompt Structure

```python
TURKISH_REFINEMENT_PROMPT = """
Sen eğitim içeriklerini iyileştiren bir uzman asistansın.

GÖREVIN:
- Verilen chunk'ı anlamsal olarak daha tutarlı hale getir
- Gereksiz tekrarları kaldır
- Başlık ve liste formatlarını iyileştir
- Türkçe dilbilgisi kurallarına uy
- İçeriğin eğitsel değerini artır

KURALLAR:
1. ✅ Anlamı koruyarak daha akıcı hale getir
2. ✅ Teknik terimleri doğru kullan
3. ✅ Markdown formatını koru
4. ❌ İçerik ekleme veya çıkarma
5. ❌ Ana anlamı değiştirme

CHUNK: {chunk_content}

İYİLEŞTİRİLMİŞ CHUNK:
"""
```

### Language Detection

The system automatically detects Turkish content and applies appropriate processing:

```python
def detect_language(text: str) -> str:
    """Detect if text is Turkish for optimized processing"""
    turkish_indicators = [
        'türkiye', 'ğ', 'ı', 'ş', 'ç', 'ö', 'ü',
        'olan', 'olan', 'için', 'ile', 'bu', 'da', 'de'
    ]

    text_lower = text.lower()
    turkish_score = sum(1 for indicator in turkish_indicators if indicator in text_lower)

    return "turkish" if turkish_score >= 3 else "english"
```

## 📈 Performance Optimization

### Caching System

The system implements intelligent caching to avoid duplicate LLM calls:

```python
# Cache key generation
cache_key = hashlib.md5(f"{chunk_content}_{model_name}_{config_hash}".encode()).hexdigest()

# Cache TTL configuration
cache_ttl_seconds = 3600  # 1 hour default
```

### Batch Processing

Chunks are processed in batches for optimal performance:

```python
batch_size = 5  # Process 5 chunks simultaneously
max_workers = 3  # Use 3 parallel threads

# Batch processing with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for batch in chunk_batches:
        future = executor.submit(process_batch, batch)
        futures.append(future)
```

### Smart Filtering

Only chunks that would benefit from refinement are processed:

```python
def needs_refinement(chunk: str) -> bool:
    """Determine if a chunk needs LLM refinement"""
    # Skip very short chunks
    if len(chunk.strip()) < 100:
        return False

    # Skip already well-formatted chunks
    if has_clear_structure(chunk) and has_good_flow(chunk):
        return False

    return True
```

## 🔍 Troubleshooting

### Common Issues

#### 1. LLM Service Unavailable

**Symptom:** Chunks are not being refined, original chunks returned

**Solution:**

```bash
# Check model inference service
curl http://localhost:8002/health

# Verify service is running
docker ps | grep model-inference
```

#### 2. Slow Processing

**Symptom:** Document processing takes very long

**Solutions:**

- Reduce batch size: `batch_size=2`
- Use faster model: `llm_model_name="llama-3.1-8b-instant"`
- Disable post-processing for large documents: `use_llm_post_processing=false`

#### 3. Memory Issues

**Symptom:** Out of memory errors during processing

**Solutions:**

- Reduce `max_workers`: `max_workers=1`
- Smaller `batch_size`: `batch_size=2`
- Process documents individually instead of in bulk

#### 4. Rate Limiting

**Symptom:** 429 errors from LLM service

**Solutions:**

- Increase `request_timeout_seconds`: `request_timeout_seconds=60`
- Add delays between requests: `max_retries=3`
- Enable caching: `enable_caching=True`

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('chunk_post_processor')
logger.setLevel(logging.DEBUG)
```

### Health Checks

Monitor system health with these endpoints:

```bash
# API Gateway health
curl http://localhost:8000/health

# Document processing service health
curl http://localhost:8080/health

# Model inference service health
curl http://localhost:8002/health
```

## 📊 Monitoring and Analytics

### Performance Metrics

Monitor these key metrics:

```python
# Processing time per chunk
avg_processing_time_ms = total_time / chunk_count

# Cache hit rate
cache_hit_rate = cache_hits / total_requests

# Success rate
success_rate = successful_chunks / total_chunks

# LLM API usage
api_calls_per_minute = api_calls / elapsed_minutes
```

### Logging

The system provides structured logging:

```python
logger.info(f"✅ LLM post-processing successful: {len(chunks)} chunks refined")
logger.warning(f"⚠️ LLM service timeout, using original chunks")
logger.error(f"❌ LLM post-processing failed: {error_message}")
```

## 🔮 Future Enhancements

Planned improvements for the LLM post-processing feature:

1. **Multi-language Support**: Extend beyond Turkish to support other languages
2. **Custom Prompts**: Allow users to define custom refinement prompts
3. **Quality Metrics**: Implement automatic quality assessment of refined chunks
4. **A/B Testing**: Compare original vs. refined chunks for effectiveness
5. **Streaming Processing**: Real-time chunk refinement as documents are uploaded

## 📚 Related Documentation

- [Lightweight Turkish Chunking Architecture](./Lightweight_Turkish_Chunking_Architecture.md)
- [Docker Services Reference](./DOCKER_SERVICES_REFERENCE.md)
- [API Gateway Documentation](../src/api/README.md)
- [Model Inference Service](../services/model_inference_service/README.md)

## 🤝 Contributing

To contribute to the LLM post-processing feature:

1. **Testing**: Run comprehensive tests:

   ```bash
   python rag3_for_local/tests/test_llm_post_processing.py
   python rag3_for_local/tests/test_llm_post_processing_integration.py
   ```

2. **Development**: Follow the architecture patterns established in:

   - `src/text_processing/chunk_post_processor.py`
   - `src/text_processing/text_chunker.py`

3. **Documentation**: Update this document when adding new features

---

**Last Updated:** 2025-11-17  
**Version:** 1.0.0  
**Maintainer:** RAG3 Development Team
