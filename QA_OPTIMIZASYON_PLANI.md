# QA Pair Matching Optimizasyon Planı

## 🎯 Mevcut Sorun

**Şu anki yaklaşım:**
1. Topic ID'lere göre 50 QA pair veritabanından çekiliyor
2. Her QA pair için similarity hesaplanıyor (batch embedding ile optimize edildi ama hala 50 QA pair çekiliyor)
3. Top 5 seçiliyor

**Sorunlar:**
- 50 QA pair çekmek gereksiz (sadece 5 kullanılıyor)
- Topic bazlı filtreleme yeterli değil (semantic similarity daha iyi)
- Embedding hesaplama her seferinde yapılıyor

## 🚀 Önerilen Çözümler

### Çözüm 1: QA Question Embedding'lerini Veritabanında Saklamak (ÖNERİLEN)

**Avantajlar:**
- Embedding'ler önceden hesaplanıp saklanıyor
- Query embedding'i bir kez hesaplanıyor
- Cosine similarity veritabanında veya memory'de hızlı hesaplanıyor
- Topic filtreleme + semantic search kombinasyonu

**Yaklaşım:**
1. QA pair oluşturulduğunda/question güncellendiğinde embedding hesaplanıp saklanıyor
2. Query geldiğinde:
   - Query embedding'i hesaplanıyor (1 kez)
   - Topic ID'lere göre QA pair'ler çekiliyor (embedding'leriyle birlikte)
   - Cosine similarity hesaplanıyor (numpy ile hızlı)
   - Top 5 seçiliyor

**Veritabanı değişikliği:**
```sql
ALTER TABLE topic_qa_pairs 
ADD COLUMN question_embedding BLOB,  -- Embedding vector (JSON veya binary)
ADD COLUMN embedding_model VARCHAR(100),  -- Hangi model kullanıldı
ADD COLUMN embedding_dim INTEGER,  -- Embedding boyutu
ADD COLUMN embedding_updated_at TIMESTAMP;  -- Ne zaman güncellendi
```

### Çözüm 2: ChromaDB'de QA Question'ları İndex'lemek

**Avantajlar:**
- ChromaDB'nin optimize edilmiş vector search'ü
- Topic bazlı filtreleme metadata ile yapılabilir
- Çok hızlı semantic search

**Yaklaşım:**
1. QA question'ları ChromaDB'de ayrı bir collection'da saklamak
2. Query geldiğinde ChromaDB'de vector search yapmak
3. Topic ID metadata ile filtreleme

**Dezavantajlar:**
- ChromaDB'ye bağımlılık
- Collection yönetimi ekstra iş
- Veritabanı ile senkronizasyon sorunu

### Çözüm 3: Hybrid Yaklaşım (Keyword + Embedding)

**Avantajlar:**
- Önce keyword ile filtreleme (hızlı)
- Sonra embedding ile similarity (doğru)

**Yaklaşım:**
1. Query'den keyword'ler çıkarılıyor
2. QA question'larda keyword araması yapılıyor (SQL LIKE veya full-text search)
3. Filtrelenmiş QA pair'ler için embedding similarity hesaplanıyor

## 📊 Karşılaştırma

| Yaklaşım | Hız | Doğruluk | Karmaşıklık | Önerilen |
|----------|-----|----------|-------------|----------|
| Mevcut (50 QA çek) | ⭐⭐ | ⭐⭐⭐ | ⭐ | ❌ |
| Embedding DB'de sakla | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ |
| ChromaDB index | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⚠️ |
| Hybrid (keyword+embedding) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ |

## 🎯 Önerilen Uygulama: Çözüm 1 (Embedding DB'de Sakla)

**Neden:**
- En pratik ve hızlı çözüm
- Mevcut veritabanı yapısına minimal değişiklik
- Topic filtreleme + semantic search kombinasyonu
- Embedding'ler önceden hesaplanıp saklanıyor (QA oluşturulduğunda)

**Uygulama Adımları:**
1. Migration: `question_embedding` kolonu ekle
2. QA oluşturulduğunda/güncellendiğinde embedding hesapla ve sakla
3. Query'de: Topic ID'lere göre QA pair'leri çek (embedding'leriyle)
4. Query embedding'i hesapla (1 kez)
5. Cosine similarity hesapla (numpy ile hızlı)
6. Top 5 seç

**Beklenen Performans:**
- Önce: 50 QA çek + 50 embedding hesapla = ~2-3 saniye
- Sonra: 50 QA çek (embedding'leriyle) + 1 query embedding + similarity hesapla = ~200-300ms
- **10x hızlanma bekleniyor**



