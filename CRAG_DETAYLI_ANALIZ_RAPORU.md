# 🔍 CRAG (Corrective RAG) Detaylı Analiz Raporu

**Tarih:** 26 Kasım 2025  
**Soru:** "mitoz ve mayoz farkları"  
**Durum:** CRAG REJECT - Sistem çalışmıyor gibi görünüyor

---

## 📋 İÇİNDEKİLER

1. [CRAG Nedir?](#1-crag-nedir)
2. [CRAG İşlem Akışı](#2-crag-işlem-akışı)
3. [Skor Hesaplama Mekanizması](#3-skor-hesaplama-mekanizması)
4. [Threshold Değerleri ve Karar Mantığı](#4-threshold-değerleri-ve-karar-mantığı)
5. [Gerçek Senaryo Analizi](#5-gerçek-senaryo-analizi)
6. [Sorunlar ve Nedenler](#6-sorunlar-ve-nedenler)
7. [Öneriler ve Çözümler](#7-öneriler-ve-çözümler)

---

## 1. CRAG NEDİR?

**CRAG (Corrective RAG)**, RAG sistemlerinde **yanlış veya alakasız bilgilerin filtrelenmesi** için kullanılan bir değerlendirme mekanizmasıdır.

### Amaç:
- ✅ **İyi dokümanları kabul et** (yüksek relevance)
- ❌ **Kötü dokümanları reddet** (düşük relevance)
- 🔍 **Orta dokümanları filtrele** (threshold ile)

### Temel Mantık:
```
Vector Search → Chunks Bulunur → CRAG Değerlendirme → Karar Verilir
```

---

## 2. CRAG İŞLEM AKIŞI

### 2.1. Adım Adım İşlem

```
┌─────────────────────────────────────────────────────────────┐
│ 1. HYBRID RETRIEVAL                                         │
│    - Vector Search: 5 chunk bulundu                        │
│    - KB Retrieval: 3 KB item bulundu (%80 confidence)      │
│    - QA Matching: 0 eşleşme                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CRAG EVALUATION (Sadece Chunks için)                    │
│    - Reranker Service'e gönderilir                         │
│    - Query: "mitoz ve mayoz farkları"                      │
│    - Documents: 5 chunk içeriği                            │
│    - Reranker: Alibaba gte-rerank-v2                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. RERANKER SKORLARI                                        │
│    Chunk 1: 0.2293 (22.93%)                                │
│    Chunk 2: 0.1289 (12.89%)                                │
│    Chunk 3: 0.1181 (11.81%)                                │
│    Chunk 4: 0.1828 (18.28%)                                │
│    Chunk 5: 0.1593 (15.93%)                                │
│    Max Score: 0.2293                                        │
│    Avg Score: 0.1637                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. THRESHOLD KARŞILAŞTIRMA                                  │
│    incorrect_threshold = 0.3 (30%)                         │
│    Max Score (0.2293) < 0.3 → REJECT                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. KARAR: REJECT                                            │
│    ❌ Tüm chunks reddedildi                                │
│    ❌ KB bilgisi kullanılmadı (KB fallback yoktu)         │
│    ❌ Cevap üretilmedi                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. SKOR HESAPLAMA MEKANİZMASI

### 3.1. Reranker Servisi

**Alibaba gte-rerank-v2** kullanılıyor:

```python
# Reranker Service
POST /rerank
{
  "query": "mitoz ve mayoz farkları",
  "documents": [
    "## MITOZ\n<table>...",  # Chunk 1
    "İğ iplikleri kaybolur...",  # Chunk 2
    "## MAYOZDA KROMOZOM...",  # Chunk 3
    "## SORULAR\n1. Hayvansal...",  # Chunk 4
    "### EŞEYSİZ ÜREME...",  # Chunk 5
  ],
  "reranker_type": "alibaba"
}
```

### 3.2. Alibaba API Response

```json
{
  "output": {
    "results": [
      {"index": 0, "relevance_score": 0.2293, "document": "..."},
      {"index": 1, "relevance_score": 0.1289, "document": "..."},
      {"index": 2, "relevance_score": 0.1181, "document": "..."},
      {"index": 3, "relevance_score": 0.1828, "document": "..."},
      {"index": 4, "relevance_score": 0.1593, "document": "..."}
    ]
  }
}
```

### 3.3. Skor Aralığı

- **Alibaba gte-rerank-v2**: 0.0 - 1.0 arası (0-100%)
- **BGE Reranker**: 0.0 - 1.0 arası
- **MS-MARCO**: -5.0 ile +5.0 arası (normalize edilir)

---

## 4. THRESHOLD DEĞERLERİ VE KARAR MANTIĞI

### 4.1. Mevcut Threshold'lar (Alibaba için)

```python
correct_threshold = 0.7    # ACCEPT: max_score >= 0.7 (70%)
incorrect_threshold = 0.3  # REJECT: max_score < 0.3 (30%)
filter_threshold = 0.5     # FILTER: 0.3 <= max_score < 0.7, 
                           #         individual docs >= 0.5
```

### 4.2. Karar Mantığı

```python
if max_score >= 0.7:
    action = "accept"  # ✅ Tüm dokümanlar kabul
elif max_score < 0.3:
    action = "reject"  # ❌ Tüm dokümanlar reddedilir
else:
    action = "filter"  # 🔍 Threshold >= 0.5 olanlar filtrelenir
```

### 4.3. Gerçek Senaryo

```
Max Score: 0.2293 (22.93%)
Threshold: 0.3 (30%)

0.2293 < 0.3 → REJECT ❌
```

**Sonuç:** Tüm chunks reddedildi, sistem "bilgi bulunamadı" dedi.

---

## 5. GERÇEK SENARYO ANALİZİ

### 5.1. Vector Search Sonuçları

| Chunk | İçerik Preview | Vector Score | CRAG Score | Durum |
|-------|---------------|--------------|------------|-------|
| 1 | `## MITOZ\n<table>...` | 100.0% | 22.93% | ❌ REJECT |
| 2 | `İğ iplikleri kaybolur...` | 99.4% | 12.89% | ❌ REJECT |
| 3 | `## MAYOZDA KROMOZOM...` | 96.7% | 11.81% | ❌ REJECT |
| 4 | `## SORULAR\n1. Hayvansal...` | 83.7% | 18.28% | ❌ REJECT |
| 5 | `### EŞEYSİZ ÜREME...` | 81.2% | 15.93% | ❌ REJECT |

### 5.2. Analiz

**Vector Search:** ✅ Çok iyi çalışıyor (100%, 99.4%, 96.7% skorlar)
- Chunk 1: "MITOZ" ve "MAYOZ" tablosu içeriyor → **MÜKEMMEL EŞLEŞME**
- Chunk 2-3: Mayoz ve mitoz detayları → **İYİ EŞLEŞME**

**CRAG (Reranker):** ❌ Çok düşük skorlar (22.93%, 12.89%, 11.81%)
- Chunk 1 bile sadece 22.93% aldı
- Tüm chunks 30% threshold'un altında

### 5.3. Neden Bu Kadar Düşük Skor?

#### Olası Nedenler:

1. **Query Formatı Sorunu**
   - Query: "mitoz ve mayoz farkları"
   - Reranker "farkları" kelimesini yeterince anlamıyor olabilir
   - "karşılaştırma", "fark", "benzerlik" gibi terimler daha iyi skor alabilir

2. **Chunk İçeriği Formatı**
   - Chunk 1: HTML table formatında (`<table>`, `<th>`, `<td>`)
   - Reranker HTML tag'lerini düzgün parse edemiyor olabilir
   - Markdown veya plain text daha iyi skor alabilir

3. **Reranker Model Limiti**
   - Alibaba gte-rerank-v2 genel amaçlı bir model
   - Biyoloji terimleri için optimize edilmemiş
   - Türkçe "farkları" kelimesi için düşük skor verebilir

4. **Threshold Çok Yüksek**
   - 0.3 (30%) threshold çok katı olabilir
   - 0.2293 gibi skorlar aslında "orta" seviye olabilir
   - Ama sistem bunu "çok düşük" olarak değerlendiriyor

---

## 6. SORUNLAR VE NEDENLER

### 6.1. Ana Sorun: CRAG Çok Katı

**Problem:**
- Vector search mükemmel sonuçlar buluyor (100%, 99.4%)
- Ama CRAG bunları reddediyor (22.93% < 30%)
- Sonuç: Sistem çalışmıyor gibi görünüyor

**Neden:**
1. **Threshold çok yüksek**: 0.3 (30%) çok katı
2. **Reranker modeli yetersiz**: HTML table'ları düzgün anlamıyor
3. **Query formatı**: "farkları" kelimesi reranker için yeterince açık değil

### 6.2. KB Bilgisi Kullanılmadı

**Problem:**
- KB %80 confidence ile bulundu
- Ama CRAG reject olduğunda KB kullanılmadı
- KB fallback logic yoktu (şimdi eklendi)

**Neden:**
- CRAG sadece chunks'ı değerlendiriyor
- KB bilgisi CRAG'e dahil edilmiyor
- CRAG reject olduğunda KB de atlanıyordu

### 6.3. Vector Search vs CRAG Uyumsuzluğu

**Problem:**
- Vector search: Semantic similarity (embedding-based)
- CRAG: Cross-encoder relevance (reranker-based)
- İki yöntem farklı skorlar veriyor

**Neden:**
- Vector search: "mitoz" ve "mayoz" kelimelerini yakın buluyor
- CRAG: Query-document pair'ini değerlendiriyor, format sorunları var

---

## 7. ÖNERİLER VE ÇÖZÜMLER

### 7.1. ✅ Yapılan İyileştirmeler

1. **KB Fallback Logic Eklendi**
   - CRAG reject olsa bile, KB confidence >= 0.7 ise KB kullanılıyor
   - "mitoz ve mayoz farkları" sorusu için KB %80 confidence → KB kullanılacak

2. **Detaylı Debug Bilgileri**
   - Her chunk için CRAG skorları gösteriliyor
   - Threshold status bilgileri eklendi
   - Neden reject edildiği açıklanıyor

### 7.2. 🔧 Önerilen İyileştirmeler

#### A. Threshold'ları Gevşetmek

```python
# Mevcut (Çok Katı)
incorrect_threshold = 0.3  # 30%

# Önerilen (Daha Toleranslı)
incorrect_threshold = 0.2  # 20%
filter_threshold = 0.3     # 30% (0.5'ten düşürüldü)
```

**Avantaj:**
- 0.2293 gibi skorlar artık reject edilmeyecek
- Orta seviye dokümanlar kullanılabilecek

**Dezavantaj:**
- Daha düşük kaliteli dokümanlar da kabul edilebilir

#### B. Chunk İçeriğini Temizlemek

```python
# HTML tag'lerini kaldır
import re
def clean_chunk_content(content: str) -> str:
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    # Remove markdown table syntax
    content = re.sub(r'\|.*\|', '', content)
    return content.strip()
```

**Avantaj:**
- Reranker daha temiz içerik görecek
- Skorlar artabilir

#### C. Query Expansion

```python
# "farkları" → "karşılaştırma", "fark", "benzerlik"
query_expansions = {
    "farkları": ["karşılaştırma", "fark", "benzerlik", "ayrım"],
    "nedir": ["tanım", "açıklama", "anlam"],
    "nasıl": ["yöntem", "süreç", "adımlar"]
}
```

**Avantaj:**
- Reranker daha iyi anlayacak
- Skorlar artabilir

#### D. Hybrid Scoring

```python
# Vector score + CRAG score kombinasyonu
final_score = (vector_score * 0.6) + (crag_score * 0.4)

# Eğer vector score çok yüksekse, CRAG'i daha az önemli yap
if vector_score > 0.9:
    final_score = (vector_score * 0.8) + (crag_score * 0.2)
```

**Avantaj:**
- Vector search'in iyi sonuçları korunur
- CRAG sadece filtreleme için kullanılır

#### E. CRAG'i Sadece Filtreleme İçin Kullanmak

```python
# CRAG'i reject için değil, sadece filtreleme için kullan
if crag_result["action"] == "reject":
    # Reject etme, sadece düşük skorlu olanları filtrele
    filtered_docs = [doc for doc in chunks 
                     if doc.get("crag_score", 0) >= 0.2]
    # Eğer hiç doküman kalmadıysa, vector search sonuçlarını kullan
    if not filtered_docs:
        filtered_docs = chunks[:3]  # Top 3'ü al
```

**Avantaj:**
- CRAG sadece filtreleme yapar, reject yapmaz
- Vector search'in iyi sonuçları korunur

---

## 8. SONUÇ VE TAVSİYELER

### 8.1. Mevcut Durum

- ✅ **Vector Search:** Mükemmel çalışıyor
- ❌ **CRAG:** Çok katı, iyi sonuçları reddediyor
- ✅ **KB Fallback:** Eklendi, artık KB kullanılacak

### 8.2. Öncelikli Çözümler

1. **Threshold'ları gevşet** (0.3 → 0.2)
2. **KB fallback kullan** (✅ Yapıldı)
3. **Chunk içeriğini temizle** (HTML tag'leri kaldır)
4. **CRAG'i sadece filtreleme için kullan** (reject yapma)

### 8.3. Uzun Vadeli Çözümler

1. **Daha iyi reranker modeli** (Türkçe için optimize edilmiş)
2. **Query expansion** (farkları → karşılaştırma, fark, benzerlik)
3. **Hybrid scoring** (vector + CRAG kombinasyonu)
4. **Domain-specific reranker** (Biyoloji için özel model)

---

## 9. DEBUG BİLGİLERİ

### 9.1. Mevcut Debug Bilgileri

Artık debug panelinde şunlar görünüyor:

```json
{
  "crag_evaluation": {
    "action": "reject",
    "max_score": 0.2293,
    "avg_score": 0.1637,
    "thresholds": {
      "correct": 0.7,
      "incorrect": 0.3,
      "filter": 0.5
    },
    "detailed_chunk_scores": [
      {
        "chunk_index": 0,
        "content_preview": "## MITOZ\n<table>...",
        "original_similarity": 1.0,
        "crag_score": 0.2293,
        "threshold_status": {
          "rejected": true
        }
      }
    ]
  }
}
```

### 9.2. Yeni Eklenen Bilgiler

- ✅ Her chunk için detaylı skor bilgisi
- ✅ Threshold status (above_correct, above_filter, rejected)
- ✅ Content preview (ilk 150 karakter)
- ✅ Original similarity vs CRAG score karşılaştırması

---

## 10. TEST ÖNERİLERİ

### 10.1. Threshold Testi

```python
# Test 1: Threshold 0.2'ye düşür
incorrect_threshold = 0.2  # 0.2293 > 0.2 → ACCEPT olmalı

# Test 2: Threshold 0.15'e düşür
incorrect_threshold = 0.15  # Daha toleranslı

# Test 3: Threshold 0.25'e düşür
incorrect_threshold = 0.25  # Orta seviye
```

### 10.2. Query Format Testi

```python
# Test 1: "mitoz ve mayoz farkları"
# Test 2: "mitoz mayoz karşılaştırma"
# Test 3: "mitoz ile mayoz arasındaki farklar"
# Test 4: "mitoz mayoz benzerlik fark"
```

### 10.3. Chunk Temizleme Testi

```python
# Test 1: HTML tag'leri kaldır
# Test 2: Markdown table syntax'ını kaldır
# Test 3: Sadece plain text gönder
```

---

**Rapor Hazırlayan:** AI Assistant  
**Tarih:** 26 Kasım 2025  
**Versiyon:** 1.0



