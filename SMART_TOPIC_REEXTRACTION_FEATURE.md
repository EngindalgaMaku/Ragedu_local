# Smart Topic Re-Extraction Feature

**Tarih:** 20 Kasım 2025  
**Feature:** Smart Topic Re-Extraction  
**Endpoint:** `POST /api/aprag/topics/re-extract/{session_id}`

---

## 🎯 Problem

### Mevcut Topic Extraction Sorunu:

```
Döküman: 149 chunk (toplam ~50,000 karakter)
LLM Limit: 12,000 karakter (~6,000 token - Groq limiti)

Mevcut Davranış:
chunks_text[:12000]  # Sadece ilk 12k karakter kullanılır

Sonuç:
❌ Sadece ilk ~30 chunk analiz edilir
❌ Geriye kalan 119 chunk'taki konular ATLANIR
❌ Dökümanın %20'si → konular çıkarılır
❌ Dökümanın %80'i → GÖZ ARDI EDİLİR
```

**Örnek:**
```
Döküman: Hücre Biyolojisi (149 chunk)

İlk 30 chunk:
✅ Hücre Nedir
✅ Hücre Teorisi
✅ Hücre Zarı
✅ Hücre Organelleri

Chunk 31-149 (ATLANIR!):
❌ Mitokondri Detayları
❌ Endoplazmik Retikulum
❌ Golgi Aygıtı
❌ Hücre Bölünmesi
❌ Mitoz ve Mayoz
```

---

## ✅ Çözüm: Smart Topic Re-Extraction

### Yeni Yaklaşım:

```python
@router.post("/re-extract/{session_id}")
async def re_extract_topics_smart(
    session_id: str,
    method: str = "full"  # full, partial, merge
):
    """
    Tüm dökümanı analiz eder - hiçbir chunk atlanmaz!
    
    Workflow:
    1. Tüm chunk'ları al (149 chunk)
    2. Batch'lere böl (her batch 12k char)
       → Batch 1: Chunk 1-30
       → Batch 2: Chunk 31-60
       → Batch 3: Chunk 61-90
       → Batch 4: Chunk 91-120
       → Batch 5: Chunk 121-149
    3. Her batch için ayrı topic extraction
    4. Duplicate'leri merge et
    5. Sıralı liste oluştur
    6. Database'e kaydet
    """
```

### Akış Diyagramı:

```
149 Chunk
    ↓
Split to Batches
    ↓
Batch 1 (30 chunk) → LLM → ["Hücre Nedir", "Hücre Teorisi", "Hücre Zarı"]
Batch 2 (30 chunk) → LLM → ["Organeller", "Mitokondri", "ER"]
Batch 3 (30 chunk) → LLM → ["Golgi", "Lizozom", "Ribozom"]
Batch 4 (30 chunk) → LLM → ["Hücre Bölünmesi", "Mitoz"]
Batch 5 (29 chunk) → LLM → ["Mayoz", "Hücre Döngüsü"]
    ↓
Merge Similar Topics
    ↓
["Hücre Nedir", "Hücre Teorisi", "Hücre Zarı", "Hücre Organelleri",
 "Mitokondri", "ER", "Golgi", "Lizozom", "Ribozom",
 "Hücre Bölünmesi", "Mitoz", "Mayoz", "Hücre Döngüsü"]
    ↓
Re-order (1-13)
    ↓
Save to Database
```

---

## 🔧 Kullanım

### Method 1: Full (Tam Yenileme)

```bash
# Eski konuları SİL, tüm dökümanı yeniden analiz et
curl -X POST http://localhost:8007/api/aprag/topics/re-extract/abc123?method=full

# Response:
{
  "success": true,
  "method": "full",
  "batches_processed": 5,
  "raw_topics_extracted": 35,  # Her batch'ten ~7 topic
  "merged_topics_count": 13,    # Duplicate'ler temizlendi
  "saved_topics_count": 13,
  "chunks_analyzed": 149        # TÜM CHUNK'LAR! ✅
}
```

**Ne Zaman Kullan:**
- İlk extraction yanlış gittiyse
- Döküman değiştiyse
- Daha detaylı topic yapısı istiyorsanız

### Method 2: Partial (Eksikleri Ekle)

```bash
# Mevcut konuları KORU, sadece eksik olanları ekle
curl -X POST http://localhost:8007/api/aprag/topics/re-extract/abc123?method=partial

# Response:
{
  "success": true,
  "method": "partial",
  "existing_topics_count": 8,   # Mevcut korundu
  "new_topics_added": 5,         # Yeni eklendi
  "chunks_analyzed": 149
}
```

**Ne Zaman Kullan:**
- Mevcut konular iyi ama eksikler var
- Manuel düzenlediğiniz konuları kaybetmek istemiyorsanız

---

## 📊 Karşılaştırma

| Özellik | Old Extraction | Smart Re-Extraction |
|---------|---------------|---------------------|
| **Analiz Edilen** | İlk 12k char | TÜM döküman |
| **Chunk Coverage** | %20 (~30/149) | %100 (149/149) ✅ |
| **Topic Count** | 3-5 | 10-15 |
| **Süre** | 30 saniye | 2-3 dakika |
| **Accuracy** | Orta | Yüksek ✨ |
| **Detay Level** | Düşük | Yüksek |

---

## 💡 Avantajlar

### 1. Tam Kapsama ✅
```
Old: 149 chunk → 30 kullanıldı (20%)
New: 149 chunk → 149 kullanıldı (100%)
```

### 2. Batch Processing 🔄
```
5 batch × 30 saniye = 150 saniye (2.5 dakika)
Her batch bağımsız → Paralel işlenebilir (gelecek)
```

### 3. Duplicate Merging 🔀
```
Raw: 35 topic (5 batch × 7)
Merged: 13 unique topic
Similarity threshold: 70% word overlap
```

### 4. Smart Ordering 📋
```
Topics otomatik sıralanır:
1. Temel kavramlar (beginner)
2. Ara konular (intermediate)
3. İleri konular (advanced)
```

---

## 🚀 Frontend Entegrasyonu

### EnhancedTopicManagementPanel'e Ekle:

```tsx
// Yeni buton: "Konuları Yeniden Çıkar (Gelişmiş)"
<button
  onClick={() => handleSmartReExtraction("full")}
  disabled={reExtracting}
  className="px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg"
>
  {reExtracting ? (
    <>🔄 Tüm Döküman Analiz Ediliyor...</>
  ) : (
    <>🔬 Gelişmiş Yeniden Çıkar</>
  )}
</button>

// Handler:
const handleSmartReExtraction = async (method: "full" | "partial") => {
  try {
    setReExtracting(true);
    
    const response = await fetch(
      `${URLS.API_GATEWAY}/api/aprag/topics/re-extract/${sessionId}?method=${method}`,
      { method: "POST" }
    );
    
    const data = await response.json();
    
    if (data.success) {
      setSuccess(
        `✅ ${data.merged_topics_count} konu oluşturuldu! ` +
        `(${data.chunks_analyzed} chunk analiz edildi - ${data.batches_processed} batch)`
      );
      await fetchTopics();
    }
  } catch (e) {
    setError("Re-extraction başarısız");
  } finally {
    setReExtracting(false);
  }
};
```

---

## 📈 Beklenen İyileştirme

### Senaryo: Hücre Biyolojisi Dökümanı

**Old Extraction:**
```
Chunks: 149
Analyzed: 30 (20%)
Topics: 5
├─ Hücre Nedir
├─ Hücre Teorisi
├─ Hücre Zarı
├─ Prokaryot/Ökaryot
└─ Hücre Organelleri (genel)
```

**Smart Re-Extraction:**
```
Chunks: 149
Analyzed: 149 (100%) ✅
Topics: 13
├─ Hücre Nedir
├─ Hücre Teorisi
├─ Hücre Zarı
│  ├─ Fosfolipid Yapısı
│  └─ Membran Proteinleri
├─ Prokaryot ve Ökaryot Hücreler
├─ Hücre Organelleri
│  ├─ Mitokondri
│  ├─ Endoplazmik Retikulum
│  ├─ Golgi Aygıtı
│  ├─ Lizozom
│  └─ Ribozom
├─ Hücre Bölünmesi
├─ Mitoz
└─ Mayoz

Topic Coverage: +160% 🔥
Detail Level: +300% 🔥
```

---

## ⚡ Performance

| Operation | Time | Cost |
|-----------|------|------|
| **Old Extraction** | 30s | $0.001 |
| **Smart Re-Extraction (5 batch)** | 150s | $0.005 |
| **Paralel (future)** | 40s | $0.005 |

**Maliyet:** Minimal - $0.005 per session (Groq API)  
**Fayda:** Maksimum - %100 döküman coverage

---

## 🎓 Kullanım Senaryoları

### Senaryo 1: İlk Extraction Yetersiz

```
Öğretmen: "Sadece 5 konu çıkardı, ama döküman çok daha zengin!"

Çözüm:
→ "Gelişmiş Yeniden Çıkar" (Full method)
→ 13 konu çıkar ✅
```

### Senaryo 2: Manuel Konular + Otomatik Tamamlama

```
Öğretmen manuel 3 konu ekledi, gerisi otomatik olsun

Çözüm:
→ "Eksikleri Ekle" (Partial method)
→ Mevcut 3 korunur
→ +7 yeni konu eklenir
→ Toplam 10 konu ✅
```

### Senaryo 3: Döküman Güncellendi

```
Döküman revize edildi, yeni bölümler eklendi

Çözüm:
→ "Gelişmiş Yeniden Çıkar" (Full method)
→ Tüm döküman fresh analiz
→ Güncel konu yapısı ✅
```

---

## 🛠️ Implementation Status

- [x] ✅ Backend API (`re-extract` endpoint)
- [x] ✅ Batch splitting logic
- [x] ✅ Duplicate merging
- [x] ✅ Smart ordering
- [ ] ⏳ Frontend UI integration (pending)
- [ ] ⏳ Parallel batch processing (future)
- [ ] ⏳ Progress tracking (future)

---

## 📝 API Documentation

### Request:

```bash
POST /api/aprag/topics/re-extract/{session_id}?method={full|partial}
```

### Response:

```json
{
  "success": true,
  "method": "full",
  "session_id": "abc123",
  "batches_processed": 5,
  "raw_topics_extracted": 35,
  "merged_topics_count": 13,
  "saved_topics_count": 13,
  "chunks_analyzed": 149
}
```

### Error Handling:

```json
{
  "detail": "No chunks found for session"
}
```

---

**Status:** ✅ IMPLEMENTED - Ready for Frontend Integration  
**Priority:** 🔴 HIGH - Critical for large documents  
**Impact:** 🔥 +160% topic coverage






