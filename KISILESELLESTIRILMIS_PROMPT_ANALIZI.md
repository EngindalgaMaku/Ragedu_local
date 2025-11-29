# Kişiselleştirilmiş Prompt Analizi

## Özet

Debug raporunda **iki farklı prompt** kullanılıyor:

1. **Standart RAG Prompt** (Section [5.1]): İlk yanıt için kullanılan prompt - kişiselleştirme içermez
2. **Kişiselleştirilmiş Prompt**: Orijinal yanıtı kişiselleştirmek için kullanılan prompt - raporda görünmez

## Kişiselleştirilmiş Prompt Yapısı

Kişiselleştirilmiş prompt `_generate_personalization_prompt()` fonksiyonu tarafından oluşturulur ve şu bölümlerden oluşur:

### 1. Öğrenci Profili Bölümü

```
📊 ÖĞRENCİ PROFİLİ:
- Anlama Seviyesi: {understanding_level}
- Zorluk Seviyesi: {difficulty_level}
- Açıklama Stili: {explanation_style}
- Örnekler Gerekli: {needs_examples}
```

**Rapordaki Değerler:**
- `understanding_level`: "intermediate"
- `difficulty_level`: "elementary" (ZPD'den gelen öneri)
- `explanation_style`: "balanced"
- `needs_examples`: false

### 2. ZPD (Zone of Proximal Development) Bölümü

```
🎯 ZPD (Zone of Proximal Development):
- Mevcut Seviye: {current_level}
- Önerilen Seviye: {recommended_level}
- Başarı Oranı: {success_rate}
```

**Rapordaki Değerler:**
- `current_level`: "intermediate"
- `recommended_level`: "elementary"
- `success_rate`: 0.0 (0%)

### 3. Bloom Taksonomisi Bölümü

```
🧠 Bloom Taksonomisi:
- Tespit Edilen Seviye: {level} (Seviye {level_index})
- Güven: {confidence}
```

**Rapordaki Değerler:**
- `level`: "remember"
- `level_index`: 1
- `confidence`: 1.0 (100%)

### 4. Bilişsel Yük Bölümü

```
⚖️ Bilişsel Yük:
- Toplam Yük: {total_load}
- Sadeleştirme Gerekli: {needs_simplification}
```

**Rapordaki Değerler:**
- `total_load`: 0.089 (çok düşük)
- `needs_simplification`: false

### 5. Pedagogical Instructions (En Önemli Bölüm)

```
🎓 PEDAGOJİK TALİMATLAR (ÇOK ÖNEMLİ - MUTLAKA UYGULA):
{pedagogical_instructions}
```

**Rapordaki Değerler (Section [9.5]):**
```
--- BLOOM SEVİYE TALİMATI ---
Bu soru Bloom Taksonomisi Seviye 1 (remember) gerektiriyor.
Öğrencinin mevcut seviyesi: elementary

📝 Yanıt Stratejisi:
- Kısa, net ve doğrudan tanım ver
- Hafızayı destekleyici ipuçları ekle
- Anahtar kelimeleri vurgula
```

### 6. Kişiselleştirme Talimatları

Prompt'a eklenen dinamik talimatlar:

**Difficulty Level = "elementary" olduğu için:**
- "Temel kavramları önce açıkla"
- "Teknik terimleri basit dille açıkla"
- "Daha basit kelimeler kullan"

**Explanation Style = "balanced" olduğu için:**
- (Ne "detailed" ne de "concise" - orta seviye)

**Needs Examples = false olduğu için:**
- Örnek eklenmez

## Parametrelerin Prompt'a Yansıması

### 1. ZPD Parametreleri → Prompt'a Yansıma

| Parametre | Değer | Prompt'a Yansıması |
|-----------|-------|-------------------|
| `recommended_level` | "elementary" | → `difficulty_level` olarak kullanılır → "Temel kavramları önce açıkla" talimatı eklenir |
| `current_level` | "intermediate" | → ZPD bölümünde gösterilir |
| `success_rate` | 0.0 | → ZPD bölümünde gösterilir |

### 2. Bloom Parametreleri → Prompt'a Yansıma

| Parametre | Değer | Prompt'a Yansıması |
|-----------|-------|-------------------|
| `level` | "remember" | → Bloom bölümünde gösterilir + Pedagogical Instructions oluşturulur |
| `level_index` | 1 | → Bloom bölümünde gösterilir |
| `confidence` | 1.0 | → Bloom bölümünde gösterilir |

**Bloom Instructions Oluşturma:**
- `detect_bloom_level()` → Bloom seviyesi tespit edilir
- `generate_bloom_instructions()` → Özel talimatlar oluşturulur
- Talimatlar `pedagogical_instructions` string'ine eklenir

### 3. Cognitive Load Parametreleri → Prompt'a Yansıma

| Parametre | Değer | Prompt'a Yansıması |
|-----------|-------|-------------------|
| `total_load` | 0.089 | → Bilişsel Yük bölümünde gösterilir |
| `needs_simplification` | false | → Bilişsel Yük bölümünde gösterilir (simplification talimatı eklenmez) |

### 4. Student Profile Parametreleri → Prompt'a Yansıma

| Parametre | Değer | Prompt'a Yansıması |
|-----------|-------|-------------------|
| `understanding_level` | "intermediate" | → Öğrenci Profili bölümünde gösterilir |
| `explanation_style` | "balanced" | → Öğrenci Profili bölümünde gösterilir (talimat eklenmez) |
| `needs_examples` | false | → Öğrenci Profili bölümünde gösterilir (örnek talimatı eklenmez) |

## Tam Prompt Örneği (Bu Sorgu İçin)

```
Sen bir eğitim asistanısın. Aşağıdaki cevabı öğrencinin öğrenme profiline ve pedagojik analiz sonuçlarına göre kişiselleştir.

📊 ÖĞRENCİ PROFİLİ:
- Anlama Seviyesi: intermediate
- Zorluk Seviyesi: elementary
- Açıklama Stili: balanced
- Örnekler Gerekli: Hayır

🎯 ZPD (Zone of Proximal Development):
- Mevcut Seviye: intermediate
- Önerilen Seviye: elementary
- Başarı Oranı: 0.0%

🧠 Bloom Taksonomisi:
- Tespit Edilen Seviye: remember (Seviye 1)
- Güven: 100.0%

⚖️ Bilişsel Yük:
- Toplam Yük: 0.09
- Sadeleştirme Gerekli: Hayır

📝 ORİJİNAL SORU:
tomurcuklanma nedir

📄 ORİJİNAL CEVAP:
Yanıt oluşturulamadı. Lütfen tekrar deneyin.

⚠️ ÇOK ÖNEMLİ - DOĞRULUK KURALLARI:
- SADECE orijinal cevapta ve ders materyallerinde bulunan bilgileri kullan
- Orijinal cevapta olmayan yeni bilgiler EKLEME
- Orijinal cevabın içeriğini koru, sadece sunumunu değiştir
- Emin olmadığın bilgileri uydurma veya tahmin etme

🔧 KİŞİSELLEŞTİRME TALİMATLARI:
- Temel kavramları önce açıkla
- Teknik terimleri basit dille açıkla
- Daha basit kelimeler kullan

🎓 PEDAGOJİK TALİMATLAR (ÇOK ÖNEMLİ - MUTLAKA UYGULA):

--- BLOOM SEVİYE TALİMATI ---
Bu soru Bloom Taksonomisi Seviye 1 (remember) gerektiriyor.
Öğrencinin mevcut seviyesi: elementary

📝 Yanıt Stratejisi:
- Kısa, net ve doğrudan tanım ver
- Hafızayı destekleyici ipuçları ekle
- Anahtar kelimeleri vurgula

----------------------------

✅ ÖNEMLİ: Kişiselleştirilmiş cevabı SADECE TÜRKÇE olarak ver. Orijinal cevabın içeriğini koru, ancak sunumunu, detay seviyesini ve zorluk seviyesini öğrenci profiline ve pedagojik talimatlara göre ayarla. Cevabı başlık veya madde listesi olmadan, düz metin olarak ver.

⚠️ ÇOK ÖNEMLİ: Aynı bilgiyi veya cümleyi tekrar etme. Her cümle yeni bir bilgi veya açıklama içermeli. Gereksiz tekrarlardan kaçın.
```

## Sonuç

**Kişiselleştirilmiş prompt var** ve parametreler şu şekilde yansıyor:

1. ✅ **ZPD** → `recommended_level` → Difficulty talimatları
2. ✅ **Bloom** → `level` + `level_index` → Özel Bloom talimatları
3. ✅ **Cognitive Load** → `total_load` + `needs_simplification` → Simplification talimatları (gerekirse)
4. ✅ **Student Profile** → `understanding_level`, `explanation_style`, `needs_examples` → Profil bilgileri

**Ancak raporda görünmüyor** çünkü:
- Bu prompt ayrı bir LLM çağrısında kullanılıyor (`/api/personalization` endpoint'i)
- Debug output'a dahil edilmemiş
- Sadece `pedagogical_instructions` string'i raporda görünüyor (Section [9.5])

**Öneri:** Debug output'a kişiselleştirilmiş prompt'u da eklemek için `personalization.py` dosyasında logging eklenebilir.

