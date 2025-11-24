# Interaction ID ve Konu Bilgisi Açıklaması

## Interaction ID Nedir?

**Interaction ID (örn: 78)**, her öğrenci sorusu için oluşturulan benzersiz bir kayıt numarasıdır.

### Nasıl Oluşturulur?
1. Öğrenci bir soru sorar
2. Sistem RAG ile cevap üretir
3. `createAPRAGInteraction` fonksiyonu çağrılır
4. `student_interactions` tablosuna yeni bir kayıt eklenir
5. Veritabanı otomatik olarak bir `interaction_id` oluşturur (örn: 78)

### Interaction'da Ne Tutulur?
- `user_id`: Öğrenci ID'si
- `session_id`: Oturum ID'si
- `query`: Öğrencinin sorusu
- `original_response`: Sistemin verdiği cevap
- `personalized_response`: Kişiselleştirilmiş cevap (varsa)
- `timestamp`: Soru zamanı
- `sources`: Kullanılan kaynaklar
- `emoji_feedback`: Öğrencinin verdiği emoji geri bildirimi (😊, 👍, 😐, ❌)

## Konu Bilgisi Nasıl Tutulur?

### 1. Konu Sınıflandırması (Topic Classification)

Soru sorulduktan sonra, sistem otomatik olarak soruyu bir konuya sınıflandırır:

**Adımlar:**
1. `classifyQuestion` fonksiyonu çağrılır
2. `classify_question_with_llm` fonksiyonu LLM kullanarak soruyu analiz eder
3. Mevcut konular listesi (`course_topics` tablosu) ile karşılaştırılır
4. En uygun konu seçilir (topic_id, topic_title, confidence_score ile)

**LLM Sınıflandırması:**
- Soru analiz edilir
- Anahtar kelimeler çıkarılır
- Mevcut konularla eşleştirilir
- Güven skoru (confidence_score) hesaplanır (0.0 - 1.0 arası)

### 2. Veritabanında Saklama

Konu bilgisi iki yerde tutulur:

**a) `question_topic_mapping` Tablosu:**
```sql
- interaction_id: 78
- topic_id: 571
- confidence_score: 0.95
- question_complexity: "basic"
- question_type: "factual"
```

**b) `topic_progress` Tablosu:**
```sql
- user_id: 5
- session_id: "32ba88c0..."
- topic_id: 571
- questions_asked: +1 (artırılır)
```

### 3. Chat History'de Gösterim

Konu bilgisi chat history'de şu şekilde gösterilir:
- `topic_id`: Konu ID'si
- `topic_title`: Konu başlığı (örn: "Kan Grupları")
- `confidence_score`: Güven skoru (örn: 0.95 = %95)

## Örnek Akış

1. **Öğrenci sorar**: "Kana rengini ne verir?"
2. **Interaction oluşturulur**: `interaction_id = 78`
3. **Cevap üretilir**: RAG sistemi cevap verir
4. **Konu sınıflandırması**: LLM soruyu analiz eder
   - Topic: "Kan Grupları" (topic_id: 571)
   - Confidence: 0.95
5. **Kayıt edilir**:
   - `question_topic_mapping`: interaction_id=78, topic_id=571
   - `topic_progress`: user_id=5, topic_id=571, questions_asked+=1
6. **Chat history'de gösterilir**: Konu başlığı ve güven skoru ile

## Sorun Giderme

**Konu bilgisi görünmüyorsa:**
1. `classifyQuestion` fonksiyonu çağrıldı mı kontrol edin
2. `question_topic_mapping` tablosunda kayıt var mı kontrol edin
3. `course_topics` tablosunda konu var mı kontrol edin
4. Chat history'den topic bilgisi doğru parse ediliyor mu kontrol edin

