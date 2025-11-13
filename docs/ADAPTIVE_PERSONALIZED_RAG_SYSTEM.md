# Aktif Öğrenme ile Kişiselleştirilmiş RAG Sistemi
## Adaptive Personalized RAG System (APRAG)

## 📋 İçindekiler
1. [Genel Bakış](#genel-bakış)
2. [Sistem Mimarisi](#sistem-mimarisi)
3. [Modül Yapısı](#modül-yapısı)
4. [Veri Modeli](#veri-modeli)
5. [API Tasarımı](#api-tasarımı)
6. [Frontend Bileşenleri](#frontend-bileşenleri)
7. [Backend Servisleri](#backend-servisleri)
8. [Veri Akışı](#veri-akışı)
9. [Konfigürasyon ve Yönetim](#konfigürasyon-ve-yönetim)
10. [Güvenlik ve Performans](#güvenlik-ve-performans)
11. [Test Stratejisi](#test-stratejisi)
12. [Uygulama Planı](#uygulama-planı)
13. [Topic-Based Learning Path Tracking](#topic-based-learning-path-tracking)

---

## 🎯 Genel Bakış

### Amaç
Öğrencilerin her sorgu ve cevap etkileşiminden öğrenen, kişiselleştirilmiş RAG cevapları üreten ve aktif öğrenme prensiplerini kullanan bir sistem modülü.

### Temel Özellikler
1. **Öğrenci Etkileşim Kaydı**: Her sorgu ve cevap kaydedilir
2. **Geri Bildirim Toplama**: Öğrenci değerlendirmeleri (anlama, yeterlilik vb.)
3. **Kişiselleştirilmiş Cevap Üretimi**: Öğrenci profiline göre adapte edilmiş cevaplar
4. **Akıllı Öneriler**: Kişiselleştirilmiş soru ve cevap önerileri
5. **Sürekli Öğrenme**: Sistem kendini sürekli geliştirir

### Modülerlik ve Kontrol
- ✅ Admin panelden açıp kapatılabilir
- ✅ Ders oturumu bazında aktif/pasif edilebilir
- ✅ Mevcut sistemi etkilemez (feature flag sistemi)
- ✅ Geriye dönük uyumluluk korunur

---

## 🏗️ Sistem Mimarisi

### Genel Yapı
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Student UI   │  │ Feedback UI  │  │ Admin Panel  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│                    API Gateway                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  APRAG Module Controller (Feature Flag Check)        │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
┌─────────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
│ APRAG Service  │ │ Document    │ │ Model          │
│ (New Module)   │ │ Processing  │ │ Inference      │
│                │ │ Service     │ │ Service        │
└────────┬───────┘ └─────────────┘ └────────────────┘
         │
┌─────────▼──────────────────────────────────────────┐
│              Database (PostgreSQL)                  │
│  ┌──────────────────────────────────────────────┐  │
│  │  student_interactions                        │  │
│  │  student_feedback                            │  │
│  │  student_profiles                            │  │
│  │  personalized_responses                      │  │
│  │  learning_patterns                           │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Modül İzolasyonu
- APRAG modülü ayrı bir servis olarak çalışır
- Feature flag ile kontrol edilir
- Mevcut RAG akışına middleware olarak entegre edilir
- Modül kapalıyken mevcut sistem normal çalışır

---

## 🔧 Modül Yapısı

### 1. Feature Flag Sistemi
```python
# config/feature_flags.py
class FeatureFlags:
    APRAG_ENABLED = "aprag_enabled"  # Global enable/disable
    APRAG_PER_SESSION = "aprag_per_session"  # Session-level control
    APRAG_FEEDBACK_COLLECTION = "aprag_feedback_collection"
    APRAG_PERSONALIZATION = "aprag_personalization"
    APRAG_RECOMMENDATIONS = "aprag_recommendations"
```

### 2. Modül Bileşenleri

#### 2.1. Interaction Logger
- Sorgu ve cevapları kaydeder
- Metadata toplar (timestamp, session_id, user_id, etc.)
- Performans metrikleri kaydeder

#### 2.2. Feedback Collector
- Öğrenci geri bildirimlerini toplar
- Ölçekler: Anlama seviyesi, yeterlilik, memnuniyet
- İsteğe bağlı açık uçlu yorumlar

#### 2.3. Student Profile Manager
- Öğrenci öğrenme profili oluşturur
- Zayıf/güçlü yönleri tespit eder
- Öğrenme stillerini analiz eder

#### 2.4. Personalization Engine
- Öğrenci profiline göre cevap adaptasyonu
- Zorluk seviyesi ayarlama
- Açıklama detay seviyesi optimizasyonu

#### 2.5. Recommendation System
- Kişiselleştirilmiş soru önerileri
- İlgili konu önerileri
- Öğrenme yolu önerileri

#### 2.6. Learning Pattern Analyzer
- Öğrenci davranış kalıplarını analiz eder
- Başarı/başarısızlık trendlerini tespit eder
- Adaptif öğrenme stratejileri önerir

---

## 💾 Veri Modeli

### 1. student_interactions
```sql
CREATE TABLE student_interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    original_response TEXT NOT NULL,
    personalized_response TEXT,  -- NULL if personalization disabled
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER,
    model_used VARCHAR(255),
    chain_type VARCHAR(50),
    sources JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_session (user_id, session_id),
    INDEX idx_timestamp (timestamp)
);
```

### 2. student_feedback
```sql
CREATE TABLE student_feedback (
    feedback_id SERIAL PRIMARY KEY,
    interaction_id INTEGER REFERENCES student_interactions(interaction_id),
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Ölçekler (1-5 Likert)
    understanding_level INTEGER CHECK (understanding_level BETWEEN 1 AND 5),
    answer_adequacy INTEGER CHECK (answer_adequacy BETWEEN 1 AND 5),
    satisfaction_level INTEGER CHECK (satisfaction_level BETWEEN 1 AND 5),
    difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 5),
    
    -- Boolean değerlendirmeler
    topic_understood BOOLEAN,
    answer_helpful BOOLEAN,
    needs_more_explanation BOOLEAN,
    
    -- Açık uçlu geri bildirim
    comment TEXT,
    
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_interaction (interaction_id),
    INDEX idx_user (user_id)
);
```

### 3. student_profiles
```sql
CREATE TABLE student_profiles (
    profile_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Öğrenme metrikleri
    average_understanding DECIMAL(3,2),
    average_satisfaction DECIMAL(3,2),
    total_interactions INTEGER DEFAULT 0,
    total_feedback_count INTEGER DEFAULT 0,
    
    -- Güçlü/zayıf yönler
    strong_topics JSONB,  -- {"topic": "score"}
    weak_topics JSONB,
    
    -- Öğrenme stili
    preferred_explanation_style VARCHAR(50),  -- detailed, concise, examples, etc.
    preferred_difficulty_level VARCHAR(20),   -- beginner, intermediate, advanced
    
    -- Kişiselleştirme ayarları
    personalization_enabled BOOLEAN DEFAULT true,
    feedback_collection_enabled BOOLEAN DEFAULT true,
    
    -- Metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_session (user_id, session_id)
);
```

### 4. personalized_responses
```sql
CREATE TABLE personalized_responses (
    response_id SERIAL PRIMARY KEY,
    interaction_id INTEGER REFERENCES student_interactions(interaction_id),
    user_id VARCHAR(255) NOT NULL,
    
    -- Kişiselleştirme detayları
    original_response TEXT NOT NULL,
    personalized_response TEXT NOT NULL,
    personalization_factors JSONB,  -- Hangi faktörler kullanıldı
    
    -- Adaptasyon detayları
    difficulty_adjustment VARCHAR(20),
    explanation_level VARCHAR(20),
    added_examples BOOLEAN,
    added_visual_aids BOOLEAN,
    
    -- Feedback sonrası güncellemeler
    updated_after_feedback BOOLEAN DEFAULT false,
    feedback_incorporated JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_interaction (interaction_id),
    INDEX idx_user (user_id)
);
```

### 5. learning_patterns
```sql
CREATE TABLE learning_patterns (
    pattern_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Kalıp türü
    pattern_type VARCHAR(50),  -- improvement, struggle, mastery, etc.
    pattern_description TEXT,
    
    -- İlgili konular
    related_topics JSONB,
    
    -- Trend verileri
    trend_data JSONB,  -- Time series data
    
    -- Öneriler
    recommendations JSONB,
    
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score DECIMAL(3,2),
    INDEX idx_user_session (user_id, session_id)
);
```

### 6. recommendations
```sql
CREATE TABLE recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Öneri türü
    recommendation_type VARCHAR(50),  -- question, topic, learning_path, etc.
    
    -- Öneri içeriği
    title VARCHAR(255),
    description TEXT,
    content JSONB,
    
    -- Öncelik ve skor
    priority INTEGER,
    relevance_score DECIMAL(3,2),
    
    -- Durum
    status VARCHAR(20) DEFAULT 'pending',  -- pending, shown, accepted, dismissed
    shown_at TIMESTAMP,
    accepted_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_session (user_id, session_id),
    INDEX idx_status (status)
);
```

---

## 🔌 API Tasarımı

### 1. Feature Flag Endpoints

#### GET /api/aprag/status
```json
{
  "enabled": true,
  "global_enabled": true,
  "session_enabled": true,
  "features": {
    "feedback_collection": true,
    "personalization": true,
    "recommendations": true
  }
}
```

#### POST /api/aprag/toggle
```json
{
  "enabled": true,
  "scope": "global" | "session",
  "session_id": "optional"
}
```

### 2. Interaction Endpoints

#### POST /api/aprag/interactions
```json
{
  "user_id": "string",
  "session_id": "string",
  "query": "string",
  "response": "string",
  "metadata": {}
}
```

#### GET /api/aprag/interactions/:user_id
```json
{
  "interactions": [...],
  "total": 100,
  "page": 1
}
```

### 3. Feedback Endpoints

#### POST /api/aprag/feedback
```json
{
  "interaction_id": 123,
  "understanding_level": 4,
  "answer_adequacy": 5,
  "satisfaction_level": 4,
  "topic_understood": true,
  "answer_helpful": true,
  "comment": "Çok yardımcı oldu"
}
```

#### GET /api/aprag/feedback/:interaction_id
```json
{
  "feedback": {...},
  "profile_impact": {...}
}
```

### 4. Profile Endpoints

#### GET /api/aprag/profile/:user_id
```json
{
  "user_id": "string",
  "average_understanding": 4.2,
  "strong_topics": {...},
  "weak_topics": {...},
  "preferred_style": "detailed"
}
```

#### PUT /api/aprag/profile/:user_id
```json
{
  "personalization_enabled": true,
  "preferred_explanation_style": "detailed"
}
```

### 5. Personalization Endpoints

#### POST /api/aprag/personalize
```json
{
  "user_id": "string",
  "session_id": "string",
  "query": "string",
  "original_response": "string",
  "context": {}
}
```

Response:
```json
{
  "personalized_response": "string",
  "personalization_factors": {...},
  "difficulty_adjustment": "intermediate"
}
```

### 6. Recommendation Endpoints

#### GET /api/aprag/recommendations/:user_id
```json
{
  "recommendations": [
    {
      "type": "question",
      "title": "Önerilen Soru",
      "content": "...",
      "relevance_score": 0.85
    }
  ]
}
```

#### POST /api/aprag/recommendations/:recommendation_id/accept
```json
{
  "status": "accepted"
}
```

### 7. Analytics Endpoints

#### GET /api/aprag/analytics/:user_id
```json
{
  "total_interactions": 50,
  "average_understanding": 4.2,
  "improvement_trend": "positive",
  "learning_patterns": [...]
}
```

---

## 🎨 Frontend Bileşenleri

### 1. Student Components

#### FeedbackModal
- Öğrenci geri bildirimi toplama
- Likert ölçekleri
- Boolean değerlendirmeler
- Açık uçlu yorumlar

#### PersonalizedResponseDisplay
- Kişiselleştirilmiş cevap gösterimi
- Adaptasyon detaylarını gösterir
- Orijinal cevap ile karşılaştırma

#### RecommendationPanel
- Kişiselleştirilmiş öneriler
- Soru önerileri
- Konu önerileri
- Öğrenme yolu önerileri

#### LearningProgressWidget
- Öğrenme ilerlemesi
- Güçlü/zayıf yönler
- Trend grafikleri

### 2. Admin Components

#### APRAGSettingsPanel
- Modül açma/kapama
- Ders oturumu bazında kontrol
- Özellik bazında kontrol

#### APRAGAnalyticsDashboard
- Sistem geneli istatistikler
- Öğrenci bazlı analitikler
- Öğrenme kalıpları görselleştirme

#### FeedbackReviewPanel
- Geri bildirimleri inceleme
- Toplu analiz
- Export özellikleri

---

## ⚙️ Backend Servisleri

### 1. APRAG Service (Yeni)

#### Yapı
```
services/aprag_service/
├── main.py                 # FastAPI app
├── models/
│   ├── interaction.py
│   ├── feedback.py
│   ├── profile.py
│   └── recommendation.py
├── services/
│   ├── interaction_logger.py
│   ├── feedback_collector.py
│   ├── profile_manager.py
│   ├── personalization_engine.py
│   ├── recommendation_engine.py
│   └── pattern_analyzer.py
├── database/
│   ├── migrations/
│   └── queries.py
├── config/
│   └── feature_flags.py
└── utils/
    └── helpers.py
```

#### Ana Servisler

##### InteractionLogger
```python
class InteractionLogger:
    def log_interaction(
        self, 
        user_id: str, 
        session_id: str, 
        query: str, 
        response: str,
        metadata: dict
    ) -> Interaction
```

##### FeedbackCollector
```python
class FeedbackCollector:
    def collect_feedback(
        self, 
        interaction_id: int, 
        feedback_data: FeedbackData
    ) -> Feedback
    
    def update_profile_from_feedback(
        self, 
        user_id: str, 
        feedback: Feedback
    ) -> StudentProfile
```

##### PersonalizationEngine
```python
class PersonalizationEngine:
    def personalize_response(
        self, 
        user_id: str, 
        original_response: str,
        context: dict
    ) -> PersonalizedResponse
    
    def adapt_difficulty(
        self, 
        profile: StudentProfile, 
        response: str
    ) -> str
    
    def adjust_explanation_level(
        self, 
        profile: StudentProfile, 
        response: str
    ) -> str
```

##### RecommendationEngine
```python
class RecommendationEngine:
    def generate_question_recommendations(
        self, 
        user_id: str, 
        session_id: str
    ) -> List[Recommendation]
    
    def generate_topic_recommendations(
        self, 
        user_id: str
    ) -> List[Recommendation]
```

### 2. Mevcut Servislere Entegrasyon

#### Document Processing Service
- RAG cevabı üretildikten sonra APRAG middleware'i çağrılır
- Feature flag kontrolü yapılır
- Kişiselleştirme varsa uygulanır

#### API Gateway
- Feature flag kontrolü
- APRAG endpoint'lerini yönlendirme
- Mevcut akışa middleware ekleme

---

## 🔄 Veri Akışı

### 1. Normal RAG Akışı (APRAG Kapalı)
```
Student Query → API Gateway → Document Processing → Model Inference → Response → Student
```

### 2. APRAG Aktif Akışı
```
Student Query 
  → API Gateway (Feature Flag Check)
  → Document Processing (Original Response)
  → APRAG Service:
      ├─ Interaction Logger (Kayıt)
      ├─ Personalization Engine (Kişiselleştirme)
      └─ Response Enhancement
  → Personalized Response → Student
  → Feedback Collection (Async)
  → Profile Update (Async)
  → Recommendation Generation (Async)
```

### 3. Feedback Akışı
```
Student Feedback
  → Feedback Collector
  → Profile Update
  → Pattern Analysis
  → Recommendation Update
  → Learning Pattern Detection
```

---

## ⚙️ Konfigürasyon ve Yönetim

### 1. Admin Panel Ayarları

#### Global Settings
- APRAG Modülü: Açık/Kapalı
- Geri Bildirim Toplama: Açık/Kapalı
- Kişiselleştirme: Açık/Kapalı
- Öneriler: Açık/Kapalı

#### Session-Level Settings
- Her ders oturumu için ayrı kontrol
- Varsayılan ayarları miras alır

#### Feature-Specific Settings
- Kişiselleştirme parametreleri
- Öneri algoritması ayarları
- Geri bildirim ölçekleri

### 2. Environment Variables
```env
APRAG_ENABLED=true
APRAG_SERVICE_URL=http://aprag-service:8000
APRAG_DB_HOST=localhost
APRAG_DB_NAME=aprag_db
APRAG_FEEDBACK_ENABLED=true
APRAG_PERSONALIZATION_ENABLED=true
APRAG_RECOMMENDATIONS_ENABLED=true
```

### 3. Database Migrations
- Alembic kullanılacak
- Ayrı migration dosyaları
- Geri alınabilir (rollback)

---

## 🔒 Güvenlik ve Performans

### Güvenlik
1. **Veri İzolasyonu**: Öğrenci verileri izole edilir
2. **Yetkilendirme**: Sadece ilgili öğrenci kendi verilerini görebilir
3. **Anonimleştirme**: Analitik için anonimleştirilmiş veri
4. **Rate Limiting**: Feedback ve interaction endpoint'leri için
5. **Input Validation**: Tüm girdiler doğrulanır

### Performans
1. **Async Processing**: Feedback ve profil güncellemeleri async
2. **Caching**: Öğrenci profilleri cache'lenir
3. **Database Indexing**: Sık sorgulanan alanlar indexlenir
4. **Batch Processing**: Toplu işlemler için batch API'ler
5. **Lazy Loading**: Öneriler lazy load edilir

---

## 🧪 Test Stratejisi

### 1. Unit Tests
- Her servis için unit testler
- Mock veritabanı kullanımı
- Edge case'lerin test edilmesi

### 2. Integration Tests
- Servisler arası entegrasyon
- API endpoint testleri
- Veritabanı işlemleri

### 3. Feature Flag Tests
- Modül kapalıyken mevcut sistemin çalışması
- Modül açıkken yeni özelliklerin çalışması
- Geçiş senaryoları

### 4. Performance Tests
- Yük testleri
- Response time testleri
- Database query optimizasyonu

---

## 📅 Uygulama Planı

### Faz 1: Temel Altyapı (1-2 hafta)
1. ✅ Feature flag sistemi
2. ✅ Veritabanı şemaları
3. ✅ APRAG Service temel yapısı
4. ✅ API Gateway entegrasyonu
5. ✅ Admin panel ayarları

### Faz 2: Interaction Logging (1 hafta)
1. ✅ Interaction logger servisi
2. ✅ API endpoint'leri
3. ✅ Frontend entegrasyonu
4. ✅ Testler

### Faz 3: Feedback Collection (1 hafta)
1. ✅ Feedback collector servisi
2. ✅ Feedback UI bileşenleri
3. ✅ Profile update mekanizması
4. ✅ Testler

### Faz 4: Personalization Engine (2 hafta)
1. ✅ Personalization engine
2. ✅ Profile manager
3. ✅ Response adaptation algoritmaları
4. ✅ Frontend gösterimi
5. ✅ Testler

### Faz 5: Recommendation System (2 hafta)
1. ✅ Recommendation engine
2. ✅ Question recommendation algoritması
3. ✅ Topic recommendation algoritması
4. ✅ Frontend recommendation panel
5. ✅ Testler

### Faz 6: Analytics ve Pattern Analysis (1-2 hafta)
1. ✅ Pattern analyzer
2. ✅ Analytics dashboard
3. ✅ Learning pattern detection
4. ✅ Visualization components
5. ✅ Testler

### Faz 7: Topic-Based Learning Path Tracking (2-3 hafta)
1. ⏳ Topic extraction from chunks (LLM-based analysis)
2. ⏳ Question-to-topic mapping
3. ⏳ Progress tracking per topic
4. ⏳ Next topic recommendation
5. ⏳ Teacher topic management UI
6. ⏳ Student progress visualization
7. ⏳ Testler

### Faz 8: Optimizasyon ve Dokümantasyon (1 hafta)
1. ⏳ Performans optimizasyonu
2. ⏳ Güvenlik audit
3. ⏳ Dokümantasyon tamamlama
4. ⏳ Kullanıcı kılavuzu
5. ⏳ Final testler

---

## 📚 Topic-Based Learning Path Tracking

### Genel Bakış

Topic-Based Learning Path Tracking, öğrencilerin ders içeriğindeki konuları sıralı ve yapılandırılmış bir şekilde öğrenmelerini sağlayan bir özelliktir. Sistem, ders oturumundaki chunk'ları analiz ederek konu yapısını otomatik olarak çıkarır ve öğrencinin her sorusunu ilgili konuya eşleştirir. Böylece öğrencinin hangi konuda olduğu, o konudaki ilerlemesi ve sonraki konuya geçmeye hazır olup olmadığı takip edilir.

### Temel Özellikler

1. **Otomatik Konu Çıkarımı**: Chunk'lar LLM ile analiz edilerek konu başlıkları, alt başlıklar ve sıralamalar belirlenir
2. **Soru-Konu Eşleştirme**: Öğrenci soruları otomatik olarak ilgili konuya eşleştirilir
3. **İlerleme Takibi**: Her konu için öğrencinin soru sayısı, anlama seviyesi ve hazır olma durumu takip edilir
4. **Sonraki Konu Önerisi**: Öğrenci bir konuyu yeterince öğrendiğinde sonraki konuya geçiş önerilir
5. **Öğretmen Kontrolü**: Öğretmen konu yapısını görüntüleyebilir, düzenleyebilir ve öğrenci ilerlemelerini takip edebilir

### Veri Modeli

#### Yeni Tablolar

```sql
-- Course Topics Table
CREATE TABLE IF NOT EXISTS course_topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(255) NOT NULL,
    
    -- Topic hierarchy
    topic_title VARCHAR(255) NOT NULL,
    parent_topic_id INTEGER,  -- NULL for main topics
    topic_order INTEGER,  -- Order within parent or session
    
    -- Topic metadata
    description TEXT,
    keywords TEXT,  -- JSON array of keywords
    estimated_difficulty VARCHAR(20),  -- beginner, intermediate, advanced
    estimated_time_minutes INTEGER,
    
    -- Prerequisites
    prerequisites TEXT,  -- JSON array of topic_ids
    
    -- Chunk references
    related_chunk_ids TEXT,  -- JSON array of chunk IDs
    
    -- LLM extraction metadata
    extraction_method VARCHAR(50),  -- llm_analysis, manual, hybrid
    extraction_confidence DECIMAL(3,2),  -- 0.00 to 1.00
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_topic_id) REFERENCES course_topics(topic_id) ON DELETE SET NULL
);

-- Topic Progress Table
CREATE TABLE IF NOT EXISTS topic_progress (
    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    topic_id INTEGER NOT NULL,
    
    -- Progress metrics
    questions_asked INTEGER DEFAULT 0,
    average_understanding DECIMAL(3,2),  -- From feedback
    average_satisfaction DECIMAL(3,2),
    last_question_timestamp TIMESTAMP,
    
    -- Mastery assessment
    mastery_level VARCHAR(20),  -- not_started, learning, mastered, needs_review
    mastery_score DECIMAL(3,2),  -- 0.00 to 1.00
    
    -- Readiness for next topic
    is_ready_for_next BOOLEAN DEFAULT FALSE,
    readiness_score DECIMAL(3,2),  -- 0.00 to 1.00
    
    -- Time tracking
    time_spent_minutes INTEGER DEFAULT 0,
    first_question_timestamp TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, session_id, topic_id),
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
);

-- Question-Topic Mapping Table
CREATE TABLE IF NOT EXISTS question_topic_mapping (
    mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    
    -- Mapping confidence
    confidence_score DECIMAL(3,2),  -- 0.00 to 1.00
    mapping_method VARCHAR(50),  -- llm_classification, embedding_similarity, keyword_match
    
    -- Question analysis
    question_complexity VARCHAR(20),  -- basic, intermediate, advanced
    question_type VARCHAR(50),  -- factual, conceptual, application, analysis
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
);
```

### Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│              Teacher Panel (Session Selection)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  APRAG Settings for Session                           │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ Topic Extraction Trigger                       │   │   │
│  │  └──────────────────┬───────────────────────────┘   │   │
│  └──────────────────────┼───────────────────────────────┘   │
└─────────────────────────┼────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              APRAG Service                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Topic Extraction Engine                              │   │
│  │  ├─ Chunk Analysis (LLM)                             │   │
│  │  ├─ Topic Hierarchy Generation                       │   │
│  │  └─ Topic Ordering & Prerequisites                    │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                      │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  Question Classification Engine                        │   │
│  │  ├─ Question → Topic Mapping (LLM)                    │   │
│  │  ├─ Question Complexity Analysis                      │   │
│  │  └─ Confidence Scoring                                │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                      │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  Progress Tracking Engine                              │   │
│  │  ├─ Topic Progress Calculation                        │   │
│  │  ├─ Mastery Assessment                                │   │
│  │  └─ Readiness for Next Topic                          │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                      │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  Next Topic Recommendation Engine                     │   │
│  │  ├─ Prerequisite Check                                │   │
│  │  ├─ Readiness Evaluation                              │   │
│  │  └─ Topic Recommendation                             │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Student Interface                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Progress Dashboard                                   │   │
│  │  ├─ Current Topic                                     │   │
│  │  ├─ Progress Bar                                      │   │
│  │  ├─ Next Topic Suggestion                            │   │
│  │  └─ Topic Completion Status                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Veri Akışı

#### 1. Topic Extraction Flow (Öğretmen Oturum Seçtiğinde)

```
Teacher Selects Session
  ↓
APRAG Settings Panel Appears
  ↓
Teacher Triggers "Extract Topics"
  ↓
APRAG Service:
  ├─ Fetch all chunks for session
  ├─ Group chunks by similarity (optional)
  ├─ LLM Analysis:
  │  ├─ Extract main topics
  │  ├─ Extract sub-topics
  │  ├─ Determine topic order
  │  ├─ Identify prerequisites
  │  └─ Assign difficulty levels
  ├─ Save to course_topics table
  └─ Return topic structure
  ↓
Teacher Reviews/Edits Topics (Optional)
  ↓
Topics Ready for Student Learning
```

#### 2. Question Classification Flow (Öğrenci Soru Sorduğunda)

```
Student Asks Question
  ↓
RAG Query Processed
  ↓
APRAG Middleware:
  ├─ Question Classification:
  │  ├─ LLM: "Which topic does this question belong to?"
  │  ├─ Embedding Similarity: Compare with topic keywords
  │  └─ Confidence Scoring
  ├─ Map to topic_id
  ├─ Analyze question complexity
  ├─ Save to question_topic_mapping
  └─ Update topic_progress
  ↓
Response Enhanced with Topic Info
  ↓
Student Sees:
  ├─ Answer
  ├─ Current Topic
  ├─ Progress in Topic
  └─ Next Topic Suggestion (if ready)
```

#### 3. Progress Tracking Flow

```
After Each Question:
  ↓
Update topic_progress:
  ├─ Increment questions_asked
  ├─ Update average_understanding (from feedback)
  ├─ Update average_satisfaction
  ├─ Calculate mastery_score:
  │  ├─ Based on questions_asked
  │  ├─ Based on understanding_level
  │  ├─ Based on question_complexity
  │  └─ Based on time_spent
  ├─ Determine mastery_level:
  │  ├─ not_started: questions_asked = 0
  │  ├─ learning: 0 < mastery_score < 0.7
  │  ├─ mastered: mastery_score >= 0.7
  │  └─ needs_review: Recent low understanding
  └─ Calculate readiness_for_next:
     ├─ Check mastery_score >= threshold
     ├─ Check prerequisites completed
     └─ Check minimum questions asked
  ↓
If ready_for_next:
  └─ Recommend next topic
```

### API Tasarımı

#### Topic Management Endpoints

```python
# Topic Extraction
POST /api/aprag/topics/extract
{
    "session_id": "string",
    "extraction_method": "llm_analysis",  # or "manual", "hybrid"
    "options": {
        "include_subtopics": true,
        "min_confidence": 0.7,
        "max_topics": 50
    }
}

Response:
{
    "topics": [
        {
            "topic_id": 1,
            "topic_title": "Kimyasal Bağlar",
            "parent_topic_id": null,
            "topic_order": 1,
            "description": "...",
            "keywords": ["kovalent", "iyonik", "bağ"],
            "estimated_difficulty": "intermediate",
            "prerequisites": [],
            "extraction_confidence": 0.92
        }
    ],
    "total_topics": 15,
    "extraction_time_ms": 3500
}

# Get Topics for Session
GET /api/aprag/topics/session/{session_id}
Response:
{
    "topics": [...],
    "hierarchy": {...}  # Tree structure
}

# Update Topic
PUT /api/aprag/topics/{topic_id}
{
    "topic_title": "string",
    "topic_order": 1,
    "description": "string",
    "prerequisites": [2, 3]
}
```

#### Question Classification Endpoints

```python
# Classify Question to Topic
POST /api/aprag/topics/classify-question
{
    "question": "string",
    "session_id": "string",
    "interaction_id": 123
}

Response:
{
    "topic_id": 5,
    "topic_title": "Kimyasal Bağlar",
    "confidence_score": 0.89,
    "question_complexity": "intermediate",
    "question_type": "conceptual"
}

# Get Question-Topic Mappings
GET /api/aprag/topics/questions/{interaction_id}
Response:
{
    "mappings": [
        {
            "topic_id": 5,
            "topic_title": "Kimyasal Bağlar",
            "confidence_score": 0.89
        }
    ]
}
```

#### Progress Tracking Endpoints

```python
# Get Student Progress
GET /api/aprag/topics/progress/{user_id}/{session_id}
Response:
{
    "progress": [
        {
            "topic_id": 1,
            "topic_title": "Atom Yapısı",
            "mastery_level": "mastered",
            "mastery_score": 0.85,
            "questions_asked": 8,
            "average_understanding": 4.2,
            "is_ready_for_next": true,
            "time_spent_minutes": 45
        }
    ],
    "current_topic": {
        "topic_id": 2,
        "topic_title": "Periyodik Tablo",
        "progress_percentage": 65
    },
    "next_recommended_topic": {
        "topic_id": 3,
        "topic_title": "Kimyasal Bağlar",
        "readiness_score": 0.92
    }
}

# Get Topic Progress Details
GET /api/aprag/topics/{topic_id}/progress/{user_id}
Response:
{
    "topic": {...},
    "progress": {...},
    "questions": [...],  # All questions asked about this topic
    "recommendations": [...]
}
```

### Frontend Bileşenleri

#### 1. Teacher: Topic Management Panel

**Location**: Session detail page, APRAG settings section

**Features**:
- "Extract Topics" button
- Topic hierarchy tree view
- Edit topic titles, order, prerequisites
- Manual topic creation
- Topic activation/deactivation
- Preview extracted topics before saving

**UI Components**:
- `TopicExtractionPanel.tsx`
- `TopicHierarchyTree.tsx`
- `TopicEditor.tsx`

#### 2. Student: Progress Dashboard

**Location**: Main chat interface, sidebar or top bar

**Features**:
- Current topic indicator
- Progress bar for current topic
- Topic completion status (not started, learning, mastered)
- Next topic recommendation card
- Topic navigation (jump to specific topic)

**UI Components**:
- `TopicProgressCard.tsx`
- `NextTopicRecommendation.tsx`
- `TopicNavigation.tsx`

#### 3. Enhanced Question Display

**Features**:
- Show which topic the question belongs to
- Show progress in that topic
- Show related topics

### LLM Prompt Tasarımı

#### Topic Extraction Prompt

```
Sen bir eğitim içeriği analiz uzmanısın. Aşağıdaki ders materyallerini analiz ederek konu yapısını çıkar.

DERS MATERYALLERİ:
{chunks}

LÜTFEN ŞUNLARI YAP:
1. Ana konu başlıklarını belirle (5-15 arası)
2. Her ana konu için alt başlıkları belirle
3. Konuları öğrenme sırasına göre sırala
4. Her konu için önkoşul konuları belirle
5. Her konunun zorluk seviyesini belirle (beginner, intermediate, advanced)
6. Her konu için 3-5 anahtar kelime belirle

ÇIKTI FORMATI (JSON):
{
  "topics": [
    {
      "topic_title": "Ana Konu Başlığı",
      "order": 1,
      "difficulty": "intermediate",
      "keywords": ["kelime1", "kelime2"],
      "prerequisites": [],
      "subtopics": [
        {
          "topic_title": "Alt Konu",
          "order": 1,
          "keywords": ["alt1", "alt2"]
        }
      ],
      "related_chunks": [1, 5, 12]
    }
  ]
}

Sadece JSON çıktısı ver, başka açıklama yapma.
```

#### Question Classification Prompt

```
Aşağıdaki öğrenci sorusunu, verilen konu listesine göre sınıflandır.

ÖĞRENCİ SORUSU:
{question}

KONU LİSTESİ:
{topics}

LÜTFEN ŞUNLARI YAP:
1. Sorunun hangi konuya ait olduğunu belirle
2. Sorunun karmaşıklık seviyesini belirle (basic, intermediate, advanced)
3. Sorunun türünü belirle (factual, conceptual, application, analysis)
4. Güven skoru ver (0.0 - 1.0)

ÇIKTI FORMATI (JSON):
{
  "topic_id": 5,
  "topic_title": "Kimyasal Bağlar",
  "confidence_score": 0.89,
  "question_complexity": "intermediate",
  "question_type": "conceptual",
  "reasoning": "Soruda kovalent bağların özellikleri soruluyor..."
}

Sadece JSON çıktısı ver.
```

### Algoritma Detayları

#### Mastery Score Calculation

```python
def calculate_mastery_score(topic_progress):
    """
    Calculate mastery score (0.0 - 1.0) for a topic
    
    Factors:
    - Questions asked (weight: 0.2)
    - Average understanding (weight: 0.4)
    - Question complexity distribution (weight: 0.2)
    - Time spent (weight: 0.2)
    """
    score = 0.0
    
    # Questions asked component (normalized to 0-1)
    # Minimum 3 questions, optimal 10+ questions
    questions_score = min(1.0, topic_progress.questions_asked / 10.0)
    score += questions_score * 0.2
    
    # Understanding component (from feedback, 1-5 scale)
    if topic_progress.average_understanding:
        understanding_score = (topic_progress.average_understanding - 1) / 4.0
        score += understanding_score * 0.4
    
    # Complexity distribution (more advanced questions = better mastery)
    # This would require analyzing question_topic_mapping
    complexity_score = calculate_complexity_score(topic_progress)
    score += complexity_score * 0.2
    
    # Time spent (normalized, optimal 30-60 minutes)
    if topic_progress.time_spent_minutes:
        time_score = min(1.0, topic_progress.time_spent_minutes / 60.0)
        score += time_score * 0.2
    
    return min(1.0, score)
```

#### Readiness for Next Topic

```python
def is_ready_for_next_topic(topic_progress, next_topic):
    """
    Determine if student is ready to move to next topic
    
    Conditions:
    1. Current topic mastery_score >= 0.7
    2. All prerequisites completed (mastery_score >= 0.7)
    3. Minimum questions asked (>= 3)
    4. Recent activity (last question within 7 days)
    """
    # Check current topic mastery
    if topic_progress.mastery_score < 0.7:
        return False
    
    # Check minimum questions
    if topic_progress.questions_asked < 3:
        return False
    
    # Check prerequisites
    for prereq_topic_id in next_topic.prerequisites:
        prereq_progress = get_topic_progress(user_id, session_id, prereq_topic_id)
        if not prereq_progress or prereq_progress.mastery_score < 0.7:
            return False
    
    # Check recent activity
    if topic_progress.last_question_timestamp:
        days_since_last = (now() - topic_progress.last_question_timestamp).days
        if days_since_last > 7:
            return False
    
    return True
```

### Öğretmen UI Akışı

1. **Session Selection**:
   - Öğretmen bir ders oturumu seçer
   - APRAG ayarları paneli görünür (APRAG enabled ise)

2. **Topic Extraction**:
   - "Konuları Çıkar" butonuna tıklar
   - LLM chunk'ları analiz eder
   - Konu yapısı gösterilir
   - Öğretmen düzenleyebilir/kaydedebilir

3. **Topic Management**:
   - Konu hiyerarşisi ağaç görünümünde
   - Sürükle-bırak ile sıralama
   - Konu düzenleme modal'ı
   - Önkoşul seçimi

### Öğrenci UI Akışı

1. **Progress Indicator**:
   - Chat interface'te üstte veya yanda
   - Mevcut konu gösterilir
   - İlerleme çubuğu
   - Tamamlanan konular listesi

2. **Question Context**:
   - Her soru sorulduğunda hangi konuya ait olduğu gösterilir
   - O konudaki ilerleme güncellenir

3. **Next Topic Recommendation**:
   - Konu tamamlandığında sonraki konu önerilir
   - "Sonraki Konuya Geç" butonu
   - Önkoşul kontrolü mesajı (eğer hazır değilse)

### Performans ve Optimizasyon

1. **Topic Extraction**:
   - Batch processing (tüm chunk'lar bir seferde)
   - Caching (bir kez çıkarıldıktan sonra)
   - Incremental updates (yeni chunk eklendiğinde)

2. **Question Classification**:
   - Fast LLM model kullanımı (llama-3.1-8b-instant)
   - Embedding-based pre-filtering
   - Confidence threshold (0.7 altı manual review)

3. **Progress Calculation**:
   - Async calculation (non-blocking)
   - Caching (her soru sonrası değil, periyodik)
   - Batch updates

### Güvenlik ve Veri İzolasyonu

- Öğrenciler sadece kendi progress'lerini görebilir
- Öğretmenler sadece kendi oturumlarının topic'lerini yönetebilir
- Topic extraction sadece öğretmen tarafından tetiklenebilir
- Rate limiting: Topic extraction için (abuse prevention)

### Test Senaryoları

1. **Topic Extraction**:
   - Farklı ders türlerinde (kimya, matematik, tarih)
   - Farklı chunk sayılarında (10, 50, 200)
   - Hiyerarşik yapı doğruluğu

2. **Question Classification**:
   - Belirsiz sorular (multiple topics)
   - Yeni konular (extraction'ta olmayan)
   - Edge cases (çok kısa/uzun sorular)

3. **Progress Tracking**:
   - Mastery score accuracy
   - Readiness calculation correctness
   - Prerequisite validation

---

## 📝 Notlar

### Mevcut Sistemle Uyumluluk
- Tüm değişiklikler feature flag ile korunur
- Mevcut RAG akışı hiç değişmez
- APRAG kapalıyken hiçbir ek işlem yapılmaz
- Geriye dönük uyumluluk %100 korunur

### Genişletilebilirlik
- Yeni kişiselleştirme algoritmaları eklenebilir
- Yeni öneri türleri eklenebilir
- Yeni geri bildirim ölçekleri eklenebilir
- Plugin mimarisi düşünülebilir

### Makale İçin Önemli Noktalar
- Aktif öğrenme prensipleri
- Kişiselleştirme algoritmaları
- Sürekli gelişim mekanizması
- Öğrenci başarısına etkisi
- Sistem performansı ve ölçeklenebilirlik

---

## 🚀 Başlangıç

Bu dokümantasyon tamamlandıktan sonra:
1. Veritabanı şemaları oluşturulacak
2. APRAG Service temel yapısı kurulacak
3. Feature flag sistemi entegre edilecek
4. Adım adım özellikler eklenecek

**Önemli**: Her faz sonunda mevcut sistemin çalıştığı doğrulanacak ve feature flag ile kontrol edilecek.

