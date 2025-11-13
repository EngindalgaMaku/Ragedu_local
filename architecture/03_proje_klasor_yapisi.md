# Proje Klasör Yapısı

## Kişiselleştirilmiş Ders Notu ve Kaynak Asistanı

### Genel Proje Organizasyonu

Bu klasör yapısı **eğitim odaklı** olarak tasarlanmıştır. Her klasör belirli bir işlevi temsil eder ve öğrencilerin sistemi kolayca anlayıp katkıda bulunabilmelerini sağlar.

```
rag3/
├── README.md                           # Ana proje tanıtımı ve kurulum kılavuzu
├── requirements.txt                    # Python bağımlılıkları
├── .env.example                        # Çevre değişkenleri şablonu
├── .gitignore                          # Git göz ardı dosyaları
├── docker-compose.yml                  # Docker yapılandırması (opsiyonel)
│
├── architecture/                       # 🏗️ Mimari dokümantasyon
│   ├── 01_project_requirements_analysis.md
│   ├── 02_rag_system_architecture.md
│   ├── 03_proje_klasor_yapisi.md
│   ├── 04_teknik_pipeline_tasarimi.md
│   ├── 05_sistem_bilesenleri.md
│   └── diagrams/                       # Sistem diyagramları
│       ├── system_architecture.png
│       ├── data_flow.png
│       └── component_diagram.png
│
├── src/                                # 💻 Ana kaynak kodlar
│   ├── __init__.py
│   ├── main.py                         # FastAPI ana uygulama
│   ├── config/                         # Yapılandırma dosyaları
│   │   ├── __init__.py
│   │   ├── settings.py                 # Ana ayarlar
│   │   └── logging_config.py           # Loglama yapılandırması
│   │
│   ├── core/                           # 🧠 RAG sistem çekirdeği
│   │   ├── __init__.py
│   │   ├── document_processor.py       # Doküman işleme
│   │   ├── text_chunker.py            # Metin parçalama
│   │   ├── embedding_generator.py      # Embedding üretimi
│   │   ├── vector_store.py            # Vektör veritabanı
│   │   ├── retriever.py               # Bilgi geri getirme
│   │   └── response_generator.py       # Yanıt üretimi
│   │
│   ├── api/                            # 🌐 API katmanı
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py           # Doküman yönetimi endpoint'leri
│   │   │   ├── query.py               # Soru-cevap endpoint'leri
│   │   │   └── analytics.py           # Analitik endpoint'leri
│   │   ├── models/                     # Pydantic modeller
│   │   │   ├── __init__.py
│   │   │   ├── document_models.py
│   │   │   ├── query_models.py
│   │   │   └── analytics_models.py
│   │   └── dependencies.py             # API bağımlılıkları
│   │
│   ├── ui/                             # 🎨 Kullanıcı arayüzü
│   │   ├── __init__.py
│   │   ├── streamlit_app.py           # Ana Streamlit uygulaması
│   │   ├── components/                 # UI bileşenleri
│   │   │   ├── __init__.py
│   │   │   ├── document_upload.py     # Doküman yükleme arayüzü
│   │   │   ├── query_interface.py     # Soru sorma arayüzü
│   │   │   ├── analytics_dashboard.py # Analitik paneli
│   │   │   └── system_explanation.py  # Sistem açıklaması
│   │   └── static/                     # Statik dosyalar
│   │       ├── css/
│   │       │   └── custom_styles.css
│   │       └── images/
│   │           └── logo.png
│   │
│   ├── utils/                          # 🔧 Yardımcı fonksiyonlar
│   │   ├── __init__.py
│   │   ├── file_handler.py            # Dosya işlemleri
│   │   ├── text_preprocessing.py      # Metin ön işleme
│   │   ├── similarity_metrics.py      # Benzerlik hesaplamaları
│   │   └── validation.py              # Veri doğrulama
│   │
│   └── analytics/                      # 📊 Analitik ve izleme
│       ├── __init__.py
│       ├── tracker.py                 # Kullanım takibi
│       ├── metrics.py                 # Performans metrikleri
│       └── visualizer.py              # Veri görselleştirme
│
├── data/                               # 📁 Veri dosyaları
│   ├── raw/                           # Ham veriler
│   │   ├── sample_documents/          # Örnek kurs materyalleri
│   │   │   ├── cs101_introduction.pdf
│   │   │   ├── math_basics.pptx
│   │   │   └── programming_guide.docx
│   │   └── README.md                  # Veri açıklamaları
│   │
│   ├── processed/                     # İşlenmiş veriler
│   │   ├── chunks/                    # Metin parçaları
│   │   ├── embeddings/                # Vektör temsilleri
│   │   └── metadata/                  # Meta veriler
│   │
│   └── vector_db/                     # Vektör veritabanı
│       ├── faiss_index/
│       └── chroma_db/
│
├── tests/                              # 🧪 Test dosyaları
│   ├── __init__.py
│   ├── conftest.py                    # Pytest yapılandırması
│   ├── unit/                          # Birim testleri
│   │   ├── __init__.py
│   │   ├── test_document_processor.py
│   │   ├── test_text_chunker.py
│   │   ├── test_embedding_generator.py
│   │   ├── test_retriever.py
│   │   └── test_response_generator.py
│   │
│   ├── integration/                   # Entegrasyon testleri
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   └── test_full_pipeline.py
│   │
│   └── test_data/                     # Test verileri
│       ├── sample_pdf.pdf
│       ├── sample_queries.json
│       └── expected_responses.json
│
├── notebooks/                          # 📓 Jupyter notebook'lar (eğitim amaçlı)
│   ├── 01_veri_kesfetme.ipynb         # Veri analizi ve keşif
│   ├── 02_embedding_deneyimleri.ipynb # Embedding modelleri karşılaştırması
│   ├── 03_retrieval_optimizasyonu.ipynb # Geri getirme optimizasyonu
│   ├── 04_prompt_muhendisligi.ipynb   # Prompt engineering deneyleri
│   └── 05_sistem_performansi.ipynb   # Performans analizi
│
├── scripts/                           # 🔨 Yardımcı scriptler
│   ├── setup_project.py              # Proje kurulum scripti
│   ├── process_documents.py          # Toplu doküman işleme
│   ├── create_sample_data.py         # Örnek veri üretimi
│   ├── benchmark_system.py           # Sistem benchmark'ı
│   └── deploy.py                     # Deployment scripti
│
├── docs/                              # 📚 Dokümantasyon
│   ├── kurulum_kilavuzu.md           # Kurulum talimatları
│   ├── kullanici_kilavuzu.md         # Kullanıcı kılavuzu
│   ├── gelistirici_kilavuzu.md       # Geliştirici kılavuzu
│   ├── api_dokumantasyonu.md         # API referansı
│   ├── sorun_giderme.md              # Troubleshooting
│   └── sss.md                        # Sık sorulan sorular
│
├── research/                          # 🔬 Araştırma materyalleri
│   ├── literature_review/            # Literatür taraması
│   │   ├── related_works.md
│   │   ├── comparison_table.md
│   │   └── references.bib
│   │
│   ├── report/                       # Araştırma raporu
│   │   ├── sections/                 # Rapor bölümleri
│   │   │   ├── 01_giris.md
│   │   │   ├── 02_literatur_taramasi.md
│   │   │   ├── 03_yontem.md
│   │   │   ├── 04_deneysel_tasarim.md
│   │   │   ├── 05_sonuclar.md
│   │   │   ├── 06_kisitlamalar.md
│   │   │   ├── 07_etik_uyumluluk.md
│   │   │   └── 08_kaynaklar.md
│   │   ├── images/                   # Rapor görselleri
│   │   └── main_report.md            # Ana rapor dosyası
│   │
│   └── presentation/                 # Sunum materyalleri
│       ├── slides/                   # Sunum slaytları
│       │   ├── 01_problem_tanimi.md
│       │   ├── 02_yontem.md
│       │   ├── 03_deneyim.md
│       │   ├── 04_sonuclar.md
│       │   └── 05_gelecek_calisma.md
│       ├── assets/                   # Sunum görselleri
│       └── presentation.pptx         # Hazır sunum
│
├── logs/                              # 📝 Log dosyaları
│   ├── application.log
│   ├── api_access.log
│   └── system_performance.log
│
└── outputs/                           # 📤 Çıktı dosyaları
    ├── demo_results/                  # Demo sonuçları
    ├── benchmark_reports/             # Benchmark raporları
    └── analytics_exports/             # Analitik dışa aktarımları
```

## Klasör Açıklamaları

### 🏗️ `architecture/` - Mimari Dokümantasyon

**Amaç:** Sistem mimarisi ve tasarım kararlarının dokümantasyonu

- Sistem gereksinimlerinin analizi
- RAG mimarisinin detaylı tasarımı
- Teknik pipeline açıklamaları
- Sistem bileşenlerinin spesifikasyonları

### 💻 `src/` - Ana Kaynak Kodlar

**Amaç:** Projenin tüm kaynak kodlarının organize edilmesi

#### `core/` - RAG Sistem Çekirdeği

- **Eğitim Odaklı Tasarım:** Her modül bağımsız olarak anlaşılabilir
- **Modüler Yapı:** Bileşenler arası net arayüzler
- **Dokümantasyon:** Extensive code comments ve docstrings

#### `api/` - RESTful API Katmanı

- **FastAPI:** Modern, otomatik dokümantasyon
- **Type Hints:** Kod güvenliği ve eğitim değeri
- **Pydantic Modeller:** Veri validasyonu ve serialization

#### `ui/` - Streamlit Kullanıcı Arayüzü

- **Eğitim Dostu:** Görsel ve interaktif
- **Komponent Tabanlı:** Yeniden kullanılabilir UI parçaları
- **Sistem Açıklaması:** Built-in eğitim materyalleri

### 📁 `data/` - Veri Yönetimi

**Amaç:** Veri pipeline'ının tüm aşamalarını organize etme

- **raw/**: Orijinal dokümanlar ve veri setleri
- **processed/**: İşlenmiş ve hazırlanmış veriler
- **vector_db/**: Vektör veritabanı dosyaları

### 🧪 `tests/` - Test Süitleri

**Amaç:** Sistem kalitesi ve güvenilirliği

- **Birim Testleri:** Her bileşen için ayrı testler
- **Entegrasyon Testleri:** End-to-end sistem testleri
- **Test Verileri:** Tutarlı test senaryoları

### 📓 `notebooks/` - Eğitim Notebook'ları

**Amaç:** Interaktif keşif ve öğrenme materyalleri

- Veri analizi ve görselleştirme
- Model karşılaştırmaları ve optimizasyon
- Konsept açıklamaları ve deneyimler

### 🔬 `research/` - Akademik Materyaller

**Amaç:** Araştırma sürecinin tüm çıktıları

- **Literatür taraması ve referanslar**
- **Akademik rapor bölümleri**
- **Sunum materyalleri**

## Eğitim Odaklı Tasarım Prensipleri

### 1. **Şeffaflık**

- Her klasörde README.md ile açıklama
- Kod organizasyonu net ve anlaşılır
- İşlem adımları görünür ve takip edilebilir

### 2. **Modülerlik**

- Bağımsız çalışabilen komponenler
- Net arayüzler ve bağımlılıklar
- Kolay test ve geliştirme

### 3. **Dokümantasyon**

- Kapsamlı README dosyaları
- Code-level dokümantasyon
- Eğitim materyalleri entegrasyonu

### 4. **Çoğaltılabilirlik**

- Deterministik sonuçlar
- Version control friendly
- Clear dependency management

Bu klasör yapısı, öğrencilerin sistemi kolayca anlayıp geliştirmelerini sağlarken, aynı zamanda akademik standartlara uygun bir proje organizasyonu sunar.
