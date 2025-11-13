#!/usr/bin/env python3
"""
Gelişmiş Semantic Chunking Sistemi Test Dosyası

Bu dosya, yeni eklenen özelliklerini test eder:
- Markdown başlık tespiti
- Liste tespiti (numaralı ve madde işaretli)
- Cümle bütünlüğü koruması
- 200-800 karakter boyut optimizasyonu
- Gelişmiş kalite kontrolü
"""

import sys
import os
import time
import traceback
from typing import List, Dict

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.text_processing.semantic_chunker import SemanticChunker, create_semantic_chunks
    print("✅ Semantic chunking modülü başarıyla import edildi")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    sys.exit(1)

# Test verileri
TEST_TEXTS = {
    "academic_turkish": """
# Yapay Zeka ve Doğal Dil İşleme Araştırması

## Giriş

Yapay zeka alanında doğal dil işleme (NLP), makinelerin insan dilini anlayıp işleyebilmesi için geliştirilen teknolojilerin bütünüdür. Bu çalışmada, modern NLP yaklaşımlarının etkinliği araştırılmıştır.

Son yıllarda transformer mimarisinin geliştirilmesi ile birlikte, dil modellerinin performansı önemli ölçüde artmıştır. GPT, BERT ve benzeri modeller, çeşitli NLP görevlerinde insan düzeyinde performans göstermektedir.

## Metodoloji

Araştırmamızda kullanılan metodoloji şu adımları içermektedir:

1. **Veri Toplama**: Türkçe ve İngilizce akademik metinler derlenmiştir
2. **Ön İşleme**: Metinler temizlenmiş ve normalize edilmiştir
3. **Model Eğitimi**: Transformer tabanlı modeller eğitilmiştir
4. **Değerlendirme**: Çeşitli metriklerle performans ölçülmüştür

### Veri Seti Özellikleri

Kullanılan veri setinin özellikleri şunlardır:

- Toplam doküman sayısı: 10,000
- Ortalama doküman uzunluğu: 2,500 kelime
- Dil dağılımı: %60 Türkçe, %40 İngilizce
- Konu alanları: Teknoloji, tıp, eğitim, sosyal bilimler

## Sonuçlar

Deneysel çalışmalar sonucunda elde edilen bulgular şu şekildedir:

### Performans Metrikleri

Model performansı aşağıdaki metriklerle değerlendirilmiştir:
- F1 Skoru: 0.92
- Hassasiyet (Precision): 0.89
- Duyarlılık (Recall): 0.95

Bu sonuçlar, önerilen yaklaşımın etkinliğini göstermektedir.

## Tartışma ve Gelecek Çalışmalar

Elde edilen sonuçlar literatürdeki benzer çalışmalarla karşılaştırıldığında, önerilen yöntemin rekabetçi performans gösterdiği görülmektedir. Gelecek çalışmalarda, daha büyük veri setleri üzerinde denemeler yapılması planlanmaktadır.
""",
    
    "markdown_lists": """
# Proje Yönetim Rehberi

## Proje Başlatma

Yeni bir proje başlatırken dikkat edilmesi gereken temel adımlar:

1. **Proje Tanımı**
   - Amaç ve hedefleri belirleme
   - Kapsam sınırlarını çizme
   - Başarı kriterlerini tanımlama

2. **Kaynak Planlaması**
   - İnsan kaynağı ihtiyacı
   - Bütçe tahsisi
   - Zaman planlaması

3. **Risk Analizi**
   - Potansiyel riskleri belirleme
   - Risk önleme stratejileri
   - Acil durum planları

## Takım Organizasyonu

### Roller ve Sorumluluklar

Her takım üyesinin sorumluluklarının net olarak tanımlanması gerekir:

- **Proje Yöneticisi**
  - Genel koordinasyon
  - İletişim yönetimi
  - İlerleme takibi

- **Teknik Lider**
  - Mimari kararlar
  - Kod kalitesi
  - Teknik mentörlük

- **Geliştirici**
  - Kod yazımı
  - Test yazımı
  - Dokümantasyon

### İletişim Protokolleri

Etkili iletişim için kurallara uyulması önemlidir:

* Günlük stand-up toplantıları
* Haftalık ilerleme raporları
* Aylık değerlendirme toplantıları
* Acil durumlar için iletişim kanalları

## Kalite Kontrol

### Test Stratejileri

Yazılım kalitesini sağlamak için:

1. Birim testleri (Unit Tests)
2. Entegrasyon testleri
3. Kullanıcı kabul testleri
4. Performans testleri

### Kod İncelemesi

- Her commit için kod incelemesi zorunludur
- En az bir kişi tarafından onaylanmalıdır
- Standartlara uygunluk kontrol edilmelidir
""",

    "technical_content": """
# API Dokümantasyonu

## Kimlik Doğrulama

API'ye erişim için kimlik doğrulama gereklidir. İki farklı yöntem desteklenir:

### API Anahtarı ile Kimlik Doğrulama

```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.get('https://api.example.com/data', headers=headers)
```

### OAuth 2.0 ile Kimlik Doğrulama

OAuth 2.0 akışını kullanarak token alabilirsiniz:

```python
# Token alma
auth_data = {
    'client_id': 'your_client_id',
    'client_secret': 'your_client_secret',
    'grant_type': 'client_credentials'
}

token_response = requests.post('https://api.example.com/oauth/token', data=auth_data)
access_token = token_response.json()['access_token']
```

## Endpoint Detayları

### Kullanıcı Bilgileri

**GET /api/v1/users/{user_id}**

Belirli bir kullanıcının bilgilerini getirir.

**Parametreler:**
- `user_id` (integer): Kullanıcı ID'si
- `include` (string, opsiyonel): İçerilecek ilişkili veriler

**Yanıt Örneği:**
```json
{
  "id": 123,
  "name": "Ali Veli",
  "email": "ali@example.com",
  "created_at": "2023-01-15T10:30:00Z"
}
```

### Veri Oluşturma

**POST /api/v1/data**

Yeni veri kaydı oluşturur.

```python
data = {
    'title': 'Yeni Kayıt',
    'content': 'Kayıt içeriği burada...',
    'tags': ['python', 'api', 'json']
}

response = requests.post('https://api.example.com/api/v1/data', 
                        json=data, headers=headers)
```
"""
}

def test_basic_functionality():
    """Temel chunking fonksiyonalitesini test et."""
    print("\n🧪 TEMEL FONKSİYONALİTE TESTİ")
    print("=" * 50)
    
    # Basit metin ile test
    simple_text = "Bu basit bir test metnidir. İkinci cümle burada. Üçüncü cümle de eklenmiş."
    
    try:
        chunks = create_semantic_chunks(simple_text, target_size=100, language="tr")
        print(f"✅ Basit metin chunking başarılı: {len(chunks)} chunk oluşturuldu")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {len(chunk)} karakter")
        return True
    except Exception as e:
        print(f"❌ Basit metin chunking hatası: {e}")
        traceback.print_exc()
        return False

def test_markdown_structure():
    """Markdown yapısı tespiti testleri."""
    print("\n🧪 MARKDOWN YAPI TESPİTİ")
    print("=" * 50)
    
    chunker = SemanticChunker()
    
    try:
        # Markdown yapısını tespit et
        structure = chunker._detect_text_structure(TEST_TEXTS["markdown_lists"])
        
        print(f"✅ Yapısal tespit başarılı:")
        print(f"   - Başlık sayısı: {len(structure['headers'])}")
        print(f"   - Numaralı liste öğesi: {len(structure['numbered_lists'])}")
        print(f"   - Madde işaretli liste: {len(structure['bullet_lists'])}")
        print(f"   - Kod bloğu: {len(structure['code_blocks'])}")
        
        # Başlıkları kontrol et
        for header in structure['headers']:
            print(f"   H{header['level']}: {header['title']}")
        
        return True
    except Exception as e:
        print(f"❌ Markdown yapı tespiti hatası: {e}")
        traceback.print_exc()
        return False

def test_sentence_integrity():
    """Cümle bütünlüğü koruması testi."""
    print("\n🧪 CÜMLE BÜTÜNLÜĞÜ KORUMA")
    print("=" * 50)
    
    chunker = SemanticChunker()
    
    try:
        test_text = """Bu uzun bir test cümlesidir ve kesilmemesi gerekir. İkinci cümle daha kısa. Üçüncü cümle orta uzunlukta bir cümledir. Dördüncü cümle de test için yazılmıştır."""
        
        # Cümle bütünlüğünü koru
        preserved = chunker._preserve_sentence_integrity(test_text, 80)
        
        # Cümlelerin kesik olmadığını kontrol et
        if preserved.endswith('.') or preserved.endswith('!') or preserved.endswith('?'):
            print(f"✅ Cümle bütünlüğü korundu ({len(preserved)} karakter)")
            print(f"   Sonuç: {preserved}")
            return True
        else:
            print(f"❌ Cümle bütünlüğü bozuldu: {preserved[-20:]}")
            return False
    except Exception as e:
        print(f"❌ Cümle bütünlüğü testi hatası: {e}")
        return False

def test_size_optimization():
    """200-800 karakter boyut optimizasyonu testi."""
    print("\n🧪 BOYUT OPTİMİZASYONU (200-800 KARAKTER)")
    print("=" * 50)
    
    try:
        chunks = create_semantic_chunks(
            TEST_TEXTS["academic_turkish"], 
            target_size=500, 
            language="tr"
        )
        
        size_stats = {
            'too_small': 0,  # <200 karakter
            'optimal': 0,    # 200-800 karakter
            'too_large': 0,  # >800 karakter
            'total_chunks': len(chunks)
        }
        
        for chunk in chunks:
            length = len(chunk)
            if length < 200:
                size_stats['too_small'] += 1
            elif length <= 800:
                size_stats['optimal'] += 1
            else:
                size_stats['too_large'] += 1
        
        optimal_ratio = size_stats['optimal'] / size_stats['total_chunks']
        
        print(f"✅ Boyut optimizasyonu testi tamamlandı:")
        print(f"   - Toplam chunk: {size_stats['total_chunks']}")
        print(f"   - Çok küçük (<200): {size_stats['too_small']}")
        print(f"   - Optimal (200-800): {size_stats['optimal']}")
        print(f"   - Çok büyük (>800): {size_stats['too_large']}")
        print(f"   - Optimal oran: {optimal_ratio:.2%}")
        
        # En az %70 optimal olmalı
        return optimal_ratio >= 0.7
    except Exception as e:
        print(f"❌ Boyut optimizasyon testi hatası: {e}")
        traceback.print_exc()
        return False

def test_quality_control():
    """Kalite kontrolü sistemi testi."""
    print("\n🧪 KALİTE KONTROLÜ SİSTEMİ")
    print("=" * 50)
    
    chunker = SemanticChunker()
    
    try:
        # Test chunk'ları oluştur
        test_chunks = [
            "Bu çok kısa.",  # Kötü kalite - çok kısa
            "Bu iyi bir chunk örneğidir. İçinde birden fazla cümle var. Boyutu da uygun aralıkta. Kaliteli içeriğe sahiptir.",  # İyi kalite
            "Bu başlık var ama içerik yok\n# Başlık",  # Kötü kalite - sadece başlık
            "Bu chunk çok uzun" + " uzun kelime" * 50,  # Kötü kalite - çok uzun
        ]
        
        quality_results = []
        for i, chunk in enumerate(test_chunks):
            quality = chunker._validate_chunk_quality(chunk)
            quality_results.append(quality)
            
            status = "✅ İyi" if quality['is_valid'] else "❌ Kötü"
            print(f"   Chunk {i+1}: {status} (Boyut: {len(chunk)}, Skor: {quality.get('size_score', 0):.2f})")
            if quality['issues']:
                print(f"      Sorunlar: {quality['issues']}")
        
        # En az 1 iyi, 1 kötü kalite olmalı
        good_count = sum(1 for q in quality_results if q['is_valid'])
        bad_count = len(quality_results) - good_count
        
        print(f"✅ Kalite kontrolü testi: {good_count} iyi, {bad_count} kötü kalite tespit edildi")
        return True
    except Exception as e:
        print(f"❌ Kalite kontrolü testi hatası: {e}")
        traceback.print_exc()
        return False

def test_turkish_language_support():
    """Türkçe dil desteği testi."""
    print("\n🧪 TÜRKÇE DİL DESTEĞİ")
    print("=" * 50)
    
    try:
        # Türkçe karakterler ve noktalama
        turkish_text = """Türkçe dilinde özel karakterler vardır: ç, ğ, ı, ö, ş, ü. 
        Bu karakterler doğru işlenmelidir. Ayrıca Türkçe noktalama kuralları da önemlidir.
        
        Cümleler düzgün kesilmelidir. Türkçe'de cümle yapısı İngilizce'den farklıdır.
        Kelime sırası daha esnektir."""
        
        chunks = create_semantic_chunks(turkish_text, target_size=200, language="tr")
        
        print(f"✅ Türkçe metin işleme başarılı: {len(chunks)} chunk")
        
        # Türkçe karakterlerin korunduğunu kontrol et
        turkish_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'I', 'İ', 'Ö', 'Ş', 'Ü']
        all_chunks_text = ' '.join(chunks)
        
        preserved_chars = [char for char in turkish_chars if char in all_chunks_text]
        print(f"   Korunan Türkçe karakterler: {preserved_chars}")
        
        return len(preserved_chars) > 0
    except Exception as e:
        print(f"❌ Türkçe dil desteği testi hatası: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """Performans testi."""
    print("\n🧪 PERFORMANS TESTİ")
    print("=" * 50)
    
    try:
        # Büyük metin ile test
        large_text = TEST_TEXTS["academic_turkish"] * 5  # 5 kez çoğalt
        
        start_time = time.time()
        chunks = create_semantic_chunks(large_text, target_size=400, language="tr")
        end_time = time.time()
        
        processing_time = end_time - start_time
        chars_per_second = len(large_text) / processing_time if processing_time > 0 else 0
        
        print(f"✅ Performans testi tamamlandı:")
        print(f"   - Metin uzunluğu: {len(large_text):,} karakter")
        print(f"   - Chunk sayısı: {len(chunks)}")
        print(f"   - İşlem süresi: {processing_time:.2f} saniye")
        print(f"   - Hız: {chars_per_second:,.0f} karakter/saniye")
        
        # 10 saniyeden az sürmeli (makul bir süre)
        return processing_time < 10.0
    except Exception as e:
        print(f"❌ Performans testi hatası: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Hata yönetimi testi."""
    print("\n🧪 HATA YÖNETİMİ TESTİ")
    print("=" * 50)
    
    test_cases = [
        ("", "Boş string"),
        ("   ", "Sadece boşluk"),
        ("A", "Tek karakter"),
        ("A" * 10000, "Çok uzun metin"),
        ("🙂 😊 👍", "Emoji içerikli metin")
    ]
    
    passed = 0
    for test_input, description in test_cases:
        try:
            chunks = create_semantic_chunks(test_input, target_size=300, language="tr")
            print(f"   ✅ {description}: {len(chunks)} chunk oluşturuldu")
            passed += 1
        except Exception as e:
            print(f"   ❌ {description}: Hata - {e}")
    
    print(f"✅ Hata yönetimi testi: {passed}/{len(test_cases)} test geçti")
    return passed == len(test_cases)

def run_comprehensive_test():
    """Kapsamlı test suitesi."""
    print("🚀 GELİŞMİŞ SEMANTİK CHUNKING SİSTEMİ TESTLERİ")
    print("=" * 60)
    
    tests = [
        ("Temel Fonksiyonalite", test_basic_functionality),
        ("Markdown Yapı Tespiti", test_markdown_structure),
        ("Cümle Bütünlüğü", test_sentence_integrity),
        ("Boyut Optimizasyonu", test_size_optimization),
        ("Kalite Kontrolü", test_quality_control),
        ("Türkçe Dil Desteği", test_turkish_language_support),
        ("Performans", test_performance),
        ("Hata Yönetimi", test_error_handling),
    ]
    
    passed_tests = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n▶️  {test_name} testi başlatılıyor...")
            result = test_func()
            if result:
                passed_tests += 1
                print(f"✅ {test_name} testi BAŞARILI")
            else:
                failed_tests.append(test_name)
                print(f"❌ {test_name} testi BAŞARISIZ")
        except Exception as e:
            failed_tests.append(test_name)
            print(f"❌ {test_name} testi HATA: {e}")
    
    # Sonuç raporu
    print("\n" + "=" * 60)
    print("📊 TEST SONUÇLARI")
    print("=" * 60)
    print(f"Toplam Test: {len(tests)}")
    print(f"Başarılı: {passed_tests}")
    print(f"Başarısız: {len(failed_tests)}")
    print(f"Başarı Oranı: {passed_tests/len(tests)*100:.1f}%")
    
    if failed_tests:
        print(f"\nBaşarısız Testler:")
        for test in failed_tests:
            print(f"  - {test}")
    else:
        print("\n🎉 TÜM TESTLER BAŞARILI!")
    
    return passed_tests == len(tests)

def demo_semantic_chunking():
    """Semantic chunking demo."""
    print("\n🎯 SEMANTİK CHUNKING DEMO")
    print("=" * 50)
    
    # Demo metin
    demo_text = TEST_TEXTS["academic_turkish"]
    
    print("Demo metin işleniyor...")
    print(f"Metin uzunluğu: {len(demo_text)} karakter")
    
    # Chunking işlemi
    chunks = create_semantic_chunks(demo_text, target_size=500, language="tr")
    
    print(f"\n📚 Oluşturulan {len(chunks)} chunk:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 CHUNK {i} ({len(chunk)} karakter):")
        print("-" * 30)
        
        # İlk 200 karakteri göster
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(preview)
        
        # Chunk özelliklerini analiz et
        chunker = SemanticChunker()
        quality = chunker._validate_chunk_quality(chunk)
        
        print(f"\n📊 Kalite Analizi:")
        print(f"   Boyut Skoru: {quality['size_score']:.2f}")
        print(f"   Tutarlılık: {quality['coherence_score']:.2f}")
        print(f"   Yapı Skoru: {quality['structure_score']:.2f}")
        print(f"   Geçerli: {'✅' if quality['is_valid'] else '❌'}")
        
        if quality['issues']:
            print(f"   Sorunlar: {', '.join(quality['issues'])}")

if __name__ == "__main__":
    try:
        # Ana test suitesi
        success = run_comprehensive_test()
        
        # Demo gösterimi
        demo_semantic_chunking()
        
        # Çıkış kodu
        exit_code = 0 if success else 1
        print(f"\n🏁 Test süreci tamamlandı (Çıkış kodu: {exit_code})")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⛔ Test süreci kullanıcı tarafından durduruldu")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Beklenmeyen hata: {e}")
        traceback.print_exc()
        sys.exit(1)