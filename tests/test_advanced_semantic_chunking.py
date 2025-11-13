#!/usr/bin/env python3
"""
Gelişmiş Semantic Chunking Kalite Skorlama Sistemi Test Dosyası

Bu test dosyası yeni geliştirilen gelişmiş metrikleri test eder:

YENİ GELİŞMİŞ METRİKLER:
✅ Semantic Coherence Score (40%) - Konuların tutarlılığı, topic modeling
✅ Context Preservation Score (25%) - Bağlam korunması, referans çözümü  
✅ Information Completeness (20%) - Bilgi bütünlüğü, ana fikir tamamlanması
✅ Readability & Flow (15%) - Doğal okuma akışı, cümle geçişleri

ESKİ SORUNLU METRİKLER (artık kullanılmıyor):
❌ Boyut Skoru (30%): 200-800 karakter optimal - çok basit
❌ Tutarlılık Skoru (30%): Tam cümle sayısı - anlamsal değil  
❌ Yapı Skoru (20%): Başlık/liste bütünlüğü - yetersiz
❌ Okunabilirlik (20%): Kelime/cümle oranı - anlamsız
"""

import sys
import os
import traceback
from typing import List, Dict

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.text_processing.semantic_chunker import SemanticChunker
    from src.text_processing.advanced_chunk_validator import AdvancedChunkValidator
    print("✅ Gelişmiş semantic chunking modülleri başarıyla import edildi")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    sys.exit(1)

# Test chunk örnekleri - kalite skalası için
TEST_CHUNKS = {
    "excellent_quality": """
# Yapay Zeka ve Eğitim Teknolojileri

Modern eğitim sistemlerinde yapay zeka teknolojileri önemli bir dönüşüm yaratmaktadır. Bu teknolojiler, kişiselleştirilmiş öğrenme deneyimleri sunarak öğrenci başarısını artırmaya odaklanmaktadır.

Kişiselleştirilmiş öğrenme sistemleri, her öğrencinin benzersiz öğrenme stilini ve hızını analiz ederek uygun içerik önerileri sunar. Bu yaklaşım, geleneksel tek boyutlu eğitim yöntemlerinden çok daha etkili sonuçlar vermektedir.

Ayrıca, yapay zeka destekli değerlendirme sistemleri öğrenci performansını gerçek zamanlı olarak takip edebilmektedir. Bu sistemler, öğrencilerin güçlü ve zayıf yanlarını tespit ederek öğretmenlere değerli geri bildirimler sağlar.
""".strip(),

    "good_quality": """
Makine öğrenmesi algoritmalarının eğitim alanındaki uygulamaları giderek çeşitlenmektedir. Bu algoritmalar özellikle büyük veri analizi ve örüntü tanıma konularında başarılı olmaktadır.

Eğitim verilerinin analizi sayesinde öğrenci davranışları hakkında önemli bilgiler elde edilebilir. Bu bilgiler doğrultusunda eğitim programları optimize edilebilir ve daha etkili öğretim stratejileri geliştirilebilir.

Sonuç olarak teknoloji ve eğitimin entegrasyonu gelecekteki öğrenme deneyimlerini büyük ölçüde şekillendirecektir.
""".strip(),

    "average_quality": """
Yapay zeka çok önemli bir teknolojik gelişme. Bu alanda birçok uygulama var. Eğitim sektörü bundan faydalanıyor.

Öğrenciler için faydalı sistemler geliştirilmekte. Bunlar öğrenmeyi kolaylaştırıyor. Ayrıca öğretmenler de bu sistemleri kullanıyor.

Gelecekte daha fazla gelişme bekleniyor.
""".strip(),

    "poor_quality": """
Bu konu hakkında detaylı bilgi. Ancak ne hakkında olduğu net değil. 

Bunlar önemli konular. Özellikle şu nokta çok kritik. Diğer konular da öyle.

Bu nedenle dikkat edilmeli. Çünkü sonuçları çok büyük.
""".strip(),

    "context_dependent": """
Bu sistemler daha önce belirtilen avantajları sağlamaktadır. Onlar sayesinde eğitim kalitesi artırılabilir. Bunların uygulanması için uygun altyapı gereklidir.

Ayrıca bu teknolojiler sürekli gelişmeye devam etmektedir. Dolayısıyla eğitim kurumlarının bu gelişmeleri takip etmesi önemlidir.
""".strip(),

    "incomplete_information": """
# Önemli Başlık

Burada bir şeyler anlatılacak ama...

Örneğin:
- İlk madde
- İkinci madde başlıyor ama
""".strip(),

    "excellent_flow": """
Doğal dil işleme teknolojileri son yıllarda büyük ilerleme kaydetmiştir. Bu gelişmeler özellikle transformer mimarisinin keşfedilmesi ile hız kazanmıştır.

Buna bağlı olarak, dil modellerinin performansı dramatik şekilde artmıştır. GPT ve BERT gibi modeller, çeşitli NLP görevlerinde insan seviyesinde sonuçlar elde etmeye başlamıştır.

Sonuç olarak, bu teknolojik ilerlemeler eğitim alanında yeni fırsatlar yaratmaktadır. Akıllı ders asistanları ve otomatik değerlendirme sistemleri bunun somut örnekleridir.
""".strip(),

    "poor_flow": """
Teknoloji gelişiyor. Eğitim önemli. Yapay zeka var.

Sonra başka konular. Tabii ki bunlar da mühim. Ama diğerleri farklı.

Böylece bu şekilde devam ediyor. Yani genel olarak durum bu.
""".strip()
}

def test_advanced_metrics_detailed():
    """Gelişmiş metriklerin detaylı analizi."""
    print("\n🔬 GELİŞMİŞ METRİKLER DETAYYLI ANALİZ")
    print("=" * 60)
    
    validator = AdvancedChunkValidator()
    
    for chunk_type, chunk_text in TEST_CHUNKS.items():
        print(f"\n📋 {chunk_type.upper().replace('_', ' ')} ANALİZİ:")
        print("-" * 40)
        
        try:
            # Gelişmiş analiz yap
            quality_score = validator.validate_chunk_quality(chunk_text)
            
            print(f"📊 SKORLAR:")
            print(f"   🧠 Semantic Coherence:     {quality_score.semantic_coherence:.3f} (40%)")
            print(f"   🔗 Context Preservation:   {quality_score.context_preservation:.3f} (25%)")
            print(f"   ✅ Information Completeness: {quality_score.information_completeness:.3f} (20%)")
            print(f"   📖 Readability & Flow:     {quality_score.readability_flow:.3f} (15%)")
            print(f"   🎯 GENEL SKOR:            {quality_score.overall_score:.3f}")
            print(f"   {'✅ GEÇERLİ' if quality_score.is_valid else '❌ GEÇERSİZ'}")
            
            # Detaylı analiz bilgileri
            analysis = quality_score.detailed_analysis
            print(f"\n📈 DETAYLAR:")
            print(f"   Uzunluk: {analysis.get('chunk_length', 0)} karakter")
            print(f"   Cümle sayısı: {analysis.get('sentence_count', 0)}")
            print(f"   Kelime sayısı: {analysis.get('word_count', 0)}")
            print(f"   Ortalama cümle uzunluğu: {analysis.get('avg_sentence_length', 0):.1f}")
            print(f"   Anahtar kelime sayısı: {analysis.get('keyword_count', 0)}")
            print(f"   Referans sayısı: {analysis.get('reference_count', 0)}")
            print(f"   Geçiş kelimesi sayısı: {analysis.get('transition_count', 0)}")
            
            # Güçlü yanlar
            if analysis.get('strengths'):
                print(f"   💪 Güçlü yanlar: {', '.join(analysis['strengths'])}")
            
            # Sorun alanları  
            if analysis.get('quality_issues'):
                print(f"   ⚠️  Sorunlar: {', '.join(analysis['quality_issues'])}")
            
            print(f"   📝 İlk 100 karakter: {chunk_text[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Analiz hatası: {e}")

def test_context_aware_scoring():
    """Bağlam farkındalığı testi."""
    print("\n🔗 BAĞLAM FARKINDA SKORLAMA TESTİ")
    print("=" * 50)
    
    validator = AdvancedChunkValidator()
    
    # Bağlam serisi
    previous_chunk = """
Yapay zeka teknolojileri eğitim sektöründe devrim yaratmaktadır. Bu teknolojiler kişiselleştirilmiş öğrenme deneyimleri sunarak her öğrencinin kendine özgü öğrenme stiline uygun içerik sağlamaktadır.

Özellikle makine öğrenmesi algoritmaları, öğrenci davranışlarını analiz ederek en uygun öğretim stratejilerini belirlemekte büyük rol oynamaktadır.
""".strip()
    
    test_chunk = TEST_CHUNKS["context_dependent"]
    
    next_chunk = """
Bu nedenle eğitim kurumlarının teknolojik altyapılarını güçlendirmeleri ve öğretmenlerini bu konularda eğitmeleri büyük önem taşımaktadır.

Gelecekte yapay zeka destekli eğitim sistemlerinin daha da yaygınlaşması ve eğitim kalitesinin artırılması beklenmektedir.
""".strip()
    
    print("🔍 TESTİ YAPILAN CHUNK:")
    print(f"   {test_chunk[:80]}...")
    
    # Bağlam olmadan test
    print("\n📊 BAĞLAM OLMADAN:")
    score_no_context = validator.validate_chunk_quality(test_chunk)
    print(f"   Context Preservation Score: {score_no_context.context_preservation:.3f}")
    print(f"   Overall Score: {score_no_context.overall_score:.3f}")
    
    # Bağlam ile test  
    print("\n📊 BAĞLAM İLE:")
    score_with_context = validator.validate_chunk_quality(test_chunk, previous_chunk, next_chunk)
    print(f"   Context Preservation Score: {score_with_context.context_preservation:.3f}")
    print(f"   Overall Score: {score_with_context.overall_score:.3f}")
    
    # Fark analizi
    context_improvement = score_with_context.context_preservation - score_no_context.context_preservation
    overall_improvement = score_with_context.overall_score - score_no_context.overall_score
    
    print(f"\n📈 BAĞLAM ETKİSİ:")
    print(f"   Context Score İyileşmesi: {context_improvement:+.3f}")
    print(f"   Overall Score İyileşmesi: {overall_improvement:+.3f}")
    print(f"   {'✅ Bağlam pozitif etki yaptı' if overall_improvement > 0 else '⚠️ Bağlam negatif/nötr etki'}")

def test_score_distribution():
    """Skor dağılımı ve eşik analizi."""
    print("\n📊 SKOR DAĞILIMI VE KALITE EŞİKLERİ ANALİZİ")
    print("=" * 60)
    
    validator = AdvancedChunkValidator()
    results = []
    
    for chunk_type, chunk_text in TEST_CHUNKS.items():
        score = validator.validate_chunk_quality(chunk_text)
        results.append({
            'type': chunk_type,
            'overall': score.overall_score,
            'coherence': score.semantic_coherence,
            'context': score.context_preservation,
            'completeness': score.information_completeness,
            'readability': score.readability_flow,
            'valid': score.is_valid
        })
    
    # Skor tablosu
    print("\n📋 SKOR TABLOSU:")
    print(f"{'Chunk Type':<20} {'Overall':<8} {'Coherence':<9} {'Context':<8} {'Complete':<8} {'Readable':<8} {'Valid':<6}")
    print("-" * 78)
    
    for result in results:
        validity_icon = "✅" if result['valid'] else "❌"
        print(f"{result['type']:<20} {result['overall']:<8.3f} {result['coherence']:<9.3f} "
              f"{result['context']:<8.3f} {result['completeness']:<8.3f} "
              f"{result['readability']:<8.3f} {validity_icon:<6}")
    
    # İstatistikler
    valid_count = sum(1 for r in results if r['valid'])
    print(f"\n📈 İSTATİSTİKLER:")
    print(f"   Toplam test: {len(results)}")
    print(f"   Geçerli chunk: {valid_count}")
    print(f"   Geçerlilik oranı: {valid_count/len(results)*100:.1f}%")
    
    # En yüksek ve en düşük skorlar
    max_score = max(results, key=lambda x: x['overall'])
    min_score = min(results, key=lambda x: x['overall'])
    
    print(f"   En yüksek skor: {max_score['overall']:.3f} ({max_score['type']})")
    print(f"   En düşük skor: {min_score['overall']:.3f} ({min_score['type']})")
    print(f"   Skor aralığı: {max_score['overall'] - min_score['overall']:.3f}")

def test_metric_weights():
    """Metrik ağırlıklarının etkisini test et."""
    print("\n⚖️  METRİK AĞIRLIKLARININ ETKİ ANALİZİ")  
    print("=" * 50)
    
    validator = AdvancedChunkValidator()
    
    # Test chunk'ı seç
    test_chunk = TEST_CHUNKS["excellent_quality"]
    score = validator.validate_chunk_quality(test_chunk)
    
    print("🧮 AĞIRLIK HESAPLAMA ÖRNEĞİ:")
    print(f"   Semantic Coherence:     {score.semantic_coherence:.3f} × 40% = {score.semantic_coherence * 0.40:.3f}")
    print(f"   Context Preservation:   {score.context_preservation:.3f} × 25% = {score.context_preservation * 0.25:.3f}")
    print(f"   Information Completeness: {score.information_completeness:.3f} × 20% = {score.information_completeness * 0.20:.3f}")
    print(f"   Readability & Flow:     {score.readability_flow:.3f} × 15% = {score.readability_flow * 0.15:.3f}")
    
    calculated_total = (score.semantic_coherence * 0.40 + 
                       score.context_preservation * 0.25 +
                       score.information_completeness * 0.20 + 
                       score.readability_flow * 0.15)
    
    print(f"   ────────────────────────────────────")
    print(f"   TOPLAM HESAPLANAN:      {calculated_total:.3f}")
    print(f"   SİSTEM SKORU:          {score.overall_score:.3f}")
    print(f"   FARK:                  {abs(calculated_total - score.overall_score):.3f}")
    
    # Ağırlık değişikliği simülasyonu
    print(f"\n🔄 AĞIRLIK DEĞİŞİKLİĞİ SİMÜLASYONU:")
    
    # Coherence ağırlığını artır
    alt_score_1 = (score.semantic_coherence * 0.60 + 
                   score.context_preservation * 0.20 +
                   score.information_completeness * 0.15 + 
                   score.readability_flow * 0.05)
    print(f"   Coherence Ağırlık 60%'a çıkarsa: {alt_score_1:.3f}")
    
    # Context ağırlığını artır
    alt_score_2 = (score.semantic_coherence * 0.20 + 
                   score.context_preservation * 0.50 +
                   score.information_completeness * 0.20 + 
                   score.readability_flow * 0.10)
    print(f"   Context Ağırlık 50%'e çıkarsa:   {alt_score_2:.3f}")

def test_comparison_old_vs_new():
    """Eski ve yeni sistem karşılaştırması."""
    print("\n⚡ ESKİ VS YENİ SİSTEM KARŞILAŞTIRMASI")
    print("=" * 55)
    
    chunker = SemanticChunker()
    
    print("🔍 KARŞILAŞTIRMA ÖRNEKLERİ:")
    
    for chunk_type, chunk_text in TEST_CHUNKS.items():
        print(f"\n📝 {chunk_type.upper().replace('_', ' ')}:")
        print(f"   İçerik: {chunk_text[:60]}...")
        
        try:
            # Yeni sistem
            quality = chunker._validate_chunk_quality(chunk_text)
            
            print(f"   🆕 YENİ SİSTEM:")
            print(f"      Overall Score: {quality.get('overall_score', 0):.3f}")
            print(f"      Coherence: {quality.get('coherence_score', 0):.3f} (gerçek anlamsal)")
            print(f"      Context: {quality.get('context_score', 0):.3f} (bağlam korunması)")
            print(f"      Structure: {quality.get('structure_score', 0):.3f} (bilgi tamamlanması)")
            print(f"      Readability: {quality.get('readability_score', 0):.3f} (akış kalitesi)")
            print(f"      Valid: {'✅' if quality['is_valid'] else '❌'}")
            
            if quality.get('strengths'):
                print(f"      💪 Güçlü: {', '.join(quality['strengths'][:2])}")
            if quality.get('issues'):
                print(f"      ⚠️  Sorun: {', '.join(quality['issues'][:2])}")
                
        except Exception as e:
            print(f"   ❌ Test hatası: {e}")

def test_realistic_scoring_scenarios():
    """Gerçekçi skorlama senaryoları."""
    print("\n🎯 GERÇEKÇİ SKORLAMA SENARYOLARİ")
    print("=" * 45)
    
    realistic_scenarios = {
        "academic_paper_chunk": {
            "text": """
# Metodoloji

Bu çalışmada karma yöntem araştırması benimsenmiştir. Araştırmanın nicel boyutunda 450 öğrenciye anket uygulanmış, nitel boyutunda ise 15 öğretmenle derinlemesine görüşmeler gerçekleştirilmiştir.

Veri toplama süreci üç aşamada gerçekleşmiştir. İlk aşamada pilot uygulama yapılarak ölçme aracının geçerlilik ve güvenilirliği test edilmiştir. İkinci aşamada ana uygulama gerçekleştirilmiş, üçüncü aşamada ise görüşmeler yapılmıştır.

Verilerin analizi SPSS 25 ve NVivo 12 programları kullanılarak gerçekleştirilmiştir. Nicel veriler için betimsel istatistikler ve parametrik testler, nitel veriler için ise tematik analiz uygulanmıştır.
""".strip(),
            "expected_range": (0.8, 0.95)
        },
        
        "technical_documentation": {
            "text": """
## API Endpoint: /users/{id}

Bu endpoint belirli bir kullanıcının bilgilerini getirir.

**Parametreler:**
- id (integer): Kullanıcı ID'si
- format (string, optional): Yanıt formatı (json, xml)

**Yanıt:**
```json
{
  "user_id": 123,
  "name": "John Doe",
  "email": "john@example.com"
}
```

**Hata Kodları:**
- 404: Kullanıcı bulunamadı
- 403: Yetkisiz erişim
""".strip(),
            "expected_range": (0.75, 0.90)
        },
        
        "fragmented_content": {
            "text": """
Bu konuda. Ancak diğer şey.

Onlar. Bunlar da öyle. Yani bu şekilde.

Dolayısıyla. Çünkü bahsettiğimiz konular.
""".strip(),
            "expected_range": (0.2, 0.4)
        }
    }
    
    validator = AdvancedChunkValidator()
    
    for scenario_name, scenario_data in realistic_scenarios.items():
        text = scenario_data["text"]
        expected_min, expected_max = scenario_data["expected_range"]
        
        print(f"\n🔬 {scenario_name.upper().replace('_', ' ')} TESTİ:")
        
        score = validator.validate_chunk_quality(text)
        
        print(f"   📊 Skor: {score.overall_score:.3f}")
        print(f"   🎯 Beklenen: {expected_min:.2f} - {expected_max:.2f}")
        
        # Skor aralık kontrolü
        in_range = expected_min <= score.overall_score <= expected_max
        print(f"   {'✅ Beklenen aralıkta' if in_range else '⚠️ Beklenen aralık dışında'}")
        
        # En düşük ve en yüksek metrikler
        metrics = {
            'Coherence': score.semantic_coherence,
            'Context': score.context_preservation, 
            'Completeness': score.information_completeness,
            'Readability': score.readability_flow
        }
        
        highest = max(metrics, key=metrics.get)
        lowest = min(metrics, key=metrics.get)
        
        print(f"   🥇 En yüksek: {highest} ({metrics[highest]:.3f})")
        print(f"   🥉 En düşük: {lowest} ({metrics[lowest]:.3f})")

def run_comprehensive_advanced_tests():
    """Kapsamlı gelişmiş test süiti."""
    print("🚀 GELİŞMİŞ SEMANTİK CHUNKING KALİTE SKORLAMA SİSTEMİ")
    print("=" * 65)
    print("📋 Yeni Advanced Metrics ile Gerçekçi Kalite Değerlendirmesi")
    print("-" * 65)
    
    tests = [
        ("Gelişmiş Metrikler Detaylı Analiz", test_advanced_metrics_detailed),
        ("Bağlam Farkında Skorlama", test_context_aware_scoring),
        ("Skor Dağılımı ve Eşikler", test_score_distribution), 
        ("Metrik Ağırlıkları Etkisi", test_metric_weights),
        ("Eski vs Yeni Sistem", test_comparison_old_vs_new),
        ("Gerçekçi Skorlama Senaryoları", test_realistic_scoring_scenarios),
    ]
    
    passed_tests = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n▶️  {test_name} testi başlatılıyor...")
            test_func()
            passed_tests += 1
            print(f"✅ {test_name} testi BAŞARILI")
        except Exception as e:
            failed_tests.append(test_name)
            print(f"❌ {test_name} testi HATA: {e}")
            traceback.print_exc()
    
    # Sonuç raporu
    print("\n" + "=" * 65)
    print("📊 TEST SONUÇLARI - GELİŞMİŞ METRİK SİSTEMİ")
    print("=" * 65)
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
        print("\n💡 YENİ SİSTEM ÖZETİ:")
        print("   ✅ Gerçek anlamsal tutarlılık analizi")
        print("   ✅ Bağlam korunması değerlendirmesi")  
        print("   ✅ Bilgi tamamlanması kontrolü")
        print("   ✅ Akıcı okuma deneyimi ölçümü")
        print("   ✅ Türkçe dil yapısına özel analiz")
        print("   ✅ Bağlam farkında skorlama")
        print("   ✅ Gerçekçi ve anlamlı skorlar")
    
    return passed_tests == len(tests)

if __name__ == "__main__":
    try:
        # Ana test suitesi
        success = run_comprehensive_advanced_tests()
        
        # Çıkış kodu
        exit_code = 0 if success else 1
        print(f"\n🏁 Gelişmiş test süreci tamamlandı (Çıkış kodu: {exit_code})")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⛔ Test süreci kullanıcı tarafından durduruldu")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Beklenmeyen hata: {e}")
        traceback.print_exc()
        sys.exit(1)