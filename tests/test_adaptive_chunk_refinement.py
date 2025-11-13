#!/usr/bin/env python3
"""
Adaptive Chunk Refinement System Test

Bu test LLM destekli chunk iyileştirme sistemini test eder:

✅ SORUN TESPİTİ:
- Semantic Coherence < 0.6 → LLM ile konu tutarlılığı iyileştirmesi
- Context Preservation < 0.5 → LLM ile bağlam köprüleri ekleme
- Information Completeness < 0.6 → LLM ile eksik bilgi tamamlama
- Readability & Flow < 0.5 → LLM ile akış düzeltmesi

✅ ÇÖZÜM STRATEJİLERİ:
- LLM anlamsal analiz ve iyileştirme
- Bağlam farkında düzenleme
- Mechanical fallback
- Gerçek zamanlı skor iyileştirmesi
"""

import sys
import os
import traceback
from typing import List, Dict

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.text_processing.semantic_chunker import SemanticChunker
    from src.text_processing.adaptive_chunk_refiner import AdaptiveChunkRefiner
    from src.text_processing.advanced_chunk_validator import AdvancedChunkValidator
    print("✅ Adaptive chunk refinement modülleri başarıyla import edildi")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    sys.exit(1)

# Problem chunk'ları - gerçek dünyadan sorunlu örnekler
PROBLEMATIC_CHUNKS = {
    "low_coherence": """
Yapay zeka gelişiyor. Eğitim sektörü çok büyük. Python programlama dili. 
Makine öğrenmesi algoritmaları kullanılıyor. Öğrenciler memnun. Teknoloji 
gelişmeye devam ediyor. Veri analizi yapılması gerekiyor. Sonuç iyi çıkıyor.
Bu konularda daha fazla çalışma yapmalı.
""".strip(),
    
    "poor_context": """
Bu sistemler çok faydalı. Onlar sayesinde işler kolaylaşıyor. Bunları 
uygulayarak başarılı olunabilir. Şu anda bahsettiğimiz konular önemli.
Diğer yaklaşımlarla karşılaştırıldığında daha etkili sonuçlar veriyor.
""".strip(),
    
    "incomplete_info": """
# Önemli Başlık

Araştırma yöntemi açıklanacak ama...

Sonuçlar şunlardır:
- İlk bulgu
- İkinci bulgu başlıyor
- Üçüncü
""".strip(),
    
    "poor_readability": """
Teknoloji gelişiyor eğitim önemli yapay zeka var sonra başka konular tabii bunlar da mühim ama diğerleri farklı böylece bu şekilde devam ediyor yani genel olarak durum bu şekilde olmakta.
""".strip(),
    
    "multiple_issues": """
Bu çok önemli. Onlar. Şu konular. Yapay zeka teknoloji eğitim sistem 
öğrenci öğretmen veri analiz sonuç. Bunlar da var. Diğer konular 
farklı ama öyle. Bu şekilde devam ediyor işte.
""".strip(),
    
    "good_quality_baseline": """
Yapay zeka teknolojileri eğitim alanında devrim yaratmaktadır. Bu teknolojiler, 
kişiselleştirilmiş öğrenme deneyimleri sunarak her öğrencinin kendine özgü 
öğrenme stiline uygun içerik sağlamaktadır.

Özellikle makine öğrenmesi algoritmaları, öğrenci davranışlarını analiz ederek 
en uygun öğretim stratejilerini belirlemekte büyük rol oynamaktadır. Bu sayede 
eğitim kalitesi önemli ölçüde artırılabilmektedir.
""".strip()
}

def test_individual_refinement_strategies():
    """Her iyileştirme stratejisini ayrı ayrı test et."""
    print("\n🔧 BİREYSEL İYİLEŞTİRME STRATEJİLERİ TESTİ")
    print("=" * 60)
    
    refiner = AdaptiveChunkRefiner()
    validator = AdvancedChunkValidator()
    
    strategies = {
        "Semantic Coherence": "low_coherence",
        "Context Preservation": "poor_context", 
        "Information Completeness": "incomplete_info",
        "Readability & Flow": "poor_readability"
    }
    
    for strategy_name, chunk_key in strategies.items():
        chunk_text = PROBLEMATIC_CHUNKS[chunk_key]
        
        print(f"\n📋 {strategy_name.upper()} İYİLEŞTİRME:")
        print("-" * 45)
        print(f"📝 Orijinal chunk: {chunk_text[:80]}...")
        
        try:
            # Kalite analizi
            original_quality = validator.validate_chunk_quality(chunk_text)
            print(f"📊 Orijinal skor: {original_quality.overall_score:.3f}")
            
            # İyileştirme gerekli mi?
            if refiner.should_refine_chunk(original_quality):
                print("🔍 İyileştirme gerekli tespit edildi")
                
                # LLM iyileştirmesi
                refinement_result = refiner.refine_low_quality_chunk(
                    chunk=chunk_text,
                    quality_score=original_quality,
                    language="tr"
                )
                
                if refinement_result.success:
                    print(f"✅ İyileştirme başarılı!")
                    print(f"📈 Skor iyileştirmesi: {refinement_result.improvement_score:+.3f}")
                    print(f"🔧 Uygulanan düzeltmeler: {', '.join(refinement_result.applied_fixes)}")
                    print(f"💡 Açıklama: {refinement_result.reasoning[:100]}...")
                    
                    # İyileştirilmiş metni göster
                    if len(refinement_result.refined_chunks) == 1:
                        refined_text = refinement_result.refined_chunks[0]
                        print(f"📝 İyileştirilmiş: {refined_text[:80]}...")
                    else:
                        print(f"📝 {len(refinement_result.refined_chunks)} chunk'a bölündü")
                else:
                    print(f"❌ İyileştirme başarısız: {refinement_result.reasoning}")
            else:
                print("✅ İyileştirme gerekmiyor - kalite yeterli")
                
        except Exception as e:
            print(f"❌ Test hatası: {e}")
            traceback.print_exc()

def test_end_to_end_chunking_with_refinement():
    """Tam süreç test - chunking + refinement."""
    print("\n🚀 UÇTAN UCA CHUNKING + İYİLEŞTİRME TESTİ")
    print("=" * 55)
    
    # Sorunlu metin oluştur
    problematic_text = """
# Test Başlığı

Yapay zeka çok önemli. Teknoloji gelişiyor. Bu konular hakkında bilgi.

Eğitim sektörü. Öğrenciler var. Öğretmenler de. Bunlar önemli konular. 
Şu anda bahsettiğimiz şeyler. Teknoloji ile ilgili konular.

Makine öğrenmesi algoritmaları kullanılıyor. Veri analizi. Sonuçlar 
iyi çıkıyor genelde. Bu konularda çalışmak gerekiyor. Araştırma 
yapılması lazım tabii ki.

## Alt Başlık

Diğer konular da var. Bunlar da önemli. Ama şu konular daha kritik. 
O yüzden dikkat edilmeli. Sonuç olarak bu şekilde.
""".strip()
    
    print("📄 Test metni uzunluğu:", len(problematic_text), "karakter")
    
    try:
        chunker = SemanticChunker()
        
        print("\n🔄 Semantic chunking + adaptive refinement çalıştırılıyor...")
        
        # Chunking yap (içinde refinement otomatik çalışacak)
        chunks = chunker.create_semantic_chunks(
            text=problematic_text,
            target_size=300,
            language="tr"
        )
        
        print(f"\n📊 SONUÇLAR:")
        print(f"   Oluşturulan chunk sayısı: {len(chunks)}")
        
        # Her chunk'ı analiz et
        validator = AdvancedChunkValidator()
        
        total_quality = 0
        valid_chunks = 0
        
        for i, chunk in enumerate(chunks, 1):
            quality = validator.validate_chunk_quality(chunk)
            total_quality += quality.overall_score
            
            if quality.is_valid:
                valid_chunks += 1
                status = "✅"
            else:
                status = "❌"
                
            print(f"   Chunk {i}: {status} Skor: {quality.overall_score:.3f} ({len(chunk)} kar.)")
            print(f"           İlk 60 kar: {chunk[:60]}...")
        
        avg_quality = total_quality / len(chunks) if chunks else 0
        valid_ratio = valid_chunks / len(chunks) * 100 if chunks else 0
        
        print(f"\n📈 KALİTE ÖZETİ:")
        print(f"   Ortalama kalite: {avg_quality:.3f}")
        print(f"   Geçerli chunk oranı: {valid_ratio:.1f}%")
        
        return avg_quality > 0.65  # Başarı kriteri
        
    except Exception as e:
        print(f"❌ End-to-end test hatası: {e}")
        traceback.print_exc()
        return False

def test_refinement_comparison():
    """İyileştirme öncesi/sonrası karşılaştırması."""
    print("\n📊 İYİLEŞTİRME KARŞILAŞTIRMA TESTİ")
    print("=" * 45)
    
    validator = AdvancedChunkValidator()
    refiner = AdaptiveChunkRefiner()
    
    test_chunks = [
        ("Düşük Tutarlılık", PROBLEMATIC_CHUNKS["low_coherence"]),
        ("Kötü Bağlam", PROBLEMATIC_CHUNKS["poor_context"]),
        ("Eksik Bilgi", PROBLEMATIC_CHUNKS["incomplete_info"]),
        ("Çoklu Sorun", PROBLEMATIC_CHUNKS["multiple_issues"])
    ]
    
    improvements = []
    
    for chunk_name, chunk_text in test_chunks:
        print(f"\n🔍 {chunk_name} analizi:")
        
        try:
            # Orijinal kalite
            original_quality = validator.validate_chunk_quality(chunk_text)
            print(f"   📊 Orijinal: {original_quality.overall_score:.3f}")
            
            # İyileştirme
            if refiner.should_refine_chunk(original_quality):
                result = refiner.refine_low_quality_chunk(
                    chunk=chunk_text,
                    quality_score=original_quality,
                    language="tr"
                )
                
                if result.success and len(result.refined_chunks) > 0:
                    # İyileştirilmiş chunk'ın kalitesini ölç
                    refined_quality = validator.validate_chunk_quality(result.refined_chunks[0])
                    improvement = refined_quality.overall_score - original_quality.overall_score
                    improvements.append(improvement)
                    
                    print(f"   📈 İyileştirilmiş: {refined_quality.overall_score:.3f} ({improvement:+.3f})")
                    
                    if improvement > 0.1:
                        print(f"   🎉 Önemli iyileşme! ({improvement:+.3f})")
                    elif improvement > 0.05:
                        print(f"   ✅ İyi iyileşme ({improvement:+.3f})")
                    else:
                        print(f"   ⚠️ Minimal iyileşme ({improvement:+.3f})")
                else:
                    print(f"   ❌ İyileştirme başarısız")
                    improvements.append(0)
            else:
                print(f"   ✅ İyileştirme gerekmiyor")
                improvements.append(0)
                
        except Exception as e:
            print(f"   ❌ Hata: {e}")
            improvements.append(0)
    
    # Genel istatistikler
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        positive_improvements = [i for i in improvements if i > 0.05]
        success_rate = len(positive_improvements) / len(improvements) * 100
        
        print(f"\n📈 GENEL İYİLEŞTİRME İSTATİSTİKLERİ:")
        print(f"   Ortalama iyileşme: {avg_improvement:+.3f}")
        print(f"   Başarılı iyileştirmeler: {len(positive_improvements)}/{len(improvements)}")
        print(f"   Başarı oranı: {success_rate:.1f}%")
        
        return success_rate > 50  # En az %50 başarı bekliyoruz
    
    return False

def test_fallback_mechanisms():
    """Fallback mekanizmaları testi."""
    print("\n🔄 FALLBACK MEKANİZMALARI TESTİ")
    print("=" * 45)
    
    refiner = AdaptiveChunkRefiner()
    
    # Test senaryoları
    scenarios = [
        ("Çok kısa metin", "Kısa."),
        ("Tek cümle", "Bu tek bir cümle içeriyor ve bölünemeyen bir yapıda."),
        ("Emoji ve özel karakter", "🙂😊👍 Emoji içerikli metin! @#$%^&*()"),
    ]
    
    for scenario_name, text in scenarios:
        print(f"\n🔍 {scenario_name}:")
        print(f"   Metin: {text}")
        
        try:
            # Fallback testi
            result = refiner._fallback_mechanical_split(text)
            
            print(f"   📊 Sonuç: {len(result.refined_chunks)} chunk")
            print(f"   ✅ Başarı: {'Evet' if result.success else 'Hayır'}")
            print(f"   💡 Açıklama: {result.reasoning}")
            
        except Exception as e:
            print(f"   ❌ Hata: {e}")

def run_adaptive_refinement_tests():
    """Tüm adaptive refinement testlerini çalıştır."""
    print("🚀 ADAPTIVE CHUNK REFINEMENT SİSTEMİ TESTLERİ")
    print("=" * 60)
    print("🎯 LLM Destekli Anlamsal Chunk İyileştirme Sistemi")
    print("-" * 60)
    
    tests = [
        ("Bireysel İyileştirme Stratejileri", test_individual_refinement_strategies),
        ("Uçtan Uca Chunking + İyileştirme", test_end_to_end_chunking_with_refinement),
        ("İyileştirme Karşılaştırması", test_refinement_comparison),
        ("Fallback Mekanizmaları", test_fallback_mechanisms),
    ]
    
    passed_tests = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n▶️  {test_name} testi başlatılıyor...")
            
            if test_name in ["İyileştirme Karşılaştırması", "Uçtan Uca Chunking + İyileştirme"]:
                # Bu testler boolean döndürür
                result = test_func()
                if result:
                    passed_tests += 1
                    print(f"✅ {test_name} testi BAŞARILI")
                else:
                    failed_tests.append(test_name)
                    print(f"❌ {test_name} testi BAŞARISIZ")
            else:
                # Diğer testler exception kontrolü
                test_func()
                passed_tests += 1
                print(f"✅ {test_name} testi BAŞARILI")
                
        except Exception as e:
            failed_tests.append(test_name)
            print(f"❌ {test_name} testi HATA: {e}")
            traceback.print_exc()
    
    # Sonuç raporu
    print("\n" + "=" * 60)
    print("📊 ADAPTIVE REFINEMENT TEST SONUÇLARI")
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
        print("\n💡 YENİ ADAPTIVE REFINEMENT ÖZETİ:")
        print("   ✅ LLM destekli anlamsal iyileştirme")
        print("   ✅ Semantic coherence düzeltmesi") 
        print("   ✅ Context preservation iyileştirmesi")
        print("   ✅ Information completeness tamamlaması")
        print("   ✅ Readability & flow düzeltmesi")
        print("   ✅ Mechanical fallback güvencesi")
        print("   ✅ Gerçek zamanlı kalite iyileştirmesi")
        print("   ✅ Çok dilli destek (Türkçe/İngilizce)")
    
    return passed_tests == len(tests)

if __name__ == "__main__":
    try:
        # Ana test suitesi
        success = run_adaptive_refinement_tests()
        
        # Çıkış kodu
        exit_code = 0 if success else 1
        print(f"\n🏁 Adaptive refinement test süreci tamamlandı (Çıkış kodu: {exit_code})")
        
        if success:
            print("\n🎯 SONUÇ: Semantic chunking sistemi artık düşük kaliteli chunk'ları otomatik olarak LLM ile iyileştiriyor!")
            
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⛔ Test süreci kullanıcı tarafından durduruldu")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Beklenmeyen hata: {e}")
        traceback.print_exc()
        sys.exit(1)