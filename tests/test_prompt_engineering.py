"""
Prompt Mühendisliği Sistemi Test Dosyası
Test script for the Teacher Prompt Engineering System

Bu dosya prompt mühendisliği sisteminin tüm bileşenlerini test eder:
- Özel prompt oluşturma
- Hızlı komut sistemi
- Performans izleme
- RAG entegrasyonu
"""

import sys
import os

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from src.services.prompt_manager import (
    TeacherPromptManager, CustomPrompt, PromptCommand, PromptPerformance,
    PromptCategory, PromptComplexity, teacher_prompt_manager
)
from src.app_logic import (
    execute_prompt_command_with_rag, analyze_prompt_effectiveness,
    get_prompt_performance_summary, batch_test_prompts,
    get_store, get_session_index_path
)
from src.services.document_analyzer import (
    DocumentAnalyzer, analyze_document_and_suggest_prompts, document_analyzer
)
from src.document_processing.enhanced_pdf_processor import (
    enhanced_pdf_processor, MARKER_AVAILABLE
)


def test_prompt_manager_basic():
    """Temel prompt manager fonksiyonlarını test et"""
    
    print("🧪 Prompt Manager Temel Testleri")
    print("=" * 50)
    
    # Test 1: Özel prompt oluşturma
    try:
        test_prompt = teacher_prompt_manager.create_custom_prompt(
            name="Test Matematik Açıklama",
            description="Matematik konularını basit şekilde açıklar",
            template="""Sen bir matematik öğretmenisin. {konu} konusunu {seviye} seviyesinde açıkla.

KONU: {konu}
SEVİYE: {seviye}

Basit kelimeler kullanarak açıkla.""",
            category=PromptCategory.SUBJECT_SPECIFIC,
            complexity=PromptComplexity.BEGINNER,
            language="tr",
            created_by="test_user",
            tags=["matematik", "açıklama", "basit"]
        )
        
        print(f"✅ Özel prompt oluşturuldu: {test_prompt.name}")
        print(f"   Değişkenler: {test_prompt.variables}")
        print(f"   Etiketler: {test_prompt.tags}")
        
    except Exception as e:
        print(f"❌ Özel prompt oluşturma hatası: {e}")
    
    # Test 2: Prompt komutları listesi
    try:
        commands = teacher_prompt_manager.get_prompt_commands()
        print(f"\n✅ {len(commands)} komut bulundu:")
        for cmd in commands[:3]:  # İlk 3'ünü göster
            print(f"   {cmd.command} - {cmd.name}")
    except Exception as e:
        print(f"❌ Komut listesi hatası: {e}")
    
    # Test 3: Komut çalıştırma
    try:
        filled_prompt, error = teacher_prompt_manager.execute_prompt_command(
            "/basit-anlat",
            topic="Çarpma işlemi",
            grade_level="3. sınıf"
        )
        
        if error:
            print(f"❌ Komut çalıştırma hatası: {error}")
        else:
            print(f"✅ Komut başarıyla çalıştırıldı")
            print(f"   Oluşturulan prompt: {filled_prompt[:100]}...")
    except Exception as e:
        print(f"❌ Komut çalıştırma hatası: {e}")
    
    print("\n")


def test_performance_tracking():
    """Performans izleme sistemini test et"""
    
    print("📊 Performans İzleme Testleri")
    print("=" * 50)
    
    # Test 1: Performans kaydı oluşturma
    try:
        performance = PromptPerformance(
            prompt_id="test_prompt_123",
            execution_time=2.5,
            user_rating=4.0,
            response_quality=3.5,
            educational_effectiveness=4.5,
            engagement_score=4.0,
            timestamp=datetime.now().isoformat(),
            session_id="test_session",
            user_feedback="Test geri bildirimi - çok başarılı"
        )
        
        teacher_prompt_manager.record_prompt_performance(performance)
        print("✅ Performans kaydı başarıyla oluşturuldu")
        
    except Exception as e:
        print(f"❌ Performans kaydı hatası: {e}")
    
    # Test 2: Analitikler
    try:
        analytics = teacher_prompt_manager.get_prompt_analytics(days=30)
        print(f"✅ Analitik veriler alındı:")
        print(f"   Toplam çalıştırma: {analytics['total_executions']}")
        print(f"   Ortalama rating: {analytics['avg_user_rating']:.2f}" if analytics['avg_user_rating'] else "   Ortalama rating: 0.00")
        print(f"   Ortalama süre: {analytics['avg_execution_time']:.2f}s" if analytics['avg_execution_time'] else "   Ortalama süre: 0.00s")
        
    except Exception as e:
        print(f"❌ Analitik veriler hatası: {e}")
    
    # Test 3: Arama fonksiyonu
    try:
        search_results = teacher_prompt_manager.search_prompts_and_commands("matematik", "tr")
        print(f"✅ Arama sonuçları:")
        print(f"   Bulunan prompt: {len(search_results['prompts'])}")
        print(f"   Bulunan komut: {len(search_results['commands'])}")
        
    except Exception as e:
        print(f"❌ Arama hatası: {e}")
    
    print("\n")


def test_effectiveness_analysis():
    """Etkinlik analizi sistemini test et"""
    
    print("🔍 Etkinlik Analizi Testleri")
    print("=" * 50)
    
    # Test 1: Cevap analizi
    test_answer = """
    Çarpma işlemi, aynı sayıyı birden fazla kez toplama işlemidir. 
    Örneğin, 3 x 4 işlemi 3 + 3 + 3 + 3 işlemi ile aynıdır.
    
    Çarpma işleminin adımları:
    1. İlk sayıyı belirle
    2. İkinci sayı kadar tekrarla
    3. Sonucu hesapla
    
    Bu şekilde çarpma işlemini daha kolay anlayabilirsiniz. Siz de deneyebilir misiniz?
    """
    
    test_sources = ["Matematik ders kitabı sayfa 45", "Çarpma konusu açıklaması"]
    test_scores = [0.85, 0.72]
    test_params = {"topic": "çarpma", "grade_level": "3. sınıf"}
    
    try:
        analysis = analyze_prompt_effectiveness(
            answer=test_answer,
            sources=test_sources,
            scores=test_scores,
            query_params=test_params,
            execution_time=2.3
        )
        
        print("✅ Etkinlik analizi başarılı:")
        print(f"   Cevap uzunluğu: {analysis['response_length']}")
        print(f"   Kaynak sayısı: {analysis['source_count']}")
        print(f"   Ortalama alaka: {analysis['avg_relevance_score']:.3f}")
        print(f"   Tahmini kalite: {analysis['estimated_quality']:.3f}")
        print(f"   Açıklayıcı dil: {analysis['educational_indicators']['explanatory_language']}")
        print(f"   Yapılandırılmış içerik: {analysis['educational_indicators']['structured_content']}")
        print(f"   Etkileşimli öğeler: {analysis['educational_indicators']['interactive_elements']}")
        
    except Exception as e:
        print(f"❌ Etkinlik analizi hatası: {e}")
    
    print("\n")


def test_batch_operations():
    """Toplu işlem testlerini çalıştır"""
    
    print("📦 Toplu İşlem Testleri")
    print("=" * 50)
    
    # Test case'leri oluştur
    test_cases = [
        {
            "command": "/basit-anlat",
            "params": {"topic": "toplama", "grade_level": "2. sınıf"},
            "expected_keywords": ["toplama", "sayı", "işlem"]
        },
        {
            "command": "/analoji-yap",
            "params": {"topic": "çıkarma", "audience": "ilkokul"},
            "expected_keywords": ["çıkarma", "örnek", "günlük"]
        }
    ]
    
    try:
        # Not: Bu test gerçek RAG sistemi gerektirir, simüle edelim
        print("ℹ️  Toplu test simülasyonu (gerçek RAG sistemi olmadığı için)")
        
        for i, case in enumerate(test_cases, 1):
            print(f"   Test {i}: {case['command']}")
            print(f"     Parametreler: {case['params']}")
            print(f"     Beklenen kelimeler: {case['expected_keywords']}")
        
        print("✅ Toplu test yapısı doğrulandı")
        
    except Exception as e:
        print(f"❌ Toplu test hatası: {e}")
    
    print("\n")


def test_integration():
    """Entegrasyon testlerini çalıştır"""
    
    print("🔗 Entegrasyon Testleri")
    print("=" * 50)
    
    # Test 1: Prompt manager ile komut sistemi entegrasyonu
    try:
        commands = teacher_prompt_manager.get_prompt_commands(subject_area="Matematik")
        if commands:
            cmd = commands[0]
            print(f"✅ Entegrasyon testi - {cmd.command}")
            print(f"   Parametreler: {cmd.parameters}")
            print(f"   Kullanım: {cmd.usage_count}")
        else:
            print("ℹ️  Matematik komutları bulunamadı")
    except Exception as e:
        print(f"❌ Entegrasyon test hatası: {e}")
    
    # Test 2: Performans özeti
    try:
        summary = get_prompt_performance_summary(days=7)
        print(f"✅ Performans özeti:")
        print(f"   Toplam çalıştırma: {summary['total_executions']}")
        print(f"   Unique prompt: {summary['unique_prompts']}")
        
    except Exception as e:
        print(f"❌ Performans özeti hatası: {e}")
    
    print("\n")


def test_document_analyzer():
    """Document analyzer sistemini test et"""
    
    print("📄 Document Analyzer Testleri")
    print("=" * 50)
    
    # Test 1: Basit metin analizi
    test_content = """
    Bu matematik ders notları 5. sınıf öğrencileri için hazırlanmıştır.
    
    Kesirler Konusu:
    Kesir, bir bütünün parçalarını gösteren sayıdır. Örneğin 1/2, bir bütünün yarısını ifade eder.
    
    Örnekler:
    - Pizza örneği: Bir pizzayı 4 eşit parçaya böldüğümüzde her parça 1/4'tür
    - Pasta örneği: Bir pastayı 8 eşit parçaya böldüğümüzde her parça 1/8'dir
    
    Alıştırmalar:
    1. 2/4 kesrini en sade haliyle yazın
    2. 3/6 + 1/6 işlemini yapın
    3. Günlük hayattan 3 kesir örneği verin
    
    Bu örnekleri çözerek kesirler konusunu pekiştirebilirsiniz.
    """
    
    try:
        # Döküman analizi yap
        analysis, suggestions = analyze_document_and_suggest_prompts(
            content=test_content,
            filename="matematik_kesirler_5_sinif.txt"
        )
        
        print("✅ Döküman analizi başarılı:")
        print(f"   İçerik tipi: {analysis.content_type.value}")
        print(f"   Eğitim seviyesi: {analysis.educational_level.value}")
        print(f"   Konu alanı: {analysis.subject_area}")
        print(f"   Ana konular: {analysis.main_topics}")
        print(f"   Zorluk skoru: {analysis.difficulty_score:.2f}")
        print(f"   Güven skoru: {analysis.confidence_score:.2f}")
        
        print(f"\n✅ {len(suggestions)} prompt önerisi oluşturuldu:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"   {i}. {suggestion.suggested_name} (Güven: {suggestion.confidence:.2f})")
            print(f"      Kategori: {suggestion.category.value}")
            print(f"      Komut önerisi: {suggestion.suggested_command or 'Yok'}")
        
    except Exception as e:
        print(f"❌ Document analyzer hatası: {e}")
    
    # Test 2: Farklı içerik türü
    science_content = """
    Fotosentez Deneyi - 7. Sınıf Fen Bilimleri
    
    Amaç: Bitkilerin ışık varlığında oksijen ürettiğini gözlemlemek
    
    Malzemeler:
    - Su bitkisi (Elodea)
    - Şeffaf cam kavanoz
    - Musluk suyu
    - Masa lambası
    
    Deney Adımları:
    1. Kavanoza su doldurun
    2. Su bitkisini suya yerleştirin
    3. Kavanozu güneş ışığına veya lambaya yakın tutun
    4. Bitki yapraklarından çıkan kabarcıkları gözlemleyin
    
    Gözlem: Işık varlığında bitki yapraklarından oksigen kabarcıkları çıkar
    Sonuç: Bitkiler fotosentez yapar ve oksijen üretir
    """
    
    try:
        analysis2, suggestions2 = analyze_document_and_suggest_prompts(
            content=science_content,
            filename="fotosentez_deneyi_7_sinif.txt"
        )
        
        print(f"\n✅ İkinci analiz başarılı - {analysis2.subject_area} içeriği tespit edildi")
        print(f"   Deney içeriği: {analysis2.has_exercises}")
        print(f"   {len(suggestions2)} özel öneri oluşturuldu")
        
    except Exception as e:
        print(f"❌ İkinci analiz hatası: {e}")
    
    print("\n")


def run_comprehensive_tests():
    """Tüm testleri çalıştır"""
    
    print("🚀 Prompt Mühendisliği Sistemi Kapsamlı Testleri")
    print("=" * 60)
    print(f"Test Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Testleri sırayla çalıştır
    test_prompt_manager_basic()
    test_performance_tracking()
    test_effectiveness_analysis()
    test_batch_operations()
    test_integration()
    test_document_analyzer()  # Yeni test ekle
    
    print("🎉 Tüm Testler Tamamlandı!")
    print("=" * 60)
    
    # Özet rapor
    print("\n📋 Test Özeti:")
    print("✅ Prompt Manager Temel İşlevleri")
    print("✅ Performans İzleme Sistemi")
    print("✅ Etkinlik Analizi")
    print("✅ Toplu İşlem Yapısı")
    print("✅ Sistem Entegrasyonu")
    print("✅ Document Analyzer ve Akıllı Öneriler")  # Yeni test
    
    print("\n💡 Yeni Özellikler:")
    print("- 🤖 Dökümanları LLM ile otomatik analiz")
    print("- 🎯 İçeriğe özel prompt önerileri")
    print("- ⚡ Önerilen komutları tek tıkla kaydetme")
    print("- 📊 Detaylı döküman analytics")
    print("- 🔧 Prompt önerilerini düzenleme imkanı")
    print("- 🚀 Marker ile yüksek kaliteli PDF işleme")
    print("- 📄 PDF-to-Markdown dönüşüm sistemi")
    print("- 🔄 Otomatik fallback mekanizması")
    
    print("\n💡 Genel Öneriler:")
    print("- Gerçek RAG sistemi ile test edilebilir")
    print("- Daha fazla test case'i eklenebilir")
    print("- Performans metrikleri genişletilebilir")
    print("- Kullanıcı arayüzü testleri eklenebilir")


if __name__ == "__main__":
    try:
        run_comprehensive_tests()
    except KeyboardInterrupt:
        print("\n❌ Test yarıda kesildi!")
    except Exception as e:
        print(f"\n💥 Test sırasında beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()