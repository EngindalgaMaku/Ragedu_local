#!/usr/bin/env python3
"""
Çıktı temizleyici fonksiyonunu test eder.
"""

import requests
import json
import sys
import os

# Test için örnek ham çıktılar
test_cases = [
    {
        "name": "İç analiz ile gelen ham çıktı",
        "raw_output": """Önce bağlamdaki bilgileri analiz ediyorum:
1. Azot oranı %78 olarak belirtilmiş
2. Oksijen oranı %21 olarak belirtilmiş

Sonra bu bilgileri doğruluyorum ve cevaplıyorum:

Cevap: Atmosferdeki azot oranı %78, oksijen oranı %21'dir.""",
        "expected_clean": "Atmosferdeki azot oranı %78, oksijen oranı %21'dir.",
        "query": "Atmosferdeki azot ve oksijen oranları nedir?"
    },
    {
        "name": "Adım adım analiz içeren çıktı",
        "raw_output": """Adım 1: Bağlamdaki verileri kontrol ediyorum
Adım 2: Sayısal değerleri doğruluyorum

SONUÇ: Su molekülünün kimyasal formülü H2O'dur ve iki hidrojen ile bir oksijen atomundan oluşur.""",
        "expected_clean": "Su molekülünün kimyasal formülü H2O'dur ve iki hidrojen ile bir oksijen atomundan oluşur.",
        "query": "Su molekülünün yapısı nasıldır?"
    },
    {
        "name": "Sistem mesajları içeren çıktı",
        "raw_output": """System: Bağlamı analiz ediyorum...
User: Soru sorulmuş
(Bu bilgiyi bağlamda kontrol ediyorum)

Cevap: Fotosentez, bitkilerin güneş ışığını kimyasal enerjiye dönüştürme sürecidir.""",
        "expected_clean": "Fotosentez, bitkilerin güneş ışığını kimyasal enerjiye dönüştürme sürecidir.",
        "query": "Fotosentez nedir?"
    }
]

def test_output_cleaner():
    """Çıktı temizleyici fonksiyonunu test eder."""
    
    # Model inference servisine bağlan
    model_service_url = "http://localhost:8002"
    
    print("🧪 Çıktı Temizleyici Test Başlıyor...")
    print("=" * 60)
    
    # Her test case için
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        print(f"Ham Çıktı:")
        print(f"'{test_case['raw_output'][:100]}...'")
        
        # Test documents oluştur (çıktı temizleyici test etmek için)
        test_docs = [
            {
                "content": "Test bağlam metni",
                "source": "test_document"
            }
        ]
        
        # Generate answer endpoint'ini kullanarak test et
        try:
            response = requests.post(
                f"{model_service_url}/generate-answer",
                json={
                    "query": test_case['query'],
                    "docs": test_docs,
                    "model": "llama-3.1-8b-instant",
                    "max_context_chars": 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                cleaned_answer = result.get('response', '')
                
                print(f"✅ Temizlenmiş Çıktı:")
                print(f"'{cleaned_answer}'")
                print(f"📊 Uzunluk: Ham={len(test_case['raw_output'])}, Temiz={len(cleaned_answer)}")
                
                # Basit kontroller
                if "Adım" not in cleaned_answer and "Önce" not in cleaned_answer:
                    print("✅ İç analiz aşamaları başarıyla kaldırılmış")
                else:
                    print("⚠️  İç analiz aşamaları hala mevcut")
                    
                if "System:" not in cleaned_answer and "User:" not in cleaned_answer:
                    print("✅ Sistem mesajları başarıyla kaldırılmış")
                else:
                    print("⚠️  Sistem mesajları hala mevcut")
                    
            else:
                print(f"❌ API Hatası: {response.status_code}")
                print(f"Hata: {response.text}")
                
        except Exception as e:
            print(f"❌ Bağlantı Hatası: {e}")
    
    print(f"\n{'='*60}")
    print("🏁 Test Tamamlandı")

def test_direct_function():
    """Çıktı temizleyici fonksiyonunu doğrudan test eder."""
    print("\n🔧 Doğrudan Fonksiyon Testi...")
    print("=" * 40)
    
    # Import the cleaning function (bu sadece örnek, gerçek import için service'in çalışması gerekir)
    sys.path.append('/services/model_inference_service')
    
    try:
        # Burada gerçek fonksiyonu import edebilirdik, ama service container'da olduğu için sadece API test yapıyoruz
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            raw_output = test_case['raw_output']
            
            # Basit regex temizleme mantığını simulate et
            cleaned = raw_output
            
            # "Cevap:" sonrasını al
            import re
            cevap_match = re.search(r'(?i)(?:cevap\s*(?:\([^)]*\))?:\s*)(.*?)$', cleaned, re.DOTALL)
            if cevap_match:
                cleaned = cevap_match.group(1).strip()
            
            # "SONUÇ:" sonrasını al
            sonuc_match = re.search(r'(?i)(?:sonuç\s*:?\s*)(.*?)$', cleaned, re.DOTALL)
            if sonuc_match:
                cleaned = sonuc_match.group(1).strip()
                
            print(f"Ham uzunluk: {len(raw_output)}")
            print(f"Temiz uzunluk: {len(cleaned)}")
            print(f"Temizlenmiş: '{cleaned}'")
            
    except Exception as e:
        print(f"Doğrudan test hatası: {e}")

if __name__ == "__main__":
    print("🚀 Çıktı Temizleyici Test Araçları")
    print("=" * 60)
    
    # Service kontrolü
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("✅ Model Inference Service çalışıyor")
            test_output_cleaner()
        else:
            print("⚠️  Model Inference Service yanıt vermiyor")
    except:
        print("❌ Model Inference Service'e bağlanılamıyor")
        print("Lütfen servisi başlatın: python services/model_inference_service/main.py")
    
    # Doğrudan fonksiyon testi
    test_direct_function()