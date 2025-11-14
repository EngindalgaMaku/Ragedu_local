#!/usr/bin/env python3
"""
Çıktı temizleyici fonksiyonunu debug eder.
"""

import requests
import re

def debug_clean_function():
    """Çıktı temizleyici fonksiyonunu manuel test eder."""
    
    print("🔍 Çıktı Temizleyici Debug Test")
    print("=" * 50)
    
    # Test için örnek ham çıktı
    sample_raw_output = """1. Bağlam metnindeki sayısal verileri ve önemli bilgileri analiz ediyorum:

- Atmosferin bileşimi şu şekildedir:
 - Azot (%78): En büyük bileşendir
 - Oksigen (%21): Solunumda kullanılır
 - Argon (%0.93): Asal gaz

2. Doğrulanmış bilgileri kullanarak cevabı veriyorum:

Atmosferin en büyük bileşeni Azot'tır."""

    print("📄 Ham Çıktı:")
    print(f"'{sample_raw_output}'")
    print(f"Uzunluk: {len(sample_raw_output)}")
    
    # Manuel temizleme simülasyonu
    print("\n🔧 Manuel Temizleme Adımları:")
    
    original = sample_raw_output.strip()
    
    # 1. Cümleleri ayır
    sentences = []
    for sentence in re.split(r'[.!?]\s*', original):
        sentence = sentence.strip()
        if sentence and len(sentence) > 5:
            sentences.append(sentence)
    
    print(f"1. Toplam cümle sayısı: {len(sentences)}")
    for i, sentence in enumerate(sentences):
        print(f"   {i+1}: '{sentence}'")
    
    # 2. Son 5 cümleye bak
    candidates = sentences[-5:] if len(sentences) >= 5 else sentences
    print(f"2. Son 5 cümle adayı: {len(candidates)}")
    
    # 3. Temiz cümleyi bul
    forbidden_words = [
        'analiz', 'adım', 'kontrol', 'bağlam', 'inceleme', 
        'tespit', 'doğrula', 'önce', 'sonra', 'cevaplayacağım',
        'değerlendirme', 'hesaplay', 'çıkarım', 'sonuç çıkar'
    ]
    
    print("3. Cümle analizi:")
    for i, sentence in enumerate(reversed(candidates)):
        print(f"   Cümle {len(candidates)-i}: '{sentence}'")
        
        # Uzunluk kontrolü
        if len(sentence) < 10:
            print("     ❌ Çok kısa")
            continue
            
        # Numara kontrolü
        if re.match(r'^\d+[\.\)]\s*', sentence):
            print("     ❌ Numara ile başlıyor")
            continue
            
        # Yasaklı kelime kontrolü
        found_forbidden = [word for word in forbidden_words if word in sentence.lower()]
        if found_forbidden:
            print(f"     ❌ Yasaklı kelimeler: {', '.join(found_forbidden)}")
            continue
        else:
            print("     ✅ Temiz cümle bulundu!")
            final_answer = sentence + '.'
            break
    else:
        print("     ⚠️  Temiz cümle bulunamadı, son cümleyi al")
        final_answer = sentences[-1] + '.'
    
    print(f"\n✅ Final Cevap:")
    print(f"'{final_answer}'")
    print(f"Uzunluk: {len(final_answer)}")

def test_with_model_service():
    """Model servisindeki temizleyiciyi test et."""
    
    print("\n🔧 Model Servisindeki Temizleyiciyi Test")
    print("=" * 50)
    
    model_service_url = "http://localhost:8002"
    
    # Basit test
    test_docs = [
        {
            "content": "Azot atmosferin %78'ini oluşturur. Oksijen %21'dir.",
            "source": "test.pdf"
        }
    ]
    
    try:
        response = requests.post(
            f"{model_service_url}/generate-answer",
            json={
                "query": "Atmosferin en büyük bileşeni nedir?",
                "docs": test_docs,
                "model": "llama-3.1-8b-instant",
                "max_context_chars": 500
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            
            print(f"📄 Model Servisi Cevabı:")
            print(f"'{answer}'")
            print(f"Uzunluk: {len(answer)}")
            
            # Analiz kelimelerini kontrol et
            analysis_words = ['analiz', 'bağlam', 'adım', 'kontrol', 'tespit']
            found = [word for word in analysis_words if word in answer.lower()]
            
            if found:
                print(f"⚠️  İç analiz kelimeleri mevcut: {', '.join(found)}")
            else:
                print("✅ Temiz çıktı!")
                
            # Numara kontrolü
            if any(f"{i}." in answer for i in range(1, 6)):
                print("⚠️  Numaralı adımlar mevcut")
            else:
                print("✅ Numaralı adımlar yok")
                
        else:
            print(f"❌ Hata: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}")

if __name__ == "__main__":
    debug_clean_function()
    test_with_model_service()