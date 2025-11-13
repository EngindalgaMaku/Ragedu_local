#!/usr/bin/env python3
"""
Gerçek RAG sorgusu ile çıktı temizleyici fonksiyonunu test eder.
"""

import requests
import json
import sys

def test_real_rag_query():
    """Gerçek RAG sorgusu ile test yapar."""
    
    api_gateway_url = "http://localhost:8001"
    
    print("🧪 Gerçek RAG Sorgusu ile Çıktı Temizleyici Testi")
    print("=" * 60)
    
    # Test senaryoları - gerçek soru ve bağlam
    test_queries = [
        {
            "question": "Atmosferin bileşimi nedir?",
            "session_id": "test_session_output_cleaner"
        },
        {
            "question": "Fotosentez nasıl gerçekleşir?",
            "session_id": "test_session_output_cleaner"
        },
        {
            "question": "Hücre zarının işlevi nedir?",
            "session_id": "test_session_output_cleaner"
        }
    ]
    
    # Her test sorgusu için
    for i, test in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: '{test['question']}'")
        print("-" * 50)
        
        try:
            # RAG sorgusu yap
            response = requests.post(
                f"{api_gateway_url}/query",
                json={
                    "session_id": test["session_id"],
                    "query": test["question"],
                    "top_k": 5,
                    "use_rerank": True,
                    "min_score": 0.1
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                sources = result.get('sources', [])
                
                print(f"📄 Cevap Uzunluğu: {len(answer)} karakter")
                print(f"📂 Kaynak Sayısı: {len(sources)}")
                print(f"✅ Ham Cevap:")
                print(f"'{answer[:200]}...'")
                
                # İç analiz işaretlerini kontrol et
                analysis_indicators = [
                    "Önce", "Sonra", "Adım", "Analiz", "Kontrol", 
                    "System:", "User:", "Bağlam", "inceledim", "tespit"
                ]
                
                found_indicators = []
                for indicator in analysis_indicators:
                    if indicator.lower() in answer.lower():
                        found_indicators.append(indicator)
                
                if found_indicators:
                    print(f"⚠️  İç analiz bulundu: {', '.join(found_indicators)}")
                    print("🧹 Çıktı temizleyici çalışıyor...")
                else:
                    print("✅ Temiz çıktı - iç analiz bulunamadı")
                
                # Temizlenmiş versiyonu göster (eğer varsa)
                if found_indicators:
                    print("🔧 Temizleme öncesi/sonrası karşılaştırması yapılıyor...")
                
            else:
                print(f"❌ API Hatası: {response.status_code}")
                print(f"Hata: {response.text}")
                
        except Exception as e:
            print(f"❌ Bağlantı Hatası: {e}")
    
    print(f"\n{'='*60}")
    print("🏁 Gerçek RAG Test Tamamlandı")

def test_model_inference_directly():
    """Model inference servisini doğrudan test eder."""
    
    model_service_url = "http://localhost:8002"
    
    print("\n🔧 Model Inference Servisini Doğrudan Test")
    print("=" * 50)
    
    # Gerçekçi bağlam metni
    realistic_context = """
    Atmosfer, Dünya'yı çevreleyen gaz karışımıdır. Atmosferin bileşimi şu şekildedir:
    - Azot (%78): En büyük bileşendir
    - Oksijen (%21): Solunumda kullanılır  
    - Argon (%0.93): Asal gaz
    - Karbondioksit (%0.04): Sera gazı
    - Diğer gazlar (%0.03): Neon, helyum vb.
    
    Fotosentez, bitkilerin güneş ışığını kullanarak karbondioksit ve sudan glikoz üretmesi sürecidir.
    Bu süreçte oksijen açığa çıkar.
    """
    
    test_docs = [
        {
            "content": realistic_context,
            "source": "biology_textbook.pdf",
            "page": 1
        }
    ]
    
    test_questions = [
        "Atmosferin en büyük bileşeni nedir?",
        "Fotosentez sonucu ne üretilir?",
        "Oksijen atmosferin yüzde kaçını oluşturur?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Model Test {i}: '{question}'")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{model_service_url}/generate-answer",
                json={
                    "query": question,
                    "docs": test_docs,
                    "model": "llama-3.1-8b-instant",
                    "max_context_chars": 2000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                cleaned_answer = result.get('response', '')
                
                print(f"✅ Temizlenmiş Cevap ({len(cleaned_answer)} karakter):")
                print(f"'{cleaned_answer}'")
                
                # İç analiz hala var mı kontrol et
                analysis_words = ["önce", "sonra", "adım", "analiz", "kontrol", "bağlam",
                                "tespit", "inceleme", "doğrula", "cevaplayacağım"]
                analysis_found = any(word in cleaned_answer.lower() for word in analysis_words)
                
                if analysis_found:
                    found_words = [word for word in analysis_words if word in cleaned_answer.lower()]
                    print(f"⚠️  İç analiz hala mevcut: {', '.join(found_words)}")
                    print("🔧 Çıktı temizleyici daha da geliştirilebilir")
                else:
                    print("✅ Mükemmel! İç analiz başarıyla kaldırıldı")
                    
                # Cevabın kalitesini kontrol et
                if len(cleaned_answer) < 10:
                    print("⚠️  Cevap çok kısa - temizleyici çok agresif olabilir")
                elif len(cleaned_answer) > 150:
                    print("⚠️  Cevap hala uzun - daha fazla temizleme gerekebilir")
                else:
                    print("✅ Cevap uzunluğu optimal")
                    
                # Numaralı adımlar kontrol et
                if any(char in cleaned_answer for char in ['1.', '2.', '3.', '4.']):
                    print("⚠️  Numaralı adımlar hala mevcut")
                else:
                    print("✅ Numaralı adımlar temizlendi")
                    
            else:
                print(f"❌ Model Service Hatası: {response.status_code}")
                print(f"Hata: {response.text}")
                
        except Exception as e:
            print(f"❌ Model Service Bağlantı Hatası: {e}")

if __name__ == "__main__":
    print("🚀 Gerçek RAG Sorgusu ile Çıktı Temizleyici Test")
    print("=" * 60)
    
    # API Gateway kontrolü
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Gateway çalışıyor")
            test_real_rag_query()
        else:
            print("⚠️  API Gateway yanıt vermiyor")
    except:
        print("❌ API Gateway'e bağlanılamıyor")
    
    # Model Inference Service kontrolü
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("✅ Model Inference Service çalışıyor")
            test_model_inference_directly()
        else:
            print("⚠️  Model Inference Service yanıt vermiyor")
    except:
        print("❌ Model Inference Service'e bağlanılamıyor")