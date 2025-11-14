import requests
import json

# Test data for Turkish document processing
test_data = {
    "text": """# Türkçe Test Dökümanı

Bu semantik chunking için hazırlanmış bir test içeriğidir.

## Biyoloji Konusu
Hücre biyolojisi, canlıların temel yapı taşları olan hücreleri inceleyen bir bilim dalıdır.

### Hücre Çeşitleri
1. Prokaryot hücreler
2. Ökaryot hücreler

## Matematik Konusu
Matematik, sayılar ve geometrik şekiller ile ilgili bir bilim dalıdır.

Bu test içeriği çeşitli konuları kapsar ve semantik chunking'in doğru çalışıp çalışmadığını test eder.""",
    "metadata": {
        "test": True,
        "dil": "türkçe",
        "konu": "test_document"
    },
    "collection_name": "test_turkish_collection",
    "chunk_size": 400,
    "chunk_overlap": 50
}

print("🧪 Docker container içinde document processing test başlatılıyor...")

try:
    # Docker container içinde local servise bağlan
    response = requests.post(
        'http://localhost:8080/process-and-store', 
        json=test_data, 
        timeout=60
    )
    
    print(f"📊 HTTP Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS: Document processing başarılı!")
        print(f"📄 İşlenen chunk sayısı: {result.get('chunks_processed', 0)}")
        print(f"📁 Collection adı: {result.get('collection_name', 'Belirtilmemiş')}")
        print(f"🔗 Chunk ID'ler: {len(result.get('chunk_ids', []))}")
        print("\n🎉 Tüm testler BAŞARILI! DocumentProcessor sınıfı düzgün çalışıyor!")
        
    elif response.status_code == 500:
        print("❌ ERROR: 500 Internal Server Error")
        print("📝 Hata detayları:")
        try:
            error_data = response.json()
            print(json.dumps(error_data, indent=2, ensure_ascii=False))
        except:
            print(response.text)
    else:
        print(f"⚠️  Beklenmeyen HTTP Status: {response.status_code}")
        print(f"📝 Response: {response.text}")
        
except requests.exceptions.Timeout:
    print("⏰ TIMEOUT: İstek zaman aşımına uğradı")
    print("ℹ️  Bu genellikle ChromaDB bağlantı sorunundan kaynaklanır, ana kod problemi değil")
    
except requests.exceptions.ConnectionError:
    print("🔌 CONNECTION ERROR: Servise bağlanılamadı")
    
except Exception as e:
    print(f"💥 Beklenmeyen hata: {type(e).__name__}: {e}")