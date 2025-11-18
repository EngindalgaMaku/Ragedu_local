#!/usr/bin/env python3
"""
Test scripti - Advanced Chunking Sistemi Kalite Kontrolü
Bu script Türkçe metinler üzerinde chunking kalitesini test eder.
"""

import sys
import os
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import chunking functionality
try:
    from src.text_processing.text_chunker import chunk_text
    print("✅ Advanced chunking sistemi yüklendi")
except ImportError as e:
    print(f"❌ Advanced chunking sistemi yüklenemedi: {e}")
    print("Fallback basic chunking kullanılacak")
    chunk_text = None

def test_turkish_document_chunking():
    """
    Türkçe doküman chunking kalitesini test et
    """
    print("\n" + "="*80)
    print("🧪 TÜRKÇE DOKÜMAN CHUNKING KALİTE TESTİ")
    print("="*80)
    
    # Test dokümanı - Problematik Türkçe metin (gerçek dünya örneği)
    turkish_test_doc = """
# Atmosferin Yapısı ve Özellikleri

Dünya'nın yüzeyini çevreleyen gaz katmanına **atmosfer** denir. Atmosfer, yaşamın sürekliliği için son derece önemlidir.

## Atmosferin Katmanları

### 1. Troposfer
Yerden itibaren 0-18 km arasındaki katman troposferdir. Bu katmanda:
- Hava sıcaklığı yükseklikle birlikte azalır
- Bulutlar ve yağışlar oluşur  
- İklim olayları meydana gelir
- Canlıların yaşadığı en önemli katmandır

Troposferdeki gazların dağılımı şöyledir:
* Azot (N₂): %78.09
* Oksijen (O₂): %20.95  
* Argon (Ar): %0.93
* Karbondioksit (CO₂): %0.04

### 2. Stratosfer  
18-50 km yükseklik arasında yer alır. Ozon tabakası bu katmanda bulunur.

#### Ozon Tabakasının Önemi
Ozon (O₃) molekülleri güneşten gelen zararlı ultraviyole ışınları emer. Bu sayede:

1. Canlılar UV ışınlarından korunur
2. Dünya yüzeyindeki sıcaklık dengesi sağlanır
3. Ekosistem korunmuş olur

### 3. Mezosfer ve Termosfer
Daha üst katmanlarda basınç ve yoğunluk azalır, sıcaklık değişimleri yaşanır.

## Atmosferin İşlevleri

Atmosfer yaşam için kritik işlevlere sahiptir:

**Koruyucu İşlevler:**
- Meteoritlerden koruma sağlar
- Zararlı radyasyonu süzer  
- Sıcaklık dengesini korur

**Yaşam Destek İşlevleri:**
- Solunumu mümkün kılar
- Su döngüsünü sağlar
- İklimi düzenler

Bu nedenlerle atmosferin korunması büyük önem taşır.
"""

    print(f"📝 Test metni uzunluğu: {len(turkish_test_doc)} karakter")
    print(f"📝 Satır sayısı: {len(turkish_test_doc.split(chr(10)))}")
    
    if not chunk_text:
        print("❌ Chunking sistemi mevcut değil, test atlanıyor")
        return
    
    # Test parametreleri
    chunk_size = 800  # Orta boyutlu chunk'lar
    chunk_overlap = 150  # Makul overlap
    
    print(f"⚙️ Chunk boyutu: {chunk_size}, Overlap: {chunk_overlap}")
    
    # Farklı stratejileri test et
    strategies_to_test = [
        ("markdown", "Markdown yapısal chunking"),
        ("sentence", "Türkçe cümle chunking"), 
        ("semantic", "Semantik chunking"),
        ("hybrid", "Hibrit chunking")
    ]
    
    results = {}
    
    for strategy, description in strategies_to_test:
        try:
            print(f"\n🔬 {description} testi başlıyor...")
            
            chunks = chunk_text(
                text=turkish_test_doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=strategy,
                language="tr"
            )
            
            if not chunks:
                print(f"❌ {strategy}: Chunk oluşturulamadı")
                continue
            
            print(f"✅ {strategy}: {len(chunks)} chunk oluşturuldu")
            
            # Kalite analizi
            quality_score = analyze_chunk_quality(chunks, strategy)
            results[strategy] = {
                'chunks': chunks,
                'count': len(chunks),
                'quality_score': quality_score,
                'description': description
            }
            
            # İlk chunk'ın önizlemesi
            first_chunk = chunks[0]
            print(f"📄 İlk chunk önizleme ({len(first_chunk)} karakter):")
            print(f"   '{first_chunk[:100]}...'")
            
        except Exception as e:
            print(f"❌ {strategy} stratejisinde hata: {str(e)}")
    
    # Sonuçları karşılaştır
    print("\n" + "="*80)
    print("📊 CHUNKING KALİTE KARŞILAŞTIRMASI")
    print("="*80)
    
    for strategy, result in results.items():
        print(f"\n🔹 {result['description']}")
        print(f"   Chunk sayısı: {result['count']}")
        print(f"   Kalite skoru: {result['quality_score']:.2f}/10")
        print(f"   Strateji: {strategy}")
    
    # En iyi stratejiyi belirle
    if results:
        best_strategy = max(results.keys(), key=lambda k: results[k]['quality_score'])
        print(f"\n🏆 EN İYİ STRATEJİ: {results[best_strategy]['description']}")
        print(f"   Kalite skoru: {results[best_strategy]['quality_score']:.2f}/10")
        
        # En iyi stratejinin detaylı analizi
        best_chunks = results[best_strategy]['chunks']
        print(f"\n📊 Detaylı chunk analizi:")
        for i, chunk in enumerate(best_chunks[:3]):  # İlk 3 chunk
            print(f"\n--- Chunk {i+1} ({len(chunk)} karakter) ---")
            lines = chunk.strip().split('\n')
            print(f"İlk satır: {lines[0][:80]}...")
            if len(lines) > 1:
                print(f"Son satır: {lines[-1][:80]}...")

def analyze_chunk_quality(chunks: list, strategy: str) -> float:
    """
    Chunk kalitesini analiz et ve 0-10 arası skor ver
    """
    if not chunks:
        return 0.0
    
    score = 0.0
    max_score = 10.0
    
    # 1. Chunk boyut tutarlılığı (2 puan)
    sizes = [len(chunk) for chunk in chunks]
    avg_size = sum(sizes) / len(sizes)
    size_consistency = 1 - (max(sizes) - min(sizes)) / (avg_size + 1)
    score += size_consistency * 2
    
    # 2. Kesik kelime/cümle kontrolü (3 puan)
    broken_chunks = 0
    for chunk in chunks:
        # Küçük harfle başlayan chunk'lar şüpheli
        first_line = chunk.strip().split('\n')[0].strip()
        if first_line and first_line[0].islower():
            # Ama başlık işareti yoksa kesinlikle kötü
            if not first_line.startswith('#'):
                broken_chunks += 1
        
        # Noktalama ile biten ama anlamsız başlayan
        if first_line and first_line.startswith(('.', ',', ';', ':', ')', ']')):
            broken_chunks += 1
    
    chunk_quality = max(0, 1 - (broken_chunks / len(chunks)))
    score += chunk_quality * 3
    
    # 3. Markdown yapı korunumu (2 puan)
    structure_score = 0
    for chunk in chunks:
        # Başlık korunmuş mu?
        if '# ' in chunk or '## ' in chunk or '### ' in chunk:
            structure_score += 0.5
        # Liste yapısı korunmuş mu?
        if '- ' in chunk or '* ' in chunk or '1. ' in chunk:
            structure_score += 0.3
    
    structure_score = min(2.0, structure_score)
    score += structure_score
    
    # 4. Konu bütünlüğü (3 puan)
    topic_coherence = 0
    for chunk in chunks:
        lines = chunk.strip().split('\n')
        has_title = any(line.strip().startswith('#') for line in lines)
        has_content = any(len(line.strip()) > 30 for line in lines if not line.strip().startswith('#'))
        
        if has_title and has_content:
            topic_coherence += 0.5
        elif has_content:  # Başlık yoksa ama içerik var
            topic_coherence += 0.3
    
    topic_coherence = min(3.0, topic_coherence)
    score += topic_coherence
    
    return min(max_score, score)

def test_problematic_cases():
    """
    Özellikle sorunlu durumları test et
    """
    print("\n" + "="*80)
    print("🚨 PROBLEMATİK DURUMLAR TESTİ")
    print("="*80)
    
    if not chunk_text:
        print("❌ Chunking sistemi mevcut değil, test atlanıyor")
        return
    
    # Sorunlu metin örnekleri
    problematic_texts = {
        "uzun_liste": """
# Önemli Kimyasal Elementler

## Periyodik Tablodaki Ana Elementler

1. Hidrojen (H): Evrendeki en yaygın element
2. Helyum (He): İkinci en yaygın element  
3. Lityum (Li): En hafif metal
4. Berilyum (Be): Sert ve hafif metal
5. Bor (B): Yarı metal özellikler
6. Karbon (C): Organik bileşiklerin temeli
7. Azot (N): Atmosferin %78'i
8. Oksijen (O): Yaşam için gerekli
9. Flor (F): En elektronegativite element
10. Neon (Ne): Soy gaz grubu
""",
        
        "kisa_paragraflar": """
Fizik nedir?

Fizik doğadaki olayları inceler.

Neden önemlidir?

Teknolojinin temelini oluşturur.

Hangi dalları var?

Mekanik, termodinamik, optik.

Kimlerle ilgilidir?

Matematik ve kimya ile bağlantılı.
""",
        
        "karma_yapi": """
### Soru 1: Atmosfer nedir?

Atmosfer, Dünya'nın etrafındaki gaz tabakasıdır.

```python
# Atmosfer bileşenleri
atmosfer = {
    "azot": 78.09,
    "oksijen": 20.95,
    "argon": 0.93
}
```

### Soru 2: Ozon nedir?

Ozon (O₃) molekülü.

**Özellikleri:**
- UV ışın emici
- Stratosfer katmanında  
- Yaşam için koruyucu

### Soru 3: İklim nasıl oluşur?

İklim faktörleri: sıcaklık, nem, basınç.
"""
    }
    
    for test_name, test_text in problematic_texts.items():
        print(f"\n🔍 Test: {test_name}")
        print(f"   Metin uzunluğu: {len(test_text)} karakter")
        
        try:
            chunks = chunk_text(
                text=test_text,
                chunk_size=300,  # Küçük chunk size - zorlu test
                chunk_overlap=50,
                strategy="markdown",
                language="tr"
            )
            
            print(f"   ✅ {len(chunks)} chunk oluşturuldu")
            
            # İlk chunk'a bak
            if chunks:
                first_chunk = chunks[0].strip()
                first_line = first_chunk.split('\n')[0]
                print(f"   📄 İlk chunk başlangıcı: '{first_line[:60]}...'")
                
                # Kesik başlangıç kontrolü
                if first_line and first_line[0].islower() and not first_line.startswith('#'):
                    print(f"   ⚠️ Kesik başlangıç tespit edildi!")
                else:
                    print(f"   ✅ Temiz başlangıç")
            
        except Exception as e:
            print(f"   ❌ Hata: {str(e)}")

if __name__ == "__main__":
    print("🧪 Advanced Chunking Sistemi Kalite Test Araracı")
    print("Türkçe doküman işleme kalitesini test eder")
    
    # Ana testleri çalıştır
    test_turkish_document_chunking()
    test_problematic_cases()
    
    print("\n" + "="*80)
    print("✅ Test tamamlandı!")
    print("="*80)