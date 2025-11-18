#!/usr/bin/env python3
"""
Basit Chunking Testi - Sadece mevcut fonksiyonları kullanır
"""

import sys
import re
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Basic chunking fonksiyonlarını doğrudan import et
try:
    from src.text_processing.text_chunker import chunk_text
    print("✅ Basic chunking sistemi yüklendi")
    chunking_available = True
except ImportError as e:
    print(f"❌ Chunking sistemi yüklenemedi: {e}")
    chunking_available = False

def test_basic_chunking():
    """
    Temel chunking fonksiyonlarını test et
    """
    print("\n" + "="*80)
    print("🧪 TEMEL CHUNKING TESTİ")
    print("="*80)
    
    # Basit Türkçe test metni
    test_text = """
# Atmosfer Katmanları

## Troposfer
Dünya yüzeyinden 18 km yüksekliğe kadar olan katmandır. Bu katmanda hava sıcaklığı yükseklik arttıkça azalır.

### Troposferin Özellikleri
- Bulutlar bu katmanda oluşur
- Yağışlar meydana gelir
- Canlılar bu katmanda yaşar

## Stratosfer  
18-50 km arasındaki katmandır. Ozon tabakası burada bulunur.

### Ozon Tabakası
Ozon (O₃) molekülleri ultraviyole ışınları emer. Bu sayede canlıları korur.

## Sonuç
Atmosfer katmanları yaşam için çok önemlidir.
"""

    print(f"📝 Test metni uzunluğu: {len(test_text)} karakter")
    
    if not chunking_available:
        print("❌ Chunking sistemi mevcut değil")
        return
    
    # Farklı stratejileri test et
    test_strategies = ["char", "markdown", "sentence"]
    
    for strategy in test_strategies:
        print(f"\n🔬 {strategy.upper()} stratejisi testi:")
        
        try:
            chunks = chunk_text(
                text=test_text,
                chunk_size=300,
                chunk_overlap=50,
                strategy=strategy,
                language="tr"
            )
            
            if chunks:
                print(f"  ✅ {len(chunks)} chunk oluşturuldu")
                
                # İlk chunk kontrolü
                first_chunk = chunks[0].strip()
                first_line = first_chunk.split('\n')[0].strip()
                print(f"  📄 İlk chunk: '{first_line[:50]}...'")
                
                # Kesik başlangıç kontrolü
                if first_line and len(first_line) > 0:
                    if first_line[0].islower() and not first_line.startswith('#'):
                        print(f"  ⚠️ Kesik başlangıç tespit edildi!")
                    else:
                        print(f"  ✅ Temiz başlangıç")
                
                # Chunk boyutları
                sizes = [len(chunk) for chunk in chunks]
                print(f"  📏 Chunk boyutları: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")
                
            else:
                print(f"  ❌ Chunk oluşturulamadı")
                
        except Exception as e:
            print(f"  ❌ Hata: {str(e)}")

def test_problematic_turkish_text():
    """
    Problematik Türkçe metni test et
    """
    print("\n" + "="*80)
    print("🚨 PROBLEMATİK TÜRKÇE METİN TESTİ")
    print("="*80)
    
    # Görüntülerde gördüğümüz gibi problem yaşayan metin türü
    problematic_text = """
çiziminde kullanılır.
Konik Projeksiyon: Kutuplar ve çevresindeki bölgelerin
çiziminde kullanılır.
Düzlem (Ufki) Projeksiyon: Kutuplar ve çevresi için daha
uygundur.Bu projeksiyonla elde edilen haritalarda biçim ve: Not: Haritalar
çeşitli yollarla (film, fotokopi vs.) büyütülüp
alan bozulmaları çok fazladır. Bu haritalar daha çok küçültülürse ölçekleri de değişir. Ancak bu haritalar
Türkiye'de, izoüstim alan ile gerçek alan arasındaki farkın en az
olduğu bölgeler Doğu Anadolu ve Karadeniz, en az
fazla olduğu bölgeler Doğu Anadolu ve Karadeniz, en az
Ölçek:Haritadaki küçültme oranını ölçekli gerçek alan
uzunluklar arasındaki oran
"""

    print(f"📝 Problematik metin uzunluğu: {len(problematic_text)} karakter")
    
    if not chunking_available:
        print("❌ Chunking sistemi mevcut değil")
        return
    
    # Markdown stratejisi ile test et
    try:
        chunks = chunk_text(
            text=problematic_text,
            chunk_size=200,
            chunk_overlap=30,
            strategy="markdown",
            language="tr"
        )
        
        print(f"✅ {len(chunks)} chunk oluşturuldu")
        
        for i, chunk in enumerate(chunks):
            lines = chunk.strip().split('\n')
            first_line = lines[0].strip() if lines else ""
            
            print(f"\n--- Chunk {i+1} ({len(chunk)} karakter) ---")
            print(f"İlk satır: '{first_line[:60]}...'")
            
            # Kesik kelime kontrolü
            if first_line and len(first_line) > 0:
                if (first_line[0].islower() and 
                    not first_line.startswith('#') and
                    not first_line.startswith('-') and
                    not first_line.startswith('*')):
                    print("⚠️ KESİK BAŞLANGIÇ TESPİT EDİLDİ!")
                else:
                    print("✅ Temiz başlangıç")
            
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")

def analyze_chunk_boundaries(chunks):
    """
    Chunk sınırlarını analiz et
    """
    print("\n📊 Chunk Sınır Analizi:")
    
    boundary_issues = 0
    for i, chunk in enumerate(chunks):
        lines = chunk.strip().split('\n')
        if not lines:
            continue
            
        first_line = lines[0].strip()
        last_line = lines[-1].strip()
        
        # İlk satır problemleri
        if first_line and first_line[0].islower():
            if not (first_line.startswith('#') or 
                   first_line.startswith('-') or
                   first_line.startswith('*') or
                   first_line.startswith('.')):  # Numaralı liste
                print(f"  🔸 Chunk {i+1}: Kesik başlangıç - '{first_line[:30]}...'")
                boundary_issues += 1
        
        # Son satır problemleri  
        if last_line and not (last_line.endswith('.') or 
                             last_line.endswith('!') or
                             last_line.endswith('?') or
                             last_line.endswith(':') or
                             last_line.strip() == ''):
            # Bu normal olabilir, çok kritik değil
            pass
    
    if boundary_issues == 0:
        print("  ✅ Chunk sınırları temiz!")
    else:
        print(f"  ⚠️ {boundary_issues} chunk'ta sınır problemi tespit edildi")
    
    return boundary_issues

if __name__ == "__main__":
    print("🧪 Basit Chunking Sistemi Test Aracı")
    print("Mevcut chunking fonksiyonlarını test eder")
    
    # Testleri çalıştır
    test_basic_chunking()
    test_problematic_turkish_text()
    
    print("\n" + "="*80)
    print("✅ Test tamamlandı!")
    print("💡 Öneriler:")
    print("   - Kesik başlangıçlar düzeltilmeli")
    print("   - Türkçe noktalama kuralları daha iyi uygulanmalı")
    print("   - Konu bütünlüğü korunmalı")
    print("="*80)