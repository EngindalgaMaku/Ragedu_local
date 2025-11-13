#!/usr/bin/env python3
"""
Düzeltilmiş chunking algoritmasını test eden script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.text_processing.text_chunker import chunk_text

# Gerçek biyoloji metni - tam versiyon
test_markdown = """
# 9. SINIF BİYOLOJİ DERS NOTLARI

## 1.ÜNİTE: YAŞAM BİLİMİ BİYOLOJİ

### 1. BÖLÜM: BİYOLOJİ VE CANLILARIN ORTAK ÖZELLİKLERİ

**BİYOLOJİ:** Canlıların yapılarını, yaşamsal faaliyetlerini, davranışlarını, gelişmelerini, yeryüzündeki dağılışlarını, birbirleriyle ve çevreleriyle olan ilişkilerini inceleyen bilim dalıdır.

## CANLILARIN ORTAK ÖZELLİKLERİ

### 1. HÜCRESEL YAPI
Hücre canlının en küçük yapı birimidir. Bütün canlılar bir veya daha fazla sayıda hücreden oluşurlar.
- **Tek hücreli:** bakteri, amip
- **Çok hücreli:** bitki, hayvan

### 2. BESLENME
Canlıların enerji ihtiyacını karşılamak ve yaşamlarını sürdürmek için gerekli maddeleri almasına **BESLENME** denir.

**Beslenme Türleri:**
- **OTOTROF:** Kendi besinlerini kendileri üreten canlılar (bitkiler, bazı bakteriler)
- **HETEROTROF:** Besinlerini dışarıdan hazır alan canlılar (hayvanlar, mantarlar)

### 3. SOLUNUM
Hücre içinde besinlerin parçalanarak elde edilen ATP molekülünden enerji açığa çıkarılmasına solunum denir.
- **Oksijenli solunum:** Oksijen kullanarak
- **Oksijensiz solunum:** Oksijen kullanmadan

### 4. ATP ÜRETME VE TÜKETME
- **FOSFORİLASYON:** ATP üretme
- **DEFOSFORİLASYON:** ATP tüketme

### 5. METABOLİZMA
Canlı vücudunda gerçekleşen yapım ve yıkım olaylarının tamamına metabolizma denir.
- **ANABOLİZMA:** Küçük moleküllerin birleştirilerek büyük molekül üretilmesi (yapım)
- **KATABOLİZMA:** Büyük moleküllerin parçalanarak küçültülmesi (yıkım)

### 6. BOŞALTIM
Metabolizma sonucu oluşan atıkların vücuttan uzaklaştırılmasına boşaltım denir.
- **Bitkiler:** terleme, yaprak dökme
- **Hayvanlar:** terleme, soluk verme, idrar oluşturma

### 7. HAREKET
- **PASİF HAREKET:** Yön değiştirerek yapılan (bitkiler)
- **AKTİF HAREKET:** Yer değiştirerek yapılan (hayvanlar, bakteriler)
"""

def analyze_chunks(chunks):
    """Chunk kalitesini analiz et"""
    print("🔍 CHUNK KALİTE ANALİZİ")
    print("=" * 50)
    
    issues = []
    
    for i, chunk in enumerate(chunks):
        lines = chunk.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Kelime kesikliği kontrolü
        if first_line and len(first_line) > 10:
            if not first_line[0].isupper() and not first_line.startswith('#') and not first_line.startswith('-'):
                issues.append(f"⚠️ Chunk {i+1}: Kelime kesikliği - '{first_line[:30]}...'")
        
        # Çok küçük chunk kontrolü  
        if len(chunk) < 100:
            issues.append(f"⚠️ Chunk {i+1}: Çok küçük ({len(chunk)} karakter)")
        
        print(f"📄 **CHUNK {i+1}** - {len(chunk)} karakter")
        print(f"   🔸 İlk satır: '{first_line[:50]}...'")
        print()
    
    if issues:
        print(f"🚨 {len(issues)} kalite sorunu:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ Kalite sorunları tespit edilmedi!")
    
    return len(issues) == 0

def main():
    print("🧪 DÜZELTİLMİŞ CHUNKING SİSTEMİ TESTİ")
    print("=" * 60)
    
    # Chunking yap
    chunks = chunk_text(test_markdown, chunk_size=400, chunk_overlap=80, strategy="markdown")
    
    print(f"📊 SONUÇLAR:")
    print(f"   📚 Toplam chunk: {len(chunks)}")
    if chunks:
        lengths = [len(c) for c in chunks]
        print(f"   📏 Ortalama boyut: {sum(lengths)/len(lengths):.0f} karakter")
        print(f"   📐 Min-Max: {min(lengths)}-{max(lengths)} karakter")
    
    print("\n" + "="*60)
    
    # Kalite analizi
    is_good_quality = analyze_chunks(chunks)
    
    print("\n" + "="*60)
    print(f"🎯 GENEL SONUÇ: {'✅ BAŞARILI' if is_good_quality else '❌ İYİLEŞTİRME GEREKİYOR'}")
    
    # Chunk içeriklerini göster
    print(f"\n📋 CHUNK İÇERİKLERİ:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i+1} ---")
        print(chunk)
        print("-" * 40)

if __name__ == "__main__":
    main()