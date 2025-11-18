#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggressive Test to Reproduce Critical Issues
============================================

This test uses larger documents with smaller chunk sizes to force the 
overlap and list fragmentation issues to occur.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_processing.lightweight_chunker import (
    LightweightSemanticChunker,
    ChunkingConfig
)

def create_large_test_document():
    """Create a large test document with multiple lists and content sections"""
    
    return """# Türkiye'nin Kapsamlı Coğrafi Analizi

## Giriş
Türkiye, Anadolu ve Trakya yarımadalarında yer alan, üç kıta arasında köprü görevi gören stratejik öneme sahip bir ülkedir. Bu analiz, ülkenin coğrafi özelliklerini detaylı olarak incelemektedir.

## Türkiye'nin Komşu Ülkeleri

### Avrupa Komşuları
Türkiye'nin Avrupa kıtasındaki komşuları şunlardır:

1. Yunanistan - Ege Denizi kıyısında ve Trakya bölgesinde sınır
2. Bulgaristan - Trakya bölgesinde kısa bir sınır hattı
3. Bu ülkeler Avrupa Birliği üyesi olup ekonomik entegrasyonda önemlidir
4. Sınır kapıları yoğun ticari faaliyetlere ev sahipliği yapar

### Asya Komşuları
Doğu ve güneydoğu sınırlarımızdaki ülkeler:

1. Gürcistan - Karadeniz kıyısında, Artvin ve Ardahan illeriyle sınır
2. Ermenistan - Ağrı ve Kars illeri üzerinden sınır (kapalı sınır)
3. İran - En uzun doğu sınırımız, 534 km uzunluğunda
4. Irak - Güneydoğu sınırımız, 367 km uzunluğunda
5. Suriye - En uzun kara sınırımız, 822 km uzunluğunda

## İklim Özellikleri

Türkiye'de görülen iklim tipleri ve özellikleri:

### Akdeniz İklimi
- Kıyı kesimlerinde etkili
- Yazları sıcak ve kurak
- Kışları ılık ve yağışlı
- Turizm için ideal koşullar
- Narenciye üretimi yaygın

### Karasal İklim
- İç Anadolu'da baskın
- Büyük sıcaklık farkları
- Az yağış alma
- Tahıl üretimi yaygın
- Hayvancılık gelişmiş

### Karadeniz İklimi
- Karadeniz kıyısında
- Yıl boyunca yağışlı
- Sıcaklık farkları az
- Çay ve fındık üretimi
- Yoğun orman örtüsü

## Doğal Kaynaklar

### Maden Kaynakları
Türkiye'nin önemli maden yatakları:

1. Kömür rezervleri - Zonguldak, Soma, Beypazarı yöreleri
2. Demir cevheri - Divriği, Hekimhan, Hasançelebi bölgeleri  
3. Krom madeni - Dünya rezervinin %38'i ülkemizde
4. Bor minerali - Dünya üretiminin %73'ü Türkiye'den
5. Bakır yatakları - Murgul, Ergani, Siirt civarı
6. Altın rezervleri - Kışladağ, Ovacık, İzmir-Bergama
7. Mermer yatakları - Afyon, Bilecik, Marmara bölgesi

### Enerji Kaynakları
- Hidroelektrik potansiyeli: 433 milyar kWh/yıl
- Rüzgar enerjisi kapasitesi giderek artıyor
- Güneş enerjisi potansiyeli yüksek
- Jeotermal enerji kaynakları zengin
- Doğalgaz rezervleri sınırlı
- Petrol rezervleri az

## Tarım ve Hayvancılık

### Bitkisel Üretim
Türkiye'nin önemli tarım ürünleri:

1. Tahıl üretimi - Buğday, arpa, mısır, çeltik
2. Endüstri bitkileri - Pamuk, tütün, şeker pancarı
3. Sebze üretimi - Domates, biber, patlıcan, soğan
4. Meyve üretimi - Elma, narenciye, üzüm, kayısı
5. Yağlı tohum - Ayçiçeği, susam, aspir

Her bölgenin kendine özgü tarımsal potansiyeli bulunmaktadır.

### Hayvancılık Sektörü
- Küçükbaş hayvancılık yaygın
- Büyükbaş hayvancılık gelişiyor  
- Beyaz et üretimi artıyor
- Süt üretimi yeterli seviyede
- Su ürünleri avcılığı önemli

## Sonuç

Türkiye'nin coğrafi konumu, iklim çeşitliliği ve doğal kaynakları ülkeye büyük avantajlar sağlamaktadır. Bu potansiyelin doğru değerlendirilmesi ile ekonomik kalkınma hızlanacaktır."""

def test_overlap_issues_aggressive():
    """Test overlap issues with very small chunk sizes"""
    print("\n🔍 AGGRESSIVE TEST: Overlap Issues with Small Chunk Sizes")
    print("=" * 60)
    
    test_text = create_large_test_document()
    print(f"Document size: {len(test_text):,} characters")
    
    # Use very small chunk sizes to force chunking
    chunker = LightweightSemanticChunker()
    chunks = chunker.create_semantic_chunks(
        text=test_text,
        target_size=200,  # Very small chunks
        overlap_ratio=0.3   # High overlap ratio
    )
    
    print(f"Created {len(chunks)} chunks with high overlap")
    
    # Detailed analysis of each chunk transition
    overlap_issues = []
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].strip()
        next_chunk = chunks[i + 1].strip()
        
        print(f"\n--- Analyzing Chunks {i} -> {i+1} ---")
        print(f"Chunk {i} ending: ...{current_chunk[-100:]}")
        print(f"Chunk {i+1} starting: {next_chunk[:100]}...")
        
        # Split into sentences for precise overlap analysis
        current_sentences = [s.strip() for s in current_chunk.split('.') if s.strip()]
        next_sentences = [s.strip() for s in next_chunk.split('.') if s.strip()]
        
        # Check for exact sentence duplicates
        duplicated_sentences = []
        for sent in current_sentences[-3:]:  # Last 3 sentences of current
            if sent in next_sentences[:3]:   # First 3 sentences of next
                duplicated_sentences.append(sent)
        
        if duplicated_sentences:
            overlap_issues.append({
                'chunk_pair': (i, i+1),
                'duplicated_sentences': duplicated_sentences
            })
            print(f"❌ OVERLAP ISSUE: {len(duplicated_sentences)} duplicated sentences")
            for dup in duplicated_sentences:
                print(f"   Duplicate: '{dup[:50]}...'")
        else:
            print("✅ No sentence duplication")
    
    return len(overlap_issues) == 0

def test_list_fragmentation_aggressive():
    """Test list fragmentation with documents containing many lists"""
    print("\n🔍 AGGRESSIVE TEST: List Fragmentation")
    print("=" * 60)
    
    test_text = create_large_test_document()
    
    # Very small chunk size to force list splitting
    chunker = LightweightSemanticChunker()
    chunks = chunker.create_semantic_chunks(
        text=test_text,
        target_size=150,  # Extremely small to force splits
        overlap_ratio=0.1
    )
    
    print(f"Created {len(chunks)} chunks with small size")
    
    # Analyze each chunk for fragmented lists
    fragmentation_issues = []
    for i, chunk in enumerate(chunks):
        print(f"\n--- Analyzing Chunk {i} ---")
        lines = chunk.split('\n')
        
        # Find numbered list items
        numbered_items = []
        for line_no, line in enumerate(lines):
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                item_num = int(line.split('.')[0])
                numbered_items.append((line_no, item_num, line))
        
        if numbered_items:
            print(f"Found {len(numbered_items)} numbered items:")
            numbers = [item[1] for item in numbered_items]
            for item in numbered_items:
                print(f"   {item[1]}. {item[2][:50]}...")
            
            # Check for fragmentation patterns
            if numbers and numbers[0] != 1:
                fragmentation_issues.append({
                    'chunk': i,
                    'type': 'missing_start',
                    'starts_with': numbers[0],
                    'items': numbered_items
                })
                print(f"❌ LIST FRAGMENTATION: Starts with {numbers[0]} instead of 1")
            
            # Check for gaps
            for j in range(len(numbers) - 1):
                if numbers[j+1] - numbers[j] > 1:
                    missing = list(range(numbers[j] + 1, numbers[j+1]))
                    fragmentation_issues.append({
                        'chunk': i,
                        'type': 'missing_items',
                        'missing_numbers': missing,
                        'items': numbered_items
                    })
                    print(f"❌ LIST GAP: Missing items {missing}")
        
        # Check for bulleted lists
        bullet_items = [line for line in lines if line.strip().startswith('-')]
        if bullet_items:
            print(f"Found {len(bullet_items)} bullet items")
            if len(bullet_items) < 2:  # Suspiciously small bullet list
                print("⚠️ Possibly fragmented bullet list")
    
    return len(fragmentation_issues) == 0

def run_aggressive_tests():
    """Run all aggressive tests"""
    print("=" * 70)
    print("AGGRESSIVE TESTING: REPRODUCING CRITICAL ISSUES")
    print("=" * 70)
    
    overlap_passed = test_overlap_issues_aggressive()
    list_passed = test_list_fragmentation_aggressive()
    
    print("\n" + "=" * 70)
    print("AGGRESSIVE TEST RESULTS")
    print("=" * 70)
    
    print(f"Overlap Issues: {'✅ PASS' if overlap_passed else '❌ ISSUES FOUND'}")
    print(f"List Fragmentation: {'✅ PASS' if list_passed else '❌ ISSUES FOUND'}")
    
    if overlap_passed and list_passed:
        print("\n✅ No critical issues reproduced - System working well!")
        return True
    else:
        print("\n❌ Critical issues reproduced - FIXES NEEDED!")
        return False

if __name__ == "__main__":
    success = run_aggressive_tests()
    sys.exit(0 if success else 1)