#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Cases for Critical Overlap and List Handling Issues
========================================================

This test specifically reproduces the critical issues mentioned in user feedback:
1. Overlap Problem - Lines from end of one chunk repeating in next chunk
2. List Item Fragmentation - Numbered lists split incorrectly 
3. Multiple Line Repetition - Multiple lines repeated in next chunk
4. List Structure Preservation - Complete lists should stay together

Author: Fixing Critical Issues
Date: 2025-11-17
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_processing.lightweight_chunker import (
    LightweightSemanticChunker,
    ChunkingConfig,
    create_semantic_chunks
)

def test_overlap_line_repetition():
    """Test case to reproduce overlap line repetition issues"""
    print("\n🔍 Testing Overlap Line Repetition Issues...")
    
    test_text = """# Türkiye'nin İklim Özellikleri

Türkiye'de üç farklı iklim tipi görülür. Bu iklim çeşitliliği ülkenin coğrafi konumundan kaynaklanır. Akdeniz iklimi güney kıyılarında etkilidir.

## Akdeniz İklimi
Akdeniz iklimi, güney kıyılarında görülür. Yazları sıcak ve kurak, kışları ılık ve yağışlıdır. Bu iklim tipi turizm için idealdir.

## Karasal İklim
Karasal iklim, iç bölgelerde etkilidir. Sıcaklık farkları büyüktür. Yaz ayları sıcak, kış ayları soğuk geçer."""

    # Create chunks with overlap
    chunker = LightweightSemanticChunker()
    chunks = chunker.create_semantic_chunks(
        text=test_text,
        target_size=200,  # Small size to force chunking
        overlap_ratio=0.2
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Check for line repetition between chunks
    overlap_issues = []
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].strip()
        next_chunk = chunks[i + 1].strip()
        
        # Split into lines
        current_lines = current_chunk.split('\n')
        next_lines = next_chunk.split('\n')
        
        # Check for exact line matches
        current_last_lines = [line.strip() for line in current_lines[-3:] if line.strip()]
        next_first_lines = [line.strip() for line in next_lines[:3] if line.strip()]
        
        # Find duplicated lines
        duplicated_lines = []
        for line in current_last_lines:
            if line in next_first_lines:
                duplicated_lines.append(line)
        
        if duplicated_lines:
            overlap_issues.append({
                'chunk_pair': (i, i + 1),
                'duplicated_lines': duplicated_lines,
                'current_chunk_end': current_lines[-2:],
                'next_chunk_start': next_lines[:2]
            })
    
    # Report results
    if overlap_issues:
        print(f"❌ Found {len(overlap_issues)} overlap line repetition issues:")
        for issue in overlap_issues:
            chunk_i, chunk_j = issue['chunk_pair']
            print(f"   • Between chunks {chunk_i} and {chunk_j}:")
            for dup_line in issue['duplicated_lines']:
                print(f"     Duplicated: '{dup_line[:50]}...'")
        return False
    else:
        print("✅ No line repetition found in overlaps")
        return True

def test_list_fragmentation():
    """Test case to reproduce list fragmentation issues"""
    print("\n🔍 Testing List Item Fragmentation...")
    
    test_text = """# Türkiye'nin Komşu Ülkeleri

Türkiye'nin komşu ülkeleri şunlardır:

## Batı Komşuları
1. Yunanistan - Ege Denizi ve Trakya sınırı
2. Bulgaristan - Trakya bölgesinden sınır
3. Bu ülkeler Avrupa Birliği üyesidir

## Doğu Komşuları  
1. Gürcistan - Karadeniz kıyısında
2. Ermenistan - Ağrı ve Kars illeri
3. İran - Doğu Anadolu sınırı
4. Irak - Güneydoğu sınırımız

## Güney Komşuları
1. Suriye - En uzun kara sınırımız
2. Bu sınır 822 km uzunluğundadır

Her komşu ülke ile farklı ilişkilerimiz vardır."""

    # Create chunks that might split lists
    chunker = LightweightSemanticChunker()
    chunks = chunker.create_semantic_chunks(
        text=test_text,
        target_size=150,  # Very small to force list splitting
        overlap_ratio=0.1
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Analyze list fragmentation
    fragmentation_issues = []
    
    for i, chunk in enumerate(chunks):
        lines = chunk.split('\n')
        
        # Find numbered list items
        numbered_items = []
        for j, line in enumerate(lines):
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                numbered_items.append((j, line))
        
        # Check if we have incomplete numbered sequences
        if numbered_items:
            numbers = [int(item[1].split('.')[0]) for item in numbered_items]
            
            # Check for gaps in numbering (indicates fragmentation)
            for k in range(len(numbers) - 1):
                if numbers[k + 1] - numbers[k] > 1:
                    fragmentation_issues.append({
                        'chunk_index': i,
                        'issue_type': 'numbering_gap',
                        'missing_numbers': list(range(numbers[k] + 1, numbers[k + 1])),
                        'found_items': numbered_items
                    })
            
            # Check if list starts with number > 1 (indicates start fragmentation)
            if numbers[0] > 1:
                fragmentation_issues.append({
                    'chunk_index': i,
                    'issue_type': 'list_start_fragmented',
                    'first_number': numbers[0],
                    'expected_start': 1,
                    'found_items': numbered_items
                })
    
    # Report results
    if fragmentation_issues:
        print(f"❌ Found {len(fragmentation_issues)} list fragmentation issues:")
        for issue in fragmentation_issues:
            if issue['issue_type'] == 'numbering_gap':
                print(f"   • Chunk {issue['chunk_index']}: Missing items {issue['missing_numbers']}")
            elif issue['issue_type'] == 'list_start_fragmented':
                print(f"   • Chunk {issue['chunk_index']}: List starts at {issue['first_number']} instead of 1")
        return False
    else:
        print("✅ No list fragmentation found")
        return True

def test_bulleted_list_preservation():
    """Test bulleted list preservation"""
    print("\n🔍 Testing Bulleted List Preservation...")
    
    test_text = """# Türkiye'nin Doğal Kaynakları

Türkiye zengin doğal kaynaklara sahiptir:

## Maden Kaynakları
- Kömür rezervleri (Zonguldak, Soma)
- Demir cevheri (Divriği, Hekimhan)  
- Krom rezervleri (dünya 2. sırası)
- Bor minerali (dünya 1. sırası)
- Bakır yatakları (Murgul, Ergani)
- Altın rezervleri (Kışladağ, Ovacık)

## Enerji Kaynakları
- Hidroelektrik potansiyeli yüksek
- Rüzgar enerjisi kapasitesi büyük
- Güneş enerjisi potansiyeli var
- Doğalgaz rezervleri sınırlı

Bu kaynaklar ekonomimiz için çok önemlidir."""

    chunker = LightweightSemanticChunker()
    chunks = chunker.create_semantic_chunks(
        text=test_text,
        target_size=180,
        overlap_ratio=0.1
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Check for bulleted list fragmentation
    list_issues = []
    
    for i, chunk in enumerate(chunks):
        lines = chunk.split('\n')
        bullet_lines = [line.strip() for line in lines if line.strip().startswith('-')]
        
        # If we have bullet points, check if list seems complete
        if bullet_lines:
            # Look for list context (should have a header above)
            has_context = any('##' in line or ':' in line for line in lines[:3])
            
            if len(bullet_lines) < 3 and not has_context:
                # Might be a fragmented list
                list_issues.append({
                    'chunk_index': i,
                    'bullet_count': len(bullet_lines),
                    'bullets': bullet_lines,
                    'has_context': has_context
                })
    
    if list_issues:
        print(f"❌ Found {len(list_issues)} potential bulleted list issues:")
        for issue in list_issues:
            print(f"   • Chunk {issue['chunk_index']}: Only {issue['bullet_count']} bullets, context: {issue['has_context']}")
        return False
    else:
        print("✅ Bulleted lists preserved well")
        return True

def test_multiple_line_repetition():
    """Test for multiple consecutive lines being repeated"""
    print("\n🔍 Testing Multiple Line Repetition...")
    
    test_text = """# Türkiye'nin Bölgeleri

## Marmara Bölgesi
Marmara Bölgesi, Türkiye'nin kuzeybatısında yer alır. 
Bu bölge ülkenin en kalabalık bölgesidir.
İstanbul bu bölgenin en büyük şehridir.
Sanayi açısından da çok gelişmiştir.

## Ege Bölgesi  
Ege Bölgesi, batı kıyılarımızda bulunur.
Turizm ve tarım açısından önemlidir.
İzmir bölgenin merkezi konumundadır."""

    chunker = LightweightSemanticChunker() 
    chunks = chunker.create_semantic_chunks(
        text=test_text,
        target_size=120,
        overlap_ratio=0.25  # Higher overlap to test repetition
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Check for multiple consecutive line repetition
    repetition_issues = []
    
    for i in range(len(chunks) - 1):
        current_lines = chunks[i].strip().split('\n')
        next_lines = chunks[i + 1].strip().split('\n') 
        
        # Find consecutive repeated lines
        consecutive_matches = []
        for j in range(len(current_lines) - 1):
            current_line = current_lines[j].strip()
            next_line = current_lines[j + 1].strip()
            
            if current_line in [l.strip() for l in next_lines]:
                next_line_idx = [l.strip() for l in next_lines].index(current_line)
                if (next_line_idx + 1 < len(next_lines) and 
                    next_lines[next_line_idx + 1].strip() == next_line):
                    consecutive_matches.append((current_line, next_line))
        
        if len(consecutive_matches) > 1:
            repetition_issues.append({
                'chunk_pair': (i, i + 1),
                'repeated_sequences': consecutive_matches
            })
    
    if repetition_issues:
        print(f"❌ Found {len(repetition_issues)} multiple line repetition issues:")
        for issue in repetition_issues:
            chunk_i, chunk_j = issue['chunk_pair']
            print(f"   • Chunks {chunk_i}-{chunk_j}: {len(issue['repeated_sequences'])} consecutive lines repeated")
        return False
    else:
        print("✅ No multiple line repetition found")
        return True

def run_all_tests():
    """Run all critical issue tests"""
    print("=" * 70)
    print("TESTING CRITICAL OVERLAP AND LIST HANDLING ISSUES")
    print("=" * 70)
    
    # Run all tests
    test_results = []
    
    test_results.append(("Overlap Line Repetition", test_overlap_line_repetition()))
    test_results.append(("List Item Fragmentation", test_list_fragmentation()))  
    test_results.append(("Bulleted List Preservation", test_bulleted_list_preservation()))
    test_results.append(("Multiple Line Repetition", test_multiple_line_repetition()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All critical issue tests PASSED - No issues found!")
        return True
    else:
        print(f"❌ {total - passed} critical issues found - FIXES NEEDED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)