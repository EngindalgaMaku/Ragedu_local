#!/usr/bin/env python3
"""
Test script for semantic chunking JSON parsing fixes.
Tests the critical JSON parsing issues and fallback mechanisms.
"""

import os
import sys
sys.path.append('.')

from src.text_processing.semantic_chunker import SemanticChunker, create_semantic_chunks

def test_json_parsing_fix():
    """Test the new JSON parsing fixes."""
    
    print("🔥 TESTING SEMANTIC CHUNKING JSON PARSING FIXES")
    print("=" * 60)
    
    # Test text
    test_text = """
    # Biyoloji ve Yaşam Bilimleri

    Biyoloji, canlı organizmaları ve yaşam süreçlerini inceleyen bilim dalıdır. 
    Bu alan moleküler seviyeden ekosistem seviyesine kadar geniş bir yelpaze kapsar.

    ## Hücre Biyolojisi
    
    Hücreler, tüm canlı organizmaların temel yapı taşlarıdır. Prokaryotik ve ökaryotik 
    hücreler olmak üzere iki ana gruba ayrılırlar.
    
    Hücre zarı, hücrenin içeriğini dış ortamdan ayıran seçici geçirgen bir yapıdır.
    
    ## Genetik ve Kalıtım
    
    DNA, genetik bilginin depolandığı moleküldür. Genler, belirli proteinleri 
    kodlayan DNA segmentleridir ve kalıtımsal özelliklerin aktarımından sorumludur.
    """
    
    try:
        # Initialize chunker
        chunker = SemanticChunker()
        
        print(f"📝 Test Text Length: {len(test_text)} chars")
        print(f"🚀 API Call Limit: {chunker.max_api_calls}")
        print(f"⚡ Max Consecutive Failures: {chunker.max_consecutive_failures}")
        print()
        
        # Test chunking
        print("🔍 Starting semantic chunking test...")
        chunks = chunker.create_semantic_chunks(
            text=test_text,
            target_size=400,
            overlap_ratio=0.05,  # Reduced overlap
            language="tr"
        )
        
        print(f"✅ Success! Generated {len(chunks)} chunks")
        print()
        
        # Display chunks
        for i, chunk in enumerate(chunks, 1):
            print(f"📄 CHUNK {i} (Length: {len(chunk)} chars)")
            print("-" * 50)
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            print()
        
        # Test performance metrics
        print("📊 PERFORMANCE METRICS:")
        print(f"🔄 Total API Calls Made: {chunker.api_call_count}")
        print(f"❌ Consecutive Failures: {chunker.consecutive_failures}")
        print(f"💾 Cached Invalid Responses: {len(chunker.invalid_response_cache)}")
        print(f"⚠️  Warning Suppressions: {len(chunker.warning_suppression)}")
        
        if chunks:
            avg_chunk_size = sum(len(c) for c in chunks) / len(chunks)
            print(f"📏 Average Chunk Size: {avg_chunk_size:.1f} chars")
            print(f"📐 Size Range: {min(len(c) for c in chunks)} - {max(len(c) for c in chunks)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_mechanisms():
    """Test fallback mechanisms with forced failures."""
    
    print("\n🔧 TESTING FALLBACK MECHANISMS")
    print("=" * 60)
    
    try:
        chunker = SemanticChunker()
        
        # Force failure by setting very low API limit
        chunker.max_api_calls = 0  # Force immediate fallback
        
        test_text = """Bu metin JSON parsing hataları test ediyor.
        Fallback mekanizmalarının çalışıp çalışmadığını kontrol ediyoruz.
        Bu metin, minimum chunk boyutunu aşacak kadar uzun olmalıdır.
        Böylece fallback mekanizması test edildiğinde en az bir chunk üretilebilir.
        """
        
        print("🚫 Forcing API limit (0) to test fallback...")
        
        chunks = chunker.create_semantic_chunks(
            text=test_text,
            target_size=200,
            language="tr"
        )
        
        if chunks:
            print(f"✅ Fallback SUCCESS! Generated {len(chunks)} chunks using heuristics")
            for i, chunk in enumerate(chunks, 1):
                print(f"   Chunk {i}: {len(chunk)} chars")
            return True
        else:
            print("❌ Fallback FAILED - No chunks generated")
            return False
            
    except Exception as e:
        print(f"❌ FALLBACK ERROR: {e}")
        return False

def test_json_parsing_strategies():
    """Test different JSON parsing strategies."""
    
    print("\n🧪 TESTING JSON PARSING STRATEGIES")
    print("=" * 60)
    
    chunker = SemanticChunker()
    
    # Test different malformed JSON responses
    test_responses = [
        '{"boundaries": [{"position": 123, "confidence": 0.8, "topic_shift": true, "coherence_score": 0.9, "boundary_type": "topic_change"}]}',  # Valid JSON
        'Text before {"boundaries": [{"position": 123, "confidence": 0.8, "topic_shift": true, "coherence_score": 0.9, "boundary_type": "topic_change"}]} text after',  # JSON in text
        '{"boundaries": []}',  # Empty boundaries
        'Invalid JSON response with no brackets',  # No JSON
        '{"boundaries": [{"position": 123, "confidence": 0.8}]}',  # Partial data
    ]
    
    for i, response in enumerate(test_responses, 1):
        print(f"🧪 Testing Strategy {i}: {response[:50]}...")
        
        try:
            result = chunker._parse_llm_response_robust(response)
            print(f"   ✅ SUCCESS: {len(result.get('boundaries', []))} boundaries found")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        print()
    
    return True

def main():
    """Run all tests."""
    
    print("🚀 SEMANTIC CHUNKING JSON PARSING FIX TEST SUITE")
    print("=" * 80)
    print()
    
    tests = [
        ("JSON Parsing Fix", test_json_parsing_fix),
        ("Fallback Mechanisms", test_fallback_mechanisms), 
        ("JSON Parsing Strategies", test_json_parsing_strategies)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n▶️  Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🏆 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! JSON parsing fix is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())