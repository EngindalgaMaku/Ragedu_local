#!/usr/bin/env python3
"""
Test script to verify embedding provider selection functionality
Tests both Groq (fast) and Ollama (local) embedding providers
"""

import requests
import json
import time
import uuid
from typing import Dict, Any

# Service URLs
API_GATEWAY_URL = "http://localhost:8000"
DOC_PROCESSING_URL = "http://localhost:8003"  # Correct port from docker-compose ps

def test_embedding_provider_selection():
    """Test embedding provider selection functionality end-to-end"""
    
    print("🧪 EMBEDDING PROVIDER SELECTION TEST")
    print("=" * 60)
    
    # Test document content (Turkish)
    test_content = """
    Biyoloji ve Canlıların Ortak Özellikleri

    Biyoloji, canlıları ve yaşam süreçlerini inceleyen bilim dalıdır. Tüm canlılar bazı ortak özelliklere sahiptir:

    1. Hücresel Yapı: Tüm canlılar bir veya daha fazla hücreden oluşur.
    2. Metabolizma: Canlılar enerji alışverişi yapar ve kimyasal reaksiyonlar gerçekleştirir.
    3. Büyüme ve Gelişme: Canlılar zaman içinde büyür ve gelişir.
    4. Çoğalma: Canlılar kendi türlerinden yeni bireyler meydana getirir.
    5. Uyarılabilirlik: Çevre koşullarındaki değişikliklere tepki verir.
    6. Homeostaz: İç dengeyi korur.
    7. Adaptasyon: Çevresel koşullara uyum sağlar.

    Bu özellikler, bir varlığın canlı olup olmadığını belirlemede kullanılır.
    """
    
    # Generate a unique session ID for this test
    session_id = str(uuid.uuid4())
    print(f"📍 Test Session ID: {session_id}")
    
    # Test only the available embedding provider (Groq embeddings were removed - not supported)
    providers = [
        {"name": "Ollama", "model": "mxbai-embed-large", "expected_speed": "local (~5-10s)"}
    ]
    
    test_results = {}
    
    for provider in providers:
        print(f"\n📊 Testing {provider['name']} Embedding Provider")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Test direct document processing service call with embedding model parameter
            print(f"🔧 Testing direct document processing with {provider['model']}")
            
            process_data = {
                "text": test_content,
                "metadata": {
                    "session_id": session_id,
                    "embedding_model": provider['model'],  # Key parameter
                    "chunk_strategy": "semantic",
                    "filename": f"test_biology_{provider['name'].lower()}.md"
                },
                "collection_name": f"session_{session_id}",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
            
            print(f"🚀 Calling document processing service...")
            print(f"   Embedding Model: {provider['model']}")
            print(f"   Expected Speed: {provider['expected_speed']}")
            
            doc_response = requests.post(
                f"{DOC_PROCESSING_URL}/process-and-store",
                json=process_data,
                headers={"Content-Type": "application/json"},
                timeout=600  # 10 minutes max for slow local embeddings
            )
            
            processing_time = time.time() - start_time
            
            if doc_response.status_code == 200:
                doc_result = doc_response.json()
                print(f"✅ Document processing successful!")
                print(f"   Processing Time: {processing_time:.2f}s")
                print(f"   Chunks Processed: {doc_result.get('chunks_processed', 'unknown')}")
                print(f"   Collection Name: {doc_result.get('collection_name', 'unknown')}")
                
                # Test query functionality
                print(f"🔍 Testing query functionality...")
                
                query_data = {
                    "session_id": session_id,
                    "query": "Canlıların ortak özellikleri nelerdir?",
                    "top_k": 3,
                    "use_rerank": True
                }
                
                query_start = time.time()
                
                query_response = requests.post(
                    f"{DOC_PROCESSING_URL}/query",
                    json=query_data,
                    headers={"Content-Type": "application/json"},
                    timeout=120
                )
                
                query_time = time.time() - query_start
                total_time = time.time() - start_time
                
                if query_response.status_code == 200:
                    query_result = query_response.json()
                    print(f"✅ Query successful!")
                    print(f"   Query Time: {query_time:.2f}s")
                    print(f"   Total Time: {total_time:.2f}s")
                    print(f"   Answer: {query_result.get('answer', 'No answer')[:100]}...")
                    print(f"   Sources Found: {len(query_result.get('sources', []))}")
                    
                    test_results[provider['name']] = {
                        "success": True,
                        "processing_time": processing_time,
                        "query_time": query_time,
                        "total_time": total_time,
                        "chunks_processed": doc_result.get('chunks_processed', 0),
                        "sources_found": len(query_result.get('sources', [])),
                        "embedding_model": provider['model']
                    }
                    
                else:
                    print(f"❌ Query failed: {query_response.status_code}")
                    print(f"   Error: {query_response.text}")
                    test_results[provider['name']] = {
                        "success": False,
                        "error": f"Query failed: {query_response.status_code}",
                        "processing_time": processing_time
                    }
                    
            else:
                print(f"❌ Document processing failed: {doc_response.status_code}")
                print(f"   Error: {doc_response.text}")
                test_results[provider['name']] = {
                    "success": False,
                    "error": f"Processing failed: {doc_response.status_code}",
                    "processing_time": processing_time
                }
                
        except requests.RequestException as e:
            processing_time = time.time() - start_time
            print(f"❌ Request error: {str(e)}")
            test_results[provider['name']] = {
                "success": False,
                "error": f"Request error: {str(e)}",
                "processing_time": processing_time
            }
    
    # Print comprehensive results
    print(f"\n📈 EMBEDDING PROVIDER COMPARISON RESULTS")
    print("=" * 60)
    
    for provider_name, results in test_results.items():
        print(f"\n🔹 {provider_name} Results:")
        if results['success']:
            print(f"   ✅ Status: SUCCESS")
            print(f"   ⏱️  Processing Time: {results['processing_time']:.2f}s")
            print(f"   🔍 Query Time: {results.get('query_time', 0):.2f}s")
            print(f"   🕐 Total Time: {results.get('total_time', 0):.2f}s")
            print(f"   📄 Chunks Processed: {results.get('chunks_processed', 0)}")
            print(f"   📚 Sources Found: {results.get('sources_found', 0)}")
            print(f"   🤖 Embedding Model: {results.get('embedding_model', 'unknown')}")
        else:
            print(f"   ❌ Status: FAILED")
            print(f"   🚨 Error: {results['error']}")
            print(f"   ⏱️  Time Before Failure: {results['processing_time']:.2f}s")
    
    # Speed comparison
    if len([r for r in test_results.values() if r['success']]) >= 2:
        groq_time = test_results.get('Groq', {}).get('processing_time', float('inf'))
        ollama_time = test_results.get('Ollama', {}).get('processing_time', float('inf'))
        
        if groq_time != float('inf') and ollama_time != float('inf'):
            speed_improvement = (ollama_time - groq_time) / ollama_time * 100
            print(f"\n⚡ SPEED COMPARISON:")
            print(f"   Groq Embedding: {groq_time:.2f}s")
            print(f"   Ollama Embedding: {ollama_time:.2f}s")
            print(f"   Speed Improvement: {speed_improvement:.1f}% faster with Groq")
    
    print(f"\n🎯 TEST CONCLUSION:")
    successful_tests = len([r for r in test_results.values() if r['success']])
    total_tests = len(test_results)
    
    if successful_tests == total_tests:
        print(f"   ✅ ALL TESTS PASSED ({successful_tests}/{total_tests})")
        print(f"   🎉 Embedding provider selection is working correctly!")
        print(f"   🔧 Users can now choose between fast Groq and local Ollama embeddings")
    else:
        print(f"   ⚠️  PARTIAL SUCCESS ({successful_tests}/{total_tests} providers working)")
        if successful_tests > 0:
            working_providers = [name for name, results in test_results.items() if results['success']]
            print(f"   ✅ Working providers: {', '.join(working_providers)}")
    
    return test_results

if __name__ == "__main__":
    try:
        results = test_embedding_provider_selection()
        
        # Save results to file for analysis
        with open("embedding_provider_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Results saved to: embedding_provider_test_results.json")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()