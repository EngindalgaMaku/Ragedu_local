#!/usr/bin/env python3
"""
Complete end-to-end workflow validation test
Tests: Document processing → Chunk storage → RAG queries → Chunk retrieval
"""

import requests
import json
import uuid
import time

# Service URLs
DOCUMENT_PROCESSING_URL = "http://localhost:8003"

def test_complete_workflow():
    """Test the complete document processing and RAG workflow"""
    print("🔄 COMPLETE WORKFLOW VALIDATION")
    print("=" * 60)
    
    # Create test session
    test_session_id = str(uuid.uuid4()).replace("-", "")
    test_filename = "complete_workflow_test.md"
    
    # Test document with Turkish content (to match original error logs)
    test_content = """# Biyoloji Dersi - Hücre Yapısı

Bu belge hücre yapısı ile ilgili temel bilgileri içermektedir.

## Hücre Zarı

Hücre zarı, hücreyi dış ortamdan ayıran seçici geçirgen bir yapıdır. 
Fosfolipid çift katmandan oluşur ve hücrenin şeklini korur.

## Sitoplazma 

Sitoplazma, hücre zarı ile çekirdek arasındaki jel benzeri maddedir.
Hücresel aktivitelerin çoğu burada gerçekleşir.

## Çekirdek

Çekirdek, hücrenin kontrol merkezidir. DNA'yı içerir ve hücresel 
aktiviteleri yönetir.

### DNA ve Kromozomlar

DNA, genetik bilgiyi taşıyan moleküldür. Kromozomlar halinde organize olmuştur.

## Mitokondri

Mitokondri, hücrenin enerji santralıdır. ATP üretiminden sorumludur.

Bu yapılar hücrenin temel bileşenleridir ve yaşam için gereklidir."""
    
    print(f"🧪 Test Session ID: {test_session_id}")
    print(f"📄 Test Document: {test_filename}")
    print(f"📝 Content Length: {len(test_content)} characters")
    
    # Step 1: Process document
    print(f"\n{'='*20} STEP 1: DOCUMENT PROCESSING {'='*20}")
    
    payload = {
        "text": test_content,
        "metadata": {
            "session_id": test_session_id,
            "source_file": test_filename,
            "filename": test_filename,
            "embedding_model": "mxbai-embed-large",
            "chunk_strategy": "semantic",
            "subject": "Biyoloji",
            "topic": "Hücre Yapısı"
        },
        "collection_name": f"session_{test_session_id}",
        "chunk_size": 400,
        "chunk_overlap": 50
    }
    
    try:
        process_response = requests.post(
            f"{DOCUMENT_PROCESSING_URL}/process-and-store",
            json=payload,
            timeout=120
        )
        
        print(f"📊 Processing Status: {process_response.status_code}")
        
        if process_response.status_code == 200:
            result = process_response.json()
            chunks_created = result.get("chunks_processed", 0)
            print(f"✅ Document processed successfully")
            print(f"📦 Chunks created: {chunks_created}")
            print(f"🎯 Collection: {result.get('collection_name', 'Unknown')}")
        else:
            print(f"❌ Processing failed: {process_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False
    
    # Step 2: Retrieve and validate chunks
    print(f"\n{'='*20} STEP 2: CHUNK VALIDATION {'='*20}")
    
    try:
        chunks_response = requests.get(
            f"{DOCUMENT_PROCESSING_URL}/sessions/{test_session_id}/chunks",
            timeout=30
        )
        
        print(f"📊 Chunks Status: {chunks_response.status_code}")
        
        if chunks_response.status_code == 200:
            chunks_data = chunks_response.json()
            chunks = chunks_data.get("chunks", [])
            
            print(f"✅ Retrieved {len(chunks)} chunks")
            
            # Validate chunk titles
            proper_titles = 0
            for i, chunk in enumerate(chunks):
                document_name = chunk.get("document_name", "Unknown")
                chunk_text = chunk.get("chunk_text", "")[:100] + "..."
                
                print(f"  📄 Chunk {i+1}: '{document_name}' ({len(chunk.get('chunk_text', ''))} chars)")
                print(f"      Content: {chunk_text}")
                
                if document_name != "Unknown" and test_filename in document_name:
                    proper_titles += 1
            
            if proper_titles == len(chunks):
                print(f"🎉 All {len(chunks)} chunks have proper titles!")
            else:
                print(f"⚠️  Only {proper_titles}/{len(chunks)} chunks have proper titles")
                return False
                
        else:
            print(f"❌ Chunks retrieval failed: {chunks_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chunks error: {e}")
        return False
    
    # Step 3: Test RAG queries
    print(f"\n{'='*20} STEP 3: RAG QUERY TESTING {'='*20}")
    
    test_queries = [
        "Hücre zarı nedir?",
        "Mitokondri hangi işlevi görür?",
        "DNA nerede bulunur?"
    ]
    
    successful_queries = 0
    
    for i, query in enumerate(test_queries):
        print(f"\n📝 Query {i+1}: '{query}'")
        
        try:
            rag_payload = {
                "session_id": test_session_id,
                "query": query,
                "top_k": 3,
                "use_rerank": True,
                "model": "llama-3.1-8b-instant"
            }
            
            rag_response = requests.post(
                f"{DOCUMENT_PROCESSING_URL}/query",
                json=rag_payload,
                timeout=60
            )
            
            print(f"📊 Query Status: {rag_response.status_code}")
            
            if rag_response.status_code == 200:
                rag_result = rag_response.json()
                answer = rag_result.get("answer", "")
                sources = rag_result.get("sources", [])
                
                print(f"✅ Query successful")
                print(f"💬 Answer length: {len(answer)} characters")
                print(f"📚 Sources found: {len(sources)}")
                
                if len(answer) > 20 and len(sources) > 0:
                    successful_queries += 1
                    print(f"🎯 Answer preview: {answer[:100]}...")
                else:
                    print(f"⚠️  Weak response: short answer or no sources")
            else:
                print(f"❌ Query failed: {rag_response.text}")
                
        except Exception as e:
            print(f"❌ Query error: {e}")
    
    print(f"\n📊 RAG Query Results: {successful_queries}/{len(test_queries)} successful")
    
    # Final assessment
    print(f"\n{'='*20} FINAL ASSESSMENT {'='*20}")
    
    if successful_queries >= 2:  # At least 2/3 queries should work
        print("🎉 COMPLETE WORKFLOW VALIDATION: ✅ PASSED")
        print("📋 All key components working:")
        print("  ✅ Document processing with semantic chunking")
        print("  ✅ Metadata handling and chunk titles")
        print("  ✅ ChromaDB storage and retrieval") 
        print("  ✅ RAG query functionality")
        return True
    else:
        print("❌ COMPLETE WORKFLOW VALIDATION: ❌ FAILED")
        print(f"   Issues detected in RAG query functionality")
        return False

def main():
    success = test_complete_workflow()
    
    if success:
        print(f"\n🏆 ALL SYSTEMS OPERATIONAL!")
        print("The document processing pipeline is fully functional.")
    else:
        print(f"\n⚠️  ISSUES DETECTED!")
        print("Some components need attention.")

if __name__ == "__main__":
    main()