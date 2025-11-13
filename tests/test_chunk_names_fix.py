#!/usr/bin/env python3
"""
Test chunk names fix - Check if document names show correctly instead of "unknown"
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_chunk_names_fix():
    print("🧪 Testing Chunk Names Fix...")
    
    # Use existing session or create a new one
    print("📝 Checking for existing sessions...")
    list_response = requests.get(f"{API_BASE_URL}/sessions")
    
    if list_response.status_code == 200:
        sessions = list_response.json()
        if sessions:
            session_id = sessions[0]["session_id"]
            print(f"✅ Using existing session: {session_id}")
        else:
            print("❌ No sessions found")
            return
    else:
        print(f"❌ Failed to list sessions: {list_response.status_code}")
        return
    
    # Create test content with proper filename
    test_content = """# Fizik ve Hareket Kanunları

## Newton'un Hareket Kanunları

### Birinci Kanun (Eylemsizlik Kanunu)
Bir cisim, üzerine net bir kuvvet etki etmediği sürece durgunluk halinde durgunluk halinde kalmaya, hareket halinde ise aynı hızla doğru çizgide hareket etmeye devam eder.

### İkinci Kanun (F = ma)
Bir cisme uygulanan net kuvvet, cismin kütlesi ile ivmesinin çarpımına eşittir. Bu, kuvvet, kütle ve ivme arasındaki temel ilişkiyi tanımlar.

### Üçüncü Kanun (Etki-Tepki)
Her etkiye eşit ve zıt yönde bir tepki vardır. Bu kanun, kuvvetlerin çiftler halinde geldiğini açıklar.

Bu kanunlar modern fiziğin temelini oluşturur."""
    
    try:
        # Direct to document processing service to ensure our fix is tested
        print("📤 Processing content with specific filename...")
        
        payload = {
            'session_id': session_id,
            'text': test_content,
            'metadata': {
                'session_id': session_id,
                'source_files': ['fizik_hareket_kanunlari.md'],  # Specific filename that should appear
                'embedding_model': 'mixedbread-ai/mxbai-embed-large-v1',
                'chunk_strategy': 'semantic'
            },
            'collection_name': f'session_{session_id}',
            'chunk_size': 300,
            'chunk_overlap': 50
        }
        
        process_response = requests.post(
            f"{API_BASE_URL.replace('8000', '8003')}/process-and-store",
            json=payload,
            timeout=120
        )
        
        if process_response.status_code == 200:
            result = process_response.json()
            print(f"✅ Content processed: {result.get('chunks_processed', 0)} chunks")
        else:
            print(f"❌ Processing failed: {process_response.status_code}")
            print(f"Response: {process_response.text}")
            return
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Now test chunk retrieval
        print("🔍 Testing chunk retrieval...")
        chunks_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/chunks")
        
        if chunks_response.status_code == 200:
            chunks_data = chunks_response.json()
            chunks = chunks_data.get('chunks', [])
            
            print(f"📊 Retrieved {len(chunks)} chunks")
            
            # Check document names
            document_names = set()
            for i, chunk in enumerate(chunks):
                doc_name = chunk.get('document_name', 'Unknown')
                document_names.add(doc_name)
                print(f"📝 Chunk {i+1}: '{doc_name}' - {chunk.get('chunk_text', '')[:50]}...")
            
            # Analyze results
            print(f"\n📋 Document names found: {list(document_names)}")
            
            if "fizik_hareket_kanunlari.md" in document_names:
                print("✅ SUCCESS: Document name fix is working! Specific filename appears in chunks.")
            elif "Unknown" in document_names and len(document_names) == 1:
                print("❌ FAILED: Still showing 'Unknown' - fix not working properly")
            else:
                print(f"⚠️ PARTIAL: Mixed results - {document_names}")
                
        else:
            print(f"❌ Failed to retrieve chunks: {chunks_response.status_code}")
            print(f"Response: {chunks_response.text}")
            
    except Exception as e:
        print(f"❌ Test error: {str(e)}")

def main():
    print("🚀 Testing Chunk Names Fix")
    print("=" * 50)
    test_chunk_names_fix()
    print("=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    main()