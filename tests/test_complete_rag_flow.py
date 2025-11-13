"""
Complete RAG Flow Test - API Gateway to Document Processing Service
Tests the full pipeline after fixing the routing issue
"""
import requests
import json
import time

def test_complete_rag_flow():
    print("🔍 Testing Complete RAG Flow via API Gateway...")
    print("=" * 60)
    
    # Test configuration
    API_GATEWAY_URL = "http://localhost:8000"
    session_id = "5a3c7780d9a52090c426e9f81326cc74"
    
    # Step 1: Test API Gateway health
    print("\n📋 Step 1: Testing API Gateway health...")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/health")
        print(f"API Gateway Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ API Gateway health check failed: {e}")
        return
    
    # Step 2: Test document processing via API Gateway
    print("\n📋 Step 2: Testing document processing via API Gateway...")
    try:
        # Prepare test data - simulating what the frontend would send
        test_payload = {
            "text": """Bu kapsamlı bir test belgesidir. ChromaDB vektör veritabanı, embeddings'lerin verimli şekilde depolanması ve sorgulanması için kullanılır.
            Semantik arama yetenekleri sağlar ve farklı döküman formatlarını destekler. RAG uygulamaları için hızlı benzerlik araması yapabilir.""",
            "metadata": {
                "session_id": session_id,
                "embedding_model": "mxbai-embed-large", 
                "chunk_strategy": "semantic"
            },
            "collection_name": f"session_{session_id}",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        # Send to document processing via API Gateway
        response = requests.post(
            f"{API_GATEWAY_URL}/documents/process-and-store",
            data={
                "session_id": session_id,
                "markdown_files": json.dumps(["test_doc.md"]),  # Simulating file list
                "chunk_strategy": "semantic",
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "embedding_model": "mxbai-embed-large"
            },
            timeout=120
        )
        
        print(f"Document Processing Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Document processing successful!")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # Extract key information
            chunks_processed = result.get("chunks_processed", 0)
            collection_name = result.get("collection_name", "unknown")
            print(f"📊 Processed: {chunks_processed} chunks in collection '{collection_name}'")
            
        else:
            print(f"❌ Document processing failed: {response.status_code}")
            print(f"Error: {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return
    
    # Step 3: Test RAG query via API Gateway (CRITICAL TEST)
    print("\n📋 Step 3: Testing RAG query via API Gateway...")
    try:
        query_payload = {
            "session_id": session_id,
            "query": "ChromaDB nedir ve nasıl çalışır?",
            "top_k": 3,
            "use_rerank": True,
            "min_score": 0.1,
            "max_context_chars": 8000
        }
        
        # Send RAG query via API Gateway (should now route to document-processing-service)
        response = requests.post(
            f"{API_GATEWAY_URL}/rag/query",
            json=query_payload,
            timeout=60
        )
        
        print(f"RAG Query Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("🎉 RAG Query successful!")
            print(f"Answer: {result.get('answer', 'No answer')}")
            print(f"Sources: {len(result.get('sources', []))} sources found")
            
            # Check for 'undefined' values
            if 'undefined' in str(result):
                print("⚠️  WARNING: 'undefined' values detected in response!")
            else:
                print("✅ No 'undefined' values in response!")
                
        else:
            print(f"❌ RAG query failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ RAG query error: {e}")
    
    print("\n" + "=" * 60)
    print("🚀 Complete RAG flow test finished!")

if __name__ == "__main__":
    test_complete_rag_flow()