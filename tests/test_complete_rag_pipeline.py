#!/usr/bin/env python3
"""
Complete RAG Pipeline Integration Test
Tests the entire workflow from document upload to RAG query
"""

import requests
import json
import time
import sys
from pathlib import Path

# Service URLs (updated for local docker-compose environment)
API_GATEWAY_URL = "http://localhost:8000"
DOCUMENT_PROCESSOR_URL = "http://localhost:8003"
# RAG functionality is handled by Document Processing Service, not a separate RAG service
RAG_SERVICE_URL = DOCUMENT_PROCESSOR_URL

def test_service_health():
    """Test all service health endpoints"""
    print("🩺 Testing service health...")
    
    services = {
        "API Gateway": f"{API_GATEWAY_URL}/health/services",
        "RAG Service": f"{RAG_SERVICE_URL}/health"
    }
    
    for service_name, health_url in services.items():
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                print(f"  ✅ {service_name}: OK")
                if service_name == "API Gateway":
                    health_data = response.json()
                    for svc, status in health_data.get("services", {}).items():
                        status_icon = "✅" if status.get("status") == "ok" else "❌"
                        print(f"    {status_icon} {svc}: {status.get('status', 'unknown')}")
            else:
                print(f"  ❌ {service_name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ {service_name}: {str(e)}")
    
    print()

def test_document_conversion():
    """Test document conversion to markdown"""
    print("📄 Testing document conversion...")
    
    # Create a test PDF content (mock)
    test_content = b"Mock PDF content for testing"
    
    try:
        files = {"file": ("test.pdf", test_content, "application/pdf")}
        response = requests.post(
            f"{API_GATEWAY_URL}/documents/convert-document-to-markdown",
            files=files,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Document converted: {result.get('markdown_filename')}")
            return result.get('markdown_filename')
        else:
            print(f"  ❌ Conversion failed: HTTP {response.status_code}")
            print(f"     Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"  ❌ Conversion error: {str(e)}")
        return None

def test_document_processing(markdown_files, session_id="test_session_123"):
    """Test document processing and storage"""
    print("⚙️ Testing document processing...")
    
    if not markdown_files:
        print("  ⏭️ Skipping - no markdown files to process")
        return False
    
    try:
        data = {
            "session_id": session_id,
            "markdown_files": json.dumps([markdown_files]),
            "chunk_strategy": "semantic",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "embedding_model": "mixedbread-ai/mxbai-embed-large-v1"
        }
        
        response = requests.post(
            f"{API_GATEWAY_URL}/documents/process-and-store",
            data=data,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Documents processed: {result.get('chunks_processed', 0)} chunks")
            return True
        else:
            print(f"  ❌ Processing failed: HTTP {response.status_code}")
            print(f"     Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Processing error: {str(e)}")
        return False

def test_rag_query(session_id="test_session_123"):
    """Test RAG query functionality"""
    print("🤖 Testing RAG query...")
    
    test_queries = [
        "Bu belge neden bahsediyor?",
        "Ana konu nedir?",
        "What is the main topic?"
    ]
    
    for query in test_queries:
        print(f"  📝 Query: '{query}'")
        
        try:
            data = {
                "session_id": session_id,
                "query": query,
                "top_k": 5,
                "use_rerank": True,
                "min_score": 0.1,
                "max_context_chars": 8000,
                "model": "llama-3.1-8b-instant"
            }
            
            response = requests.post(
                f"{API_GATEWAY_URL}/rag/query",
                json=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', 'No answer')
                sources_count = len(result.get('sources', []))
                
                print(f"    ✅ Answer: {answer[:100]}...")
                print(f"    📚 Sources: {sources_count}")
                
                if "lightweight model inference service" in answer.lower():
                    print(f"    ⚠️ WARNING: Still getting old error message!")
                    return False
                    
            else:
                print(f"    ❌ Query failed: HTTP {response.status_code}")
                print(f"       Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"    ❌ Query error: {str(e)}")
            return False
        
        time.sleep(1)  # Rate limiting
    
    return True

def test_document_processor_directly():
    """Test Document Processing Service directly (handles RAG functionality)"""
    print("🎯 Testing Document Processing Service directly...")
    
    try:
        # Test health
        response = requests.get(f"{DOCUMENT_PROCESSOR_URL}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✅ Document Processing Service health: {health}")
        else:
            print(f"  ❌ Document Processing Service health check failed: HTTP {response.status_code}")
            return False
            
        # Test direct query (this service handles RAG functionality)
        data = {
            "session_id": "test_session_123",
            "query": "Test query",
            "top_k": 3
        }
        
        response = requests.post(f"{DOCUMENT_PROCESSOR_URL}/query", json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Direct RAG query successful")
            print(f"     Answer: {result.get('answer', 'No answer')[:100]}...")
            
            # Check if we're still getting the old error message
            answer = result.get('answer', '').lower()
            if "lightweight model inference service" in answer:
                print(f"  ⚠️ WARNING: Still getting old error message!")
                return False
            return True
        else:
            print(f"  ❌ Direct RAG query failed: HTTP {response.status_code}")
            print(f"     Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Document Processing Service test error: {str(e)}")
        return False

def main():
    """Run complete pipeline test"""
    print("🚀 Starting Complete RAG Pipeline Integration Test")
    print("=" * 60)
    
    # Test 1: Service Health
    test_service_health()
    
    # Test 2: Document Processing Service Direct Test (handles RAG)
    rag_direct_success = test_document_processor_directly()
    print()
    
    # Test 3: Document Conversion
    markdown_file = test_document_conversion()
    print()
    
    # Test 4: Document Processing
    processing_success = test_document_processing(markdown_file)
    print()
    
    # Test 5: RAG Query via API Gateway
    query_success = test_rag_query()
    print()
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Document Processor Direct: {'PASS' if rag_direct_success else 'FAIL'}")
    print(f"✅ Document Conversion:    {'PASS' if markdown_file else 'FAIL'}")
    print(f"✅ Document Processing:    {'PASS' if processing_success else 'FAIL'}")
    print(f"✅ RAG Query via Gateway:  {'PASS' if query_success else 'FAIL'}")
    
    overall_success = all([rag_direct_success, processing_success, query_success])
    print(f"\n🎯 OVERALL: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if not overall_success:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check that all services are deployed and running")
        print("2. Verify service URLs in this script")
        print("3. Check CloudRun logs for detailed error messages")
        print("4. Ensure RAG_SERVICE_URL is set correctly in API Gateway")
        
        sys.exit(1)
    else:
        print("\n🎉 RAG Pipeline is working correctly!")

if __name__ == "__main__":
    main()