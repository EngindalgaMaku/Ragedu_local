#!/usr/bin/env python3
"""
Reranker Diagnosis Script - Add validation logs to confirm which reranker is being used
"""
import requests
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_reranker_services():
    """Check which reranker services are active and their configurations"""
    
    print("🔍 RERANKER SYSTEM DIAGNOSIS")
    print("="*50)
    
    # Check reranker service
    try:
        response = requests.get("http://localhost:8008/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ RERANKER SERVICE: Active")
            print(f"   Type: {info.get('reranker_type')}")
            print(f"   Configured: {info.get('configured_type')}")
            print(f"   Alibaba Available: {info.get('alibaba_available')}")
        else:
            print(f"❌ RERANKER SERVICE: Error {response.status_code}")
    except Exception as e:
        print(f"❌ RERANKER SERVICE: Offline - {e}")
    
    # Check model inference service reranker
    try:
        response = requests.post(
            "http://localhost:8002/rerank",
            json={"query": "test", "documents": ["test doc"]},
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ MODEL INFERENCE RERANKER: Active (proxy mode)")
        else:
            print(f"❌ MODEL INFERENCE RERANKER: Error {response.status_code}")
    except Exception as e:
        print(f"❌ MODEL INFERENCE RERANKER: Offline - {e}")
    
    # Check environment variables
    print(f"\n🔧 CONFIGURATION:")
    print(f"   USE_RERANKER_SERVICE: {os.getenv('USE_RERANKER_SERVICE', 'NOT SET')}")
    print(f"   RERANKER_TYPE: {os.getenv('RERANKER_TYPE', 'NOT SET')}")
    print(f"   ALIBABA_API_KEY: {'SET' if os.getenv('ALIBABA_API_KEY') else 'NOT SET'}")
    
    # Check if local ReRanker class is being imported
    try:
        from src.rag.re_ranker import ReRanker
        local_reranker = ReRanker()
        print(f"⚠️  LOCAL RERANKER: Active (THIS IS THE PROBLEM!)")
        print(f"   Model Info: {local_reranker.get_model_info()}")
    except Exception as e:
        print(f"✅ LOCAL RERANKER: Disabled - {e}")

def test_reranker_workflow():
    """Test the actual reranker workflow"""
    print(f"\n🧪 TESTING RERANKER WORKFLOW:")
    print("="*50)
    
    query = "mitoz bölünme nedir"
    documents = [
        "Mitoz, hücre bölünmesinin bir türüdür.",
        "Mitoz bölünmede kromozomlar eşit olarak dağıtılır.",
        "Bu süreç hücre çoğalması için gereklidir."
    ]
    
    # Test with direct reranker service call
    try:
        response = requests.post(
            "http://localhost:8008/rerank",
            json={
                "query": query,
                "documents": documents,
                "reranker_type": "alibaba"  # Explicitly request Alibaba
            },
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ DIRECT ALIBABA RERANKER: Success")
            print(f"   Used Type: {result.get('reranker_type')}")
            print(f"   Processing Time: {result.get('processing_time_ms')}ms")
        else:
            print(f"❌ DIRECT ALIBABA RERANKER: Error {response.status_code}")
    except Exception as e:
        print(f"❌ DIRECT ALIBABA RERANKER: Failed - {e}")
    
    # Test with document processing service (CRAG evaluation)
    try:
        response = requests.post(
            "http://localhost:8003/query",
            json={
                "query": query,
                "collection_name": "test_collection",
                "top_k": 3,
                "use_rerank": True,  # Enable CRAG evaluation
                "use_hybrid_search": True
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ DOCUMENT PROCESSING (CRAG): Success")
            # Look for reranker type in the response
            if 'documents' in result:
                for doc in result['documents']:
                    if 'metadata' in doc and 'reranker_type' in doc['metadata']:
                        print(f"   CRAG Reranker Type: {doc['metadata']['reranker_type']}")
                        break
        else:
            print(f"❌ DOCUMENT PROCESSING (CRAG): Error {response.status_code}")
    except Exception as e:
        print(f"❌ DOCUMENT PROCESSING (CRAG): Failed - {e}")

if __name__ == "__main__":
    check_reranker_services()
    test_reranker_workflow()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("="*50)
    print("1. Set USE_RERANKER_SERVICE=true in docker-compose.yml")
    print("2. Disable local ReRanker in src/rag/rag_chains.py")
    print("3. Ensure ALIBABA_API_KEY is properly set")
    print("4. Restart services to apply changes")