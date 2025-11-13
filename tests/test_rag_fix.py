#!/usr/bin/env python3
"""
Quick test to verify RAG pipeline is working after ChromaDB fix
"""

import requests
import json

# Test RAG query through API Gateway
def test_rag_query():
    url = "https://api-gateway-1051060211087.europe-west1.run.app/rag/query"
    
    payload = {
        "session_id": "test_session",
        "query": "Test query",
        "top_k": 3,
        "use_rerank": True,
        "min_score": 0.1,
        "max_context_chars": 8000,
        "model": "llama-3.1-8b-instant"
    }
    
    try:
        print("🤖 Testing RAG query through API Gateway...")
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', 'No answer')
            sources_count = len(result.get('sources', []))
            
            print(f"✅ Answer: {answer}")
            print(f"📚 Sources: {sources_count}")
            
            # Check if we still get the old error message
            if "lightweight model inference service" in answer.lower():
                print("❌ WARNING: Still getting old error message!")
                return False
            else:
                print("✅ SUCCESS: RAG system is working!")
                return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing RAG Pipeline Fix")
    print("=" * 40)
    test_rag_query()