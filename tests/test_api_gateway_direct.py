#!/usr/bin/env python3
"""
Direct API Gateway Test - Check if routing is working correctly
"""
import requests
import json

API_GATEWAY_URL = "https://api-gateway-awe3elsvra-ew.a.run.app"

def test_gateway_config():
    """Check API Gateway configuration"""
    print("🔍 Checking API Gateway configuration...")
    
    try:
        response = requests.get(f"{API_GATEWAY_URL}/", timeout=10)
        if response.status_code == 200:
            config = response.json()
            print("✅ API Gateway config:")
            print(json.dumps(config, indent=2))
        else:
            print(f"❌ Failed to get config: HTTP {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_rag_query():
    """Test RAG query through API Gateway"""
    print("\n🤖 Testing RAG query through API Gateway...")
    
    try:
        data = {
            "session_id": "test_session_123",
            "query": "Bu bir test sorusudur",
            "top_k": 3
        }
        
        response = requests.post(
            f"{API_GATEWAY_URL}/rag/query",
            json=data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '')
            
            if "lightweight model inference service" in answer.lower():
                print("⚠️ WARNING: Still getting old error message!")
                return False
            else:
                print("✅ Getting correct response!")
                return True
        else:
            print(f"❌ Query failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 API Gateway Direct Test")
    print("=" * 50)
    
    test_gateway_config()
    test_rag_query()