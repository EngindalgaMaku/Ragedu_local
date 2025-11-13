"""
Test RAG Query Only - via API Gateway
Tests just the RAG query functionality that the user is experiencing
"""
import requests
import json

def test_rag_query_via_api_gateway():
    print("🔍 Testing RAG Query via API Gateway...")
    print("=" * 50)
    
    # Test configuration
    API_GATEWAY_URL = "http://localhost:8000"
    session_id = "5a3c7780d9a52090c426e9f81326cc74"  # From our working test
    
    # Step 1: Test RAG query via API Gateway
    print("\n📋 Testing RAG query via API Gateway...")
    try:
        query_payload = {
            "session_id": session_id,
            "query": "ChromaDB nedir ve nasıl çalışır?",
            "top_k": 3,
            "use_rerank": True,
            "min_score": 0.1,
            "max_context_chars": 8000
        }
        
        print(f"📤 Sending query to: {API_GATEWAY_URL}/rag/query")
        print(f"📤 Payload: {json.dumps(query_payload, indent=2, ensure_ascii=False)}")
        
        # Send RAG query via API Gateway (should now route to document-processing-service)
        response = requests.post(
            f"{API_GATEWAY_URL}/rag/query",
            json=query_payload,
            timeout=60
        )
        
        print(f"\n📥 RAG Query Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("🎉 RAG Query successful!")
            print(f"📄 Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # Extract key information
            answer = result.get('answer', 'No answer')
            sources = result.get('sources', [])
            
            print(f"\n📊 Analysis:")
            print(f"✅ Answer: {answer[:100]}...")
            print(f"✅ Sources: {len(sources)} sources found")
            
            # Check for 'undefined' values - THIS IS THE CRITICAL TEST
            response_text = json.dumps(result, ensure_ascii=False)
            if 'undefined' in response_text.lower():
                print("❌ WARNING: 'undefined' values detected in response!")
                print("This is the issue the user is experiencing.")
            else:
                print("✅ SUCCESS: No 'undefined' values in response!")
                print("The routing fix worked correctly.")
                
        else:
            print(f"❌ RAG query failed: {response.status_code}")
            print(f"Error details: {response.text}")
            
            # Check if this shows routing is still wrong
            if response.status_code == 404:
                print("💡 This may indicate routing issues or service unavailability.")
            elif response.status_code == 500:
                print("💡 This may indicate internal service errors.")
                
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("💡 Check if API Gateway is running on port 8000")
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout error: {e}")
        print("💡 Request took too long - may indicate service issues")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print("🚀 RAG query test finished!")

if __name__ == "__main__":
    test_rag_query_via_api_gateway()