# test_hybrid_rag.py - Doğrudan endpoint'i test etmek için
import requests
import json
import sys

# Önce bir session ID alalım
print("=" * 60)
print("STEP 1: Getting session ID...")
print("=" * 60)
try:
    response = requests.get("http://localhost:3000/api/sessions", timeout=10)
    if response.status_code == 200:
        sessions = response.json()
        if sessions and len(sessions) > 0:
            session_id = sessions[0].get('session_id')
            session_name = sessions[0].get('name', 'Unknown')
            print(f"✅ Found session: {session_name} (ID: {session_id})")
        else:
            print("❌ No sessions found. Please create a session first.")
            sys.exit(1)
    else:
        print(f"❌ Failed to get sessions: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error getting sessions: {e}")
    sys.exit(1)

# Test payload
payload = {
    "session_id": session_id,
    "query": "mitoz bölünme nedir",
    "top_k": 5,
    "use_kb": True,
    "use_qa_pairs": True,
    "use_crag": True,
    "model": "qwen-plus",
    "embedding_model": "text-embedding-v4",
    "max_tokens": 2048,
    "temperature": 0.7
}

# Test 1: API Gateway üzerinden (frontend'in kullandığı)
print("\n" + "=" * 60)
print("TEST 1: API Gateway üzerinden (http://localhost:3000)")
print("=" * 60)
try:
    response = requests.post(
        "http://localhost:3000/api/aprag/hybrid-rag/query",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    print(f"Status Code: {response.status_code}")
    print(f"Status Text: {response.reason}")
    print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
    print(f"Content-Length: {response.headers.get('Content-Length', 'N/A')}")
    
    print(f"\nResponse Body (first 2000 chars):")
    print(response.text[:2000])
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"\n✅ JSON Response Keys: {list(json_response.keys())}")
            if 'answer' in json_response:
                print(f"Answer length: {len(json_response['answer'])} chars")
                print(f"Answer preview: {json_response['answer'][:200]}...")
            if 'sources' in json_response:
                print(f"Sources count: {len(json_response.get('sources', []))}")
            if 'debug_info' in json_response:
                print(f"Debug info available: {json_response['debug_info'] is not None}")
        except Exception as e:
            print(f"\n❌ JSON Parse Error: {e}")
            print(f"Response text type: {type(response.text)}")
    else:
        print(f"\n❌ Error Response (first 500 chars): {response.text[:500]}")
except Exception as e:
    print(f"❌ Request Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Doğrudan aprag-service'e (bypass API Gateway)
print("\n" + "=" * 60)
print("TEST 2: Doğrudan aprag-service'e (http://localhost:8007)")
print("=" * 60)
try:
    response = requests.post(
        "http://localhost:8007/api/aprag/hybrid-rag/query",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    print(f"Status Code: {response.status_code}")
    print(f"Status Text: {response.reason}")
    print(f"\nResponse Body (first 2000 chars):")
    print(response.text[:2000])
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"\n✅ JSON Response Keys: {list(json_response.keys())}")
            if 'answer' in json_response:
                print(f"Answer length: {len(json_response['answer'])} chars")
        except Exception as e:
            print(f"\n❌ JSON Parse Error: {e}")
except Exception as e:
    print(f"❌ Request Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)




