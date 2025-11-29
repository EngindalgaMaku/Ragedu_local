# Test doğrudan API Gateway'e (Next.js bypass)
import requests
import json

# Test payload
payload = {
    "session_id": "6f3318202dd81b5fcab7b6621a6f4807",
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

print("=" * 60)
print("TEST: Doğrudan API Gateway'e (http://localhost:8000)")
print("=" * 60)
try:
    response = requests.post(
        "http://localhost:8000/api/aprag/hybrid-rag/query",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    print(f"Status Code: {response.status_code}")
    print(f"Status Text: {response.reason}")
    print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"\n✅ JSON Response Keys: {list(json_response.keys())}")
            if 'answer' in json_response:
                print(f"Answer length: {len(json_response['answer'])} chars")
        except Exception as e:
            print(f"\n❌ JSON Parse Error: {e}")
            print(f"Response text (first 500): {response.text[:500]}")
    else:
        print(f"\n❌ Error Response: {response.text[:500]}")
except Exception as e:
    print(f"❌ Request Failed: {e}")
    import traceback
    traceback.print_exc()




