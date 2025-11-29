import requests
import json

# Test the classify-question endpoint
url = "http://localhost:8007/api/aprag/topics/classify-question"
headers = {"Content-Type": "application/json"}

# Test data
payload = {
    "question": "What is photosynthesis?",
    "session_id": "6f3318202dd81b5fcab7b6621a6f4807",
    "user_id": "test_user_1",
    "interaction_id": 12345
}

try:
    print(f"Sending request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)
    
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response: {e.response.text}")
