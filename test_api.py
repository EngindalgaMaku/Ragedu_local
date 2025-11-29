import requests
import json

# API endpoint
url = "http://localhost:8007/api/aprag/topics/classify-question"

# Headers
headers = {
    "Content-Type": "application/json"
}

# Request payload
payload = {
    "question": "What is photosynthesis?",
    "session_id": "6f3318202dd81b5fcab7b6621a6f4807"
}

try:
    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    # Print response
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    
except Exception as e:
    print(f"Error: {e}")
