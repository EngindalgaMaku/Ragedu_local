import requests
import json

url = "http://localhost:8007/api/aprag/topics/extract"
headers = {"Content-Type": "application/json"}
payload = {
    "session_id": "6f3318202dd81b5fcab7b6621a6f4807",
    "options": {
        "include_subtopics": True
    }
}

try:
    print("Extracting topics...")
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)
except Exception as e:
    print(f"Error: {e}")
