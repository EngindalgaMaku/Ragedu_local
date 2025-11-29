import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

url = "http://localhost:8007/api/aprag/topics/classify-question"
headers = {"Content-Type": "application/json"}
payload = {
    "question": "What is photosynthesis?",
    "session_id": "6f3318202dd81b5fcab7b6621a6f4807",
    "user_id": "test_user_1",
    "interaction_id": 999  # A test interaction ID
}

try:
    # First, get the topics to see what we're working with
    topics_url = f"http://localhost:8007/api/aprag/topics/session/{payload['session_id']}"
    topics_response = requests.get(topics_url, headers=headers)
    logger.info(f"Topics Response Status: {topics_response.status_code}")
    logger.info(f"Topics: {topics_response.text}")
    
    # Now call the classify endpoint
    logger.info("Calling classify-question endpoint...")
    response = requests.post(url, headers=headers, json=payload)
    
    logger.info(f"Status Code: {response.status_code}")
    logger.info(f"Response: {response.text}")
    
    # Print the response in a more readable format
    if response.status_code == 200:
        try:
            data = response.json()
            print("\n=== Classification Result ===")
            print(f"Topic ID: {data.get('topic_id', 'Not found')}")
            print(f"Confidence: {data.get('confidence_score', 'N/A')}")
            print(f"Question Type: {data.get('question_type', 'N/A')}")
            print(f"Complexity: {data.get('question_complexity', 'N/A')}")
        except Exception as e:
            print(f"Error parsing response: {e}")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
