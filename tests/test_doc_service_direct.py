import requests

# Test the document processing service directly
url = 'http://localhost:8003/process-and-store'

# JSON data as expected by the document processing service
data = {
    "text": "This is a test document content for processing.",
    "metadata": {"title": "test document"},
    "collection_name": "test_collection",
    "chunk_size": 500,
    "chunk_overlap": 100
}

try:
    response = requests.post(url, json=data, timeout=30)
    print(f'Status: {response.status_code}')
    print(f'Response: {response.text}')
except Exception as e:
    print(f'Error: {e}')