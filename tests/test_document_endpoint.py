import requests
import json

# Test the document processing endpoint
url = 'http://localhost:8000/documents/process-and-store'

# Form data as expected by the endpoint
data = {
    'session_id': 'test',
    'markdown_files': json.dumps([{'title': 'test', 'content': 'test content'}])
}

try:
    response = requests.post(url, data=data, timeout=30)
    print(f'Status: {response.status_code}')
    print(f'Response: {response.text}')
except Exception as e:
    print(f'Error: {e}')