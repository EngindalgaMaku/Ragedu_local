import requests
import json

# Test data
test_data = {
    'text': '# Test Header\nThis is some test content for semantic chunking.\n\n## Sub Header\nMore content here.',
    'metadata': {'test': True},
    'collection_name': 'test_collection',
    'chunk_size': 500,
    'chunk_overlap': 50
}

try:
    response = requests.post('http://localhost:8080/process-and-store', json=test_data, timeout=10)
    print(f'Status Code: {response.status_code}')
    if response.status_code == 200:
        print('SUCCESS: Text processing is working!')
        result = response.json()
        print(f'Chunks processed: {result.get("chunks_processed", 0)}')
        print(f'Collection name: {result.get("collection_name")}')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Request failed: {e}')