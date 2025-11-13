import requests

# Test API Gateway PDF endpoint
files = {'file': ('test.pdf', b'%PDF-1.4\ntest content', 'application/pdf')}

try:
    response = requests.post('http://localhost:8000/documents/convert-document-to-markdown', files=files, timeout=30)
    print(f'Status: {response.status_code}')
    print(f'Response: {response.text[:1000]}')
except Exception as e:
    print(f'Error: {e}')
