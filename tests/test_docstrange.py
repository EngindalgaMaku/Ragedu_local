import requests

# Test DocStrange service directly
files = {'file': ('test.pdf', b'%PDF-1.4\ntest content', 'application/pdf')}

try:
    response = requests.post('http://localhost:8005/convert/pdf-to-markdown', files=files, timeout=10)
    print(f'Status: {response.status_code}')
    print(f'Response: {response.text[:500]}')
except Exception as e:
    print(f'Error: {e}')
