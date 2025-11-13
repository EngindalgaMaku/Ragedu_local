import requests
import time

endpoints = ['/health', '/models', '/models/list']

for endpoint in endpoints:
    try:
        print(f'Testing {endpoint}...')
        response = requests.get(f'http://localhost:8000{endpoint}', timeout=5)
        print(f'  Status: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            if 'models' in data:
                print(f'  Models: {len(data["models"])} found')
                for model in data['models']:
                    print(f'    - {model}')
            else:
                print(f'  Response: {data}')
        else:
            print(f'  Error: {response.text[:200]}')
    except requests.exceptions.Timeout:
        print('  TIMEOUT after 5s')
    except Exception as e:
        print(f'  ERROR: {e}')
    print()