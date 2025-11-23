import requests
sessions = requests.get('http://localhost:8000/sessions').json()
print('Sessions:')
for s in (sessions if isinstance(sessions, list) else sessions.get('sessions', [])):
    print(f'  - {s.get("session_id", s.get("id", "unknown"))}: {s.get("name", "Unnamed")}')



