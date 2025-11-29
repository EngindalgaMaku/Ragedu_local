import os
import requests
import json

# API endpoint to get session details
session_id = "6f3318202dd81b5fcab7b6621a6f4807"
api_url = f"http://localhost:8000/sessions/{session_id}"

try:
    print(f"Fetching session details from: {api_url}")
    response = requests.get(api_url, timeout=10)
    
    if response.status_code == 200:
        session_data = response.json()
        print("Session data:")
        print(json.dumps(session_data, indent=2, ensure_ascii=False))
        
        # Check for model in rag_settings
        rag_settings = session_data.get("rag_settings")
        if rag_settings and isinstance(rag_settings, str):
            try:
                rag_settings = json.loads(rag_settings)
                print("\nRAG Settings:")
                print(json.dumps(rag_settings, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("\nCould not parse rag_settings as JSON:")
                print(rag_settings)
        
        # Check metadata.rag_settings
        metadata = session_data.get("metadata")
        if metadata and isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
                print("\nMetadata:")
                print(json.dumps(metadata, indent=2, ensure_ascii=False))
                
                if "rag_settings" in metadata:
                    rag_meta = metadata["rag_settings"]
                    if isinstance(rag_meta, str):
                        try:
                            rag_meta = json.loads(rag_meta)
                            print("\nRAG Settings from Metadata:")
                            print(json.dumps(rag_meta, indent=2, ensure_ascii=False))
                        except json.JSONDecodeError:
                            print("\nCould not parse metadata.rag_settings as JSON:")
                            print(rag_meta)
            except json.JSONDecodeError:
                print("\nCould not parse metadata as JSON:")
                print(metadata)
    else:
        print(f"Error: {response.status_code} - {response.text}")

except Exception as e:
    print(f"Error: {e}")
