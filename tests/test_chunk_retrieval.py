#!/usr/bin/env python3
"""
Test script to check chunk retrieval with source_files metadata
"""
import requests
import json

def test_chunk_retrieval():
    """Test retrieving chunks to see if source_files is properly parsed"""
    
    session_id = "test_session_123"
    
    print(f"🧪 Testing chunk retrieval for session: {session_id}")
    
    try:
        # Call the chunks endpoint
        response = requests.get(
            f"http://localhost:8003/sessions/{session_id}/chunks",
            timeout=30
        )
        
        print(f"📤 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            chunks = data.get("chunks", [])
            
            print(f"📤 Retrieved {len(chunks)} chunks")
            
            if chunks:
                first_chunk = chunks[0]
                print(f"📤 First chunk document_name: {first_chunk.get('document_name')}")
                print(f"📤 First chunk metadata: {json.dumps(first_chunk.get('chunk_metadata', {}), indent=2)}")
                
                # Check if document_name is now correct (not "Unknown")
                if first_chunk.get('document_name') != "Unknown":
                    print("✅ Document name is correctly retrieved!")
                else:
                    print("❌ Document name is still showing as 'Unknown'")
            else:
                print("⚠️ No chunks found")
                
            print(f"📤 Full response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

if __name__ == "__main__":
    test_chunk_retrieval()