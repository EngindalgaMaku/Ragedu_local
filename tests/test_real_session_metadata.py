#!/usr/bin/env python3
"""
Test script with realistic session ID format to verify metadata fix
"""
import requests
import json
import hashlib

def generate_session_id():
    """Generate a realistic 32-char hex session ID"""
    import random
    import string
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    return hashlib.md5(random_string.encode()).hexdigest()

def test_realistic_metadata():
    """Test with realistic session ID and metadata"""
    
    # Generate a proper 32-char hex session ID
    session_id = generate_session_id()
    print(f"🧪 Testing with realistic session ID: {session_id}")
    
    # Test data
    test_text = """
    # Biology Chapter 1: Living Organisms
    
    Living organisms share several common characteristics:
    
    ## Basic Properties
    1. Organization - All living things are organized
    2. Metabolism - Chemical reactions for energy
    3. Growth - Increase in size and complexity
    4. Reproduction - Ability to produce offspring
    
    ## Cellular Structure
    All living organisms are made of cells, which are the basic unit of life.
    """
    
    # Test metadata with source_files 
    test_metadata = {
        "session_id": session_id,
        "embedding_model": "mxbai-embed-large", 
        "chunk_strategy": "semantic",
        "source_files": ["Biology_Chapter_1.md", "Living_Organisms.pdf"]  # Multiple files
    }
    
    # Request payload  
    payload = {
        "text": test_text.strip(),
        "metadata": test_metadata,
        "collection_name": f"session_{session_id}",  # Proper format
        "chunk_size": 800,
        "chunk_overlap": 100
    }
    
    print("📤 Processing document with source_files metadata...")
    
    try:
        # 1. Process and store
        response = requests.post(
            "http://localhost:8003/process-and-store",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Document processing failed: {response.text}")
            return
            
        print(f"✅ Document processed successfully: {response.json()}")
        
        # 2. Retrieve chunks
        print(f"📤 Retrieving chunks for session: {session_id}")
        
        chunks_response = requests.get(
            f"http://localhost:8003/sessions/{session_id}/chunks",
            timeout=30
        )
        
        if chunks_response.status_code != 200:
            print(f"❌ Chunk retrieval failed: {chunks_response.text}")
            return
            
        data = chunks_response.json()
        chunks = data.get("chunks", [])
        
        print(f"📤 Retrieved {len(chunks)} chunks")
        
        if chunks:
            for i, chunk in enumerate(chunks):
                document_name = chunk.get('document_name', 'N/A')
                metadata = chunk.get('chunk_metadata', {})
                
                print(f"📄 Chunk {i+1}:")
                print(f"   Document Name: {document_name}")
                print(f"   Stored source_files: {metadata.get('source_files', 'NOT FOUND')}")
                
                # Check if fix worked
                if document_name != "Unknown":
                    print(f"   ✅ SUCCESS: Document name correctly showing as '{document_name}'")
                else:
                    print(f"   ❌ FAILED: Document name still shows as 'Unknown'")
                
        else:
            print("⚠️ No chunks found")
                
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_realistic_metadata()