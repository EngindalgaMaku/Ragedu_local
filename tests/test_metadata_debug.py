#!/usr/bin/env python3
"""
Test script to debug metadata processing in document-processing-service
"""
import requests
import json

def test_metadata_processing():
    """Test the document processing endpoint with metadata containing source_files"""
    
    # Test data
    test_text = """
    # Test Document
    This is a test document to verify metadata processing.
    It contains some sample text for chunking.
    """
    
    # Test metadata with source_files (this should be the issue)
    test_metadata = {
        "session_id": "test_session_123",
        "embedding_model": "mxbai-embed-large", 
        "chunk_strategy": "semantic",
        "source_files": ["test_document.md"]  # This is a list - should be causing the issue
    }
    
    # Request payload
    payload = {
        "text": test_text.strip(),
        "metadata": test_metadata,
        "collection_name": "session_test_session_123",
        "chunk_size": 500,
        "chunk_overlap": 100
    }
    
    print("🧪 Testing document processing with source_files metadata...")
    print(f"📤 Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Call the document processing service (port 8003 from docker-compose)
        response = requests.post(
            "http://localhost:8003/process-and-store",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📤 Response status: {response.status_code}")
        print(f"📤 Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Document processing succeeded!")
        else:
            print(f"❌ Document processing failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

if __name__ == "__main__":
    test_metadata_processing()