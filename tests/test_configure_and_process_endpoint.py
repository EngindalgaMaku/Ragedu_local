#!/usr/bin/env python3
"""
Test script for the new /rag/configure-and-process endpoint
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_configure_and_process_endpoint():
    """Test the new RAG configure and process endpoint"""
    
    print("🧪 Testing /rag/configure-and-process endpoint...")
    
    # First, let's get available sessions
    print("\n📋 Getting available sessions...")
    sessions_response = requests.get(f"{BASE_URL}/sessions")
    
    if sessions_response.status_code != 200:
        print(f"❌ Failed to get sessions: {sessions_response.status_code}")
        return False
    
    sessions = sessions_response.json()
    print(f"✅ Found {len(sessions)} sessions")
    
    if not sessions:
        print("📝 Creating a test session...")
        # Create a test session
        session_data = {
            "name": "RAG Test Session",
            "description": "Test session for RAG configure and process endpoint",
            "category": "general",
            "created_by": "test_user",
            "grade_level": "9",
            "subject_area": "Computer Science",
            "learning_objectives": ["Test RAG processing"],
            "tags": ["test", "rag"],
            "is_public": False
        }
        
        create_response = requests.post(f"{BASE_URL}/sessions", json=session_data)
        if create_response.status_code != 200:
            print(f"❌ Failed to create session: {create_response.status_code}")
            print(f"Response: {create_response.text}")
            return False
        
        session = create_response.json()
        session_id = session["session_id"]
        print(f"✅ Created test session: {session_id}")
    else:
        session_id = sessions[0]["session_id"]
        print(f"✅ Using existing session: {session_id}")
    
    # Get available markdown files
    print("\n📄 Getting available markdown files...")
    markdown_response = requests.get(f"{BASE_URL}/documents/list-markdown")
    
    if markdown_response.status_code != 200:
        print(f"❌ Failed to get markdown files: {markdown_response.status_code}")
        return False
    
    markdown_data = markdown_response.json()
    markdown_files = markdown_data["markdown_files"]
    print(f"✅ Found {len(markdown_files)} markdown files: {markdown_files}")
    
    if not markdown_files:
        print("❌ No markdown files available for testing")
        return False
    
    # Test the configure and process endpoint
    print("\n🔄 Testing RAG configure and process endpoint...")
    
    # Use the first available markdown file for testing
    test_files = markdown_files[:1]  # Use only the first file for testing
    
    test_data = {
        "session_id": session_id,
        "markdown_files": test_files,
        "chunk_strategy": "markdown",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "embedding_model": "mxbai-embed-large"
    }
    
    print(f"🔍 Request data: {json.dumps(test_data, indent=2)}")
    
    configure_response = requests.post(f"{BASE_URL}/rag/configure-and-process", json=test_data)
    
    print(f"📊 Response status: {configure_response.status_code}")
    print(f"📊 Response data: {json.dumps(configure_response.json(), indent=2)}")
    
    if configure_response.status_code != 200:
        print(f"❌ Failed to configure and process RAG: {configure_response.status_code}")
        print(f"Error details: {configure_response.text}")
        return False
    
    response_data = configure_response.json()
    
    # Verify response structure
    required_fields = ["success", "processed_files", "total_chunks", "message"]
    for field in required_fields:
        if field not in response_data:
            print(f"❌ Missing required field in response: {field}")
            return False
    
    if response_data["success"]:
        print(f"✅ Successfully processed {response_data['processed_files']} documents")
        print(f"✅ Created {response_data['total_chunks']} chunks")
        print(f"✅ Message: {response_data['message']}")
    else:
        print(f"❌ Processing failed: {response_data.get('message', 'Unknown error')}")
        if response_data.get("errors"):
            print(f"Errors: {response_data['errors']}")
        return False
    
    # Verify session was updated
    print("\n📈 Checking session metadata update...")
    updated_session_response = requests.get(f"{BASE_URL}/sessions/{session_id}")
    
    if updated_session_response.status_code == 200:
        updated_session = updated_session_response.json()
        print(f"✅ Session document count: {updated_session['document_count']}")
        print(f"✅ Session total chunks: {updated_session['total_chunks']}")
    else:
        print(f"⚠️ Could not verify session update: {updated_session_response.status_code}")
    
    # Test with invalid session
    print("\n🚫 Testing with invalid session...")
    invalid_test_data = {
        "session_id": "invalid_session_id",
        "markdown_files": test_files,
        "chunk_strategy": "markdown",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "embedding_model": "mxbai-embed-large"
    }
    
    invalid_response = requests.post(f"{BASE_URL}/rag/configure-and-process", json=invalid_test_data)
    
    if invalid_response.status_code == 404:
        print("✅ Correctly rejected invalid session")
    else:
        print(f"❌ Expected 404 for invalid session, got {invalid_response.status_code}")
        return False
    
    # Test with non-existent files
    print("\n🚫 Testing with non-existent files...")
    nonexistent_test_data = {
        "session_id": session_id,
        "markdown_files": ["nonexistent_file.md"],
        "chunk_strategy": "markdown",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "embedding_model": "mxbai-embed-large"
    }
    
    nonexistent_response = requests.post(f"{BASE_URL}/rag/configure-and-process", json=nonexistent_test_data)
    
    if nonexistent_response.status_code == 200:
        nonexistent_data = nonexistent_response.json()
        if not nonexistent_data["success"] and nonexistent_data.get("errors"):
            print("✅ Correctly handled non-existent file")
        else:
            print(f"❌ Expected failure for non-existent file, got success: {nonexistent_data}")
            return False
    else:
        print(f"❌ Expected 200 with error handling, got {nonexistent_response.status_code}")
        return False
    
    print("\n🎉 All tests passed! The /rag/configure-and-process endpoint is working correctly.")
    return True

def test_different_chunk_strategies():
    """Test different chunking strategies"""
    
    print("\n🔧 Testing different chunk strategies...")
    
    # Get sessions and markdown files
    sessions_response = requests.get(f"{BASE_URL}/sessions")
    if sessions_response.status_code != 200 or not sessions_response.json():
        print("❌ No sessions available for strategy testing")
        return False
    
    session_id = sessions_response.json()[0]["session_id"]
    
    markdown_response = requests.get(f"{BASE_URL}/documents/list-markdown")
    if markdown_response.status_code != 200 or not markdown_response.json()["markdown_files"]:
        print("❌ No markdown files available for strategy testing")
        return False
    
    test_file = markdown_response.json()["markdown_files"][0]
    
    # Test different configurations
    test_configs = [
        {
            "name": "Small chunks",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "chunk_strategy": "markdown"
        },
        {
            "name": "Large chunks", 
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "chunk_strategy": "markdown"
        }
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing {config['name']} configuration...")
        
        test_data = {
            "session_id": session_id,
            "markdown_files": [test_file],
            "chunk_strategy": config["chunk_strategy"],
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "embedding_model": "mxbai-embed-large"
        }
        
        response = requests.post(f"{BASE_URL}/rag/configure-and-process", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"✅ {config['name']}: {data['total_chunks']} chunks created")
            else:
                print(f"❌ {config['name']} failed: {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ {config['name']} request failed: {response.status_code}")
            return False
    
    print("✅ All chunk strategy tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting RAG configure and process endpoint tests...\n")
    
    try:
        # Main endpoint test
        if not test_configure_and_process_endpoint():
            print("❌ Main endpoint test failed")
            exit(1)
        
        # Strategy tests
        if not test_different_chunk_strategies():
            print("❌ Strategy tests failed") 
            exit(1)
        
        print("\n🎊 All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running on localhost:8000")
        exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        exit(1)