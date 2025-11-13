#!/usr/bin/env python3
"""
Test script for the new markdown documents endpoint
"""
import requests
import json

# API base URL - using Docker container
API_URL = "http://localhost:8000"

def test_add_markdown_documents():
    """Test the new add-markdown-documents endpoint"""
    
    print("🧪 Testing new add-markdown-documents endpoint...")
    
    # First, let's check what markdown files are available
    print("📋 Checking available markdown files...")
    try:
        response = requests.get(f"{API_URL}/documents/list-markdown")
        if response.status_code == 200:
            files = response.json()
            print(f"✅ Found {files['count']} markdown files:")
            for file in files['markdown_files']:
                print(f"  - {file}")
            
            if files['count'] == 0:
                print("❌ No markdown files found to test with")
                return
                
        else:
            print(f"❌ Failed to list markdown files: {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Error listing markdown files: {e}")
        return
    
    # Get list of sessions to test with
    print("\n📋 Getting available sessions...")
    try:
        response = requests.get(f"{API_URL}/sessions")
        if response.status_code == 200:
            sessions = response.json()
            print(f"✅ Found {len(sessions)} sessions")
            
            if len(sessions) == 0:
                print("❌ No sessions found to test with")
                print("💡 Please create a session first")
                return
                
            # Use the first session
            test_session = sessions[0]
            session_id = test_session['session_id']
            print(f"📝 Using session: {test_session['name']} ({session_id})")
            
        else:
            print(f"❌ Failed to get sessions: {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Error getting sessions: {e}")
        return
    
    # Test the new endpoint with first available markdown file
    print(f"\n🚀 Testing add-markdown-documents endpoint...")
    test_files = files['markdown_files'][:1]  # Use just the first file for testing
    
    payload = {
        "session_id": session_id,
        "markdown_files": test_files
    }
    
    try:
        response = requests.post(
            f"{API_URL}/sessions/add-markdown-documents",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint test successful!")
            print(f"📊 Result:")
            print(f"  - Success: {result['success']}")
            print(f"  - Processed files: {result['processed_count']}")
            print(f"  - Total chunks added: {result['total_chunks_added']}")
            print(f"  - Message: {result['message']}")
            
            if result.get('errors'):
                print(f"⚠️  Errors:")
                for error in result['errors']:
                    print(f"    - {error}")
            
        else:
            print(f"❌ Endpoint test failed: {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")

def test_health():
    """Test if API is accessible"""
    print("🏥 Testing API health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Testing RAG3 Markdown Documents Endpoint")
    print("=" * 50)
    
    if test_health():
        test_add_markdown_documents()
    
    print("\n✅ Test completed!")