#!/usr/bin/env python3
"""
Test script for the new GET /documents/markdown/{filename} endpoint
"""
import requests
import json

# API base URL
API_URL = "http://localhost:8000"

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

def test_get_markdown_content():
    """Test the new GET /documents/markdown/{filename} endpoint"""
    
    print("🧪 Testing GET /documents/markdown/{filename} endpoint...")
    
    # First, get the list of available markdown files
    print("📋 Getting available markdown files...")
    try:
        response = requests.get(f"{API_URL}/documents/list-markdown")
        if response.status_code == 200:
            files_data = response.json()
            markdown_files = files_data['markdown_files']
            print(f"✅ Found {files_data['count']} markdown files: {markdown_files}")
            
            if files_data['count'] == 0:
                print("❌ No markdown files found to test with")
                return
                
        else:
            print(f"❌ Failed to list markdown files: {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Error listing markdown files: {e}")
        return
    
    # Test 1: Get content of existing file
    if markdown_files:
        test_file = markdown_files[0]
        print(f"\n📖 Test 1: Getting content of existing file '{test_file}'...")
        
        try:
            response = requests.get(f"{API_URL}/documents/markdown/{test_file}")
            if response.status_code == 200:
                result = response.json()
                content = result['content']
                print("✅ Successfully retrieved markdown content!")
                print(f"📊 Content length: {len(content)} characters")
                print(f"📄 First 200 characters: {content[:200]}...")
            else:
                print(f"❌ Failed to get markdown content: {response.status_code}")
                print("Response:", response.text)
                
        except Exception as e:
            print(f"❌ Error getting markdown content: {e}")
    
    # Test 2: Test with non-existent file (should return 404)
    print(f"\n📖 Test 2: Testing with non-existent file...")
    try:
        response = requests.get(f"{API_URL}/documents/markdown/nonexistent_file.md")
        if response.status_code == 404:
            print("✅ Correctly returned 404 for non-existent file")
        else:
            print(f"⚠️  Expected 404 but got {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print(f"❌ Error testing non-existent file: {e}")
    
    # Test 3: Test path traversal protection
    print(f"\n🔐 Test 3: Testing path traversal protection...")
    path_traversal_attempts = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        "../../src/api/main.py",
        "../uploads/secret.txt"
    ]
    
    for attempt in path_traversal_attempts:
        try:
            response = requests.get(f"{API_URL}/documents/markdown/{attempt}")
            if response.status_code in [400, 404]:
                print(f"✅ Path traversal blocked for: {attempt}")
            else:
                print(f"⚠️  Potential security issue - got {response.status_code} for: {attempt}")
        except Exception as e:
            print(f"❌ Error testing path traversal for {attempt}: {e}")
    
    # Test 4: Test filename without .md extension
    if markdown_files:
        test_file_without_ext = markdown_files[0].replace('.md', '')
        print(f"\n📖 Test 4: Testing filename without .md extension: '{test_file_without_ext}'...")
        
        try:
            response = requests.get(f"{API_URL}/documents/markdown/{test_file_without_ext}")
            if response.status_code == 200:
                result = response.json()
                print("✅ Successfully handled filename without .md extension!")
                print(f"📊 Content length: {len(result['content'])} characters")
            else:
                print(f"❌ Failed to handle filename without extension: {response.status_code}")
                print("Response:", response.text)
        except Exception as e:
            print(f"❌ Error testing filename without extension: {e}")

def test_complete_workflow():
    """Test the complete workflow: list files -> get content"""
    print("\n🔄 Testing complete workflow...")
    
    try:
        # Step 1: List markdown files
        list_response = requests.get(f"{API_URL}/documents/list-markdown")
        if list_response.status_code != 200:
            print("❌ Failed to list markdown files")
            return
        
        files_data = list_response.json()
        if files_data['count'] == 0:
            print("❌ No markdown files to test with")
            return
        
        # Step 2: Get content of each file
        for filename in files_data['markdown_files']:
            content_response = requests.get(f"{API_URL}/documents/markdown/{filename}")
            if content_response.status_code == 200:
                content_data = content_response.json()
                print(f"✅ Successfully retrieved content for {filename} ({len(content_data['content'])} chars)")
            else:
                print(f"❌ Failed to get content for {filename}: {content_response.status_code}")
        
        print("✅ Complete workflow test successful!")
        
    except Exception as e:
        print(f"❌ Error in complete workflow test: {e}")

if __name__ == "__main__":
    print("🎯 Testing New GET /documents/markdown/{filename} Endpoint")
    print("=" * 60)
    
    if test_health():
        test_get_markdown_content()
        test_complete_workflow()
    else:
        print("❌ API is not available. Make sure the server is running.")
    
    print("\n✅ All tests completed!")