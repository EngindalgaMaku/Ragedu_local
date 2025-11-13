#!/usr/bin/env python3
"""
Test script to verify the data transformation fix is working correctly
after restarting the API Gateway container.
"""

import requests
import json

def test_process_and_store_endpoint():
    """Test the /documents/process-and-store endpoint"""
    print("Testing /documents/process-and-store endpoint...")
    
    url = "http://localhost:8000/documents/process-and-store"
    
    # Test data in form format (as expected by the endpoint)
    data = {
        'session_id': 'test_session',
        'markdown_files': '["BIYOLOJI VE CANLILARIN ORTAK ÖZELLIKLERI -I.md"]',
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'chunk_strategy': 'semantic',
        'embedding_model': 'mixedbread-ai/mxbai-embed-large-v1'
    }
    
    try:
        response = requests.post(url, data=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text[:1000]}...")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Endpoint responded successfully!")
            try:
                result = response.json()
                print("✅ SUCCESS: Valid JSON response received")
                return True
            except json.JSONDecodeError:
                print("⚠️  Warning: Response is not valid JSON")
                return False
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API Gateway")
        return False
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_health_endpoint():
    """Test that the API Gateway is running"""
    print("Testing API Gateway health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Gateway is healthy")
            return True
        else:
            print(f"❌ API Gateway health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Gateway health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing API Gateway Data Transformation Fix\n")
    
    # First check if API Gateway is healthy
    if test_health_endpoint():
        print()
        # Then test the actual endpoint
        success = test_process_and_store_endpoint()
        
        print("\n" + "="*50)
        if success:
            print("✅ DATA TRANSFORMATION FIX IS WORKING!")
        else:
            print("❌ DATA TRANSFORMATION FIX NEEDS INVESTIGATION")
    else:
        print("❌ API Gateway is not responding. Container may not be running properly.")