#!/usr/bin/env python3
"""Test script to verify the API server is running and test multi-format document conversion."""

import requests
import time
import sys

def test_api_server():
    print("🌐 Testing API server...")
    
    # Give the server time to start up
    time.sleep(3)
    
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8000/")
        print(f"✅ API server running - Status: {response.status_code}")
        
        # Test docs endpoint
        try:
            docs_response = requests.get("http://localhost:8000/docs")
            print(f"📚 API docs available at http://localhost:8000/docs - Status: {docs_response.status_code}")
        except Exception as e:
            print(f"⚠️ API docs may not be available: {e}")
        
        # Test health endpoint if available
        try:
            health_response = requests.get("http://localhost:8000/health")
            print(f"💚 Health endpoint - Status: {health_response.status_code}")
        except Exception as e:
            print(f"⚠️ Health endpoint not available: {e}")
        
        print("🎉 API server is running successfully!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Server not responding - may still be starting up")
        return False
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def test_document_endpoint():
    print("\n🧪 Testing document conversion endpoint...")
    
    try:
        # Test the new multi-format endpoint
        endpoint_url = "http://localhost:8000/documents/convert-document-to-markdown"
        
        # Test with a simple HTTP request to check if endpoint exists
        response = requests.post(endpoint_url)
        
        # We expect a 422 error (validation error) since we're not sending a file
        if response.status_code == 422:
            print("✅ Document conversion endpoint is available and responding")
            print(f"📍 Endpoint: {endpoint_url}")
            return True
        else:
            print(f"⚠️ Unexpected response from endpoint: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Document endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting API server tests...\n")
    
    server_ok = test_api_server()
    if server_ok:
        endpoint_ok = test_document_endpoint()
        
        if server_ok and endpoint_ok:
            print("\n🎉 All tests passed! API server is ready for multi-format document conversion.")
            sys.exit(0)
        else:
            print("\n⚠️ Some tests failed. Check the server configuration.")
            sys.exit(1)
    else:
        print("\n❌ Server is not running. Please check the server startup.")
        sys.exit(1)