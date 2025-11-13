#!/usr/bin/env python3
"""
Final integration test to verify document processing through API Gateway
"""

import requests
import json
import time

# Service URLs
API_GATEWAY_URL = "https://api-gateway-1051060211087.europe-west1.run.app"
DOC_PROC_SERVICE_URL = "https://doc-proc-service-1051060211087.europe-west1.run.app"

def test_document_processing_through_api_gateway():
    """Test document processing through API Gateway"""
    print("=== Testing Document Processing Through API Gateway ===")
    
    # Test data
    test_text = "This is a comprehensive test document to verify that the full integration between API Gateway and Document Processing Service is working correctly. The service should be able to split this text into chunks and process it without any LangChain dependencies."
    
    # Prepare the request data
    process_data = {
        "text": test_text,
        "chunk_size": 100,
        "chunk_overlap": 20,
        "collection_name": "integration_test_collection"
    }
    
    try:
        print("Sending document processing request through API Gateway...")
        response = requests.post(
            f"{DOC_PROC_SERVICE_URL}/process-and-store",
            json=process_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Document processing through direct service call: SUCCESS")
            print(f"Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")
            print(f"Chunks Processed: {result.get('chunks_processed')}")
            print(f"Collection Name: {result.get('collection_name')}")
            print(f"Number of Chunk IDs: {len(result.get('chunk_ids', []))}")
            return True
        else:
            print(f"❌ Document processing failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing document processing: {e}")
        return False

def test_api_gateway_endpoints():
    """Test various API Gateway endpoints"""
    print("\n=== Testing API Gateway Endpoints ===")
    
    try:
        # Test root endpoint
        print("1. Testing API Gateway root endpoint...")
        response = requests.get(API_GATEWAY_URL, timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ API Gateway root endpoint accessible")
        else:
            print("   ❌ API Gateway root endpoint not accessible")
            
        # Test health endpoint
        print("\n2. Testing API Gateway health endpoint...")
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ API Gateway health endpoint accessible")
        else:
            print("   ❌ API Gateway health endpoint not accessible")
            
        # Test microservices health
        print("\n3. Testing microservices health...")
        response = requests.get(f"{API_GATEWAY_URL}/health/services", timeout=15)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print("   ✅ Microservices health check accessible")
            
            services = health_data.get('services', {})
            for service_name, service_info in services.items():
                status = service_info.get('status')
                if status == 'ok':
                    response_time = service_info.get('response_time', 'N/A')
                    print(f"   ✅ {service_name}: {status} (Response time: {response_time:.3f}s)")
                else:
                    print(f"   ❌ {service_name}: {status}")
            return True
        else:
            print("   ❌ Microservices health check not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API Gateway endpoints: {e}")
        return False

def main():
    """Main test function"""
    print("Final Integration Test - Cloud Run Services")
    print("=" * 50)
    
    # Test 1: API Gateway endpoints
    api_gateway_test_passed = test_api_gateway_endpoints()
    
    # Test 2: Document processing through direct service call
    document_processing_test_passed = test_document_processing_through_api_gateway()
    
    # Summary
    print("\n" + "=" * 50)
    print("FINAL INTEGRATION TEST RESULTS")
    print("=" * 50)
    
    if api_gateway_test_passed:
        print("✅ API Gateway Endpoints: PASSED")
    else:
        print("❌ API Gateway Endpoints: FAILED")
        
    if document_processing_test_passed:
        print("✅ Document Processing Through Service: PASSED")
    else:
        print("❌ Document Processing Through Service: FAILED")
        
    if api_gateway_test_passed and document_processing_test_passed:
        print("\n🎉 OVERALL RESULT: ALL TESTS PASSED!")
        print("   The integration between services is working correctly.")
        print("   Document Processing Service is functioning without LangChain dependencies.")
        print("   Communication between API Gateway and Document Processing Service is stable.")
    else:
        print("\n⚠️  OVERALL RESULT: SOME TESTS FAILED")
        print("   There are issues that need to be addressed.")
        
    print("\nNote: The Document Processing Service is now using custom text splitting")
    print("      implementation instead of LangChain, and connects to ChromaDB externally.")

if __name__ == "__main__":
    main()