#!/usr/bin/env python3
"""
Test script to check communication between API Gateway and Document Processing Service
"""

import requests
import json
import time

# API Gateway URL (assuming it's running locally on port 8080)
API_GATEWAY_URL = "http://localhost:8080"

# Document Processing Service URL (the one we just deployed)
DOC_PROC_SERVICE_URL = "https://doc-proc-service-1051060211087.europe-west1.run.app"

def test_direct_service_health():
    """Test the document processing service directly"""
    print("=== Testing Document Processing Service Directly ===")
    try:
        response = requests.get(f"{DOC_PROC_SERVICE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ Document Processing Service is healthy")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Document Processing Service health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Document Processing Service: {e}")
        return False

def test_api_gateway_health():
    """Test API Gateway health endpoints"""
    print("\n=== Testing API Gateway Health ===")
    try:
        # Test basic health
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=5)
        print(f"API Gateway Basic Health - Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ API Gateway is running")
        else:
            print("❌ API Gateway basic health check failed")
            
        # Test microservices health
        response = requests.get(f"{API_GATEWAY_URL}/health/services", timeout=10)
        print(f"Microservices Health - Status Code: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Gateway microservices health check successful")
            print(f"Gateway Status: {health_data.get('gateway')}")
            
            services = health_data.get('services', {})
            for service_name, service_info in services.items():
                status = service_info.get('status')
                url = service_info.get('url')
                if status == 'ok':
                    print(f"  ✅ {service_name}: {status} ({url})")
                else:
                    print(f"  ❌ {service_name}: {status} ({url})")
                    error = service_info.get('error')
                    if error:
                        print(f"    Error: {error}")
            return True
        else:
            print("❌ API Gateway microservices health check failed")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API Gateway: {e}")
        return False

def test_document_processing_endpoint():
    """Test the document processing endpoint through API Gateway"""
    print("\n=== Testing Document Processing Endpoint ===")
    
    # Sample text to process
    sample_text = "This is a sample document for testing the document processing service. It contains multiple sentences to demonstrate text chunking and processing capabilities."
    
    # Create a simple process request
    process_request = {
        "text": sample_text,
        "metadata": {"source": "test", "document_id": "test-001"},
        "collection_name": "test_collection",
        "chunk_size": 100,
        "chunk_overlap": 20
    }
    
    try:
        # Test direct call to document processing service
        print("Testing direct call to Document Processing Service...")
        response = requests.post(
            f"{DOC_PROC_SERVICE_URL}/process-and-store",
            json=process_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Direct Call Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ Direct call to Document Processing Service successful")
            response_data = response.json()
            print(f"Response: {json.dumps(response_data, indent=2)}")
        else:
            print(f"❌ Direct call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed to test document processing endpoint: {e}")

def main():
    """Main test function"""
    print("Testing Communication Between API Gateway and Document Processing Service")
    print("=" * 80)
    
    # Test direct service health
    service_healthy = test_direct_service_health()
    
    # Test API gateway health
    gateway_healthy = test_api_gateway_health()
    
    # Test document processing endpoint
    test_document_processing_endpoint()
    
    print("\n" + "=" * 80)
    if service_healthy:
        print("✅ Document Processing Service is accessible")
    else:
        print("❌ Document Processing Service is not accessible")
        
    if gateway_healthy:
        print("✅ API Gateway is running and can check microservice health")
    else:
        print("❌ API Gateway health checks failed")
        
    print("\nTest completed!")

if __name__ == "__main__":
    main()