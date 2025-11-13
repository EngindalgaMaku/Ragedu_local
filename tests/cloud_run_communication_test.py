#!/usr/bin/env python3
"""
Test communication between services deployed on Cloud Run
"""

import requests
import json
import time

# Cloud Run service URLs
API_GATEWAY_URL = "https://api-gateway-1051060211087.europe-west1.run.app"
DOC_PROC_SERVICE_URL = "https://doc-proc-service-1051060211087.europe-west1.run.app"

def test_direct_service_communication():
    """Test direct communication with Document Processing Service"""
    print("=== Testing Direct Document Processing Service Communication ===")
    
    # Test health endpoint
    try:
        print("1. Testing Document Processing Service health...")
        response = requests.get(f"{DOC_PROC_SERVICE_URL}/health", timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print("   ✅ Document Processing Service is accessible")
            print(f"   Status: {health_data.get('status')}")
            # Updated check for the new health response format
            text_processing_available = health_data.get('text_processing_available')
            if text_processing_available is not None:
                print(f"   Text Processing Available: {text_processing_available}")
            else:
                print(f"   LangChain Available: {health_data.get('langchain_available')}")
            print(f"   Model Service Connected: {health_data.get('model_service_connected')}")
        else:
            print(f"   ❌ Document Processing Service health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to connect to Document Processing Service: {e}")
        return False
    
    # Test root endpoint
    try:
        print("\n2. Testing Document Processing Service root endpoint...")
        response = requests.get(DOC_PROC_SERVICE_URL, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            root_data = response.json()
            print("   ✅ Document Processing Service root endpoint accessible")
            print(f"   Message: {root_data.get('message')}")
        else:
            print(f"   ❌ Document Processing Service root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing root endpoint: {e}")
    
    return True

def test_api_gateway_communication():
    """Test API Gateway communication with microservices"""
    print("\n=== Testing API Gateway Communication ===")
    
    # Test API Gateway health
    try:
        print("1. Testing API Gateway health...")
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print("   ✅ API Gateway is accessible")
            print(f"   Service: {health_data.get('service')}")
            print(f"   Status: {health_data.get('status')}")
        else:
            print(f"   ❌ API Gateway health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to connect to API Gateway: {e}")
        return False
    
    # Test microservices health through API Gateway
    try:
        print("\n2. Testing microservices health through API Gateway...")
        response = requests.get(f"{API_GATEWAY_URL}/health/services", timeout=15)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print("   ✅ API Gateway microservices health check successful")
            print(f"   Gateway Status: {health_data.get('gateway')}")
            
            services = health_data.get('services', {})
            for service_name, service_info in services.items():
                status = service_info.get('status')
                url = service_info.get('url')
                response_time = service_info.get('response_time', 'N/A')
                
                if status == 'ok':
                    print(f"   ✅ {service_name}: {status} (Response time: {response_time:.3f}s)")
                else:
                    print(f"   ❌ {service_name}: {status}")
                    error = service_info.get('error')
                    if error:
                        print(f"      Error: {error}")
            
            # Check specifically for document processor
            doc_proc_status = services.get('document_processor', {}).get('status')
            if doc_proc_status == 'ok':
                print("\n   🎉 Document Processing Service communication through API Gateway: SUCCESSFUL")
                return True
            else:
                print("\n   ❌ Document Processing Service communication through API Gateway: FAILED")
                return False
        else:
            print(f"   ❌ API Gateway microservices health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error testing microservices health: {e}")
        return False

def test_end_to_end_communication():
    """Test end-to-end communication flow"""
    print("\n=== Testing End-to-End Communication Flow ===")
    
    try:
        # Test if we can get the API documentation
        print("1. Testing API Gateway documentation...")
        response = requests.get(f"{API_GATEWAY_URL}/docs", timeout=10)
        print(f"   Documentation Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ API Gateway documentation accessible")
        else:
            print("   ℹ️  API Gateway documentation not accessible (this is OK)")
            
        # Test the root endpoint to see what's available
        print("\n2. Testing API Gateway root endpoint...")
        response = requests.get(API_GATEWAY_URL, timeout=10)
        print(f"   Root Endpoint Status Code: {response.status_code}")
        
        if response.status_code == 200:
            root_data = response.json()
            print("   ✅ API Gateway root endpoint accessible")
            print("   Available microservices:")
            microservices = root_data.get('microservices', {})
            for name, url in microservices.items():
                print(f"     - {name}: {url}")
        else:
            print(f"   ❌ API Gateway root endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error in end-to-end communication test: {e}")

def main():
    """Main test function"""
    print("Cloud Run Services Communication Test")
    print("=" * 50)
    print(f"API Gateway URL: {API_GATEWAY_URL}")
    print(f"Document Processing Service URL: {DOC_PROC_SERVICE_URL}")
    print("=" * 50)
    
    # Test 1: Direct service communication
    print("\n📋 TEST 1: DIRECT SERVICE COMMUNICATION")
    direct_test_passed = test_direct_service_communication()
    
    # Test 2: API Gateway communication
    print("\n📋 TEST 2: API GATEWAY COMMUNICATION")
    gateway_test_passed = test_api_gateway_communication()
    
    # Test 3: End-to-end communication
    print("\n📋 TEST 3: END-TO-END COMMUNICATION")
    test_end_to_end_communication()
    
    # Summary
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    
    if direct_test_passed:
        print("✅ Direct Document Processing Service Communication: PASSED")
    else:
        print("❌ Direct Document Processing Service Communication: FAILED")
        
    if gateway_test_passed:
        print("✅ API Gateway to Document Processing Service Communication: PASSED")
    else:
        print("❌ API Gateway to Document Processing Service Communication: FAILED")
        
    if direct_test_passed and gateway_test_passed:
        print("\n🎉 OVERALL RESULT: COMMUNICATION BETWEEN SERVICES IS WORKING!")
        print("   The API Gateway can successfully communicate with the Document Processing Service.")
    else:
        print("\n⚠️  OVERALL RESULT: SOME COMMUNICATION ISSUES DETECTED")
        print("   There are issues with service communication that need to be addressed.")
        
    print("\nNote: Even though communication is working, the Document Processing Service")
    print("      reports that LangChain is not available, which may affect document processing.")

if __name__ == "__main__":
    main()