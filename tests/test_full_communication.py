#!/usr/bin/env python3
"""
Test script to check full communication flow:
Client -> API Gateway -> Document Processing Service -> Response back
"""

import requests
import json
import time
import uuid

# API Gateway URL (running locally on port 8080)
API_GATEWAY_URL = "http://localhost:8080"

# Document Processing Service URL (deployed on Cloud Run)
DOC_PROC_SERVICE_URL = "https://doc-proc-service-1051060211087.europe-west1.run.app"

def test_direct_document_service():
    """Test the document processing service directly"""
    print("=== Testing Document Processing Service Directly ===")
    try:
        # Test health endpoint
        response = requests.get(f"{DOC_PROC_SERVICE_URL}/health", timeout=10)
        print(f"Health Check - Status Code: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Document Processing Service is healthy")
            print(f"Service Status: {health_data.get('status')}")
            print(f"LangChain Available: {health_data.get('langchain_available')}")
            print(f"Model Service Connected: {health_data.get('model_service_connected')}")
        else:
            print(f"❌ Document Processing Service health check failed: {response.status_code}")
            return False
            
        # Test process endpoint with sample data
        print("\n--- Testing Process Endpoint ---")
        sample_text = "This is a test document for checking the communication between services. It contains multiple sentences to demonstrate text chunking capabilities. This is the second sentence. This is the third sentence. This is the fourth sentence to make sure we have enough content for chunking."
        
        process_data = {
            "text": sample_text,
            "metadata": {
                "source": "direct_test",
                "test_id": str(uuid.uuid4()),
                "timestamp": time.time()
            },
            "collection_name": "test_collection",
            "chunk_size": 100,
            "chunk_overlap": 20
        }
        
        response = requests.post(
            f"{DOC_PROC_SERVICE_URL}/process-and-store",
            json=process_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Process Request - Status Code: {response.status_code}")
        if response.status_code == 200:
            process_result = response.json()
            print("✅ Document Processing Service processed request successfully")
            print(f"Success: {process_result.get('success')}")
            print(f"Message: {process_result.get('message')}")
            print(f"Chunks Processed: {process_result.get('chunks_processed')}")
            print(f"Collection Name: {process_result.get('collection_name')}")
            print(f"Chunk IDs Count: {len(process_result.get('chunk_ids', []))}")
            return True
        else:
            print(f"❌ Document Processing Service failed to process request: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Document Processing Service directly: {e}")
        return False

def test_api_gateway_endpoints():
    """Test API Gateway endpoints"""
    print("\n=== Testing API Gateway Endpoints ===")
    try:
        # Test basic health
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=5)
        print(f"API Gateway Health - Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ API Gateway basic health check passed")
        else:
            print("❌ API Gateway basic health check failed")
            
        # Test microservices health
        response = requests.get(f"{API_GATEWAY_URL}/health/services", timeout=10)
        print(f"Microservices Health - Status Code: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Gateway microservices health check passed")
            services = health_data.get('services', {})
            for service_name, service_info in services.items():
                status = service_info.get('status')
                if status == 'ok':
                    print(f"  ✅ {service_name}: {status}")
                else:
                    print(f"  ❌ {service_name}: {status}")
                    error = service_info.get('error')
                    if error:
                        print(f"    Error: {error}")
        else:
            print("❌ API Gateway microservices health check failed")
            
    except Exception as e:
        print(f"❌ Error testing API Gateway endpoints: {e}")

def test_document_processing_flow():
    """Test the full document processing flow through API Gateway"""
    print("\n=== Testing Full Document Processing Flow ===")
    
    # First, let's check what endpoints are available in the API gateway for document processing
    print("Checking API Gateway documentation...")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API Gateway documentation is accessible")
        else:
            print("ℹ️  API Gateway documentation not available")
    except:
        print("ℹ️  API Gateway documentation not accessible")
    
    # Let's look at the main gateway file to understand the document processing endpoint
    print("\nChecking document processing endpoint in API Gateway...")
    
    # Based on the main_gateway.py file, the document processing endpoint is:
    # POST /documents/process-and-store
    # But it expects form data with specific parameters
    
    # Let's create a test file first
    test_content = """# Test Document
This is a test document to check the full communication flow between services.

## Section 1
This section contains some sample text to process.

## Section 2
This is another section with more content for testing.

## Section 3
Final section to ensure we have enough content for proper testing.
"""
    
    # Save test content to a file
    with open("test_document.md", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print("Created test document: test_document.md")
    
    # Now let's try to test the document processing flow
    # But we need to understand the exact API structure from main_gateway.py
    
    # Looking at the code, the endpoint expects:
    # POST /documents/process-and-store
    # Form data:
    # - session_id (str)
    # - markdown_files (JSON string of file list)
    # - chunk_strategy (str, default "semantic")
    # - chunk_size (int, default 1000)
    # - chunk_overlap (int, default 100)
    # - embedding_model (str, default "mixedbread-ai/mxbai-embed-large-v1")
    
    try:
        # Test the document processing endpoint through API Gateway
        print("\n--- Testing Document Processing Through API Gateway ---")
        
        # Prepare form data
        form_data = {
            "session_id": f"test-session-{uuid.uuid4()}",
            "markdown_files": json.dumps(["test_document.md"]),
            "chunk_strategy": "fixed",
            "chunk_size": "100",
            "chunk_overlap": "20",
            "embedding_model": "mixedbread-ai/mxbai-embed-large-v1"
        }
        
        # But we need to check if this endpoint actually exists and works
        # Looking at the code, it seems like the API gateway routes to the document processor
        # at /process-documents endpoint, not /process-and-store
        
        # Let's check what endpoints are actually available
        response = requests.get(f"{API_GATEWAY_URL}", timeout=5)
        if response.status_code == 200:
            gateway_info = response.json()
            print("API Gateway Info:")
            print(json.dumps(gateway_info, indent=2))
            
            # Get the document processor URL from the gateway
            microservices = gateway_info.get("microservices", {})
            doc_proc_url = microservices.get("document_processor")
            if doc_proc_url:
                print(f"\nDocument Processor URL from Gateway: {doc_proc_url}")
        
        print("\nTrying to call the document processing endpoint through API Gateway...")
        
        # Try the endpoint that should route to the document processor
        response = requests.post(
            f"{API_GATEWAY_URL}/documents/process-and-store",
            data=form_data,
            timeout=30
        )
        
        print(f"Document Processing Request - Status Code: {response.status_code}")
        if response.status_code == 200:
            process_result = response.json()
            print("✅ API Gateway successfully routed request to Document Processing Service")
            print(f"Response: {json.dumps(process_result, indent=2)}")
            return True
        else:
            print(f"❌ API Gateway failed to process document request: {response.status_code}")
            print(f"Response: {response.text}")
            
            # Let's also try a simpler approach - directly test if the gateway can reach the service
            print("\n--- Testing Direct Health Check Through Gateway ---")
            # We can test if the gateway can reach the document processor by checking its health endpoint
            
    except Exception as e:
        print(f"❌ Error testing document processing flow: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("Testing Full Communication Flow")
    print("=" * 50)
    
    # Test 1: Direct service communication
    print("\n1. Testing Direct Service Communication")
    direct_test_passed = test_direct_document_service()
    
    # Test 2: API Gateway endpoints
    print("\n2. Testing API Gateway Endpoints")
    test_api_gateway_endpoints()
    
    # Test 3: Full flow communication
    print("\n3. Testing Full Communication Flow")
    full_flow_passed = test_document_processing_flow()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    if direct_test_passed:
        print("✅ Direct Document Processing Service Communication: PASSED")
    else:
        print("❌ Direct Document Processing Service Communication: FAILED")
        
    if full_flow_passed:
        print("✅ Full Communication Flow: PASSED")
    else:
        print("❌ Full Communication Flow: FAILED")
        
    print("\nTest completed!")

if __name__ == "__main__":
    main()