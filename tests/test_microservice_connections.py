#!/usr/bin/env python3
"""
Test script to verify connections between API Gateway and microservices
"""

import requests
import os
import time
from pathlib import Path

# Microservice URLs from environment variables (same as in main_gateway.py)
PDF_PROCESSOR_URL = os.getenv('PDF_PROCESSOR_URL', 'https://pdf-processor-1051060211087.europe-west1.run.app')
DOCUMENT_PROCESSOR_URL = os.getenv('DOCUMENT_PROCESSOR_URL', 'https://doc-proc-service-1051060211087.europe-west1.run.app')
MODEL_INFERENCE_URL = os.getenv('MODEL_INFERENCE_URL', 'https://model-inferencer-1051060211087.europe-west1.run.app')

# Test PDF file path
TEST_PDF_PATH = "test_documents/test_document.pdf"

def test_api_gateway_health():
    """Test API Gateway health endpoint"""
    print("Testing API Gateway health...")
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Gateway health check: PASSED")
            return True
        else:
            print(f"❌ API Gateway health check: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ API Gateway health check: FAILED (Error: {e})")
        return False

def test_microservices_health():
    """Test health of all microservices"""
    print("\nTesting microservices health...")
    services = {
        "PDF Processor": PDF_PROCESSOR_URL,
        "Document Processor": DOCUMENT_PROCESSOR_URL,
        "Model Inference": MODEL_INFERENCE_URL
    }
    
    results = {}
    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} health check: PASSED")
                results[name] = True
            else:
                print(f"❌ {name} health check: FAILED (Status: {response.status_code})")
                results[name] = False
        except Exception as e:
            print(f"❌ {name} health check: FAILED (Error: {e})")
            results[name] = False
    
    return results

def test_api_gateway_microservice_connections():
    """Test API Gateway connections to microservices"""
    print("\nTesting API Gateway connections to microservices...")
    try:
        response = requests.get("http://localhost:8080/health/services", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Gateway microservice connection test: PASSED")
            
            # Print detailed results
            services = data.get("services", {})
            for name, service_info in services.items():
                status = service_info.get("status", "unknown")
                if status == "ok":
                    print(f"  ✅ Connection to {name}: OK")
                else:
                    print(f"  ❌ Connection to {name}: FAILED - {service_info.get('error', 'Unknown error')}")
            return True
        else:
            print(f"❌ API Gateway microservice connection test: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ API Gateway microservice connection test: FAILED (Error: {e})")
        return False

def test_pdf_processing_endpoint():
    """Test PDF processing endpoint through API Gateway"""
    print("\nTesting PDF processing endpoint...")
    
    # Check if test PDF exists
    if not Path(TEST_PDF_PATH).exists():
        print(f"⚠️  Test PDF not found at {TEST_PDF_PATH}. Skipping PDF processing test.")
        return False
    
    try:
        with open(TEST_PDF_PATH, 'rb') as pdf_file:
            files = {'file': ('test_document.pdf', pdf_file, 'application/pdf')}
            response = requests.post(
                "http://localhost:8080/documents/convert-document-to-markdown",
                files=files,
                timeout=60
            )
            
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ PDF processing endpoint test: PASSED")
                return True
            else:
                print(f"❌ PDF processing endpoint test: FAILED - {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ PDF processing endpoint test: FAILED (Status: {response.status_code})")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ PDF processing endpoint test: FAILED (Error: {e})")
        return False

def test_model_endpoint():
    """Test model endpoint through API Gateway"""
    print("\nTesting model endpoint...")
    try:
        response = requests.get("http://localhost:8080/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            if models:
                print(f"✅ Model endpoint test: PASSED ({len(models)} models available)")
                return True
            else:
                print("⚠️  Model endpoint test: PARTIAL (No models returned)")
                return True
        else:
            print(f"❌ Model endpoint test: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Model endpoint test: FAILED (Error: {e})")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Microservice Connection Test")
    print("=" * 60)
    
    # Test API Gateway
    gateway_ok = test_api_gateway_health()
    
    if not gateway_ok:
        print("\n❌ API Gateway is not running. Please start the API Gateway service.")
        return
    
    # Test microservices
    microservices_results = test_microservices_health()
    
    # Test API Gateway connections
    connections_ok = test_api_gateway_microservice_connections()
    
    # Test specific endpoints
    pdf_test_ok = test_pdf_processing_endpoint()
    model_test_ok = test_model_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    print(f"API Gateway: {'✅ OK' if gateway_ok else '❌ FAILED'}")
    for name, result in microservices_results.items():
        print(f"{name}: {'✅ OK' if result else '❌ FAILED'}")
    print(f"Gateway-Microservice Connections: {'✅ OK' if connections_ok else '❌ FAILED'}")
    print(f"PDF Processing Endpoint: {'✅ OK' if pdf_test_ok else '❌ FAILED'}")
    print(f"Model Endpoint: {'✅ OK' if model_test_ok else '❌ FAILED'}")
    
    overall_success = (
        gateway_ok and 
        all(microservices_results.values()) and 
        connections_ok and 
        pdf_test_ok and 
        model_test_ok
    )
    
    if overall_success:
        print("\n🎉 All tests passed! Microservice connections are working properly.")
    else:
        print("\n⚠️  Some tests failed. Please check the individual test results above.")

if __name__ == "__main__":
    main()