#!/usr/bin/env python3
"""
Microservice Connection Report
This script generates a detailed report of the microservice connections in the RAG3 system.
"""

import requests
import time
from datetime import datetime

def test_api_gateway():
    """Test API Gateway functionality"""
    print("🔍 Testing API Gateway...")
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ API Gateway root endpoint accessible")
            print(f"      Service: {data.get('service')}")
            print(f"      Version: {data.get('version')}")
            
            # Display microservice URLs
            microservices = data.get('microservices', {})
            print("      Configured Microservices:")
            for name, url in microservices.items():
                print(f"         {name}: {url}")
            return True
        else:
            print(f"   ❌ API Gateway root endpoint failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"   ❌ API Gateway not accessible: {e}")
        return False

def test_pdf_microservice():
    """Test PDF Processing Microservice"""
    print("\n🔍 Testing PDF Processing Microservice...")
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ PDF Processing Microservice health check passed")
            print(f"      Status: {data.get('status')}")
            print(f"      Marker available: {data.get('marker_available')}")
            return True
        else:
            print(f"   ❌ PDF Processing Microservice health check failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"   ❌ PDF Processing Microservice not accessible: {e}")
        return False

def test_api_gateway_to_pdf_connection():
    """Test API Gateway connection to PDF Processing Microservice"""
    print("\n🔍 Testing API Gateway to PDF Processing connection...")
    try:
        # Test health/services endpoint which checks all microservice connections
        response = requests.get("http://localhost:8080/health/services", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ API Gateway microservice connection test passed")
            
            services = data.get('services', {})
            pdf_service = services.get('pdf_processor', {})
            
            if pdf_service.get('status') == 'ok':
                print("   ✅ API Gateway to PDF Processing connection: SUCCESSFUL")
                print(f"      Response time: {pdf_service.get('response_time', 'N/A')}s")
                return True
            else:
                print("   ❌ API Gateway to PDF Processing connection: FAILED")
                print(f"      Error: {pdf_service.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   ❌ API Gateway microservice connection test failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"   ❌ API Gateway microservice connection test failed: {e}")
        return False

def test_model_microservice():
    """Test Model Inference Microservice (if running)"""
    print("\n🔍 Testing Model Inference Microservice...")
    try:
        # Try to connect to the default port
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Model Inference Microservice accessible")
            print(f"      Status: {data.get('status')}")
            print(f"      Ollama available: {data.get('ollama_available')}")
            print(f"      Groq available: {data.get('groq_available')}")
            return True
        else:
            print(f"   ⚠️  Model Inference Microservice returned status {response.status_code}")
            return False
    except Exception as e:
        print("   ⚠️  Model Inference Microservice not accessible (may not be running)")
        print(f"      Error: {e}")
        return False

def test_document_microservice():
    """Test Document Processing Microservice (if running)"""
    print("\n🔍 Testing Document Processing Microservice...")
    try:
        # Try to connect to a common port
        response = requests.get("http://localhost:8081/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Document Processing Microservice accessible")
            print(f"      Status: {data.get('status')}")
            return True
        else:
            print(f"   ⚠️  Document Processing Microservice returned status {response.status_code}")
            return False
    except Exception as e:
        print("   ⚠️  Document Processing Microservice not accessible (may not be running)")
        print(f"      Error: {e}")
        return False

def generate_report():
    """Generate a comprehensive connection report"""
    print("=" * 70)
    print("RAG3 Microservice Connection Report")
    print("=" * 70)
    print(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test all services
    api_gateway_ok = test_api_gateway()
    pdf_service_ok = test_pdf_microservice()
    gateway_to_pdf_ok = test_api_gateway_to_pdf_connection()
    model_service_ok = test_model_microservice()
    document_service_ok = test_document_microservice()
    
    # Summary
    print("\n" + "=" * 70)
    print("CONNECTION SUMMARY")
    print("=" * 70)
    
    print(f"API Gateway Service: {'✅ RUNNING' if api_gateway_ok else '❌ NOT RUNNING'}")
    print(f"PDF Processing Service: {'✅ RUNNING' if pdf_service_ok else '❌ NOT RUNNING'}")
    print(f"API Gateway to PDF Processing: {'✅ CONNECTED' if gateway_to_pdf_ok else '❌ DISCONNECTED'}")
    print(f"Model Inference Service: {'✅ RUNNING' if model_service_ok else '⚠️  NOT RUNNING'}")
    print(f"Document Processing Service: {'✅ RUNNING' if document_service_ok else '⚠️  NOT RUNNING'}")
    
    # Overall status
    core_services_running = api_gateway_ok and pdf_service_ok and gateway_to_pdf_ok
    all_services_running = core_services_running and model_service_ok and document_service_ok
    
    print("\n" + "=" * 70)
    if core_services_running:
        print("🎉 CORE MICROSERVICES ARE CONNECTED AND FUNCTIONAL")
        if all_services_running:
            print("🌟 ALL MICROSERVICES ARE CONNECTED AND FUNCTIONAL")
        else:
            print("⚠️  Some non-critical services are not running")
    else:
        print("❌ CORE MICROSERVICES HAVE CONNECTION ISSUES")
    print("=" * 70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if not api_gateway_ok:
        print("   - Start the API Gateway service (src/api/main_gateway.py)")
    if not pdf_service_ok:
        print("   - Start the PDF Processing service (services/pdf_processing_service/main.py)")
    if not gateway_to_pdf_ok:
        print("   - Check network connectivity between API Gateway and PDF Processing service")
    if not model_service_ok:
        print("   - Start the Model Inference service (services/model_inference_service/main.py)")
        print("   - Ensure Ollama or Groq API key is properly configured")
    if not document_service_ok:
        print("   - Install ChromaDB dependency: pip install chromadb")
        print("   - Start the Document Processing service (services/document_processing_service/main.py)")
    
    print("\nFor detailed logs, check the terminal output of each service.")

if __name__ == "__main__":
    generate_report()