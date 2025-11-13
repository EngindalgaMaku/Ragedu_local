#!/usr/bin/env python3
"""
Test script for PDF Processing Service integration.
Tests the main API's PDF processing endpoint to ensure it properly delegates to the microservice.
"""

import requests
import json
import os
from pathlib import Path

def test_pdf_microservice_integration():
    """Test the /documents/convert-document-to-markdown endpoint"""
    
    # Main API endpoint - using port 8090 for new API instance
    main_api_url = "http://localhost:8090"
    endpoint = f"{main_api_url}/documents/convert-document-to-markdown"
    
    print("🔧 Testing PDF Microservice Integration...")
    print(f"📍 Main API URL: {main_api_url}")
    print(f"📍 Endpoint: {endpoint}")
    
    # Check if we have a test PDF file
    test_pdf_path = Path("test_documents/test_document.pdf")
    if not test_pdf_path.exists():
        print("⚠️  No test PDF found. Creating a simple test scenario...")
        print("ℹ️  You can create a test PDF file at: test_documents/test_document.pdf")
        return False
    
    try:
        # Read the test PDF
        with open(test_pdf_path, 'rb') as f:
            files = {'file': ('test_document.pdf', f, 'application/pdf')}
            
            print("📤 Sending PDF to main API...")
            response = requests.post(endpoint, files=files, timeout=60)
            
            print(f"📨 Response Status: {response.status_code}")
            print(f"📨 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS! PDF processing completed via microservice")
                print(f"📄 Markdown filename: {result.get('markdown_filename')}")
                print(f"📊 Metadata: {result.get('metadata')}")
                return True
            else:
                print(f"❌ FAILED! Status: {response.status_code}")
                print(f"📄 Response: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError as e:
        print(f"❌ CONNECTION ERROR: Cannot connect to main API at {main_api_url}")
        print("ℹ️  Make sure the main API is running on port 8080")
        print(f"📄 Error: {e}")
        return False
        
    except requests.exceptions.Timeout as e:
        print(f"❌ TIMEOUT ERROR: Request timed out")
        print(f"📄 Error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False

def test_pdf_service_directly():
    """Test the PDF Processing Service directly"""
    
    pdf_service_url = "http://localhost:8001"  # Default local development URL
    endpoint = f"{pdf_service_url}/process"
    
    print("\n🔧 Testing PDF Processing Service directly...")
    print(f"📍 PDF Service URL: {pdf_service_url}")
    
    try:
        # Test health endpoint first
        health_response = requests.get(f"{pdf_service_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ PDF Service Health: {health_data}")
        else:
            print(f"⚠️  PDF Service health check failed: {health_response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ PDF Processing Service is not available")
        print("ℹ️  Make sure the PDF Processing Service is running on port 8001")
        print("ℹ️  You can start it with: docker-compose up pdf-processor")
        return False
        
    return True

def check_configuration():
    """Check configuration settings"""
    print("\n🔧 Checking Configuration...")
    
    try:
        # Import configuration
        from src.config import get_pdf_processor_url
        
        pdf_url = get_pdf_processor_url()
        print(f"📍 Configured PDF Processor URL: {pdf_url}")
        
        # Test environment variable
        env_url = os.getenv('PDF_PROCESSOR_URL', 'Not set')
        print(f"🌍 Environment PDF_PROCESSOR_URL: {env_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 PDF MICROSERVICE INTEGRATION TEST")
    print("=" * 60)
    
    # Check configuration
    config_ok = check_configuration()
    
    # Test PDF service directly
    service_ok = test_pdf_service_directly()
    
    # Test main API integration
    if config_ok and service_ok:
        integration_ok = test_pdf_microservice_integration()
    else:
        print("\n⚠️  Skipping integration test due to configuration or service issues")
        integration_ok = False
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Configuration: {'✅' if config_ok else '❌'}")
    print(f"PDF Service:   {'✅' if service_ok else '❌'}")
    print(f"Integration:   {'✅' if integration_ok else '❌'}")
    
    if config_ok and service_ok and integration_ok:
        print("\n🎉 ALL TESTS PASSED! PDF microservice integration is working!")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        
    return config_ok and service_ok and integration_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)