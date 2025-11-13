#!/usr/bin/env python3
"""
Simple test script to verify connections between API Gateway and running microservices
"""

import requests
import time

def test_api_gateway_root():
    """Test API Gateway root endpoint"""
    print("Testing API Gateway root endpoint...")
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Gateway root endpoint: PASSED")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print(f"❌ API Gateway root endpoint: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ API Gateway root endpoint: FAILED (Error: {e})")
        return False

def test_microservices_info():
    """Test API Gateway microservices info"""
    print("\nTesting API Gateway microservices info...")
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            microservices = data.get('microservices', {})
            print("✅ API Gateway microservices info: PASSED")
            for name, url in microservices.items():
                print(f"   {name}: {url}")
            return True
        else:
            print(f"❌ API Gateway microservices info: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ API Gateway microservices info: FAILED (Error: {e})")
        return False

def test_pdf_processor_health():
    """Test PDF Processor health endpoint"""
    print("\nTesting PDF Processor health endpoint...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ PDF Processor health endpoint: PASSED")
            print(f"   Status: {data.get('status')}")
            print(f"   Marker available: {data.get('marker_available')}")
            return True
        else:
            print(f"❌ PDF Processor health endpoint: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ PDF Processor health endpoint: FAILED (Error: {e})")
        return False

def test_api_gateway_microservice_health():
    """Test API Gateway microservice health check"""
    print("\nTesting API Gateway microservice health check...")
    try:
        response = requests.get("http://localhost:8080/health/services", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Gateway microservice health check: PASSED")
            print(f"   Gateway status: {data.get('gateway')}")
            
            services = data.get('services', {})
            for name, info in services.items():
                status = info.get('status')
                if status == 'ok':
                    print(f"   ✅ {name}: {status} (Response time: {info.get('response_time', 'N/A')}s)")
                else:
                    print(f"   ❌ {name}: {status} - {info.get('error', 'Unknown error')}")
            return True
        else:
            print(f"❌ API Gateway microservice health check: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ API Gateway microservice health check: FAILED (Error: {e})")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Simple Microservice Connection Test")
    print("=" * 60)
    
    # Wait a moment for services to fully start
    print("Waiting for services to start...")
    time.sleep(2)
    
    # Test API Gateway
    gateway_root_ok = test_api_gateway_root()
    microservices_info_ok = test_microservices_info()
    
    # Test PDF Processor
    pdf_processor_ok = test_pdf_processor_health()
    
    # Test API Gateway connections
    gateway_connections_ok = test_api_gateway_microservice_health()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    print(f"API Gateway Root Endpoint: {'✅ OK' if gateway_root_ok else '❌ FAILED'}")
    print(f"API Gateway Microservices Info: {'✅ OK' if microservices_info_ok else '❌ FAILED'}")
    print(f"PDF Processor Health: {'✅ OK' if pdf_processor_ok else '❌ FAILED'}")
    print(f"API Gateway Microservice Connections: {'✅ OK' if gateway_connections_ok else '❌ FAILED'}")
    
    overall_success = (
        gateway_root_ok and 
        microservices_info_ok and 
        pdf_processor_ok and 
        gateway_connections_ok
    )
    
    if overall_success:
        print("\n🎉 All tests passed! Microservice connections are working properly.")
    else:
        print("\n⚠️  Some tests failed. Please check the individual test results above.")

if __name__ == "__main__":
    main()