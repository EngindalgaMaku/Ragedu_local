#!/usr/bin/env python3
"""
Simple script to verify microservice connections
"""

import requests
import time

def check_connection(url, service_name):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} is responding")
            return True
        else:
            print(f"❌ {service_name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name} is not accessible: {e}")
        return False

def main():
    """Main function to verify all connections"""
    print("=" * 50)
    print("Microservice Connection Verification")
    print("=" * 50)
    
    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(3)
    
    # Check API Gateway
    print("\n🔍 Checking API Gateway...")
    api_gateway_ok = check_connection("http://localhost:8080/", "API Gateway")
    
    # Check PDF Processing Service
    print("\n🔍 Checking PDF Processing Service...")
    pdf_service_ok = check_connection("http://localhost:8001/health", "PDF Processing Service")
    
    # Check Document Processing Service
    print("\n🔍 Checking Document Processing Service...")
    doc_service_ok = check_connection("http://localhost:8081/health", "Document Processing Service")
    
    # Check Model Inference Service
    print("\n🔍 Checking Model Inference Service...")
    model_service_ok = check_connection("http://localhost:8002/health", "Model Inference Service")
    
    # Check API Gateway connections to microservices
    print("\n🔍 Checking API Gateway to Microservices connections...")
    if api_gateway_ok:
        try:
            response = requests.get("http://localhost:8080/health/services", timeout=10)
            if response.status_code == 200:
                data = response.json()
                services = data.get('services', {})
                print("✅ API Gateway can reach microservices:")
                for name, info in services.items():
                    status = info.get('status')
                    if status == 'ok':
                        print(f"   ✅ {name}")
                    else:
                        print(f"   ❌ {name}: {info.get('error', 'Unknown error')}")
            else:
                print(f"❌ API Gateway microservice check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ API Gateway microservice check failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("CONNECTION SUMMARY")
    print("=" * 50)
    print(f"API Gateway: {'✅ OK' if api_gateway_ok else '❌ FAILED'}")
    print(f"PDF Processing Service: {'✅ OK' if pdf_service_ok else '❌ FAILED'}")
    print(f"Document Processing Service: {'✅ OK' if doc_service_ok else '❌ FAILED'}")
    print(f"Model Inference Service: {'✅ OK' if model_service_ok else '❌ FAILED'}")
    
    if api_gateway_ok and pdf_service_ok:
        print("\n🎉 Core services are connected and functional!")
        print("You can now use the RAG3 system through the API Gateway.")
    else:
        print("\n⚠️  Some services are not connected properly.")
        print("Please check the individual service status above.")

if __name__ == "__main__":
    main()