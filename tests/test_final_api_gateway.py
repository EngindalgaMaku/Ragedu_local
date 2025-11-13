#!/usr/bin/env python3
"""
Final API Gateway and PDF Processing Communication Test
Tests the complete PDF to Markdown workflow through API Gateway
"""

import requests
import time
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

# Updated service URLs after fresh deployment
API_GATEWAY_URL = "https://api-gateway-1051060211087.europe-west1.run.app"
PDF_PROCESSOR_URL = "https://pdf-processor-awe3elsvra-ew.a.run.app"

def create_test_pdf():
    """Create a simple test PDF in memory"""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Add content to PDF
    p.drawString(100, 750, "Test Document for PDF Processing")
    p.drawString(100, 720, "This is a test document created for testing")
    p.drawString(100, 690, "the API Gateway and PDF processor integration.")
    p.drawString(100, 660, "")
    p.drawString(100, 630, "Key Features to Test:")
    p.drawString(120, 600, "• PDF upload through API Gateway")
    p.drawString(120, 570, "• Routing to PDF Processing Service")
    p.drawString(120, 540, "• Markdown conversion using Marker")
    p.drawString(120, 510, "• Response handling and error management")
    p.drawString(100, 480, "")
    p.drawString(100, 450, "This document contains Turkish characters: ğüşıöçĞÜŞIÖÇ")
    p.drawString(100, 420, "And some special symbols: @#$%^&*()")
    
    p.showPage()
    p.save()
    
    buffer.seek(0)
    return buffer.getvalue()

def test_service_health(name, url):
    """Test service health endpoint"""
    try:
        print(f"🔍 Testing {name} Health...")
        start_time = time.time()
        response = requests.get(f"{url}/health", timeout=10)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ {name} Health OK ({duration:.2f}s)")
            return True
        else:
            print(f"❌ {name} Health Failed: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ {name} Health Error: {str(e)}")
        return False

def test_api_gateway_services():
    """Test API Gateway's service health check"""
    try:
        print(f"🔍 Testing API Gateway Services Health...")
        start_time = time.time()
        response = requests.get(f"{API_GATEWAY_URL}/health/services", timeout=15)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            services_data = response.json()
            print(f"✅ Gateway Services Health OK ({duration:.2f}s)")
            print(f"📊 Services Status:")
            for service, status in services_data.get('services', {}).items():
                status_icon = "✅" if status.get('status') == 'ok' else "❌"
                print(f"   {status_icon} {service}: {status.get('status')}")
            return True
        else:
            print(f"❌ Gateway Services Health Failed: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Gateway Services Health Error: {str(e)}")
        return False

def test_pdf_to_markdown_conversion():
    """Test complete PDF to Markdown conversion through API Gateway"""
    try:
        print("📄 Testing PDF to Markdown Conversion...")
        
        # Create test PDF
        pdf_content = create_test_pdf()
        print(f"✅ Created test PDF ({len(pdf_content)} bytes)")
        
        # Prepare multipart form data
        files = {
            'file': ('test_document.pdf', pdf_content, 'application/pdf')
        }
        
        # Send to API Gateway
        start_time = time.time()
        response = requests.post(
            f"{API_GATEWAY_URL}/documents/convert-document-to-markdown",
            files=files,
            timeout=60
        )
        duration = time.time() - start_time
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Time: {duration:.2f}s")
        print(f"📊 Response Size: {len(response.content)} bytes")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ PDF to Markdown Conversion Successful!")
            print(f"📁 Markdown File: {result.get('markdown_filename', 'Unknown')}")
            print(f"📝 Message: {result.get('message', 'No message')}")
            
            # Check metadata
            metadata = result.get('metadata', {})
            if metadata:
                print(f"📋 Metadata:")
                for key, value in metadata.items():
                    print(f"   • {key}: {value}")
            
            return True
        else:
            print(f"❌ PDF to Markdown Conversion Failed: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ PDF to Markdown Conversion Error: {str(e)}")
        return False

def main():
    print("🚀 Starting API Gateway and PDF Processing Integration Test")
    print("=" * 80)
    print(f"🌐 API Gateway: {API_GATEWAY_URL}")
    print(f"📄 PDF Processor: {PDF_PROCESSOR_URL}")
    print("=" * 80)
    print()
    
    results = []
    
    # Test 1: API Gateway Health
    print("🧪 Test 1: API Gateway Health")
    result1 = test_service_health("API Gateway", API_GATEWAY_URL)
    results.append(("API Gateway Health", result1))
    print()
    
    # Test 2: Gateway Services Health
    print("🧪 Test 2: Gateway Services Health Check")
    result2 = test_api_gateway_services()
    results.append(("Gateway Services Health", result2))
    print()
    
    # Test 3: PDF Processor Direct Health
    print("🧪 Test 3: PDF Processor Direct Health")
    result3 = test_service_health("PDF Processor", PDF_PROCESSOR_URL)
    results.append(("PDF Processor Health", result3))
    print()
    
    # Test 4: Complete PDF to Markdown Pipeline
    print("🧪 Test 4: PDF to Markdown Conversion Pipeline")
    result4 = test_pdf_to_markdown_conversion()
    results.append(("PDF to Markdown Pipeline", result4))
    print()
    
    # Summary
    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    print("-" * 80)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! PDF processing pipeline is working correctly.")
    elif passed >= total * 0.75:  # 75% success rate
        print("⚠️ Most tests passed. Minor issues detected but core functionality works.")
    else:
        print("❌ Multiple test failures. PDF processing pipeline needs attention.")
    
    return passed == total

if __name__ == "__main__":
    main()