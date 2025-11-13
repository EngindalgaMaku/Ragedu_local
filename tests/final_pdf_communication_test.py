#!/usr/bin/env python3
"""
Final PDF Communication Test with Correct URLs
Tests PDF processing through API Gateway using actual deployed URLs
"""
import requests
import json
import time
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# CORRECT Service URLs from gcloud run services list
API_GATEWAY_URL = "https://api-gateway-awe3elsvra-ew.a.run.app"
PDF_PROCESSOR_URL = "https://pdf-processor-awe3elsvra-ew.a.run.app"

def create_test_pdf():
    """Create a test PDF file for testing"""
    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 100, "API Gateway PDF Processing Test")
        
        # Add some content
        c.setFont("Helvetica", 12)
        content = [
            "Bu bir PDF'den Markdown'a dönüşüm testidir.",
            "API Gateway üzerinden PDF Processor'a gönderilecek.",
            "",
            "Test içeriği:",
            "• Türkçe karakter desteği: çöğüşıüö",
            "• Liste formatları",
            "• Başlık ve paragraf yapıları",
            "",
            "Bu test başarılı olursa PDF processing mimarisi çalışıyor demektir.",
        ]
        
        y_position = height - 150
        for line in content:
            c.drawString(100, y_position, line)
            y_position -= 20
            
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print(f"❌ Error creating test PDF: {e}")
        return None

def test_api_gateway_health():
    """Test API Gateway health endpoint"""
    try:
        print("🔍 Testing API Gateway Health...")
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Gateway Health: {data}")
            return True
        else:
            print(f"❌ API Gateway Health Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ API Gateway Health Error: {e}")
        return False

def test_api_gateway_services():
    """Test API Gateway services health check"""
    try:
        print("🔍 Testing API Gateway Services Health...")
        response = requests.get(f"{API_GATEWAY_URL}/health/services", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Gateway Services Health:")
            for service, info in data.get('services', {}).items():
                status = info.get('status', 'unknown')
                url = info.get('url', 'N/A')
                print(f"   🔗 {service}: {status} ({url})")
            return True
        else:
            print(f"❌ Gateway Services Health Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Gateway Services Health Error: {e}")
        return False

def test_pdf_to_markdown_conversion():
    """Test PDF to Markdown conversion through API Gateway"""
    try:
        print("📄 Testing PDF to Markdown Conversion...")
        
        # Create test PDF
        pdf_content = create_test_pdf()
        if not pdf_content:
            return False
            
        print(f"✅ Created test PDF ({len(pdf_content)} bytes)")
        
        # Send PDF to API Gateway
        files = {
            'file': ('test_document.pdf', pdf_content, 'application/pdf')
        }
        
        response = requests.post(
            f"{API_GATEWAY_URL}/documents/convert-document-to-markdown",
            files=files,
            timeout=60
        )
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ PDF to Markdown Conversion Successful!")
            print(f"   📄 Markdown file: {data.get('markdown_filename')}")
            print(f"   📊 Success: {data.get('success')}")
            print(f"   💬 Message: {data.get('message')}")
            print(f"   🔍 Metadata: {data.get('metadata', {})}")
            
            # Test retrieving the created markdown file
            if data.get('markdown_filename'):
                return test_markdown_retrieval(data['markdown_filename'])
            return True
        else:
            print(f"❌ PDF to Markdown Conversion Failed: {response.status_code}")
            print(f"   Response: {response.text[:500]}...")
            return False
            
    except Exception as e:
        print(f"❌ PDF to Markdown Conversion Error: {e}")
        return False

def test_markdown_retrieval(filename):
    """Test retrieving markdown file content"""
    try:
        print(f"📖 Testing Markdown File Retrieval: {filename}")
        
        response = requests.get(
            f"{API_GATEWAY_URL}/documents/markdown/{filename}",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', '')
            print(f"✅ Markdown file retrieved successfully!")
            print(f"   📝 Content length: {len(content)} characters")
            print(f"   📄 Content preview:\n{content[:300]}...")
            return True
        else:
            print(f"❌ Markdown Retrieval Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Markdown Retrieval Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Final PDF Processing Communication Test")
    print("=" * 70)
    print(f"🌐 API Gateway: {API_GATEWAY_URL}")
    print(f"🔗 PDF Processor: {PDF_PROCESSOR_URL}")
    print("=" * 70)
    
    tests = [
        ("API Gateway Health", test_api_gateway_health),
        ("Gateway Services Health", test_api_gateway_services),
        ("PDF to Markdown Conversion", test_pdf_to_markdown_conversion),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        start_time = time.time()
        
        try:
            success = test_func()
            duration = time.time() - start_time
            
            if success:
                print(f"✅ PASSED: {test_name} ({duration:.2f}s)")
                passed += 1
            else:
                print(f"❌ FAILED: {test_name} ({duration:.2f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ ERROR: {test_name} ({duration:.2f}s) - {e}")
    
    # Results summary
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! PDF processing communication is working!")
        print("✅ API Gateway can successfully communicate with PDF processor")
        print("✅ PDF to Markdown conversion is functional")
        print("✅ End-to-end pipeline is working correctly")
    elif passed > 0:
        print("⚠️  Some tests passed, some failed. Partial functionality detected.")
    else:
        print("❌ All tests failed. PDF processing pipeline has issues.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)