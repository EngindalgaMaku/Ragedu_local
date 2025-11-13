#!/usr/bin/env python3
"""
API Gateway Only PDF Communication Test
Tests PDF processing through API Gateway (no direct service access)
"""
import requests
import json
import time
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io

# Service URLs
API_GATEWAY_URL = "https://api-gateway-1051060211087.europe-west1.run.app"

def create_test_pdf():
    """Create a test PDF file for testing"""
    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 100, "Test Document for PDF Processing")
        
        # Add some content
        c.setFont("Helvetica", 12)
        content = [
            "Bu bir test belgesidir.",
            "PDF'den Markdown'a dönüştürme işlemini test etmek için kullanılır.",
            "",
            "İçerik özellikleri:",
            "• Türkçe karakter desteği: çöğüşıüö",
            "• Basit metin formatları",
            "• Liste öğeleri",
            "",
            "Test başarılı olursa bu içerik markdown formatında geri dönecektir.",
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
            print(f"✅ Gateway Services Health: {json.dumps(data, indent=2)}")
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
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ PDF to Markdown Conversion Successful!")
            print(f"   📄 Markdown file: {data.get('markdown_filename')}")
            print(f"   📊 Metadata: {data.get('metadata', {})}")
            
            # Test retrieving the created markdown file
            if data.get('markdown_filename'):
                return test_markdown_retrieval(data['markdown_filename'])
            return True
        else:
            print(f"❌ PDF to Markdown Conversion Failed: {response.status_code}")
            print(f"   Response: {response.text}")
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
            print(f"   📄 Content preview:\n{content[:200]}...")
            return True
        else:
            print(f"❌ Markdown Retrieval Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Markdown Retrieval Error: {e}")
        return False

def test_markdown_listing():
    """Test listing markdown files"""
    try:
        print("📂 Testing Markdown Files Listing...")
        
        response = requests.get(f"{API_GATEWAY_URL}/documents/list-markdown", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Markdown Files Listed Successfully!")
            print(f"   📊 Total files: {data.get('count', 0)}")
            files = data.get('markdown_files', [])
            if files:
                print(f"   📄 Files: {', '.join(files[:5])}")  # Show first 5
            return True
        else:
            print(f"❌ Markdown Listing Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Markdown Listing Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting API Gateway PDF Processing Tests")
    print("=" * 60)
    
    tests = [
        ("API Gateway Health", test_api_gateway_health),
        ("Gateway Services Health", test_api_gateway_services),
        ("PDF to Markdown Conversion", test_pdf_to_markdown_conversion),
        ("Markdown Files Listing", test_markdown_listing),
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
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! PDF processing communication is working!")
        print("✅ API Gateway can successfully communicate with PDF processor")
        print("✅ PDF to Markdown conversion is functional")
    else:
        print("❌ Some tests failed. Please check service configuration.")
        
    return passed == total

if __name__ == "__main__":
    main()