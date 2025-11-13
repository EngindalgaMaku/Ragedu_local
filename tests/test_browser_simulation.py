#!/usr/bin/env python3
"""
Browser Simulation Test - Exact Frontend Request Reproduction
Tests the exact same request format that the frontend JavaScript makes
"""
import requests
import json
from pathlib import Path

def test_exact_frontend_request():
    """Test the exact same request format that frontend JavaScript makes"""
    print("🔍 Testing exact frontend request format...")
    
    # Create a test PDF content (minimal valid PDF)
    pdf_content = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \n0000000173 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n253\n%%EOF'
    
    # Prepare files exactly like FormData does in browser
    files = {
        'file': ('test_document.pdf', pdf_content, 'application/pdf')
    }
    
    # Simulate browser headers (what fetch() typically sends)
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    # Note: Content-Type is automatically set by requests when using files parameter
    
    try:
        print("📤 Sending request to http://localhost:8000/documents/convert-document-to-markdown")
        print(f"📎 File: test_document.pdf ({len(pdf_content)} bytes)")
        print(f"🌐 Headers: {headers}")
        
        response = requests.post(
            'http://localhost:8000/documents/convert-document-to-markdown',
            files=files,
            headers=headers,
            timeout=30
        )
        
        print(f"📥 Response Status: {response.status_code}")
        print(f"📥 Response Headers: {dict(response.headers)}")
        print(f"📥 Response Body: {response.text}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Browser simulation worked!")
            try:
                result_json = response.json()
                print(f"📄 Parsed Response: {json.dumps(result_json, indent=2)}")
            except json.JSONDecodeError:
                print("⚠️  Response is not valid JSON")
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print("🚨 This matches the frontend error!")
            
        return response.status_code == 200
        
    except requests.exceptions.ConnectionError as e:
        print(f"🚫 Connection Error: {e}")
        print("❌ API Gateway may not be running or accessible")
        return False
    except requests.exceptions.Timeout as e:
        print(f"⏱️  Timeout Error: {e}")
        return False
    except Exception as e:
        print(f"💥 Unexpected Error: {e}")
        return False

def test_preflight_cors():
    """Test CORS preflight request that browsers send for complex requests"""
    print("\n🔍 Testing CORS preflight request...")
    
    headers = {
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type',
        'Origin': 'http://localhost:3000'
    }
    
    try:
        response = requests.options(
            'http://localhost:8000/documents/convert-document-to-markdown',
            headers=headers,
            timeout=10
        )
        
        print(f"📥 OPTIONS Response Status: {response.status_code}")
        print(f"📥 CORS Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ CORS preflight successful")
            return True
        else:
            print(f"❌ CORS preflight failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"💥 CORS preflight error: {e}")
        return False

def test_api_health():
    """Test if API Gateway is responding"""
    print("\n🔍 Testing API Gateway health...")
    
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        print(f"📥 Health Status: {response.status_code}")
        print(f"📥 Health Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"💥 Health check error: {e}")
        return False

def main():
    print("🚀 Browser Simulation Test - Debugging Frontend Document Upload")
    print("=" * 70)
    
    # Test 1: Basic health check
    health_ok = test_api_health()
    if not health_ok:
        print("🚫 API Gateway is not responding. Cannot continue tests.")
        return
    
    # Test 2: CORS preflight
    cors_ok = test_preflight_cors()
    
    # Test 3: Exact frontend request simulation
    frontend_ok = test_exact_frontend_request()
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY:")
    print(f"🏥 API Health: {'✅' if health_ok else '❌'}")
    print(f"🌐 CORS Preflight: {'✅' if cors_ok else '❌'}")
    print(f"📱 Frontend Request: {'✅' if frontend_ok else '❌'}")
    
    if frontend_ok:
        print("\n🎉 SUCCESS: Backend works perfectly!")
        print("🤔 The issue must be elsewhere:")
        print("   • Frontend error handling/display")
        print("   • Network issues in browser")
        print("   • Browser-specific JavaScript errors")
    else:
        print("\n🚨 ISSUE CONFIRMED: Backend request fails")
        print("💡 Next steps: Check API Gateway logs and microservice connectivity")

if __name__ == "__main__":
    main()