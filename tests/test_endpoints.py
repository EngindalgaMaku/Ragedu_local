#!/usr/bin/env python3
"""
Test script for the new PDF-to-Markdown conversion endpoints
"""

import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_list_markdown_endpoint():
    """Test the /documents/list-markdown endpoint"""
    print("🔍 Testing /documents/list-markdown endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/documents/list-markdown")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Found {data['count']} markdown files:")
            for file in data['markdown_files']:
                print(f"   📄 {file}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_convert_pdf_endpoint():
    """Test the /documents/convert-pdf-to-markdown endpoint with an existing PDF"""
    print("🔍 Testing /documents/convert-pdf-to-markdown endpoint...")
    
    # Look for an existing PDF file in data/uploads
    upload_dir = Path("data/uploads")
    pdf_files = list(upload_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in data/uploads/ for testing")
        return False
    
    # Use the first PDF file found
    test_pdf = pdf_files[0]
    print(f"📄 Using test PDF: {test_pdf.name}")
    
    try:
        with open(test_pdf, 'rb') as f:
            files = {'file': (test_pdf.name, f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/documents/convert-pdf-to-markdown", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! PDF converted to: {data['markdown_filename']}")
            print(f"   📊 Processing method: {data['metadata'].get('processing_method', 'unknown')}")
            return True
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all endpoint tests"""
    print("🚀 Testing PDF-to-Markdown API endpoints")
    print("=" * 50)
    
    # Test list endpoint
    list_success = test_list_markdown_endpoint()
    print()
    
    # Test convert endpoint
    convert_success = test_convert_endpoint()
    print()
    
    # Summary
    print("📊 Test Results:")
    print(f"   📋 List endpoint: {'✅ PASS' if list_success else '❌ FAIL'}")
    print(f"   🔄 Convert endpoint: {'✅ PASS' if convert_success else '❌ FAIL'}")
    
    if list_success and convert_success:
        print("\n🎉 All tests passed! Both endpoints are working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the server logs for more details.")
        return False

if __name__ == "__main__":
    main()