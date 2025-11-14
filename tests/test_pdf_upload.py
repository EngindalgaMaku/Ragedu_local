#!/usr/bin/env python3
"""
Test script for PDF to Markdown conversion via DocStrange service
"""
import requests
import time

# Test with a real PDF file
pdf_path = "data/uploads/21164209_Biyoloji_9.pdf"

print(f"🧪 Testing PDF upload with: {pdf_path}")
print("=" * 60)

# Read PDF file
with open(pdf_path, 'rb') as f:
    pdf_content = f.read()
    print(f"✅ PDF file loaded: {len(pdf_content)} bytes")

# Prepare request
files = {'file': ('test.pdf', pdf_content, 'application/pdf')}

# Call API Gateway endpoint
api_url = "http://localhost:8000/documents/convert-document-to-markdown"

print(f"\n📤 Sending request to: {api_url}")
print("⏳ Please wait, this may take up to 10 minutes for large PDFs...")
print("=" * 60)

start_time = time.time()

try:
    response = requests.post(api_url, files=files, timeout=600)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Request completed in {elapsed_time:.2f} seconds")
    print(f"📊 Status Code: {response.status_code}")
    print("=" * 60)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS!")
        print(f"📄 Markdown file: {result.get('markdown_filename', 'N/A')}")
        print(f"📝 Message: {result.get('message', 'N/A')}")
        
        # Check if we got content
        if 'result' in result:
            content_preview = str(result['result'])[:200]
            print(f"\n📖 Content preview:\n{content_preview}...")
        
    else:
        print("❌ FAILED!")
        print(f"Error: {response.text[:500]}")
        
except requests.exceptions.Timeout:
    print("\n⏰ REQUEST TIMEOUT (10 minutes)")
    print("The PDF is too large or Nanonets is taking too long to process.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")

print("\n" + "=" * 60)
print("🔍 Check Docker logs for more details:")
print("   docker logs docstrange-service --tail 50")
print("   docker logs api-gateway --tail 50")

