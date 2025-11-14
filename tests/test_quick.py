#!/usr/bin/env python3
import requests
import time

# Use smallest PDF
pdf_path = "plan.pdf"  # Very small file

print(f"🧪 Quick test with: {pdf_path}")

with open(pdf_path, 'rb') as f:
    files = {'file': ('test.pdf', f, 'application/pdf')}
    
    print("📤 Sending request...")
    start = time.time()
    
    try:
        # Test with short timeout first
        response = requests.post(
            "http://localhost:8000/documents/convert-document-to-markdown",
            files=files,
            timeout=120  # 2 minutes only
        )
        
        elapsed = time.time() - start
        print(f"⏱️ Completed in {elapsed:.1f}s")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS - Markdown: {result.get('markdown_filename')}")
        else:
            print(f"❌ ERROR: {response.text[:300]}")
            
    except requests.Timeout:
        print(f"⏰ Timeout after {time.time()-start:.1f}s")
    except Exception as e:
        print(f"❌ Error: {e}")

# Also check logs
print("\n📋 DocStrange logs:")
import subprocess
result = subprocess.run(["docker", "logs", "docstrange-service", "--tail", "20"], 
                       capture_output=True, text=True)
print(result.stdout[-500:] if result.stdout else "No logs")

