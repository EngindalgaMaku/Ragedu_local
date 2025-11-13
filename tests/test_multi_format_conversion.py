#!/usr/bin/env python3
"""Test script for multi-format document conversion endpoint."""

import requests
import os
import io
from pathlib import Path

# Create test documents programmatically
def create_test_documents():
    """Create sample documents in different formats for testing."""
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    print("📄 Creating test documents...")
    
    # Create a simple text file (will test as a base case)
    simple_text = """# Test Document

This is a test document for multi-format conversion.

## Section 1
This section contains some basic text content.

## Section 2
- List item 1
- List item 2
- List item 3

**Bold text** and *italic text* formatting.
"""
    
    # Save as text file for reference
    with open(test_dir / "test_document.txt", "w", encoding="utf-8") as f:
        f.write(simple_text)
    
    # Create a simple HTML file (if HTML converter is available)
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Test Document</h1>
    <p>This is a test document for multi-format conversion.</p>
    
    <h2>Section 1</h2>
    <p>This section contains some basic text content.</p>
    
    <h2>Section 2</h2>
    <ul>
        <li>List item 1</li>
        <li>List item 2</li>
        <li>List item 3</li>
    </ul>
    
    <p><strong>Bold text</strong> and <em>italic text</em> formatting.</p>
</body>
</html>"""
    
    with open(test_dir / "test_document.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Created test documents in {test_dir}")
    return test_dir

def test_conversion_endpoint(file_path, expected_format):
    """Test the conversion endpoint with a specific file."""
    print(f"\n🧪 Testing conversion: {file_path.name} ({expected_format})")
    
    try:
        url = "http://localhost:8000/documents/convert-document-to-markdown"
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, get_content_type(file_path.suffix))}
            
            response = requests.post(url, files=files, timeout=30)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {expected_format} conversion successful")
            print(f"📄 Markdown length: {len(result.get('markdown', ''))}")
            
            # Show first 200 characters of markdown output
            markdown_preview = result.get('markdown', '')[:200]
            if markdown_preview:
                print(f"📋 Preview: {markdown_preview}...")
            
            return True, result
            
        elif response.status_code == 422:
            print(f"⚠️ {expected_format} format not supported (422 - Validation Error)")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False, None
            
        else:
            print(f"❌ {expected_format} conversion failed - Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False, None
            
    except requests.exceptions.Timeout:
        print(f"⏰ {expected_format} conversion timed out")
        return False, None
    except Exception as e:
        print(f"❌ {expected_format} conversion error: {e}")
        return False, None

def get_content_type(file_extension):
    """Get the appropriate content type for file extension."""
    content_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.html': 'text/html',
        '.txt': 'text/plain'
    }
    return content_types.get(file_extension, 'application/octet-stream')

def test_api_validation():
    """Test API endpoint validation with invalid inputs."""
    print("\n🔍 Testing API validation...")
    
    url = "http://localhost:8000/documents/convert-document-to-markdown"
    
    # Test with no file
    try:
        response = requests.post(url)
        if response.status_code == 422:
            print("✅ No file validation works correctly")
        else:
            print(f"⚠️ Unexpected response for no file: {response.status_code}")
    except Exception as e:
        print(f"❌ No file test error: {e}")
    
    # Test with empty file
    try:
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        response = requests.post(url, files=files)
        print(f"📋 Empty file response: {response.status_code}")
    except Exception as e:
        print(f"❌ Empty file test error: {e}")

def main():
    """Main test function."""
    print("🚀 Starting Multi-Format Document Conversion Tests\n")
    
    # Ensure server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding correctly")
            return False
        print("✅ API server is running")
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        return False
    
    # Create test documents
    test_dir = create_test_documents()
    
    # Test validation
    test_api_validation()
    
    # Test available formats
    test_files = [
        (test_dir / "test_document.txt", "TXT"),
        (test_dir / "test_document.html", "HTML"),
    ]
    
    successful_conversions = 0
    total_tests = len(test_files)
    
    print(f"\n📋 Testing {total_tests} format conversions...")
    
    for file_path, format_name in test_files:
        if file_path.exists():
            success, result = test_conversion_endpoint(file_path, format_name)
            if success:
                successful_conversions += 1
        else:
            print(f"⚠️ Test file not found: {file_path}")
    
    print(f"\n📊 CONVERSION TEST RESULTS:")
    print(f"✅ Successful: {successful_conversions}/{total_tests}")
    print(f"❌ Failed: {total_tests - successful_conversions}/{total_tests}")
    
    if successful_conversions > 0:
        print(f"\n🎉 Multi-format conversion is working!")
        print(f"   At least {successful_conversions} format(s) can be converted successfully")
        
        # Note about marker-pdf[full] capabilities
        print(f"\n📝 NOTE: This test used simple formats (TXT, HTML)")
        print(f"   For PDF, DOCX, PPTX, XLSX support, ensure marker-pdf[full] is properly installed")
        print(f"   The endpoint will accept these formats if converters are available")
        
        return True
    else:
        print(f"\n❌ No successful conversions - check server configuration")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)