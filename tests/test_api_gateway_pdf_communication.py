#!/usr/bin/env python3
"""
Test script for API Gateway and PDF Processor Communication
Tests PDF to Markdown conversion through microservices architecture
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Service URLs - Update these based on actual deployed services
API_GATEWAY_URL = "https://api-gateway-1051060211087.europe-west1.run.app"
PDF_PROCESSOR_URL = "https://pdf-processor-1051060211087.europe-west1.run.app" 
DOC_PROCESSOR_URL = "https://doc-proc-service-1051060211087.europe-west1.run.app"

class TestResults:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, name, success, details="", duration=0):
        self.tests.append({
            'name': name,
            'success': success,
            'details': details,
            'duration': duration
        })
        if success:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {len(self.tests)}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/len(self.tests)*100):.1f}%" if self.tests else "0%")
        
        for test in self.tests:
            status = "✅ PASS" if test['success'] else "❌ FAIL"
            print(f"\n{status} {test['name']} ({test['duration']:.2f}s)")
            if test['details']:
                print(f"   └── {test['details']}")

def create_test_pdf():
    """Create a simple test PDF file"""
    try:
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Add content to PDF
        p.drawString(100, 750, "Test PDF Document")
        p.drawString(100, 720, "Bu bir test PDF dosyasıdır.")
        p.drawString(100, 690, "This document contains:")
        p.drawString(120, 660, "• Turkish text for language testing")
        p.drawString(120, 630, "• Multiple lines to test chunking")
        p.drawString(120, 600, "• Special characters: çğıöşü ÇĞIÖŞÜ")
        p.drawString(100, 570, "Academic Content:")
        p.drawString(120, 540, "The theory of relativity, developed by Einstein,")
        p.drawString(120, 510, "revolutionized our understanding of space and time.")
        p.drawString(120, 480, "E = mc² is the most famous equation in physics.")
        
        # Add more content on a second page
        p.showPage()
        p.drawString(100, 750, "Page 2 - Additional Content")
        p.drawString(100, 720, "This is additional content to test multi-page processing.")
        p.drawString(100, 690, "RAG (Retrieval-Augmented Generation) systems combine:")
        p.drawString(120, 660, "1. Document processing and chunking")
        p.drawString(120, 630, "2. Vector embeddings for semantic search")
        p.drawString(120, 600, "3. Language models for generation")
        
        p.save()
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error creating test PDF: {e}")
        return None

def test_service_health(service_name, url, results):
    """Test service health endpoint"""
    start_time = time.time()
    try:
        response = requests.get(f"{url}/health", timeout=10)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            details = f"Status: {status}, Response time: {duration:.2f}s"
            results.add_test(f"{service_name} Health Check", True, details, duration)
            return True, data
        else:
            results.add_test(f"{service_name} Health Check", False, 
                           f"HTTP {response.status_code}: {response.text[:100]}", duration)
            return False, None
    except Exception as e:
        duration = time.time() - start_time
        results.add_test(f"{service_name} Health Check", False, str(e), duration)
        return False, None

def test_gateway_services_health(results):
    """Test API Gateway services health check"""
    start_time = time.time()
    try:
        response = requests.get(f"{API_GATEWAY_URL}/health/services", timeout=15)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            services_status = data.get('services', {})
            
            details = []
            all_healthy = True
            
            for service_name, service_info in services_status.items():
                status = service_info.get('status', 'unknown')
                if status != 'ok':
                    all_healthy = False
                response_time = service_info.get('response_time', 'N/A')
                details.append(f"{service_name}: {status} ({response_time}s)")
            
            results.add_test("Gateway Services Health Check", all_healthy, 
                           "; ".join(details), duration)
            return all_healthy, data
        else:
            results.add_test("Gateway Services Health Check", False,
                           f"HTTP {response.status_code}: {response.text[:100]}", duration)
            return False, None
    except Exception as e:
        duration = time.time() - start_time
        results.add_test("Gateway Services Health Check", False, str(e), duration)
        return False, None

def test_pdf_processor_direct(pdf_content, results):
    """Test PDF processor directly (bypass gateway)"""
    start_time = time.time()
    try:
        files = {'file': ('test.pdf', pdf_content, 'application/pdf')}
        response = requests.post(f"{PDF_PROCESSOR_URL}/process", 
                               files=files, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', '')
            metadata = data.get('metadata', {})
            
            details = f"Content length: {len(content)}, Method: {metadata.get('processing_method', 'unknown')}, Pages: {metadata.get('page_count', 0)}"
            results.add_test("PDF Processor Direct Test", True, details, duration)
            return True, data
        else:
            results.add_test("PDF Processor Direct Test", False,
                           f"HTTP {response.status_code}: {response.text[:200]}", duration)
            return False, None
    except Exception as e:
        duration = time.time() - start_time
        results.add_test("PDF Processor Direct Test", False, str(e), duration)
        return False, None

def test_gateway_pdf_conversion(pdf_content, results):
    """Test PDF to Markdown conversion through API Gateway"""
    start_time = time.time()
    try:
        files = {'file': ('test.pdf', pdf_content, 'application/pdf')}
        response = requests.post(f"{API_GATEWAY_URL}/documents/convert-document-to-markdown",
                               files=files, timeout=120)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            success = data.get('success', False)
            markdown_filename = data.get('markdown_filename', '')
            metadata = data.get('metadata', {})
            
            details = f"Success: {success}, File: {markdown_filename}, Processing: {metadata.get('processing_method', 'unknown')}"
            results.add_test("Gateway PDF to Markdown Conversion", success, details, duration)
            return success, data
        else:
            error_details = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                error_details += f": {error_data.get('detail', response.text[:200])}"
            except:
                error_details += f": {response.text[:200]}"
            
            results.add_test("Gateway PDF to Markdown Conversion", False, error_details, duration)
            return False, None
    except Exception as e:
        duration = time.time() - start_time
        results.add_test("Gateway PDF to Markdown Conversion", False, str(e), duration)
        return False, None

def test_markdown_retrieval(markdown_filename, results):
    """Test retrieving converted markdown file"""
    start_time = time.time()
    try:
        response = requests.get(f"{API_GATEWAY_URL}/documents/markdown/{markdown_filename}",
                              timeout=10)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', '')
            
            # Verify content quality
            quality_checks = {
                'has_content': len(content) > 100,
                'has_turkish': any(char in content.lower() for char in 'çğıöşü'),
                'has_english': 'Einstein' in content or 'theory' in content.lower(),
                'has_equation': 'mc²' in content or 'E = mc' in content
            }
            
            passed_checks = sum(quality_checks.values())
            total_checks = len(quality_checks)
            
            details = f"Content length: {len(content)}, Quality checks: {passed_checks}/{total_checks}"
            success = passed_checks >= total_checks - 1  # Allow 1 failed check
            
            results.add_test("Markdown Content Retrieval", success, details, duration)
            return success, content
        else:
            results.add_test("Markdown Content Retrieval", False,
                           f"HTTP {response.status_code}: {response.text[:100]}", duration)
            return False, None
    except Exception as e:
        duration = time.time() - start_time
        results.add_test("Markdown Content Retrieval", False, str(e), duration)
        return False, None

def test_markdown_files_listing(results):
    """Test listing markdown files"""
    start_time = time.time()
    try:
        response = requests.get(f"{API_GATEWAY_URL}/documents/list-markdown", timeout=10)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            markdown_files = data.get('markdown_files', [])
            count = data.get('count', 0)
            
            details = f"Found {count} markdown files"
            results.add_test("Markdown Files Listing", True, details, duration)
            return True, markdown_files
        else:
            results.add_test("Markdown Files Listing", False,
                           f"HTTP {response.status_code}: {response.text[:100]}", duration)
            return False, []
    except Exception as e:
        duration = time.time() - start_time
        results.add_test("Markdown Files Listing", False, str(e), duration)
        return False, []

def run_comprehensive_tests():
    """Run all tests and generate report"""
    print("🚀 Starting API Gateway & PDF Processor Communication Tests")
    print("="*60)
    
    results = TestResults()
    
    # Step 1: Test service health
    print("\n📋 Testing Service Health...")
    test_service_health("API Gateway", API_GATEWAY_URL, results)
    test_service_health("PDF Processor", PDF_PROCESSOR_URL, results)
    test_service_health("Document Processor", DOC_PROCESSOR_URL, results)
    test_gateway_services_health(results)
    
    # Step 2: Create test PDF
    print("\n📄 Creating Test PDF...")
    pdf_content = create_test_pdf()
    if not pdf_content:
        results.add_test("Test PDF Creation", False, "Failed to create test PDF")
        results.print_summary()
        return
    
    results.add_test("Test PDF Creation", True, f"Created PDF ({len(pdf_content)} bytes)")
    
    # Step 3: Test PDF processor directly
    print("\n🔧 Testing PDF Processor Direct Communication...")
    direct_success, direct_data = test_pdf_processor_direct(pdf_content, results)
    
    # Step 4: Test through API Gateway
    print("\n🌐 Testing PDF to Markdown through API Gateway...")
    gateway_success, gateway_data = test_gateway_pdf_conversion(pdf_content, results)
    
    # Step 5: Test markdown retrieval
    if gateway_success and gateway_data:
        markdown_filename = gateway_data.get('markdown_filename')
        if markdown_filename:
            print("\n📖 Testing Markdown Retrieval...")
            test_markdown_retrieval(markdown_filename, results)
    
    # Step 6: Test markdown files listing
    print("\n📂 Testing Markdown Files Listing...")
    test_markdown_files_listing(results)
    
    # Generate final report
    results.print_summary()
    
    # Detailed analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    if direct_success and gateway_success:
        print("✅ End-to-end PDF to Markdown conversion is working!")
        print("   Both direct PDF processor and API Gateway routes are functional.")
    elif direct_success and not gateway_success:
        print("⚠️ PDF processor works directly but API Gateway routing has issues.")
        print("   Check API Gateway configuration and service URLs.")
    elif not direct_success and gateway_success:
        print("⚠️ Unusual: Gateway works but direct processor doesn't.")
        print("   This might indicate URL or configuration issues.")
    else:
        print("❌ PDF processing is not working properly.")
        print("   Check service deployment and configuration.")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_tests()
        
        # Save results to file
        report_file = "api_gateway_pdf_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_tests': len(results.tests),
                'passed': results.passed,
                'failed': results.failed,
                'tests': results.tests
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Detailed test report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results.failed == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test execution failed: {e}")
        sys.exit(1)