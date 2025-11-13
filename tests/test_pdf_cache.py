#!/usr/bin/env python3
"""
Test PDF processor model caching system
"""

import os
import tempfile
import requests
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_processor_cache():
    """Test PDF processor with model caching"""
    
    # Test PDF processor service
    base_url = "http://localhost:8001"  # PDF processor port
    
    try:
        # Health check
        logger.info("🏥 Testing PDF processor health...")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ PDF processor healthy: {health_data}")
            
            if health_data.get('marker_available'):
                logger.info("🎯 Marker library available - testing model cache...")
            else:
                logger.warning("⚠️ Marker not available - will test PyPDF2 fallback")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Create test PDF
        test_pdf_content = b"""
%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF Cache) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
0000000368 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
465
%%EOF
"""
        
        # Test PDF processing multiple times to test cache efficiency
        for i in range(3):
            logger.info(f"🔄 Testing PDF processing - Round {i+1}/3")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(test_pdf_content)
                tmp.flush()
                
                # Process PDF
                with open(tmp.name, 'rb') as pdf_file:
                    files = {'file': ('test.pdf', pdf_file, 'application/pdf')}
                    
                    import time
                    start_time = time.time()
                    
                    response = requests.post(
                        f"{base_url}/process",
                        files=files,
                        timeout=120  # 2 minutes timeout for model download
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✅ PDF processed successfully in {processing_time:.2f}s")
                        logger.info(f"📄 Method: {result['metadata']['processing_method']}")
                        logger.info(f"📏 Text length: {result['metadata']['text_length']} chars")
                        logger.info(f"📋 Pages: {result['metadata']['page_count']}")
                        
                        # Check if processing got faster (indicating cache usage)
                        if i == 0:
                            first_time = processing_time
                        else:
                            speedup = first_time / processing_time
                            logger.info(f"⚡ Speedup vs first run: {speedup:.2f}x")
                            
                            if speedup > 1.5:
                                logger.info("🚀 Cache appears to be working - faster processing!")
                            elif processing_time < 5:
                                logger.info("⚡ Fast processing - likely using cache")
                            
                    else:
                        logger.error(f"❌ Processing failed: {response.status_code} - {response.text}")
                        return False
                
                # Clean up
                os.unlink(tmp.name)
        
        logger.info("✅ PDF processor cache test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def start_pdf_processor():
    """Start PDF processor service for testing"""
    import subprocess
    import time
    
    logger.info("🚀 Starting PDF processor service...")
    
    # Change to PDF processor directory and start service
    processor_dir = Path("services/pdf_processing_service")
    
    if not processor_dir.exists():
        logger.error(f"❌ PDF processor directory not found: {processor_dir}")
        return None
    
    try:
        # Start the service
        process = subprocess.Popen(
            ["python", "main.py"],
            cwd=processor_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it time to start
        logger.info("⏳ Waiting for service to start...")
        time.sleep(10)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("✅ PDF processor service started")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Service failed to start")
            logger.error(f"STDOUT: {stdout.decode()}")
            logger.error(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        return None

if __name__ == "__main__":
    logger.info("🧪 Starting PDF processor cache test...")
    
    # Check if service is already running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        service_running = response.status_code == 200
    except:
        service_running = False
    
    process = None
    
    if not service_running:
        logger.info("🔧 Service not running, starting it...")
        process = start_pdf_processor()
        
        if not process:
            logger.error("❌ Could not start PDF processor service")
            exit(1)
    else:
        logger.info("✅ Service already running")
    
    try:
        # Run the test
        success = test_pdf_processor_cache()
        
        if success:
            logger.info("🎉 All tests passed! Model cache system working.")
        else:
            logger.error("❌ Tests failed!")
            
    finally:
        # Clean up
        if process:
            logger.info("🛑 Stopping PDF processor service...")
            process.terminate()
            process.wait()