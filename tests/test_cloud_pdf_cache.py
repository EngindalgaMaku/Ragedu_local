#!/usr/bin/env python3
"""
Test PDF processor model caching in Google Cloud Run
"""

import requests
import time
import logging
import tempfile
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cloud_pdf_cache():
    """Test PDF processor caching in Cloud Run"""
    
    # PDF Processor Cloud Run URL
    pdf_processor_url = "https://pdf-processor-1051060211087.europe-west1.run.app"
    
    try:
        # Create test PDF content
        test_pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj  
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R
/Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 50 >>
stream
BT /F1 12 Tf 72 720 Td (Model Cache Test PDF) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
0000000375 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
465
%%EOF"""

        # Test multiple rounds to measure caching effect
        processing_times = []
        
        for i in range(3):
            logger.info(f"🔄 Cloud Test Round {i+1}/3")
            
            # Health check first
            logger.info("🏥 Testing health endpoint...")
            health_response = requests.get(f"{pdf_processor_url}/health", timeout=30)
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                logger.info(f"✅ Service healthy: {health_data}")
            else:
                logger.error(f"❌ Health check failed: {health_response.status_code}")
                continue
            
            # Create temporary PDF file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(test_pdf_content)
                tmp.flush()
                tmp_path = tmp.name
            
            # Process PDF and measure time
            with open(tmp_path, 'rb') as pdf_file:
                files = {'file': ('test_cache.pdf', pdf_file, 'application/pdf')}
                
                start_time = time.time()
                logger.info(f"📤 Uploading PDF to Cloud Run...")
                
                response = requests.post(
                    f"{pdf_processor_url}/process",
                    files=files,
                    timeout=300  # 5 minute timeout for first run (model download)
                )
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ PDF processed in {processing_time:.2f}s")
                    logger.info(f"📄 Method: {result['metadata']['processing_method']}")
                    logger.info(f"📏 Content length: {result['metadata']['text_length']} chars")
                    logger.info(f"📋 Pages: {result['metadata']['page_count']}")
                    
                    # Analyze caching performance
                    if i == 0:
                        if processing_time > 60:
                            logger.info("⏳ First run took long - likely downloading 258MB model")
                        else:
                            logger.info("⚡ Fast first run - cache might be working!")
                    else:
                        speedup = processing_times[0] / processing_time
                        logger.info(f"🚀 Speedup vs first run: {speedup:.2f}x")
                        
                        if speedup > 5:
                            logger.info("🎉 Excellent cache performance!")
                        elif speedup > 2:
                            logger.info("✅ Good cache performance")
                        elif processing_time < 10:
                            logger.info("⚡ Fast processing - cache likely working")
                        else:
                            logger.warning("⚠️ No significant speedup - cache might not be working")
                
                else:
                    logger.error(f"❌ Processing failed: {response.status_code}")
                    logger.error(f"Error: {response.text}")
            
            # Clean up
            os.unlink(tmp_path)
            
            # Wait between requests to avoid rate limiting
            if i < 2:
                logger.info("⏳ Waiting 10 seconds before next test...")
                time.sleep(10)
        
        # Final analysis
        if len(processing_times) >= 2:
            avg_subsequent = sum(processing_times[1:]) / len(processing_times[1:])
            total_speedup = processing_times[0] / avg_subsequent
            
            logger.info(f"\n📊 CACHE PERFORMANCE ANALYSIS:")
            logger.info(f"First run: {processing_times[0]:.2f}s")
            logger.info(f"Avg subsequent: {avg_subsequent:.2f}s")
            logger.info(f"Total speedup: {total_speedup:.2f}x")
            
            if total_speedup > 10:
                logger.info("🎉 EXCELLENT: Model cache working perfectly!")
                return True
            elif total_speedup > 3:
                logger.info("✅ GOOD: Cache providing significant speedup")
                return True
            elif avg_subsequent < 15:
                logger.info("⚡ FAST: Quick processing indicates caching")
                return True
            else:
                logger.warning("⚠️ SLOW: Cache might not be working optimally")
                return False
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🌤️ Starting Cloud PDF Processor Cache Test...")
    logger.info("🚨 Remember to update PDF_PROCESSOR_URL after deployment!")
    
    # Note: URL needs to be updated after Cloud Run deployment
    success = test_cloud_pdf_cache()
    
    if success:
        logger.info("🎉 Cloud cache test PASSED!")
    else:
        logger.error("❌ Cloud cache test FAILED!")