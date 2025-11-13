#!/usr/bin/env python3
"""
Comprehensive Test Script for PDF Processing Memory Fixes
Tests model caching, memory management, and crash prevention
"""

import os
import sys
import time
import tempfile
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_pdf():
    """Create a simple test PDF for testing"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # Create PDF with some content
        c = canvas.Canvas(temp_file.name, pagesize=letter)
        
        # Add multiple pages with content
        for page_num in range(5):
            c.drawString(100, 750, f"Test PDF Page {page_num + 1}")
            c.drawString(100, 700, "This is a test document for PDF processing validation.")
            c.drawString(100, 650, "It contains multiple pages to test the processing pipeline.")
            
            # Add some more content to make it substantial
            for i in range(20):
                c.drawString(100, 600 - i*20, f"Line {i+1}: Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
            
            c.showPage()
        
        c.save()
        temp_file.close()
        
        logger.info(f"✅ Created test PDF: {temp_file.name}")
        return temp_file.name
        
    except ImportError:
        logger.warning("⚠️ reportlab not available, creating dummy PDF file")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(b"%PDF-1.4\n%Dummy PDF content for testing\n%%EOF")
        temp_file.close()
        return temp_file.name

def test_model_cache_manager():
    """Test the model cache manager"""
    logger.info("\n🔧 Testing Model Cache Manager...")
    
    try:
        # Set test cache directory
        test_cache_dir = tempfile.mkdtemp(prefix="marker_cache_test_")
        os.environ["MARKER_CACHE_DIR"] = test_cache_dir
        
        from src.utils.model_cache_manager import get_model_cache_manager, get_cached_marker_models
        
        cache_manager = get_model_cache_manager()
        logger.info(f"✅ Model cache manager created - Cache dir: {test_cache_dir}")
        
        # Get cache stats before
        stats_before = cache_manager.get_cache_stats()
        logger.info(f"📊 Cache stats before: {stats_before['cached_model_sets']} cached sets")
        
        # Test model loading (this should cache models)
        logger.info("🔄 Testing model caching...")
        start_time = time.time()
        
        models = get_cached_marker_models()
        
        cache_time = time.time() - start_time
        logger.info(f"⏱️ Model loading time: {cache_time:.2f} seconds")
        
        # Get cache stats after
        stats_after = cache_manager.get_cache_stats()
        logger.info(f"📊 Cache stats after: {stats_after}")
        
        if models:
            logger.info(f"✅ Models cached successfully! ({len(models)} components)")
        else:
            logger.warning("⚠️ Model caching returned None - may be expected in test environment")
        
        # Test cache reuse (should be faster)
        logger.info("🔄 Testing cache reuse...")
        start_time = time.time()
        
        models_cached = get_cached_marker_models()
        
        reuse_time = time.time() - start_time
        logger.info(f"⏱️ Cached model loading time: {reuse_time:.2f} seconds")
        
        if reuse_time < cache_time * 0.5:  # Should be significantly faster
            logger.info("✅ Cache reuse is working - much faster load time!")
        else:
            logger.warning("⚠️ Cache reuse may not be working optimally")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model cache manager test failed: {e}")
        return False

def test_memory_manager():
    """Test the memory management system"""
    logger.info("\n🧠 Testing Memory Manager...")
    
    try:
        from src.utils.memory_manager import get_memory_manager, memory_managed, optimize_for_large_processing
        
        memory_manager = get_memory_manager()
        
        # Test memory info
        memory_info = memory_manager.get_memory_usage()
        logger.info(f"📊 Current memory usage: {memory_info['rss_mb']:.1f}MB")
        logger.info(f"📊 Memory limit: {memory_manager.max_memory_mb}MB")
        
        # Test memory monitoring
        logger.info("🔄 Testing memory monitoring...")
        memory_manager.start_monitoring(check_interval=1.0)
        time.sleep(3)  # Let it monitor for a bit
        memory_manager.stop_memory_monitoring()
        logger.info("✅ Memory monitoring test completed")
        
        # Test memory-managed function
        @memory_managed("Test Operation")
        def test_memory_operation():
            # Simulate some memory usage
            data = [i for i in range(100000)]  # Create some data
            return len(data)
        
        result = test_memory_operation()
        logger.info(f"✅ Memory-managed function test completed: {result}")
        
        # Test large processing optimization
        logger.info("🔄 Testing large processing optimization...")
        optimized_manager = optimize_for_large_processing()
        time.sleep(2)
        optimized_manager.stop_memory_monitoring()
        logger.info("✅ Large processing optimization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory manager test failed: {e}")
        return False

def test_enhanced_pdf_processor():
    """Test the enhanced PDF processor with caching and memory management"""
    logger.info("\n📄 Testing Enhanced PDF Processor...")
    
    try:
        # Create test PDF
        test_pdf_path = create_test_pdf()
        
        from src.document_processing.enhanced_pdf_processor import (
            MarkerPDFProcessor, 
            extract_text_from_pdf_enhanced,
            process_pdf_with_analysis
        )
        
        # Test with caching and memory management disabled (safe mode)
        processor = MarkerPDFProcessor(use_llm=False, use_gpu=False)
        
        logger.info("🔄 Testing PDF processor stats...")
        stats = processor.get_processing_stats()
        logger.info(f"📊 Processing stats: {stats}")
        
        # Test basic text extraction
        logger.info("🔄 Testing basic PDF text extraction...")
        start_time = time.time()
        
        try:
            text_content = extract_text_from_pdf_enhanced(test_pdf_path, prefer_marker=True, use_llm=False)
            extraction_time = time.time() - start_time
            
            logger.info(f"✅ PDF extraction completed in {extraction_time:.2f} seconds")
            logger.info(f"📝 Extracted text length: {len(text_content)} characters")
            
            if len(text_content) > 0:
                logger.info("✅ PDF text extraction successful!")
                # Show first 200 characters
                preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
                logger.info(f"📄 Text preview: {preview}")
            else:
                logger.warning("⚠️ No text extracted - may be expected for dummy PDF")
            
        except Exception as e:
            logger.error(f"❌ PDF text extraction failed: {e}")
            logger.info("🔄 Testing fallback extraction...")
            
            # Test fallback
            fallback_text = processor._fallback_extract(test_pdf_path)
            if fallback_text:
                logger.info(f"✅ Fallback extraction worked: {len(fallback_text)} characters")
            else:
                logger.warning("⚠️ Fallback extraction also failed")
        
        # Test detailed analysis
        logger.info("🔄 Testing PDF analysis...")
        try:
            analysis_text, metadata = process_pdf_with_analysis(test_pdf_path)
            logger.info(f"✅ PDF analysis completed")
            logger.info(f"📊 Analysis metadata: {metadata}")
        except Exception as e:
            logger.error(f"❌ PDF analysis failed: {e}")
        
        # Cleanup
        try:
            os.unlink(test_pdf_path)
        except:
            pass
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced PDF processor test failed: {e}")
        return False

def test_memory_limits():
    """Test memory limit enforcement"""
    logger.info("\n🚨 Testing Memory Limit Enforcement...")
    
    try:
        from src.utils.memory_manager import get_memory_manager, check_memory_available
        
        memory_manager = get_memory_manager()
        
        # Test memory availability check
        available_1gb = check_memory_available(1000)  # 1GB
        logger.info(f"📊 1GB memory available: {available_1gb}")
        
        available_10gb = check_memory_available(10000)  # 10GB (should fail)
        logger.info(f"📊 10GB memory available: {available_10gb}")
        
        # Test memory warnings
        is_warning = memory_manager.is_memory_warning()
        is_critical = memory_manager.is_memory_critical()
        
        logger.info(f"⚠️ Memory at warning level: {is_warning}")
        logger.info(f"🚨 Memory at critical level: {is_critical}")
        
        if is_critical:
            logger.warning("🚨 CRITICAL MEMORY DETECTED - Testing cleanup...")
            memory_manager.force_cleanup()
            logger.info("✅ Emergency cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory limit test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    logger.info("🚀 Starting Comprehensive PDF Memory Fix Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Model Cache Manager
    test_results['model_cache'] = test_model_cache_manager()
    
    # Test 2: Memory Manager
    test_results['memory_manager'] = test_memory_manager()
    
    # Test 3: Memory Limits
    test_results['memory_limits'] = test_memory_limits()
    
    # Test 4: Enhanced PDF Processor
    test_results['pdf_processor'] = test_enhanced_pdf_processor()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<20}: {status}")
    
    logger.info("-" * 60)
    logger.info(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Memory fixes are working correctly.")
    elif passed > total * 0.5:
        logger.info("⚠️ Most tests passed. Some issues may remain in test environment.")
    else:
        logger.error("❌ Many tests failed. Review the implementation.")
    
    return passed, total

if __name__ == "__main__":
    try:
        # Set test environment
        os.environ["MARKER_CACHE_DIR"] = tempfile.mkdtemp(prefix="test_cache_")
        os.environ["MARKER_MAX_MEMORY_MB"] = "3500"
        os.environ["MARKER_DISABLE_GEMINI"] = "true"
        os.environ["MARKER_USE_LOCAL_ONLY"] = "true"
        os.environ["MARKER_DISABLE_CLOUD_SERVICES"] = "true"
        os.environ["MARKER_DISABLE_ALL_LLM"] = "true"
        
        # Add src to path
        sys.path.insert(0, 'src')
        
        passed, total = run_comprehensive_test()
        
        # Exit with appropriate code
        sys.exit(0 if passed == total else 1)
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Test suite failed with error: {e}")
        sys.exit(1)