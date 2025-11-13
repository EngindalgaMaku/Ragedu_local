#!/usr/bin/env python3
"""
CRITICAL CACHE FIX TESTER
Tests the marker model cache fix to ensure models are NOT downloaded repeatedly
"""

import os
import time
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cache_environment():
    """Test that cache environment variables are properly set"""
    logger.info("🔍 Testing cache environment setup...")
    
    # CRITICAL: Import enhanced_pdf_processor first to trigger environment setup
    try:
        from src.document_processing.enhanced_pdf_processor import _setup_marker_environment
        logger.info("✅ Imported environment setup function")
    except Exception as e:
        logger.warning(f"⚠️ Could not import setup function: {e}")
        # Try importing the whole module which will trigger setup
        try:
            import src.document_processing.enhanced_pdf_processor
            logger.info("✅ Imported enhanced_pdf_processor module")
        except Exception as e2:
            logger.error(f"❌ Could not import enhanced_pdf_processor: {e2}")
    
    required_vars = [
        "TORCH_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HOME",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_OFFLINE"
    ]
    
    cache_base = os.getenv("MARKER_CACHE_DIR", "/app/models")
    logger.info(f"📁 Cache base directory: {cache_base}")
    
    missing_vars = []
    for var in required_vars:
        value = os.environ.get(var, "")
        if not value:
            missing_vars.append(var)
            logger.error(f"❌ Missing environment variable: {var}")
        else:
            logger.info(f"✅ {var}: {value}")
            
            # Check if directory exists
            if os.path.isdir(value):
                logger.info(f"📁 Directory exists: {value}")
                # Count files
                try:
                    file_count = len([f for f in Path(value).rglob("*") if f.is_file()])
                    logger.info(f"📊 Files in cache: {file_count}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not count files: {e}")
            else:
                logger.warning(f"⚠️ Directory missing: {value}")
    
    if missing_vars:
        logger.error(f"❌ Cache setup FAILED - missing variables: {missing_vars}")
        return False
    
    logger.info("✅ Cache environment setup PASSED")
    return True

def test_marker_import():
    """Test that marker imports work with cache"""
    logger.info("🔍 Testing marker library import...")
    
    try:
        # This should NOT trigger downloads if cache is working
        from marker.models import create_model_dict
        logger.info("✅ Marker import successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Marker import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Marker import error: {e}")
        return False

def test_model_loading():
    """Test actual model loading - this is where downloads would happen"""
    logger.info("🔍 Testing model loading (CRITICAL TEST)...")
    logger.info("🚨 If models download here, the cache fix FAILED!")
    
    try:
        start_time = time.time()
        
        from marker.models import create_model_dict
        
        logger.info("📥 Creating model dict - watch for download messages...")
        models = create_model_dict()
        
        load_time = time.time() - start_time
        logger.info(f"⏱️ Model loading took: {load_time:.2f} seconds")
        
        if load_time > 60:  # More than 1 minute suggests downloading
            logger.error(f"❌ Model loading took {load_time:.2f}s - likely downloading!")
            logger.error("🚨 CACHE FIX FAILED - Models are being downloaded!")
            return False
        elif load_time > 30:
            logger.warning(f"⚠️ Model loading took {load_time:.2f}s - suspicious...")
        else:
            logger.info(f"✅ Model loading was fast ({load_time:.2f}s) - likely using cache!")
            
        if models:
            logger.info(f"✅ Successfully loaded {len(models)} model components")
            return True
        else:
            logger.error("❌ No models loaded")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_pdf_processing():
    """Test that PDF processing uses cached models"""
    logger.info("🔍 Testing PDF processor initialization...")
    
    try:
        from src.document_processing.enhanced_pdf_processor import MarkerPDFProcessor
        
        start_time = time.time()
        processor = MarkerPDFProcessor(use_llm=False, use_gpu=False)
        init_time = time.time() - start_time
        
        logger.info(f"⏱️ PDF processor init took: {init_time:.2f} seconds")
        
        if init_time > 120:  # More than 2 minutes
            logger.error(f"❌ PDF processor init took {init_time:.2f}s - likely downloading models!")
            return False
        else:
            logger.info(f"✅ PDF processor init was reasonable ({init_time:.2f}s)")
            
        # Get processing stats
        stats = processor.get_processing_stats()
        logger.info(f"📊 Processing stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PDF processor test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all cache tests"""
    logger.info("🚀 STARTING COMPREHENSIVE CACHE FIX TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Environment Variables", test_cache_environment),
        ("Marker Import", test_marker_import), 
        ("Model Loading", test_model_loading),
        ("PDF Processing", test_pdf_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Final results
    logger.info("\n" + "=" * 60)
    logger.info("🏁 FINAL RESULTS")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n📊 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - CACHE FIX WORKING!")
        return True
    else:
        logger.error("🚨 SOME TESTS FAILED - CACHE FIX NOT WORKING!")
        logger.error("💡 Models will still download on every conversion!")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)