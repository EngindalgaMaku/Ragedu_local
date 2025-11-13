#!/usr/bin/env python3
"""
Test script to verify PDF service imports work correctly
"""
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pdf_service_imports():
    """Test that PDF service can import its dependencies without errors"""
    print("🔧 Testing PDF Processing Service imports...")
    
    try:
        # Add the PDF service directory to the Python path
        pdf_service_dir = os.path.join(os.getcwd(), 'services', 'pdf_processing_service')
        sys.path.insert(0, pdf_service_dir)
        
        # Test importing model_cache_manager from the PDF service
        print("📦 Testing model_cache_manager import...")
        from model_cache_manager import ModelCacheManager, get_model_cache_manager
        print("✅ model_cache_manager imported successfully")
        
        # Test creating cache manager instance
        print("🔧 Testing ModelCacheManager instantiation...")
        cache_manager = get_model_cache_manager()
        print("✅ ModelCacheManager created successfully")
        
        # Test getting cache stats
        print("📊 Testing cache stats...")
        stats = cache_manager.get_cache_stats()
        print(f"✅ Cache stats retrieved: {stats['cache_directory']}")
        print(f"   - Cloud environment: {stats['is_cloud_environment']}")
        print(f"   - Cloud storage available: {stats['cloud_storage_available']}")
        
        print("\n🎉 All PDF service imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        # Clean up sys.path
        if pdf_service_dir in sys.path:
            sys.path.remove(pdf_service_dir)

if __name__ == "__main__":
    success = test_pdf_service_imports()
    sys.exit(0 if success else 1)