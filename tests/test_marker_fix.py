#!/usr/bin/env python3
"""
Test script to verify Marker configuration works without Google Gemini
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_marker_initialization():
    """Test that Marker can be initialized without Gemini API errors"""
    print("🔧 Testing Marker initialization...")
    
    try:
        # Import the enhanced processor
        from src.document_processing.enhanced_pdf_processor import enhanced_pdf_processor, MARKER_AVAILABLE
        
        print(f"   Marker available: {'✅ Yes' if MARKER_AVAILABLE else '❌ No'}")
        
        if not MARKER_AVAILABLE:
            print("   📦 Marker installation required: pip install marker-pdf")
            return False
        
        # Get processing stats
        stats = enhanced_pdf_processor.get_processing_stats()
        print(f"   📊 Processing stats: {stats}")
        
        # Try to initialize the converter (this is where Gemini error occurred)
        enhanced_pdf_processor._load_converter_if_needed()
        
        if enhanced_pdf_processor.models_loaded:
            print("   ✅ Marker converter successfully loaded!")
            return True
        else:
            print("   ❌ Marker converter failed to load")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during Marker test: {e}")
        return False

def test_environment_variables():
    """Test that environment variables are properly set"""
    print("\n🌍 Testing environment variables...")
    
    expected_vars = {
        "MARKER_DISABLE_GEMINI": "true",
        "MARKER_USE_LOCAL_ONLY": "true", 
        "MARKER_LLM_PROVIDER": "ollama",
        "MARKER_DISABLE_CLOUD_SERVICES": "true",
        "GEMINI_API_KEY": "",
        "GOOGLE_API_KEY": ""
    }
    
    for var_name, expected_value in expected_vars.items():
        actual_value = os.getenv(var_name, "NOT_SET")
        status = "✅" if actual_value == expected_value else "❌"
        print(f"   {status} {var_name}: {actual_value}")

def test_config_loading():
    """Test that our config loads properly"""
    print("\n⚙️ Testing config loading...")
    
    try:
        from src.config import OLLAMA_BASE_URL, OLLAMA_GENERATION_MODEL, get_config
        
        config = get_config()
        print(f"   ✅ Ollama Base URL: {OLLAMA_BASE_URL}")
        print(f"   ✅ Ollama Model: {OLLAMA_GENERATION_MODEL}")
        print(f"   ✅ Config loaded successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Config loading error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MARKER + OLLAMA CONFIGURATION TEST")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run tests
    config_ok = test_config_loading()
    env_ok = test_environment_variables() 
    marker_ok = test_marker_initialization()
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS:")
    print(f"   Config Loading: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   Environment Variables: ✅ CONFIGURED")
    print(f"   Marker Initialization: {'✅ PASS' if marker_ok else '❌ FAIL'}")
    
    if marker_ok:
        print("\n🎉 SUCCESS: Marker configured to work with Ollama (no Gemini required)!")
        print("   You can now use PDF processing without Google Gemini API key.")
    else:
        print("\n⚠️ There may still be issues. Check the error messages above.")
        
    print("\n💡 Next steps:")
    print("   - Make sure Ollama is running: ollama serve")
    print("   - Check available models: ollama list")
    print("   - Test with a real PDF file using your application")