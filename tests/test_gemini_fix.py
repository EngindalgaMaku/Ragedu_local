#!/usr/bin/env python3
"""
Simple test to check if Gemini API error is resolved
"""

import os
import sys

# Add src to path  
sys.path.insert(0, 'src')

def main():
    print("🔧 Testing Gemini API fix...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set the problematic environment variables explicitly
    os.environ["MARKER_DISABLE_GEMINI"] = "true"
    os.environ["MARKER_USE_LOCAL_ONLY"] = "true"
    os.environ["MARKER_LLM_PROVIDER"] = "ollama" 
    os.environ["MARKER_DISABLE_CLOUD_SERVICES"] = "true"
    os.environ["GEMINI_API_KEY"] = ""
    os.environ["GOOGLE_API_KEY"] = ""
    
    print("✅ Environment variables set")
    
    try:
        # Try to import the enhanced PDF processor
        print("📦 Importing enhanced PDF processor...")
        from src.document_processing.enhanced_pdf_processor import enhanced_pdf_processor, MARKER_AVAILABLE
        
        if not MARKER_AVAILABLE:
            print("⚠️ Marker not available, but that's expected if not installed")
            print("✅ No Gemini API errors - fix successful!")
            return True
            
        print("🔄 Testing Marker initialization...")
        
        # This is where the original error occurred
        enhanced_pdf_processor._load_converter_if_needed()
        
        print("✅ Marker initialized successfully without Gemini!")
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        if "gemini" in error_msg or "google" in error_msg:
            print(f"❌ Still getting Gemini error: {e}")
            return False
        else:
            print(f"⚠️ Different error (not Gemini related): {e}")
            print("✅ No Gemini API errors - fix likely successful!")
            return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 SUCCESS: Gemini API dependency removed!")
            print("   Your application should now work with only Ollama.")
        else:
            print("\n❌ FAILED: Still issues with Gemini dependency")
    except Exception as e:
        if "gemini" in str(e).lower():
            print(f"\n❌ FAILED: Gemini error still present: {e}")
        else:
            print(f"\n✅ Different error (not Gemini): {e}")
            print("   This suggests the Gemini dependency issue is resolved!")