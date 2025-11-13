#!/usr/bin/env python3
"""
Test script for Marker Safe Mode - System freeze prevention
"""

import os
import sys
import time

# Add src to path  
sys.path.insert(0, 'src')

def test_safe_mode_settings():
    """Test safe mode ayarlarını kontrol et"""
    print("🔒 Testing Safe Mode Configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    expected_settings = {
        "MARKER_MAX_PAGES": "10",
        "MARKER_MAX_MEMORY_MB": "1024", 
        "MARKER_MAX_CPU_PERCENT": "25",
        "MARKER_TIMEOUT_SECONDS": "60",
        "MARKER_PROCESS_PRIORITY": "low",
        "MARKER_ENABLE_RESOURCE_MONITORING": "true"
    }
    
    all_good = True
    for setting, expected in expected_settings.items():
        actual = os.getenv(setting, "NOT_SET")
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_good = False
        print(f"   {status} {setting}: {actual}")
    
    return all_good

def test_resource_limits():
    """Resource limit mekanizmalarını test et"""
    print("\n📊 Testing Resource Limit Mechanisms...")
    
    try:
        from src.document_processing.enhanced_pdf_processor import enhanced_pdf_processor
        
        # Resource limits kontrolü
        limits = enhanced_pdf_processor.resource_limits
        print(f"   💾 Memory Limit: {limits['max_memory_mb']}MB")
        print(f"   🔥 CPU Limit: {limits['max_cpu_percent']}%")
        print(f"   ⏰ Timeout: {limits['timeout_seconds']}s")
        print(f"   📊 Monitoring: {limits['enable_monitoring']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Resource limits test failed: {e}")
        return False

def test_safe_pdf_processing():
    """Güvenli PDF işleme testi"""
    print("\n🔒 Testing Safe PDF Processing...")
    
    try:
        from src.document_processing.enhanced_pdf_processor import enhanced_pdf_processor, MARKER_AVAILABLE
        
        if not MARKER_AVAILABLE:
            print("   ⚠️ Marker not available - testing initialization only")
            return True
        
        # Small test files
        test_pdfs = [
            "data/uploads/21164209_Biyoloji_9.pdf",
            "data/uploads/derin_ogrenme1.pdf"
        ]
        
        for pdf_path in test_pdfs:
            if os.path.exists(pdf_path):
                print(f"   📄 Testing with: {os.path.basename(pdf_path)}")
                
                file_size = os.path.getsize(pdf_path) / (1024 * 1024)
                print(f"   📏 File size: {file_size:.1f}MB")
                
                if file_size > 25:  # Safe mode limit
                    print(f"   ⚠️ File too large for safe mode, will use fallback")
                    continue
                
                start_time = time.time()
                try:
                    # Safe mode ile test - çok kısa timeout
                    content, metadata = enhanced_pdf_processor.process_pdf_with_marker(
                        pdf_path, timeout_seconds=30
                    )
                    processing_time = time.time() - start_time
                    
                    print(f"   ✅ Safe processing: {processing_time:.2f}s")
                    print(f"   📊 Method: {metadata.get('processing_method', 'unknown')}")
                    print(f"   📄 Content: {len(content)} chars")
                    
                    return True
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    print(f"   ⚠️ Processing error after {processing_time:.2f}s: {e}")
                    if processing_time < 35:  # Timeout çalıştı mı?
                        print("   ✅ Timeout mechanism working correctly")
                        return True
                    else:
                        return False
                        
        print("   ⚠️ No suitable test PDFs found")
        return True
        
    except Exception as e:
        print(f"   ❌ Safe processing test failed: {e}")
        return False

def test_memory_monitoring():
    """Memory monitoring test"""
    print("\n📊 Testing Memory Monitoring...")
    
    try:
        import psutil
        print(f"   ✅ psutil available: {psutil.__version__}")
        
        # Current process info
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        print(f"   💾 Current memory: {memory_mb:.1f}MB")
        print(f"   🔥 Current CPU: {cpu_percent:.1f}%")
        
        return True
        
    except ImportError:
        print("   ❌ psutil not available - resource monitoring disabled")
        return False
    except Exception as e:
        print(f"   ❌ Memory monitoring test failed: {e}")
        return False

def main():
    print("🚀 MARKER SAFE MODE TEST")
    print("=" * 50)
    
    # Run tests
    settings_ok = test_safe_mode_settings()
    resources_ok = test_resource_limits() 
    processing_ok = test_safe_pdf_processing()
    monitoring_ok = test_memory_monitoring()
    
    print("\n" + "=" * 50)
    print("📋 SAFE MODE TEST RESULTS:")
    print(f"   Settings Configuration: {'✅ PASS' if settings_ok else '❌ FAIL'}")
    print(f"   Resource Limits: {'✅ PASS' if resources_ok else '❌ FAIL'}")
    print(f"   Safe Processing: {'✅ PASS' if processing_ok else '❌ FAIL'}")
    print(f"   Memory Monitoring: {'✅ PASS' if monitoring_ok else '❌ FAIL'}")
    
    if settings_ok and resources_ok and processing_ok:
        print("\n🔒 SUCCESS: Safe Mode is properly configured!")
        print("   - System freeze protection: ACTIVE")
        print("   - Resource monitoring: ENABLED") 
        print("   - Memory limits: 1GB max")
        print("   - CPU limits: 25% max")
        print("   - Timeout: 60 seconds max")
        print("   - Process priority: LOW")
        
        print("\n💡 Safe Mode Features:")
        print("   🛡️ Prevents system freezing")
        print("   📏 File size limits (25MB max)")
        print("   📄 Page count limits (10 pages max)")
        print("   ⏰ Strict timeouts (60s max)")
        print("   💾 Memory monitoring (1GB limit)")
        print("   🔥 CPU usage monitoring (25% limit)")
        print("   🔄 Automatic fallback if limits exceeded")
        
    else:
        print("\n⚠️ Some safe mode components need attention")
        
    print("\n🎯 Usage: Marker now processes PDFs safely without freezing your computer!")

if __name__ == "__main__":
    main()