#!/usr/bin/env python3
"""
Test script to verify Model Inference Service integration.
"""

import requests
import json
import os
import sys

# Add src to path to import our modules
sys.path.append('src')

from config import get_model_inference_url

def test_model_inference_service_health():
    """Test health endpoint"""
    try:
        model_inference_url = get_model_inference_url()
        print(f"🔗 Testing Model Inference Service at: {model_inference_url}")
        
        response = requests.get(f"{model_inference_url}/health", timeout=5)
        print(f"✅ Health check status: {response.status_code}")
        print(f"📊 Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_inference_service_generate():
    """Test generation endpoint"""
    try:
        model_inference_url = get_model_inference_url()
        
        request_data = {
            "prompt": "System: Sen yardımcı bir AI asistanısın.\n\nUser: Merhaba, nasılsın?",
            "model": "llama-3.1-8b-instant", 
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        print(f"🚀 Testing generation with request: {json.dumps(request_data, indent=2)}")
        
        response = requests.post(
            f"{model_inference_url}/models/generate",
            json=request_data,
            timeout=30
        )
        
        print(f"✅ Generation status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"📝 Generated response: {result.get('response', 'No response')}")
            print(f"🤖 Model used: {result.get('model_used', 'Unknown')}")
            return True
        else:
            print(f"❌ Generation failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def test_app_logic_integration():
    """Test app_logic integration"""
    try:
        from app_logic import call_model_inference_service
        
        print(f"🔄 Testing app_logic integration...")
        
        result = call_model_inference_service(
            prompt="System: Sen yardımcı bir AI asistanısın.\n\nUser: Test cevabı ver.",
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=50
        )
        
        if result and result.strip():
            print(f"✅ App logic integration successful!")
            print(f"📝 Response: {result}")
            return True
        else:
            print(f"❌ App logic integration failed: Empty response")
            return False
            
    except Exception as e:
        print(f"❌ App logic integration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Model Inference Service Integration Tests")
    print("=" * 50)
    
    # Set environment variable for testing
    if not os.getenv('MODEL_INFERENCE_URL'):
        os.environ['MODEL_INFERENCE_URL'] = 'http://model-inferencer:8002'
    
    tests = [
        ("Health Check", test_model_inference_service_health),
        ("Generation Test", test_model_inference_service_generate),
        ("App Logic Integration", test_app_logic_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
        print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
    
    print(f"\n📊 Test Summary:")
    print("=" * 50)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model Inference Service integration is working!")
        return True
    else:
        print("⚠️ Some tests failed. Check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)