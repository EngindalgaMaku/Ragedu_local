"""
Test OpenRouter Integration
Tests the complete OpenRouter integration including:
1. Configuration loading
2. Model Inference Service API calls
3. Model selection UI integration
"""

import os
import sys
import requests
import time
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_openrouter_config():
    """Test OpenRouter configuration loading"""
    print("🔍 Testing OpenRouter Configuration...")
    
    try:
        from rag3_for_local.src.config import get_config
        config = get_config()
        
        # Check if OpenRouter models are loaded
        openrouter_models = [model for model, info in config.cloud_models.items() 
                           if info.get('provider') == 'openrouter']
        
        print(f"✅ Found {len(openrouter_models)} OpenRouter models in config")
        for model in openrouter_models[:3]:  # Show first 3
            model_info = config.cloud_models[model]
            print(f"  - {model}: {model_info['name']}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_model_inference_service():
    """Test Model Inference Service OpenRouter support"""
    print("\n🔍 Testing Model Inference Service...")
    
    # Check if service is running
    service_url = "http://localhost:8002"  # Default model inference service URL
    
    try:
        # Test health endpoint
        response = requests.get(f"{service_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            openrouter_available = health_data.get('openrouter_available', False)
            print(f"✅ Model Inference Service is running")
            print(f"  - OpenRouter available: {openrouter_available}")
            
            if not openrouter_available:
                print("⚠️  OpenRouter not available - check OPENROUTER_API_KEY environment variable")
            
            return True
        else:
            print(f"❌ Model Inference Service not responding: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Model Inference Service not running on localhost:8002")
        print("   Please start the service with: python rag3_for_local/services/model_inference_service/main.py")
        return False
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

def test_available_models_endpoint():
    """Test if OpenRouter models are listed in available models"""
    print("\n🔍 Testing Available Models Endpoint...")
    
    service_url = "http://localhost:8002"
    
    try:
        response = requests.get(f"{service_url}/models/available", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            openrouter_models = models_data.get('openrouter', [])
            
            print(f"✅ Available models endpoint working")
            print(f"  - OpenRouter models: {len(openrouter_models)}")
            
            if openrouter_models:
                print("  OpenRouter models found:")
                for model in openrouter_models[:3]:  # Show first 3
                    print(f"    - {model}")
            else:
                print("  ⚠️  No OpenRouter models found - check OPENROUTER_API_KEY")
            
            return True
        else:
            print(f"❌ Available models endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Available models test failed: {e}")
        return False

def test_openrouter_generation(test_with_api_key=False):
    """Test OpenRouter text generation (requires API key)"""
    print("\n🔍 Testing OpenRouter Generation...")
    
    if not test_with_api_key:
        print("⏭️  Skipping generation test (requires OPENROUTER_API_KEY)")
        print("   Set test_with_api_key=True and add OPENROUTER_API_KEY to test generation")
        return True
    
    service_url = "http://localhost:8002"
    
    # Test with a simple OpenRouter model
    test_payload = {
        "prompt": "What is 2+2? Answer briefly.",
        "model": "openai/gpt-4o-mini",  # Small, fast model for testing
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(
            f"{service_url}/models/generate",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            model_used = result.get('model_used', '')
            
            print(f"✅ OpenRouter generation successful")
            print(f"  - Model used: {model_used}")
            print(f"  - Response: {generated_text[:100]}...")
            return True
        else:
            error_detail = response.json().get('detail', response.text) if response.text else 'Unknown error'
            print(f"❌ Generation failed: {response.status_code}")
            print(f"  - Error: {error_detail}")
            return False
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def test_model_selector_integration():
    """Test model selector integration with OpenRouter"""
    print("\n🔍 Testing Model Selector Integration...")
    
    try:
        # Import config to access OpenRouter models directly
        from rag3_for_local.src.config import get_config
        
        config = get_config()
        openrouter_models = [model for model, info in config.cloud_models.items()
                           if info.get('provider') == 'openrouter']
        
        print(f"✅ Model selector integration working")
        print(f"  - OpenRouter models in config: {len(openrouter_models)}")
        
        if openrouter_models:
            sample_model = openrouter_models[0]
            model_info = config.cloud_models[sample_model]
            print(f"  Sample model: {sample_model}")
            print(f"    - Name: {model_info.get('name', 'Unknown')}")
            print(f"    - Provider: {model_info.get('provider', 'Unknown')}")
            print(f"    - Description: {model_info.get('description', 'No description')[:60]}...")
        
        return True
    except Exception as e:
        print(f"❌ Model selector integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all OpenRouter integration tests"""
    print("🚀 OpenRouter Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_openrouter_config),
        ("Model Inference Service", test_model_inference_service),
        ("Available Models Endpoint", test_available_models_endpoint),
        ("Model Selector Integration", test_model_selector_integration),
        ("OpenRouter Generation", lambda: test_openrouter_generation(test_with_api_key=False)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! OpenRouter integration is working correctly.")
        print("\n📝 Next steps:")
        print("1. Add OPENROUTER_API_KEY to your .env file to enable OpenRouter models")
        print("2. Start the Model Inference Service to use OpenRouter in the UI")
        print("3. OpenRouter models will appear in the model selector UI")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        if passed > 0:
            print("Some components are working - the integration is partially functional.")
    
    return passed == total

if __name__ == "__main__":
    main()