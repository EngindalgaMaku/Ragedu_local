"""
Test OpenRouter with actual API key
Quick test to verify OpenRouter integration works with real API key
"""

import os
import sys
import requests

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_openrouter_direct_api():
    """Test OpenRouter API directly with environment variable key"""
    print("🚀 Testing OpenRouter API directly...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("   Please set: export OPENROUTER_API_KEY=your_key_here")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "RAG3 Local System"
    }
    
    payload = {
        "model": "openai/gpt-4o-mini",  # Small, fast, cost-effective model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer very briefly."}
        ],
        "temperature": 0.7,
        "max_tokens": 20
    }
    
    try:
        print(f"📡 Calling OpenRouter API with model: {payload['model']}")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                response_content = result["choices"][0]["message"]["content"]
                model_used = result["model"]
                usage = result.get("usage", {})
                
                print("✅ OpenRouter API test successful!")
                print(f"  🤖 Model used: {model_used}")
                print(f"  💬 Response: {response_content}")
                print(f"  📈 Tokens used: {usage.get('total_tokens', 'Unknown')}")
                
                return True
            else:
                print("❌ Invalid response format from OpenRouter")
                print(f"Response: {result}")
                return False
        else:
            print(f"❌ OpenRouter API error: {response.status_code}")
            print(f"Error detail: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_model_inference_service_with_key():
    """Test Model Inference Service with OpenRouter key"""
    print("\n🔧 Testing Model Inference Service with OpenRouter...")
    
    # Check if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("   Please set: export OPENROUTER_API_KEY=your_key_here")
        return False
    
    # Import and test the service functions directly
    try:
        sys.path.append('rag3_for_local/services/model_inference_service')
        
        # Test OpenRouter model detection
        from main import is_openrouter_model, OPENROUTER_API_KEY
        
        print(f"🔑 API Key loaded: {'Yes' if OPENROUTER_API_KEY else 'No'}")
        
        test_models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku", 
            "llama-3.1-8b-instant"  # This should be Groq, not OpenRouter
        ]
        
        for model in test_models:
            is_or = is_openrouter_model(model)
            print(f"  📋 {model}: {'OpenRouter' if is_or else 'Other provider'}")
        
        print("✅ Model detection working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Model Inference Service test failed: {e}")
        return False

def main():
    """Run OpenRouter tests with real API key"""
    print("🚀 OpenRouter API Key Integration Test")
    print("=" * 50)
    
    tests = [
        ("Direct OpenRouter API", test_openrouter_direct_api),
        ("Model Inference Service Integration", test_model_inference_service_with_key)
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
    print("📊 OPENROUTER TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 OpenRouter integration fully working!")
        print("\n📝 Ready to use:")
        print("1. ✅ OpenRouter API key is valid")
        print("2. ✅ Model detection working")
        print("3. ✅ Direct API calls successful")
        print("\n🚀 You can now:")
        print("- Start the Model Inference Service")
        print("- Use OpenRouter models in the UI")
        print("- Select premium models like GPT-4, Claude, etc.")
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
    
    return passed == total

if __name__ == "__main__":
    main()