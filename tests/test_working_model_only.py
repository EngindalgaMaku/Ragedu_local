#!/usr/bin/env python3
"""
Test script to verify that the system works correctly with only the confirmed working model.
"""

import requests
import json
from datetime import datetime

def test_available_models():
    """Check what models the service now reports as available"""
    print("🔍 Checking available models from service...")
    
    try:
        response = requests.get("http://localhost:8002/models/available", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Available models endpoint working:")
            print(f"   Groq models: {data.get('groq', [])}")
            print(f"   Ollama models: {data.get('ollama', [])}")
            return data
        else:
            print(f"❌ Available models endpoint failed: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting available models: {str(e)}")
        return None

def test_working_model():
    """Test only the confirmed working model"""
    model_name = "llama-3.1-8b-instant"
    
    payload = {
        "prompt": "Say 'system working correctly' if you can respond.",
        "model": model_name,
        "temperature": 0.1,
        "max_tokens": 20
    }
    
    try:
        print(f"🔍 Testing confirmed working model: {model_name}")
        
        response = requests.post(
            "http://localhost:8002/models/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            model_used = data.get('model_used', 'unknown')
            answer = data.get('response', '').strip()
            print(f"✅ {model_name}: SUCCESS")
            print(f"   Model used: {model_used}")
            print(f"   Response: '{answer}'")
            return True
        else:
            print(f"❌ {model_name}: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name}: Error - {str(e)}")
        return False

def test_api_gateway_models():
    """Test what models the API Gateway reports"""
    print("🔍 Checking models from API Gateway...")
    
    try:
        response = requests.get("http://localhost:8080/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Gateway models endpoint working:")
            models = data.get('models', [])
            for model in models:
                if model.get('provider') == 'groq':
                    print(f"   - {model.get('id', 'unknown')} ({model.get('name', 'unknown')})")
            return True
        else:
            print(f"❌ API Gateway models endpoint failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting models from API Gateway: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("WORKING MODEL VERIFICATION TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Test 1: Check what models the service reports as available
    print("1. CHECKING SERVICE AVAILABLE MODELS")
    print("-" * 40)
    available_models = test_available_models()
    print()
    
    # Test 2: Test the confirmed working model
    print("2. TESTING CONFIRMED WORKING MODEL")
    print("-" * 40)
    working_model_ok = test_working_model()
    print()
    
    # Test 3: Check API Gateway
    print("3. CHECKING API GATEWAY MODELS")
    print("-" * 40)
    api_gateway_ok = test_api_gateway_models()
    print()
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if working_model_ok:
        print("✅ The working model (llama-3.1-8b-instant) is functioning correctly")
    else:
        print("❌ The working model is NOT functioning correctly")
    
    if api_gateway_ok:
        print("✅ API Gateway is responding to model requests")
    else:
        print("❌ API Gateway is NOT responding correctly")
    
    if available_models:
        groq_models = available_models.get('groq', [])
        if len(groq_models) == 1 and 'llama-3.1-8b-instant' in groq_models:
            print("✅ Model Inference Service configuration updated correctly (restart may be needed)")
        else:
            print(f"⚠️ Model Inference Service still reports old models: {groq_models}")
            print("   Please restart the Model Inference Service to pick up configuration changes")
    
    print()
    print("🔄 Note: If you made configuration changes, restart the Model Inference Service:")
    print("   python services/model_inference_service/main.py")

if __name__ == "__main__":
    main()