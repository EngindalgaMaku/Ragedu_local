#!/usr/bin/env python3
"""
Test script to check if the Model Inference Service is working with all Groq models.
"""

import requests
import json
import sys
from datetime import datetime

def test_model_via_service(model_name: str, service_url: str = "http://localhost:8002"):
    """Test a model through the Model Inference Service"""
    
    payload = {
        "prompt": "Say 'test successful' if you can respond.",
        "model": model_name,
        "temperature": 0.1,
        "max_tokens": 20
    }
    
    try:
        print(f"🔍 Testing {model_name} via Model Inference Service...")
        
        response = requests.post(
            f"{service_url}/models/generate",
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
            
    except requests.ConnectionError:
        print(f"❌ {model_name}: Cannot connect to service at {service_url}")
        return False
    except requests.Timeout:
        print(f"❌ {model_name}: Timeout")
        return False
    except Exception as e:
        print(f"❌ {model_name}: Error - {str(e)}")
        return False

def check_service_health(service_url: str = "http://localhost:8002"):
    """Check if the Model Inference Service is running"""
    try:
        response = requests.get(f"{service_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model Inference Service is running")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Ollama available: {data.get('ollama_available', False)}")
            print(f"   Groq available: {data.get('groq_available', False)}")
            print(f"   Ollama host: {data.get('ollama_host', 'unknown')}")
            return True
        else:
            print(f"❌ Service health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach Model Inference Service: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("MODEL INFERENCE SERVICE TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    service_url = "http://localhost:8002"
    
    # Check service health first
    if not check_service_health(service_url):
        print("\n❌ Service is not available. Please start the Model Inference Service first.")
        print("   Run: python services/model_inference_service/main.py")
        sys.exit(1)
    
    print()
    
    # Test all Groq models
    models_to_test = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile", 
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b"
    ]
    
    working_models = []
    failed_models = []
    
    for model in models_to_test:
        if test_model_via_service(model, service_url):
            working_models.append(model)
        else:
            failed_models.append(model)
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY - SERVICE TEST RESULTS")
    print("=" * 60)
    
    print(f"✅ Working models through service ({len(working_models)}):")
    for model in working_models:
        print(f"   - {model}")
    
    print(f"\n❌ Failed models through service ({len(failed_models)}):")
    for model in failed_models:
        print(f"   - {model}")
    
    if failed_models:
        print(f"\n💡 Recommendation: Remove the {len(failed_models)} failed models from configuration")
        print("   Files to update:")
        print("   - services/model_inference_service/main.py")
        print("   - src/app_logic.py") 
        print("   - src/utils/model_selector.py")
    else:
        print(f"\n🎉 All {len(working_models)} models are working correctly!")

if __name__ == "__main__":
    main()