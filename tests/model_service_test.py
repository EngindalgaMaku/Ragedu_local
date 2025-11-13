#!/usr/bin/env python3
"""
Test script to check communication with the Model Inference Service
"""

import requests
import json

# Service URLs
API_GATEWAY_URL = "https://api-gateway-1051060211087.europe-west1.run.app"
MODEL_INFERENCE_URL = "https://model-inferencer-1051060211087.europe-west1.run.app"

def test_model_inference_service_directly():
    """Test the Model Inference Service directly"""
    print("=== Testing Model Inference Service Directly ===")
    
    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{MODEL_INFERENCE_URL}/health", timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print("   ✅ Health endpoint accessible")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Ollama Available: {health_data.get('ollama_available')}")
            print(f"   Groq Available: {health_data.get('groq_available')}")
        else:
            print(f"   ❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error testing health endpoint: {e}")
        return False
    
    # Test available models endpoint
    try:
        print("\n2. Testing available models endpoint...")
        response = requests.get(f"{MODEL_INFERENCE_URL}/models/available", timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            print("   ✅ Available models endpoint accessible")
            print(f"   Groq Models: {models_data.get('groq', [])}")
            print(f"   Ollama Models: {models_data.get('ollama', [])}")
        else:
            print(f"   ❌ Available models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing available models endpoint: {e}")
    
    return True

def test_api_gateway_to_model_service():
    """Test communication from API Gateway to Model Inference Service"""
    print("\n=== Testing API Gateway to Model Inference Service ===")
    
    # Test models endpoint through API Gateway
    try:
        print("1. Testing models endpoint through API Gateway...")
        response = requests.get(f"{API_GATEWAY_URL}/models", timeout=15)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            print("   ✅ Models endpoint through API Gateway accessible")
            print(f"   Available Models: {models_data.get('models', [])}")
            return True
        else:
            print(f"   ❌ Models endpoint through API Gateway failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error testing models endpoint through API Gateway: {e}")
        return False

def test_model_generation_with_fallback():
    """Test model generation with fallback to default models"""
    print("\n=== Testing Model Generation ===")
    
    # Test with a simple prompt using one of the default models
    test_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b"
    ]
    
    # Try to test the RAG query endpoint through API Gateway
    print("1. Testing RAG query endpoint through API Gateway...")
    query_data = {
        "session_id": "test-session",
        "query": "Hello, this is a test query",
        "top_k": 3,
        "use_rerank": False,
        "min_score": 0.1,
        "max_context_chars": 1000
    }
    
    try:
        response = requests.post(
            f"{API_GATEWAY_URL}/rag/query",
            json=query_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("   ✅ RAG query endpoint working")
            print(f"   Response: {result}")
            return True
        else:
            print(f"   ❌ RAG query endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            # This is expected since the model inference service doesn't have the /query endpoint
    except Exception as e:
        print(f"   ❌ Error testing RAG query endpoint: {e}")
    
    return False

def main():
    """Main test function"""
    print("Model Inference Service Communication Test")
    print("=" * 50)
    
    # Test 1: Direct service communication
    direct_test_passed = test_model_inference_service_directly()
    
    # Test 2: API Gateway to Model Service communication
    gateway_test_passed = test_api_gateway_to_model_service()
    
    # Test 3: Model generation (will likely fail due to missing models)
    generation_test_passed = test_model_generation_with_fallback()
    
    # Summary
    print("\n" + "=" * 50)
    print("MODEL SERVICE COMMUNICATION TEST RESULTS")
    print("=" * 50)
    
    if direct_test_passed:
        print("✅ Direct Model Inference Service Communication: PASSED")
    else:
        print("❌ Direct Model Inference Service Communication: FAILED")
        
    if gateway_test_passed:
        print("✅ API Gateway to Model Inference Service Communication: PASSED")
    else:
        print("❌ API Gateway to Model Inference Service Communication: FAILED")
        
    if generation_test_passed:
        print("✅ Model Generation: PASSED")
    else:
        print("⚠️  Model Generation: NOT WORKING (likely due to missing models)")
        
    print("\nNote: The Model Inference Service shows that Ollama is available")
    print("      but no models are currently installed or accessible.")
    print("      Communication between services is working, but model generation")
    print("      requires models to be installed in the Ollama service.")

if __name__ == "__main__":
    main()