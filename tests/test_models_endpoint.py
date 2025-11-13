#!/usr/bin/env python3
"""
Test script for the new /models/list endpoint.
This script tests the API endpoint that returns available language models.
"""

import requests
import json

# API configuration
API_URL = "http://localhost:8000"
ENDPOINT = "/models/list"

def test_models_list_endpoint():
    """Test the GET /models/list endpoint."""
    try:
        print(f"Testing {API_URL}{ENDPOINT}")
        print("=" * 50)
        
        # Make the API call
        response = requests.get(f"{API_URL}{ENDPOINT}")
        
        # Check response status
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            # Parse JSON response
            data = response.json()
            print("✅ Success! Response received:")
            print(f"Raw Response: {json.dumps(data, indent=2)}")
            
            # Validate response structure
            if "models" in data and isinstance(data["models"], list):
                models = data["models"]
                print(f"\n📋 Found {len(models)} models:")
                
                # Group models by type for better display
                ollama_models = []
                groq_models = []
                other_models = []
                
                for model in models:
                    if any(groq_model in model for groq_model in ["llama-3.1", "mixtral", "gemma"]):
                        groq_models.append(model)
                    elif ":" in model and not model.endswith("-cloud"):
                        ollama_models.append(model)
                    else:
                        other_models.append(model)
                
                # Display models by category
                if ollama_models:
                    print(f"\n🏠 Local Ollama Models ({len(ollama_models)}):")
                    for model in ollama_models:
                        print(f"  - {model}")
                
                if groq_models:
                    print(f"\n🌐 Groq Cloud Models ({len(groq_models)}):")
                    for model in groq_models:
                        print(f"  - {model}")
                        
                if other_models:
                    print(f"\n🔧 Other Models ({len(other_models)}):")
                    for model in other_models:
                        print(f"  - {model}")
                
                print(f"\n✅ Test PASSED - Endpoint returned {len(models)} models")
                
                # Additional validation checks
                if len(models) == 0:
                    print("⚠️  Warning: No models found. This might indicate:")
                    print("  - Ollama is not running")
                    print("  - No cloud API keys configured")
                    print("  - Configuration issues")
                
            else:
                print("❌ Test FAILED - Invalid response structure")
                print("Expected: {'models': [list of strings]}")
                
        else:
            print(f"❌ Test FAILED - HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error - Is the API server running?")
        print("Start the server with: uvicorn src.api.main:app --reload")
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")

def test_api_health():
    """Test if the API server is running by checking the health endpoint."""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print(f"⚠️  API server responded with status {response.status_code}")
            return False
    except:
        print("❌ API server is not accessible")
        return False

if __name__ == "__main__":
    print("Testing RAG3 Models List Endpoint")
    print("=" * 50)
    
    # First check if API is running
    if test_api_health():
        print()
        test_models_list_endpoint()
    else:
        print("\nPlease start the API server first:")
        print("uvicorn src.api.main:app --reload")