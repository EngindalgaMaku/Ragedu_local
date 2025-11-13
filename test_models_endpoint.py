#!/usr/bin/env python3
"""
Test the models endpoint to verify restored models are available
"""
import requests
import json

def test_models_endpoint():
    """Test the models endpoint"""
    try:
        response = requests.get('http://localhost:8080/models')
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response JSON: {json.dumps(data, indent=2)}")
            
            # Check if we have the restored models
            if 'models' in data:
                models = data['models']
                print(f"\nModels found: {len(models)}")
                for model in models:
                    print(f"  - {model}")
                    
                # Check for our restored models
                expected_models = [
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile", 
                    "openai/gpt-oss-20b",
                    "qwen/qwen3-32b"
                ]
                
                found_models = 0
                for expected in expected_models:
                    if expected in models:
                        print(f"✅ Found: {expected}")
                        found_models += 1
                    else:
                        print(f"❌ Missing: {expected}")
                
                print(f"\nSummary: {found_models}/{len(expected_models)} expected models found")
                
                if found_models == len(expected_models):
                    print("🎉 All restored models are available!")
                    return True
                else:
                    print("⚠️ Some models are missing")
                    return False
            else:
                print("❌ No 'models' field in response")
                return False
        else:
            print(f"❌ API request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_available_models_endpoint():
    """Test the available models endpoint"""
    try:
        response = requests.get('http://localhost:8080/models/available')
        print(f"\n--- Testing /models/available ---")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response JSON: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Models API Endpoints")
    print("="*50)
    
    success = test_models_endpoint()
    test_available_models_endpoint()
    
    if success:
        print("\n✅ Model restoration successful!")
    else:
        print("\n❌ Model restoration needs attention")