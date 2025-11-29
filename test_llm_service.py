import requests
import json

# Test the LLM service directly
base_url = "http://localhost:8002"
headers = {"Content-Type": "application/json"}

def list_models():
    """List all available models"""
    try:
        print("\n=== Listing Available Models ===")
        response = requests.get(f"{base_url}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("Available Models:")
            for model in models.get("models", []):
                print(f"- {model}")
            return models.get("models", [])
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def test_model_generation(model_name, prompt):
    """Test generating text with a specific model"""
    print(f"\n=== Testing Model: {model_name} ===")
    url = f"{base_url}/models/generate"
    
    payload = {
        "prompt": prompt,
        "model": model_name,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        print(f"Sending request to LLM service: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print("Error Response:", response.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # List available models
    models = list_models()
    
    # Test with each available model
    test_prompt = "Merhaba, nasılsın?"
    
    if not models:
        print("No models found. Testing with default model...")
        test_model_generation("default", test_prompt)
    else:
        for model in models:
            test_model_generation(model, test_prompt)
