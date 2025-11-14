#!/usr/bin/env python3
"""
Test Groq API directly to see if problematic models actually work
"""
import os
import requests
import json

def test_groq_model(model_name):
    """Test a specific Groq model directly via API"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ GROQ_API_KEY environment variable not set")
        return False
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Explain the importance of fast language models in one sentence."
            }
        ]
    }
    
    try:
        print(f"🧪 Testing {model_name}...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ SUCCESS: {content[:100]}...")
            return True
        else:
            error_details = response.text
            print(f"❌ FAILED: {error_details}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    """Test all the 'problematic' models"""
    models_to_test = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b", 
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant"  # Control - this one definitely works
    ]
    
    print("🚀 Testing Groq models directly...")
    results = {}
    
    for model in models_to_test:
        print(f"\n{'='*50}")
        results[model] = test_groq_model(model)
        print(f"{'='*50}")
    
    print(f"\n📊 SUMMARY:")
    for model, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"  {model}: {status}")
    
    # Count working models
    working_count = sum(results.values())
    total_count = len(results)
    print(f"\n📈 {working_count}/{total_count} models are working")

if __name__ == "__main__":
    main()