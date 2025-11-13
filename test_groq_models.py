#!/usr/bin/env python3
"""
Test script to verify which Groq models are actually available and working.
"""

import os
import sys
import requests
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_groq_model(model_name):
    """Test a specific Groq model with a simple request"""
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return False
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Say 'test successful' if you can respond."
            }
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        print(f"🔍 Testing model: {model_name}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ {model_name}: Working (Response: '{content.strip()}')")
            return True
        elif response.status_code == 404:
            print(f"❌ {model_name}: Model not found (404)")
            return False
        else:
            print(f"❌ {model_name}: HTTP {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name}: Error - {str(e)}")
        return False

def main():
    """Test all Groq models we've been trying to use"""
    
    print("=" * 60)
    print("GROQ MODEL AVAILABILITY TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # List of models to test
    models_to_test = [
        "llama-3.1-8b-instant",  # This one we know works
        "llama-3.3-70b-versatile",  # This one we know fails
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b"
    ]
    
    working_models = []
    failed_models = []
    
    for model in models_to_test:
        if test_groq_model(model):
            working_models.append(model)
        else:
            failed_models.append(model)
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Working models ({len(working_models)}):")
    for model in working_models:
        print(f"   - {model}")
    
    print(f"\n❌ Failed models ({len(failed_models)}):")
    for model in failed_models:
        print(f"   - {model}")
    
    print(f"\nRecommendation: Use only the {len(working_models)} working models in your configuration.")
    
    # Also check what models Groq actually supports
    print("\n" + "=" * 60)
    print("CHECKING GROQ MODEL LIST")
    print("=" * 60)
    
    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        models_url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {groq_api_key}"}
        
        response = requests.get(models_url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            available_models = [model['id'] for model in data.get('data', [])]
            print(f"✅ Groq API reports {len(available_models)} available models:")
            for model in sorted(available_models):
                print(f"   - {model}")
        else:
            print(f"❌ Failed to get model list: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting model list: {str(e)}")

if __name__ == "__main__":
    main()