#!/usr/bin/env python3
"""
Investigation script to check Groq API capabilities
"""

import requests
import os

def test_groq_api():
    """Test Groq API endpoints to see what's available"""
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable not set; skipping tests.")
        return
    
    print("🔍 GROQ API INVESTIGATION")
    print("=" * 50)
    
    # Test 1: Try the embedding endpoint I implemented
    print("\n1. Testing Groq Embedding Endpoint (as implemented):")
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "nomic-embed-text-v1.5",
                "input": "test text"
            },
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test 2: Try chat completions (known to work)
    print("\n2. Testing Groq Chat Completions (known endpoint):")
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test 3: Check available models
    print("\n3. Testing Groq Models List:")
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Available models: {len(models.get('data', []))}")
            for model in models.get('data', [])[:5]:  # Show first 5
                print(f"   - {model.get('id', 'unknown')}")
            
            # Check if any embedding models exist
            embedding_models = [m for m in models.get('data', []) if 'embed' in m.get('id', '').lower()]
            print(f"   Embedding models found: {len(embedding_models)}")
            for model in embedding_models:
                print(f"   - EMBEDDING: {model.get('id', 'unknown')}")
        else:
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test 4: Try different embedding endpoint patterns
    print("\n4. Testing Alternative Embedding Endpoints:")
    
    endpoints_to_try = [
        "https://api.groq.com/v1/embeddings",
        "https://api.groq.com/openai/v1/embeddings",
        "https://api.groq.com/embeddings"
    ]
    
    for endpoint in endpoints_to_try:
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "nomic-embed-text-v1.5",
                    "input": "test"
                },
                timeout=5
            )
            print(f"   {endpoint}: {response.status_code} - {response.text[:100]}")
        except Exception as e:
            print(f"   {endpoint}: Error - {str(e)}")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   Based on these tests, we can determine if Groq actually supports embeddings")
    print(f"   or if we need to use a different fast embedding provider.")

if __name__ == "__main__":
    test_groq_api()