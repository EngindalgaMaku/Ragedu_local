#!/usr/bin/env python3
"""
Test script for the new model selection feature in RAG query endpoint.
"""
import requests
import json
from typing import Optional

API_BASE_URL = "http://localhost:8000"

def test_rag_query_with_model(session_id: str, query: str, model: Optional[str] = None):
    """Test the RAG query endpoint with optional model parameter."""
    url = f"{API_BASE_URL}/rag/query"
    
    payload = {
        "session_id": session_id,
        "query": query,
        "top_k": 3,
        "use_rerank": False,
        "min_score": 0.1,
        "max_context_chars": 4000
    }
    
    if model:
        payload["model"] = model
    
    headers = {"Content-Type": "application/json"}
    
    try:
        print(f"🔍 Testing RAG query with model: {model or 'default'}")
        print(f"📝 Query: {query}")
        print(f"📊 Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"📈 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response:")
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Sources count: {len(result.get('sources', []))}")
            print(f"   Answer preview: {result.get('answer', '')[:200]}...")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Error details: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        return False

def test_available_models():
    """Test the list available models endpoint."""
    url = f"{API_BASE_URL}/models/list"
    
    try:
        print(f"🔍 Testing available models endpoint...")
        response = requests.get(url)
        
        print(f"📈 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            models = result.get("models", [])
            print(f"✅ Available models ({len(models)}):")
            for model in models:
                print(f"   - {model}")
            return models
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Error details: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        return []

def test_list_sessions():
    """Test listing sessions to find a valid session ID."""
    url = f"{API_BASE_URL}/sessions"
    
    try:
        print(f"🔍 Testing sessions endpoint...")
        response = requests.get(url)
        
        if response.status_code == 200:
            sessions = response.json()
            print(f"✅ Found {len(sessions)} sessions")
            if sessions:
                session = sessions[0]
                print(f"   Using session: {session['session_id']} - {session['name']}")
                return session['session_id']
            else:
                print("⚠️ No sessions found")
                return None
        else:
            print(f"❌ Error getting sessions: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        return None

def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 TESTING MODEL SELECTION FEATURE FOR RAG QUERY")
    print("=" * 60)
    
    # Test 1: Get available models
    print("\n" + "-" * 40)
    print("TEST 1: List Available Models")
    print("-" * 40)
    available_models = test_available_models()
    
    # Test 2: Get a session to test with
    print("\n" + "-" * 40)
    print("TEST 2: Find Test Session")
    print("-" * 40)
    session_id = test_list_sessions()
    
    if not session_id:
        print("⚠️ Cannot continue testing without a valid session. Please create a session first.")
        return
    
    test_query = "What is machine learning?"
    
    # Test 3: Query without model (should use default)
    print("\n" + "-" * 40)
    print("TEST 3: RAG Query Without Model (Default)")
    print("-" * 40)
    success_default = test_rag_query_with_model(session_id, test_query, None)
    
    # Test 4: Query with specific model (if available)
    if available_models:
        print("\n" + "-" * 40)
        print("TEST 4: RAG Query With Specific Model")
        print("-" * 40)
        
        # Try different models
        test_models = available_models[:2]  # Test first 2 models
        
        for model in test_models:
            print(f"\n🔍 Testing with model: {model}")
            success_model = test_rag_query_with_model(session_id, test_query, model)
            if success_model:
                print(f"✅ Model {model} works!")
            else:
                print(f"❌ Model {model} failed!")
    
    print("\n" + "=" * 60)
    print("🏁 TESTING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()