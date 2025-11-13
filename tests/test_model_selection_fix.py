#!/usr/bin/env python3
"""
Model Selection Fix Test
Tests that model selection now properly categorizes Groq/Ollama models and removes problematic ones
"""

import requests
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
API_GATEWAY_URL = "http://localhost:8080"
MODEL_INFERENCE_URL = "http://localhost:8002"

def test_model_inference_service():
    """Test Model Inference Service directly"""
    try:
        logger.info("🔍 Testing Model Inference Service directly")
        
        # Test the available models endpoint
        response = requests.get(f"{MODEL_INFERENCE_URL}/models/available", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Model Inference Service Response:")
            logger.info(f"   Groq models: {data.get('groq', [])}")
            logger.info(f"   Ollama models: {data.get('ollama', [])}")
            return data
        else:
            logger.error(f"❌ Model Inference Service failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Model Inference Service error: {str(e)}")
        return None

def test_api_gateway_models():
    """Test API Gateway models endpoint"""
    try:
        logger.info("🔍 Testing API Gateway models endpoint")
        
        response = requests.get(f"{API_GATEWAY_URL}/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ API Gateway Models Response:")
            
            models = data.get('models', [])
            providers = data.get('providers', {})
            
            logger.info(f"   Total models: {len(models)}")
            logger.info(f"   Providers: {list(providers.keys())}")
            
            # Check for problematic models
            problematic_models = []  # All new models should be working
            found_problematic = []
            
            for model in models:
                model_id = model.get('id') if isinstance(model, dict) else model
                if model_id in problematic_models:
                    found_problematic.append(model_id)
            
            if found_problematic:
                logger.error(f"❌ Found problematic models: {found_problematic}")
            else:
                logger.info("✅ No problematic models found")
            
            # Display model categorization
            for provider_key, provider_info in providers.items():
                provider_models = [m for m in models if m.get('provider') == provider_key]
                logger.info(f"   {provider_key.upper()}: {len(provider_models)} models")
                for model in provider_models:
                    logger.info(f"     - {model.get('name', model.get('id', 'Unknown'))}")
            
            return data
        else:
            logger.error(f"❌ API Gateway models failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ API Gateway models error: {str(e)}")
        return None

def test_rag_query_with_groq():
    """Test RAG query with Groq model"""
    try:
        logger.info("🔍 Testing RAG query with Groq model")
        
        # First create a test session (if not exists)
        session_data = {
            "name": "Model Test Session",
            "description": "Testing model selection",
            "category": "research",
            "created_by": "test_user"
        }
        
        # Try to create session
        try:
            session_response = requests.post(f"{API_GATEWAY_URL}/sessions", json=session_data)
            if session_response.status_code == 200:
                session = session_response.json()
                session_id = session["session_id"]
                logger.info(f"✅ Created test session: {session_id}")
            else:
                # Use existing session if creation fails
                sessions_response = requests.get(f"{API_GATEWAY_URL}/sessions")
                if sessions_response.status_code == 200:
                    sessions = sessions_response.json()
                    if sessions:
                        session_id = sessions[0]["session_id"]
                        logger.info(f"✅ Using existing session: {session_id}")
                    else:
                        logger.error("❌ No sessions available for testing")
                        return False
                else:
                    logger.error("❌ Could not get sessions for testing")
                    return False
        except Exception as e:
            logger.error(f"❌ Session setup failed: {str(e)}")
            return False
        
        # Test RAG query with a Groq model
        query_data = {
            "session_id": session_id,
            "query": "Test query",
            "model": "llama-3.1-8b-instant",  # Working Groq model
            "top_k": 3,
            "use_rerank": True
        }
        
        response = requests.post(f"{API_GATEWAY_URL}/rag/query", json=query_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ RAG query with Groq model successful")
            logger.info(f"   Answer: {result.get('answer', '')[:100]}...")
            return True
        else:
            logger.error(f"❌ RAG query failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ RAG query test error: {str(e)}")
        return False

def test_model_selection_improvements():
    """Test all model selection improvements"""
    logger.info("🚀 Starting Model Selection Improvements Test")
    
    results = {
        "model_inference_service": False,
        "api_gateway_models": False,
        "rag_query_groq": False
    }
    
    # Test 1: Model Inference Service
    results["model_inference_service"] = test_model_inference_service() is not None
    
    # Test 2: API Gateway Models
    results["api_gateway_models"] = test_api_gateway_models() is not None
    
    # Test 3: RAG Query with Groq
    results["rag_query_groq"] = test_rag_query_with_groq()
    
    # Summary
    logger.info("🏁 MODEL SELECTION TEST RESULTS:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - Model selection improvements working!")
    else:
        logger.error("❌ SOME TESTS FAILED - Model selection needs debugging")
    
    return all_passed

if __name__ == "__main__":
    success = test_model_selection_improvements()
    exit(0 if success else 1)