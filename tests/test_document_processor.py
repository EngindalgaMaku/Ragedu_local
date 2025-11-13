#!/usr/bin/env python3
"""
Test script for Document Processing Service
"""

import requests
import json
import time

# Configuration
SERVICE_URL = "http://localhost:8080"  # Change to your deployed service URL

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   ChromaDB available: {data.get('chromadb_available')}")
            print(f"   LangChain available: {data.get('langchain_available')}")
            print(f"   ChromaDB connected: {data.get('chromadb_connected')}")
            print(f"   Model service connected: {data.get('model_service_connected')}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_process_and_store():
    """Test the process-and-store endpoint"""
    print("\nTesting process-and-store endpoint...")
    
    # Sample text for testing
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create software or machines that exhibit human-like intelligence. 
    This can include learning from experience, understanding natural language, solving problems, and recognizing patterns.
    
    Machine Learning (ML) is a subset of AI that focuses on algorithms and statistical models that enable computers to improve at tasks 
    with experience. Deep Learning, a further subset, uses neural networks with multiple layers to model complex patterns in data.
    
    Natural Language Processing (NLP) combines computational linguistics with statistical, machine learning, and deep learning models 
    to process human language in the form of text or voice data. Applications include translation, sentiment analysis, and chatbots.
    """
    
    payload = {
        "text": sample_text,
        "metadata": {
            "source": "test_document",
            "category": "technology"
        },
        "collection_name": "test_collection",
        "chunk_size": 500,
        "chunk_overlap": 50
    }
    
    try:
        response = requests.post(
            f"{SERVICE_URL}/process-and-store",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Process and store test passed")
            print(f"   Success: {data.get('success')}")
            print(f"   Message: {data.get('message')}")
            print(f"   Chunks processed: {data.get('chunks_processed')}")
            print(f"   Collection name: {data.get('collection_name')}")
            print(f"   Chunk IDs count: {len(data.get('chunk_ids', []))}")
            return True
        else:
            print(f"❌ Process and store test failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Process and store test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Document Processing Service Test")
    print("=" * 50)
    print(f"Service URL: {SERVICE_URL}")
    
    # Wait a moment for service to start
    print("Waiting for service to start...")
    time.sleep(2)
    
    # Run tests
    health_ok = test_health()
    process_ok = test_process_and_store()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Health endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Process and store endpoint: {'✅ PASS' if process_ok else '❌ FAIL'}")
    
    if health_ok and process_ok:
        print("\n🎉 All tests passed! The Document Processing Service is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the service logs for more information.")

if __name__ == "__main__":
    main()