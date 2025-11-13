#!/usr/bin/env python3
"""
Test internal analysis fix via API Gateway
"""
import requests
import json
import time

def test_api_gateway_rag():
    """Test API Gateway RAG endpoint for internal analysis"""
    print("🔍 Testing API Gateway RAG Query for Internal Analysis...")
    
    api_gateway_url = "http://127.0.0.1:8000"
    
    # Simple test query
    test_query = {
        "session_id": "test-session-internal-analysis",
        "query": "Havadaki oksijen yüzdesi nedir?",
        "model": "llama-3.1-8b-instant",
        "use_direct_llm": True,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(
            f"{api_gateway_url}/rag/query",
            json=test_query,
            timeout=60
        )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "").strip()
            print(f"✅ API Gateway Response: {answer}")
            
            # Check if response contains internal analysis steps
            analysis_keywords = [
                "İÇSEL ANALİZ", "ÇIKTIDA GÖSTERME", "AŞAMA", "ANALİZ SÜRECİ",
                "DOĞRULAMA", "KAYNAK ANALİZİ", "AŞAMA 1", "AŞAMA 2"
            ]
            
            has_internal_analysis = any(keyword in answer.upper() for keyword in analysis_keywords)
            
            if has_internal_analysis:
                print("❌ PROBLEM: Response still contains internal analysis steps!")
                print(f"Raw response: {answer}")
                return False
            else:
                print("✅ SUCCESS: Response contains only final answer!")
                return True
        else:
            print(f"❌ API Gateway error: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error detail: {error_detail}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API Gateway: {e}")
        return False

def main():
    print("🚀 Testing Internal Analysis Fix via API Gateway")
    print("=" * 50)
    
    success = test_api_gateway_rag()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TEST PASSED! Internal analysis fix working via API Gateway.")
    else:
        print("❌ TEST FAILED! Need to check API Gateway.")
        
    return success

if __name__ == "__main__":
    main()