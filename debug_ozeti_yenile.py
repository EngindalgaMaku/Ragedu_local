#!/usr/bin/env python3
"""
Özeti Yenile Debug Script
Diagnoses issues with the KB summary refresh functionality
"""

import requests
import json
import sys
import time
from typing import Dict, Any

# Configuration
API_GATEWAY_URL = "http://localhost:8000"  # Adjust if different
APRAG_SERVICE_URL = "http://localhost:8007"  # Adjust if different
MODEL_INFERENCE_URL = "http://localhost:8002"  # Adjust if different
DOCUMENT_PROCESSING_URL = "http://localhost:8002"  # Adjust if different

def check_service(name: str, url: str, endpoint: str = "/health") -> Dict[str, Any]:
    """Check if a service is running and responsive"""
    try:
        response = requests.get(f"{url}{endpoint}", timeout=5)
        return {
            "name": name,
            "url": url,
            "status": "✅ HEALTHY" if response.status_code == 200 else f"⚠️ STATUS {response.status_code}",
            "response_time": response.elapsed.total_seconds(),
            "details": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:200]
        }
    except requests.exceptions.ConnectionError:
        return {"name": name, "url": url, "status": "❌ CONNECTION REFUSED", "error": "Service not running"}
    except requests.exceptions.Timeout:
        return {"name": name, "url": url, "status": "⏰ TIMEOUT", "error": "Service too slow"}
    except Exception as e:
        return {"name": name, "url": url, "status": f"❌ ERROR", "error": str(e)}

def test_ozeti_yenile_flow(session_id: str, topic_id: int) -> Dict[str, Any]:
    """Test the complete özeti yenile flow"""
    
    results = {
        "session_id": session_id,
        "topic_id": topic_id,
        "steps": []
    }
    
    # Step 1: Check if session exists
    print(f"🔍 Step 1: Checking session {session_id}...")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/sessions/{session_id}", timeout=10)
        if response.status_code == 200:
            session_data = response.json()
            results["steps"].append({
                "step": "session_check",
                "status": "✅ SUCCESS",
                "details": f"Session found: {session_data.get('name', 'Unknown')}"
            })
        else:
            results["steps"].append({
                "step": "session_check", 
                "status": f"❌ FAILED ({response.status_code})",
                "details": response.text
            })
            return results
    except Exception as e:
        results["steps"].append({
            "step": "session_check",
            "status": "❌ ERROR",
            "details": str(e)
        })
        return results
    
    # Step 2: Check if topic exists
    print(f"📚 Step 2: Checking topic {topic_id}...")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/api/aprag/topics/session/{session_id}", timeout=10)
        if response.status_code == 200:
            topics_data = response.json()
            topic_found = any(t.get('topic_id') == topic_id for t in topics_data.get('topics', []))
            if topic_found:
                topic = next(t for t in topics_data.get('topics', []) if t.get('topic_id') == topic_id)
                results["steps"].append({
                    "step": "topic_check",
                    "status": "✅ SUCCESS",
                    "details": f"Topic found: {topic.get('topic_title', 'Unknown')}"
                })
            else:
                results["steps"].append({
                    "step": "topic_check",
                    "status": "❌ TOPIC NOT FOUND",
                    "details": f"Topic ID {topic_id} not found in session {session_id}"
                })
                return results
        else:
            results["steps"].append({
                "step": "topic_check",
                "status": f"❌ FAILED ({response.status_code})",
                "details": response.text
            })
            return results
    except Exception as e:
        results["steps"].append({
            "step": "topic_check",
            "status": "❌ ERROR",
            "details": str(e)
        })
        return results
    
    # Step 3: Test KB extraction API call
    print(f"🚀 Step 3: Testing KB extraction for topic {topic_id}...")
    try:
        response = requests.post(
            f"{API_GATEWAY_URL}/api/aprag/knowledge/extract/{topic_id}",
            json={
                "topic_id": topic_id,
                "force_refresh": True
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                results["steps"].append({
                    "step": "kb_extraction",
                    "status": "✅ SUCCESS",
                    "details": f"KB extracted successfully. Quality score: {data.get('quality_score', 'N/A')}"
                })
            else:
                results["steps"].append({
                    "step": "kb_extraction",
                    "status": "❌ FAILED",
                    "details": f"API returned success=false: {data}"
                })
        else:
            error_text = response.text
            results["steps"].append({
                "step": "kb_extraction",
                "status": f"❌ HTTP {response.status_code}",
                "details": error_text
            })
    except Exception as e:
        results["steps"].append({
            "step": "kb_extraction",
            "status": "❌ ERROR",
            "details": str(e)
        })
    
    return results

def main():
    print("🔧 Özeti Yenile (KB Refresh) Diagnostic Tool")
    print("=" * 50)
    
    # Step 1: Check all required services
    print("\n🏥 Checking Service Health...")
    services = [
        ("API Gateway", API_GATEWAY_URL),
        ("APRAG Service", APRAG_SERVICE_URL),
        ("Model Inference", MODEL_INFERENCE_URL),
        ("Document Processing", DOCUMENT_PROCESSING_URL)
    ]
    
    all_healthy = True
    for name, url in services:
        result = check_service(name, url)
        print(f"{result['status']:20} {name:20} ({url})")
        if "❌" in result['status']:
            all_healthy = False
            print(f"   Error: {result.get('error', 'Unknown')}")
        elif result.get('details'):
            print(f"   Details: {str(result['details'])[:100]}...")
    
    if not all_healthy:
        print("\n⚠️  Some services are not healthy. Please fix service issues first.")
        return
    
    print("\n✅ All services are healthy!")
    
    # Step 2: Get test parameters
    session_id = input("\n📝 Enter Session ID (from URL): ").strip()
    if not session_id:
        print("❌ Session ID is required!")
        return
    
    try:
        topic_id = int(input("📚 Enter Topic ID: ").strip())
    except ValueError:
        print("❌ Topic ID must be a number!")
        return
    
    # Step 3: Test the complete flow
    print(f"\n🧪 Testing Özeti Yenile flow for session {session_id}, topic {topic_id}...")
    print("-" * 60)
    
    results = test_ozeti_yenile_flow(session_id, topic_id)
    
    # Display results
    print("\n📊 Test Results:")
    for step in results["steps"]:
        print(f"{step['status']:15} {step['step']:20} - {step['details']}")
    
    # Final recommendation
    failed_steps = [s for s in results["steps"] if "❌" in s['status']]
    if not failed_steps:
        print("\n🎉 All tests passed! The Özeti Yenile functionality should be working.")
        print("   If it's still not working, check the browser console for frontend errors.")
    else:
        print(f"\n🔧 Found {len(failed_steps)} issues that need to be fixed:")
        for step in failed_steps:
            print(f"   - {step['step']}: {step['details']}")
    
    print("\n💡 Tips for fixing common issues:")
    print("   - Service not running: Use docker-compose up -d <service-name>")
    print("   - Database issues: Check /app/data/ directory in Docker containers")
    print("   - Model inference issues: Check if Ollama/Groq API keys are configured")
    print("   - Frontend issues: Check browser console for JavaScript errors")

if __name__ == "__main__":
    main()