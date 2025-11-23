#!/usr/bin/env python3
"""
Topic-Chunk Relationship Repair Script
Fixes the broken relationships between topics and document chunks
"""

import requests
import json
import sys
import time
from typing import Dict, Any, List

# Configuration
API_GATEWAY_URL = "http://localhost:8000"
APRAG_SERVICE_URL = "http://localhost:8007" 

def check_topic_chunk_relationships(session_id: str) -> Dict[str, Any]:
    """Check current topic-chunk relationships"""
    
    print(f"🔍 Checking topic-chunk relationships for session {session_id}...")
    
    try:
        # Get topics for session
        topics_response = requests.get(f"{API_GATEWAY_URL}/api/aprag/topics/session/{session_id}", timeout=10)
        if topics_response.status_code != 200:
            return {"error": f"Failed to fetch topics: {topics_response.status_code}"}
        
        topics_data = topics_response.json()
        topics = topics_data.get('topics', [])
        
        # Get chunks for session
        chunks_response = requests.get(f"{API_GATEWAY_URL}/sessions/{session_id}/chunks", timeout=10)
        if chunks_response.status_code != 200:
            return {"error": f"Failed to fetch chunks: {chunks_response.status_code}"}
        
        chunks_data = chunks_response.json()
        chunks = chunks_data.get('chunks', [])
        
        # Analyze relationships
        analysis = {
            "session_id": session_id,
            "total_topics": len(topics),
            "total_chunks": len(chunks),
            "topics_with_chunk_ids": 0,
            "topics_with_keywords": 0,
            "broken_relationships": [],
            "topics": []
        }
        
        for topic in topics:
            topic_analysis = {
                "topic_id": topic.get('topic_id'),
                "topic_title": topic.get('topic_title'),
                "has_chunk_ids": bool(topic.get('related_chunk_ids')),
                "chunk_ids": topic.get('related_chunk_ids', []),
                "has_keywords": bool(topic.get('keywords')),
                "keywords": topic.get('keywords', []),
                "status": "OK"
            }
            
            # Check if topic has proper relationships
            if not topic.get('related_chunk_ids') and not topic.get('keywords'):
                topic_analysis["status"] = "BROKEN - No chunk IDs or keywords"
                analysis["broken_relationships"].append(topic_analysis)
            elif topic.get('related_chunk_ids'):
                analysis["topics_with_chunk_ids"] += 1
            elif topic.get('keywords'):
                analysis["topics_with_keywords"] += 1
            
            analysis["topics"].append(topic_analysis)
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}

def regenerate_topics_with_chunk_links(session_id: str) -> Dict[str, Any]:
    """Regenerate topics with proper chunk relationships"""
    
    print(f"🔄 Regenerating topics with chunk relationships for session {session_id}...")
    
    try:
        # Start topic extraction with full method
        response = requests.post(
            f"{API_GATEWAY_URL}/api/aprag/topics/re-extract/{session_id}",
            params={"method": "full"},
            timeout=10
        )
        
        if response.status_code != 200:
            return {"error": f"Failed to start topic regeneration: {response.status_code}", "details": response.text}
        
        data = response.json()
        if not data.get('success'):
            return {"error": "Topic regeneration failed", "details": data}
        
        job_id = data.get('job_id')
        print(f"✅ Topic regeneration job started: {job_id}")
        
        # Poll for completion
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(
                f"{API_GATEWAY_URL}/api/aprag/topics/re-extract/status/{job_id}",
                timeout=10
            )
            
            if status_response.status_code == 200:
                status = status_response.json()
                
                if status.get('status') == 'completed':
                    result = status.get('result', {})
                    print(f"✅ Topic regeneration completed!")
                    print(f"   - Topics extracted: {result.get('merged_topics_count', 0)}")
                    print(f"   - Chunks analyzed: {result.get('chunks_analyzed', 0)}")
                    print(f"   - Batches processed: {result.get('batches_processed', 0)}")
                    
                    return {
                        "success": True,
                        "job_id": job_id,
                        "result": result
                    }
                    
                elif status.get('status') == 'failed':
                    return {"error": f"Topic regeneration failed: {status.get('error', 'Unknown error')}"}
                
                elif status.get('status') == 'processing':
                    current_batch = status.get('current_batch', 0)
                    total_batches = status.get('total_batches', 0)
                    print(f"   Progress: {current_batch}/{total_batches} batches processed...")
            
            time.sleep(3)
        
        return {"error": "Topic regeneration timed out"}
        
    except Exception as e:
        return {"error": str(e)}

def test_ozeti_yenile_after_fix(session_id: str, topic_id: int) -> Dict[str, Any]:
    """Test the Özeti Yenile functionality after the fix"""
    
    print(f"🧪 Testing Özeti Yenile for topic {topic_id} after fix...")
    
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
                return {
                    "success": True,
                    "message": "Özeti Yenile now works correctly!",
                    "details": {
                        "knowledge_id": data.get('knowledge_id'),
                        "quality_score": data.get('quality_score'),
                        "extraction_time": data.get('extraction_time_seconds'),
                        "components": data.get('extracted_components', {})
                    }
                }
            else:
                return {"success": False, "error": "KB extraction returned success=false", "details": data}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    print("🔧 Topic-Chunk Relationship Repair Tool")
    print("=" * 50)
    
    # Get session ID
    session_id = input("\n📝 Enter Session ID (from URL): ").strip()
    if not session_id:
        print("❌ Session ID is required!")
        return
    
    # Step 1: Check current relationships
    print("\n" + "="*50)
    print("STEP 1: Analyzing Current Topic-Chunk Relationships")
    print("="*50)
    
    analysis = check_topic_chunk_relationships(session_id)
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    print(f"📊 Analysis Results:")
    print(f"   - Total Topics: {analysis['total_topics']}")
    print(f"   - Total Chunks: {analysis['total_chunks']}")
    print(f"   - Topics with Chunk IDs: {analysis['topics_with_chunk_ids']}")
    print(f"   - Topics with Keywords: {analysis['topics_with_keywords']}")
    print(f"   - Broken Relationships: {len(analysis['broken_relationships'])}")
    
    if analysis['broken_relationships']:
        print("\n❌ Broken Topic-Chunk Relationships Found:")
        for broken in analysis['broken_relationships']:
            print(f"   - {broken['topic_title']} (ID: {broken['topic_id']}): {broken['status']}")
    
    # Step 2: Ask for regeneration
    print("\n" + "="*50)
    print("STEP 2: Topic Regeneration")
    print("="*50)
    
    if len(analysis['broken_relationships']) > 0:
        print(f"⚠️  Found {len(analysis['broken_relationships'])} topics with broken relationships.")
        regenerate = input("🔄 Do you want to regenerate topics with proper chunk links? (y/n): ").lower().strip()
    else:
        print("✅ All topics have proper relationships, but you can still regenerate if needed.")
        regenerate = input("🔄 Do you want to regenerate topics anyway? (y/n): ").lower().strip()
    
    if regenerate == 'y':
        result = regenerate_topics_with_chunk_links(session_id)
        if "error" in result:
            print(f"❌ Regeneration failed: {result['error']}")
            if "details" in result:
                print(f"   Details: {result['details']}")
            return
        else:
            print(f"✅ Topics regenerated successfully!")
    
    # Step 3: Test the fix
    print("\n" + "="*50)
    print("STEP 3: Testing Özeti Yenile Functionality")
    print("="*50)
    
    test_fix = input("🧪 Do you want to test the Özeti Yenile functionality? (y/n): ").lower().strip()
    if test_fix == 'y':
        try:
            topic_id = int(input("📚 Enter Topic ID to test: ").strip())
        except ValueError:
            print("❌ Invalid topic ID!")
            return
        
        test_result = test_ozeti_yenile_after_fix(session_id, topic_id)
        if test_result.get('success'):
            print("🎉 SUCCESS! Özeti Yenile is now working correctly!")
            print(f"   Quality Score: {test_result['details']['quality_score']}")
            print(f"   Extraction Time: {test_result['details']['extraction_time']}s")
            components = test_result['details']['components']
            print(f"   Components: {components['summary_length']} words summary, {components['concepts_count']} concepts")
        else:
            print(f"❌ Test failed: {test_result['error']}")
            if 'details' in test_result:
                print(f"   Details: {test_result['details']}")
    
    print("\n" + "="*50)
    print("🎯 SUMMARY")
    print("="*50)
    print("✅ Topic-chunk relationship analysis completed")
    if regenerate == 'y':
        print("✅ Topics regenerated with proper chunk links")
    if test_fix == 'y':
        if test_result.get('success'):
            print("✅ Özeti Yenile functionality verified working")
        else:
            print("❌ Özeti Yenile still needs debugging")
    
    print("\n💡 Next Steps:")
    print("   1. Go to the frontend and try the 'Özeti Yenile (TR)' button")
    print("   2. Check browser console for detailed debug logs")
    print("   3. Check backend logs for detailed processing information")
    print("   4. If issues persist, ensure all services are running properly")

if __name__ == "__main__":
    main()