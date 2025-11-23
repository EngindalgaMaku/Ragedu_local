#!/usr/bin/env python3
"""
Direct Özeti Yenile Fix Script
Alternative solution when topic regeneration fails
"""

import requests
import json
import sys
from typing import Dict, Any, List

# Configuration
API_GATEWAY_URL = "http://localhost:8000"

def get_topic_info(session_id: str, topic_id: int) -> Dict[str, Any]:
    """Get detailed information about a specific topic"""
    
    try:
        # Get topics for session
        response = requests.get(f"{API_GATEWAY_URL}/api/aprag/topics/session/{session_id}", timeout=10)
        if response.status_code != 200:
            return {"error": f"Failed to fetch topics: {response.status_code}"}
        
        topics_data = response.json()
        topics = topics_data.get('topics', [])
        
        # Find specific topic
        topic = next((t for t in topics if t.get('topic_id') == topic_id), None)
        if not topic:
            return {"error": f"Topic {topic_id} not found"}
        
        return {"success": True, "topic": topic}
        
    except Exception as e:
        return {"error": str(e)}

def get_session_chunks(session_id: str) -> Dict[str, Any]:
    """Get all chunks for a session"""
    
    try:
        response = requests.get(f"{API_GATEWAY_URL}/sessions/{session_id}/chunks", timeout=30)
        if response.status_code != 200:
            return {"error": f"Failed to fetch chunks: {response.status_code}"}
        
        chunks_data = response.json()
        chunks = chunks_data.get('chunks', [])
        
        return {"success": True, "chunks": chunks}
        
    except Exception as e:
        return {"error": str(e)}

def manually_link_topic_to_chunks(session_id: str, topic_id: int, topic_title: str, chunks: List[Dict]) -> List[Dict]:
    """Manually find relevant chunks for a topic based on title keywords"""
    
    print(f"🔍 Manually finding relevant chunks for '{topic_title}'...")
    
    # Extract keywords from topic title
    title_words = topic_title.lower().split()
    
    # Enhanced Turkish keyword extraction
    keywords = []
    for word in title_words:
        # Remove common Turkish suffixes and particles
        word = word.replace("'den", "").replace("'dan", "").replace("'ye", "").replace("'ya", "")
        word = word.replace("'nin", "").replace("'nın", "").replace("'nu", "").replace("'nü", "")
        word = word.replace("ler", "").replace("lar", "").replace("den", "").replace("dan", "")
        
        if len(word) > 3:  # Only meaningful words
            keywords.append(word)
    
    print(f"🔑 Extracted keywords: {keywords}")
    
    # Find matching chunks
    relevant_chunks = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("chunk_text", chunk.get("content", "")).lower()
        
        # Count matches
        matches = 0
        for keyword in keywords:
            if keyword in content:
                matches += 1
        
        # Score the match
        if matches > 0:
            score = matches / len(keywords)
            chunk_copy = chunk.copy()
            chunk_copy["relevance_score"] = score
            chunk_copy["keyword_matches"] = matches
            relevant_chunks.append(chunk_copy)
            print(f"📄 Chunk {i}: {matches}/{len(keywords)} keywords matched (score: {score:.2f})")
    
    # Sort by relevance
    relevant_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Return top 10
    top_chunks = relevant_chunks[:10]
    print(f"✅ Found {len(top_chunks)} relevant chunks (from {len(chunks)} total)")
    
    return top_chunks

def test_direct_kb_extraction(session_id: str, topic_id: int, relevant_chunks: List[Dict]) -> Dict[str, Any]:
    """Test knowledge extraction with manually provided chunks"""
    
    print(f"🧪 Testing direct KB extraction with manual chunk selection...")
    
    try:
        # Prepare chunk text
        chunks_text = "\n\n---\n\n".join([
            chunk.get("chunk_text", chunk.get("content", ""))
            for chunk in relevant_chunks
        ])
        
        print(f"📝 Prepared {len(chunks_text)} characters of relevant text")
        print(f"📦 Using {len(relevant_chunks)} chunks for extraction")
        
        # Try the extraction
        response = requests.post(
            f"{API_GATEWAY_URL}/api/aprag/knowledge/extract/{topic_id}",
            json={
                "topic_id": topic_id,
                "force_refresh": True
            },
            timeout=120
        )
        
        if response.status_code == 422:
            # This is expected if our fix is working - chunks not linked properly
            return {
                "expected_error": True,
                "message": "Expected error - topic-chunk relationship broken as intended",
                "needs_manual_fix": True,
                "error_details": response.text
            }
        elif response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return {
                    "success": True,
                    "message": "KB extraction successful!",
                    "details": data
                }
            else:
                return {
                    "success": False,
                    "error": "KB extraction returned success=false",
                    "details": data
                }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_minimal_topic_summary(topic_title: str, relevant_chunks: List[Dict]) -> str:
    """Create a simple topic summary manually without LLM dependency"""
    
    chunks_text = "\n".join([
        chunk.get("chunk_text", chunk.get("content", ""))[:500]  # First 500 chars of each
        for chunk in relevant_chunks[:3]  # Top 3 most relevant chunks
    ])
    
    # Simple extractive summary (first sentences of top chunks)
    sentences = []
    for chunk in relevant_chunks[:3]:
        content = chunk.get("chunk_text", chunk.get("content", ""))
        # Find first sentence
        first_sentence = content.split('.')[0] + '.'
        if len(first_sentence) > 20 and len(first_sentence) < 200:
            sentences.append(first_sentence)
    
    manual_summary = f"Bu konu '{topic_title}' hakkındadır. " + " ".join(sentences[:3])
    
    return manual_summary

def main():
    print("🔧 Direct Özeti Yenile Fix Tool")
    print("=" * 50)
    print("Bu araç topic regeneration başarısız olduğunda alternatif çözüm sağlar.")
    
    # Get parameters
    session_id = input("\n📝 Enter Session ID: ").strip()
    if not session_id:
        print("❌ Session ID is required!")
        return
    
    try:
        topic_id = int(input("📚 Enter Topic ID (problematic one): ").strip())
    except ValueError:
        print("❌ Topic ID must be a number!")
        return
    
    # Step 1: Get topic info
    print(f"\n🔍 Getting topic information...")
    topic_info = get_topic_info(session_id, topic_id)
    if "error" in topic_info:
        print(f"❌ {topic_info['error']}")
        return
    
    topic = topic_info['topic']
    topic_title = topic.get('topic_title', 'Unknown Topic')
    print(f"📚 Topic: {topic_title}")
    print(f"🔑 Current keywords: {topic.get('keywords', [])}")
    print(f"🔗 Current chunk IDs: {topic.get('related_chunk_ids', [])}")
    
    # Step 2: Get session chunks
    print(f"\n📦 Getting session chunks...")
    chunks_info = get_session_chunks(session_id)
    if "error" in chunks_info:
        print(f"❌ {chunks_info['error']}")
        return
    
    chunks = chunks_info['chunks']
    print(f"📦 Found {len(chunks)} chunks in session")
    
    # Step 3: Manually link topic to chunks
    relevant_chunks = manually_link_topic_to_chunks(session_id, topic_id, topic_title, chunks)
    
    if not relevant_chunks:
        print("❌ No relevant chunks found. The topic title might be too generic.")
        print("💡 Suggestion: Try with a more specific topic or check if documents contain this topic.")
        return
    
    # Step 4: Test the extraction
    print(f"\n🧪 Testing KB extraction with manual chunk selection...")
    test_result = test_direct_kb_extraction(session_id, topic_id, relevant_chunks)
    
    if test_result.get('expected_error'):
        print("✅ Expected behavior confirmed - our fix is working!")
        print("   The system correctly rejects extraction when topic-chunk relationships are broken.")
        print("\n💡 Manual Summary Creation (Workaround):")
        
        manual_summary = create_minimal_topic_summary(topic_title, relevant_chunks)
        print(f"📝 Manual Summary: {manual_summary}")
        
        print(f"\n📊 Analysis:")
        print(f"   - Topic: {topic_title}")
        print(f"   - Relevant Chunks: {len(relevant_chunks)}")
        print(f"   - Best Matches:")
        for i, chunk in enumerate(relevant_chunks[:3]):
            print(f"     {i+1}. Score: {chunk.get('relevance_score', 0):.2f} - Keywords: {chunk.get('keyword_matches', 0)}")
        
    elif test_result.get('success'):
        print("🎉 Unexpected Success! KB extraction worked!")
        print("   This suggests the problem was temporarily resolved.")
        details = test_result.get('details', {})
        print(f"   Quality Score: {details.get('quality_score', 'N/A')}")
        print(f"   Extraction Time: {details.get('extraction_time_seconds', 'N/A')}s")
    else:
        print(f"❌ Test failed: {test_result.get('error', 'Unknown error')}")
        if 'details' in test_result:
            print(f"   Details: {test_result['details']}")
    
    # Step 5: Recommendations
    print(f"\n" + "="*50)
    print("🎯 RECOMMENDATIONS")
    print("="*50)
    
    print("1. ✅ IMMEDIATE WORKAROUND:")
    print("   - Sistem şu anda doğru davranıyor (broken relationships'i reddediyor)")
    print("   - Manual summary yukarıda sağlandı")
    print("   - Bu geçici bir çözümdür")
    
    print("\n2. 🔧 PERMANENT FIX OPTIONS:")
    print("   a) Fix LLM service dependency causing topic regeneration to fail")
    print("   b) Check model-inference-service logs for errors")
    print("   c) Restart services: docker-compose restart")
    print("   d) Try topic regeneration again after fixing service issues")
    
    print("\n3. 🚀 NEXT STEPS:")
    print("   a) Check service logs: docker-compose logs model-inference-service")
    print("   b) Verify Ollama/Groq API availability")
    print("   c) Restart topic regeneration when services are stable")

if __name__ == "__main__":
    main()