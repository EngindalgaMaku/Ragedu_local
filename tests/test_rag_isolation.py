#!/usr/bin/env python3
"""
Test script to check if RAG queries are properly isolated between sessions
"""
import requests
import json

def test_rag_query_isolation():
    """Test if RAG queries only return chunks from the correct session"""
    print("🧪 Testing RAG Query Isolation")
    print("=" * 50)
    
    # Use existing sessions from the previous test
    # We'll use some realistic session IDs from the logs
    existing_sessions = [
        ("Biology", "fade01955abee70a0fe4b7a1a775f8b1"),  # Should contain biology content
        ("Physics", "a323305ac52ba1d1be075a5496c3e2ca"),  # Should contain physics content  
        ("Math", "08d237b54d4e343d14148d3b0dfbb3e7")      # Should contain math content
    ]
    
    test_queries = [
        ("What are cells?", "Biology"),      # Biology question
        ("What is force?", "Physics"),       # Physics question  
        ("What is Pythagorean theorem?", "Math")  # Math question
    ]
    
    print("🔍 Testing cross-session RAG queries...")
    print("=" * 50)
    
    for query_text, expected_domain in test_queries:
        print(f"\n📝 Query: '{query_text}' (Expected: {expected_domain})")
        print("-" * 50)
        
        for session_name, session_id in existing_sessions:
            print(f"\n🎯 Querying {session_name} session ({session_id[:8]}...):")
            
            try:
                # RAG query payload
                rag_payload = {
                    "session_id": session_id,
                    "query": query_text,
                    "top_k": 3,
                    "use_rerank": True,
                    "min_score": 0.1,
                    "max_context_chars": 2000
                }
                
                # Send RAG query
                response = requests.post(
                    "http://localhost:8003/query",
                    json=rag_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    
                    print(f"   📤 Answer: {answer[:200]}...")
                    print(f"   📚 Sources found: {len(sources)}")
                    
                    # Check source content for contamination
                    for i, source in enumerate(sources):
                        source_text = source.get("text", "")
                        source_metadata = source.get("metadata", {})
                        stored_session_id = source_metadata.get("session_id", "N/A")
                        
                        print(f"   📄 Source {i+1}:")
                        print(f"      Session ID: {stored_session_id}")
                        print(f"      Text: {source_text[:100]}...")
                        
                        # Critical check: Is the source from the correct session?
                        if stored_session_id == session_id:
                            print(f"      ✅ Source is from correct session")
                        else:
                            print(f"      ❌ CRITICAL ERROR: Source from wrong session!")
                            print(f"         Expected: {session_id}")
                            print(f"         Found: {stored_session_id}")
                    
                    # Check if answer is relevant to the session's domain
                    answer_lower = answer.lower()
                    
                    # Domain relevance check
                    biology_terms = ["cell", "dna", "living", "organism", "photosynthesis"]
                    physics_terms = ["force", "mass", "energy", "acceleration", "light"] 
                    math_terms = ["theorem", "pythagorean", "derivative", "integral", "equation"]
                    
                    if session_name == "Biology":
                        has_bio_terms = any(term in answer_lower for term in biology_terms)
                        has_other_terms = any(term in answer_lower for term in physics_terms + math_terms)
                        
                        if expected_domain == "Biology":
                            if has_bio_terms:
                                print(f"   ✅ Correct domain: Biology answer from Biology session")
                            elif not has_bio_terms and len(sources) > 0:
                                print(f"   ⚠️ Biology session gave non-biology answer to biology question")
                        else:
                            if has_bio_terms and expected_domain != "Biology":
                                print(f"   ⚠️ Biology session gave biology answer to {expected_domain} question")
                    
                    elif session_name == "Physics":
                        has_physics_terms = any(term in answer_lower for term in physics_terms)
                        if expected_domain == "Physics":
                            if has_physics_terms:
                                print(f"   ✅ Correct domain: Physics answer from Physics session")
                            elif not has_physics_terms and len(sources) > 0:
                                print(f"   ⚠️ Physics session gave non-physics answer to physics question")
                        else:
                            if has_physics_terms and expected_domain != "Physics":
                                print(f"   ⚠️ Physics session gave physics answer to {expected_domain} question")
                    
                    elif session_name == "Math":
                        has_math_terms = any(term in answer_lower for term in math_terms)
                        if expected_domain == "Math":
                            if has_math_terms:
                                print(f"   ✅ Correct domain: Math answer from Math session")
                            elif not has_math_terms and len(sources) > 0:
                                print(f"   ⚠️ Math session gave non-math answer to math question")
                        else:
                            if has_math_terms and expected_domain != "Math":
                                print(f"   ⚠️ Math session gave math answer to {expected_domain} question")
                    
                else:
                    print(f"   ❌ RAG query failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ RAG query error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎯 RAG ISOLATION TEST SUMMARY")
    print("=" * 50)
    print("Each session should only return sources with its own session_id.")
    print("Cross-domain contamination indicates potential issues.")

if __name__ == "__main__":
    test_rag_query_isolation()