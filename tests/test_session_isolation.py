#!/usr/bin/env python3
"""
Test script to check if chunks are properly isolated between sessions
"""
import requests
import json
import hashlib
import random
import string

def generate_session_id():
    """Generate a realistic 32-char hex session ID"""
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    return hashlib.md5(random_string.encode()).hexdigest()

def create_test_session(session_name, document_content, source_files):
    """Create a test session with specific content"""
    session_id = generate_session_id()
    
    # Test metadata with source_files 
    test_metadata = {
        "session_id": session_id,
        "embedding_model": "mxbai-embed-large", 
        "chunk_strategy": "semantic",
        "source_files": source_files
    }
    
    # Request payload  
    payload = {
        "text": document_content.strip(),
        "metadata": test_metadata,
        "collection_name": f"session_{session_id}",
        "chunk_size": 300,  # Smaller chunks for testing
        "chunk_overlap": 50
    }
    
    print(f"📤 Creating {session_name} session: {session_id}")
    
    try:
        response = requests.post(
            "http://localhost:8003/process-and-store",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {session_name} created: {result['chunks_processed']} chunks")
            return session_id
        else:
            print(f"❌ {session_name} creation failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ {session_name} creation error: {str(e)}")
        return None

def check_session_chunks(session_id, session_name):
    """Check chunks for a specific session"""
    print(f"\n🔍 Checking chunks for {session_name} ({session_id}):")
    
    try:
        response = requests.get(
            f"http://localhost:8003/sessions/{session_id}/chunks",
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            chunks = data.get("chunks", [])
            
            print(f"📄 Found {len(chunks)} chunks:")
            
            for i, chunk in enumerate(chunks):
                document_name = chunk.get('document_name', 'N/A')
                chunk_text = chunk.get('chunk_text', '')[:100] + "..."
                metadata = chunk.get('chunk_metadata', {})
                stored_session_id = metadata.get('session_id', 'N/A')
                
                print(f"   Chunk {i+1}:")
                print(f"     Document: {document_name}")
                print(f"     Stored Session ID: {stored_session_id}")
                print(f"     Text Preview: {chunk_text}")
                
                # Check for session ID mismatch (this would indicate chunk mixing!)
                if stored_session_id != session_id:
                    print(f"     ⚠️ WARNING: Session ID mismatch! Expected: {session_id}, Found: {stored_session_id}")
                else:
                    print(f"     ✅ Session ID correct")
                
            return chunks
        else:
            print(f"❌ Failed to get chunks: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error getting chunks: {str(e)}")
        return []

def test_session_isolation():
    """Test if sessions are properly isolated"""
    print("🧪 Testing Session Isolation")
    print("=" * 50)
    
    # Create Session 1 - Biology
    biology_content = """
# Biology Fundamentals
Cells are the basic unit of all living organisms.
DNA contains genetic information for all living things.
Photosynthesis is the process by which plants make food.
"""
    
    session1_id = create_test_session(
        "Biology Session", 
        biology_content,
        ["biology_fundamentals.md", "cell_biology.pdf"]
    )
    
    # Create Session 2 - Physics  
    physics_content = """
# Physics Principles
Force equals mass times acceleration (F = ma).
Energy cannot be created or destroyed, only transformed.
Light travels at 299,792,458 meters per second in vacuum.
"""
    
    session2_id = create_test_session(
        "Physics Session",
        physics_content, 
        ["physics_principles.md", "mechanics.pdf"]
    )
    
    # Create Session 3 - Math
    math_content = """
# Mathematics Concepts  
The Pythagorean theorem: a² + b² = c².
Derivatives measure the rate of change.
Integrals calculate the area under curves.
"""
    
    session3_id = create_test_session(
        "Math Session",
        math_content,
        ["mathematics_concepts.md", "calculus.pdf"]
    )
    
    if not all([session1_id, session2_id, session3_id]):
        print("❌ Failed to create all test sessions")
        return
    
    print("\n" + "=" * 50)
    print("📋 CHECKING SESSION ISOLATION")
    print("=" * 50)
    
    # Check each session's chunks
    bio_chunks = check_session_chunks(session1_id, "Biology Session")
    physics_chunks = check_session_chunks(session2_id, "Physics Session")
    math_chunks = check_session_chunks(session3_id, "Math Session")
    
    print("\n" + "=" * 50) 
    print("🎯 ISOLATION ANALYSIS")
    print("=" * 50)
    
    # Check for cross-contamination
    def contains_biology_terms(text):
        bio_terms = ["cell", "dna", "photosynthesis", "living", "organism"]
        return any(term.lower() in text.lower() for term in bio_terms)
        
    def contains_physics_terms(text):
        physics_terms = ["force", "mass", "acceleration", "energy", "light"]  
        return any(term.lower() in text.lower() for term in physics_terms)
        
    def contains_math_terms(text):
        math_terms = ["pythagorean", "derivative", "integral", "theorem", "calculus"]
        return any(term.lower() in text.lower() for term in math_terms)
    
    # Check Biology session for contamination
    print("\n🔬 Biology Session Analysis:")
    for i, chunk in enumerate(bio_chunks):
        text = chunk.get('chunk_text', '')
        has_physics = contains_physics_terms(text)
        has_math = contains_math_terms(text)
        
        if has_physics or has_math:
            print(f"❌ CONTAMINATION in Biology chunk {i+1}: contains {'Physics' if has_physics else 'Math'} content")
        else:
            print(f"✅ Biology chunk {i+1}: clean")
    
    # Check Physics session for contamination  
    print("\n⚛️ Physics Session Analysis:")
    for i, chunk in enumerate(physics_chunks):
        text = chunk.get('chunk_text', '')
        has_biology = contains_biology_terms(text)
        has_math = contains_math_terms(text)
        
        if has_biology or has_math:
            print(f"❌ CONTAMINATION in Physics chunk {i+1}: contains {'Biology' if has_biology else 'Math'} content")
        else:
            print(f"✅ Physics chunk {i+1}: clean")
            
    # Check Math session for contamination
    print("\n🔢 Math Session Analysis:")
    for i, chunk in enumerate(math_chunks):
        text = chunk.get('chunk_text', '')
        has_biology = contains_biology_terms(text)
        has_physics = contains_physics_terms(text)
        
        if has_biology or has_physics:
            print(f"❌ CONTAMINATION in Math chunk {i+1}: contains {'Biology' if has_biology else 'Physics'} content")
        else:
            print(f"✅ Math chunk {i+1}: clean")
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY")
    print("=" * 50)
    print(f"Biology Session ({session1_id[:8]}...): {len(bio_chunks)} chunks")
    print(f"Physics Session ({session2_id[:8]}...): {len(physics_chunks)} chunks") 
    print(f"Math Session ({session3_id[:8]}...): {len(math_chunks)} chunks")

if __name__ == "__main__":
    test_session_isolation()