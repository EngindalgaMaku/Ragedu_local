#!/usr/bin/env python3
"""
Test script for the new /rag/chunks/{session_id} endpoint
"""
import requests
import sqlite3
from pathlib import Path

def test_chunks_endpoint():
    """Test the GET /rag/chunks/{session_id} endpoint"""
    
    # First, find a session with chunks in the database
    db_path = Path('data/analytics/sessions.db')
    if not db_path.exists():
        print("❌ Sessions database not found")
        return False
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get a session ID that has chunks
    cursor.execute("SELECT DISTINCT session_id FROM document_chunks LIMIT 1")
    result = cursor.fetchone()
    
    if not result:
        print("❌ No sessions with chunks found in database")
        conn.close()
        return False
    
    session_id = result[0]
    
    # Get expected chunk count for validation
    cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE session_id = ?", (session_id,))
    expected_count = cursor.fetchone()[0]
    
    # Get some sample data for comparison
    cursor.execute("""
        SELECT document_name, chunk_index, LENGTH(chunk_text) as text_length 
        FROM document_chunks 
        WHERE session_id = ? 
        ORDER BY document_name, chunk_index 
        LIMIT 3
    """, (session_id,))
    sample_data = cursor.fetchall()
    
    conn.close()
    
    print(f"🧪 Testing endpoint with session_id: {session_id[:8]}...")
    print(f"📊 Expected chunk count: {expected_count}")
    print("📋 Sample data from database:")
    for row in sample_data:
        print(f"   Document: {row[0]}, Chunk #{row[1]}, Text Length: {row[2]} chars")
    
    # Test the API endpoint
    api_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{api_url}/rag/chunks/{session_id}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Endpoint responded successfully!")
            print(f"📊 Response structure:")
            print(f"   - chunks: {len(data.get('chunks', []))} items")
            print(f"   - total_count: {data.get('total_count', 0)}")
            print(f"   - session_id: {data.get('session_id', 'N/A')}")
            
            # Validate response structure
            chunks = data.get('chunks', [])
            if len(chunks) == expected_count:
                print(f"✅ Chunk count matches database: {len(chunks)}")
            else:
                print(f"⚠️  Chunk count mismatch - API: {len(chunks)}, DB: {expected_count}")
            
            # Show sample chunk data from API response
            print("📋 Sample chunks from API response:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"   Chunk {i + 1}:")
                print(f"      - document_name: {chunk.get('document_name', 'N/A')}")
                print(f"      - chunk_index: {chunk.get('chunk_index', 'N/A')}")
                print(f"      - chunk_text length: {len(chunk.get('chunk_text', ''))} chars")
                print(f"      - has_metadata: {chunk.get('chunk_metadata') is not None}")
            
            return True
            
        else:
            print(f"❌ Endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_invalid_session():
    """Test the endpoint with an invalid session ID"""
    print("\n🧪 Testing with invalid session ID...")
    
    api_url = "http://localhost:8000"
    invalid_session_id = "nonexistent-session-id"
    
    try:
        response = requests.get(f"{api_url}/rag/chunks/{invalid_session_id}")
        
        if response.status_code == 404:
            print("✅ Correctly returned 404 for invalid session ID")
            return True
        else:
            print(f"⚠️  Expected 404, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing the new /rag/chunks/{session_id} endpoint")
    print("=" * 50)
    
    # Test with valid session
    success1 = test_chunks_endpoint()
    
    # Test with invalid session
    success2 = test_invalid_session()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ All tests passed! The endpoint is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")