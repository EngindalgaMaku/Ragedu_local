#!/usr/bin/env python3
"""
Comprehensive test to verify all RAG fixes are working properly:
1. Field mapping fix (undefined values resolved)
2. Session metadata updating
3. Chunk retrieval and display
4. Document count accuracy
"""

import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

def test_document_upload_and_processing():
    """Test document upload with field mapping and session metadata updates."""
    print("🧪 Testing Document Upload and Processing...")
    
    # First try to list existing sessions to see if we can use one
    print("📝 Checking for existing sessions...")
    list_response = requests.get(f"{API_BASE_URL}/sessions")
    
    if list_response.status_code == 200:
        sessions = list_response.json()
        # Look for our test session
        test_session = None
        for session in sessions:
            if session.get("name") == "Test Session - Comprehensive Fixes" and session.get("created_by") == "test_user":
                test_session = session
                break
        
        if test_session:
            session_id = test_session["session_id"]
            print(f"✅ Found existing session: {session_id}")
        else:
            # Create with unique name
            import time
            unique_name = f"Test Session - Comprehensive Fixes - {int(time.time())}"
            print(f"📝 Creating new session: {unique_name}")
            session_response = requests.post(f"{API_BASE_URL}/sessions", json={
                "name": unique_name,
                "description": "Testing all fixes including field mapping and metadata",
                "category": "biology",
                "created_by": "test_user",
                "grade_level": "9",
                "subject_area": "Biology"
            })
            
            if session_response.status_code != 200:
                print(f"❌ Failed to create session: {session_response.status_code}")
                if session_response.text:
                    try:
                        error_detail = session_response.json()
                        print(f"Error detail: {error_detail}")
                    except:
                        print(f"Response text: {session_response.text}")
                return None
                
            session_data = session_response.json()
            session_id = session_data["session_id"]
            print(f"✅ Created session: {session_id}")
    else:
        print(f"❌ Failed to list sessions: {list_response.status_code}")
        return None
    
    # Check initial session metadata
    print("🔍 Checking initial session metadata...")
    session_info_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}")
    if session_info_response.status_code == 200:
        initial_session = session_info_response.json()
        print(f"📊 Initial document_count: {initial_session.get('document_count', 'undefined')}")
        print(f"📊 Initial total_chunks: {initial_session.get('total_chunks', 'undefined')}")
    
    # Create test markdown content
    test_markdown_content = """# Biyoloji ve Canlıların Ortak Özellikleri

Canlılar birçok ortak özelliğe sahiptir:

## Temel Özellikler

1. **Hücresel Yapı**: Tüm canlılar hücrelerden oluşur
2. **Metabolizma**: Enerji üretimi ve kullanımı
3. **Büyüme ve Gelişme**: Boyut ve karmaşıklık artışı
4. **Üreme**: Neslin devamı için gerekli
5. **Uyarılabilirlik**: Çevresel değişimlere tepki
6. **Hareket**: Aktif veya pasif yer değiştirme
7. **Homeostasis**: İç dengenin korunması

## Sonuç

Bu özellikler canlıları cansızlardan ayıran temel kriterlerdir. Biyoloji bilimi bu özellikleri inceler.
"""
    
    try:
        # We need to use the API Gateway's process-and-store endpoint to trigger our fixes
        # But first we need to save the markdown file so it can be processed
        
        print("📤 Preparing markdown content for processing...")
        markdown_filename = "test_biology_content.md"
        
        # First, we need to save the markdown file to the cloud storage system
        # We'll simulate this by creating a test markdown file that the system can read
        
        # Let's try a different approach: use the API Gateway's process-and-store
        # but provide the content as if it came from markdown files
        try:
            print("📤 Processing through API Gateway (with field mapping fixes)...")
            
            # The API Gateway expects markdown_files to reference files that exist
            # Let's create the payload in the format that the API Gateway expects
            data = {
                'session_id': session_id,
                'markdown_files': json.dumps([markdown_filename]),  # Reference to files
                'chunk_strategy': 'semantic',
                'chunk_size': 500,
                'chunk_overlap': 50,
                'embedding_model': 'mixedbread-ai/mxbai-embed-large-v1'
            }
            
            # This will likely fail because we don't have the markdown files saved
            # But let's see what happens and then we can diagnose
            upload_response = requests.post(
                f"{API_BASE_URL}/documents/process-and-store",
                data=data,
                timeout=120
            )
            
            print(f"Response status: {upload_response.status_code}")
            if upload_response.status_code != 200:
                print(f"API Gateway failed: {upload_response.text}")
                
                # Fallback to direct Document Processing Service to test basic functionality
                print("📤 Falling back to direct Document Processing Service call...")
                payload = {
                    'session_id': session_id,
                    'text': test_markdown_content,
                    'metadata': {
                        'session_id': session_id,
                        'source_files': [markdown_filename],
                        'embedding_model': 'mixedbread-ai/mxbai-embed-large-v1',
                        'chunk_strategy': 'semantic'
                    },
                    'collection_name': f'session_{session_id}',
                    'chunk_size': 500,
                    'chunk_overlap': 50
                }
                
                upload_response = requests.post(
                    f"{API_BASE_URL.replace('8000', '8003')}/process-and-store",
                    json=payload,
                    timeout=60
                )
                print("⚠️ Used direct service call - field mapping and session metadata fixes will not be tested")
            
        except Exception as e:
            print(f"❌ All approaches failed: {str(e)}")
            return session_id
        
        if upload_response.status_code != 200:
            print(f"❌ Upload failed with status: {upload_response.status_code}")
            if upload_response.text:
                print(f"Response: {upload_response.text}")
            return session_id
            
        result = upload_response.json()
        print("✅ Document uploaded successfully!")
        
        # Check for the field mapping fixes
        print("\n🔍 Testing Field Mapping Fixes...")
        print(f"📊 processed_count: {result.get('processed_count', 'undefined')}")
        print(f"📊 total_chunks_added: {result.get('total_chunks_added', 'undefined')}")
        
        # The original fields should still exist for backward compatibility
        print(f"📊 chunks_processed: {result.get('chunks_processed', 'undefined')}")
        
        # Verify no undefined values in critical fields
        if result.get('processed_count') is not None and result.get('total_chunks_added') is not None:
            print("✅ Field mapping fix working - no undefined values!")
        else:
            print("❌ Field mapping fix failed - still seeing undefined values")
            
        return session_id
        
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        return session_id
        
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        return session_id

def test_session_metadata_updates(session_id):
    """Test if session metadata is properly updated after processing."""
    if not session_id:
        print("⏭️ Skipping metadata test - no session ID")
        return
        
    print("\n🔍 Testing Session Metadata Updates...")
    
    # Wait a moment for processing to complete
    time.sleep(2)
    
    # Check updated session metadata
    session_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}")
    if session_response.status_code != 200:
        print(f"❌ Failed to get session metadata: {session_response.status_code}")
        return
        
    session_data = session_response.json()
    document_count = session_data.get('document_count', 0)
    total_chunks = session_data.get('total_chunks', 0)
    
    print(f"📊 Updated document_count: {document_count}")
    print(f"📊 Updated total_chunks: {total_chunks}")
    
    if document_count > 0 and total_chunks > 0:
        print("✅ Session metadata update fix working!")
    else:
        print("❌ Session metadata update fix failed")

def test_chunks_retrieval(session_id):
    """Test if chunks can be retrieved from the session."""
    if not session_id:
        print("⏭️ Skipping chunks test - no session ID")
        return
        
    print("\n🔍 Testing Chunks Retrieval...")
    
    # Test the new chunks endpoint
    chunks_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/chunks")
    if chunks_response.status_code != 200:
        print(f"❌ Failed to retrieve chunks: {chunks_response.status_code}")
        print(f"Response: {chunks_response.text}")
        return
        
    chunks_data = chunks_response.json()
    chunks = chunks_data.get('chunks', [])
    
    print(f"📊 Retrieved {len(chunks)} chunks")
    
    if len(chunks) > 0:
        print("✅ Chunks retrieval fix working!")
        # Show first chunk as example
        first_chunk = chunks[0]
        print(f"📝 Sample chunk: {first_chunk.get('content', '')[:100]}...")
    else:
        print("❌ Chunks retrieval fix failed - no chunks found")

def test_frontend_defensive_programming():
    """Test that frontend won't crash with undefined values."""
    print("\n🔍 Testing Frontend Defensive Programming...")
    
    # This would typically require browser automation
    # For now, we'll verify our nullish coalescing fixes by checking the session page exists
    try:
        frontend_response = requests.get(FRONTEND_URL)
        if frontend_response.status_code == 200:
            print("✅ Frontend is accessible")
            print("📝 Note: Manual testing required to verify nullish coalescing operators")
        else:
            print(f"❌ Frontend not accessible: {frontend_response.status_code}")
    except Exception as e:
        print(f"⚠️ Could not test frontend: {str(e)}")

def main():
    print("🚀 Starting Comprehensive RAG Fixes Test")
    print("=" * 50)
    
    # Test 1: Document upload with field mapping
    session_id = test_document_upload_and_processing()
    
    # Test 2: Session metadata updates
    test_session_metadata_updates(session_id)
    
    # Test 3: Chunks retrieval
    test_chunks_retrieval(session_id)
    
    # Test 4: Frontend defensive programming
    test_frontend_defensive_programming()
    
    print("\n" + "=" * 50)
    print("🏁 Comprehensive test completed!")
    
    if session_id:
        print(f"🔗 Test session created: {session_id}")
        print(f"🌐 View in frontend: {FRONTEND_URL}/sessions/{session_id}")

if __name__ == "__main__":
    main()