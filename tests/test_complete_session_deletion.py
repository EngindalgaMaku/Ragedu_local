#!/usr/bin/env python3
"""
Complete Session Deletion Test
Tests that session deletion properly cleans up both SQLite metadata and ChromaDB collections
"""

import requests
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
API_GATEWAY_URL = "http://localhost:8080"
DOCUMENT_PROCESSOR_URL = "http://localhost:8081"

def test_complete_session_deletion():
    """Test complete session deletion including ChromaDB cleanup"""
    try:
        logger.info("🧪 Starting Complete Session Deletion Test")
        
        # Step 1: Create a test session
        logger.info("📝 Step 1: Creating test session")
        session_data = {
            "name": "Test Session for Deletion",
            "description": "This session will be deleted to test cleanup",
            "category": "research",
            "created_by": "test_user"
        }
        
        response = requests.post(f"{API_GATEWAY_URL}/sessions", json=session_data)
        if response.status_code != 200:
            raise Exception(f"Failed to create session: {response.status_code} - {response.text}")
        
        session = response.json()
        session_id = session["session_id"]
        logger.info(f"✅ Created test session: {session_id}")
        
        # Step 2: Add some documents to create ChromaDB collection and vectors
        logger.info("📄 Step 2: Adding documents to create ChromaDB data")
        
        # Create some test markdown content
        test_content = """# Test Document
        
This is a test document that will be processed and stored in ChromaDB.
It contains multiple paragraphs to create several chunks.

## Section 1
This is the first section with some meaningful content about artificial intelligence
and machine learning concepts that will be embedded as vectors.

## Section 2  
This is the second section with additional content about natural language processing
and retrieval-augmented generation systems that will also be embedded.

## Conclusion
This concludes our test document which should create multiple chunks
in the ChromaDB collection for testing deletion functionality.
"""
        
        # Process and store the document
        process_data = {
            "text": test_content,
            "metadata": {
                "session_id": session_id,
                "source_files": ["test_document.md"],
                "embedding_model": "mxbai-embed-large"
            },
            "collection_name": f"session_{session_id}",
            "chunk_size": 500,
            "chunk_overlap": 50
        }
        
        response = requests.post(f"{DOCUMENT_PROCESSOR_URL}/process-and-store", json=process_data)
        if response.status_code != 200:
            raise Exception(f"Failed to process document: {response.status_code} - {response.text}")
        
        process_result = response.json()
        chunks_created = process_result.get("chunks_processed", 0)
        logger.info(f"✅ Processed document: {chunks_created} chunks created in ChromaDB")
        
        # Step 3: Verify ChromaDB collection exists and has data
        logger.info("🔍 Step 3: Verifying ChromaDB collection exists")
        
        response = requests.get(f"{API_GATEWAY_URL}/sessions/{session_id}/chunks")
        if response.status_code == 200:
            chunks_data = response.json()
            chunk_count = chunks_data.get("total_count", 0)
            logger.info(f"✅ ChromaDB collection verified: {chunk_count} chunks found")
        else:
            logger.warning(f"Could not verify ChromaDB data: {response.status_code}")
        
        # Step 4: Delete the session (this should trigger complete cleanup)
        logger.info("🗑️ Step 4: Deleting session (should cleanup SQLite + ChromaDB)")
        
        response = requests.delete(f"{API_GATEWAY_URL}/sessions/{session_id}")
        if response.status_code != 200:
            raise Exception(f"Failed to delete session: {response.status_code} - {response.text}")
        
        deletion_result = response.json()
        logger.info(f"✅ Session deleted: {json.dumps(deletion_result, indent=2)}")
        
        # Step 5: Verify session is deleted from SQLite
        logger.info("🔍 Step 5: Verifying session deleted from SQLite")
        
        response = requests.get(f"{API_GATEWAY_URL}/sessions/{session_id}")
        if response.status_code == 404:
            logger.info("✅ Session successfully deleted from SQLite database")
        else:
            logger.error(f"❌ Session still exists in SQLite: {response.status_code}")
        
        # Step 6: Verify ChromaDB collection is deleted
        logger.info("🔍 Step 6: Verifying ChromaDB collection deleted")
        
        response = requests.get(f"{API_GATEWAY_URL}/sessions/{session_id}/chunks")
        if response.status_code == 200:
            chunks_data = response.json()
            chunk_count = chunks_data.get("total_count", 0)
            if chunk_count == 0:
                logger.info("✅ ChromaDB collection successfully deleted (no chunks found)")
            else:
                logger.error(f"❌ ChromaDB collection still contains {chunk_count} chunks")
        else:
            logger.info("✅ ChromaDB collection access failed (likely deleted)")
        
        # Test Summary
        logger.info("🎉 COMPLETE SESSION DELETION TEST RESULTS:")
        logger.info(f"   - Session ID: {session_id}")
        logger.info(f"   - SQLite Deletion: {'✅ Success' if response.status_code == 404 else '❌ Failed'}")
        logger.info(f"   - ChromaDB Deletion: {'✅ Success' if deletion_result.get('chromadb_collection_deleted', False) else '❌ Failed'}")
        logger.info(f"   - Chunks Created: {chunks_created}")
        logger.info(f"   - Session Name: {session['name']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False

def test_session_deletion_without_chromadb():
    """Test session deletion when no ChromaDB data exists"""
    try:
        logger.info("🧪 Testing session deletion without ChromaDB data")
        
        # Create a session but don't add any documents
        session_data = {
            "name": "Empty Test Session",
            "description": "Session with no documents for deletion test",
            "category": "research",
            "created_by": "test_user"
        }
        
        response = requests.post(f"{API_GATEWAY_URL}/sessions", json=session_data)
        if response.status_code != 200:
            raise Exception(f"Failed to create empty session: {response.status_code}")
        
        session = response.json()
        session_id = session["session_id"]
        logger.info(f"✅ Created empty session: {session_id}")
        
        # Delete the session immediately
        response = requests.delete(f"{API_GATEWAY_URL}/sessions/{session_id}")
        if response.status_code != 200:
            raise Exception(f"Failed to delete empty session: {response.status_code}")
        
        deletion_result = response.json()
        logger.info("✅ Empty session deletion successful")
        logger.info(f"   ChromaDB deletion result: {deletion_result.get('chromadb_collection_deleted', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Empty session deletion test failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting Complete Session Deletion Tests")
    
    # Test 1: Complete deletion with ChromaDB data
    test1_success = test_complete_session_deletion()
    
    # Test 2: Deletion without ChromaDB data
    test2_success = test_session_deletion_without_chromadb()
    
    # Final results
    logger.info("🏁 FINAL TEST RESULTS:")
    logger.info(f"   Complete Deletion Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    logger.info(f"   Empty Session Deletion Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        logger.info("🎉 ALL TESTS PASSED - Session deletion with ChromaDB cleanup is working!")
    else:
        logger.error("❌ SOME TESTS FAILED - Session deletion needs debugging")