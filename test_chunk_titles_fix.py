#!/usr/bin/env python3
"""
Test script to validate that chunk titles are now properly extracted from metadata
after the metadata key mismatch fix.
"""

import requests
import json
import uuid

# Test configuration
DOCUMENT_PROCESSING_URL = "http://localhost:8003"  # Fixed port - service runs on 8003
session_id = "c4cbd4689a4d9268e983430a38aa13fb"  # Use existing session from logs

def test_chunks_endpoint():
    """Test that chunks now show proper document names instead of 'Unknown'"""
    print("🧪 Testing /sessions/{session_id}/chunks endpoint...")
    
    url = f"{DOCUMENT_PROCESSING_URL}/sessions/{session_id}/chunks"
    print(f"📞 Calling: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            chunks = data.get("chunks", [])
            
            print(f"✅ Retrieved {len(chunks)} chunks")
            
            if chunks:
                print("\n📋 Chunk Analysis:")
                unknown_count = 0
                named_count = 0
                
                for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                    document_name = chunk.get("document_name", "N/A")
                    chunk_metadata = chunk.get("chunk_metadata", {})
                    
                    print(f"  Chunk {i+1}:")
                    print(f"    📄 Document Name: '{document_name}'")
                    
                    if document_name == "Unknown":
                        unknown_count += 1
                        print("    ⚠️  Still showing as 'Unknown'")
                        print(f"    🔍 Raw metadata keys: {list(chunk_metadata.keys())}")
                        
                        # Look for source file info in metadata
                        for key in ["source_files", "source_file", "filename", "document_name", "file_name"]:
                            if key in chunk_metadata:
                                print(f"    🎯 Found '{key}': {chunk_metadata[key]}")
                    else:
                        named_count += 1
                        print(f"    ✅ Successfully extracted document name!")
                    
                    print()
                
                print(f"📊 Results Summary:")
                print(f"   ✅ Named chunks: {named_count}")
                print(f"   ⚠️  Unknown chunks: {unknown_count}")
                print(f"   📈 Success rate: {named_count}/{len(chunks)} ({100*named_count/len(chunks):.1f}%)")
                
                if unknown_count == 0:
                    print("\n🎉 SUCCESS: All chunks now have proper document names!")
                    return True
                else:
                    print(f"\n❌ ISSUE: {unknown_count} chunks still show as 'Unknown'")
                    return False
            else:
                print("⚠️  No chunks found for this session")
                return False
                
        else:
            print(f"❌ Failed to get chunks: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing chunks endpoint: {e}")
        return False

def test_process_new_document():
    """Process a new document to test the complete flow"""
    print("\n🧪 Testing complete document processing flow...")
    
    # Create test document content
    test_content = """# Test Dökümanı

Bu bir test dökümanıdır. Metadata anahtarı düzeltmesini test etmek için oluşturulmuştur.

## Bölüm 1: Giriş

Test içeriği burada yer almaktadır. Bu bölümde basit bir açıklama bulunmaktadır.

## Bölüm 2: Detaylar

Daha detaylı bilgiler bu bölümde sunulmaktadır.
"""
    
    # Create new session ID for this test
    test_session_id = str(uuid.uuid4()).replace("-", "")
    test_filename = "test_metadata_fix.md"
    
    # Process document
    payload = {
        "text": test_content,
        "metadata": {
            "session_id": test_session_id,
            "source_file": test_filename,  # This is the key that was causing issues
            "filename": test_filename,
            "embedding_model": "mxbai-embed-large",
            "chunk_strategy": "semantic"
        },
        "collection_name": f"session_{test_session_id}",
        "chunk_size": 500,
        "chunk_overlap": 50
    }
    
    print(f"📤 Processing document with session_id: {test_session_id}")
    print(f"📄 Source file: {test_filename}")
    
    try:
        # Process the document
        response = requests.post(
            f"{DOCUMENT_PROCESSING_URL}/process-and-store",
            json=payload,
            timeout=120
        )
        
        print(f"📊 Process Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Processing successful: {result.get('chunks_processed', 0)} chunks created")
            
            # Now test the chunks endpoint for the new session
            print(f"\n📞 Testing chunks endpoint for new session...")
            chunks_response = requests.get(
                f"{DOCUMENT_PROCESSING_URL}/sessions/{test_session_id}/chunks",
                timeout=30
            )
            
            if chunks_response.status_code == 200:
                chunks_data = chunks_response.json()
                chunks = chunks_data.get("chunks", [])
                
                print(f"✅ Retrieved {len(chunks)} chunks from new session")
                
                # Check if document names are properly set
                success_count = 0
                for i, chunk in enumerate(chunks):
                    document_name = chunk.get("document_name", "N/A")
                    print(f"  Chunk {i+1}: '{document_name}'")
                    
                    if document_name != "Unknown" and test_filename in document_name:
                        success_count += 1
                
                if success_count == len(chunks):
                    print(f"\n🎉 PERFECT: All {len(chunks)} chunks show correct document name!")
                    return True
                else:
                    print(f"\n⚠️  Partial success: {success_count}/{len(chunks)} chunks correctly named")
                    return False
            else:
                print(f"❌ Failed to retrieve chunks: {chunks_response.status_code}")
                return False
        else:
            print(f"❌ Processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error in document processing test: {e}")
        return False

def main():
    print("🔧 CHUNK TITLE METADATA FIX VALIDATION")
    print("=" * 50)
    
    # Test 1: Check existing chunks
    print("TEST 1: Checking existing session chunks")
    existing_test_passed = test_chunks_endpoint()
    
    # Test 2: Process new document and verify
    print("\n" + "="*50)
    print("TEST 2: Processing new document to verify complete flow")
    new_test_passed = test_process_new_document()
    
    # Final results
    print("\n" + "="*50)
    print("📊 FINAL TEST RESULTS:")
    print(f"   Existing chunks test: {'✅ PASSED' if existing_test_passed else '❌ FAILED'}")
    print(f"   New document test: {'✅ PASSED' if new_test_passed else '❌ FAILED'}")
    
    if existing_test_passed and new_test_passed:
        print("\n🎉 ALL TESTS PASSED! Chunk titles metadata fix is working correctly!")
    elif new_test_passed:
        print("\n✅ New processing works! Old data may need reprocessing.")
    else:
        print("\n❌ Issues detected. Metadata fix may need additional work.")

if __name__ == "__main__":
    main()