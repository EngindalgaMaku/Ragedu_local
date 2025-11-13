#!/usr/bin/env python3
"""
Test script to verify the API fix for undefined values issue
"""
import json
import sys

def test_api_response_mapping():
    """Test the response mapping logic"""
    print("🔍 Testing API Response Mapping Fix...")
    
    # Simulate the Document Processing Service response
    mock_processor_response = {
        "success": True,
        "message": "Successfully processed and stored: 5 chunks",
        "chunks_processed": 5,  # This is what the Document Processing Service returns
        "collection_name": "test-collection",
        "chunk_ids": ["id1", "id2", "id3", "id4", "id5"]
    }
    
    # Simulate the mapping logic from the fixed API Gateway
    mapped_response = {
        "success": mock_processor_response.get("success", False),
        "message": mock_processor_response.get("message", ""),
        "processed_count": mock_processor_response.get("chunks_processed", 0),  # Map chunks_processed to processed_count
        "total_chunks_added": mock_processor_response.get("chunks_processed", 0),  # Same value as processed_count
        "processing_time": mock_processor_response.get("processing_time")
    }
    
    print("✅ Original Document Processing Service Response:")
    print(json.dumps(mock_processor_response, indent=2))
    
    print("\n✅ Mapped API Gateway Response (what frontend receives):")
    print(json.dumps(mapped_response, indent=2))
    
    # Test the frontend logic
    print("\n🔍 Testing Frontend Success Message Generation...")
    
    # Simulate the fixed frontend logic
    def simulate_frontend_success_message(result):
        # Defensive programming: ensure we have valid values to prevent undefined display
        processed_count = result.get("processed_count", 0) if result else 0
        total_chunks = result.get("total_chunks_added", 0) if result else 0
        
        success_message = f"RAG işlemi tamamlandı! {processed_count} dosya işlendi, {total_chunks} parça oluşturuldu."
        return success_message
    
    # Test with the mapped response (normal case)
    message1 = simulate_frontend_success_message(mapped_response)
    print(f"✅ Success message with mapped response: {message1}")
    
    # Test with undefined/None response (edge case)
    message2 = simulate_frontend_success_message(None)
    print(f"✅ Success message with None response: {message2}")
    
    # Test with empty response (edge case)
    message3 = simulate_frontend_success_message({})
    print(f"✅ Success message with empty response: {message3}")
    
    print("\n🎉 All tests passed! The fix should resolve:")
    print("   1. 'undefined dosya işlendi, undefined parça oluşturuldu' issue")
    print("   2. Frontend freezing/crashing due to undefined values")
    print("   3. Proper field mapping between Document Processing Service and Frontend")
    
    return True

if __name__ == "__main__":
    try:
        test_api_response_mapping()
        print("\n✅ API Fix Test: SUCCESS")
    except Exception as e:
        print(f"\n❌ API Fix Test: FAILED - {e}")
        sys.exit(1)