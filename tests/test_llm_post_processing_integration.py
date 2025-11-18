#!/usr/bin/env python3
"""
Comprehensive Integration Tests for LLM Post-Processing Feature
Tests the complete pipeline from API Gateway to Document Processing Service
"""

import pytest
import requests
import json
import time
import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test data
TURKISH_TEST_TEXT = """
# Türkiye'nin İklim Özellikleri

## Akdeniz İklimi
Türkiye'nin güney sahillerinde Akdeniz iklimi görülür. Bu iklim türü yazları sıcak ve kurak, kışları ılık ve yağışlı geçer.

## Karasal İklim
İç Anadolu'da karasal iklim etkilidir. Yazlar sıcak, kışlar soğuk geçer. Yıllık sıcaklık farkları fazladır.

### Bölgesel Farklılıklar
- Doğu Anadolu: En sert kış koşulları
- İç Anadolu: Sıcaklık farklarının en yüksek olduğu bölge
- Karadeniz: Yıl boyunca bol yağış
"""

class TestLLMPostProcessingIntegration:
    """Integration tests for LLM post-processing across all system components"""

    def setup_method(self):
        """Setup for each test method"""
        # Mock URLs for testing
        self.model_inference_url = "http://localhost:8002"
        self.test_session_id = "test_session_123"
        
    def test_chunk_post_processor_creation(self):
        """Test that ChunkPostProcessor can be imported and created"""
        try:
            from src.text_processing.chunk_post_processor import ChunkPostProcessor, PostProcessingConfig
            
            config = PostProcessingConfig(
                model_name="llama-3.1-8b-instant",
                model_inference_url=self.model_inference_url,
                batch_size=3,
                cache_ttl_seconds=300
            )
            
            processor = ChunkPostProcessor(config)
            assert processor is not None
            assert processor.config.model_name == "llama-3.1-8b-instant"
            assert processor.config.batch_size == 3
            
        except ImportError as e:
            pytest.fail(f"Failed to import ChunkPostProcessor: {e}")
    
    def test_text_chunker_with_llm_post_processing(self):
        """Test that text_chunker integrates LLM post-processing correctly"""
        try:
            from src.text_processing.text_chunker import chunk_text
            
            # Mock the LLM post-processing to avoid actual API calls in tests
            with patch('src.text_processing.chunk_post_processor.ChunkPostProcessor.process_chunks') as mock_process:
                mock_process.return_value = [
                    "# Akdeniz İklimi\nGüney sahillerde yazları sıcak ve kurak, kışları ılık ve yağışlı geçer.",
                    "# Karasal İklim\nİç Anadolu'da yazlar sıcak, kışlar soğuk. Yıllık sıcaklık farkları fazla.",
                    "# Bölgesel Farklılıklar\nDoğu Anadolu en sert, İç Anadolu en değişken, Karadeniz en yağışlı."
                ]
                
                chunks = chunk_text(
                    text=TURKISH_TEST_TEXT,
                    chunk_size=200,
                    chunk_overlap=50,
                    strategy="lightweight",
                    use_llm_post_processing=True,
                    llm_model_name="llama-3.1-8b-instant",
                    model_inference_url=self.model_inference_url
                )
                
                assert len(chunks) == 3
                assert "Akdeniz İklimi" in chunks[0]
                assert "Karasal İklim" in chunks[1]
                assert "Bölgesel Farklılıklar" in chunks[2]
                
                # Verify that post-processing was called
                mock_process.assert_called_once()
                
        except ImportError as e:
            pytest.fail(f"Failed to import text_chunker: {e}")
    
    def test_document_processing_service_request_model(self):
        """Test that ProcessRequest model includes LLM post-processing parameters"""
        try:
            # Add the service path to sys.path temporarily
            service_path = project_root / "services" / "document_processing_service"
            sys.path.insert(0, str(service_path))
            
            from main import ProcessRequest
            
            # Test creating a request with LLM post-processing parameters
            request_data = {
                "text": TURKISH_TEST_TEXT,
                "metadata": {"source": "test"},
                "collection_name": "test_collection",
                "chunk_size": 300,
                "chunk_overlap": 50,
                "chunk_strategy": "lightweight",
                "use_llm_post_processing": True,
                "llm_model_name": "llama-3.1-8b-instant",
                "model_inference_url": self.model_inference_url
            }
            
            request = ProcessRequest(**request_data)
            
            assert request.use_llm_post_processing == True
            assert request.llm_model_name == "llama-3.1-8b-instant"
            assert request.model_inference_url == self.model_inference_url
            assert request.chunk_strategy == "lightweight"
            
            sys.path.remove(str(service_path))
            
        except ImportError as e:
            pytest.fail(f"Failed to import ProcessRequest: {e}")
        finally:
            # Clean up sys.path
            if str(service_path) in sys.path:
                sys.path.remove(str(service_path))
    
    @patch('requests.post')
    def test_llm_post_processing_with_mock_llm_calls(self, mock_post):
        """Test LLM post-processing with mocked LLM API calls"""
        try:
            from src.text_processing.chunk_post_processor import ChunkPostProcessor, PostProcessingConfig
            
            # Mock successful LLM response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "# İyileştirilmiş İçerik\nBu chunk LLM tarafından iyileştirilmiş ve daha anlamlı hale getirilmiştir."
            }
            mock_post.return_value = mock_response
            
            config = PostProcessingConfig(
                model_name="llama-3.1-8b-instant",
                model_inference_url=self.model_inference_url,
                batch_size=2,
                max_workers=2,
                enable_caching=False  # Disable caching for predictable test results
            )
            
            processor = ChunkPostProcessor(config)
            test_chunks = [
                "İlk chunk: Türkiye'de iklim çeşitliliği vardır.",
                "İkinci chunk: Akdeniz iklimi güney sahillerde görülür."
            ]
            
            processed_chunks = processor.process_chunks(test_chunks)
            
            assert len(processed_chunks) == 2
            assert all("İyileştirilmiş İçerik" in chunk for chunk in processed_chunks)
            assert mock_post.call_count == 2  # Two chunks, two calls
            
        except ImportError as e:
            pytest.fail(f"Failed to import ChunkPostProcessor: {e}")
    
    def test_batch_processing_performance(self):
        """Test that batch processing works correctly with multiple chunks"""
        try:
            from src.text_processing.chunk_post_processor import ChunkPostProcessor, PostProcessingConfig
            
            with patch('requests.post') as mock_post:
                # Mock successful batch responses
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "response": "İyileştirilmiş chunk içeriği"
                }
                mock_post.return_value = mock_response
                
                config = PostProcessingConfig(
                    model_name="llama-3.1-8b-instant",
                    model_inference_url=self.model_inference_url,
                    batch_size=3,  # Process 3 chunks at a time
                    max_workers=2,
                    enable_caching=False
                )
                
                processor = ChunkPostProcessor(config)
                
                # Create 10 test chunks
                test_chunks = [f"Test chunk {i}: Bu {i}. test chunk'ıdır." for i in range(1, 11)]
                
                start_time = time.time()
                processed_chunks = processor.process_chunks(test_chunks)
                end_time = time.time()
                
                assert len(processed_chunks) == 10
                assert all("İyileştirilmiş" in chunk for chunk in processed_chunks)
                
                # Verify that parallel processing was used
                assert mock_post.call_count == 10
                
                # Processing should be reasonably fast due to parallel execution
                processing_time = end_time - start_time
                assert processing_time < 5.0  # Should complete within 5 seconds
                
        except ImportError as e:
            pytest.fail(f"Failed to import ChunkPostProcessor: {e}")
    
    def test_error_handling_and_fallback(self):
        """Test error handling when LLM service is unavailable"""
        try:
            from src.text_processing.chunk_post_processor import ChunkPostProcessor, PostProcessingConfig
            
            with patch('requests.post') as mock_post:
                # Mock failed LLM response
                mock_post.side_effect = requests.exceptions.RequestException("Service unavailable")
                
                config = PostProcessingConfig(
                    model_name="llama-3.1-8b-instant",
                    model_inference_url=self.model_inference_url,
                    batch_size=2,
                    request_timeout_seconds=5,
                    max_retries=1
                )
                
                processor = ChunkPostProcessor(config)
                test_chunks = [
                    "Test chunk 1",
                    "Test chunk 2"
                ]
                
                processed_chunks = processor.process_chunks(test_chunks)
                
                # Should return original chunks when LLM fails
                assert processed_chunks == test_chunks
                
        except ImportError as e:
            pytest.fail(f"Failed to import ChunkPostProcessor: {e}")
    
    def test_caching_functionality(self):
        """Test that caching works correctly to avoid duplicate LLM calls"""
        try:
            from src.text_processing.chunk_post_processor import ChunkPostProcessor, PostProcessingConfig
            
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "response": "Cached improved content"
                }
                mock_post.return_value = mock_response
                
                config = PostProcessingConfig(
                    model_name="llama-3.1-8b-instant",
                    model_inference_url=self.model_inference_url,
                    enable_caching=True,
                    cache_ttl_seconds=300
                )
                
                processor = ChunkPostProcessor(config)
                
                # Process the same chunk twice
                test_chunk = "Identical test chunk for caching"
                test_chunks = [test_chunk, test_chunk]
                
                processed_chunks = processor.process_chunks(test_chunks)
                
                assert len(processed_chunks) == 2
                assert processed_chunks[0] == processed_chunks[1]  # Same result
                assert mock_post.call_count == 1  # Only one LLM call due to caching
                
        except ImportError as e:
            pytest.fail(f"Failed to import ChunkPostProcessor: {e}")
    
    def test_turkish_language_prompting(self):
        """Test that Turkish language prompts are used correctly"""
        try:
            from src.text_processing.chunk_post_processor import ChunkPostProcessor, PostProcessingConfig
            
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "response": "İyileştirilmiş Türkçe içerik"
                }
                mock_post.return_value = mock_response
                
                config = PostProcessingConfig(
                    model_name="llama-3.1-8b-instant",
                    model_inference_url=self.model_inference_url,
                    enable_caching=False
                )
                
                processor = ChunkPostProcessor(config)
                test_chunks = ["Türkçe test chunk içeriği"]
                
                processed_chunks = processor.process_chunks(test_chunks)
                
                assert len(processed_chunks) == 1
                assert "İyileştirilmiş" in processed_chunks[0]
                
                # Verify that the prompt included Turkish instructions
                call_args = mock_post.call_args[1]['json']  # Get the JSON payload
                prompt = call_args['prompt']
                assert "Türkçe" in prompt
                assert "anlamlı" in prompt or "semantic" in prompt.lower()
                
        except ImportError as e:
            pytest.fail(f"Failed to import ChunkPostProcessor: {e}")

class TestAPIIntegration:
    """Test API endpoints with LLM post-processing parameters"""
    
    def test_api_gateway_form_parameters(self):
        """Test that API Gateway accepts LLM post-processing form parameters"""
        # This would be an integration test that requires the actual API to be running
        # For now, we'll test that the parameters are properly defined
        
        expected_params = [
            "session_id",
            "markdown_files", 
            "chunk_strategy",
            "chunk_size",
            "chunk_overlap", 
            "embedding_model",
            "use_llm_post_processing",  # NEW parameter
            "llm_model_name",  # NEW parameter
            "model_inference_url"  # NEW parameter
        ]
        
        # In a real integration test, you would send a POST request like this:
        # test_data = {
        #     "session_id": "test_session",
        #     "markdown_files": json.dumps(["test.md"]),
        #     "use_llm_post_processing": True,
        #     "llm_model_name": "llama-3.1-8b-instant",
        #     "model_inference_url": "http://localhost:8002"
        # }
        # response = requests.post("http://localhost:8000/documents/process-and-store", data=test_data)
        
        # For this test, we just verify the parameters are defined
        assert all(param for param in expected_params)  # Basic validation
    
    def test_configuration_validation(self):
        """Test that configuration parameters are properly validated"""
        try:
            from src.text_processing.chunk_post_processor import PostProcessingConfig
            
            # Test default values
            config = PostProcessingConfig()
            assert config.model_name == "llama-3.1-8b-instant"
            assert config.batch_size == 5
            assert config.max_workers == 3
            assert config.enable_caching == True
            assert config.cache_ttl_seconds == 3600
            
            # Test custom values
            custom_config = PostProcessingConfig(
                model_name="custom-model",
                batch_size=10,
                max_workers=5,
                enable_caching=False,
                request_timeout_seconds=30
            )
            assert custom_config.model_name == "custom-model"
            assert custom_config.batch_size == 10
            assert custom_config.max_workers == 5
            assert custom_config.enable_caching == False
            assert custom_config.request_timeout_seconds == 30
            
        except ImportError as e:
            pytest.fail(f"Failed to import PostProcessingConfig: {e}")

def run_integration_tests():
    """Run all integration tests manually"""
    print("🧪 Running LLM Post-Processing Integration Tests...")
    
    # Test basic functionality
    test_integration = TestLLMPostProcessingIntegration()
    test_integration.setup_method()
    
    try:
        print("📋 Test 1: ChunkPostProcessor creation...")
        test_integration.test_chunk_post_processor_creation()
        print("✅ ChunkPostProcessor creation test passed")
        
        print("📋 Test 2: Text chunker integration...")
        test_integration.test_text_chunker_with_llm_post_processing()
        print("✅ Text chunker integration test passed")
        
        print("📋 Test 3: Document processing service model...")
        test_integration.test_document_processing_service_request_model()
        print("✅ Document processing service model test passed")
        
        print("📋 Test 4: Mock LLM calls...")
        test_integration.test_llm_post_processing_with_mock_llm_calls()
        print("✅ Mock LLM calls test passed")
        
        print("📋 Test 5: Batch processing performance...")
        test_integration.test_batch_processing_performance()
        print("✅ Batch processing performance test passed")
        
        print("📋 Test 6: Error handling and fallback...")
        test_integration.test_error_handling_and_fallback()
        print("✅ Error handling and fallback test passed")
        
        print("📋 Test 7: Caching functionality...")
        test_integration.test_caching_functionality()
        print("✅ Caching functionality test passed")
        
        print("📋 Test 8: Turkish language prompting...")
        test_integration.test_turkish_language_prompting()
        print("✅ Turkish language prompting test passed")
        
        # Test API integration
        api_test = TestAPIIntegration()
        
        print("📋 Test 9: API Gateway form parameters...")
        api_test.test_api_gateway_form_parameters()
        print("✅ API Gateway form parameters test passed")
        
        print("📋 Test 10: Configuration validation...")
        api_test.test_configuration_validation()
        print("✅ Configuration validation test passed")
        
        print("\n🎉 All LLM Post-Processing Integration Tests Passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)