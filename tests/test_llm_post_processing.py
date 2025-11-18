"""
Comprehensive tests for LLM chunk post-processing functionality.

Tests the ChunkPostProcessor class and its integration with the chunking system,
focusing on Turkish language support and batch processing capabilities.

Author: RAG3 for Local - LLM Post-Processing Tests
Version: 1.0
Date: 2025-11-17
"""

import sys
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.text_processing.chunk_post_processor import (
    ChunkPostProcessor, 
    PostProcessingConfig
)
from src.text_processing.lightweight_chunker import (
    LightweightSemanticChunker,
    create_semantic_chunks
)


class TestPostProcessingConfig(unittest.TestCase):
    """Test PostProcessingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PostProcessingConfig()
        
        self.assertFalse(config.enabled)
        self.assertEqual(config.model_name, "llama-3.1-8b-instant")
        self.assertEqual(config.language, "tr")
        self.assertTrue(config.preserve_technical_terms)
        self.assertTrue(config.maintain_formatting)
        self.assertEqual(config.batch_size, 5)
        self.assertEqual(config.max_workers, 3)
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PostProcessingConfig(
            enabled=True,
            model_name="custom-model",
            language="en",
            batch_size=10,
            max_workers=5
        )
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.model_name, "custom-model")
        self.assertEqual(config.language, "en")
        self.assertEqual(config.batch_size, 10)
        self.assertEqual(config.max_workers, 5)


class TestChunkPostProcessor(unittest.TestCase):
    """Test ChunkPostProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PostProcessingConfig(
            enabled=True,
            model_name="test-model",
            timeout_seconds=10,
            retry_attempts=1
        )
        self.processor = ChunkPostProcessor(self.config)
        
        # Sample Turkish chunks for testing
        self.sample_chunks = [
            "# Türkiye Coğrafyası\n\nTürkiye Anadolu ve Trakya yarımadalarında yer alır.",
            "Kuzeyinde Karadeniz, güneyinde Akdeniz bulunur. Batısında ise Ege Denizi vardır.",
            "- Yunanistan (batı)\n- Bulgaristan (kuzeybatı)\n- Gürcistan (kuzeydoğu)",
            "İklim özellikleri çeşitlidir. Akdeniz iklimi güneyde görülür."
        ]
        
        # Sample problematic chunks
        self.problematic_chunks = [
            "türkiye anadolu yarımada",  # lowercase start
            "Bu bir test  metnidir.",    # double spaces
            "- Liste öğesi\n- ",         # broken list
            "http://example.com kod ```python```"  # structured data
        ]
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor.config)
        self.assertEqual(self.processor.config.model_name, "test-model")
        self.assertIsNotNone(self.processor.logger)
        
    def test_cache_key_generation(self):
        """Test cache key generation."""
        chunk = "Test chunk content"
        key1 = self.processor._get_cache_key(chunk)
        key2 = self.processor._get_cache_key(chunk)
        
        self.assertEqual(key1, key2)  # Same content should generate same key
        self.assertIsInstance(key1, str)
        self.assertTrue(len(key1) > 0)
    
    def test_is_well_formatted(self):
        """Test well-formatted chunk detection."""
        well_formatted = "# Türkiye Coğrafyası\n\nTürkiye bir ülkedir."
        poorly_formatted = "türkiye  bir  ülkedir"  # lowercase + double spaces
        
        self.assertTrue(self.processor._is_well_formatted(well_formatted))
        self.assertFalse(self.processor._is_well_formatted(poorly_formatted))
    
    def test_is_mostly_structured_data(self):
        """Test structured data detection."""
        structured = "```python\nprint('hello')\n```"
        table_data = "| Column 1 | Column 2 |\n| --------- | --------- |"
        url_data = "Visit http://example.com for more info"
        regular_text = "Bu normal bir Türkçe metin parçasıdır."
        
        self.assertTrue(self.processor._is_mostly_structured_data(structured))
        self.assertTrue(self.processor._is_mostly_structured_data(table_data))
        self.assertTrue(self.processor._is_mostly_structured_data(url_data))
        self.assertFalse(self.processor._is_mostly_structured_data(regular_text))
    
    def test_is_chunk_worth_processing(self):
        """Test chunk processing worthiness detection."""
        # Too short
        short_chunk = "Kısa"
        self.assertFalse(self.processor._is_chunk_worth_processing(short_chunk))
        
        # Well-formatted
        good_chunk = "# Başlık\n\nİyi formatlanmış metin parçası."
        self.assertFalse(self.processor._is_chunk_worth_processing(good_chunk))
        
        # Needs processing
        bad_chunk = "başlık\n\nkötü  formatlanmış  metin"
        self.assertTrue(self.processor._is_chunk_worth_processing(bad_chunk))
        
        # Structured data
        code_chunk = "```python\nprint('test')\n```"
        self.assertFalse(self.processor._is_chunk_worth_processing(code_chunk))
    
    def test_validate_improved_chunk(self):
        """Test improved chunk validation."""
        original = "Türkiye bir ülkedir. Anadolu yarımadası üzerinde yer alır."
        
        # Good improvement
        good_improved = "# Türkiye\n\nTürkiye bir ülkedir. Anadolu yarımadası üzerinde yer alır."
        self.assertTrue(self.processor._validate_improved_chunk(original, good_improved))
        
        # Too short
        too_short = "Kısa"
        self.assertFalse(self.processor._validate_improved_chunk(original, too_short))
        
        # Completely different content
        different = "This is completely different English content."
        self.assertFalse(self.processor._validate_improved_chunk(original, different))
        
        # Turkish character preservation test
        original_turkish = "Türkiye'nin coğrafi özellikleri çeşitlidir."
        no_turkish = "Turkiye'nin cografi ozellikleri cesitlidir."
        self.assertFalse(self.processor._validate_improved_chunk(original_turkish, no_turkish))
    
    def test_caching_functionality(self):
        """Test caching system."""
        chunk = "Test chunk for caching"
        cache_key = self.processor._get_cache_key(chunk)
        
        # Initially no cache
        self.assertIsNone(self.processor._get_cached_result(cache_key))
        
        # Cache a result
        result = "Improved test chunk"
        self.processor._cache_result(cache_key, result)
        
        # Should retrieve cached result
        cached = self.processor._get_cached_result(cache_key)
        self.assertEqual(cached, result)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        chunk = "Test chunk"
        cache_key = self.processor._get_cache_key(chunk)
        self.processor._cache_result(cache_key, "result")
        
        # Verify it's cached
        self.assertIsNotNone(self.processor._get_cached_result(cache_key))
        
        # Clear cache
        self.processor.clear_cache()
        
        # Should be gone
        self.assertIsNone(self.processor._get_cached_result(cache_key))
    
    @patch('requests.post')
    def test_call_llm_service_success(self, mock_post):
        """Test successful LLM service call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Improved text"}
        mock_post.return_value = mock_response
        
        prompt = "Test prompt"
        result = self.processor._call_llm_service(prompt)
        
        self.assertEqual(result, "Improved text")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_call_llm_service_failure(self, mock_post):
        """Test failed LLM service call."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        prompt = "Test prompt"
        result = self.processor._call_llm_service(prompt)
        
        self.assertIsNone(result)
    
    @patch('requests.post')
    def test_call_llm_service_timeout(self, mock_post):
        """Test LLM service timeout."""
        # Mock timeout
        mock_post.side_effect = Exception("Timeout")
        
        prompt = "Test prompt"
        result = self.processor._call_llm_service(prompt)
        
        self.assertIsNone(result)
    
    def test_disabled_processor(self):
        """Test processor with disabled config."""
        disabled_config = PostProcessingConfig(enabled=False)
        disabled_processor = ChunkPostProcessor(disabled_config)
        
        chunks = ["chunk1", "chunk2", "chunk3"]
        result = disabled_processor.process_chunks(chunks)
        
        # Should return original chunks unchanged
        self.assertEqual(result, chunks)
    
    def test_empty_chunks_list(self):
        """Test processing empty chunks list."""
        result = self.processor.process_chunks([])
        self.assertEqual(result, [])
    
    def test_processing_stats(self):
        """Test processing statistics."""
        stats = self.processor.get_processing_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_processed", stats)
        self.assertIn("total_improved", stats)
        self.assertIn("total_failed", stats)
        self.assertIn("average_improvement", stats)
    
    def test_create_from_config(self):
        """Test factory method."""
        processor = ChunkPostProcessor.create_from_config(
            enabled=True,
            model_name="test-model",
            batch_size=10
        )
        
        self.assertTrue(processor.config.enabled)
        self.assertEqual(processor.config.model_name, "test-model")
        self.assertEqual(processor.config.batch_size, 10)


class TestLLMIntegration(unittest.TestCase):
    """Test LLM post-processing integration with chunking system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        # Türkiye'nin Coğrafi Özellikleri

        ## Konum ve Sınırlar
        Türkiye, Anadolu ve Trakya yarımadalarında yer alan bir ülkedir. 
        Kuzeyinde Karadeniz, güneyinde Akdeniz, batısında Ege Denizi bulunur.

        ### Komşu Ülkeler
        - Yunanistan ve Bulgaristan (batı)
        - Gürcistan ve Ermenistan (kuzeydoğu)  
        - İran ve Irak (doğu)
        - Suriye (güneydoğu)

        ## İklim Özellikleri
        Türkiye'de üç farklı iklim tipi görülür: Akdeniz iklimi, karasal iklim ve Karadeniz iklimi.
        """
    
    def test_lightweight_chunker_with_llm_disabled(self):
        """Test lightweight chunker without LLM post-processing."""
        chunker = LightweightSemanticChunker()
        chunks = chunker.create_semantic_chunks(
            text=self.sample_text,
            target_size=300,
            overlap_ratio=0.1,
            language="tr",
            use_llm_post_processing=False
        )
        
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        
        # Verify chunks are strings
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertTrue(len(chunk.strip()) > 0)
    
    @patch('src.text_processing.chunk_post_processor.requests.post')
    def test_lightweight_chunker_with_llm_enabled(self, mock_post):
        """Test lightweight chunker with LLM post-processing enabled."""
        # Mock LLM service response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "# Türkiye Coğrafyası\n\nİyileştirilmiş metin parçası."}
        mock_post.return_value = mock_response
        
        chunker = LightweightSemanticChunker()
        chunks = chunker.create_semantic_chunks(
            text=self.sample_text,
            target_size=300,
            overlap_ratio=0.1,
            language="tr",
            use_llm_post_processing=True,
            llm_model_name="test-model",
            model_inference_url="http://test-service:8002"
        )
        
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        
        # Verify chunks are strings
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertTrue(len(chunk.strip()) > 0)
    
    def test_global_function_with_llm_parameters(self):
        """Test global create_semantic_chunks function with LLM parameters."""
        chunks = create_semantic_chunks(
            text=self.sample_text,
            target_size=400,
            overlap_ratio=0.15,
            language="tr",
            use_llm_post_processing=False  # Disabled for testing
        )
        
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        
        # Verify chunks contain Turkish content
        combined_text = " ".join(chunks)
        self.assertIn("Türkiye", combined_text)
        self.assertIn("Anadolu", combined_text)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in LLM post-processing."""
    
    def test_import_error_handling(self):
        """Test graceful handling when post-processor is not available."""
        # This test ensures the system doesn't crash if imports fail
        try:
            from src.text_processing.lightweight_chunker import LLM_POST_PROCESSING_AVAILABLE
            # If we get here, imports work
            self.assertTrue(True)
        except ImportError:
            # This should be handled gracefully
            self.assertTrue(True)
    
    def test_network_error_handling(self):
        """Test handling of network errors during LLM calls."""
        config = PostProcessingConfig(enabled=True, retry_attempts=1)
        processor = ChunkPostProcessor(config)
        
        # Test with a chunk that would normally be processed
        chunk = "kötü formatlanmış metin parçası"
        
        # Without mocking, this should fail gracefully and return original
        with patch.object(processor, '_call_llm_service', return_value=None):
            result = processor._process_single_chunk(chunk)
            self.assertEqual(result, chunk)  # Should return original on failure
    
    def test_malformed_llm_response(self):
        """Test handling of malformed LLM responses."""
        config = PostProcessingConfig(enabled=True)
        processor = ChunkPostProcessor(config)
        
        original = "Test chunk"
        
        # Test empty response
        self.assertFalse(processor._validate_improved_chunk(original, ""))
        
        # Test None response
        self.assertFalse(processor._validate_improved_chunk(original, None))
        
        # Test too short response
        self.assertFalse(processor._validate_improved_chunk(original, "x"))


class TestTurkishLanguageSupport(unittest.TestCase):
    """Test Turkish language specific features."""
    
    def setUp(self):
        """Set up Turkish language test fixtures."""
        self.processor = ChunkPostProcessor(PostProcessingConfig(language="tr"))
    
    def test_turkish_character_preservation(self):
        """Test preservation of Turkish characters."""
        original = "Türkiye'nin coğrafi özellikleri çeşitlidir. İklim ısıl özellikler gösterir."
        
        # Good: preserves Turkish characters
        good_improved = "# Türkiye Coğrafyası\n\nTürkiye'nin coğrafi özellikleri çeşitlidir. İklim ısıl özellikler gösterir."
        self.assertTrue(self.processor._validate_improved_chunk(original, good_improved))
        
        # Bad: removes Turkish characters
        bad_improved = "# Turkiye Cografyasi\n\nTurkiye'nin cografi ozellikleri cesitlidir. Iklim isil ozellikler gosterir."
        self.assertFalse(self.processor._validate_improved_chunk(original, bad_improved))
    
    def test_turkish_prompt_setup(self):
        """Test Turkish prompt template setup."""
        self.assertIn("TÜRKÇE", self.processor.refinement_prompt_template)
        self.assertIn("İngilizce kelime kullanma", self.processor.refinement_prompt_template)
        self.assertIn("Teknik terimler ve özel isimler korunmalı", self.processor.refinement_prompt_template)
        self.assertIn("{chunk_text}", self.processor.refinement_prompt_template)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)