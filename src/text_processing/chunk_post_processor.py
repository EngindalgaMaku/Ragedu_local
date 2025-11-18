"""
LLM Chunk Post-Processor for Semantic Refinement

This module provides optional LLM post-processing to improve chunk quality by:
1. Removing redundant information
2. Improving text flow and coherence  
3. Better formatting of lists, headers, and technical content
4. Maintaining Turkish language quality
5. Preserving technical terms and proper nouns

The system is designed to be completely optional and fail-safe.

Author: RAG3 for Local - LLM Post-Processing Feature
Version: 1.0
Date: 2025-11-17
"""

import hashlib
import logging
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class PostProcessingConfig:
    """Configuration for LLM post-processing of chunks."""
    
    # LLM settings
    enabled: bool = False
    model_name: str = "llama-3.1-8b-instant"
    model_inference_url: str = "http://model-inference-service:8002"
    temperature: float = 0.3
    max_tokens: int = 800
    
    # Processing settings
    batch_size: int = 5
    max_workers: int = 3
    timeout_seconds: int = 30
    retry_attempts: int = 2
    
    # Quality control
    min_improvement_threshold: float = 0.1
    max_chunk_size: int = 2048
    preserve_structure: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    cache_max_size: int = 1000
    
    # Language settings
    language: str = "tr"
    preserve_technical_terms: bool = True
    maintain_formatting: bool = True


class ChunkPostProcessor:
    """
    LLM-powered chunk post-processor for semantic refinement and formatting.
    
    This class provides optional LLM post-processing to improve chunk quality by:
    1. Removing redundant information
    2. Improving text flow and coherence  
    3. Better formatting of lists, headers, and technical content
    4. Maintaining Turkish language quality
    5. Preserving technical terms and proper nouns
    
    The system is designed to be completely optional and fail-safe.
    """
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        self.config = config or PostProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance caching
        self._cache: Dict[str, Tuple[str, datetime]] = {}
        self._processing_stats = {
            "total_processed": 0,
            "total_improved": 0,
            "total_failed": 0,
            "average_improvement": 0.0
        }
        
        # Turkish-optimized prompts
        self._setup_turkish_prompts()
        
    def _setup_turkish_prompts(self):
        """Setup Turkish-optimized prompts for chunk refinement."""
        
        self.refinement_prompt_template = """Sen bir Türkçe metin editörüsün. Görevin, verilen metin parçacığını anlamsal olarak iyileştirmek ve daha iyi formatlamak.

KATI KURALLAR:
1. KESINLIKLE TÜRKÇE cevap ver. İngilizce kelime kullanma.
2. İçeriğin ANLAMINI DEĞİŞTİRME. Sadece kaliteyi artır.
3. Teknik terimler ve özel isimler korunmalı.
4. Liste formatları ve başlıklar düzeltilsin.
5. Gereksiz tekrarlar kaldırılsın.
6. Cümle akışı iyileştirilsin.
7. Markdown formatını koru ve iyileştir.

YAPILACAKLAR:
- Bölünmüş cümleleri tamamla
- Liste öğelerini düzelt
- Başlık hiyerarşisini koru
- Gereksiz boşlukları temizle
- Cümle bağlantılarını iyileştir

METİN:
{chunk_text}

İYİLEŞTİRİLMİŞ METİN:"""

        self.quality_check_prompt = """Bu metin parçacığının kalitesini 1-10 arasında değerlendir:

METİN:
{chunk_text}

Değerlendirme kriterleri:
- Anlam bütünlüğü (1-3 puan)
- Cümle akışı (1-3 puan) 
- Biçimsel doğruluk (1-2 puan)
- Teknik doğruluk (1-2 puan)

Sadece sayısal skor ver (örn: 7.5):"""

    def _is_chunk_worth_processing(self, chunk_text: str) -> bool:
        """Determine if a chunk needs LLM processing."""
        
        # Skip very short chunks
        if len(chunk_text.strip()) < 50:
            return False
            
        # Skip if already well-formatted
        if self._is_well_formatted(chunk_text):
            return False
            
        # Skip if contains mostly code or structured data
        if self._is_mostly_structured_data(chunk_text):
            return False
            
        return True
        
    def _is_well_formatted(self, chunk_text: str) -> bool:
        """Check if chunk is already well-formatted."""
        
        # Check for common formatting issues
        issues = 0
        
        # Broken sentences (ends with lowercase)
        if chunk_text.strip() and chunk_text.strip()[-1].islower():
            issues += 1
            
        # Multiple consecutive spaces
        if "  " in chunk_text:
            issues += 1
            
        # Broken list items
        if "- \n" in chunk_text or "* \n" in chunk_text:
            issues += 1
            
        # Malformed headers
        if chunk_text.count("#") > 0 and not any(line.startswith("#") for line in chunk_text.split("\n")):
            issues += 1
            
        return issues < 2
        
    def _is_mostly_structured_data(self, chunk_text: str) -> bool:
        """Check if chunk contains mostly structured data that shouldn't be modified."""
        
        structured_indicators = [
            "```",  # Code blocks
            "| ",   # Tables
            "http",  # URLs
            "@",     # Emails
            "{",     # JSON-like structures
        ]
        
        return any(indicator in chunk_text for indicator in structured_indicators)

    def _get_cache_key(self, chunk_text: str) -> str:
        """Generate cache key for chunk."""
        return hashlib.md5(f"{chunk_text}{self.config.model_name}".encode()).hexdigest()
        
    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """Get cached result if available and not expired."""
        if not self.config.enable_caching:
            return None
            
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=self.config.cache_ttl_hours):
                return result
            else:
                del self._cache[cache_key]
                
        return None
        
    def _cache_result(self, cache_key: str, result: str):
        """Cache processing result."""
        if not self.config.enable_caching:
            return
            
        # Limit cache size
        if len(self._cache) >= self.config.cache_max_size:
            # Remove oldest entries
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[:len(sorted_items) // 2]:
                del self._cache[key]
                
        self._cache[cache_key] = (result, datetime.now())

    def _call_llm_service(self, prompt: str) -> Optional[str]:
        """Call LLM service for chunk processing."""
        
        try:
            payload = {
                "prompt": prompt,
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            response = requests.post(
                f"{self.config.model_inference_url}/models/generate",
                json=payload,
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                self.logger.warning(f"LLM service error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning("LLM service timeout")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"LLM service request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM call: {e}")
            return None

    def _process_single_chunk(self, chunk_text: str) -> str:
        """Process a single chunk with LLM refinement."""
        
        # Check cache first
        cache_key = self._get_cache_key(chunk_text)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        # Check if chunk needs processing
        if not self._is_chunk_worth_processing(chunk_text):
            return chunk_text
            
        # Attempt LLM processing with retries
        for attempt in range(self.config.retry_attempts):
            try:
                prompt = self.refinement_prompt_template.format(chunk_text=chunk_text)
                improved_chunk = self._call_llm_service(prompt)
                
                if improved_chunk and len(improved_chunk.strip()) > 0:
                    # Validate improved chunk
                    if self._validate_improved_chunk(chunk_text, improved_chunk):
                        self._cache_result(cache_key, improved_chunk)
                        self._processing_stats["total_improved"] += 1
                        return improved_chunk
                        
                # If improvement failed, try next attempt
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    
            except Exception as e:
                self.logger.warning(f"Chunk processing attempt {attempt + 1} failed: {e}")
                
        # If all attempts failed, return original
        self._processing_stats["total_failed"] += 1
        return chunk_text

    def _validate_improved_chunk(self, original: str, improved: str) -> bool:
        """Validate that the improved chunk is actually better."""
        
        # Basic sanity checks
        if not improved or len(improved.strip()) < 10:
            return False
            
        # Check size constraints
        if len(improved) > self.config.max_chunk_size:
            return False
            
        # Ensure content wasn't completely changed
        original_words = set(original.lower().split())
        improved_words = set(improved.lower().split())
        
        # At least 60% word overlap
        overlap = len(original_words & improved_words) / len(original_words) if original_words else 0
        if overlap < 0.6:
            return False
            
        # Check for Turkish language preservation
        if self.config.language == "tr":
            # Ensure Turkish characters are preserved if present
            turkish_chars = set("çğıöşüÇĞIİÖŞÜ")
            orig_turkish = any(c in original for c in turkish_chars)
            impr_turkish = any(c in improved for c in turkish_chars)
            
            if orig_turkish and not impr_turkish:
                return False
                
        return True

    def process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Process multiple chunks with LLM post-processing.
        
        Args:
            chunks: List of chunk texts to process
            
        Returns:
            List of processed chunks (same length as input)
        """
        
        if not self.config.enabled:
            return chunks
            
        if not chunks:
            return chunks
            
        self.logger.info(f"🔄 Starting LLM post-processing for {len(chunks)} chunks")
        start_time = time.time()
        
        # Initialize stats
        self._processing_stats["total_processed"] += len(chunks)
        
        try:
            # Process chunks in batches
            processed_chunks = []
            
            if self.config.max_workers > 1:
                # Parallel processing
                processed_chunks = self._process_chunks_parallel(chunks)
            else:
                # Sequential processing
                processed_chunks = self._process_chunks_sequential(chunks)
                
            # Calculate improvement statistics
            improved_count = sum(1 for orig, proc in zip(chunks, processed_chunks) if orig != proc)
            improvement_ratio = improved_count / len(chunks) if chunks else 0
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ LLM post-processing completed: {improved_count}/{len(chunks)} chunks improved in {processing_time:.2f}s")
            
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"❌ LLM post-processing failed: {e}")
            # Return original chunks on any error
            return chunks

    def _process_chunks_sequential(self, chunks: List[str]) -> List[str]:
        """Process chunks sequentially."""
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            processed_chunk = self._process_single_chunk(chunk)
            processed_chunks.append(processed_chunk)
            
        return processed_chunks

    def _process_chunks_parallel(self, chunks: List[str]) -> List[str]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        
        processed_chunks = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks for processing
            future_to_index = {
                executor.submit(self._process_single_chunk, chunk): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    processed_chunk = future.result()
                    processed_chunks[index] = processed_chunk
                except Exception as e:
                    self.logger.warning(f"Parallel processing failed for chunk {index}: {e}")
                    processed_chunks[index] = chunks[index]  # Fallback to original
                    
        return processed_chunks

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._processing_stats.copy()
        
    def clear_cache(self):
        """Clear processing cache."""
        self._cache.clear()
        self.logger.info("Processing cache cleared")

    @staticmethod
    def create_from_config(
        enabled: bool = False,
        model_name: str = "llama-3.1-8b-instant",
        model_inference_url: str = "http://model-inference-service:8002",
        batch_size: int = 5,
        **kwargs
    ) -> 'ChunkPostProcessor':
        """Create ChunkPostProcessor with custom configuration."""
        
        config = PostProcessingConfig(
            enabled=enabled,
            model_name=model_name,
            model_inference_url=model_inference_url,
            batch_size=batch_size,
            **kwargs
        )
        
        return ChunkPostProcessor(config)