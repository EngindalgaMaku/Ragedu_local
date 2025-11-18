"""
LLM Chunk Post-Processor with Grok API Optimization

This module provides Grok API-optimized LLM post-processing with:
1. Rate limiting protection
2. Batch processing with delays
3. Retry mechanisms with exponential backoff
4. Turkish language optimization
5. Fail-safe fallback handling

Author: RAG3 for Local - Grok API Edition
Version: 2.0
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
import json


@dataclass
class PostProcessingConfig:
    """Configuration for LLM post-processing with Grok API optimization."""
    
    # LLM settings
    enabled: bool = False
    model_name: str = "llama-3.1-8b-instant"
    model_inference_url: str = "http://model-inference-service:8002"
    temperature: float = 0.3
    max_tokens: int = 800
    
    # Grok API Rate Limiting
    requests_per_minute: int = 30  # Conservative limit for Grok API
    min_request_interval: float = 2.0  # Minimum 2 seconds between requests
    rate_limit_buffer: float = 0.5  # Extra delay for safety
    
    # Batch Processing
    batch_size: int = 3  # Process 3 chunks at a time (conservative)
    batch_delay: float = 5.0  # 5 second delay between batches
    max_workers: int = 1  # Sequential processing for Grok API
    
    # Retry settings
    timeout_seconds: int = 45  # Longer timeout for Grok
    retry_attempts: int = 3
    retry_backoff_multiplier: float = 2.0
    retry_initial_delay: float = 2.0
    
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
    
    # Error handling
    max_consecutive_failures: int = 3  # Stop after 3 consecutive failures
    failure_cooldown_seconds: float = 10.0


class RateLimiter:
    """Rate limiter for Grok API calls."""
    
    def __init__(self, requests_per_minute: int, min_interval: float = 2.0):
        self.requests_per_minute = requests_per_minute
        self.min_interval = min_interval
        self.last_request_time = 0.0
        self.request_count = 0
        self.minute_start = time.time()
        
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        current_time = time.time()
        
        # Check if we've started a new minute
        if current_time - self.minute_start >= 60:
            self.minute_start = current_time
            self.request_count = 0
        
        # Check requests per minute limit
        if self.request_count >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start)
            if wait_time > 0:
                logging.info(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.minute_start = time.time()
                self.request_count = 0
        
        # Check minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)
        
        # Update tracking
        self.last_request_time = time.time()
        self.request_count += 1


class GrokChunkPostProcessor:
    """
    Grok API-optimized chunk post-processor with rate limiting and batch processing.
    """
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        self.config = config or PostProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Detect provider based on model name
        self.is_ollama = 'llama3:' in self.config.model_name.lower() or ':' in self.config.model_name
        self.provider_name = "Ollama" if self.is_ollama else "Grok API"
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.requests_per_minute,
            min_interval=self.config.min_request_interval + self.config.rate_limit_buffer
        )
        
        # Performance caching
        self._cache: Dict[str, Tuple[str, datetime]] = {}
        self._processing_stats = {
            "total_processed": 0,
            "total_improved": 0,
            "total_failed": 0,
            "total_cached": 0,
            "total_skipped": 0,
            "average_improvement": 0.0,
            "total_api_calls": 0,
            "total_wait_time": 0.0
        }
        
        # Failure tracking
        self.consecutive_failures = 0
        
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
8. SADECE iyileştirilmiş metni döndür, açıklama yapma.

YAPILACAKLAR:
- Bölünmüş cümleleri tamamla
- Liste öğelerini düzelt
- Başlık hiyerarşisini koru
- Gereksiz boşlukları temizle
- Cümle bağlantılarını iyileştir

METİN:
{chunk_text}

İYİLEŞTİRİLMİŞ METİN (sadece metin, açıklama yok):"""

    def _is_chunk_worth_processing(self, chunk_text: str) -> bool:
        """Determine if a chunk needs LLM processing."""
        
        # Skip very short chunks
        if len(chunk_text.strip()) < 50:
            self.logger.debug("Skipping chunk: too short")
            return False
            
        # Skip if already well-formatted
        if self._is_well_formatted(chunk_text):
            self.logger.debug("Skipping chunk: already well-formatted")
            return False
            
        # Skip if contains mostly code or structured data
        if self._is_mostly_structured_data(chunk_text):
            self.logger.debug("Skipping chunk: mostly structured data")
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
        """Check if chunk contains mostly structured data."""
        
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
                self._processing_stats["total_cached"] += 1
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

    def _call_grok_api(self, prompt: str) -> Optional[str]:
        """
        Call Grok API via model inference service with rate limiting.
        """
        
        # Apply rate limiting
        wait_start = time.time()
        self.rate_limiter.wait_if_needed()
        wait_time = time.time() - wait_start
        self._processing_stats["total_wait_time"] += wait_time
        self._processing_stats["total_api_calls"] += 1
        
        try:
            payload = {
                "prompt": prompt,
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            self.logger.debug(f"🌐 Calling {self.provider_name} via {self.config.model_inference_url}")
            
            response = requests.post(
                f"{self.config.model_inference_url}/models/generate",
                json=payload,
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                if response_text:
                    self.consecutive_failures = 0  # Reset failure counter
                    return response_text
                else:
                    self.logger.warning("Grok API returned empty response")
                    return None
                    
            elif response.status_code == 429:
                # Rate limit exceeded
                self.logger.warning(f"🚫 Grok API rate limit exceeded (429)")
                retry_after = int(response.headers.get('Retry-After', 60))
                self.logger.info(f"⏳ Waiting {retry_after}s before retry...")
                time.sleep(retry_after)
                return None
                
            else:
                self.logger.warning(f"Grok API error: {response.status_code}")
                self.logger.warning(f"Response: {response.text[:500]}")  # Show more of error message
                self.logger.warning(f"Request URL: {self.config.model_inference_url}/models/generate")
                self.logger.warning(f"Request model: {self.config.model_name}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning(f"⏰ Grok API timeout after {self.config.timeout_seconds}s")
            self.logger.warning(f"Request URL: {self.config.model_inference_url}/models/generate")
            return None
        except requests.exceptions.ConnectionError as e:
            self.logger.warning(f"🔌 Grok API connection failed: {e}")
            self.logger.warning(f"Could not connect to: {self.config.model_inference_url}")
            self.logger.warning(f"Please ensure model-inference-service is running!")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"🔌 Grok API request failed: {e}")
            self.logger.warning(f"Request URL: {self.config.model_inference_url}/models/generate")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in Grok API call: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            return None

    def _process_single_chunk(self, chunk_text: str, chunk_index: int = 0) -> str:
        """Process a single chunk with Grok API and smart retry."""
        
        # Check cache first
        cache_key = self._get_cache_key(chunk_text)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.logger.debug(f"✅ Cache hit for chunk {chunk_index}")
            return cached_result
            
        # Check if chunk needs processing
        if not self._is_chunk_worth_processing(chunk_text):
            self._processing_stats["total_skipped"] += 1
            return chunk_text
        
        # Check if we've had too many consecutive failures
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.logger.warning(f"⚠️ Too many consecutive failures ({self.consecutive_failures}). Cooling down...")
            time.sleep(self.config.failure_cooldown_seconds)
            self.consecutive_failures = 0  # Reset after cooldown
            
        # Attempt Grok API processing with exponential backoff
        for attempt in range(self.config.retry_attempts):
            try:
                prompt = self.refinement_prompt_template.format(chunk_text=chunk_text)
                improved_chunk = self._call_grok_api(prompt)
                
                if improved_chunk and len(improved_chunk.strip()) > 0:
                    # Clean LLM response first
                    improved_chunk = self._clean_llm_response(improved_chunk)
                    
                    # Validate improved chunk
                    if self._validate_improved_chunk(chunk_text, improved_chunk):
                        self._cache_result(cache_key, improved_chunk)
                        self._processing_stats["total_improved"] += 1
                        self.logger.debug(f"✨ Chunk {chunk_index} improved successfully")
                        return improved_chunk
                        
                # If improvement failed, apply exponential backoff
                if attempt < self.config.retry_attempts - 1:
                    backoff_delay = self.config.retry_initial_delay * (self.config.retry_backoff_multiplier ** attempt)
                    self.logger.debug(f"⏳ Retry {attempt + 1} failed. Waiting {backoff_delay:.1f}s...")
                    time.sleep(backoff_delay)
                    
            except Exception as e:
                self.logger.warning(f"Chunk processing attempt {attempt + 1} failed: {e}")
                
        # If all attempts failed, return original
        self.consecutive_failures += 1
        self._processing_stats["total_failed"] += 1
        self.logger.debug(f"⚠️ Chunk {chunk_index} processing failed after {self.config.retry_attempts} attempts")
        return chunk_text

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response by removing common English prefixes/suffixes."""
        
        cleaned = response.strip()
        
        # Remove common English prefixes (case-insensitive)
        english_prefixes = [
            "here is the improved text:",
            "here is the edited text:",
            "here is the refined text:",
            "here is the improved version:",
            "here is the edited text in turkish:",
            "improved text:",
            "edited text:",
            "refined text:",
        ]
        
        for prefix in english_prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Remove common English suffixes and notes
        english_suffixes = [
            "let me know if you need further improvements",
            "is there anything else",
        ]
        
        for suffix in english_suffixes:
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[:-(len(suffix))].strip()
                break
        
        # Remove parenthetical English notes at the end (e.g., "(Note: I only edited...)")
        import re
        # Match parentheses containing "note:" or common English phrases at the end
        note_pattern = r'\s*\((?:note:|i only|this is|here|without changing).*?\)\s*$'
        cleaned = re.sub(note_pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove leading/trailing quotes if present
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        
        return cleaned
    
    def _validate_improved_chunk(self, original: str, improved: str) -> bool:
        """Validate that the improved chunk is actually better."""
        
        # Basic sanity checks (text already cleaned by caller)
        if not improved or len(improved.strip()) < 10:
            return False
            
        # Check size constraints
        if len(improved) > self.config.max_chunk_size:
            self.logger.debug("Validation failed: chunk too large")
            return False
            
        # Ensure content wasn't completely changed
        original_words = set(original.lower().split())
        improved_words = set(improved.lower().split())
        
        # At least 60% word overlap
        overlap = len(original_words & improved_words) / len(original_words) if original_words else 0
        if overlap < 0.6:
            self.logger.debug(f"Validation failed: insufficient word overlap ({overlap:.2%})")
            return False
            
        # Check for Turkish language preservation
        if self.config.language == "tr":
            # Ensure Turkish characters are preserved if present
            turkish_chars = set("çğıöşüÇĞIİÖŞÜ")
            orig_turkish = any(c in original for c in turkish_chars)
            impr_turkish = any(c in improved for c in turkish_chars)
            
            if orig_turkish and not impr_turkish:
                self.logger.debug("Validation failed: Turkish characters lost")
                return False
                
        return True

    def process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Process multiple chunks with Grok API batch processing and rate limiting.
        """
        
        if not self.config.enabled:
            self.logger.info("LLM post-processing disabled")
            return chunks
            
        if not chunks:
            return chunks
            
        self.logger.info(f"🚀 Starting {self.provider_name} post-processing for {len(chunks)} chunks")
        self.logger.info(f"📋 Config: batch_size={self.config.batch_size}, batch_delay={self.config.batch_delay}s, rate_limit={self.config.requests_per_minute}/min")
        start_time = time.time()
        
        # Initialize stats
        self._processing_stats["total_processed"] += len(chunks)
        
        try:
            # Process chunks in batches with delays
            processed_chunks = []
            
            for batch_start in range(0, len(chunks), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(chunks))
                batch = chunks[batch_start:batch_end]
                batch_num = (batch_start // self.config.batch_size) + 1
                total_batches = (len(chunks) + self.config.batch_size - 1) // self.config.batch_size
                
                self.logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                
                # Process batch (sequential for Grok API)
                batch_results = []
                for i, chunk in enumerate(batch):
                    global_index = batch_start + i
                    result = self._process_single_chunk(chunk, global_index)
                    batch_results.append(result)
                
                processed_chunks.extend(batch_results)
                
                # Delay between batches (except after last batch)
                if batch_end < len(chunks):
                    self.logger.info(f"⏸️  Batch delay: waiting {self.config.batch_delay}s...")
                    time.sleep(self.config.batch_delay)
                
            # Calculate improvement statistics
            improved_count = sum(1 for orig, proc in zip(chunks, processed_chunks) if orig != proc)
            improvement_ratio = improved_count / len(chunks) if chunks else 0
            
            processing_time = time.time() - start_time
            
            # Log comprehensive stats
            stats = self.get_processing_stats()
            self.logger.info(f"""
✅ {self.provider_name} post-processing completed!
   📊 Results: {improved_count}/{len(chunks)} chunks improved ({improvement_ratio:.1%})
   ⏱️  Time: {processing_time:.1f}s (avg: {processing_time/len(chunks):.2f}s/chunk)
   💾 Cache hits: {stats['total_cached']}
   ⏭️  Skipped: {stats['total_skipped']}
   🌐 API calls: {stats['total_api_calls']}
   ⏳ Wait time: {stats['total_wait_time']:.1f}s
   ❌ Failures: {stats['total_failed']}
            """)
            
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"❌ {self.provider_name} post-processing failed: {e}")
            # Return original chunks on any error
            return chunks

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._processing_stats.copy()
        
    def clear_cache(self):
        """Clear processing cache."""
        self._cache.clear()
        self.logger.info("Processing cache cleared")

    @staticmethod
    def create_for_grok(
        enabled: bool = True,
        model_name: str = "llama-3.1-8b-instant",
        model_inference_url: str = "http://model-inference-service:8002",
        batch_size: int = 3,
        requests_per_minute: int = 30,
        **kwargs
    ) -> 'GrokChunkPostProcessor':
        """Create Grok-optimized ChunkPostProcessor."""
        
        config = PostProcessingConfig(
            enabled=enabled,
            model_name=model_name,
            model_inference_url=model_inference_url,
            batch_size=batch_size,
            requests_per_minute=requests_per_minute,
            **kwargs
        )
        
        return GrokChunkPostProcessor(config)


# Backward compatibility - export both classes
ChunkPostProcessor = GrokChunkPostProcessor


