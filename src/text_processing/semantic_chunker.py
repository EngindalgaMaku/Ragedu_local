"""
Advanced Semantic Chunking Module with Embedding-based Analysis - Phase 1.

This module provides state-of-the-art chunking capabilities using embedding models
to analyze semantic boundaries, topic coherence, and natural text structure for
optimal chunk creation. Enhanced with Turkish language support and adaptive
boundary detection.

Features:
- Embedding-based semantic coherence analysis
- Adaptive boundary detection using sentence similarity
- Enhanced Turkish sentence boundary patterns
- Cross-chunk relationship analysis
- Performance optimizations with caching
"""

from typing import List, Dict, Optional, Tuple, Union
import re
import numpy as np
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

# Core dependencies
try:
    # Try relative imports first (when used as package)
    from ..config import get_config
    from ..utils.logger import get_logger
except ImportError:
    # Fallback to absolute imports (for testing and standalone use)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import get_config
    from src.utils.logger import get_logger

# Advanced dependencies for Phase 1
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import nltk
    from cachetools import LRUCache
    import psutil
    EMBEDDING_SUPPORT = True
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
except ImportError as e:
    EMBEDDING_SUPPORT = False
    print(f"Warning: Advanced semantic chunking features not available: {e}")


@dataclass
class SemanticChunk:
    """Enhanced chunk with semantic metadata."""
    text: str
    start_index: int
    end_index: int
    sentence_count: int
    word_count: int
    coherence_score: float
    topic_consistency: float
    embedding_vector: Optional[np.ndarray] = None
    keywords: Optional[List[str]] = None
    language: str = "auto"


@dataclass
class ChunkBoundary:
    """Semantic boundary information."""
    position: int
    sentence_index: int
    similarity_score: float
    boundary_strength: float
    is_natural_break: bool
    reason: str

class AdvancedSemanticChunker:
    """
    Advanced semantic chunker with embedding-based analysis and Turkish language support.
    
    This implementation uses embedding models to analyze semantic coherence, detect
    natural topic boundaries, and create chunks with optimal semantic integrity.
    Features adaptive boundary detection and cross-chunk relationship analysis.
    
    Key Features:
    - Embedding-based semantic similarity analysis
    - Adaptive chunk boundary detection
    - Turkish language optimizations
    - Performance caching and memory management
    - Cross-chunk coherence evaluation
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.config = get_config()
        self.logger = get_logger(__name__, self.config)
        
        # Flexible character limits for chunks
        self.min_chunk_size = 150
        self.max_chunk_size = 1024
        self.target_coherence_threshold = 0.75
        
        # Enhanced sentence boundary patterns for Turkish
        self.sentence_boundary_patterns = {
            'turkish': re.compile(r'(?<=[.!?…])\s+(?=[A-ZÇĞIŞÖÜ])|(?<=\.)\s+(?=\d)|(?<=[.!?])\s*\n+\s*(?=[A-ZÇĞIŞÖÜ])'),
            'english': re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\.)\s+(?=\d)|(?<=[.!?])\s*\n+\s*(?=[A-Z])'),
            'general': re.compile(r'(?<=[.!?…])\s+')
        }
        
        # Turkish specific patterns
        self.turkish_sentence_starters = re.compile(r'^(Bu|Şu|O|Bunlar|Şunlar|Onlar|Böyle|Şöyle|Öyle|Ancak|Fakat|Ama|Lakin|Ayrıca|Dahası|Sonuç olarak|Bu nedenle)\s+', re.IGNORECASE)
        self.turkish_abbreviations = {'Dr.', 'Prof.', 'Doç.', 'Yrd.', 'vs.', 'vb.', 'örn.', 'yak.', 'yakl.', 'krş.', 'bkz.'}
        
        # Initialize embedding model and cache
        self.embedding_model = None
        self.embedding_cache = LRUCache(maxsize=1000)  # Cache for performance
        self.sentence_cache = LRUCache(maxsize=500)
        
        if EMBEDDING_SUPPORT:
            try:
                self._initialize_embedding_model(embedding_model)
            except Exception as e:
                self.logger.warning(f"Failed to initialize embedding model: {e}")
                EMBEDDING_SUPPORT = False
        
        if not EMBEDDING_SUPPORT:
            self.logger.warning("Falling back to pattern-based chunking without embeddings")
            
    def _initialize_embedding_model(self, model_name: str):
        """Initialize the sentence transformer model with error handling."""
        try:
            self.logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def create_semantic_chunks(
        self,
        text: str,
        target_size: int = 512,
        overlap_ratio: float = 0.1,
        language: str = "auto",
        use_embedding_analysis: bool = True
    ) -> List[SemanticChunk]:
        """
        Creates advanced semantic chunks using embedding-based analysis.
        
        This method combines traditional sentence-based chunking with embedding analysis
        to identify optimal semantic boundaries and maintain topic coherence.
        
        Args:
            text: The input text to be chunked
            target_size: The desired character length for each chunk
            overlap_ratio: The percentage of overlap between consecutive chunks
            language: Language of the text ("tr", "en", or "auto")
            use_embedding_analysis: Whether to use embedding-based boundary detection
            
        Returns:
            A list of SemanticChunk objects with enhanced metadata
        """
        if not text or not text.strip():
            return []
            
        self.logger.info(f"Starting advanced semantic chunking for text of length {len(text)} (embedding_analysis: {use_embedding_analysis})")
        
        # Language detection and sentence splitting
        detected_language = self._detect_language(text) if language == "auto" else language
        sentences = self._split_into_sentences(text, detected_language)
        
        if not sentences:
            self.logger.warning("No sentences found in the provided text")
            return []
            
        # Choose chunking strategy based on embedding support and user preference
        if use_embedding_analysis and EMBEDDING_SUPPORT and self.embedding_model:
            return self._create_embedding_based_chunks(sentences, target_size, overlap_ratio, detected_language)
        else:
            return self._create_traditional_chunks(sentences, target_size, overlap_ratio, detected_language)
    
    def _create_embedding_based_chunks(
        self,
        sentences: List[str],
        target_size: int,
        overlap_ratio: float,
        language: str
    ) -> List[SemanticChunk]:
        """Create chunks using embedding-based semantic boundary detection."""
        
        self.logger.info("Using embedding-based semantic chunking")
        
        # Step 1: Generate sentence embeddings
        sentence_embeddings = self._get_sentence_embeddings(sentences)
        
        # Step 2: Detect semantic boundaries
        semantic_boundaries = self._detect_semantic_boundaries(sentences, sentence_embeddings, target_size)
        
        # Step 3: Create chunks based on boundaries
        chunks = []
        current_start = 0
        
        for boundary in semantic_boundaries:
            chunk_end = boundary.sentence_index
            
            # Create chunk
            chunk_sentences = sentences[current_start:chunk_end + 1]
            chunk_text = " ".join(chunk_sentences)
            
            # Calculate chunk metrics
            coherence_score = self._calculate_chunk_coherence(
                sentence_embeddings[current_start:chunk_end + 1]
            )
            
            chunk = SemanticChunk(
                text=chunk_text,
                start_index=current_start,
                end_index=chunk_end,
                sentence_count=len(chunk_sentences),
                word_count=len(chunk_text.split()),
                coherence_score=coherence_score,
                topic_consistency=boundary.similarity_score,
                embedding_vector=np.mean(sentence_embeddings[current_start:chunk_end + 1], axis=0),
                keywords=self._extract_chunk_keywords(chunk_text),
                language=language
            )
            
            chunks.append(chunk)
            
            # Calculate overlap for next chunk
            if overlap_ratio > 0:
                overlap_sentences = int((chunk_end - current_start + 1) * overlap_ratio)
                current_start = max(current_start + 1, chunk_end + 1 - overlap_sentences)
            else:
                current_start = chunk_end + 1
                
            if current_start >= len(sentences):
                break
                
        # Handle remaining sentences
        if current_start < len(sentences):
            remaining_sentences = sentences[current_start:]
            remaining_text = " ".join(remaining_sentences)
            
            if len(remaining_text) >= self.min_chunk_size:
                remaining_embeddings = sentence_embeddings[current_start:]
                coherence_score = self._calculate_chunk_coherence(remaining_embeddings)
                
                final_chunk = SemanticChunk(
                    text=remaining_text,
                    start_index=current_start,
                    end_index=len(sentences) - 1,
                    sentence_count=len(remaining_sentences),
                    word_count=len(remaining_text.split()),
                    coherence_score=coherence_score,
                    topic_consistency=0.8,  # Default for final chunk
                    embedding_vector=np.mean(remaining_embeddings, axis=0),
                    keywords=self._extract_chunk_keywords(remaining_text),
                    language=language
                )
                chunks.append(final_chunk)
            else:
                # Merge with previous chunk if too small
                if chunks:
                    last_chunk = chunks[-1]
                    merged_text = last_chunk.text + " " + remaining_text
                    merged_embeddings = np.concatenate([
                        sentence_embeddings[last_chunk.start_index:last_chunk.end_index + 1],
                        sentence_embeddings[current_start:]
                    ])
                    
                    chunks[-1] = SemanticChunk(
                        text=merged_text,
                        start_index=last_chunk.start_index,
                        end_index=len(sentences) - 1,
                        sentence_count=last_chunk.sentence_count + len(remaining_sentences),
                        word_count=len(merged_text.split()),
                        coherence_score=self._calculate_chunk_coherence(merged_embeddings),
                        topic_consistency=last_chunk.topic_consistency,
                        embedding_vector=np.mean(merged_embeddings, axis=0),
                        keywords=self._extract_chunk_keywords(merged_text),
                        language=language
                    )
        
        self.logger.info(f"Generated {len(chunks)} embedding-based semantic chunks")
        return chunks
    
    def _create_traditional_chunks(
        self,
        sentences: List[str],
        target_size: int,
        overlap_ratio: float,
        language: str
    ) -> List[SemanticChunk]:
        """Create chunks using traditional sentence-based method as fallback."""
        
        self.logger.info("Using traditional sentence-based chunking")
        
        chunks = []
        current_sentence_index = 0
        num_sentences = len(sentences)
        
        while current_sentence_index < num_sentences:
            chunk_start_index = current_sentence_index
            chunk_end_index = chunk_start_index
            current_chunk_char_count = 0
            
            # Greedily add sentences to form a chunk
            while chunk_end_index < num_sentences:
                sentence_len = len(sentences[chunk_end_index])
                
                if current_chunk_char_count > 0 and (current_chunk_char_count + sentence_len) > self.max_chunk_size:
                    break
                
                current_chunk_char_count += sentence_len + 1
                chunk_end_index += 1
                
                if current_chunk_char_count >= target_size:
                    break
            
            # Extend if too small
            while current_chunk_char_count < self.min_chunk_size and chunk_end_index < num_sentences:
                current_chunk_char_count += len(sentences[chunk_end_index]) + 1
                chunk_end_index += 1
            
            # Create chunk
            chunk_sentences = sentences[chunk_start_index:chunk_end_index]
            chunk_text = " ".join(chunk_sentences)
            
            if chunk_text:
                chunk = SemanticChunk(
                    text=chunk_text,
                    start_index=chunk_start_index,
                    end_index=chunk_end_index - 1,
                    sentence_count=len(chunk_sentences),
                    word_count=len(chunk_text.split()),
                    coherence_score=0.7,  # Default score for traditional chunking
                    topic_consistency=0.6,  # Default score
                    embedding_vector=None,
                    keywords=self._extract_chunk_keywords(chunk_text),
                    language=language
                )
                chunks.append(chunk)
            
            # Calculate overlap
            if chunk_end_index >= num_sentences:
                break
                
            num_sentences_in_chunk = chunk_end_index - chunk_start_index
            overlap_sentence_count = int(num_sentences_in_chunk * overlap_ratio)
            
            if overlap_ratio > 0 and overlap_sentence_count == 0 and num_sentences_in_chunk > 1:
                overlap_sentence_count = 1
            
            next_sentence_index = chunk_end_index - overlap_sentence_count
            
            if next_sentence_index <= current_sentence_index:
                current_sentence_index += 1
            else:
                current_sentence_index = next_sentence_index
        
        self.logger.info(f"Generated {len(chunks)} traditional sentence-based chunks")
        return chunks


    # Helper Methods for Advanced Semantic Analysis
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the text (Turkish/English)."""
        # Simple heuristic-based language detection
        turkish_indicators = ['bir', 'bu', 'şu', 'olan', 've', 'ile', 'için', 'de', 'da', 'den', 'dan']
        english_indicators = ['the', 'and', 'of', 'to', 'in', 'is', 'that', 'for', 'with', 'as']
        
        words = text.lower().split()[:100]  # Check first 100 words
        
        turkish_count = sum(1 for word in words if word in turkish_indicators)
        english_count = sum(1 for word in words if word in english_indicators)
        
        if turkish_count > english_count * 1.2:
            return "tr"
        elif english_count > turkish_count * 1.2:
            return "en"
        else:
            return "general"
    
    def _split_into_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences using language-specific patterns."""
        
        # Use appropriate pattern based on language
        if language in self.sentence_boundary_patterns:
            pattern = self.sentence_boundary_patterns[language]
        else:
            pattern = self.sentence_boundary_patterns['general']
        
        # Handle Turkish abbreviations
        if language == "tr":
            # Temporarily replace abbreviations to avoid wrong splits
            temp_text = text
            for abbr in self.turkish_abbreviations:
                temp_text = temp_text.replace(abbr, abbr.replace('.', '<DOT>'))
            
            sentences = pattern.split(temp_text.strip())
            
            # Restore abbreviations
            sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
        else:
            sentences = pattern.split(text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter very short sentences (likely fragments)
        filtered_sentences = []
        for sentence in sentences:
            if len(sentence.split()) >= 3 or len(sentence) >= 20:  # Minimum word/char threshold
                filtered_sentences.append(sentence)
            elif filtered_sentences:
                # Merge with previous sentence if too short
                filtered_sentences[-1] += " " + sentence
        
        return filtered_sentences
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for sentences with EFFICIENT batch processing and caching.
        
        Bu method Hugging Face model'e minimum request sayısı ile maximum efficiency sağlar:
        - Cache'den mevcut embeddings'i alır
        - Sadece cache'de olmayan sentences'ları batch olarak encode eder
        - Tek seferde batch request yapar, efficient kullanım sağlar
        """
        if not EMBEDDING_SUPPORT or not self.embedding_model:
            return np.random.rand(len(sentences), 384)  # Fallback random embeddings
        
        embeddings = []
        sentences_to_encode = []
        indices_to_encode = []
        cache_hits = 0
        
        # Step 1: Check cache for existing embeddings
        for i, sentence in enumerate(sentences):
            cache_key = hashlib.md5(sentence.encode()).hexdigest()
            
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                cache_hits += 1
            else:
                # Store placeholder and remember what needs encoding
                embeddings.append(None)
                sentences_to_encode.append(sentence)
                indices_to_encode.append(i)
        
        # Step 2: Batch encode missing sentences (EFFICIENT!)
        if sentences_to_encode:
            try:
                self.logger.debug(f"Batch encoding {len(sentences_to_encode)} sentences")
                batch_embeddings = self.embedding_model.encode(sentences_to_encode)
                
                # Store in cache and update embeddings list
                for i, (sentence, embedding) in enumerate(zip(sentences_to_encode, batch_embeddings)):
                    cache_key = hashlib.md5(sentence.encode()).hexdigest()
                    self.embedding_cache[cache_key] = embedding
                    
                    original_index = indices_to_encode[i]
                    embeddings[original_index] = embedding
                    
            except Exception as e:
                self.logger.error(f"Batch encoding failed: {e}")
                # Fallback: fill missing embeddings with zeros
                for idx in indices_to_encode:
                    embeddings[idx] = np.zeros(384)
        
        # Log efficiency stats
        total_sentences = len(sentences)
        if cache_hits > 0:
            self.logger.debug(f"Embedding efficiency: {cache_hits}/{total_sentences} cache hits ({cache_hits/total_sentences*100:.1f}%), {len(sentences_to_encode)} new encodings")
        
        return np.array(embeddings)
    
    def _detect_semantic_boundaries(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        target_size: int
    ) -> List[ChunkBoundary]:
        """Detect optimal semantic boundaries using embedding similarity."""
        
        boundaries = []
        current_chunk_size = 0
        similarity_threshold = 0.7  # Configurable threshold
        
        for i in range(len(sentences) - 1):
            current_chunk_size += len(sentences[i]) + 1
            
            # Calculate similarity with next sentence
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0, 0]
            
            # Determine boundary strength
            boundary_strength = 1 - similarity
            
            # Check if this is a natural break point
            is_natural_break = self._is_natural_sentence_boundary(sentences[i], sentences[i + 1])
            
            # Decision logic for boundary detection
            should_break = False
            reason = ""
            
            # Size-based boundary
            if current_chunk_size >= target_size:
                if similarity < similarity_threshold or is_natural_break:
                    should_break = True
                    reason = "size_threshold_with_semantic_break"
                elif current_chunk_size >= self.max_chunk_size:
                    should_break = True
                    reason = "max_size_exceeded"
            
            # Strong semantic boundary regardless of size
            elif similarity < 0.5 and boundary_strength > 0.5:
                should_break = True
                reason = "strong_semantic_boundary"
            
            # Natural language boundary with moderate similarity drop
            elif is_natural_break and similarity < 0.8:
                should_break = True
                reason = "natural_language_boundary"
            
            if should_break:
                boundary = ChunkBoundary(
                    position=sum(len(s) + 1 for s in sentences[:i + 1]),
                    sentence_index=i,
                    similarity_score=similarity,
                    boundary_strength=boundary_strength,
                    is_natural_break=is_natural_break,
                    reason=reason
                )
                boundaries.append(boundary)
                current_chunk_size = 0
        
        # Ensure we have at least one boundary if text is long
        if not boundaries and len(sentences) > 10:
            mid_point = len(sentences) // 2
            boundary = ChunkBoundary(
                position=sum(len(s) + 1 for s in sentences[:mid_point]),
                sentence_index=mid_point - 1,
                similarity_score=0.6,
                boundary_strength=0.4,
                is_natural_break=False,
                reason="fallback_midpoint"
            )
            boundaries.append(boundary)
        
        return boundaries
    
    def _is_natural_sentence_boundary(self, current_sentence: str, next_sentence: str) -> bool:
        """Check if there's a natural boundary between two sentences."""
        
        # Turkish-specific boundary indicators
        if self.turkish_sentence_starters.match(next_sentence):
            return True
        
        # Paragraph breaks
        if '\n\n' in current_sentence or next_sentence.startswith('\n'):
            return True
        
        # Topic transition words
        transition_words = ['however', 'moreover', 'furthermore', 'meanwhile', 'consequently',
                          'ancak', 'fakat', 'ayrıca', 'dahası', 'bu nedenle', 'sonuç olarak']
        
        next_lower = next_sentence.lower()
        for word in transition_words:
            if next_lower.startswith(word.lower()):
                return True
        
        # Question-answer patterns
        if current_sentence.strip().endswith('?') and not next_sentence.startswith(('Yes', 'No', 'Evet', 'Hayır')):
            return True
        
        return False
    
    def _calculate_chunk_coherence(self, chunk_embeddings: np.ndarray) -> float:
        """Calculate semantic coherence score for a chunk."""
        
        if len(chunk_embeddings) < 2:
            return 1.0  # Single sentence is perfectly coherent
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(chunk_embeddings)):
            for j in range(i + 1, len(chunk_embeddings)):
                sim = cosine_similarity(
                    chunk_embeddings[i].reshape(1, -1),
                    chunk_embeddings[j].reshape(1, -1)
                )[0, 0]
                similarities.append(sim)
        
        # Return average similarity as coherence score
        return float(np.mean(similarities)) if similarities else 0.5
    
    def _extract_chunk_keywords(self, text: str) -> List[str]:
        """Extract key terms from chunk text."""
        
        # Simple keyword extraction (can be enhanced with TF-IDF later)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        common_words = {'bir', 'bu', 'şu', 've', 'ile', 'için', 'olan', 'the', 'and', 'of', 'to', 'in', 'is', 'that'}
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]


# Backward Compatibility Layer
class SemanticChunker(AdvancedSemanticChunker):
    """Backward compatibility wrapper for the original SemanticChunker."""
    
    def __init__(self):
        super().__init__()
        
    def create_semantic_chunks(
        self,
        text: str,
        target_size: int = 512,
        overlap_ratio: float = 0.1,
        language: str = "auto"
    ) -> List[str]:
        """Backward compatible method returning text chunks only."""
        
        try:
            semantic_chunks = super().create_semantic_chunks(
                text, target_size, overlap_ratio, language, use_embedding_analysis=True
            )
            # Convert SemanticChunk objects to strings for backward compatibility
            return [chunk.text for chunk in semantic_chunks]
        except Exception as e:
            self.logger.error(f"Advanced semantic chunking failed: {e}")
            # Fallback to traditional chunking
            semantic_chunks = super().create_semantic_chunks(
                text, target_size, overlap_ratio, language, use_embedding_analysis=False
            )
            return [chunk.text for chunk in semantic_chunks]


def create_semantic_chunks(
    text: str,
    target_size: int = 800,
    overlap_ratio: float = 0.1,
    language: str = "auto",
    fallback_strategy: str = "markdown"  # Kept for compatibility (not used)
) -> List[str]:
    """
    Main function to create semantic chunks with advanced embedding analysis.
    
    This function uses the new AdvancedSemanticChunker with embedding-based
    boundary detection and Turkish language optimizations.
    
    Args:
        text: Input text to chunk
        target_size: Target size for chunks
        overlap_ratio: Ratio of overlap between chunks
        language: Language of the text ("tr", "en", or "auto")
        fallback_strategy: Kept for compatibility (not used)
    
    Returns:
        List of semantically coherent text chunks
    """
    chunker = SemanticChunker()
    
    try:
        return chunker.create_semantic_chunks(text, target_size, overlap_ratio, language)
    except Exception as e:
        chunker.logger.error(f"Advanced semantic chunking failed unexpectedly: {e}")
        # Last resort: return the whole text as one chunk to avoid data loss
        return [text] if text.strip() else []