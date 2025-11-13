"""
Semantic Chunking Module with LLM-based Text Analysis.

This module provides advanced chunking capabilities using LLM to analyze semantic 
boundaries, topic coherence, and natural text structure for optimal chunk creation.
"""

from typing import List
import re
from ..config import get_config
from ..utils.logger import get_logger

class SemanticChunker:
    """
    A robust text chunker that prioritizes sentence integrity and avoids content duplication.
    
    This implementation uses a cursor-based approach on a list of sentences. This ensures
    that chunks are built from complete sentences and that content is not accidentally
    duplicated. Overlap is handled by starting the next chunk a few sentences before
    the previous one ended, providing context without copying text.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__, self.config)
        # Flexible character limits for chunks
        self.min_chunk_size = 150
        self.max_chunk_size = 1024
        
        # A robust regex to split text into sentences. It splits after a period,
        # exclamation mark, or question mark that is followed by whitespace or
        # is at the end of the text. The lookbehind `(?<=[.!?])` ensures the
        # punctuation is kept with the sentence.
        self.sentence_boundary_pattern = re.compile(r'(?<=[.!?])\s+')

    def create_semantic_chunks(
        self,
        text: str,
        target_size: int = 512,
        overlap_ratio: float = 0.1,
        language: str = "auto" # Kept for signature compatibility
    ) -> List[str]:
        """
        Creates sentence-aware chunks from text, ensuring no content duplication.

        This method works by advancing a cursor (`current_sentence_index`) through a
        list of sentences derived from the input text. For each chunk, it greedily
        adds sentences until a `target_size` is reached, without exceeding `max_chunk_size`.
        This guarantees that chunks do not start or end mid-sentence.

        Overlap is achieved not by copying text, but by strategically setting the
        starting index for the next chunk. The next chunk begins a few sentences
        before the end of the current one, based on `overlap_ratio`.

        Args:
            text: The input text to be chunked.
            target_size: The desired character length for each chunk.
            overlap_ratio: The percentage of overlap between consecutive chunks.
                         For example, 0.1 means a 10% overlap in terms of sentences.

        Returns:
            A list of text chunks, where each chunk is a coherent block of complete sentences.
        """
        if not text or not text.strip():
            return []

        self.logger.info(f"Starting robust, sentence-based chunking for text of length {len(text)}.")
        
        # 1. Split the entire text into sentences first.
        sentences = self.sentence_boundary_pattern.split(text.strip())
        # Filter out any empty strings that might result from the split.
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            self.logger.warning("No sentences found in the provided text.")
            return []

        chunks = []
        current_sentence_index = 0
        num_sentences = len(sentences)

        # 2. Iterate through sentences with a cursor.
        while current_sentence_index < num_sentences:
            chunk_start_index = current_sentence_index
            chunk_end_index = chunk_start_index
            current_chunk_char_count = 0
            
            # 3. Greedily add sentences to form a chunk.
            while chunk_end_index < num_sentences:
                sentence_len = len(sentences[chunk_end_index])
                
                # Stop if adding the next sentence would exceed max_chunk_size,
                # but only if the current chunk has some content already.
                if current_chunk_char_count > 0 and (current_chunk_char_count + sentence_len) > self.max_chunk_size:
                    break
                
                current_chunk_char_count += sentence_len + 1  # +1 for the joining space
                chunk_end_index += 1
                
                # Stop if we've reached the target size.
                if current_chunk_char_count >= target_size:
                    break

            # If the chunk is still too small, extend it further.
            while current_chunk_char_count < self.min_chunk_size and chunk_end_index < num_sentences:
                 current_chunk_char_count += len(sentences[chunk_end_index]) + 1
                 chunk_end_index += 1

            # 4. Assemble the chunk from the selected sentences.
            chunk_text = " ".join(sentences[chunk_start_index:chunk_end_index])
            if chunk_text:
                chunks.append(chunk_text)

            # If we've processed all sentences, exit.
            if chunk_end_index >= num_sentences:
                break

            # 5. Calculate the start of the next chunk to create overlap.
            num_sentences_in_chunk = chunk_end_index - chunk_start_index
            overlap_sentence_count = int(num_sentences_in_chunk * overlap_ratio)
            
            # Ensure at least one sentence of overlap if requested and possible.
            if overlap_ratio > 0 and overlap_sentence_count == 0 and num_sentences_in_chunk > 1:
                overlap_sentence_count = 1
            
            next_sentence_index = chunk_end_index - overlap_sentence_count
            
            # 6. Advance the cursor, ensuring forward progress.
            if next_sentence_index <= current_sentence_index:
                # This prevents an infinite loop if overlap is too large or chunk is too small.
                current_sentence_index += 1
            else:
                current_sentence_index = next_sentence_index
        
        self.logger.info(f"Generated {len(chunks)} unique, sentence-aligned chunks.")
        return chunks


def create_semantic_chunks(
    text: str,
    target_size: int = 800,
    overlap_ratio: float = 0.1,
    language: str = "auto",
    fallback_strategy: str = "markdown" # This is no longer used by the new chunker
) -> List[str]:
    """
    Main function to create semantic chunks.
    
    This function now uses a robust, sentence-based chunking strategy that
    prevents content duplication and ensures sentence integrity.
    
    Args:
        text: Input text to chunk
        target_size: Target size for chunks
        overlap_ratio: Ratio of overlap between chunks
        language: Language of the text (kept for compatibility)
    
    Returns:
        List of sentence-aligned text chunks.
    """
    chunker = SemanticChunker()
    
    try:
        # The new method is robust and doesn't require a fallback.
        return chunker.create_semantic_chunks(text, target_size, overlap_ratio, language)
    except Exception as e:
        # Log the error, but the new chunker is less likely to fail.
        chunker.logger.error(f"Robust semantic chunking failed unexpectedly: {e}")
        # As a last resort, return the whole text as one chunk to avoid data loss.
        return [text] if text.strip() else []