#!/usr/bin/env python3
"""
Script to fix the text_chunker.py import issues and Turkish sentence boundary detection.
"""

import os

# The fixed text_chunker.py content
fixed_content = '''"""
Text chunking module - FIXED VERSION.

This module provides functions for splitting large texts into smaller,
manageable chunks with improved Turkish support and consolidated import handling.
"""

from typing import List, Literal, Sequence
import re
import logging
from functools import lru_cache

# Safe import handling with proper fallbacks
try:
    # Try relative imports first (when used as package)
    from .. import config
    from ..utils.helpers import setup_logging
    logger = setup_logging()
except ImportError:
    # Fallback to absolute imports (for testing and standalone use)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    try:
        from src import config
        from src.utils.helpers import setup_logging
        logger = setup_logging()
    except ImportError:
        # Final fallback - create basic config and logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Create a basic config object with default values
        class Config:
            CHUNK_SIZE = 1000
            CHUNK_OVERLAP = 200
        
        config = Config()

# Import semantic chunking functionality with safe fallback
SEMANTIC_CHUNKING_AVAILABLE = False
try:
    from .semantic_chunker import create_semantic_chunks
    SEMANTIC_CHUNKING_AVAILABLE = True
    logger.info("✅ Semantic chunking functionality available")
except ImportError:
    logger.warning("⚠️ Semantic chunking not available - using enhanced markdown fallback")
    # Create safe fallback function
    def create_semantic_chunks(text, target_size=1000, overlap_ratio=0.1, language="auto", fallback_strategy="markdown"):
        """Fallback semantic chunking using enhanced markdown strategy."""
        chunk_overlap = int(target_size * overlap_ratio)
        return _chunk_by_markdown_structure(text, target_size, chunk_overlap)

# Import AST-based markdown parser with safe fallback  
AST_MARKDOWN_AVAILABLE = False
try:
    from .ast_markdown_parser import ASTMarkdownParser, MarkdownSection
    AST_MARKDOWN_AVAILABLE = True
    logger.info("✅ AST Markdown parser available")
except ImportError:
    logger.warning("⚠️ AST Markdown parser not available - using enhanced markdown fallback")
    # Create safe fallback classes
    class MarkdownSection:
        def __init__(self, title="", content="", subsections=None):
            self.title = title
            self.content = content
            self.subsections = subsections or []
    
    class ASTMarkdownParser:
        def parse_markdown_to_ast(self, text):
            return []
        
        def create_semantic_sections(self, ast_nodes):
            return []

def _group_units(units: Sequence[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Group sentence/paragraph units into chunks close to chunk_size (by characters)."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for u in units:
        u = u.strip()
        if not u:
            continue
        u_len = len(u) + (1 if current else 0)  # space/newline join cost
        if current_len + u_len <= chunk_size:
            current.append(u)
            current_len += u_len
        else:
            if current:
                chunks.append(" ".join(current))
            # start new chunk with this unit (trim if single unit is too big)
            if len(u) > chunk_size:
                # hard split overly long unit
                for start in range(0, len(u), chunk_size):
                    part = u[start:start + chunk_size]
                    chunks.append(part)
                current = []
                current_len = 0
            else:
                current = [u]
                current_len = len(u)
    if current:
        chunks.append(" ".join(current))

    if chunk_overlap > 0 and chunks:
        # apply character-level overlap between consecutive chunks
        overlapped: List[str] = []
        prev_tail = ""
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                tail = prev_tail[-chunk_overlap:] if prev_tail else ""
                overlapped.append((tail + " " + ch).strip())
            prev_tail = ch
        return overlapped
    return chunks


def _split_turkish_sentences(text: str) -> List[str]:
    """
    IMPROVED Turkish sentence boundary detection with better patterns.
    
    Enhanced to handle:
    - Turkish punctuation (. ! ? … ; :)
    - Turkish capital letters (Ç Ğ I İ Ö Ş Ü)
    - Abbreviations (Dr. vs. vd. gibi)
    - Numbers and dates
    - Quote and parenthesis handling
    """
    if not text.strip():
        return []
    
    # Common Turkish abbreviations that should not trigger sentence breaks
    turkish_abbrevs = {
        'Dr.', 'Prof.', 'Doç.', 'vs.', 'vd.', 'vb.', 'öz.', 'sy.', 'sh.', 'ss.', 'st.', 
        'No.', 'nr.', 'Tel.', 'Fax.', 'E-mail.', 'Web.', 'www.', 'http.', 'https.',
        'km.', 'cm.', 'mm.', 'm.', 'gr.', 'kg.', 'lt.', 'ml.', 'TL.', '$', '€',
        'Ltd.', 'A.Ş.', 'Ltd.Şti.', 'Koop.', 'der.', 'yay.', 'çev.', 'ed.'
    }
    
    # Enhanced Turkish sentence pattern - FIXED VERSION
    # Matches sentence endings [.!?…;:] followed by whitespace/newline and Turkish uppercase/digit
    # This replaces the problematic regex at lines 785-819 in the original code
    sentence_endings = r'([.!?…;:])'
    next_word_pattern = r'\\s*(?=\\s*[A-ZÇĞIİÖŞÜ\\d\\n]|$)'
    
    # Split text while preserving punctuation
    parts = re.split(f'{sentence_endings}{next_word_pattern}', text, flags=re.MULTILINE)
    
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            # We have base text and punctuation
            base_text = parts[i].strip()
            punct = parts[i + 1] if i + 1 < len(parts) else ''
            
            if base_text:
                sentence_candidate = base_text + punct
                
                # Check if this is a false positive (abbreviation)
                is_abbreviation = False
                for abbrev in turkish_abbrevs:
                    if sentence_candidate.rstrip().endswith(abbrev):
                        is_abbreviation = True
                        break
                
                # If it's an abbreviation, continue building the sentence
                if is_abbreviation and i + 2 < len(parts):
                    # Look ahead to next part
                    next_part = parts[i + 2].strip()
                    if next_part and not next_part[0].isupper():
                        # Continue with this sentence
                        i += 2
                        continue
                
                # Valid sentence boundary - Turkish minimum length is 15 characters
                if len(sentence_candidate) >= 15:
                    sentences.append(sentence_candidate)
                elif sentences and len(sentences[-1]) < 200:
                    # Merge short sentences with previous one (Turkish sentence structure)
                    sentences[-1] = sentences[-1] + ' ' + sentence_candidate
                else:
                    sentences.append(sentence_candidate)
            
            i += 2
        else:
            # Last part without punctuation
            if parts[i].strip():
                remaining = parts[i].strip()
                if len(remaining) >= 10:
                    sentences.append(remaining)
                elif sentences:
                    sentences[-1] = sentences[-1] + ' ' + remaining
            i += 1
    
    # Clean up sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) >= 10:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def _chunk_by_markdown_structure(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    ENHANCED Markdown structure-aware chunking with improved Turkish support.
    
    This fixes the markdown structure preservation issues by:
    - Preserving header-content relationships
    - Ensuring chunks don't split in the middle of topics
    - Using Turkish-aware sentence boundary detection
    - Smart overlap that respects markdown structure
    """
    if not text.strip():
        return []
    
    # Normalize text - remove excessive empty lines
    text = re.sub(r'\\n\\s*\\n\\s*\\n+', '\\n\\n', text)
    lines = [line.rstrip() for line in text.split('\\n')]
    
    sections = []
    current_section = []
    current_section_size = 0
    current_header = None
    
    def add_section_safe():
        """Add current section safely with minimum size check"""
        if current_section:
            section_text = '\\n'.join(current_section).strip()
            if len(section_text) > 50:  # Minimum chunk size
                sections.append(section_text)
    
    def smart_overlap_markdown(prev_text: str, overlap_size: int) -> str:
        """Create smart overlap preserving Turkish sentences and markdown structure"""
        if len(prev_text) <= overlap_size:
            return prev_text
        
        # Priority 1: Complete lines (preserves markdown structure like headers, lists)
        lines = prev_text.split('\\n')
        selected_lines = []
        current_len = 0
        
        # Take complete lines from the end
        for line in reversed(lines):
            line_len = len(line) + 1
            if current_len + line_len <= overlap_size * 1.5:  # Allow flexibility
                selected_lines.insert(0, line)
                current_len += line_len
            else:
                break
        
        if selected_lines and len(selected_lines) > 0:
            return '\\n'.join(selected_lines)
        
        # Priority 2: Turkish sentence boundaries
        sentences = _split_turkish_sentences(prev_text)
        if len(sentences) > 1:
            # Take last 1-2 sentences as overlap
            last_sentences = sentences[-2:] if len(sentences) > 2 else [sentences[-1]]
            overlap_text = ' '.join(last_sentences)
            if len(overlap_text) <= overlap_size * 2:
                return overlap_text
        
        # Priority 3: Word boundaries (fallback)
        words = prev_text.split()
        if len(words) <= 3:
            return prev_text
        
        overlap_words = []
        current_len = 0
        
        for word in reversed(words):
            if current_len + len(word) + 1 <= overlap_size:
                overlap_words.insert(0, word)
                current_len += len(word) + 1
            else:
                break
        
        return ' '.join(overlap_words) if overlap_words else ""
    
    # Process lines into sections with header-content preservation
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Main headers (## or # ) - these start new topics
        if line.startswith('##') or line.startswith('# '):
            add_section_safe()  # Save previous section
            current_section = [line]  # Start with header
            current_section_size = len(line)
            current_header = line
            i += 1
            
            # Collect content under this header - CRITICAL FOR TOPIC COHERENCE
            while i < len(lines) and not (lines[i].startswith('##') or lines[i].startswith('# ')):
                next_line = lines[i]
                line_size = len(next_line) + 1
                
                # Size check - if adding this line would exceed limit
                if current_section_size + line_size > chunk_size and len(current_section) > 1:
                    add_section_safe()  # Save current chunk
                    # IMPORTANT: Start new chunk with header to maintain topic context
                    current_section = [current_header] if current_header else []
                    current_section_size = len(current_header) if current_header else 0
                
                current_section.append(next_line)
                current_section_size += line_size
                i += 1
        
        # Sub headers (###) - keep with main topic when possible
        elif line.startswith('###'):
            if current_section_size < 200 and current_section:
                # Small section, keep sub-header with current content
                current_section.append(line)
                current_section_size += len(line) + 1
            else:
                add_section_safe()
                # Start new section with main header context + sub header
                current_section = [current_header, line] if current_header else [line]
                current_section_size = len(current_header or '') + len(line) + 1
            i += 1
        
        # Code blocks - keep intact
        elif line.startswith('```'):
            code_block = [line]
            code_size = len(line)
            i += 1
            
            # Collect entire code block
            while i < len(lines):
                code_line = lines[i]
                code_block.append(code_line)
                code_size += len(code_line) + 1
                
                if code_line.startswith('```'):
                    break
                i += 1
            
            # Add code block to current section or as separate section
            if current_section_size + code_size <= chunk_size:
                current_section.extend(code_block)
                current_section_size += code_size
            else:
                add_section_safe()
                sections.append('\\n'.join(code_block))
                current_section = []
                current_section_size = 0
        
        # Lists - keep together when possible
        elif re.match(r'^[\\s]*[-\\*\\+][\\s]|^[\\s]*\\d+\\.[\\s]', line):
            list_items = []
            list_size = 0
            
            # Collect complete list
            while i < len(lines):
                list_line = lines[i]
                
                if not re.match(r'^[\\s]*[-\\*\\+][\\s]|^[\\s]*\\d+\\.[\\s]|^[\\s]*$', list_line):
                    break
                
                list_items.append(list_line)
                list_size += len(list_line) + 1
                i += 1
            
            # Add list to current section or as separate section
            if current_section_size + list_size <= chunk_size:
                current_section.extend(list_items)
                current_section_size += list_size
            else:
                add_section_safe()
                sections.append('\\n'.join(list_items))
                current_section = []
                current_section_size = 0
            continue
        
        # Regular paragraph lines
        else:
            line_size = len(line) + 1
            
            if current_section_size + line_size > chunk_size and current_section:
                add_section_safe()
                # Maintain topic context by including header
                current_section = [current_header] if current_header else []
                current_section_size = len(current_header) if current_header else 0
            
            current_section.append(line)
            current_section_size += line_size
        
        i += 1
    
    # Add final section
    add_section_safe()
    
    # Merge very small sections to avoid fragmenting topics
    final_sections = []
    for section in sections:
        if len(section) < 100 and final_sections and len(final_sections[-1]) < chunk_size * 0.8:
            final_sections[-1] = final_sections[-1] + '\\n\\n' + section
        else:
            final_sections.append(section)
    
    # Apply smart overlap with markdown structure awareness
    if chunk_overlap > 0 and len(final_sections) > 1:
        logger.info(f"Applying Turkish-aware smart overlap: {chunk_overlap} characters")
        overlapped_sections = []
        
        for i, section in enumerate(final_sections):
            if i == 0:
                overlapped_sections.append(section)
            else:
                prev_section = final_sections[i-1]
                overlap_text = smart_overlap_markdown(prev_section, chunk_overlap)
                
                if overlap_text:
                    overlapped_section = overlap_text + '\\n\\n' + section
                else:
                    overlapped_section = section
                
                overlapped_sections.append(overlapped_section)
        
        return overlapped_sections
    
    return final_sections


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    strategy: Literal["char", "paragraph", "sentence", "markdown", "semantic", "hybrid"] = "markdown",
    language: str = "auto",
    use_embedding_refinement: bool = True
) -> List[str]:
    """
    UNIFIED text chunking with Turkish support and consolidated import handling.

    Args:
        text: The input text to be chunked.
        chunk_size: The desired maximum size of each chunk (in characters).
        chunk_overlap: The desired overlap between consecutive chunks (in characters).
        strategy: Chunking strategy to use:
                  - "char": Character-based chunking with word boundaries
                  - "paragraph": Paragraph-based chunking
                  - "sentence": Turkish-aware sentence-based chunking
                  - "markdown": Enhanced markdown structure-aware chunking (DEFAULT)
                  - "semantic": LLM-based semantic chunking (with safe fallback)
                  - "hybrid": Combination of markdown + semantic analysis
        language: Language of the text ("tr", "en", or "auto")
        use_embedding_refinement: Whether to use embedding-based refinement

    Returns:
        A list of text chunks optimized for Turkish content and markdown structure.
    """
    # Use config defaults if not provided
    if chunk_size is None:
        chunk_size = getattr(config, 'CHUNK_SIZE', 1000)
    if chunk_overlap is None:
        chunk_overlap = getattr(config, 'CHUNK_OVERLAP', 200)

    if not text:
        logger.warning("Input text is empty. Returning no chunks.")
        return []

    logger.info(f"Chunking text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, strategy={strategy}")

    # Normalize newlines
    normalized = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")

    if strategy == "char":
        # Smart character chunking with Turkish word boundaries
        chunks: List[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            
            # Find word boundary for Turkish text
            if end < len(normalized):
                # Look backwards for safe cut point
                while end > start and normalized[end] not in ' \\n\\t.,!?;:…':
                    end -= 1
                
                # If no suitable cut point found, force cut
                if end <= start:
                    end = start + chunk_size
            
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - chunk_overlap if chunk_overlap > 0 else end
            
            if start >= len(normalized):
                break
        
        return chunks
    
    elif strategy == "paragraph":
        paragraphs = [p.strip() for p in normalized.split("\\n\\n")]
        return _group_units(paragraphs, chunk_size, chunk_overlap)
    
    elif strategy == "sentence":
        # Enhanced Turkish sentence chunking
        sentences = _split_turkish_sentences(normalized)
        return _group_units(sentences, chunk_size, chunk_overlap)
    
    elif strategy == "markdown":
        # Enhanced markdown structure-aware chunking - DEFAULT and RECOMMENDED
        return _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
    
    elif strategy == "semantic":
        # Semantic chunking with safe fallback
        if SEMANTIC_CHUNKING_AVAILABLE:
            try:
                overlap_ratio = chunk_overlap / chunk_size if chunk_overlap > 0 else 0.1
                chunks = create_semantic_chunks(
                    text=normalized,
                    target_size=chunk_size,
                    overlap_ratio=overlap_ratio,
                    language=language,
                    fallback_strategy="markdown"
                )
                logger.info(f"✅ Semantic chunking successful: {len(chunks)} chunks")
                return chunks
            except Exception as e:
                logger.error(f"❌ Semantic chunking failed: {e}")
                logger.info("⚠️ Falling back to enhanced markdown strategy")
                return _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
        else:
            logger.info("⚠️ Semantic chunking not available, using enhanced markdown")
            return _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
    
    elif strategy == "hybrid":
        # Hybrid strategy: Start with markdown, enhance with semantic analysis if available
        logger.info("Applying hybrid chunking strategy (markdown + semantic)")
        
        try:
            # Get initial structural chunks using enhanced markdown
            structural_chunks = _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
            
            if not SEMANTIC_CHUNKING_AVAILABLE:
                logger.info("⚠️ Semantic analysis not available for hybrid, using pure enhanced markdown")
                return structural_chunks
            
            # Apply semantic refinement to large chunks only
            refined_chunks = []
            
            for chunk in structural_chunks:
                if len(chunk) > chunk_size * 0.7:  # Only refine larger chunks
                    try:
                        overlap_ratio = chunk_overlap / chunk_size if chunk_overlap > 0 else 0.1
                        semantic_sub_chunks = create_semantic_chunks(
                            text=chunk,
                            target_size=chunk_size,
                            overlap_ratio=overlap_ratio,
                            language=language,
                            fallback_strategy="markdown"
                        )
                        
                        if len(semantic_sub_chunks) > 1 and all(len(sc) >= 100 for sc in semantic_sub_chunks):
                            refined_chunks.extend(semantic_sub_chunks)
                            logger.debug(f"Refined chunk into {len(semantic_sub_chunks)} semantic sub-chunks")
                        else:
                            refined_chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Semantic refinement failed: {e}")
                        refined_chunks.append(chunk)
                else:
                    refined_chunks.append(chunk)
            
            logger.info(f"✅ Hybrid strategy: {len(structural_chunks)} structural -> {len(refined_chunks)} refined chunks")
            return refined_chunks
            
        except Exception as e:
            logger.error(f"❌ Hybrid chunking strategy failed: {e}")
            logger.info("⚠️ Falling back to pure enhanced markdown strategy")
            return _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
    
    else:
        logger.warning(f"Unknown chunking strategy '{strategy}', falling back to enhanced markdown.")
        return _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)


if __name__ == '__main__':
    # Test the enhanced chunking system
    sample_turkish_text = """
    # Türkiye'nin Coğrafi Özellikleri

    ## Konum ve Sınırlar
    Türkiye, Anadolu ve Trakya yarımadalarında yer alan bir ülkedir. Kuzeyinde Karadeniz, güneyinde Akdeniz, batısında Ege Denizi bulunur.

    ### Komşu Ülkeler
    - Yunanistan ve Bulgaristan (batı)
    - Gürcistan ve Ermenistan (kuzeydoğu)  
    - İran ve Irak (doğu)
    - Suriye (güneydoğu)

    ## İklim Özellikleri
    Türkiye'de üç farklı iklim tipi görülür: Akdeniz iklimi, karasal iklim ve Karadeniz iklimi. Bu durum ülkenin zengin biyolojik çeşitliliğini destekler.

    ### Akdeniz İklimi
    Güney kıyılarında görülür. Yaz ayları sıcak ve kurak, kış ayları ılık ve yağışlıdır.
    """
    
    print("--- Testing Enhanced Turkish Chunking ---")
    
    # Test enhanced markdown strategy
    chunks = chunk_text(sample_turkish_text, chunk_size=300, chunk_overlap=50, strategy="markdown")
    
    if chunks:
        print(f"Successfully created {len(chunks)} chunks with enhanced markdown strategy.")
        for i, chunk in enumerate(chunks):
            print(f"\\n--- Chunk {i+1} (Length: {len(chunk)}) ---")
            print(chunk)
            print("---")
    else:
        print("Failed to create chunks.")
'''

# Write the fixed content to text_chunker.py
with open("src/text_processing/text_chunker.py", "w", encoding="utf-8") as f:
    f.write(fixed_content)

print("✅ Fixed text_chunker.py with enhanced Turkish support and resolved import issues")
