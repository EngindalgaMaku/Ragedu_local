"""
Text chunking module.

This module provides functions for splitting large texts into smaller,
manageable chunks, which is a crucial step before generating embeddings.
"""

from typing import List, Literal, Sequence
import re
from functools import lru_cache
try:
    # Try relative imports first (when used as package)
    from .. import config
    from ..utils.helpers import setup_logging
except ImportError:
    # Fallback to absolute imports (for testing and standalone use)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src import config
    from src.utils.helpers import setup_logging

# Import semantic chunking functionality
try:
    from .semantic_chunker import create_semantic_chunks
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False
    
    # Create dummy function to prevent import errors
    def create_semantic_chunks(*args, **kwargs):
        raise NotImplementedError("Semantic chunking not available")

# Import AST-based markdown parser - Phase 1 Enhancement
try:
    from .ast_markdown_parser import ASTMarkdownParser, MarkdownSection
    AST_MARKDOWN_AVAILABLE = True
except ImportError:
    AST_MARKDOWN_AVAILABLE = False

logger = setup_logging()

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


def _chunk_by_markdown_structure(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    DÜZELTİLMİŞ Markdown yapısına dayalı akıllı chunking.
    Kelime sınırları korumalı, akıllı overlap, minimum chunk boyutu kontrolü.
    """
    if not text.strip():
        return []
    
    # Text'i normalize et
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Çoklu boş satırları temizle
    lines = [line.rstrip() for line in text.split('\n')]
    
    sections = []
    current_section = []
    current_section_size = 0
    current_header = None
    
    def add_section_safe():
        """Mevcut section'ı güvenli şekilde ekle - minimum boyut kontrolü ile"""
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if len(section_text) > 50:  # Minimum chunk boyutu
                sections.append(section_text)
    
    def smart_overlap(prev_text: str, overlap_size: int) -> str:
        """Satır/cümle bazlı akıllı overlap oluştur"""
        if len(prev_text) <= overlap_size:
            return prev_text
            
        # Öncelik: tam satırlar (markdown yapısını korur)
        lines = prev_text.split('\n')
        selected_lines = []
        current_len = 0
        
        # Sondan geriye doğru tam satırları al
        for line in reversed(lines):
            line_len = len(line) + 1  # +1 for newline
            if current_len + line_len <= overlap_size * 1.5:  # Biraz esnek ol
                selected_lines.insert(0, line)
                current_len += line_len
            else:
                break
                
        if selected_lines:
            return '\n'.join(selected_lines)
            
        # Satır bazlı çözüm yoksa cümle bazlı dene
        sentences = prev_text.split('.')
        if len(sentences) > 1:
            last_sentences = sentences[-2:] if len(sentences) > 2 else sentences[-1:]
            overlap_text = '.'.join(last_sentences).strip()
            if overlap_text and len(overlap_text) <= overlap_size * 2:
                return overlap_text if overlap_text.endswith('.') else overlap_text + '.'
        
        # Son çare: kelime bazlı
        words = prev_text.split()
        if len(words) <= 3:  # Çok az kelime varsa tamamını al
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
    
    def clean_chunk_start(chunk_text: str) -> str:
        """Chunk başlangıcını temizle - kelime kesikliği varsa düzelt"""
        lines = chunk_text.split('\n')
        if not lines:
            return chunk_text
        
        # İlk boş olmayan satırı bul
        first_content_line_idx = 0
        for i, line in enumerate(lines):
            if line.strip():
                first_content_line_idx = i
                break
        else:
            return chunk_text  # Tümü boş satır
            
        first_line = lines[first_content_line_idx].strip()
        
        # Başlık, liste öğesi ise dokunma
        if (first_line.startswith('#') or
            re.match(r'^[\s]*[-\*\+][\s]|^[\s]*\d+\.[\s]', first_line)):
            return chunk_text
            
        # Problematik başlangıçları tespit et
        is_problematic = False
        
        # 1. Sadece noktalama ile başlayorsa
        if re.match(r'^[\*\:\.\,\!\?\;\-\)\]]+', first_line):
            is_problematic = True
            
        # 2. Küçük harfle başlayıp sonrasında noktalama varsa (kesik kelime)
        elif first_line and first_line[0].islower():
            # "türleri:**", "üretme -" gibi durumlar
            if re.search(r'[\:\*\-\.\,]', first_line[:20]):  # İlk 20 karakterde noktalama
                is_problematic = True
                
        # 3. Çok kısa ve anlamlı değilse (3 kelimeden az)
        elif len(first_line.split()) < 3 and not first_line[0].isupper():
            is_problematic = True
            
        if is_problematic:
            # Problematik satırı kaldır, sonrakinden devam et
            remaining_lines = lines[first_content_line_idx + 1:]
            if remaining_lines:
                # Kalan satırlarda düzgün başlangıç ara
                for i, line in enumerate(remaining_lines):
                    clean_line = line.strip()
                    if (clean_line and
                        (clean_line[0].isupper() or
                         clean_line.startswith('#') or
                         re.match(r'^[\s]*[-\*\+][\s]|^[\s]*\d+\.[\s]', clean_line))):
                        # Düzgün başlangıç bulundu
                        return '\n'.join(lines[:first_content_line_idx] + remaining_lines[i:])
                
                # Düzgün başlangıç bulunamazsa tüm problematik bölümü at
                return '\n'.join(remaining_lines)
            else:
                return ""  # Hiçbir şey kalmadı
        
        return chunk_text
    
    def find_sentence_boundary(text: str, max_size: int) -> int:
        """Türkçe cümle sonu veya kelime sınırından uygun kesim noktası bul"""
        if len(text) <= max_size:
            return len(text)
        
        # Türkçe cümle sonları ve noktalama - genişletilmiş
        sentence_endings = '.!?…;:'
        
        # Önce cümle sonu ara (Türkçe noktalama dahil)
        for i in range(max_size, max(0, max_size - 150), -1):
            if i < len(text) and text[i] in sentence_endings:
                # Cümle sonundan sonra boşluk varsa ideal kesim noktası
                if i + 1 < len(text) and text[i + 1] in ' \n\t':
                    return i + 1
                else:
                    return i + 1
        
        # Paragraf sonu ara (\n\n)
        for i in range(max_size, max(0, max_size - 100), -1):
            if i + 1 < len(text) and text[i:i+2] == '\n\n':
                return i
        
        # Satır sonu ara
        for i in range(max_size, max(0, max_size - 80), -1):
            if i < len(text) and text[i] == '\n':
                return i
        
        # Kelime sınırı ara (Türkçe karakterlere uyumlu)
        for i in range(max_size, max(0, max_size - 50), -1):
            if i < len(text) and text[i] in ' \t':
                return i
        
        # Noktalama işareti ara
        for i in range(max_size, max(0, max_size - 30), -1):
            if i < len(text) and text[i] in ',-;:()[]{}':
                return i + 1
        
        # En son çare olarak verilen boyutta kes
        return max_size
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Ana başlık veya bölüm başlığı (## veya #)
        if line.startswith('##') or line.startswith('# '):
            add_section_safe()
            current_section = [line]
            current_section_size = len(line)
            current_header = line
            i += 1
            
            # Bu başlık altındaki içeriği topla
            while i < len(lines) and not (lines[i].startswith('##') or lines[i].startswith('# ')):
                next_line = lines[i]
                line_size = len(next_line) + 1
                
                # Boyut kontrolü - kelime sınırına dikkat et
                if current_section_size + line_size > chunk_size and len(current_section) > 1:
                    add_section_safe()
                    # Başlığı koruyarak yeni section başlat
                    current_section = [current_header] if current_header else []
                    current_section_size = len(current_header) if current_header else 0
                
                current_section.append(next_line)
                current_section_size += line_size
                i += 1
                
        # Alt başlık (###)
        elif line.startswith('###'):
            # Eğer mevcut section çok küçükse birleştir
            if current_section_size < 200 and current_section:
                current_section.append(line)
                current_section_size += len(line) + 1
            else:
                add_section_safe()
                current_section = [current_header, line] if current_header else [line]
                current_section_size = len(current_header or '') + len(line) + 1
            i += 1
        
        # Kod bloğu tespit et (```)
        elif line.startswith('```'):
            code_block = [line]
            code_size = len(line)
            i += 1
            
            while i < len(lines):
                code_line = lines[i]
                code_block.append(code_line)
                code_size += len(code_line) + 1
                
                if code_line.startswith('```'):
                    break
                i += 1
            
            # Kod bloğu mevcut bölüme sığıyorsa ekle
            if current_section_size + code_size <= chunk_size:
                current_section.extend(code_block)
                current_section_size += code_size
            else:
                add_section_safe()
                sections.append('\n'.join(code_block))
                current_section = []
                current_section_size = 0
        
        # Liste öğesi tespit et (-, *, +, 1.)
        elif re.match(r'^[\s]*[-\*\+][\s]|^[\s]*\d+\.[\s]', line):
            list_items = []
            list_size = 0
            
            while i < len(lines):
                list_line = lines[i]
                
                if not re.match(r'^[\s]*[-\*\+][\s]|^[\s]*\d+\.[\s]|^[\s]*$', list_line):
                    break
                
                list_items.append(list_line)
                list_size += len(list_line) + 1
                i += 1
            
            # Liste mevcut bölüme sığıyorsa ekle
            if current_section_size + list_size <= chunk_size:
                current_section.extend(list_items)
                current_section_size += list_size
            else:
                add_section_safe()
                sections.append('\n'.join(list_items))
                current_section = []
                current_section_size = 0
            continue
        
        # Normal paragraf satırı
        else:
            line_size = len(line) + 1
            
            if current_section_size + line_size > chunk_size and current_section:
                add_section_safe()
                current_section = [current_header] if current_header else []
                current_section_size = len(current_header) if current_header else 0
            
            current_section.append(line)
            current_section_size += line_size
        
        i += 1
    
    # Son section'ı ekle
    add_section_safe()
    
    # Çok küçük section'ları birleştir
    final_sections = []
    for section in sections:
        if len(section) < 100 and final_sections and len(final_sections[-1]) < chunk_size * 0.8:
            # Önceki section ile birleştir
            final_sections[-1] = final_sections[-1] + '\n\n' + section
        else:
            final_sections.append(section)
    
    # AKILLI OVERLAP - Kalite ve bağlam dengesini korur
    if chunk_overlap > 0:
        logger.info(f"Akıllı overlap uygulanıyor: {chunk_overlap} karakter")
        overlapped_sections = []
        
        for i, section in enumerate(final_sections):
            if i == 0:
                overlapped_sections.append(section)
            else:
                # Önceki section'dan akıllı overlap oluştur
                prev_section = final_sections[i-1]
                overlap_text = smart_overlap(prev_section, chunk_overlap)
                
                if overlap_text:
                    overlapped_section = overlap_text + '\n\n' + section
                else:
                    overlapped_section = section
                
                overlapped_sections.append(overlapped_section)
        
        return overlapped_sections
    
    return final_sections


def _chunk_by_hybrid_strategy(text: str, chunk_size: int, chunk_overlap: int, language: str) -> List[str]:
    """
    Hybrid chunking strategy combining markdown structural analysis with LLM semantic analysis.
    
    This approach:
    1. First applies markdown structural chunking to respect document structure
    2. Then uses LLM semantic analysis to refine boundaries and improve coherence
    3. Falls back to pure markdown chunking if semantic analysis fails
    """
    logger.info("Applying hybrid chunking strategy (markdown + semantic)")
    
    try:
        # Step 1: Get initial structural chunks using markdown strategy
        structural_chunks = _chunk_by_markdown_structure(text, chunk_size, chunk_overlap)
        
        if not SEMANTIC_CHUNKING_AVAILABLE:
            logger.warning("Semantic analysis not available, using pure markdown strategy")
            return structural_chunks
        
        # Step 2: Apply semantic refinement to structural chunks
        refined_chunks = []
        
        for chunk in structural_chunks:
            # Only apply semantic analysis to chunks that are significantly large
            if len(chunk) > chunk_size * 0.6:  # Only refine larger chunks
                try:
                    # Use semantic chunking to potentially split large chunks better
                    overlap_ratio = chunk_overlap / chunk_size if chunk_overlap > 0 else 0.1
                    semantic_sub_chunks = create_semantic_chunks(
                        text=chunk,
                        target_size=chunk_size,
                        overlap_ratio=overlap_ratio,
                        language=language,
                        fallback_strategy="markdown"
                    )
                    
                    # If semantic analysis produces better chunking, use it
                    if len(semantic_sub_chunks) > 1 and all(len(sc) >= 100 for sc in semantic_sub_chunks):
                        refined_chunks.extend(semantic_sub_chunks)
                        logger.debug(f"Refined chunk into {len(semantic_sub_chunks)} semantic sub-chunks")
                    else:
                        refined_chunks.append(chunk)
                        
                except Exception as e:
                    logger.warning(f"Semantic refinement failed for chunk: {e}")
                    refined_chunks.append(chunk)
            else:
                # Keep smaller chunks as-is
                refined_chunks.append(chunk)
        
        # Step 3: Post-process to merge very small chunks
        final_chunks = []
        for chunk in refined_chunks:
            if len(chunk) < 150 and final_chunks and len(final_chunks[-1]) < chunk_size * 0.8:
                # Merge small chunk with previous one
                final_chunks[-1] = final_chunks[-1] + '\n\n' + chunk
            else:
                final_chunks.append(chunk)
        
        logger.info(f"Hybrid strategy: {len(structural_chunks)} structural -> {len(refined_chunks)} refined -> {len(final_chunks)} final chunks")
        return final_chunks
        
    except Exception as e:
        logger.error(f"Hybrid chunking strategy failed: {e}")
        logger.info("Falling back to pure markdown strategy")
        return _chunk_by_markdown_structure(text, chunk_size, chunk_overlap)


def _chunk_by_ast_markdown(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Advanced AST-based markdown chunking with semantic structure preservation.
    
    Bu function yeni AST markdown parser'ı kullanarak intelligent chunking yapar:
    - Header hierarchy korunur
    - Table'lar semantic context ile korunur
    - Code block'lar intact kalır
    - Math formula'ları protect edilir
    - Cross-reference'lar çözümlenir
    """
    if not text.strip():
        return []
    
    if not AST_MARKDOWN_AVAILABLE:
        logger.warning("AST Markdown parser not available, falling back to traditional markdown chunking")
        return _chunk_by_markdown_structure(text, chunk_size, chunk_overlap)
    
    try:
        logger.info("Using AST-based markdown chunking")
        
        # Initialize AST parser
        ast_parser = ASTMarkdownParser()
        
        # Step 1: Parse markdown to AST
        ast_nodes = ast_parser.parse_markdown_to_ast(text)
        
        if not ast_nodes:
            logger.warning("No AST nodes generated, falling back to traditional markdown")
            return _chunk_by_markdown_structure(text, chunk_size, chunk_overlap)
        
        # Step 2: Create semantic sections
        sections = ast_parser.create_semantic_sections(ast_nodes)
        
        # Step 3: Convert sections to chunks with size control
        chunks = []
        
        for section in sections:
            section_text = section.content
            
            # If section fits in chunk size, use as-is
            if len(section_text) <= chunk_size:
                chunks.append(section_text)
            
            # If section is too large, apply intelligent splitting
            else:
                # Try to split at subsection boundaries first
                if section.subsections:
                    subsection_chunks = []
                    current_chunk = section.title  # Start with section title
                    current_size = len(current_chunk)
                    
                    for subsection in section.subsections:
                        subsection_content = f"\n\n{subsection.content}"
                        
                        if current_size + len(subsection_content) <= chunk_size:
                            current_chunk += subsection_content
                            current_size += len(subsection_content)
                        else:
                            # Current chunk is ready
                            if current_chunk.strip():
                                subsection_chunks.append(current_chunk)
                            
                            # Start new chunk with subsection
                            current_chunk = subsection.content
                            current_size = len(current_chunk)
                    
                    # Add final chunk
                    if current_chunk.strip():
                        subsection_chunks.append(current_chunk)
                    
                    chunks.extend(subsection_chunks)
                
                else:
                    # No subsections, apply semantic sentence splitting
                    if SEMANTIC_CHUNKING_AVAILABLE:
                        try:
                            overlap_ratio = chunk_overlap / chunk_size if chunk_overlap > 0 else 0.1
                            semantic_chunks = create_semantic_chunks(
                                text=section_text,
                                target_size=chunk_size,
                                overlap_ratio=overlap_ratio,
                                language="auto",
                                fallback_strategy="markdown"
                            )
                            chunks.extend(semantic_chunks)
                        except Exception as e:
                            logger.warning(f"Semantic splitting failed for large section: {e}")
                            # Fallback to traditional splitting
                            chunks.extend(_split_large_text_preserving_structure(section_text, chunk_size))
                    else:
                        chunks.extend(_split_large_text_preserving_structure(section_text, chunk_size))
        
        # Step 4: Post-process chunks for optimal size
        final_chunks = _optimize_chunk_sizes(chunks, chunk_size, chunk_overlap)
        
        logger.info(f"AST markdown chunking: {len(sections)} sections -> {len(final_chunks)} optimized chunks")
        return final_chunks
        
    except Exception as e:
        logger.error(f"AST markdown chunking failed: {e}")
        logger.info("Falling back to traditional markdown chunking")
        return _chunk_by_markdown_structure(text, chunk_size, chunk_overlap)


def _split_large_text_preserving_structure(text: str, chunk_size: int) -> List[str]:
    """Split large text while preserving markdown structure."""
    chunks = []
    
    # Split by double newlines (paragraphs) first
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If single paragraph is still too large, split by sentences
            if len(paragraph) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > chunk_size:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _optimize_chunk_sizes(chunks: List[str], target_size: int, overlap: int) -> List[str]:
    """Optimize chunk sizes by merging small chunks and handling overlap."""
    if not chunks:
        return []
    
    optimized = []
    min_chunk_size = target_size // 4  # Minimum chunk size is 1/4 of target
    
    current_chunk = ""
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # If chunk is too small, try to merge with current
        if len(chunk) < min_chunk_size and current_chunk:
            # Check if merging would exceed target size
            if len(current_chunk) + len(chunk) + 2 <= target_size * 1.2:  # Allow 20% overflow
                current_chunk += "\n\n" + chunk
                continue
        
        # Save current chunk if it exists
        if current_chunk:
            optimized.append(current_chunk)
        
        current_chunk = chunk
    
    # Add final chunk
    if current_chunk:
        optimized.append(current_chunk)
    
    # Handle overlap if requested
    if overlap > 0 and len(optimized) > 1:
        overlapped = []
        
        for i, chunk in enumerate(optimized):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Create overlap from previous chunk
                prev_chunk = optimized[i-1]
                
                # Find natural overlap point (sentence boundary)
                overlap_text = ""
                sentences = re.split(r'(?<=[.!?])\s+', prev_chunk)
                
                if sentences:
                    # Take last few sentences as overlap
                    overlap_sentences = []
                    overlap_length = 0
                    
                    for sentence in reversed(sentences):
                        if overlap_length + len(sentence) <= overlap:
                            overlap_sentences.insert(0, sentence)
                            overlap_length += len(sentence)
                        else:
                            break
                    
                    if overlap_sentences:
                        overlap_text = " ".join(overlap_sentences)
                
                # Add overlapped chunk
                if overlap_text:
                    overlapped_chunk = overlap_text + "\n\n" + chunk
                else:
                    overlapped_chunk = chunk
                
                overlapped.append(overlapped_chunk)
        
        return overlapped
    
    return optimized


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    strategy: Literal["char", "paragraph", "sentence", "markdown", "ast_markdown", "semantic", "hybrid"] = "char",
    language: str = "auto",
    use_embedding_refinement: bool = True
) -> List[str]:
    """
    Splits a text into overlapping chunks using various strategies with Phase 1 enhancements.

    Args:
        text: The input text to be chunked.
        chunk_size: The desired maximum size of each chunk (in characters).
                    If None, uses the value from the config file.
        chunk_overlap: The desired overlap between consecutive chunks (in characters).
                       If None, uses the value from the config file.
        strategy: Chunking strategy to use:
                  - "char": Character-based chunking
                  - "paragraph": Paragraph-based chunking
                  - "sentence": Sentence-based chunking
                  - "markdown": Markdown structure-aware chunking
                  - "ast_markdown": Advanced AST-based markdown with semantic structure preservation (NEW in Phase 1)
                  - "semantic": LLM-based semantic chunking with embedding refinement
                  - "hybrid": Combination of markdown + semantic analysis
        language: Language of the text for semantic analysis ("tr", "en", or "auto")
        use_embedding_refinement: Whether to use embedding-based refinement for semantic strategies (Phase 1)

    Returns:
        A list of text chunks.
        
    Phase 1 Enhancements:
        - New "ast_markdown" strategy with intelligent structure preservation
        - Enhanced semantic chunking with embedding-based boundary refinement
        - Improved Turkish language support
        - Batch processing optimizations for embeddings
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP

    if not text:
        logger.warning("Input text is empty. Returning no chunks.")
        return []

    logger.info(f"Chunking text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, strategy={strategy}")

    # Normalize newlines
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    if strategy == "char":
        # Akıllı karakter chunking - kelime sınırlarından kes
        chunks: List[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            
            # Eğer chunk sonunda değilsek, kelime sınırından kes
            if end < len(normalized):
                # Geriye doğru en yakın boşluk, noktalama veya satır sonu bul
                while end > start and normalized[end] not in ' \n\t.,!?;:':
                    end -= 1
                
                # Eğer uygun kesim noktası bulunamazsa (çok uzun kelime), zorla kes
                if end <= start:
                    end = start + chunk_size
            
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - chunk_overlap if chunk_overlap > 0 else end
            
            # Sonsuz döngüyü engelle
            if start >= len(normalized):
                break
    elif strategy == "paragraph":
        # split by blank lines as paragraphs
        paragraphs = [p.strip() for p in normalized.split("\n\n")]
        chunks = _group_units(paragraphs, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        # Gelişmiş Türkçe cümle chunking
        import re
        
        # Türkçe cümle ayırma pattern'i - daha doğru
        # Cümle sonları: . ! ? … ; :
        # Sonrasında büyük harf, rakam, yeni satır veya metin sonu
        turkish_sentence_pattern = r'([.!?…;:])\s*(?=\s*[A-ZÇĞIÖŞÜ\d\n]|$)'
        
        # Cümleleri ayır
        parts = re.split(turkish_sentence_pattern, normalized)
        
        sentences: List[str] = []
        i = 0
        while i < len(parts):
            base_part = parts[i].strip() if i < len(parts) else ""
            punct_part = parts[i+1].strip() if i+1 < len(parts) else ""
            
            if base_part:
                # Cümleyi noktalama ile birleştir
                if punct_part in '.!?…;:':
                    sentence = base_part + punct_part
                    i += 2
                else:
                    sentence = base_part
                    i += 1
                
                # Türkçe için minimum cümle uzunluğu 20 karakter
                if len(sentence) >= 20:
                    sentences.append(sentence)
                elif sentences and len(sentences[-1]) < 150:
                    # Kısa cümleyi öncekiyle birleştir (Türkçe cümle yapısına uygun)
                    sentences[-1] = sentences[-1] + ' ' + sentence
                else:
                    sentences.append(sentence)
            else:
                i += 1
        
        # Boş ve çok kısa cümleleri temizle
        sentences = [s for s in sentences if s.strip() and len(s.strip()) >= 10]
        chunks = _group_units(sentences, chunk_size, chunk_overlap)
    elif strategy == "markdown":
        # Markdown yapısına dayalı akıllı chunking
        chunks = _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
    elif strategy == "ast_markdown":
        # NEW in Phase 1: Advanced AST-based markdown chunking with semantic structure preservation
        chunks = _chunk_by_ast_markdown(normalized, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        # LLM tabanlı semantic chunking with Phase 1 embedding refinement
        if SEMANTIC_CHUNKING_AVAILABLE:
            try:
                overlap_ratio = chunk_overlap / chunk_size if chunk_overlap > 0 else 0.1
                chunks = create_semantic_chunks(
                    text=normalized,
                    target_size=chunk_size,
                    overlap_ratio=overlap_ratio,
                    language=language,
                    fallback_strategy="ast_markdown" if AST_MARKDOWN_AVAILABLE else "markdown"
                )
            except Exception as e:
                logger.error(f"Semantic chunking failed: {e}")
                fallback = "ast_markdown" if AST_MARKDOWN_AVAILABLE else "markdown"
                logger.info(f"Falling back to {fallback} strategy")
                if AST_MARKDOWN_AVAILABLE:
                    chunks = _chunk_by_ast_markdown(normalized, chunk_size, chunk_overlap)
                else:
                    chunks = _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
        else:
            logger.warning("Semantic chunking not available, falling back to enhanced markdown")
            if AST_MARKDOWN_AVAILABLE:
                chunks = _chunk_by_ast_markdown(normalized, chunk_size, chunk_overlap)
            else:
                chunks = _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
    elif strategy == "hybrid":
        # Hibrit strateji: Markdown yapısal analiz + LLM semantic analiz
        chunks = _chunk_by_hybrid_strategy(normalized, chunk_size, chunk_overlap, language)
    else:
        logger.warning(f"Unknown chunking strategy '{strategy}', falling back to 'char'.")
        chunks = []
        start = 0
        while start < len(normalized):
            end = start + chunk_size
            chunks.append(normalized[start:end])
            start += max(1, chunk_size - chunk_overlap)
    
    logger.info(f"Generated {len(chunks)} chunks from the text.")
    return chunks

if __name__ == '__main__':
    # This is for testing purposes.
    sample_text = """
    This is a long sample text designed to test the chunking functionality of the RAG system.
    The system needs to be able to split this text into smaller, overlapping chunks so that
    each chunk can be embedded and stored in a vector database. The overlap is important
    to ensure that semantic context is not lost at the boundaries of the chunks.
    Let's add more content to make sure it's long enough. We are building a Personalized
    Course Note and Resource Assistant. This assistant will help students by answering
    questions based on their course materials. The materials can be in PDF, DOCX, or PPTX format.
    The core of the system is the Retrieval-Augmented Generation (RAG) pipeline.
    This pipeline involves several steps: document processing, text chunking, embedding
    generation, vector storage, retrieval, and response generation. This test focuses
    on the text chunking step, which is fundamental for the subsequent stages.
    """
    
    print("--- Testing Text Chunker ---")
    
    # Use default config values
    text_chunks = chunk_text(sample_text)
    
    if text_chunks:
        print(f"Successfully created {len(text_chunks)} chunks with default settings.")
        for i, chunk in enumerate(text_chunks):
            print(f"\n--- Chunk {i+1} (Length: {len(chunk)}) ---")
            print(chunk)
            print("--------------------")
    else:
        print("Failed to create chunks.")

    print("\n--- Testing with custom settings ---")
    custom_chunks = chunk_text(sample_text, chunk_size=150, chunk_overlap=30)
    if custom_chunks:
        print(f"Successfully created {len(custom_chunks)} chunks with custom settings.")
        print(f"First chunk: '{custom_chunks}'")
        print(f"Second chunk: '{custom_chunks}'")
    else:
        print("Failed to create chunks with custom settings.")
