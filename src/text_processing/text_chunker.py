"""
Text chunking module.

This module provides functions for splitting large texts into smaller,
manageable chunks, which is a crucial step before generating embeddings.
"""

from typing import List, Literal, Sequence
import re
from .. import config
from ..utils.helpers import setup_logging

# Import semantic chunking functionality
try:
    from .semantic_chunker import create_semantic_chunks
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

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
        """Cümle sonu veya kelime sınırından uygun kesim noktası bul"""
        if len(text) <= max_size:
            return len(text)
        
        # Önce cümle sonu ara
        for i in range(max_size, max(0, max_size - 100), -1):
            if text[i] in '.!?':
                return i + 1
        
        # Cümle sonu bulunamazsa kelime sınırı ara
        for i in range(max_size, max(0, max_size - 50), -1):
            if text[i] in ' \n\t':
                return i
        
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
    
    # ÇOK BASIT VE TEMİZ YAKLAŞIM: Overlap'ı devre dışı bırak
    # Overlap'siz ama kaliteli chunk'lar, overlap'li ama kötü chunk'lardan iyidir
    if chunk_overlap > 0:
        logger.warning("Overlap devre dışı bırakıldı - kelime kesikleri önlemek için")
    
    return final_sections
    
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


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    strategy: Literal["char", "paragraph", "sentence", "markdown", "semantic", "hybrid"] = "char",
    language: str = "auto",
) -> List[str]:
    """
    Splits a text into overlapping chunks using various strategies.

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
                  - "semantic": LLM-based semantic chunking
                  - "hybrid": Combination of markdown + semantic analysis
        language: Language of the text for semantic analysis ("tr", "en", or "auto")

    Returns:
        A list of text chunks.
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
        # Gelişmiş cümle chunking - Türkçe desteği ile
        import re
        
        # Türkçe noktalama ve cümle sonları için gelişmiş regex
        sentence_endings = r"[\.\!\?]+(?:\s+|$)"
        parts = re.split(f"({sentence_endings})", normalized)
        
        sentences: List[str] = []
        i = 0
        while i < len(parts):
            base = parts[i].strip() if i < len(parts) else ""
            punct = parts[i+1].strip() if i+1 < len(parts) else ""
            
            if base:
                # Cümleyi birleştir
                sentence = (base + " " + punct).strip()
                # Çok kısa cümlecikleri birleştir
                if sentences and len(sentence) < 50:
                    sentences[-1] = sentences[-1] + " " + sentence
                else:
                    sentences.append(sentence)
            i += 2
        
        # Boş cümleleri temizle
        sentences = [s for s in sentences if s.strip()]
        chunks = _group_units(sentences, chunk_size, chunk_overlap)
    elif strategy == "markdown":
        # Markdown yapısına dayalı akıllı chunking
        chunks = _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        # LLM tabanlı semantic chunking
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
            except Exception as e:
                logger.error(f"Semantic chunking failed: {e}")
                logger.info("Falling back to markdown strategy")
                chunks = _chunk_by_markdown_structure(normalized, chunk_size, chunk_overlap)
        else:
            logger.warning("Semantic chunking not available, falling back to markdown")
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
