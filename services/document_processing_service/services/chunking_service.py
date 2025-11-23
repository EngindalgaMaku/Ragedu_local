"""
Text chunking service
Handles text splitting and chunk processing with unified chunking system
"""
import re
import sys
from pathlib import Path
from typing import List
from fastapi import HTTPException
from utils.logger import logger

# Import UNIFIED chunking system with LLM post-processing support
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
try:
    from src.text_processing.text_chunker import chunk_text
    from src.text_processing.lightweight_chunker import create_semantic_chunks
    UNIFIED_CHUNKING_AVAILABLE = True
    logger.info("✅ UNIFIED chunking system imported successfully with Turkish support")
except ImportError as e:
    UNIFIED_CHUNKING_AVAILABLE = False
    logger.warning(f"⚠️ CRITICAL: Unified chunking system not available: {e}")


def chunk_text_with_strategy(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "lightweight",
    use_llm_post_processing: bool = False,
    llm_model_name: str = "llama-3.1-8b-instant",
    model_inference_url: str = None
) -> List[str]:
    """
    Split text into chunks using the specified strategy
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy ('lightweight' or 'semantic')
        use_llm_post_processing: Whether to use LLM for chunk refinement
        llm_model_name: LLM model for post-processing
        model_inference_url: Model inference service URL
        
    Returns:
        List of text chunks
        
    Raises:
        HTTPException: If chunking fails
    """
    if not UNIFIED_CHUNKING_AVAILABLE:
        logger.error("❌ CRITICAL: Unified chunking system not available and no fallback exists")
        raise HTTPException(
            status_code=500,
            detail="Critical system error: Unified chunking system not available"
        )
    
    logger.info(
        f"🚀 USING UNIFIED CHUNKING SYSTEM: strategy='{strategy}', "
        f"size={chunk_size}, overlap={chunk_overlap}, "
        f"llm_post_processing={use_llm_post_processing}"
    )
    
    try:
        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            use_llm_post_processing=use_llm_post_processing,
            llm_model_name=llm_model_name,
            model_inference_url=model_inference_url
        )
        
        if use_llm_post_processing:
            logger.info(
                f"✅ Unified chunking with LLM post-processing successful: "
                f"{len(chunks)} chunks created"
            )
        else:
            logger.info(
                f"✅ Unified chunking successful: {len(chunks)} chunks created"
            )
        
        return chunks
        
    except Exception as e:
        logger.error(f"❌ CRITICAL: Unified chunking failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Critical chunking system failure: {str(e)}"
        )


def extract_chunk_title_from_content(content: str, fallback_title: str) -> str:
    """
    Extract meaningful title from chunk content
    
    Features:
    - Detects markdown headers (# Header)
    - Extracts first meaningful sentence
    - Truncates to 70 characters
    
    Args:
        content: Chunk content
        fallback_title: Title to use if extraction fails
        
    Returns:
        Extracted or fallback title
    """
    lines = content.split('\n')
    
    # Look for headers first
    for line in lines:
        header_match = re.match(r'^#{1,6}\s+(.+)$', line.strip())
        if header_match:
            return header_match.group(1).strip()
    
    # Look for first meaningful sentence
    for line in lines:
        if line.strip():
            title = line.strip()[:70]
            if len(line.strip()) > 70:
                title += '...'
            return title
    
    return fallback_title






