import os
import uuid
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import requests
import logging
import chromadb
from chromadb.config import Settings

# Import advanced semantic chunking system
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from src.text_processing.text_chunker import chunk_text
    ADVANCED_CHUNKING_AVAILABLE = True
    logging.getLogger(__name__).info("✅ Advanced semantic chunking system imported successfully")
except ImportError as e:
    ADVANCED_CHUNKING_AVAILABLE = False
    logging.getLogger(__name__).warning(f"⚠️ Advanced chunking not available: {e}")

# Import langdetect for language detection
from langdetect import detect, LangDetectException

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing Service",
    description="Text processing and external service integration microservice",
    version="1.0.0"
)

# Pydantic models
class ProcessRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}
    collection_name: Optional[str] = "documents"
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    chunk_strategy: Optional[str] = "semantic"  # NEW: Enable advanced semantic chunking by default

class CRAGEvaluationRequest(BaseModel):
    query: str
    retrieved_docs: List[Dict[str, Any]]

class CRAGEvaluationResponse(BaseModel):
    success: bool
    evaluation: Dict[str, Any]
    filtered_docs: List[Dict[str, Any]]

class ProcessResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int
    collection_name: str
    chunk_ids: List[str]

class RAGQueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5
    use_rerank: bool = True
    min_score: float = 0.1
    max_context_chars: int = 8000
    model: Optional[str] = None
    chain_type: Optional[str] = "stuff"
    embedding_model: Optional[str] = None
    max_tokens: Optional[int] = 2048  # Answer length: 1024 (short), 2048 (normal), 4096 (detailed)
    conversation_history: Optional[List[Dict[str, str]]] = None  # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    chain_type: Optional[str] = None

# Environment variables - Google Cloud Run compatible
# For Docker: use service names (e.g., http://model-inference-service:8003)
# For Cloud Run: use full URLs (e.g., https://model-inference-xxx.run.app)
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", os.getenv("MODEL_INFERENCE_URL", None))
if not MODEL_INFERENCER_URL:
    MODEL_INFERENCE_HOST = os.getenv("MODEL_INFERENCE_HOST", "model-inference-service")
    MODEL_INFERENCE_PORT = os.getenv("MODEL_INFERENCE_PORT", "8003")
    if MODEL_INFERENCE_HOST.startswith("http://") or MODEL_INFERENCE_HOST.startswith("https://"):
        MODEL_INFERENCER_URL = MODEL_INFERENCE_HOST
    else:
        MODEL_INFERENCER_URL = f"http://{MODEL_INFERENCE_HOST}:{MODEL_INFERENCE_PORT}"

CHROMADB_URL = os.getenv("CHROMADB_URL", None)
if not CHROMADB_URL:
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "chromadb-service")
    CHROMADB_PORT = os.getenv("CHROMADB_PORT", "8004")
    if CHROMADB_HOST.startswith("http://") or CHROMADB_HOST.startswith("https://"):
        CHROMADB_URL = CHROMADB_HOST
    else:
        CHROMADB_URL = f"http://{CHROMADB_HOST}:{CHROMADB_PORT}"

CHROMA_SERVICE_URL = os.getenv("CHROMA_SERVICE_URL", CHROMADB_URL)  # Use CHROMADB_URL as fallback
PORT = int(os.getenv("PORT", "8080"))  # Cloud Run default port
MIN_SIMILARITY_DEFAULT = float(os.getenv("MIN_SIMILARITY_DEFAULT", "0.5"))

# ChromaDB Client Setup - Google Cloud Run compatible
def get_chroma_client():
    """Get ChromaDB client with connection to our service"""
    try:
        # Parse CHROMA_SERVICE_URL properly for HttpClient
        logger.info(f"🔍 DIAGNOSTIC: Creating ChromaDB client with URL: {CHROMA_SERVICE_URL}")
        
        # Check if URL is Cloud Run format (https://xxx.run.app) or Docker format (http://host:port)
        if CHROMA_SERVICE_URL.startswith("https://"):
            # Cloud Run: use full URL
            from urllib.parse import urlparse
            parsed = urlparse(CHROMA_SERVICE_URL)
            host = parsed.hostname
            port = parsed.port or 443  # HTTPS default port
            use_https = True
        elif CHROMA_SERVICE_URL.startswith("http://"):
            # Docker or local: extract host and port
            chroma_url = CHROMA_SERVICE_URL.replace("http://", "")
            if ":" in chroma_url:
                host_parts = chroma_url.split(":")
                host = host_parts[0]
                port = int(host_parts[1])
            else:
                host = chroma_url
                port = 8000
            use_https = False
        else:
            # Fallback: assume Docker format
            chroma_url = CHROMA_SERVICE_URL.replace("http://", "").replace("https://", "")
            if ":" in chroma_url:
                host_parts = chroma_url.split(":")
                host = host_parts[0]
                port = int(host_parts[1])
            else:
                host = chroma_url
                port = 8000
            use_https = False
        
        logger.info(f"🔍 DIAGNOSTIC: Connecting to ChromaDB at host='{host}', port={port}, https={use_https}")
        
        # Create HttpClient with proper host/port configuration
        # Note: chromadb.HttpClient doesn't support HTTPS directly for Cloud Run
        # For Cloud Run, you may need to use a different client or proxy
        if use_https:
            # For Cloud Run, try to use the full URL
            # ChromaDB HttpClient may need special configuration for HTTPS
            logger.warning("⚠️ HTTPS detected for ChromaDB. Ensure ChromaDB service supports HTTPS or use HTTP proxy.")
            # Try to connect - ChromaDB HttpClient may handle HTTPS URLs
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False)
            )
        
        logger.info(f"✅ ChromaDB client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create ChromaDB client: {e}")
        raise

class DocumentProcessor:
    def __init__(self):
        pass
        
    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Split text into semantic chunks that respect sentence boundaries and markdown structure
        Uses advanced semantic chunking instead of basic regex splitting
        """
        if not text:
            return []
        
        # Clean up the text
        text = text.strip()
        if not text:
            return []
        
        # Detect language
        try:
            language = detect(text)
            logger.info(f"Detected language: {language}")
        except LangDetectException:
            language = "en"
            logger.warning("Could not detect language, defaulting to English")
        
        # Use markdown-aware semantic chunking
        chunks = self.create_markdown_aware_semantic_chunks(
            text=text,
            target_size=chunk_size,
            overlap_ratio=chunk_overlap / chunk_size if chunk_size > 0 else 0.1,
            language=language
        )
        
        logger.info(f"Text split into {len(chunks)} chunks using markdown-aware semantic chunking")
        return chunks
    
    def create_markdown_aware_semantic_chunks(
        self,
        text: str,
        target_size: int = 1000,
        overlap_ratio: float = 0.1,
        language: str = "auto"
    ) -> List[str]:
        """
        Creates intelligent semantic chunks based on markdown structure and content.
        
        Features:
        1. Header-based chunking (respects topic boundaries)
        2. Question-answer grouping
        3. List preservation
        4. Sentence boundary respect
        5. Named chunks based on content
        """
        
        # Use advanced semantic chunking
        return self._create_intelligent_semantic_chunks(text, target_size, overlap_ratio, language)
    
    def _create_intelligent_semantic_chunks(
        self,
        text: str,
        target_size: int = 1000,
        overlap_ratio: float = 0.1,
        language: str = "auto"
    ) -> List[str]:
        """
        Advanced semantic chunking with topic awareness
        """
        
        # First, identify document structure
        sections = self._identify_document_sections(text)
        
        if not sections:
            logger.warning("No sections found, falling back to basic chunking")
            return self._fallback_chunking(text, target_size)
        
        chunks = []
        
        for section in sections:
            section_chunks = self._process_section_intelligently(
                section, target_size, overlap_ratio
            )
            chunks.extend(section_chunks)
        
        logger.info(f"Generated {len(chunks)} intelligent semantic chunks")
        return chunks
    
    def _split_into_markdown_blocks(self, text: str) -> List[str]:
        """
        Split text into semantic blocks that respect markdown structure.
        
        Returns a list of text blocks where each block is a semantically
        meaningful unit (paragraph, header, code block, list, etc.)
        """
        lines = text.split('\n')
        blocks = []
        current_block_lines = []
        in_code_block = False
        in_list = False
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Handle code blocks (preserve them as single blocks)
            if stripped_line.startswith('```'):
                if in_code_block:
                    # End of code block
                    current_block_lines.append(line)
                    blocks.append('\n'.join(current_block_lines))
                    current_block_lines = []
                    in_code_block = False
                    in_list = False
                else:
                    # Start of code block - save any previous block
                    if current_block_lines:
                        blocks.append('\n'.join(current_block_lines))
                        current_block_lines = []
                    current_block_lines.append(line)
                    in_code_block = True
                    in_list = False
                continue
            
            # If we're in a code block, just add the line
            if in_code_block:
                current_block_lines.append(line)
                continue
            
            # Handle markdown headers (they start new blocks)
            if stripped_line.startswith('#'):
                # Save previous block
                if current_block_lines:
                    blocks.append('\n'.join(current_block_lines))
                    current_block_lines = []
                current_block_lines.append(line)
                in_list = False
                continue
            
            # Handle lists
            if re.match(r'^\s*[\-\+\*]\s', line) or re.match(r'^\s*\d+\.\s', line):
                if not in_list and current_block_lines:
                    # Start of new list - save previous block
                    blocks.append('\n'.join(current_block_lines))
                    current_block_lines = []
                current_block_lines.append(line)
                in_list = True
                continue
            
            # Handle empty lines
            if not stripped_line:
                if current_block_lines and not in_list:
                    # Empty line ends a paragraph (but not a list)
                    current_block_lines.append(line)
                    blocks.append('\n'.join(current_block_lines))
                    current_block_lines = []
                    in_list = False
                elif current_block_lines:
                    # Keep empty line as part of current block
                    current_block_lines.append(line)
                continue
            
            # Regular text line
            if in_list and not re.match(r'^\s*[\-\+\*]\s', line) and not re.match(r'^\s*\d+\.\s', line):
                # End of list
                if current_block_lines:
                    blocks.append('\n'.join(current_block_lines))
                    current_block_lines = []
                in_list = False
            
            current_block_lines.append(line)
        
        # Don't forget the last block
        if current_block_lines:
            blocks.append('\n'.join(current_block_lines))
        
        # Filter out empty blocks
        blocks = [block.strip() for block in blocks if block.strip()]
        
        return blocks
    
    def get_embeddings(self, texts: List[str], embedding_model: str = "nomic-embed-text") -> List[List[float]]:
        """Get embeddings from local model inference service (Ollama)"""
        try:
            return self._get_local_embeddings(texts, embedding_model)
                
        except Exception as e:
            # Don't raise HTTPException - let caller handle fallback
            logger.warning(f"Embedding service error with {embedding_model}: {str(e)}")
            raise  # Re-raise to allow fallback mechanism
    
    def _get_local_embeddings(self, texts: List[str], embedding_model: str) -> List[List[float]]:
        """Get embeddings from model inference service"""
        embed_url = f"{MODEL_INFERENCER_URL}/embed"
        
        logger.info(f"Getting embeddings for {len(texts)} texts using model: {embedding_model}")
        
        # Send all texts in a single request for efficiency
        # Increased timeout to handle multiple chunks with slow local Ollama embeddings
        response = requests.post(
            embed_url,
            json={"texts": texts, "model": embedding_model},
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes for multiple chunks with slow local embeddings
        )
        
        if response.status_code != 200:
            logger.error(f"Local embedding error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get local embeddings: {response.status_code}"
            )
        
        embedding_data = response.json()
        embeddings = embedding_data.get("embeddings", [])
        
        if len(embeddings) != len(texts):
            raise HTTPException(
                status_code=500,
                detail=f"Embedding count ({len(embeddings)}) doesn't match text count ({len(texts)})"
            )
        
        logger.info(f"Successfully retrieved {len(embeddings)} local embeddings")
        return embeddings

    def _identify_document_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Identifies sections in a markdown document based on headers.
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []

        for i, line in enumerate(lines):
            header_match = re.match(r'^(#+)\s+(.*)', line)
            if header_match:
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    current_section['end_line'] = i - 1
                    if current_section['content']:
                        sections.append(current_section)

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    'title': title,
                    'level': level,
                    'start_line': i,
                    'type': 'header' if 'soru' not in title.lower() else 'question'
                }
                current_content = []
            else:
                if current_section is None:
                    current_section = {
                        'title': 'Giriş', 'level': 0, 'start_line': 0, 'type': 'text'
                    }
                    current_content = []
                current_content.append(line)
        
        if current_section and current_content:
            current_section['content'] = '\n'.join(current_content).strip()
            current_section['end_line'] = len(lines) - 1
            if current_section['content']:
                sections.append(current_section)
        
        if not sections and text.strip():
            sections.append({
                'title': 'Genel İçerik', 'level': 0, 'start_line': 0,
                'end_line': len(lines) - 1, 'content': text, 'type': 'text'
            })

        return sections

    def _process_section_intelligently(
        self,
        section: dict,
        target_size: int,
        overlap_ratio: float
    ) -> List[str]:
        """
        Process a section intelligently based on its type and content
        """
        content = section['content']
        section_type = section.get('type', 'text')
        title = section.get('title', 'Untitled')
        
        # If section is small enough, keep as single chunk
        if len(content) <= target_size * 1.2:
            chunk_title = self._generate_chunk_title(section_type, title, content)
            return [f"# {chunk_title}\n\n{content}"]
        
        # For larger sections, split intelligently
        if section_type == 'header':
            return self._split_header_section(content, title, target_size, overlap_ratio)
        elif section_type == 'question':
            return self._split_question_section(content, title, target_size, overlap_ratio)
        else:
            return self._split_generic_section(content, title, target_size, overlap_ratio)
    
    def _generate_chunk_title(self, section_type: str, title: str, content: str) -> str:
        """
        Generate meaningful chunk titles based on content
        """
        if section_type == 'header':
            return title
        elif section_type == 'question':
            return f"Soru: {title}"
        else:
            # Extract first meaningful sentence or phrase
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10 and not line.startswith('#'):
                    # Take first 50 characters as title
                    return line[:50] + ('...' if len(line) > 50 else '')
            return "Metin Bölümü"
    
    def _split_header_section(
        self,
        content: str,
        title: str,
        target_size: int,
        overlap_ratio: float
    ) -> List[str]:
        """
        Split header section while preserving sub-structure
        """
        chunks = []
        lines = content.split('\n')
        current_chunk_lines = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # Check if this is a sub-header
            if re.match(r'^#{2,6}\s+', line.strip()):
                # If we have accumulated content, save it as a chunk
                if current_chunk_lines and current_size > 200:
                    chunk_content = '\n'.join(current_chunk_lines)
                    chunk_title = self._extract_chunk_title_from_content(chunk_content, title)
                    chunks.append(f"# {chunk_title}\n\n{chunk_content}")
                    
                    # Start new chunk with overlap
                    overlap_lines = int(len(current_chunk_lines) * overlap_ratio)
                    current_chunk_lines = current_chunk_lines[-overlap_lines:] if overlap_lines > 0 else []
                    current_size = sum(len(line) + 1 for line in current_chunk_lines)
            
            current_chunk_lines.append(line)
            current_size += line_size
            
            # If chunk is getting too large, split at sentence boundary
            if current_size > target_size:
                split_point = self._find_sentence_boundary(current_chunk_lines, target_size)
                if split_point > 0:
                    chunk_lines = current_chunk_lines[:split_point]
                    chunk_content = '\n'.join(chunk_lines)
                    chunk_title = self._extract_chunk_title_from_content(chunk_content, title)
                    chunks.append(f"# {chunk_title}\n\n{chunk_content}")
                    
                    # Continue with remaining lines plus overlap
                    overlap_lines = int(split_point * overlap_ratio)
                    current_chunk_lines = current_chunk_lines[split_point-overlap_lines:]
                    current_size = sum(len(line) + 1 for line in current_chunk_lines)
        
        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk_title = self._extract_chunk_title_from_content(chunk_content, title)
            chunks.append(f"# {chunk_title}\n\n{chunk_content}")
        
        return chunks
    
    def _split_question_section(
        self,
        content: str,
        title: str,
        target_size: int,
        overlap_ratio: float
    ) -> List[str]:
        """
        Split question sections, keeping Q&A pairs together
        """
        chunks = []
        lines = content.split('\n')
        current_qa_pair = []
        current_size = 0
        
        for line in lines:
            current_qa_pair.append(line)
            current_size += len(line) + 1
            
            # If we hit a new question and current chunk is large enough
            if (line.strip().endswith('?') and
                len(current_qa_pair) > 1 and
                current_size > target_size * 0.7):
                
                chunk_content = '\n'.join(current_qa_pair)
                chunk_title = f"Soru: {current_qa_pair[0].strip()[:50]}..."
                chunks.append(f"# {chunk_title}\n\n{chunk_content}")
                
                current_qa_pair = []
                current_size = 0
        
        # Last chunk
        if current_qa_pair:
            chunk_content = '\n'.join(current_qa_pair)
            chunk_title = f"Soru: {title}"
            chunks.append(f"# {chunk_title}\n\n{chunk_content}")
        
        return chunks
    
    def _split_generic_section(
        self,
        content: str,
        title: str,
        target_size: int,
        overlap_ratio: float
    ) -> List[str]:
        """
        Split generic text sections at sentence boundaries
        """
        chunks = []
        sentences = self._split_into_sentences(content)
        
        current_chunk_sentences = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence) + 1
            
            # If adding this sentence would exceed target size
            if current_size + sentence_size > target_size and current_chunk_sentences:
                chunk_content = ' '.join(current_chunk_sentences)
                chunk_title = self._extract_chunk_title_from_content(chunk_content, title)
                chunks.append(f"# {chunk_title}\n\n{chunk_content}")
                
                # Start new chunk with overlap
                overlap_count = int(len(current_chunk_sentences) * overlap_ratio)
                current_chunk_sentences = current_chunk_sentences[-overlap_count:] if overlap_count > 0 else []
                current_size = sum(len(s) + 1 for s in current_chunk_sentences)
            
            current_chunk_sentences.append(sentence)
            current_size += sentence_size
        
        # Last chunk
        if current_chunk_sentences:
            chunk_content = ' '.join(current_chunk_sentences)
            chunk_title = self._extract_chunk_title_from_content(chunk_content, title)
            chunks.append(f"# {chunk_title}\n\n{chunk_content}")
        
        return chunks
    
    def _extract_chunk_title_from_content(self, content: str, fallback_title: str) -> str:
        """
        Extract meaningful title from chunk content
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
                return line.strip()[:70] + ('...' if len(line.strip()) > 70 else '')
        return fallback_title

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling Turkish punctuation
        """
        # Turkish sentence endings
        sentence_endings = r'[.!?…]\s+'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _find_sentence_boundary(self, lines: List[str], target_size: int) -> int:
        """
        Find the best sentence boundary to split at
        """
        current_size = 0
        best_split = len(lines) // 2  # fallback
        
        for i, line in enumerate(lines):
            current_size += len(line) + 1
            
            # Look for sentence endings
            if (re.search(r'[.!?]\s*$', line.strip()) and
                current_size >= target_size * 0.7):
                return i + 1
            
            if current_size >= target_size:
                return max(1, i)  # Don't return 0
        
        return best_split
    
    def _fallback_chunking(self, text: str, target_size: int) -> List[str]:
        """
        Fallback to basic chunking if advanced parsing fails
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1
            
            if current_size + word_size > target_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(word)
            current_size += word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class CRAGEvaluator:
    """
    Corrective RAG (CRAG) Evaluator - REAL IMPLEMENTATION
    
    Uses a cross-encoder model via the model-inference-service to get
    actual relevance scores for query-document pairs.
    """
    
    def __init__(self, model_inference_url: str):
        self.model_inference_url = model_inference_url
        self.rerank_url = f"{self.model_inference_url}/rerank"
        self.correct_threshold = 0.5    # Calibrated for ms-marco-MiniLM-L-6-v2
        self.incorrect_threshold = 0.01 # Very low confidence threshold for rejecting all docs
        self.filter_threshold = 0.1     # Individual document filter threshold
        logger.info(f"CRAGEvaluator initialized with rerank URL: {self.rerank_url}")

    def evaluate_retrieved_docs(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate retrieved documents using a real cross-encoder model.
        """
        if not retrieved_docs:
            return {"action": "reject", "confidence": 0.0, "avg_score": 0.0, "filtered_docs": [], "evaluation_scores": []}

        # Prepare documents for the rerank service
        docs_to_rerank = [doc.get("content") or doc.get("text", "") for doc in retrieved_docs]
        
        try:
            logger.info(f"▶️ Calling rerank service for CRAG evaluation. Query: '{query[:50]}...', Docs: {len(docs_to_rerank)}")
            response = requests.post(
                self.rerank_url,
                json={"query": query, "documents": docs_to_rerank},
                timeout=60
            )
            response.raise_for_status()
            rerank_results = response.json().get("results", [])
            logger.info(f"◀️ Rerank service returned {len(rerank_results)} results.")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ CRITICAL: Rerank service call failed: {e}. Cannot perform CRAG evaluation.")
            # Fail open: if reranker fails, accept the documents to not block the user.
            # This is a production-friendly choice.
            return {"action": "accept", "confidence": 0.5, "avg_score": 0.5, "filtered_docs": retrieved_docs, "evaluation_scores": [], "error": str(e)}

        # Process rerank results
        evaluation_scores = []
        updated_docs = []
        for i, doc in enumerate(retrieved_docs):
            # Find the corresponding rerank result
            rerank_score = 0.0
            for res in rerank_results:
                if res.get("index") == i:
                    rerank_score = res.get("relevance_score", 0.0)
                    break
            
            # The final score is the cross-encoder's relevance score
            final_score = rerank_score
            
            # Update the document with the new, more accurate score
            doc["crag_score"] = final_score
            # Keep original 'score' as the similarity score for comparison
            updated_docs.append(doc)

            evaluation_scores.append({
                "index": i,
                "final_score": round(final_score, 4)
            })

        # Sort documents by the new CRAG score
        updated_docs.sort(key=lambda x: x["crag_score"], reverse=True)
        
        # --- CRAG Decision Logic based on REAL scores ---
        if not updated_docs:
            return {"action": "reject", "confidence": 0.0, "avg_score": 0.0, "filtered_docs": [], "evaluation_scores": []}

        scores = [doc["crag_score"] for doc in updated_docs]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        if max_score >= self.correct_threshold:
            action = "accept"
            filtered_docs = updated_docs
            logger.info(f"✅ CRAG ACCEPT: Max score {max_score:.3f} is high.")
        elif max_score < self.incorrect_threshold:
            action = "reject"
            filtered_docs = []
            logger.info(f"❌ CRAG REJECT: Max score {max_score:.3f} is very low.")
        else:
            action = "filter"
            filtered_docs = [doc for doc in updated_docs if doc["crag_score"] >= self.filter_threshold]
            logger.info(f"🔍 CRAG FILTER: {len(filtered_docs)}/{len(updated_docs)} docs passed filter (threshold: {self.filter_threshold})")
            if not filtered_docs:
                action = "reject" # If filtering removes all docs, it's a rejection.
                logger.info("❌ CRAG REJECT: All documents were filtered out.")

        return {
            "action": action,
            "confidence": round(max_score, 3),
            "avg_score": round(avg_score, 3),
            "filtered_docs": filtered_docs,
            "evaluation_scores": evaluation_scores,
            "thresholds": {
                "correct": self.correct_threshold,
                "incorrect": self.incorrect_threshold,
                "filter": self.filter_threshold
            }
        }

# Global processor instance
processor = DocumentProcessor()

@app.on_event("startup")
async def startup_event():
    """Fast startup - connections are lazy loaded"""
    logger.info("Document Processing Service starting...")
    logger.info(f"Service running on port {PORT}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Document Processing Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        model_service_status = False
        
        # Test model inference service
        try:
            health_response = requests.get(f"{MODEL_INFERENCER_URL}/health", timeout=5)
            model_service_status = health_response.status_code == 200
        except Exception as e:
            logger.debug(f"Model service health check failed: {e}")
        
        return {
            "status": "healthy",
            "text_processing_available": True,  # We now use our own implementation
            "model_service_connected": model_service_status,
            "model_inferencer_url": MODEL_INFERENCER_URL,
            "port": PORT
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "healthy",  # Base service is running
            "error": str(e),
            "text_processing_available": True,  # We now use our own implementation
            "model_service_connected": False,
            "port": PORT
        }

@app.post("/process-and-store", response_model=ProcessResponse)
async def process_and_store(request: ProcessRequest):
    """
    Process text block and return chunks with embeddings
    
    1. Split text into chunks using regex-based approach
    2. Get embeddings for each chunk
    3. Return processed data (storage happens in external service)
    """
    try:
        logger.info(f"Starting text processing. Text length: {len(request.text)} characters")
        
        # CRITICAL FIX: Use advanced semantic chunking instead of basic split_text
        chunk_size = request.chunk_size or 1000
        chunk_overlap = request.chunk_overlap or 200
        chunk_strategy = request.chunk_strategy or "semantic"
        
        if ADVANCED_CHUNKING_AVAILABLE and chunk_strategy != "char":
            # Use advanced semantic chunking system with AST markdown parsing
            logger.info(f"🚀 USING ADVANCED SEMANTIC CHUNKING: strategy='{chunk_strategy}', size={chunk_size}, overlap={chunk_overlap}")
            try:
                chunks = chunk_text(
                    text=request.text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    strategy=chunk_strategy,
                    language="auto",
                    use_embedding_refinement=True
                )
                logger.info(f"✅ Advanced chunking successful: {len(chunks)} semantic chunks created")
            except Exception as e:
                logger.warning(f"⚠️ Advanced chunking failed, falling back to basic: {e}")
                chunks = processor.split_text(request.text, chunk_size, chunk_overlap)
        else:
            # Fallback to basic chunking for char strategy or when advanced chunking unavailable
            logger.info(f"📝 Using basic chunking: strategy='{chunk_strategy}' (advanced_available={ADVANCED_CHUNKING_AVAILABLE})")
            chunks = processor.split_text(request.text, chunk_size, chunk_overlap)
        
        if not chunks:
            logger.warning("Text could not be split into any chunks.")
            raise HTTPException(status_code=400, detail="Text could not be split into chunks")
        
        logger.info(f"Successfully split text into {len(chunks)} chunks.")

        # Get embeddings - check for embedding model preference in metadata
        embedding_model = request.metadata.get("embedding_model", "nomic-embed-text")
        logger.info(f"Using embedding model: {embedding_model}")
        embeddings = processor.get_embeddings(chunks, embedding_model)
        
        if len(embeddings) != len(chunks):
            logger.error(f"Mismatch between chunk count ({len(chunks)}) and embedding count ({len(embeddings)}).")
            raise HTTPException(
                status_code=500,
                detail="Embedding count doesn't match chunk count"
            )
        
        # Generate chunk IDs
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Use default collection name if None, and ensure ChromaDB compatibility
        collection_name = request.collection_name or "documents"
        logger.info(f"🔍 DIAGNOSTIC: Initial collection_name: '{collection_name}'")
        
        # If collection name starts with "session_", use just the UUID part (ChromaDB rejects "session_" prefix)
        if collection_name.startswith("session_"):
            session_id = collection_name[8:]  # Remove "session_" prefix
            logger.info(f"🔍 DIAGNOSTIC: Extracted session_id: '{session_id}' (length: {len(session_id)})")
            
            # Convert 32-char hex string to proper UUID format (8-4-4-4-12)
            if len(session_id) == 32 and session_id.replace('-', '').isalnum():
                original_collection_name = collection_name
                collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
                logger.info(f"🔍 DIAGNOSTIC: Transformed '{original_collection_name}' -> '{collection_name}'")
                logger.info(f"Using pure UUID as collection name: {collection_name}")
            elif len(session_id) == 36:  # Already formatted UUID
                collection_name = session_id
                logger.info(f"🔍 DIAGNOSTIC: Session ID already in UUID format: '{collection_name}'")
                logger.info(f"Using existing UUID as collection name: {collection_name}")
            else:
                logger.warning(f"🔍 DIAGNOSTIC: Unusual session_id format: '{session_id}' (length: {len(session_id)}, isalnum: {session_id.replace('-', '').isalnum()})")
        
        # Store chunks and embeddings in ChromaDB
        if not CHROMA_SERVICE_URL:
            logger.error("ChromaDB service URL not configured")
            raise HTTPException(
                status_code=500,
                detail="ChromaDB service URL not configured"
            )
        
        try:
            # Create a list of metadata for each chunk
            # Sanitize metadata to ensure all values are ChromaDB-compliant types (str, int, float, bool)
            # Convert lists and dicts to JSON strings for compatibility
            logger.info(f"🔍 METADATA DEBUG: Raw metadata received: {request.metadata}")
            logger.info(f"🔍 METADATA DEBUG: Raw metadata type: {type(request.metadata)}")
            logger.info(f"🔍 METADATA DEBUG: Raw metadata length: {len(request.metadata) if request.metadata else 0}")
            
            if not request.metadata:
                logger.warning(f"🔍 METADATA DEBUG: No metadata received in request!")
                sanitized_metadata = {}
            else:
                sanitized_metadata = {}
                for key, value in request.metadata.items():
                    logger.info(f"🔍 METADATA DEBUG: Processing key='{key}', value={repr(value)}, type={type(value)}")
                    if isinstance(value, (str, int, float, bool)):
                        sanitized_metadata[key] = value
                        logger.info(f"🔍 METADATA DEBUG: Added {key}={value} (primitive type)")
                    elif isinstance(value, (list, dict)):
                        # Convert lists and dicts to JSON strings
                        json_value = json.dumps(value)
                        sanitized_metadata[key] = json_value
                        logger.info(f"🔍 METADATA DEBUG: Converted metadata key '{key}' from {type(value)} to JSON string: {json_value}")
                    else:
                        logger.warning(f"🔍 METADATA DEBUG: Excluding non-compliant metadata key '{key}' of type {type(value)}.")
            
            logger.info(f"🔍 METADATA DEBUG: Final sanitized metadata: {sanitized_metadata}")

            # Create the list of metadatas for each chunk with position info
            chunk_metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = sanitized_metadata.copy()
                chunk_metadata["chunk_index"] = i + 1
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["chunk_length"] = len(chunk)
                
                # IMPORTANT: Add session_id to metadata for extra security/validation
                # This ensures even if collection names are somehow mixed, we can filter by session_id
                chunk_metadata["session_id"] = collection_name  # collection_name is the session_id
                
                # Add first few words as chunk preview for identification
                chunk_preview = chunk.strip()[:100].replace('\n', ' ').replace('\r', '')
                if len(chunk_preview) == 100:
                    chunk_preview += "..."
                chunk_metadata["chunk_preview"] = chunk_preview
                
                # Extract chunk title from content (if chunk starts with #)
                chunk_title = processor._extract_chunk_title_from_content(chunk, f"Bölüm {i + 1}")
                chunk_metadata["chunk_title"] = chunk_title
                
                chunk_metadatas.append(chunk_metadata)

            # NEW: Use ChromaDB Python Client instead of HTTP requests
            logger.info(f"🚀 NEW APPROACH: Using ChromaDB Python Client for collection '{collection_name}'")
            
            # Get ChromaDB client
            client = get_chroma_client()
            logger.info(f"✅ ChromaDB client connected successfully")
            
            # DETAILED PAYLOAD LOGGING - Analyze exact data being sent to ChromaDB
            logger.info(f"🔍 PAYLOAD ANALYSIS: Preparing data for ChromaDB")
            logger.info(f"🔍 PAYLOAD: chunks count: {len(chunks)}")
            logger.info(f"🔍 PAYLOAD: embeddings count: {len(embeddings)}")
            logger.info(f"🔍 PAYLOAD: metadatas count: {len(chunk_metadatas)}")
            logger.info(f"🔍 PAYLOAD: ids count: {len(chunk_ids)}")
            
            if chunks:
                logger.info(f"🔍 PAYLOAD: first chunk preview: {chunks[0][:100]}...")
            if embeddings:
                logger.info(f"🔍 PAYLOAD: first embedding sample: {embeddings[0][:5]}...")
            if chunk_metadatas:
                logger.info(f"🔍 PAYLOAD: first metadata: {json.dumps(chunk_metadatas[0])}")
            if chunk_ids:
                logger.info(f"🔍 PAYLOAD: first id: {chunk_ids[0]}")
                
            # Get or create collection using ChromaDB client with cosine distance
            logger.info(f"🔧 Getting or creating collection '{collection_name}' with cosine distance")
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"created_by": "document_processing_service", "hnsw:space": "cosine"}
            )
            logger.info(f"✅ Collection '{collection_name}' ready with cosine distance")
            
            # Add documents to collection
            logger.info(f"🔧 Adding {len(chunks)} documents to collection")
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            logger.info(f"🎉 SUCCESS: Added {len(chunks)} documents to collection '{collection_name}'")

        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to store chunks in ChromaDB collection '{collection_name}' using ChromaDB client: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store chunks in ChromaDB collection '{collection_name}': {str(e)}"
            )
        
        logger.info(f"Processing completed. {len(chunks)} chunks processed and stored.")
        
        return ProcessResponse(
            success=True,
            message=f"Successfully processed and stored: {len(chunks)} chunks",
            chunks_processed=len(chunks),
            collection_name=collection_name,
            chunk_ids=chunk_ids
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    RAG Query endpoint - Uses ChromaDB for retrieval and Model Inference for generation
    """
    try:
        logger.info(f"RAG query received for session: {request.session_id}")
        chain_type = (request.chain_type or "stuff").lower()
        # Fetch collection
        # Try to call ChromaDB service directly
        try:
            # Search in ChromaDB (use pure UUID as collection name - no "session_" prefix)
            session_id = request.session_id
            logger.info(f"🔍 DIAGNOSTIC: RAG Query - Initial session_id: '{session_id}' (length: {len(session_id)})")
            
            if len(session_id) == 32 and session_id.replace('-', '').isalnum():
                # Convert 32-char hex string to proper UUID format
                collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
                logger.info(f"🔍 DIAGNOSTIC: RAG Query - Transformed session_id '{session_id}' -> collection_name '{collection_name}'")
                logger.info(f"Using pure UUID as collection name for query: {collection_name}")
            elif len(session_id) == 36:  # Already formatted UUID
                collection_name = session_id
                logger.info(f"🔍 DIAGNOSTIC: RAG Query - Session ID already in UUID format: '{collection_name}'")
                logger.info(f"Using existing UUID as collection name for query: {collection_name}")
            else:
                # Fallback - use session_id as-is
                collection_name = session_id
                logger.warning(f"🔍 DIAGNOSTIC: RAG Query - Using session_id as-is for collection name (unusual format): '{collection_name}' (length: {len(session_id)})")
                logger.info(f"Using session_id as collection name for query: {collection_name}")
            
            # Use ChromaDB Python Client for querying
            logger.info(f"🔍 Using ChromaDB Python Client to query collection '{collection_name}'")
            
            try:
                client = get_chroma_client()
                
                # Try to get collection - if it doesn't exist, check alternative formats
                try:
                    collection = client.get_collection(name=collection_name)
                    logger.info(f"✅ Found collection '{collection_name}'")
                except Exception as collection_error:
                    # Try alternative collection name formats
                    logger.warning(f"⚠️ Collection '{collection_name}' not found. Trying alternatives...")
                    alternative_names = []
                    
                    # If collection_name is UUID format, try with session_ prefix
                    if '-' in collection_name and len(collection_name) == 36:
                        uuid_part = collection_name.replace('-', '')
                        alternative_names.append(f"session_{uuid_part}")
                    
                    # If session_id is 32-char hex, try both formats
                    # IMPORTANT: Try session_ prefix FIRST (this is how chunks are stored)
                    if len(session_id) == 32:
                        # Try with session_ prefix FIRST (most likely format)
                        alternative_names.insert(0, f"session_{session_id}")
                        # Also try UUID format
                        uuid_format = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
                        if uuid_format != collection_name:
                            alternative_names.append(uuid_format)
                    
                    # Try each alternative
                    collection = None
                    for alt_name in alternative_names:
                        try:
                            logger.info(f"🔍 Trying alternative collection name: '{alt_name}'")
                            collection = client.get_collection(name=alt_name)
                            logger.info(f"✅ Found collection with alternative name: '{alt_name}'")
                            collection_name = alt_name  # Update collection_name for consistency
                            break
                        except:
                            continue
                    
                    if collection is None:
                        # List all collections to help debug
                        try:
                            all_collections = client.list_collections()
                            collection_names = [c.name for c in all_collections]
                            logger.error(f"❌ Collection not found. Available collections: {collection_names}")
                            raise Exception(f"Collection '{collection_name}' not found. Available: {collection_names}")
                        except Exception as list_error:
                            logger.error(f"❌ Could not list collections: {list_error}")
                            raise Exception(f"Collection '{collection_name}' not found and could not list alternatives: {collection_error}")
                
                # Get embeddings for the query using our model inference service
                # Prefer embedding model provided with the request; fallback to default
                embedding_model = (request.embedding_model or os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text"))
                logger.info(f"🔍 Getting embeddings for query via model inference service using {embedding_model}")
                
                # Try multiple embedding models in order of preference
                embedding_models_to_try = [
                    embedding_model,  # Try requested model first
                    "nomic-embed-text",  # Fallback to Ollama model
                    "sentence-transformers/all-MiniLM-L6-v2",  # Try HuggingFace again
                    "BAAI/bge-small-en-v1.5"  # Last resort HuggingFace
                ]
                
                query_embeddings = None
                successful_model = None
                
                for model_to_try in embedding_models_to_try:
                    try:
                        logger.info(f"🔄 Trying embedding model: {model_to_try}")
                        query_embeddings = processor.get_embeddings([request.query], model_to_try)
                        if query_embeddings and len(query_embeddings) > 0 and len(query_embeddings[0]) > 0:
                            successful_model = model_to_try
                            logger.info(f"✅ Successfully got {len(query_embeddings)} query embeddings using {model_to_try}")
                            break
                        else:
                            logger.warning(f"⚠️ Empty embeddings from {model_to_try}")
                    except Exception as emb_error:
                        logger.warning(f"⚠️ Failed to get embeddings with {model_to_try}: {emb_error}")
                        continue
                
                if not query_embeddings or not query_embeddings[0]:
                    raise Exception(f"Failed to generate query embeddings with any model. Tried: {', '.join(embedding_models_to_try)}")
                
                # Query the collection using embeddings (not query_texts)
                search_results = collection.query(
                    query_embeddings=query_embeddings,
                    n_results=request.top_k
                )
                
                # Extract documents from ChromaDB response
                documents = search_results.get('documents', [[]])[0]
                metadatas = search_results.get('metadatas', [[]])[0]
                distances = search_results.get('distances', [[]])[0]
                
                total_found = len(documents)
                logger.info(f"🔍 Query results: {total_found} documents found in collection '{collection_name}'")
                
                # Format context for generation
                context_docs = []
                # Don't filter by similarity - include all retrieved docs since similarity scores vary greatly by embedding model
                # SECURITY: Extra validation - filter by session_id in metadata to ensure we only get documents from this session
                filtered_count = 0
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    # SECURITY CHECK: Verify session_id matches (extra layer of protection)
                    # Even though we query a specific collection, this ensures no data leakage
                    if metadata.get("session_id") and metadata.get("session_id") != collection_name:
                        logger.warning(f"⚠️ SECURITY: Document {i} has mismatched session_id in metadata: {metadata.get('session_id')} != {collection_name}. Skipping.")
                        filtered_count += 1
                        continue
                    
                    # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
                    # Convert to similarity score (1 = identical, 0 = orthogonal)
                    # Use formula: similarity = 1 - distance for cosine distance
                    distance = distances[i] if i < len(distances) else float('inf')
                    logger.info(f"🔍 SIMILARITY DEBUG: Document {i} raw distance: {distance}")
                    
                    # Convert cosine distance to similarity score (0-1)
                    # Use formula: similarity = max(0.0, 1.0 - distance)
                    if distance == float('inf'):
                        similarity = 0.0
                    else:
                        similarity = max(0.0, 1.0 - distance)
                    
                    logger.info(f"🔍 SIMILARITY DEBUG: Document {i} calculated similarity: {similarity}")

                    context_docs.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": similarity
                    })
                
                # CRAG Evaluation with detailed scoring
                crag_evaluator = CRAGEvaluator(model_inference_url=MODEL_INFERENCER_URL)
                crag_evaluation_result = crag_evaluator.evaluate_retrieved_docs(
                    query=request.query,
                    retrieved_docs=context_docs
                )
                
                logger.info(f"🔍 CRAG EVALUATION: {crag_evaluation_result}")
                
                # Apply CRAG decision
                if crag_evaluation_result["action"] == "reject":
                    logger.info("❌ CRAG: Query rejected - low relevance to documents")
                    return RAGQueryResponse(
                        answer="⚠️ **DERS KAPSAMINDA DEĞİL**\n\nSorduğunuz soru ders dökümanlarıyla ilgili görünmüyor. Lütfen ders materyalleri kapsamında sorular sorunuz.",
                        sources=[],
                        chain_type=chain_type
                    )
                elif crag_evaluation_result["action"] == "filter":
                    logger.info(f"🔍 CRAG: Filtering documents - keeping {len(crag_evaluation_result['filtered_docs'])} docs")
                    context_docs = crag_evaluation_result["filtered_docs"]
                else:
                    logger.info("✅ CRAG: Good relevance - using all documents")
                
                # Generate answer using Model Inference Service
                if context_docs:
                    context_text = "\n\n".join([doc["content"] for doc in context_docs])
                    
                    # Truncate if too long
                    max_length = 4000
                    if len(context_text) > max_length:
                        context_text = context_text[:max_length] + "..."
                    
                    # Create Turkish RAG prompt with INTERNAL VERIFICATION - NO VISIBLE ANALYSIS
                    system_prompt = (
                        "Sen yalnızca sağlanan BAĞLAM metnini kullanarak sorulara TÜRKÇE cevap veren bir yapay zeka asistanısın.\n\n"
                        "ÇALIŞMA PRENSİBİN:\n"
                        "Cevap vermeden önce zihninde şunları yap (ama çıktıda HİÇBİR ZAMAN gösterme):\n"
                        "• Bağlamdaki tüm sayısal verileri (yüzdeler, miktarlar, sayılar) tespit et\n"
                        "• Bu verilerin hangi konularla ilgili olduğunu belirle\n"
                        "• Çelişkili bilgi varsa en güvenilir olanı seç\n"
                        "• Bilgilerin tutarlılığını kontrol et\n\n"
                        "ÇIKTI KURALLARI:\n"
                        "1. KESINLIKLE SADECE TÜRKÇE cevap ver\n"
                        "2. Zihninde doğruladığın sayısal verileri AYNEN kullan\n"
                        "3. Kendi bilgini kullanma, sadece bağlamdaki bilgileri kullan\n"
                        "4. Sorunun cevabı bağlamda yoksa: 'Bu bilgi ders dökümanlarında bulunamamıştır.'\n"
                        "5. SADECE NİHAİ CEVABI YAZ - analiz sürecini, adımları, düşünceleri gösterme\n\n"
                        "Örnek: Bağlamda 'azot %78' yazıyorsa kesinlikle %78 yaz, başka değer yazma."
                    )
                    
                    # Build conversation context if available
                    context_parts = [f"System: {system_prompt}\n"]
                    if request.conversation_history:
                        context_parts.append("Önceki konuşma bağlamı:\n")
                        for msg in request.conversation_history[-4:]:  # Last 4 messages for context
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "user":
                                context_parts.append(f"Öğrenci: {content}\n")
                            elif role == "assistant":
                                context_parts.append(f"Asistan: {content}\n")
                        context_parts.append("\n")
                    
                    # Enhanced prompt - direct answer only
                    context_parts.append(f"User: Bağlam Metni:\n{context_text}\n\n")
                    context_parts.append(f"Soru: {request.query}\n\n")
                    context_parts.append("Cevap (içsel analizden sonra sadece nihai cevabı yaz):")
                    full_prompt = "".join(context_parts)
                    
                    # Generate answer
                    # Lower temperature for more accurate, deterministic answers that stick to context
                    gen_request = {
                        "prompt": full_prompt,
                        "model": request.model or "llama-3.1-8b-instant",
                        "temperature": 0.3,  # Reduced from 0.7 to 0.3 for more accurate, context-faithful answers
                        "max_tokens": request.max_tokens or 1024  # Use client-provided max_tokens (512=short, 1024=normal, 2048=detailed)
                    }
                    
                    try:
                        gen_response = requests.post(
                            f"{MODEL_INFERENCER_URL}/models/generate",
                            json=gen_request,
                            timeout=180  # 3 minutes for Ollama models (CPU can be slow)
                        )
                        
                        if gen_response.status_code == 200:
                            gen_result = gen_response.json()
                            answer = gen_result.get("response", "").strip()
                            
                            return RAGQueryResponse(
                                answer=answer,
                                sources=context_docs,
                                chain_type=chain_type
                            )
                        else:
                            logger.error(f"Generation failed: {gen_response.status_code}")
                            return RAGQueryResponse(
                                answer="Üzgünüm, cevap oluşturulurken bir hata oluştu.",
                                sources=context_docs,
                                chain_type=chain_type
                            )
                    except requests.RequestException as req_error:
                        logger.error(f"Model inference service request failed: {str(req_error)}")
                        return RAGQueryResponse(
                            answer="Üzgünüm, cevap oluşturma servisi şu anda kullanılamıyor.",
                            sources=context_docs,
                            chain_type=chain_type
                        )
                else:
                    return RAGQueryResponse(
                        answer="Bu oturum için ilgili bilgi bulunamadı.",
                        sources=[],
                        chain_type=chain_type
                    )
                    
            except Exception as chromadb_error:
                logger.error(f"ChromaDB query error: {str(chromadb_error)}")
                return RAGQueryResponse(
                    answer="Bu oturum için dökümanlar bulunamadı. Lütfen önce dökümanları yükleyiniz.",
                    sources=[],
                    chain_type=chain_type
                )
            
                
        except Exception as e:
            logger.error(f"RAG query error: {str(e)}")
            return RAGQueryResponse(
                answer="Üzgünüm, sorunuzu işlerken bir hata oluştu. Lütfen tekrar deneyiniz.",
                sources=[],
                chain_type=request.chain_type or "stuff"
            )
            
    except Exception as e:
        logger.error(f"RAG query endpoint error: {str(e)}")
        return RAGQueryResponse(
            answer="Üzgünüm, sistemde beklenmeyen bir hata oluştu. Lütfen tekrar deneyiniz.",
            sources=[],
            chain_type=request.chain_type or "stuff"
        )

class RetrieveRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = 5
    embedding_model: Optional[str] = None

class RetrieveResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total: int

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """
    Retrieve documents without generation - for testing RAG retrieval quality.
    Returns only the retrieved documents with their scores.
    """
    try:
        logger.info(f"Retrieve request for collection: {request.collection_name}, query: {request.query[:50]}...")
        
        # Get ChromaDB client
        client = get_chroma_client()
        
        # Try to get collection - handle both session_ prefix and plain UUID formats
        collection = None
        collection_name = request.collection_name
        
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"✅ Found collection '{collection_name}'")
        except Exception as collection_error:
            # Try alternative collection name formats
            logger.warning(f"⚠️ Collection '{collection_name}' not found. Trying alternatives...")
            alternative_names = []
            
            # If collection_name has session_ prefix, try without it
            if collection_name.startswith("session_"):
                session_part = collection_name.replace("session_", "")
                alternative_names.append(session_part)
                # Try UUID format
                if len(session_part) == 32:
                    uuid_format = f"{session_part[:8]}-{session_part[8:12]}-{session_part[12:16]}-{session_part[16:20]}-{session_part[20:]}"
                    alternative_names.append(uuid_format)
            else:
                # Try with session_ prefix
                alternative_names.append(f"session_{collection_name}")
            
            # Try each alternative
            for alt_name in alternative_names:
                try:
                    logger.info(f"🔍 Trying alternative collection name: '{alt_name}'")
                    collection = client.get_collection(name=alt_name)
                    logger.info(f"✅ Found collection with alternative name: '{alt_name}'")
                    collection_name = alt_name
                    break
                except:
                    continue
            
            if collection is None:
                # List all collections to help debug
                try:
                    all_collections = client.list_collections()
                    collection_names = [c.name for c in all_collections]
                    logger.error(f"❌ Collection not found. Available collections: {collection_names}")
                    return RetrieveResponse(success=False, results=[], total=0)
                except Exception as list_error:
                    logger.error(f"❌ Could not list collections: {list_error}")
                    return RetrieveResponse(success=False, results=[], total=0)
        
        # Get embeddings for the query
        embedding_model = request.embedding_model or "nomic-embed-text"
        logger.info(f"🔍 Getting embeddings for query using {embedding_model}")
        
        query_embeddings = processor.get_embeddings([request.query], embedding_model)
        
        if not query_embeddings or not query_embeddings[0]:
            logger.error("Failed to generate query embeddings")
            return RetrieveResponse(success=False, results=[], total=0)
        
        # Query the collection
        search_results = collection.query(
            query_embeddings=query_embeddings,
            n_results=request.top_k
        )
        
        # Extract and format results
        documents = search_results.get('documents', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]
        
        results = []
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else float('inf')
            
            # Convert distance to similarity score (1 - distance for cosine)
            similarity_score = max(0, 1 - (distance / 2))  # Normalize to 0-1 range
            
            results.append({
                "text": doc,
                "score": round(similarity_score, 4),
                "metadata": metadata,
                "distance": round(distance, 4)
            })
        
        logger.info(f"✅ Retrieved {len(results)} documents from collection '{collection_name}'")
        
        return RetrieveResponse(
            success=True,
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error in retrieve endpoint: {e}", exc_info=True)
        return RetrieveResponse(success=False, results=[], total=0)

@app.get("/sessions/{session_id}/chunks")
async def get_session_chunks(session_id: str):
    """
    Get chunks for a specific session from ChromaDB
    """
    try:
        logger.info(f"Getting chunks for session: {session_id}")
        
        # Convert session_id to collection name format
        if len(session_id) == 32 and session_id.replace('-', '').isalnum():
            # Convert 32-char hex string to proper UUID format
            collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
            logger.info(f"Converted session_id '{session_id}' -> collection_name '{collection_name}'")
        elif len(session_id) == 36:  # Already formatted UUID
            collection_name = session_id
            logger.info(f"Session ID already in UUID format: '{collection_name}'")
        else:
            collection_name = session_id
            logger.warning(f"Using session_id as-is for collection name: '{collection_name}'")
        
        try:
            # Get ChromaDB client and collection
            client = get_chroma_client()
            collection = client.get_collection(name=collection_name)
            
            # Get all documents from the collection
            results = collection.get()
            
            # Format chunks for frontend
            chunks = []
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            ids = results.get('ids', [])
            
            for i, document in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
                
                # Robust metadata parsing - check multiple possible keys for source information
                source_files = ["Unknown"]
                source_value = None
                
                # Check multiple possible keys that could contain source file information
                for key in ["source_files", "source_file", "filename", "document_name", "file_name"]:
                    if metadata.get(key):
                        source_value = metadata.get(key)
                        logger.debug(f"🔍 METADATA FIX: Found source info in key '{key}': {source_value}")
                        break
                
                if source_value:
                    try:
                        # Try to parse as JSON string first (for source_files array)
                        parsed_value = json.loads(source_value)
                        if isinstance(parsed_value, list):
                            source_files = [str(item) for item in parsed_value if item]
                        else:
                            source_files = [str(parsed_value)]
                        logger.debug(f"🔍 METADATA FIX: Parsed JSON source_files: {source_files}")
                    except (json.JSONDecodeError, TypeError):
                        # If it's not JSON, treat as string
                        source_files = [str(source_value)]
                        logger.debug(f"🔍 METADATA FIX: Using string source_files: {source_files}")
                else:
                    logger.warning(f"🔍 METADATA FIX: No source file information found in metadata keys: {list(metadata.keys())}")
                
                document_name = source_files[0] if source_files and source_files[0] != "Unknown" else "Unknown"
                logger.debug(f"🔍 METADATA FIX: Final document_name: '{document_name}'")
                
                chunks.append({
                    "document_name": document_name,
                    "chunk_index": i + 1,
                    "chunk_text": document,
                    "chunk_metadata": metadata,
                    "chunk_id": chunk_id
                })
            
            logger.info(f"Retrieved {len(chunks)} chunks for session {session_id}")
            
            return {
                "chunks": chunks,
                "total_count": len(chunks),
                "session_id": session_id
            }
            
        except Exception as chromadb_error:
            logger.error(f"ChromaDB error for session {session_id}: {str(chromadb_error)}")
            return {
                "chunks": [],
                "total_count": 0,
                "session_id": session_id
            }
            
    except Exception as e:
        logger.error(f"Error getting chunks for session {session_id}: {str(e)}")
        return {
            "chunks": [],
            "total_count": 0,
            "session_id": session_id
        }

@app.post("/sessions/{session_id}/reprocess")
async def reprocess_session_documents(
    session_id: str,
    request: dict = Body(...)
):
    """
    Re-process existing documents in a session with a new embedding model.
    This will:
    1. Get all existing chunks from ChromaDB
    2. Group them by source file
    3. Re-embed with new embedding model
    4. Delete old chunks and add new ones
    """
    try:
        embedding_model = request.get("embedding_model", "nomic-embed-text")
        source_files = request.get("source_files", None)  # Optional: filter by specific files
        chunk_size = request.get("chunk_size", 1000)
        chunk_overlap = request.get("chunk_overlap", 200)
        
        logger.info(f"Re-processing session {session_id} with embedding model: {embedding_model}")
        
        # Convert session_id to collection name format
        # Try multiple collection name formats (same logic as query endpoint)
        if len(session_id) == 32 and session_id.replace('-', '').isalnum():
            collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
        elif len(session_id) == 36:
            collection_name = session_id
        else:
            collection_name = session_id
        
        # Get ChromaDB client
        client = get_chroma_client()
        
        # Try to get collection - if it doesn't exist, check alternative formats
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"✅ Found collection '{collection_name}' for reprocessing")
        except Exception as collection_error:
            # Try alternative collection name formats
            logger.warning(f"⚠️ Collection '{collection_name}' not found. Trying alternatives...")
            alternative_names = []
            
            # IMPORTANT: Try session_ prefix FIRST (this is how chunks are stored)
            if len(session_id) == 32:
                alternative_names.insert(0, f"session_{session_id}")
            
            # If collection_name is UUID format, try with session_ prefix (without dashes)
            if '-' in collection_name and len(collection_name) == 36:
                uuid_part = collection_name.replace('-', '')
                if f"session_{uuid_part}" not in alternative_names:
                    alternative_names.append(f"session_{uuid_part}")
            
            # Try each alternative
            collection = None
            for alt_name in alternative_names:
                try:
                    logger.info(f"🔍 Trying alternative collection name: '{alt_name}'")
                    collection = client.get_collection(name=alt_name)
                    logger.info(f"✅ Found collection with alternative name: '{alt_name}'")
                    collection_name = alt_name  # Update collection_name for consistency
                    break
                except:
                    continue
            
            if collection is None:
                # List all collections to help debug
                try:
                    all_collections = client.list_collections()
                    collection_names = [c.name for c in all_collections]
                    logger.error(f"❌ Collection not found. Available collections: {collection_names}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Collection '{collection_name}' not found. Available: {collection_names}"
                    )
                except Exception as list_error:
                    logger.error(f"❌ Could not list collections: {list_error}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Collection '{collection_name}' not found and could not list alternatives: {collection_error}"
                    )
        
        # Get all existing chunks
        results = collection.get()
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        
        if len(documents) == 0:
            return {
                "success": False,
                "message": "No documents found to re-process",
                "chunks_processed": 0
            }
        
        # Group chunks by source file
        file_chunks = {}
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
            
            # Get source file name
            source_file = None
            for key in ["source_file", "filename", "document_name", "file_name"]:
                if metadata.get(key):
                    source_file = metadata.get(key)
                    break
            
            if not source_file:
                source_file = "unknown"
            
            # Filter by source_files if specified
            if source_files and source_file not in source_files:
                continue
            
            if source_file not in file_chunks:
                file_chunks[source_file] = []
            
            file_chunks[source_file].append({
                "text": doc,
                "metadata": metadata,
                "id": chunk_id
            })
        
        logger.info(f"Found {len(file_chunks)} unique source files to re-process")
        
        # Process each file
        total_chunks_processed = 0
        successful_files = []
        failed_files = []
        
        for source_file, chunks in file_chunks.items():
            try:
                logger.info(f"Re-processing {len(chunks)} chunks from file: {source_file}")
                
                # Try to get original file from API gateway first
                combined_text = None
                api_gateway_url = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
                try:
                    response = requests.get(
                        f"{api_gateway_url}/documents/markdown/{source_file}",
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        combined_text = data.get("content", "")
                        if combined_text and combined_text.strip():
                            logger.info(f"Retrieved original file content for {source_file} from API gateway")
                except Exception as e:
                    logger.warning(f"Could not retrieve original file {source_file} from API gateway: {e}. Using combined chunks.")
                
                # Fallback: combine chunks back into original text
                if not combined_text:
                    combined_text = "\n\n".join([chunk["text"] for chunk in chunks])
                
                # Re-split into chunks
                new_chunks = processor.split_text(combined_text, chunk_size, chunk_overlap)
                
                if not new_chunks:
                    logger.warning(f"No chunks generated for {source_file}")
                    continue
                
                # Get new embeddings with new model
                # IMPORTANT: If HuggingFace model is selected, ALL chunks must use the same model
                # to ensure consistent similarity scores. No fallback to Ollama!
                logger.info(f"Getting embeddings for {len(new_chunks)} chunks using model: {embedding_model}")
                
                # Check if this is a HuggingFace model
                is_hf_model = "/" in embedding_model and not embedding_model.startswith("openai/")
                
                # Retry mechanism for rate limiting (HuggingFace API)
                max_retries = 3 if is_hf_model else 1
                retry_delay = 5  # seconds
                new_embeddings = None
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        new_embeddings = processor.get_embeddings(new_chunks, embedding_model)
                        if len(new_embeddings) == len(new_chunks):
                            break  # Success
                        else:
                            raise Exception(f"Embedding count mismatch: {len(new_chunks)} chunks but {len(new_embeddings)} embeddings")
                    except Exception as emb_error:
                        last_error = emb_error
                        if is_hf_model and attempt < max_retries - 1:
                            logger.warning(f"Embedding attempt {attempt + 1}/{max_retries} failed (possibly rate limiting): {emb_error}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            # If HuggingFace model was selected, fail completely (no fallback to Ollama)
                            if is_hf_model:
                                raise Exception(
                                    f"Failed to generate embeddings with HuggingFace model '{embedding_model}' after {max_retries} attempts. "
                                    f"ALL embeddings must use the same model to ensure consistent similarity scores. "
                                    f"Error: {str(emb_error)}. Please retry later or choose a different model."
                                )
                            else:
                                raise emb_error
                
                if new_embeddings is None or len(new_embeddings) != len(new_chunks):
                    raise Exception(f"Embedding count mismatch: {len(new_chunks)} chunks but {len(new_embeddings)} embeddings")
                
                # Generate new chunk IDs
                new_chunk_ids = [str(uuid.uuid4()) for _ in new_chunks]
                
                # Prepare metadata
                new_metadatas = []
                for i, chunk in enumerate(new_chunks):
                    chunk_metadata = {
                        "session_id": collection_name,
                        "source_file": source_file,
                        "filename": source_file,
                        "embedding_model": embedding_model,
                        "chunk_index": i + 1,
                        "total_chunks": len(new_chunks),
                        "chunk_length": len(chunk),
                        "reprocessed": True,
                        "reprocessed_at": datetime.now().isoformat()
                    }
                    new_metadatas.append(chunk_metadata)
                
                # Delete old chunks for this file
                old_ids_to_delete = [chunk["id"] for chunk in chunks]
                if old_ids_to_delete:
                    collection.delete(ids=old_ids_to_delete)
                    logger.info(f"Deleted {len(old_ids_to_delete)} old chunks for {source_file}")
                
                # Add new chunks
                collection.add(
                    documents=new_chunks,
                    embeddings=new_embeddings,
                    metadatas=new_metadatas,
                    ids=new_chunk_ids
                )
                
                total_chunks_processed += len(new_chunks)
                successful_files.append(source_file)
                logger.info(f"✅ Successfully re-processed {source_file}: {len(old_ids_to_delete)} old chunks -> {len(new_chunks)} new chunks")
                
            except Exception as e:
                logger.error(f"Error re-processing {source_file}: {str(e)}")
                failed_files.append(f"{source_file}: {str(e)}")
        
        return {
            "success": len(failed_files) == 0,
            "message": f"Re-processed {len(successful_files)} files, {len(failed_files)} failed",
            "chunks_processed": total_chunks_processed,
            "successful_files": successful_files,
            "failed_files": failed_files if failed_files else None,
            "embedding_model": embedding_model
        }
        
    except Exception as e:
        logger.error(f"Error in reprocess_session_documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to re-process documents: {str(e)}"
        )

@app.delete("/sessions/{session_id}/collection")
async def delete_session_collection(session_id: str):
    """
    Delete ChromaDB collection for a specific session
    This is called when a session is deleted to clean up all associated vectors
    """
    collection_name = ""
    try:
        logger.info(f"Deleting ChromaDB collection for session: {session_id}")
        
        # Convert session_id to collection name format
        if len(session_id) == 32 and session_id.replace('-', '').isalnum():
            collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
            logger.info(f"Converted session_id '{session_id}' -> collection_name '{collection_name}'")
        elif len(session_id) == 36:
            collection_name = session_id
            logger.info(f"Session ID already in UUID format: '{collection_name}'")
        else:
            collection_name = session_id
            logger.warning(f"Using session_id as-is for collection name: '{collection_name}'")
        
        client = get_chroma_client()
        
        try:
            client.get_collection(name=collection_name)
            client.delete_collection(name=collection_name)
            logger.info(f"✅ Successfully deleted ChromaDB collection: '{collection_name}'")
            return {
                "success": True,
                "message": f"ChromaDB collection '{collection_name}' deleted successfully",
                "session_id": session_id,
                "collection_name": collection_name
            }
        except Exception as get_error:
            if "does not exist" in str(get_error) or "not found" in str(get_error).lower():
                logger.info(f"Collection '{collection_name}' does not exist - already cleaned up")
                return {
                    "success": True,
                    "message": f"ChromaDB collection '{collection_name}' was already deleted or did not exist",
                    "session_id": session_id,
                    "collection_name": collection_name
                }
            else:
                raise get_error

    except Exception as e:
        logger.error(f"Error during collection deletion for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during collection deletion for session {session_id}: {str(e)}"
        )
    

@app.post("/crag-evaluate", response_model=CRAGEvaluationResponse)
async def evaluate_with_crag(request: CRAGEvaluationRequest):
    """
    CRAG Evaluation endpoint for RAG Testing.
    
    Evaluates retrieved documents using CRAG (Corrective RAG) methodology
    without heavy ML dependencies.
    """
    try:
        logger.info(f"CRAG Evaluation request for query: {request.query[:50]}...")
        
        # Initialize CRAG evaluator
        crag_evaluator = CRAGEvaluator(model_inference_url=MODEL_INFERENCER_URL)
        
        # Perform CRAG evaluation
        evaluation_result = crag_evaluator.evaluate_retrieved_docs(
            query=request.query,
            retrieved_docs=request.retrieved_docs
        )
        
        logger.info(f"✅ CRAG Evaluation completed: action={evaluation_result['action']}, confidence={evaluation_result['confidence']}")
        
        return CRAGEvaluationResponse(
            success=True,
            evaluation=evaluation_result,
            filtered_docs=evaluation_result.get("filtered_docs", request.retrieved_docs)
        )
        
    except Exception as e:
        logger.error(f"CRAG Evaluation error: {e}", exc_info=True)
        # Return fallback evaluation
        avg_score = sum(doc.get("score", 0) for doc in request.retrieved_docs) / len(request.retrieved_docs) if request.retrieved_docs else 0
        
        fallback_evaluation = {
            "action": "reject" if avg_score < 0.3 else "accept",
            "confidence": avg_score,
            "avg_score": avg_score,
            "error": str(e),
            "method": "fallback"
        }
        
        return CRAGEvaluationResponse(
            success=False,
            evaluation=fallback_evaluation,
            filtered_docs=request.retrieved_docs if avg_score >= 0.3 else []
        )

    
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)