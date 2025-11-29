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

# Import UNIFIED chunking system with LLM post-processing support
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from src.text_processing.text_chunker import chunk_text
    from src.text_processing.lightweight_chunker import create_semantic_chunks
    UNIFIED_CHUNKING_AVAILABLE = True
    logging.getLogger(__name__).info("✅ UNIFIED chunking system imported successfully with Turkish support, zero ML dependencies, and LLM post-processing")
except ImportError as e:
    UNIFIED_CHUNKING_AVAILABLE = False
    logging.getLogger(__name__).warning(f"⚠️ CRITICAL: Unified chunking system not available: {e}")

# Import langdetect for language detection
from langdetect import detect, LangDetectException

# Import BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
    logging.getLogger(__name__).info("✅ BM25 for hybrid search available")
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None
    logging.getLogger(__name__).warning("⚠️ BM25 not available - install rank-bm25")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Turkish stopwords for better keyword search
TURKISH_STOPWORDS = {
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü',
    'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için',
    'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mi', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede',
    'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani'
}

def tokenize_turkish(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize Turkish text for BM25 search
    - Lowercase conversion
    - Remove punctuation
    - Optional stopword removal
    - Keep numbers and special characters (for product codes, dates, etc.)
    """
    # Lowercase
    text = text.lower()
    
    # Split by whitespace and basic punctuation (but keep numbers intact)
    tokens = re.findall(r'\b[\w\d]+\b', text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [t for t in tokens if t not in TURKISH_STOPWORDS and len(t) > 1]
    
    return tokens

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
    chunk_strategy: Optional[str] = "lightweight"  # NEW: Enable lightweight Turkish chunking by default
    use_llm_post_processing: Optional[bool] = False  # NEW: Optional LLM post-processing for chunk refinement
    llm_model_name: Optional[str] = "llama-3.1-8b-instant"  # NEW: LLM model for post-processing
    model_inference_url: Optional[str] = None  # NEW: Override model inference URL for LLM post-processing

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
    use_hybrid_search: Optional[bool] = True  # Enable hybrid search (semantic + BM25)
    bm25_weight: Optional[float] = 0.3  # Weight for BM25 score (0.3 = 30% keyword, 70% semantic)

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    chain_type: Optional[str] = None
    correction: Optional[Dict[str, Any]] = None  # NEW: For self-correction details

# Environment variables - Google Cloud Run compatible
# For Docker: use service names (e.g., http://model-inference-service:8003)
# For Cloud Run: use full URLs (e.g., https://model-inference-xxx.run.app)
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", os.getenv("MODEL_INFERENCE_URL", None))
if not MODEL_INFERENCER_URL:
    MODEL_INFERENCE_HOST = os.getenv("MODEL_INFERENCE_HOST", "model-inference-service")
    MODEL_INFERENCE_PORT = os.getenv("MODEL_INFERENCE_PORT", "8002")
    if MODEL_INFERENCE_HOST.startswith("http://") or MODEL_INFERENCE_HOST.startswith("https://"):
        MODEL_INFERENCER_URL = MODEL_INFERENCE_HOST
    else:
        MODEL_INFERENCER_URL = f"http://{MODEL_INFERENCE_HOST}:{MODEL_INFERENCE_PORT}"

CHROMADB_URL = os.getenv("CHROMADB_URL", None)
if not CHROMADB_URL:
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "chromadb-service")
    CHROMADB_PORT = os.getenv("CHROMADB_PORT", "8000")
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
        # Configure ChromaDB settings with longer timeouts for bulk operations
        chroma_settings = Settings(
            anonymized_telemetry=False,
            chroma_client_auth_provider=None,
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            # Increase timeouts for bulk update operations (default is 60s)
            # These help with SSL handshake timeouts during high load
        )
        
        if use_https:
            # For Cloud Run, try to use the full URL
            # ChromaDB HttpClient may need special configuration for HTTPS
            logger.warning("⚠️ HTTPS detected for ChromaDB. Ensure ChromaDB service supports HTTPS or use HTTP proxy.")
            # Try to connect - ChromaDB HttpClient may handle HTTPS URLs
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=chroma_settings
            )
        else:
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=chroma_settings
            )
        
        logger.info(f"✅ ChromaDB client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create ChromaDB client: {e}")
        raise

# REMOVED: Internal DocumentProcessor class - now using unified external chunking system only

def get_embeddings_direct(texts: List[str], embedding_model: str = "text-embedding-v4") -> List[List[float]]:
    """
    Direct embedding function for use with unified chunking system.
    Get embeddings from local model inference service (Ollama)
    """
    try:
        embed_url = f"{MODEL_INFERENCER_URL}/embed"
        
        logger.info(f"Getting embeddings for {len(texts)} texts using model: {embedding_model}")
        
        # Send all texts in a single request for efficiency
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
        
    except Exception as e:
        logger.warning(f"Embedding service error with {embedding_model}: {str(e)}")
        raise  # Re-raise to allow fallback mechanism

def extract_chunk_title_from_content(content: str, fallback_title: str) -> str:
    """
    Extract meaningful title from chunk content for unified system compatibility
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

class CRAGEvaluator:
    """
    Corrective RAG (CRAG) Evaluator - REAL IMPLEMENTATION
    
    Uses a cross-encoder model via the model-inference-service to get
    actual relevance scores for query-document pairs.
    """
    
    def __init__(self, model_inference_url: str):
        self.model_inference_url = model_inference_url
        self.rerank_url = f"{self.model_inference_url}/rerank"
        self.correct_threshold = 3.0    # Stricter: Only truly relevant docs (ms-marco scores 0-10)
        self.incorrect_threshold = 1.0  # Filter out low-relevance docs
        self.filter_threshold = 2.0     # Individual document filter threshold (raised from 0.1)
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

# REMOVED: Global processor instance - now using unified external chunking system only

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
        
        # CRITICAL UPGRADE: Use unified chunking system with LLM post-processing support
        # Get embedding model from metadata to adjust chunk sizes
        embedding_model = request.metadata.get("embedding_model", "text-embedding-v4")
        
        # Adjust chunk sizes for Alibaba embedding models (they handle larger chunks better)
        is_alibaba_embedding = (
            embedding_model and (
                embedding_model.startswith("text-embedding-") or
                "alibaba" in embedding_model.lower() or
                "dashscope" in embedding_model.lower()
            )
        )
        
        if is_alibaba_embedding:
            # Alibaba embeddings (text-embedding-v4, etc.) can handle larger chunks
            # Increase chunk size significantly for better context retention
            default_chunk_size = 2500  # Increased from 1000 to 2500
            default_chunk_overlap = 500  # Increased from 200 to 500
            logger.info(f"🔵 Alibaba embedding detected ({embedding_model}): Using larger chunk sizes (size={default_chunk_size}, overlap={default_chunk_overlap})")
        else:
            # Default sizes for Ollama/local models
            default_chunk_size = 1000
            default_chunk_overlap = 200
            logger.info(f"⚪ Standard embedding model ({embedding_model}): Using standard chunk sizes (size={default_chunk_size}, overlap={default_chunk_overlap})")
        
        chunk_size = request.chunk_size or default_chunk_size
        chunk_overlap = request.chunk_overlap or default_chunk_overlap
        chunk_strategy = request.chunk_strategy or "lightweight"
        use_llm_post_processing = request.use_llm_post_processing or False
        llm_model_name = request.llm_model_name or "llama-3.1-8b-instant"
        model_inference_url = request.model_inference_url or MODEL_INFERENCER_URL
        
        if UNIFIED_CHUNKING_AVAILABLE:
            # Use UNIFIED chunking system with Turkish support, zero ML dependencies, and optional LLM post-processing
            logger.info(f"🚀 USING UNIFIED CHUNKING SYSTEM: strategy='{chunk_strategy}', size={chunk_size}, overlap={chunk_overlap}, llm_post_processing={use_llm_post_processing}")
            try:
                chunks = chunk_text(
                    text=request.text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    strategy=chunk_strategy,
                    use_llm_post_processing=use_llm_post_processing,
                    llm_model_name=llm_model_name,
                    model_inference_url=model_inference_url
                )
                
                if use_llm_post_processing:
                    logger.info(f"✅ Unified chunking with LLM post-processing successful: {len(chunks)} chunks created with enhanced Turkish support and semantic refinement")
                else:
                    logger.info(f"✅ Unified chunking successful: {len(chunks)} chunks created with Turkish support and zero ML dependencies")
            except Exception as e:
                logger.error(f"❌ CRITICAL: Unified chunking failed: {e}")
                logger.info("⚠️ No fallback available - unified chunking system is required")
                raise HTTPException(
                    status_code=500,
                    detail=f"Critical chunking system failure: {str(e)}"
                )
        else:
            # No fallback - unified chunking is required
            logger.error("❌ CRITICAL: Unified chunking system not available and no fallback exists")
            raise HTTPException(
                status_code=500,
                detail="Critical system error: Unified chunking system not available"
            )
        
        if not chunks:
            logger.warning("Text could not be split into any chunks.")
            raise HTTPException(status_code=400, detail="Text could not be split into chunks")
        
        logger.info(f"Successfully split text into {len(chunks)} chunks.")

        # Get embeddings - check for embedding model preference in metadata
        # (embedding_model already extracted above for chunk size adjustment)
        logger.info(f"Using embedding model: {embedding_model}")
        embeddings = get_embeddings_direct(chunks, embedding_model)
        
        if len(embeddings) != len(chunks):
            logger.error(f"Mismatch between chunk count ({len(chunks)}) and embedding count ({len(embeddings)}).")
            raise HTTPException(
                status_code=500,
                detail="Embedding count doesn't match chunk count"
            )
        
        # Generate chunk IDs
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        
        # CRITICAL FIX: Use SAME collection for all files in a session (no timestamp per file!)
        collection_name = request.collection_name or "documents"
        logger.info(f"🔍 DIAGNOSTIC: Initial collection_name: '{collection_name}'")
        
        # If collection name starts with "session_", convert to UUID format WITHOUT timestamp
        # This ensures all files in the same session go to the SAME collection
        if collection_name.startswith("session_"):
            session_id = collection_name[8:]  # Remove "session_" prefix
            logger.info(f"🔍 DIAGNOSTIC: Extracted session_id: '{session_id}' (length: {len(session_id)})")
            
            # Convert 32-char hex string to proper UUID format (8-4-4-4-12)
            # NO TIMESTAMP - all files in same session use same collection
            if len(session_id) == 32 and session_id.replace('-', '').isalnum():
                original_collection_name = collection_name
                base_uuid = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
                collection_name = base_uuid  # NO TIMESTAMP - use same collection for all files
                logger.info(f"🔍 FIXED NAMING: Transformed '{original_collection_name}' -> '{collection_name}' (NO TIMESTAMP)")
                logger.info(f"✅ Using UUID collection name (same for all files in session): {collection_name}")
            elif len(session_id) == 36:  # Already formatted UUID
                collection_name = session_id  # NO TIMESTAMP - use same collection for all files
                logger.info(f"🔍 FIXED NAMING: Session ID already in UUID format, using as-is (NO TIMESTAMP): '{collection_name}'")
                logger.info(f"✅ Using UUID collection name (same for all files in session): {collection_name}")
            else:
                logger.warning(f"🔍 DIAGNOSTIC: Unusual session_id format: '{session_id}' (length: {len(session_id)}, isalnum: {session_id.replace('-', '').isalnum()})")
                collection_name = session_id  # Use as-is, no timestamp
                logger.info(f"✅ Using session_id as collection name (NO TIMESTAMP): {collection_name}")
        else:
            # For non-session collections, keep as-is (no timestamp)
            logger.info(f"🔍 Using collection name as-is (non-session): '{collection_name}'")
        
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
                
                # CRITICAL: Store embedding model in metadata to detect dimension mismatches
                chunk_metadata["embedding_model"] = embedding_model
                
                # IMPORTANT: Add session_id to metadata for extra security/validation
                # This ensures even if collection names are somehow mixed, we can filter by session_id
                chunk_metadata["session_id"] = collection_name  # collection_name is the session_id
                
                # Add first few words as chunk preview for identification
                chunk_preview = chunk.strip()[:100].replace('\n', ' ').replace('\r', '')
                if len(chunk_preview) == 100:
                    chunk_preview += "..."
                chunk_metadata["chunk_preview"] = chunk_preview
                
                # Extract chunk title from content (if chunk starts with #)
                chunk_title = extract_chunk_title_from_content(chunk, f"Bölüm {i + 1}")
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
            
            # CRITICAL: Check if collection already has embeddings with different model/dimension
            try:
                existing_samples = collection.get(limit=1, include=["metadatas", "embeddings"])
                if existing_samples and existing_samples.get('metadatas') and len(existing_samples['metadatas']) > 0:
                    existing_model = existing_samples['metadatas'][0].get('embedding_model')
                    if existing_model and existing_model != embedding_model:
                        # Check embedding dimensions
                        existing_embeddings = existing_samples.get('embeddings', [])
                        if existing_embeddings and len(existing_embeddings) > 0:
                            existing_dim = len(existing_embeddings[0])
                            new_dim = len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
                            if existing_dim != new_dim:
                                error_msg = (
                                    f"❌ EMBEDDING DIMENSION MISMATCH: "
                                    f"Collection '{collection_name}' already contains embeddings from model '{existing_model}' "
                                    f"with dimension {existing_dim}, but you're trying to add embeddings from model '{embedding_model}' "
                                    f"with dimension {new_dim}. "
                                    f"ChromaDB requires all embeddings in a collection to have the same dimension. "
                                    f"Please use the same embedding model for all documents in this session, or create a new session."
                                )
                                logger.error(error_msg)
                                raise HTTPException(status_code=400, detail=error_msg)
                            else:
                                logger.warning(
                                    f"⚠️ Different embedding model detected but same dimension: "
                                    f"existing='{existing_model}' vs new='{embedding_model}' (both {existing_dim}D). "
                                    f"This may cause retrieval quality issues. Consider using the same model."
                                )
            except HTTPException:
                raise
            except Exception as e:
                # If check fails, log warning but continue (might be empty collection)
                logger.warning(f"⚠️ Could not verify embedding model compatibility: {e}")
            
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
            logger.debug(f"🔍 DIAGNOSTIC: RAG Query - Initial session_id: '{session_id}' (length: {len(session_id)})")
            
            if len(session_id) == 32 and session_id.replace('-', '').isalnum():
                # Convert 32-char hex string to proper UUID format
                collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
                logger.debug(f"🔍 DIAGNOSTIC: RAG Query - Transformed session_id '{session_id}' -> collection_name '{collection_name}'")
                logger.info(f"Using pure UUID as collection name for query: {collection_name}")
            elif len(session_id) == 36:  # Already formatted UUID
                collection_name = session_id
                logger.debug(f"🔍 DIAGNOSTIC: RAG Query - Session ID already in UUID format: '{collection_name}'")
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
                
                # Try to get collection - if it doesn't exist, check alternative formats including timestamped
                try:
                    collection = client.get_collection(name=collection_name)
                    logger.info(f"✅ Found collection '{collection_name}'")
                except Exception as collection_error:
                    # Try alternative collection name formats including timestamped ones
                    logger.warning(f"⚠️ Collection '{collection_name}' not found. Trying alternatives...")
                    
                    # CRITICAL: First list ALL collections to find timestamped versions
                    alternative_names = []
                    all_collection_names = []
                    
                    try:
                        logger.debug(f"🔍 RAG QUERY TIMESTAMPED SEARCH: Listing all collections...")
                        all_collections = client.list_collections()
                        all_collection_names = [c.name for c in all_collections]
                        logger.debug(f"🔍 RAG QUERY TIMESTAMPED SEARCH: Available collections ({len(all_collection_names)}): {all_collection_names}")
                    except Exception as list_error:
                        logger.error(f"🔍 RAG QUERY TIMESTAMPED SEARCH FAILED: Could not list collections: {list_error}")
                    
                    # Build list of base names to search for (with and without timestamp)
                    search_patterns = [collection_name]  # Start with the exact collection_name
                    
                    # If collection_name is UUID format, also try with session_ prefix
                    if '-' in collection_name and len(collection_name) == 36:
                        uuid_part = collection_name.replace('-', '')
                        search_patterns.append(f"session_{uuid_part}")
                    
                    # If session_id is 32-char hex, try both formats
                    if len(session_id) == 32:
                        # session_ prefix format (how it's stored in process_and_store)
                        search_patterns.append(f"session_{session_id}")
                        # UUID format
                        uuid_format = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
                        search_patterns.append(uuid_format)
                    
                    logger.debug(f"🔍 RAG QUERY TIMESTAMPED SEARCH: Search patterns: {search_patterns}")
                    
                    # Now search for both exact matches AND timestamped versions
                    for pattern in search_patterns:
                        # First try exact match (non-timestamped)
                        if pattern in all_collection_names and pattern not in alternative_names:
                            alternative_names.append(pattern)
                            logger.debug(f"🔍 RAG QUERY TIMESTAMPED SEARCH: Found exact match: {pattern}")
                        
                        # Then try timestamped versions (pattern_TIMESTAMP)
                        for coll_name in all_collection_names:
                            if coll_name.startswith(pattern + "_"):
                                # Check if suffix is a timestamp (all digits)
                                suffix = coll_name[len(pattern)+1:]
                                if suffix.isdigit() and coll_name not in alternative_names:
                                    alternative_names.append(coll_name)
                                    logger.debug(f"🔍 RAG QUERY TIMESTAMPED SEARCH: Found timestamped version: {coll_name} (pattern: {pattern})")
                    
                    logger.debug(f"🔍 RAG QUERY TIMESTAMPED SEARCH: Total alternatives found: {len(alternative_names)}")
                    logger.debug(f"🔍 RAG QUERY TIMESTAMPED SEARCH: Alternatives list: {alternative_names}")
                    
                    # Try each alternative (prefer timestamped versions - they're more recent)
                    # Sort by timestamp (newest first) if multiple timestamped versions exist
                    timestamped_alternatives = []
                    non_timestamped_alternatives = []
                    
                    for alt_name in alternative_names:
                        # Check if it ends with _TIMESTAMP
                        parts = alt_name.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            timestamped_alternatives.append((alt_name, int(parts[1])))
                        else:
                            non_timestamped_alternatives.append(alt_name)
                    
                    # Sort timestamped by timestamp (newest first)
                    timestamped_alternatives.sort(key=lambda x: x[1], reverse=True)
                    
                    # Try timestamped first (newest), then non-timestamped
                    sorted_alternatives = [name for name, _ in timestamped_alternatives] + non_timestamped_alternatives
                    
                    logger.debug(f"🔍 RAG QUERY TIMESTAMPED SEARCH: Trying alternatives in order: {sorted_alternatives}")
                    
                    # Try each alternative
                    collection = None
                    for alt_name in sorted_alternatives:
                        try:
                            logger.info(f"🔍 Trying alternative collection name: '{alt_name}'")
                            collection = client.get_collection(name=alt_name)
                            logger.info(f"✅ Found collection with alternative name: '{alt_name}'")
                            collection_name = alt_name  # Update collection_name for consistency
                            break
                        except Exception as alt_error:
                            logger.debug(f"Failed to get collection '{alt_name}': {alt_error}")
                            continue
                    
                    if collection is None:
                        logger.error(f"❌ Collection not found with any alternative names")
                        logger.error(f"❌ Tried alternatives: {sorted_alternatives}")
                        logger.error(f"❌ Available collections: {all_collection_names}")
                        raise Exception(f"Collection '{collection_name}' not found. Tried alternatives: {sorted_alternatives}")
                
                # CRITICAL: Check collection's embedding dimension first
                collection_dimension = None
                collection_embedding_model = None
                
                # Strategy 1: Try to get embedding_model from metadata first (FASTEST)
                try:
                    logger.info(f"🔍 Attempting to get metadata from collection '{collection_name}'...")
                    sample_meta = collection.get(limit=1, include=["metadatas"])
                    logger.info(f"🔍 Got sample_meta: {type(sample_meta)}, keys: {list(sample_meta.keys()) if isinstance(sample_meta, dict) else 'not a dict'}")
                    
                    # SAFE: Check if sample_meta exists without using truth value
                    if sample_meta is not None:
                        # Safely extract metadata
                        try:
                            meta_key = 'metadatas'
                            if meta_key in sample_meta:
                                metadatas_raw = sample_meta[meta_key]
                                logger.info(f"🔍 Found 'metadatas' key, type: {type(metadatas_raw)}")
                                
                                # Convert to list safely
                                import numpy as np
                                if isinstance(metadatas_raw, np.ndarray):
                                    metadatas_list = metadatas_raw.tolist()
                                    logger.info(f"🔍 Converted NumPy array to list, length: {len(metadatas_list)}")
                                elif isinstance(metadatas_raw, (list, tuple)):
                                    metadatas_list = list(metadatas_raw)
                                    logger.info(f"🔍 Converted list/tuple, length: {len(metadatas_list)}")
                                else:
                                    metadatas_list = []
                                    logger.warning(f"⚠️ Unexpected metadatas type: {type(metadatas_raw)}")
                                
                                if len(metadatas_list) > 0:
                                    logger.info(f"🔍 First metadata type: {type(metadatas_list[0])}, is dict: {isinstance(metadatas_list[0], dict)}")
                                    if isinstance(metadatas_list[0], dict):
                                        first_meta = metadatas_list[0]
                                        logger.info(f"🔍 First metadata keys: {list(first_meta.keys())}")
                                        collection_embedding_model = first_meta.get('embedding_model')
                                        if collection_embedding_model:
                                            logger.info(f"✅ Found embedding model in metadata: {collection_embedding_model}")
                                            
                                        # Map model name to dimension (FASTEST approach)
                                        model_lower = collection_embedding_model.lower()
                                        if 'text-embedding-v4' in model_lower:
                                            collection_dimension = 1024  # FIXED: v4 is 1024D, not 2048D
                                        elif 'nomic-embed' in model_lower:
                                            collection_dimension = 768  # Usually 768
                                        elif 'all-mpnet-base-v2' in model_lower:
                                            collection_dimension = 768
                                        elif 'all-minilm' in model_lower or 'bge-small' in model_lower:
                                            collection_dimension = 384
                                            
                                            if collection_dimension:
                                                logger.info(f"📏 Collection dimension (from model): {collection_dimension}D")
                                        else:
                                            logger.warning(f"⚠️ No 'embedding_model' key in metadata")
                                    else:
                                        logger.warning(f"⚠️ First metadata is not a dict: {type(metadatas_list[0])}")
                                else:
                                    logger.warning(f"⚠️ metadatas_list is empty")
                            else:
                                logger.warning(f"⚠️ 'metadatas' key not found in sample_meta. Available keys: {list(sample_meta.keys()) if isinstance(sample_meta, dict) else 'not a dict'}")
                        except Exception as meta_err:
                            logger.error(f"❌ Error extracting metadata: {meta_err}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.warning(f"⚠️ sample_meta is None")
                except Exception as meta_check_err:
                    logger.error(f"❌ Could not get metadata: {meta_check_err}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Strategy 2: If dimension still unknown, try to get from embeddings directly
                if not collection_dimension:
                    try:
                        sample_emb = collection.get(limit=1, include=["embeddings"])
                        # SAFE: Check if sample_emb exists without using truth value
                        if sample_emb is not None:
                            try:
                                emb_key = 'embeddings'
                                if emb_key in sample_emb:
                                    embeddings_raw = sample_emb[emb_key]
                                    
                                    # Convert to list safely
                                    import numpy as np
                                    if isinstance(embeddings_raw, np.ndarray):
                                        embeddings_list = embeddings_raw.tolist()
                                    elif isinstance(embeddings_raw, (list, tuple)):
                                        embeddings_list = list(embeddings_raw)
                                    else:
                                        embeddings_list = []
                                    
                                    if len(embeddings_list) > 0:
                                        first_emb = embeddings_list[0]
                                        # Convert to list if needed
                                        if isinstance(first_emb, np.ndarray):
                                            first_emb = first_emb.tolist()
                                        elif not isinstance(first_emb, (list, tuple)):
                                            first_emb = list(first_emb) if hasattr(first_emb, '__iter__') and not isinstance(first_emb, (str, bytes)) else []
                                        
                                        if isinstance(first_emb, (list, tuple)) and len(first_emb) > 0:
                                            collection_dimension = len(first_emb)
                                            logger.info(f"📏 Collection dimension (from embedding): {collection_dimension}D")
                            except Exception as emb_err:
                                logger.warning(f"⚠️ Error extracting embedding dimension: {emb_err}")
                    except Exception as emb_check_err:
                        logger.warning(f"⚠️ Could not get embeddings: {emb_check_err}")
                
                if not collection_dimension:
                    logger.warning("⚠️ Could not determine collection dimension. Will try multiple models.")
                
                # Get embeddings for the query using our model inference service
                # CRITICAL: If we know collection dimension, ONLY use models with matching dimension
                preferred_model = collection_embedding_model or request.embedding_model or os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
                logger.info(f"🔍 Collection dimension: {collection_dimension}, preferred model: {preferred_model}")
                
                # Define models by dimension
                models_by_dimension = {
                    1024: ["text-embedding-v4"],  # Alibaba 1024D only v4
                    768: ["nomic-embed-text", "sentence-transformers/all-mpnet-base-v2"],  # 768D
                    384: ["nomic-embed-text", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],  # 384D
                }
                
                # Build embedding models list based on collection dimension
                embedding_models_to_try = []
                
                if collection_dimension:
                    # CRITICAL: Only use models with matching dimension
                    logger.info(f"⚠️ Collection requires {collection_dimension}D embeddings. Filtering models by dimension...")
                    
                    # First, try preferred model if it matches dimension
                    # Check if preferred model is in the correct dimension list
                    matching_models = models_by_dimension.get(collection_dimension, [])
                    if preferred_model in matching_models:
                        embedding_models_to_try.append(preferred_model)
                    
                    # Add all models with matching dimension
                    embedding_models_to_try.extend([m for m in matching_models if m != preferred_model])
                    
                    # If requested model is different and matches dimension, add it
                    if request.embedding_model and request.embedding_model != preferred_model:
                        if request.embedding_model in matching_models:
                            embedding_models_to_try.insert(1, request.embedding_model)  # Insert after preferred
                    
                    # Remove duplicates while preserving order
                    embedding_models_to_try = list(dict.fromkeys(embedding_models_to_try))
                    
                    if not embedding_models_to_try:
                        raise Exception(
                            f"❌ No embedding models available for {collection_dimension}D dimension. "
                            f"Collection was created with model: {collection_embedding_model or 'unknown'}. "
                            f"Please use a model that produces {collection_dimension}D embeddings."
                        )
                    
                    logger.info(f"✅ Will try {len(embedding_models_to_try)} models with {collection_dimension}D: {', '.join(embedding_models_to_try)}")
                else:
                    # Unknown dimension, try preferred model first, then common models
                    logger.warning("⚠️ Collection dimension unknown. Trying preferred model and common fallbacks...")
                    embedding_models_to_try = [preferred_model]
                    if request.embedding_model and request.embedding_model != preferred_model:
                        embedding_models_to_try.append(request.embedding_model)
                    embedding_models_to_try.extend([
                        "text-embedding-v3",  # Try Alibaba 1024D
                        "text-embedding-v2",  # Try Alibaba 1024D
                        "nomic-embed-text",
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "BAAI/bge-small-en-v1.5"
                    ])
                    embedding_models_to_try = list(dict.fromkeys([m for m in embedding_models_to_try if m]))
                
                query_embeddings = None
                successful_model = None
                query_dimension = None
                
                for model_to_try in embedding_models_to_try:
                    try:
                        logger.info(f"🔄 Trying embedding model: {model_to_try}")
                        query_embeddings = get_embeddings_direct([request.query], model_to_try)
                        if query_embeddings and len(query_embeddings) > 0 and len(query_embeddings[0]) > 0:
                            query_dimension = len(query_embeddings[0])
                            
                            # Check dimension match if we know collection dimension
                            if collection_dimension and query_dimension != collection_dimension:
                                logger.warning(
                                    f"⚠️ Dimension mismatch: Collection expects {collection_dimension}D, "
                                    f"but {model_to_try} produces {query_dimension}D. Trying next model..."
                                )
                                query_embeddings = None
                                continue
                            
                            successful_model = model_to_try
                            logger.info(f"✅ Successfully got {len(query_embeddings)} query embeddings using {model_to_try} (dimension: {query_dimension})")
                            break
                        else:
                            logger.warning(f"⚠️ Empty embeddings from {model_to_try}")
                    except Exception as emb_error:
                        logger.warning(f"⚠️ Failed to get embeddings with {model_to_try}: {emb_error}")
                        continue
                
                if not query_embeddings or not query_embeddings[0]:
                    error_msg = f"Failed to generate query embeddings with any model. Tried: {', '.join(embedding_models_to_try)}"
                    if collection_dimension:
                        error_msg += f" Collection requires {collection_dimension}D embeddings."
                    raise Exception(error_msg)
                
                # Final dimension check
                if collection_dimension and query_dimension != collection_dimension:
                    raise Exception(
                        f"❌ EMBEDDING DIMENSION MISMATCH: Collection '{collection_name}' requires {collection_dimension}D embeddings, "
                        f"but query embedding has {query_dimension}D. Please use the same embedding model that was used to create the collection. "
                        f"Collection was created with model: {collection_embedding_model or 'unknown'}"
                    )
                
                # Check if reranker will be used (via CRAG evaluator)
                # CRAG evaluator always uses reranker, so we need to check session settings
                use_reranker = True  # Default: CRAG evaluator uses reranker
                try:
                    # Try to get session rag_settings to check if reranker is explicitly enabled
                    session_response = requests.get(
                        f"{os.getenv('API_GATEWAY_URL', 'http://api-gateway:8001')}/sessions/{request.session_id}",
                        timeout=5
                    )
                    if session_response.status_code == 200:
                        session_data = session_response.json()
                        rag_settings = session_data.get('rag_settings', {})
                        # If use_reranker_service is explicitly False, don't use reranker
                        if rag_settings.get('use_reranker_service') is False:
                            use_reranker = False
                            logger.info("Reranker disabled in session settings")
                        else:
                            logger.info("Reranker will be used (CRAG evaluator)")
                    else:
                        logger.info(f"Could not fetch session settings (status: {session_response.status_code}), assuming reranker will be used")
                except Exception as e:
                    logger.warning(f"Could not check session settings for reranker: {e}, assuming reranker will be used")
                
                # Query the collection using embeddings (not query_texts)
                # Determine how many documents to fetch initially
                # Reranker needs more candidates to select from (5x more or at least 25-50)
                # Hybrid search also needs more for reranking (3x)
                if use_reranker:
                    # Reranker will be used: fetch 5x more documents (or at least 25-50)
                    n_results_fetch = max(request.top_k * 5, 25)
                    n_results_fetch = min(n_results_fetch, 50)  # Cap at 50 for performance
                    logger.info(f"🔄 Reranker enabled: fetching {n_results_fetch} documents (top_k={request.top_k})")
                elif request.use_hybrid_search and BM25_AVAILABLE:
                    # Hybrid search reranking: fetch 3x more
                    n_results_fetch = request.top_k * 3
                    logger.info(f"🔄 Hybrid search enabled: fetching {n_results_fetch} documents (top_k={request.top_k})")
                else:
                    # No reranking: just fetch what we need
                    n_results_fetch = request.top_k
                    logger.info(f"🔄 No reranking: fetching {n_results_fetch} documents (top_k={request.top_k})")
                
                search_results = collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results_fetch
                )
                
                # Extract documents from ChromaDB response
                documents = search_results.get('documents', [[]])[0]
                metadatas = search_results.get('metadatas', [[]])[0]
                distances = search_results.get('distances', [[]])[0]
                
                total_found = len(documents)
                logger.info(f"🔍 Semantic search: {total_found} documents found in collection '{collection_name}'")
                
                # 🔥 HYBRID SEARCH: Combine Semantic + BM25 Keyword Search
                if request.use_hybrid_search and BM25_AVAILABLE and len(documents) > 0:
                    try:
                        logger.info("🔥 Applying HYBRID SEARCH: Semantic + BM25")
                        
                        # Tokenize query for BM25
                        query_tokens = tokenize_turkish(request.query, remove_stopwords=True)
                        logger.info(f"🔍 Query tokens (stopwords removed): {query_tokens}")
                        
                        # Tokenize all documents for BM25
                        tokenized_docs = [tokenize_turkish(doc, remove_stopwords=True) for doc in documents]
                        
                        # Calculate BM25 scores
                        bm25 = BM25Okapi(tokenized_docs)
                        bm25_scores = bm25.get_scores(query_tokens)
                        
                        # Normalize BM25 scores to 0-1 range
                        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                        normalized_bm25_scores = [score / max_bm25 for score in bm25_scores]
                        
                        # Calculate semantic similarity scores from distances
                        semantic_scores = []
                        for distance in distances:
                            if distance == float('inf'):
                                semantic_scores.append(0.0)
                            else:
                                semantic_scores.append(max(0.0, 1.0 - distance))
                        
                        # 2025 UPDATE: Reciprocal Rank Fusion (RRF) Logic implemented below
                        # Previous weighted average logic removed for clarity

                        
                        # Sort by hybrid score (descending) and take top_k
                        # 2025 UPDATE: Switched to Reciprocal Rank Fusion (RRF)
                        # RRF is more robust than weighted average for combining scores with different distributions
                        # Formula: score = 1 / (k + rank)
                        
                        # 1. Get Ranks for Semantic Scores
                        # semantic_scores is already aligned with documents list
                        # Create (index, score) pairs
                        semantic_pairs = [(i, s) for i, s in enumerate(semantic_scores)]
                        # Sort by score descending (higher is better)
                        semantic_pairs.sort(key=lambda x: x[1], reverse=True)
                        # Create rank map: index -> rank (0-based)
                        semantic_ranks = {index: rank for rank, (index, _) in enumerate(semantic_pairs)}
                        
                        # 2. Get Ranks for BM25 Scores
                        # bm25_scores is aligned with documents list
                        bm25_pairs = [(i, s) for i, s in enumerate(bm25_scores)]
                        bm25_pairs.sort(key=lambda x: x[1], reverse=True)
                        bm25_ranks = {index: rank for rank, (index, _) in enumerate(bm25_pairs)}
                        
                        # 3. Calculate RRF Scores
                        rrf_k = 60  # Standard constant for RRF
                        hybrid_scores = []
                        
                        for i in range(len(documents)):
                            sem_rank = semantic_ranks.get(i, len(documents))
                            kw_rank = bm25_ranks.get(i, len(documents))
                            
                            # RRF Score calculation
                            # Weighted RRF: We can still honor the weights by multiplying the RRF components
                            # semantic_weight affects the semantic component
                            # bm25_weight affects the keyword component
                            
                            rrf_sem = (1.0 / (rrf_k + sem_rank))
                            rrf_bm25 = (1.0 / (rrf_k + kw_rank))
                            
                            # Apply weights (normalized)
                            # Default request.bm25_weight is usually 0.3
                            w_sem = 1.0 - request.bm25_weight
                            w_bm25 = request.bm25_weight
                            
                            final_score = (w_sem * rrf_sem) + (w_bm25 * rrf_bm25)
                            
                            hybrid_scores.append({
                                'index': i,
                                'hybrid_score': final_score,
                                'semantic_score': semantic_scores[i],
                                'bm25_score': bm25_scores[i] if i < len(bm25_scores) else 0.0,
                                'semantic_rank': sem_rank,
                                'bm25_rank': kw_rank
                            })
                        
                        # Sort by hybrid score (descending) and take top_k
                        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
                        top_k_indices = [item['index'] for item in hybrid_scores[:request.top_k]]
                        
                        # Reorder documents, metadatas, distances based on hybrid ranking
                        documents = [documents[i] for i in top_k_indices]
                        metadatas = [metadatas[i] for i in top_k_indices]
                        distances = [distances[i] for i in top_k_indices]
                        
                        # Update scores in hybrid_scores for logging
                        hybrid_scores = hybrid_scores[:request.top_k]
                        
                        logger.info(f"✅ HYBRID SEARCH: Reranked to top {request.top_k} documents")
                        logger.info(f"📊 Top 3 hybrid scores: {[(s['hybrid_score'], s['semantic_score'], s['bm25_score']) for s in hybrid_scores[:3]]}")
                        
                    except Exception as hybrid_error:
                        logger.warning(f"⚠️ Hybrid search failed, falling back to semantic only: {hybrid_error}")
                        # Continue with semantic-only results
                else:
                    if not BM25_AVAILABLE:
                        logger.info("ℹ️ BM25 not available - using semantic search only")
                    hybrid_scores = None
                
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
                # Get reranker_type from session rag_settings if available
                # Note: We already checked session settings above for use_reranker flag
                reranker_type = "alibaba"  # DEFAULT: Use Alibaba reranker
                if use_reranker:
                    try:
                        # Try to get session rag_settings from API Gateway for reranker_type
                        session_response = requests.get(
                            f"{os.getenv('API_GATEWAY_URL', 'http://api-gateway:8001')}/sessions/{request.session_id}",
                            timeout=5
                        )
                        if session_response.status_code == 200:
                            session_data = session_response.json()
                            rag_settings = session_data.get('rag_settings', {})
                            if rag_settings.get('reranker_type'):
                                reranker_type = rag_settings.get('reranker_type')
                                logger.info(f"Using reranker_type from session rag_settings: {reranker_type}")
                            else:
                                logger.info(f"No reranker_type in session rag_settings, using default: {reranker_type}")
                        else:
                            logger.info(f"Could not fetch session rag_settings (status: {session_response.status_code}), using default: {reranker_type}")
                    except Exception as e:
                        logger.warning(f"Could not fetch session rag_settings for reranker_type: {e}, using default: {reranker_type}")
                else:
                    logger.info("Reranker disabled, skipping CRAG evaluation")
                
                # CRAG Evaluation (only if reranker is enabled)
                if use_reranker:
                    # Import new CRAGEvaluator from services
                    from services.crag_evaluator import CRAGEvaluator as NewCRAGEvaluator
                    crag_evaluator = NewCRAGEvaluator(model_inference_url=MODEL_INFERENCER_URL, reranker_type=reranker_type)
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
                else:
                    # No reranker: use all retrieved documents (already limited to top_k)
                    logger.info("✅ No reranker: using all retrieved documents")
                
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
                    
                    # ---------------------------------------------------------
                    # 2025 PHASE 2: SELF-CORRECTION & VERIFICATION LOOP
                    # ---------------------------------------------------------
                    # Step 1: Generate initial answer
                    # Step 2: Verify consistency with context
                    # Step 3: If inconsistent, generate corrected answer
                    
                    correction_details = None
                    
                    # --- PHASE 1: INITIAL GENERATION ---
                    
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
                            initial_answer = gen_result.get("response", "").strip()
                            final_answer = initial_answer
                            
                            # --- PHASE 2: SELF-VERIFICATION ---
                            # Only verify if we have a substantial answer and context
                            if len(initial_answer) > 50 and len(context_text) > 100:
                                logger.info("🔍 PHASE 2: Starting Self-Verification...")
                                
                                verification_prompt = (
                                    "Sen titiz bir doğrulama uzmanısın. Görevin, verilen CEVAP ile BAĞLAM arasındaki tutarlılığı VE mantıksal doğruluğu kontrol etmektir.\n\n"
                                    f"BAĞLAM:\n{context_text[:2000]}...\n\n" # Truncate for verification to save tokens
                                    f"SORU: {request.query}\n"
                                    f"ÜRETİLEN CEVAP: {initial_answer}\n\n"
                                    "GÖREV:\n"
                                    "1. Cevaptaki bilgiler bağlamla örtüşüyor mu? (ÖNCELİKLİ)\n"
                                    "2. Sayısal veriler (tarih, yüzde, miktar) bağlamdakiyle birebir aynı mı?\n"
                                    "3. [KENDİ BİLGİNLE KONTROL]: Cevap mantıklı ve genel doğrularla tutarlı mı? Bağlam yanlış anlaşılmış olabilir mi?\n"
                                    "   - Eğer bağlam yetersizse veya yanlış anlaşılmışsa ve cevap mantıksızsa, kendi genel bilginle durumu analiz et.\n"
                                    "   - ANCAK: Bağlamda net bir veri varsa (örn: 'Ali'nin yaşı 5'), genel bilgine uymasa bile bağlamı kabul et.\n\n"
                                    "KRİTİK: Eğer is_consistent=false ise, MUTLAKA corrected_answer üret!\n\n"
                                    "ÇIKTI FORMATI (JSON):\n"
                                    "{\n"
                                    '  "is_consistent": true/false,\n'
                                    '  "issues": ["Hata 1: Bağlamda X var ama cevap Y demiş", "Mantık Hatası: Bu sonuç bu veriden çıkmaz"],\n'
                                    '  "corrected_answer": "EĞER is_consistent=false: MUTLAKA düzeltilmiş cevap yaz. Bağlamı temel al, mantık hatalarını kendi bilginle düzelt. Eğer bağlamda cevap yoksa: \'Bu bilgi ders dökümanlarında bulunamamıştır.\' yaz. | EĞER is_consistent=true: boş string (\"\") yaz"\n'
                                    "}\n\n"
                                    "ÖNEMLI: corrected_answer asla null veya eksik olmasın. Eğer tutarsızlık varsa MUTLAKA düzeltilmiş cevap ver.\n"
                                    "Sadece JSON çıktısı ver, başka hiçbir şey yazma."
                                )
                                
                                verify_request = {
                                    "prompt": verification_prompt,
                                    "model": request.model or "llama-3.1-8b-instant",
                                    "temperature": 0.1, # Very low temp for strict logic
                                    "max_tokens": 1024,
                                    "json_mode": True # Force JSON output if supported by model
                                }
                                
                                try:
                                    verify_response = requests.post(
                                        f"{MODEL_INFERENCER_URL}/models/generate",
                                        json=verify_request,
                                        timeout=60
                                    )
                                    
                                    if verify_response.status_code == 200:
                                        verify_result = verify_response.json()
                                        verify_text = verify_result.get("response", "").strip()
                                        
                                        # Parse JSON response
                                        import json
                                        try:
                                            # Find JSON part if wrapped in markdown
                                            json_start = verify_text.find('{')
                                            json_end = verify_text.rfind('}') + 1
                                            if json_start >= 0 and json_end > json_start:
                                                verify_json = json.loads(verify_text[json_start:json_end])
                                                
                                                if not verify_json.get("is_consistent", True):
                                                    logger.warning(f"⚠️ SELF-CORRECTION: Inconsistency found: {verify_json.get('issues')}")
                                                    
                                                    corrected_answer = verify_json.get("corrected_answer", "").strip()
                                                    
                                                    # Accept any non-empty corrected answer (more tolerant threshold)
                                                    if corrected_answer and len(corrected_answer) > 10 and corrected_answer.lower() not in ["null", "none", "yok"]:
                                                        final_answer = corrected_answer
                                                        correction_details = {
                                                            "original_answer": initial_answer,
                                                            "issues": verify_json.get("issues", []),
                                                            "was_corrected": True
                                                        }
                                                        logger.info(f"✅ SELF-CORRECTION: Answer updated with verified content. New answer length: {len(corrected_answer)}")
                                                    else:
                                                        logger.warning(f"⚠️ SELF-CORRECTION: Issues found but no valid corrected answer provided. Got: '{corrected_answer}'")
                                                        
                                                        # FALLBACK: Generate a corrected answer ourselves
                                                        # If the answer is inconsistent with context, provide a safe fallback
                                                        fallback_answer = "Üzgünüm, verdiğim ilk cevap ders materyalleriyle tutarlı değildi. Bu sorunun cevabı sağlanan ders dökümanlarında net olarak bulunamamaktadır. Lütfen soruyu daha spesifik hale getirin veya farklı bir kaynak kullanın."
                                                        
                                                        final_answer = fallback_answer
                                                        correction_details = {
                                                            "original_answer": initial_answer,
                                                            "issues": verify_json.get("issues", []),
                                                            "was_corrected": True
                                                        }
                                                        logger.info("✅ SELF-CORRECTION: Used fallback corrected answer due to inconsistency.")
                                                else:
                                                    logger.info("✅ SELF-CORRECTION: Verification passed. Answer is consistent.")
                                                    # Explicitly state that verification passed
                                                    correction_details = {
                                                        "original_answer": initial_answer,
                                                        "issues": [],
                                                        "was_corrected": False
                                                    }
                                            else:
                                                logger.warning("⚠️ SELF-CORRECTION: Could not parse JSON from verification response.")
                                        except Exception as json_err:
                                            logger.warning(f"⚠️ SELF-CORRECTION: JSON parse error: {json_err}")
                                            
                                except Exception as verify_err:
                                    logger.warning(f"⚠️ SELF-CORRECTION: Verification request failed: {verify_err}")

                            return RAGQueryResponse(
                                answer=final_answer,
                                sources=context_docs,
                                chain_type=chain_type,
                                correction=correction_details
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
            # Try alternative collection name formats including timestamped ones
            logger.warning(f"⚠️ Collection '{collection_name}' not found. Trying alternatives...")
            
            # CRITICAL: First list ALL collections to find timestamped versions
            alternative_names = []
            all_collection_names = []
            
            try:
                logger.debug(f"🔍 RETRIEVE TIMESTAMPED SEARCH: Listing all collections...")
                all_collections = client.list_collections()
                all_collection_names = [c.name for c in all_collections]
                logger.debug(f"🔍 RETRIEVE TIMESTAMPED SEARCH: Available collections ({len(all_collection_names)}): {all_collection_names}")
            except Exception as list_error:
                logger.error(f"🔍 RETRIEVE TIMESTAMPED SEARCH FAILED: Could not list collections: {list_error}")
            
            # Build list of base names to search for (with and without timestamp)
            search_patterns = [collection_name]  # Start with the exact collection_name
            
            # If collection_name has session_ prefix, try without it
            if collection_name.startswith("session_"):
                session_part = collection_name.replace("session_", "")
                search_patterns.append(session_part)
                # Try UUID format
                if len(session_part) == 32:
                    uuid_format = f"{session_part[:8]}-{session_part[8:12]}-{session_part[12:16]}-{session_part[16:20]}-{session_part[20:]}"
                    search_patterns.append(uuid_format)
            else:
                # Try with session_ prefix
                search_patterns.append(f"session_{collection_name}")
                # If it's a 32-char hex, try UUID format too
                if len(collection_name) == 32 and collection_name.replace('-', '').isalnum():
                    uuid_format = f"{collection_name[:8]}-{collection_name[8:12]}-{collection_name[12:16]}-{collection_name[16:20]}-{collection_name[20:]}"
                    search_patterns.append(uuid_format)
            
            logger.debug(f"🔍 RETRIEVE TIMESTAMPED SEARCH: Search patterns: {search_patterns}")
            
            # Now search for both exact matches AND timestamped versions
            for pattern in search_patterns:
                # First try exact match (non-timestamped)
                if pattern in all_collection_names and pattern not in alternative_names:
                    alternative_names.append(pattern)
                    logger.debug(f"🔍 RETRIEVE TIMESTAMPED SEARCH: Found exact match: {pattern}")
                
                # Then try timestamped versions (pattern_TIMESTAMP)
                for coll_name in all_collection_names:
                    if coll_name.startswith(pattern + "_"):
                        # Check if suffix is a timestamp (all digits)
                        suffix = coll_name[len(pattern)+1:]
                        if suffix.isdigit() and coll_name not in alternative_names:
                            alternative_names.append(coll_name)
                            logger.debug(f"🔍 RETRIEVE TIMESTAMPED SEARCH: Found timestamped version: {coll_name} (pattern: {pattern})")
            
            logger.debug(f"🔍 RETRIEVE TIMESTAMPED SEARCH: Total alternatives found: {len(alternative_names)}")
            
            # Try each alternative (prefer timestamped versions - they're more recent)
            # Sort by timestamp (newest first) if multiple timestamped versions exist
            timestamped_alternatives = []
            non_timestamped_alternatives = []
            
            for alt_name in alternative_names:
                # Check if it ends with _TIMESTAMP
                parts = alt_name.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    timestamped_alternatives.append((alt_name, int(parts[1])))
                else:
                    non_timestamped_alternatives.append(alt_name)
            
            # Sort timestamped by timestamp (newest first)
            timestamped_alternatives.sort(key=lambda x: x[1], reverse=True)
            
            # Try timestamped first (newest), then non-timestamped
            sorted_alternatives = [name for name, _ in timestamped_alternatives] + non_timestamped_alternatives
            
            logger.debug(f"🔍 RETRIEVE TIMESTAMPED SEARCH: Trying alternatives in order: {sorted_alternatives}")
            
            # Try each alternative
            for alt_name in sorted_alternatives:
                try:
                    logger.info(f"🔍 Trying alternative collection name: '{alt_name}'")
                    collection = client.get_collection(name=alt_name)
                    logger.info(f"✅ Found collection with alternative name: '{alt_name}'")
                    collection_name = alt_name
                    break
                except Exception as alt_error:
                    logger.debug(f"Failed to get collection '{alt_name}': {alt_error}")
                    continue
            
            if collection is None:
                logger.error(f"❌ Collection not found with any alternative names")
                logger.error(f"❌ Tried alternatives: {sorted_alternatives}")
                logger.error(f"❌ Available collections: {all_collection_names}")
                return RetrieveResponse(success=False, results=[], total=0)
        
        # Get embeddings for the query
        embedding_model = request.embedding_model or "text-embedding-v4"
        logger.info(f"🔍 Getting embeddings for query using {embedding_model}")
        
        query_embeddings = get_embeddings_direct([request.query], embedding_model)
        
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
    
    # Get ChromaDB client and collection
    client = get_chroma_client()
    
    # Try to get collection - if it doesn't exist, check alternative formats including timestamped ones
    collection = None
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"✅ Found collection '{collection_name}'")
    except Exception as collection_error:
        # Try alternative collection name formats including timestamped ones
        logger.warning(f"⚠️ Collection '{collection_name}' not found. Original error: {collection_error}")
        logger.warning(f"⚠️ Starting alternative collection name search...")
        
        # CRITICAL: First list ALL collections to find timestamped versions
        alternative_names = []
        all_collection_names = []
        
        try:
            logger.debug(f"🔍 TIMESTAMPED SEARCH: Listing all collections...")
            all_collections = client.list_collections()
            all_collection_names = [c.name for c in all_collections]
            logger.debug(f"🔍 TIMESTAMPED SEARCH: Available collections ({len(all_collection_names)}): {all_collection_names}")
        except Exception as list_error:
            logger.error(f"🔍 TIMESTAMPED SEARCH FAILED: Could not list collections: {list_error}")
        
        # Build list of base names to search for (with and without timestamp)
        search_patterns = [collection_name]  # Start with the exact collection_name
        
        # If collection_name is UUID format, also try with session_ prefix
        if '-' in collection_name and len(collection_name) == 36:
            uuid_part = collection_name.replace('-', '')
            search_patterns.append(f"session_{uuid_part}")
        
        # If session_id is 32-char hex, try both formats
        if len(session_id) == 32:
            # session_ prefix format (how it's stored in process_and_store)
            search_patterns.append(f"session_{session_id}")
            # UUID format
            uuid_format = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
            search_patterns.append(uuid_format)
        
        logger.debug(f"🔍 TIMESTAMPED SEARCH: Search patterns: {search_patterns}")
        
        # Now search for both exact matches AND timestamped versions
        for pattern in search_patterns:
            # First try exact match (non-timestamped)
            if pattern in all_collection_names and pattern not in alternative_names:
                alternative_names.append(pattern)
                logger.debug(f"🔍 TIMESTAMPED SEARCH: Found exact match: {pattern}")
            
            # Then try timestamped versions (pattern_TIMESTAMP)
            for coll_name in all_collection_names:
                if coll_name.startswith(pattern + "_"):
                    # Check if suffix is a timestamp (all digits)
                    suffix = coll_name[len(pattern)+1:]
                    if suffix.isdigit() and coll_name not in alternative_names:
                        alternative_names.append(coll_name)
                        logger.debug(f"🔍 TIMESTAMPED SEARCH: Found timestamped version: {coll_name} (pattern: {pattern})")
        
        logger.debug(f"🔍 TIMESTAMPED SEARCH: Total alternatives found: {len(alternative_names)}")
        logger.debug(f"🔍 TIMESTAMPED SEARCH: Alternatives list: {alternative_names}")
        
        # Try each alternative (prefer timestamped versions - they're more recent)
        # Sort by timestamp (newest first) if multiple timestamped versions exist
        timestamped_alternatives = []
        non_timestamped_alternatives = []
        
        for alt_name in alternative_names:
            # Check if it ends with _TIMESTAMP
            parts = alt_name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                timestamped_alternatives.append((alt_name, int(parts[1])))
            else:
                non_timestamped_alternatives.append(alt_name)
        
        # Sort timestamped by timestamp (newest first)
        timestamped_alternatives.sort(key=lambda x: x[1], reverse=True)
        
        # Try timestamped first (newest), then non-timestamped
        sorted_alternatives = [name for name, _ in timestamped_alternatives] + non_timestamped_alternatives
        
        logger.debug(f"🔍 TIMESTAMPED SEARCH: Trying alternatives in order: {sorted_alternatives}")
        
        for alt_name in sorted_alternatives:
            try:
                logger.info(f"🔍 Trying alternative collection name: '{alt_name}'")
                collection = client.get_collection(name=alt_name)
                logger.info(f"✅ Found collection with alternative name: '{alt_name}'")
                collection_name = alt_name  # Update collection_name for consistency
                break
            except Exception as alt_error:
                logger.debug(f"Failed to get collection '{alt_name}': {alt_error}")
                continue
        
        if collection is None:
            logger.error(f"❌ Collection not found with any alternative names. Original error: {collection_error}")
            logger.error(f"❌ Tried alternatives: {sorted_alternatives}")
            # Return empty result
            return {
                "chunks": [],
                "total_count": 0,
                "session_id": session_id
            }
    
    # If we reach here, we have a valid collection
    # CRITICAL FIX: Since all files in a session use the SAME collection (no timestamp),
    # we only need to get chunks from this single collection
    # However, we should also check for any timestamped versions that might exist from old code
    all_collections_for_session = []
    
    # First, add the found collection
    all_collections_for_session.append(collection)
    
    # Then, find all timestamped versions of this collection (for backward compatibility)
    # This ensures we get chunks from ALL files, even if some were processed with old code
    try:
        all_collections = client.list_collections()
        all_collection_names = [c.name for c in all_collections]
        
        # Build base pattern to search for
        base_patterns = [collection_name]
        if len(session_id) == 32:
            uuid_format = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
            base_patterns.append(uuid_format)
            base_patterns.append(f"session_{session_id}")
        
        # Find all collections that match this session (including timestamped versions from old code)
        for pattern in base_patterns:
            for coll_name in all_collection_names:
                if coll_name == pattern:
                    # Exact match - already added
                    continue
                elif coll_name.startswith(pattern + "_"):
                    # Timestamped version (from old code)
                    suffix = coll_name[len(pattern)+1:]
                    if suffix.isdigit():
                        try:
                            alt_collection = client.get_collection(name=coll_name)
                            if alt_collection not in all_collections_for_session:
                                all_collections_for_session.append(alt_collection)
                                logger.info(f"✅ Found additional timestamped collection (old code): {coll_name}")
                        except Exception as e:
                            logger.warning(f"Could not load collection {coll_name}: {e}")
    except Exception as e:
        logger.warning(f"Could not list all collections: {e}")
    
    # Get all documents from ALL collections
    # CRITICAL: Group chunks by document_name to ensure we get chunks from ALL files
    all_chunks = []
    chunks_by_document = {}  # Track chunks per document for logging
    
    for coll in all_collections_for_session:
        results = coll.get()
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
            
            # Track chunks per document for logging
            if document_name not in chunks_by_document:
                chunks_by_document[document_name] = 0
            chunks_by_document[document_name] += 1
            
            # CRITICAL FIX: Use chunk_index from metadata if available, otherwise use global index
            chunk_index_from_metadata = metadata.get("chunk_index")
            if chunk_index_from_metadata is not None:
                # Use metadata chunk_index (preserves original file-based indexing)
                chunk_index = int(chunk_index_from_metadata)
            else:
                # Fallback: use global index (all_chunks length + 1)
                chunk_index = len(all_chunks) + 1
            
            all_chunks.append({
                "document_name": document_name,
                "chunk_index": chunk_index,  # Use metadata chunk_index or global index
                "chunk_text": document,
                "chunk_metadata": metadata,
                "chunk_id": chunk_id
            })
    
    # Sort chunks by document_name and chunk_index for consistent display
    all_chunks.sort(key=lambda x: (x["document_name"], x["chunk_index"]))
    
    # Re-number chunks globally (1, 2, 3, ...) for display
    for i, chunk in enumerate(all_chunks):
        chunk["display_index"] = i + 1  # Global display index
        # Keep original chunk_index in metadata for reference
    
    # Log summary of chunks per document
    logger.info(f"📊 Chunks per document: {chunks_by_document}")
    logger.info(f"✅ Retrieved {len(all_chunks)} chunks from {len(all_collections_for_session)} collection(s) for session {session_id}")
    logger.info(f"📁 Found chunks from {len(chunks_by_document)} unique document(s)")
    
    return {
        "chunks": all_chunks,
        "total_count": len(all_chunks),
        "session_id": session_id
    }

@app.get("/sessions/{session_id}/chunks-with-embeddings")
async def get_session_chunks_with_embeddings(session_id: str):
    """
    Get chunks WITH EMBEDDINGS for a specific session from ChromaDB.
    Used for backup/restore operations.
    """
    logger.info(f"Getting chunks with embeddings for session: {session_id}")
    
    # Convert session_id to collection name format
    if len(session_id) == 32 and session_id.replace('-', '').isalnum():
        collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
    elif len(session_id) == 36:
        collection_name = session_id
    else:
        collection_name = f"session_{session_id}"
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as collection_error:
        logger.warning(f"Collection '{collection_name}' not found, trying alternatives...")
        # Try alternative collection names
        all_collections = client.list_collections()
        all_collection_names = [c.name for c in all_collections]
        
        alternative_names = []
        base_patterns = [collection_name, session_id, f"session_{session_id}"]
        
        for pattern in base_patterns:
            if pattern in all_collection_names:
                alternative_names.append(pattern)
            for coll_name in all_collection_names:
                if coll_name.startswith(pattern + "_") and coll_name not in alternative_names:
                    suffix = coll_name[len(pattern)+1:]
                    if suffix.isdigit():
                        alternative_names.append(coll_name)
        
        collection = None
        for alt_name in alternative_names:
            try:
                collection = client.get_collection(name=alt_name)
                collection_name = alt_name
                break
            except:
                continue
        
        if collection is None:
            logger.error(f"Collection not found for session {session_id}")
            return {
                "chunks": [],
                "total_count": 0,
                "session_id": session_id
            }
    
    # Get chunks WITH embeddings
    try:
        results = collection.get(include=['embeddings', 'documents', 'metadatas'])
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        embeddings = results.get('embeddings', [])
        
        chunks_with_embeddings = []
        for i, document in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
            embedding = embeddings[i] if i < len(embeddings) else None
            
            # Extract document name
            document_name = metadata.get("document_name") or metadata.get("source_file") or metadata.get("filename") or "Unknown"
            
            chunks_with_embeddings.append({
                "chunk_id": chunk_id,
                "chunk_text": document,
                "chunk_metadata": metadata,
                "document_name": document_name,
                "chunk_index": metadata.get("chunk_index", i + 1),
                "embedding": embedding  # Include embedding for restore
            })
        
        logger.info(f"✅ Retrieved {len(chunks_with_embeddings)} chunks with embeddings for session {session_id}")
        
        return {
            "chunks": chunks_with_embeddings,
            "total_count": len(chunks_with_embeddings),
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error getting chunks with embeddings: {e}", exc_info=True)
        return {
            "chunks": [],
            "total_count": 0,
            "session_id": session_id,
            "error": str(e)
        }

@app.post("/sessions/{session_id}/restore-chunks")
async def restore_session_chunks(session_id: str, request: dict = Body(...)):
    """
    Restore chunks WITH EMBEDDINGS to ChromaDB for a session.
    Used for backup/restore operations.
    
    Request body:
    {
        "chunks": [
            {
                "chunk_id": "...",
                "chunk_text": "...",
                "chunk_metadata": {...},
                "document_name": "...",
                "chunk_index": 1,
                "embedding": [0.1, 0.2, ...]  # Optional, will regenerate if missing
            }
        ],
        "original_session_id": "..."
    }
    """
    logger.info(f"Restoring chunks for session: {session_id}")
    
    chunks = request.get("chunks", [])
    original_session_id = request.get("original_session_id", session_id)
    
    if not chunks:
        return {
            "success": False,
            "error": "No chunks provided",
            "chunks_restored": 0
        }
    
    # Convert session_id to collection name format
    if len(session_id) == 32 and session_id.replace('-', '').isalnum():
        collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
    elif len(session_id) == 36:
        collection_name = session_id
    else:
        collection_name = f"session_{session_id}"
    
    try:
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            # Clear existing chunks if restoring to same session
            if session_id == original_session_id:
                logger.info(f"Clearing existing chunks in collection {collection_name}")
                existing = collection.get()
                if existing.get("ids"):
                    collection.delete(ids=existing["ids"])
        except:
            # Collection doesn't exist, create it
            logger.info(f"Creating new collection: {collection_name}")
            collection = client.create_collection(
                name=collection_name,
                metadata={"session_id": session_id, "restored_from": original_session_id}
            )
        
        # Prepare chunks for insertion
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        has_embeddings = False
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            chunk_text = chunk.get("chunk_text") or chunk.get("text") or ""
            chunk_metadata = chunk.get("chunk_metadata") or chunk.get("metadata") or {}
            embedding = chunk.get("embedding")
            
            # Ensure metadata has required fields
            chunk_metadata["session_id"] = session_id
            chunk_metadata["document_name"] = chunk.get("document_name") or chunk_metadata.get("document_name") or "Unknown"
            chunk_metadata["chunk_index"] = chunk.get("chunk_index") or chunk_metadata.get("chunk_index") or len(documents) + 1
            
            documents.append(chunk_text)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
            
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                embeddings.append(embedding)
                has_embeddings = True
            else:
                embeddings.append(None)  # Will be generated by ChromaDB
        
        # Add chunks to collection
        if has_embeddings and all(e is not None for e in embeddings):
            # Add with embeddings
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"✅ Added {len(chunks)} chunks with embeddings to collection {collection_name}")
        else:
            # Add without embeddings (ChromaDB will generate them)
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✅ Added {len(chunks)} chunks (embeddings will be generated) to collection {collection_name}")
        
        return {
            "success": True,
            "chunks_restored": len(chunks),
            "session_id": session_id,
            "collection_name": collection_name,
            "has_embeddings": has_embeddings
        }
        
    except Exception as e:
        logger.error(f"Error restoring chunks: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "chunks_restored": 0
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
        # CRITICAL FIX: Try to get embedding_model from session settings first, then from request
        embedding_model_from_request = request.get("embedding_model", None)
        source_files = request.get("source_files", None)  # Optional: filter by specific files
        chunk_size = request.get("chunk_size", 1000)
        chunk_overlap = request.get("chunk_overlap", 200)
        
        # Get session RAG settings from API Gateway to retrieve the correct embedding model
        embedding_model = None
        try:
            api_gateway_url = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
            session_response = requests.get(
                f"{api_gateway_url}/sessions/{session_id}",
                timeout=10
            )
            if session_response.status_code == 200:
                session_data = session_response.json()
                rag_settings = session_data.get("rag_settings", {})
                session_embedding_model = rag_settings.get("embedding_model")
                if session_embedding_model:
                    embedding_model = session_embedding_model
                    logger.info(f"✅ Retrieved embedding model from session settings: {embedding_model}")
        except Exception as e:
            logger.warning(f"⚠️ Could not retrieve session settings for embedding model: {e}")
        
        # Fallback to request parameter, then to default
        if not embedding_model:
            embedding_model = embedding_model_from_request or "text-embedding-v4"
            logger.info(f"Using embedding model from request/default: {embedding_model}")
        
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
        
        # Try to get collection - if it doesn't exist, check alternative formats including timestamped ones
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"✅ Found collection '{collection_name}' for reprocessing")
        except Exception as collection_error:
            # Try alternative collection name formats including timestamped ones
            logger.warning(f"⚠️ Collection '{collection_name}' not found for reprocessing. Original error: {collection_error}")
            logger.warning(f"⚠️ Starting alternative collection name search for reprocessing...")
            
            # CRITICAL: First list ALL collections to find timestamped versions
            alternative_names = []
            all_collection_names = []
            
            try:
                logger.info(f"🔍 REPROCESS TIMESTAMPED SEARCH: Listing all collections...")
                all_collections = client.list_collections()
                all_collection_names = [c.name for c in all_collections]
                logger.info(f"🔍 REPROCESS TIMESTAMPED SEARCH: Available collections ({len(all_collection_names)}): {all_collection_names}")
            except Exception as list_error:
                logger.error(f"🔍 REPROCESS TIMESTAMPED SEARCH FAILED: Could not list collections: {list_error}")
            
            # Build list of base names to search for (with and without timestamp)
            search_patterns = [collection_name]  # Start with the exact collection_name
            
            # If collection_name is UUID format, also try with session_ prefix
            if '-' in collection_name and len(collection_name) == 36:
                uuid_part = collection_name.replace('-', '')
                search_patterns.append(f"session_{uuid_part}")
            
            # If session_id is 32-char hex, try both formats
            if len(session_id) == 32:
                # session_ prefix format (how it's stored in process_and_store)
                search_patterns.append(f"session_{session_id}")
                # UUID format
                uuid_format = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
                search_patterns.append(uuid_format)
            
            logger.info(f"🔍 REPROCESS TIMESTAMPED SEARCH: Search patterns: {search_patterns}")
            
            # Now search for both exact matches AND timestamped versions
            for pattern in search_patterns:
                # First try exact match (non-timestamped)
                if pattern in all_collection_names and pattern not in alternative_names:
                    alternative_names.append(pattern)
                    logger.info(f"🔍 REPROCESS TIMESTAMPED SEARCH: Found exact match: {pattern}")
                
                # Then try timestamped versions (pattern_TIMESTAMP)
                for coll_name in all_collection_names:
                    if coll_name.startswith(pattern + "_"):
                        # Check if suffix is a timestamp (all digits)
                        suffix = coll_name[len(pattern)+1:]
                        if suffix.isdigit() and coll_name not in alternative_names:
                            alternative_names.append(coll_name)
                            logger.info(f"🔍 REPROCESS TIMESTAMPED SEARCH: Found timestamped version: {coll_name} (pattern: {pattern})")
            
            logger.info(f"🔍 REPROCESS TIMESTAMPED SEARCH: Total alternatives found: {len(alternative_names)}")
            logger.info(f"🔍 REPROCESS TIMESTAMPED SEARCH: Alternatives list: {alternative_names}")
            
            # Try each alternative (prefer timestamped versions - they're more recent)
            # Sort by timestamp (newest first) if multiple timestamped versions exist
            timestamped_alternatives = []
            non_timestamped_alternatives = []
            
            for alt_name in alternative_names:
                # Check if it ends with _TIMESTAMP
                parts = alt_name.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    timestamped_alternatives.append((alt_name, int(parts[1])))
                else:
                    non_timestamped_alternatives.append(alt_name)
            
            # Sort timestamped by timestamp (newest first)
            timestamped_alternatives.sort(key=lambda x: x[1], reverse=True)
            
            # Try timestamped first (newest), then non-timestamped
            sorted_alternatives = [name for name, _ in timestamped_alternatives] + non_timestamped_alternatives
            
            logger.info(f"🔍 REPROCESS TIMESTAMPED SEARCH: Trying alternatives in order: {sorted_alternatives}")
            
            # Try each alternative
            collection = None
            for alt_name in sorted_alternatives:
                try:
                    logger.info(f"🔍 Trying alternative collection name for reprocessing: '{alt_name}'")
                    collection = client.get_collection(name=alt_name)
                    logger.info(f"✅ Found collection with alternative name for reprocessing: '{alt_name}'")
                    collection_name = alt_name  # Update collection_name for consistency
                    break
                except Exception as alt_error:
                    logger.debug(f"Failed to get collection '{alt_name}': {alt_error}")
                    continue
            
            if collection is None:
                logger.error(f"❌ Collection not found with any alternative names for reprocessing. Original error: {collection_error}")
                logger.error(f"❌ Tried alternatives: {sorted_alternatives}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{collection_name}' not found. Tried alternatives: {sorted_alternatives}"
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
                
                # Check if we should re-chunk or just re-embed
                re_chunk = request.get("re_chunk", False)  # New parameter to force re-chunking
                
                if re_chunk:
                    # Re-chunk the file content
                    logger.info(f"🔄 REPROCESS: Re-chunking file {source_file} with new chunking parameters")
                    
                    # CRITICAL: Get original file content from session data directory
                    # Try to read the original file from data/markdown directory
                    import os
                    from pathlib import Path
                    
                    # Try multiple possible locations for the original file
                    possible_paths = [
                        Path(f"data/markdown/{source_file}"),
                        Path(f"data/markdown/{session_id}/{source_file}"),
                        Path(f"../data/markdown/{source_file}"),
                        Path(f"../data/markdown/{session_id}/{source_file}"),
                    ]
                    
                    original_text = None
                    for path in possible_paths:
                        if path.exists():
                            try:
                                with open(path, 'r', encoding='utf-8') as f:
                                    original_text = f.read()
                                logger.info(f"✅ Found original file at {path}")
                                break
                            except Exception as e:
                                logger.warning(f"Could not read {path}: {e}")
                                continue
                    
                    # Fallback: Reconstruct from chunks (not ideal but better than nothing)
                    if not original_text:
                        logger.warning(f"⚠️ Original file not found, reconstructing from chunks (may have issues)")
                        # Remove overlap by deduplicating
                        chunk_texts = [chunk["text"] for chunk in chunks]
                        # Try to remove obvious duplicates at boundaries
                        deduplicated = []
                        for i, text in enumerate(chunk_texts):
                            if i == 0:
                                deduplicated.append(text)
                            else:
                                # Check if start of current chunk is in previous chunk
                                prev_text = deduplicated[-1]
                                # Find where current chunk actually starts (not in previous)
                                # Simple heuristic: if first 100 chars of current are in previous, skip them
                                first_100 = text[:100]
                                if first_100 in prev_text:
                                    # Find the position after the duplicate
                                    pos = prev_text.find(first_100)
                                    if pos != -1:
                                        # Start from after the duplicate part
                                        overlap_len = len(first_100)
                                        new_start = overlap_len
                                        # But be smarter - find sentence boundary
                                        sentences = text.split('.')
                                        if len(sentences) > 1:
                                            # Skip first sentence if it's duplicate
                                            if sentences[0].strip() in prev_text:
                                                new_text = '.'.join(sentences[1:]).strip()
                                                if new_text:
                                                    deduplicated.append(new_text)
                                                    continue
                                deduplicated.append(text)
                        original_text = "\n\n".join(deduplicated)
                    
                    # Re-chunk with new parameters
                    from services.chunking_service import chunk_text_with_strategy
                    new_chunks = chunk_text_with_strategy(
                        text=original_text,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        strategy="lightweight",
                        use_llm_post_processing=False
                    )
                    logger.info(f"✅ REPROCESS: Re-chunked into {len(new_chunks)} new chunks")
                else:
                    # ✅ CRITICAL FIX: Do NOT re-chunk! Just keep existing chunks and re-embed them
                    # This preserves LLM improvements and metadata
                    logger.info(f"✅ REPROCESS: Keeping existing {len(chunks)} chunks (preserving LLM improvements and metadata)")
                    logger.info(f"✅ REPROCESS: Only re-embedding with new model: {embedding_model}")
                    
                    # Extract chunk texts (preserving order and content)
                    new_chunks = [chunk["text"] for chunk in chunks]
                
                if not new_chunks:
                    logger.warning(f"No chunks found for {source_file}")
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
                        new_embeddings = get_embeddings_direct(new_chunks, embedding_model)
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
                
                # ✅ CRITICAL FIX: Keep original chunk IDs and metadata (preserving LLM improvements!)
                logger.info(f"✅ REPROCESS: Preserving original chunk IDs and metadata (including LLM improvements)")
                
                # Use original chunk IDs
                new_chunk_ids = [chunk["id"] for chunk in chunks]
                
                # Preserve original metadata and only update embedding-related fields
                new_metadatas = []
                for i, chunk_data in enumerate(chunks):
                    # Start with original metadata (preserves llm_improved, llm_model, etc.)
                    chunk_metadata = chunk_data["metadata"].copy() if chunk_data.get("metadata") else {}
                    
                    # Only update embedding-related fields
                    chunk_metadata["embedding_model"] = embedding_model
                    chunk_metadata["reprocessed"] = True
                    chunk_metadata["reprocessed_at"] = datetime.now().isoformat()
                    
                    # Update chunk_length if needed
                    chunk_metadata["chunk_length"] = len(new_chunks[i])
                    
                    new_metadatas.append(chunk_metadata)
                    
                    # Log if this chunk was LLM improved
                    if chunk_metadata.get("llm_improved"):
                        logger.debug(f"✅ Preserved LLM improvement for chunk {i+1} (model: {chunk_metadata.get('llm_model', 'unknown')})")
                
                # Count how many LLM-improved chunks we're preserving
                llm_improved_count = sum(1 for m in new_metadatas if m.get("llm_improved"))
                logger.info(f"✅ REPROCESS: Preserving {llm_improved_count}/{len(new_metadatas)} LLM-improved chunks")
                
                # ✅ CRITICAL FIX: Use UPDATE instead of DELETE+ADD to safely preserve chunks
                # This prevents data loss if operation fails mid-way
                logger.info(f"✅ REPROCESS: Updating {len(new_chunk_ids)} chunks in-place (safer than delete+add)")
                
                try:
                    # Update all chunks in one go (atomic operation)
                    collection.update(
                        ids=new_chunk_ids,
                        documents=new_chunks,
                        embeddings=new_embeddings,
                        metadatas=new_metadatas
                    )
                    logger.info(f"✅ Successfully updated {len(new_chunk_ids)} chunks for {source_file}")
                except Exception as update_error:
                    # If update fails (e.g., IDs don't exist), fall back to delete+add
                    logger.warning(f"⚠️ Update failed for {source_file}, falling back to delete+add: {update_error}")
                    
                    old_ids_to_delete = [chunk["id"] for chunk in chunks]
                    if old_ids_to_delete:
                        collection.delete(ids=old_ids_to_delete)
                        logger.info(f"Deleted {len(old_ids_to_delete)} old chunks for {source_file}")
                    
                    collection.add(
                        documents=new_chunks,
                        embeddings=new_embeddings,
                        metadatas=new_metadatas,
                        ids=new_chunk_ids
                    )
                
                total_chunks_processed += len(new_chunks)
                successful_files.append(source_file)
                logger.info(f"✅ Successfully re-processed {source_file}: {len(new_chunks)} chunks updated with new embeddings")
                
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
    




class ImproveSingleChunkRequest(BaseModel):
    chunk_text: str
    language: Optional[str] = "tr"
    model_name: Optional[str] = "llama-3.1-8b-instant"
    session_id: Optional[str] = None  # NEW: For ChromaDB update
    chunk_id: Optional[str] = None  # NEW: For ChromaDB update (if known)
    document_name: Optional[str] = None  # NEW: Alternative to chunk_id
    chunk_index: Optional[int] = None  # NEW: Alternative to chunk_id

class ImproveSingleChunkResponse(BaseModel):
    success: bool
    original_text: str
    improved_text: Optional[str] = None
    message: Optional[str] = None
    processing_time_ms: Optional[float] = None

class ImproveAllChunksRequest(BaseModel):
    language: Optional[str] = "tr"
    model_name: Optional[str] = "llama-3.1-8b-instant"
    skip_already_improved: Optional[bool] = True

class ImproveAllChunksResponse(BaseModel):
    success: bool
    total_chunks: int
    processed: int
    improved: int
    failed: int
    skipped: int
    message: str
    processing_time_ms: float


@app.post("/chunks/improve-single", response_model=ImproveSingleChunkResponse)
async def improve_single_chunk(request: ImproveSingleChunkRequest):
    """
    Improve a single chunk using LLM post-processing.
    This is more efficient than batch processing for on-demand improvements.
    """
    start_time = time.time()
    
    try:
        logger.info(f"🤖 Improving single chunk ({len(request.chunk_text)} chars) with {request.model_name}")
        
        # Import the standard post-processor (single chunk, simpler)
        try:
            from src.text_processing.chunk_post_processor_grok import GrokChunkPostProcessor, PostProcessingConfig
        except ImportError:
            try:
                from src.text_processing.chunk_post_processor import ChunkPostProcessor as GrokChunkPostProcessor, PostProcessingConfig
            except ImportError:
                return ImproveSingleChunkResponse(
                    success=False,
                    original_text=request.chunk_text,
                    improved_text=None,
                    message="LLM post-processing not available",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
        
        # Create post-processor configuration
        config = PostProcessingConfig(
            enabled=True,
            model_name=request.model_name,
            model_inference_url=MODEL_INFERENCER_URL,
            language=request.language,
            timeout_seconds=30,
            retry_attempts=2
        )
        
        # Create and use post-processor
        post_processor = GrokChunkPostProcessor(config)
        
        # IMPORTANT: Temporarily disable "worth processing" check for manual improvements
        # User explicitly wants to improve this chunk, so we should always try
        original_check = post_processor._is_chunk_worth_processing
        post_processor._is_chunk_worth_processing = lambda x: True  # Always process
        
        try:
            # Process single chunk
            improved_chunks = post_processor.process_chunks([request.chunk_text])
        finally:
            # Restore original check
            post_processor._is_chunk_worth_processing = original_check
        
        if improved_chunks and len(improved_chunks) > 0:
            improved_text = improved_chunks[0]
            
            # Check if actually improved (not just echoed back)
            if improved_text != request.chunk_text and len(improved_text.strip()) > 10:
                processing_time = (time.time() - start_time) * 1000
                logger.info(f"✅ Chunk improved successfully in {processing_time:.0f}ms")
                
                # ✅ NEW: Update ChromaDB if session_id provided
                if request.session_id:
                    try:
                        # Get ChromaDB client and collection
                        client = get_chroma_client()
                        
                        # Find collection (handle various formats)
                        collection_name = request.session_id
                        if len(request.session_id) == 32:
                            collection_name = f"{request.session_id[:8]}-{request.session_id[8:12]}-{request.session_id[12:16]}-{request.session_id[16:20]}-{request.session_id[20:]}"
                        
                        collection = None
                        try:
                            collection = client.get_collection(name=collection_name)
                        except:
                            # Try timestamped versions
                            try:
                                all_collections = client.list_collections()
                                for coll in all_collections:
                                    if coll.name.startswith(collection_name):
                                        collection = client.get_collection(name=coll.name)
                                        break
                            except:
                                pass
                        
                        if collection:
                            # Find chunk by ID or by document_name + chunk_index
                            target_chunk_id = None
                            existing = None
                            
                            if request.chunk_id:
                                # Direct ID lookup
                                existing = collection.get(ids=[request.chunk_id], include=['embeddings', 'metadatas'])
                                if existing and existing['ids']:
                                    target_chunk_id = request.chunk_id
                            elif request.document_name is not None and request.chunk_index is not None:
                                # Search by document_name and chunk_index in metadata
                                all_chunks = collection.get(include=['embeddings', 'metadatas'])
                                for i, metadata in enumerate(all_chunks.get('metadatas', [])):
                                    doc_name = metadata.get('document_name') or metadata.get('filename') or metadata.get('source_file')
                                    chunk_idx = metadata.get('chunk_index')
                                    if doc_name == request.document_name and chunk_idx == request.chunk_index:
                                        target_chunk_id = all_chunks['ids'][i]
                                        existing = {
                                            'ids': [target_chunk_id],
                                            'embeddings': [all_chunks['embeddings'][i]],
                                            'metadatas': [metadata]
                                        }
                                        break
                            
                            if existing and existing['ids'] and target_chunk_id:
                                # Update with improved text, preserving embedding and metadata
                                updated_metadata = existing['metadatas'][0].copy() if existing['metadatas'] else {}
                                updated_metadata['llm_improved'] = True
                                updated_metadata['llm_model'] = request.model_name
                                updated_metadata['improvement_timestamp'] = datetime.now().isoformat()
                                
                                # Update ChromaDB (preserve original embedding!)
                                collection.update(
                                    ids=[target_chunk_id],
                                    documents=[improved_text],
                                    metadatas=[updated_metadata],
                                    embeddings=[existing['embeddings'][0]]  # Preserve original embedding
                                )
                                
                                logger.info(f"✅ Chunk updated in ChromaDB (ID: {target_chunk_id}, doc: {request.document_name}, idx: {request.chunk_index})")
                            else:
                                logger.warning(f"⚠️ Chunk not found in ChromaDB (doc: {request.document_name}, idx: {request.chunk_index})")
                        else:
                            logger.warning(f"⚠️ Collection not found for session {request.session_id}")
                    except Exception as db_error:
                        logger.error(f"❌ Failed to update ChromaDB: {db_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                        # Don't fail the whole request if DB update fails
                
                return ImproveSingleChunkResponse(
                    success=True,
                    original_text=request.chunk_text,
                    improved_text=improved_text,
                    message="Chunk improved successfully",
                    processing_time_ms=processing_time
                )
            else:
                return ImproveSingleChunkResponse(
                    success=False,
                    original_text=request.chunk_text,
                    improved_text=None,
                    message="LLM did not improve the chunk",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
        else:
            return ImproveSingleChunkResponse(
                success=False,
                original_text=request.chunk_text,
                improved_text=None,
                message="LLM post-processing failed",
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
    except Exception as e:
        logger.error(f"❌ Error improving single chunk: {e}")
        return ImproveSingleChunkResponse(
            success=False,
            original_text=request.chunk_text,
            improved_text=None,
            message=f"Error: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000
        )


@app.post("/sessions/{session_id}/chunks/improve-all", response_model=ImproveAllChunksResponse)
async def improve_all_chunks(session_id: str, request: ImproveAllChunksRequest):
    """
    Improve all chunks in a session using LLM post-processing.
    Skips chunks that are already marked as improved (unless skip_already_improved=False).
    Updates chunk metadata to mark improved chunks.
    """
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Starting bulk chunk improvement for session {session_id} with {request.model_name}")
        
        # Get ChromaDB client
        client = get_chroma_client()
        
        # Find the collection (handle timestamped names)
        collection_name = session_id
        if len(session_id) == 32:
            collection_name = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
        
        collection = None
        try:
            collection = client.get_collection(name=collection_name)
        except:
            # Try alternatives including timestamped
            alternative_names = [f"session_{session_id}"]
            
            # Try timestamped versions
            try:
                all_collections = client.list_collections()
                for coll in all_collections:
                    if coll.name.startswith(collection_name + "_"):
                        alternative_names.insert(0, coll.name)
            except:
                pass
            
            for alt_name in alternative_names:
                try:
                    collection = client.get_collection(name=alt_name)
                    collection_name = alt_name
                    break
                except:
                    continue
        
        if not collection:
            return ImproveAllChunksResponse(
                success=False,
                total_chunks=0,
                processed=0,
                improved=0,
                failed=0,
                skipped=0,
                message=f"Collection not found for session {session_id}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get all chunks from collection INCLUDING embeddings
        results = collection.get(include=['documents', 'metadatas', 'embeddings'])
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        embeddings = results.get('embeddings', [])  # Get existing embeddings!
        
        total_chunks = len(documents)
        logger.info(f"📊 Found {total_chunks} chunks to process")
        logger.info(f"📊 Retrieved {len(embeddings)} existing embeddings (will be preserved)")
        
        # Import post-processor
        try:
            from src.text_processing.chunk_post_processor_grok import GrokChunkPostProcessor, PostProcessingConfig
        except ImportError:
            try:
                from src.text_processing.chunk_post_processor import ChunkPostProcessor as GrokChunkPostProcessor, PostProcessingConfig
            except ImportError:
                return ImproveAllChunksResponse(
                    success=False,
                    total_chunks=total_chunks,
                    processed=0,
                    improved=0,
                    failed=0,
                    skipped=0,
                    message="LLM post-processing not available",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
        
        # Create post-processor
        config = PostProcessingConfig(
            enabled=True,
            model_name=request.model_name,
            model_inference_url=MODEL_INFERENCER_URL,
            language=request.language,
            timeout_seconds=30,
            retry_attempts=2
        )
        post_processor = GrokChunkPostProcessor(config)
        
        # Process chunks
        processed_count = 0
        improved_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i, (doc, metadata, chunk_id, embedding) in enumerate(zip(documents, metadatas, ids, embeddings)):
            # Check if already improved
            if request.skip_already_improved and metadata.get('llm_improved'):
                logger.debug(f"⏭️  Skipping chunk {i+1}/{total_chunks} (already improved)")
                skipped_count += 1
                continue
            
            # Force processing (disable worth check)
            original_check = post_processor._is_chunk_worth_processing
            post_processor._is_chunk_worth_processing = lambda x: True
            
            try:
                # Process chunk
                improved_chunks = post_processor.process_chunks([doc])
                
                if improved_chunks and len(improved_chunks) > 0:
                    improved_text = improved_chunks[0]
                    
                    # Check if actually improved
                    if improved_text != doc and len(improved_text.strip()) > 10:
                        # Update chunk in ChromaDB
                        updated_metadata = metadata.copy()
                        updated_metadata['llm_improved'] = True
                        updated_metadata['llm_model'] = request.model_name
                        updated_metadata['improvement_timestamp'] = datetime.now().isoformat()
                        
                        # IMPORTANT: Update document and metadata while PRESERVING original embeddings!
                        # We pass the original embedding back to ChromaDB to prevent automatic re-calculation
                        # This avoids: 1) Model download, 2) Dimension mismatch, 3) Slow processing
                        
                        # Retry logic for ChromaDB update (SSL timeout handling)
                        max_retries = 5  # Increased from 3
                        retry_delay = 3.0  # Increased from 2.0
                        
                        for retry in range(max_retries):
                            try:
                                collection.update(
                                    ids=[chunk_id],
                                    documents=[improved_text],
                                    metadatas=[updated_metadata],
                                    embeddings=[embedding]  # CRITICAL: Pass original embedding to prevent re-calculation!
                                )
                                if retry > 0:
                                    logger.info(f"✅ ChromaDB update succeeded on attempt {retry+1}")
                                break  # Success, exit retry loop
                            except Exception as update_error:
                                if retry < max_retries - 1:
                                    logger.warning(f"⚠️ ChromaDB update failed (attempt {retry+1}/{max_retries}): {update_error}")
                                    # Exponential backoff: 3s, 6s, 9s, 12s
                                    wait_time = retry_delay * (retry + 1)
                                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                                    time.sleep(wait_time)
                                else:
                                    logger.error(f"❌ ChromaDB update failed after {max_retries} attempts")
                                    raise  # Re-raise on final attempt
                        
                        improved_count += 1
                        logger.info(f"✅ Chunk {i+1}/{total_chunks} improved successfully")
                    else:
                        failed_count += 1
                        logger.warning(f"⚠️ Chunk {i+1}/{total_chunks} - LLM did not improve")
                else:
                    failed_count += 1
                    logger.warning(f"❌ Chunk {i+1}/{total_chunks} - Processing failed")
                
                processed_count += 1
                
            except Exception as e:
                failed_count += 1
                processed_count += 1
                logger.error(f"❌ Chunk {i+1}/{total_chunks} error: {e}")
            finally:
                # Restore original check
                post_processor._is_chunk_worth_processing = original_check
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"""
✅ Bulk chunk improvement completed!
   📊 Total: {total_chunks} chunks
   ✅ Improved: {improved_count}
   ❌ Failed: {failed_count}
   ⏭️  Skipped: {skipped_count}
   ⏱️  Time: {processing_time:.0f}ms
        """)
        
        return ImproveAllChunksResponse(
            success=True,
            total_chunks=total_chunks,
            processed=processed_count,
            improved=improved_count,
            failed=failed_count,
            skipped=skipped_count,
            message=f"Processed {processed_count} chunks: {improved_count} improved, {failed_count} failed, {skipped_count} skipped",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Error in bulk chunk improvement: {e}")
        return ImproveAllChunksResponse(
            success=False,
            total_chunks=0,
            processed=0,
            improved=0,
            failed=0,
            skipped=0,
            message=f"Error: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000
        )

    
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)