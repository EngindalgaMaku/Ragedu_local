"""
Pydantic models for request/response schemas
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class ProcessRequest(BaseModel):
    """Request model for text processing and storage"""
    text: str
    metadata: Optional[Dict[str, Any]] = {}
    collection_name: Optional[str] = "documents"
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    chunk_strategy: Optional[str] = "lightweight"  # Enable lightweight Turkish chunking by default
    use_llm_post_processing: Optional[bool] = False  # Optional LLM post-processing for chunk refinement
    llm_model_name: Optional[str] = "llama-3.1-8b-instant"  # LLM model for post-processing
    model_inference_url: Optional[str] = None  # Override model inference URL for LLM post-processing


class ProcessResponse(BaseModel):
    """Response model for text processing"""
    success: bool
    message: str
    chunks_processed: int
    collection_name: str
    chunk_ids: List[str]


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries"""
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
    conversation_history: Optional[List[Dict[str, str]]] = None  # [{"role": "user", "content": "..."}]
    use_hybrid_search: Optional[bool] = False  # Disabled: Alibaba embedding + reranker is sufficient
    bm25_weight: Optional[float] = 0.3  # Weight for BM25 score (0.3 = 30% keyword, 70% semantic)
    skip_llm: Optional[bool] = False  # If True, skip LLM generation and return only chunks (for hybrid retriever)


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str
    sources: List[Dict[str, Any]] = []
    chain_type: Optional[str] = None
    correction: Optional[Dict[str, Any]] = None  # For self-correction details


class RerankRequest(BaseModel):
    """Request model for document reranking"""
    query: str
    documents: List[Dict[str, Any]]


class RerankResponse(BaseModel):
    """Response model for document reranking"""
    success: bool
    reranked_docs: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]
    max_score: float
    avg_score: float
    reranker_type: str


class ImproveSingleChunkRequest(BaseModel):
    """Request model for improving a single chunk"""
    chunk_text: str
    language: Optional[str] = "tr"
    model_name: Optional[str] = "llama-3.1-8b-instant"
    session_id: Optional[str] = None  # For ChromaDB update
    chunk_id: Optional[str] = None  # For ChromaDB update (if known)
    document_name: Optional[str] = None  # Alternative to chunk_id
    chunk_index: Optional[int] = None  # Alternative to chunk_id
    update_chromadb: Optional[bool] = True  # Whether to update ChromaDB with improved text


class ImproveSingleChunkResponse(BaseModel):
    """Response model for single chunk improvement"""
    success: bool
    original_text: str
    improved_text: Optional[str] = None
    message: Optional[str] = None
    processing_time_ms: Optional[float] = None


class ImproveAllChunksRequest(BaseModel):
    """Request model for improving all chunks in a session"""
    language: Optional[str] = "tr"
    model_name: Optional[str] = "llama-3.1-8b-instant"
    skip_already_improved: Optional[bool] = True


class ImproveAllChunksResponse(BaseModel):
    """Response model for bulk chunk improvement"""
    success: bool
    total_chunks: int
    processed: int
    improved: int
    failed: int
    skipped: int
    message: str
    processing_time_ms: float


class RetrieveRequest(BaseModel):
    """Request model for document retrieval (testing RAG retrieval quality)"""
    query: str
    collection_name: str
    top_k: int = 5
    embedding_model: Optional[str] = None


class RetrieveResponse(BaseModel):
    """Response model for document retrieval"""
    success: bool
    results: List[Dict[str, Any]]
    total: int
