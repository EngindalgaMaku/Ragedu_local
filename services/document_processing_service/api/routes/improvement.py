"""
Chunk improvement endpoints
"""
from fastapi import APIRouter
from models.schemas import (
    ImproveSingleChunkRequest,
    ImproveSingleChunkResponse,
    ImproveAllChunksRequest,
    ImproveAllChunksResponse
)
from services.chunk_improver import improve_single_chunk, improve_all_chunks_in_session
from utils.logger import logger

router = APIRouter()


@router.post("/chunks/improve-single", response_model=ImproveSingleChunkResponse)
async def improve_chunk_endpoint(request: ImproveSingleChunkRequest):
    """
    Improve a single chunk using LLM post-processing
    
    Features:
    - Turkish language support
    - Configurable LLM model
    - Optional ChromaDB update
    """
    logger.info(f"📝 Improving single chunk ({len(request.chunk_text)} chars)")
    
    result = improve_single_chunk(
        chunk_text=request.chunk_text,
        language=request.language,
        model_name=request.model_name,
        session_id=request.session_id,
        chunk_id=request.chunk_id,
        document_name=request.document_name,
        chunk_index=request.chunk_index,
        update_chromadb=request.update_chromadb
    )
    
    return ImproveSingleChunkResponse(**result)


@router.post("/sessions/{session_id}/chunks/improve", response_model=ImproveSingleChunkResponse)
async def improve_session_chunk(session_id: str, request: ImproveSingleChunkRequest):
    """
    Improve a single chunk within a specific session
    """
    logger.info(f"📝 Improving chunk in session {session_id}")
    
    result = improve_single_chunk(
        chunk_text=request.chunk_text,
        language=request.language,
        model_name=request.model_name,
        session_id=session_id,
        chunk_id=request.chunk_id,
        document_name=request.document_name,
        chunk_index=request.chunk_index,
        update_chromadb=request.update_chromadb
    )
    
    return ImproveSingleChunkResponse(**result)


@router.post("/sessions/{session_id}/chunks/improve-all", response_model=ImproveAllChunksResponse)
async def improve_all_chunks_endpoint(session_id: str, request: ImproveAllChunksRequest):
    """
    Improve all chunks in a session using LLM post-processing
    
    Features:
    - Bulk processing
    - Skip already improved chunks
    - Retry mechanism
    - Embedding preservation
    """
    logger.info(f"🚀 Starting bulk chunk improvement for session {session_id}")
    
    result = improve_all_chunks_in_session(
        session_id=session_id,
        language=request.language,
        model_name=request.model_name,
        skip_already_improved=request.skip_already_improved
    )
    
    return ImproveAllChunksResponse(**result)






