"""
Rerank endpoint
"""
from fastapi import APIRouter
from models.schemas import RerankRequest, RerankResponse
from services.reranker import Reranker
from config import MODEL_INFERENCER_URL
from utils.logger import logger

router = APIRouter()


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents based on query relevance.
    
    This endpoint reranks documents using a cross-encoder model (reranker service).
    It only performs reranking and sorting - no accept/reject/filter decisions.
    """
    try:
        logger.info(f"Rerank request for query: {request.query[:50]}...")
        
        # Initialize Reranker
        reranker = Reranker(model_inference_url=MODEL_INFERENCER_URL)
        
        # Perform reranking
        rerank_result = reranker.rerank_documents(
            query=request.query,
            documents=request.documents
        )
        
        logger.info(
            f"✅ Rerank completed: max_score={rerank_result['max_score']:.4f}, "
            f"avg_score={rerank_result['avg_score']:.4f}, "
            f"documents={len(rerank_result['reranked_docs'])}"
        )
        
        return RerankResponse(
            success=True,
            reranked_docs=rerank_result["reranked_docs"],
            scores=rerank_result["scores"],
            max_score=rerank_result["max_score"],
            avg_score=rerank_result["avg_score"],
            reranker_type=rerank_result["reranker_type"]
        )
        
    except Exception as e:
        logger.error(f"Rerank error: {e}", exc_info=True)
        
        # Return fallback: return documents in original order
        return RerankResponse(
            success=False,
            reranked_docs=request.documents,
            scores=[],
            max_score=0.0,
            avg_score=0.0,
            reranker_type="unknown"
        )



