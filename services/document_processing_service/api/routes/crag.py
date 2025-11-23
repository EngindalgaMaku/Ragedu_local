"""
CRAG evaluation endpoint
"""
from fastapi import APIRouter
from models.schemas import CRAGEvaluationRequest, CRAGEvaluationResponse
from services.crag_evaluator import CRAGEvaluator
from config import MODEL_INFERENCER_URL
from utils.logger import logger

router = APIRouter()


@router.post("/crag-evaluate", response_model=CRAGEvaluationResponse)
@router.post("/crag/evaluate", response_model=CRAGEvaluationResponse)
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
        
        logger.info(
            f"✅ CRAG Evaluation completed: action={evaluation_result['action']}, "
            f"confidence={evaluation_result['confidence']}"
        )
        
        return CRAGEvaluationResponse(
            success=True,
            evaluation=evaluation_result,
            filtered_docs=evaluation_result.get("filtered_docs", request.retrieved_docs)
        )
        
    except Exception as e:
        logger.error(f"CRAG Evaluation error: {e}", exc_info=True)
        
        # Return fallback evaluation
        avg_score = (
            sum(doc.get("score", 0) for doc in request.retrieved_docs) /
            len(request.retrieved_docs)
            if request.retrieved_docs else 0
        )
        
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






