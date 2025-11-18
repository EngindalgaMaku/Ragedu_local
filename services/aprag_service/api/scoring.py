"""
CACS Scoring Endpoints
Conversation-Aware Content Scoring for personalized document ranking

This module requires APRAG and CACS to be enabled.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import json

# Import business logic and database
try:
    from business_logic.cacs import get_cacs_scorer, CACSScorer
    from database.database import DatabaseManager
    from config.feature_flags import FeatureFlags
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from business_logic.cacs import get_cacs_scorer, CACSScorer
    from database.database import DatabaseManager
    from config.feature_flags import FeatureFlags

# DB manager will be injected via dependency
db_manager = None

logger = logging.getLogger(__name__)
router = APIRouter()


class DocumentInput(BaseModel):
    """Single document input"""
    doc_id: str = Field(..., description="Document identifier")
    content: Optional[str] = Field(None, description="Document content (optional)")
    base_score: float = Field(0.5, ge=0.0, le=1.0, description="RAG similarity score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DocumentScoreRequest(BaseModel):
    """Request model for scoring documents"""
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    query: str = Field(..., description="Current query text")
    documents: List[DocumentInput] = Field(..., min_items=1, description="Documents to score")


class DocumentScoreResponse(BaseModel):
    """Response model for scored document"""
    doc_id: str
    final_score: float
    base_score: float
    personal_score: float
    global_score: float
    context_score: float
    breakdown: Dict[str, float]
    cacs_enabled: bool = True
    rank: Optional[int] = None  # Will be set after sorting


class ScoringStatusResponse(BaseModel):
    """Response model for scoring status"""
    cacs_enabled: bool
    aprag_enabled: bool
    egitsel_kbrag_enabled: bool
    message: str


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


@router.get("/status", response_model=ScoringStatusResponse)
async def get_scoring_status():
    """
    Get CACS scoring status
    
    Returns current feature flag status for CACS and dependencies.
    """
    aprag_enabled = FeatureFlags.is_aprag_enabled()
    egitsel_kbrag_enabled = FeatureFlags.is_egitsel_kbrag_enabled()
    cacs_enabled = FeatureFlags.is_cacs_enabled()
    
    if not aprag_enabled:
        message = "APRAG is disabled - CACS unavailable"
    elif not egitsel_kbrag_enabled:
        message = "Eğitsel-KBRAG is disabled - CACS unavailable"
    elif not cacs_enabled:
        message = "CACS is disabled"
    else:
        message = "CACS is ready and enabled"
    
    return ScoringStatusResponse(
        cacs_enabled=cacs_enabled,
        aprag_enabled=aprag_enabled,
        egitsel_kbrag_enabled=egitsel_kbrag_enabled,
        message=message
    )


@router.post("/score", response_model=List[DocumentScoreResponse])
async def score_documents(
    request: DocumentScoreRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Score documents using CACS algorithm
    
    This endpoint takes RAG documents and re-scores them based on:
    - Base score: Semantic similarity (from RAG)
    - Personal score: Student's history and profile
    - Global score: Feedback from all students
    - Context score: Conversation context relevance
    
    Documents are returned sorted by final_score (highest first).
    
    **Requires:** APRAG, Eğitsel-KBRAG, and CACS to be enabled.
    """
    
    # Check if CACS is enabled
    if not FeatureFlags.is_cacs_enabled():
        logger.warning("CACS scoring requested but feature is disabled")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CACS scoring is not enabled. Check feature flags: APRAG_ENABLED, ENABLE_EGITSEL_KBRAG, ENABLE_CACS"
        )
    
    try:
        logger.info(f"CACS scoring request for user {request.user_id}, session {request.session_id}, "
                   f"{len(request.documents)} documents")
        
        # Get CACS scorer instance
        scorer = get_cacs_scorer()
        
        if scorer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CACS scorer initialization failed"
            )
        
        # Fetch student profile
        profile = db.execute_query(
            "SELECT * FROM student_profiles WHERE user_id = ? AND session_id = ?",
            (request.user_id, request.session_id)
        )
        student_profile = profile[0] if profile else {}
        
        if not student_profile:
            logger.info(f"No profile found for user {request.user_id}, using defaults")
        
        # Fetch conversation history (last 20 interactions)
        history = db.execute_query(
            """
            SELECT * FROM student_interactions 
            WHERE user_id = ? AND session_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            (request.user_id, request.session_id)
        )
        
        logger.debug(f"Found {len(history)} past interactions for user {request.user_id}")
        
        # Fetch global scores for all documents
        global_scores = {}
        for doc in request.documents:
            doc_id = doc.doc_id
            score_data = db.execute_query(
                "SELECT * FROM document_global_scores WHERE doc_id = ?",
                (doc_id,)
            )
            if score_data:
                global_scores[doc_id] = score_data[0]
        
        logger.debug(f"Found global scores for {len(global_scores)} documents")
        
        # Score each document
        scored_documents = []
        for doc in request.documents:
            try:
                score_result = scorer.calculate_score(
                    doc_id=doc.doc_id,
                    base_score=doc.base_score,
                    student_profile=student_profile,
                    conversation_history=history,
                    global_scores=global_scores,
                    current_query=request.query
                )
                
                # Create response object
                scored_doc = DocumentScoreResponse(
                    doc_id=doc.doc_id,
                    final_score=score_result['final_score'],
                    base_score=score_result['base_score'],
                    personal_score=score_result['personal_score'],
                    global_score=score_result['global_score'],
                    context_score=score_result['context_score'],
                    breakdown=score_result['breakdown'],
                    cacs_enabled=score_result.get('cacs_enabled', True)
                )
                
                scored_documents.append(scored_doc)
                
            except Exception as e:
                logger.error(f"Error scoring document {doc.doc_id}: {e}")
                # Fallback: use base score only
                scored_documents.append(DocumentScoreResponse(
                    doc_id=doc.doc_id,
                    final_score=doc.base_score,
                    base_score=doc.base_score,
                    personal_score=0.5,
                    global_score=0.5,
                    context_score=0.5,
                    breakdown={
                        'base_weight': 1.0,
                        'personal_weight': 0.0,
                        'global_weight': 0.0,
                        'context_weight': 0.0
                    },
                    cacs_enabled=False
                ))
        
        # Sort by final_score (highest first)
        scored_documents.sort(key=lambda x: x.final_score, reverse=True)
        
        # Add rank information
        for rank, doc in enumerate(scored_documents, start=1):
            doc.rank = rank
        
        logger.info(f"Successfully scored and ranked {len(scored_documents)} documents for user {request.user_id}")
        
        # Log top 3 for debugging
        for i, doc in enumerate(scored_documents[:3], start=1):
            logger.debug(f"  Rank {i}: {doc.doc_id} - Score: {doc.final_score:.3f} "
                        f"(base:{doc.base_score:.2f}, personal:{doc.personal_score:.2f}, "
                        f"global:{doc.global_score:.2f}, context:{doc.context_score:.2f})")
        
        return scored_documents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CACS scoring failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CACS scoring failed: {str(e)}"
        )


@router.post("/score-and-save")
async def score_and_save_documents(
    request: DocumentScoreRequest,
    save_to_db: bool = True,
    db: DatabaseManager = Depends(get_db)
):
    """
    Score documents and optionally save scores to database
    
    This endpoint scores documents and saves the CACS scores to the
    `cacs_scores` table for future analysis.
    
    **Note:** interaction_id is required in metadata to save scores.
    """
    
    # Check if CACS is enabled
    if not FeatureFlags.is_cacs_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CACS scoring is not enabled"
        )
    
    try:
        # Score documents using existing endpoint
        scored_documents = await score_documents(request, db)
        
        # Save scores to database if requested
        if save_to_db:
            saved_count = 0
            for doc in scored_documents:
                # Get interaction_id from request metadata (if provided)
                interaction_id = None
                doc_input = next((d for d in request.documents if d.doc_id == doc.doc_id), None)
                if doc_input and doc_input.metadata:
                    interaction_id = doc_input.metadata.get('interaction_id')
                
                if interaction_id:
                    try:
                        db.execute_insert(
                            """
                            INSERT INTO cacs_scores 
                            (interaction_id, doc_id, final_score, base_score, personal_score,
                             global_score, context_score, weight_base, weight_personal,
                             weight_global, weight_context)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                interaction_id,
                                doc.doc_id,
                                doc.final_score,
                                doc.base_score,
                                doc.personal_score,
                                doc.global_score,
                                doc.context_score,
                                doc.breakdown.get('base_weight', 0.30),
                                doc.breakdown.get('personal_weight', 0.25),
                                doc.breakdown.get('global_weight', 0.25),
                                doc.breakdown.get('context_weight', 0.20)
                            )
                        )
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to save CACS score for doc {doc.doc_id}: {e}")
            
            logger.info(f"Saved {saved_count}/{len(scored_documents)} CACS scores to database")
        
        return {
            "scored_documents": scored_documents,
            "saved_to_db": save_to_db,
            "saved_count": saved_count if save_to_db else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Score and save failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Score and save failed: {str(e)}"
        )

