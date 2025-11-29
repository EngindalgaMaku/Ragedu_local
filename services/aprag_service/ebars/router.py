"""
EBARS API Router
Endpoints for Emoji-Based Adaptive Response System
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import os

from database.database import DatabaseManager

def get_db() -> DatabaseManager:
    """Get database manager dependency"""
    from database.database import DatabaseManager
    db_path = os.getenv("APRAG_DB_PATH", os.getenv("DATABASE_PATH", "/app/data/rag_assistant.db"))
    return DatabaseManager(db_path)
from config.feature_flags import is_feature_enabled
from .feedback_handler import FeedbackHandler
from .score_calculator import ComprehensionScoreCalculator
from .prompt_adapter import PromptAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ebars", tags=["EBARS"])


# ============================================================================
# Request/Response Models
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request model for emoji feedback"""
    user_id: str
    session_id: str
    emoji: str  # '👍', '😊', '😐', '❌'
    interaction_id: Optional[int] = None
    query_text: Optional[str] = None


class AdaptivePromptRequest(BaseModel):
    """Request model for adaptive prompt generation"""
    user_id: str
    session_id: str
    base_prompt: Optional[str] = None
    query: Optional[str] = None
    original_response: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def check_ebars_enabled(session_id: Optional[str] = None) -> bool:
    """Check if EBARS feature is enabled"""
    return is_feature_enabled("ebars", session_id=session_id)


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/feedback")
async def process_emoji_feedback(
    request: FeedbackRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Process emoji feedback and update comprehension score.
    
    This endpoint:
    1. Updates comprehension score based on emoji
    2. Adjusts difficulty level if needed
    3. Records feedback in history
    4. Returns updated state
    """
    try:
        # Check if EBARS is enabled
        if not check_ebars_enabled(request.session_id):
            raise HTTPException(
                status_code=403,
                detail="EBARS feature is disabled for this session"
            )
        
        # Process feedback
        handler = FeedbackHandler(db)
        result = handler.process_feedback(
            user_id=request.user_id,
            session_id=request.session_id,
            emoji=request.emoji,
            interaction_id=request.interaction_id,
            query_text=request.query_text
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Failed to process feedback')
            )
        
        return {
            "success": True,
            "message": "Feedback processed successfully",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{user_id}/{session_id}")
async def get_ebars_state(
    user_id: str,
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get current EBARS state for a student.
    
    Returns:
        - Current comprehension score
        - Current difficulty level
        - Prompt parameters
        - Statistics
    """
    try:
        # Check if EBARS is enabled
        if not check_ebars_enabled(session_id):
            raise HTTPException(
                status_code=403,
                detail="EBARS feature is disabled for this session"
            )
        
        handler = FeedbackHandler(db)
        state = handler.get_current_state(user_id, session_id)
        
        return {
            "success": True,
            "data": state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting EBARS state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompt/generate")
async def generate_adaptive_prompt(
    request: AdaptivePromptRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Generate adaptive prompt based on student's comprehension score.
    
    This endpoint generates a prompt that instructs the LLM to adapt
    the response according to the student's current difficulty level.
    """
    try:
        # Check if EBARS is enabled
        if not check_ebars_enabled(request.session_id):
            raise HTTPException(
                status_code=403,
                detail="EBARS feature is disabled for this session"
            )
        
        handler = FeedbackHandler(db)
        prompt = handler.generate_adaptive_prompt(
            user_id=request.user_id,
            session_id=request.session_id,
            base_prompt=request.base_prompt,
            query=request.query,
            original_response=request.original_response
        )
        
        # Get current state for context
        state = handler.get_current_state(request.user_id, request.session_id)
        
        return {
            "success": True,
            "prompt": prompt,
            "comprehension_score": state['comprehension_score'],
            "difficulty_level": state['difficulty_level'],
            "prompt_parameters": state['prompt_parameters']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating adaptive prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/score/{user_id}/{session_id}")
async def get_comprehension_score(
    user_id: str,
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get current comprehension score for a student.
    """
    try:
        # Check if EBARS is enabled
        if not check_ebars_enabled(session_id):
            raise HTTPException(
                status_code=403,
                detail="EBARS feature is disabled for this session"
            )
        
        calculator = ComprehensionScoreCalculator(db)
        score = calculator.get_score(user_id, session_id)
        difficulty = calculator.get_difficulty_level(user_id, session_id)
        
        return {
            "success": True,
            "comprehension_score": score,
            "difficulty_level": difficulty
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comprehension score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score/reset/{user_id}/{session_id}")
async def reset_comprehension_score(
    user_id: str,
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Reset comprehension score to default (50.0).
    Useful for testing or when student wants to start fresh.
    """
    try:
        # Check if EBARS is enabled
        if not check_ebars_enabled(session_id):
            raise HTTPException(
                status_code=403,
                detail="EBARS feature is disabled for this session"
            )
        
        with db.get_connection() as conn:
            conn.execute("""
                UPDATE student_comprehension_scores
                SET comprehension_score = 50.0,
                    current_difficulty_level = 'normal',
                    consecutive_positive_count = 0,
                    consecutive_negative_count = 0,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
            """, (user_id, session_id))
            conn.commit()
        
        return {
            "success": True,
            "message": "Comprehension score reset to default (50.0)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting comprehension score: {e}")
        raise HTTPException(status_code=500, detail=str(e))

