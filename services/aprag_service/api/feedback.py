"""
Feedback collection endpoints
Collects student feedback on interactions
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Import database manager
try:
    from database.database import DatabaseManager
    from main import db_manager
except ImportError:
    # Fallback import
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from database.database import DatabaseManager
    db_manager = None


class FeedbackCreate(BaseModel):
    """Request model for creating feedback"""
    interaction_id: int
    user_id: str
    session_id: str
    understanding_level: Optional[int] = Field(None, ge=1, le=5)
    answer_adequacy: Optional[int] = Field(None, ge=1, le=5)
    satisfaction_level: Optional[int] = Field(None, ge=1, le=5)
    difficulty_level: Optional[int] = Field(None, ge=1, le=5)
    topic_understood: Optional[bool] = None
    answer_helpful: Optional[bool] = None
    needs_more_explanation: Optional[bool] = None
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response model for feedback"""
    feedback_id: int
    interaction_id: int
    user_id: str
    session_id: str
    understanding_level: Optional[int]
    answer_adequacy: Optional[int]
    satisfaction_level: Optional[int]
    difficulty_level: Optional[int]
    topic_understood: Optional[bool]
    answer_helpful: Optional[bool]
    needs_more_explanation: Optional[bool]
    comment: Optional[str]
    timestamp: str


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


@router.post("", status_code=201)
async def create_feedback(feedback: FeedbackCreate, db: DatabaseManager = Depends(get_db)):
    """
    Create feedback for an interaction
    
    This endpoint collects student feedback on the quality and helpfulness
    of RAG responses for personalization and improvement.
    """
    try:
        logger.info(f"Collecting feedback for interaction {feedback.interaction_id}, user {feedback.user_id}")
        
        # Verify interaction exists
        interaction_check = db.execute_query(
            "SELECT interaction_id, user_id, session_id FROM student_interactions WHERE interaction_id = ?",
            (feedback.interaction_id,)
        )
        
        if not interaction_check:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        interaction = interaction_check[0]
        
        # Verify user_id and session_id match
        if interaction["user_id"] != feedback.user_id or interaction["session_id"] != feedback.session_id:
            raise HTTPException(
                status_code=403,
                detail="User ID or session ID does not match the interaction"
            )
        
        # Insert feedback into database
        query = """
            INSERT INTO student_feedback 
            (interaction_id, user_id, session_id, understanding_level, answer_adequacy,
             satisfaction_level, difficulty_level, topic_understood, answer_helpful,
             needs_more_explanation, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        feedback_id = db.execute_insert(
            query,
            (
                feedback.interaction_id,
                feedback.user_id,
                feedback.session_id,
                feedback.understanding_level,
                feedback.answer_adequacy,
                feedback.satisfaction_level,
                feedback.difficulty_level,
                feedback.topic_understood,
                feedback.answer_helpful,
                feedback.needs_more_explanation,
                feedback.comment
            )
        )
        
        logger.info(f"Successfully collected feedback {feedback_id} for interaction {feedback.interaction_id}")
        
        # Update student profile with feedback data (async, non-blocking)
        try:
            _update_profile_from_feedback(db, feedback)
        except Exception as e:
            logger.warning(f"Failed to update profile from feedback (non-critical): {e}")
        
        return {
            "feedback_id": feedback_id,
            "message": "Feedback collected successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to collect feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect feedback: {str(e)}"
        )


def _update_profile_from_feedback(db: DatabaseManager, feedback: FeedbackCreate):
    """Update student profile based on feedback (internal helper)"""
    try:
        # Get or create student profile
        profile_check = db.execute_query(
            "SELECT profile_id, total_feedback_count, average_understanding, average_satisfaction FROM student_profiles WHERE user_id = ? AND session_id = ?",
            (feedback.user_id, feedback.session_id)
        )
        
        if not profile_check:
            # Create new profile
            db.execute_insert(
                """
                INSERT INTO student_profiles 
                (user_id, session_id, total_feedback_count, average_understanding, average_satisfaction)
                VALUES (?, ?, 1, ?, ?)
                """,
                (
                    feedback.user_id,
                    feedback.session_id,
                    feedback.understanding_level or 0,
                    feedback.satisfaction_level or 0
                )
            )
        else:
            # Update existing profile
            profile = profile_check[0]
            current_count = profile["total_feedback_count"] or 0
            current_understanding = profile["average_understanding"] or 0
            current_satisfaction = profile["average_satisfaction"] or 0
            
            # Calculate new averages
            new_understanding = (
                (current_understanding * current_count + (feedback.understanding_level or 0)) / (current_count + 1)
                if feedback.understanding_level else current_understanding
            )
            new_satisfaction = (
                (current_satisfaction * current_count + (feedback.satisfaction_level or 0)) / (current_count + 1)
                if feedback.satisfaction_level else current_satisfaction
            )
            
            db.execute_update(
                """
                UPDATE student_profiles 
                SET total_feedback_count = total_feedback_count + 1,
                    average_understanding = ?,
                    average_satisfaction = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
                """,
                (new_understanding, new_satisfaction, feedback.user_id, feedback.session_id)
            )
            
    except Exception as e:
        logger.warning(f"Profile update failed (non-critical): {e}")


@router.get("/{interaction_id}")
async def get_feedback(interaction_id: int, db: DatabaseManager = Depends(get_db)):
    """
    Get feedback for a specific interaction
    """
    try:
        query = "SELECT * FROM student_feedback WHERE interaction_id = ?"
        results = db.execute_query(query, (interaction_id,))
        
        if not results:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        return results[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feedback: {str(e)}"
        )


@router.get("/session/{session_id}")
async def get_session_feedback(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get all feedback for a session
    """
    try:
        query = """
            SELECT f.*, i.query, i.original_response
            FROM student_feedback f
            JOIN student_interactions i ON f.interaction_id = i.interaction_id
            WHERE f.session_id = ?
            ORDER BY f.timestamp DESC
            LIMIT ? OFFSET ?
        """
        
        feedback_list = db.execute_query(query, (session_id, limit, offset))
        
        # Get total count
        count_query = "SELECT COUNT(*) as count FROM student_feedback WHERE session_id = ?"
        count_result = db.execute_query(count_query, (session_id,))
        total = count_result[0]["count"] if count_result else 0
        
        return {
            "feedback": feedback_list,
            "total": total,
            "count": len(feedback_list),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get session feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session feedback: {str(e)}"
        )

