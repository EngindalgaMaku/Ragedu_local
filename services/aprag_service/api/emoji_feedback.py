"""
Emoji-based Micro-Feedback Endpoints (Faz 4)
Quick and easy feedback collection using emojis

This module requires APRAG and Emoji Feedback to be enabled.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime

# Import database and dependencies
try:
    from database.database import DatabaseManager
    from config.feature_flags import FeatureFlags
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from database.database import DatabaseManager
    from config.feature_flags import FeatureFlags

# DB manager will be injected via dependency
db_manager = None

logger = logging.getLogger(__name__)
router = APIRouter()


# Emoji to score mapping (Eğitsel-KBRAG standard)
EMOJI_SCORE_MAP = {
    '😊': 0.7,  # Anladım (I understood)
    '👍': 1.0,  # Mükemmel (Excellent)
    '😐': 0.2,  # Karışık (Confused)
    '❌': 0.0,  # Anlamadım (I didn't understand)
}

# Emoji descriptions
EMOJI_DESCRIPTIONS = {
    '😊': 'Anladım - Cevap anlaşılır',
    '👍': 'Mükemmel - Çok açıklayıcı',
    '😐': 'Karışık - Ek açıklama gerekli',
    '❌': 'Anlamadım - Alternatif yaklaşım gerekli',
}


class EmojiFeedbackCreate(BaseModel):
    """Request model for emoji feedback"""
    interaction_id: int = Field(..., description="Interaction ID to provide feedback for")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    emoji: str = Field(..., description="Emoji feedback: 😊, 👍, 😐, or ❌")
    comment: Optional[str] = Field(None, max_length=500, description="Optional comment")


class MultiFeedbackCreate(BaseModel):
    """Request model for multi-dimensional feedback"""
    interaction_id: int = Field(..., description="Interaction ID to provide feedback for")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    
    # Multi-dimensional scores (1-5 scale)
    understanding: int = Field(..., ge=1, le=5, description="Understanding level (1-5)")
    relevance: int = Field(..., ge=1, le=5, description="Relevance level (1-5)")
    clarity: int = Field(..., ge=1, le=5, description="Clarity level (1-5)")
    
    # Optional emoji and comment
    emoji: Optional[str] = Field(None, description="Optional emoji feedback")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional comment")


class MultiFeedbackResponse(BaseModel):
    """Response model for multi-dimensional feedback"""
    message: str
    interaction_id: int
    feedback_entry_id: int
    dimensions: Dict[str, int]
    overall_score: float
    profile_updated: bool = False


class MultiDimensionalStats(BaseModel):
    """Response model for multi-dimensional feedback statistics"""
    user_id: str
    session_id: str
    
    # Dimension averages
    avg_understanding: Optional[float] = None
    avg_relevance: Optional[float] = None
    avg_clarity: Optional[float] = None
    avg_overall: Optional[float] = None
    
    # Feedback counts
    total_feedback_count: int
    dimension_feedback_count: int
    emoji_only_count: int
    
    # Distributions
    understanding_distribution: Dict[str, int]
    relevance_distribution: Dict[str, int]
    clarity_distribution: Dict[str, int]
    
    # Trend analysis
    improvement_trend: str = "insufficient_data"
    weak_dimensions: List[str] = []
    strong_dimensions: List[str] = []


class EmojiFeedbackResponse(BaseModel):
    """Response model for emoji feedback"""
    message: str
    emoji: str
    score: float
    description: str
    interaction_id: int
    profile_updated: bool = False


class EmojiStatsResponse(BaseModel):
    """Response model for emoji feedback statistics"""
    user_id: str
    session_id: str
    total_feedback_count: int
    emoji_distribution: Dict[str, int]
    avg_score: float
    most_common_emoji: Optional[str] = None
    recent_trend: str = "neutral"  # positive, negative, neutral


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


@router.get("/emojis")
async def get_available_emojis():
    """
    Get available emoji options and their meanings
    
    Returns the emoji options students can use for quick feedback.
    """
    return {
        "emojis": [
            {
                "emoji": "😊",
                "name": "anladim",
                "description": EMOJI_DESCRIPTIONS['😊'],
                "score": EMOJI_SCORE_MAP['😊']
            },
            {
                "emoji": "👍",
                "name": "mukemmel",
                "description": EMOJI_DESCRIPTIONS['👍'],
                "score": EMOJI_SCORE_MAP['👍']
            },
            {
                "emoji": "😐",
                "name": "karisik",
                "description": EMOJI_DESCRIPTIONS['😐'],
                "score": EMOJI_SCORE_MAP['😐']
            },
            {
                "emoji": "❌",
                "name": "anlamadim",
                "description": EMOJI_DESCRIPTIONS['❌'],
                "score": EMOJI_SCORE_MAP['❌']
            }
        ]
    }


@router.post("", response_model=EmojiFeedbackResponse, status_code=201)
async def create_emoji_feedback(
    feedback: EmojiFeedbackCreate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Submit emoji-based feedback for an interaction
    
    **Eğitsel-KBRAG Micro-Feedback Mechanism**
    
    This endpoint allows students to provide quick feedback using emojis:
    - 😊 Anladım (I understood)
    - 👍 Mükemmel (Excellent)
    - 😐 Karışık (Confused)
    - ❌ Anlamadım (I didn't understand)
    
    **Effects:**
    - Updates interaction with emoji feedback
    - Updates student profile (real-time)
    - Updates document global scores
    - Triggers adaptive responses if needed
    """
    
    # Check if emoji feedback is enabled
    if not FeatureFlags.is_emoji_feedback_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emoji feedback is not enabled"
        )
    
    try:
        # Validate emoji
        if feedback.emoji not in EMOJI_SCORE_MAP:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid emoji. Must be one of: {list(EMOJI_SCORE_MAP.keys())}"
            )
        
        # Get emoji score
        emoji_score = EMOJI_SCORE_MAP[feedback.emoji]
        
        logger.info(f"Emoji feedback {feedback.emoji} for interaction {feedback.interaction_id} "
                   f"by user {feedback.user_id} (score: {emoji_score})")
        
        # Check if interaction exists
        interaction = db.execute_query(
            "SELECT * FROM student_interactions WHERE interaction_id = ?",
            (feedback.interaction_id,)
        )
        
        if not interaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interaction {feedback.interaction_id} not found"
            )
        
        # Update interaction with emoji feedback
        db.execute_update(
            """
            UPDATE student_interactions 
            SET emoji_feedback = ?,
                feedback_score = ?,
                emoji_feedback_timestamp = CURRENT_TIMESTAMP,
                emoji_comment = ?
            WHERE interaction_id = ?
            """,
            (feedback.emoji, emoji_score, feedback.comment, feedback.interaction_id)
        )
        
        logger.info(f"Updated interaction {feedback.interaction_id} with emoji {feedback.emoji}")
        
        # Update global document scores (if sources available)
        interaction_data = interaction[0]
        sources_json = interaction_data.get('sources')
        
        if sources_json:
            try:
                if isinstance(sources_json, str):
                    sources = json.loads(sources_json)
                else:
                    sources = sources_json
                
                for source in sources:
                    doc_id = source.get('doc_id') or source.get('document_id')
                    if doc_id:
                        _update_global_score(db, doc_id, emoji_score, feedback.emoji)
                
                logger.debug(f"Updated global scores for {len(sources)} documents")
            except Exception as e:
                logger.warning(f"Failed to update global scores: {e}")
        
        # Update student profile (real-time)
        profile_updated = _update_profile_from_emoji(
            db,
            feedback.user_id,
            feedback.session_id,
            emoji_score,
            feedback.emoji
        )
        
        # Update emoji summary table
        _update_emoji_summary(db, feedback.user_id, feedback.session_id, feedback.emoji, emoji_score)
        
        logger.info(f"Emoji feedback {feedback.emoji} successfully recorded")
        
        return EmojiFeedbackResponse(
            message="Geri bildiriminiz kaydedildi. Teşekkürler!",
            emoji=feedback.emoji,
            score=emoji_score,
            description=EMOJI_DESCRIPTIONS[feedback.emoji],
            interaction_id=feedback.interaction_id,
            profile_updated=profile_updated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create emoji feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record emoji feedback: {str(e)}"
        )


@router.get("/stats/{user_id}/{session_id}", response_model=EmojiStatsResponse)
async def get_emoji_stats(
    user_id: str,
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get emoji feedback statistics for a user/session
    
    Returns distribution of emoji feedback and trends.
    """
    
    if not FeatureFlags.is_emoji_feedback_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emoji feedback is not enabled"
        )
    
    try:
        # Get emoji summary
        summary = db.execute_query(
            """
            SELECT emoji, emoji_count, avg_score
            FROM emoji_feedback_summary
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id)
        )
        
        if not summary:
            # No feedback yet
            return EmojiStatsResponse(
                user_id=user_id,
                session_id=session_id,
                total_feedback_count=0,
                emoji_distribution={},
                avg_score=0.5,
                most_common_emoji=None,
                recent_trend="neutral"
            )
        
        # Calculate statistics
        emoji_distribution = {}
        total_count = 0
        weighted_score_sum = 0.0
        
        for row in summary:
            emoji = row['emoji']
            count = row['emoji_count']
            avg_score = row['avg_score']
            
            emoji_distribution[emoji] = count
            total_count += count
            weighted_score_sum += count * avg_score
        
        avg_score = weighted_score_sum / total_count if total_count > 0 else 0.5
        
        # Most common emoji
        most_common_emoji = max(emoji_distribution, key=emoji_distribution.get) if emoji_distribution else None
        
        # Recent trend (last 5 feedbacks)
        recent_feedbacks = db.execute_query(
            """
            SELECT emoji_feedback, feedback_score
            FROM student_interactions
            WHERE user_id = ? AND session_id = ? AND emoji_feedback IS NOT NULL
            ORDER BY emoji_feedback_timestamp DESC
            LIMIT 5
            """,
            (user_id, session_id)
        )
        
        recent_trend = "neutral"
        if recent_feedbacks and len(recent_feedbacks) >= 3:
            recent_scores = [f['feedback_score'] for f in recent_feedbacks if f.get('feedback_score') is not None]
            if recent_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                if recent_avg >= 0.7:
                    recent_trend = "positive"
                elif recent_avg <= 0.3:
                    recent_trend = "negative"
        
        logger.info(f"Emoji stats for user {user_id}: {total_count} feedbacks, avg score: {avg_score:.2f}")
        
        return EmojiStatsResponse(
            user_id=user_id,
            session_id=session_id,
            total_feedback_count=total_count,
            emoji_distribution=emoji_distribution,
            avg_score=avg_score,
            most_common_emoji=most_common_emoji,
            recent_trend=recent_trend
        )
        
    except Exception as e:
        logger.error(f"Failed to get emoji stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get emoji stats: {str(e)}"
        )


@router.post("/detailed-feedback", response_model=MultiFeedbackResponse, status_code=201)
async def create_detailed_feedback(
    feedback: MultiFeedbackCreate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Submit multi-dimensional feedback for an interaction
    
    **Multi-Dimensional Assessment System**
    
    This endpoint allows students to provide detailed feedback across three dimensions:
    - Understanding: How well did you understand the explanation? (1-5)
    - Relevance: How relevant was the answer to your question? (1-5)
    - Clarity: How clear and well-explained was the answer? (1-5)
    
    **Effects:**
    - Updates interaction with detailed feedback dimensions
    - Updates student profile with multi-dimensional data
    - Updates analytics tables for detailed reporting
    """
    
    # Check if emoji feedback is enabled (multi-dimensional is part of emoji feedback system)
    if not FeatureFlags.is_emoji_feedback_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback system is not enabled"
        )
    
    try:
        logger.info(f"Multi-dimensional feedback for interaction {feedback.interaction_id} "
                   f"by user {feedback.user_id}: U={feedback.understanding}, "
                   f"R={feedback.relevance}, C={feedback.clarity}")
        
        # Check if interaction exists
        interaction = db.execute_query(
            "SELECT * FROM student_interactions WHERE interaction_id = ?",
            (feedback.interaction_id,)
        )
        
        if not interaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interaction {feedback.interaction_id} not found"
            )
        
        # Calculate overall score (average of three dimensions)
        overall_score = (feedback.understanding + feedback.relevance + feedback.clarity) / 3.0
        
        # Validate emoji if provided
        emoji_score = None
        if feedback.emoji and feedback.emoji not in EMOJI_SCORE_MAP:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid emoji. Must be one of: {list(EMOJI_SCORE_MAP.keys())}"
            )
        elif feedback.emoji:
            emoji_score = EMOJI_SCORE_MAP[feedback.emoji]
        
        # Create detailed feedback entry
        entry_id = db.execute_insert(
            """
            INSERT INTO detailed_feedback_entries
            (interaction_id, user_id, session_id, understanding_score, relevance_score,
             clarity_score, overall_score, emoji_feedback, emoji_score, comment, feedback_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'multi_dimensional')
            """,
            (
                feedback.interaction_id, feedback.user_id, feedback.session_id,
                feedback.understanding, feedback.relevance, feedback.clarity,
                overall_score, feedback.emoji, emoji_score, feedback.comment
            )
        )
        
        # Update interaction with feedback dimensions JSON
        feedback_dimensions = {
            "understanding": feedback.understanding,
            "relevance": feedback.relevance,
            "clarity": feedback.clarity,
            "overall_score": overall_score,
            "emoji": feedback.emoji,
            "emoji_score": emoji_score,
            "feedback_type": "multi_dimensional"
        }
        
        db.execute_update(
            """
            UPDATE student_interactions
            SET feedback_dimensions = ?,
                feedback_score = ?,
                emoji_feedback = ?,
                emoji_feedback_timestamp = CURRENT_TIMESTAMP
            WHERE interaction_id = ?
            """,
            (json.dumps(feedback_dimensions), overall_score/5.0, feedback.emoji, feedback.interaction_id)
        )
        
        logger.info(f"Updated interaction {feedback.interaction_id} with multi-dimensional feedback")
        
        # Update multi-dimensional profile
        profile_updated = _update_profile_from_multi_feedback(
            db,
            feedback.user_id,
            feedback.session_id,
            feedback.understanding,
            feedback.relevance,
            feedback.clarity,
            overall_score
        )
        
        logger.info(f"Multi-dimensional feedback successfully recorded (entry_id: {entry_id})")
        
        return MultiFeedbackResponse(
            message="Detaylı geri bildiriminiz kaydedildi. Teşekkürler!",
            interaction_id=feedback.interaction_id,
            feedback_entry_id=entry_id,
            dimensions={
                "understanding": feedback.understanding,
                "relevance": feedback.relevance,
                "clarity": feedback.clarity
            },
            overall_score=overall_score,
            profile_updated=profile_updated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create multi-dimensional feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record detailed feedback: {str(e)}"
        )


@router.get("/multi-stats/{user_id}/{session_id}", response_model=MultiDimensionalStats)
async def get_multi_dimensional_stats(
    user_id: str,
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get multi-dimensional feedback statistics for a user/session
    
    Returns detailed analytics across Understanding, Relevance, and Clarity dimensions.
    """
    
    if not FeatureFlags.is_emoji_feedback_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback system is not enabled"
        )
    
    try:
        # Get multi-dimensional summary
        summary = db.execute_query(
            """
            SELECT * FROM multi_feedback_summary
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id)
        )
        
        if not summary:
            # No feedback yet
            return MultiDimensionalStats(
                user_id=user_id,
                session_id=session_id,
                total_feedback_count=0,
                dimension_feedback_count=0,
                emoji_only_count=0,
                understanding_distribution={},
                relevance_distribution={},
                clarity_distribution={}
            )
        
        row = summary[0]
        
        # Parse JSON distributions
        understanding_dist = {}
        relevance_dist = {}
        clarity_dist = {}
        
        try:
            if row.get('understanding_distribution'):
                understanding_dist = json.loads(row['understanding_distribution'])
            if row.get('relevance_distribution'):
                relevance_dist = json.loads(row['relevance_distribution'])
            if row.get('clarity_distribution'):
                clarity_dist = json.loads(row['clarity_distribution'])
        except:
            pass
        
        # Parse weak/strong dimensions
        weak_dims = []
        strong_dims = []
        
        try:
            if row.get('weak_dimensions'):
                weak_dims = json.loads(row['weak_dimensions'])
            if row.get('strong_dimensions'):
                strong_dims = json.loads(row['strong_dimensions'])
        except:
            pass
        
        logger.info(f"Multi-dimensional stats for user {user_id}: "
                   f"{row.get('dimension_feedback_count', 0)} detailed feedbacks")
        
        return MultiDimensionalStats(
            user_id=user_id,
            session_id=session_id,
            avg_understanding=row.get('avg_understanding'),
            avg_relevance=row.get('avg_relevance'),
            avg_clarity=row.get('avg_clarity'),
            avg_overall=row.get('avg_overall'),
            total_feedback_count=row.get('total_feedback_count', 0),
            dimension_feedback_count=row.get('dimension_feedback_count', 0),
            emoji_only_count=row.get('emoji_only_count', 0),
            understanding_distribution=understanding_dist,
            relevance_distribution=relevance_dist,
            clarity_distribution=clarity_dist,
            improvement_trend=row.get('improvement_trend', 'insufficient_data'),
            weak_dimensions=weak_dims,
            strong_dimensions=strong_dims
        )
        
    except Exception as e:
        logger.error(f"Failed to get multi-dimensional stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get multi-dimensional stats: {str(e)}"
        )


def _update_profile_from_multi_feedback(
    db: DatabaseManager,
    user_id: str,
    session_id: str,
    understanding: int,
    relevance: int,
    clarity: int,
    overall_score: float
) -> bool:
    """Update student profile based on multi-dimensional feedback"""
    try:
        # Get current profile
        profile = db.execute_query(
            "SELECT * FROM student_profiles WHERE user_id = ? AND session_id = ?",
            (user_id, session_id)
        )
        
        if profile:
            # Update existing profile with weighted averages
            row = profile[0]
            
            current_understanding = row.get('average_understanding', 3.0)
            current_satisfaction = row.get('average_satisfaction', 3.0)
            feedback_count = row.get('total_feedback_count', 0)
            
            # Update understanding (direct from understanding score)
            new_understanding = (current_understanding * feedback_count + understanding) / (feedback_count + 1)
            
            # Update satisfaction (average of relevance and clarity)
            satisfaction_score = (relevance + clarity) / 2.0
            new_satisfaction = (current_satisfaction * feedback_count + satisfaction_score) / (feedback_count + 1)
            
            db.execute_update(
                """
                UPDATE student_profiles
                SET average_understanding = ?,
                    average_satisfaction = ?,
                    total_feedback_count = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
                """,
                (new_understanding, new_satisfaction, feedback_count + 1, user_id, session_id)
            )
            
            logger.debug(f"Updated profile for user {user_id}: understanding={new_understanding:.2f}, "
                        f"satisfaction={new_satisfaction:.2f}")
            return True
            
        else:
            # Create new profile
            satisfaction_score = (relevance + clarity) / 2.0
            
            db.execute_insert(
                """
                INSERT INTO student_profiles
                (user_id, session_id, average_understanding, average_satisfaction,
                 total_interactions, total_feedback_count, last_updated)
                VALUES (?, ?, ?, ?, 1, 1, CURRENT_TIMESTAMP)
                """,
                (user_id, session_id, understanding, satisfaction_score)
            )
            
            logger.debug(f"Created profile for user {user_id} with multi-dimensional data")
            return True
            
    except Exception as e:
        logger.warning(f"Failed to update profile from multi-dimensional feedback: {e}")
        return False


def _update_global_score(db: DatabaseManager, doc_id: str, emoji_score: float, emoji: str):
    """Update global document score with emoji feedback"""
    try:
        # Check if document exists in global scores
        existing = db.execute_query(
            "SELECT * FROM document_global_scores WHERE doc_id = ?",
            (doc_id,)
        )
        
        if existing:
            # Update existing
            row = existing[0]
            total = row['total_feedback_count'] + 1
            
            # Update positive/negative counts
            if emoji_score >= 0.7:
                positive = row['positive_feedback_count'] + 1
                negative = row['negative_feedback_count']
            elif emoji_score <= 0.2:
                positive = row['positive_feedback_count']
                negative = row['negative_feedback_count'] + 1
            else:
                positive = row['positive_feedback_count']
                negative = row['negative_feedback_count']
            
            # Update avg emoji score
            current_avg = row.get('avg_emoji_score', 0.5)
            new_avg = (current_avg * (total - 1) + emoji_score) / total
            
            db.execute_update(
                """
                UPDATE document_global_scores
                SET total_feedback_count = ?,
                    positive_feedback_count = ?,
                    negative_feedback_count = ?,
                    avg_emoji_score = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE doc_id = ?
                """,
                (total, positive, negative, new_avg, doc_id)
            )
        else:
            # Insert new
            positive = 1 if emoji_score >= 0.7 else 0
            negative = 1 if emoji_score <= 0.2 else 0
            
            db.execute_insert(
                """
                INSERT INTO document_global_scores
                (doc_id, total_feedback_count, positive_feedback_count, 
                 negative_feedback_count, avg_emoji_score, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (doc_id, 1, positive, negative, emoji_score)
            )
        
        logger.debug(f"Updated global score for doc {doc_id}")
        
    except Exception as e:
        logger.warning(f"Failed to update global score for doc {doc_id}: {e}")


def _update_profile_from_emoji(
    db: DatabaseManager,
    user_id: str,
    session_id: str,
    emoji_score: float,
    emoji: str
) -> bool:
    """Update student profile based on emoji feedback (real-time)"""
    try:
        # Get current profile
        profile = db.execute_query(
            "SELECT * FROM student_profiles WHERE user_id = ? AND session_id = ?",
            (user_id, session_id)
        )
        
        if profile:
            # Update existing profile
            row = profile[0]
            
            # Update average understanding
            current_avg = row.get('average_understanding', 3.0)
            feedback_count = row.get('total_feedback_count', 0)
            
            # Convert emoji_score (0-1) to 1-5 scale
            understanding_score = 1 + (emoji_score * 4)
            
            new_avg = (current_avg * feedback_count + understanding_score) / (feedback_count + 1)
            new_count = feedback_count + 1
            
            # Update satisfaction similarly
            current_sat = row.get('average_satisfaction', 3.0)
            satisfaction_score = 1 + (emoji_score * 4)
            new_sat = (current_sat * feedback_count + satisfaction_score) / (feedback_count + 1)
            
            db.execute_update(
                """
                UPDATE student_profiles
                SET average_understanding = ?,
                    average_satisfaction = ?,
                    total_feedback_count = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
                """,
                (new_avg, new_sat, new_count, user_id, session_id)
            )
            
            logger.debug(f"Updated profile for user {user_id}: avg understanding {new_avg:.2f}")
            return True
        else:
            # Create new profile
            understanding_score = 1 + (emoji_score * 4)
            
            db.execute_insert(
                """
                INSERT INTO student_profiles
                (user_id, session_id, average_understanding, average_satisfaction,
                 total_interactions, total_feedback_count, last_updated)
                VALUES (?, ?, ?, ?, 1, 1, CURRENT_TIMESTAMP)
                """,
                (user_id, session_id, understanding_score, understanding_score)
            )
            
            logger.debug(f"Created profile for user {user_id}")
            return True
            
    except Exception as e:
        logger.warning(f"Failed to update profile from emoji: {e}")
        return False


def _update_emoji_summary(
    db: DatabaseManager,
    user_id: str,
    session_id: str,
    emoji: str,
    emoji_score: float
):
    """Update emoji summary table for analytics"""
    try:
        # Check if entry exists
        existing = db.execute_query(
            """
            SELECT * FROM emoji_feedback_summary
            WHERE user_id = ? AND session_id = ? AND emoji = ?
            """,
            (user_id, session_id, emoji)
        )
        
        if existing:
            # Update existing
            row = existing[0]
            new_count = row['emoji_count'] + 1
            current_avg = row['avg_score']
            new_avg = (current_avg * (new_count - 1) + emoji_score) / new_count
            
            db.execute_update(
                """
                UPDATE emoji_feedback_summary
                SET emoji_count = ?,
                    avg_score = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ? AND emoji = ?
                """,
                (new_count, new_avg, user_id, session_id, emoji)
            )
        else:
            # Insert new
            db.execute_insert(
                """
                INSERT INTO emoji_feedback_summary
                (user_id, session_id, emoji, emoji_count, avg_score, last_updated)
                VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, session_id, emoji, emoji_score)
            )
        
        logger.debug(f"Updated emoji summary for {user_id}: {emoji}")
        
    except Exception as e:
        logger.warning(f"Failed to update emoji summary: {e}")

