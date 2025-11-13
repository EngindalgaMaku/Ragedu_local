"""
Student profile management endpoints
Manages personalized learning profiles
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import json

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


class ProfileResponse(BaseModel):
    """Response model for student profile"""
    user_id: str
    session_id: str
    average_understanding: Optional[float]
    average_satisfaction: Optional[float]
    total_interactions: int
    total_feedback_count: int
    strong_topics: Optional[Dict[str, Any]]
    weak_topics: Optional[Dict[str, Any]]
    preferred_explanation_style: Optional[str]
    preferred_difficulty_level: Optional[str]


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


@router.get("/{user_id}")
async def get_profile(
    user_id: str,
    session_id: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get student profile
    
    Args:
        user_id: User ID
        session_id: Optional session ID filter
    """
    try:
        if session_id:
            query = """
                SELECT * FROM student_profiles 
                WHERE user_id = ? AND session_id = ?
            """
            params = (user_id, session_id)
        else:
            # Get most recent profile
            query = """
                SELECT * FROM student_profiles 
                WHERE user_id = ?
                ORDER BY last_updated DESC
                LIMIT 1
            """
            params = (user_id,)
        
        results = db.execute_query(query, params)
        
        if not results:
            # Return default profile
            return ProfileResponse(
                user_id=user_id,
                session_id=session_id or "",
                average_understanding=None,
                average_satisfaction=None,
                total_interactions=0,
                total_feedback_count=0,
                strong_topics=None,
                weak_topics=None,
                preferred_explanation_style=None,
                preferred_difficulty_level=None
            )
        
        profile = results[0]
        
        # Parse JSON fields
        strong_topics = None
        weak_topics = None
        if profile.get("strong_topics"):
            try:
                strong_topics = json.loads(profile["strong_topics"])
            except:
                strong_topics = None
        if profile.get("weak_topics"):
            try:
                weak_topics = json.loads(profile["weak_topics"])
            except:
                weak_topics = None
        
        return ProfileResponse(
            user_id=profile["user_id"],
            session_id=profile["session_id"],
            average_understanding=float(profile["average_understanding"]) if profile.get("average_understanding") else None,
            average_satisfaction=float(profile["average_satisfaction"]) if profile.get("average_satisfaction") else None,
            total_interactions=profile.get("total_interactions", 0) or 0,
            total_feedback_count=profile.get("total_feedback_count", 0) or 0,
            strong_topics=strong_topics,
            weak_topics=weak_topics,
            preferred_explanation_style=profile.get("preferred_explanation_style"),
            preferred_difficulty_level=profile.get("preferred_difficulty_level")
        )
        
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get profile: {str(e)}"
        )


@router.get("/{user_id}/{session_id}")
async def get_session_profile(
    user_id: str,
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get student profile for a specific session
    """
    return await get_profile(user_id, session_id, db)


@router.put("/{user_id}/{session_id}")
async def update_profile(
    user_id: str,
    session_id: str,
    profile_data: Dict[str, Any],
    db: DatabaseManager = Depends(get_db)
):
    """
    Update student profile settings
    """
    try:
        # Check if profile exists
        existing = db.execute_query(
            "SELECT profile_id FROM student_profiles WHERE user_id = ? AND session_id = ?",
            (user_id, session_id)
        )
        
        # Prepare update data
        update_fields = []
        update_values = []
        
        if "preferred_explanation_style" in profile_data:
            update_fields.append("preferred_explanation_style = ?")
            update_values.append(profile_data["preferred_explanation_style"])
        
        if "preferred_difficulty_level" in profile_data:
            update_fields.append("preferred_difficulty_level = ?")
            update_values.append(profile_data["preferred_difficulty_level"])
        
        if "strong_topics" in profile_data:
            update_fields.append("strong_topics = ?")
            update_values.append(json.dumps(profile_data["strong_topics"]))
        
        if "weak_topics" in profile_data:
            update_fields.append("weak_topics = ?")
            update_values.append(json.dumps(profile_data["weak_topics"]))
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        
        update_fields.append("last_updated = CURRENT_TIMESTAMP")
        update_values.extend([user_id, session_id])
        
        if existing:
            # Update existing profile
            query = f"""
                UPDATE student_profiles 
                SET {', '.join(update_fields)}
                WHERE user_id = ? AND session_id = ?
            """
            db.execute_update(query, tuple(update_values))
        else:
            # Create new profile
            query = """
                INSERT INTO student_profiles 
                (user_id, session_id, preferred_explanation_style, preferred_difficulty_level, strong_topics, weak_topics)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            db.execute_insert(
                query,
                (
                    user_id,
                    session_id,
                    profile_data.get("preferred_explanation_style"),
                    profile_data.get("preferred_difficulty_level"),
                    json.dumps(profile_data.get("strong_topics")) if profile_data.get("strong_topics") else None,
                    json.dumps(profile_data.get("weak_topics")) if profile_data.get("weak_topics") else None,
                )
            )
        
        return {"message": "Profile updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update profile: {str(e)}"
        )

