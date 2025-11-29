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
        # Ensure user_id is always a string (FastAPI might pass it as int from path)
        user_id = str(user_id)
        if session_id:
            session_id = str(session_id)
        
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
            # Auto-create profile if it doesn't exist
            logger.info(f"Profile not found for user {user_id}, session {session_id}, creating default profile...")
            try:
                db.execute_insert(
                    """
                    INSERT INTO student_profiles
                    (user_id, session_id, average_understanding, average_satisfaction,
                     total_interactions, total_feedback_count, last_updated, created_at)
                    VALUES (?, ?, 3.0, NULL, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (user_id, session_id or "")
                )
                logger.info(f"✅ Created default profile for user {user_id}, session {session_id}")
                # Return the newly created profile
                return ProfileResponse(
                    user_id=user_id,
                    session_id=session_id or "",
                    average_understanding=3.0,
                    average_satisfaction=None,  # NULL until multi-dimensional feedback is provided
                    total_interactions=0,
                    total_feedback_count=0,
                    strong_topics=None,
                    weak_topics=None,
                    preferred_explanation_style=None,
                    preferred_difficulty_level=None
                )
            except Exception as create_error:
                logger.warning(f"Failed to auto-create profile: {create_error}")
                # Return default profile if creation fails
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
        
        # Ensure user_id and session_id are strings (not None)
        profile_user_id = str(profile.get("user_id") or user_id)
        profile_session_id = str(profile.get("session_id") or session_id or "")
        
        # Parse JSON fields
        strong_topics = None
        weak_topics = None
        if profile.get("strong_topics"):
            try:
                if isinstance(profile["strong_topics"], str):
                    strong_topics = json.loads(profile["strong_topics"])
                else:
                    strong_topics = profile["strong_topics"]
            except:
                strong_topics = None
        if profile.get("weak_topics"):
            try:
                if isinstance(profile["weak_topics"], str):
                    weak_topics = json.loads(profile["weak_topics"])
                else:
                    weak_topics = profile["weak_topics"]
            except:
                weak_topics = None
        
        return ProfileResponse(
            user_id=profile_user_id,
            session_id=profile_session_id,
            average_understanding=float(profile["average_understanding"]) if profile.get("average_understanding") is not None else None,
            average_satisfaction=float(profile["average_satisfaction"]) if profile.get("average_satisfaction") is not None else None,
            total_interactions=int(profile.get("total_interactions", 0) or 0),
            total_feedback_count=int(profile.get("total_feedback_count", 0) or 0),
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
    # Ensure both are strings (FastAPI might pass them as int from path)
    user_id = str(user_id)
    session_id = str(session_id)
    return await get_profile(user_id, session_id, db)


@router.post("/{user_id}/{session_id}/reset")
async def reset_profile(
    user_id: str,
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Reset student profile and learning parameters to defaults
    This will reset the profile statistics but keep the profile record
    """
    try:
        # Ensure both are strings
        user_id = str(user_id)
        session_id = str(session_id)
        
        # Reset profile to defaults
        db.execute_update(
            """
            UPDATE student_profiles
            SET average_understanding = 3.0,
                average_satisfaction = 3.0,
                total_interactions = 0,
                total_feedback_count = 0,
                strong_topics = NULL,
                weak_topics = NULL,
                preferred_explanation_style = NULL,
                preferred_difficulty_level = NULL,
                last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id)
        )
        
        # Delete all interactions for this user and session
        db.execute_update(
            """
            DELETE FROM student_interactions
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id)
        )
        
        logger.info(f"Reset profile for user {user_id}, session {session_id}")
        
        return {
            "message": "Profile reset successfully",
            "user_id": user_id,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Failed to reset profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset profile: {str(e)}"
        )


@router.get("/{user_id}/{session_id}/pedagogical-state")
async def get_pedagogical_state(
    user_id: str,
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get student's current pedagogical state (ZPD, Bloom, Cognitive Load)
    This is used to display the student's current learning parameters before making a query.
    """
    try:
        # Ensure both are strings
        user_id = str(user_id)
        session_id = str(session_id)
        
        # Get profile
        profile = await get_profile(user_id, session_id, db)
        profile_dict = {
            "user_id": profile.user_id,
            "session_id": profile.session_id,
            "average_understanding": profile.average_understanding,
            "average_satisfaction": profile.average_satisfaction,
            "total_interactions": profile.total_interactions,
            "total_feedback_count": profile.total_feedback_count,
            "strong_topics": profile.strong_topics,
            "weak_topics": profile.weak_topics,
            "preferred_explanation_style": profile.preferred_explanation_style,
            "preferred_difficulty_level": profile.preferred_difficulty_level,
        }
        
        # Get recent interactions (last 20)
        # Note: student_interactions table doesn't have 'response' column
        # We'll get query and other available fields
        query = """
            SELECT interaction_id, query, feedback_type, feedback_value, 
                   pedagogical_context, timestamp as created_at
            FROM student_interactions
            WHERE user_id = ? AND session_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        """
        recent_interactions = db.execute_query(query, (user_id, session_id))
        
        # Import pedagogical calculators
        from business_logic.pedagogical import (
            get_zpd_calculator,
            get_bloom_detector,
            get_cognitive_load_manager
        )
        
        # Calculate ZPD
        zpd_calculator = get_zpd_calculator()
        zpd_info = zpd_calculator.calculate_zpd_level(recent_interactions, profile_dict)
        
        # Calculate Bloom (from last interaction if available)
        bloom_detector = get_bloom_detector()
        if recent_interactions and recent_interactions[0].get("query"):
            bloom_info = bloom_detector.detect_bloom_level(recent_interactions[0]["query"])
        else:
            bloom_info = {
                "level": "remember",
                "level_index": 1,
                "confidence": 0.0
            }
        
        # Calculate Cognitive Load
        # Since we don't have response in student_interactions, use default values
        cognitive_load_manager = get_cognitive_load_manager()
        # Try to get response from student_feedback or use default
        cognitive_load_info = {
            "total_load": 0.5,
            "needs_simplification": False
        }
        # If we have pedagogical_context, we might extract cognitive load info from there
        if recent_interactions and recent_interactions[0].get("pedagogical_context"):
            try:
                ped_context = recent_interactions[0]["pedagogical_context"]
                if isinstance(ped_context, str):
                    import json
                    ped_context = json.loads(ped_context)
                if isinstance(ped_context, dict) and "cognitive_load" in ped_context:
                    cognitive_load_info["total_load"] = ped_context.get("cognitive_load", 0.5)
                    cognitive_load_info["needs_simplification"] = ped_context.get("needs_simplification", False)
            except:
                pass
        
        # Get personalization factors from profile
        personalization_factors = {
            "understanding_level": "intermediate" if profile.average_understanding is None else (
                "beginner" if profile.average_understanding < 2.0 else
                "elementary" if profile.average_understanding < 3.0 else
                "intermediate" if profile.average_understanding < 4.0 else
                "advanced" if profile.average_understanding < 4.5 else "expert"
            ),
            "difficulty_level": profile.preferred_difficulty_level or "intermediate",
            "explanation_style": profile.preferred_explanation_style or "balanced"
        }
        
        # Ensure profile stats have default values if None
        profile_stats = {
            "total_interactions": profile.total_interactions or 0,
            "total_feedback_count": profile.total_feedback_count or 0,
            "average_understanding": profile.average_understanding if profile.average_understanding is not None else 3.0,
            "average_satisfaction": profile.average_satisfaction if profile.average_satisfaction is not None else 3.0,
        }
        
        return {
            "user_id": user_id,
            "session_id": session_id,
            "zpd": zpd_info,
            "bloom": bloom_info,
            "cognitive_load": cognitive_load_info,
            "personalization_factors": personalization_factors,
            "profile_stats": profile_stats,
            "last_updated": recent_interactions[0].get("created_at") if recent_interactions else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get pedagogical state: {e}")
        # Return defaults
        return {
            "user_id": user_id,
            "session_id": session_id,
            "zpd": {
                "current_level": "intermediate",
                "recommended_level": "intermediate",
                "success_rate": 0.5,
                "level_index": 2
            },
            "bloom": {
                "level": "remember",
                "level_index": 1,
                "confidence": 0.0
            },
            "cognitive_load": {
                "total_load": 0.5,
                "needs_simplification": False
            },
            "personalization_factors": {
                "understanding_level": "intermediate",
                "difficulty_level": "intermediate",
                "explanation_style": "balanced"
            },
            "profile_stats": {
                "total_interactions": 0,
                "total_feedback_count": 0,
                "average_understanding": 3.0,
                "average_satisfaction": 3.0,
            },
            "last_updated": None
        }


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

