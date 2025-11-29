"""
Session Settings API Endpoints
Allows teachers to control educational features per session
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import requests
import os
from datetime import datetime

# Import database and dependencies
try:
    from database.database import DatabaseManager
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from database.database import DatabaseManager

db_manager = None
logger = logging.getLogger(__name__)
router = APIRouter()

# API Gateway URL for fetching session metadata
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api-gateway:8001")


class SessionSettings(BaseModel):
    """Session settings model"""
    session_id: str
    user_id: str
    
    # Main Educational Features
    enable_progressive_assessment: bool = Field(False, description="Progressive Assessment Flow")
    enable_personalized_responses: bool = Field(False, description="AI-Powered Response Personalization")
    enable_multi_dimensional_feedback: bool = Field(False, description="Multi-dimensional Student Feedback")
    enable_topic_analytics: bool = Field(True, description="Topic-Based Learning Analytics")
    
    # Advanced APRAG Component Settings
    enable_cacs: bool = Field(True, description="CACS Document Scoring")
    enable_zpd: bool = Field(True, description="Zone of Proximal Development")
    enable_bloom: bool = Field(True, description="Bloom Taxonomy Detection")
    enable_cognitive_load: bool = Field(True, description="Cognitive Load Management")
    enable_emoji_feedback: bool = Field(True, description="Emoji-Based Feedback")
    enable_ebars: bool = Field(False, description="EBARS - Emoji-Based Adaptive Response System")


class SessionSettingsUpdate(BaseModel):
    """Session settings update request"""
    enable_progressive_assessment: Optional[bool] = None
    enable_personalized_responses: Optional[bool] = None
    enable_multi_dimensional_feedback: Optional[bool] = None
    enable_topic_analytics: Optional[bool] = None
    enable_cacs: Optional[bool] = None
    enable_zpd: Optional[bool] = None
    enable_bloom: Optional[bool] = None
    enable_cognitive_load: Optional[bool] = None
    enable_emoji_feedback: Optional[bool] = None
    enable_ebars: Optional[bool] = None


class SessionSettingsResponse(BaseModel):
    """Session settings response"""
    success: bool
    message: str
    settings: SessionSettings
    updated_at: str


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


@router.get("/{session_id}", response_model=SessionSettingsResponse)
async def get_session_settings(
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get session settings for a specific session
    
    Returns current settings or creates defaults if none exist.
    """
    
    try:
        logger.info(f"Getting session settings for session: {session_id}")
        
        # Get existing settings
        result = db.execute_query(
            "SELECT * FROM session_settings WHERE session_id = ?",
            (session_id,)
        )
        
        if result:
            settings_data = result[0]
            settings = SessionSettings(
                session_id=settings_data['session_id'],
                user_id=settings_data['user_id'],
                enable_progressive_assessment=bool(settings_data['enable_progressive_assessment']),
                enable_personalized_responses=bool(settings_data['enable_personalized_responses']),
                enable_multi_dimensional_feedback=bool(settings_data['enable_multi_dimensional_feedback']),
                enable_topic_analytics=bool(settings_data['enable_topic_analytics']),
                enable_cacs=bool(settings_data['enable_cacs']),
                enable_zpd=bool(settings_data['enable_zpd']),
                enable_bloom=bool(settings_data['enable_bloom']),
                enable_cognitive_load=bool(settings_data['enable_cognitive_load']),
                enable_emoji_feedback=bool(settings_data['enable_emoji_feedback']),
                enable_ebars=bool(settings_data.get('enable_ebars', False))
            )
            
            return SessionSettingsResponse(
                success=True,
                message="Session settings retrieved successfully",
                settings=settings,
                updated_at=settings_data['updated_at']
            )
        else:
            # Create default settings for session
            # Get the session owner (teacher) from API Gateway
            # Do NOT use student_interactions user_id as that would be a student, not the teacher
            default_user_id = None
            candidate_user_id = None
            
            try:
                # Fetch session metadata from API Gateway to get the teacher (created_by)
                session_response = requests.get(
                    f"{API_GATEWAY_URL}/sessions/{session_id}",
                    timeout=5
                )
                if session_response.status_code == 200:
                    session_data = session_response.json()
                    created_by = session_data.get('created_by')
                    if created_by:
                        candidate_user_id = created_by
                        logger.info(f"Found session owner (teacher) from API Gateway: {created_by}")
            except Exception as e:
                logger.warning(f"Could not fetch session metadata from API Gateway: {e}")
            
            # Verify that the candidate user_id exists in users table (FOREIGN KEY constraint)
            if candidate_user_id:
                try:
                    user_check = db.execute_query(
                        "SELECT username FROM users WHERE username = ? LIMIT 1",
                        (candidate_user_id,)
                    )
                    if user_check and len(user_check) > 0:
                        default_user_id = candidate_user_id
                        logger.info(f"Verified user exists in users table: {default_user_id}")
                    else:
                        logger.warning(f"User '{candidate_user_id}' from session not found in users table, will use admin")
                except Exception as e:
                    logger.warning(f"Could not verify user in users table: {e}")
            
            # Fallback: Use admin username
            # Note: We don't check users table because it's in auth-service, not aprag-service
            # The user_id is just stored as a string identifier for the teacher who manages the settings
            if not default_user_id:
                default_user_id = "admin"
                logger.info(f"Using 'admin' as fallback for session {session_id}")
            
            # Note: We don't verify user_id in users table anymore because:
            # 1. users table is in auth-service, not aprag-service
            # 2. FOREIGN KEY constraint has been removed from session_settings table
            # 3. We still try to get valid user_id from session metadata or use admin
            
            # Create default settings
            # First check if enable_ebars column exists, if not add it
            try:
                with db.get_connection() as conn:
                    cursor = conn.execute("PRAGMA table_info(session_settings)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if 'enable_ebars' not in columns:
                        conn.execute("ALTER TABLE session_settings ADD COLUMN enable_ebars BOOLEAN DEFAULT 0")
                        conn.commit()
                        logger.info("Added enable_ebars column to session_settings table")
            except Exception as e:
                logger.warning(f"Could not add enable_ebars column: {e}")
            
            db.execute_insert(
                """
                INSERT INTO session_settings 
                (session_id, user_id, enable_progressive_assessment, enable_personalized_responses,
                 enable_multi_dimensional_feedback, enable_topic_analytics, enable_cacs, enable_zpd,
                 enable_bloom, enable_cognitive_load, enable_emoji_feedback, enable_ebars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, default_user_id, False, False, False, True, True, True, True, True, True, False)
            )
            
            settings = SessionSettings(
                session_id=session_id,
                user_id=default_user_id,
                enable_progressive_assessment=False,
                enable_personalized_responses=False,
                enable_multi_dimensional_feedback=False,
                enable_topic_analytics=True,
                enable_cacs=True,
                enable_zpd=True,
                enable_bloom=True,
                enable_cognitive_load=True,
                enable_emoji_feedback=True,
                enable_ebars=False
            )
            
            return SessionSettingsResponse(
                success=True,
                message="Default session settings created",
                settings=settings,
                updated_at=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Failed to get session settings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session settings: {str(e)}"
        )


@router.put("/{session_id}", response_model=SessionSettingsResponse)
async def update_session_settings(
    session_id: str,
    updates: SessionSettingsUpdate,
    user_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Update session settings for a specific session
    
    Only updates the fields that are provided in the request.
    """
    
    try:
        logger.info(f"Updating session settings for session: {session_id} by user: {user_id}")
        
        # Get current settings first
        current_result = db.execute_query(
            "SELECT * FROM session_settings WHERE session_id = ?",
            (session_id,)
        )
        
        if current_result:
            # Update existing settings
            current = current_result[0]
            
            # Build update query dynamically based on provided fields
            update_fields = []
            update_values = []
            
            if updates.enable_progressive_assessment is not None:
                update_fields.append("enable_progressive_assessment = ?")
                update_values.append(updates.enable_progressive_assessment)
                
            if updates.enable_personalized_responses is not None:
                update_fields.append("enable_personalized_responses = ?")
                update_values.append(updates.enable_personalized_responses)
                
            if updates.enable_multi_dimensional_feedback is not None:
                update_fields.append("enable_multi_dimensional_feedback = ?")
                update_values.append(updates.enable_multi_dimensional_feedback)
                
            if updates.enable_topic_analytics is not None:
                update_fields.append("enable_topic_analytics = ?")
                update_values.append(updates.enable_topic_analytics)
                
            if updates.enable_cacs is not None:
                update_fields.append("enable_cacs = ?")
                update_values.append(updates.enable_cacs)
                
            if updates.enable_zpd is not None:
                update_fields.append("enable_zpd = ?")
                update_values.append(updates.enable_zpd)
                
            if updates.enable_bloom is not None:
                update_fields.append("enable_bloom = ?")
                update_values.append(updates.enable_bloom)
                
            if updates.enable_cognitive_load is not None:
                update_fields.append("enable_cognitive_load = ?")
                update_values.append(updates.enable_cognitive_load)
                
            if updates.enable_emoji_feedback is not None:
                update_fields.append("enable_emoji_feedback = ?")
                update_values.append(updates.enable_emoji_feedback)
            
            if updates.enable_ebars is not None:
                # Ensure column exists
                try:
                    with db.get_connection() as conn:
                        cursor = conn.execute("PRAGMA table_info(session_settings)")
                        columns = [row[1] for row in cursor.fetchall()]
                        if 'enable_ebars' not in columns:
                            conn.execute("ALTER TABLE session_settings ADD COLUMN enable_ebars BOOLEAN DEFAULT 0")
                            conn.commit()
                            logger.info("Added enable_ebars column to session_settings table")
                except Exception as e:
                    logger.warning(f"Could not add enable_ebars column: {e}")
                
                update_fields.append("enable_ebars = ?")
                update_values.append(updates.enable_ebars)
            
            if not update_fields:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No fields to update"
                )
            
            # Add updated_at and user_id to the update
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            update_fields.append("user_id = ?")
            update_values.append(user_id)
            update_values.append(session_id)  # For WHERE clause
            
            update_query = f"""
                UPDATE session_settings 
                SET {', '.join(update_fields)}
                WHERE session_id = ?
            """
            
            db.execute_update(update_query, tuple(update_values))
            
        else:
            # Create new settings with provided updates
            # Ensure enable_ebars column exists
            try:
                with db.get_connection() as conn:
                    cursor = conn.execute("PRAGMA table_info(session_settings)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if 'enable_ebars' not in columns:
                        conn.execute("ALTER TABLE session_settings ADD COLUMN enable_ebars BOOLEAN DEFAULT 0")
                        conn.commit()
                        logger.info("Added enable_ebars column to session_settings table")
            except Exception as e:
                logger.warning(f"Could not add enable_ebars column: {e}")
            
            db.execute_insert(
                """
                INSERT INTO session_settings 
                (session_id, user_id, enable_progressive_assessment, enable_personalized_responses,
                 enable_multi_dimensional_feedback, enable_topic_analytics, enable_cacs, enable_zpd,
                 enable_bloom, enable_cognitive_load, enable_emoji_feedback, enable_ebars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, 
                    user_id,
                    updates.enable_progressive_assessment or False,
                    updates.enable_personalized_responses or False,
                    updates.enable_multi_dimensional_feedback or False,
                    updates.enable_topic_analytics if updates.enable_topic_analytics is not None else True,
                    updates.enable_cacs if updates.enable_cacs is not None else True,
                    updates.enable_zpd if updates.enable_zpd is not None else True,
                    updates.enable_bloom if updates.enable_bloom is not None else True,
                    updates.enable_cognitive_load if updates.enable_cognitive_load is not None else True,
                    updates.enable_emoji_feedback if updates.enable_emoji_feedback is not None else True,
                    updates.enable_ebars if updates.enable_ebars is not None else False
                )
            )
        
        # Return updated settings
        return await get_session_settings(session_id, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session settings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update session settings: {str(e)}"
        )


@router.post("/{session_id}/reset", response_model=SessionSettingsResponse)
async def reset_session_settings(
    session_id: str,
    user_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Reset session settings to defaults
    
    This will reset all educational features to their default state.
    """
    
    try:
        logger.info(f"Resetting session settings for session: {session_id} by user: {user_id}")
        
        # Delete existing settings
        db.execute_update(
            "DELETE FROM session_settings WHERE session_id = ?",
            (session_id,)
        )
        
        # Ensure enable_ebars column exists
        try:
            with db.get_connection() as conn:
                cursor = conn.execute("PRAGMA table_info(session_settings)")
                columns = [row[1] for row in cursor.fetchall()]
                if 'enable_ebars' not in columns:
                    conn.execute("ALTER TABLE session_settings ADD COLUMN enable_ebars BOOLEAN DEFAULT 0")
                    conn.commit()
                    logger.info("Added enable_ebars column to session_settings table")
        except Exception as e:
            logger.warning(f"Could not add enable_ebars column: {e}")
        
        # Create default settings
        db.execute_insert(
            """
            INSERT INTO session_settings 
            (session_id, user_id, enable_progressive_assessment, enable_personalized_responses,
             enable_multi_dimensional_feedback, enable_topic_analytics, enable_cacs, enable_zpd,
             enable_bloom, enable_cognitive_load, enable_emoji_feedback, enable_ebars)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, False, False, False, True, True, True, True, True, True, False)
        )
        
        # Return new default settings
        return await get_session_settings(session_id, db)
        
    except Exception as e:
        logger.error(f"Failed to reset session settings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset session settings: {str(e)}"
        )


@router.get("/{session_id}/presets")
async def get_settings_presets():
    """
    Get predefined settings presets for teachers
    
    Returns common configuration presets for different teaching scenarios.
    """
    
    presets = {
        "conservative": {
            "name": "Muhafazakar",
            "description": "Temel özellikler, minimum AI müdahalesi",
            "settings": {
                "enable_progressive_assessment": False,
                "enable_personalized_responses": False,
                "enable_multi_dimensional_feedback": False,
                "enable_topic_analytics": True,
                "enable_cacs": True,
                "enable_zpd": False,
                "enable_bloom": False,
                "enable_cognitive_load": False,
                "enable_emoji_feedback": True,
                "enable_ebars": False
            }
        },
        "balanced": {
            "name": "Dengeli",
            "description": "Temel kişiselleştirme ile dengeli yaklaşım",
            "settings": {
                "enable_progressive_assessment": False,
                "enable_personalized_responses": True,
                "enable_multi_dimensional_feedback": False,
                "enable_topic_analytics": True,
                "enable_cacs": True,
                "enable_zpd": True,
                "enable_bloom": True,
                "enable_cognitive_load": True,
                "enable_emoji_feedback": True,
                "enable_ebars": False
            }
        },
        "advanced": {
            "name": "İleri Düzey",
            "description": "Tüm AI özellikler ve ilerici değerlendirme",
            "settings": {
                "enable_progressive_assessment": True,
                "enable_personalized_responses": True,
                "enable_multi_dimensional_feedback": True,
                "enable_topic_analytics": True,
                "enable_cacs": True,
                "enable_zpd": True,
                "enable_bloom": True,
                "enable_cognitive_load": True,
                "enable_emoji_feedback": True,
                "enable_ebars": True
            }
        }
    }
    
    return {
        "success": True,
        "presets": presets
    }


@router.post("/{session_id}/apply-preset/{preset_name}", response_model=SessionSettingsResponse)
async def apply_settings_preset(
    session_id: str,
    preset_name: str,
    user_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Apply a predefined settings preset
    
    Available presets: conservative, balanced, advanced
    """
    
    try:
        logger.info(f"Applying preset '{preset_name}' to session: {session_id} by user: {user_id}")
        
        # Get presets
        presets_response = await get_settings_presets()
        presets = presets_response["presets"]
        
        if preset_name not in presets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Preset '{preset_name}' not found. Available: {list(presets.keys())}"
            )
        
        preset_settings = presets[preset_name]["settings"]
        
        # Convert to update object
        updates = SessionSettingsUpdate(**preset_settings)
        
        # Apply the preset
        return await update_session_settings(session_id, updates, user_id, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply preset: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply preset: {str(e)}"
        )