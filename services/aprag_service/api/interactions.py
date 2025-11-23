"""
Interaction logging endpoints
Records student queries and responses
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
import httpx
import os
from datetime import datetime

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


class InteractionCreate(BaseModel):
    """Request model for creating an interaction"""
    user_id: str
    session_id: str
    query: str
    response: str
    personalized_response: Optional[str] = None
    processing_time_ms: Optional[int] = None
    model_used: Optional[str] = None
    chain_type: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class InteractionResponse(BaseModel):
    """Response model for interaction"""
    interaction_id: int
    user_id: str
    session_id: str
    query: str
    original_response: str
    personalized_response: Optional[str]
    timestamp: str
    processing_time_ms: Optional[int]
    model_used: Optional[str]
    chain_type: Optional[str]


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


async def get_user_info_from_auth_service(user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch user information from Auth service for given user IDs
    
    Args:
        user_ids: List of user IDs to fetch
        
    Returns:
        Dictionary mapping user_id -> user_info
    """
    if not user_ids:
        return {}
    
    try:
        # Auth service URL from environment or default
        auth_service_url = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
        
        user_info_map = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for user_id in user_ids:
                try:
                    # Try to get user by username first (most common case)
                    response = await client.get(f"{auth_service_url}/users/by-username/{user_id}")
                    
                    if response.status_code == 200:
                        user_data = response.json()
                        user_info_map[user_id] = {
                            "id": user_data.get("id"),
                            "username": user_data.get("username"),
                            "first_name": user_data.get("first_name", ""),
                            "last_name": user_data.get("last_name", ""),
                            "student_name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip()
                        }
                    elif user_id.isdigit():
                        # Fallback: try by user ID if username lookup failed
                        response = await client.get(f"{auth_service_url}/users/{user_id}")
                        
                        if response.status_code == 200:
                            user_data = response.json()
                            user_info_map[user_id] = {
                                "id": user_data.get("id"),
                                "username": user_data.get("username"),
                                "first_name": user_data.get("first_name", ""),
                                "last_name": user_data.get("last_name", ""),
                                "student_name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip()
                            }
                        else:
                            logger.warning(f"User not found in auth service: {user_id}")
                            user_info_map[user_id] = None
                    else:
                        logger.warning(f"User not found in auth service: {user_id}")
                        user_info_map[user_id] = None
                        
                except httpx.RequestError as e:
                    logger.error(f"Failed to fetch user {user_id} from auth service: {e}")
                    user_info_map[user_id] = None
                    
    except Exception as e:
        logger.error(f"Failed to connect to auth service: {e}")
        return {}
    
    return user_info_map


@router.post("", status_code=201)
async def create_interaction(interaction: InteractionCreate, db: DatabaseManager = Depends(get_db)):
    """
    Create a new student interaction record
    
    This endpoint is called after a RAG query is processed
    to log the interaction for learning and personalization.
    """
    try:
        logger.info(f"Logging interaction for user {interaction.user_id}, session {interaction.session_id}")
        
        # Prepare sources as JSON string
        sources_json = json.dumps(interaction.sources) if interaction.sources else None
        
        # Prepare metadata as JSON string
        metadata_json = json.dumps(interaction.metadata) if interaction.metadata else None
        
        # Insert interaction into database
        query = """
            INSERT INTO student_interactions 
            (user_id, session_id, query, original_response, personalized_response,
             processing_time_ms, model_used, chain_type, sources, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        interaction_id = db.execute_insert(
            query,
            (
                interaction.user_id,
                interaction.session_id,
                interaction.query,
                interaction.response,
                interaction.personalized_response,
                interaction.processing_time_ms,
                interaction.model_used,
                interaction.chain_type,
                sources_json,
                metadata_json
            )
        )
        
        logger.info(f"Successfully logged interaction {interaction_id} for user {interaction.user_id}")
        
        return {
            "interaction_id": interaction_id,
            "message": "Interaction logged successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to log interaction: {str(e)}"
        )


@router.get("/{user_id}")
async def get_user_interactions(
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get interactions for a user
    
    Args:
        user_id: User ID
        session_id: Optional session ID filter
        limit: Maximum number of results
        offset: Offset for pagination
    """
    try:
        if session_id:
            query = """
                SELECT * FROM student_interactions
                WHERE user_id = ? AND session_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            params = (user_id, session_id, limit, offset)
        else:
            query = """
                SELECT * FROM student_interactions
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            params = (user_id, limit, offset)
        
        interactions = db.execute_query(query, params)
        
        # Parse JSON fields
        for interaction in interactions:
            if interaction.get("sources"):
                try:
                    interaction["sources"] = json.loads(interaction["sources"])
                except:
                    interaction["sources"] = []
            if interaction.get("metadata"):
                try:
                    interaction["metadata"] = json.loads(interaction["metadata"])
                except:
                    interaction["metadata"] = {}
        
        # Get total count
        if session_id:
            count_query = "SELECT COUNT(*) as count FROM student_interactions WHERE user_id = ? AND session_id = ?"
            count_params = (user_id, session_id)
        else:
            count_query = "SELECT COUNT(*) as count FROM student_interactions WHERE user_id = ?"
            count_params = (user_id,)
        
        count_result = db.execute_query(count_query, count_params)
        total = count_result[0]["count"] if count_result else 0
        
        return {
            "interactions": interactions,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get interactions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get interactions: {str(e)}"
        )


@router.get("/session/{session_id}")
async def get_session_interactions(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get all interactions for a session with student names and topic information
    Uses Auth service for user data via HTTP API
    """
    try:
        # Simple query to get interactions with topic info (no user JOIN needed)
        query = """
            SELECT
                si.interaction_id,
                si.user_id,
                si.session_id,
                si.query,
                si.original_response,
                si.personalized_response,
                si.timestamp,
                si.processing_time_ms,
                si.model_used,
                si.chain_type,
                si.sources,
                si.metadata,
                si.created_at,
                -- Topic information from question-topic mapping
                ct.topic_title,
                qtm.confidence_score as topic_confidence,
                qtm.question_complexity,
                qtm.question_type
            FROM student_interactions si
            LEFT JOIN question_topic_mapping qtm ON si.interaction_id = qtm.interaction_id
            LEFT JOIN course_topics ct ON qtm.topic_id = ct.topic_id AND ct.is_active = TRUE
            WHERE si.session_id = ?
            ORDER BY si.timestamp DESC
            LIMIT ? OFFSET ?
        """
        
        interactions = db.execute_query(query, (session_id, limit, offset))
        
        # Extract unique user IDs from interactions
        user_ids = list(set([interaction.get("user_id") for interaction in interactions if interaction.get("user_id")]))
        
        # Fetch user information from Auth service
        user_info_map = await get_user_info_from_auth_service(user_ids)
        
        # Parse JSON fields and merge user information
        for interaction in interactions:
            # Parse JSON fields
            if interaction.get("sources"):
                try:
                    interaction["sources"] = json.loads(interaction["sources"])
                except:
                    interaction["sources"] = []
            if interaction.get("metadata"):
                try:
                    interaction["metadata"] = json.loads(interaction["metadata"])
                except:
                    interaction["metadata"] = {}
            
            # Get user info from Auth service
            user_id = interaction.get("user_id")
            user_info = user_info_map.get(user_id)
            
            if user_info:
                # User found in Auth service
                interaction["first_name"] = user_info.get("first_name", "")
                interaction["last_name"] = user_info.get("last_name", "")
                interaction["username"] = user_info.get("username", "")
                interaction["student_name"] = user_info.get("student_name", f"Öğrenci ({user_id})")
            else:
                # User not found in Auth service
                interaction["first_name"] = ""
                interaction["last_name"] = ""
                interaction["username"] = ""
                interaction["student_name"] = f"Öğrenci (ID: {user_id})"
                logger.warning(f"User not found in Auth service: {user_id}")
            
            # Format topic information
            if interaction.get("topic_title"):
                interaction["topic_info"] = {
                    "title": interaction["topic_title"],
                    "confidence": interaction.get("topic_confidence"),
                    "question_complexity": interaction.get("question_complexity"),
                    "question_type": interaction.get("question_type")
                }
            else:
                interaction["topic_info"] = None
        
        # Get total count
        count_query = "SELECT COUNT(*) as count FROM student_interactions WHERE session_id = ?"
        count_result = db.execute_query(count_query, (session_id,))
        total = count_result[0]["count"] if count_result else 0
        
        # Create debug info
        failed_user_lookups = [uid for uid in user_ids if user_info_map.get(uid) is None]
        
        return {
            "interactions": interactions,
            "total": total,
            "count": len(interactions),
            "limit": limit,
            "offset": offset,
            "debug_info": {
                "auth_service_called": True,
                "total_user_ids": len(user_ids),
                "successful_user_lookups": len([uid for uid in user_ids if user_info_map.get(uid) is not None]),
                "failed_user_lookups": failed_user_lookups
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get session interactions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session interactions: {str(e)}"
        )


@router.get("/detail/{interaction_id}")
async def get_interaction(interaction_id: int, db: DatabaseManager = Depends(get_db)):
    """
    Get a specific interaction by ID
    """
    try:
        query = "SELECT * FROM student_interactions WHERE interaction_id = ?"
        results = db.execute_query(query, (interaction_id,))
        
        if not results:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        interaction = results[0]
        
        # Parse JSON fields
        if interaction.get("sources"):
            try:
                interaction["sources"] = json.loads(interaction["sources"])
            except:
                interaction["sources"] = []
        if interaction.get("metadata"):
            try:
                interaction["metadata"] = json.loads(interaction["metadata"])
            except:
                interaction["metadata"] = {}
        
        return interaction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get interaction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get interaction: {str(e)}"
        )

