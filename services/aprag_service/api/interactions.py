"""
Interaction logging endpoints
Records student queries and responses
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
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
    Get all interactions for a session
    """
    try:
        query = """
            SELECT * FROM student_interactions
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        
        interactions = db.execute_query(query, (session_id, limit, offset))
        
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
        count_query = "SELECT COUNT(*) as count FROM student_interactions WHERE session_id = ?"
        count_result = db.execute_query(count_query, (session_id,))
        total = count_result[0]["count"] if count_result else 0
        
        return {
            "interactions": interactions,
            "total": total,
            "count": len(interactions),
            "limit": limit,
            "offset": offset
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

