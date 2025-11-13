"""
Recommendation endpoints
Generates personalized recommendations for students
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import requests
import os
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Import database manager and profiles
try:
    from database.database import DatabaseManager
    from main import db_manager
    from api.profiles import get_profile
except ImportError:
    # Fallback import
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from database.database import DatabaseManager
    from api.profiles import get_profile
    db_manager = None

# Model Inference Service URL - Google Cloud Run compatible
# For Docker: use service name (e.g., http://model-inference-service:8002)
# For Cloud Run: use full URL (e.g., https://model-inference-xxx.run.app)
MODEL_INFERENCE_URL = os.getenv("MODEL_INFERENCE_URL", None)
if not MODEL_INFERENCE_URL:
    MODEL_INFERENCE_HOST = os.getenv("MODEL_INFERENCE_HOST", "model-inference-service")
    MODEL_INFERENCE_PORT = os.getenv("MODEL_INFERENCE_PORT", "8002")
    # Check if host is a full URL (Cloud Run)
    if MODEL_INFERENCE_HOST.startswith("http://") or MODEL_INFERENCE_HOST.startswith("https://"):
        MODEL_INFERENCE_URL = MODEL_INFERENCE_HOST
    else:
        # Docker service name format
        MODEL_INFERENCE_URL = f"http://{MODEL_INFERENCE_HOST}:{MODEL_INFERENCE_PORT}"


class Recommendation(BaseModel):
    """Recommendation model"""
    recommendation_id: Optional[int] = None
    recommendation_type: str
    title: str
    description: str
    content: Dict[str, Any]
    priority: int
    relevance_score: float
    status: Optional[str] = "pending"


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    recommendations: List[Recommendation]
    total: int


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


def _generate_question_recommendations(
    user_id: str,
    session_id: str,
    profile: Dict[str, Any],
    db: DatabaseManager
) -> List[Dict[str, Any]]:
    """
    Generate personalized question recommendations based on student profile and interactions
    """
    recommendations = []
    
    try:
        # Get recent interactions for the session
        interactions = db.execute_query(
            """
            SELECT query, original_response, timestamp, sources
            FROM student_interactions
            WHERE user_id = ? AND session_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
            """,
            (user_id, session_id)
        )
        
        if not interactions or len(interactions) < 2:
            # Not enough interactions, generate generic questions
            return [
                {
                    "recommendation_type": "question",
                    "title": "Temel Kavramları Sor",
                    "description": "Bu dersin temel kavramlarını öğrenmek için sorular sorun",
                    "content": {
                        "suggested_questions": [
                            "Bu dersin temel konuları nelerdir?",
                            "Önemli kavramları açıklar mısın?",
                            "Kısa bir özet hazırla"
                        ]
                    },
                    "priority": 5,
                    "relevance_score": 0.7
                }
            ]
        
        # Analyze interactions to find weak areas
        weak_topics = profile.get("weak_topics")
        if weak_topics and isinstance(weak_topics, dict):
            # Generate questions about weak topics
            for topic, score in list(weak_topics.items())[:3]:
                recommendations.append({
                    "recommendation_type": "question",
                    "title": f"{topic} Hakkında Soru Sor",
                    "description": f"Bu konuda daha fazla bilgi edinmek için sorular sorun",
                    "content": {
                        "topic": topic,
                        "suggested_questions": [
                            f"{topic} nedir?",
                            f"{topic} hakkında daha fazla bilgi ver",
                            f"{topic} ile ilgili örnekler ver"
                        ]
                    },
                    "priority": 8,
                    "relevance_score": 0.9 - (float(score) / 10)  # Lower score = higher priority
                })
        
        # Generate follow-up questions based on recent interactions
        if interactions:
            latest_interaction = interactions[0]
            latest_query = latest_interaction.get("query", "")
            latest_response = latest_interaction.get("original_response", "")
            
            if latest_query and latest_response:
                # Use LLM to generate follow-up questions
                try:
                    prompt = f"""Öğrenci şu soruyu sormuş: "{latest_query}"
Ve şu cevabı almış: "{latest_response[:500]}"

Bu konuyla ilgili öğrencinin öğrenmesini destekleyecek 3 takip sorusu öner. Sorular Türkçe olmalı ve öğrencinin konuyu daha iyi anlamasına yardımcı olmalı.

Sadece soruları liste halinde ver, başka açıklama yapma:"""

                    model_response = requests.post(
                        f"{MODEL_INFERENCE_URL}/models/generate",
                        json={
                            "prompt": prompt,
                            "model": "llama-3.1-8b-instant",
                            "max_tokens": 200,
                            "temperature": 0.7,
                        },
                        timeout=5
                    )
                    
                    if model_response.status_code == 200:
                        result = model_response.json()
                        questions_text = result.get("response", result.get("text", ""))
                        
                        # Parse questions (simple extraction)
                        questions = []
                        for line in questions_text.split('\n'):
                            line = line.strip()
                            if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit() or '?' in line):
                                # Clean up the question
                                question = line.lstrip('- •0123456789. ').strip()
                                if question and len(question) > 10:
                                    questions.append(question)
                        
                        if questions:
                            recommendations.append({
                                "recommendation_type": "question",
                                "title": "Takip Soruları",
                                "description": "Son sorduğunuz konuyla ilgili önerilen sorular",
                                "content": {
                                    "suggested_questions": questions[:3]
                                },
                                "priority": 7,
                                "relevance_score": 0.85
                            })
                except Exception as e:
                    logger.debug(f"Could not generate LLM-based questions: {e}")
        
        # If no recommendations generated, add generic ones
        if not recommendations:
            recommendations.append({
                "recommendation_type": "question",
                "title": "Daha Fazla Soru Sor",
                "description": "Öğrenmenizi desteklemek için farklı açılardan sorular sorun",
                "content": {
                    "suggested_questions": [
                        "Bu konunun pratik uygulamaları nelerdir?",
                        "Bu konuyla ilgili örnekler ver",
                        "Bu konunun önemli noktaları nelerdir?"
                    ]
                },
                "priority": 5,
                "relevance_score": 0.7
            })
        
    except Exception as e:
        logger.warning(f"Error generating question recommendations: {e}")
        # Return generic recommendations on error
        return [{
            "recommendation_type": "question",
            "title": "Sorular Sor",
            "description": "Ders materyalleri hakkında sorular sorun",
            "content": {"suggested_questions": ["Bu dersin temel konuları neler?"]},
            "priority": 5,
            "relevance_score": 0.6
        }]
    
    return recommendations


def _generate_topic_recommendations(
    user_id: str,
    session_id: str,
    profile: Dict[str, Any],
    db: DatabaseManager
) -> List[Dict[str, Any]]:
    """
    Generate personalized topic recommendations based on student profile
    """
    recommendations = []
    
    try:
        # Get strong and weak topics from profile
        strong_topics = profile.get("strong_topics")
        weak_topics = profile.get("weak_topics")
        
        # Recommend reviewing weak topics
        if weak_topics and isinstance(weak_topics, dict):
            for topic, score in list(weak_topics.items())[:3]:
                recommendations.append({
                    "recommendation_type": "topic",
                    "title": f"{topic} Konusunu Tekrar Gözden Geçir",
                    "description": f"Bu konuda daha fazla pratik yapmanız önerilir",
                    "content": {
                        "topic": topic,
                        "action": "review",
                        "reason": "weak_area"
                    },
                    "priority": 9,
                    "relevance_score": 0.95 - (float(score) / 10)
                })
        
        # Recommend exploring related topics for strong areas
        if strong_topics and isinstance(strong_topics, dict):
            for topic, score in list(strong_topics.items())[:2]:
                if float(score) >= 8.0:  # Very strong topic
                    recommendations.append({
                        "recommendation_type": "topic",
                        "title": f"{topic} ile İlgili İleri Konular",
                        "description": f"Bu konuda iyisiniz, ileri seviye konuları keşfedebilirsiniz",
                        "content": {
                            "topic": topic,
                            "action": "explore_advanced",
                            "reason": "strong_area"
                        },
                        "priority": 6,
                        "relevance_score": 0.75
                    })
        
        # If no specific recommendations, suggest general topics
        if not recommendations:
            recommendations.append({
                "recommendation_type": "topic",
                "title": "Yeni Konular Keşfet",
                "description": "Ders materyallerindeki farklı konuları keşfedin",
                "content": {
                    "action": "explore",
                    "reason": "general"
                },
                "priority": 5,
                "relevance_score": 0.6
            })
        
    except Exception as e:
        logger.warning(f"Error generating topic recommendations: {e}")
    
    return recommendations


@router.get("/{user_id}")
async def get_recommendations(
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 10,
    db: DatabaseManager = Depends(get_db)
) -> RecommendationResponse:
    """
    Get personalized recommendations for a user
    
    Args:
        user_id: User ID
        session_id: Optional session ID filter
        limit: Maximum number of recommendations
    """
    try:
        # Get student profile
        try:
            profile_result = await get_profile(user_id, session_id, db)
            profile_dict = profile_result.dict() if hasattr(profile_result, 'dict') else profile_result
        except Exception as e:
            logger.warning(f"Could not get profile: {e}")
            profile_dict = {}
        
        # Generate recommendations
        all_recommendations = []
        
        if session_id:
            # Question recommendations
            question_recs = _generate_question_recommendations(user_id, session_id, profile_dict, db)
            all_recommendations.extend(question_recs)
            
            # Topic recommendations
            topic_recs = _generate_topic_recommendations(user_id, session_id, profile_dict, db)
            all_recommendations.extend(topic_recs)
        
        # Get existing recommendations from database
        if session_id:
            existing_query = """
                SELECT * FROM recommendations
                WHERE user_id = ? AND session_id = ? AND status = 'pending'
                ORDER BY priority DESC, relevance_score DESC
                LIMIT ?
            """
            existing_params = (user_id, session_id, limit)
        else:
            existing_query = """
                SELECT * FROM recommendations
                WHERE user_id = ? AND status = 'pending'
                ORDER BY priority DESC, relevance_score DESC
                LIMIT ?
            """
            existing_params = (user_id, limit)
        
        existing_recs = db.execute_query(existing_query, existing_params)
        
        # Convert existing recommendations to dict format
        for rec in existing_recs:
            content = {}
            if rec.get("content"):
                try:
                    content = json.loads(rec["content"])
                except:
                    content = {}
            
            all_recommendations.append({
                "recommendation_id": rec["recommendation_id"],
                "recommendation_type": rec["recommendation_type"],
                "title": rec["title"],
                "description": rec["description"],
                "content": content,
                "priority": rec.get("priority", 5),
                "relevance_score": float(rec.get("relevance_score", 0.5)) if rec.get("relevance_score") else 0.5,
                "status": rec.get("status", "pending")
            })
        
        # Sort by priority and relevance score
        all_recommendations.sort(key=lambda x: (x["priority"], x["relevance_score"]), reverse=True)
        
        # Limit results
        all_recommendations = all_recommendations[:limit]
        
        # Save new recommendations to database (if not already saved)
        for rec in all_recommendations:
            if not rec.get("recommendation_id") and session_id:
                try:
                    rec_id = db.execute_insert(
                        """
                        INSERT INTO recommendations
                        (user_id, session_id, recommendation_type, title, description, content, priority, relevance_score, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            session_id,
                            rec["recommendation_type"],
                            rec["title"],
                            rec["description"],
                            json.dumps(rec["content"]),
                            rec["priority"],
                            rec["relevance_score"],
                            rec.get("status", "pending")
                        )
                    )
                    rec["recommendation_id"] = rec_id
                except Exception as e:
                    logger.debug(f"Could not save recommendation: {e}")
        
        return RecommendationResponse(
            recommendations=all_recommendations,
            total=len(all_recommendations)
        )
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@router.post("/{recommendation_id}/accept")
async def accept_recommendation(
    recommendation_id: int,
    db: DatabaseManager = Depends(get_db)
):
    """
    Mark a recommendation as accepted
    """
    try:
        db.execute_update(
            """
            UPDATE recommendations
            SET status = 'accepted', accepted_at = CURRENT_TIMESTAMP
            WHERE recommendation_id = ?
            """,
            (recommendation_id,)
        )
        
        return {"message": "Recommendation accepted", "recommendation_id": recommendation_id}
        
    except Exception as e:
        logger.error(f"Failed to accept recommendation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to accept recommendation: {str(e)}"
        )


@router.post("/{recommendation_id}/dismiss")
async def dismiss_recommendation(
    recommendation_id: int,
    db: DatabaseManager = Depends(get_db)
):
    """
    Mark a recommendation as dismissed
    """
    try:
        db.execute_update(
            """
            UPDATE recommendations
            SET status = 'dismissed'
            WHERE recommendation_id = ?
            """,
            (recommendation_id,)
        )
        
        return {"message": "Recommendation dismissed", "recommendation_id": recommendation_id}
        
    except Exception as e:
        logger.error(f"Failed to dismiss recommendation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to dismiss recommendation: {str(e)}"
        )

