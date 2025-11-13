"""
Personalization endpoints
Generates personalized responses based on student profiles
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import requests
import os
import json

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


class PersonalizeRequest(BaseModel):
    """Request model for personalization"""
    user_id: str
    session_id: str
    query: str
    original_response: str
    context: Optional[Dict[str, Any]] = None


class PersonalizeResponse(BaseModel):
    """Response model for personalized response"""
    personalized_response: str
    personalization_factors: Dict[str, Any]
    difficulty_adjustment: Optional[str]
    explanation_level: Optional[str]


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


def _analyze_student_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze student profile and extract personalization factors
    
    Returns:
        Dictionary with personalization factors
    """
    factors = {
        "understanding_level": "intermediate",
        "explanation_style": "balanced",
        "difficulty_level": "intermediate",
        "needs_examples": False,
        "needs_visual_aids": False,
    }
    
    # Analyze understanding level
    avg_understanding = profile.get("average_understanding")
    if avg_understanding:
        if avg_understanding >= 4.0:
            factors["understanding_level"] = "high"
            factors["difficulty_level"] = "advanced"
        elif avg_understanding >= 3.0:
            factors["understanding_level"] = "intermediate"
            factors["difficulty_level"] = "intermediate"
        else:
            factors["understanding_level"] = "low"
            factors["difficulty_level"] = "beginner"
            factors["needs_examples"] = True
    
    # Analyze satisfaction level
    avg_satisfaction = profile.get("average_satisfaction")
    if avg_satisfaction and avg_satisfaction < 3.0:
        factors["needs_examples"] = True
        factors["explanation_style"] = "detailed"
    
    # Use preferred settings if available
    if profile.get("preferred_explanation_style"):
        factors["explanation_style"] = profile["preferred_explanation_style"]
    
    if profile.get("preferred_difficulty_level"):
        factors["difficulty_level"] = profile["preferred_difficulty_level"]
    
    # Analyze feedback patterns
    total_feedback = profile.get("total_feedback_count", 0)
    if total_feedback > 0:
        # Check recent feedback for patterns
        try:
            from api.feedback import get_session_feedback
            # This would require async context, so we'll use a simpler approach
            # Check if student has low understanding or needs more explanation
            if avg_understanding and avg_understanding < 3.0:
                factors["needs_examples"] = True
                factors["explanation_style"] = "detailed"
        except:
            pass
    
    return factors


def _generate_personalization_prompt(
    original_response: str,
    query: str,
    factors: Dict[str, Any]
) -> str:
    """
    Generate a prompt for LLM to personalize the response
    """
    prompt = f"""Aşağıdaki cevabı öğrencinin öğrenme profiline göre kişiselleştir:

ÖĞRENCİ PROFİLİ:
- Anlama Seviyesi: {factors['understanding_level']}
- Zorluk Seviyesi: {factors['difficulty_level']}
- Açıklama Stili: {factors['explanation_style']}
- Örnekler Gerekli: {'Evet' if factors['needs_examples'] else 'Hayır'}

ORİJİNAL SORU:
{query}

ORİJİNAL CEVAP:
{original_response}

LÜTFEN CEVABI ŞU ŞEKİLDE KİŞİSELLEŞTİR:
"""
    
    if factors["explanation_style"] == "detailed":
        prompt += "- Daha detaylı açıklamalar ekle\n"
        prompt += "- Her adımı açıkça belirt\n"
    elif factors["explanation_style"] == "concise":
        prompt += "- Daha kısa ve öz bir açıklama yap\n"
        prompt += "- Gereksiz detayları çıkar\n"
    
    if factors["needs_examples"]:
        prompt += "- Pratik örnekler ekle\n"
        prompt += "- Günlük hayattan örnekler ver\n"
    
    if factors["difficulty_level"] == "beginner":
        prompt += "- Temel kavramları önce açıkla\n"
        prompt += "- Teknik terimleri basit dille açıkla\n"
    elif factors["difficulty_level"] == "advanced":
        prompt += "- Daha derinlemesine bilgi ver\n"
        prompt += "- İleri seviye detaylar ekle\n"
    
    prompt += "\nKişiselleştirilmiş cevabı sadece Türkçe olarak ver. Orijinal cevabın içeriğini koru, sadece sunumunu ve detay seviyesini öğrenci profiline göre ayarla."
    
    return prompt


@router.post("", status_code=200)
async def personalize_response(
    request: PersonalizeRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Personalize a RAG response based on student profile
    
    This endpoint takes an original RAG response and adapts it
    based on the student's learning profile, preferences, and history.
    """
    try:
        logger.info(f"Personalizing response for user {request.user_id}, session {request.session_id}")
        
        # Get student profile
        try:
            profile_result = await get_profile(request.user_id, request.session_id, db)
            profile_dict = profile_result.dict() if hasattr(profile_result, 'dict') else profile_result
        except Exception as e:
            logger.warning(f"Could not get profile, using defaults: {e}")
            profile_dict = {
                "average_understanding": None,
                "average_satisfaction": None,
                "total_interactions": 0,
                "total_feedback_count": 0,
                "preferred_explanation_style": None,
                "preferred_difficulty_level": None,
            }
        
        # Analyze profile to get personalization factors
        factors = _analyze_student_profile(profile_dict)
        
        # If no significant personalization needed, return original
        if (
            profile_dict.get("total_feedback_count", 0) < 3 and
            not profile_dict.get("preferred_explanation_style") and
            not profile_dict.get("preferred_difficulty_level")
        ):
            logger.info("Insufficient profile data, returning original response")
            return PersonalizeResponse(
                personalized_response=request.original_response,
                personalization_factors=factors,
                difficulty_adjustment=None,
                explanation_level=None
            )
        
        # Generate personalization prompt
        personalization_prompt = _generate_personalization_prompt(
            request.original_response,
            request.query,
            factors
        )
        
        # Call model inference service to personalize
        try:
            # Use a simple model for personalization (fast, lightweight)
            model_response = requests.post(
                f"{MODEL_INFERENCE_URL}/models/generate",
                json={
                    "prompt": personalization_prompt,
                    "model": "llama-3.1-8b-instant",  # Fast model for personalization
                    "max_tokens": min(len(request.original_response.split()) * 2, 2048),  # Allow some expansion, but cap at 2048
                    "temperature": 0.3,  # Lower temperature for more consistent personalization
                },
                timeout=10
            )
            
            if model_response.status_code == 200:
                result = model_response.json()
                personalized_text = result.get("response", result.get("text", request.original_response))
                
                # Clean up the response (remove prompt artifacts if any)
                if "Kişiselleştirilmiş cevabı" in personalized_text:
                    # Extract only the personalized response part
                    parts = personalized_text.split("Kişiselleştirilmiş cevabı")
                    if len(parts) > 1:
                        personalized_text = parts[-1].strip()
                
                logger.info(f"Successfully personalized response for user {request.user_id}")
                
                return PersonalizeResponse(
                    personalized_response=personalized_text,
                    personalization_factors=factors,
                    difficulty_adjustment=factors.get("difficulty_level"),
                    explanation_level=factors.get("explanation_style")
                )
            else:
                logger.warning(f"Model inference failed: {model_response.status_code}")
                # Fallback to original
                return PersonalizeResponse(
                    personalized_response=request.original_response,
                    personalization_factors=factors,
                    difficulty_adjustment=None,
                    explanation_level=None
                )
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Model inference service unavailable: {e}")
            # Fallback to original response
            return PersonalizeResponse(
                personalized_response=request.original_response,
                personalization_factors=factors,
                difficulty_adjustment=None,
                explanation_level=None
            )
        
    except Exception as e:
        logger.error(f"Personalization failed: {e}")
        # Return original response on error
        return PersonalizeResponse(
            personalized_response=request.original_response,
            personalization_factors={},
            difficulty_adjustment=None,
            explanation_level=None
        )

