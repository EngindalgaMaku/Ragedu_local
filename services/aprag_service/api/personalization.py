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
    from config.feature_flags import FeatureFlags
    from business_logic.pedagogical import (
        get_zpd_calculator,
        get_bloom_detector,
        get_cognitive_load_manager
    )
except ImportError:
    # Fallback import
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from database.database import DatabaseManager
    from api.profiles import get_profile
    from config.feature_flags import FeatureFlags
    from business_logic.pedagogical import (
        get_zpd_calculator,
        get_bloom_detector,
        get_cognitive_load_manager
    )
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
    # Faz 3: Pedagogical additions
    zpd_info: Optional[Dict[str, Any]] = None
    bloom_info: Optional[Dict[str, Any]] = None
    cognitive_load: Optional[Dict[str, Any]] = None
    pedagogical_instructions: Optional[str] = None


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
    prompt += "\n\nMARKDOWN FORMATI KULLAN: Önemli kavramları **kalın** yaz, listeler için `-` veya `*` kullan, kod için `backtick` kullan, başlıklar için `##` kullan. Formatlı ve okunabilir bir cevap ver."
    
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
    
    **Faz 3: Now includes pedagogical monitoring:**
    - ZPD Calculator: Determines optimal difficulty level
    - Bloom Detector: Identifies cognitive level of question
    - Cognitive Load: Optimizes response complexity
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
        
        # Analyze profile to get personalization factors (legacy v1)
        factors = _analyze_student_profile(profile_dict)
        
        # === FAZ 3: PEDAGOGICAL MONITORING ===
        zpd_info = None
        bloom_info = None
        cognitive_load_info = None
        pedagogical_instructions = ""
        
        # 1. ZPD Calculator (if enabled)
        if FeatureFlags.is_zpd_enabled():
            try:
                zpd_calc = get_zpd_calculator()
                if zpd_calc:
                    # Get recent interactions for ZPD calculation
                    recent_interactions = db.execute_query(
                        """
                        SELECT * FROM student_interactions
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 20
                        """,
                        (request.user_id, request.session_id)
                    )
                    
                    zpd_info = zpd_calc.calculate_zpd_level(
                        recent_interactions=recent_interactions,
                        student_profile=profile_dict
                    )
                    
                    # Update factors with ZPD recommendations
                    factors['difficulty_level'] = zpd_info['recommended_level']
                    
                    logger.info(f"ZPD: {zpd_info['current_level']} → {zpd_info['recommended_level']} "
                              f"(success: {zpd_info['success_rate']:.2f})")
            except Exception as e:
                logger.warning(f"ZPD calculation failed: {e}")
        
        # 2. Bloom Taxonomy Detection (if enabled)
        if FeatureFlags.is_bloom_enabled():
            try:
                bloom_det = get_bloom_detector()
                if bloom_det:
                    bloom_info = bloom_det.detect_bloom_level(request.query)
                    
                    # Generate Bloom instructions for LLM
                    bloom_instructions = bloom_det.generate_bloom_instructions(
                        detected_level=bloom_info['level'],
                        student_zpd_level=factors['difficulty_level']
                    )
                    pedagogical_instructions += bloom_instructions
                    
                    logger.info(f"Bloom: Level {bloom_info['level_index']} ({bloom_info['level']}) "
                              f"- confidence: {bloom_info['confidence']:.2f}")
            except Exception as e:
                logger.warning(f"Bloom detection failed: {e}")
        
        # 3. Cognitive Load Management (if enabled)
        if FeatureFlags.is_cognitive_load_enabled():
            try:
                cog_mgr = get_cognitive_load_manager()
                if cog_mgr:
                    cognitive_load_info = cog_mgr.calculate_cognitive_load(
                        response=request.original_response,
                        query=request.query
                    )
                    
                    # Generate simplification instructions if needed
                    if cognitive_load_info['needs_simplification']:
                        simplification_instructions = cog_mgr.generate_simplification_instructions(
                            cognitive_load_info
                        )
                        pedagogical_instructions += simplification_instructions
                    
                    logger.info(f"Cognitive Load: {cognitive_load_info['total_load']:.2f} "
                              f"(simplify: {cognitive_load_info['needs_simplification']})")
            except Exception as e:
                logger.warning(f"Cognitive load calculation failed: {e}")
        
        # === END FAZ 3 ===
        
        # If no significant personalization needed, return original
        if (
            profile_dict.get("total_feedback_count", 0) < 3 and
            not profile_dict.get("preferred_explanation_style") and
            not profile_dict.get("preferred_difficulty_level")
        ):
            logger.info("Insufficient profile data, returning original response (with pedagogical info if available)")
            return PersonalizeResponse(
                personalized_response=request.original_response,
                personalization_factors=factors,
                difficulty_adjustment=None,
                explanation_level=None,
                zpd_info=zpd_info,
                bloom_info=bloom_info,
                cognitive_load=cognitive_load_info,
                pedagogical_instructions=pedagogical_instructions if pedagogical_instructions else None
            )
        
        # Generate personalization prompt (v1 + pedagogical enhancements)
        personalization_prompt = _generate_personalization_prompt(
            request.original_response,
            request.query,
            factors
        )
        
        # Add pedagogical instructions if available
        if pedagogical_instructions:
            personalization_prompt += pedagogical_instructions
        
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
                    explanation_level=factors.get("explanation_style"),
                    # Faz 3: Pedagogical info
                    zpd_info=zpd_info,
                    bloom_info=bloom_info,
                    cognitive_load=cognitive_load_info,
                    pedagogical_instructions=pedagogical_instructions if pedagogical_instructions else None
                )
            else:
                logger.warning(f"Model inference failed: {model_response.status_code}")
                # Fallback to original
                return PersonalizeResponse(
                    personalized_response=request.original_response,
                    personalization_factors=factors,
                    difficulty_adjustment=None,
                    explanation_level=None,
                    zpd_info=zpd_info,
                    bloom_info=bloom_info,
                    cognitive_load=cognitive_load_info,
                    pedagogical_instructions=pedagogical_instructions if pedagogical_instructions else None
                )
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Model inference service unavailable: {e}")
            # Fallback to original response
            return PersonalizeResponse(
                personalized_response=request.original_response,
                personalization_factors=factors,
                difficulty_adjustment=None,
                explanation_level=None,
                zpd_info=zpd_info,
                bloom_info=bloom_info,
                cognitive_load=cognitive_load_info,
                pedagogical_instructions=pedagogical_instructions if pedagogical_instructions else None
            )
        
    except Exception as e:
        logger.error(f"Personalization failed: {e}")
        import traceback
        traceback.print_exc()
        # Return original response on error
        return PersonalizeResponse(
            personalized_response=request.original_response,
            personalization_factors={},
            difficulty_adjustment=None,
            explanation_level=None,
            zpd_info=None,
            bloom_info=None,
            cognitive_load=None,
            pedagogical_instructions=None
        )


@router.post("/personalize", response_model=PersonalizeResponse)
async def personalize_response_endpoint(
    request: PersonalizeRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Generate personalized response for student
    
    This endpoint analyzes student profile and adapts the response:
    - Adjusts difficulty based on ZPD level
    - Modifies explanation style based on Bloom taxonomy
    - Manages cognitive load
    - Applies APRAG personalization factors
    
    **Required for Gateway Integration**
    """
    
    # Check if APRAG is enabled
    if not FeatureFlags.is_aprag_enabled():
        raise HTTPException(
            status_code=503,
            detail="APRAG personalization is not enabled"
        )
    
    try:
        logger.info(f"Personalization request for user {request.user_id}, session {request.session_id}")
        
        # Get student profile
        try:
            profile_result = await get_profile(request.user_id, request.session_id, db)
            profile_dict = profile_result.dict() if hasattr(profile_result, 'dict') else profile_result
        except Exception as e:
            logger.warning(f"Could not get profile: {e}")
            profile_dict = {}
        
        # Get recent interactions for context
        recent_interactions = db.execute_query(
            """
            SELECT * FROM student_interactions
            WHERE user_id = ? AND session_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            (request.user_id, request.session_id)
        )
        
        # Analyze student profile
        factors = _analyze_student_profile(profile_dict)
        
        # Get pedagogical analysis if enabled
        zpd_info = None
        bloom_info = None
        cognitive_load_info = None
        pedagogical_instructions = None
        
        if FeatureFlags.is_egitsel_kbrag_enabled():
            # ZPD Calculator
            if FeatureFlags.is_zpd_enabled():
                try:
                    zpd_calc = get_zpd_calculator()
                    zpd_result = zpd_calc.calculate_zpd_level(recent_interactions or [], profile_dict)
                    zpd_info = zpd_result
                    factors["zpd_level"] = zpd_result.get("current_level", "intermediate")
                except Exception as e:
                    logger.warning(f"ZPD calculation failed: {e}")
            
            # Bloom Taxonomy
            if FeatureFlags.is_bloom_enabled():
                try:
                    bloom_detector = get_bloom_detector()
                    bloom_result = bloom_detector.detect_bloom_level(request.query)
                    bloom_info = bloom_result
                    factors["bloom_level"] = bloom_result.get("level", "understand")
                except Exception as e:
                    logger.warning(f"Bloom detection failed: {e}")
            
            # Cognitive Load
            if FeatureFlags.is_cognitive_load_enabled():
                try:
                    cognitive_manager = get_cognitive_load_manager()
                    load_result = cognitive_manager.analyze_cognitive_load(
                        request.original_response,
                        profile_dict
                    )
                    cognitive_load_info = load_result
                    factors["cognitive_load"] = load_result.get("load_level", 0.5)
                except Exception as e:
                    logger.warning(f"Cognitive load analysis failed: {e}")
            
            # Generate pedagogical instructions
            pedagogical_instructions = _generate_pedagogical_instructions(
                zpd_info,
                bloom_info,
                cognitive_load_info
            )
        
        # For now, return original response with metadata
        # In future, we can use LLM to actually rewrite the response
        logger.info(f"Personalization complete for user {request.user_id}")
        
        return PersonalizeResponse(
            personalized_response=request.original_response,  # TODO: Actual personalization with LLM
            personalization_factors=factors,
            difficulty_adjustment=zpd_info.get("recommended_level") if zpd_info else None,
            explanation_level=bloom_info.get("level") if bloom_info else None,
            zpd_info=zpd_info,
            bloom_info=bloom_info,
            cognitive_load=cognitive_load_info,
            pedagogical_instructions=pedagogical_instructions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Personalization endpoint failed: {e}", exc_info=True)
        # Return original response on error (graceful degradation)
        return PersonalizeResponse(
            personalized_response=request.original_response,
            personalization_factors={},
            difficulty_adjustment=None,
            explanation_level=None,
            zpd_info=None,
            bloom_info=None,
            cognitive_load=None,
            pedagogical_instructions=None
        )

