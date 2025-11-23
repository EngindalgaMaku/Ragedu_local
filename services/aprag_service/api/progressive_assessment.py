"""
Progressive Assessment Flow API (ADIM 3)
Implements progressive, adaptive assessment flow that provides deeper learning insights

Flow Stages:
1. Initial Response (Immediate) - Emoji feedback (existing)
2. Follow-up Assessment (30 sec delay) - Confidence + Application
3. Deeper Analysis (Optional, triggered by low scores) - Concept mapping + Clarification

Progressive flow provides deeper insights into student learning progress.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime, timedelta
from enum import Enum

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

db_manager = None
logger = logging.getLogger(__name__)
router = APIRouter()


class AssessmentStage(str, Enum):
    """Assessment flow stages"""
    INITIAL = "initial"
    FOLLOW_UP = "follow_up"
    DEEP_ANALYSIS = "deep_analysis"


class ConfidenceLevel(int, Enum):
    """Confidence level scale"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


class ProgressiveAssessment(BaseModel):
    """Progressive assessment data model"""
    interaction_id: int
    user_id: str
    session_id: str
    stage: AssessmentStage
    confidence_level: Optional[int] = Field(None, ge=1, le=5, description="Confidence level (1-5)")
    has_questions: Optional[bool] = Field(None, description="Does student have more questions?")
    application_understanding: Optional[str] = Field(None, max_length=1000, description="How would you use this knowledge?")
    confusion_areas: Optional[List[str]] = Field(None, description="Areas that need clarification")
    requested_topics: Optional[List[str]] = Field(None, description="Related topics student wants to explore")
    alternative_explanation_request: Optional[str] = Field(None, max_length=500, description="Request for alternative explanation")
    timestamp: Optional[datetime] = None


class FollowUpAssessmentCreate(BaseModel):
    """Request model for follow-up assessment (30 sec delay)"""
    interaction_id: int = Field(..., description="Original interaction ID")
    user_id: str
    session_id: str
    
    # Follow-up questions
    has_questions: bool = Field(..., description="Başka soru var mı?")
    confidence_level: int = Field(..., ge=1, le=5, description="Bu konuda kendini ne kadar güvende hissediyorsun? (1-5)")
    application_understanding: str = Field(..., max_length=1000, description="Bu bilgiyi gerçek hayatta nasıl kullanırsın?")
    
    # Optional comment
    comment: Optional[str] = Field(None, max_length=500)


class DeepAnalysisCreate(BaseModel):
    """Request model for deep analysis assessment"""
    interaction_id: int = Field(..., description="Original interaction ID")
    user_id: str
    session_id: str
    
    # Deep analysis questions
    confusion_areas: List[str] = Field(..., description="Hangi kısmı daha açık olmyabiliriz?")
    requested_topics: Optional[List[str]] = Field(None, description="Related topics to explore")
    alternative_explanation_request: Optional[str] = Field(None, max_length=500, description="Alternative explanation request")
    
    # Additional context
    comment: Optional[str] = Field(None, max_length=1000)


class ProgressiveAssessmentResponse(BaseModel):
    """Response model for progressive assessment"""
    assessment_id: int
    message: str
    stage: AssessmentStage
    interaction_id: int
    next_stage_available: bool
    next_stage_delay: Optional[int] = None  # seconds
    insights_generated: bool = False
    recommended_actions: Optional[List[str]] = None


class AssessmentInsights(BaseModel):
    """Response model for assessment insights"""
    user_id: str
    session_id: str
    
    # Learning progression indicators
    confidence_trend: str = "stable"  # improving, declining, stable, insufficient_data
    application_readiness: str = "moderate"  # low, moderate, high, excellent
    concept_mastery_level: float = Field(0.5, ge=0.0, le=1.0)
    
    # Areas needing attention
    weak_areas: List[str] = []
    confusion_patterns: List[str] = []
    knowledge_gaps: List[str] = []
    
    # Recommendations
    recommended_topics: List[str] = []
    suggested_exercises: List[str] = []
    next_learning_steps: List[str] = []
    
    # Analytics
    total_assessments: int
    follow_up_completion_rate: float
    deep_analysis_trigger_rate: float
    average_confidence: float
    
    # Adaptive triggers
    needs_immediate_intervention: bool = False
    intervention_reasons: List[str] = []


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


@router.post("/follow-up", response_model=ProgressiveAssessmentResponse, status_code=201)
async def create_follow_up_assessment(
    assessment: FollowUpAssessmentCreate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Submit follow-up assessment (Stage 2 - 30 second delay)
    
    This endpoint handles the second stage of progressive assessment:
    - Confidence level check
    - Additional questions inquiry
    - Application understanding evaluation
    
    **Adaptive Triggering:**
    - Confidence < 3 → Trigger deep analysis
    - has_questions = true → Additional support needed
    - Poor application understanding → Concept reinforcement
    """
    
    if not FeatureFlags.is_progressive_assessment_enabled(assessment.session_id):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Progressive assessment is not enabled for this session"
        )
    
    try:
        logger.info(f"Follow-up assessment for interaction {assessment.interaction_id} "
                   f"by user {assessment.user_id}: confidence={assessment.confidence_level}")
        
        # Verify original interaction exists
        interaction = db.execute_query(
            "SELECT * FROM student_interactions WHERE interaction_id = ?",
            (assessment.interaction_id,)
        )
        
        if not interaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interaction {assessment.interaction_id} not found"
            )
        
        # Insert follow-up assessment
        assessment_id = db.execute_insert(
            """
            INSERT INTO progressive_assessments
            (interaction_id, user_id, session_id, stage, confidence_level,
             has_questions, application_understanding, comment, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                assessment.interaction_id, assessment.user_id, assessment.session_id,
                AssessmentStage.FOLLOW_UP.value, assessment.confidence_level,
                assessment.has_questions, assessment.application_understanding,
                assessment.comment
            )
        )
        
        # Update original interaction with follow-up data
        follow_up_data = {
            "stage": "follow_up",
            "confidence_level": assessment.confidence_level,
            "has_questions": assessment.has_questions,
            "application_understanding": assessment.application_understanding
        }
        
        db.execute_update(
            """
            UPDATE student_interactions
            SET progressive_assessment_data = ?,
                progressive_assessment_stage = ?
            WHERE interaction_id = ?
            """,
            (json.dumps(follow_up_data), "follow_up", assessment.interaction_id)
        )
        
        # Determine if deep analysis is needed
        needs_deep_analysis = _should_trigger_deep_analysis(
            assessment.confidence_level,
            assessment.has_questions,
            assessment.application_understanding
        )
        
        # Update student profile with confidence data
        _update_profile_with_progressive_data(
            db, assessment.user_id, assessment.session_id,
            assessment.confidence_level, assessment.has_questions,
            assessment.application_understanding
        )
        
        # Generate recommendations
        recommended_actions = _generate_follow_up_recommendations(
            assessment.confidence_level,
            assessment.has_questions,
            assessment.application_understanding
        )
        
        logger.info(f"Follow-up assessment {assessment_id} created. Deep analysis needed: {needs_deep_analysis}")
        
        return ProgressiveAssessmentResponse(
            assessment_id=assessment_id,
            message="Follow-up değerlendirmeniz kaydedildi. Teşekkürler!",
            stage=AssessmentStage.FOLLOW_UP,
            interaction_id=assessment.interaction_id,
            next_stage_available=needs_deep_analysis,
            next_stage_delay=0,  # Deep analysis available immediately if needed
            insights_generated=True,
            recommended_actions=recommended_actions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create follow-up assessment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record follow-up assessment: {str(e)}"
        )


@router.post("/deep-analysis", response_model=ProgressiveAssessmentResponse, status_code=201)
async def create_deep_analysis_assessment(
    assessment: DeepAnalysisCreate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Submit deep analysis assessment (Stage 3 - Optional, triggered by low scores)
    
    This endpoint handles the third stage of progressive assessment:
    - Confusion areas identification
    - Concept mapping requests
    - Alternative explanation requests
    - Related topic exploration
    """
    
    if not FeatureFlags.is_progressive_assessment_enabled(assessment.session_id):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Progressive assessment is not enabled for this session"
        )
    
    try:
        logger.info(f"Deep analysis assessment for interaction {assessment.interaction_id} "
                   f"by user {assessment.user_id}: {len(assessment.confusion_areas)} confusion areas")
        
        # Verify original interaction exists
        interaction = db.execute_query(
            "SELECT * FROM student_interactions WHERE interaction_id = ?",
            (assessment.interaction_id,)
        )
        
        if not interaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interaction {assessment.interaction_id} not found"
            )
        
        # Insert deep analysis assessment
        assessment_id = db.execute_insert(
            """
            INSERT INTO progressive_assessments
            (interaction_id, user_id, session_id, stage, confusion_areas,
             requested_topics, alternative_explanation_request, comment, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                assessment.interaction_id, assessment.user_id, assessment.session_id,
                AssessmentStage.DEEP_ANALYSIS.value, json.dumps(assessment.confusion_areas),
                json.dumps(assessment.requested_topics) if assessment.requested_topics else None,
                assessment.alternative_explanation_request, assessment.comment
            )
        )
        
        # Update original interaction with deep analysis data
        deep_analysis_data = {
            "stage": "deep_analysis",
            "confusion_areas": assessment.confusion_areas,
            "requested_topics": assessment.requested_topics,
            "alternative_explanation_request": assessment.alternative_explanation_request
        }
        
        db.execute_update(
            """
            UPDATE student_interactions
            SET progressive_assessment_data = ?,
                progressive_assessment_stage = ?
            WHERE interaction_id = ?
            """,
            (json.dumps(deep_analysis_data), "deep_analysis", assessment.interaction_id)
        )
        
        # Generate personalized recommendations based on confusion areas
        recommended_actions = _generate_deep_analysis_recommendations(
            assessment.confusion_areas,
            assessment.requested_topics,
            assessment.alternative_explanation_request
        )
        
        # Update analytics for concept mapping
        _update_concept_confusion_analytics(
            db, assessment.user_id, assessment.session_id,
            assessment.confusion_areas, assessment.requested_topics
        )
        
        logger.info(f"Deep analysis assessment {assessment_id} created with {len(recommended_actions)} recommendations")
        
        return ProgressiveAssessmentResponse(
            assessment_id=assessment_id,
            message="Detaylı analiz değerlendirmeniz kaydedildi. Size özel öneriler hazırlandı!",
            stage=AssessmentStage.DEEP_ANALYSIS,
            interaction_id=assessment.interaction_id,
            next_stage_available=False,  # This is the final stage
            insights_generated=True,
            recommended_actions=recommended_actions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create deep analysis assessment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record deep analysis assessment: {str(e)}"
        )


@router.get("/insights/{user_id}", response_model=AssessmentInsights)
async def get_assessment_insights(
    user_id: str,
    session_id: Optional[str] = None,
    days: int = 7,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get comprehensive assessment insights for a user
    
    Provides deep learning analytics based on progressive assessment data:
    - Confidence trends
    - Application readiness
    - Concept mastery levels
    - Learning recommendations
    """
    
    try:
        # Base query conditions
        where_conditions = ["user_id = ?"]
        params = [user_id]
        
        if session_id:
            where_conditions.append("session_id = ?")
            params.append(session_id)
        
        where_conditions.append("timestamp >= datetime('now', '-{} days')".format(days))
        where_clause = " AND ".join(where_conditions)
        
        # Get progressive assessment data
        assessments = db.execute_query(
            f"""
            SELECT * FROM progressive_assessments
            WHERE {where_clause}
            ORDER BY timestamp DESC
            """,
            tuple(params)
        )
        
        if not assessments:
            return AssessmentInsights(
                user_id=user_id,
                session_id=session_id or "all",
                total_assessments=0,
                follow_up_completion_rate=0.0,
                deep_analysis_trigger_rate=0.0,
                average_confidence=2.5
            )
        
        # Analyze assessment data
        insights = _analyze_progressive_assessments(assessments, user_id, session_id or "all")
        
        logger.info(f"Generated insights for user {user_id}: {insights.total_assessments} assessments analyzed")
        
        return insights
        
    except Exception as e:
        logger.error(f"Failed to get assessment insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get assessment insights: {str(e)}"
        )


@router.get("/check-trigger/{interaction_id}")
async def check_progressive_trigger(
    interaction_id: int,
    db: DatabaseManager = Depends(get_db)
):
    """
    Check if progressive assessment should be triggered for an interaction
    
    Returns trigger conditions based on:
    - Initial emoji feedback score
    - Time since interaction
    - User's assessment history
    """
    
    try:
        # Get interaction with emoji feedback
        interaction = db.execute_query(
            """
            SELECT interaction_id, user_id, session_id, emoji_feedback, 
                   feedback_score, emoji_feedback_timestamp, timestamp
            FROM student_interactions
            WHERE interaction_id = ?
            """,
            (interaction_id,)
        )
        
        if not interaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interaction {interaction_id} not found"
            )
        
        interaction_data = interaction[0]
        
        # Check if follow-up assessment already exists
        existing_assessment = db.execute_query(
            "SELECT * FROM progressive_assessments WHERE interaction_id = ?",
            (interaction_id,)
        )
        
        if existing_assessment:
            return {
                "trigger_follow_up": False,
                "trigger_deep_analysis": False,
                "reason": "Assessment already completed",
                "next_stage": None
            }
        
        # Determine trigger conditions
        emoji_score = interaction_data.get('feedback_score', 0.5)
        emoji_feedback = interaction_data.get('emoji_feedback')
        
        # Trigger follow-up if:
        # 1. Low emoji scores (😐, ❌)
        # 2. 30+ seconds have passed since interaction
        # 3. No existing progressive assessment
        
        should_trigger_follow_up = False
        should_trigger_deep = False
        
        if emoji_feedback in ['😐', '❌'] or emoji_score <= 0.5:
            should_trigger_follow_up = True
            
        # Check time delay (30 seconds for follow-up)
        if interaction_data.get('timestamp'):
            interaction_time = datetime.fromisoformat(interaction_data['timestamp'].replace('Z', '+00:00'))
            time_diff = datetime.now() - interaction_time
            
            if time_diff.total_seconds() >= 30:
                should_trigger_follow_up = True
        
        return {
            "trigger_follow_up": should_trigger_follow_up,
            "trigger_deep_analysis": should_trigger_deep,
            "emoji_feedback": emoji_feedback,
            "emoji_score": emoji_score,
            "interaction_id": interaction_id,
            "user_id": interaction_data['user_id'],
            "session_id": interaction_data['session_id']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check progressive trigger: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check trigger conditions: {str(e)}"
        )


def _should_trigger_deep_analysis(confidence_level: int, has_questions: bool, application_text: str) -> bool:
    """Determine if deep analysis should be triggered"""
    # Trigger deep analysis if:
    # 1. Confidence level is low (< 3)
    # 2. Student has more questions
    # 3. Application understanding is insufficient
    
    if confidence_level < 3:
        return True
    
    if has_questions:
        return True
    
    # Simple heuristic: if application text is too short or contains uncertainty words
    if len(application_text.strip()) < 20:
        return True
    
    uncertainty_words = ['bilmiyorum', 'emin değilim', 'anlamadım', 'karışık', 'zor']
    if any(word in application_text.lower() for word in uncertainty_words):
        return True
    
    return False


def _generate_follow_up_recommendations(confidence_level: int, has_questions: bool, application_text: str) -> List[str]:
    """Generate recommendations based on follow-up assessment"""
    recommendations = []
    
    if confidence_level < 3:
        recommendations.append("Bu konuyu tekrar gözden geçirmenizi öneririz")
        recommendations.append("Ek kaynaklar ve örnekler sizin için hazırlanacak")
    
    if has_questions:
        recommendations.append("Sorularınızı sormaktan çekinmeyin")
        recommendations.append("Konu uzmanlarından destek alabilirsiniz")
    
    if len(application_text.strip()) < 20:
        recommendations.append("Pratik uygulamalar üzerinde çalışmanızı öneririz")
        recommendations.append("Gerçek hayat örnekleri ile pekiştirebiliriz")
    
    if confidence_level >= 4 and not has_questions:
        recommendations.append("Harika! İleri seviye konulara geçebilirsiniz")
        recommendations.append("Bu konudaki bilginizi başkalarıyla paylaşabilirsiniz")
    
    return recommendations


def _generate_deep_analysis_recommendations(confusion_areas: List[str], requested_topics: Optional[List[str]], alt_explanation: Optional[str]) -> List[str]:
    """Generate recommendations based on deep analysis"""
    recommendations = []
    
    if confusion_areas:
        recommendations.append(f"{len(confusion_areas)} alanda ek açıklama hazırlanacak")
        recommendations.append("Size özel kişiselleştirilmiş açıklamalar oluşturuldu")
    
    if requested_topics:
        recommendations.append(f"{len(requested_topics)} ilgili konu önerisi not edildi")
        recommendations.append("Bağlantılı konular öncelik sırasına alındı")
    
    if alt_explanation:
        recommendations.append("Alternatif açıklama tarzı tercihleriniz kaydedildi")
        recommendations.append("Gelecek açıklamalar bu tercihe göre uyarlanacak")
    
    recommendations.append("Öğrenme profiliniz güncellendi")
    recommendations.append("Size özel öğrenme rotası hazırlanıyor")
    
    return recommendations


def _update_profile_with_progressive_data(db: DatabaseManager, user_id: str, session_id: str, confidence: int, has_questions: bool, application: str):
    """Update student profile with progressive assessment data"""
    try:
        # Get current profile
        profile = db.execute_query(
            "SELECT * FROM student_profiles WHERE user_id = ? AND session_id = ?",
            (user_id, session_id)
        )
        
        confidence_score = confidence / 5.0  # Convert to 0-1 scale
        application_readiness = min(len(application.strip()) / 100.0, 1.0)  # Simple heuristic
        
        if profile:
            # Update existing
            row = profile[0]
            current_confidence = row.get('average_confidence', 0.5)
            current_readiness = row.get('application_readiness', 0.5)
            assessment_count = row.get('progressive_assessment_count', 0)
            
            new_confidence = (current_confidence * assessment_count + confidence_score) / (assessment_count + 1)
            new_readiness = (current_readiness * assessment_count + application_readiness) / (assessment_count + 1)
            
            db.execute_update(
                """
                UPDATE student_profiles
                SET average_confidence = ?,
                    application_readiness = ?,
                    progressive_assessment_count = ?,
                    has_active_questions = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
                """,
                (new_confidence, new_readiness, assessment_count + 1, has_questions, user_id, session_id)
            )
        else:
            # Create new profile
            db.execute_insert(
                """
                INSERT INTO student_profiles
                (user_id, session_id, average_confidence, application_readiness,
                 progressive_assessment_count, has_active_questions, last_updated)
                VALUES (?, ?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, session_id, confidence_score, application_readiness, has_questions)
            )
        
    except Exception as e:
        logger.warning(f"Failed to update profile with progressive data: {e}")


def _update_concept_confusion_analytics(db: DatabaseManager, user_id: str, session_id: str, confusion_areas: List[str], requested_topics: Optional[List[str]]):
    """Update analytics for concept confusion patterns"""
    try:
        for area in confusion_areas:
            # Insert confusion area record
            db.execute_insert(
                """
                INSERT INTO concept_confusion_log
                (user_id, session_id, confusion_area, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, session_id, area)
            )
        
        if requested_topics:
            for topic in requested_topics:
                # Insert requested topic record
                db.execute_insert(
                    """
                    INSERT INTO requested_topics_log
                    (user_id, session_id, requested_topic, timestamp)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (user_id, session_id, topic)
                )
        
    except Exception as e:
        logger.warning(f"Failed to update concept confusion analytics: {e}")


def _analyze_progressive_assessments(assessments: List[Dict], user_id: str, session_id: str) -> AssessmentInsights:
    """Analyze progressive assessments and generate insights"""
    
    total_assessments = len(assessments)
    follow_up_count = len([a for a in assessments if a['stage'] == 'follow_up'])
    deep_analysis_count = len([a for a in assessments if a['stage'] == 'deep_analysis'])
    
    # Calculate rates
    follow_up_rate = follow_up_count / total_assessments if total_assessments > 0 else 0
    deep_analysis_rate = deep_analysis_count / total_assessments if total_assessments > 0 else 0
    
    # Analyze confidence trends
    confidence_values = [a['confidence_level'] for a in assessments if a.get('confidence_level')]
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 2.5
    
    confidence_trend = "stable"
    if len(confidence_values) >= 3:
        recent_avg = sum(confidence_values[:3]) / 3
        older_avg = sum(confidence_values[-3:]) / 3
        if recent_avg > older_avg + 0.5:
            confidence_trend = "improving"
        elif recent_avg < older_avg - 0.5:
            confidence_trend = "declining"
    
    # Extract confusion areas and patterns
    confusion_areas = []
    for a in assessments:
        if a.get('confusion_areas'):
            try:
                areas = json.loads(a['confusion_areas'])
                confusion_areas.extend(areas)
            except:
                pass
    
    # Generate recommendations
    weak_areas = list(set(confusion_areas))[:5]  # Top 5 most common
    
    # Determine if intervention needed
    needs_intervention = False
    intervention_reasons = []
    
    if avg_confidence < 2.5:
        needs_intervention = True
        intervention_reasons.append("Düşük güven seviyesi")
    
    if deep_analysis_rate > 0.7:
        needs_intervention = True
        intervention_reasons.append("Sık detaylı analiz ihtiyacı")
    
    if confidence_trend == "declining":
        needs_intervention = True
        intervention_reasons.append("Güven seviyesinde düşüş")
    
    return AssessmentInsights(
        user_id=user_id,
        session_id=session_id,
        confidence_trend=confidence_trend,
        concept_mastery_level=avg_confidence / 5.0,
        weak_areas=weak_areas,
        confusion_patterns=weak_areas,
        total_assessments=total_assessments,
        follow_up_completion_rate=follow_up_rate,
        deep_analysis_trigger_rate=deep_analysis_rate,
        average_confidence=avg_confidence,
        needs_immediate_intervention=needs_intervention,
        intervention_reasons=intervention_reasons,
        recommended_topics=weak_areas,
        next_learning_steps=[
            "Zayıf alanları tekrar et",
            "Pratik uygulamalar yap",
            "Ek kaynakları incele"
        ] if weak_areas else [
            "İleri seviye konulara geç",
            "Bilgiyi başkalarıyla paylaş"
        ]
    )