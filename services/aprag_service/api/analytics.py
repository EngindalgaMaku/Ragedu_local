"""
Analytics endpoints
Provides analytics and insights on student learning
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

router = APIRouter()

# Import database manager
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


class AnalyticsResponse(BaseModel):
    """Response model for analytics"""
    total_interactions: int
    total_feedback: int
    average_understanding: Optional[float]
    average_satisfaction: Optional[float]
    improvement_trend: str
    learning_patterns: List[Dict[str, Any]]
    topic_performance: Dict[str, Any]
    engagement_metrics: Dict[str, Any]
    time_analysis: Dict[str, Any]


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


def _calculate_improvement_trend(
    interactions: List[Dict[str, Any]],
    feedback: List[Dict[str, Any]]
) -> str:
    """
    Calculate improvement trend based on recent interactions and feedback
    Returns: 'improving', 'stable', 'declining', or 'insufficient_data'
    """
    if len(feedback) < 3:
        return "insufficient_data"
    
    # Sort feedback by timestamp
    sorted_feedback = sorted(
        feedback,
        key=lambda x: x.get("timestamp", "") or "",
        reverse=False
    )
    
    # Calculate average understanding for first half and second half
    mid_point = len(sorted_feedback) // 2
    first_half = sorted_feedback[:mid_point]
    second_half = sorted_feedback[mid_point:]
    
    first_avg = sum(
        float(f.get("understanding_level", 0) or 0)
        for f in first_half
    ) / len(first_half) if first_half else 0
    
    second_avg = sum(
        float(f.get("understanding_level", 0) or 0)
        for f in second_half
    ) / len(second_half) if second_half else 0
    
    if second_avg > first_avg + 0.3:
        return "improving"
    elif second_avg < first_avg - 0.3:
        return "declining"
    else:
        return "stable"


def _detect_learning_patterns(
    interactions: List[Dict[str, Any]],
    feedback: List[Dict[str, Any]],
    profile: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Detect learning patterns from interactions and feedback
    """
    patterns = []
    
    # Pattern 1: Active questioning
    if len(interactions) >= 5:
        recent_interactions = interactions[-10:]
        questions_per_day = len(recent_interactions) / 7  # Approximate
        if questions_per_day >= 2:
            patterns.append({
                "pattern_type": "active_questioning",
                "description": "Öğrenci aktif olarak sorular soruyor",
                "strength": "high" if questions_per_day >= 3 else "medium",
                "recommendation": "Bu aktif öğrenme yaklaşımını sürdürün"
            })
    
    # Pattern 2: Consistent feedback
    if len(feedback) >= 3:
        avg_understanding = sum(
            float(f.get("understanding_level", 0) or 0)
            for f in feedback
        ) / len(feedback)
        if avg_understanding >= 4.0:
            patterns.append({
                "pattern_type": "high_understanding",
                "description": "Öğrenci konuları iyi anlıyor",
                "strength": "high",
                "recommendation": "İleri seviye konulara geçebilirsiniz"
            })
        elif avg_understanding < 3.0:
            patterns.append({
                "pattern_type": "needs_support",
                "description": "Öğrenci ek destek ihtiyacı gösteriyor",
                "strength": "high",
                "recommendation": "Daha detaylı açıklamalar ve örnekler önerilir"
            })
    
    # Pattern 3: Topic preferences
    weak_topics = profile.get("weak_topics")
    strong_topics = profile.get("strong_topics")
    
    if weak_topics and isinstance(weak_topics, dict):
        top_weak = sorted(
            weak_topics.items(),
            key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0
        )[:3]
        if top_weak:
            patterns.append({
                "pattern_type": "weak_areas",
                "description": f"Zayıf alanlar: {', '.join([t[0] for t in top_weak])}",
                "strength": "medium",
                "recommendation": "Bu konularda daha fazla pratik yapın"
            })
    
    if strong_topics and isinstance(strong_topics, dict):
        top_strong = sorted(
            strong_topics.items(),
            key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 10,
            reverse=True
        )[:3]
        if top_strong:
            patterns.append({
                "pattern_type": "strong_areas",
                "description": f"Güçlü alanlar: {', '.join([t[0] for t in top_strong])}",
                "strength": "high",
                "recommendation": "Bu konularda ileri seviye çalışmalar yapabilirsiniz"
            })
    
    # Pattern 4: Feedback consistency
    if len(feedback) >= 5:
        satisfaction_scores = [
            float(f.get("satisfaction_level", 0) or 0)
            for f in feedback
        ]
        if all(s >= 4.0 for s in satisfaction_scores[-3:]):
            patterns.append({
                "pattern_type": "high_satisfaction",
                "description": "Son geri bildirimlerde yüksek memnuniyet",
                "strength": "high",
                "recommendation": "Mevcut öğrenme yaklaşımı etkili"
            })
    
    return patterns


def _analyze_topic_performance(
    interactions: List[Dict[str, Any]],
    feedback: List[Dict[str, Any]],
    profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze performance by topic
    """
    topic_stats = {}
    
    # Extract topics from interactions (simple keyword-based)
    for interaction in interactions:
        query = interaction.get("query", "").lower()
        response = interaction.get("original_response", "").lower()
        
        # Simple topic extraction (can be enhanced with NLP)
        common_topics = ["kimya", "fizik", "matematik", "biyoloji", "tarih", "edebiyat"]
        for topic in common_topics:
            if topic in query or topic in response:
                if topic not in topic_stats:
                    topic_stats[topic] = {"count": 0, "avg_understanding": 0.0}
                topic_stats[topic]["count"] += 1
    
    # Add profile-based topic info
    weak_topics = profile.get("weak_topics")
    strong_topics = profile.get("strong_topics")
    
    return {
        "weak_topics": weak_topics if weak_topics else {},
        "strong_topics": strong_topics if strong_topics else {},
        "interaction_topics": topic_stats,
        "total_topics_covered": len(topic_stats)
    }


def _calculate_engagement_metrics(
    interactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate engagement metrics
    """
    if not interactions:
        return {
            "total_interactions": 0,
            "avg_per_day": 0,
            "most_active_day": None,
            "engagement_level": "low"
        }
    
    # Group by date
    daily_counts = {}
    for interaction in interactions:
        timestamp = interaction.get("timestamp")
        if timestamp:
            try:
                date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                daily_counts[date] = daily_counts.get(date, 0) + 1
            except:
                pass
    
    total = len(interactions)
    days_active = len(daily_counts)
    avg_per_day = total / days_active if days_active > 0 else 0
    
    most_active_day = max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else None
    
    # Determine engagement level
    if avg_per_day >= 3:
        engagement_level = "high"
    elif avg_per_day >= 1:
        engagement_level = "medium"
    else:
        engagement_level = "low"
    
    return {
        "total_interactions": total,
        "days_active": days_active,
        "avg_per_day": round(avg_per_day, 2),
        "most_active_day": str(most_active_day) if most_active_day else None,
        "engagement_level": engagement_level
    }


def _analyze_time_patterns(
    interactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze time-based patterns
    """
    if not interactions:
        return {
            "peak_hour": None,
            "peak_day": None,
            "time_distribution": {}
        }
    
    hour_counts = {}
    day_counts = {}
    
    for interaction in interactions:
        timestamp = interaction.get("timestamp")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = dt.hour
                day = dt.strftime("%A")
                
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
                day_counts[day] = day_counts.get(day, 0) + 1
            except:
                pass
    
    peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None
    peak_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else None
    
    return {
        "peak_hour": peak_hour,
        "peak_day": peak_day,
        "hour_distribution": hour_counts,
        "day_distribution": day_counts
    }


@router.get("/{user_id}")
async def get_analytics(
    user_id: str,
    session_id: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
) -> AnalyticsResponse:
    """
    Get analytics for a user
    
    Args:
        user_id: User ID
        session_id: Optional session ID filter
    """
    try:
        # Get interactions
        if session_id:
            interactions_query = """
                SELECT * FROM student_interactions
                WHERE user_id = ? AND session_id = ?
                ORDER BY timestamp DESC
            """
            interactions_params = (user_id, session_id)
        else:
            interactions_query = """
                SELECT * FROM student_interactions
                WHERE user_id = ?
                ORDER BY timestamp DESC
            """
            interactions_params = (user_id,)
        
        interactions = db.execute_query(interactions_query, interactions_params)
        
        # Get feedback
        if session_id:
            feedback_query = """
                SELECT sf.*, si.query, si.original_response
                FROM student_feedback sf
                JOIN student_interactions si ON sf.interaction_id = si.interaction_id
                WHERE sf.user_id = ? AND sf.session_id = ?
                ORDER BY sf.timestamp DESC
            """
            feedback_params = (user_id, session_id)
        else:
            feedback_query = """
                SELECT sf.*, si.query, si.original_response
                FROM student_feedback sf
                JOIN student_interactions si ON sf.interaction_id = si.interaction_id
                WHERE sf.user_id = ?
                ORDER BY sf.timestamp DESC
            """
            feedback_params = (user_id,)
        
        feedback = db.execute_query(feedback_query, feedback_params)
        
        # Get profile
        try:
            profile_result = await get_profile(user_id, session_id, db)
            profile_dict = profile_result.dict() if hasattr(profile_result, 'dict') else profile_result
        except:
            profile_dict = {}
        
        # Calculate metrics
        total_interactions = len(interactions)
        total_feedback = len(feedback)
        
        avg_understanding = profile_dict.get("average_understanding")
        avg_satisfaction = profile_dict.get("average_satisfaction")
        
        # Calculate improvement trend
        improvement_trend = _calculate_improvement_trend(interactions, feedback)
        
        # Detect learning patterns
        learning_patterns = _detect_learning_patterns(interactions, feedback, profile_dict)
        
        # Analyze topic performance
        topic_performance = _analyze_topic_performance(interactions, feedback, profile_dict)
        
        # Calculate engagement metrics
        engagement_metrics = _calculate_engagement_metrics(interactions)
        
        # Analyze time patterns
        time_analysis = _analyze_time_patterns(interactions)
        
        return AnalyticsResponse(
            total_interactions=total_interactions,
            total_feedback=total_feedback,
            average_understanding=float(avg_understanding) if avg_understanding else None,
            average_satisfaction=float(avg_satisfaction) if avg_satisfaction else None,
            improvement_trend=improvement_trend,
            learning_patterns=learning_patterns,
            topic_performance=topic_performance,
            engagement_metrics=engagement_metrics,
            time_analysis=time_analysis
        )
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.get("/{user_id}/summary")
async def get_analytics_summary(
    user_id: str,
    session_id: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get a summary of analytics (lightweight version)
    """
    try:
        analytics = await get_analytics(user_id, session_id, db)
        
        return {
            "total_interactions": analytics.total_interactions,
            "average_understanding": analytics.average_understanding,
            "improvement_trend": analytics.improvement_trend,
            "engagement_level": analytics.engagement_metrics.get("engagement_level"),
            "key_patterns": analytics.learning_patterns[:3]  # Top 3 patterns
        }
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics summary: {str(e)}"
        )

