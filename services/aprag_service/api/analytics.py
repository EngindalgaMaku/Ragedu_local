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


# ============================================================================
# TOPIC ANALYTICS ENDPOINTS
# ============================================================================

class TopicAnalyticsResponse(BaseModel):
    """Response model for topic analytics"""
    session_id: str
    topics: List[Dict[str, Any]]
    summary: Dict[str, Any]
    heatmap_data: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]


@router.get("/topics/mastery-overview/{session_id}")
async def get_topic_mastery_overview(
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get comprehensive topic mastery overview for a session
    
    Returns heatmap data showing student performance across topics
    """
    try:
        # Get topic mastery analytics from view
        query = """
            SELECT
                topic_id, topic_title, estimated_difficulty,
                students_attempted, total_interactions,
                avg_mastery_level, avg_understanding, avg_satisfaction,
                performance_level, difficulty_appropriateness,
                has_knowledge_base, qa_pairs_count,
                first_interaction, last_interaction
            FROM topic_mastery_analytics
            WHERE session_id = ?
            ORDER BY topic_id
        """
        
        mastery_data = db.execute_query(query, (session_id,))
        
        # Calculate summary statistics
        total_topics = len(mastery_data)
        topics_with_students = len([t for t in mastery_data if t['students_attempted'] > 0])
        
        avg_mastery = sum(t['avg_mastery_level'] for t in mastery_data) / total_topics if total_topics > 0 else 0
        avg_understanding = sum(t['avg_understanding'] for t in mastery_data) / total_topics if total_topics > 0 else 0
        
        # Performance distribution
        performance_dist = {}
        for topic in mastery_data:
            level = topic['performance_level'] or 'unknown'
            performance_dist[level] = performance_dist.get(level, 0) + 1
        
        # Create heatmap data structure
        heatmap_data = []
        for topic in mastery_data:
            heatmap_data.append({
                'topic_id': topic['topic_id'],
                'topic_title': topic['topic_title'],
                'difficulty': topic['estimated_difficulty'],
                'students_attempted': topic['students_attempted'],
                'mastery_score': round(topic['avg_mastery_level'], 2),
                'understanding_score': round(topic['avg_understanding'], 2),
                'satisfaction_score': round(topic['avg_satisfaction'], 2),
                'performance_level': topic['performance_level'],
                'status': 'needs_attention' if topic['avg_understanding'] < 3.0 else 'good'
            })
        
        summary = {
            'total_topics': total_topics,
            'topics_with_engagement': topics_with_students,
            'engagement_rate': round(topics_with_students / total_topics * 100, 1) if total_topics > 0 else 0,
            'avg_mastery_level': round(avg_mastery, 2),
            'avg_understanding': round(avg_understanding, 2),
            'performance_distribution': performance_dist,
            'topics_needing_attention': len([t for t in mastery_data if t['avg_understanding'] < 3.0])
        }
        
        return {
            'success': True,
            'session_id': session_id,
            'summary': summary,
            'topics': mastery_data,
            'heatmap_data': heatmap_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get topic mastery overview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic mastery overview: {str(e)}"
        )


@router.get("/topics/student-progress/{user_id}")
async def get_student_topic_progress(
    user_id: str,
    session_id: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get individual student progress across topics
    """
    try:
        # Build query with optional session filter
        base_query = """
            SELECT
                user_id, topic_id, session_id, topic_title, estimated_difficulty,
                mastery_level, completion_percentage, time_spent_minutes,
                learning_velocity, total_interactions, feedback_count,
                avg_understanding, avg_satisfaction, progress_trend,
                last_interaction_date, prerequisite_status
            FROM student_topic_progress_analytics
            WHERE user_id = ?
        """
        
        params = [user_id]
        if session_id:
            base_query += " AND session_id = ?"
            params.append(session_id)
            
        base_query += " ORDER BY topic_id"
        
        progress_data = db.execute_query(base_query, tuple(params))
        
        # Calculate learning trends
        trends = {
            'improving': len([p for p in progress_data if p['progress_trend'] == 'improving']),
            'stable': len([p for p in progress_data if p['progress_trend'] == 'stable']),
            'declining': len([p for p in progress_data if p['progress_trend'] == 'declining']),
            'new': len([p for p in progress_data if p['progress_trend'] == 'new'])
        }
        
        # Mastery distribution
        mastery_dist = {}
        for progress in progress_data:
            level = progress['mastery_level'] or 0
            range_key = f"{int(level)}-{int(level)+1}"
            mastery_dist[range_key] = mastery_dist.get(range_key, 0) + 1
        
        # Topics needing attention
        weak_topics = [
            p for p in progress_data
            if p['avg_understanding'] and p['avg_understanding'] < 3.0
        ]
        
        strong_topics = [
            p for p in progress_data
            if p['avg_understanding'] and p['avg_understanding'] >= 4.0
        ]
        
        return {
            'success': True,
            'user_id': user_id,
            'session_id': session_id,
            'progress_data': progress_data,
            'summary': {
                'total_topics': len(progress_data),
                'avg_mastery': round(sum(p['mastery_level'] or 0 for p in progress_data) / len(progress_data), 2) if progress_data else 0,
                'trends': trends,
                'mastery_distribution': mastery_dist,
                'weak_topics_count': len(weak_topics),
                'strong_topics_count': len(strong_topics),
                'total_time_spent': sum(p['time_spent_minutes'] or 0 for p in progress_data)
            },
            'weak_topics': weak_topics[:5],  # Top 5 weakest
            'strong_topics': strong_topics[:5]  # Top 5 strongest
        }
        
    except Exception as e:
        logger.error(f"Failed to get student topic progress: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get student topic progress: {str(e)}"
        )


@router.get("/topics/difficulty-analysis")
async def get_topic_difficulty_analysis(
    session_id: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
):
    """
    Analyze topic difficulty vs student performance correlation
    """
    try:
        # Build query with optional session filter
        base_query = """
            SELECT
                topic_id, session_id, topic_title, estimated_difficulty,
                total_questions, avg_question_complexity, students_attempted,
                avg_understanding_score, avg_satisfaction_score, success_rate,
                avg_time_spent, kb_quality_score, available_qa_pairs,
                difficulty_recommendation
            FROM topic_difficulty_analysis
        """
        
        params = []
        if session_id:
            base_query += " WHERE session_id = ?"
            params.append(session_id)
            
        base_query += " ORDER BY estimated_difficulty, avg_understanding_score"
        
        difficulty_data = db.execute_query(base_query, tuple(params))
        
        # Analyze difficulty appropriateness
        difficulty_analysis = {
            'beginner': {'appropriate': 0, 'too_hard': 0, 'too_easy': 0},
            'intermediate': {'appropriate': 0, 'too_hard': 0, 'too_easy': 0},
            'advanced': {'appropriate': 0, 'too_hard': 0, 'too_easy': 0}
        }
        
        recommendations = []
        
        for topic in difficulty_data:
            difficulty = topic['estimated_difficulty']
            understanding = topic['avg_understanding_score']
            
            if difficulty in difficulty_analysis:
                if understanding < 2.5:
                    difficulty_analysis[difficulty]['too_hard'] += 1
                elif understanding > 4.5:
                    difficulty_analysis[difficulty]['too_easy'] += 1
                else:
                    difficulty_analysis[difficulty]['appropriate'] += 1
            
            # Generate specific recommendations
            if topic['difficulty_recommendation'] and topic['difficulty_recommendation'] != 'appropriate_level':
                recommendations.append({
                    'topic_id': topic['topic_id'],
                    'topic_title': topic['topic_title'],
                    'current_difficulty': topic['estimated_difficulty'],
                    'recommendation': topic['difficulty_recommendation'],
                    'understanding_score': topic['avg_understanding_score'],
                    'student_count': topic['students_attempted'],
                    'priority': 'high' if topic['avg_understanding_score'] < 2.5 else 'medium'
                })
        
        # Sort recommendations by priority
        recommendations.sort(key=lambda x: (x['priority'] == 'high', -x['understanding_score']))
        
        return {
            'success': True,
            'session_id': session_id,
            'difficulty_data': difficulty_data,
            'difficulty_analysis': difficulty_analysis,
            'recommendations': recommendations,
            'summary': {
                'total_topics_analyzed': len(difficulty_data),
                'topics_needing_adjustment': len(recommendations),
                'avg_success_rate': round(sum(t['success_rate'] or 0 for t in difficulty_data) / len(difficulty_data), 2) if difficulty_data else 0,
                'topics_with_insufficient_data': len([t for t in difficulty_data if t['students_attempted'] < 3])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get topic difficulty analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic difficulty analysis: {str(e)}"
        )


@router.get("/topics/recommendation-insights")
async def get_topic_recommendation_insights(
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get AI-powered topic recommendations and intervention strategies
    """
    try:
        # Get recommendation insights from view
        query = """
            SELECT
                topic_id, topic_title, estimated_difficulty,
                topic_strength_level, engagement_level,
                recommended_intervention, learning_path_suggestion,
                student_count, avg_understanding, avg_satisfaction,
                qa_pairs_count, kb_quality
            FROM topic_recommendation_insights
            WHERE session_id = ?
            ORDER BY
                CASE topic_strength_level
                    WHEN 'critical_weakness' THEN 1
                    WHEN 'moderate_weakness' THEN 2
                    ELSE 3
                END,
                avg_understanding ASC
        """
        
        insights = db.execute_query(query, (session_id,))
        
        # Categorize recommendations
        critical_topics = [i for i in insights if i['topic_strength_level'] == 'critical_weakness']
        moderate_issues = [i for i in insights if i['topic_strength_level'] == 'moderate_weakness']
        strong_topics = [i for i in insights if i['topic_strength_level'] == 'strength']
        
        # Prioritized intervention strategies
        interventions = {}
        for insight in insights:
            intervention = insight['recommended_intervention']
            if intervention not in interventions:
                interventions[intervention] = []
            interventions[intervention].append({
                'topic_id': insight['topic_id'],
                'topic_title': insight['topic_title'],
                'priority': insight['topic_strength_level'],
                'student_count': insight['student_count'],
                'understanding_score': insight['avg_understanding']
            })
        
        # Learning path recommendations
        learning_paths = {}
        for insight in insights:
            path = insight['learning_path_suggestion']
            if path not in learning_paths:
                learning_paths[path] = []
            learning_paths[path].append({
                'topic_id': insight['topic_id'],
                'topic_title': insight['topic_title'],
                'current_difficulty': insight['estimated_difficulty'],
                'understanding_score': insight['avg_understanding']
            })
        
        # Overall session health
        session_health = {
            'critical_topics': len(critical_topics),
            'moderate_issues': len(moderate_issues),
            'strong_topics': len(strong_topics),
            'overall_score': round((len(strong_topics) * 3 + len(moderate_issues) * 2 + len(critical_topics) * 1) / len(insights) if insights else 0, 1)
        }
        
        return {
            'success': True,
            'session_id': session_id,
            'insights': insights,
            'session_health': session_health,
            'critical_topics': critical_topics,
            'moderate_issues': moderate_issues,
            'strong_topics': strong_topics,
            'intervention_strategies': interventions,
            'learning_path_recommendations': learning_paths,
            'action_items': {
                'immediate_attention': [t['topic_title'] for t in critical_topics[:3]],
                'monitor_closely': [t['topic_title'] for t in moderate_issues[:3]],
                'leverage_strengths': [t['topic_title'] for t in strong_topics[:3]]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get topic recommendation insights: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic recommendation insights: {str(e)}"
        )


@router.get("/topics/session-overview/{session_id}")
async def get_session_topic_overview(
    session_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get comprehensive topic analytics overview for a session
    Combines all analytics into a single dashboard-ready response
    """
    try:
        # Get mastery overview
        mastery_data = await get_topic_mastery_overview(session_id, db)
        
        # Get difficulty analysis
        difficulty_data = await get_topic_difficulty_analysis(session_id, db)
        
        # Get recommendations
        recommendations_data = await get_topic_recommendation_insights(session_id, db)
        
        # Combine into comprehensive overview
        overview = {
            'success': True,
            'session_id': session_id,
            'generated_at': datetime.utcnow().isoformat(),
            
            # Mastery analytics
            'mastery_overview': mastery_data,
            
            # Difficulty analysis
            'difficulty_analysis': difficulty_data,
            
            # AI recommendations
            'recommendations': recommendations_data,
            
            # Executive summary
            'executive_summary': {
                'total_topics': mastery_data['summary']['total_topics'],
                'engagement_rate': mastery_data['summary']['engagement_rate'],
                'avg_understanding': mastery_data['summary']['avg_understanding'],
                'topics_needing_attention': mastery_data['summary']['topics_needing_attention'],
                'session_health_score': recommendations_data['session_health']['overall_score'],
                'immediate_actions_needed': len(recommendations_data['critical_topics']),
                'key_strengths': len(recommendations_data['strong_topics']),
                'overall_status': (
                    'excellent' if recommendations_data['session_health']['overall_score'] >= 2.5
                    else 'good' if recommendations_data['session_health']['overall_score'] >= 2.0
                    else 'needs_improvement'
                )
            }
        }
        
        return overview
        
    except Exception as e:
        logger.error(f"Failed to get session topic overview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session topic overview: {str(e)}"
        )

