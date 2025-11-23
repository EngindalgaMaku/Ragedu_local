-- Topic Analytics Database Views
-- Creates comprehensive views for topic-level analytics dashboard

-- ============================================================================
-- TOPIC MASTERY ANALYTICS VIEW
-- ============================================================================
CREATE VIEW IF NOT EXISTS topic_mastery_analytics AS
SELECT 
    ct.topic_id,
    ct.session_id,
    ct.topic_title,
    ct.description,
    ct.estimated_difficulty,
    ct.keywords,
    -- Student engagement metrics
    COUNT(DISTINCT si.user_id) as students_attempted,
    COUNT(si.interaction_id) as total_interactions,
    
    -- Mastery level metrics (from topic_progress if available)
    COALESCE(AVG(tp.mastery_level), 0.0) as avg_mastery_level,
    COALESCE(MIN(tp.mastery_level), 0.0) as min_mastery_level,
    COALESCE(MAX(tp.mastery_level), 0.0) as max_mastery_level,
    
    -- Feedback metrics
    COALESCE(AVG(sf.understanding_level), 0.0) as avg_understanding,
    COALESCE(AVG(sf.satisfaction_level), 0.0) as avg_satisfaction,
    COUNT(sf.feedback_id) as total_feedback_count,
    
    -- Question-topic mapping reliability
    COALESCE(AVG(qtm.confidence_score), 0.0) as avg_topic_confidence,
    COALESCE(AVG(qtm.question_complexity), 0.0) as avg_question_complexity,
    COUNT(qtm.mapping_id) as mapped_questions_count,
    
    -- Knowledge base metrics
    CASE WHEN tkb.knowledge_id IS NOT NULL THEN 1 ELSE 0 END as has_knowledge_base,
    tkb.content_quality_score,
    COUNT(tqa.qa_id) as qa_pairs_count,
    
    -- Time-based metrics
    MIN(si.timestamp) as first_interaction,
    MAX(si.timestamp) as last_interaction,
    
    -- Performance indicators
    CASE 
        WHEN AVG(sf.understanding_level) >= 4.0 THEN 'high'
        WHEN AVG(sf.understanding_level) >= 3.0 THEN 'medium'
        WHEN AVG(sf.understanding_level) >= 2.0 THEN 'low'
        ELSE 'very_low'
    END as performance_level,
    
    -- Difficulty vs Success correlation
    CASE 
        WHEN ct.estimated_difficulty = 'beginner' AND AVG(sf.understanding_level) >= 4.0 THEN 'appropriate'
        WHEN ct.estimated_difficulty = 'intermediate' AND AVG(sf.understanding_level) >= 3.5 THEN 'appropriate'
        WHEN ct.estimated_difficulty = 'advanced' AND AVG(sf.understanding_level) >= 3.0 THEN 'appropriate'
        WHEN AVG(sf.understanding_level) < 2.0 THEN 'too_difficult'
        WHEN AVG(sf.understanding_level) >= 4.5 THEN 'too_easy'
        ELSE 'needs_review'
    END as difficulty_appropriateness,
    
    ct.created_at,
    ct.updated_at

FROM course_topics ct
LEFT JOIN student_interactions si ON si.session_id = ct.session_id
LEFT JOIN question_topic_mapping qtm ON qtm.topic_id = ct.topic_id 
    AND qtm.interaction_id = si.interaction_id
LEFT JOIN student_feedback sf ON sf.interaction_id = si.interaction_id
LEFT JOIN topic_progress tp ON tp.topic_id = ct.topic_id AND tp.user_id = si.user_id
LEFT JOIN topic_knowledge_base tkb ON tkb.topic_id = ct.topic_id
LEFT JOIN topic_qa_pairs tqa ON tqa.topic_id = ct.topic_id AND tqa.is_active = TRUE

WHERE ct.is_active = TRUE
GROUP BY 
    ct.topic_id, ct.session_id, ct.topic_title, ct.description, 
    ct.estimated_difficulty, ct.keywords, ct.created_at, ct.updated_at,
    tkb.knowledge_id, tkb.content_quality_score;

-- ============================================================================
-- STUDENT TOPIC PROGRESS VIEW
-- ============================================================================
CREATE VIEW IF NOT EXISTS student_topic_progress_analytics AS
SELECT 
    tp.user_id,
    tp.topic_id,
    ct.session_id,
    ct.topic_title,
    ct.estimated_difficulty,
    
    -- Current progress metrics
    tp.mastery_level,
    tp.completion_percentage,
    tp.time_spent_minutes,
    tp.last_interaction_date,
    
    -- Learning velocity (interactions per day)
    CASE 
        WHEN tp.time_spent_minutes > 0 
        THEN CAST(COUNT(si.interaction_id) AS FLOAT) / (tp.time_spent_minutes / 1440.0)
        ELSE 0.0
    END as learning_velocity,
    
    -- Interaction metrics
    COUNT(si.interaction_id) as total_interactions,
    COUNT(sf.feedback_id) as feedback_count,
    COALESCE(AVG(sf.understanding_level), 0.0) as avg_understanding,
    COALESCE(AVG(sf.satisfaction_level), 0.0) as avg_satisfaction,
    
    -- Progress trend (improvement over time)
    CASE 
        WHEN LAG(tp.mastery_level) OVER (
            PARTITION BY tp.user_id, tp.topic_id 
            ORDER BY tp.updated_at
        ) IS NULL THEN 'new'
        WHEN tp.mastery_level > LAG(tp.mastery_level) OVER (
            PARTITION BY tp.user_id, tp.topic_id 
            ORDER BY tp.updated_at
        ) THEN 'improving'
        WHEN tp.mastery_level < LAG(tp.mastery_level) OVER (
            PARTITION BY tp.user_id, tp.topic_id 
            ORDER BY tp.updated_at
        ) THEN 'declining'
        ELSE 'stable'
    END as progress_trend,
    
    -- Prerequisite completion status
    CASE 
        WHEN ct.prerequisite_topics IS NOT NULL 
        THEN 'has_prerequisites'
        ELSE 'no_prerequisites'
    END as prerequisite_status,
    
    tp.created_at,
    tp.updated_at

FROM topic_progress tp
JOIN course_topics ct ON ct.topic_id = tp.topic_id
LEFT JOIN student_interactions si ON si.user_id = tp.user_id 
    AND si.session_id = ct.session_id
LEFT JOIN student_feedback sf ON sf.interaction_id = si.interaction_id
    AND sf.user_id = tp.user_id

WHERE ct.is_active = TRUE
GROUP BY 
    tp.user_id, tp.topic_id, ct.session_id, ct.topic_title, 
    ct.estimated_difficulty, tp.mastery_level, tp.completion_percentage,
    tp.time_spent_minutes, tp.last_interaction_date, ct.prerequisite_topics,
    tp.created_at, tp.updated_at;

-- ============================================================================
-- TOPIC DIFFICULTY ANALYSIS VIEW
-- ============================================================================
CREATE VIEW IF NOT EXISTS topic_difficulty_analysis AS
SELECT 
    ct.topic_id,
    ct.session_id,
    ct.topic_title,
    ct.estimated_difficulty,
    
    -- Question complexity metrics
    COUNT(qtm.mapping_id) as total_questions,
    COALESCE(AVG(qtm.question_complexity), 0.0) as avg_question_complexity,
    COALESCE(AVG(qtm.confidence_score), 0.0) as avg_mapping_confidence,
    
    -- Student performance metrics
    COUNT(DISTINCT si.user_id) as students_attempted,
    COALESCE(AVG(sf.understanding_level), 0.0) as avg_understanding_score,
    COALESCE(AVG(sf.satisfaction_level), 0.0) as avg_satisfaction_score,
    
    -- Success rate by difficulty
    CASE 
        WHEN COUNT(sf.feedback_id) > 0 
        THEN CAST(COUNT(CASE WHEN sf.understanding_level >= 4.0 THEN 1 END) AS FLOAT) / COUNT(sf.feedback_id)
        ELSE 0.0
    END as success_rate,
    
    -- Time investment
    COALESCE(AVG(tp.time_spent_minutes), 0.0) as avg_time_spent,
    COALESCE(AVG(si.processing_time_ms), 0.0) as avg_response_time,
    
    -- Knowledge base quality correlation
    COALESCE(tkb.content_quality_score, 0.0) as kb_quality_score,
    COUNT(tqa.qa_id) as available_qa_pairs,
    
    -- Recommendation flags
    CASE 
        WHEN AVG(sf.understanding_level) < 2.5 AND ct.estimated_difficulty = 'beginner' 
        THEN 'needs_simplification'
        WHEN AVG(sf.understanding_level) > 4.5 AND ct.estimated_difficulty = 'advanced'
        THEN 'can_increase_difficulty'
        WHEN AVG(sf.understanding_level) < 3.0 
        THEN 'needs_support_materials'
        WHEN COUNT(sf.feedback_id) < 5 
        THEN 'insufficient_data'
        ELSE 'appropriate_level'
    END as difficulty_recommendation,
    
    ct.created_at,
    ct.updated_at

FROM course_topics ct
LEFT JOIN question_topic_mapping qtm ON qtm.topic_id = ct.topic_id
LEFT JOIN student_interactions si ON si.session_id = ct.session_id
    AND si.interaction_id = qtm.interaction_id
LEFT JOIN student_feedback sf ON sf.interaction_id = si.interaction_id
LEFT JOIN topic_progress tp ON tp.topic_id = ct.topic_id
LEFT JOIN topic_knowledge_base tkb ON tkb.topic_id = ct.topic_id
LEFT JOIN topic_qa_pairs tqa ON tqa.topic_id = ct.topic_id AND tqa.is_active = TRUE

WHERE ct.is_active = TRUE
GROUP BY 
    ct.topic_id, ct.session_id, ct.topic_title, ct.estimated_difficulty,
    tkb.content_quality_score, ct.created_at, ct.updated_at;

-- ============================================================================
-- TOPIC RECOMMENDATION INSIGHTS VIEW
-- ============================================================================
CREATE VIEW IF NOT EXISTS topic_recommendation_insights AS
SELECT 
    ct.session_id,
    ct.topic_id,
    ct.topic_title,
    ct.estimated_difficulty,
    
    -- Weakness identification
    CASE 
        WHEN AVG(sf.understanding_level) < 2.5 THEN 'critical_weakness'
        WHEN AVG(sf.understanding_level) < 3.5 THEN 'moderate_weakness' 
        ELSE 'strength'
    END as topic_strength_level,
    
    -- Student engagement level
    CASE 
        WHEN COUNT(DISTINCT si.user_id) = 0 THEN 'no_engagement'
        WHEN COUNT(DISTINCT si.user_id) <= 2 THEN 'low_engagement'
        WHEN COUNT(DISTINCT si.user_id) <= 5 THEN 'medium_engagement'
        ELSE 'high_engagement'
    END as engagement_level,
    
    -- Intervention recommendations
    CASE 
        WHEN AVG(sf.understanding_level) < 2.5 AND COUNT(tqa.qa_id) < 5
        THEN 'add_more_practice_questions'
        WHEN AVG(sf.understanding_level) < 3.0 AND tkb.content_quality_score < 0.7
        THEN 'improve_knowledge_base'
        WHEN COUNT(DISTINCT si.user_id) <= 2 AND ct.estimated_difficulty = 'beginner'
        THEN 'make_more_accessible'
        WHEN AVG(sf.satisfaction_level) < 3.0
        THEN 'review_teaching_approach'
        WHEN AVG(sf.understanding_level) >= 4.5
        THEN 'add_advanced_content'
        ELSE 'maintain_current_approach'
    END as recommended_intervention,
    
    -- Optimal learning path suggestions
    CASE 
        WHEN ct.estimated_difficulty = 'beginner' AND AVG(sf.understanding_level) >= 4.0
        THEN 'ready_for_intermediate'
        WHEN ct.estimated_difficulty = 'intermediate' AND AVG(sf.understanding_level) >= 4.0
        THEN 'ready_for_advanced'
        WHEN AVG(sf.understanding_level) < 3.0
        THEN 'needs_prerequisite_review'
        ELSE 'continue_current_level'
    END as learning_path_suggestion,
    
    -- Supporting metrics
    COUNT(DISTINCT si.user_id) as student_count,
    COALESCE(AVG(sf.understanding_level), 0.0) as avg_understanding,
    COALESCE(AVG(sf.satisfaction_level), 0.0) as avg_satisfaction,
    COUNT(tqa.qa_id) as qa_pairs_count,
    COALESCE(tkb.content_quality_score, 0.0) as kb_quality,
    
    CURRENT_TIMESTAMP as analysis_timestamp

FROM course_topics ct
LEFT JOIN student_interactions si ON si.session_id = ct.session_id
LEFT JOIN question_topic_mapping qtm ON qtm.topic_id = ct.topic_id 
    AND qtm.interaction_id = si.interaction_id
LEFT JOIN student_feedback sf ON sf.interaction_id = si.interaction_id
LEFT JOIN topic_knowledge_base tkb ON tkb.topic_id = ct.topic_id
LEFT JOIN topic_qa_pairs tqa ON tqa.topic_id = ct.topic_id AND tqa.is_active = TRUE

WHERE ct.is_active = TRUE
GROUP BY 
    ct.session_id, ct.topic_id, ct.topic_title, ct.estimated_difficulty,
    tkb.content_quality_score;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================
-- Note: These would be created on the actual tables, not views
-- Including here for reference

-- CREATE INDEX IF NOT EXISTS idx_student_interactions_session_user ON student_interactions(session_id, user_id);
-- CREATE INDEX IF NOT EXISTS idx_student_feedback_interaction ON student_feedback(interaction_id);
-- CREATE INDEX IF NOT EXISTS idx_topic_progress_user_topic ON topic_progress(user_id, topic_id);
-- CREATE INDEX IF NOT EXISTS idx_question_topic_mapping_topic ON question_topic_mapping(topic_id);
-- CREATE INDEX IF NOT EXISTS idx_course_topics_session ON course_topics(session_id, is_active);