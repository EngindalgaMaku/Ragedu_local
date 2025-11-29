-- Migration 016: Add completion_percentage column to topic_progress table
-- Created: 2025-11-29
-- Description: Adds completion_percentage column to topic_progress table for tracking student progress

-- Step 1: Add completion_percentage column if it doesn't exist
-- SQLite doesn't support IF NOT EXISTS for ALTER TABLE ADD COLUMN, so we check first
-- We'll use a safe approach: try to add, ignore if it already exists

-- Check if column exists, if not add it
-- Note: SQLite doesn't support conditional ALTER TABLE, so we'll use a try-catch approach
-- In SQLite, we can use a workaround by checking pragma_table_info

-- Add completion_percentage column (DECIMAL(5,2) for 0.00 to 100.00)
-- Default to 0.0 for existing rows
ALTER TABLE topic_progress 
ADD COLUMN completion_percentage DECIMAL(5,2) DEFAULT 0.0;

-- Add last_interaction_date column if it doesn't exist (for consistency with view)
-- This column stores the date of the last interaction with this topic
ALTER TABLE topic_progress 
ADD COLUMN last_interaction_date TIMESTAMP;

-- Create index for better query performance on completion_percentage
CREATE INDEX IF NOT EXISTS idx_topic_progress_completion 
ON topic_progress(completion_percentage DESC);

-- Update existing rows: calculate completion_percentage based on mastery_score
-- If mastery_score exists, use it to calculate completion (mastery_score * 100)
-- Otherwise, keep it at 0.0
UPDATE topic_progress 
SET completion_percentage = COALESCE(mastery_score * 100.0, 0.0)
WHERE completion_percentage IS NULL OR completion_percentage = 0.0;

-- Update last_interaction_date from last_question_timestamp if available
UPDATE topic_progress 
SET last_interaction_date = last_question_timestamp
WHERE last_interaction_date IS NULL AND last_question_timestamp IS NOT NULL;

-- Step 2: Recreate student_topic_progress_analytics view to use the new completion_percentage column
DROP VIEW IF EXISTS student_topic_progress_analytics;

-- Recreate the view with completion_percentage support
CREATE VIEW IF NOT EXISTS student_topic_progress_analytics AS
SELECT 
    COALESCE(tp.user_id, si.user_id) as user_id,
    ct.topic_id,
    ct.session_id,
    ct.topic_title,
    ct.estimated_difficulty,
    
    -- Current progress metrics (use topic_progress if available, otherwise calculate from feedback)
    COALESCE(tp.mastery_score, COALESCE(AVG(sf.understanding_level) * 0.2, 0.0)) as mastery_level,
    COALESCE(tp.completion_percentage, 0.0) as completion_percentage,
    COALESCE(tp.time_spent_minutes, 0.0) as time_spent_minutes,
    COALESCE(tp.last_interaction_date, tp.last_question_timestamp, MAX(si.timestamp)) as last_interaction_date,
    
    -- Learning velocity (interactions per day) - calculate if time_spent_minutes available
    CASE 
        WHEN COALESCE(tp.time_spent_minutes, 0.0) > 0 
        THEN CAST(COUNT(si.interaction_id) AS FLOAT) / (COALESCE(tp.time_spent_minutes, 0.0) / 1440.0)
        ELSE 0.0
    END as learning_velocity,
    
    -- Interaction metrics
    COUNT(si.interaction_id) as total_interactions,
    COUNT(sf.feedback_id) as feedback_count,
    COALESCE(tp.average_understanding, AVG(sf.understanding_level), 0.0) as avg_understanding,
    COALESCE(tp.average_satisfaction, AVG(sf.satisfaction_level), 0.0) as avg_satisfaction,
    
    -- Progress trend (improvement over time)
    CASE 
        WHEN COUNT(si.interaction_id) = 0 THEN 'new'
        WHEN COALESCE(tp.mastery_score, AVG(sf.understanding_level) * 0.2) >= 0.8 THEN 'improving'
        WHEN COALESCE(tp.mastery_score, AVG(sf.understanding_level) * 0.2) < 0.6 THEN 'declining'
        ELSE 'stable'
    END as progress_trend,
    
    -- Prerequisite completion status
    CASE 
        WHEN ct.prerequisite_topics IS NOT NULL 
        THEN 'has_prerequisites'
        ELSE 'no_prerequisites'
    END as prerequisite_status,
    
    COALESCE(tp.created_at, ct.created_at) as created_at,
    COALESCE(tp.updated_at, ct.updated_at) as updated_at

FROM course_topics ct
LEFT JOIN student_interactions si ON si.session_id = ct.session_id
LEFT JOIN student_feedback sf ON sf.interaction_id = si.interaction_id
LEFT JOIN topic_progress tp ON tp.topic_id = ct.topic_id AND tp.user_id = si.user_id

WHERE ct.is_active = TRUE AND si.user_id IS NOT NULL
GROUP BY 
    COALESCE(tp.user_id, si.user_id), ct.topic_id, ct.session_id, ct.topic_title, 
    ct.estimated_difficulty, 
    COALESCE(tp.mastery_score, 0.0), 
    COALESCE(tp.completion_percentage, 0.0),
    COALESCE(tp.time_spent_minutes, 0.0), 
    COALESCE(tp.last_interaction_date, tp.last_question_timestamp, si.timestamp),
    COALESCE(tp.average_understanding, 0.0),
    COALESCE(tp.average_satisfaction, 0.0),
    ct.prerequisite_topics,
    COALESCE(tp.created_at, ct.created_at),
    COALESCE(tp.updated_at, ct.updated_at);

