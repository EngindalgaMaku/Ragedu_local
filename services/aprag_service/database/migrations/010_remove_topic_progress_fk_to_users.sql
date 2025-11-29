-- Migration 010: Remove foreign key constraint from topic_progress to users table
-- Created: 2025-11-28
-- Description: Removes the problematic foreign key constraint from topic_progress.user_id to users table
--              This is needed because aprag-service database doesn't have a users table
--              (users table is in auth-service database)

-- Enable foreign key constraints temporarily (will be disabled during migration)
PRAGMA foreign_keys = OFF;

-- Step 1: Create new topic_progress table without foreign key to users
CREATE TABLE IF NOT EXISTS topic_progress_new (
    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,  -- Changed from INTEGER to VARCHAR, no FK constraint
    session_id VARCHAR(255) NOT NULL,
    topic_id INTEGER NOT NULL,
    
    -- Progress metrics
    questions_asked INTEGER DEFAULT 0,
    average_understanding DECIMAL(3,2),  -- From feedback (1-5 scale normalized)
    average_satisfaction DECIMAL(3,2),   -- From feedback (1-5 scale normalized)
    last_question_timestamp TIMESTAMP,
    
    -- Mastery assessment
    mastery_level VARCHAR(20),  -- not_started, learning, mastered, needs_review
    mastery_score DECIMAL(3,2),  -- 0.00 to 1.00
    
    -- Readiness for next topic
    is_ready_for_next BOOLEAN DEFAULT FALSE,
    readiness_score DECIMAL(3,2),  -- 0.00 to 1.00
    
    -- Time tracking
    time_spent_minutes INTEGER DEFAULT 0,
    first_question_timestamp TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, session_id, topic_id),
    -- REMOVED: FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
    -- Only keep FK to course_topics
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
);

-- Step 2: Copy existing data (if any)
-- Handle both old schema (id, average_understanding REAL) and new schema (progress_id, average_understanding DECIMAL)
-- Use explicit column mapping to handle schema differences
-- First check what columns exist in the old table
INSERT INTO topic_progress_new 
(progress_id, user_id, session_id, topic_id, questions_asked, average_understanding, 
 average_satisfaction, last_question_timestamp, mastery_level, mastery_score, 
 is_ready_for_next, readiness_score, time_spent_minutes, first_question_timestamp, 
 created_at, updated_at)
SELECT 
    -- Handle both 'id' and 'progress_id' column names (check which exists)
    CASE 
        WHEN EXISTS (SELECT 1 FROM pragma_table_info('topic_progress') WHERE name='progress_id') 
        THEN progress_id
        WHEN EXISTS (SELECT 1 FROM pragma_table_info('topic_progress') WHERE name='id') 
        THEN id
        ELSE NULL
    END as progress_id,
    CAST(user_id AS VARCHAR(255)) as user_id,  -- Ensure VARCHAR type
    session_id, 
    topic_id, 
    COALESCE(questions_asked, 0) as questions_asked, 
    CAST(COALESCE(average_understanding, 0.0) AS DECIMAL(3,2)) as average_understanding,  -- Convert REAL to DECIMAL
    CAST(COALESCE(average_satisfaction, NULL) AS DECIMAL(3,2)) as average_satisfaction,  -- May not exist in old schema
    COALESCE(last_question_timestamp, first_interaction_timestamp, first_question_timestamp) as last_question_timestamp,  -- Handle different column names
    mastery_level, 
    CAST(COALESCE(mastery_score, 0.0) AS DECIMAL(3,2)) as mastery_score,  -- Handle both column names
    COALESCE(is_ready_for_next, 0) as is_ready_for_next, 
    CAST(COALESCE(readiness_score, 0.0) AS DECIMAL(3,2)) as readiness_score, 
    COALESCE(time_spent_minutes, 0) as time_spent_minutes, 
    COALESCE(first_question_timestamp, first_interaction_timestamp, created_at) as first_question_timestamp,  -- Handle different column names
    COALESCE(created_at, CURRENT_TIMESTAMP) as created_at, 
    COALESCE(updated_at, created_at, CURRENT_TIMESTAMP) as updated_at
FROM topic_progress
WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='topic_progress');

-- Step 3: Drop old table
DROP TABLE IF EXISTS topic_progress;

-- Step 4: Rename new table
ALTER TABLE topic_progress_new RENAME TO topic_progress;

-- Step 5: Recreate indexes
CREATE INDEX IF NOT EXISTS idx_topic_progress_user_topic ON topic_progress(user_id, topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_progress_session ON topic_progress(session_id);
CREATE INDEX IF NOT EXISTS idx_topic_progress_topic ON topic_progress(topic_id);

-- Re-enable foreign key constraints
PRAGMA foreign_keys = ON;


