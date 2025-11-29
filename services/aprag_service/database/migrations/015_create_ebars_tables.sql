-- Migration 015: EBARS (Emoji-Based Adaptive Response System) Tables
-- Date: 2025-11-29
-- Description: Create tables for EBARS comprehension score tracking and adaptive response system
-- Dependency: APRAG service must be enabled

-- ===========================================
-- 1. Create student_comprehension_scores table
-- ===========================================
-- This table stores the comprehension score (0-100) for each student-session pair
-- The score is updated based on emoji feedback and used to determine difficulty level

CREATE TABLE IF NOT EXISTS student_comprehension_scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    
    -- Comprehension score (0-100)
    comprehension_score DECIMAL(5,2) NOT NULL DEFAULT 50.0,
    
    -- Current difficulty level (derived from score)
    current_difficulty_level VARCHAR(20) NOT NULL DEFAULT 'normal',
    -- Values: 'very_struggling', 'struggling', 'normal', 'good', 'excellent'
    
    -- Statistics
    total_feedback_count INTEGER DEFAULT 0,
    positive_feedback_count INTEGER DEFAULT 0,
    negative_feedback_count INTEGER DEFAULT 0,
    
    -- Recent feedback pattern tracking (for adaptive adjustments)
    consecutive_positive_count INTEGER DEFAULT 0,
    consecutive_negative_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_feedback_at TIMESTAMP,
    
    -- Unique constraint: one score per user-session pair
    UNIQUE(user_id, session_id)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_comprehension_scores_user_session 
ON student_comprehension_scores(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_comprehension_scores_difficulty 
ON student_comprehension_scores(current_difficulty_level);

CREATE INDEX IF NOT EXISTS idx_comprehension_scores_score 
ON student_comprehension_scores(comprehension_score DESC);

-- ===========================================
-- 2. Create ebars_feedback_history table
-- ===========================================
-- This table stores the history of emoji feedbacks and their impact on comprehension score
-- Used for analysis and debugging

CREATE TABLE IF NOT EXISTS ebars_feedback_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    interaction_id INTEGER,
    
    -- Emoji feedback
    emoji_feedback TEXT NOT NULL,
    -- Values: '👍' (Mükemmel), '😊' (Anladım), '😐' (Karışık), '❌' (Anlamadım)
    
    -- Score changes
    previous_score DECIMAL(5,2) NOT NULL,
    score_delta DECIMAL(5,2) NOT NULL,
    new_score DECIMAL(5,2) NOT NULL,
    
    -- Difficulty level changes
    previous_difficulty_level VARCHAR(20),
    new_difficulty_level VARCHAR(20),
    difficulty_changed BOOLEAN DEFAULT FALSE,
    
    -- Adaptive adjustment info
    adjustment_type VARCHAR(50),
    -- Values: 'proactive_increase', 'reactive_decrease', 'normal_update', 'immediate_drop', 'immediate_raise'
    
    -- Metadata
    query_text TEXT,
    response_preview TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE SET NULL
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_ebars_feedback_user_session 
ON ebars_feedback_history(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_ebars_feedback_timestamp 
ON ebars_feedback_history(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_ebars_feedback_interaction 
ON ebars_feedback_history(interaction_id);

-- ===========================================
-- 3. Create ebars_prompt_cache table
-- ===========================================
-- This table caches the generated prompts for each difficulty level
-- to avoid regenerating them on every request

CREATE TABLE IF NOT EXISTS ebars_prompt_cache (
    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    
    -- Difficulty level this prompt is for
    difficulty_level VARCHAR(20) NOT NULL,
    
    -- Cached prompt parameters (JSON)
    prompt_parameters TEXT NOT NULL,
    
    -- Full prompt text (optional, for debugging)
    full_prompt_text TEXT,
    
    -- Usage statistics
    times_used INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, session_id, difficulty_level)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ebars_prompt_cache_user_session 
ON ebars_prompt_cache(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_ebars_prompt_cache_difficulty 
ON ebars_prompt_cache(difficulty_level);

-- ===========================================
-- Migration complete
-- ===========================================

SELECT 'Migration 015 complete - EBARS tables ready' AS status;

