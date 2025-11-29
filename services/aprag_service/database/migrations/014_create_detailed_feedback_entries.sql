-- Migration 014: Create detailed_feedback_entries table
-- Created: 2025-11-28
-- Description: Creates table for storing multi-dimensional feedback entries

PRAGMA foreign_keys = OFF;

-- Create detailed_feedback_entries table
CREATE TABLE IF NOT EXISTS detailed_feedback_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Multi-dimensional scores (1-5 scale)
    understanding_score INTEGER NOT NULL CHECK(understanding_score >= 1 AND understanding_score <= 5),
    relevance_score INTEGER NOT NULL CHECK(relevance_score >= 1 AND relevance_score <= 5),
    clarity_score INTEGER NOT NULL CHECK(clarity_score >= 1 AND clarity_score <= 5),
    overall_score DECIMAL(3,2) NOT NULL,  -- Average of three dimensions (1-5 scale)
    
    -- Optional emoji feedback
    emoji_feedback TEXT,  -- 😊, 👍, 😐, ❌
    emoji_score DECIMAL(3,2),  -- 0.0 to 1.0
    
    -- Optional comment
    comment TEXT,
    
    -- Feedback type
    feedback_type VARCHAR(50) DEFAULT 'multi_dimensional',  -- multi_dimensional, emoji_only, etc.
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key to interaction (optional, may not exist in all cases)
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE CASCADE
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_detailed_feedback_interaction 
ON detailed_feedback_entries(interaction_id);

CREATE INDEX IF NOT EXISTS idx_detailed_feedback_user_session 
ON detailed_feedback_entries(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_detailed_feedback_type 
ON detailed_feedback_entries(feedback_type);

CREATE INDEX IF NOT EXISTS idx_detailed_feedback_created 
ON detailed_feedback_entries(created_at DESC);

-- Create multi_feedback_summary table for analytics
CREATE TABLE IF NOT EXISTS multi_feedback_summary (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Dimension averages
    avg_understanding DECIMAL(3,2),
    avg_relevance DECIMAL(3,2),
    avg_clarity DECIMAL(3,2),
    avg_overall DECIMAL(3,2),
    
    -- Feedback counts
    total_feedback_count INTEGER DEFAULT 0,
    dimension_feedback_count INTEGER DEFAULT 0,  -- Multi-dimensional feedbacks
    emoji_only_count INTEGER DEFAULT 0,  -- Emoji-only feedbacks
    
    -- Distributions (JSON strings)
    understanding_distribution TEXT,  -- JSON: {"1": 2, "2": 1, "3": 5, "4": 10, "5": 8}
    relevance_distribution TEXT,
    clarity_distribution TEXT,
    
    -- Trend analysis
    improvement_trend VARCHAR(50) DEFAULT 'insufficient_data',  -- improving, stable, declining, insufficient_data
    weak_dimensions TEXT,  -- JSON array: ["relevance", "clarity"]
    strong_dimensions TEXT,  -- JSON array: ["understanding"]
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, session_id)
);

-- Create index for multi_feedback_summary
CREATE INDEX IF NOT EXISTS idx_multi_feedback_summary_user_session 
ON multi_feedback_summary(user_id, session_id);

PRAGMA foreign_keys = ON;

-- Verify tables were created
SELECT 'detailed_feedback_entries and multi_feedback_summary tables created successfully' as status;



