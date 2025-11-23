-- Migration 006: Add multi-dimensional feedback support
-- Created: 2025-11-23
-- Description: Extends student_interactions table with feedback_dimensions JSON field
--              for Understanding, Relevance, and Clarity assessment

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- Add feedback_dimensions column to student_interactions table
-- ============================================================================
-- This will store JSON data: {"understanding": 4, "relevance": 5, "clarity": 3, "emoji": "👍"}

ALTER TABLE student_interactions 
ADD COLUMN feedback_dimensions TEXT;

-- ============================================================================
-- Create multi-dimensional feedback summary table for analytics
-- ============================================================================
CREATE TABLE IF NOT EXISTS multi_feedback_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Dimension averages (1-5 scale)
    avg_understanding DECIMAL(3,2),
    avg_relevance DECIMAL(3,2), 
    avg_clarity DECIMAL(3,2),
    avg_overall DECIMAL(3,2),
    
    -- Feedback counts
    total_feedback_count INTEGER DEFAULT 0,
    dimension_feedback_count INTEGER DEFAULT 0,
    emoji_only_count INTEGER DEFAULT 0,
    
    -- Dimension distributions (JSON strings)
    understanding_distribution TEXT, -- JSON: {"1": 2, "2": 1, "3": 5, "4": 8, "5": 4}
    relevance_distribution TEXT,
    clarity_distribution TEXT,
    
    -- Trend analysis
    improvement_trend VARCHAR(20), -- improving, stable, declining, insufficient_data
    weak_dimensions TEXT, -- JSON array: ["understanding", "clarity"]
    strong_dimensions TEXT, -- JSON array: ["relevance"]
    
    -- Metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint: one summary per user per session
    UNIQUE(user_id, session_id),
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- Create detailed feedback entries table
-- ============================================================================
CREATE TABLE IF NOT EXISTS detailed_feedback_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Multi-dimensional scores (1-5 scale)
    understanding_score INTEGER CHECK (understanding_score BETWEEN 1 AND 5),
    relevance_score INTEGER CHECK (relevance_score BETWEEN 1 AND 5),
    clarity_score INTEGER CHECK (clarity_score BETWEEN 1 AND 5),
    overall_score DECIMAL(3,2), -- Calculated average
    
    -- Original emoji feedback (for compatibility)
    emoji_feedback VARCHAR(10),
    emoji_score DECIMAL(3,2),
    
    -- Optional comment
    comment TEXT,
    
    -- Feedback type
    feedback_type VARCHAR(20) DEFAULT 'multi_dimensional', -- multi_dimensional, emoji_only
    
    -- Metadata
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Multi feedback summary indexes
CREATE INDEX IF NOT EXISTS idx_multi_feedback_user_session ON multi_feedback_summary(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_multi_feedback_updated ON multi_feedback_summary(last_updated);

-- Detailed feedback entries indexes
CREATE INDEX IF NOT EXISTS idx_detailed_feedback_interaction ON detailed_feedback_entries(interaction_id);
CREATE INDEX IF NOT EXISTS idx_detailed_feedback_user_session ON detailed_feedback_entries(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_detailed_feedback_submitted ON detailed_feedback_entries(submitted_at);
CREATE INDEX IF NOT EXISTS idx_detailed_feedback_type ON detailed_feedback_entries(feedback_type);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Update multi_feedback_summary last_updated timestamp
CREATE TRIGGER IF NOT EXISTS update_multi_feedback_updated_at
    AFTER UPDATE ON multi_feedback_summary
    FOR EACH ROW
    WHEN NEW.last_updated <= OLD.last_updated
BEGIN
    UPDATE multi_feedback_summary SET last_updated = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Auto-update multi_feedback_summary when detailed feedback is added
CREATE TRIGGER IF NOT EXISTS update_multi_summary_on_detailed_feedback
    AFTER INSERT ON detailed_feedback_entries
    FOR EACH ROW
BEGIN
    -- Insert or update summary
    INSERT INTO multi_feedback_summary (
        user_id, session_id, 
        avg_understanding, avg_relevance, avg_clarity, avg_overall,
        total_feedback_count, dimension_feedback_count,
        last_updated
    )
    VALUES (
        NEW.user_id, NEW.session_id,
        NEW.understanding_score, NEW.relevance_score, NEW.clarity_score, NEW.overall_score,
        1, 
        CASE WHEN NEW.feedback_type = 'multi_dimensional' THEN 1 ELSE 0 END,
        CURRENT_TIMESTAMP
    )
    ON CONFLICT(user_id, session_id) DO UPDATE SET
        avg_understanding = (
            (avg_understanding * dimension_feedback_count + NEW.understanding_score) / 
            (dimension_feedback_count + CASE WHEN NEW.feedback_type = 'multi_dimensional' THEN 1 ELSE 0 END)
        ),
        avg_relevance = (
            (avg_relevance * dimension_feedback_count + NEW.relevance_score) / 
            (dimension_feedback_count + CASE WHEN NEW.feedback_type = 'multi_dimensional' THEN 1 ELSE 0 END)
        ),
        avg_clarity = (
            (avg_clarity * dimension_feedback_count + NEW.clarity_score) / 
            (dimension_feedback_count + CASE WHEN NEW.feedback_type = 'multi_dimensional' THEN 1 ELSE 0 END)
        ),
        avg_overall = (
            (avg_overall * total_feedback_count + NEW.overall_score) / 
            (total_feedback_count + 1)
        ),
        total_feedback_count = total_feedback_count + 1,
        dimension_feedback_count = dimension_feedback_count + CASE WHEN NEW.feedback_type = 'multi_dimensional' THEN 1 ELSE 0 END,
        last_updated = CURRENT_TIMESTAMP;
END;

-- ============================================================================
-- INITIAL DATA MIGRATION
-- ============================================================================

-- Migrate existing emoji feedback to new format (optional - preserves existing data)
UPDATE student_interactions 
SET feedback_dimensions = json_object(
    'emoji', emoji_feedback,
    'emoji_score', feedback_score,
    'understanding', NULL,
    'relevance', NULL,
    'clarity', NULL,
    'overall_score', feedback_score,
    'feedback_type', 'emoji_only'
)
WHERE emoji_feedback IS NOT NULL AND feedback_dimensions IS NULL;

-- Verify migration
SELECT 'Multi-dimensional feedback migration completed successfully' as status;