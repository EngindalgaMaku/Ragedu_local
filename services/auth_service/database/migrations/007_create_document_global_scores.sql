-- Migration 007: Create document_global_scores table for CACS
-- Created: 2025-11-27
-- Description: Stores global feedback scores for documents used in CACS (Conversation-Aware Content Scoring)

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- DOCUMENT GLOBAL SCORES TABLE
-- ============================================================================
-- Stores aggregated feedback scores for documents across all students
-- Used by CACS algorithm to calculate global_score component
CREATE TABLE IF NOT EXISTS document_global_scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id VARCHAR(255) NOT NULL UNIQUE,  -- Document identifier (filename, chunk_id, etc.)
    
    -- Feedback statistics
    total_feedback_count INTEGER DEFAULT 0,
    positive_feedback_count INTEGER DEFAULT 0,  -- Positive feedback (score >= 3)
    negative_feedback_count INTEGER DEFAULT 0,  -- Negative feedback (score < 3)
    
    -- Score aggregations
    average_score DECIMAL(5,3) DEFAULT 0.5,  -- Average feedback score (0-1)
    total_score_sum DECIMAL(10,3) DEFAULT 0.0,  -- Sum of all scores for calculation
    
    -- Usage statistics
    total_uses INTEGER DEFAULT 0,  -- How many times this document was used
    last_used_at TIMESTAMP,  -- Last time this document was used in a query
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_global_scores_doc_id ON document_global_scores(doc_id);
CREATE INDEX IF NOT EXISTS idx_global_scores_avg_score ON document_global_scores(average_score DESC);
CREATE INDEX IF NOT EXISTS idx_global_scores_total_feedback ON document_global_scores(total_feedback_count DESC);
CREATE INDEX IF NOT EXISTS idx_global_scores_last_used ON document_global_scores(last_used_at DESC);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_global_scores_updated_at
    AFTER UPDATE ON document_global_scores
    FOR EACH ROW
    WHEN NEW.updated_at <= OLD.updated_at
BEGIN
    UPDATE document_global_scores SET updated_at = CURRENT_TIMESTAMP WHERE score_id = NEW.score_id;
END;

-- Verify table was created successfully
SELECT 'document_global_scores table created successfully' as status;



