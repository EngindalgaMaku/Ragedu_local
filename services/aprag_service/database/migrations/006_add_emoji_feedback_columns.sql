-- Migration 006: Add emoji feedback columns to student_interactions
-- Faz 4: Emoji Tabanlı Mikro-Geri Bildirim
-- Created: 2025-11-17

-- Add emoji_feedback column to student_interactions
ALTER TABLE student_interactions 
ADD COLUMN emoji_feedback TEXT DEFAULT NULL;

-- Add emoji_feedback_timestamp
ALTER TABLE student_interactions 
ADD COLUMN emoji_feedback_timestamp TIMESTAMP DEFAULT NULL;

-- Create index for faster emoji feedback queries
CREATE INDEX IF NOT EXISTS idx_emoji_feedback 
ON student_interactions(emoji_feedback, emoji_feedback_timestamp);

-- Create emoji_feedback_summary table for analytics
CREATE TABLE IF NOT EXISTS emoji_feedback_summary (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    emoji TEXT NOT NULL,
    emoji_count INTEGER DEFAULT 1,
    avg_score REAL DEFAULT 0.5,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, session_id, emoji)
);

-- Create index for emoji summary queries
CREATE INDEX IF NOT EXISTS idx_emoji_summary_user_session 
ON emoji_feedback_summary(user_id, session_id);

-- Add comment column for additional feedback
ALTER TABLE student_interactions 
ADD COLUMN emoji_comment TEXT DEFAULT NULL;















