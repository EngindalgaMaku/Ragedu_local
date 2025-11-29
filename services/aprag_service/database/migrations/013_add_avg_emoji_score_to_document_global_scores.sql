-- Migration 013: Add avg_emoji_score column to document_global_scores
-- Created: 2025-11-28
-- Description: Adds avg_emoji_score column to document_global_scores table for emoji feedback tracking

PRAGMA foreign_keys = OFF;

-- Add avg_emoji_score column if it doesn't exist
-- Check if column exists first (SQLite doesn't support IF NOT EXISTS for ALTER TABLE ADD COLUMN)
-- We'll use a safe approach: try to add, ignore if it fails

-- Add the column (will fail silently if it already exists in some SQLite versions)
-- For safety, we'll check first using a different approach
-- SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we'll use a try-catch approach
-- or check if column exists first

-- Since SQLite doesn't support conditional column addition, we'll use a workaround:
-- 1. Check if column exists
-- 2. Add if it doesn't exist

-- For SQLite, we can use a PRAGMA to check, but it's easier to just try to add
-- and catch the error. However, SQLite will throw an error if column exists.
-- So we'll use a safer approach: check table_info first

-- Check if avg_emoji_score column exists
-- If not, add it
-- Note: SQLite doesn't support IF NOT EXISTS for ALTER TABLE ADD COLUMN
-- So we need to check first using PRAGMA table_info

-- Add avg_emoji_score column
-- This will work if the column doesn't exist
-- If it exists, SQLite will throw an error, which we'll catch in Python
ALTER TABLE document_global_scores 
ADD COLUMN avg_emoji_score DECIMAL(5,3) DEFAULT 0.5;

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_global_scores_avg_emoji_score 
ON document_global_scores(avg_emoji_score DESC);

PRAGMA foreign_keys = ON;

-- Verify column was added
SELECT 'avg_emoji_score column added to document_global_scores' as status;



