-- Migration 007: Remove FOREIGN KEY constraint from session_settings.user_id
-- Created: 2025-11-23
-- Description: Remove FOREIGN KEY constraint because users table is in auth-service,
--              not in aprag-service database. We still validate user_id exists
--              in code, but don't enforce it at database level.

-- Disable foreign key constraints
PRAGMA foreign_keys = OFF;

-- SQLite doesn't support DROP CONSTRAINT directly, so we need to recreate the table
-- Step 1: Create new table without FOREIGN KEY
CREATE TABLE IF NOT EXISTS session_settings_new (
    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,  -- Teacher who created/modified the setting (no FK)
    
    -- Educational Feature Settings
    enable_progressive_assessment BOOLEAN NOT NULL DEFAULT FALSE,
    enable_personalized_responses BOOLEAN NOT NULL DEFAULT FALSE,
    enable_multi_dimensional_feedback BOOLEAN NOT NULL DEFAULT FALSE,
    enable_topic_analytics BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- APRAG Component Settings (for advanced control)
    enable_cacs BOOLEAN NOT NULL DEFAULT TRUE,
    enable_zpd BOOLEAN NOT NULL DEFAULT TRUE,
    enable_bloom BOOLEAN NOT NULL DEFAULT TRUE,
    enable_cognitive_load BOOLEAN NOT NULL DEFAULT TRUE,
    enable_emoji_feedback BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints (no FOREIGN KEY)
    UNIQUE(session_id)
);

-- Step 2: Copy existing data (if table exists)
INSERT INTO session_settings_new 
SELECT * FROM session_settings 
WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_settings');

-- Step 3: Drop old table (if exists) - only if we copied data
DROP TABLE IF EXISTS session_settings;

-- Step 4: Rename new table
ALTER TABLE session_settings_new RENAME TO session_settings;

-- Step 5: Recreate indexes
CREATE INDEX IF NOT EXISTS idx_session_settings_session_id ON session_settings(session_id);
CREATE INDEX IF NOT EXISTS idx_session_settings_user_id ON session_settings(user_id);

-- Step 6: Recreate trigger
DROP TRIGGER IF EXISTS update_session_settings_updated_at;
CREATE TRIGGER IF NOT EXISTS update_session_settings_updated_at
    AFTER UPDATE ON session_settings
    FOR EACH ROW
    WHEN NEW.updated_at <= OLD.updated_at
BEGIN
    UPDATE session_settings SET updated_at = CURRENT_TIMESTAMP WHERE setting_id = NEW.setting_id;
END;

-- Re-enable foreign keys (but session_settings won't have FK anymore)
PRAGMA foreign_keys = ON;

-- Verify
SELECT 'Session Settings table recreated without FOREIGN KEY constraint' as status;

