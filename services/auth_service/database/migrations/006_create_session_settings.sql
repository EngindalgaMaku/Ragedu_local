-- Migration 006: Create Session Settings Table
-- Created: 2025-11-23
-- Description: Teacher-controllable session settings for educational features

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- SESSION SETTINGS TABLE
-- ============================================================================
-- Stores session-specific feature settings controllable by teachers
CREATE TABLE IF NOT EXISTS session_settings (
    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,  -- Teacher who created/modified the setting
    
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
    
    -- Constraints
    UNIQUE(session_id),
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_session_settings_session_id ON session_settings(session_id);
CREATE INDEX IF NOT EXISTS idx_session_settings_user_id ON session_settings(user_id);

-- ============================================================================
-- TRIGGER FOR AUTOMATIC UPDATES
-- ============================================================================
-- Update session_settings updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_session_settings_updated_at
    AFTER UPDATE ON session_settings
    FOR EACH ROW
    WHEN NEW.updated_at <= OLD.updated_at
BEGIN
    UPDATE session_settings SET updated_at = CURRENT_TIMESTAMP WHERE setting_id = NEW.setting_id;
END;

-- ============================================================================
-- DEFAULT SETTINGS FOR EXISTING SESSIONS
-- ============================================================================
-- NOTE: We do NOT create default settings from student_interactions
-- because student_interactions.user_id is a STUDENT ID, not a TEACHER ID.
-- Session settings must have the TEACHER's user_id (session owner).
-- Settings will be created automatically when accessed via the API,
-- using the session's created_by field (teacher) from the sessions table.

-- Verify table was created successfully
SELECT 'Session Settings table created successfully' as status;