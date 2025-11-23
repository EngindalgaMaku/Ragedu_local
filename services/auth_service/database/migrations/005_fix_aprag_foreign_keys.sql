-- Migration 005: Fix APRAG foreign key constraints
-- Created: 2025-11-18
-- Description: Fix user_id foreign keys to reference users(id) instead of users(username)
-- This allows using numeric user IDs instead of usernames

-- Enable foreign key constraints
PRAGMA foreign_keys = OFF;

-- ============================================================================
-- 1. FIX STUDENT_INTERACTIONS TABLE
-- ============================================================================

-- Create new table with correct foreign key
CREATE TABLE IF NOT EXISTS student_interactions_new (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    original_response TEXT NOT NULL,
    personalized_response TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER,
    model_used VARCHAR(255),
    chain_type VARCHAR(50),
    sources TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Copy existing data (if any)
INSERT INTO student_interactions_new 
SELECT * FROM student_interactions WHERE 1=0;  -- Copy structure only, no data yet

-- Drop old table
DROP TABLE IF EXISTS student_interactions;

-- Rename new table
ALTER TABLE student_interactions_new RENAME TO student_interactions;

-- ============================================================================
-- 2. FIX STUDENT_FEEDBACK TABLE
-- ============================================================================

-- Create new table with correct foreign key
CREATE TABLE IF NOT EXISTS student_feedback_new (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Likert scale ratings (1-5)
    understanding_level INTEGER CHECK (understanding_level BETWEEN 1 AND 5),
    answer_adequacy INTEGER CHECK (answer_adequacy BETWEEN 1 AND 5),
    satisfaction_level INTEGER CHECK (satisfaction_level BETWEEN 1 AND 5),
    difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 5),
    
    -- Boolean evaluations
    topic_understood BOOLEAN,
    answer_helpful BOOLEAN,
    needs_more_explanation BOOLEAN,
    
    -- Open-ended feedback
    comment TEXT,
    
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Copy existing data (if any)
INSERT INTO student_feedback_new 
SELECT * FROM student_feedback WHERE 1=0;  -- Copy structure only

-- Drop old table
DROP TABLE IF EXISTS student_feedback;

-- Rename new table
ALTER TABLE student_feedback_new RENAME TO student_feedback;

-- ============================================================================
-- 3. FIX STUDENT_PROFILES TABLE
-- ============================================================================

-- Create new table with correct foreign key
CREATE TABLE IF NOT EXISTS student_profiles_new (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Learning metrics
    average_understanding DECIMAL(3,2),
    average_satisfaction DECIMAL(3,2),
    total_interactions INTEGER DEFAULT 0,
    total_feedback_count INTEGER DEFAULT 0,
    
    -- Strong/weak topics (JSON strings)
    strong_topics TEXT,
    weak_topics TEXT,
    
    -- Learning style preferences
    preferred_explanation_style VARCHAR(50),
    preferred_difficulty_level VARCHAR(20),
    
    -- Personalization settings
    personalization_enabled BOOLEAN DEFAULT TRUE,
    feedback_collection_enabled BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint: one profile per user per session
    UNIQUE(user_id, session_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Copy existing data (if any)
INSERT INTO student_profiles_new 
SELECT * FROM student_profiles WHERE 1=0;  -- Copy structure only

-- Drop old table
DROP TABLE IF EXISTS student_profiles;

-- Rename new table
ALTER TABLE student_profiles_new RENAME TO student_profiles;

-- ============================================================================
-- 4. FIX PERSONALIZED_RESPONSES TABLE
-- ============================================================================

-- Create new table with correct foreign key
CREATE TABLE IF NOT EXISTS personalized_responses_new (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    
    -- Personalization details
    original_response TEXT NOT NULL,
    personalized_response TEXT NOT NULL,
    personalization_factors TEXT,
    
    -- Adaptation details
    difficulty_adjustment VARCHAR(20),
    explanation_level VARCHAR(20),
    added_examples BOOLEAN DEFAULT FALSE,
    added_visual_aids BOOLEAN DEFAULT FALSE,
    
    -- Feedback incorporation
    updated_after_feedback BOOLEAN DEFAULT FALSE,
    feedback_incorporated TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Copy existing data (if any)
INSERT INTO personalized_responses_new 
SELECT * FROM personalized_responses WHERE 1=0;  -- Copy structure only

-- Drop old table
DROP TABLE IF EXISTS personalized_responses;

-- Rename new table
ALTER TABLE personalized_responses_new RENAME TO personalized_responses;

-- ============================================================================
-- 5. FIX LEARNING_PATTERNS TABLE
-- ============================================================================

-- Create new table with correct foreign key
CREATE TABLE IF NOT EXISTS learning_patterns_new (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Pattern type and description
    pattern_type VARCHAR(50),
    pattern_description TEXT,
    
    -- Related topics (JSON string)
    related_topics TEXT,
    
    -- Trend data (JSON string)
    trend_data TEXT,
    
    -- Recommendations (JSON string)
    recommendations TEXT,
    
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score DECIMAL(3,2),
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Copy existing data (if any)
INSERT INTO learning_patterns_new 
SELECT * FROM learning_patterns WHERE 1=0;  -- Copy structure only

-- Drop old table
DROP TABLE IF EXISTS learning_patterns;

-- Rename new table
ALTER TABLE learning_patterns_new RENAME TO learning_patterns;

-- ============================================================================
-- 6. FIX RECOMMENDATIONS TABLE
-- ============================================================================

-- Create new table with correct foreign key
CREATE TABLE IF NOT EXISTS recommendations_new (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Recommendation type
    recommendation_type VARCHAR(50),
    
    -- Recommendation content
    title VARCHAR(255),
    description TEXT,
    content TEXT,
    
    -- Priority and score
    priority INTEGER,
    relevance_score DECIMAL(3,2),
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending',
    shown_at TIMESTAMP,
    accepted_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Copy existing data (if any)
INSERT INTO recommendations_new 
SELECT * FROM recommendations WHERE 1=0;  -- Copy structure only

-- Drop old table
DROP TABLE IF EXISTS recommendations;

-- Rename new table
ALTER TABLE recommendations_new RENAME TO recommendations;

-- ============================================================================
-- RE-CREATE INDEXES
-- ============================================================================

-- Student Interactions indexes
CREATE INDEX IF NOT EXISTS idx_interactions_user_session ON student_interactions(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON student_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON student_interactions(session_id);

-- Student Feedback indexes
CREATE INDEX IF NOT EXISTS idx_feedback_interaction ON student_feedback(interaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON student_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON student_feedback(timestamp);

-- Student Profiles indexes
CREATE INDEX IF NOT EXISTS idx_profiles_user_session ON student_profiles(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_profiles_user ON student_profiles(user_id);

-- Personalized Responses indexes
CREATE INDEX IF NOT EXISTS idx_responses_interaction ON personalized_responses(interaction_id);
CREATE INDEX IF NOT EXISTS idx_responses_user ON personalized_responses(user_id);

-- Learning Patterns indexes
CREATE INDEX IF NOT EXISTS idx_patterns_user_session ON learning_patterns(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON learning_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_detected ON learning_patterns(detected_at);

-- Recommendations indexes
CREATE INDEX IF NOT EXISTS idx_recommendations_user_session ON recommendations(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status);
CREATE INDEX IF NOT EXISTS idx_recommendations_type ON recommendations(recommendation_type);
CREATE INDEX IF NOT EXISTS idx_recommendations_priority ON recommendations(priority DESC);

-- ============================================================================
-- RE-CREATE TRIGGERS
-- ============================================================================

-- Update student_profiles last_updated timestamp
CREATE TRIGGER IF NOT EXISTS update_profiles_updated_at
    AFTER UPDATE ON student_profiles
    FOR EACH ROW
    WHEN NEW.last_updated <= OLD.last_updated
BEGIN
    UPDATE student_profiles SET last_updated = CURRENT_TIMESTAMP WHERE profile_id = NEW.profile_id;
END;

-- Auto-increment total_interactions in student_profiles
CREATE TRIGGER IF NOT EXISTS increment_interaction_count
    AFTER INSERT ON student_interactions
    FOR EACH ROW
BEGIN
    UPDATE student_profiles 
    SET total_interactions = total_interactions + 1,
        last_updated = CURRENT_TIMESTAMP
    WHERE user_id = NEW.user_id AND session_id = NEW.session_id;
END;

-- Auto-increment total_feedback_count in student_profiles
CREATE TRIGGER IF NOT EXISTS increment_feedback_count
    AFTER INSERT ON student_feedback
    FOR EACH ROW
BEGIN
    UPDATE student_profiles 
    SET total_feedback_count = total_feedback_count + 1,
        last_updated = CURRENT_TIMESTAMP
    WHERE user_id = NEW.user_id AND session_id = NEW.session_id;
END;

-- Re-enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Verify migration
SELECT 'Migration 005 complete - Foreign keys fixed to use user ID instead of username' AS status;












