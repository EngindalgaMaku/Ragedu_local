-- Migration 003: Create APRAG (Adaptive Personalized RAG) tables
-- Created: 2025-01-XX
-- Description: Tables for student interactions, feedback, profiles, and personalization

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- 1. STUDENT INTERACTIONS TABLE
-- ============================================================================
-- Stores all student queries and responses for learning session interactions
CREATE TABLE IF NOT EXISTS student_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    original_response TEXT NOT NULL,
    personalized_response TEXT,  -- NULL if personalization disabled
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER,
    model_used VARCHAR(255),
    chain_type VARCHAR(50),
    sources TEXT,  -- JSON string
    metadata TEXT,  -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes will be created separately
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- 2. STUDENT FEEDBACK TABLE
-- ============================================================================
-- Stores student feedback on interactions (understanding, satisfaction, etc.)
CREATE TABLE IF NOT EXISTS student_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    user_id VARCHAR(255) NOT NULL,
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
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- 3. STUDENT PROFILES TABLE
-- ============================================================================
-- Stores personalized learning profiles for each student per session
CREATE TABLE IF NOT EXISTS student_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Learning metrics
    average_understanding DECIMAL(3,2),
    average_satisfaction DECIMAL(3,2),
    total_interactions INTEGER DEFAULT 0,
    total_feedback_count INTEGER DEFAULT 0,
    
    -- Strong/weak topics (JSON strings)
    strong_topics TEXT,  -- JSON: {"topic": "score"}
    weak_topics TEXT,    -- JSON: {"topic": "score"}
    
    -- Learning style preferences
    preferred_explanation_style VARCHAR(50),  -- detailed, concise, examples, etc.
    preferred_difficulty_level VARCHAR(20),   -- beginner, intermediate, advanced
    
    -- Personalization settings
    personalization_enabled BOOLEAN DEFAULT TRUE,
    feedback_collection_enabled BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint: one profile per user per session
    UNIQUE(user_id, session_id),
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- 4. PERSONALIZED RESPONSES TABLE
-- ============================================================================
-- Stores personalized response adaptations and their factors
CREATE TABLE IF NOT EXISTS personalized_responses (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    
    -- Personalization details
    original_response TEXT NOT NULL,
    personalized_response TEXT NOT NULL,
    personalization_factors TEXT,  -- JSON: which factors were used
    
    -- Adaptation details
    difficulty_adjustment VARCHAR(20),
    explanation_level VARCHAR(20),
    added_examples BOOLEAN DEFAULT FALSE,
    added_visual_aids BOOLEAN DEFAULT FALSE,
    
    -- Feedback incorporation
    updated_after_feedback BOOLEAN DEFAULT FALSE,
    feedback_incorporated TEXT,  -- JSON: feedback data incorporated
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- 5. LEARNING PATTERNS TABLE
-- ============================================================================
-- Stores detected learning patterns and trends
CREATE TABLE IF NOT EXISTS learning_patterns (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Pattern type and description
    pattern_type VARCHAR(50),  -- improvement, struggle, mastery, etc.
    pattern_description TEXT,
    
    -- Related topics (JSON string)
    related_topics TEXT,  -- JSON array
    
    -- Trend data (JSON string)
    trend_data TEXT,  -- JSON: time series data
    
    -- Recommendations (JSON string)
    recommendations TEXT,  -- JSON array
    
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score DECIMAL(3,2),
    
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- 6. RECOMMENDATIONS TABLE
-- ============================================================================
-- Stores personalized recommendations for students
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Recommendation type
    recommendation_type VARCHAR(50),  -- question, topic, learning_path, etc.
    
    -- Recommendation content
    title VARCHAR(255),
    description TEXT,
    content TEXT,  -- JSON string
    
    -- Priority and score
    priority INTEGER,
    relevance_score DECIMAL(3,2),
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending',  -- pending, shown, accepted, dismissed
    shown_at TIMESTAMP,
    accepted_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- 7. FEATURE FLAGS TABLE (for database-backed flags)
-- ============================================================================
-- Stores feature flag settings for global and session-level control
CREATE TABLE IF NOT EXISTS aprag_feature_flags (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    flag_key VARCHAR(100) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    scope VARCHAR(20) NOT NULL,  -- global, session, user
    scope_id VARCHAR(255),  -- session_id or user_id if scope is not global
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint: one flag setting per key per scope
    UNIQUE(flag_key, scope, scope_id)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
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

-- Feature Flags indexes
CREATE INDEX IF NOT EXISTS idx_flags_key_scope ON aprag_feature_flags(flag_key, scope, scope_id);
CREATE INDEX IF NOT EXISTS idx_flags_scope_id ON aprag_feature_flags(scope, scope_id);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Update student_profiles last_updated timestamp
CREATE TRIGGER IF NOT EXISTS update_profiles_updated_at
    AFTER UPDATE ON student_profiles
    FOR EACH ROW
    WHEN NEW.last_updated <= OLD.last_updated
BEGIN
    UPDATE student_profiles SET last_updated = CURRENT_TIMESTAMP WHERE profile_id = NEW.profile_id;
END;

-- Update feature flags updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_flags_updated_at
    AFTER UPDATE ON aprag_feature_flags
    FOR EACH ROW
    WHEN NEW.updated_at <= OLD.updated_at
BEGIN
    UPDATE aprag_feature_flags SET updated_at = CURRENT_TIMESTAMP WHERE flag_id = NEW.flag_id;
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

-- Verify tables were created successfully
SELECT 'APRAG tables created successfully' as status;

