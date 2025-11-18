-- Migration 005: Eğitsel-KBRAG Tables and Columns
-- Date: 2025-11-17
-- Description: Add pedagogical features to APRAG service
-- Dependency: APRAG service must be enabled

-- ===========================================
-- 0. Ensure base tables exist (from migration 003)
-- ===========================================

-- Create student_interactions table if it doesn't exist
CREATE TABLE IF NOT EXISTS student_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    original_response TEXT NOT NULL,
    personalized_response TEXT,
    processing_time_ms INTEGER,
    model_used TEXT,
    chain_type TEXT,
    sources TEXT,
    metadata TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create student_profiles table if it doesn't exist
CREATE TABLE IF NOT EXISTS student_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    average_understanding REAL,
    average_satisfaction REAL,
    total_interactions INTEGER DEFAULT 0,
    total_feedback_count INTEGER DEFAULT 0,
    strong_topics TEXT,
    weak_topics TEXT,
    preferred_explanation_style TEXT,
    preferred_difficulty_level TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, session_id)
);

-- Create student_feedback table if it doesn't exist
CREATE TABLE IF NOT EXISTS student_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    understanding_level INTEGER,
    answer_adequacy INTEGER,
    satisfaction_level INTEGER,
    difficulty_level INTEGER,
    topic_understood BOOLEAN,
    answer_helpful BOOLEAN,
    needs_more_explanation BOOLEAN,
    comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id)
);

-- ===========================================
-- 1. Extend student_interactions table
-- ===========================================

-- Check and add columns one by one (SQLite doesn't support ADD COLUMN IF NOT EXISTS)
-- We'll use a more robust approach

-- Add ZPD (Zone of Proximal Development) level
ALTER TABLE student_interactions 
ADD COLUMN zpd_level TEXT DEFAULT NULL;

-- Add Cognitive Load score
ALTER TABLE student_interactions 
ADD COLUMN cognitive_load_score REAL DEFAULT NULL;

-- Add CACS (Conversation-Aware Content Scoring) score
ALTER TABLE student_interactions 
ADD COLUMN cacs_score REAL DEFAULT NULL;

-- Add Emoji feedback
ALTER TABLE student_interactions 
ADD COLUMN emoji_feedback TEXT DEFAULT NULL;

-- Add feedback score (normalized 0-1)
ALTER TABLE student_interactions 
ADD COLUMN feedback_score REAL DEFAULT NULL;

-- ===========================================
-- 2. Extend student_profiles table
-- ===========================================

-- Add current ZPD level
ALTER TABLE student_profiles 
ADD COLUMN current_zpd_level TEXT DEFAULT 'intermediate';

-- Add success rate (0-1)
ALTER TABLE student_profiles 
ADD COLUMN success_rate REAL DEFAULT 0.5;

-- Add average Bloom level
ALTER TABLE student_profiles 
ADD COLUMN avg_bloom_level TEXT DEFAULT 'understand';

-- Add learning velocity (how fast student progresses)
ALTER TABLE student_profiles 
ADD COLUMN learning_velocity REAL DEFAULT 1.0;

-- Add last ZPD update timestamp
ALTER TABLE student_profiles 
ADD COLUMN last_zpd_update TIMESTAMP DEFAULT NULL;

-- ===========================================
-- 3. Create document_global_scores table
-- ===========================================

CREATE TABLE IF NOT EXISTS document_global_scores (
    doc_id TEXT PRIMARY KEY,
    total_feedback_count INTEGER DEFAULT 0,
    avg_emoji_score REAL DEFAULT 0.5,
    positive_feedback_count INTEGER DEFAULT 0,
    negative_feedback_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional metrics
    avg_understanding_level REAL DEFAULT 0.5,
    times_used INTEGER DEFAULT 0,
    
    -- CACS score components (aggregated)
    avg_base_score REAL DEFAULT 0.5,
    avg_personal_score REAL DEFAULT 0.5,
    avg_context_score REAL DEFAULT 0.5
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_doc_global_scores_last_updated 
ON document_global_scores(last_updated DESC);

CREATE INDEX IF NOT EXISTS idx_doc_global_scores_avg_emoji 
ON document_global_scores(avg_emoji_score DESC);

-- ===========================================
-- 4. Create cacs_scores table (detailed scoring history)
-- ===========================================

CREATE TABLE IF NOT EXISTS cacs_scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    
    -- CACS score components
    final_score REAL NOT NULL,
    base_score REAL NOT NULL,
    personal_score REAL NOT NULL,
    global_score REAL NOT NULL,
    context_score REAL NOT NULL,
    
    -- Weights used
    weight_base REAL DEFAULT 0.30,
    weight_personal REAL DEFAULT 0.25,
    weight_global REAL DEFAULT 0.25,
    weight_context REAL DEFAULT 0.20,
    
    -- Metadata
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id)
);

-- Index for fast queries
CREATE INDEX IF NOT EXISTS idx_cacs_scores_interaction 
ON cacs_scores(interaction_id);

CREATE INDEX IF NOT EXISTS idx_cacs_scores_doc 
ON cacs_scores(doc_id);

-- ===========================================
-- 5. Create pedagogical_analytics table
-- ===========================================

CREATE TABLE IF NOT EXISTS pedagogical_analytics (
    analytics_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    
    -- ZPD metrics
    zpd_level TEXT,
    zpd_confidence REAL,
    zpd_success_rate REAL,
    
    -- Bloom metrics
    bloom_level TEXT,
    bloom_confidence REAL,
    
    -- Cognitive Load metrics
    cognitive_load REAL,
    needs_simplification BOOLEAN DEFAULT 0,
    
    -- Timestamp
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id, session_id) REFERENCES student_profiles(user_id, session_id)
);

-- Index for analytics queries
CREATE INDEX IF NOT EXISTS idx_pedagogical_analytics_user_session 
ON pedagogical_analytics(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_pedagogical_analytics_recorded 
ON pedagogical_analytics(recorded_at DESC);

-- ===========================================
-- 6. Create feature_flags table (for session-specific control)
-- ===========================================

CREATE TABLE IF NOT EXISTS feature_flags (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    session_id TEXT DEFAULT NULL,
    user_id TEXT DEFAULT NULL,
    feature_enabled BOOLEAN DEFAULT 1,
    
    -- Metadata
    enabled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    disabled_at TIMESTAMP DEFAULT NULL,
    reason TEXT DEFAULT NULL,
    
    UNIQUE(feature_name, session_id, user_id)
);

-- Index for feature flag lookups
CREATE INDEX IF NOT EXISTS idx_feature_flags_lookup 
ON feature_flags(feature_name, session_id, user_id);

-- ===========================================
-- 7. Insert default feature flags
-- ===========================================

-- Default APRAG flag (globally enabled)
INSERT OR IGNORE INTO feature_flags (feature_name, feature_enabled, reason)
VALUES ('aprag', 1, 'Default APRAG service enabled');

-- Default Eğitsel-KBRAG flag (globally enabled)
INSERT OR IGNORE INTO feature_flags (feature_name, feature_enabled, reason)
VALUES ('egitsel_kbrag', 1, 'Default Eğitsel-KBRAG enabled');

-- ===========================================
-- 8. Create indexes on existing tables for performance
-- ===========================================

-- Index for student_interactions queries
CREATE INDEX IF NOT EXISTS idx_student_interactions_bloom 
ON student_interactions(bloom_level);

CREATE INDEX IF NOT EXISTS idx_student_interactions_zpd 
ON student_interactions(zpd_level);

CREATE INDEX IF NOT EXISTS idx_student_interactions_emoji 
ON student_interactions(emoji_feedback);

-- Index for student_profiles queries
CREATE INDEX IF NOT EXISTS idx_student_profiles_zpd 
ON student_profiles(current_zpd_level);

CREATE INDEX IF NOT EXISTS idx_student_profiles_success 
ON student_profiles(success_rate DESC);

-- ===========================================
-- Migration complete
-- ===========================================

-- Verify tables exist
SELECT 'Migration 005 complete - Eğitsel-KBRAG tables ready' AS status;

