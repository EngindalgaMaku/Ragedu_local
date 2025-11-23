-- Migration 005: Eğitsel-KBRAG Tables and Columns (SAFE VERSION)
-- Date: 2025-11-17
-- Description: Add pedagogical features to APRAG service
-- Note: This version is safe to run multiple times

-- ===========================================
-- 0. Ensure base tables exist
-- ===========================================

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
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Eğitsel-KBRAG columns (added in this migration)
    bloom_level TEXT DEFAULT NULL,
    zpd_level TEXT DEFAULT NULL,
    cognitive_load_score REAL DEFAULT NULL,
    cacs_score REAL DEFAULT NULL,
    emoji_feedback TEXT DEFAULT NULL,
    feedback_score REAL DEFAULT NULL
);

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
    -- Eğitsel-KBRAG columns
    current_zpd_level TEXT DEFAULT 'intermediate',
    success_rate REAL DEFAULT 0.5,
    avg_bloom_level TEXT DEFAULT 'understand',
    learning_velocity REAL DEFAULT 1.0,
    last_zpd_update TIMESTAMP DEFAULT NULL,
    UNIQUE(user_id, session_id)
);

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
-- 1. Create document_global_scores table
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

-- ===========================================
-- 2. Create cacs_scores table
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

-- ===========================================
-- 3. Create pedagogical_analytics table
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

-- ===========================================
-- 4. Create feature_flags table
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

-- ===========================================
-- 5. Create indexes
-- ===========================================

-- document_global_scores indexes
CREATE INDEX IF NOT EXISTS idx_doc_global_scores_last_updated 
ON document_global_scores(last_updated DESC);

CREATE INDEX IF NOT EXISTS idx_doc_global_scores_avg_emoji 
ON document_global_scores(avg_emoji_score DESC);

-- cacs_scores indexes
CREATE INDEX IF NOT EXISTS idx_cacs_scores_interaction 
ON cacs_scores(interaction_id);

CREATE INDEX IF NOT EXISTS idx_cacs_scores_doc 
ON cacs_scores(doc_id);

-- pedagogical_analytics indexes
CREATE INDEX IF NOT EXISTS idx_pedagogical_analytics_user_session 
ON pedagogical_analytics(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_pedagogical_analytics_recorded 
ON pedagogical_analytics(recorded_at DESC);

-- feature_flags indexes
CREATE INDEX IF NOT EXISTS idx_feature_flags_lookup 
ON feature_flags(feature_name, session_id, user_id);

-- student_interactions indexes (new columns)
CREATE INDEX IF NOT EXISTS idx_student_interactions_bloom 
ON student_interactions(bloom_level);

CREATE INDEX IF NOT EXISTS idx_student_interactions_zpd 
ON student_interactions(zpd_level);

CREATE INDEX IF NOT EXISTS idx_student_interactions_emoji 
ON student_interactions(emoji_feedback);

-- student_profiles indexes (new columns)
CREATE INDEX IF NOT EXISTS idx_student_profiles_zpd 
ON student_profiles(current_zpd_level);

CREATE INDEX IF NOT EXISTS idx_student_profiles_success 
ON student_profiles(success_rate DESC);

-- ===========================================
-- 6. Insert default feature flags
-- ===========================================

INSERT OR IGNORE INTO feature_flags (feature_name, feature_enabled, reason)
VALUES ('aprag', 1, 'Default APRAG service enabled');

INSERT OR IGNORE INTO feature_flags (feature_name, feature_enabled, reason)
VALUES ('egitsel_kbrag', 1, 'Default Eğitsel-KBRAG enabled');

-- ===========================================
-- Migration complete
-- ===========================================

SELECT 'Migration 005 complete - Eğitsel-KBRAG tables ready' AS status;















