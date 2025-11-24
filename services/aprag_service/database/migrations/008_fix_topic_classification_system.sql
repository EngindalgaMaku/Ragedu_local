-- Topic Classification System Database Schema Fixes
-- Migration 008: Fix all missing tables and columns for topic classification and progress tracking

-- ============================================================================
-- 1. Fix student_interactions table with all required columns
-- ============================================================================

-- Create student_interactions table if not exists with ALL required columns
CREATE TABLE IF NOT EXISTS student_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT,
    original_response TEXT,
    personalized_response TEXT,
    adaptive_context TEXT,
    processing_time_ms INTEGER,
    model_used TEXT,
    chain_type TEXT,
    sources TEXT,
    emoji_feedback TEXT,
    feedback_score INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Add missing columns to student_interactions if they don't exist
ALTER TABLE student_interactions ADD COLUMN original_response TEXT DEFAULT NULL;
ALTER TABLE student_interactions ADD COLUMN personalized_response TEXT DEFAULT NULL;  
ALTER TABLE student_interactions ADD COLUMN adaptive_context TEXT DEFAULT NULL;
ALTER TABLE student_interactions ADD COLUMN sources TEXT DEFAULT NULL;
ALTER TABLE student_interactions ADD COLUMN emoji_feedback TEXT DEFAULT NULL;
ALTER TABLE student_interactions ADD COLUMN feedback_score INTEGER DEFAULT NULL;

-- ============================================================================
-- 2. Create topic_progress table for student learning analytics
-- ============================================================================

CREATE TABLE IF NOT EXISTS topic_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    topic_id INTEGER NOT NULL,
    questions_asked INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    mastery_level REAL DEFAULT 0.0,
    mastery_score REAL DEFAULT 0.0,
    average_understanding REAL DEFAULT 0.0,
    is_ready_for_next BOOLEAN DEFAULT FALSE,
    readiness_score REAL DEFAULT 0.0,
    time_spent_minutes INTEGER DEFAULT 0,
    first_interaction_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    last_question_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, session_id, topic_id)
);

-- ============================================================================
-- 3. Create question_topic_mapping table for classification tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS question_topic_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    topic_id INTEGER NOT NULL,
    confidence_score REAL NOT NULL,
    mapping_method TEXT DEFAULT 'llm_classification',
    question_complexity TEXT,
    question_type TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id)
);

-- ============================================================================
-- 4. Create/update feature_flags table for system control
-- ============================================================================

CREATE TABLE IF NOT EXISTS feature_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    flag_name TEXT,
    feature_name TEXT,
    is_enabled BOOLEAN NOT NULL DEFAULT 1,
    config_data TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add session-specific feature flags
CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_flags_session_feature 
ON feature_flags(session_id, feature_name) WHERE session_id IS NOT NULL;

-- Add global feature flags  
CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_flags_global_flag
ON feature_flags(flag_name) WHERE session_id IS NULL;

-- ============================================================================
-- 5. Create course_topics table if not exists (for topic classification)
-- ============================================================================

CREATE TABLE IF NOT EXISTS course_topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    topic_title TEXT NOT NULL,
    parent_topic_id INTEGER,
    topic_order INTEGER DEFAULT 0,
    description TEXT,
    keywords TEXT, -- JSON array
    estimated_difficulty TEXT DEFAULT 'intermediate',
    prerequisites TEXT, -- JSON array 
    related_chunk_ids TEXT, -- JSON array
    extraction_method TEXT DEFAULT 'llm_analysis',
    extraction_confidence REAL DEFAULT 0.75,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_topic_id) REFERENCES course_topics(topic_id)
);

-- ============================================================================
-- 6. Create performance indexes
-- ============================================================================

-- Student interactions indexes
CREATE INDEX IF NOT EXISTS idx_student_interactions_user_session ON student_interactions(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_student_interactions_session ON student_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_student_interactions_created ON student_interactions(created_at);

-- Topic progress indexes  
CREATE INDEX IF NOT EXISTS idx_topic_progress_user_session ON topic_progress(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_topic_progress_topic ON topic_progress(topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_progress_updated ON topic_progress(updated_at);

-- Question mapping indexes
CREATE INDEX IF NOT EXISTS idx_question_mapping_interaction ON question_topic_mapping(interaction_id);
CREATE INDEX IF NOT EXISTS idx_question_mapping_topic ON question_topic_mapping(topic_id);
CREATE INDEX IF NOT EXISTS idx_question_mapping_created ON question_topic_mapping(created_at);

-- Course topics indexes
CREATE INDEX IF NOT EXISTS idx_course_topics_session ON course_topics(session_id);
CREATE INDEX IF NOT EXISTS idx_course_topics_parent ON course_topics(parent_topic_id);
CREATE INDEX IF NOT EXISTS idx_course_topics_active ON course_topics(is_active);
CREATE INDEX IF NOT EXISTS idx_course_topics_order ON course_topics(topic_order);

-- ============================================================================
-- 7. Insert default feature flags for topic classification system
-- ============================================================================

-- Global system flags
INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description, created_at) 
VALUES ('aprag_enabled', 1, 'Enable APRAG adaptive personalized system', CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description, created_at) 
VALUES ('topic_classification', 1, 'Enable automatic topic classification for questions', CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description, created_at) 
VALUES ('hybrid_rag', 1, 'Enable hybrid RAG with knowledge base integration', CURRENT_TIMESTAMP);

-- Test session specific flags
INSERT OR IGNORE INTO feature_flags (session_id, feature_name, is_enabled, created_at) 
VALUES ('32ba88c06967b82bf467ef09380a4892', 'aprag_enabled', 1, CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO feature_flags (session_id, feature_name, is_enabled, created_at) 
VALUES ('32ba88c06967b82bf467ef09380a4892', 'topic_classification', 1, CURRENT_TIMESTAMP);

-- ============================================================================
-- 8. Create triggers for automatic updates
-- ============================================================================

-- Update topic_progress.updated_at on changes
CREATE TRIGGER IF NOT EXISTS update_topic_progress_timestamp
AFTER UPDATE ON topic_progress
BEGIN
    UPDATE topic_progress 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

-- Update course_topics.updated_at on changes  
CREATE TRIGGER IF NOT EXISTS update_course_topics_timestamp
AFTER UPDATE ON course_topics
BEGIN
    UPDATE course_topics 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE topic_id = NEW.topic_id;
END;

-- ============================================================================
-- 9. Create views for analytics and reporting
-- ============================================================================

-- Topic classification analytics view
CREATE VIEW IF NOT EXISTS topic_classification_analytics AS
SELECT 
    qtm.id,
    qtm.interaction_id,
    qtm.question,
    qtm.topic_id,
    qtm.confidence_score,
    qtm.mapping_method,
    qtm.question_complexity,
    qtm.question_type,
    qtm.created_at,
    
    -- Topic information
    ct.topic_title,
    ct.session_id,
    ct.estimated_difficulty,
    
    -- Student interaction context
    si.user_id,
    si.query as full_query,
    si.response as system_response,
    si.processing_time_ms,
    si.model_used,
    
    -- Progress information
    tp.questions_asked,
    tp.mastery_level,
    tp.average_understanding
    
FROM question_topic_mapping qtm
LEFT JOIN course_topics ct ON qtm.topic_id = ct.topic_id
LEFT JOIN student_interactions si ON qtm.interaction_id = si.interaction_id  
LEFT JOIN topic_progress tp ON si.user_id = tp.user_id 
    AND ct.session_id = tp.session_id 
    AND qtm.topic_id = tp.topic_id;

-- Student progress summary view
CREATE VIEW IF NOT EXISTS student_progress_summary AS
SELECT 
    tp.user_id,
    tp.session_id,
    COUNT(DISTINCT tp.topic_id) as topics_engaged,
    SUM(tp.questions_asked) as total_questions,
    AVG(tp.mastery_level) as avg_mastery,
    AVG(tp.average_understanding) as avg_understanding,
    COUNT(CASE WHEN tp.is_ready_for_next = 1 THEN 1 END) as topics_ready_for_next,
    MAX(tp.updated_at) as last_activity
FROM topic_progress tp
WHERE tp.questions_asked > 0
GROUP BY tp.user_id, tp.session_id;