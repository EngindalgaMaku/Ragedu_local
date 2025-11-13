-- Migration 004: Create Topic-Based Learning Path Tracking tables
-- Created: 2025-01-XX
-- Description: Tables for course topics, topic progress, and question-topic mapping

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- 1. COURSE TOPICS TABLE
-- ============================================================================
-- Stores topic hierarchy extracted from course chunks
CREATE TABLE IF NOT EXISTS course_topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(255) NOT NULL,
    
    -- Topic hierarchy
    topic_title VARCHAR(255) NOT NULL,
    parent_topic_id INTEGER,  -- NULL for main topics
    topic_order INTEGER,  -- Order within parent or session
    
    -- Topic metadata
    description TEXT,
    keywords TEXT,  -- JSON array of keywords
    estimated_difficulty VARCHAR(20),  -- beginner, intermediate, advanced
    estimated_time_minutes INTEGER,
    
    -- Prerequisites
    prerequisites TEXT,  -- JSON array of topic_ids
    
    -- Chunk references
    related_chunk_ids TEXT,  -- JSON array of chunk IDs
    
    -- LLM extraction metadata
    extraction_method VARCHAR(50),  -- llm_analysis, manual, hybrid
    extraction_confidence DECIMAL(3,2),  -- 0.00 to 1.00
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Note: session_id references document processing sessions, not user authentication sessions
    -- These sessions are managed by document processing service and may be ephemeral
    FOREIGN KEY (parent_topic_id) REFERENCES course_topics(topic_id) ON DELETE SET NULL
);

-- ============================================================================
-- 2. TOPIC PROGRESS TABLE
-- ============================================================================
-- Stores student progress for each topic
CREATE TABLE IF NOT EXISTS topic_progress (
    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    topic_id INTEGER NOT NULL,
    
    -- Progress metrics
    questions_asked INTEGER DEFAULT 0,
    average_understanding DECIMAL(3,2),  -- From feedback (1-5 scale normalized)
    average_satisfaction DECIMAL(3,2),   -- From feedback (1-5 scale normalized)
    last_question_timestamp TIMESTAMP,
    
    -- Mastery assessment
    mastery_level VARCHAR(20),  -- not_started, learning, mastered, needs_review
    mastery_score DECIMAL(3,2),  -- 0.00 to 1.00
    
    -- Readiness for next topic
    is_ready_for_next BOOLEAN DEFAULT FALSE,
    readiness_score DECIMAL(3,2),  -- 0.00 to 1.00
    
    -- Time tracking
    time_spent_minutes INTEGER DEFAULT 0,
    first_question_timestamp TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, session_id, topic_id),
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE,
    -- Note: session_id references document processing sessions, not user authentication sessions
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
);

-- ============================================================================
-- 3. QUESTION-TOPIC MAPPING TABLE
-- ============================================================================
-- Maps student questions to topics
CREATE TABLE IF NOT EXISTS question_topic_mapping (
    mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    
    -- Mapping confidence
    confidence_score DECIMAL(3,2),  -- 0.00 to 1.00
    mapping_method VARCHAR(50),  -- llm_classification, embedding_similarity, keyword_match
    
    -- Question analysis
    question_complexity VARCHAR(20),  -- basic, intermediate, advanced
    question_type VARCHAR(50),  -- factual, conceptual, application, analysis
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Course Topics indexes
CREATE INDEX IF NOT EXISTS idx_course_topics_session_id ON course_topics(session_id);
CREATE INDEX IF NOT EXISTS idx_course_topics_parent_id ON course_topics(parent_topic_id);
CREATE INDEX IF NOT EXISTS idx_course_topics_order ON course_topics(session_id, topic_order);
CREATE INDEX IF NOT EXISTS idx_course_topics_active ON course_topics(session_id, is_active);

-- Topic Progress indexes
CREATE INDEX IF NOT EXISTS idx_topic_progress_user_session ON topic_progress(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_topic_progress_topic_id ON topic_progress(topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_progress_mastery ON topic_progress(user_id, session_id, mastery_level);
CREATE INDEX IF NOT EXISTS idx_topic_progress_ready ON topic_progress(user_id, session_id, is_ready_for_next);

-- Question-Topic Mapping indexes
CREATE INDEX IF NOT EXISTS idx_question_topic_interaction ON question_topic_mapping(interaction_id);
CREATE INDEX IF NOT EXISTS idx_question_topic_topic_id ON question_topic_mapping(topic_id);
CREATE INDEX IF NOT EXISTS idx_question_topic_confidence ON question_topic_mapping(topic_id, confidence_score);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update updated_at timestamp for course_topics
CREATE TRIGGER IF NOT EXISTS update_course_topics_timestamp
AFTER UPDATE ON course_topics
FOR EACH ROW
BEGIN
    UPDATE course_topics
    SET updated_at = CURRENT_TIMESTAMP
    WHERE topic_id = NEW.topic_id;
END;

-- Update updated_at timestamp for topic_progress
CREATE TRIGGER IF NOT EXISTS update_topic_progress_timestamp
AFTER UPDATE ON topic_progress
FOR EACH ROW
BEGIN
    UPDATE topic_progress
    SET updated_at = CURRENT_TIMESTAMP
    WHERE progress_id = NEW.progress_id;
END;

-- Verify tables were created successfully
SELECT 'Topic tables created successfully' as status;



