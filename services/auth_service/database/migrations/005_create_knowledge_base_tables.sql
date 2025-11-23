-- Migration 005: Create Knowledge Base Enhancement Tables
-- Created: 2025-11-20
-- Description: Tables for structured topic knowledge, QA pairs, and enhanced learning materials
-- Purpose: KB-Enhanced RAG - Structured knowledge layer on top of chunks

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- 1. TOPIC KNOWLEDGE BASE TABLE
-- ============================================================================
-- Stores structured knowledge extracted from chunks for each topic
-- This provides a "knowledge card" for each topic with summaries, concepts, etc.

CREATE TABLE IF NOT EXISTS topic_knowledge_base (
    knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    
    -- Core Knowledge Components
    topic_summary TEXT NOT NULL,  -- 200-300 word comprehensive summary
    key_concepts TEXT,  -- JSON array: [{"term": "...", "definition": "...", "importance": "high"}]
    learning_objectives TEXT,  -- JSON array: [{"level": "remember", "objective": "..."}]
    
    -- Structured Content
    definitions TEXT,  -- JSON object: {"term1": "definition1", "term2": "definition2"}
    formulas TEXT,  -- JSON array: [{"name": "F=ma", "explanation": "...", "variables": {...}}]
    examples TEXT,  -- JSON array: [{"title": "...", "scenario": "...", "explanation": "..."}]
    
    -- Relationships & Context
    related_topics TEXT,  -- JSON array: [topic_id1, topic_id2, ...] - topics that should be learned together
    prerequisite_concepts TEXT,  -- JSON array: ["concept1", "concept2"] - what student should know first
    real_world_applications TEXT,  -- JSON array: ["application1", "application2"]
    common_misconceptions TEXT,  -- JSON array: [{"misconception": "...", "correction": "..."}]
    
    -- Metadata
    content_quality_score DECIMAL(3,2) DEFAULT 0.80,  -- LLM extraction quality (0.00 to 1.00)
    extraction_model VARCHAR(100),  -- Which LLM model was used
    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    is_validated BOOLEAN DEFAULT FALSE,  -- Has a teacher reviewed this?
    validation_date TIMESTAMP,
    validator_user_id VARCHAR(255),
    
    -- Stats
    view_count INTEGER DEFAULT 0,
    usefulness_rating DECIMAL(3,2),  -- Average student rating
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE,
    FOREIGN KEY (validator_user_id) REFERENCES users(username) ON DELETE SET NULL,
    
    -- Ensure one knowledge base entry per topic
    UNIQUE(topic_id)
);

-- ============================================================================
-- 2. TOPIC QA PAIRS TABLE
-- ============================================================================
-- Stores pre-generated question-answer pairs for each topic
-- These can be used for:
-- - Direct answer matching (if student asks exact same question)
-- - Practice question generation
-- - Assessment creation
-- - Rapid response without LLM inference

CREATE TABLE IF NOT EXISTS topic_qa_pairs (
    qa_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    
    -- Question-Answer Content
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    explanation TEXT,  -- Detailed step-by-step explanation
    
    -- Classification
    difficulty_level VARCHAR(20) NOT NULL,  -- beginner, intermediate, advanced
    question_type VARCHAR(50) NOT NULL,  -- factual, conceptual, application, analysis, synthesis, evaluation
    bloom_taxonomy_level VARCHAR(50),  -- remember, understand, apply, analyze, evaluate, create
    cognitive_complexity VARCHAR(20),  -- low, medium, high
    
    -- Source Information
    source_chunk_ids TEXT,  -- JSON array: chunk IDs this was generated from
    extraction_method VARCHAR(50) DEFAULT 'llm_generated',  -- llm_generated, manual, synthetic, student_question
    extraction_model VARCHAR(100),  -- Which LLM model generated this
    
    -- Quality Metrics
    quality_score DECIMAL(3,2) DEFAULT 0.75,  -- Initial quality estimate
    is_validated BOOLEAN DEFAULT FALSE,
    validator_user_id VARCHAR(255),
    validation_notes TEXT,
    
    -- Usage Statistics
    times_asked INTEGER DEFAULT 0,  -- How many times students asked similar question
    times_matched INTEGER DEFAULT 0,  -- How many times this QA was used in response
    average_student_rating DECIMAL(3,2),  -- Student feedback on this answer
    success_rate DECIMAL(3,2),  -- Did students understand after this answer?
    average_followup_count INTEGER DEFAULT 0,  -- How many follow-up questions typically?
    
    -- Related Content
    related_qa_ids TEXT,  -- JSON array: [qa_id1, qa_id2] - similar/related questions
    related_concepts TEXT,  -- JSON array: ["concept1", "concept2"]
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_featured BOOLEAN DEFAULT FALSE,  -- Featured questions for review
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE,
    FOREIGN KEY (validator_user_id) REFERENCES users(username) ON DELETE SET NULL
);

-- ============================================================================
-- 3. TOPIC PREREQUISITES GRAPH TABLE
-- ============================================================================
-- Explicit prerequisite relationships between topics
-- Enables curriculum-aware learning path generation

CREATE TABLE IF NOT EXISTS topic_prerequisites (
    prerequisite_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,  -- The topic that requires prerequisites
    prerequisite_topic_id INTEGER NOT NULL,  -- The topic that must be learned first
    
    -- Strength of prerequisite relationship
    importance_level VARCHAR(20) DEFAULT 'required',  -- required, recommended, optional
    strength_score DECIMAL(3,2) DEFAULT 0.80,  -- How critical is this prerequisite (0-1)
    
    -- Metadata
    extraction_method VARCHAR(50) DEFAULT 'llm_analysis',  -- llm_analysis, manual, curriculum_standard
    confidence_score DECIMAL(3,2) DEFAULT 0.75,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE,
    FOREIGN KEY (prerequisite_topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE,
    
    -- Prevent duplicate prerequisites
    UNIQUE(topic_id, prerequisite_topic_id),
    
    -- Prevent self-reference
    CHECK(topic_id != prerequisite_topic_id)
);

-- ============================================================================
-- 4. QA SIMILARITY CACHE TABLE
-- ============================================================================
-- Cache for frequently asked question similarities
-- Speeds up QA matching without re-computing embeddings

CREATE TABLE IF NOT EXISTS qa_similarity_cache (
    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text TEXT NOT NULL,
    question_text_hash VARCHAR(64) NOT NULL,  -- MD5 hash for quick lookup
    
    -- Matched QA Pairs
    matched_qa_ids TEXT NOT NULL,  -- JSON array: [{"qa_id": 123, "similarity": 0.92}, ...]
    
    -- Cache Metadata
    embedding_model VARCHAR(100),  -- Which embedding model was used
    cache_hits INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,  -- Cache expiry (30 days default)
    
    -- Quick lookup by hash
    UNIQUE(question_text_hash)
);

-- ============================================================================
-- 5. STUDENT QA INTERACTIONS TABLE
-- ============================================================================
-- Track which QA pairs were shown to students and their feedback
-- Helps improve QA quality over time

CREATE TABLE IF NOT EXISTS student_qa_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    qa_id INTEGER NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    -- Student's original question
    original_question TEXT NOT NULL,
    similarity_score DECIMAL(3,2),  -- How similar to stored question
    
    -- Interaction Data
    was_helpful BOOLEAN,  -- Did student find this helpful?
    student_rating INTEGER,  -- 1-5 stars
    had_followup BOOLEAN DEFAULT FALSE,  -- Did student ask follow-up?
    followup_question TEXT,  -- What was the follow-up?
    
    -- Response metadata
    response_time_ms INTEGER,  -- How fast was the response
    response_source VARCHAR(50),  -- direct_qa, hybrid, chunk_only
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (qa_id) REFERENCES topic_qa_pairs(qa_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Topic Knowledge Base indexes
CREATE INDEX IF NOT EXISTS idx_topic_kb_topic_id ON topic_knowledge_base(topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_kb_validated ON topic_knowledge_base(is_validated);
CREATE INDEX IF NOT EXISTS idx_topic_kb_quality ON topic_knowledge_base(content_quality_score DESC);

-- Topic QA Pairs indexes
CREATE INDEX IF NOT EXISTS idx_topic_qa_topic_id ON topic_qa_pairs(topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_qa_difficulty ON topic_qa_pairs(difficulty_level);
CREATE INDEX IF NOT EXISTS idx_topic_qa_type ON topic_qa_pairs(question_type);
CREATE INDEX IF NOT EXISTS idx_topic_qa_active ON topic_qa_pairs(is_active);
CREATE INDEX IF NOT EXISTS idx_topic_qa_featured ON topic_qa_pairs(is_featured);
CREATE INDEX IF NOT EXISTS idx_topic_qa_usage ON topic_qa_pairs(times_asked DESC);
CREATE INDEX IF NOT EXISTS idx_topic_qa_rating ON topic_qa_pairs(average_student_rating DESC);

-- Topic Prerequisites indexes
CREATE INDEX IF NOT EXISTS idx_topic_prereq_topic ON topic_prerequisites(topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_prereq_prerequisite ON topic_prerequisites(prerequisite_topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_prereq_importance ON topic_prerequisites(importance_level);

-- QA Similarity Cache indexes
CREATE INDEX IF NOT EXISTS idx_qa_cache_hash ON qa_similarity_cache(question_text_hash);
CREATE INDEX IF NOT EXISTS idx_qa_cache_expires ON qa_similarity_cache(expires_at);

-- Student QA Interactions indexes
CREATE INDEX IF NOT EXISTS idx_student_qa_qa_id ON student_qa_interactions(qa_id);
CREATE INDEX IF NOT EXISTS idx_student_qa_user ON student_qa_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_student_qa_session ON student_qa_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_student_qa_helpful ON student_qa_interactions(was_helpful);
CREATE INDEX IF NOT EXISTS idx_student_qa_rating ON student_qa_interactions(student_rating DESC);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update updated_at timestamp for topic_knowledge_base
CREATE TRIGGER IF NOT EXISTS update_topic_kb_timestamp
AFTER UPDATE ON topic_knowledge_base
FOR EACH ROW
BEGIN
    UPDATE topic_knowledge_base
    SET last_updated = CURRENT_TIMESTAMP
    WHERE knowledge_id = NEW.knowledge_id;
END;

-- Update updated_at timestamp for topic_qa_pairs
CREATE TRIGGER IF NOT EXISTS update_topic_qa_timestamp
AFTER UPDATE ON topic_qa_pairs
FOR EACH ROW
BEGIN
    UPDATE topic_qa_pairs
    SET updated_at = CURRENT_TIMESTAMP
    WHERE qa_id = NEW.qa_id;
END;

-- Increment view_count when knowledge base is accessed
CREATE TRIGGER IF NOT EXISTS increment_kb_view_count
AFTER INSERT ON student_qa_interactions
FOR EACH ROW
WHEN NEW.response_source = 'direct_qa'
BEGIN
    UPDATE topic_knowledge_base
    SET view_count = view_count + 1
    WHERE topic_id = (SELECT topic_id FROM topic_qa_pairs WHERE qa_id = NEW.qa_id);
END;

-- Increment times_asked when QA pair is used
CREATE TRIGGER IF NOT EXISTS increment_qa_times_asked
AFTER INSERT ON student_qa_interactions
FOR EACH ROW
BEGIN
    UPDATE topic_qa_pairs
    SET times_asked = times_asked + 1
    WHERE qa_id = NEW.qa_id;
END;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Popular QA Pairs (for analytics)
CREATE VIEW IF NOT EXISTS v_popular_qa_pairs AS
SELECT 
    qa.qa_id,
    qa.topic_id,
    t.topic_title,
    qa.question,
    qa.answer,
    qa.difficulty_level,
    qa.times_asked,
    qa.average_student_rating,
    qa.success_rate,
    COUNT(sqa.interaction_id) as interaction_count,
    AVG(CASE WHEN sqa.was_helpful = 1 THEN 1.0 ELSE 0.0 END) as helpfulness_rate
FROM topic_qa_pairs qa
LEFT JOIN course_topics t ON qa.topic_id = t.topic_id
LEFT JOIN student_qa_interactions sqa ON qa.qa_id = sqa.qa_id
WHERE qa.is_active = TRUE
GROUP BY qa.qa_id
ORDER BY qa.times_asked DESC, qa.average_student_rating DESC;

-- View: Topic Learning Paths (prerequisite chains)
CREATE VIEW IF NOT EXISTS v_topic_learning_paths AS
SELECT 
    t1.topic_id,
    t1.topic_title,
    t1.topic_order,
    tp.prerequisite_topic_id,
    t2.topic_title as prerequisite_title,
    tp.importance_level,
    tp.strength_score
FROM course_topics t1
LEFT JOIN topic_prerequisites tp ON t1.topic_id = tp.topic_id
LEFT JOIN course_topics t2 ON tp.prerequisite_topic_id = t2.topic_id
WHERE t1.is_active = TRUE
ORDER BY t1.topic_order, tp.strength_score DESC;

-- View: Knowledge Base Quality Report
CREATE VIEW IF NOT EXISTS v_kb_quality_report AS
SELECT 
    kb.knowledge_id,
    kb.topic_id,
    t.topic_title,
    kb.content_quality_score,
    kb.is_validated,
    kb.view_count,
    kb.usefulness_rating,
    COUNT(qa.qa_id) as qa_pairs_count,
    AVG(qa.quality_score) as avg_qa_quality,
    kb.last_updated
FROM topic_knowledge_base kb
LEFT JOIN course_topics t ON kb.topic_id = t.topic_id
LEFT JOIN topic_qa_pairs qa ON kb.topic_id = qa.topic_id AND qa.is_active = TRUE
GROUP BY kb.knowledge_id
ORDER BY kb.content_quality_score DESC, kb.view_count DESC;

-- ============================================================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================================================

-- Sample: Insert a test knowledge base entry (commented out for production)
/*
INSERT INTO topic_knowledge_base (
    topic_id, topic_summary, key_concepts, learning_objectives,
    content_quality_score, extraction_model
) VALUES (
    1,  -- Assuming topic_id 1 exists from previous migration
    'Hücre, tüm canlıların temel yapı ve işlev birimidir. Robert Hooke 1665 yılında mikroskopla mantar hücrelerini inceleyerek ilk kez hücre terimini kullanmıştır.',
    '[{"term": "Hücre", "definition": "Canlıların temel yapı birimi", "importance": "high"}, {"term": "Hücre Teorisi", "definition": "Tüm canlılar hücrelerden oluşur", "importance": "high"}]',
    '[{"level": "remember", "objective": "Öğrenci hücre tanımını yapabilmeli"}, {"level": "understand", "objective": "Öğrenci hücre teorisinin temel ilkelerini açıklayabilmeli"}]',
    0.85,
    'llama-3.1-8b-instant'
);
*/

-- Verify tables were created successfully
SELECT 'Knowledge Base tables created successfully' as status;
SELECT COUNT(*) as kb_tables_count FROM sqlite_master 
WHERE type='table' AND name IN (
    'topic_knowledge_base', 
    'topic_qa_pairs',
    'topic_prerequisites',
    'qa_similarity_cache',
    'student_qa_interactions'
);






