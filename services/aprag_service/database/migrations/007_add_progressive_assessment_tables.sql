-- Progressive Assessment System Database Schema
-- Migration 007: Add tables for progressive assessment flow

-- Main progressive assessments table
CREATE TABLE IF NOT EXISTS progressive_assessments (
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    stage TEXT NOT NULL CHECK(stage IN ('initial', 'follow_up', 'deep_analysis')),
    
    -- Follow-up assessment data
    confidence_level INTEGER CHECK(confidence_level >= 1 AND confidence_level <= 5),
    has_questions BOOLEAN,
    application_understanding TEXT,
    
    -- Deep analysis assessment data
    confusion_areas TEXT, -- JSON array
    requested_topics TEXT, -- JSON array
    alternative_explanation_request TEXT,
    
    -- Common fields
    comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (interaction_id) REFERENCES student_interactions(interaction_id)
);

-- Add progressive assessment columns to student_interactions table
ALTER TABLE student_interactions ADD COLUMN progressive_assessment_data TEXT DEFAULT NULL;
ALTER TABLE student_interactions ADD COLUMN progressive_assessment_stage TEXT DEFAULT NULL;

-- Add progressive assessment columns to student_profiles table
ALTER TABLE student_profiles ADD COLUMN average_confidence REAL DEFAULT 2.5;
ALTER TABLE student_profiles ADD COLUMN application_readiness REAL DEFAULT 0.5;
ALTER TABLE student_profiles ADD COLUMN progressive_assessment_count INTEGER DEFAULT 0;
ALTER TABLE student_profiles ADD COLUMN has_active_questions BOOLEAN DEFAULT FALSE;

-- Concept confusion tracking
CREATE TABLE IF NOT EXISTS concept_confusion_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    confusion_area TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Requested topics tracking
CREATE TABLE IF NOT EXISTS requested_topics_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    requested_topic TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Progressive assessment analytics summary table
CREATE TABLE IF NOT EXISTS progressive_assessment_summary (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    
    -- Assessment counts
    total_assessments INTEGER DEFAULT 0,
    follow_up_count INTEGER DEFAULT 0,
    deep_analysis_count INTEGER DEFAULT 0,
    
    -- Confidence metrics
    avg_confidence REAL DEFAULT 2.5,
    confidence_trend TEXT DEFAULT 'stable', -- improving, declining, stable
    min_confidence INTEGER DEFAULT 1,
    max_confidence INTEGER DEFAULT 5,
    
    -- Application readiness
    avg_application_readiness REAL DEFAULT 0.5,
    application_improvement_trend TEXT DEFAULT 'stable',
    
    -- Confusion patterns
    most_common_confusions TEXT, -- JSON array
    confusion_frequency INTEGER DEFAULT 0,
    
    -- Learning insights
    concept_mastery_level REAL DEFAULT 0.5,
    needs_intervention BOOLEAN DEFAULT FALSE,
    intervention_reasons TEXT, -- JSON array
    
    -- Timestamps
    last_assessment TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, session_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_progressive_assessments_interaction ON progressive_assessments(interaction_id);
CREATE INDEX IF NOT EXISTS idx_progressive_assessments_user_session ON progressive_assessments(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_progressive_assessments_stage ON progressive_assessments(stage);
CREATE INDEX IF NOT EXISTS idx_progressive_assessments_timestamp ON progressive_assessments(timestamp);

CREATE INDEX IF NOT EXISTS idx_concept_confusion_user_session ON concept_confusion_log(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_concept_confusion_area ON concept_confusion_log(confusion_area);
CREATE INDEX IF NOT EXISTS idx_concept_confusion_timestamp ON concept_confusion_log(timestamp);

CREATE INDEX IF NOT EXISTS idx_requested_topics_user_session ON requested_topics_log(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_requested_topics_topic ON requested_topics_log(requested_topic);
CREATE INDEX IF NOT EXISTS idx_requested_topics_timestamp ON requested_topics_log(timestamp);

CREATE INDEX IF NOT EXISTS idx_progressive_summary_user_session ON progressive_assessment_summary(user_id, session_id);

-- Triggers for automatic summary updates
CREATE TRIGGER IF NOT EXISTS update_progressive_summary_after_assessment
AFTER INSERT ON progressive_assessments
BEGIN
    INSERT OR REPLACE INTO progressive_assessment_summary (
        user_id, 
        session_id,
        total_assessments,
        follow_up_count,
        deep_analysis_count,
        avg_confidence,
        last_assessment,
        last_updated
    )
    SELECT 
        NEW.user_id,
        NEW.session_id,
        COUNT(*) as total_assessments,
        SUM(CASE WHEN stage = 'follow_up' THEN 1 ELSE 0 END) as follow_up_count,
        SUM(CASE WHEN stage = 'deep_analysis' THEN 1 ELSE 0 END) as deep_analysis_count,
        AVG(CASE WHEN confidence_level IS NOT NULL THEN confidence_level ELSE 2.5 END) as avg_confidence,
        MAX(timestamp) as last_assessment,
        CURRENT_TIMESTAMP
    FROM progressive_assessments
    WHERE user_id = NEW.user_id AND session_id = NEW.session_id;
END;

-- Trigger for confusion area frequency updates
CREATE TRIGGER IF NOT EXISTS update_confusion_frequency
AFTER INSERT ON concept_confusion_log
BEGIN
    UPDATE concept_confusion_log 
    SET frequency = (
        SELECT COUNT(*) 
        FROM concept_confusion_log 
        WHERE user_id = NEW.user_id 
        AND session_id = NEW.session_id 
        AND confusion_area = NEW.confusion_area
    )
    WHERE user_id = NEW.user_id 
    AND session_id = NEW.session_id 
    AND confusion_area = NEW.confusion_area;
END;

-- Trigger for requested topic frequency updates
CREATE TRIGGER IF NOT EXISTS update_topic_request_frequency
AFTER INSERT ON requested_topics_log
BEGIN
    UPDATE requested_topics_log 
    SET frequency = (
        SELECT COUNT(*) 
        FROM requested_topics_log 
        WHERE user_id = NEW.user_id 
        AND session_id = NEW.session_id 
        AND requested_topic = NEW.requested_topic
    )
    WHERE user_id = NEW.user_id 
    AND session_id = NEW.session_id 
    AND requested_topic = NEW.requested_topic;
END;

-- View for easy progressive assessment analytics
CREATE VIEW IF NOT EXISTS progressive_assessment_analytics AS
SELECT 
    pa.user_id,
    pa.session_id,
    pa.assessment_id,
    pa.interaction_id,
    pa.stage,
    pa.confidence_level,
    pa.has_questions,
    pa.application_understanding,
    pa.confusion_areas,
    pa.requested_topics,
    pa.timestamp,
    
    -- Interaction context
    si.query as original_query,
    si.original_response,
    si.emoji_feedback,
    si.feedback_score,
    
    -- Progressive assessment summary data
    pas.avg_confidence as user_avg_confidence,
    pas.concept_mastery_level,
    pas.needs_intervention
    
FROM progressive_assessments pa
LEFT JOIN student_interactions si ON pa.interaction_id = si.interaction_id
LEFT JOIN progressive_assessment_summary pas ON pa.user_id = pas.user_id AND pa.session_id = pas.session_id;

-- Insert initial feature flag for progressive assessment
INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description, created_at) 
VALUES ('progressive_assessment', 1, 'Enable progressive assessment flow system', CURRENT_TIMESTAMP);