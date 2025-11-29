-- Migration 009: Create Module Extraction Tables
-- Creates tables for curriculum-aware module organization system

BEGIN TRANSACTION;

-- =============================================================================
-- 1. Courses Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS courses (
    course_id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code VARCHAR(50) NOT NULL,
    course_name VARCHAR(255) NOT NULL,
    course_description TEXT,

    -- Curriculum Information
    curriculum_standard VARCHAR(100) NOT NULL, -- e.g., "MEB_2018", "IB_DP", "AP"
    subject_area VARCHAR(100) NOT NULL, -- e.g., "biology", "mathematics", "physics"
    grade_level VARCHAR(50) NOT NULL, -- e.g., "9", "10-11", "high_school"
    academic_year VARCHAR(20), -- e.g., "2024-2025"

    -- Course Metadata
    total_hours INTEGER,
    difficulty_level VARCHAR(20) DEFAULT 'intermediate',
    language VARCHAR(10) DEFAULT 'tr',

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    UNIQUE(course_code)
);

-- Create indexes for courses
CREATE INDEX IF NOT EXISTS idx_courses_curriculum ON courses(curriculum_standard, subject_area, grade_level);
CREATE INDEX IF NOT EXISTS idx_courses_active ON courses(is_active, subject_area);

-- =============================================================================
-- 2. Course Modules Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS course_modules (
    module_id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL,
    session_id VARCHAR(255), -- Link to document processing session (optional)

    -- Module Identity
    module_code VARCHAR(100) NOT NULL,
    module_title VARCHAR(255) NOT NULL,
    module_description TEXT,

    -- Curriculum Alignment
    curriculum_unit_code VARCHAR(100), -- Official curriculum unit code from MEB
    learning_standards TEXT, -- JSON: Official learning standards/objectives

    -- Module Organization
    module_order INTEGER NOT NULL,
    parent_module_id INTEGER, -- For nested modules (optional)

    -- Academic Properties
    estimated_duration_hours INTEGER,
    difficulty_level VARCHAR(20) DEFAULT 'intermediate',
    prerequisites TEXT, -- JSON: [module_id1, module_id2, ...]
    learning_outcomes TEXT, -- JSON: Expected learning outcomes

    -- Assessment Information
    assessment_methods TEXT, -- JSON: ["quiz", "project", "exam"]
    passing_criteria TEXT, -- JSON: Criteria for module completion

    -- Content Metadata
    topic_count INTEGER DEFAULT 0,
    total_content_chunks INTEGER DEFAULT 0,
    estimated_reading_time_minutes INTEGER,

    -- Extraction Information
    extraction_method VARCHAR(50) DEFAULT 'llm_analysis',
    extraction_model VARCHAR(100),
    extraction_confidence DECIMAL(3,2),
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign Keys
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_module_id) REFERENCES course_modules(module_id) ON DELETE SET NULL,

    -- Constraints
    UNIQUE(course_id, module_code)
);

-- Create indexes for course_modules
CREATE INDEX IF NOT EXISTS idx_modules_course ON course_modules(course_id, module_order);
CREATE INDEX IF NOT EXISTS idx_modules_session ON course_modules(session_id);
CREATE INDEX IF NOT EXISTS idx_modules_active ON course_modules(is_active, course_id);

-- =============================================================================
-- 3. Module-Topic Relationships Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS module_topic_relationships (
    relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
    module_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,

    -- Relationship Properties
    relationship_type VARCHAR(50) DEFAULT 'contains', -- contains, references, prerequisite
    topic_order_in_module INTEGER,
    importance_level VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical

    -- Learning Progression
    is_prerequisite BOOLEAN DEFAULT FALSE,
    prerequisite_topics TEXT, -- JSON: [topic_id1, topic_id2] within this module

    -- Content Analysis
    content_coverage_percentage DECIMAL(5,2), -- How much of topic is covered in this module
    estimated_study_time_minutes INTEGER,

    -- Extraction Metadata
    extraction_confidence DECIMAL(3,2),
    extraction_method VARCHAR(50) DEFAULT 'llm_analysis',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign Keys
    FOREIGN KEY (module_id) REFERENCES course_modules(module_id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE,

    -- Constraints
    UNIQUE(module_id, topic_id),
    CHECK(content_coverage_percentage >= 0 AND content_coverage_percentage <= 100)
);

-- Create indexes for module_topic_relationships
CREATE INDEX IF NOT EXISTS idx_module_topics_module ON module_topic_relationships(module_id, topic_order_in_module);
CREATE INDEX IF NOT EXISTS idx_module_topics_topic ON module_topic_relationships(topic_id);
CREATE INDEX IF NOT EXISTS idx_module_topics_importance ON module_topic_relationships(module_id, importance_level);

-- =============================================================================
-- 4. Module Progress Tracking Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS module_progress (
    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL, -- References users table (INTEGER)
    session_id VARCHAR(255) NOT NULL, -- Document processing session
    module_id INTEGER NOT NULL,

    -- Progress Metrics
    completion_status VARCHAR(20) DEFAULT 'not_started', -- not_started, in_progress, completed, mastered
    completion_percentage DECIMAL(5,2) DEFAULT 0.0,
    topics_completed INTEGER DEFAULT 0,
    topics_total INTEGER DEFAULT 0,

    -- Time Tracking
    time_spent_minutes INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    last_activity_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Learning Assessment
    average_topic_understanding DECIMAL(3,2), -- Average from topic progress
    module_assessment_score DECIMAL(5,2), -- Overall module assessment
    mastery_level VARCHAR(20), -- not_assessed, developing, proficient, advanced

    -- Learning Path
    is_prerequisite_met BOOLEAN DEFAULT TRUE,
    next_recommended_module_id INTEGER,

    -- Adaptive Learning
    difficulty_adjustment VARCHAR(20) DEFAULT 'none', -- easier, none, harder
    personalized_path TEXT, -- JSON: Customized learning sequence

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign Keys
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (module_id) REFERENCES course_modules(module_id) ON DELETE CASCADE,
    FOREIGN KEY (next_recommended_module_id) REFERENCES course_modules(module_id) ON DELETE SET NULL,

    -- Constraints
    UNIQUE(user_id, session_id, module_id),
    CHECK(completion_percentage >= 0 AND completion_percentage <= 100)
);

-- Create indexes for module_progress
CREATE INDEX IF NOT EXISTS idx_module_progress_user_session ON module_progress(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_module_progress_module ON module_progress(module_id);
CREATE INDEX IF NOT EXISTS idx_module_progress_status ON module_progress(completion_status);
CREATE INDEX IF NOT EXISTS idx_module_progress_activity ON module_progress(last_activity_at DESC);

-- =============================================================================
-- 5. Curriculum Standards Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS curriculum_standards (
    standard_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Standard Identity
    curriculum_type VARCHAR(100) NOT NULL, -- e.g., "MEB_2018", "IB_DP"
    subject_area VARCHAR(100) NOT NULL,
    grade_level VARCHAR(50) NOT NULL,

    -- Standard Details
    standard_code VARCHAR(100) NOT NULL, -- Official curriculum code
    standard_title VARCHAR(500) NOT NULL,
    standard_description TEXT,

    -- Hierarchy
    parent_standard_id INTEGER, -- For nested standards
    standard_level INTEGER DEFAULT 1, -- 1=main, 2=sub, 3=detailed

    -- Learning Specifications
    expected_outcomes TEXT, -- JSON: Learning outcomes
    assessment_criteria TEXT, -- JSON: How to assess this standard
    time_allocation_hours INTEGER,

    -- Additional Metadata
    official_source_url TEXT, -- Link to official curriculum document
    last_updated_official DATE, -- When was this standard last updated officially

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign Keys
    FOREIGN KEY (parent_standard_id) REFERENCES curriculum_standards(standard_id) ON DELETE SET NULL,

    -- Constraints
    UNIQUE(curriculum_type, subject_area, grade_level, standard_code)
);

-- Create indexes for curriculum_standards
CREATE INDEX IF NOT EXISTS idx_curriculum_standards_type ON curriculum_standards(curriculum_type, subject_area, grade_level);
CREATE INDEX IF NOT EXISTS idx_curriculum_standards_code ON curriculum_standards(standard_code);
CREATE INDEX IF NOT EXISTS idx_curriculum_standards_active ON curriculum_standards(is_active);

-- =============================================================================
-- 6. Module Templates Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS module_templates (
    template_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Template Identity
    template_name VARCHAR(255) NOT NULL,
    curriculum_type VARCHAR(100) NOT NULL,
    subject_area VARCHAR(100) NOT NULL,
    grade_levels TEXT, -- JSON: ["9", "10", "11"] - applicable grade levels

    -- Template Content
    module_structure_template TEXT NOT NULL, -- JSON: Module organization template
    prompt_template TEXT NOT NULL, -- LLM prompt template for module extraction
    learning_outcomes_template TEXT, -- JSON: Standard learning outcomes structure
    assessment_template TEXT, -- JSON: Assessment methods and criteria

    -- Template Configuration
    default_module_duration_hours INTEGER DEFAULT 20,
    max_topics_per_module INTEGER DEFAULT 15,
    min_topics_per_module INTEGER DEFAULT 3,
    prerequisite_depth INTEGER DEFAULT 2, -- How deep to analyze prerequisites

    -- Template Metadata
    template_version VARCHAR(20) DEFAULT '1.0',
    created_by VARCHAR(255),
    is_default BOOLEAN DEFAULT FALSE, -- Is this the default template for this subject/grade?

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    UNIQUE(curriculum_type, subject_area, template_name)
);

-- Create indexes for module_templates
CREATE INDEX IF NOT EXISTS idx_module_templates_curriculum ON module_templates(curriculum_type, subject_area);
CREATE INDEX IF NOT EXISTS idx_module_templates_default ON module_templates(is_default, is_active);

-- =============================================================================
-- 7. Module Extraction Jobs Table (for tracking background jobs)
-- =============================================================================
CREATE TABLE IF NOT EXISTS module_extraction_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    course_id INTEGER,

    -- Job Details
    extraction_method VARCHAR(50) DEFAULT 'llm_analysis',
    extraction_options TEXT, -- JSON: Extraction configuration

    -- Job Status
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    progress_percentage DECIMAL(5,2) DEFAULT 0.0,
    current_operation VARCHAR(255),
    
    -- Job Results
    modules_created INTEGER DEFAULT 0,
    topics_processed INTEGER DEFAULT 0,
    extraction_confidence DECIMAL(3,2),
    results TEXT, -- JSON: Detailed results
    error_message TEXT,

    -- Timing
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign Key
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE SET NULL
);

-- Create indexes for module_extraction_jobs
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_session ON module_extraction_jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_status ON module_extraction_jobs(status, created_at DESC);

-- =============================================================================
-- 8. Insert Sample Data
-- =============================================================================

-- Insert sample Turkish MEB curriculum standards
INSERT OR IGNORE INTO curriculum_standards (
    curriculum_type, subject_area, grade_level, standard_code, standard_title,
    expected_outcomes, time_allocation_hours
) VALUES
(
    'MEB_2018', 'biology', '9', 'B.9.1.1',
    'Canlıların temel birimi olan hücreyi tanıyabilir',
    '["Hücre teorisini açıklayabilir", "Prokaryot ve ökaryot hücreleri karşılaştırabilir", "Hücre organellerinin görevlerini açıklayabilir"]',
    12
),
(
    'MEB_2018', 'biology', '9', 'B.9.1.2',
    'Hücre zarının yapı ve görevlerini açıklayabilir',
    '["Hücre zarının akıcı mozaik yapısını açıklayabilir", "Osmoz ve difüzyon olaylarını karşılaştırabilir", "Aktif ve pasif taşıma türlerini ayırt edebilir"]',
    8
),
(
    'MEB_2018', 'biology', '9', 'B.9.2.1',
    'Mitoz ve mayoz bölünme süreçlerini karşılaştırır',
    '["Mitoz bölünme aşamalarını açıklayabilir", "Mayoz bölünmenin önemini kavrar", "Hücre döngüsünü anlayabilir"]',
    16
),
(
    'MEB_2018', 'mathematics', '10', 'M.10.1.1',
    'Fonksiyon kavramını anlar ve örnekler verir',
    '["Fonksiyon tanımını yapar", "Günlük yaşamdan fonksiyon örnekleri verir", "Fonksiyonları grafikle gösterir"]',
    20
),
(
    'MEB_2018', 'physics', '9', 'F.9.1.1',
    'Fizik biliminin doğası ve çalışma alanlarını açıklar',
    '["Fizik biliminin önemini kavrar", "Fizik dallarını tanır", "Bilimsel yöntemi uygular"]',
    10
);

-- Insert sample course
INSERT OR IGNORE INTO courses (
    course_code, course_name, curriculum_standard, subject_area, grade_level, total_hours
) VALUES
(
    'BIO_9_MEB', '9. Sınıf Biyoloji', 'MEB_2018', 'biology', '9', 144
),
(
    'MAT_10_MEB', '10. Sınıf Matematik', 'MEB_2018', 'mathematics', '10', 180
),
(
    'FIZ_9_MEB', '9. Sınıf Fizik', 'MEB_2018', 'physics', '9', 108
);

-- =============================================================================
-- 9. Create Analytics Views
-- =============================================================================

-- Module completion overview
CREATE VIEW IF NOT EXISTS v_module_completion_overview AS
SELECT
    cm.module_id,
    cm.module_title,
    c.course_name,
    cm.estimated_duration_hours,
    COUNT(DISTINCT mp.user_id) as total_students,
    COUNT(DISTINCT CASE WHEN mp.completion_status = 'completed' THEN mp.user_id END) as completed_students,
    ROUND(
        COUNT(DISTINCT CASE WHEN mp.completion_status = 'completed' THEN mp.user_id END) * 100.0 /
        NULLIF(COUNT(DISTINCT mp.user_id), 0), 2
    ) as completion_rate_percentage,
    AVG(mp.time_spent_minutes) as avg_time_spent_minutes,
    AVG(mp.completion_percentage) as avg_completion_percentage
FROM course_modules cm
LEFT JOIN courses c ON cm.course_id = c.course_id
LEFT JOIN module_progress mp ON cm.module_id = mp.module_id
WHERE cm.is_active = TRUE
GROUP BY cm.module_id, cm.module_title, c.course_name, cm.estimated_duration_hours;

-- Student module progress dashboard
CREATE VIEW IF NOT EXISTS v_student_module_progress AS
SELECT
    mp.user_id,
    c.course_name,
    cm.module_title,
    cm.module_order,
    mp.completion_status,
    mp.completion_percentage,
    mp.topics_completed,
    cm.topic_count as topics_total,
    mp.time_spent_minutes,
    mp.average_topic_understanding,
    mp.last_activity_at,
    CASE
        WHEN mp.completion_status = 'completed' THEN 'Completed'
        WHEN mp.completion_status = 'in_progress' THEN 'In Progress'
        WHEN mp.completion_status = 'not_started' THEN 'Not Started'
        ELSE 'Unknown'
    END as status_display
FROM module_progress mp
JOIN course_modules cm ON mp.module_id = cm.module_id
JOIN courses c ON cm.course_id = c.course_id
WHERE cm.is_active = TRUE
ORDER BY mp.user_id, c.course_name, cm.module_order;

-- Curriculum alignment overview
CREATE VIEW IF NOT EXISTS v_curriculum_alignment AS
SELECT
    c.course_code,
    c.course_name,
    c.curriculum_standard,
    c.subject_area,
    c.grade_level,
    COUNT(DISTINCT cm.module_id) as total_modules,
    COUNT(DISTINCT cs.standard_id) as aligned_standards,
    ROUND(
        COUNT(DISTINCT cs.standard_id) * 100.0 /
        NULLIF(COUNT(DISTINCT cm.module_id), 0), 2
    ) as alignment_percentage
FROM courses c
LEFT JOIN course_modules cm ON c.course_id = cm.course_id AND cm.is_active = TRUE
LEFT JOIN curriculum_standards cs ON c.curriculum_standard = cs.curriculum_type
    AND c.subject_area = cs.subject_area
    AND c.grade_level = cs.grade_level
WHERE c.is_active = TRUE
GROUP BY c.course_id, c.course_code, c.course_name, c.curriculum_standard, c.subject_area, c.grade_level;

COMMIT;