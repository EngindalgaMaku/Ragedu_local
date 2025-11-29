"""
Integration Test for Module Extraction Service
Tests the complete module extraction pipeline from database to API
"""

import pytest
import asyncio
import json
import os
import tempfile
import sqlite3
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import DatabaseManager
from services.module_extraction_service import ModuleExtractionService
from validators.module_quality_validator import ModuleQualityValidator, ValidationSeverity
from templates.curriculum_templates import CurriculumTemplateManager
from processors.llm_module_organizer import LLMModuleOrganizer
from config.feature_flags import FeatureFlags
from utils.logging_config import get_module_logger


class TestModuleExtractionIntegration:
    """Comprehensive integration tests for module extraction"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary test database"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Initialize database with schema
        conn = sqlite3.connect(path)
        self._create_test_schema(conn)
        conn.close()
        
        yield path
        
        # Cleanup
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def db_manager(self, temp_db):
        """Create database manager with test database"""
        return DatabaseManager(temp_db)

    @pytest.fixture
    def extraction_service(self, db_manager):
        """Create module extraction service"""
        return ModuleExtractionService(db_manager)

    def _create_test_schema(self, conn):
        """Create test database schema"""
        # Create required tables
        schema_queries = [
            # Course topics table
            """
            CREATE TABLE course_topics (
                topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                topic_title TEXT NOT NULL,
                parent_topic_id INTEGER,
                topic_order INTEGER DEFAULT 0,
                description TEXT,
                keywords TEXT,
                estimated_difficulty TEXT DEFAULT 'intermediate',
                prerequisites TEXT,
                related_chunk_ids TEXT,
                extraction_method TEXT DEFAULT 'llm_analysis',
                extraction_confidence REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_topic_id) REFERENCES course_topics(topic_id)
            )
            """,
            
            # Course modules table
            """
            CREATE TABLE course_modules (
                module_id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                module_code TEXT NOT NULL,
                module_title TEXT NOT NULL,
                module_description TEXT,
                module_order INTEGER DEFAULT 0,
                estimated_duration_hours REAL DEFAULT 20.0,
                difficulty_level TEXT DEFAULT 'intermediate',
                prerequisites TEXT,
                learning_outcomes TEXT,
                assessment_methods TEXT,
                topic_count INTEGER DEFAULT 0,
                extraction_method TEXT DEFAULT 'llm_analysis',
                extraction_model TEXT,
                extraction_confidence REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Module topic relationships table
            """
            CREATE TABLE module_topic_relationships (
                relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                relationship_type TEXT DEFAULT 'contains',
                topic_order_in_module INTEGER DEFAULT 0,
                importance_level TEXT DEFAULT 'medium',
                content_coverage_percentage REAL DEFAULT 100.0,
                extraction_confidence REAL DEFAULT 0.0,
                extraction_method TEXT DEFAULT 'llm_analysis',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (module_id) REFERENCES course_modules(module_id) ON DELETE CASCADE,
                FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
            )
            """,
            
            # Courses table
            """
            CREATE TABLE courses (
                course_id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL UNIQUE,
                course_name TEXT NOT NULL,
                course_description TEXT,
                curriculum_standard TEXT DEFAULT 'MEB_2018',
                subject_area TEXT NOT NULL,
                grade_level TEXT NOT NULL,
                total_hours INTEGER DEFAULT 144,
                difficulty_level TEXT DEFAULT 'intermediate',
                language TEXT DEFAULT 'tr',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Curriculum standards table
            """
            CREATE TABLE curriculum_standards (
                standard_id INTEGER PRIMARY KEY AUTOINCREMENT,
                standard_code TEXT NOT NULL,
                curriculum_type TEXT NOT NULL DEFAULT 'MEB_2018',
                subject_area TEXT NOT NULL,
                grade_level TEXT NOT NULL,
                standard_title TEXT NOT NULL,
                description TEXT,
                expected_outcomes TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(standard_code, curriculum_type)
            )
            """,
            
            # Module progress table
            """
            CREATE TABLE module_progress (
                progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                module_id INTEGER NOT NULL,
                completion_status TEXT DEFAULT 'not_started',
                completion_percentage REAL DEFAULT 0.0,
                time_spent_minutes REAL DEFAULT 0.0,
                topics_total INTEGER DEFAULT 0,
                topics_completed INTEGER DEFAULT 0,
                current_topic_id INTEGER,
                is_prerequisite_met BOOLEAN DEFAULT TRUE,
                last_accessed_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (module_id) REFERENCES course_modules(module_id) ON DELETE CASCADE,
                FOREIGN KEY (current_topic_id) REFERENCES course_topics(topic_id),
                UNIQUE(user_id, session_id, module_id)
            )
            """
        ]
        
        for query in schema_queries:
            conn.execute(query)
        
        # Insert test data
        self._insert_test_data(conn)
        conn.commit()

    def _insert_test_data(self, conn):
        """Insert test data"""
        # Insert test course
        conn.execute("""
            INSERT INTO courses (
                course_code, course_name, course_description,
                curriculum_standard, subject_area, grade_level
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "BIO10", "10. Sınıf Biyoloji", "Lise biyoloji dersi",
            "MEB_2018", "Biology", "10"
        ))
        
        course_id = conn.lastrowid
        
        # Insert test topics
        test_topics = [
            {
                "session_id": "test_session_123",
                "topic_title": "Hücre Yapısı ve Organelleri",
                "topic_order": 1,
                "description": "Hücre temel yapısı ve organeller",
                "keywords": ["hücre", "organeller", "sitoplazma"],
                "estimated_difficulty": "intermediate"
            },
            {
                "session_id": "test_session_123",
                "topic_title": "DNA ve RNA",
                "topic_order": 2,
                "description": "Genetik materyal yapısı",
                "keywords": ["DNA", "RNA", "genetik"],
                "estimated_difficulty": "advanced"
            },
            {
                "session_id": "test_session_123",
                "topic_title": "Protein Sentezi",
                "topic_order": 3,
                "description": "Protein üretim süreçleri",
                "keywords": ["protein", "sentez", "ribosom"],
                "estimated_difficulty": "advanced"
            },
            {
                "session_id": "test_session_123",
                "topic_title": "Hücresel Solunum",
                "topic_order": 4,
                "description": "Enerji üretimi süreçleri",
                "keywords": ["solunum", "ATP", "mitokondri"],
                "estimated_difficulty": "intermediate"
            },
            {
                "session_id": "test_session_123",
                "topic_title": "Fotosentez",
                "topic_order": 5,
                "description": "Bitkilerde besin üretimi",
                "keywords": ["fotosentez", "kloroplast", "glikoz"],
                "estimated_difficulty": "intermediate"
            }
        ]
        
        for topic in test_topics:
            conn.execute("""
                INSERT INTO course_topics (
                    session_id, topic_title, topic_order, description, 
                    keywords, estimated_difficulty
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                topic["session_id"],
                topic["topic_title"],
                topic["topic_order"],
                topic["description"],
                json.dumps(topic["keywords"]),
                topic["estimated_difficulty"]
            ))
        
        # Insert curriculum standards
        standards = [
            {
                "standard_code": "B.10.1.1",
                "subject_area": "Biology",
                "grade_level": "10",
                "title": "Hücre Yapısı ve İşlevi",
                "outcomes": ["Hücre organellerini tanır", "Organellerin işlevlerini açıklar"]
            },
            {
                "standard_code": "B.10.2.1",
                "subject_area": "Biology", 
                "grade_level": "10",
                "title": "Genetik Materyal",
                "outcomes": ["DNA ve RNA yapısını açıklar", "Genetik bilgi akışını tanımlar"]
            }
        ]
        
        for standard in standards:
            conn.execute("""
                INSERT INTO curriculum_standards (
                    standard_code, curriculum_type, subject_area, grade_level,
                    standard_title, expected_outcomes
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                standard["standard_code"],
                "MEB_2018",
                standard["subject_area"],
                standard["grade_level"],
                standard["title"],
                json.dumps(standard["outcomes"])
            ))

    def test_database_schema_creation(self, temp_db):
        """Test that database schema is created correctly"""
        conn = sqlite3.connect(temp_db)
        
        # Check that all required tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN (
                'course_modules', 'module_topic_relationships', 
                'curriculum_standards', 'module_progress'
            )
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            'course_modules', 'module_topic_relationships',
            'curriculum_standards', 'module_progress'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found in database"
        
        conn.close()

    def test_feature_flags_integration(self):
        """Test feature flags for module extraction"""
        # Test module extraction feature flag
        assert hasattr(FeatureFlags, 'MODULE_EXTRACTION_ENABLED')
        assert hasattr(FeatureFlags, 'is_module_extraction_enabled')
        
        # Test feature flag functionality
        enabled = FeatureFlags.is_module_extraction_enabled()
        assert isinstance(enabled, bool)
        
        # Test hierarchical dependency (module extraction requires APRAG)
        FeatureFlags.set_flag(FeatureFlags.APRAG_ENABLED, False)
        FeatureFlags.set_flag(FeatureFlags.MODULE_EXTRACTION_ENABLED, True)
        
        # Should return False because APRAG is disabled
        assert not FeatureFlags.is_module_extraction_enabled()

    def test_curriculum_template_manager(self):
        """Test curriculum template manager functionality"""
        template_manager = CurriculumTemplateManager()
        
        # Test available subjects
        subjects = template_manager.get_available_subjects()
        assert "Biology" in subjects
        assert isinstance(subjects, list)
        
        # Test available grades for Biology
        grades = template_manager.get_available_grades("Biology")
        assert "10" in grades
        assert isinstance(grades, list)
        
        # Test template retrieval
        template = template_manager.get_template("Biology", "10")
        assert template is not None
        assert "prompt_template" in template
        assert "curriculum_standards" in template

    @pytest.mark.asyncio
    async def test_module_quality_validator(self):
        """Test module quality validation"""
        validator = ModuleQualityValidator()
        
        # Test module set with various quality issues
        test_modules = [
            {
                "module_title": "Test Module 1",
                "module_description": "Valid module",
                "topics": [
                    {"topic_id": 1, "topic_title": "Topic 1"},
                    {"topic_id": 2, "topic_title": "Topic 2"}
                ],
                "estimated_duration_hours": 20,
                "difficulty_level": "intermediate"
            },
            {
                # Missing required fields - should generate validation issues
                "topics": [],
                "estimated_duration_hours": -5  # Invalid duration
            },
            {
                "module_title": "Very Very Long Module Title That Exceeds The Maximum Recommended Length For Educational Modules",
                "topics": [{"topic_id": 3}],  # Too few topics
                "estimated_duration_hours": 100  # Too long duration
            }
        ]
        
        course_context = {
            "subject": "Biology",
            "grade_level": "10",
            "curriculum_standard": "MEB_2018",
            "total_hours": 144
        }
        
        validation_options = {
            "apply_auto_fixes": True,
            "strict_curriculum_validation": True
        }
        
        # Run validation
        is_valid, issues, corrected_modules = await validator.validate_module_set(
            test_modules, course_context, validation_options
        )
        
        # Assertions
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
        assert len(corrected_modules) == len(test_modules)
        
        # Should have validation issues
        assert len(issues) > 0
        
        # Check that auto-fixes were applied
        for module in corrected_modules:
            assert "module_title" in module
            assert "estimated_duration_hours" in module
            assert module["estimated_duration_hours"] > 0

    def test_llm_module_organizer_initialization(self):
        """Test LLM module organizer initialization"""
        organizer = LLMModuleOrganizer()
        
        # Test initialization
        assert organizer is not None
        assert hasattr(organizer, 'organize_topics_into_modules')

    @pytest.mark.asyncio 
    async def test_module_extraction_service_basic_functionality(self, extraction_service, db_manager):
        """Test basic module extraction service functionality"""
        # Enable feature flags for testing
        FeatureFlags.set_flag(FeatureFlags.APRAG_ENABLED, True)
        FeatureFlags.set_flag(FeatureFlags.MODULE_EXTRACTION_ENABLED, True)
        
        # Test service initialization
        assert extraction_service.db_manager is not None
        
        # Test loading session topics
        topics = await extraction_service._load_session_topics("test_session_123")
        assert len(topics) == 5  # We inserted 5 test topics
        assert all("topic_title" in topic for topic in topics)
        
        # Test loading course info
        course_info = await extraction_service._load_course_info(1)  # Course ID 1
        assert course_info["course_name"] == "10. Sınıf Biyoloji"
        assert course_info["subject_area"] == "Biology"

    def test_logging_system_integration(self):
        """Test centralized logging system"""
        # Test logger creation
        logger = get_module_logger("test_module")
        assert logger is not None
        
        # Test logging operations
        logger.log_operation_start("TEST_OPERATION", test_param="value")
        logger.log_operation_success("TEST_OPERATION", duration_ms=100.5, result="success")
        logger.log_operation_warning("TEST_OPERATION", "Test warning", context="test")
        
        # Test specialized logging
        logger.log_llm_operation("GENERATION", "test-model", tokens=150)
        logger.log_database_operation("INSERT", "test_table", affected_rows=5)
        logger.log_curriculum_operation("TEMPLATE_LOAD", "Biology", grade="10")
        
        # Test validation logging
        logger.log_validation_issue("warning", "structure", "Test validation issue", module_id=1)

    @pytest.mark.asyncio
    async def test_api_integration_mock(self):
        """Test API integration with mock requests"""
        # This would test the actual API endpoints if we had a running server
        # For now, we test the API structure and response formats
        
        from api.modules import ModuleExtractionRequest, ModuleValidationRequest
        
        # Test request models
        extraction_request = ModuleExtractionRequest(
            session_id="test_session_123",
            extraction_strategy="curriculum_aligned",
            options={"max_modules": 10},
            course_context={"subject": "Biology", "grade_level": "10"}
        )
        
        assert extraction_request.session_id == "test_session_123"
        assert extraction_request.extraction_strategy == "curriculum_aligned"
        
        validation_request = ModuleValidationRequest(
            modules=[{"module_title": "Test Module", "topics": []}],
            course_context={"subject": "Biology"},
            validation_options={"apply_auto_fixes": True}
        )
        
        assert len(validation_request.modules) == 1
        assert validation_request.course_context["subject"] == "Biology"

    def test_error_handling_system(self):
        """Test centralized error handling"""
        from utils.logging_config import ErrorHandler, get_module_logger
        
        logger = get_module_logger("test_error_handler")
        error_handler = ErrorHandler(logger)
        
        # Test LLM error handling
        llm_timeout_error = Exception("Request timeout after 30s")
        llm_error_info = error_handler.handle_llm_error(llm_timeout_error, {"model": "test"})
        
        assert llm_error_info["success"] == False
        assert llm_error_info["error_type"] == "timeout"
        assert llm_error_info["fallback_available"] == True
        
        # Test database error handling
        db_error = Exception("FOREIGN KEY constraint failed")
        db_error_info = error_handler.handle_database_error(db_error, "INSERT", {"table": "test"})
        
        assert db_error_info["success"] == False
        assert db_error_info["error_type"] == "foreign_key_constraint"
        
        # Test error response creation
        error_response = error_handler.create_error_response(llm_error_info, 504)
        assert error_response["success"] == False
        assert error_response["status_code"] == 504
        assert "timestamp" in error_response["error"]

    @pytest.mark.asyncio
    async def test_end_to_end_mock_extraction(self, extraction_service):
        """Test end-to-end module extraction with mocked LLM"""
        # Enable feature flags
        FeatureFlags.set_flag(FeatureFlags.APRAG_ENABLED, True)
        FeatureFlags.set_flag(FeatureFlags.MODULE_EXTRACTION_ENABLED, True)
        
        # Mock the LLM organizer to avoid actual API calls
        with patch.object(extraction_service, 'module_organizer') as mock_organizer:
            # Mock the organize method
            mock_organizer.organize_topics_into_modules = AsyncMock(return_value=[
                {
                    "module_title": "Hücre Biyolojisi Temelleri",
                    "module_description": "Hücre yapısı ve temel süreçler",
                    "module_order": 1,
                    "estimated_duration_hours": 25,
                    "difficulty_level": "intermediate",
                    "learning_outcomes": ["Hücre yapısını açıklar", "Organellerin işlevlerini bilir"],
                    "assessment_methods": ["quiz", "assignment"],
                    "topics": [
                        {"topic_id": 1, "topic_order_in_module": 1, "importance_level": "high"},
                        {"topic_id": 2, "topic_order_in_module": 2, "importance_level": "medium"}
                    ]
                },
                {
                    "module_title": "Genetik ve Protein Sentezi",
                    "module_description": "Genetik materyal ve protein üretimi",
                    "module_order": 2,
                    "estimated_duration_hours": 30,
                    "difficulty_level": "advanced",
                    "learning_outcomes": ["DNA/RNA yapısını açıklar", "Protein sentezini anlatır"],
                    "assessment_methods": ["exam", "project"],
                    "topics": [
                        {"topic_id": 2, "topic_order_in_module": 1, "importance_level": "high"},
                        {"topic_id": 3, "topic_order_in_module": 2, "importance_level": "high"}
                    ]
                }
            ])
            
            # Mock the validator to avoid complex validation
            with patch.object(extraction_service, 'module_validator') as mock_validator:
                mock_validator.validate_module_set = AsyncMock(return_value=(
                    True, [], mock_organizer.organize_topics_into_modules.return_value
                ))
                
                # Run extraction
                result = await extraction_service.extract_modules_from_session(
                    session_id="test_session_123",
                    course_id=1,
                    strategy="curriculum_aligned",
                    options={"max_modules": 10}
                )
                
                # Verify result
                assert result["success"] == True
                assert result["modules_created"] == 2
                assert result["extraction_strategy"] == "curriculum_aligned"
                assert "extraction_time_seconds" in result
                assert "module_ids" in result

    def test_complete_system_health_check(self):
        """Test complete system health and component integration"""
        # Test all major components can be imported and initialized
        components_to_test = [
            "ModuleExtractionService",
            "ModuleQualityValidator", 
            "CurriculumTemplateManager",
            "LLMModuleOrganizer",
            "FeatureFlags",
            "get_module_logger"
        ]
        
        for component in components_to_test:
            try:
                if component == "ModuleExtractionService":
                    from services.module_extraction_service import ModuleExtractionService
                    # Test initialization with mock DB
                    service = ModuleExtractionService()
                    assert service is not None
                
                elif component == "ModuleQualityValidator":
                    from validators.module_quality_validator import ModuleQualityValidator
                    validator = ModuleQualityValidator()
                    assert validator is not None
                
                elif component == "CurriculumTemplateManager":
                    from templates.curriculum_templates import CurriculumTemplateManager
                    manager = CurriculumTemplateManager()
                    assert manager is not None
                
                elif component == "LLMModuleOrganizer":
                    from processors.llm_module_organizer import LLMModuleOrganizer
                    organizer = LLMModuleOrganizer()
                    assert organizer is not None
                
                elif component == "FeatureFlags":
                    from config.feature_flags import FeatureFlags
                    # Test basic functionality
                    assert hasattr(FeatureFlags, 'is_module_extraction_enabled')
                
                elif component == "get_module_logger":
                    from utils.logging_config import get_module_logger
                    logger = get_module_logger("test")
                    assert logger is not None
                
            except Exception as e:
                pytest.fail(f"Component {component} failed to initialize: {e}")

    def test_configuration_completeness(self):
        """Test that all required configuration is present"""
        # Test feature flags configuration
        flag_methods = [
            'is_module_extraction_enabled',
            'is_module_quality_validation_enabled', 
            'is_module_curriculum_alignment_enabled'
        ]
        
        for method in flag_methods:
            assert hasattr(FeatureFlags, method), f"Missing feature flag method: {method}"
        
        # Test that all required default values are set
        required_flags = [
            'MODULE_EXTRACTION_ENABLED',
            'MODULE_QUALITY_VALIDATION_ENABLED',
            'MODULE_CURRICULUM_ALIGNMENT_ENABLED'
        ]
        
        for flag in required_flags:
            assert hasattr(FeatureFlags, flag), f"Missing feature flag constant: {flag}"
            assert flag in FeatureFlags._defaults, f"Missing default value for flag: {flag}"


# Run tests if this file is executed directly
if __name__ == "__main__":
    # Simple test runner for basic functionality
    import traceback
    
    print("🧪 Running Module Extraction Integration Tests...")
    
    test_instance = TestModuleExtractionIntegration()
    
    # Test database creation
    try:
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Test schema creation
        conn = sqlite3.connect(temp_path)
        test_instance._create_test_schema(conn)
        conn.close()
        
        print("✅ Database schema creation - PASSED")
        
        # Cleanup
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Database schema creation - FAILED: {e}")
    
    # Test feature flags
    try:
        test_instance.test_feature_flags_integration()
        print("✅ Feature flags integration - PASSED")
    except Exception as e:
        print(f"❌ Feature flags integration - FAILED: {e}")
    
    # Test curriculum templates
    try:
        test_instance.test_curriculum_template_manager()
        print("✅ Curriculum template manager - PASSED")
    except Exception as e:
        print(f"❌ Curriculum template manager - FAILED: {e}")
    
    # Test logging system
    try:
        test_instance.test_logging_system_integration()
        print("✅ Logging system integration - PASSED")
    except Exception as e:
        print(f"❌ Logging system integration - FAILED: {e}")
    
    # Test error handling
    try:
        test_instance.test_error_handling_system()
        print("✅ Error handling system - PASSED")  
    except Exception as e:
        print(f"❌ Error handling system - FAILED: {e}")
    
    # Test system health check
    try:
        test_instance.test_complete_system_health_check()
        print("✅ Complete system health check - PASSED")
    except Exception as e:
        print(f"❌ Complete system health check - FAILED: {e}")
        print(f"Error details: {traceback.format_exc()}")
    
    # Test configuration completeness
    try:
        test_instance.test_configuration_completeness()
        print("✅ Configuration completeness - PASSED")
    except Exception as e:
        print(f"❌ Configuration completeness - FAILED: {e}")
    
    print("\n🎯 Integration test suite completed!")
    print("📋 To run full async tests, use: pytest test_module_extraction_integration.py")