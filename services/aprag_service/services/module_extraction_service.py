"""
Core Module Extraction Service
Handles curriculum-aware module organization using LLM integration
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from database.database import DatabaseManager
from config.feature_flags import FeatureFlags

# Import centralized logging and error handling
from utils.logging_config import get_module_logger, log_execution_time, with_error_handling

# Initialize module-specific logger
module_logger = get_module_logger(__name__)
logger = logging.getLogger(__name__)


class ModuleExtractionService:
    """
    Main service for extracting and organizing educational modules
    Integrates with existing APRAG service architecture
    """

    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the module extraction service
        
        Args:
            db_manager: Database manager instance (optional)
        """
        self.db_manager = db_manager or self._get_default_db_manager()
        self.curriculum_analyzer = CurriculumAnalyzer(self.db_manager)
        self.module_organizer = ModuleOrganizer()
        self.module_validator = ModuleValidator()
        
        logger.info("Module Extraction Service initialized")

    def _get_default_db_manager(self) -> DatabaseManager:
        """Get default database manager"""
        import os
        db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
        return DatabaseManager(db_path)

    async def extract_modules_from_session(
        self,
        session_id: str,
        course_id: int,
        strategy: str = "curriculum_aligned",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main method to extract modules from a session's topics
        
        Args:
            session_id: Document processing session ID
            course_id: Target course ID
            strategy: Extraction strategy ('curriculum_aligned', 'semantic_clustering', 'hybrid')
            options: Additional extraction options
            
        Returns:
            Dictionary with extraction results
        """
        # Check if module extraction is enabled
        if not FeatureFlags.is_aprag_enabled(session_id):
            raise ValueError("APRAG module is disabled")
        
        start_time = datetime.now()
        logger.info(f"Starting module extraction for session {session_id}, course {course_id}")
        
        try:
            # 1. Validate inputs
            await self._validate_extraction_inputs(session_id, course_id)
            
            # 2. Load session topics
            topics = await self._load_session_topics(session_id)
            if not topics:
                raise ValueError(f"No topics found for session {session_id}")
            
            # 3. Load course and curriculum information
            course_info = await self._load_course_info(course_id)
            curriculum_template = await self.curriculum_analyzer.get_curriculum_template(course_info)
            
            # 4. Organize topics into modules using selected strategy
            if strategy == "curriculum_aligned":
                module_candidates = await self._extract_curriculum_aligned_modules(
                    topics, course_info, curriculum_template, options or {}
                )
            elif strategy == "semantic_clustering":
                module_candidates = await self._extract_semantic_modules(
                    topics, course_info, options or {}
                )
            elif strategy == "hybrid":
                module_candidates = await self._extract_hybrid_modules(
                    topics, course_info, curriculum_template, options or {}
                )
            else:
                raise ValueError(f"Unknown extraction strategy: {strategy}")
            
            # 5. Validate module structure and quality
            is_valid, validation_issues, corrected_modules = await self.module_validator.validate_module_set(
                module_candidates, course_info, options or {}
            )
            
            # After validation, distribute unused topics to modules that have no topics
            all_topic_ids = set(t['topic_id'] for t in topics)
            used_topic_ids = set()
            for module in corrected_modules:
                module_topics = module.get('topics', [])
                for topic_ref in module_topics:
                    topic_id = topic_ref.get('topic_id')
                    if topic_id:
                        used_topic_ids.add(topic_id)
            
            unused_topic_ids = all_topic_ids - used_topic_ids
            if unused_topic_ids:
                logger.info(f"Found {len(unused_topic_ids)} unused topics. Distributing to modules with no topics...")
                unused_topics = [t for t in topics if t['topic_id'] in unused_topic_ids]
                
                # Find modules with no topics
                empty_modules = [m for m in corrected_modules if not m.get('topics') or len(m.get('topics', [])) == 0]
                
                if empty_modules and unused_topics:
                    # Distribute unused topics to empty modules
                    topics_per_module = max(1, len(unused_topics) // len(empty_modules))
                    topic_index = 0
                    
                    for module in empty_modules:
                        if topic_index >= len(unused_topics):
                            break
                        
                        # Add topics to this module
                        module_topics = module.get('topics', [])
                        for i in range(topics_per_module):
                            if topic_index >= len(unused_topics):
                                break
                            
                            topic = unused_topics[topic_index]
                            module_topics.append({
                                'topic_id': topic['topic_id'],
                                'topic_order_in_module': len(module_topics) + 1,
                                'importance_level': 'medium'
                            })
                            topic_index += 1
                        
                        module['topics'] = module_topics
                    
                    # Distribute remaining topics
                    remaining_topics = unused_topics[topic_index:]
                    if remaining_topics:
                        # Add to modules that have fewest topics
                        modules_by_topic_count = sorted(corrected_modules, key=lambda m: len(m.get('topics', [])))
                        for i, topic in enumerate(remaining_topics):
                            target_module = modules_by_topic_count[i % len(modules_by_topic_count)]
                            module_topics = target_module.get('topics', [])
                            module_topics.append({
                                'topic_id': topic['topic_id'],
                                'topic_order_in_module': len(module_topics) + 1,
                                'importance_level': 'medium'
                            })
                            target_module['topics'] = module_topics
                    
                    logger.info(f"Distributed {len(unused_topics)} unused topics to {len(empty_modules)} empty modules")
            
            if not is_valid:
                # validation_issues is a list of dicts, not objects
                error_issues = [issue for issue in validation_issues if issue.get('severity') == "error"]
                if error_issues:
                    error_messages = [issue.get('message', str(issue)) for issue in error_issues]
                    raise ValueError(f"Module validation failed: {error_messages}")
            
            # 6. Create module records in database
            created_modules = await self._create_module_records(
                corrected_modules, course_id, session_id
            )
            
            # 7. Establish topic-module relationships
            relationship_count = await self._create_topic_module_relationships(
                created_modules, topics
            )
            
            # 8. Calculate curriculum alignment score
            alignment_score = await self._calculate_curriculum_alignment(
                created_modules, course_info
            )
            
            # 9. Initialize progress tracking
            await self._initialize_module_progress(created_modules, session_id)
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "modules_created": len(created_modules),
                "topics_organized": len(topics),
                "relationships_created": relationship_count,
                "alignment_score": alignment_score,
                "extraction_strategy": strategy,
                "extraction_time_seconds": extraction_time,
                "validation_issues": len(validation_issues),
                "module_ids": [module["module_id"] for module in created_modules],
                "course_id": course_id,
                "session_id": session_id
            }
            
            logger.info(f"Module extraction completed successfully: {result}")
            return result
            
        except Exception as e:
            extraction_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Module extraction failed after {extraction_time}s: {e}")
            logger.error(f"Session: {session_id}, Course: {course_id}, Strategy: {strategy}")
            raise

    async def _validate_extraction_inputs(self, session_id: str, course_id: int):
        """Validate extraction inputs"""
        # Check if session exists and has topics
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM course_topics WHERE session_id = ? AND is_active = TRUE",
                (session_id,)
            )
            topic_count = dict(cursor.fetchone())["count"]
            
            if topic_count == 0:
                raise ValueError(f"No topics found for session {session_id}")
            
            # Check if course exists
            cursor = conn.execute(
                "SELECT course_id FROM courses WHERE course_id = ? AND is_active = TRUE",
                (course_id,)
            )
            if not cursor.fetchone():
                raise ValueError(f"Course {course_id} not found or inactive")
            
            logger.info(f"Validation passed: {topic_count} topics found for session {session_id}")

    async def _load_session_topics(self, session_id: str) -> List[Dict[str, Any]]:
        """Load topics for the session with full details"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    topic_id, topic_title, parent_topic_id, topic_order,
                    description, keywords, estimated_difficulty, prerequisites,
                    related_chunk_ids, extraction_confidence
                FROM course_topics
                WHERE session_id = ? AND is_active = TRUE
                ORDER BY topic_order, topic_id
            """, (session_id,))
            
            topics = []
            for row in cursor.fetchall():
                topic = dict(row)
                # Parse JSON fields
                topic["keywords"] = json.loads(topic["keywords"]) if topic["keywords"] else []
                topic["prerequisites"] = json.loads(topic["prerequisites"]) if topic["prerequisites"] else []
                topic["related_chunk_ids"] = json.loads(topic["related_chunk_ids"]) if topic["related_chunk_ids"] else []
                topics.append(topic)
            
            logger.info(f"Loaded {len(topics)} topics for session {session_id}")
            return topics

    async def _load_course_info(self, course_id: int) -> Dict[str, Any]:
        """Load course information with curriculum standards"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    course_id, course_code, course_name, course_description,
                    curriculum_standard, subject_area, grade_level,
                    total_hours, difficulty_level, language
                FROM courses
                WHERE course_id = ? AND is_active = TRUE
            """, (course_id,))
            
            course = cursor.fetchone()
            if not course:
                raise ValueError(f"Course {course_id} not found")
            
            course_info = dict(course)
            
            # Load related curriculum standards
            cursor = conn.execute("""
                SELECT standard_id, standard_code, standard_title, expected_outcomes
                FROM curriculum_standards
                WHERE curriculum_type = ? AND subject_area = ? AND grade_level = ? AND is_active = TRUE
                ORDER BY standard_code
            """, (
                course_info["curriculum_standard"],
                course_info["subject_area"],
                course_info["grade_level"]
            ))
            
            standards = []
            for row in cursor.fetchall():
                standard = dict(row)
                standard["expected_outcomes"] = json.loads(standard["expected_outcomes"]) if standard["expected_outcomes"] else []
                standards.append(standard)
            
            course_info["curriculum_standards"] = standards
            
            logger.info(f"Loaded course info: {course_info['course_name']} with {len(standards)} standards")
            return course_info

    async def _extract_curriculum_aligned_modules(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        curriculum_template: Dict[str, Any],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract modules aligned with curriculum standards"""
        logger.info("Starting curriculum-aligned module extraction")
        
        # Use LLM-based module organizer with curriculum template
        module_candidates = await self.module_organizer.organize_topics_into_modules(
            topics=topics,
            course_info=course_info,
            template=curriculum_template,
            strategy="curriculum_aligned",
            options=options
        )
        
        logger.info(f"Generated {len(module_candidates)} curriculum-aligned module candidates")
        return module_candidates

    async def _extract_semantic_modules(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract modules using semantic clustering"""
        logger.info("Starting semantic clustering module extraction")
        
        # Use semantic clustering approach
        module_candidates = await self.module_organizer.organize_topics_into_modules(
            topics=topics,
            course_info=course_info,
            template=None,
            strategy="semantic_clustering",
            options=options
        )
        
        logger.info(f"Generated {len(module_candidates)} semantic module candidates")
        return module_candidates

    async def _extract_hybrid_modules(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        curriculum_template: Dict[str, Any],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract modules using hybrid approach (curriculum + semantic)"""
        logger.info("Starting hybrid module extraction")
        
        # Combine curriculum alignment with semantic analysis
        module_candidates = await self.module_organizer.organize_topics_into_modules(
            topics=topics,
            course_info=course_info,
            template=curriculum_template,
            strategy="hybrid",
            options=options
        )
        
        logger.info(f"Generated {len(module_candidates)} hybrid module candidates")
        return module_candidates

    async def _create_module_records(
        self,
        module_candidates: List[Dict[str, Any]],
        course_id: int,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Create module records in the database"""
        logger.info(f"Creating {len(module_candidates)} module records")
        
        created_modules = []
        
        with self.db_manager.get_connection() as conn:
            for i, module_data in enumerate(module_candidates):
                # Generate module code
                module_code = f"MOD_{course_id}_{i+1:02d}"
                
                # Insert module record
                cursor = conn.execute("""
                    INSERT INTO course_modules (
                        course_id, session_id, module_code, module_title, module_description,
                        module_order, estimated_duration_hours, difficulty_level,
                        prerequisites, learning_outcomes, assessment_methods,
                        extraction_method, extraction_model, extraction_confidence,
                        is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    course_id,
                    session_id,
                    module_code,
                    module_data.get("module_title", f"Module {i+1}"),
                    module_data.get("module_description"),
                    module_data.get("module_order", i+1),
                    module_data.get("estimated_duration_hours", 20),
                    module_data.get("difficulty_level", "intermediate"),
                    json.dumps(module_data.get("prerequisites", [])),
                    json.dumps(module_data.get("learning_outcomes", [])),
                    json.dumps(module_data.get("assessment_methods", ["quiz"])),
                    "llm_analysis",
                    module_data.get("extraction_model", "unknown"),
                    module_data.get("extraction_confidence", 0.8),
                    True
                ))
                
                module_id = cursor.lastrowid
                
                created_module = {
                    "module_id": module_id,
                    "module_code": module_code,
                    "module_title": module_data.get("module_title"),
                    "module_order": module_data.get("module_order", i+1),
                    "topics": module_data.get("topics", [])
                }
                
                created_modules.append(created_module)
            
            conn.commit()
            
        logger.info(f"Successfully created {len(created_modules)} module records")
        return created_modules

    async def _create_topic_module_relationships(
        self,
        created_modules: List[Dict[str, Any]],
        original_topics: List[Dict[str, Any]]
    ) -> int:
        """Create topic-module relationships"""
        logger.info("Creating topic-module relationships")
        
        relationship_count = 0
        topic_id_map = {topic["topic_id"]: topic for topic in original_topics}
        
        with self.db_manager.get_connection() as conn:
            for module in created_modules:
                module_id = module["module_id"]
                module_topics = module.get("topics", [])
                
                for j, topic_ref in enumerate(module_topics):
                    topic_id = topic_ref.get("topic_id")
                    if topic_id and topic_id in topic_id_map:
                        # Insert relationship
                        conn.execute("""
                            INSERT INTO module_topic_relationships (
                                module_id, topic_id, relationship_type, topic_order_in_module,
                                importance_level, content_coverage_percentage,
                                extraction_confidence, extraction_method
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            module_id,
                            topic_id,
                            topic_ref.get("relationship_type", "contains"),
                            j + 1,
                            topic_ref.get("importance_level", "medium"),
                            topic_ref.get("content_coverage_percentage", 100.0),
                            topic_ref.get("extraction_confidence", 0.8),
                            "llm_analysis"
                        ))
                        relationship_count += 1
            
            conn.commit()
        
        logger.info(f"Created {relationship_count} topic-module relationships")
        return relationship_count

    async def _calculate_curriculum_alignment(
        self,
        created_modules: List[Dict[str, Any]],
        course_info: Dict[str, Any]
    ) -> float:
        """Calculate curriculum alignment score"""
        if not course_info.get("curriculum_standards"):
            return 0.0
        
        total_standards = len(course_info["curriculum_standards"])
        covered_standards = 0
        
        # Simple alignment calculation
        # In a more sophisticated version, this would analyze the actual content alignment
        for module in created_modules:
            # Check if module covers any curriculum standards
            if module.get("learning_outcomes"):
                covered_standards += 1
        
        alignment_score = min(covered_standards / total_standards, 1.0) if total_standards > 0 else 0.0
        
        logger.info(f"Calculated curriculum alignment score: {alignment_score:.2f}")
        return alignment_score

    async def _initialize_module_progress(
        self,
        created_modules: List[Dict[str, Any]],
        session_id: str
    ):
        """Initialize module progress for existing users in the session"""
        logger.info("Initializing module progress tracking")
        
        # Get users who already have progress in this session
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT user_id 
                FROM topic_progress 
                WHERE session_id = ?
            """, (session_id,))
            
            user_ids = [row[0] for row in cursor.fetchall()]
            
            if not user_ids:
                logger.info("No existing users found for progress initialization")
                return
            
            # Initialize module progress for each user
            for user_id in user_ids:
                for module in created_modules:
                    # Count topics in this module
                    cursor = conn.execute("""
                        SELECT COUNT(*) as topic_count
                        FROM module_topic_relationships
                        WHERE module_id = ?
                    """, (module["module_id"],))
                    
                    topic_count = dict(cursor.fetchone())["topic_count"]
                    
                    # Insert module progress record
                    conn.execute("""
                        INSERT OR IGNORE INTO module_progress (
                            user_id, session_id, module_id,
                            completion_status, topics_total, is_prerequisite_met
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        user_id,
                        session_id,
                        module["module_id"],
                        "not_started",
                        topic_count,
                        True  # Assume prerequisites are met for now
                    ))
            
            conn.commit()
            
        logger.info(f"Initialized progress tracking for {len(user_ids)} users across {len(created_modules)} modules")

    # Public utility methods
    async def get_course_modules(self, course_id: int, session_id: str = None) -> List[Dict[str, Any]]:
        """Get all modules for a course"""
        with self.db_manager.get_connection() as conn:
            query = """
                SELECT 
                    module_id, module_code, module_title, module_description,
                    module_order, estimated_duration_hours, difficulty_level,
                    topic_count, extraction_confidence, is_active, created_at
                FROM course_modules
                WHERE course_id = ? AND is_active = TRUE
            """
            params = [course_id]
            
            if session_id:
                query += " AND (session_id = ? OR session_id IS NULL)"
                params.append(session_id)
            
            query += " ORDER BY module_order, module_id"
            
            cursor = conn.execute(query, params)
            modules = [dict(row) for row in cursor.fetchall()]
            
        return modules

    async def get_module_topics(self, module_id: int) -> List[Dict[str, Any]]:
        """Get all topics for a module"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    t.topic_id, t.topic_title, t.estimated_difficulty,
                    r.topic_order_in_module, r.importance_level,
                    r.content_coverage_percentage, r.extraction_confidence
                FROM course_topics t
                JOIN module_topic_relationships r ON t.topic_id = r.topic_id
                WHERE r.module_id = ? AND t.is_active = TRUE
                ORDER BY r.topic_order_in_module, t.topic_id
            """, (module_id,))
            
            topics = [dict(row) for row in cursor.fetchall()]
        
        return topics


class CurriculumAnalyzer:
    """Analyzes curriculum standards and provides templates"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def get_curriculum_template(self, course_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get curriculum template for course"""
        curriculum_type = course_info.get("curriculum_standard", "")
        subject_area = course_info.get("subject_area", "")
        grade_level = course_info.get("grade_level", "")
        
        # Try to find existing template
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT template_id, template_name, prompt_template, 
                       module_structure_template, learning_outcomes_template
                FROM module_templates
                WHERE curriculum_type = ? AND subject_area = ? 
                AND grade_levels LIKE ? AND is_active = TRUE
                ORDER BY is_default DESC, template_id
                LIMIT 1
            """, (curriculum_type, subject_area, f'%"{grade_level}"%'))
            
            template = cursor.fetchone()
            
        if template:
            template_dict = dict(template)
            # Parse JSON fields
            template_dict["module_structure_template"] = json.loads(
                template_dict["module_structure_template"]
            ) if template_dict["module_structure_template"] else {}
            template_dict["learning_outcomes_template"] = json.loads(
                template_dict["learning_outcomes_template"]
            ) if template_dict["learning_outcomes_template"] else []
            
            logger.info(f"Found curriculum template: {template_dict['template_name']}")
            return template_dict
        
        # Return default template
        default_template = {
            "template_name": "default",
            "prompt_template": self._get_default_prompt_template(course_info),
            "module_structure_template": self._get_default_structure_template(),
            "learning_outcomes_template": []
        }
        
        logger.info("Using default curriculum template")
        return default_template
    
    def _get_default_prompt_template(self, course_info: Dict[str, Any]) -> str:
        """Get default prompt template for course"""
        # Don't format here - let llm_module_organizer format it with all placeholders
        # This avoids double-formatting issues with JSON examples
        return """
Sen bir eğitim uzmanısın. {grade}. sınıf {subject} konularını resmi müfredata uygun modüllere organize et.

ÖNEMLİ: Aşağıdaki konuları modüllere organize etmelisin. Sadece konu listesi döndürme!
Her modül bir başlık ve açıklama içermeli, ve içinde ilgili konuları gruplamalı.

MEVCUT KONULAR:
{{topics_list}}

MODÜL ORGANİZASYON PRENSİPLERİ:
1. Konuları mantıklı öğrenme sırası takip etmeli
2. Her modül 3-15 konu içermeli
3. Zorluk derecesi kademeli artmalı
4. Önkoşul ilişkileri belirtilmeli

ÇIKTI FORMATI (JSON):
JSON çıktısı şu formatta olmalı:
- Ana obje: "modules" anahtarı ile başlamalı (array)
- Her modül: "module_title", "module_description", "module_order", "estimated_duration_hours", "learning_outcomes", "topics" içermeli
- Her topic: "topic_id", "topic_order_in_module", "importance_level" içermeli

ÖNEMLİ KURALLAR:
- MUTLAKA modül organizasyonu yap: Konuları mantıklı gruplara ayır ve her gruba bir modül başlığı ver
- Sadece konu listesi döndürme! Konuları modüllere organize etmelisin
- JSON çıktısında İNGİLİZCE anahtar kelimeler kullan:
  * "modules" (Türkçe "moduller" değil)
  * "module_title" (Türkçe "baslik" değil)
  * "module_description" (Türkçe "aciklama" değil)
  * "topics" (Türkçe "konular" değil)
  * "topic_id" (Türkçe "konu_id" değil)
- Her modül en az 3-12 konu içermeli

Sadece geçerli JSON çıktısı ver, açıklama yapma.
"""
    
    def _get_default_structure_template(self) -> Dict[str, Any]:
        """Get default module structure template"""
        return {
            "max_modules": 10,
            "min_topics_per_module": 3,
            "max_topics_per_module": 15,
            "difficulty_progression": True,
            "prerequisite_analysis": True
        }


class ModuleOrganizer:
    """Organizes topics into modules using various strategies"""
    
    def __init__(self):
        """Initialize module organizer with LLM organizer"""
        from processors.llm_module_organizer import LLMModuleOrganizer
        
        self.llm_organizer = LLMModuleOrganizer()
        logger.info("LLM Module Organizer initialized")
    
    async def organize_topics_into_modules(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        template: Optional[Dict[str, Any]],
        strategy: str,
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Organize topics into modules based on strategy using LLM
        
        Raises exception if LLM organization fails - no fallback
        """
        logger.info(f"Organizing {len(topics)} topics using {strategy} strategy with LLM")
        
        if not self.llm_organizer:
            raise ValueError("LLM Module Organizer is not initialized. Cannot organize modules without LLM.")
        
        try:
            logger.info("Starting LLM-based organization")
            modules = await self.llm_organizer.organize_topics_into_modules(
                topics=topics,
                course_info=course_info,
                template=template,
                strategy=strategy,
                options=options
            )
            
            if not modules or len(modules) == 0:
                raise ValueError(
                    "LLM organizasyonu başarısız oldu: Modül oluşturulamadı. "
                    "LLM servisi boş sonuç döndü. Lütfen LLM servisini kontrol edin veya daha sonra tekrar deneyin."
                )
            
            # Validate that modules have proper titles
            for i, module in enumerate(modules):
                if not module.get("module_title") or module.get("module_title", "").strip() == "":
                    raise ValueError(
                        f"LLM organizasyonu başarısız oldu: Modül {i+1} için başlık oluşturulamadı. "
                        "LLM servisi eksik veri döndü. Lütfen tekrar deneyin."
                    )
                # Check for generic titles
                title = module.get("module_title", "").strip()
                if title.lower().startswith("modül ") and title.replace("modül ", "").strip().isdigit():
                    raise ValueError(
                        f"LLM organizasyonu başarısız oldu: Modül {i+1} için generic başlık oluşturuldu ('{title}'). "
                        "LLM servisi anlamlı başlık üretemedi. Lütfen LLM servisini kontrol edin veya tekrar deneyin."
                    )
            
            logger.info(f"LLM organization successful: {len(modules)} modules created with proper titles")
            return modules
            
        except ValueError as e:
            # Re-raise ValueError - it already contains LLM response if available
            # Just add some context if it doesn't have LLM response info
            error_str = str(e)
            if "LLM Yanıtı" not in error_str:
                error_msg = (
                    f"{error_str}\n\n"
                    "Lütfen şunları kontrol edin:\n"
                    "1. LLM servisi çalışıyor mu? (Model Inference Service)\n"
                    "2. LLM servisine bağlantı var mı?\n"
                    "3. Yeterli konu var mı? (En az 3-5 konu önerilir)\n"
                    "4. Daha sonra tekrar deneyin"
                )
                raise ValueError(error_msg) from e
            else:
                # Already has LLM response, re-raise as-is
                raise
        except Exception as e:
            error_msg = (
                f"LLM organizasyonu başarısız oldu: {str(e)}\n\n"
                "Lütfen şunları kontrol edin:\n"
                "1. LLM servisi çalışıyor mu? (Model Inference Service)\n"
                "2. LLM servisine bağlantı var mı?\n"
                "3. Yeterli konu var mı? (En az 3-5 konu önerilir)\n"
                "4. Daha sonra tekrar deneyin"
            )
            logger.error(f"LLM organization failed: {e}")
            raise ValueError(error_msg) from e
    
    async def _organize_curriculum_aligned(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        template: Dict[str, Any],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Organize topics based on curriculum alignment"""
        # This would integrate with the LLM-based module organizer
        # For now, implementing a simple version
        
        max_topics_per_module = options.get("max_topics_per_module", 8)
        modules = []
        
        # Group topics into modules of max_topics_per_module size
        for i in range(0, len(topics), max_topics_per_module):
            module_topics = topics[i:i + max_topics_per_module]
            
            # Generate meaningful title from topic titles
            topic_titles = [t.get("topic_title", "") for t in module_topics if t.get("topic_title")]
            if topic_titles:
                # Use first topic title as base, or combine first few
                if len(topic_titles) == 1:
                    module_title = topic_titles[0]
                elif len(topic_titles) <= 3:
                    # Combine first 2-3 topic titles
                    module_title = " - ".join(topic_titles[:2])
                    if len(module_title) > 80:
                        module_title = topic_titles[0][:77] + "..."
                else:
                    # Use first topic + count
                    module_title = f"{topic_titles[0]} ve İlgili Konular"
                    if len(module_title) > 80:
                        module_title = topic_titles[0][:77] + "..."
            else:
                # Fallback if no topic titles
                subject = course_info.get("subject_area", "Ders")
                module_title = f"{subject} Modülü {len(modules) + 1}"
            
            # Generate description from topics
            if topic_titles:
                desc_topics = ", ".join(topic_titles[:3])
                if len(topic_titles) > 3:
                    desc_topics += f" ve {len(topic_titles) - 3} konu daha"
                module_description = f"{desc_topics} konularını içeren eğitim modülü"
            else:
                module_description = f"Eğitim modülü - {len(module_topics)} konu içerir"
            
            module = {
                "module_title": module_title,
                "module_description": module_description,
                "module_order": len(modules) + 1,
                "estimated_duration_hours": len(module_topics) * 2,
                "difficulty_level": "intermediate",
                "learning_outcomes": [f"{module_title} kapsamındaki temel kavramları öğrenme"],
                "assessment_methods": ["quiz", "assignment"],
                "topics": [
                    {
                        "topic_id": topic["topic_id"],
                        "topic_order_in_module": j + 1,
                        "importance_level": "medium",
                        "relationship_type": "contains"
                    }
                    for j, topic in enumerate(module_topics)
                ]
            }
            modules.append(module)
        
        logger.info(f"Created {len(modules)} curriculum-aligned modules")
        return modules
    
    async def _organize_semantic_clustering(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Organize topics using semantic clustering"""
        # Placeholder for semantic clustering implementation
        return await self._organize_simple(topics, course_info, options)
    
    async def _organize_hybrid(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        template: Optional[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Organize topics using hybrid approach"""
        # Placeholder for hybrid implementation
        return await self._organize_curriculum_aligned(topics, course_info, template or {}, options)
    
    async def _organize_simple(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Simple topic organization as fallback"""
        max_topics_per_module = options.get("max_topics_per_module", 6)
        modules = []
        
        for i in range(0, len(topics), max_topics_per_module):
            module_topics = topics[i:i + max_topics_per_module]
            
            # Generate meaningful title from topic titles
            topic_titles = [t.get("topic_title", "") for t in module_topics if t.get("topic_title")]
            if topic_titles:
                if len(topic_titles) == 1:
                    module_title = topic_titles[0]
                else:
                    module_title = f"{topic_titles[0]} ve İlgili Konular"
                    if len(module_title) > 80:
                        module_title = topic_titles[0][:77] + "..."
            else:
                subject = course_info.get("subject_area", "Ders")
                module_title = f"{subject} Modülü {len(modules) + 1}"
            
            # Generate description
            if topic_titles:
                desc = ", ".join(topic_titles[:2])
                if len(topic_titles) > 2:
                    desc += f" ve {len(topic_titles) - 2} konu daha"
                module_description = f"{desc} konularını içeren modül"
            else:
                module_description = f"Basit modül - {len(module_topics)} konu"
            
            module = {
                "module_title": module_title,
                "module_description": module_description,
                "module_order": len(modules) + 1,
                "estimated_duration_hours": len(module_topics) * 3,
                "difficulty_level": "intermediate",
                "learning_outcomes": [f"{module_title} kapsamındaki temel kavramları öğrenme"],
                "assessment_methods": ["quiz"],
                "topics": [
                    {
                        "topic_id": topic["topic_id"],
                        "topic_order_in_module": j + 1,
                        "importance_level": "medium",
                        "relationship_type": "contains"
                    }
                    for j, topic in enumerate(module_topics)
                ]
            }
            modules.append(module)
        
        return modules


class ModuleValidator:
    """Validates module structure and quality"""
    
    async def validate_module_set(
        self,
        modules: List[Dict[str, Any]],
        course_context: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate a set of modules
        
        Returns:
            Tuple of (is_valid, validation_issues, corrected_modules)
        """
        logger.info(f"Validating {len(modules)} modules")
        
        validation_issues = []
        corrected_modules = []
        
        for i, module in enumerate(modules):
            is_valid, issues, corrected_module = await self._validate_individual_module(
                module, course_context, i + 1
            )
            
            validation_issues.extend(issues)
            corrected_modules.append(corrected_module)
        
        # Check for critical issues
        error_issues = [issue for issue in validation_issues if issue.get("severity") == "error"]
        is_overall_valid = len(error_issues) == 0
        
        logger.info(f"Validation completed: {len(validation_issues)} issues found, valid: {is_overall_valid}")
        return is_overall_valid, validation_issues, corrected_modules
    
    async def _validate_individual_module(
        self,
        module: Dict[str, Any],
        course_context: Dict[str, Any],
        module_number: int
    ) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
        """Validate individual module"""
        issues = []
        corrected_module = module.copy()
        
        # Check required fields
        required_fields = ["module_title"]
        for field in required_fields:
            if field not in module or not module[field]:
                issues.append({
                    "severity": "error",
                    "message": f"Module {module_number} missing required field: {field}",
                    "field": field
                })
                
                # Auto-correct
                if field == "module_title":
                    corrected_module["module_title"] = f"Modül {module_number}"
        
        # Topics is not strictly required - it can be empty initially and filled later
        if "topics" not in module or not module.get("topics"):
            issues.append({
                "severity": "warning",  # Changed from "error" to "warning"
                "message": f"Module {module_number} has no topics (will be filled from unused topics)",
                "field": "topics"
            })
            corrected_module["topics"] = []  # Initialize empty list
        
        # Check topic count
        topic_count = len(module.get("topics", []))
        if topic_count < 2:
            issues.append({
                "severity": "warning",
                "message": f"Module {module_number} has too few topics ({topic_count})",
                "field": "topics"
            })
        elif topic_count > 20:
            issues.append({
                "severity": "warning",
                "message": f"Module {module_number} has too many topics ({topic_count})",
                "field": "topics"
            })
        
        # Ensure required defaults
        defaults = {
            "module_description": f"Eğitim modülü {module_number}",
            "module_order": module_number,
            "estimated_duration_hours": max(topic_count * 2, 10),
            "difficulty_level": "intermediate",
            "learning_outcomes": ["Temel öğrenme hedefleri"],
            "assessment_methods": ["quiz"]
        }
        
        for key, default_value in defaults.items():
            if key not in corrected_module or not corrected_module[key]:
                corrected_module[key] = default_value
        
        has_errors = any(issue.get("severity") == "error" for issue in issues)
        return not has_errors, issues, corrected_module