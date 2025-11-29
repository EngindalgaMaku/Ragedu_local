"""
Module-Based Learning Path Management API
Handles curriculum-aware module extraction, organization, and progress tracking
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import logging
import json
from datetime import datetime
import requests
import os
import asyncio
import threading
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()

# Import database manager
from database.database import DatabaseManager

# Import services
try:
    import sys
    import os
    # Add parent directory to path to import from config
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.insert(0, parent_dir)
    from config.feature_flags import FeatureFlags
    
    # Import module extraction components
    from services.module_extraction_service import ModuleExtractionService
    from validators.module_quality_validator import ModuleQualityValidator, ValidationSeverity
    from templates.curriculum_templates import CurriculumTemplateManager
    from processors.llm_module_organizer import LLMModuleOrganizer
    
except ImportError as e:
    logger.warning(f"Could not import some components: {e}")
    # Fallback: Define minimal version if parent config not available
    class FeatureFlags:
        @staticmethod
        def is_aprag_enabled(session_id=None):
            """Fallback implementation when feature flags config is not available"""
            return os.getenv("APRAG_ENABLED", "true").lower() == "true"
        
        @staticmethod
        def is_module_extraction_enabled(session_id=None):
            """Check if module extraction is enabled"""
            return os.getenv("MODULE_EXTRACTION_ENABLED", "true").lower() == "true"

# Environment variables
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", os.getenv("MODEL_INFERENCE_URL", "http://model-inference-service:8002"))
CHROMA_SERVICE_URL = os.getenv("CHROMA_SERVICE_URL", os.getenv("CHROMADB_URL", "http://chromadb-service:8004"))
DOCUMENT_PROCESSING_URL = os.getenv("DOCUMENT_PROCESSING_URL", "http://document-processing-service:8002")


# ============================================================================
# Request/Response Models
# ============================================================================

class ModuleExtractionRequest(BaseModel):
    """Request model for module extraction"""
    session_id: str
    extraction_strategy: str = "curriculum_aligned"  # curriculum_aligned, semantic_clustering, hybrid
    curriculum_prompt: Optional[str] = None  # User-provided curriculum instruction/prompt
    options: Optional[Dict[str, Any]] = {
        "include_module_relationships": True,
        "min_confidence": 0.7,
        "max_modules": 15,
        "validate_quality": True,
        "apply_auto_fixes": True
    }
    course_context: Optional[Dict[str, Any]] = None  # Course metadata (subject, grade, curriculum)


class ModuleUpdateRequest(BaseModel):
    """Request model for updating a module"""
    module_title: Optional[str] = None
    module_description: Optional[str] = None
    module_order: Optional[int] = None
    estimated_duration_hours: Optional[float] = None
    difficulty_level: Optional[str] = None
    learning_outcomes: Optional[List[str]] = None
    assessment_methods: Optional[List[str]] = None
    prerequisites: Optional[List[int]] = None
    is_active: Optional[bool] = None


class ModuleProgressUpdateRequest(BaseModel):
    """Request model for updating module progress"""
    completion_percentage: Optional[float] = None
    status: Optional[str] = None  # not_started, in_progress, completed, mastered
    time_spent_minutes: Optional[float] = None
    current_topic_id: Optional[int] = None


class ModuleValidationRequest(BaseModel):
    """Request model for module validation"""
    modules: List[Dict[str, Any]]
    course_context: Dict[str, Any]
    validation_options: Optional[Dict[str, Any]] = {
        "apply_auto_fixes": True,
        "strict_curriculum_validation": True
    }


class CreateModuleFromRAGRequest(BaseModel):
    """Request model for creating a module from RAG search"""
    session_id: str
    module_title: str  # User-provided module name
    module_description: Optional[str] = None
    top_k: int = 20  # Number of chunks to retrieve via RAG
    similarity_threshold: float = 0.6  # Minimum similarity score for chunks
    use_hybrid_search: bool = True  # Use hybrid search (semantic + BM25)


class ModuleResponse(BaseModel):
    """Response model for a module"""
    module_id: int
    session_id: str
    module_title: str
    module_description: Optional[str]
    module_order: int
    estimated_duration_hours: float
    difficulty_level: Optional[str]
    learning_outcomes: Optional[List[str]]
    assessment_methods: Optional[List[str]]
    prerequisites: Optional[List[int]]
    extraction_confidence: Optional[float]
    is_active: bool
    created_at: str
    updated_at: Optional[str]


class ModuleProgressResponse(BaseModel):
    """Response model for module progress"""
    user_id: str
    module_id: int
    completion_percentage: float
    status: str
    time_spent_minutes: float
    current_topic_id: Optional[int]
    last_accessed_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class ValidationIssueResponse(BaseModel):
    """Response model for validation issues"""
    severity: str
    category: str
    message: str
    suggested_fix: str
    affected_items: List[str]
    auto_fixable: bool
    module_id: Optional[int]


# ============================================================================
# Helper Functions
# ============================================================================

def get_db() -> DatabaseManager:
    """Get database manager dependency"""
    db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
    return DatabaseManager(db_path)


def get_extraction_service() -> ModuleExtractionService:
    """Get module extraction service instance"""
    try:
        db_manager = get_db()
        return ModuleExtractionService(db_manager)
    except Exception as e:
        logger.error(f"Failed to initialize ModuleExtractionService: {e}")
        raise HTTPException(status_code=500, detail="Service initialization failed")


def get_quality_validator() -> ModuleQualityValidator:
    """Get module quality validator instance"""
    try:
        return ModuleQualityValidator()
    except Exception as e:
        logger.error(f"Failed to initialize ModuleQualityValidator: {e}")
        raise HTTPException(status_code=500, detail="Validator initialization failed")


# Global dict to track extraction jobs
module_extraction_jobs = {}


def run_module_extraction_in_background(
    job_id: str,
    session_id: str,
    extraction_strategy: str,
    options: Dict[str, Any],
    course_context: Dict[str, Any],
    curriculum_prompt: Optional[str] = None
):
    """
    Run module extraction in background thread
    Updates job status in global dict
    """
    try:
        module_extraction_jobs[job_id]["status"] = "processing"
        module_extraction_jobs[job_id]["message"] = "Module extraction başlatılıyor..."
        
        # Get extraction service
        extraction_service = get_extraction_service()
        
        # Extract modules
        module_extraction_jobs[job_id]["message"] = "Modüller çıkarılıyor..."
        
        # Extract course_id from course_context or use default
        course_id = course_context.get("course_id") if course_context else None
        if course_id is None:
            # Try to get course_id from session or use default
            # Default to 1 if not provided (you may want to adjust this)
            course_id = 1
            logger.warning(f"No course_id provided in course_context, using default: {course_id}")
        
        # Add curriculum_prompt to options if provided
        if curriculum_prompt:
            if options is None:
                options = {}
            options['curriculum_prompt'] = curriculum_prompt
        
        result = asyncio.run(extraction_service.extract_modules_from_session(
            session_id=session_id,
            course_id=course_id,
            strategy=extraction_strategy,
            options=options
        ))
        
        module_extraction_jobs[job_id]["status"] = "completed"
        module_extraction_jobs[job_id]["message"] = "Module extraction tamamlandı!"
        module_extraction_jobs[job_id]["result"] = result
        
        logger.info(f"✅ Background module extraction completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Background module extraction failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        module_extraction_jobs[job_id]["status"] = "failed"
        module_extraction_jobs[job_id]["error"] = str(e)


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/extract")
async def extract_modules(request: ModuleExtractionRequest):
    """
    Extract curriculum-aware modules from session content
    """
    # Check if module extraction is enabled
    if not FeatureFlags.is_module_extraction_enabled(request.session_id):
        raise HTTPException(
            status_code=403,
            detail="Module extraction is disabled. Please enable it from admin settings."
        )
    
    # Also check if APRAG is enabled (prerequisite)
    if not FeatureFlags.is_aprag_enabled(request.session_id):
        raise HTTPException(
            status_code=403,
            detail="APRAG module is disabled. Module extraction requires APRAG to be enabled."
        )
    
    try:
        # Create job for background processing
        job_id = str(uuid.uuid4())
        module_extraction_jobs[job_id] = {
            "job_id": job_id,
            "session_id": request.session_id,
            "strategy": request.extraction_strategy,
            "status": "starting",
            "message": "Module extraction işi başlatılıyor...",
            "progress_percentage": 0,
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Start background thread
        thread = threading.Thread(
            target=run_module_extraction_in_background,
            args=(
                job_id,
                request.session_id,
                request.extraction_strategy,
                request.options or {},
                request.course_context or {},
                request.curriculum_prompt
            ),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started background module extraction job: {job_id} for session {request.session_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "session_id": request.session_id,
            "extraction_strategy": request.extraction_strategy,
            "message": "Module extraction arka planda başlatıldı. Progress takip edebilirsiniz.",
            "status_check_url": f"/api/aprag/modules/extract/status/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting module extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Module extraction failed to start: {str(e)}")


@router.get("/extract/status/{job_id}")
async def get_extraction_status(job_id: str):
    """
    Get status of background module extraction job
    """
    if job_id not in module_extraction_jobs:
        raise HTTPException(status_code=404, detail="Extraction job not found")
    
    job = module_extraction_jobs[job_id]
    return {
        "job_id": job_id,
        "session_id": job["session_id"],
        "strategy": job["strategy"],
        "status": job["status"],  # starting, processing, completed, failed
        "message": job["message"],
        "progress_percentage": job["progress_percentage"],
        "result": job["result"],
        "error": job["error"]
    }


@router.get("/job/{job_id}/status")
async def get_extraction_job_status(job_id: str):
    """
    Get status of background module extraction job (alternative endpoint for frontend compatibility)
    """
    if job_id not in module_extraction_jobs:
        raise HTTPException(status_code=404, detail="Extraction job not found")
    
    job = module_extraction_jobs[job_id]
    result = job.get("result", {})
    
    # Map backend status to frontend expected status
    status = job["status"]
    if status == "processing":
        status = "in_progress"
    elif status == "starting":
        status = "in_progress"
    
    # Format response to match frontend expectations
    return {
        "job_id": job_id,
        "session_id": job["session_id"],
        "status": status,  # in_progress, completed, failed
        "message": job["message"],
        "progress_percentage": job.get("progress_percentage", 0),
        "created_at": job.get("created_at", datetime.now().isoformat()),
        "completed_at": datetime.now().isoformat() if job["status"] == "completed" else None,
        "result_summary": {
            "modules_extracted": result.get("modules_created", 0) if result else 0,
            "topics_organized": result.get("topics_organized", 0) if result else 0,
            "alignment_score": result.get("alignment_score", 0.0) if result else 0.0
        } if result else None,
        "error_details": {
            "error_message": job.get("error"),
            "error_type": "extraction_error"
        } if job.get("error") else None,
        "result": result,
        "error": job.get("error")
    }


@router.get("/session/{session_id}")
async def get_session_modules(session_id: str):
    """
    Get all modules for a session
    """
    # Check if module extraction is enabled
    if not FeatureFlags.is_module_extraction_enabled(session_id):
        return {
            "success": False,
            "modules": [],
            "total": 0,
            "message": "Module extraction is disabled"
        }
    
    db = get_db()
    
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    module_id, session_id, module_title, module_description,
                    module_order, estimated_duration_hours, difficulty_level,
                    learning_outcomes, assessment_methods, prerequisites,
                    extraction_confidence, is_active, created_at, updated_at
                FROM course_modules
                WHERE session_id = ? AND is_active = TRUE
                ORDER BY module_order, module_id
            """, (session_id,))
            
            modules = []
            for row in cursor.fetchall():
                module = dict(row)
                # Parse JSON fields
                module["learning_outcomes"] = json.loads(module["learning_outcomes"]) if module["learning_outcomes"] else []
                module["assessment_methods"] = json.loads(module["assessment_methods"]) if module["assessment_methods"] else []
                module["prerequisites"] = json.loads(module["prerequisites"]) if module["prerequisites"] else []
                modules.append(module)
            
            return {
                "success": True,
                "modules": modules,
                "total": len(modules)
            }
            
    except Exception as e:
        logger.error(f"Error fetching modules for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch modules: {str(e)}")


@router.get("/{module_id}")
async def get_module(module_id: int):
    """
    Get a specific module with its topics
    """
    db = get_db()
    
    try:
        with db.get_connection() as conn:
            # Get module details
            cursor = conn.execute("""
                SELECT 
                    module_id, session_id, module_title, module_description,
                    module_order, estimated_duration_hours, difficulty_level,
                    learning_outcomes, assessment_methods, prerequisites,
                    extraction_confidence, is_active, created_at, updated_at
                FROM course_modules
                WHERE module_id = ? AND is_active = TRUE
            """, (module_id,))
            
            module_row = cursor.fetchone()
            if not module_row:
                raise HTTPException(status_code=404, detail="Module not found")
            
            module = dict(module_row)
            session_id = module["session_id"]
            
            # Check if module extraction is enabled
            if not FeatureFlags.is_module_extraction_enabled(session_id):
                raise HTTPException(
                    status_code=403,
                    detail="Module extraction is disabled"
                )
            
            # Parse JSON fields
            module["learning_outcomes"] = json.loads(module["learning_outcomes"]) if module["learning_outcomes"] else []
            module["assessment_methods"] = json.loads(module["assessment_methods"]) if module["assessment_methods"] else []
            module["prerequisites"] = json.loads(module["prerequisites"]) if module["prerequisites"] else []
            
            # Get module topics
            cursor = conn.execute("""
                SELECT 
                    t.topic_id, t.topic_title, mtr.topic_order_in_module,
                    mtr.importance_level, t.estimated_difficulty, t.keywords
                FROM module_topic_relationships mtr
                JOIN course_topics t ON mtr.topic_id = t.topic_id
                WHERE mtr.module_id = ? AND t.is_active = TRUE
                ORDER BY mtr.topic_order_in_module, t.topic_id
            """, (module_id,))
            
            topics = []
            for row in cursor.fetchall():
                topic = dict(row)
                topic["keywords"] = json.loads(topic["keywords"]) if topic["keywords"] else []
                topics.append(topic)
            
            module["topics"] = topics
            module["topic_count"] = len(topics)
            
            return {
                "success": True,
                "module": module
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching module {module_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch module: {str(e)}")


@router.put("/{module_id}")
async def update_module(module_id: int, request: ModuleUpdateRequest):
    """
    Update a module
    """
    db = get_db()
    
    # Get session_id from module first to check feature flags
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT session_id FROM course_modules WHERE module_id = ?", (module_id,))
        module = cursor.fetchone()
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")
        
        session_id = dict(module)["session_id"]
        
        # Check if module extraction is enabled
        if not FeatureFlags.is_module_extraction_enabled(session_id):
            raise HTTPException(
                status_code=403,
                detail="Module extraction is disabled. Cannot update modules."
            )
    
    try:
        with db.get_connection() as conn:
            # Build update query dynamically
            updates = []
            params = []
            
            if request.module_title is not None:
                updates.append("module_title = ?")
                params.append(request.module_title)
            
            if request.module_description is not None:
                updates.append("module_description = ?")
                params.append(request.module_description)
            
            if request.module_order is not None:
                updates.append("module_order = ?")
                params.append(request.module_order)
            
            if request.estimated_duration_hours is not None:
                updates.append("estimated_duration_hours = ?")
                params.append(request.estimated_duration_hours)
            
            if request.difficulty_level is not None:
                updates.append("difficulty_level = ?")
                params.append(request.difficulty_level)
            
            if request.learning_outcomes is not None:
                updates.append("learning_outcomes = ?")
                params.append(json.dumps(request.learning_outcomes, ensure_ascii=False))
            
            if request.assessment_methods is not None:
                updates.append("assessment_methods = ?")
                params.append(json.dumps(request.assessment_methods, ensure_ascii=False))
            
            if request.prerequisites is not None:
                updates.append("prerequisites = ?")
                params.append(json.dumps(request.prerequisites))
            
            if request.is_active is not None:
                updates.append("is_active = ?")
                params.append(request.is_active)
            
            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(module_id)
            
            cursor = conn.execute(f"""
                UPDATE course_modules
                SET {', '.join(updates)}
                WHERE module_id = ?
            """, params)
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Module not found")
            
            conn.commit()
            
            return {"success": True, "message": "Module updated successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating module {module_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update module: {str(e)}")


@router.delete("/{module_id}")
async def delete_module(module_id: int):
    """
    Delete a module and its relationships (cascading deletes handled by database)
    """
    db = get_db()
    
    # Get module info first
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT session_id, module_title FROM course_modules WHERE module_id = ?", (module_id,))
        module = cursor.fetchone()
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")
        
        module_dict = dict(module)
        session_id = module_dict["session_id"]
        module_title = module_dict.get("module_title", "Unknown")
        
        # Check if module extraction is enabled
        if not FeatureFlags.is_module_extraction_enabled(session_id):
            raise HTTPException(
                status_code=403,
                detail="Module extraction is disabled. Cannot delete modules."
            )
    
    try:
        with db.get_connection() as conn:
            # Delete module (cascading deletes will handle related records)
            cursor = conn.execute("DELETE FROM course_modules WHERE module_id = ?", (module_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Module not found")
            
            conn.commit()
            
            logger.info(f"Module {module_id} ('{module_title}') deleted successfully")
            
            return {
                "success": True,
                "message": f"Module '{module_title}' deleted successfully",
                "module_id": module_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting module {module_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete module: {str(e)}")


@router.post("/validate")
async def validate_modules(request: ModuleValidationRequest):
    """
    Validate a set of modules for quality and compliance
    """
    try:
        validator = get_quality_validator()
        
        is_valid, issues, corrected_modules = await validator.validate_module_set(
            modules=request.modules,
            course_context=request.course_context,
            validation_options=request.validation_options
        )
        
        # Convert issues to response format
        issues_response = []
        for issue in issues:
            issues_response.append({
                "severity": issue.severity.value,
                "category": issue.category,
                "message": issue.message,
                "suggested_fix": issue.suggested_fix,
                "affected_items": issue.affected_items,
                "auto_fixable": issue.auto_fixable,
                "module_id": issue.module_id
            })
        
        return {
            "success": True,
            "is_valid": is_valid,
            "issues": issues_response,
            "corrected_modules": corrected_modules,
            "validation_summary": {
                "total_issues": len(issues),
                "errors": len([i for i in issues if i.severity == ValidationSeverity.ERROR]),
                "warnings": len([i for i in issues if i.severity == ValidationSeverity.WARNING]),
                "info": len([i for i in issues if i.severity == ValidationSeverity.INFO])
            }
        }
        
    except Exception as e:
        logger.error(f"Error validating modules: {e}")
        raise HTTPException(status_code=500, detail=f"Module validation failed: {str(e)}")


@router.get("/progress/{user_id}/{session_id}")
async def get_module_progress(user_id: str, session_id: str):
    """
    Get student progress for all modules in a session
    """
    # Check if module extraction is enabled
    if not FeatureFlags.is_module_extraction_enabled(session_id):
        return {
            "success": False,
            "progress": [],
            "current_module": None,
            "next_recommended_module": None,
            "message": "Module extraction is disabled"
        }
    
    db = get_db()
    
    try:
        # Convert user_id to integer
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid user_id: {user_id}")
            raise HTTPException(status_code=400, detail=f"Invalid user_id: {user_id}")
        
        with db.get_connection() as conn:
            # Get all modules with progress
            cursor = conn.execute("""
                SELECT 
                    m.module_id,
                    m.module_title,
                    m.module_order,
                    m.estimated_duration_hours,
                    COALESCE(p.completion_percentage, 0.0) as completion_percentage,
                    COALESCE(p.status, 'not_started') as status,
                    COALESCE(p.time_spent_minutes, 0.0) as time_spent_minutes,
                    p.current_topic_id,
                    p.last_accessed_at,
                    p.started_at,
                    p.completed_at,
                    CASE 
                        WHEN p.completion_percentage >= 80 THEN 1
                        ELSE 0
                    END as is_ready_for_next
                FROM course_modules m
                LEFT JOIN module_progress p ON m.module_id = p.module_id 
                    AND p.user_id = ? AND p.session_id = ?
                WHERE m.session_id = ? AND m.is_active = TRUE
                ORDER BY m.module_order, m.module_id
            """, (user_id_int, session_id, session_id))
            
            progress = []
            current_module = None
            next_recommended = None
            
            for row in cursor.fetchall():
                module_progress = dict(row)
                
                # Determine current module (first incomplete module)
                if (current_module is None and 
                    module_progress.get("status", "not_started") not in ["completed", "mastered"]):
                    current_module = module_progress
                
                # Find next recommended module (first module ready for next)
                if (next_recommended is None and 
                    module_progress.get("is_ready_for_next")):
                    next_recommended = module_progress
                
                progress.append(module_progress)
            
            return {
                "success": True,
                "progress": progress,
                "current_module": current_module,
                "next_recommended_module": next_recommended,
                "total_modules": len(progress)
            }
            
    except Exception as e:
        logger.error(f"Error fetching module progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch module progress: {str(e)}")


@router.put("/progress/{user_id}/{module_id}")
async def update_module_progress(
    user_id: str,
    module_id: int,
    request: ModuleProgressUpdateRequest
):
    """
    Update student progress for a specific module
    """
    db = get_db()
    
    # Get session_id from module first
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT session_id FROM course_modules WHERE module_id = ?", (module_id,))
        module = cursor.fetchone()
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")
        
        session_id = dict(module)["session_id"]
        
        # Check if module extraction is enabled
        if not FeatureFlags.is_module_extraction_enabled(session_id):
            raise HTTPException(
                status_code=403,
                detail="Module extraction is disabled. Cannot update progress."
            )
    
    try:
        # Convert user_id to integer
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid user_id: {user_id}")
            raise HTTPException(status_code=400, detail=f"Invalid user_id: {user_id}")
        
        with db.get_connection() as conn:
            # Build update query dynamically
            updates = []
            params = []
            
            if request.completion_percentage is not None:
                updates.append("completion_percentage = ?")
                params.append(request.completion_percentage)
            
            if request.status is not None:
                updates.append("status = ?")
                params.append(request.status)
            
            if request.time_spent_minutes is not None:
                updates.append("time_spent_minutes = ?")
                params.append(request.time_spent_minutes)
            
            if request.current_topic_id is not None:
                updates.append("current_topic_id = ?")
                params.append(request.current_topic_id)
            
            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            # Always update last_accessed_at
            updates.append("last_accessed_at = CURRENT_TIMESTAMP")
            
            # Set started_at if status becomes in_progress and not already set
            if request.status == "in_progress":
                updates.append("started_at = COALESCE(started_at, CURRENT_TIMESTAMP)")
            
            # Set completed_at if status becomes completed or mastered
            if request.status in ["completed", "mastered"]:
                updates.append("completed_at = CURRENT_TIMESTAMP")
            
            params.extend([user_id_int, session_id, module_id])
            
            # Use INSERT OR REPLACE for upsert
            conn.execute(f"""
                INSERT OR REPLACE INTO module_progress (
                    user_id, session_id, module_id, {', '.join(updates)}
                ) VALUES (?, ?, ?, {', '.join(['?' for _ in updates])})
            """, params)
            
            conn.commit()
            
            return {"success": True, "message": "Module progress updated successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating module progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update module progress: {str(e)}")


@router.get("/curriculum-standards")
async def get_curriculum_standards():
    """
    Get available curriculum standards for module extraction
    """
    db = get_db()
    
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    standard_id, standard_code, subject, grade_level, 
                    title, description, learning_outcomes
                FROM curriculum_standards
                WHERE is_active = TRUE
                ORDER BY subject, grade_level, standard_code
            """)
            
            standards = []
            for row in cursor.fetchall():
                standard = dict(row)
                standard["learning_outcomes"] = json.loads(standard["learning_outcomes"]) if standard["learning_outcomes"] else []
                standards.append(standard)
            
            # Group by subject and grade
            grouped_standards = {}
            for standard in standards:
                subject = standard["subject"]
                grade = standard["grade_level"]
                
                if subject not in grouped_standards:
                    grouped_standards[subject] = {}
                if grade not in grouped_standards[subject]:
                    grouped_standards[subject][grade] = []
                
                grouped_standards[subject][grade].append(standard)
            
            return {
                "success": True,
                "standards": standards,
                "grouped_standards": grouped_standards,
                "total": len(standards)
            }
            
    except Exception as e:
        logger.error(f"Error fetching curriculum standards: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch curriculum standards: {str(e)}")


@router.get("/templates/subjects")
async def get_available_templates():
    """
    Get available curriculum template subjects and grades
    """
    try:
        template_manager = CurriculumTemplateManager()
        
        subjects = template_manager.get_available_subjects()
        subject_grades = {}
        
        for subject in subjects:
            grades = template_manager.get_available_grades(subject)
            subject_grades[subject] = grades
        
        return {
            "success": True,
            "subjects": subjects,
            "subject_grades": subject_grades,
            "total_subjects": len(subjects)
        }
        
    except Exception as e:
        logger.error(f"Error fetching curriculum templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch templates: {str(e)}")


@router.get("/analytics/session/{session_id}")
async def get_session_module_analytics(session_id: str):
    """
    Get analytics and insights for modules in a session
    """
    # Check if module extraction is enabled
    if not FeatureFlags.is_module_extraction_enabled(session_id):
        return {
            "success": False,
            "analytics": {},
            "message": "Module extraction is disabled"
        }
    
    db = get_db()
    
    try:
        with db.get_connection() as conn:
            # Use analytics view from migration
            cursor = conn.execute("""
                SELECT * FROM module_completion_analytics
                WHERE session_id = ?
            """, (session_id,))
            
            analytics_data = [dict(row) for row in cursor.fetchall()]
            
            # Get curriculum alignment data
            cursor = conn.execute("""
                SELECT * FROM curriculum_alignment_view
                WHERE session_id = ?
            """, (session_id,))
            
            alignment_data = [dict(row) for row in cursor.fetchall()]
            
            return {
                "success": True,
                "analytics": {
                    "completion_analytics": analytics_data,
                    "curriculum_alignment": alignment_data,
                    "session_id": session_id
                }
            }
            
    except Exception as e:
        logger.error(f"Error fetching session analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")


@router.post("/create-from-rag")
async def create_module_from_rag(request: CreateModuleFromRAGRequest):
    """
    Create a module from RAG-based semantic search
    
    Workflow:
    1. User provides module title
    2. Use RAG to find relevant chunks via semantic search
    3. Extract topics from chunks (via metadata or related_chunk_ids)
    4. Create module with found topics
    """
    db = get_db()
    
    try:
        # Step 1: Use RAG to find relevant chunks
        logger.info(f"🔍 Searching chunks for module: {request.module_title}")
        
        # Call document processing service for RAG search
        rag_response = requests.post(
            f"{DOCUMENT_PROCESSING_URL}/query",
            json={
                "query": request.module_title,
                "session_id": request.session_id,
                "top_k": request.top_k,
                "use_hybrid_search": request.use_hybrid_search,
                "use_rerank": False,  # We just need chunks, not LLM answer
                "chain_type": "stuff"
            },
            timeout=30
        )
        
        if rag_response.status_code != 200:
            raise HTTPException(
                status_code=rag_response.status_code,
                detail=f"RAG search failed: {rag_response.text}"
            )
        
        rag_data = rag_response.json()
        sources = rag_data.get("sources", [])
        
        if not sources:
            raise HTTPException(
                status_code=404,
                detail=f"No relevant chunks found for module title: {request.module_title}"
            )
        
        logger.info(f"✅ Found {len(sources)} chunks via RAG")
        
        # Step 2: Extract chunk IDs and metadata
        chunk_ids = []
        chunk_metadata_list = []
        
        for source in sources:
            # Extract chunk ID from source metadata
            chunk_id = source.get("chunk_id") or source.get("id")
            if chunk_id:
                chunk_ids.append(str(chunk_id))
            
            # Store metadata for topic extraction
            chunk_metadata_list.append({
                "chunk_id": chunk_id,
                "metadata": source.get("metadata", {}),
                "score": source.get("score", 0.0)
            })
        
        # Step 3: Find topics related to these chunks
        # Method 1: Check course_topics table for related_chunk_ids
        with db.get_connection() as conn:
            # Get all topics for this session
            cursor = conn.execute("""
                SELECT topic_id, topic_title, related_chunk_ids, description, keywords
                FROM course_topics
                WHERE session_id = ? AND is_active = 1
            """, (request.session_id,))
            
            all_topics = cursor.fetchall()
            topic_dict = {row[0]: {
                "topic_id": row[0],
                "topic_title": row[1],
                "related_chunk_ids": json.loads(row[2]) if row[2] else [],
                "description": row[3],
                "keywords": json.loads(row[4]) if row[4] else []
            } for row in all_topics}
            
            # Find topics that have any of our chunk IDs
            matched_topic_ids = []
            for topic_id, topic_data in topic_dict.items():
                topic_chunk_ids = [str(cid) for cid in topic_data["related_chunk_ids"]]
                # Check if any chunk ID matches
                if any(cid in topic_chunk_ids for cid in chunk_ids):
                    matched_topic_ids.append(topic_id)
            
            # If no topics found via chunk IDs, try semantic matching with topic titles
            if not matched_topic_ids:
                logger.info("No topics found via chunk IDs, trying semantic matching...")
                # Use module title to find similar topics
                for topic_id, topic_data in topic_dict.items():
                    topic_title = topic_data["topic_title"].lower()
                    module_title_lower = request.module_title.lower()
                    # Simple keyword matching
                    if any(word in topic_title for word in module_title_lower.split() if len(word) > 3):
                        matched_topic_ids.append(topic_id)
            
            if not matched_topic_ids:
                raise HTTPException(
                    status_code=404,
                    detail=f"No topics found related to chunks for module: {request.module_title}"
                )
            
            logger.info(f"✅ Found {len(matched_topic_ids)} topics related to module")
            
            # Step 4: Get next module order
            cursor = conn.execute("""
                SELECT COALESCE(MAX(module_order), 0) + 1
                FROM modules
                WHERE session_id = ?
            """, (request.session_id,))
            next_order = cursor.fetchone()[0]
            
            # Step 5: Create module
            cursor = conn.execute("""
                INSERT INTO modules (
                    session_id, module_title, module_description, module_order,
                    estimated_duration_hours, difficulty_level, is_active,
                    extraction_method, extraction_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.session_id,
                request.module_title,
                request.module_description or f"Modül: {request.module_title}",
                next_order,
                0.0,  # Will be calculated from topics
                "intermediate",
                1,
                "rag_based",
                json.dumps({
                    "chunk_count": len(chunk_ids),
                    "topic_count": len(matched_topic_ids),
                    "rag_query": request.module_title,
                    "similarity_threshold": request.similarity_threshold
                })
            ))
            
            module_id = cursor.lastrowid
            
            # Step 6: Link topics to module
            for idx, topic_id in enumerate(matched_topic_ids, 1):
                cursor.execute("""
                    INSERT INTO module_topics (
                        module_id, topic_id, topic_order_in_module, importance_level
                    ) VALUES (?, ?, ?, ?)
                """, (module_id, topic_id, idx, "high"))
            
            conn.commit()
            
            logger.info(f"✅ Created module {module_id} with {len(matched_topic_ids)} topics")
            
            return {
                "success": True,
                "module_id": module_id,
                "module_title": request.module_title,
                "topics_added": len(matched_topic_ids),
                "chunks_used": len(chunk_ids),
                "message": f"Modül '{request.module_title}' başarıyla oluşturuldu"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating module from RAG: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create module: {str(e)}")