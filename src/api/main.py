"""
Clean API Gateway - Only Routing & Session Management
No heavy dependencies like ChromaDB, FAISS, or ML libraries
"""
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
import uuid
import logging
from pathlib import Path
from datetime import datetime
import time
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import httpx
import io
import time
try:
    from PyPDF2 import PdfReader, PdfWriter  # lightweight pure-python
except Exception:
    PdfReader = None
    PdfWriter = None

# Session Management Integration
from src.services.session_manager import (
    professional_session_manager,
    SessionCategory,
    SessionStatus,
    SessionMetadata
)

# Cloud Storage Manager Import
from src.utils.cloud_storage_manager import cloud_storage_manager

# SQLite database manager for markdown categories
from src.database.database import get_db_manager

db_manager = get_db_manager()

app = FastAPI(title="RAG3 API Gateway", version="1.0.0",
              description="Pure API Gateway - Routes requests to microservices")

# CREDENTIALS-COMPATIBLE CORS configuration (no wildcard allowed with credentials)
logger.info("[API GATEWAY] Setting up CORS with credentials support (no wildcard)")

origins = [
    # Local development
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
    
    # Docker container networking
    "http://frontend:3000",
    "http://api-gateway:8000",
    
    # GUARANTEED server deployment origins (46.62.254.131)
    "http://46.62.254.131:3000",
    "http://46.62.254.131:8000",
    "http://46.62.254.131:8006",
    "http://46.62.254.131:8007",
    
    # HTTPS variants
    "https://46.62.254.131:3000",
    "https://46.62.254.131:8000",
    "https://46.62.254.131:8006",
    "https://46.62.254.131:8007"
]

logger.info(f"[API GATEWAY CORS] Credentials-compatible origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Microservice URLs from environment variables - Google Cloud Run compatible
# For Docker: use service names (e.g., http://document-processing-service:8080)
# For Cloud Run: use full URLs (e.g., https://document-processing-xxx.run.app)
PDF_PROCESSOR_URL = os.getenv('PDF_PROCESSOR_URL', None)
if not PDF_PROCESSOR_URL:
    PDF_PROCESSOR_HOST = os.getenv('PDF_PROCESSOR_HOST', 'docstrange-service')
    PDF_PROCESSOR_PORT = os.getenv('PDF_PROCESSOR_PORT', '80')
    if PDF_PROCESSOR_HOST.startswith('http://') or PDF_PROCESSOR_HOST.startswith('https://'):
        PDF_PROCESSOR_URL = PDF_PROCESSOR_HOST
    else:
        PDF_PROCESSOR_URL = f'http://{PDF_PROCESSOR_HOST}:{PDF_PROCESSOR_PORT}'

DOCUMENT_PROCESSOR_URL = os.getenv('DOCUMENT_PROCESSOR_URL', None)
if not DOCUMENT_PROCESSOR_URL:
    DOCUMENT_PROCESSOR_PORT = int(os.getenv('DOCUMENT_PROCESSOR_PORT', '8080'))
    DOCUMENT_PROCESSOR_HOST = os.getenv('DOCUMENT_PROCESSOR_HOST', 'document-processing-service')
    if DOCUMENT_PROCESSOR_HOST.startswith('http://') or DOCUMENT_PROCESSOR_HOST.startswith('https://'):
        DOCUMENT_PROCESSOR_URL = DOCUMENT_PROCESSOR_HOST
    else:
        DOCUMENT_PROCESSOR_URL = f'http://{DOCUMENT_PROCESSOR_HOST}:{DOCUMENT_PROCESSOR_PORT}'
# Import centralized configuration with fallback
try:
    import sys
    from pathlib import Path
    # Add project root to path for config import
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "config"
    if str(config_path) not in sys.path:
        sys.path.insert(0, str(config_path))
    # Also try rag3_for_local/config
    rag3_config_path = project_root / "rag3_for_local" / "config"
    if str(rag3_config_path) not in sys.path:
        sys.path.insert(0, str(rag3_config_path))
    from ports import AUTH_SERVICE_URL, API_GATEWAY_URL, get_service_url
    logger.info("Successfully imported additional ports configuration")
except (ImportError, Exception) as e:
    logger.warning(f"Could not import additional ports configuration ({e}), using fallbacks")
    # Use environment variables with sensible defaults
    AUTH_SERVICE_PORT = int(os.getenv('AUTH_SERVICE_PORT', '8006'))
    API_GATEWAY_PORT = int(os.getenv('API_GATEWAY_PORT', os.getenv('PORT', '8000')))
    MARKER_API_PORT = int(os.getenv('MARKER_API_PORT', '8090'))
    
    # For Cloud Run: AUTH_SERVICE_URL should be full URL (e.g., https://auth-service-xxx.run.app)
    # For Docker: use service name (e.g., http://auth-service:8006)
    AUTH_SERVICE_URL = os.getenv('AUTH_SERVICE_URL', f'http://auth-service:{AUTH_SERVICE_PORT}')
    API_GATEWAY_URL = os.getenv('API_GATEWAY_URL', f'http://localhost:{API_GATEWAY_PORT}')
    
    def get_service_url(service_name, use_docker_names=True):
        service_map = {
            "marker_api": f"http://marker-api:{MARKER_API_PORT}" if use_docker_names else f"http://localhost:{MARKER_API_PORT}"
        }
        return service_map.get(service_name, f"http://localhost:{API_GATEWAY_PORT}")

# Model Inference Service - Google Cloud Run compatible
# If MODEL_INFERENCE_URL is set (Cloud Run), use it directly
# Otherwise, construct from host and port (Docker)
MODEL_INFERENCE_URL = os.getenv('MODEL_INFERENCE_URL', None)
if not MODEL_INFERENCE_URL:
    MODEL_INFERENCE_PORT = int(os.getenv('MODEL_INFERENCE_PORT', '8002'))
    MODEL_INFERENCE_HOST = os.getenv('MODEL_INFERENCE_HOST', 'model-inference-service')
    # Check if host is a full URL (Cloud Run)
    if MODEL_INFERENCE_HOST.startswith('http://') or MODEL_INFERENCE_HOST.startswith('https://'):
        MODEL_INFERENCE_URL = MODEL_INFERENCE_HOST
    else:
        MODEL_INFERENCE_URL = f'http://{MODEL_INFERENCE_HOST}:{MODEL_INFERENCE_PORT}'

# Auth Service - Google Cloud Run compatible
# If AUTH_SERVICE_URL is set (Cloud Run), use it directly
# Otherwise, construct from host and port (Docker)
if 'AUTH_SERVICE_URL' not in locals() or not AUTH_SERVICE_URL or AUTH_SERVICE_URL.startswith('http://auth-service'):
    AUTH_SERVICE_URL_ENV = os.getenv('AUTH_SERVICE_URL', None)
    if AUTH_SERVICE_URL_ENV:
        AUTH_SERVICE_URL = AUTH_SERVICE_URL_ENV
    else:
        AUTH_SERVICE_PORT = int(os.getenv('AUTH_SERVICE_PORT', '8006'))
        AUTH_SERVICE_HOST = os.getenv('AUTH_SERVICE_HOST', 'auth-service')
        if AUTH_SERVICE_HOST.startswith('http://') or AUTH_SERVICE_HOST.startswith('https://'):
            AUTH_SERVICE_URL = AUTH_SERVICE_HOST
        else:
            AUTH_SERVICE_URL = f'http://{AUTH_SERVICE_HOST}:{AUTH_SERVICE_PORT}'

# Marker API - Google Cloud Run compatible
MARKER_API_URL = os.getenv('MARKER_API_URL', None)
if not MARKER_API_URL:
    MARKER_API_PORT = int(os.getenv('MARKER_API_PORT', '8090'))
    MARKER_API_HOST = os.getenv('MARKER_API_HOST', 'marker-api')
    if MARKER_API_HOST.startswith('http://') or MARKER_API_HOST.startswith('https://'):
        MARKER_API_URL = MARKER_API_HOST
    else:
        MARKER_API_URL = get_service_url("marker_api", use_docker_names=True)

# APRAG Service - Google Cloud Run compatible
# If APRAG_SERVICE_URL is set (Cloud Run), use it directly
# Otherwise, construct from host and port (Docker)
APRAG_SERVICE_URL = os.getenv('APRAG_SERVICE_URL', None)
if not APRAG_SERVICE_URL:
    APRAG_SERVICE_PORT = int(os.getenv('APRAG_SERVICE_PORT', '8007'))
    APRAG_SERVICE_HOST = os.getenv('APRAG_SERVICE_HOST', 'aprag-service')
    if APRAG_SERVICE_HOST.startswith('http://') or APRAG_SERVICE_HOST.startswith('https://'):
        APRAG_SERVICE_URL = APRAG_SERVICE_HOST
    else:
        APRAG_SERVICE_URL = f'http://{APRAG_SERVICE_HOST}:{APRAG_SERVICE_PORT}'

# Add Main API Server URL for RAG queries - Google Cloud Run için PORT environment variable desteği
MAIN_API_URL = os.getenv('MAIN_API_URL', API_GATEWAY_URL)

# Follow-up suggestion settings
SUGGESTION_COUNT = int(os.getenv('SUGGESTION_COUNT', '3'))

# Pydantic Models
class CreateSessionRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    category: str
    created_by: str = "system"
    grade_level: Optional[str] = ""
    subject_area: Optional[str] = ""
    learning_objectives: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    is_public: bool = False

class SessionResponse(BaseModel):
    session_id: str
    name: str
    description: str
    category: str
    status: str
    created_by: str
    created_at: str
    updated_at: str
    last_accessed: str
    grade_level: str
    subject_area: str
    learning_objectives: List[str]
    tags: List[str]
    document_count: int
    total_chunks: int
    query_count: int
    user_rating: float
    is_public: bool
    backup_count: int
    rag_settings: Optional[Dict[str, Any]] = None

class RAGQueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5
    use_rerank: bool = True
    min_score: float = 0.1
    max_context_chars: int = 8000
    model: Optional[str] = None
    use_direct_llm: bool = False
    chain_type: Optional[str] = None
    embedding_model: Optional[str] = None
    max_tokens: Optional[int] = 2048  # Answer length: 1024 (short), 2048 (normal), 4096 (detailed)
    conversation_history: Optional[List[Dict[str, str]]] = None  # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    processing_time_ms: Optional[int] = None
    suggestions: List[str] = []

class SuggestionRequest(BaseModel):
    question: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = []

class PDFToMarkdownResponse(BaseModel):
    success: bool
    message: str
    markdown_filename: Optional[str] = None
    metadata: Optional[dict] = None

class MarkdownListResponse(BaseModel):
    markdown_files: List[str]
    count: int


class MarkdownCategory(BaseModel):
    id: int
    name: str
    description: Optional[str] = None


class MarkdownCategoryCreateUpdate(BaseModel):
    name: str
    description: Optional[str] = None


class MarkdownFileWithCategory(BaseModel):
    filename: str
    category_id: Optional[int] = None
    category_name: Optional[str] = None

@app.get("/")
def root():
    return {
        "service": "RAG3 API Gateway", 
        "status": "ok",
        "version": "1.0.0",
        "microservices": {
            "pdf_processor": PDF_PROCESSOR_URL,
            "document_processor": DOCUMENT_PROCESSOR_URL,
            "model_inference": MODEL_INFERENCE_URL,
            "auth_service": AUTH_SERVICE_URL,
            "aprag_service": APRAG_SERVICE_URL
        }
    }

@app.get("/health")
def health():
    """Health check for API Gateway"""
    return {"status": "ok", "service": "api-gateway"}

@app.get("/health/services")
async def check_microservices():
    """Check health of all microservices"""
    services = {
        "pdf_processor": PDF_PROCESSOR_URL,
        "document_processor": DOCUMENT_PROCESSOR_URL,
        "model_inference": MODEL_INFERENCE_URL,
        "auth_service": AUTH_SERVICE_URL,
        "aprag_service": APRAG_SERVICE_URL
    }
    
    results = {}
    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            results[name] = {
                "status": "ok" if response.status_code == 200 else "error",
                "url": url,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            results[name] = {
                "status": "error",
                "url": url,
                "error": str(e)
            }
    
    return {"gateway": "ok", "services": results}

# Session Management - Real Implementation with SQLite Database
def _convert_metadata_to_response(metadata: SessionMetadata) -> SessionResponse:
    """Convert SessionMetadata to SessionResponse"""
    # Provide default RAG settings if none exist
    default_rag_settings = {
        "top_k": 5,
        "use_rerank": True,
        "min_score": 0.5,
        "max_context_chars": 8000,
        "model": None,
        "chain_type": None,
        "use_direct_llm": False,
        "embedding_model": os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
    }
    rag_settings = metadata.rag_settings or default_rag_settings
    
    return SessionResponse(
        session_id=metadata.session_id,
        name=metadata.name,
        description=metadata.description,
        category=metadata.category.value,
        status=metadata.status.value,
        created_by=metadata.created_by,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        last_accessed=metadata.last_accessed,
        grade_level=metadata.grade_level,
        subject_area=metadata.subject_area,
        learning_objectives=metadata.learning_objectives,
        tags=metadata.tags,
        document_count=metadata.document_count,
        total_chunks=metadata.total_chunks,
        query_count=metadata.query_count,
        user_rating=metadata.user_rating,
        is_public=metadata.is_public,
        backup_count=metadata.backup_count,
        rag_settings=rag_settings,
    )

@app.get("/sessions", response_model=List[SessionResponse])
def list_sessions(created_by: Optional[str] = None, category: Optional[str] = None,
                  status: Optional[str] = None, limit: int = 50, request: Request = None):
    """List sessions from SQLite database"""
    try:
        # Determine requester and role
        current_user = _get_current_user(request)
        # Convert string parameters to enums if provided
        category_enum = None
        if category:
            try:
                category_enum = SessionCategory(category)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        status_enum = None
        if status:
            try:
                status_enum = SessionStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        sessions: List[SessionMetadata]
        if _is_admin(current_user):
            # Admin sees all
            sessions = professional_session_manager.list_sessions(
                created_by=None, category=category_enum, status=status_enum, limit=limit
            )
            logger.info(f"[SESSION LIST] Admin user - returning {len(sessions)} sessions")
        elif _is_teacher(current_user):
            # Teachers see only their own sessions
            # Fetch and filter by owner keys to be tolerant of historical data
            all_sessions = professional_session_manager.list_sessions(
                created_by=None, category=category_enum, status=status_enum, limit=limit
            )
            owner_keys = set(_user_owner_keys(current_user))
            logger.info(f"[SESSION LIST] Teacher user - owner_keys: {owner_keys}, all_sessions count: {len(all_sessions)}")
            logger.info(f"[SESSION LIST] Current user info: {current_user}")
            
            # Debug: log first few session created_by values BEFORE filtering
            if len(all_sessions) > 0:
                sample_created_by = [s.created_by for s in all_sessions[:5]]
                logger.info(f"[SESSION LIST] Sample created_by values from DB: {sample_created_by}")
            
            sessions = [s for s in all_sessions if s.created_by in owner_keys]
            logger.info(f"[SESSION LIST] Filtered sessions count: {len(sessions)}")
            
            # If no sessions found but sessions exist, log mismatch details
            if len(sessions) == 0 and len(all_sessions) > 0:
                all_created_by = set([s.created_by for s in all_sessions])
                logger.warning(f"[SESSION LIST] No matching sessions! Owner keys: {owner_keys}, All created_by values: {all_created_by}")
                logger.warning(f"[SESSION LIST] Mismatch detected - trying case-insensitive and partial matching...")
                
                # Try case-insensitive matching
                owner_keys_lower = {k.lower() for k in owner_keys}
                sessions = [s for s in all_sessions if s.created_by and s.created_by.lower() in owner_keys_lower]
                if len(sessions) > 0:
                    logger.info(f"[SESSION LIST] Found {len(sessions)} sessions with case-insensitive matching")
        else:
            # Students: show all active (or filter by provided status)
            sessions = professional_session_manager.list_sessions(
                created_by=None, category=category_enum, status=status_enum, limit=limit
            )
            logger.info(f"[SESSION LIST] Student user - returning {len(sessions)} sessions")
        return [_convert_metadata_to_response(session) for session in sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.post("/sessions", response_model=SessionResponse)
def create_session(req: CreateSessionRequest, request: Request):
    """Create new session in SQLite database"""
    try:
        # Convert category string to enum
        try:
            category_enum = SessionCategory(req.category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {req.category}")

        current_user = _get_current_user(request)
        creator_keys = _user_owner_keys(current_user)
        created_by = creator_keys[0] if creator_keys else (req.created_by or "system")
        
        # Create session using session manager
        metadata = professional_session_manager.create_session(
            name=req.name,
            description=req.description or "",
            category=category_enum,
            created_by=created_by,
            grade_level=req.grade_level or "",
            subject_area=req.subject_area or "",
            learning_objectives=req.learning_objectives or [],
            tags=req.tags or [],
            is_public=req.is_public
        )
        return _convert_metadata_to_response(metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, request: Request):
    """Get session details from SQLite database"""
    try:
        metadata = _require_owner_or_admin(request, session_id)
        return _convert_metadata_to_response(metadata)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, create_backup: bool = True, deleted_by: Optional[str] = None, request: Request = None):
    """Delete session from SQLite database and ChromaDB collection with enhanced retry logic"""
    try:
        # Access control
        metadata = _require_owner_or_admin(request, session_id)
        
        # ENHANCED ChromaDB collection deletion with retry logic
        chromadb_deleted = False
        chromadb_attempts = 0
        max_chromadb_retries = 3
        
        logger.info(f"🔄 Starting enhanced ChromaDB cleanup for session {session_id}")
        
        while not chromadb_deleted and chromadb_attempts < max_chromadb_retries:
            chromadb_attempts += 1
            try:
                logger.info(f"🔄 ChromaDB deletion attempt {chromadb_attempts}/{max_chromadb_retries}")
                response = requests.delete(
                    f"{DOCUMENT_PROCESSOR_URL}/sessions/{session_id}/collection",
                    timeout=45  # Increased timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    chromadb_deleted = result.get("success", False)
                    logger.info(f"✅ ChromaDB collection deletion SUCCESS for session {session_id}: {result.get('message', 'Success')}")
                    break
                else:
                    logger.warning(f"⚠️ ChromaDB deletion attempt {chromadb_attempts} failed: {response.status_code} - {response.text}")
                    if chromadb_attempts < max_chromadb_retries:
                        import time
                        time.sleep(2 * chromadb_attempts)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                logger.error(f"⚠️ ChromaDB deletion attempt {chromadb_attempts} error: {str(e)}")
                if chromadb_attempts < max_chromadb_retries:
                    import time
                    time.sleep(2 * chromadb_attempts)  # Exponential backoff
        
        if not chromadb_deleted:
            logger.error(f"❌ CRITICAL: ChromaDB collection deletion FAILED after {max_chromadb_retries} attempts for session {session_id}")
            # Continue with session deletion but warn user
        
        # Delete session from SQLite database
        success = professional_session_manager.delete_session(
            session_id=session_id,
            create_backup=create_backup,
            deleted_by=deleted_by or (metadata.created_by)
        )
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Return detailed status
        return {
            "deleted": True,
            "session_id": session_id,
            "chromadb_collection_deleted": chromadb_deleted,
            "chromadb_deletion_attempts": chromadb_attempts,
            "message": f"Session '{metadata.name}' deleted. ChromaDB cleanup: {'✅ SUCCESS' if chromadb_deleted else '❌ FAILED - may require manual cleanup'}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

class StatusUpdateRequest(BaseModel):
    status: str

@app.patch("/sessions/{session_id}/status")
def update_session_status(session_id: str, request: StatusUpdateRequest, req: Request):
    """Update session status (active/inactive)"""
    try:
        # Validate status
        status = request.status
        valid_statuses = ["active", "inactive"]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )
        try:
            status_enum = SessionStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        # Access control
        _require_owner_or_admin(req, session_id)
        # Update session status
        success = professional_session_manager.update_session_status(session_id, status_enum)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        # Get updated session
        updated_metadata = professional_session_manager.get_session_metadata(session_id)
        if not updated_metadata:
            raise HTTPException(status_code=404, detail="Session not found after update")
        return {
            "success": True,
            "session_id": session_id,
            "new_status": status,
            "updated_session": _convert_metadata_to_response(updated_metadata)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session status: {str(e)}")

@app.get("/sessions/{session_id}/chunks")
def get_session_chunks(session_id: str):
    """Get chunks for a session from Document Processing Service"""
    try:
        response = requests.get(
            f"{DOCUMENT_PROCESSOR_URL}/sessions/{session_id}/chunks",
            timeout=30
        )
        
        if response.status_code == 404:
            # Return empty chunks if session not found in Document Processing Service
            return {"chunks": [], "total_count": 0, "session_id": session_id}
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch chunks from Document Processing Service: {response.text}"
            )
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        # Return empty chunks if Document Processing Service is unavailable
        return {"chunks": [], "total_count": 0, "session_id": session_id}

class GenerateQuestionsRequest(BaseModel):
    count: int = 5

@app.post("/sessions/{session_id}/generate-questions")
async def generate_course_questions(session_id: str, request: Request, req: GenerateQuestionsRequest):
    """Generate course-specific questions based on session content chunks"""
    limit = req.count
    try:
        # Access control - anyone with access to session can generate questions
        current_user = _get_current_user(request)
        session_metadata = professional_session_manager.get_session_metadata(session_id)
        if not session_metadata:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Students can access any active session, teachers need ownership
        if not _is_admin(current_user):
            if _is_teacher(current_user):
                owner_keys = set(_user_owner_keys(current_user))
                if session_metadata.created_by not in owner_keys:
                    raise HTTPException(status_code=403, detail="You do not have access to this session")
            # For students, we allow access to active sessions (no ownership check)
        
        # Get chunks from document processing service
        chunks_response = requests.get(
            f"{DOCUMENT_PROCESSOR_URL}/sessions/{session_id}/chunks",
            timeout=30
        )
        
        if chunks_response.status_code != 200:
            logger.warning(f"Failed to get chunks for session {session_id}: {chunks_response.text}")
            # Fallback to generic questions if chunks not available
            return {"questions": [
                "Bu dersin temel konuları neler?",
                "Kısa bir özet hazırla",
                "Önemli kavramları listele",
                "Bu konudaki örnekler nelerdir?"
            ]}
        
        chunks_data = chunks_response.json()
        chunks = chunks_data.get("chunks", [])
        
        if not chunks:
            logger.warning(f"No chunks found for session {session_id}")
            return {"questions": [
                "Bu dersin temel konuları neler?",
                "Kısa bir özet hazırla",
                "Önemli kavramları listele"
            ]}
        
        # Sample chunks for analysis - RASTGELE SAMPLING with time-based seed (avoid token limits)
        import random
        import time
        # Use current time as seed to ensure different selection each time
        random.seed(int(time.time() * 1000) % 10000)
        
        if len(chunks) > 20:
            # Rastgele 20 chunk seç (daha çeşitli sorular için)
            sample_chunks = random.sample(chunks, 20)
        elif len(chunks) > 10:
            # Orta boyutta chunk listesi için rastgele 15 seç
            sample_chunks = random.sample(chunks, min(15, len(chunks)))
        else:
            # Küçük listelerde tümünü karıştır
            sample_chunks = chunks.copy()
            random.shuffle(sample_chunks)
        
        # Extract content from chunks with better randomization
        content_samples = []
        for chunk in sample_chunks:
            chunk_text = chunk.get("chunk_text", "")
            if chunk_text and len(chunk_text.strip()) > 50:  # Skip very short chunks
                content_samples.append(chunk_text[:300])  # Limit chunk size
        
        if not content_samples:
            logger.warning(f"No valid content found in chunks for session {session_id}")
            return {"questions": [
                "Bu dersin temel konuları neler?",
                "Ana kavramları açıklar mısın?",
                "Bu konu hakkında özet yaz"
            ]}
        
        # Shuffle content samples and use random subset for context
        random.shuffle(content_samples)
        # Use more samples for better diversity, limit by character count
        selected_samples = []
        total_chars = 0
        max_context_chars = 2000  # Increased context window for better question diversity
        
        for sample in content_samples:
            if total_chars + len(sample) <= max_context_chars:
                selected_samples.append(sample)
                total_chars += len(sample)
            if len(selected_samples) >= 8:  # Max 8 different chunks for variety
                break
        
        content_context = "\n\n".join(selected_samples)
        
        logger.info(f"Generated context from {len(selected_samples)} random chunks ({total_chars} chars) for session {session_id}")
        
        # Generate questions using LLM
        prompt = (
            "Sen bir eğitim uzmanısın. Aşağıdaki ders içeriğini analiz ederek, "
            "öğrencilerin sorabilecekleri en mantıklı ve yararlı soruları üret.\n\n"
            "KATI KURALLAR:\n"
            "1. KESINLIKLE TÜRKÇE sorular üret. Hiçbir durumda İngilizce kelime kullanma.\n"
            "2. Sorular, verilen DERS İÇERİĞİNE DOĞRUDAN dayalı olmalı.\n"
            "3. İçerikte geçen spesifik kavramlar, konular ve detaylar üzerine sorular oluştur.\n"
            "4. Genel veya alakasız sorular üretme. Her soru içerikle doğrudan bağlantılı olmalı.\n"
            "5. Öğrenci seviyesinde, anlaşılır sorular üret.\n"
            "6. Her soru tek satırda, soru işaretiyle bitsin.\n"
            "7. 5 adet soru üret. Başka açıklama yapma.\n\n"
            "DERS İÇERİĞİ:\n"
            f"{content_context}\n\n"
            "Bu ders içeriğine dayanarak 5 adet Türkçe soru üret:\n"
        )
        
        generation_request = {
            "prompt": prompt,
            "model": os.getenv("DEFAULT_SUGGESTION_MODEL", "llama-3.1-8b-instant"),
            "temperature": 0.7,
            "max_tokens": 500,
        }
        
        response = requests.post(
            f"{MODEL_INFERENCE_URL}/models/generate",
            json=generation_request,
            timeout=45
        )
        
        if response.status_code != 200:
            logger.error(f"LLM generation failed: {response.status_code} - {response.text}")
            # Fallback questions based on content analysis
            return {"questions": [
                "Bu derste işlenen konular nelerdir?",
                "Ana kavramları açıklar mısın?",
                "Bu konudaki örnekleri verir misin?",
                "Bu ders hakkında özet yazar mısın?",
                "Bu konuyla ilgili sorular sorabilirim"
            ]}
        
        result = response.json()
        generated_text = result.get("response", "")
        
        # Parse questions from response
        lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
        questions = []
        
        for line in lines:
            # Clean up line
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering and bullet points
            line = line.lstrip("0123456789.-•* ")
            
            # Ensure it ends with question mark
            if not line.endswith("?"):
                line += "?"
                
            # Skip very short or long questions
            if len(line) < 10 or len(line) > 150:
                continue
                
            # Skip English questions (basic check)
            if any(word in line.lower() for word in ["what", "how", "why", "when", "where", "the", "and", "is", "are"]):
                continue
            
            questions.append(line)
            
            if len(questions) >= limit:
                break
        
        # If we don't have enough good questions, add some fallback ones
        if len(questions) < 3:
            fallback_questions = [
                "Bu derste işlenen temel konular nelerdir?",
                "Ders materyalindeki önemli kavramları açıklar mısın?",
                "Bu konu hakkında detaylı bilgi verir misin?",
                "Bu dersten ne öğrenebilirim?",
                "Ders içeriğindeki örnekleri açıklar mısın?"
            ]
            for fq in fallback_questions:
                if fq not in questions and len(questions) < limit:
                    questions.append(fq)
        
        logger.info(f"Generated {len(questions)} questions for session {session_id}")
        return {"questions": questions[:limit]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate course questions: {str(e)}")
        # Return fallback questions on any error
        return {"questions": [
            "Bu dersin temel konuları neler?",
            "Kısa bir özet hazırla",
            "Önemli kavramları listele",
            "Bu konu hakkında ne öğrenebilirim?"
        ]}

@app.post("/sessions/{session_id}/reprocess")
async def reprocess_session_documents(session_id: str, request: Request):
    """Re-process existing documents in a session with a new embedding model"""
    try:
        # Access control
        _require_owner_or_admin(request, session_id)
        
        # Get request body
        body = await request.json()
        
        # Forward to document processing service
        response = requests.post(
            f"{DOCUMENT_PROCESSOR_URL}/sessions/{session_id}/reprocess",
            json=body,
            timeout=600  # 10 minutes for large documents
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        
        # Update session metadata if successful
        if result.get("success"):
            chunks_processed = result.get("chunks_processed", 0)
            # For reprocessing, document_count stays the same (we're just re-embedding existing documents)
            # Get current session metadata to preserve document_count
            current_metadata = professional_session_manager.get_session_metadata(session_id)
            current_document_count = current_metadata.document_count if current_metadata else 0
            
            professional_session_manager.update_session_counts(
                session_id=session_id,
                document_count=current_document_count,
                total_chunks=chunks_processed
            )
            
            # IMPORTANT: Save the embedding model to session rag_settings
            # This ensures chat queries use the same embedding model
            embedding_model = body.get("embedding_model")
            if embedding_model:
                current_settings = professional_session_manager.get_session_rag_settings(session_id) or {}
                current_settings["embedding_model"] = embedding_model
                professional_session_manager.save_session_rag_settings(
                    session_id=session_id,
                    settings=current_settings,
                    user_id=None  # System update
                )
                logger.info(f"Updated session {session_id} rag_settings with embedding_model: {embedding_model}")
        
        return result
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to communicate with Document Processing Service: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to re-process documents: {str(e)}")

# Document Processing - Route to PDF Processor Service
@app.post("/api/documents/convert-document-to-markdown", response_model=PDFToMarkdownResponse)
@app.post("/documents/convert-document-to-markdown", response_model=PDFToMarkdownResponse)
async def convert_document_to_markdown(file: UploadFile = File(...)):
    """Convert document to markdown - Route to PDF Processing Service"""
    supported_extensions = ['.pdf', '.docx', '.pptx', '.xlsx']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
        )
    
    try:
        # Read uploaded file content
        content = await file.read()
        
        # Route to PDF Processing Service
        files = {'file': (file.filename, content, file.content_type)}
        
        response = requests.post(
            f"{PDF_PROCESSOR_URL}/convert/pdf-to-markdown",
            files=files,
            timeout=600  # 10 minutes for large PDF processing (includes DocStrange polling)
        )
        
        if response.status_code != 200:
            error_detail = f"PDF processor service error: {response.status_code}"
            if response.text:
                try:
                    error_json = response.json()
                    error_detail = error_json.get('detail', error_detail)
                except:
                    error_detail = f"{error_detail} - {response.text[:200]}"
                    
            raise HTTPException(status_code=500, detail=error_detail)
        
        # Parse response from PDF processor (DocStrange format)
        processor_result = response.json()
        
        # Handle DocStrange response format
        markdown_content = None
        if 'result' in processor_result and isinstance(processor_result['result'], list):
            full_text = ""
            for item in processor_result['result']:
                full_text += item.get('markdown', '') + "\n\n"
            if full_text.strip():
                markdown_content = full_text.strip()
        else:
            # Fallback to direct content field
            markdown_content = processor_result.get('content')
        
        metadata = processor_result.get('metadata', {"source": "docstrange-service"})
        
        if not markdown_content or not markdown_content.strip():
            raise HTTPException(
                status_code=500, 
                detail="Failed to extract content from PDF. The document may be image-based or corrupted."
            )
        
        # Save markdown content using cloud storage manager
        base_filename = Path(file.filename).stem
        markdown_filename = f"{base_filename}.md"
        
        success = cloud_storage_manager.save_markdown_file(markdown_filename, markdown_content)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save markdown file")
        
        return PDFToMarkdownResponse(
            success=True,
            message=f"Document successfully converted to Markdown and saved to {'cloud storage' if cloud_storage_manager.is_cloud else 'local storage'}",
            markdown_filename=markdown_filename,
            metadata=metadata
        )
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="PDF Processing Service is not available. Please try again later."
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="PDF dönüştürme işlemi zaman aşımına uğradı (10 dakika). Lütfen daha küçük bir dosya deneyin veya dosyayı bölümlere ayırın."
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with PDF Processing Service: {str(e)}"
        )

# Batch Processing Jobs Tracking (similar to KB extraction)
BATCH_PROCESSING_JOBS: Dict[str, Any] = {}

# Document Processing - Route to Document Processor Service  
@app.post("/api/documents/process-and-store")
async def process_and_store_documents(
    session_id: str = Form(...),
    markdown_files: str = Form(...),  # JSON string of file list
    chunk_strategy: str = Form("lightweight"),
    chunk_size: int = Form(500),  # Reduced from 1000 to 500 for more chunks
    chunk_overlap: int = Form(100),
    embedding_model: str = Form("mixedbread-ai/mxbai-embed-large-v1"),
    use_llm_post_processing: bool = Form(False),  # NEW: Optional LLM post-processing for chunk refinement
    llm_model_name: str = Form("llama-3.1-8b-instant"),  # NEW: LLM model for post-processing
    model_inference_url: str = Form(None)  # NEW: Override model inference URL for LLM post-processing
):
    """Process documents and store vectors - Route to Document Processing Service"""
    try:
        # IMPORTANT: Check session rag_settings for embedding_model first
        # If session has saved embedding_model in rag_settings, use it instead of Form parameter
        session_rag_settings = professional_session_manager.get_session_rag_settings(session_id)
        if session_rag_settings and session_rag_settings.get("embedding_model"):
            embedding_model = session_rag_settings["embedding_model"]
            logger.info(f"Using embedding_model from session rag_settings: {embedding_model}")
        else:
            logger.info(f"Using embedding_model from Form parameter: {embedding_model}")
        
        # Adjust chunk sizes for Alibaba embedding models (they handle larger chunks better)
        is_alibaba_embedding = (
            embedding_model and (
                embedding_model.startswith("text-embedding-") or
                "alibaba" in embedding_model.lower() or
                "dashscope" in embedding_model.lower()
            )
        )
        
        if is_alibaba_embedding:
            # Alibaba embeddings can handle larger chunks - adjust if not explicitly set
            if chunk_size == 500:  # Only adjust if using default value
                chunk_size = 2500
                logger.info(f"🔵 Alibaba embedding detected ({embedding_model}): Adjusted chunk_size to {chunk_size}")
            if chunk_overlap == 100:  # Only adjust if using default value
                chunk_overlap = 500
                logger.info(f"🔵 Alibaba embedding detected ({embedding_model}): Adjusted chunk_overlap to {chunk_overlap}")
        else:
            logger.info(f"⚪ Standard embedding model ({embedding_model}): Using provided chunk sizes (size={chunk_size}, overlap={chunk_overlap})")
        
        # Parse markdown files list
        files_list = json.loads(markdown_files)
        
        # Process each markdown file separately
        total_processed = 0
        total_chunks = 0
        successful_files = []
        failed_files = []
        
        for filename in files_list:
            try:
                content = cloud_storage_manager.get_markdown_file_content(filename)
                if content and content.strip():
                    # Process each file individually - CRITICAL FIX: Add chunk_strategy and LLM post-processing to root payload
                    payload = {
                        "text": content,  # Individual file content
                        "metadata": {
                            "session_id": session_id,
                            "source_file": filename,  # Single file
                            "filename": filename,     # Also add filename for compatibility
                            "embedding_model": embedding_model,
                            "chunk_strategy": chunk_strategy
                        },
                        "collection_name": f"session_{session_id}",
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunk_strategy": chunk_strategy,  # CRITICAL: Pass chunk_strategy to enable semantic chunking
                        "use_llm_post_processing": use_llm_post_processing,  # NEW: Optional LLM post-processing
                        "llm_model_name": llm_model_name,  # NEW: LLM model for post-processing
                        "model_inference_url": model_inference_url or MODEL_INFERENCE_URL  # NEW: Model inference URL for LLM post-processing
                    }
                    
                    file_response = requests.post(
                        f"{DOCUMENT_PROCESSOR_URL}/process-and-store",
                        json=payload,
                        timeout=600
                    )
                    
                    if file_response.status_code == 200:
                        file_result = file_response.json()
                        total_processed += 1
                        total_chunks += file_result.get("chunks_processed", 0)
                        successful_files.append(filename)
                    else:
                        failed_files.append(f"{filename}: {file_response.text}")
                        
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                failed_files.append(f"{filename}: {str(e)}")
        
        if total_processed == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Could not process any files. Errors: {'; '.join(failed_files)}"
            )
        
        # Return combined results
        processor_result = {
            "success": True,
            "processed_count": total_processed,
            "chunks_created": total_chunks,
            "message": f"Successfully processed {total_processed} files",
            "successful_files": successful_files,
            "failed_files": failed_files if failed_files else None
        }
        
        # Update session metadata after successful processing
        if processor_result.get("success", False):
            try:
                chunks_added = processor_result.get("chunks_created", 0)
                files_processed = processor_result.get("processed_count", 0)
                
                # Get current session metadata
                session_metadata = professional_session_manager.get_session_metadata(session_id)
                if session_metadata:
                    # Update session with new counts
                    new_document_count = session_metadata.document_count + files_processed
                    new_total_chunks = session_metadata.total_chunks + chunks_added
                    
                    # Update session in database
                    professional_session_manager.update_session_counts(
                        session_id=session_id,
                        document_count=new_document_count,
                        total_chunks=new_total_chunks
                    )
                    
                    logger.info(f"Updated session {session_id}: {files_processed} documents, {chunks_added} chunks added")
                    
                # IMPORTANT: Save the embedding model to session rag_settings
                # This ensures chat queries use the same embedding model as documents
                try:
                    current_settings = professional_session_manager.get_session_rag_settings(session_id) or {}
                    current_settings["embedding_model"] = embedding_model
                    professional_session_manager.save_session_rag_settings(
                        session_id=session_id,
                        settings=current_settings,
                        user_id=None  # System update
                    )
                    logger.info(f"Saved embedding_model '{embedding_model}' to session {session_id} rag_settings")
                except Exception as settings_error:
                    logger.error(f"Failed to save embedding_model to session rag_settings: {str(settings_error)}")
                    # Don't fail the whole operation if settings save fails
                    
            except Exception as update_error:
                logger.error(f"Failed to update session metadata: {str(update_error)}")
                # Don't fail the whole operation if metadata update fails
        
        # Map the field names to match what frontend expects
        return {
            "success": processor_result.get("success", False),
            "message": processor_result.get("message", ""),
            "processed_count": processor_result.get("processed_count", 0),
            "total_chunks_added": processor_result.get("chunks_created", 0),
            "processing_time": processor_result.get("processing_time")
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with Document Processing Service: {str(e)}"
        )


# Batch Processing Jobs Tracking (similar to KB extraction)
BATCH_PROCESSING_JOBS: Dict[str, Any] = {}


# Batch Processing Endpoint - Background Job
@app.post("/api/documents/process-and-store-batch")
async def process_and_store_documents_batch(
    session_id: str = Form(...),
    markdown_files: str = Form(...),  # JSON string of file list
    chunk_strategy: str = Form("lightweight"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100),
    embedding_model: str = Form("mixedbread-ai/mxbai-embed-large-v1"),
    use_llm_post_processing: bool = Form(False),
    llm_model_name: str = Form("llama-3.1-8b-instant"),
    model_inference_url: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Process markdown files in batch with background job tracking.
    Returns job_id immediately; progress can be tracked via GET /api/documents/process-and-store-batch/status/{job_id}
    """
    try:
        # Parse markdown files list
        files_list = json.loads(markdown_files)
        
        if not files_list or len(files_list) == 0:
            raise HTTPException(status_code=400, detail="No markdown files provided")
        
        # Get session rag_settings for embedding_model
        session_rag_settings = professional_session_manager.get_session_rag_settings(session_id)
        if session_rag_settings and session_rag_settings.get("embedding_model"):
            embedding_model = session_rag_settings["embedding_model"]
            logger.info(f"Using embedding_model from session rag_settings: {embedding_model}")
        
        # Adjust chunk sizes for Alibaba embedding models
        is_alibaba_embedding = (
            embedding_model and (
                embedding_model.startswith("text-embedding-") or
                "alibaba" in embedding_model.lower() or
                "dashscope" in embedding_model.lower()
            )
        )
        
        if is_alibaba_embedding:
            if chunk_size == 500:
                chunk_size = 2500
            if chunk_overlap == 100:
                chunk_overlap = 500
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        BATCH_PROCESSING_JOBS[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "running",  # running | completed | failed
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "total_files": len(files_list),
            "processed_successfully": 0,
            "errors_count": 0,
            "current_file": None,
            "current_batch": 0,
            "total_batches": 0,
            "total_chunks": 0,
            "results": [],
            "errors": [],
        }
        
        # Prepare job data
        job_data = {
            "session_id": session_id,
            "files_list": files_list,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "use_llm_post_processing": use_llm_post_processing,
            "llm_model_name": llm_model_name,
            "model_inference_url": model_inference_url or MODEL_INFERENCE_URL,
        }
        
        # Schedule background job
        if background_tasks:
            background_tasks.add_task(
                _run_batch_processing_job,
                job_id,
                job_data
            )
        else:
            # Fallback: run inline (for testing)
            await _run_batch_processing_job(job_id, job_data)
        
        return {
            "success": True,
            "message": "Batch processing started in background",
            "job_id": job_id,
            "total_files": len(files_list),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start batch processing: {str(e)}")


async def _run_batch_processing_job(job_id: str, job_data: Dict[str, Any]):
    """
    Internal background job that performs batch markdown processing.
    Updates BATCH_PROCESSING_JOBS[job_id] with progress.
    """
    job = BATCH_PROCESSING_JOBS.get(job_id)
    if not job:
        logger.warning(f"[BATCH PROCESSING JOB] Job {job_id} not found at start")
        return
    
    try:
        files_list = job_data["files_list"]
        session_id = job_data["session_id"]
        results = []
        errors = []
        total_chunks = 0
        
        # Process files in batches (e.g., 10 files per batch)
        BATCH_SIZE = 10
        total_batches = (len(files_list) + BATCH_SIZE - 1) // BATCH_SIZE
        current_batch = 0
        
        for idx, filename in enumerate(files_list):
            # Calculate current batch number (1-indexed)
            if idx % BATCH_SIZE == 0:
                current_batch += 1
                job["current_batch"] = current_batch
                job["total_batches"] = total_batches
                logger.info(f"[BATCH PROCESSING JOB {job_id}] Starting batch {current_batch}/{total_batches}")
            try:
                job["current_file"] = filename
                logger.info(f"[BATCH PROCESSING JOB {job_id}] Processing file: {filename}")
                
                # Get file content
                content = cloud_storage_manager.get_markdown_file_content(filename)
                if not content or not content.strip():
                    errors.append({
                        "filename": filename,
                        "error": "File is empty or not found"
                    })
                    job["errors_count"] = len(errors)
                    continue
                
                # Process file
                payload = {
                    "text": content,
                    "metadata": {
                        "session_id": session_id,
                        "source_file": filename,
                        "filename": filename,
                        "embedding_model": job_data["embedding_model"],
                        "chunk_strategy": job_data["chunk_strategy"]
                    },
                    "collection_name": f"session_{session_id}",
                    "chunk_size": job_data["chunk_size"],
                    "chunk_overlap": job_data["chunk_overlap"],
                    "chunk_strategy": job_data["chunk_strategy"],
                    "use_llm_post_processing": job_data["use_llm_post_processing"],
                    "llm_model_name": job_data["llm_model_name"],
                    "model_inference_url": job_data["model_inference_url"]
                }
                
                file_response = requests.post(
                    f"{DOCUMENT_PROCESSOR_URL}/process-and-store",
                    json=payload,
                    timeout=600
                )
                
                if file_response.status_code == 200:
                    file_result = file_response.json()
                    chunks_processed = file_result.get("chunks_processed", 0)
                    total_chunks += chunks_processed
                    
                    results.append({
                        "filename": filename,
                        "chunks_processed": chunks_processed,
                        "success": True
                    })
                    
                    job["processed_successfully"] = len(results)
                    job["total_chunks"] = total_chunks
                    job["results"] = results[-10:]  # Keep last 10 results
                else:
                    error_msg = file_response.text
                    errors.append({
                        "filename": filename,
                        "error": error_msg
                    })
                    job["errors_count"] = len(errors)
                    job["errors"] = errors[-10:]  # Keep last 10 errors
                    
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                logger.error(f"[BATCH PROCESSING JOB {job_id}] Error processing file {filename}: {e}")
                
                errors.append({
                    "filename": filename,
                    "error": str(e)
                })
                job["errors_count"] = len(errors)
                job["errors"] = errors[-10:]
        
        # Update session metadata
        if len(results) > 0:
            try:
                session_metadata = professional_session_manager.get_session_metadata(session_id)
                if session_metadata:
                    new_document_count = session_metadata.document_count + len(results)
                    new_total_chunks = session_metadata.total_chunks + total_chunks
                    
                    professional_session_manager.update_session_counts(
                        session_id=session_id,
                        document_count=new_document_count,
                        total_chunks=new_total_chunks
                    )
                    
                    # Save embedding model to session rag_settings
                    current_settings = professional_session_manager.get_session_rag_settings(session_id) or {}
                    current_settings["embedding_model"] = job_data["embedding_model"]
                    professional_session_manager.save_session_rag_settings(
                        session_id=session_id,
                        settings=current_settings,
                        user_id=None
                    )
            except Exception as update_error:
                logger.error(f"Failed to update session metadata: {str(update_error)}")
        
        # Mark job as completed
        job["status"] = "completed" if len(errors) == 0 else "completed_with_errors"
        job["completed_at"] = datetime.now().isoformat()
        job["current_file"] = None
        
        logger.info(f"[BATCH PROCESSING JOB {job_id}] Completed: {len(results)} successful, {len(errors)} errors, {total_chunks} total chunks")
        
    except Exception as e:
        logger.error(f"[BATCH PROCESSING JOB {job_id}] Fatal error: {e}")
        job["status"] = "failed"
        job["completed_at"] = datetime.now().isoformat()
        job["current_file"] = None


@app.get("/api/documents/process-and-store-batch/status/{job_id}")
async def get_batch_processing_status(job_id: str):
    """Get status of a batch processing job"""
    job = BATCH_PROCESSING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "success": True,
        "job": job
    }

# Helper to fetch current user from Auth Service using the incoming Authorization header
def _get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        resp = requests.get(f"{AUTH_SERVICE_URL}/auth/me", headers={"Authorization": auth_header}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        logger.warning(f"Auth user fetch failed: {e}")
        return None

# Profile Management Endpoints
@app.get("/profile", tags=["Profile"])
async def get_profile(request: Request):
    """Get current user profile"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        resp = requests.get(
            f"{AUTH_SERVICE_URL}/users/me",
            headers={"Authorization": auth_header},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Unauthorized")
        else:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")

@app.put("/profile", tags=["Profile"])
async def update_profile(request: Request, profile_data: dict = Body(...)):
    """Update current user profile"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Only allow updating username, email, first_name, last_name
        allowed_fields = {"username", "email", "first_name", "last_name"}
        filtered_data = {k: v for k, v in profile_data.items() if k in allowed_fields}
        
        resp = requests.put(
            f"{AUTH_SERVICE_URL}/users/me",
            json=filtered_data,
            headers={"Authorization": auth_header, "Content-Type": "application/json"},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Unauthorized")
        else:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@app.put("/profile/change-password", tags=["Profile"])
async def change_password(request: Request, password_data: dict = Body(...)):
    """Change current user password"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if "old_password" not in password_data or "new_password" not in password_data:
            raise HTTPException(status_code=400, detail="old_password and new_password are required")
        
        resp = requests.put(
            f"{AUTH_SERVICE_URL}/auth/change-password",
            json=password_data,
            headers={"Authorization": auth_header, "Content-Type": "application/json"},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Unauthorized")
        elif resp.status_code == 400:
            raise HTTPException(status_code=400, detail=resp.json().get("detail", "Invalid password"))
        else:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")

def _get_role_name(user: Optional[Dict[str, Any]]) -> str:
    if not user:
        return ""
    role = user.get("role_name") or user.get("role") or ""
    return str(role).lower()

def _is_admin(user: Optional[Dict[str, Any]]) -> bool:
    return _get_role_name(user) in {"admin", "superadmin"}

def _is_teacher(user: Optional[Dict[str, Any]]) -> bool:
    return _get_role_name(user) in {"teacher", "ogretmen", "instructor"}

def _user_owner_keys(user: Optional[Dict[str, Any]]) -> List[str]:
    if not user:
        return []
    vals: List[str] = []
    if user.get("id") is not None:
        vals.append(str(user.get("id")))
    if user.get("username"):
        vals.append(str(user.get("username")))
    if user.get("email"):
        vals.append(str(user.get("email")))
    return [v for v in vals if v]

def _require_owner_or_admin(request: Request, session_id: str) -> SessionMetadata:
    metadata = professional_session_manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")
    user = _get_current_user(request)
    if _is_admin(user):
        return metadata
    # Teachers must own the session
    if _is_teacher(user):
        owner_keys = set(_user_owner_keys(user))
        if metadata.created_by not in owner_keys:
            raise HTTPException(status_code=403, detail="You do not have access to this session")
    return metadata

class RAGSettings(BaseModel):
    model: Optional[str] = None
    provider: Optional[str] = None  # AI Provider (groq, deepseek, etc.)
    chain_type: Optional[str] = None
    top_k: Optional[int] = None
    use_rerank: Optional[bool] = None
    min_score: Optional[float] = None
    max_context_chars: Optional[int] = None
    use_direct_llm: Optional[bool] = None
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None  # Embedding provider (ollama, huggingface)
    use_reranker_service: Optional[bool] = None
    reranker_type: Optional[str] = None

@app.get("/sessions/{session_id}/rag-settings")
def get_rag_settings(session_id: str, request: Request):
    try:
        _require_owner_or_admin(request, session_id)
        settings = professional_session_manager.get_session_rag_settings(session_id)
        return settings or {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rag settings: {str(e)}")

@app.patch("/sessions/{session_id}/rag-settings")
def update_rag_settings(session_id: str, req: RAGSettings, request: Request):
    try:
        current_user = _get_current_user(request)
        _require_owner_or_admin(request, session_id)
        uid = current_user.get("id") if current_user else None
        
        # Get all fields from request (including None values for explicit clearing)
        request_dict = req.model_dump(exclude_unset=True)  # exclude_unset: only include fields that were explicitly set
        
        # Debug logging - use print for immediate visibility
        import logging
        logger = logging.getLogger(__name__)
        print(f"[RAG SETTINGS UPDATE] Session: {session_id}, Request data: {request_dict}")
        logger.info(f"[RAG SETTINGS UPDATE] Session: {session_id}, Request data: {request_dict}")
        
        # Merge with existing settings to preserve other fields
        existing_settings = professional_session_manager.get_session_rag_settings(session_id) or {}
        print(f"[RAG SETTINGS UPDATE] Existing settings: {existing_settings}")
        logger.info(f"[RAG SETTINGS UPDATE] Existing settings: {existing_settings}")
        
        merged_settings = {**existing_settings, **request_dict}
        
        # If use_reranker_service is explicitly False, remove reranker_type
        if 'use_reranker_service' in request_dict and request_dict['use_reranker_service'] is False:
            merged_settings.pop('reranker_type', None)
        
        # Remove None values only for final save (but keep False, 0, empty strings)
        final_settings = {k: v for k, v in merged_settings.items() if v is not None}
        
        print(f"[RAG SETTINGS UPDATE] Final settings to save: {final_settings}")
        logger.info(f"[RAG SETTINGS UPDATE] Final settings to save: {final_settings}")
        
        success = professional_session_manager.save_session_rag_settings(session_id, final_settings, uid)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Return the saved settings from database
        saved_settings = professional_session_manager.get_session_rag_settings(session_id) or {}
        logger.info(f"[RAG SETTINGS UPDATE] Saved settings returned: {saved_settings}")
        
        return {"success": True, "session_id": session_id, "rag_settings": saved_settings}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update rag settings: {str(e)}")




def _generate_followup_suggestions_sync(question: str, answer: str, sources: List[Dict[str, Any]]) -> List[str]:
    """Generate short, clickable follow-up questions in Turkish using the model-inference service."""
    try:
        if not answer:
            return []
        src_titles = []
        for s in (sources or []):
            md = s.get("metadata", {}) if isinstance(s, dict) else {}
            title = md.get("source_file") or md.get("filename") or ""
            if title:
                src_titles.append(str(title))
        context_hint = ("Kaynaklar: " + ", ".join(src_titles[:5])) if src_titles else ""
        # Extract key concepts and details from answer for context-aware suggestions
        answer_keywords = []
        if len(answer) > 0:
            # Simple keyword extraction: look for important concepts
            sentences = answer.split('.')
            for sent in sentences[:5]:  # First 5 sentences
                if len(sent.strip()) > 20:  # Substantial sentences
                    answer_keywords.append(sent.strip()[:100])  # First 100 chars
        
        answer_summary = "\n".join(answer_keywords[:3]) if answer_keywords else answer[:200]
        
        prompt = (
            "Sen bir eğitim asistanısın. Aşağıda bir öğrencinin sorusu ve asistanın Türkçe cevabı var. "
            "Görevin: Bu soru ve cevaba DOĞRUDAN BAĞLI, aynı konu bağlamında takip soruları üretmek.\n\n"
            "KATI KURALLAR:\n"
            "1. KESINLIKLE TÜRKÇE SORULAR ÖNER. Hiçbir durumda İngilizce kelime, cümle veya ifade kullanma.\n"
            "2. Önerdiğin sorular, verilen SORU ve CEVAP ile DOĞRUDAN İLGİLİ olmalı. Aynı konu bağlamında kalmalı.\n"
            "3. Cevapta bahsedilen kavramlar, örnekler, detaylar üzerine takip soruları oluştur.\n"
            "4. Cevapta geçen spesifik bilgileri, örnekleri, kavramları kullanarak sorular üret.\n"
            "5. Genel veya konuyla alakasız sorular önerme. Her soru, verilen cevabın bir yönüne bağlı olmalı.\n"
            "6. 'Bu kavramın temel özellikleri neler?' gibi generic sorular önerme. Spesifik ve konuya bağlı sorular üret.\n"
            "7. Her soru tek satır, doğal Türkçe cümleler olmalı. Numara veya işaret kullanma.\n"
            "8. 3-5 soru öner. Sadece soruları sırayla yaz. Başka açıklama yapma.\n\n"
            f"Soru: {question}\n\n"
            f"Cevap Özeti (cevapla doğrudan ilgili kısımlar):\n{answer_summary}\n\n"
            f"{context_hint}\n\n"
            "Bu soru ve cevaba DOĞRUDAN BAĞLI, aynı konu bağlamında, cevaptaki spesifik bilgileri kullanarak Türkçe takip soruları öner:\n\n"
            "Öneriler:"
        )
        generation_request = {
            "prompt": prompt,
            "model": os.getenv("DEFAULT_SUGGESTION_MODEL", "llama-3.1-8b-instant"),
            "temperature": 0.1,  # Lower temperature for more focused, consistent suggestions
            "max_tokens": 400,  # Increased for better suggestions
        }
        resp = requests.post(
            f"{MODEL_INFERENCE_URL}/models/generate",
            json=generation_request,
            timeout=30,
        )
        if resp.status_code != 200:
            logger.info(f"suggestions: model-inference non-200 {resp.status_code}")
            return []
        text = (resp.json() or {}).get("response", "")
        
        # Remove English introductory phrases that LLM might add
        english_intros = [
            "here are the follow-up questions",
            "here are some follow-up questions",
            "here are",
            "follow-up questions:",
            "suggestions:",
            "öneriler:",
            "takip soruları:"
        ]
        text_lower = text.lower()
        for intro in english_intros:
            if text_lower.startswith(intro):
                # Remove the intro line
                lines_temp = text.split("\n")
                if lines_temp:
                    text = "\n".join(lines_temp[1:])  # Skip first line
                break
        
        # Split into lines and clean bullets/numbers
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        cleaned: List[str] = []
        for l in lines:
            # Skip lines that are English introductory phrases
            l_lower = l.lower().strip()
            skip_line = False
            for intro in english_intros:
                if intro in l_lower and len(l_lower) < 50:  # Short lines that contain intro phrases
                    skip_line = True
                    break
            if skip_line:
                continue
            
            l = l.lstrip("-•*0123456789. ")
            if len(l) > 2 and l not in cleaned:
                cleaned.append(l)
            if len(cleaned) >= SUGGESTION_COUNT:
                break
        cleaned = cleaned[:SUGGESTION_COUNT]
        if not cleaned or len(cleaned) < 2:
            # If we don't have enough quality suggestions, return empty
            # Better to have no suggestions than generic ones
            logger.warning(f"suggestions: only {len(cleaned)} suggestions generated, may not be context-aware")
        logger.info(f"suggestions generated: {len(cleaned)} items")
        return cleaned
    except Exception as e:
        logger.warning(f"suggestions: exception during generation: {e}")
        # Return empty instead of generic fallback - better UX
        return []

# Suggestions endpoint for async fetching on the frontend
@app.post("/rag/suggestions")
async def generate_suggestions(req: SuggestionRequest) -> Dict[str, Any]:
    """
    Generate follow-up question suggestions asynchronously.
    Accepts the original question, generated answer, and optional sources.
    """
    try:
        suggestions = _generate_followup_suggestions_sync(
            question=req.question,
            answer=req.answer,
            sources=req.sources or [],
        )
        return {"suggestions": suggestions}
    except Exception as e:
        logger.warning(f"/rag/suggestions failed: {e}")
        return {"suggestions": []}

# RAG Query with Reranking
@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest, request: Request):
    """
    RAG Query with reranking to improve document relevance ordering.
    Uses reranker service to sort retrieved documents by relevance.
    """
    # Start timing from the very beginning of the request
    request_start_time = time.time()
    try:
        # Load saved RAG settings for this session (teacher-defined)
        saved_settings: Dict[str, Any] = professional_session_manager.get_session_rag_settings(req.session_id) or {}
        # Compute effective params (frontend can override; students will omit)
        effective = {
            "top_k": req.top_k or saved_settings.get("top_k", 5),
            "use_rerank": req.use_rerank if req.use_rerank is not None else saved_settings.get("use_rerank", True),
            "min_score": req.min_score or saved_settings.get("min_score", 0.1),
            "max_context_chars": req.max_context_chars or saved_settings.get("max_context_chars", 8000),
            "model": req.model or saved_settings.get("model"),
            "chain_type": req.chain_type or saved_settings.get("chain_type"),
            # FIX: Frontend explicitly sends False, respect it (don't use 'or' which treats False as falsy)
            "use_direct_llm": req.use_direct_llm if req.use_direct_llm is not None else bool(saved_settings.get("use_direct_llm")),
            "embedding_model": req.embedding_model or saved_settings.get("embedding_model"),
        }
        
        # If use_direct_llm is True, route directly to Model Inference Service
        if effective["use_direct_llm"]:
            if not effective["model"]:
                raise HTTPException(
                    status_code=400,
                    detail="Model is required when using direct LLM mode"
                )
            # Create a Turkish educational assistant prompt with STRICT rules
            system_prompt = (
                "Sen, öğrencilere yardımcı olan eğitimli bir yapay zeka asistanısın. "
                "KATI KURALLAR:\n"
                "1. KESINLIKLE TÜRKÇE CEVAP VER. Hiçbir durumda İngilizce kelime, cümle veya ifade kullanma.\n"
                "2. Eğer bir kavramın İngilizce ismi varsa bile, Türkçe karşılığını kullan veya Türkçe açıklama yap.\n"
                "3. Teknik terimler için bile Türkçe karşılıkları tercih et.\n"
                "4. Soruları açık, anlaşılır ve eğitici bir şekilde yanıtla.\n"
                "5. Öğrencilerin öğrenme sürecini destekle.\n"
                "6. Eğer bir soruya tam olarak cevap veremiyorsan, dürüst ol ve önerilerde bulun, ama her şeyi TÜRKÇE yap.\n"
                "Bu kurallara kesinlikle uy. İngilizce kullanmak yasaktır."
            )
            
            # Build conversation context if available
            context_parts = [f"System: {system_prompt}\n"]
            if req.conversation_history:
                for msg in req.conversation_history[-4:]:  # Last 4 messages for context
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        context_parts.append(f"User: {content}\n")
                    elif role == "assistant":
                        context_parts.append(f"Assistant: {content}\n")
            
            context_parts.append(f"User: {req.query}\n\nCevap:")
            full_prompt = "\n".join(context_parts)
            generation_request = {
                "prompt": full_prompt,
                "model": effective["model"],
                "temperature": 0.7,
                "max_tokens": req.max_tokens or 1024
            }
            response = requests.post(
                f"{MODEL_INFERENCE_URL}/models/generate",
                json=generation_request,
                timeout=120
            )
            # Calculate total time from request start to response received
            elapsed_ms = int((time.time() - request_start_time) * 1000)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Model inference failed: {response.text}"
                )
            result = response.json()
            # Update counts and log (direct LLM mode)
            try:
                # Track student entry and increment query count
                current_user = _get_current_user(request)
                student_identifier = (
                    str(current_user.get("id")) if current_user and current_user.get("id") is not None
                    else (current_user.get("username") if current_user else "student")
                )
                
                # Track student entry first
                professional_session_manager.track_student_entry(req.session_id, student_identifier)
                
                # Then update query count
                meta = professional_session_manager.get_session_metadata(req.session_id)
                if meta:
                    professional_session_manager.update_session_metadata(
                        req.session_id,
                        query_count=int(meta.query_count or 0) + 1,
                        last_accessed=datetime.now().isoformat()
                    )
                    logger.info(f"[QUERY COUNT] Updated query_count to {int(meta.query_count or 0) + 1} for session {req.session_id}")
            except Exception as update_err:
                logger.warning(f"Failed to update query_count/student_entry for session {req.session_id}: {update_err}")
            try:
                from src.analytics.database import ExperimentDatabase
                db = ExperimentDatabase()
                rag_params = {
                    "use_direct_llm": True,
                    "model": effective["model"],
                    "top_k": None,
                    "min_score": None,
                    "max_context_chars": None,
                    "chain_type": None,
                }
                config_hash = db.add_or_get_rag_configuration(rag_params)
                current_user = _get_current_user(request)
                user_identifier = str(current_user.get("id")) if current_user and current_user.get("id") is not None else (current_user.get("username") if current_user else "student")
                db.add_interaction(
                    user_id=user_identifier,
                    query=req.query,
                    response=result.get("response", ""),
                    retrieved_context=[],
                    rag_config_hash=config_hash,
                    session_id=req.session_id,
                    uncertainty_score=None,
                    feedback_requested=False,
                    processing_time_ms=elapsed_ms,
                    success=True,
                    error_message=None,
                    chain_type="direct_llm",
                )
            except Exception as log_err:
                logger.warning(f"Failed to log interaction (direct LLM): {log_err}")
            
            # APRAG Integration: Personalization and Interaction Logging
            final_answer = result.get("response", "")
            try:
                from src.utils.aprag_middleware import personalize_response_async, log_interaction_async, get_user_id_from_request
                user_id = get_user_id_from_request(request)
                
                # Try personalization (non-blocking, with timeout)
                if user_id != "anonymous":
                    try:
                        personalized = await personalize_response_async(
                            user_id=user_id,
                            session_id=req.session_id,
                            query=req.query,
                            original_response=final_answer,
                            context={"model": effective["model"], "chain_type": "direct_llm"}
                        )
                        if personalized:
                            final_answer = personalized
                            logger.info(f"APRAG: Personalized response for user {user_id}")
                    except Exception as pers_err:
                        logger.debug(f"APRAG personalization failed (non-critical): {pers_err}")
                
                # Log interaction (async, non-blocking)
                asyncio.create_task(log_interaction_async(
                    user_id=user_id,
                    session_id=req.session_id,
                    query=req.query,
                    response=result.get("response", ""),
                    personalized_response=final_answer if final_answer != result.get("response", "") else None,
                    processing_time_ms=elapsed_ms,
                    model_used=effective["model"],
                    chain_type="direct_llm",
                    sources=[],
                    metadata={"use_direct_llm": True}
                ))
            except Exception as aprag_err:
                logger.debug(f"APRAG integration failed (non-critical): {aprag_err}")
            
            suggestions = _generate_followup_suggestions_sync(
                question=req.query,
                answer=final_answer,
                sources=[],
            )
            return RAGQueryResponse(answer=final_answer, sources=[], processing_time_ms=elapsed_ms, suggestions=suggestions)
        
        # RAG Query with Reranking
        logger.info(f"🔍 Processing RAG query with reranking: '{req.query}'")
        
        # Step 1: Perform retrieval
        collection_name = f"session_{req.session_id}"
        try:
            retrieval_response = requests.post(
                f"{DOCUMENT_PROCESSOR_URL}/retrieve",
                json={
                    "query": req.query,
                    "collection_name": collection_name,
                    "top_k": effective["top_k"] * 2,  # Get more docs for reranking
                    "embedding_model": effective["embedding_model"]
                },
                timeout=30
            )
            
            if retrieval_response.status_code != 200:
                raise HTTPException(
                    status_code=retrieval_response.status_code,
                    detail=f"Retrieval failed: {retrieval_response.text}"
                )
            
            retrieval_result = retrieval_response.json()
            raw_results = retrieval_result.get("results", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Document processor communication failed: {e}")
            # Fallback to full service routing
            payload = {
                **req.model_dump(),
                "top_k": effective["top_k"],
                "use_rerank": effective["use_rerank"],
                "min_score": effective["min_score"],
                "max_context_chars": effective["max_context_chars"],
                "model": effective["model"],
                "chain_type": effective["chain_type"],
                "embedding_model": effective["embedding_model"],
            }
            response = requests.post(
                f"{DOCUMENT_PROCESSOR_URL}/query",
                json=payload,
                timeout=120
            )
            elapsed_ms = int((time.time() - request_start_time) * 1000)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Query processing failed: {response.text}"
                )
            result = response.json()
            logger.info("⚠️ Used fallback routing due to retrieval service issue")
        else:
            # Route to Document Processing Service for query processing
            logger.info("🔍 Routing to Document Processing Service for query processing")
            payload = {
                **req.model_dump(),
                "top_k": effective["top_k"],
                "use_rerank": effective["use_rerank"],
                "min_score": effective["min_score"],
                "max_context_chars": effective["max_context_chars"],
                "model": effective["model"],
                "chain_type": effective["chain_type"],
                "embedding_model": effective["embedding_model"],
            }
            response = requests.post(
                f"{DOCUMENT_PROCESSOR_URL}/query",
                json=payload,
                timeout=120
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Query processing failed: {response.text}"
                )
            result = response.json()
            
        elapsed_ms = int((time.time() - request_start_time) * 1000)
        
        # Legacy fallback pattern matching (kept for backward compatibility)
        answer = result.get("answer", "")
        no_context_patterns = [
            "cannot answer",
            "not found in the provided context",
            "no information",
            "not mentioned in the context",
            "cannot find",
            "no relevant information",
            "based on the provided context, i cannot",
            "the context does not contain",
            "i don't have information",
            "not available in the context"
        ]
        answer_lower = answer.lower()
        if any(pattern in answer_lower for pattern in no_context_patterns) or len(answer.strip()) < 20:
            result["answer"] = "⚠️ **DERS KAPSAMINDA DEĞİL**\n\nSorduğunuz soru ders dökümanlarında bulunamamıştır. Eğer sorunuzun ders içeriğiyle ilgili olduğunu düşünüyorsanız öğretmeninize bildiriniz.\n\n📚 *Lütfen ders materyalleri kapsamında sorular sorunuz.*"
            result["sources"] = []
        try:
            # Track student entry and increment query count (RAG mode)
            current_user = _get_current_user(request)
            student_identifier = (
                str(current_user.get("id")) if current_user and current_user.get("id") is not None
                else (current_user.get("username") if current_user else "student")
            )
            
            # Track student entry first
            professional_session_manager.track_student_entry(req.session_id, student_identifier)
            
            # Then update query count
            meta = professional_session_manager.get_session_metadata(req.session_id)
            if meta:
                professional_session_manager.update_session_metadata(
                    req.session_id,
                    query_count=int(meta.query_count or 0) + 1,
                    last_accessed=datetime.now().isoformat()
                )
                logger.info(f"[QUERY COUNT] Updated query_count to {int(meta.query_count or 0) + 1} for session {req.session_id}")
        except Exception as update_err:
            logger.warning(f"Failed to update query_count/student_entry for session {req.session_id}: {update_err}")
        try:
            from src.analytics.database import ExperimentDatabase
            db = ExperimentDatabase()
            rag_params = {
                "use_direct_llm": False,
                "model": effective["model"] or "",
                "top_k": effective["top_k"],
                "min_score": effective["min_score"],
                "max_context_chars": effective["max_context_chars"],
                "chain_type": payload.get("chain_type") or result.get("chain_type"),
            }
            config_hash = db.add_or_get_rag_configuration(rag_params)
            current_user = _get_current_user(request)
            user_identifier = str(current_user.get("id")) if current_user and current_user.get("id") is not None else (current_user.get("username") if current_user else "student")
            db.add_interaction(
                user_id=user_identifier,
                query=req.query,
                response=result.get("answer", ""),
                retrieved_context=result.get("sources", []),
                rag_config_hash=config_hash,
                session_id=req.session_id,
                uncertainty_score=None,
                feedback_requested=False,
                processing_time_ms=elapsed_ms,
                success=True,
                error_message=None,
                chain_type=result.get("chain_type") or (payload.get("chain_type") if payload.get("chain_type") else "rag"),
            )
        except Exception as log_err:
            logger.warning(f"Failed to log interaction: {log_err}")
        
        # APRAG Integration: Personalization and Interaction Logging
        final_answer = result.get("answer", "")
        final_sources = result.get("sources", [])
        try:
            from src.utils.aprag_middleware import personalize_response_async, log_interaction_async, get_user_id_from_request
            user_id = get_user_id_from_request(request)
            
            # Try personalization (non-blocking, with timeout)
            if user_id != "anonymous":
                try:
                    personalized = await personalize_response_async(
                        user_id=user_id,
                        session_id=req.session_id,
                        query=req.query,
                        original_response=final_answer,
                        context={
                            "model": effective["model"],
                            "chain_type": result.get("chain_type") or effective["chain_type"],
                            "top_k": effective["top_k"],
                            "sources_count": len(final_sources)
                        }
                    )
                    if personalized:
                        final_answer = personalized
                        logger.info(f"APRAG: Personalized response for user {user_id}")
                except Exception as pers_err:
                    logger.debug(f"APRAG personalization failed (non-critical): {pers_err}")
            
            # Log interaction (async, non-blocking)
            asyncio.create_task(log_interaction_async(
                user_id=user_id,
                session_id=req.session_id,
                query=req.query,
                response=result.get("answer", ""),
                personalized_response=final_answer if final_answer != result.get("answer", "") else None,
                processing_time_ms=elapsed_ms,
                model_used=effective["model"],
                chain_type=result.get("chain_type") or effective["chain_type"],
                sources=final_sources,
                metadata={
                    "top_k": effective["top_k"],
                    "use_rerank": effective["use_rerank"],
                    "min_score": effective["min_score"]
                }
            ))
        except Exception as aprag_err:
            logger.debug(f"APRAG integration failed (non-critical): {aprag_err}")
        
        suggestions = _generate_followup_suggestions_sync(
            question=req.query,
            answer=final_answer,
            sources=final_sources,
        )
        return RAGQueryResponse(answer=final_answer, sources=final_sources, processing_time_ms=elapsed_ms, suggestions=suggestions)
        
    except requests.exceptions.RequestException as e:
        # Log failure interaction if possible
        try:
            from src.analytics.database import ExperimentDatabase
            db = ExperimentDatabase()
            rag_params = {"use_direct_llm": bool(getattr(req, 'use_direct_llm', False)), "model": req.model or ""}
            config_hash = db.add_or_get_rag_configuration(rag_params)
            current_user = _get_current_user(request)
            user_identifier = str(current_user.get("id")) if current_user and current_user.get("id") is not None else (current_user.get("username") if current_user else "student")
            db.add_interaction(
                user_id=user_identifier,
                query=req.query,
                response="",
                retrieved_context=[],
                rag_config_hash=config_hash,
                session_id=req.session_id,
                uncertainty_score=None,
                feedback_requested=False,
                processing_time_ms=None,
                success=False,
                error_message=str(e),
                chain_type="direct_llm" if getattr(req, 'use_direct_llm', False) else "rag",
            )
        except Exception as _log_err:
            logger.warning(f"Failed to log failed interaction: {_log_err}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with service: {str(e)}"
        )

# Simple endpoints that don't require microservices
@app.get("/api/documents/list-markdown", response_model=MarkdownListResponse)
@app.get("/documents/list-markdown", response_model=MarkdownListResponse)
def list_markdown_files():
    """List markdown files - using cloud storage manager"""
    try:
        md_files = cloud_storage_manager.list_markdown_files()
        return MarkdownListResponse(markdown_files=md_files, count=len(md_files))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list markdown files: {str(e)}")


@app.get("/api/markdown-categories", response_model=List[MarkdownCategory])
def get_markdown_categories():
    """Return all markdown categories from SQLite, ordered by name."""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, name, description
                FROM markdown_categories
                ORDER BY name COLLATE NOCASE
                """
            )
            rows = cursor.fetchall()
            return [MarkdownCategory(id=row["id"], name=row["name"], description=row["description"]) for row in rows]
    except Exception as e:
        logger.error(f"Failed to list markdown categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to list markdown categories")


@app.post("/api/markdown-categories", response_model=MarkdownCategory)
def create_markdown_category(payload: MarkdownCategoryCreateUpdate):
    """Create a new markdown category."""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO markdown_categories (name, description)
                VALUES (?, ?)
                """,
                (payload.name.strip(), payload.description or None),
            )
            conn.commit()
            new_id = cursor.lastrowid

            row = conn.execute(
                "SELECT id, name, description FROM markdown_categories WHERE id = ?",
                (new_id,),
            ).fetchone()
            return MarkdownCategory(id=row["id"], name=row["name"], description=row["description"])
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Kategori adı zaten kullanılıyor")
    except Exception as e:
        logger.error(f"Failed to create markdown category: {e}")
        raise HTTPException(status_code=500, detail="Failed to create markdown category")


@app.put("/api/markdown-categories/{category_id}", response_model=MarkdownCategory)
def update_markdown_category(category_id: int, payload: MarkdownCategoryCreateUpdate):
    """Update an existing markdown category."""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE markdown_categories
                SET name = ?, description = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (payload.name.strip(), payload.description or None, category_id),
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Kategori bulunamadı")

            row = conn.execute(
                "SELECT id, name, description FROM markdown_categories WHERE id = ?",
                (category_id,),
            ).fetchone()
            return MarkdownCategory(id=row["id"], name=row["name"], description=row["description"])
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Kategori adı zaten kullanılıyor")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update markdown category: {e}")
        raise HTTPException(status_code=500, detail="Failed to update markdown category")


@app.delete("/api/markdown-categories/{category_id}")
def delete_markdown_category(category_id: int):
    """Delete a markdown category.

    markdown_file_categories has a foreign key with ON DELETE SET NULL,
    so linked markdown files will keep their filenames but lose category.
    """
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM markdown_categories WHERE id = ?",
                (category_id,),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Kategori bulunamadı")
            return {"deleted": True, "category_id": category_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete markdown category: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete markdown category")


class AssignMarkdownCategoryRequest(BaseModel):
    filenames: List[str]
    category_id: Optional[int] = None


@app.post("/api/markdown-files/assign-category")
def assign_markdown_category(payload: AssignMarkdownCategoryRequest):
    """Assign or clear category for one or more markdown files.

    - If category_id is provided, upsert mapping for given filenames.
    - If category_id is null, clear category mapping for given filenames.
    """
    if not payload.filenames:
        raise HTTPException(status_code=400, detail="En az bir dosya adı gerekli")

    safe_filenames = [os.path.basename(f) for f in payload.filenames]

    try:
        with db_manager.get_connection() as conn:
            if payload.category_id is None:
                # Clear category: delete mappings
                conn.executemany(
                    "DELETE FROM markdown_file_categories WHERE filename = ?",
                    [(name,) for name in safe_filenames],
                )
            else:
                # Ensure category exists
                exists = conn.execute(
                    "SELECT 1 FROM markdown_categories WHERE id = ?",
                    (payload.category_id,),
                ).fetchone()
                if not exists:
                    raise HTTPException(status_code=404, detail="Kategori bulunamadı")

                # Upsert mappings
                conn.executemany(
                    """
                    INSERT INTO markdown_file_categories (filename, category_id, created_at, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(filename) DO UPDATE SET
                        category_id = excluded.category_id,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    [(name, payload.category_id) for name in safe_filenames],
                )

            conn.commit()

        return {"success": True, "filenames": safe_filenames, "category_id": payload.category_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign markdown category: {e}")
        raise HTTPException(status_code=500, detail="Failed to assign markdown category")


@app.get("/api/markdown-files/with-categories", response_model=List[MarkdownFileWithCategory])
def list_markdown_files_with_categories(category_id: Optional[int] = None):
    """List markdown files together with optional category metadata.

    Data source for admin UIs that need both filename and category name.
    """
    try:
        # Base list of markdown files from storage
        md_files = cloud_storage_manager.list_markdown_files()

        # Prepare mapping from filename -> (category_id, category_name)
        category_map: Dict[str, Dict[str, Any]] = {}
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT mfc.filename, mfc.category_id, mc.name as category_name
                FROM markdown_file_categories mfc
                LEFT JOIN markdown_categories mc ON mc.id = mfc.category_id
                """
            )
            for row in cursor.fetchall():
                category_map[row["filename"]] = {
                    "category_id": row["category_id"],
                    "category_name": row["category_name"],
                }

        results: List[MarkdownFileWithCategory] = []
        for name in md_files:
            meta = category_map.get(name, {})
            item = MarkdownFileWithCategory(
                filename=name,
                category_id=meta.get("category_id"),
                category_name=meta.get("category_name"),
            )
            results.append(item)

        # Optional filtering by category_id
        if category_id is not None:
            results = [r for r in results if r.category_id == category_id]

        return results
    except Exception as e:
        logger.error(f"Failed to list markdown files with categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to list markdown files with categories")

@app.get("/api/documents/markdown/{filename}")
@app.get("/documents/markdown/{filename}")
def get_markdown_file_content(filename: str):
    """Get markdown file content - using cloud storage manager"""
    try:
        # Path traversal protection
        safe_filename = os.path.basename(filename).replace('..', '').replace('/', '').replace('\\', '')
        if not safe_filename.lower().endswith('.md'):
            safe_filename += '.md'
        
        content = cloud_storage_manager.get_markdown_file_content(safe_filename)
        if content is None:
            raise HTTPException(status_code=404, detail=f"Markdown file '{safe_filename}' not found")
        
        return {"content": content}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read markdown file: {str(e)}")

# Delete one markdown file
@app.delete("/api/documents/markdown/{filename}")
@app.delete("/documents/markdown/{filename}")
def delete_markdown_file(filename: str):
    try:
        safe_filename = os.path.basename(filename)
        ok = cloud_storage_manager.delete_markdown_file(safe_filename)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Markdown file '{safe_filename}' not found")
        return {"deleted": True, "filename": safe_filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete markdown file: {str(e)}")

# Delete many (bulk) or all markdown files
@app.delete("/api/documents/markdown")
@app.delete("/documents/markdown")
def delete_markdown_bulk(filenames: list[str] | None = Body(default=None), delete_all: bool = False):
    try:
        if delete_all:
            count = cloud_storage_manager.delete_all_markdown_files()
            return {"deleted": True, "count": count}
        if not filenames:
            return {"deleted": False, "count": 0}
        deleted = 0
        for name in filenames:
            safe = os.path.basename(name)
            if cloud_storage_manager.delete_markdown_file(safe):
                deleted += 1
        return {"deleted": True, "count": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete markdown files: {str(e)}")

# Session Export
@app.get("/sessions/{session_id}/export")
def export_session(session_id: str, format: str = "zip", request: Request = None):
    try:
        # Only owner or admin can export
        _require_owner_or_admin(request, session_id)
        
        # Get session metadata to include RAG settings
        metadata = professional_session_manager.get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Export base session data
        path = professional_session_manager.export_session(session_id, export_format=format)
        if not os.path.exists(path):
            raise HTTPException(status_code=500, detail="Export failed")
        
        # If ZIP format, add markdown files and RAG settings
        if format.lower() == "zip":
            import zipfile
            import tempfile
            import shutil
            
            # Create a temporary ZIP file
            temp_zip_path = path + ".tmp"
            shutil.copy(path, temp_zip_path)
            
            # Get markdown files associated with this session
            markdown_files = []
            try:
                # Get chunks from document-processing-service to find source files
                response = requests.get(
                    f"{DOCUMENT_PROCESSOR_URL}/sessions/{session_id}/chunks",
                    timeout=30
                )
                if response.status_code == 200:
                    chunks_data = response.json()
                    # Extract unique source_file names from metadata
                    source_files = set()
                    if isinstance(chunks_data, dict) and "chunks" in chunks_data:
                        for chunk in chunks_data["chunks"]:
                            metadata_item = chunk.get("metadata", {})
                            source_file = metadata_item.get("source_file") or metadata_item.get("filename")
                            if source_file:
                                source_files.add(source_file)
                    
                    # Read markdown files from cloud storage
                    for filename in source_files:
                        try:
                            content = cloud_storage_manager.get_markdown_file_content(filename)
                            if content:
                                markdown_files.append((filename, content))
                        except Exception as e:
                            logger.warning(f"Failed to read markdown file {filename}: {e}")
            except Exception as e:
                logger.warning(f"Failed to get chunks for session {session_id}: {e}")
            
            # Add markdown files and RAG settings to ZIP
            with zipfile.ZipFile(temp_zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                # Add markdown files
                for filename, content in markdown_files:
                    zipf.writestr(f"markdown_files/{filename}", content.encode('utf-8'))
                
                # Read existing session_data.json and update with RAG settings
                if "session_data.json" in zipf.namelist():
                    session_data = json.loads(zipf.read("session_data.json").decode('utf-8'))
                    # Ensure RAG settings are included
                    if metadata.rag_settings:
                        if "metadata" not in session_data:
                            session_data["metadata"] = {}
                        session_data["metadata"]["rag_settings"] = metadata.rag_settings
                    # Write updated session_data.json
                    zipf.writestr("session_data.json", 
                                json.dumps(session_data, indent=2, ensure_ascii=False).encode('utf-8'))
            
            # Replace original with updated ZIP
            shutil.move(temp_zip_path, path)
        
        filename = os.path.basename(path)
        media = "application/zip" if filename.endswith(".zip") else "application/json"
        return FileResponse(path, media_type=media, filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export session: {str(e)}")

# Session Import
@app.post("/sessions/import")
async def import_session(file: UploadFile = File(...), auto_reindex: bool = True, request: Request = None):
    try:
        current_user = _get_current_user(request)
        if not _is_teacher(current_user) and not _is_admin(current_user):
            raise HTTPException(status_code=403, detail="Only teachers/admins can import sessions")

        contents = await file.read()
        tmp_dir = Path("/tmp/rag3/import")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / file.filename
        with open(tmp_path, "wb") as f:
            f.write(contents)

        # Parse data
        session_meta = None
        rag_settings = None
        markdown_files: List[str] = []
        if str(tmp_path).endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(tmp_path, 'r') as zipf:
                # session_data.json
                if "session_data.json" in zipf.namelist():
                    import json as _json
                    data = _json.loads(zipf.read("session_data.json").decode("utf-8"))
                    session_meta = data.get("metadata") or {}
                    # Extract RAG settings from metadata
                    if session_meta and "rag_settings" in session_meta:
                        rag_settings = session_meta.get("rag_settings")
                
                # collect markdown files from markdown_files/ directory
                for name in zipf.namelist():
                    if name.startswith("markdown_files/") and name.lower().endswith('.md'):
                        content = zipf.read(name).decode('utf-8', errors='ignore')
                        safe_name = os.path.basename(name)
                        cloud_storage_manager.save_markdown_file(safe_name, content)
                        markdown_files.append(safe_name)
                    # Also check for .md files at root level (backward compatibility)
                    elif "/" not in name and name.lower().endswith('.md'):
                        content = zipf.read(name).decode('utf-8', errors='ignore')
                        safe_name = os.path.basename(name)
                        cloud_storage_manager.save_markdown_file(safe_name, content)
                        if safe_name not in markdown_files:
                            markdown_files.append(safe_name)
        else:
            # json only
            import json as _json
            session_meta = _json.loads(contents.decode("utf-8"))
            session_meta = session_meta.get("metadata", session_meta)

        if not session_meta:
            raise HTTPException(status_code=400, detail="Invalid export package: missing metadata")

        # Create new session
        try:
            cat = SessionCategory(session_meta.get("category", "general"))
        except Exception:
            cat = SessionCategory.GENERAL
        created = professional_session_manager.create_session(
            name=session_meta.get("name", "Imported Session"),
            description=session_meta.get("description", ""),
            category=cat,
            created_by=(current_user.get("id") if current_user and current_user.get("id") is not None else (current_user.get("username") if current_user else "import")),
            grade_level=session_meta.get("grade_level", ""),
            subject_area=session_meta.get("subject_area", ""),
            learning_objectives=session_meta.get("learning_objectives", []),
            tags=session_meta.get("tags", []),
            is_public=bool(session_meta.get("is_public", False)),
        )

        # Restore RAG settings if available
        if rag_settings:
            try:
                # Ensure rag_settings is a dict
                if isinstance(rag_settings, str):
                    rag_settings = json.loads(rag_settings)
                if isinstance(rag_settings, dict):
                    current_user_id = current_user.get("id") if current_user and current_user.get("id") else None
                    professional_session_manager.save_session_rag_settings(
                        created.session_id,
                        rag_settings,
                        user_id=current_user_id
                    )
            except Exception as e:
                logger.warning(f"Failed to restore RAG settings: {e}")
        
        # Reindex markdowns if any
        processed = 0
        if auto_reindex and markdown_files:
            # Use embedding model from RAG settings if available
            default_embedding = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
            embedding_model = default_embedding
            if rag_settings and isinstance(rag_settings, dict):
                embedding_model = rag_settings.get("embedding_model", default_embedding)
            
            # simpler: call our own function by HTTP form, mimic frontend
            try:
                form = {
                    "session_id": created.session_id,
                    "markdown_files": json.dumps(markdown_files),
                    "chunk_strategy": "lightweight",
                    "chunk_size": 1500,
                    "chunk_overlap": 150,
                    "embedding_model": embedding_model,
                }
                requests.post(f"{MAIN_API_URL}/documents/process-and-store", data=form, timeout=600)
            except Exception as e:
                logger.warning(f"Failed to reindex markdown files: {e}")

        return {"success": True, "new_session_id": created.session_id, "imported_markdowns": len(markdown_files)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import session: {str(e)}")

# Model endpoints - Route to Model Inference Service
@app.get("/models")
def get_models():
    """Get available models with provider categorization - Route to Model Inference Service"""
    try:
        # Try to get structured model list from Model Inference Service
        response = requests.get(f"{MODEL_INFERENCE_URL}/models/available", timeout=30)
        if response.status_code == 200:
            model_data = response.json()
            
            # Create a combined list with provider information for UI
            all_models = []
            
            # Add Groq models (cloud - fast)
            for model in model_data.get("groq", []):
                # Filter out models that are known to be problematic
                if model not in []:  # All new models should be working
                    all_models.append({
                        "id": model,
                        "name": model,
                        "provider": "groq",
                        "type": "cloud",
                        "description": "Groq (Hızlı)"
                    })
            
            # Add Ollama models (local)
            for model in model_data.get("ollama", []):
                all_models.append({
                    "id": model,
                    "name": model.replace(":latest", ""),  # Clean up model names
                    "provider": "ollama",
                    "type": "local",
                    "description": "Ollama (Yerel)"
                })
            
            # Add HuggingFace models (free)
            for model in model_data.get("huggingface", []):
                all_models.append({
                    "id": model,
                    "name": model.split("/")[-1] if "/" in model else model,  # Clean up model names
                    "provider": "huggingface",
                    "type": "cloud",
                    "description": "HuggingFace (Ücretsiz)"
                })
            
            # Add OpenRouter models (free)
            for model in model_data.get("openrouter", []):
                all_models.append({
                    "id": model,
                    "name": model.split("/")[-1].split(":")[0] if "/" in model else model.split(":")[0],  # Clean up model names
                    "provider": "openrouter",
                    "type": "cloud",
                    "description": "OpenRouter (Ücretsiz)"
                })
            
            # Add DeepSeek models
            for model in model_data.get("deepseek", []):
                all_models.append({
                    "id": model,
                    "name": model.replace("deepseek-", "DeepSeek ").title(),  # Clean up model names
                    "provider": "deepseek",
                    "type": "cloud",
                    "description": "DeepSeek (Premium)"
                })
            
            # Add Alibaba DashScope models
            for model in model_data.get("alibaba", []):
                all_models.append({
                    "id": model,
                    "name": model.replace("qwen-", "Qwen ").title(),  # Clean up model names
                    "provider": "alibaba",
                    "type": "cloud",
                    "description": "Alibaba DashScope (Qwen)"
                })
            
            return {
                "models": all_models,
                "providers": {
                    "groq": {
                        "name": "Groq",
                        "description": "Hızlı Cloud Modelleri",
                        "icon": "🚀",
                        "models": model_data.get("groq", [])
                    },
                    "huggingface": {
                        "name": "HuggingFace",
                        "description": "Ücretsiz Modeller",
                        "icon": "🤗",
                        "models": model_data.get("huggingface", [])
                    },
                    "ollama": {
                        "name": "Ollama",
                        "description": "Yerel Modeller",
                        "icon": "🏠",
                        "models": model_data.get("ollama", [])
                    },
                    "openrouter": {
                        "name": "OpenRouter",
                        "description": "Ücretsiz Modeller",
                        "icon": "🌐",
                        "models": model_data.get("openrouter", [])
                    },
                    "deepseek": {
                        "name": "DeepSeek",
                        "description": "Premium Modeller",
                        "icon": "🔮",
                        "models": model_data.get("deepseek", [])
                    },
                    "alibaba": {
                        "name": "Alibaba (Cloud - Qwen)",
                        "description": "Alibaba Cloud DashScope Qwen Modelleri",
                        "icon": "🛒",
                        "models": model_data.get("alibaba", [])
                    }
                }
            }
        else:
            # Fallback - try the debug endpoint for more info
            debug_response = requests.get(f"{MODEL_INFERENCE_URL}/debug/models", timeout=10)
            if debug_response.status_code == 200:
                debug_data = debug_response.json()
                logger.info(f"Model service debug info: {debug_data}")
            
            # Fallback to only confirmed working model
            return {
                "models": [
                    {
                        "id": "llama-3.1-8b-instant",
                        "name": "Llama 3.1 8B (Instant)",
                        "provider": "groq",
                        "type": "cloud",
                        "description": "Groq (Hızlı)"
                    }
                ],
                "providers": {
                    "groq": {
                        "name": "Groq",
                        "description": "Hızlı cloud modelleri",
                        "models": ["llama-3.1-8b-instant"]
                    }
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get models from Model Inference Service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")

@app.get("/models/embedding")
def get_embedding_models():
    """Get available embedding models from Ollama, HuggingFace, and Alibaba"""
    try:
        response = requests.get(f"{MODEL_INFERENCE_URL}/models/embedding", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch embedding models")
    except requests.RequestException as e:
        logger.error(f"Error fetching embedding models: {e}")
        raise HTTPException(status_code=503, detail=f"Model inference service unavailable: {str(e)}")

@app.post("/documents/convert-document-to-markdown")
async def convert_document_to_markdown(
    file: UploadFile = File(...),
    use_fallback: str = Form(default="false")
):
    """
    Convert PDF/DOCX/PPTX to Markdown using DocStrange service
    Supports two extraction methods:
    - Nanonets API (default): Good for scanned/complex documents
    - pdfplumber (fallback): Fast for simple text-based PDFs
    """
    try:
        logger.info(f"[DocConverter] Converting {file.filename}, use_fallback={use_fallback}")
        
        # Read file content
        file_content = await file.read()
        
        # Prepare form data for DocStrange
        files = {'file': (file.filename, file_content, file.content_type)}
        data = {'use_fallback': use_fallback}
        
        # Call DocStrange service
        response = requests.post(
            f"{PDF_PROCESSOR_URL}/convert/pdf-to-markdown",
            files=files,
            data=data,
            timeout=600  # 10 minutes for large files
        )
        
        if not response.ok:
            error_msg = response.text
            logger.error(f"[DocConverter] DocStrange error: {error_msg}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Document conversion failed: {error_msg}"
            )
        
        # Parse response
        result = response.json()
        logger.info(f"[DocConverter] DocStrange response: {result.keys()}")
        
        # Extract markdown content
        markdown_content = None
        if 'result' in result and isinstance(result['result'], list) and len(result['result']) > 0:
            markdown_content = result['result'][0].get('markdown', '')
        elif 'markdown' in result:
            markdown_content = result['markdown']
        elif 'content' in result:
            markdown_content = result['content']
        
        if not markdown_content or not markdown_content.strip():
            raise HTTPException(
                status_code=500,
                detail="Document processed but no content extracted. The file may be corrupted or contain only images."
            )
        
        # Save markdown file
        markdown_dir = Path("data/markdown")
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename
        base_filename = os.path.splitext(os.path.basename(file.filename))[0]
        safe_filename = base_filename.replace('..', '').replace('/', '').replace('\\', '') + '.md'
        
        # Handle duplicate filenames
        counter = 1
        final_filename = safe_filename
        while (markdown_dir / final_filename).exists():
            final_filename = f"{base_filename}_{counter}.md"
            counter += 1
        
        # Save file using cloud storage manager
        cloud_storage_manager.save_markdown_file(final_filename, markdown_content)
        
        logger.info(f"[DocConverter] ✅ Saved as {final_filename} ({len(markdown_content)} chars)")
        
        return {
            "success": True,
            "message": f"Document converted successfully",
            "markdown_filename": final_filename,
            "extraction_method": result.get("extraction_method", "unknown"),
            "content_preview": markdown_content[:200] + "..." if len(markdown_content) > 200 else markdown_content
        }
        
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Document conversion timeout. Please try with 'Fast' method or use smaller files."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DocConverter] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert document: {str(e)}")

@app.post("/api/documents/convert-marker")
@app.post("/documents/convert-marker")
async def convert_document_marker(file: UploadFile = File(...)):
    """
    Convert PDF/DOC/PPT to Markdown using Marker API
    Highest quality conversion with OCR and layout preservation
    Slower than other methods but best for complex documents
    """
    try:
        logger.info(f"[Marker] Converting {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Prepare multipart request for Marker API
        # Marker expects the field name 'pdf_file' (see marker-api docs)
        files = {
            'pdf_file': (file.filename, file_content, file.content_type or 'application/pdf')
        }
        
        # Determine strategy based on file size/pages
        file_size_mb = round(len(file_content) / (1024 * 1024), 2)
        total_pages = None
        if PdfReader is not None:
            try:
                total_pages = len(PdfReader(io.BytesIO(file_content)).pages)
            except Exception:
                total_pages = None
        logger.info(f"[Marker] Incoming file size ~{file_size_mb}MB, pages={total_pages}")

        def _marker_ready(max_wait_s: int = 15) -> bool:
            deadline = time.time() + max_wait_s
            while time.time() < deadline:
                try:
                    # try health endpoint first, then root
                    h = requests.get(f"{MARKER_API_URL}/health", timeout=2)
                    if h.ok:
                        return True
                except Exception:
                    pass
                try:
                    r = requests.get(f"{MARKER_API_URL}/", timeout=2)
                    if r.ok:
                        return True
                except Exception:
                    pass
                time.sleep(1)
            return False

        def call_marker(pdf_bytes: bytes) -> str:
            # Ensure service is up before calling
            if not _marker_ready(10):
                raise requests.exceptions.RequestException("Marker service not ready")
            local_files = {'pdf_file': (file.filename, pdf_bytes, file.content_type or 'application/pdf')}
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    resp = requests.post(f"{MARKER_API_URL}/convert", files=local_files, timeout=900)
                    if not resp.ok:
                        raise requests.exceptions.RequestException(resp.text)
                    data = resp.json()
                    md = None
                    if isinstance(data, dict):
                        md = data.get('markdown') or data.get('content') or data.get('text')
                    if not md or not str(md).strip():
                        raise requests.exceptions.RequestException("Marker returned empty content")
                    return str(md)
                except Exception as e:
                    last_err = e
                    # brief backoff then retry
                    time.sleep(1 + attempt)
            raise requests.exceptions.RequestException(f"Marker failed after retries: {last_err}")

        markdown_content = None
        # For medium/large PDFs (size>5MB or pages>10) do chunked conversion to avoid OOM
        if (file_size_mb and file_size_mb > 5) or (total_pages and total_pages > 10):
            if PdfReader is None or PdfWriter is None:
                logger.warning("[Marker] PyPDF2 not available; proceeding single-shot may OOM")
                markdown_content = call_marker(file_content)
            else:
                logger.info("[Marker] Chunked conversion enabled (safe small chunks)")
                chunk_size = 3  # smaller chunks to reduce memory footprint
                reader = PdfReader(io.BytesIO(file_content))
                parts: list[str] = []
                for start in range(0, len(reader.pages), chunk_size):
                    end = min(start + chunk_size, len(reader.pages))
                    writer = PdfWriter()
                    for i in range(start, end):
                        writer.add_page(reader.pages[i])
                    buf = io.BytesIO()
                    writer.write(buf)
                    buf.seek(0)
                    try:
                        part_md = call_marker(buf.read())
                    except Exception as _chunk_err:
                        logger.warning(f"[Marker] Chunk {start}-{end} failed, retrying 1-page splits: {_chunk_err}")
                        # Retry per-page if a chunk fails
                        for i in range(start, end):
                            writer_single = PdfWriter()
                            writer_single.add_page(reader.pages[i])
                            buf_single = io.BytesIO()
                            writer_single.write(buf_single)
                            buf_single.seek(0)
                            single_md = call_marker(buf_single.read())
                            parts.append(single_md.strip())
                        continue
                    parts.append(part_md.strip())
                markdown_content = "\n\n".join(parts)
        else:
            # Single-shot for small/medium files
            markdown_content = call_marker(file_content)

        # Guard: do not save empty content
        if not markdown_content or not str(markdown_content).strip():
            raise HTTPException(status_code=502, detail="Marker returned empty content")

        # Create markdown directory if needed
        markdown_dir = Path("data/markdown")
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename
        base_filename = os.path.splitext(os.path.basename(file.filename))[0]
        safe_filename = base_filename.replace('..', '').replace('/', '').replace('\\', '') + '.md'
        
        # Handle duplicate filenames
        counter = 1
        final_filename = safe_filename
        while (markdown_dir / final_filename).exists():
            final_filename = f"{base_filename}_{counter}.md"
            counter += 1
        
        # Save file using cloud storage manager
        cloud_storage_manager.save_markdown_file(final_filename, markdown_content)
        
        logger.info(f"[Marker] ✅ Saved as {final_filename} ({len(markdown_content)} chars)")
        
        return {
            "success": True,
            "message": "Document converted successfully with Marker",
            "markdown_filename": final_filename,
            "extraction_method": "marker",
            "content_preview": markdown_content[:200] + "..." if len(markdown_content) > 200 else markdown_content
        }
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Marker conversion timeout (15 min). Document may be too complex. Try 'Fast' method."
        )
    except Exception as e:
        # Fallback to DocStrange (pdfplumber) on any Marker failure (OOM/connection closed/etc.)
        try:
            logger.warning(f"[Marker] Failed ({e}). Falling back to pdfplumber via DocStrange.")
            # Reuse the already read file_content
            files_fallback = {'file': (file.filename, file_content, file.content_type or 'application/pdf')}
            # Fallback to Nanonets (not pdfplumber) as requested for teacher flow
            data_fallback = {'use_fallback': 'false'}
            resp_fb = requests.post(
                f"{PDF_PROCESSOR_URL}/convert/pdf-to-markdown",
                files=files_fallback,
                data=data_fallback,
                timeout=600
            )
            if not resp_fb.ok:
                raise HTTPException(status_code=resp_fb.status_code, detail=f"Fallback conversion failed: {resp_fb.text}")
            fb = resp_fb.json()
            # Extract markdown
            markdown_content = None
            if 'result' in fb and isinstance(fb['result'], list) and fb['result']:
                markdown_content = fb['result'][0].get('markdown', '')
            elif 'markdown' in fb:
                markdown_content = fb['markdown']
            elif 'content' in fb:
                markdown_content = fb['content']
            if not markdown_content or not str(markdown_content).strip():
                raise HTTPException(status_code=502, detail="Fallback returned empty content")
            # Save
            markdown_dir = Path("data/markdown")
            markdown_dir.mkdir(parents=True, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(file.filename))[0]
            safe_filename = base_filename.replace('..', '').replace('/', '').replace('\\', '') + '.md'
            counter = 1
            final_filename = safe_filename
            while (markdown_dir / final_filename).exists():
                final_filename = f"{base_filename}_{counter}.md"
                counter += 1
            cloud_storage_manager.save_markdown_file(final_filename, markdown_content)
            logger.info(f"[Marker→Fallback] ✅ Saved as {final_filename} ({len(markdown_content)} chars)")
            return {
                "success": True,
                "message": "Marker failed; used fast fallback (pdfplumber) successfully",
                "markdown_filename": final_filename,
                "extraction_method": "pdfplumber_fallback_marker_error",
                "content_preview": markdown_content[:200] + "..." if len(markdown_content) > 200 else markdown_content
            }
        except HTTPException:
            raise
        except Exception as fb_err:
            logger.error(f"[Marker] Fallback also failed: {fb_err}")
            raise HTTPException(status_code=500, detail=f"Failed to convert document with Marker: {str(e)}; Fallback error: {str(fb_err)}")

@app.post("/api/documents/upload-markdown")
@app.post("/documents/upload-markdown")
async def upload_markdown_file(file: UploadFile = File(...)):
    """Upload markdown file directly - no conversion needed"""
    try:
        # Validate file extension
        if not file.filename or not file.filename.lower().endswith('.md'):
            raise HTTPException(
                status_code=400,
                detail="Only .md (Markdown) files are allowed"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate it's text content
        try:
            content_str = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be valid UTF-8 encoded text"
            )
        
        # Create markdown directory if it doesn't exist
        markdown_dir = Path("data/markdown")
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename
        safe_filename = os.path.basename(file.filename).replace('..', '').replace('/', '').replace('\\', '')
        if not safe_filename.lower().endswith('.md'):
            safe_filename += '.md'
        
        # Handle duplicate filenames
        base_name = safe_filename[:-3]  # Remove .md extension
        counter = 1
        final_filename = safe_filename
        
        while (markdown_dir / final_filename).exists():
            final_filename = f"{base_name}_{counter}.md"
            counter += 1
        
        # Save file
        file_path = markdown_dir / final_filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_str)
        
        return {
            "success": True,
            "message": f"Markdown file uploaded successfully",
            "markdown_filename": final_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload markdown file: {str(e)}")

@app.get("/test")
def test_endpoint():
    """Simple test endpoint"""
    return {"status": "success", "message": "API Gateway is working"}


# --- Auth Service Proxy ---
async def _proxy_request(request: Request, target_url: str):
    """Generic proxy for forwarding requests to a target service."""
    client = httpx.AsyncClient()
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    
    # Prepare request data
    headers = dict(request.headers)
    # httpx uses 'host' header from the target_url, so we can remove it
    headers.pop("host", None)
    
    content = await request.body()
    
    try:
        # Forward the request
        rp = await client.request(
            method=request.method,
            url=f"{target_url}{url.path}",
            headers=headers,
            params=request.query_params,
            content=content,
            timeout=60.0,
        )
        
        # Return the response from the target service
        return Response(
            content=rp.content,
            status_code=rp.status_code,
            headers=dict(rp.headers),
        )
    except httpx.RequestError as e:
        logger.error(f"Proxy request to {target_url} failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Service unavailable: {e}"
        )
    finally:
        await client.aclose()

@app.api_route("/auth/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_auth(request: Request):
    """Proxy for all /auth routes to the Auth Service."""
    return await _proxy_request(request, AUTH_SERVICE_URL)

@app.api_route("/users/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_users(request: Request):
    """Proxy for all /users routes to the Auth Service."""
    return await _proxy_request(request, AUTH_SERVICE_URL)

@app.api_route("/roles/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_roles(request: Request):
    """Proxy for all /roles routes to the Auth Service."""
    return await _proxy_request(request, AUTH_SERVICE_URL)

@app.api_route("/aprag/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_aprag_no_prefix(request: Request, path: str):
    """Proxy for /aprag routes (when /api prefix is stripped by Next.js rewrites) to the APRAG Service."""
    # APRAG service expects /api/aprag/... paths, so we add the /api prefix
    target_path = f"/api/aprag/{path}"
    logger.info(f"Proxying {request.method} request to APRAG service (no /api prefix): {APRAG_SERVICE_URL}{target_path}")
    
    # Create a modified request with the correct path
    client = httpx.AsyncClient()
    headers = dict(request.headers)
    headers.pop("host", None)
    content = await request.body()
    
    try:
        rp = await client.request(
            method=request.method,
            url=f"{APRAG_SERVICE_URL}{target_path}",
            headers=headers,
            params=request.query_params,
            content=content,
            timeout=60.0,
        )
        return Response(
            content=rp.content,
            status_code=rp.status_code,
            headers=dict(rp.headers),
        )
    except httpx.RequestError as e:
        logger.error(f"Proxy request to {APRAG_SERVICE_URL}{target_path} failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Service unavailable: {e}"
        )
    finally:
        await client.aclose()

# Analytics: recent interactions for teachers
@app.get("/analytics/recent-interactions")
def recent_interactions(limit: int = 20, page: int = 1, session_id: Optional[str] = None, q: Optional[str] = None):
    """Return recent student interactions for teacher dashboards with pagination and search."""
    try:
        from src.analytics.database import ExperimentDatabase
        db = ExperimentDatabase()
        offset = max(0, (page - 1)) * max(1, limit)
        with db.get_connection() as conn:
            cursor = conn.cursor()
            base_query = (
                """
                SELECT i.interaction_id, i.user_id, i.session_id, i.timestamp, i.query, i.response,
                       i.processing_time_ms, i.success, i.error_message, i.chain_type, rc.rag_params
                FROM interactions i
                LEFT JOIN rag_configurations rc ON rc.config_hash = i.rag_config_hash
                {where}
                ORDER BY i.timestamp DESC
                LIMIT ? OFFSET ?
                """
            )
            where_clauses = []
            params: list[Any] = []
            if session_id:
                where_clauses.append("i.session_id = ?")
                params.append(session_id)
            if q:
                where_clauses.append("(i.query LIKE ? OR i.response LIKE ?)")
                params.extend([f"%{q}%", f"%{q}%"]) 
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            cursor.execute(base_query.format(where=where_sql), (*params, limit, offset))
            rows = cursor.fetchall()

            # total count
            count_query = f"SELECT COUNT(*) as cnt FROM interactions i {' ' + where_sql if where_sql else ''}"
            cursor.execute(count_query, tuple(params))
            count_row = cursor.fetchone()
            total = count_row["cnt"] if count_row else 0

            items = []
            for row in rows:
                # Parse rag_params JSON to extract model and top_k
                model = None
                top_k = None
                chain_type = None
                try:
                    chain_type = row["chain_type"]
                except Exception:
                    chain_type = None
                try:
                    if row["rag_params"]:
                        rp = json.loads(row["rag_params"]) if isinstance(row["rag_params"], str) else row["rag_params"]
                        model = rp.get("model")
                        top_k = rp.get("top_k")
                        chain_type = chain_type or rp.get("chain_type")
                except Exception:
                    pass
                try:
                    success_val = row["success"] if "success" in row.keys() else None
                except Exception:
                    success_val = None
                try:
                    error_val = row["error_message"] if "error_message" in row.keys() else None
                except Exception:
                    error_val = None
                items.append({
                    "interaction_id": row["interaction_id"],
                    "user_id": row["user_id"],
                    "session_id": row["session_id"],
                    "timestamp": row["timestamp"],
                    "query": row["query"],
                    "response": row["response"],
                    "processing_time_ms": row["processing_time_ms"],
                    "model": model,
                    "top_k": top_k,
                    "success": success_val,
                    "error_message": error_val,
                    "chain_type": chain_type,
                })
            return {"items": items, "count": total, "page": page, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recent interactions: {str(e)}")

@app.delete("/analytics/recent-interactions")
def delete_recent_interactions(session_id: Optional[str] = None, request: Request = None):
    try:
        current_user = _get_current_user(request)
        if not (_is_teacher(current_user) or _is_admin(current_user)):
            raise HTTPException(status_code=403, detail="Not authorized")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        from src.analytics.database import ExperimentDatabase
        db = ExperimentDatabase()
        deleted = db.delete_interactions_by_session(session_id)
        return {"success": True, "deleted": deleted, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete interactions: {str(e)}")


# ===== MODEL ENDPOINTS =====

@app.get("/api/models")
def get_available_models(request: Request):
    """
    Get available LLM models - uses the same logic as /models endpoint
    This maintains backward compatibility for frontend
    """
    # Use the same function as /models endpoint which has provider categorization
    return get_models()

@app.get("/api/models/embedding")
def get_available_embedding_models(request: Request):
    """
    Get available embedding models - uses the same logic as /models/embedding endpoint
    This maintains backward compatibility for frontend
    """
    # Use the same function as /models/embedding endpoint
    return get_embedding_models()

# ==================== API PREFIX ENDPOINTS (for Next.js rewrites) ====================

@app.get("/api/sessions", response_model=List[SessionResponse])
def api_list_sessions(created_by: Optional[str] = None, category: Optional[str] = None,
                     status: Optional[str] = None, limit: int = 50, request: Request = None):
    """List sessions from SQLite database - API prefix version"""
    return list_sessions(created_by, category, status, limit, request)

@app.post("/api/sessions", response_model=SessionResponse)
def api_create_session(req: CreateSessionRequest, request: Request):
    """Create new session in SQLite database - API prefix version"""
    return create_session(req, request)

@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
def api_get_session(session_id: str, request: Request):
    """Get session details from SQLite database - API prefix version"""
    return get_session(session_id, request)

@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str, create_backup: bool = True, deleted_by: Optional[str] = None, request: Request = None):
    """Delete session from SQLite database - API prefix version"""
    return delete_session(session_id, create_backup, deleted_by, request)

@app.patch("/api/sessions/{session_id}/status")
def api_update_session_status(session_id: str, request: StatusUpdateRequest, req: Request):
    """Update session status (active/inactive) - API prefix version"""
    return update_session_status(session_id, request, req)

@app.get("/api/sessions/{session_id}/chunks")
def api_get_session_chunks(session_id: str):
    """Get chunks for a session from Document Processing Service - API prefix version"""
    return get_session_chunks(session_id)

@app.get("/api/sessions/{session_id}/stats")
def get_session_stats(session_id: str):
    """Get lightweight session statistics without loading full chunks data"""
    try:
        logger.info(f"🔍 [STATS API] Getting lightweight stats for session {session_id}")
        
        # Get session metadata first for basic counts
        session_metadata = professional_session_manager.get_session_metadata(session_id)
        if not session_metadata:
            logger.warning(f"⚠️ [STATS API] Session {session_id} not found in database")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_characters": 0,
                "llm_improved": 0,
                "session_id": session_id,
                "source": "empty_session"
            }
        
        # Try to get basic stats from document processor without full chunk data
        try:
            # Use a lightweight stats endpoint if available
            stats_response = requests.get(
                f"{DOCUMENT_PROCESSOR_URL}/sessions/{session_id}/stats",
                timeout=10
            )
            
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                logger.info(f"✅ [STATS API] Got stats from document processor: {stats_data}")
                return {
                    "total_documents": stats_data.get("document_count", session_metadata.document_count),
                    "total_chunks": stats_data.get("chunk_count", session_metadata.total_chunks),
                    "total_characters": stats_data.get("total_characters", 0),
                    "llm_improved": stats_data.get("llm_improved_count", 0),
                    "session_id": session_id,
                    "source": "document_processor_stats"
                }
            else:
                logger.info(f"📊 [STATS API] Document processor stats endpoint returned {stats_response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.info(f"📊 [STATS API] Document processor unavailable for stats: {e}")
        
        # Fallback: Use session metadata counts
        logger.info(f"📊 [STATS API] Using session metadata fallback for {session_id}")
        return {
            "total_documents": session_metadata.document_count,
            "total_chunks": session_metadata.total_chunks,
            "total_characters": 0,  # Not available in metadata, will show 0 until chunks loaded
            "llm_improved": 0,      # Not available in metadata, will show 0 until chunks loaded
            "session_id": session_id,
            "source": "session_metadata"
        }
        
    except Exception as e:
        logger.error(f"❌ [STATS API] Error getting session stats: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_characters": 0,
            "llm_improved": 0,
            "session_id": session_id,
            "source": "error"
        }

@app.post("/api/sessions/{session_id}/generate-questions")
async def api_generate_course_questions(session_id: str, request: Request, req: GenerateQuestionsRequest):
    """Generate course-specific questions - API prefix version"""
    return await generate_course_questions(session_id, request, req)

@app.post("/api/sessions/{session_id}/reprocess")
async def api_reprocess_session_documents(session_id: str, request: Request):
    """Re-process existing documents in a session - API prefix version"""
    return await reprocess_session_documents(session_id, request)

@app.get("/api/sessions/{session_id}/rag-settings")
def api_get_rag_settings(session_id: str, request: Request):
    """Get RAG settings - API prefix version"""
    return get_rag_settings(session_id, request)

@app.patch("/api/sessions/{session_id}/rag-settings")
def api_update_rag_settings(session_id: str, req: RAGSettings, request: Request):
    """Update RAG settings - API prefix version"""
    return update_rag_settings(session_id, req, request)

@app.get("/api/sessions/{session_id}/export")
def api_export_session(session_id: str, format: str = "zip", request: Request = None):
    """Export session - API prefix version"""
    return export_session(session_id, format, request)

@app.post("/api/sessions/import")
async def api_import_session(file: UploadFile = File(...), auto_reindex: bool = True, request: Request = None):
    """Import session - API prefix version"""
    return await import_session(file, auto_reindex, request)

@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def api_rag_query(req: RAGQueryRequest, request: Request):
    """RAG Query with CRAG evaluation - API prefix version"""
    return await rag_query(req, request)

@app.post("/api/rag/suggestions")
async def api_generate_suggestions(req: SuggestionRequest) -> Dict[str, Any]:
    """Generate follow-up question suggestions - API prefix version"""
    return await generate_suggestions(req)

# Include RAG Tests Router
from src.api.rag_tests_routes import router as rag_tests_router
app.include_router(rag_tests_router)
logger.info("✅ RAG Tests routes registered")

# APRAG Service Proxy Endpoints

@app.post("/api/aprag/interactions")
async def create_aprag_interaction_proxy(request: Request):
    """Proxy to APRAG service for creating interactions"""
    try:
        body = await request.json()
        logger.info(f"🎓 Creating APRAG interaction for user {body.get('user_id', 'unknown')}")
        
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/interactions",
            json=body,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            # Log error but don't fail - chat should continue
            logger.warning(f"APRAG service error: {response.status_code} - {response.text}")
            # Return a fallback response so chat can continue
            return {"interaction_id": -1, "message": "APRAG service error, interaction not logged"}
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        # Return a fallback response so chat can continue
        return {"interaction_id": -1, "message": "APRAG service unavailable, interaction not logged"}
    except Exception as e:
        logger.warning(f"Error in APRAG interaction proxy: {e}")
        # Return a fallback response so chat can continue
        return {"interaction_id": -1, "message": "Error logging interaction"}

@app.post("/api/aprag/adaptive-query")
async def aprag_adaptive_query_proxy(request: Request):
    """Proxy to APRAG service for adaptive query with personalization"""
    try:
        body = await request.json()
        logger.info(f"🎯 APRAG adaptive query for user {body.get('user_id', 'unknown')}")
        
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/adaptive-query",
            json=body,
            timeout=60  # Adaptive query can take longer
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            # Log error but return original response as fallback
            logger.warning(f"APRAG adaptive query error: {response.status_code} - {response.text}")
            # Return the original RAG response as fallback
            return {
                "personalized_response": body.get("rag_response", ""),
                "interaction_id": -1,
                "pedagogical_context": {
                    "zpd_recommended": "medium",
                    "bloom_level": "understand",
                    "cognitive_load": "moderate"
                },
                "cacs_applied": False
            }
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable for adaptive query: {e}")
        # Return original RAG response as fallback
        return {
            "personalized_response": body.get("rag_response", ""),
            "interaction_id": -1,
            "pedagogical_context": {
                "zpd_recommended": "medium",
                "bloom_level": "understand",
                "cognitive_load": "moderate"
            },
            "cacs_applied": False
        }
    except Exception as e:
        logger.warning(f"Error in APRAG adaptive query proxy: {e}")
        # Return generic fallback
        return {
            "personalized_response": "Cevap hazırlandı.",
            "interaction_id": -1,
            "pedagogical_context": {
                "zpd_recommended": "medium",
                "bloom_level": "understand",
                "cognitive_load": "moderate"
            },
            "cacs_applied": False
        }

@app.get("/api/aprag/profiles/{user_id}/{session_id}")
async def get_student_profile_proxy(user_id: str, session_id: str):
    """Proxy to APRAG service for getting student profile"""
    try:
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/profiles/{user_id}/{session_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"APRAG profile error: {response.status_code} - {response.text}")
            # Return default profile
            return {
                "user_id": user_id,
                "session_id": session_id,
                "average_understanding": None,
                "average_satisfaction": None,
                "total_interactions": 0,
                "total_feedback_count": 0,
                "strong_topics": None,
                "weak_topics": None,
                "preferred_explanation_style": None,
                "preferred_difficulty_level": None
            }
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable for profile: {e}")
        # Return default profile
        return {
            "user_id": user_id,
            "session_id": session_id,
            "average_understanding": None,
            "average_satisfaction": None,
            "total_interactions": 0,
            "total_feedback_count": 0,
            "strong_topics": None,
            "weak_topics": None,
            "preferred_explanation_style": None,
            "preferred_difficulty_level": None
        }
    except Exception as e:
        logger.warning(f"Error in APRAG profile proxy: {e}")
        # Return default profile
        return {
            "user_id": user_id,
            "session_id": session_id,
            "average_understanding": None,
            "average_satisfaction": None,
            "total_interactions": 0,
            "total_feedback_count": 0,
            "strong_topics": None,
            "weak_topics": None,
            "preferred_explanation_style": None,
            "preferred_difficulty_level": None
        }

@app.get("/api/aprag/interactions/session/{session_id}")
async def get_session_interactions_proxy(session_id: str, request: Request, limit: int = 50, offset: int = 0):
    """Proxy to APRAG service for getting session interactions"""
    try:
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/interactions/session/{session_id}",
            params={"limit": limit, "offset": offset},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"interactions": [], "total": 0, "count": 0, "limit": limit, "offset": offset}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        # Return empty result if service unavailable
        return {"interactions": [], "total": 0, "count": 0, "limit": limit, "offset": offset}

@app.get("/api/aprag/interactions/{user_id}")
async def get_user_interactions_proxy(user_id: str, request: Request, session_id: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Proxy to APRAG service for getting user interactions"""
    try:
        params = {"limit": limit, "offset": offset}
        if session_id:
            params["session_id"] = session_id
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/interactions/{user_id}",
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")

# APRAG Feedback Proxy Endpoints
@app.post("/api/aprag/feedback")
async def create_feedback_proxy(request: Request):
    """Proxy to APRAG service for creating feedback"""
    try:
        body = await request.json()
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/feedback",
            json=body,
            timeout=10
        )
        if response.status_code == 201:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")

@app.get("/api/aprag/feedback/session/{session_id}")
async def get_session_feedback_proxy(session_id: str, request: Request, limit: int = 50, offset: int = 0):
    """Proxy to APRAG service for getting session feedback"""
    try:
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/feedback/session/{session_id}",
            params={"limit": limit, "offset": offset},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"feedback": [], "total": 0, "count": 0, "limit": limit, "offset": offset}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        return {"feedback": [], "total": 0, "count": 0, "limit": limit, "offset": offset}

# APRAG Personalization Proxy Endpoint
@app.post("/api/aprag/personalize")
async def personalize_response_proxy(request: Request):
    """Proxy to APRAG service for personalizing responses"""
    try:
        body = await request.json()
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/personalize",
            json=body,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")

# APRAG Recommendations Proxy Endpoints
@app.get("/api/aprag/recommendations/{user_id}")
async def get_recommendations_proxy(
    user_id: str,
    request: Request,
    session_id: Optional[str] = None,
    limit: int = 10
):
    """Proxy to APRAG service for getting recommendations"""
    try:
        params = {"limit": limit}
        if session_id:
            params["session_id"] = session_id
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/recommendations/{user_id}",
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        return {"recommendations": [], "total": 0}

@app.post("/api/aprag/recommendations/{recommendation_id}/accept")
async def accept_recommendation_proxy(recommendation_id: int, request: Request):
    """Proxy to APRAG service for accepting recommendations"""
    try:
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/recommendations/{recommendation_id}/accept",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")

@app.post("/api/aprag/recommendations/{recommendation_id}/dismiss")
async def dismiss_recommendation_proxy(recommendation_id: int, request: Request):
    """Proxy to APRAG service for dismissing recommendations"""
    try:
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/recommendations/{recommendation_id}/dismiss",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")

# APRAG Analytics Proxy Endpoints
@app.get("/api/aprag/analytics/{user_id}")
async def get_analytics_proxy(
    user_id: str,
    request: Request,
    session_id: Optional[str] = None
):
    """Proxy to APRAG service for getting analytics"""
    try:
        params = {}
        if session_id:
            params["session_id"] = session_id
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/analytics/{user_id}",
            params=params,
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        return {
            "total_interactions": 0,
            "total_feedback": 0,
            "average_understanding": None,
            "average_satisfaction": None,
            "improvement_trend": "insufficient_data",
            "learning_patterns": [],
            "topic_performance": {},
            "engagement_metrics": {},
            "time_analysis": {}
        }

@app.get("/api/aprag/analytics/{user_id}/summary")
async def get_analytics_summary_proxy(
    user_id: str,
    request: Request,
    session_id: Optional[str] = None
):
    """Proxy to APRAG service for getting analytics summary"""
    try:
        params = {}
        if session_id:
            params["session_id"] = session_id
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/analytics/{user_id}/summary",
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        return {
            "total_interactions": 0,
            "average_understanding": None,
            "improvement_trend": "insufficient_data",
            "engagement_level": "low",
            "key_patterns": []
        }

# APRAG Settings Proxy Endpoints
@app.get("/api/aprag/settings/status")
async def get_aprag_settings_status_proxy(request: Request, session_id: Optional[str] = None):
    """Proxy to APRAG service for getting settings status"""
    try:
        params = {}
        if session_id:
            params["session_id"] = session_id
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/settings/status",
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        return {
            "enabled": False,
            "global_enabled": False,
            "session_enabled": None,
            "features": {
                "feedback_collection": False,
                "personalization": False,
                "recommendations": False,
                "analytics": False,
            }
        }

@app.post("/api/aprag/settings/toggle")
async def toggle_aprag_setting_proxy(request: Request):
    """Proxy to APRAG service for toggling settings"""
    try:
        body = await request.json()
        logger.info(f"[APRAG PROXY] toggle_aprag_setting_proxy called")
        logger.info(f"[APRAG PROXY] Target URL: {APRAG_SERVICE_URL}/api/aprag/settings/toggle")
        logger.info(f"[APRAG PROXY] Request body: {body}")
        
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/settings/toggle",
            json=body,
            timeout=10
        )
        
        logger.info(f"[APRAG PROXY] Response status: {response.status_code}")
        logger.info(f"[APRAG PROXY] Response headers: {dict(response.headers)}")
        logger.info(f"[APRAG PROXY] Response text (first 500 chars): {response.text[:500]}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"[APRAG PROXY] APRAG service returned error {response.status_code}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"[APRAG PROXY] Request to APRAG service failed: {e}")
        logger.error(f"[APRAG PROXY] APRAG_SERVICE_URL: {APRAG_SERVICE_URL}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")
    except Exception as e:
        logger.error(f"[APRAG PROXY] Unexpected error in proxy: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

# APRAG Topics Proxy Endpoints
@app.post("/api/aprag/topics/extract")
async def extract_topics_proxy(request: Request):
    """Proxy to APRAG service for topic extraction"""
    try:
        body = await request.json()
        
        # Forward directly to APRAG service (let it handle availability checks)
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/topics/extract",
            json=body,
            timeout=120  # Topic extraction can take time
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")

@app.get("/api/aprag/topics/session/{session_id}")
async def get_session_topics_proxy(session_id: str, request: Request):
    """Proxy to APRAG service for getting session topics"""
    try:
        # Forward directly to APRAG service (let it handle availability checks)
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/topics/session/{session_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        return {"success": False, "topics": [], "total": 0}

@app.put("/api/aprag/topics/{topic_id}")
async def update_topic_proxy(topic_id: int, request: Request):
    """Proxy to APRAG service for updating a topic"""
    try:
        # Check APRAG status - need to get session_id from topic first
        # But we'll let the APRAG service handle the check since it has the topic
        body = await request.json()
        response = requests.put(
            f"{APRAG_SERVICE_URL}/api/aprag/topics/{topic_id}",
            json=body,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")

@app.post("/api/aprag/topics/classify-question")
async def classify_question_proxy(request: Request):
    """Proxy to APRAG service for question classification"""
    logger.info(f"✅ [ROUTE MATCH] classify-question endpoint matched")
    try:
        body = await request.json()
        logger.info(f"📤 Proxying classify-question request to APRAG service: {APRAG_SERVICE_URL}/api/aprag/topics/classify-question")
        
        # Forward directly to APRAG service (let it handle availability checks)
        response = requests.post(
            f"{APRAG_SERVICE_URL}/api/aprag/topics/classify-question",
            json=body,
            timeout=60  # LLM classification can take time
        )
        logger.info(f"📥 APRAG service response: status={response.status_code}")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"❌ APRAG service returned {response.status_code}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        raise HTTPException(status_code=503, detail="APRAG service unavailable")

@app.get("/api/aprag/topics/progress/{user_id}/{session_id}")
async def get_student_progress_proxy(user_id: str, session_id: str, request: Request):
    """Proxy to APRAG service for getting student progress"""
    try:
        # Forward directly to APRAG service (let it handle availability checks)
        response = requests.get(
            f"{APRAG_SERVICE_URL}/api/aprag/topics/progress/{user_id}/{session_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.exceptions.RequestException as e:
        logger.warning(f"APRAG service unavailable: {e}")
        return {
            "success": False,
            "progress": [],
            "current_topic": None,
            "next_recommended_topic": None
        }


@app.post("/chunks/improve-single")
async def improve_single_chunk_proxy(request: Request):
    """Proxy to Document Processing Service for single chunk LLM improvement"""
    try:
        body = await request.json()
        
        logger.info(f"🤖 Proxying single chunk improvement request to document processing service")
        
        # Forward to document processing service
        response = requests.post(
            f"{DOCUMENT_PROCESSOR_URL}/chunks/improve-single",
            json=body,
            timeout=60  # LLM processing can take time
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Document processing service error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Document processing service unavailable: {e}")
        raise HTTPException(status_code=503, detail="Document processing service unavailable")


@app.post("/sessions/{session_id}/chunks/improve-all")
async def improve_all_chunks_proxy(session_id: str, request: Request):
    """Proxy to Document Processing Service for bulk chunk LLM improvement"""
    try:
        body = await request.json()
        
        logger.info(f"🚀 Proxying bulk chunk improvement request for session {session_id}")
        
        # Forward to document processing service (with longer timeout for bulk processing)
        response = requests.post(
            f"{DOCUMENT_PROCESSOR_URL}/sessions/{session_id}/chunks/improve-all",
            json=body,
            timeout=600  # 10 minutes for bulk processing
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Document processing service error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Document processing service unavailable: {e}")
        raise HTTPException(status_code=503, detail="Document processing service unavailable")


# ==================== Student Chat History Endpoints ====================

class StudentChatMessageCreate(BaseModel):
    user: str
    bot: str
    sources: Optional[List[Dict[str, Any]]] = None
    durationMs: Optional[int] = None
    session_id: str
    suggestions: Optional[List[str]] = None
    aprag_interaction_id: Optional[int] = None

class StudentChatMessageResponse(BaseModel):
    id: int
    user: str
    bot: str
    sources: Optional[List[Dict[str, Any]]] = None
    durationMs: Optional[int] = None
    session_id: str
    timestamp: str
    suggestions: Optional[List[str]] = None
    aprag_interaction_id: Optional[int] = None

import sqlite3

def get_student_db_connection():
    """Get database connection for student chat history"""
    db_path = os.getenv("DATABASE_PATH", "/app/data/rag_assistant.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_student_chat_table():
    """Initialize student chat history table if it doesn't exist"""
    try:
        conn = get_student_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                bot_message TEXT NOT NULL,
                sources TEXT,
                duration_ms INTEGER,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                suggestions TEXT,
                aprag_interaction_id INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_student_chat_session 
            ON student_chat_history(session_id)
        """)
        conn.commit()
        conn.close()
        logger.info("✅ Student chat history table initialized")
    except Exception as e:
        logger.error(f"Failed to initialize student chat table: {e}")

# Initialize table on startup
init_student_chat_table()

@app.get("/api/students/chat-history/{session_id}", response_model=List[StudentChatMessageResponse])
async def get_student_chat_history(session_id: str, request: Request):
    """Get chat history for a specific session"""
    try:
        conn = get_student_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, user_message as user, bot_message as bot, sources, duration_ms as durationMs,
                   session_id, timestamp, suggestions, aprag_interaction_id
            FROM student_chat_history
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            message = dict(row)
            # Parse JSON fields
            if message.get('sources'):
                try:
                    message['sources'] = json.loads(message['sources'])
                except:
                    message['sources'] = None
            if message.get('suggestions'):
                try:
                    message['suggestions'] = json.loads(message['suggestions'])
                except:
                    message['suggestions'] = None
            history.append(message)
        
        return history
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/students/chat-message", response_model=StudentChatMessageResponse)
async def save_student_chat_message(message: StudentChatMessageCreate, request: Request):
    """Save a student chat message"""
    try:
        conn = get_student_db_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.utcnow().isoformat() + "Z"
        sources_json = json.dumps(message.sources) if message.sources else None
        suggestions_json = json.dumps(message.suggestions) if message.suggestions else None
        
        cursor.execute("""
            INSERT INTO student_chat_history 
            (user_message, bot_message, sources, duration_ms, session_id, timestamp, suggestions, aprag_interaction_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.user,
            message.bot,
            sources_json,
            message.durationMs,
            message.session_id,
            timestamp,
            suggestions_json,
            message.aprag_interaction_id
        ))
        
        message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Return the saved message
        response = {
            "id": message_id,
            "user": message.user,
            "bot": message.bot,
            "sources": message.sources,
            "durationMs": message.durationMs,
            "session_id": message.session_id,
            "timestamp": timestamp,
            "suggestions": message.suggestions,
            "aprag_interaction_id": message.aprag_interaction_id
        }
        
        logger.info(f"✅ Saved chat message for session {message.session_id}")
        return response
    except Exception as e:
        logger.error(f"Failed to save chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/students/chat-history/{session_id}")
async def clear_student_chat_history(session_id: str, request: Request):
    """Clear chat history for a specific session"""
    try:
        conn = get_student_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM student_chat_history WHERE session_id = ?", (session_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Cleared {deleted_count} messages for session {session_id}")
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Backup and Restore Routes ====================

from src.api.backup_restore import router as backup_restore_router
app.include_router(backup_restore_router)

# ==================== Auth Service Proxy Routes ====================

@app.api_route("/api/auth/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_auth_api(request: Request, path: str):
    """Proxy for all /api/auth routes to the Auth Service."""
    # Auth service expects /auth/... paths, so we need to strip /api prefix
    # Create a modified request with /auth/... path
    client = httpx.AsyncClient()
    headers = dict(request.headers)
    headers.pop("host", None)
    content = await request.body()
    
    # Build target URL - remove /api prefix
    target_path = f"/auth/{path}"
    target_url = f"{AUTH_SERVICE_URL}{target_path}"
    
    logger.info(f"Proxying {request.method} request to Auth service: {target_url}")
    
    try:
        rp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=request.query_params,
            content=content,
            timeout=60.0,
        )
        return Response(
            content=rp.content,
            status_code=rp.status_code,
            headers=dict(rp.headers),
        )
    except httpx.RequestError as e:
        logger.error(f"Proxy request to {target_url} failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Service unavailable: {e}"
        )
    finally:
        await client.aclose()

# ==================== APRAG Service Proxy Routes ====================

@app.api_route("/api/aprag/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_aprag_service(path: str, request: Request):
    """
    Proxy all APRAG service requests to the APRAG microservice
    Handles emoji-feedback, adaptive-query, personalization, etc.
    
    NOTE: This is a catch-all route. Specific routes (like /api/aprag/topics/classify-question)
    should be defined BEFORE this route to take precedence.
    """
    # Skip if this is a route that should be handled by a specific endpoint
    if path == "topics/classify-question" and request.method == "POST":
        # This should have been handled by the specific route above
        # If we reach here, the specific route wasn't matched
        logger.warning(f"⚠️ [ROUTE WARNING] classify-question caught by catch-all route. This shouldn't happen!")
    
    try:
        # Get request body if present
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
            except Exception as body_err:
                logger.warning(f"Could not read request body: {body_err}")
                body = None
        
        # Forward headers (including Authorization)
        headers = dict(request.headers)
        # Remove host header to avoid conflicts
        headers.pop("host", None)
        
        # Build target URL
        target_url = f"{APRAG_SERVICE_URL}/api/aprag/{path}"
        
        logger.info(f"Proxying {request.method} request to APRAG service: {target_url}")
        
        # Forward request to APRAG service
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
                params=request.query_params
            )
        
        # Log response details for debugging
        content_length = len(response.content) if response.content else 0
        content_type = response.headers.get("content-type", "application/json")
        logger.info(f"APRAG service response: status={response.status_code}, content_length={content_length}, content_type={content_type}")
        
        # Prepare response headers (exclude content-encoding to avoid issues)
        response_headers = {}
        # Copy only safe headers
        safe_headers = ["content-type", "cache-control", "expires"]
        for header_name in safe_headers:
            if header_name in response.headers:
                response_headers[header_name] = response.headers[header_name]
        
        # Ensure content-type is set
        if "content-type" not in response_headers:
            response_headers["content-type"] = "application/json"
        
        # Return response from APRAG service
        try:
            # Use JSONResponse for JSON content to avoid encoding issues
            if content_type and "json" in content_type.lower():
                from fastapi.responses import JSONResponse
                import json
                try:
                    json_data = response.json()
                    return JSONResponse(
                        content=json_data,
                        status_code=response.status_code,
                        headers=response_headers
                    )
                except Exception as json_err:
                    logger.warning(f"Could not parse as JSON, returning raw content: {json_err}")
                    # Fallback to raw content
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers=response_headers,
                        media_type=content_type
                    )
            else:
                # For non-JSON content, use Response
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=content_type
                )
        except Exception as response_err:
            logger.error(f"Error creating Response object: {response_err}")
            logger.error(f"Response status: {response.status_code}")
            logger.error(f"Response content type: {content_type}")
            logger.error(f"Response content length: {content_length}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to create response: {str(response_err)}")
        
    except httpx.TimeoutException:
        logger.error(f"APRAG service timeout for path: {path}")
        raise HTTPException(status_code=504, detail="APRAG service timeout")
    except httpx.RequestError as e:
        logger.error(f"APRAG service request error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=503, detail=f"APRAG service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"APRAG service proxy error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"APRAG service proxy failed: {str(e)}")


@app.api_route("/aprag/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_aprag_service_no_prefix(path: str, request: Request):
    """
    Proxy APRAG service requests without /api prefix (for Next.js rewrites that strip /api)
    """
    try:
        # Get request body if present
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
            except Exception:
                body = None
        
        # Forward headers (including Authorization)
        headers = dict(request.headers)
        # Remove host header to avoid conflicts
        headers.pop("host", None)
        
        # Build target URL - add /api prefix back since Next.js rewrites strip it
        target_url = f"{APRAG_SERVICE_URL}/api/aprag/{path}"
        
        logger.info(f"Proxying {request.method} request (no /api prefix) to APRAG service: {target_url}")
        
        # Forward request to APRAG service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
                params=request.query_params
            )
        
        # Return response from APRAG service
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type")
        )
        
    except httpx.TimeoutException:
        logger.error(f"APRAG service timeout for path: {path}")
        raise HTTPException(status_code=504, detail="APRAG service timeout")
    except httpx.RequestError as e:
        logger.error(f"APRAG service request error: {e}")
        raise HTTPException(status_code=503, detail=f"APRAG service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"APRAG proxy error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"APRAG proxy error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # API_GATEWAY_PORT is already defined above in the fallback logic
    port = int(os.environ.get("PORT", API_GATEWAY_PORT))
    print(f"🚀 Starting RAG3 API Gateway on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)