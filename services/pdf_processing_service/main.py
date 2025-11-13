import os
import tempfile
import logging
import traceback
import sys
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from contextlib import asynccontextmanager

# PDF Service is standalone - no need to import from main project

# Set environment variables BEFORE any other imports
def _setup_marker_environment():
    """Setup marker environment variables before any marker imports."""
    cache_base = os.getenv("MARKER_CACHE_DIR", "/app/models")
    cache_vars = {
        "TORCH_HOME": f"{cache_base}/torch",
        "HUGGINGFACE_HUB_CACHE": f"{cache_base}/huggingface",
        "TRANSFORMERS_CACHE": f"{cache_base}/transformers",
        "HF_HOME": f"{cache_base}/hf_home",
        "TRANSFORMERS_OFFLINE": "0",
        "HF_HUB_OFFLINE": "0",
        "MARKER_CACHE_DIR": cache_base,
        "MARKER_DISABLE_GEMINI": "true",
        "MARKER_USE_LOCAL_ONLY": "true",
        "MARKER_DISABLE_ALL_LLM": "true",
    }
    for key, value in cache_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    logging.info(f"🔧 Marker environment configured. Cache base: {cache_base}")

_setup_marker_environment()

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
import PyPDF2

# Import local model cache manager
try:
    from model_cache_manager import get_cached_marker_models, get_model_cache_manager
    CACHE_MANAGER_AVAILABLE = True
    logging.info("✅ Model Cache Manager imported successfully.")
except ImportError as e:
    logging.warning(f"⚠️ Model Cache Manager not found: {e}. Using direct model loading.")
    CACHE_MANAGER_AVAILABLE = False

# Dynamic import for Marker
try:
    from marker.converters.pdf import PdfConverter
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
    logging.info("✅ Marker library loaded successfully.")
except ImportError:
    MARKER_AVAILABLE = False
    logging.warning("⚠️ Marker library not found. Fallback to PyPDF2 will be used.")

# Global state variables
service_ready = threading.Event()
models_loading = threading.Event()
processor_instance = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logging.info("🚀 Starting PDF Processing Service...")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if MARKER_AVAILABLE:
        logging.info("🔧 Marker available - initializing processor with async model loading...")
        # Start async model loading
        models_loading.set()
        background_task = asyncio.create_task(async_load_models())
        
        # Don't wait for models to finish loading before serving traffic
        # Health checks will indicate when ready
    else:
        logging.info("⚠️ PDF Processing Service started with PyPDF2 fallback only")
        service_ready.set()
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down PDF Processing Service...")

# --- FastAPI App Definition ---
app = FastAPI(
    title="PDF Processing Service",
    description="A microservice to convert PDF files to high-quality Markdown using Marker.",
    version="1.0.0",
    lifespan=lifespan
)

class PDFMetadata(BaseModel):
    source_file: str
    processing_method: str
    text_length: int
    page_count: int

class PDFProcessingResponse(BaseModel):
    content: str
    metadata: PDFMetadata

# --- Async Model Loading ---
async def async_load_models():
    """Load models asynchronously with timeout handling"""
    global processor_instance
    
    def load_models_sync():
        """Synchronous model loading in thread"""
        global processor_instance
        try:
            logging.info("⏳ Starting async model loading...")
            start_time = time.time()
            
            processor_instance = MarkerProcessor()
            
            load_time = time.time() - start_time
            if processor_instance.models_loaded:
                logging.info(f"✅ Models loaded successfully in {load_time:.1f}s")
                service_ready.set()
            else:
                logging.error("❌ Model loading failed")
        
        except Exception as e:
            logging.error(f"❌ Model loading failed: {e}")
            logging.error(traceback.format_exc())
        finally:
            models_loading.clear()
    
    # Run model loading in thread with timeout
    loop = asyncio.get_event_loop()
    timeout_seconds = int(os.getenv("MODEL_LOAD_TIMEOUT", "600"))  # 10 minutes default
    
    try:
        # Run in executor with timeout
        await asyncio.wait_for(
            loop.run_in_executor(None, load_models_sync),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        logging.error(f"❌ Model loading timed out after {timeout_seconds}s")
        models_loading.clear()
    except Exception as e:
        logging.error(f"❌ Unexpected error during model loading: {e}")
        models_loading.clear()

# --- Enhanced PDF Processor with Better Error Handling ---
def fallback_pdf_extract(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extracts text from a PDF using PyPDF2 as a fallback with improved error handling."""
    logging.info(f"📄 Using fallback PDF extraction for {os.path.basename(pdf_path)}")
    
    try:
        # Validate PDF file first
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=400, detail="PDF file not found")
        
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="PDF file is empty")
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="PDF file too large (max 50MB)")
        
        with open(pdf_path, 'rb') as f:
            try:
                reader = PyPDF2.PdfReader(f)
            except PyPDF2.errors.PdfReadError as e:
                logging.error(f"🚨 PDF read error: {e}")
                raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file")
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                logging.error("🔒 PDF is password protected")
                raise HTTPException(status_code=400, detail="Password-protected PDFs are not supported")
            
            text = ""
            page_count = len(reader.pages)
            
            if page_count == 0:
                raise HTTPException(status_code=400, detail="PDF contains no pages")
            
            if page_count > 500:  # Reasonable page limit
                logging.warning(f"⚠️ Large PDF with {page_count} pages, processing may be slow")
            
            # Extract text with progress logging
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    text += page_text
                    
                    if (i + 1) % 50 == 0:  # Log progress every 50 pages
                        logging.info(f"📄 Processed {i + 1}/{page_count} pages")
                        
                except Exception as e:
                    logging.warning(f"⚠️ Error extracting page {i + 1}: {e}")
                    # Continue with other pages
            
            # Validate extracted text
            if not text.strip():
                logging.warning("⚠️ No text extracted from PDF - may be image-based")
                text = "[No extractable text found - PDF may contain only images]"
            
            metadata = {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "pypdf2_fallback",
                "text_length": len(text),
                "page_count": page_count,
                "file_size_mb": file_size / (1024 * 1024),
                "processing_time": datetime.now().isoformat()
            }
            
            logging.info(f"✅ Fallback extraction completed: {len(text)} chars from {page_count} pages")
            return text, metadata
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"❌ Fallback PDF extraction failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# --- Enhanced Marker PDF Processor ---
class MarkerProcessor:
    def __init__(self):
        self.converter = None
        self.models_loaded = False
        self.cache_manager = None
        self.load_start_time = None
        if MARKER_AVAILABLE:
            self._load_converter()

    def _load_converter(self):
        """Load converter with comprehensive error handling and timeout"""
        if self.models_loaded:
            return
        
        self.load_start_time = time.time()
        
        try:
            if CACHE_MANAGER_AVAILABLE:
                logging.info("🔄 Loading Marker models using cache manager...")
                self.cache_manager = get_model_cache_manager()
                
                # Always try to get cached models first
                artifact_dict = get_cached_marker_models(force_download=False)
                
                if artifact_dict:
                    logging.info("✅ Using cached/restored Marker models")
                else:
                    # Cache manager handles download and caching internally
                    logging.error("❌ Cache manager failed to provide models")
                    # Fallback to direct loading only as last resort
                    logging.info("🔄 Attempting direct model loading as fallback...")
                    from marker.models import create_model_dict
                    artifact_dict = create_model_dict()
            else:
                logging.info("🔄 Loading Marker models directly (no cache manager)...")
                from marker.models import create_model_dict
                artifact_dict = create_model_dict()
            
            if not artifact_dict:
                raise Exception("Model dictionary is empty or None")
            
            logging.info(f"📊 Model dictionary contains {len(artifact_dict)} components")
            
            self.converter = PdfConverter(artifact_dict=artifact_dict)
            self.models_loaded = True
            
            load_time = time.time() - self.load_start_time
            logging.info(f"✅ Marker models loaded successfully in {load_time:.1f}s!")
            
            if CACHE_MANAGER_AVAILABLE and self.cache_manager:
                stats = self.cache_manager.get_cache_stats()
                logging.info(f"📊 Cache stats: {stats['total_cache_size_mb']:.1f}MB cached, {stats['cached_model_sets']} model sets")
            
        except Exception as e:
            load_time = time.time() - self.load_start_time if self.load_start_time else 0
            logging.error(f"❌ Failed to load Marker models after {load_time:.1f}s: {e}")
            logging.error(f"📋 Error details: {traceback.format_exc()}")
            self.models_loaded = False

    def process(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process PDF with Marker with comprehensive error handling and fallbacks"""
        if not self.models_loaded or not self.converter:
            logging.warning("🔄 Marker not ready, using fallback")
            return fallback_pdf_extract(pdf_path)

        start_time = time.time()
        try:
            # Validate file before processing
            if not os.path.exists(pdf_path):
                raise ValueError(f"PDF file not found: {pdf_path}")
            
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                raise ValueError("PDF file is empty")
            
            logging.info(f"🔄 Processing with Marker: {os.path.basename(pdf_path)} ({file_size / (1024*1024):.1f}MB)")
            
            # Process with timeout
            rendered = self.converter(pdf_path)
            
            if not rendered:
                raise ValueError("Marker returned empty result")
            
            markdown_text, _, _ = text_from_rendered(rendered)
            
            if not markdown_text or not markdown_text.strip():
                logging.warning("⚠️ Marker produced empty text, falling back to PyPDF2")
                return fallback_pdf_extract(pdf_path)
            
            page_count = len(rendered.children) if hasattr(rendered, 'children') else 0
            processing_time = time.time() - start_time

            metadata = {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "marker",
                "text_length": len(markdown_text),
                "page_count": page_count,
                "file_size_mb": file_size / (1024 * 1024),
                "processing_time_seconds": processing_time,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logging.info(f"✅ Marker processing completed in {processing_time:.1f}s: {len(markdown_text)} chars from {page_count} pages")
            return markdown_text, metadata
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"❌ Marker processing failed after {processing_time:.1f}s: {e}")
            logging.error(f"📋 Error details: {traceback.format_exc()}")
            logging.info("🔄 Falling back to PyPDF2 due to Marker error")
            return fallback_pdf_extract(pdf_path)

# --- Health Check Endpoints ---
@app.get("/health")
async def health_check():
    """Basic health check - service is alive"""
    return {
        "status": "ok",
        "service": "pdf-processing",
        "timestamp": datetime.now().isoformat(),
        "marker_available": MARKER_AVAILABLE
    }

@app.get("/health/ready")
async def readiness_check():
    """Readiness check - service is ready to process requests"""
    global processor_instance
    
    is_ready = service_ready.is_set()
    models_still_loading = models_loading.is_set()
    
    status = {
        "status": "ready" if is_ready else "not_ready",
        "service": "pdf-processing",
        "timestamp": datetime.now().isoformat(),
        "details": {
            "marker_available": MARKER_AVAILABLE,
            "models_loaded": processor_instance.models_loaded if processor_instance else False,
            "models_loading": models_still_loading,
            "cache_manager_available": CACHE_MANAGER_AVAILABLE
        }
    }
    
    if not is_ready:
        return status, 503  # Service Unavailable
    
    return status

@app.get("/health/live")
async def liveness_check():
    """Liveness check - service is alive and functioning"""
    global processor_instance
    
    # Basic liveness - if we can respond, we're alive
    status = {
        "status": "alive",
        "service": "pdf-processing",
        "timestamp": datetime.now().isoformat(),
        "uptime_info": {
            "marker_available": MARKER_AVAILABLE,
            "processor_available": processor_instance is not None,
            "cache_available": CACHE_MANAGER_AVAILABLE
        }
    }
    
    return status

@app.post("/process", response_model=PDFProcessingResponse)
async def process_pdf_endpoint(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Process PDF file and convert to Markdown with comprehensive error handling
    and MD saving failure checks
    """
    global processor_instance
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are supported.")
    
    # Validate file size before processing
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    max_size = 50 * 1024 * 1024  # 50MB
    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB.")
    
    tmp_path = None
    md_content = None
    processing_metadata = None
    
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        logging.info(f"📄 Processing PDF: {file.filename} ({file_size / (1024*1024):.2f}MB)")
        
        # Process the PDF
        if processor_instance and processor_instance.models_loaded:
            logging.info("🔄 Using Marker processor")
            md_content, processing_metadata = processor_instance.process(tmp_path)
        else:
            logging.info("🔄 Using PyPDF2 fallback processor")
            md_content, processing_metadata = fallback_pdf_extract(tmp_path)
        
        # Validate MD content was generated
        if not md_content or not md_content.strip():
            logging.error("❌ MD content generation failed - empty result")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate markdown content from PDF. The file may be corrupted or contain only images."
            )
        
        # Validate MD content quality
        md_length = len(md_content.strip())
        if md_length < 50:  # Very short content might indicate processing failure
            logging.warning(f"⚠️ Generated MD content is very short ({md_length} characters)")
            
        logging.info(f"✅ MD generation successful: {md_length} characters")
        
        # Enhanced metadata for frontend
        processing_metadata.update({
            "original_filename": file.filename,
            "file_size_bytes": file_size,
            "md_generation_successful": True,
            "md_length": md_length,
            "processing_timestamp": datetime.now().isoformat(),
            "service_version": "1.0.0"
        })
        
        # Create response
        response = PDFProcessingResponse(content=md_content, metadata=processing_metadata)
        
        # Optional: Save MD to persistent storage for frontend retrieval
        if background_tasks:
            background_tasks.add_task(
                save_md_for_frontend_access,
                file.filename,
                md_content,
                processing_metadata
            )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error during PDF processing: {e}")
        logging.error(f"📋 Error details: {traceback.format_exc()}")
        
        # Return detailed error for debugging
        error_metadata = {
            "error": True,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "original_filename": file.filename if file.filename else "unknown",
            "file_size_bytes": file_size,
            "processing_timestamp": datetime.now().isoformat(),
            "md_generation_successful": False
        }
        
        raise HTTPException(
            status_code=500,
            detail={
                "message": "PDF processing failed",
                "error": str(e),
                "metadata": error_metadata
            }
        )
    
    finally:
        # Always clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logging.debug(f"🗑️ Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logging.warning(f"⚠️ Could not clean up temporary file: {e}")

async def save_md_for_frontend_access(filename: str, md_content: str, metadata: Dict[str, Any]):
    """
    Save generated MD content for potential frontend access
    This is a placeholder for future integration with persistent storage
    """
    try:
        # For now, just log the successful generation
        # In production, this could save to cloud storage, database, etc.
        logging.info(f"💾 MD content ready for frontend access: {filename} ({len(md_content)} chars)")
        
        # Example: Could implement cloud storage saving here
        # cloud_storage_manager = get_cloud_storage_manager()
        # md_key = f"processed_documents/{filename}_{datetime.now().isoformat()}.md"
        # cloud_storage_manager.upload_blob(md_key, md_content.encode())
        
    except Exception as e:
        logging.warning(f"⚠️ Could not save MD for frontend access: {e}")

@app.get("/status")
async def service_status():
    """Get detailed service status including model loading state"""
    global processor_instance
    
    return {
        "service": "pdf-processing",
        "status": "operational" if service_ready.is_set() else "starting",
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "marker_available": MARKER_AVAILABLE,
            "cache_manager_available": CACHE_MANAGER_AVAILABLE,
            "models_loaded": processor_instance.models_loaded if processor_instance else False,
            "models_loading": models_loading.is_set(),
            "fallback_available": True
        },
        "cache_info": processor_instance.cache_manager.get_cache_stats() if processor_instance and processor_instance.cache_manager else None,
        "version": "1.0.0"
    }

# Add startup logging enhancement
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logging.info(f"📊 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    
    # Enhanced logging configuration
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["default"],
        },
    }
    
    logging.info(f"🚀 Starting PDF Processing Service on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_config=log_config,
        access_log=True
    )