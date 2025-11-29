"""
APRAG Service - Main FastAPI Application
Adaptive Personalized RAG System for educational assistance
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from typing import Optional

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
sys.path.insert(0, parent_dir)

try:
    from config.feature_flags import FeatureFlags
except ImportError:
    # Fallback: Define minimal version if parent config not available
    class FeatureFlags:
        @staticmethod
        def is_aprag_enabled(session_id=None):
            """Fallback implementation when feature flags config is not available"""
            return os.getenv("APRAG_ENABLED", "true").lower() == "true"
        
        @staticmethod
        def is_egitsel_kbrag_enabled(session_id=None):
            """Fallback for Eğitsel-KBRAG"""
            return os.getenv("EGITSEL_KBRAG_ENABLED", "false").lower() == "true"
        
        @staticmethod
        def is_module_extraction_enabled(session_id=None):
            """Fallback for module extraction"""
            return os.getenv("MODULE_EXTRACTION_ENABLED", "true").lower() == "true"
        
        @staticmethod
        def is_cacs_enabled(session_id=None):
            """Fallback for CACS"""
            return os.getenv("CACS_ENABLED", "false").lower() == "true"
        
        @staticmethod
        def is_zpd_enabled(session_id=None):
            """Fallback for ZPD"""
            return os.getenv("ZPD_ENABLED", "false").lower() == "true"
        
        @staticmethod
        def is_bloom_enabled(session_id=None):
            """Fallback for Bloom taxonomy"""
            return os.getenv("BLOOM_ENABLED", "false").lower() == "true"
        
        @staticmethod
        def is_cognitive_load_enabled(session_id=None):
            """Fallback for cognitive load"""
            return os.getenv("COGNITIVE_LOAD_ENABLED", "false").lower() == "true"
        
        @staticmethod
        def is_emoji_feedback_enabled(session_id=None):
            """Fallback for emoji feedback"""
            return os.getenv("EMOJI_FEEDBACK_ENABLED", "false").lower() == "true"
        
        @staticmethod
        def is_progressive_assessment_enabled(session_id=None):
            """Fallback for progressive assessment"""
            return os.getenv("PROGRESSIVE_ASSESSMENT_ENABLED", "false").lower() == "true"
        
        @staticmethod
        def is_module_quality_validation_enabled(session_id=None):
            """Fallback for module quality validation"""
            return os.getenv("MODULE_QUALITY_VALIDATION_ENABLED", "true").lower() == "true"
        
        @staticmethod
        def is_module_curriculum_alignment_enabled(session_id=None):
            """Fallback for module curriculum alignment"""
            return os.getenv("MODULE_CURRICULUM_ALIGNMENT_ENABLED", "true").lower() == "true"
        
        @staticmethod
        def load_from_database(db_manager):
            """Fallback method for database loading"""
            pass

# Import database and API modules
from database.database import DatabaseManager
from api import interactions, feedback, profiles, personalization, recommendations, analytics, settings, topics, knowledge_extraction, hybrid_rag_query, session_settings, modules, async_hybrid_rag_query

# Import CACS scoring (Faz 2 - Eğitsel-KBRAG)
try:
    from api import scoring
    SCORING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CACS scoring module not available: {e}")
    SCORING_AVAILABLE = False

# Import Emoji Feedback (Faz 4 - Eğitsel-KBRAG)
try:
    from api import emoji_feedback
    EMOJI_FEEDBACK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Emoji feedback module not available: {e}")
    EMOJI_FEEDBACK_AVAILABLE = False

# Import EBARS (Emoji-Based Adaptive Response System)
try:
    from ebars import router as ebars_router
    EBARS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"EBARS module not available: {e}")
    EBARS_AVAILABLE = False

# Import Progressive Assessment (ADIM 3 - Progressive Assessment Flow)
try:
    from api import progressive_assessment
    PROGRESSIVE_ASSESSMENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Progressive assessment module not available: {e}")
    PROGRESSIVE_ASSESSMENT_AVAILABLE = False

# Import Adaptive Query (Faz 5 - Eğitsel-KBRAG Full Pipeline)
try:
    from api import adaptive_query
    ADAPTIVE_QUERY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Adaptive query module not available: {e}")
    ADAPTIVE_QUERY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database manager
db_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global db_manager
    
    # Startup
    logger.info("Starting APRAG Service...")
    
    # Initialize database
    db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
    db_manager = DatabaseManager(db_path)
    
    # Load feature flags from database
    try:
        FeatureFlags.load_from_database(db_manager)
        logger.info("Feature flags loaded from database")
    except Exception as e:
        logger.warning(f"Could not load feature flags from database: {e}")
        logger.info("Using default feature flag values")
    
    # Check if APRAG is enabled
    if not FeatureFlags.is_aprag_enabled():
        logger.warning("APRAG module is disabled. Service will start but features will be inactive.")
    else:
        logger.info("APRAG module is enabled")
    
    yield
    
    # Shutdown
    logger.info("Shutting down APRAG Service...")


# Create FastAPI app
app = FastAPI(
    title="APRAG Service",
    description="Adaptive Personalized RAG System for Educational Assistance",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration - Enhanced with external IP support
_cors_env = os.getenv("CORS_ORIGINS", "")
if _cors_env and _cors_env.strip():
    cors_origins = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
else:
    # Fallback CORS origins with external IP support for Docker deployment
    logger.warning("CORS_ORIGINS environment variable not set, using fallback configuration")
    cors_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
        "http://host.docker.internal:3000",
        "http://frontend:3000",
        "http://api-gateway:8000",
        "http://auth-service:8006",
        "http://46.62.254.131:3000",  # External IP frontend
        "http://46.62.254.131:8000",  # External IP API gateway
        "http://46.62.254.131:8006",  # External IP auth service
        "http://46.62.254.131:8007",  # External IP aprag service (self)
        "*"  # Allow all as last resort
    ]

# Ensure external server IP origins are always included for Docker deployment
external_origins = [
    "http://46.62.254.131:3000",
    "http://46.62.254.131:8000",
    "http://46.62.254.131:8006",
    "http://46.62.254.131:8007"
]
for origin in external_origins:
    if origin not in cors_origins:
        cors_origins.append(origin)

logger.info(f"APRAG Service CORS Origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "aprag-service",
        "version": "1.0.0",
        "aprag_enabled": FeatureFlags.is_aprag_enabled(),
        "egitsel_kbrag_enabled": FeatureFlags.is_egitsel_kbrag_enabled(),
        "module_extraction_enabled": FeatureFlags.is_module_extraction_enabled(),
        "features": {
            "cacs": FeatureFlags.is_cacs_enabled(),
            "zpd": FeatureFlags.is_zpd_enabled(),
            "bloom": FeatureFlags.is_bloom_enabled(),
            "cognitive_load": FeatureFlags.is_cognitive_load_enabled(),
            "emoji_feedback": FeatureFlags.is_emoji_feedback_enabled(),
            "progressive_assessment": FeatureFlags.is_progressive_assessment_enabled(),
            "ebars": FeatureFlags.is_ebars_enabled(),
            "module_extraction": FeatureFlags.is_module_extraction_enabled(),
            "module_quality_validation": FeatureFlags.is_module_quality_validation_enabled(),
            "module_curriculum_alignment": FeatureFlags.is_module_curriculum_alignment_enabled()
        }
    }


# Include routers
app.include_router(interactions.router, prefix="/api/aprag/interactions", tags=["Interactions"])
app.include_router(feedback.router, prefix="/api/aprag/feedback", tags=["Feedback"])
app.include_router(profiles.router, prefix="/api/aprag/profiles", tags=["Profiles"])
app.include_router(personalization.router, prefix="/api/aprag/personalize", tags=["Personalization"])
app.include_router(recommendations.router, prefix="/api/aprag/recommendations", tags=["Recommendations"])
app.include_router(analytics.router, prefix="/api/aprag/analytics", tags=["Analytics"])
app.include_router(settings.router, prefix="/api/aprag/settings", tags=["Settings"])
app.include_router(topics.router, prefix="/api/aprag/topics", tags=["Topics"])
app.include_router(modules.router, prefix="/api/aprag/modules", tags=["Modules"])
app.include_router(knowledge_extraction.router, prefix="/api/aprag/knowledge", tags=["Knowledge Extraction"])
app.include_router(hybrid_rag_query.router, prefix="/api/aprag/hybrid-rag", tags=["Hybrid RAG"])
app.include_router(async_hybrid_rag_query.router, prefix="/api/aprag/async-rag", tags=["Async Hybrid RAG"])
app.include_router(session_settings.router, prefix="/api/aprag/session-settings", tags=["Session Settings"])

# Include Eğitsel-KBRAG routers (use Depends(get_db) for db access)
if SCORING_AVAILABLE and FeatureFlags.is_cacs_enabled():
    app.include_router(scoring.router, prefix="/api/aprag/scoring", tags=["CACS Scoring"])
    logger.info("CACS Scoring endpoints enabled")

if EMOJI_FEEDBACK_AVAILABLE and FeatureFlags.is_emoji_feedback_enabled():
    app.include_router(emoji_feedback.router, prefix="/api/aprag/emoji-feedback", tags=["Emoji Feedback"])
    logger.info("Emoji Feedback endpoints enabled")

if PROGRESSIVE_ASSESSMENT_AVAILABLE and FeatureFlags.is_progressive_assessment_enabled():
    app.include_router(progressive_assessment.router, prefix="/api/aprag/progressive-assessment", tags=["Progressive Assessment"])
    logger.info("Progressive Assessment endpoints enabled")

if ADAPTIVE_QUERY_AVAILABLE and FeatureFlags.is_egitsel_kbrag_enabled():
    app.include_router(adaptive_query.router, prefix="/api/aprag/adaptive-query", tags=["Adaptive Query"])
    logger.info("Adaptive Query (Full Pipeline) endpoints enabled")

# Include EBARS router (Emoji-Based Adaptive Response System)
if EBARS_AVAILABLE:
    app.include_router(ebars_router.router, prefix="/api/aprag", tags=["EBARS"])
    logger.info("EBARS (Emoji-Based Adaptive Response System) endpoints enabled")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8007"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,  # Direct app reference instead of string
        host=host,
        port=port,
        reload=False,  # Disable reload for stability
        log_level="info"
    )

