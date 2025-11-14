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
        def load_from_database(db_manager):
            """Fallback method for database loading"""
            pass

# Import database and API modules
from database.database import DatabaseManager
from api import interactions, feedback, profiles, personalization, recommendations, analytics, settings, topics

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
        "aprag_enabled": FeatureFlags.is_aprag_enabled()
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


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8007"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

