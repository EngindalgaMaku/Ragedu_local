"""
Main FastAPI Application for RAG Education Assistant Auth Service
Comprehensive authentication microservice with JWT tokens, role-based permissions, and session management
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

# Import auth components
from auth.auth_manager import AuthManager
from auth.middleware import (
    AuthenticationMiddleware, RateLimitMiddleware, SecurityHeadersMiddleware
)
from auth.dependencies import get_auth_manager
from api.auth import router as auth_router
from api.users import router as users_router
from api.roles import router as roles_router
from api.admin import router as admin_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
class Config:
    """Application configuration"""
    # Server configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", os.getenv("AUTH_SERVICE_PORT", "8006")))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # JWT configuration
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    # Database configuration
    DATABASE_PATH = os.getenv("DATABASE_PATH", "data/rag_assistant.db")
    
    # CORS configuration - Enhanced with external IP support
    _cors_env = os.getenv("CORS_ORIGINS", "")
    if _cors_env and _cors_env.strip():
        CORS_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
    else:
        # Fallback CORS origins with external IP support for Docker deployment
        logger.warning("CORS_ORIGINS environment variable not set, using fallback configuration")
        CORS_ORIGINS = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://0.0.0.0:3000",
            "http://host.docker.internal:3000",
            "http://frontend:3000",
            "http://api-gateway:8000",
            "http://46.62.254.131:3000",  # External IP frontend
            "http://46.62.254.131:8000",  # External IP API gateway
            "http://46.62.254.131:8006",  # External IP auth service (self)
            "http://localhost:8000",     # Local API gateway
            "http://127.0.0.1:8000"      # Local API gateway
        ]
    
    # Ensure external server IP origins are always included for Docker deployment
    external_origins = [
        "http://46.62.254.131:3000",
        "http://46.62.254.131:8000",
        "http://46.62.254.131:8006"
    ]
    for origin in external_origins:
        if origin not in CORS_ORIGINS:
            CORS_ORIGINS.append(origin)
    
    CORS_METHODS = ["*"]  # Allow all methods including PATCH
    CORS_HEADERS = [
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ]
    CORS_CREDENTIALS = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
    
    # Rate limiting configuration - Geliştirme için gevşetildi
    RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_RPM", "300"))
    RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "50"))
    
    # Security configuration
    REQUIRE_AUTH_BY_DEFAULT = os.getenv("REQUIRE_AUTH", "true").lower() == "true"
    
    # Service information
    SERVICE_NAME = "RAG Education Assistant - Auth Service"
    VERSION = "1.0.0"
    DESCRIPTION = """
    Comprehensive authentication and authorization microservice for RAG Education Assistant.
    
    ## Features
    
    * **JWT Authentication**: Secure token-based authentication with access and refresh tokens
    * **Role-Based Access Control**: Flexible permission system with predefined roles
    * **Session Management**: Complete session lifecycle management
    * **User Management**: Full CRUD operations for user accounts
    * **Security Features**: Rate limiting, password validation, security headers
    * **Admin Interface**: Administrative endpoints for system management
    
    ## Authentication
    
    Use the `/auth/login` endpoint to authenticate and receive JWT tokens.
    Include the access token in the Authorization header: `Bearer <token>`
    
    ## Default Roles
    
    * **Admin**: Full system access
    * **Teacher**: Session and document management
    * **Student**: Read-only access to sessions and documents
    """


# Initialize configuration
config = Config()

# Global auth manager instance
auth_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    Handles startup and shutdown procedures
    """
    # Startup
    logger.info(f"Starting {config.SERVICE_NAME} v{config.VERSION}")
    
    try:
        # Initialize authentication manager
        global auth_manager
        auth_manager = AuthManager(
            secret_key=config.SECRET_KEY,
            algorithm=config.ALGORITHM,
            access_token_expire_minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES,
            refresh_token_expire_days=config.REFRESH_TOKEN_EXPIRE_DAYS,
            db_path=config.DATABASE_PATH
        )
        
        # Store auth manager in app state for dependency injection
        app.state.auth_manager = auth_manager
        
        # Create default roles if they don't exist
        auth_manager.role_model.create_default_roles()
        
        # Clean up expired sessions on startup
        cleaned_sessions = auth_manager.cleanup_expired_sessions()
        logger.info(f"Cleaned up {cleaned_sessions} expired sessions on startup")
        
        logger.info("Auth service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down auth service...")
    
    try:
        # Clean up expired sessions on shutdown
        if auth_manager:
            cleaned_sessions = auth_manager.cleanup_expired_sessions()
            logger.info(f"Cleaned up {cleaned_sessions} expired sessions on shutdown")
        
        logger.info("Auth service shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title=config.SERVICE_NAME,
    description=config.DESCRIPTION,
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    debug=config.DEBUG
)


# Custom dependency override for auth manager
def get_auth_manager_override():
    """Override for auth manager dependency"""
    return auth_manager


app.dependency_overrides[get_auth_manager] = get_auth_manager_override


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent JSON response"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_exception",
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "detail": "Request validation failed",
            "validation_errors": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": "An internal server error occurred"
        }
    )


# Add middleware (order is important - last added runs first)

# CORS Configuration - Fixed for credentials support
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # Use specific origins instead of wildcard
    allow_credentials=config.CORS_CREDENTIALS,  # Allow credentials
    allow_methods=config.CORS_METHODS,
    allow_headers=config.CORS_HEADERS,
)

# Global OPTIONS handler
@app.options("/{full_path:path}")
async def options_handler():
    """Handle all OPTIONS requests"""
    return {"status": "OK"}

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=config.RATE_LIMIT_REQUESTS_PER_MINUTE,
    burst_requests=config.RATE_LIMIT_BURST
)

# Authentication middleware (applied to protected routes)
# NOTE: Disabled - we use dependency injection for authentication
# Endpoints use get_current_user, get_current_active_user, require_admin etc.
# This prevents middleware conflicts


# Health check endpoint (no authentication required)
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns service status and basic information
    """
    try:
        # Test database connection
        if auth_manager:
            session_count = auth_manager.session_model.get_active_sessions_count()
            db_status = "connected"
        else:
            session_count = 0
            db_status = "not_initialized"
    except Exception as e:
        logger.error(f"Health check database error: {e}")
        db_status = "error"
        session_count = 0
    
    return {
        "status": "healthy",
        "service": config.SERVICE_NAME,
        "version": config.VERSION,
        "database": db_status,
        "active_sessions": session_count,
        "environment": "development" if config.DEBUG else "production"
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint
    Returns basic service information
    """
    return {
        "message": f"Welcome to {config.SERVICE_NAME}",
        "version": config.VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


# Include API routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(roles_router)
app.include_router(admin_router)


# Additional utility endpoints
@app.get("/info", tags=["Information"])
async def service_info():
    """
    Service information endpoint
    Returns detailed service configuration and status
    """
    try:
        # Get service statistics if auth manager is available
        stats = {}
        if auth_manager:
            stats = {
                "total_users": auth_manager.user_model.get_user_count(),
                "active_users": auth_manager.user_model.get_user_count(is_active=True),
                "total_roles": auth_manager.role_model.get_role_count(),
                "active_sessions": auth_manager.session_model.get_active_sessions_count()
            }
    except Exception as e:
        logger.error(f"Error getting service stats: {e}")
        stats = {"error": "Could not retrieve statistics"}
    
    return {
        "service": config.SERVICE_NAME,
        "version": config.VERSION,
        "description": "Authentication and authorization microservice",
        "features": [
            "JWT Authentication",
            "Role-Based Access Control",
            "Session Management",
            "User Management",
            "Rate Limiting",
            "Security Headers",
            "CORS Support"
        ],
        "endpoints": {
            "authentication": "/auth",
            "user_management": "/users",
            "role_management": "/roles",
            "documentation": "/docs",
            "health_check": "/health"
        },
        "statistics": stats,
        "configuration": {
            "jwt_algorithm": config.ALGORITHM,
            "access_token_expire_minutes": config.ACCESS_TOKEN_EXPIRE_MINUTES,
            "refresh_token_expire_days": config.REFRESH_TOKEN_EXPIRE_DAYS,
            "cors_enabled": bool(config.CORS_ORIGINS),
            "rate_limiting": {
                "requests_per_minute": config.RATE_LIMIT_REQUESTS_PER_MINUTE,
                "burst_requests": config.RATE_LIMIT_BURST
            }
        }
    }


# Development server runner
if __name__ == "__main__":
    logger.info(f"Starting {config.SERVICE_NAME} in development mode")
    logger.info(f"Server will be available at http://{config.HOST}:{config.PORT}")
    logger.info(f"API documentation at http://{config.HOST}:{config.PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info" if not config.DEBUG else "debug",
        access_log=True
    )