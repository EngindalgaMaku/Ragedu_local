"""
Document Processing Service - Main Application
Modular architecture with clean separation of concerns
"""
import uvicorn
from fastapi import FastAPI
from config import PORT
from utils.logger import logger, setup_logging
from api.routes import register_routes

# Setup logging
setup_logging(level="INFO")

# Create FastAPI app
app = FastAPI(
    title="Document Processing Service",
    description="Modular text processing and external service integration microservice",
    version="2.0.0"
)

# Register all routes
register_routes(app)


@app.on_event("startup")
async def startup_event():
    """Fast startup - connections are lazy loaded"""
    logger.info("✅ Document Processing Service starting...")
    logger.info(f"📍 Service running on port {PORT}")
    logger.info(f"📦 Modular architecture loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Document Processing Service shutting down...")


if __name__ == "__main__":
    logger.info(f"🚀 Starting server on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

