"""
Health check endpoints
"""
import requests
from fastapi import APIRouter
from config import MODEL_INFERENCER_URL, PORT
from utils.logger import logger

router = APIRouter()


@router.get("/")
async def root():
    """Simple health check endpoint"""
    return {
        "message": "Document Processing Service is running",
        "status": "healthy"
    }


@router.get("/health")
async def health_check():
    """Detailed health check with dependency status"""
    try:
        model_service_status = False
        
        # Test model inference service
        try:
            health_response = requests.get(f"{MODEL_INFERENCER_URL}/health", timeout=5)
            model_service_status = health_response.status_code == 200
        except Exception as e:
            logger.debug(f"Model service health check failed: {e}")
        
        return {
            "status": "healthy",
            "text_processing_available": True,  # Using unified chunking system
            "model_service_connected": model_service_status,
            "model_inferencer_url": MODEL_INFERENCER_URL,
            "port": PORT
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "healthy",  # Base service is running
            "error": str(e),
            "text_processing_available": True,
            "model_service_connected": False,
            "port": PORT
        }






