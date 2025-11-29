"""
API Routes for Document Processing Service
"""
from fastapi import APIRouter

def register_routes(app):
    """
    Register all route modules with the FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    from .health import router as health_router
    from .processing import router as processing_router
    from .query import router as query_router
    from .rerank import router as rerank_router
    from .improvement import router as improvement_router
    from .sessions import router as sessions_router
    
    app.include_router(health_router, tags=["Health"])
    app.include_router(processing_router, tags=["Processing"])
    app.include_router(query_router, tags=["Query"])
    app.include_router(rerank_router, tags=["Rerank"])
    app.include_router(improvement_router, tags=["Improvement"])
    app.include_router(sessions_router, tags=["Sessions"])

