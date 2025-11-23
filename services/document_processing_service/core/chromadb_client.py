"""
ChromaDB client management
Handles connection to ChromaDB service with Docker and Cloud Run support
"""
import chromadb
from chromadb.config import Settings
from urllib.parse import urlparse
from utils.logger import logger
from config import CHROMA_SERVICE_URL


def get_chroma_client():
    """
    Get ChromaDB client with connection to our service
    
    Supports:
    - Docker: http://chromadb-service:8000
    - Cloud Run: https://xxx.run.app
    - Local: http://localhost:8000
    
    Returns:
        ChromaDB HttpClient instance
        
    Raises:
        Exception: If client creation fails
    """
    try:
        logger.info(f"🔍 DIAGNOSTIC: Creating ChromaDB client with URL: {CHROMA_SERVICE_URL}")
        
        # Check if URL is Cloud Run format (https://xxx.run.app) or Docker format (http://host:port)
        if CHROMA_SERVICE_URL.startswith("https://"):
            # Cloud Run: use full URL
            parsed = urlparse(CHROMA_SERVICE_URL)
            host = parsed.hostname
            port = parsed.port or 443  # HTTPS default port
            use_https = True
        elif CHROMA_SERVICE_URL.startswith("http://"):
            # Docker or local: extract host and port
            chroma_url = CHROMA_SERVICE_URL.replace("http://", "")
            if ":" in chroma_url:
                host_parts = chroma_url.split(":")
                host = host_parts[0]
                port = int(host_parts[1])
            else:
                host = chroma_url
                port = 8000
            use_https = False
        else:
            # Fallback: assume Docker format
            chroma_url = CHROMA_SERVICE_URL.replace("http://", "").replace("https://", "")
            if ":" in chroma_url:
                host_parts = chroma_url.split(":")
                host = host_parts[0]
                port = int(host_parts[1])
            else:
                host = chroma_url
                port = 8000
            use_https = False
        
        logger.info(f"🔍 DIAGNOSTIC: Connecting to ChromaDB at host='{host}', port={port}, https={use_https}")
        
        # Configure ChromaDB settings with longer timeouts for bulk operations
        chroma_settings = Settings(
            anonymized_telemetry=False,
            chroma_client_auth_provider=None,
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
        )
        
        if use_https:
            # For Cloud Run, try to use the full URL
            logger.warning("⚠️ HTTPS detected for ChromaDB. Ensure ChromaDB service supports HTTPS or use HTTP proxy.")
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=chroma_settings
            )
        else:
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=chroma_settings
            )
        
        logger.info(f"✅ ChromaDB client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create ChromaDB client: {e}")
        raise






