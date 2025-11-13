import os
import uuid
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing Service",
    description="Text processing and ChromaDB integration microservice",
    version="1.0.0"
)

# Pydantic models
class ProcessRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}
    collection_name: Optional[str] = "documents"
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

class ProcessResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int
    collection_name: str
    chunk_ids: List[str]

# Environment variables - Cloud Run URLs
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", "https://model-inferencer-1051060211087.europe-west1.run.app")
CHROMADB_URL = os.getenv("CHROMADB_URL", "https://chromadb-1051060211087.europe-west1.run.app")
PORT = int(os.getenv("PORT", "8080"))  # Cloud Run default port

# Try to import ChromaDB, but make it optional
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with: pip install chromadb")

# Try to import LangChain, but make it optional
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    RecursiveCharacterTextSplitter = None
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain")

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = None
        self.chroma_client = None
        
    def initialize_text_splitter(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize text splitter"""
        if not LANGCHAIN_AVAILABLE:
            raise HTTPException(status_code=500, detail="LangChain not available. Text splitting not possible.")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def initialize_chroma_client(self):
        """Initialize ChromaDB client"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Processing will continue without vector storage.")
            self.chroma_client = None
            return False
            
        try:
            # Parse the CHROMADB_URL to extract host and port
            if CHROMADB_URL.startswith("http"):
                # Remove protocol
                url_without_protocol = CHROMADB_URL.split("://", 1)[1]
                # Split host and port
                if ":" in url_without_protocol:
                    host_parts = url_without_protocol.split(":", 1)
                    host = host_parts[0]
                    port_part = host_parts[1].split("/")[0]  # Handle paths after port
                    port = int(port_part) if port_part.isdigit() else 80
                else:
                    host = url_without_protocol.split("/")[0]
                    port = 80  # Default HTTP port
            else:
                # Assume it's already in host:port format
                if ":" in CHROMADB_URL:
                    host_parts = CHROMADB_URL.split(":", 1)
                    host = host_parts[0]
                    port_part = host_parts[1]
                    port = int(port_part) if port_part.isdigit() else 80
                else:
                    host = CHROMADB_URL
                    port = 80
            
            # Initialize ChromaDB client with proper settings
            self.chroma_client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    chroma_api_impl="rest",
                    chroma_server_host=host,
                    chroma_server_http_port=str(port)
                )
            )
            
            # Test connection
            self.chroma_client.heartbeat()
            logger.info(f"ChromaDB client successfully connected: {CHROMADB_URL}")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to ChromaDB: {str(e)}")
            logger.info("Service will run without ChromaDB. Vector storage operations will not work.")
            self.chroma_client = None
            return False
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if not self.text_splitter:
            raise HTTPException(status_code=500, detail="Text splitter not initialized")
        
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from model inference service"""
        try:
            embed_url = f"{MODEL_INFERENCER_URL}/embed"
            
            # Send all texts in a single request for efficiency
            response = requests.post(
                embed_url,
                json={"texts": texts},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Embedding error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get embeddings: {response.status_code}"
                )
            
            embedding_data = response.json()
            embeddings = embedding_data.get("embeddings", [])
            
            if len(embeddings) != len(texts):
                raise HTTPException(
                    status_code=500,
                    detail=f"Embedding count ({len(embeddings)}) doesn't match text count ({len(texts)})"
                )
            
            logger.info(f"Successfully retrieved {len(embeddings)} embeddings")
            return embeddings
            
        except requests.RequestException as e:
            logger.error(f"Model inference service request error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model inference service error: {str(e)}"
            )
    
    def store_in_chromadb(self, chunks: List[str], embeddings: List[List[float]],
                         metadata: Dict[str, Any], collection_name: str) -> List[str]:
        """Store chunks and embeddings in ChromaDB"""
        if not CHROMADB_AVAILABLE:
            raise HTTPException(status_code=500, detail="ChromaDB not available")
            
        # Initialize ChromaDB connection if needed
        if not self.chroma_client:
            logger.info("Initializing ChromaDB connection...")
            if not self.initialize_chroma_client():
                raise HTTPException(status_code=500, detail="Could not connect to ChromaDB")
        
        try:
            # Get or create collection
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except Exception:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Document chunks with embeddings"}
                )
            
            # Generate unique IDs for chunks
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            
            # Prepare metadata for each chunk
            chunk_metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "chunk_id": chunk_ids[i]
                })
                chunk_metadatas.append(chunk_metadata)
            
            # Add to ChromaDB
            collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"Stored {len(chunks)} chunks in ChromaDB. Collection: {collection_name}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"ChromaDB storage error: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"ChromaDB storage error: {str(e)}"
            )

# Global processor instance
processor = DocumentProcessor()

@app.on_event("startup")
async def startup_event():
    """Fast startup - connections are lazy loaded"""
    logger.info("Document Processing Service starting...")
    logger.info(f"Service running on port {PORT}")
    logger.info("ChromaDB connection will be established when needed")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Document Processing Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        chromadb_status = False
        model_service_status = False
        
        # Test ChromaDB connection
        try:
            if processor.chroma_client is None:
                chromadb_status = processor.initialize_chroma_client()
            else:
                processor.chroma_client.heartbeat()
                chromadb_status = True
        except Exception as e:
            logger.debug(f"ChromaDB health check failed: {e}")
        
        # Test model inference service
        try:
            health_response = requests.get(f"{MODEL_INFERENCER_URL}/health", timeout=3)
            model_service_status = health_response.status_code == 200
        except Exception as e:
            logger.debug(f"Model service health check failed: {e}")
        
        return {
            "status": "healthy",
            "chromadb_available": CHROMADB_AVAILABLE,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "chromadb_connected": chromadb_status,
            "model_service_connected": model_service_status,
            "model_inferencer_url": MODEL_INFERENCER_URL,
            "chromadb_url": CHROMADB_URL,
            "port": PORT
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "healthy",  # Base service is running
            "error": str(e),
            "chromadb_connected": False,
            "model_service_connected": False,
            "port": PORT
        }

@app.post("/process-and-store", response_model=ProcessResponse)
async def process_and_store(request: ProcessRequest):
    """
    Process text block and store in ChromaDB
    
    1. Split text into chunks
    2. Get embeddings for each chunk
    3. Store in ChromaDB
    """
    try:
        logger.info(f"Starting text processing. Text length: {len(request.text)} characters")
        
        # Initialize text splitter
        processor.initialize_text_splitter(request.chunk_size, request.chunk_overlap)
        
        # Split text into chunks
        chunks = processor.split_text(request.text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Text could not be split into chunks")
        
        # Get embeddings
        embeddings = processor.get_embeddings(chunks)
        
        if len(embeddings) != len(chunks):
            raise HTTPException(
                status_code=500, 
                detail="Embedding count doesn't match chunk count"
            )
        
        # Store in ChromaDB
        chunk_ids = processor.store_in_chromadb(
            chunks=chunks,
            embeddings=embeddings,
            metadata=request.metadata,
            collection_name=request.collection_name
        )
        
        logger.info(f"Processing completed. {len(chunks)} chunks processed.")
        
        return ProcessResponse(
            success=True,
            message=f"Successfully processed: {len(chunks)} chunks stored",
            chunks_processed=len(chunks),
            collection_name=request.collection_name,
            chunk_ids=chunk_ids
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)