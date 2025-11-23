"""
Reranker Service - Mikroservis
BGE-Reranker-V2-M3, MS-MARCO ve Alibaba DashScope desteği ile seçimli reranking
"""
import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Reranker Service",
    description="Document reranking service with BGE-Reranker-V2-M3, MS-MARCO and Alibaba DashScope support",
    version="1.0.0"
)

# Configuration
RERANKER_TYPE = os.getenv("RERANKER_TYPE", "bge")  # "bge", "ms-marco", or "alibaba"
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
MS_MARCO_MODEL_NAME = os.getenv("MS_MARCO_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
ALIBABA_RERANKER_MODEL = os.getenv("ALIBABA_RERANKER_MODEL", "gte-rerank-v2")
# Alibaba DashScope reranker API endpoint
ALIBABA_API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/reranking/rerank"

# Global model instances
bge_reranker = None
ms_marco_reranker = None
alibaba_reranker_available = bool(ALIBABA_API_KEY)
current_reranker = None
current_reranker_type = None


# Request/Response Models
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None  # Optional: return top_k results
    reranker_type: Optional[str] = None  # Override default reranker type per request


class RerankResult(BaseModel):
    document: str
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]
    reranker_type: str
    processing_time_ms: float


# Model Loading Functions
def load_bge_reranker():
    """Load BGE-Reranker-V2-M3 model"""
    global bge_reranker
    if bge_reranker is None:
        try:
            logger.info(f"🔄 Loading BGE Reranker: {BGE_MODEL_NAME}...")
            from FlagEmbedding import FlagReranker
            bge_reranker = FlagReranker(BGE_MODEL_NAME, use_fp16=True)
            logger.info("✅ BGE Reranker loaded successfully")
        except ImportError:
            logger.error("❌ FlagEmbedding not installed. Install with: pip install FlagEmbedding")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load BGE Reranker: {e}")
            raise
    return bge_reranker


def load_ms_marco_reranker():
    """Load MS-MARCO reranker"""
    global ms_marco_reranker
    if ms_marco_reranker is None:
        try:
            logger.info(f"🔄 Loading MS-MARCO Reranker: {MS_MARCO_MODEL_NAME}...")
            from sentence_transformers import CrossEncoder
            ms_marco_reranker = CrossEncoder(MS_MARCO_MODEL_NAME)
            logger.info("✅ MS-MARCO Reranker loaded successfully")
        except ImportError:
            logger.error("❌ sentence-transformers not installed")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load MS-MARCO Reranker: {e}")
            raise
    return ms_marco_reranker


def rerank_with_alibaba(query: str, documents: List[str]) -> List[float]:
    """
    Rerank documents using Alibaba DashScope API
    
    Args:
        query: Search query
        documents: List of documents to rerank
        
    Returns:
        List of relevance scores (0-1 range)
    """
    if not ALIBABA_API_KEY:
        raise ValueError("ALIBABA_API_KEY is not set")
    
    try:
        # Prepare request payload
        payload = {
            "model": ALIBABA_RERANKER_MODEL,
            "query": query,
            "documents": documents
        }
        
        headers = {
            "Authorization": f"Bearer {ALIBABA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make API request
        response = requests.post(
            ALIBABA_API_BASE,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract scores from response
        # Alibaba API returns results with scores
        if "output" in result and "results" in result["output"]:
            scores = []
            for item in result["output"]["results"]:
                scores.append(float(item.get("relevance_score", 0.0)))
            return scores
        else:
            # Fallback: try to extract from different response format
            logger.warning(f"Unexpected Alibaba API response format: {result}")
            return [0.5] * len(documents)  # Default scores
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Alibaba reranker API error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to rerank with Alibaba: {e}")
        raise


def initialize_reranker():
    """Initialize the selected reranker based on RERANKER_TYPE"""
    global current_reranker, current_reranker_type
    
    try:
        if RERANKER_TYPE.lower() == "bge":
            current_reranker = load_bge_reranker()
            current_reranker_type = "bge"
            logger.info("✅ Using BGE-Reranker-V2-M3")
        else:
            current_reranker = load_ms_marco_reranker()
            current_reranker_type = "ms-marco"
            logger.info("✅ Using MS-MARCO Reranker")
    except Exception as e:
        logger.error(f"❌ Failed to initialize reranker: {e}")
        # Try fallback
        if RERANKER_TYPE.lower() == "bge":
            logger.warning("⚠️ BGE failed, trying MS-MARCO fallback...")
            try:
                current_reranker = load_ms_marco_reranker()
                current_reranker_type = "ms-marco-fallback"
                logger.info("✅ Using MS-MARCO as fallback")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
                current_reranker = None
                current_reranker_type = None
        else:
            current_reranker = None
            current_reranker_type = None


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Reranker Service starting up...")
    logger.info(f"📋 Configuration: RERANKER_TYPE={RERANKER_TYPE}")
    initialize_reranker()


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "reranker_type": current_reranker_type or "none",
        "reranker_available": current_reranker is not None,
        "configured_type": RERANKER_TYPE
    }


@app.get("/info")
async def get_info():
    """Get reranker information"""
    return {
        "reranker_type": current_reranker_type or "none",
        "reranker_available": current_reranker is not None or alibaba_reranker_available,
        "configured_type": RERANKER_TYPE,
        "bge_model": BGE_MODEL_NAME if RERANKER_TYPE.lower() == "bge" else None,
        "ms_marco_model": MS_MARCO_MODEL_NAME if RERANKER_TYPE.lower() == "ms-marco" else None,
        "alibaba_model": ALIBABA_RERANKER_MODEL if alibaba_reranker_available else None,
        "alibaba_available": alibaba_reranker_available,
        "supports_multilingual": current_reranker_type in ["bge", "alibaba"]
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents based on query relevance
    
    Supports BGE-Reranker-V2-M3, MS-MARCO, and Alibaba DashScope models
    Can override reranker type per request via reranker_type parameter
    """
    # Determine which reranker to use (request override or default)
    reranker_type_to_use = (request.reranker_type or RERANKER_TYPE).lower()
    
    # Normalize reranker type (handle "gte-rerank-v2" as "alibaba")
    if reranker_type_to_use in ["gte-rerank-v2", "alibaba"]:
        reranker_type_to_use = "alibaba"
    
    logger.info(f"🔍 RERANKER SELECTION: Requested type='{request.reranker_type}', Using='{reranker_type_to_use}', Default='{RERANKER_TYPE}'")
    
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    start_time = time.time()
    
    try:
        scores = []
        actual_type = reranker_type_to_use
        
        # Get scores based on reranker type
        if reranker_type_to_use == "bge":
            # Load BGE reranker if needed
            if bge_reranker is None:
                load_bge_reranker()
            if bge_reranker is None:
                raise HTTPException(
                    status_code=503,
                    detail="BGE reranker is not available. Check service logs."
                )
            
            # Prepare query-document pairs
            pairs = [[request.query, doc] for doc in request.documents]
            # BGE returns scores directly (0-1 range typically)
            scores = bge_reranker.compute_score(pairs)
            # Handle both single score and list of scores
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            else:
                scores = [float(s) for s in scores]
            actual_type = "bge"
            logger.info(f"✅ Using BGE reranker for {len(request.documents)} documents")
            
        elif reranker_type_to_use == "alibaba":
            # Use Alibaba DashScope API
            if not ALIBABA_API_KEY:
                raise HTTPException(
                    status_code=503,
                    detail="Alibaba reranker is not available. ALIBABA_API_KEY is not set."
                )
            
            scores = rerank_with_alibaba(request.query, request.documents)
            actual_type = "alibaba"
            logger.info(f"✅ Using Alibaba reranker (gte-rerank-v2) for {len(request.documents)} documents")
            
        else:
            # Default to MS-MARCO
            if ms_marco_reranker is None:
                load_ms_marco_reranker()
            if ms_marco_reranker is None:
                raise HTTPException(
                    status_code=503,
                    detail="MS-MARCO reranker is not available. Check service logs."
                )
            
            # Prepare query-document pairs
            pairs = [[request.query, doc] for doc in request.documents]
            # MS-MARCO returns scores (can be negative or positive)
            scores = ms_marco_reranker.predict(pairs)
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            else:
                scores = [float(s) for s in scores]
            actual_type = "ms-marco"
            logger.info(f"✅ Using MS-MARCO reranker for {len(request.documents)} documents")
        
        # Create results
        results = []
        for i, (doc, score) in enumerate(zip(request.documents, scores)):
            results.append(RerankResult(
                document=doc,
                index=i,
                relevance_score=float(score)
            ))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply top_k if specified
        if request.top_k is not None and request.top_k > 0:
            results = results[:request.top_k]
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"✅ Reranked {len(request.documents)} documents "
            f"({len(results)} returned) in {processing_time:.2f}ms using {actual_type} reranker"
        )
        
        logger.info(f"📊 RERANKER USED: {actual_type} (requested: {request.reranker_type or 'default'}, default config: {RERANKER_TYPE})")
        
        return RerankResponse(
            results=results,
            reranker_type=actual_type,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error during reranking with {reranker_type_to_use}: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@app.post("/switch-reranker")
async def switch_reranker(new_type: str):
    """
    Switch reranker type at runtime (for testing/comparison)
    
    Note: This requires restarting the service for full effect
    """
    global current_reranker, current_reranker_type
    
    if new_type.lower() not in ["bge", "ms-marco"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid reranker type. Use 'bge' or 'ms-marco'"
        )
    
    try:
        if new_type.lower() == "bge":
            current_reranker = load_bge_reranker()
            current_reranker_type = "bge"
        else:
            current_reranker = load_ms_marco_reranker()
            current_reranker_type = "ms-marco"
        
        return {
            "success": True,
            "message": f"Switched to {new_type} reranker",
            "reranker_type": current_reranker_type
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch reranker: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8008))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

