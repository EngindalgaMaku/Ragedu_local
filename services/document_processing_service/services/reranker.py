"""
Reranker Service
Uses cross-encoder model for document reranking
Supports both new reranker-service and legacy model-inference-service reranker
"""
import os
import requests
from typing import Dict, List, Any, Optional
from utils.logger import logger


class Reranker:
    """
    Document Reranker - Simplified version
    
    Uses a cross-encoder model via reranker-service (new) or model-inference-service (legacy)
    to rerank documents based on query relevance.
    
    This is a simplified version that only performs reranking without accept/reject/filter decisions.
    The reranker scores are used directly for sorting documents.
    """
    
    def __init__(self, model_inference_url: str, reranker_type: Optional[str] = None):
        """
        Initialize Reranker
        
        Args:
            model_inference_url: URL of model inference service (for legacy reranker)
            reranker_type: Optional reranker type override ("bge", "alibaba", or "ms-marco")
        """
        self.model_inference_url = model_inference_url
        self._session_reranker_type = reranker_type
        
        # Check if new reranker service should be used
        self.use_reranker_service = os.getenv("USE_RERANKER_SERVICE", "false").lower() == "true"
        self.reranker_service_url = os.getenv(
            "RERANKER_SERVICE_URL",
            "http://reranker-service:8008"
        )
        
        # Legacy reranker URL (model-inference-service)
        self.legacy_rerank_url = f"{self.model_inference_url}/rerank"
        
        # Determine which reranker to use
        if self.use_reranker_service:
            self.rerank_url = f"{self.reranker_service_url}/rerank"
            logger.info(f"✅ Reranker: Using NEW reranker-service at {self.rerank_url}")
        else:
            self.rerank_url = self.legacy_rerank_url
            logger.info(f"✅ Reranker: Using LEGACY reranker at {self.rerank_url}")
        
        # Get reranker type
        if self._session_reranker_type:
            self.reranker_type = self._session_reranker_type
            logger.info(f"Using reranker_type from session: {self.reranker_type}")
        else:
            self.reranker_type = self._get_reranker_type()
        
        logger.info(
            f"Reranker initialized with rerank URL: {self.rerank_url}, "
            f"type: {self.reranker_type}"
        )

    def _get_reranker_type(self) -> str:
        """Get reranker type from service"""
        try:
            if self.use_reranker_service:
                response = requests.get(f"{self.reranker_service_url}/info", timeout=5)
                if response.status_code == 200:
                    info = response.json()
                    return info.get("reranker_type", "alibaba")  # Default to Alibaba for new service
                return "alibaba"  # Default for new reranker service
            return "ms-marco"  # Default/legacy
        except Exception as e:
            logger.warning(f"Could not determine reranker type: {e}, using default")
            # If using new reranker service but can't connect, default to Alibaba
            if self.use_reranker_service:
                return "alibaba"
            return "ms-marco"

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of documents with content/text field
            
        Returns:
            Dictionary with reranking results:
            - reranked_docs: Documents sorted by rerank score (highest first)
            - scores: Individual document scores with metadata
            - max_score: Maximum rerank score
            - avg_score: Average rerank score
            - reranker_type: Type of reranker used
        """
        if not documents:
            return {
                "reranked_docs": [],
                "scores": [],
                "max_score": 0.0,
                "avg_score": 0.0,
                "reranker_type": self.reranker_type
            }

        # Prepare documents for the rerank service
        docs_to_rerank = [doc.get("content") or doc.get("text", "") for doc in documents]
        
        try:
            logger.info(
                f"▶️ Calling rerank service. "
                f"Query: '{query[:50]}...', Docs: {len(docs_to_rerank)}, "
                f"Service: {'NEW' if self.use_reranker_service else 'LEGACY'}"
            )
            
            # Call appropriate reranker service
            if self.use_reranker_service:
                # New reranker service - get reranker_type from session settings if available
                reranker_type = getattr(self, '_session_reranker_type', None)
                payload = {"query": query, "documents": docs_to_rerank}
                if reranker_type:
                    # Normalize reranker type (handle "gte-rerank-v2" as "alibaba")
                    if reranker_type in ["gte-rerank-v2", "alibaba"]:
                        payload["reranker_type"] = "alibaba"
                    else:
                        payload["reranker_type"] = reranker_type
                    logger.info(f"📤 Sending reranker request with type: {payload['reranker_type']} (original: {reranker_type})")
                else:
                    # Default to Alibaba when no session reranker_type is specified
                    payload["reranker_type"] = "alibaba"
                    logger.info("📤 No reranker_type in session, defaulting to Alibaba")
                
                response = requests.post(
                    self.rerank_url,
                    json=payload,
                    timeout=15
                )
            else:
                # Legacy reranker (model-inference-service)
                response = requests.post(
                    self.rerank_url,
                    json={"query": query, "documents": docs_to_rerank},
                    timeout=15
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if self.use_reranker_service:
                # New service format
                rerank_results = result.get("results", [])
                reranker_type = result.get("reranker_type", "ms-marco")
                # Normalize reranker type (handle "gte-rerank-v2" as "alibaba")
                if reranker_type in ["gte-rerank-v2", "alibaba"]:
                    reranker_type = "alibaba"
                logger.info(f"📥 Rerank service response: {len(rerank_results)} results, type={reranker_type}")
            else:
                # Legacy service format
                rerank_results = result.get("results", [])
                reranker_type = "ms-marco"  # Legacy always uses MS-MARCO
                logger.info(f"📥 Legacy rerank service returned {len(rerank_results)} results")
            
            # Update reranker type if changed
            if reranker_type != self.reranker_type:
                logger.info(f"🔄 Reranker type changed from {self.reranker_type} to {reranker_type}")
                self.reranker_type = reranker_type
            
            logger.info(f"✅ RERANKER CONFIRMED: Using {reranker_type} reranker")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ CRITICAL: Rerank service call failed: {e}. Returning original order.")
            # Fail open: if reranker fails, return documents in original order
            return {
                "reranked_docs": documents,
                "scores": [],
                "max_score": 0.0,
                "avg_score": 0.0,
                "reranker_type": self.reranker_type,
                "error": str(e)
            }

        # Process rerank results
        document_scores = []
        updated_docs = []
        for i, doc in enumerate(documents):
            # Find the corresponding rerank result
            rerank_score = 0.0
            for res in rerank_results:
                if res.get("index") == i:
                    rerank_score = res.get("relevance_score", 0.0)
                    break
            
            # Normalize score for frontend display (0-1 range)
            # Alibaba/BGE: already 0-1, MS-MARCO: normalize from -5 to +5 range to 0-1
            if reranker_type in ["bge", "alibaba", "gte-rerank-v2"]:
                # Alibaba and BGE scores are already in 0-1 range
                normalized_score = max(0.0, min(1.0, rerank_score))
            else:
                # MS-MARCO scores: typically -5 to +5, normalize to 0-1
                normalized_score = max(0.0, min(1.0, (rerank_score + 5) / 10))
            
            # Store rerank scores in document
            doc["rerank_score"] = normalized_score  # For frontend display (0-1)
            doc["rerank_score_raw"] = rerank_score  # Raw score
            doc["metadata"]["reranker_type"] = reranker_type  # Store reranker type
            
            updated_docs.append(doc)

            # Get chunk content preview for debugging
            chunk_content = doc.get("content", "") or doc.get("text", "")
            content_preview = chunk_content[:150] + "..." if len(chunk_content) > 150 else chunk_content
            
            document_scores.append({
                "index": i,
                "score": round(rerank_score, 4),
                "normalized_score": round(normalized_score, 4),
                "content_preview": content_preview,
                "original_similarity": doc.get("score", 0.0),
                "reranker_type": reranker_type
            })

        # Sort documents by rerank score (highest first)
        updated_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        # Calculate statistics
        scores = [s["score"] for s in document_scores]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        logger.info(
            f"📊 Rerank completed: max_score={max_score:.4f}, "
            f"avg_score={avg_score:.4f}, documents={len(updated_docs)}"
        )
        
        return {
            "reranked_docs": updated_docs,
            "scores": document_scores,
            "max_score": max_score,
            "avg_score": avg_score,
            "reranker_type": reranker_type
        }



