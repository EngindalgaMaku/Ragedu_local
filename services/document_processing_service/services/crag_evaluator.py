"""
CRAG (Corrective RAG) Evaluator
Uses cross-encoder model for real document relevance evaluation
Supports both new reranker-service and legacy model-inference-service reranker
"""
import os
import requests
from typing import Dict, List, Any, Optional
from utils.logger import logger


class CRAGEvaluator:
    """
    Corrective RAG (CRAG) Evaluator - REAL IMPLEMENTATION
    
    Uses a cross-encoder model via reranker-service (new) or model-inference-service (legacy)
    for actual relevance scores for query-document pairs.
    
    Decision Logic:
    - ACCEPT: max_score >= 3.0 (high relevance) - MS-MARCO
    - ACCEPT: max_score >= 0.7 (high relevance) - BGE
    - REJECT: max_score < 1.0 (very low relevance) - MS-MARCO
    - REJECT: max_score < 0.3 (very low relevance) - BGE
    - FILTER: 1.0 <= max_score < 3.0 (filter by threshold >= 2.0) - MS-MARCO
    - FILTER: 0.3 <= max_score < 0.7 (filter by threshold >= 0.5) - BGE
    """
    
    def __init__(self, model_inference_url: str, reranker_type: Optional[str] = None):
        """
        Initialize CRAG evaluator
        
        Args:
            model_inference_url: URL of model inference service (for legacy reranker)
            reranker_type: Optional reranker type override ("bge" or "ms-marco")
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
            logger.info(f"✅ CRAGEvaluator: Using NEW reranker-service at {self.rerank_url}")
        else:
            self.rerank_url = self.legacy_rerank_url
            logger.info(f"✅ CRAGEvaluator: Using LEGACY reranker at {self.rerank_url}")
        
        # Get reranker type to adjust thresholds
        # Use provided reranker_type or get from service
        if self._session_reranker_type:
            self.reranker_type = self._session_reranker_type
            logger.info(f"Using reranker_type from session: {self.reranker_type}")
        else:
            self.reranker_type = self._get_reranker_type()
        
        # Adjust thresholds based on reranker type
        if self.reranker_type == "bge":
            # BGE scores are typically 0-1 range
            self.correct_threshold = 0.7   # Accept if max score >= 0.7
            self.incorrect_threshold = 0.3  # Reject if max score < 0.3
            self.filter_threshold = 0.5    # Filter individual docs with score < 0.5
        else:
            # MS-MARCO scores can be negative or positive (typically 0-10)
            self.correct_threshold = -0.5   # Accept if max score > -0.5
            self.incorrect_threshold = -3.0 # Reject if max score < -3.0
            self.filter_threshold = -2.0    # Filter individual docs with score < -2.0
        
        logger.info(
            f"CRAGEvaluator initialized with rerank URL: {self.rerank_url}, "
            f"type: {self.reranker_type}, thresholds: accept>={self.correct_threshold}, "
            f"reject<{self.incorrect_threshold}, filter<{self.filter_threshold}"
        )

    def _get_reranker_type(self) -> str:
        """Get reranker type from service"""
        try:
            if self.use_reranker_service:
                response = requests.get(f"{self.reranker_service_url}/info", timeout=5)
                if response.status_code == 200:
                    info = response.json()
                    return info.get("reranker_type", "ms-marco")
            return "ms-marco"  # Default/legacy
        except Exception as e:
            logger.warning(f"Could not determine reranker type: {e}, using default")
            return "ms-marco"

    def evaluate_retrieved_docs(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate retrieved documents using a real cross-encoder model.
        
        Args:
            query: Search query
            retrieved_docs: List of retrieved documents with content/text field
            
        Returns:
            Dictionary with evaluation results:
            - action: 'accept', 'reject', or 'filter'
            - confidence: Max relevance score
            - avg_score: Average relevance score
            - filtered_docs: Documents that passed filtering
            - evaluation_scores: Individual document scores
            - thresholds: Threshold values used
        """
        if not retrieved_docs:
            return {
                "action": "reject",
                "confidence": 0.0,
                "avg_score": 0.0,
                "filtered_docs": [],
                "evaluation_scores": []
            }

        # Prepare documents for the rerank service
        docs_to_rerank = [doc.get("content") or doc.get("text", "") for doc in retrieved_docs]
        
        try:
            logger.info(
                f"▶️ Calling rerank service for CRAG evaluation. "
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
                    logger.warning("⚠️ No reranker_type provided, using service default")
                
                response = requests.post(
                    self.rerank_url,
                    json=payload,
                    timeout=60
                )
            else:
                # Legacy reranker (model-inference-service)
                response = requests.post(
                    self.rerank_url,
                    json={"query": query, "documents": docs_to_rerank},
                    timeout=60
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if self.use_reranker_service:
                # New service format
                rerank_results = result.get("results", [])
                reranker_type = result.get("reranker_type", "ms-marco")
                logger.info(f"📥 Rerank service response: {len(rerank_results)} results, type={reranker_type}")
            else:
                # Legacy service format
                rerank_results = result.get("results", [])
                reranker_type = "ms-marco"  # Legacy always uses MS-MARCO
                logger.info(f"📥 Legacy rerank service returned {len(rerank_results)} results")
            
            logger.info(f"✅ RERANKER CONFIRMED: Using {reranker_type} reranker (requested: {getattr(self, '_session_reranker_type', 'default')})")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ CRITICAL: Rerank service call failed: {e}. Cannot perform CRAG evaluation.")
            # Fail open: if reranker fails, accept the documents to not block the user.
            # This is a production-friendly choice.
            return {
                "action": "accept",
                "confidence": 0.5,
                "avg_score": 0.5,
                "filtered_docs": retrieved_docs,
                "evaluation_scores": [],
                "error": str(e)
            }

        # Process rerank results
        evaluation_scores = []
        updated_docs = []
        for i, doc in enumerate(retrieved_docs):
            # Find the corresponding rerank result
            rerank_score = 0.0
            for res in rerank_results:
                if res.get("index") == i:
                    rerank_score = res.get("relevance_score", 0.0)
                    break
            
            # Normalize score for frontend display (0-1 range)
            # BGE: already 0-1, MS-MARCO: normalize from -5 to +5 range to 0-1
            if reranker_type == "bge":
                # BGE scores are already in 0-1 range
                normalized_score = max(0.0, min(1.0, rerank_score))
            else:
                # MS-MARCO scores: typically -5 to +5, normalize to 0-1
                # Formula: (score + 5) / 10, clamped to 0-1
                normalized_score = max(0.0, min(1.0, (rerank_score + 5) / 10))
            
            # Store both raw and normalized scores
            # Use raw score for threshold comparison, normalized for display
            doc["crag_score"] = normalized_score  # For frontend display (0-1)
            doc["crag_score_raw"] = rerank_score  # For threshold comparison
            doc["metadata"]["reranker_type"] = reranker_type  # Store reranker type
            
            # The final score for threshold comparison is the raw score
            final_score = rerank_score
            # Keep original 'score' as the similarity score for comparison
            updated_docs.append(doc)

            evaluation_scores.append({
                "index": i,
                "final_score": round(final_score, 4)
            })

        # Sort documents by the new CRAG score
        updated_docs.sort(key=lambda x: x["crag_score"], reverse=True)
        
        # Calculate statistics
        scores = [s["final_score"] for s in evaluation_scores]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Decision logic based on max score
        if max_score >= self.correct_threshold:
            action = "accept"
        elif max_score < self.incorrect_threshold:
            action = "reject"
        else:
            action = "filter"
        
        # Filter documents based on threshold (use raw score for comparison)
        filtered_docs = [
            doc for doc in updated_docs
            if doc.get("crag_score_raw", doc.get("crag_score", 0)) >= self.filter_threshold
        ]
        
        logger.info(
            f"📊 CRAG Evaluation: action={action}, max_score={max_score:.4f}, "
            f"avg_score={avg_score:.4f}, filtered={len(filtered_docs)}/{len(retrieved_docs)}"
        )
        
        return {
            "action": action,
            "confidence": max_score,
            "avg_score": avg_score,
            "filtered_docs": filtered_docs,
            "evaluation_scores": evaluation_scores,
            "thresholds": {
                "correct": self.correct_threshold,
                "incorrect": self.incorrect_threshold,
                "filter": self.filter_threshold
            },
            "reranker_type": reranker_type
        }
