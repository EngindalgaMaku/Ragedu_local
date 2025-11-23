"""
Hybrid Search: Semantic + BM25 Keyword Search
Uses Reciprocal Rank Fusion (RRF) for combining results
"""
from typing import List, Dict, Any, Optional
from core.turkish_utils import tokenize_turkish
from utils.logger import logger

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None
    logger.warning("⚠️ BM25 not available - install rank-bm25")


def perform_hybrid_search(
    query: str,
    documents: List[str],
    distances: List[float],
    top_k: int = 5,
    bm25_weight: float = 0.3
) -> Dict[str, Any]:
    """
    Perform hybrid search combining semantic and BM25 keyword search
    
    Uses Reciprocal Rank Fusion (RRF) algorithm:
    - More robust than weighted average
    - Handles different score distributions well
    - Formula: score = 1 / (k + rank)
    
    Args:
        query: Search query
        documents: List of document texts
        distances: Cosine distances from semantic search
        top_k: Number of results to return
        bm25_weight: Weight for BM25 component (0.3 = 30% keyword, 70% semantic)
        
    Returns:
        Dictionary with:
        - reranked_indices: Indices of top-k documents
        - hybrid_scores: Detailed scores for each document
    """
    if not BM25_AVAILABLE or not documents:
        logger.warning("BM25 not available or no documents provided")
        return {
            "reranked_indices": list(range(min(top_k, len(documents)))),
            "hybrid_scores": None
        }
    
    try:
        logger.info("🔥 Applying HYBRID SEARCH: Semantic + BM25")
        
        # Tokenize query for BM25
        query_tokens = tokenize_turkish(query, remove_stopwords=True)
        logger.info(f"🔍 Query tokens (stopwords removed): {query_tokens}")
        
        # Tokenize all documents for BM25
        tokenized_docs = [tokenize_turkish(doc, remove_stopwords=True) for doc in documents]
        
        # Calculate BM25 scores
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        normalized_bm25_scores = [score / max_bm25 for score in bm25_scores]
        
        # Calculate semantic similarity scores from distances
        semantic_scores = []
        for distance in distances:
            if distance == float('inf'):
                semantic_scores.append(0.0)
            else:
                semantic_scores.append(max(0.0, 1.0 - distance))
        
        # --- Reciprocal Rank Fusion (RRF) ---
        
        # 1. Get Ranks for Semantic Scores
        semantic_pairs = [(i, s) for i, s in enumerate(semantic_scores)]
        semantic_pairs.sort(key=lambda x: x[1], reverse=True)
        semantic_ranks = {index: rank for rank, (index, _) in enumerate(semantic_pairs)}
        
        # 2. Get Ranks for BM25 Scores
        bm25_pairs = [(i, s) for i, s in enumerate(bm25_scores)]
        bm25_pairs.sort(key=lambda x: x[1], reverse=True)
        bm25_ranks = {index: rank for rank, (index, _) in enumerate(bm25_pairs)}
        
        # 3. Calculate RRF Scores
        rrf_k = 60  # Standard constant for RRF
        hybrid_scores = []
        
        for i in range(len(documents)):
            sem_rank = semantic_ranks.get(i, len(documents))
            kw_rank = bm25_ranks.get(i, len(documents))
            
            # RRF Score calculation with weights
            rrf_sem = (1.0 / (rrf_k + sem_rank))
            rrf_bm25 = (1.0 / (rrf_k + kw_rank))
            
            # Apply weights
            w_sem = 1.0 - bm25_weight
            w_bm25 = bm25_weight
            
            final_score = (w_sem * rrf_sem) + (w_bm25 * rrf_bm25)
            
            hybrid_scores.append({
                'index': i,
                'hybrid_score': final_score,
                'semantic_score': semantic_scores[i],
                'bm25_score': bm25_scores[i] if i < len(bm25_scores) else 0.0,
                'semantic_rank': sem_rank,
                'bm25_rank': kw_rank
            })
        
        # Sort by hybrid score (descending) and take top_k
        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_k_indices = [item['index'] for item in hybrid_scores[:top_k]]
        
        logger.info(f"✅ HYBRID SEARCH: Reranked to top {top_k} documents")
        logger.info(f"📊 Top 3 hybrid scores: {[(s['hybrid_score'], s['semantic_score'], s['bm25_score']) for s in hybrid_scores[:3]]}")
        
        return {
            "reranked_indices": top_k_indices,
            "hybrid_scores": hybrid_scores[:top_k]
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Hybrid search failed: {e}")
        # Fallback to semantic only
        return {
            "reranked_indices": list(range(min(top_k, len(documents)))),
            "hybrid_scores": None
        }






