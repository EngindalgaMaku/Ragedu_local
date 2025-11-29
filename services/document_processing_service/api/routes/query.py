"""
RAG query endpoints
"""
import requests
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.schemas import RAGQueryRequest, RAGQueryResponse, RetrieveRequest, RetrieveResponse
from core.chromadb_client import get_chroma_client
from core.embedding_service import get_embeddings_direct
from services.hybrid_search import perform_hybrid_search
from services.reranker import Reranker
from utils.helpers import format_collection_name
from utils.logger import logger
from config import MODEL_INFERENCER_URL, DEFAULT_EMBEDDING_MODEL
import os

router = APIRouter()


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    RAG Query endpoint
    
    Workflow:
    1. Generate query embedding
    2. Search ChromaDB (semantic)
    3. Optional: Hybrid search (semantic + BM25)
    4. Optional: CRAG evaluation
    5. Generate answer using LLM
    6. Optional: Self-correction
    
    Features:
    - Hybrid search support
    - CRAG quality evaluation
    - Conversation history
    - Multi-model fallback
    """
    try:
        logger.info(f"🔍 RAG query received for session: {request.session_id}")
        chain_type = (request.chain_type or "stuff").lower()
        
        # Get session RAG settings from API Gateway to use correct model
        session_rag_settings = {}
        try:
            import os
            API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
            session_response = requests.get(
                f"{API_GATEWAY_URL}/sessions/{request.session_id}",
                timeout=5
            )
            if session_response.status_code == 200:
                session_data = session_response.json()
                session_rag_settings = session_data.get("rag_settings", {}) or {}
                logger.info(f"✅ Loaded session RAG settings: model={session_rag_settings.get('model')}, embedding_model={session_rag_settings.get('embedding_model')}")
            else:
                logger.warning(f"⚠️ Could not load session settings: {session_response.status_code}")
        except Exception as settings_err:
            logger.warning(f"⚠️ Error loading session RAG settings: {settings_err}")
        
        # Determine effective model: request > session settings > default
        effective_model = request.model or session_rag_settings.get("model") or None
        effective_embedding_model = request.embedding_model or session_rag_settings.get("embedding_model")
        
        if effective_model:
            logger.info(f"🔍 Using model: {effective_model} (from request: {request.model}, session: {session_rag_settings.get('model')})")
        if effective_embedding_model:
            logger.info(f"🔍 Using embedding model: {effective_embedding_model}")
        
        # Step 1: Find collection
        client = get_chroma_client()
        collection_name = format_collection_name(request.session_id, add_timestamp=False)
        
        # Try to get collection (with timestamped alternatives)
        collection = _find_collection_with_alternatives(client, collection_name, request.session_id)
        
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection not found for session {request.session_id}")
        
        logger.info(f"✅ Found collection: {collection.name}")
        
        # Step 2: Check collection's embedding dimension and model
        collection_dimension = None
        collection_embedding_model = None
        
        # Strategy 1: Try to get embedding_model from metadata first (FASTEST, avoids NumPy issues)
        try:
            sample_meta = collection.get(limit=1, include=["metadatas"])
            if sample_meta is not None and 'metadatas' in sample_meta:
                metadatas_raw = sample_meta['metadatas']
                # Convert to list safely
                import numpy as np
                if isinstance(metadatas_raw, np.ndarray):
                    metadatas_list = metadatas_raw.tolist()
                elif isinstance(metadatas_raw, (list, tuple)):
                    metadatas_list = list(metadatas_raw)
                else:
                    metadatas_list = []
                
                if len(metadatas_list) > 0 and isinstance(metadatas_list[0], dict):
                    collection_embedding_model = metadatas_list[0].get('embedding_model')
                    if collection_embedding_model:
                        logger.info(f"🔍 Found embedding model in metadata: {collection_embedding_model}")
                        
                        # Map model name to dimension (FASTEST approach)
                        model_lower = collection_embedding_model.lower()
                        if 'text-embedding-v4' in model_lower or 'text-embedding-v3' in model_lower or 'text-embedding-v2' in model_lower:
                            collection_dimension = 1024  # FIXED: All Alibaba v2/v3/v4 are 1024D
                        elif 'nomic-embed' in model_lower:
                            collection_dimension = 768
                        elif 'all-mpnet-base-v2' in model_lower:
                            collection_dimension = 768
                        elif 'all-minilm' in model_lower or 'bge-small' in model_lower:
                            collection_dimension = 384
                        
                        if collection_dimension:
                            logger.info(f"📏 Collection dimension (from model): {collection_dimension}D")
        except Exception as meta_err:
            logger.warning(f"⚠️ Error getting metadata: {meta_err}")
        
        # Strategy 2: If dimension still unknown, try to get from embeddings directly
        if not collection_dimension:
            try:
                sample_emb = collection.get(limit=1, include=["embeddings"])
                if sample_emb is not None and 'embeddings' in sample_emb:
                    embeddings_raw = sample_emb['embeddings']
                    # Convert to list safely
                    import numpy as np
                    if isinstance(embeddings_raw, np.ndarray):
                        embeddings_list = embeddings_raw.tolist()
                    elif isinstance(embeddings_raw, (list, tuple)):
                        embeddings_list = list(embeddings_raw)
                    else:
                        embeddings_list = []
                    
                    if len(embeddings_list) > 0:
                        first_emb = embeddings_list[0]
                        if isinstance(first_emb, np.ndarray):
                            first_emb = first_emb.tolist()
                        elif not isinstance(first_emb, (list, tuple)):
                            first_emb = list(first_emb) if hasattr(first_emb, '__iter__') and not isinstance(first_emb, (str, bytes)) else []
                        
                        if isinstance(first_emb, (list, tuple)) and len(first_emb) > 0:
                            collection_dimension = len(first_emb)
                            logger.info(f"📏 Collection dimension (from embedding): {collection_dimension}D")
            except Exception as emb_err:
                logger.warning(f"⚠️ Error getting embedding dimension: {emb_err}")
        
        # Step 3: Get query embeddings
        # FORCE default model (text-embedding-v4) - ignore collection metadata completely
        default_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        
        # Only use request.embedding_model if explicitly provided, otherwise ALWAYS use default
        if request.embedding_model:
            preferred_model = request.embedding_model
            logger.info(f"🔍 Using explicitly requested embedding model: {preferred_model}")
        else:
            # FORCE default model - completely ignore collection metadata
            preferred_model = default_model
            logger.info(f"🔍 FORCING default embedding model: {preferred_model} (ignoring collection metadata: {collection_embedding_model})")
            if collection_dimension:
                logger.info(f"🔍 Collection dimension: {collection_dimension}D, but using default model: {preferred_model}")
        
        logger.info(f"🔍 Getting query embeddings using model: {preferred_model}")
        query_embeddings = _get_query_embeddings_with_fallback(
            request.query, 
            preferred_model,
            required_dimension=collection_dimension
        )
        
        # Step 3: Semantic search
        n_results_fetch = request.top_k * 3 if request.use_hybrid_search else request.top_k
        search_results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results_fetch
        )
        
        documents = search_results.get('documents', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]
        
        logger.info(f"🔍 Semantic search: {len(documents)} documents found")
        
        # Step 4: Hybrid search (optional)
        if request.use_hybrid_search and len(documents) > 0:
            hybrid_result = perform_hybrid_search(
                query=request.query,
                documents=documents,
                distances=distances,
                top_k=request.top_k,
                bm25_weight=request.bm25_weight
            )
            
            if hybrid_result["reranked_indices"]:
                # Reorder results
                indices = hybrid_result["reranked_indices"]
                documents = [documents[i] for i in indices]
                metadatas = [metadatas[i] for i in indices]
                distances = [distances[i] for i in indices]
        
        # Step 5: Format context documents with keyword filtering and title boosting
        context_docs = _format_context_docs(documents, metadatas, distances, collection.name, query=request.query)
        
        if not context_docs:
            return RAGQueryResponse(
                answer="Üzgünüm, bu soruyla ilgili yeterli bilgi bulamadım.",
                sources=[],
                chain_type=chain_type
            )
        
        # Step 6: Rerank (optional)
        if request.use_rerank and context_docs:
            logger.info(f"🔍 Rerank enabled: Applying reranking to {len(context_docs)} documents")
            context_docs = _apply_rerank(request.query, context_docs)
        else:
            logger.info(f"⏭️ Rerank disabled: Skipping reranking")
        
        # Step 7: Generate answer (skip if skip_llm=True)
        if request.skip_llm:
            logger.info(f"⏭️ Skipping LLM generation (skip_llm=True), returning only chunks")
            return RAGQueryResponse(
                answer="",  # Empty answer when skip_llm=True
                sources=context_docs,  # Return chunks as sources
                chain_type=chain_type
            )
        
        # Step 7: Generate answer
        answer, sources = _generate_answer_with_llm(
            query=request.query,
            context_docs=context_docs,
            model=effective_model,  # Use effective model from session settings
            max_tokens=request.max_tokens,
            conversation_history=request.conversation_history,
            chain_type=chain_type,
            max_context_chars=request.max_context_chars
        )
        
        return RAGQueryResponse(
            answer=answer,
            sources=sources,
            chain_type=chain_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RAG query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG query error: {str(e)}")


def _find_collection_with_alternatives(client, collection_name: str, session_id: str):
    """Find collection with alternative naming patterns including UUID formats"""
    try:
        return client.get_collection(name=collection_name)
    except:
        # Try alternatives including timestamped and UUID formats
        alternative_names = []
        search_patterns = [collection_name]
        
        # Convert between UUID formats (with/without dashes)
        if '-' in collection_name:
            # Has dashes, try without
            search_patterns.append(collection_name.replace('-', ''))
        elif len(collection_name) == 32:
            # No dashes, try with dashes (UUID format)
            uuid_format = f"{collection_name[:8]}-{collection_name[8:12]}-{collection_name[12:16]}-{collection_name[16:20]}-{collection_name[20:]}"
            search_patterns.append(uuid_format)
        
        try:
            all_collections = client.list_collections()
            all_collection_names = [c.name for c in all_collections]
            logger.info(f"🔍 Searching in {len(all_collection_names)} collections for patterns: {search_patterns}")
            
            # Search for exact matches and timestamped versions
            for pattern in search_patterns:
                # Try exact match
                if pattern in all_collection_names:
                    try:
                        logger.info(f"✅ Found exact match: {pattern}")
                        return client.get_collection(name=pattern)
                    except:
                        pass
                
                # Search for timestamped versions (pattern_TIMESTAMP)
                for coll_name in all_collection_names:
                    if coll_name.startswith(pattern + "_"):
                        suffix = coll_name[len(pattern)+1:]
                        if suffix.isdigit():
                            alternative_names.append((coll_name, int(suffix)))
            
            # Sort by timestamp (newest first)
            alternative_names.sort(key=lambda x: x[1], reverse=True)
            
            if alternative_names:
                logger.info(f"🔍 Found {len(alternative_names)} timestamped alternatives")
            
            for alt_name, timestamp in alternative_names:
                try:
                    logger.info(f"✅ Trying timestamped collection: {alt_name}")
                    return client.get_collection(name=alt_name)
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error finding collection alternatives: {e}")
        
        return None


def _get_query_embeddings_with_fallback(
    query: str, 
    preferred_model: str,
    required_dimension: int = None
) -> List[List[float]]:
    """Get query embeddings with multi-model fallback, checking dimension match"""
    # Define models by dimension
    models_by_dimension = {
        1024: ["text-embedding-v4", "text-embedding-v3", "text-embedding-v2"],  # Alibaba 1024D (v4 is also 1024D)
        768: ["nomic-embed-text", "sentence-transformers/all-mpnet-base-v2"],  # 768D
        384: ["nomic-embed-text", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],  # 384D
    }
    
    default_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
    
    if required_dimension:
        # CRITICAL: Only use models with matching dimension
        logger.info(f"⚠️ Required dimension: {required_dimension}D. Filtering models by dimension...")
        matching_models = models_by_dimension.get(required_dimension, [])
        
        # FORCE default model first - ignore dimension mismatch
        models_to_try = []
        
        # First, ALWAYS try default model (text-embedding-v4) - FORCE IT
        models_to_try.append(default_model)
        logger.info(f"🔄 FORCING default model first: {default_model} (ignoring dimension mismatch)")
        
        # Then try preferred model ONLY if it was explicitly requested in the request
        # (preferred_model should already be default_model unless request.embedding_model was set)
        if preferred_model != default_model:
            models_to_try.append(preferred_model)
            logger.info(f"🔄 Will also try preferred model: {preferred_model}")
        
        # Then add matching models for the required dimension (excluding default and preferred)
        for m in matching_models:
            if m != default_model and m != preferred_model and m not in models_to_try:
                models_to_try.append(m)
        
        if not models_to_try:
            raise Exception(
                f"❌ No embedding models available for {required_dimension}D dimension. "
                f"Please use a model that produces {required_dimension}D embeddings."
            )
        
        logger.info(f"✅ Will try {len(models_to_try)} models with {required_dimension}D: {', '.join(models_to_try)}")
    else:
        # Unknown dimension, try preferred model first, then default, then common models
        logger.warning("⚠️ Required dimension unknown. Trying preferred model and common fallbacks...")
        default_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        models_to_try = [preferred_model]
        # Add default model if different from preferred
        if default_model != preferred_model:
            models_to_try.append(default_model)
        models_to_try.extend([
            "text-embedding-v3",  # Try Alibaba 1024D
            "text-embedding-v2",  # Try Alibaba 1024D
            "nomic-embed-text",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5"
        ])
        models_to_try = list(dict.fromkeys(models_to_try))
    
    for model in models_to_try:
        try:
            logger.info(f"🔄 Trying embedding model: {model}")
            embeddings = get_embeddings_direct([query], model)
            if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
                embedding_dimension = len(embeddings[0])
                
                # Check dimension match if required
                # BUT: If this is the preferred/default model (text-embedding-v4), try to use it anyway
                # The collection might need to be recreated, but we'll try first
                if required_dimension and embedding_dimension != required_dimension:
                    if model == default_model or model == preferred_model:
                        logger.warning(
                            f"⚠️ Dimension mismatch with {model}: "
                            f"Collection requires {required_dimension}D, but got {embedding_dimension}D. "
                            f"But this is the default model, so we'll use it anyway. "
                            f"⚠️ WARNING: Collection should be recreated with {model} for proper matching."
                        )
                        # Use it anyway - collection needs to be recreated
                    else:
                        logger.warning(
                            f"⚠️ Dimension mismatch with {model}: "
                            f"Collection requires {required_dimension}D, but got {embedding_dimension}D. Trying next model..."
                        )
                        continue
                
                logger.info(f"✅ Got query embeddings using {model} (dimension: {embedding_dimension}D)")
                return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Failed with {model}: {e}")
            continue
    
    error_msg = "Failed to generate query embeddings with any model"
    if required_dimension:
        error_msg += f". Collection requires {required_dimension}D embeddings."
    raise Exception(error_msg)


def _format_context_docs(documents: List[str], metadatas: List[Dict], distances: List[float], collection_name: str, query: str = None) -> List[Dict[str, Any]]:
    """
    Format documents for context with keyword filtering and title boosting
    
    Args:
        documents: List of document contents
        metadatas: List of metadata dicts
        distances: List of similarity distances
        collection_name: Collection name for security check
        query: Optional query string for keyword filtering and title boosting
    """
    context_docs = []
    query_lower = query.lower() if query else ""
    
    # Extract keywords from query (remove stopwords)
    query_keywords = []
    if query:
        import re
        # Turkish stopwords
        stopwords = {'nedir', 'neden', 'nasıl', 'ne', 'hangi', 'kim', 'nerede', 'ne zaman', 'ile', 've', 'veya', 'için', 'gibi', 'kadar', 'daha', 'çok', 'az', 'bir', 'bu', 'şu', 'o', 'da', 'de', 'ki', 'mi', 'mı', 'mu', 'mü'}
        words = re.findall(r'\b\w+\b', query_lower)
        query_keywords = [w for w in words if w not in stopwords and len(w) > 2]
        logger.info(f"🔑 Extracted keywords from query: {query_keywords}")
    
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        
        # Security check: verify session_id
        if metadata.get("session_id") and metadata.get("session_id") != collection_name:
            logger.warning(f"⚠️ SECURITY: Mismatched session_id, skipping document {i}")
            continue
        
        # Calculate base similarity score
        distance = distances[i] if i < len(distances) else float('inf')
        base_similarity = max(0.0, 1.0 - distance) if distance != float('inf') else 0.0
        
        # Title boosting: Check if query keywords appear in chunk title
        title_boost = 0.0
        chunk_title = metadata.get("chunk_title", "").lower()
        if query_keywords and chunk_title:
            title_matches = sum(1 for kw in query_keywords if kw in chunk_title)
            if title_matches > 0:
                # Boost: +0.1 per keyword match in title, max +0.3
                title_boost = min(0.3, title_matches * 0.1)
                logger.debug(f"📌 Title boost for doc {i}: +{title_boost:.2f} (matches: {title_matches})")
        
        # Keyword filtering: Check if query keywords appear in content
        content_boost = 0.0
        content_lower = doc.lower()
        if query_keywords:
            content_matches = sum(1 for kw in query_keywords if kw in content_lower)
            if content_matches > 0:
                # Boost: +0.05 per keyword match in content, max +0.2
                content_boost = min(0.2, content_matches * 0.05)
                logger.debug(f"🔍 Content boost for doc {i}: +{content_boost:.2f} (matches: {content_matches})")
        
        # Negative keyword filtering: Penalize chunks with opposite keywords
        negative_penalty = 0.0
        if query_keywords:
            # Detect opposite keywords (e.g., "eşeyli" vs "eşeysiz")
            opposite_patterns = {
                'eşeyli': 'eşeysiz',
                'eşeysiz': 'eşeyli',
                'olumlu': 'olumsuz',
                'olumsuz': 'olumlu',
                'artı': 'eksi',
                'eksi': 'artı'
            }
            for kw in query_keywords:
                if kw in opposite_patterns:
                    opposite_kw = opposite_patterns[kw]
                    if opposite_kw in content_lower or opposite_kw in chunk_title:
                        # Penalty: -0.2 for opposite keyword
                        negative_penalty = -0.2
                        logger.debug(f"⚠️ Negative penalty for doc {i}: {negative_penalty:.2f} (opposite: {opposite_kw})")
                        break
        
        # Calculate final score with boosts and penalties
        final_score = base_similarity + title_boost + content_boost + negative_penalty
        final_score = max(0.0, min(1.0, final_score))  # Clamp to 0-1
        
        context_docs.append({
            "content": doc,
            "metadata": metadata,
            "score": final_score,
            "base_score": base_similarity,
            "title_boost": title_boost,
            "content_boost": content_boost,
            "negative_penalty": negative_penalty
        })
    
    # Sort by final score (descending)
    context_docs.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(f"📊 Formatted {len(context_docs)} documents with keyword filtering and title boosting")
    if query_keywords:
        logger.info(f"🔑 Applied keyword filtering with {len(query_keywords)} keywords")
    
    return context_docs


def _apply_crag_evaluation(query: str, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply CRAG evaluation to filter/improve documents"""
    try:
        crag_evaluator = CRAGEvaluator(model_inference_url=MODEL_INFERENCER_URL)
        evaluation_result = crag_evaluator.evaluate_retrieved_docs(
            query=query,
            retrieved_docs=context_docs
        )
        
        action = evaluation_result.get("action", "accept")
        filtered_docs = evaluation_result.get("filtered_docs", context_docs)
        
        logger.info(f"🔍 CRAG {action.upper()}: {len(filtered_docs)}/{len(context_docs)} docs")
        
        return filtered_docs if filtered_docs else context_docs
        
    except Exception as e:
        logger.warning(f"⚠️ CRAG evaluation failed: {e}, using all documents")
        return context_docs


def _generate_answer_with_llm(
    query: str,
    context_docs: List[Dict[str, Any]],
    model: str = None,
    max_tokens: int = 2048,
    conversation_history: List[Dict[str, str]] = None,
    chain_type: str = "stuff",
    max_context_chars: int = 8000
) -> tuple[str, List[Dict[str, Any]]]:
    """Generate answer using LLM via Model Inference Service"""
    try:
        # Build context
        context_parts = []
        total_chars = 0
        sources = []
        
        for doc in context_docs:
            content = doc["content"]
            if total_chars + len(content) > max_context_chars:
                break
            context_parts.append(content)
            sources.append({
                "content": content,  # Send full content for source modal
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0)
            })
            total_chars += len(content)
        
        context = "\n\n".join(context_parts)
        
        # Simple and direct prompt - no meta-analysis, just answer from context
        full_prompt = f"Aşağıdaki bilgileri kullanarak soruyu cevapla:\n\n{context}\n\nSoru: {query}\n\nCevap:"
        
        # Call LLM using /models/generate endpoint
        generate_url = f"{MODEL_INFERENCER_URL}/models/generate"
        payload = {
            "prompt": full_prompt,
            "model": model or "llama-3.1-8b-instant",
            "temperature": 0.7,
            "max_tokens": max_tokens
        }
        
        logger.info(f"🤖 Calling LLM at {generate_url} with model: {payload['model']}")
        response = requests.post(generate_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "Cevap oluşturulamadı.")
            logger.info(f"✅ Generated answer ({len(answer)} chars)")
            return answer, sources
        else:
            logger.error(f"❌ LLM generation failed: {response.status_code}")
            try:
                error_detail = response.json()
                logger.error(f"   Error details: {error_detail}")
            except:
                logger.error(f"   Response text: {response.text[:200]}")
            return "Üzgünüm, cevap oluştururken bir hata oluştu.", sources
            
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return f"Üzgünüm, bir hata oluştu: {str(e)}", []


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """
    Retrieve documents without generation - for testing RAG retrieval quality.
    Returns only the retrieved documents with their scores.
    
    This endpoint is useful for:
    - Testing retrieval quality
    - Debugging RAG performance
    - Analyzing document relevance
    """
    try:
        logger.info(f"🔍 Retrieve request: collection={request.collection_name}, query={request.query[:50]}...")
        
        # Step 1: Find collection
        client = get_chroma_client()
        collection = _find_collection_with_alternatives(client, request.collection_name, request.collection_name)
        
        if not collection:
            logger.error(f"❌ Collection not found: {request.collection_name}")
            return RetrieveResponse(success=False, results=[], total=0)
        
        logger.info(f"✅ Found collection: {collection.name}")
        
        # Step 2: Determine embedding model from collection metadata
        embedding_model = request.embedding_model
        
        if not embedding_model:
            # Get a sample document to detect the embedding model used
            try:
                sample = collection.get(limit=1, include=["metadatas"])
                if sample and sample.get('metadatas') and len(sample['metadatas']) > 0:
                    metadata = sample['metadatas'][0]
                    embedding_model = metadata.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
                    logger.info(f"📊 Detected embedding model from collection: {embedding_model}")
                else:
                    embedding_model = DEFAULT_EMBEDDING_MODEL
            except Exception as e:
                logger.warning(f"⚠️ Could not detect embedding model: {e}")
                embedding_model = DEFAULT_EMBEDDING_MODEL
        
        # Step 3: Get query embeddings with the correct model
        query_embeddings = _get_query_embeddings_with_fallback(request.query, embedding_model)
        
        # Step 3: Query the collection
        search_results = collection.query(
            query_embeddings=query_embeddings,
            n_results=request.top_k
        )
        
        # Step 4: Extract and format results
        documents = search_results.get('documents', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]
        
        results = []
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else float('inf')
            
            # Convert distance to similarity score (1 - distance for cosine)
            similarity_score = max(0.0, 1.0 - distance) if distance != float('inf') else 0.0
            
            results.append({
                "text": doc,
                "score": round(similarity_score, 4),
                "metadata": metadata,
                "distance": round(distance, 4)
            })
        
        logger.info(f"✅ Retrieved {len(results)} documents")
        
        return RetrieveResponse(
            success=True,
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"❌ Error in retrieve endpoint: {e}")
        return RetrieveResponse(success=False, results=[], total=0)


def _apply_rerank(query: str, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply reranking to context documents using reranker service.
    
    Args:
        query: Search query
        context_docs: List of context documents to rerank
        
    Returns:
        Reranked list of context documents
    """
    try:
        reranker = Reranker(model_inference_url=MODEL_INFERENCER_URL)
        rerank_result = reranker.rerank_documents(query=query, documents=context_docs)
        
        if rerank_result.get("reranked_docs"):
            logger.info(f"✅ Rerank completed: {len(rerank_result['reranked_docs'])} documents reranked")
            return rerank_result["reranked_docs"]
        else:
            logger.warning("⚠️ Rerank returned no documents, using original order")
            return context_docs
    except Exception as e:
        logger.error(f"❌ Rerank error: {e}, using original order")
        return context_docs


# TODO: Add self-correction endpoint
# TODO: Add streaming support
# TODO: Add conversation memory management

