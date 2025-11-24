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
from services.crag_evaluator import CRAGEvaluator
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
        
        # Step 1: Find collection
        client = get_chroma_client()
        collection_name = format_collection_name(request.session_id, add_timestamp=False)
        
        # Try to get collection (with timestamped alternatives)
        collection = _find_collection_with_alternatives(client, collection_name, request.session_id)
        
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection not found for session {request.session_id}")
        
        logger.info(f"✅ Found collection: {collection.name}")
        
        # Step 2: Detect embedding model from collection if not provided
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
        
        # Step 2.5: Verify query embedding dimension matches collection
        query_embeddings = _get_query_embeddings_with_fallback(request.query, embedding_model)
        
        # Check dimension compatibility
        try:
            sample_embeddings = collection.get(limit=1, include=["embeddings"])
            if sample_embeddings and sample_embeddings.get('embeddings') and len(sample_embeddings['embeddings']) > 0:
                collection_dim = len(sample_embeddings['embeddings'][0])
                query_dim = len(query_embeddings[0]) if query_embeddings and len(query_embeddings) > 0 else 0
                if collection_dim != query_dim:
                    error_msg = (
                        f"❌ EMBEDDING DIMENSION MISMATCH: "
                        f"Collection has embeddings with dimension {collection_dim}, "
                        f"but query embedding has dimension {query_dim}. "
                        f"Please use the same embedding model that was used to create the collection."
                    )
                    logger.error(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"⚠️ Could not verify embedding dimension: {e}")
        
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
        
        # Step 5: Format context documents
        context_docs = _format_context_docs(documents, metadatas, distances, collection.name)
        
        if not context_docs:
            return RAGQueryResponse(
                answer="Üzgünüm, bu soruyla ilgili yeterli bilgi bulamadım.",
                sources=[],
                chain_type=chain_type
            )
        
        # Step 6: CRAG evaluation (optional)
        if request.use_rerank:
            logger.info(f"🔍 CRAG enabled: Applying CRAG evaluation")
            context_docs = _apply_crag_evaluation(request.query, context_docs)
        else:
            logger.info(f"⏭️ CRAG disabled: Skipping CRAG evaluation")
        
        # Step 7: Generate answer
        answer, sources = _generate_answer_with_llm(
            query=request.query,
            context_docs=context_docs,
            model=request.model,
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


def _get_query_embeddings_with_fallback(query: str, preferred_model: str) -> List[List[float]]:
    """Get query embeddings with multi-model fallback"""
    models_to_try = [
        preferred_model,
        "nomic-embed-text",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5"
    ]
    
    for model in models_to_try:
        try:
            embeddings = get_embeddings_direct([query], model)
            if embeddings and len(embeddings) > 0:
                logger.info(f"✅ Got query embeddings using {model}")
                return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Failed with {model}: {e}")
            continue
    
    raise Exception("Failed to generate query embeddings with any model")


def _format_context_docs(documents: List[str], metadatas: List[Dict], distances: List[float], collection_name: str) -> List[Dict[str, Any]]:
    """Format documents for context"""
    context_docs = []
    
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        
        # Security check: verify session_id
        if metadata.get("session_id") and metadata.get("session_id") != collection_name:
            logger.warning(f"⚠️ SECURITY: Mismatched session_id, skipping document {i}")
            continue
        
        # Calculate similarity score
        distance = distances[i] if i < len(distances) else float('inf')
        similarity = max(0.0, 1.0 - distance) if distance != float('inf') else 0.0
        
        context_docs.append({
            "content": doc,
            "metadata": metadata,
            "score": similarity
        })
    
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
        # Include markdown formatting instructions
        full_prompt = f"""Aşağıdaki bilgileri kullanarak soruyu cevapla. MARKDOWN FORMATI KULLAN: Önemli kavramları **kalın** yaz, listeler için `-` veya `*` kullan, kod için `backtick` kullan, başlıklar için `##` kullan.

{context}

Soru: {query}

Cevap (markdown formatında, formatlı ve okunabilir):"""
        
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


# TODO: Add self-correction endpoint
# TODO: Add streaming support
# TODO: Add conversation memory management

