"""
Hybrid Knowledge Retriever
Combines chunk-based retrieval with structured knowledge base
KB-Enhanced RAG implementation
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import requests
import os
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Environment variables
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", "http://model-inference-service:8002")
DOCUMENT_PROCESSING_URL = os.getenv("DOCUMENT_PROCESSING_URL", "http://document-processing-service:8080")
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://chromadb-service:8000")


class HybridKnowledgeRetriever:
    """
    KB-Enhanced RAG Retriever
    
    Combines:
    1. Traditional chunk-based retrieval (vector search)
    2. Structured knowledge base (summaries, concepts)
    3. QA pairs matching (direct answers)
    
    Retrieval Strategy:
    - Query → Classify to topic
    - Retrieve chunks (traditional)
    - Check QA similarity (fast path)
    - Get KB summary (structured knowledge)
    - Merge and rank results
    """
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.qa_similarity_threshold = 0.85  # High similarity for direct answer
        self.kb_usage_threshold = 0.7  # Minimum topic classification confidence
    
    async def retrieve_for_query(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        use_kb: bool = True,
        use_qa_pairs: bool = True,
        embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval combining chunks + KB + QA
        
        Args:
            query: Student question
            session_id: Learning session ID
            top_k: Number of chunks to retrieve
            use_kb: Whether to use knowledge base
            use_qa_pairs: Whether to check QA pairs
            
        Returns:
            Dictionary with:
            - matched_topics: Classified topics
            - results: {chunks, kb, qa_pairs, merged}
            - retrieval_strategy: "hybrid_kb_rag"
            - metadata: Timing, confidence, etc.
        """
        
        retrieval_start = datetime.now()
        
        # 1. TOPIC CLASSIFICATION
        logger.info(f"🎯 Classifying query to topics: {query[:50]}...")
        topic_classification = await self._classify_to_topics(query, session_id)
        matched_topics = topic_classification.get("matched_topics", [])
        classification_confidence = topic_classification.get("confidence", 0.0)
        
        # 2. TRADITIONAL CHUNK RETRIEVAL
        logger.info(f"📄 Retrieving chunks (top_k={top_k})...")
        chunk_results = await self._retrieve_chunks(query, session_id, top_k, embedding_model)
        
        # 3. QA PAIRS MATCHING (if high topic confidence)
        qa_matches = []
        if use_qa_pairs and matched_topics and classification_confidence > 0.6:
            logger.info(f"❓ Checking QA pairs...")
            qa_matches = await self._match_qa_pairs(query, matched_topics, embedding_model)
        
        # 4. KNOWLEDGE BASE RETRIEVAL
        kb_results = []
        if use_kb and matched_topics and classification_confidence > self.kb_usage_threshold:
            logger.info(f"📚 Fetching knowledge base...")
            kb_results = await self._retrieve_knowledge_base(matched_topics)
        
        # 5. MERGE AND RANK
        logger.info(f"🔀 Merging results...")
        merged_results = self._merge_results(
            chunk_results=chunk_results,
            kb_results=kb_results,
            qa_matches=qa_matches,
            strategy="weighted_fusion"
        )
        
        retrieval_time = (datetime.now() - retrieval_start).total_seconds()
        
        return {
            "query": query,
            "matched_topics": matched_topics,
            "classification_confidence": classification_confidence,
            "results": {
                "chunks": chunk_results,
                "knowledge_base": kb_results,
                "qa_pairs": qa_matches,
                "merged": merged_results
            },
            "retrieval_strategy": "hybrid_kb_rag",
            "metadata": {
                "retrieval_time_seconds": round(retrieval_time, 3),
                "chunks_count": len(chunk_results),
                "kb_entries_count": len(kb_results),
                "qa_matches_count": len(qa_matches),
                "merged_count": len(merged_results)
            }
        }
    
    async def _classify_to_topics(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Classify query to one or more topics using LLM with improved fallback and caching
        
        Returns:
            {
                "matched_topics": [{"topic_id": 1, "topic_title": "...", "confidence": 0.9}],
                "confidence": 0.9  # Overall confidence
            }
        """
        
        try:
            logger.info(f"🎯 [TOPIC CLASSIFICATION] Starting classification for query: '{query[:50]}...'")
            
            # Check cache first
            query_hash = hashlib.md5(f"{session_id}:{query}".encode()).hexdigest()
            
            try:
                with self.db.get_connection() as conn:
                    # Try to get from cache
                    cache_result = conn.execute("""
                        SELECT classification_result, confidence
                        FROM topic_classification_cache
                        WHERE query_hash = ? AND session_id = ? AND expires_at > CURRENT_TIMESTAMP
                    """, (query_hash, session_id)).fetchone()
                    
                    if cache_result:
                        cached_data = dict(cache_result)
                        classification = json.loads(cached_data["classification_result"])
                        logger.info(f"✅ [TOPIC CLASSIFICATION] Cache hit! Confidence: {cached_data.get('confidence', 0):.2f}")
                        
                        # Update cache hits
                        conn.execute("""
                            UPDATE topic_classification_cache
                            SET cache_hits = cache_hits + 1
                            WHERE query_hash = ? AND session_id = ?
                        """, (query_hash, session_id))
                        conn.commit()
                        
                        return classification
            except Exception as cache_err:
                # Cache table might not exist, continue with normal flow
                logger.debug(f"⚠️ [TOPIC CLASSIFICATION] Cache check failed (non-critical): {cache_err}")
            
            # Get all topics for session
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT topic_id, topic_title, description, keywords, estimated_difficulty
                    FROM course_topics
                    WHERE session_id = ? AND is_active = TRUE
                    ORDER BY topic_order
                """, (session_id,))
                
                topics = []
                for row in cursor.fetchall():
                    topic_dict = dict(row)
                    # Ensure None values are converted to empty strings or empty lists
                    topic_dict["description"] = topic_dict.get("description") or ""
                    topic_dict["topic_title"] = topic_dict.get("topic_title") or ""
                    topic_dict["keywords"] = json.loads(topic_dict["keywords"]) if topic_dict["keywords"] else []
                    topics.append(topic_dict)
            
            if not topics:
                logger.warning(f"⚠️ [TOPIC CLASSIFICATION] No topics found for session: {session_id}")
                return {"matched_topics": [], "confidence": 0.0}
            
            logger.info(f"📚 [TOPIC CLASSIFICATION] Found {len(topics)} topics for session")
            
            # IMPROVED: Try keyword-based classification first (faster, more reliable)
            keyword_result = self._keyword_based_classification(query, topics)
            if keyword_result.get("matched_topics") and keyword_result.get("confidence", 0) > 0.7:
                logger.info(f"✅ [TOPIC CLASSIFICATION] Keyword-based classification successful: {len(keyword_result['matched_topics'])} topics, confidence: {keyword_result['confidence']:.2f}")
                return keyword_result
            
            # If keyword matching is not confident enough, try LLM
            logger.info(f"🤖 [TOPIC CLASSIFICATION] Keyword confidence low ({keyword_result.get('confidence', 0):.2f}), trying LLM classification...")
            
            # Prepare topics text for LLM
            topics_text = "\n".join([
                f"ID: {t['topic_id']}, Başlık: {t['topic_title']}, "
                f"Anahtar Kelimeler: {', '.join(t['keywords']) if t['keywords'] else 'Yok'}"
                for t in topics
            ])
            
            # LLM classification
            prompt = f"""Aşağıdaki öğrenci sorusunu, verilen konu listesine göre sınıflandır.

ÖĞRENCİ SORUSU:
{query}

KONU LİSTESİ:
{topics_text}

ÇIKTI FORMATI (JSON):
{{
  "matched_topics": [
    {{
      "topic_id": 5,
      "topic_title": "Hücre Zarı",
      "confidence": 0.92,
      "reasoning": "Soru hücre zarının yapısı hakkında"
    }}
  ],
  "overall_confidence": 0.92
}}

En alakalı 1-3 konu seç. Sadece JSON çıktısı ver."""

            try:
                response = requests.post(
                    f"{MODEL_INFERENCER_URL}/models/generate",
                    json={
                        "prompt": prompt,
                        "model": "llama-3.1-8b-instant",
                        "max_tokens": 512,
                        "temperature": 0.2
                    },
                    timeout=10  # Reduced timeout for faster fallback
                )
                
                if response.status_code == 200:
                    result = response.json()
                    llm_output = result.get("response", "")
                    
                    # Parse JSON
                    import re
                    json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                    if json_match:
                        classification = json.loads(json_match.group())
                        matched_topics = classification.get("matched_topics", [])
                        confidence = classification.get("overall_confidence", 0.5)
                        
                        if matched_topics:
                            logger.info(f"✅ [TOPIC CLASSIFICATION] LLM classification successful: {len(matched_topics)} topics, confidence: {confidence:.2f}")
                            
                            result = {
                                "matched_topics": matched_topics,
                                "confidence": confidence
                            }
                            
                            # Cache the LLM result
                            try:
                                with self.db.get_connection() as conn:
                                    # Create cache table if it doesn't exist
                                    conn.execute("""
                                        CREATE TABLE IF NOT EXISTS topic_classification_cache (
                                            query_hash TEXT NOT NULL,
                                            session_id TEXT NOT NULL,
                                            classification_result TEXT NOT NULL,
                                            confidence REAL NOT NULL,
                                            cache_hits INTEGER DEFAULT 0,
                                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                            expires_at TIMESTAMP NOT NULL,
                                            PRIMARY KEY (query_hash, session_id)
                                        )
                                    """)
                                    
                                    # Insert cache entry (expires in 7 days)
                                    conn.execute("""
                                        INSERT OR REPLACE INTO topic_classification_cache (
                                            query_hash, session_id, classification_result, confidence, expires_at
                                        ) VALUES (?, ?, ?, ?, datetime('now', '+7 days'))
                                    """, (
                                        query_hash,
                                        session_id,
                                        json.dumps(result, ensure_ascii=False),
                                        confidence
                                    ))
                                    conn.commit()
                                    logger.debug(f"💾 [TOPIC CLASSIFICATION] Cached LLM result for future queries")
                            except Exception as cache_err:
                                # Cache write failed, but this is non-critical
                                logger.debug(f"⚠️ [TOPIC CLASSIFICATION] Cache write failed (non-critical): {cache_err}")
                            
                            return result
                        else:
                            logger.warning(f"⚠️ [TOPIC CLASSIFICATION] LLM returned empty matched_topics")
                    else:
                        logger.warning(f"⚠️ [TOPIC CLASSIFICATION] LLM response JSON parse failed: {llm_output[:200]}")
                else:
                    logger.warning(f"⚠️ [TOPIC CLASSIFICATION] LLM service returned status {response.status_code}: {response.text[:200]}")
            except requests.exceptions.Timeout:
                logger.warning(f"⏱️ [TOPIC CLASSIFICATION] LLM classification timeout, using keyword fallback")
            except requests.exceptions.RequestException as e:
                logger.warning(f"❌ [TOPIC CLASSIFICATION] LLM service error: {e}, using keyword fallback")
            
            # Fallback: keyword matching (improved)
            logger.info(f"🔄 [TOPIC CLASSIFICATION] Falling back to keyword-based classification")
            keyword_result = self._keyword_based_classification(query, topics)
            logger.info(f"📊 [TOPIC CLASSIFICATION] Keyword fallback result: {len(keyword_result.get('matched_topics', []))} topics, confidence: {keyword_result.get('confidence', 0):.2f}")
            
            # Cache the result (even if from fallback)
            try:
                with self.db.get_connection() as conn:
                    # Create cache table if it doesn't exist
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS topic_classification_cache (
                            query_hash TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            classification_result TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            cache_hits INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP NOT NULL,
                            PRIMARY KEY (query_hash, session_id)
                        )
                    """)
                    
                    # Insert cache entry (expires in 7 days)
                    conn.execute("""
                        INSERT OR REPLACE INTO topic_classification_cache (
                            query_hash, session_id, classification_result, confidence, expires_at
                        ) VALUES (?, ?, ?, ?, datetime('now', '+7 days'))
                    """, (
                        query_hash,
                        session_id,
                        json.dumps(keyword_result, ensure_ascii=False),
                        keyword_result.get("confidence", 0.0)
                    ))
                    conn.commit()
                    logger.debug(f"💾 [TOPIC CLASSIFICATION] Cached result for future queries")
            except Exception as cache_err:
                # Cache write failed, but this is non-critical
                logger.debug(f"⚠️ [TOPIC CLASSIFICATION] Cache write failed (non-critical): {cache_err}")
            
            return keyword_result
            
        except Exception as e:
            logger.error(f"❌ [TOPIC CLASSIFICATION] Error in topic classification: {e}", exc_info=True)
            # Fallback to keyword matching
            try:
                keyword_result = self._keyword_based_classification(query, topics if 'topics' in locals() else [])
                logger.info(f"🔄 [TOPIC CLASSIFICATION] Exception fallback: {len(keyword_result.get('matched_topics', []))} topics")
                return keyword_result
            except Exception as fallback_error:
                logger.error(f"❌ [TOPIC CLASSIFICATION] Keyword fallback also failed: {fallback_error}")
                return {"matched_topics": [], "confidence": 0.0}
    
    def _keyword_based_classification(self, query: str, topics: List[Dict]) -> Dict[str, Any]:
        """
        Improved keyword-based classification with fuzzy matching
        
        Features:
        - Exact keyword matching
        - Partial word matching (for Turkish stemming)
        - Title matching (boost)
        - Description matching (lower weight)
        """
        query_lower = query.lower()
        matched = []
        
        # Extract query words (remove stopwords)
        import re
        stopwords = {'nedir', 'neden', 'nasıl', 'ne', 'hangi', 'kim', 'nerede', 'ne zaman', 
                     'ile', 've', 'veya', 'için', 'gibi', 'kadar', 'daha', 'çok', 'az', 
                     'bir', 'bu', 'şu', 'o', 'da', 'de', 'ki', 'mi', 'mı', 'mu', 'mü'}
        query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if w not in stopwords and len(w) > 2]
        
        logger.debug(f"🔑 [KEYWORD CLASSIFICATION] Query words: {query_words}")
        
        for topic in topics:
            if not topic or not isinstance(topic, dict):
                logger.warning(f"⚠️ [KEYWORD CLASSIFICATION] Skipping invalid topic: {topic}")
                continue
                
            topic_id = topic.get("topic_id")
            topic_title = str(topic.get("topic_title") or "").lower()
            topic_description = str(topic.get("description") or "").lower()
            keywords_raw = topic.get("keywords", []) or []
            # Filter out None values and ensure all are strings
            keywords = [str(kw).lower() for kw in keywords_raw if kw is not None and str(kw).strip()]
            
            # Score components
            keyword_matches = 0
            title_matches = 0
            description_matches = 0
            
            # 1. Exact keyword matching
            for kw in keywords:
                if not kw or not isinstance(kw, str):
                    continue
                kw_lower = kw.lower()
                if kw_lower in query_lower:
                    keyword_matches += 1
                # Also check if keyword appears in query words
                if kw_lower in query_words:
                    keyword_matches += 0.5  # Partial match bonus
            
            # 2. Title matching (higher weight)
            for word in query_words:
                if word in topic_title:
                    title_matches += 1
            
            # 3. Description matching (lower weight)
            for word in query_words:
                if word in topic_description:
                    description_matches += 0.3  # Lower weight
            
            # Calculate total score
            total_score = keyword_matches + (title_matches * 1.5) + description_matches
            
            if total_score > 0:
                # Normalize confidence: base on matches vs total possible
                max_possible = max(len(keywords), len(query_words), 1)
                confidence = min(total_score / max_possible, 1.0)
                
                # Boost confidence if title matches
                if title_matches > 0:
                    confidence = min(1.0, confidence * 1.2)
                
                matched.append({
                    "topic_id": topic_id,
                    "topic_title": topic.get("topic_title", ""),
                    "confidence": round(confidence, 3),
                    "match_details": {
                        "keyword_matches": keyword_matches,
                        "title_matches": title_matches,
                        "description_matches": description_matches
                    }
                })
                
                logger.debug(f"📌 [KEYWORD CLASSIFICATION] Topic '{topic.get('topic_title')}': score={total_score:.2f}, confidence={confidence:.2f}")
        
        matched.sort(key=lambda x: x["confidence"], reverse=True)
        
        result = {
            "matched_topics": matched[:3],  # Top 3
            "confidence": matched[0]["confidence"] if matched else 0.0
        }
        
        logger.info(f"📊 [KEYWORD CLASSIFICATION] Found {len(matched)} matching topics, top confidence: {result['confidence']:.2f}")
        
        return result
    
    async def _retrieve_chunks(
        self, 
        query: str, 
        session_id: str, 
        top_k: int,
        embedding_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Traditional chunk-based retrieval via document processing service
        """
        
        try:
            # Always use default embedding model if not provided
            if not embedding_model:
                embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
                logger.info(f"🔍 Using default embedding model: {embedding_model} for chunk retrieval")
            
            payload = {
                "session_id": session_id,
                "query": query,
                "top_k": top_k,
                "use_rerank": False,  # Disabled: CRAG evaluation will do reranking, avoid double reranking
                "min_score": 0.3,
                "skip_llm": True,  # Skip LLM generation, we only need chunks (performance optimization)
                "embedding_model": embedding_model  # ALWAYS include embedding_model (default or provided)
            }
            
            logger.info(f"🔍 Using embedding model: {embedding_model} for chunk retrieval")
            
            response = requests.post(
                f"{DOCUMENT_PROCESSING_URL}/query",
                json=payload,
                timeout=60  # Increased from 30 to 60 to handle slow LLM responses
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"📥 Document-processing-service response keys: {list(data.keys())}")
                
                # Document processing service returns "sources", not "chunks"
                sources = data.get("sources", [])
                logger.info(f"Retrieved {len(sources)} sources from document-processing-service")
                
                if sources:
                    logger.info(f"First source keys: {list(sources[0].keys()) if sources else 'N/A'}")
                
                # Convert sources to chunks format with keyword filtering and title boosting
                chunks = []
                query_lower = query.lower()
                
                # Extract keywords from query (remove stopwords)
                import re
                stopwords = {'nedir', 'neden', 'nasıl', 'ne', 'hangi', 'kim', 'nerede', 'ne zaman', 
                             'ile', 've', 'veya', 'için', 'gibi', 'kadar', 'daha', 'çok', 'az', 
                             'bir', 'bu', 'şu', 'o', 'da', 'de', 'ki', 'mi', 'mı', 'mu', 'mü'}
                query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if w not in stopwords and len(w) > 2]
                
                logger.debug(f"🔑 [CHUNK FILTERING] Extracted keywords from query: {query_words}")
                
                for source in sources:
                    # Handle different source formats
                    content = source.get("content") or source.get("text") or source.get("document", "")
                    base_score = source.get("score") or source.get("distance", 0.0)
                    # Convert distance to similarity if needed
                    if base_score > 1.0:  # Likely a distance, convert to similarity
                        base_score = max(0.0, 1.0 - base_score)
                    metadata = source.get("metadata") or source.get("meta", {})
                    
                    # Apply keyword filtering and title boosting
                    content_lower = content.lower()
                    chunk_title = metadata.get("chunk_title", "").lower()
                    
                    # Title boosting
                    title_boost = 0.0
                    if query_words and chunk_title:
                        title_matches = sum(1 for kw in query_words if kw in chunk_title)
                        if title_matches > 0:
                            title_boost = min(0.3, title_matches * 0.1)
                            logger.debug(f"📌 [CHUNK FILTERING] Title boost: +{title_boost:.2f} for '{chunk_title[:50]}...'")
                    
                    # Content boosting
                    content_boost = 0.0
                    if query_words:
                        content_matches = sum(1 for kw in query_words if kw in content_lower)
                        if content_matches > 0:
                            content_boost = min(0.2, content_matches * 0.05)
                    
                    # Negative keyword filtering
                    negative_penalty = 0.0
                    if query_words:
                        opposite_patterns = {
                            'eşeyli': 'eşeysiz',
                            'eşeysiz': 'eşeyli',
                            'olumlu': 'olumsuz',
                            'olumsuz': 'olumlu',
                            'artı': 'eksi',
                            'eksi': 'artı'
                        }
                        for kw in query_words:
                            if kw in opposite_patterns:
                                opposite_kw = opposite_patterns[kw]
                                if opposite_kw in content_lower or opposite_kw in chunk_title:
                                    negative_penalty = -0.2
                                    logger.debug(f"⚠️ [CHUNK FILTERING] Negative penalty: {negative_penalty:.2f} (opposite: {opposite_kw})")
                                    break
                    
                    # Calculate final score
                    final_score = base_score + title_boost + content_boost + negative_penalty
                    final_score = max(0.0, min(1.0, final_score))
                    
                    chunks.append({
                        "content": content,
                        "score": final_score,
                        "base_score": base_score,
                        "title_boost": title_boost,
                        "content_boost": content_boost,
                        "negative_penalty": negative_penalty,
                        "metadata": metadata,
                        "source": "chunk"
                    })
                
                # Sort by final score (descending)
                chunks.sort(key=lambda x: x["score"], reverse=True)
                
                logger.info(f"✅ [CHUNK FILTERING] Converted {len(chunks)} chunks with keyword filtering and title boosting")
                if query_words:
                    logger.info(f"🔑 [CHUNK FILTERING] Applied filtering with {len(query_words)} keywords")
                    top_scores = [f"{c.get('score', 0):.2%}" for c in chunks[:3]]
                    logger.info(f"📊 [CHUNK FILTERING] Top 3 scores: {top_scores}")
                
                return chunks
            else:
                error_text = response.text if hasattr(response, 'text') else "Unknown error"
                logger.warning(f"Chunk retrieval failed: {response.status_code} - {error_text}")
                logger.warning(f"Response body: {error_text[:500]}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    async def _match_qa_pairs(self, query: str, matched_topics: List[Dict], embedding_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Match query against stored QA pairs
        Uses similarity cache for performance
        
        Args:
            query: Student question
            matched_topics: List of matched topics
            embedding_model: Embedding model to use (defaults to text-embedding-v4)
        """
        # Use default embedding model if not provided
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        try:
            with self.db.get_connection() as conn:
                # Check cache - only use if embedding model matches
                cache_hit = conn.execute("""
                    SELECT matched_qa_ids, embedding_model FROM qa_similarity_cache
                    WHERE question_text_hash = ? AND expires_at > CURRENT_TIMESTAMP
                """, (query_hash,)).fetchone()
                
                if cache_hit:
                    cached_data = dict(cache_hit)
                    cached_embedding_model = cached_data.get("embedding_model")
                    
                    # Only use cache if embedding model matches
                    if cached_embedding_model == embedding_model or (not cached_embedding_model and embedding_model == os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")):
                        logger.info(f"✅ QA cache hit! (model: {embedding_model})")
                        matched_qa_ids = json.loads(cached_data["matched_qa_ids"])
                        
                        # Increment cache hits
                        conn.execute("""
                            UPDATE qa_similarity_cache
                            SET cache_hits = cache_hits + 1
                            WHERE question_text_hash = ?
                        """, (query_hash,))
                        conn.commit()
                        
                        return matched_qa_ids
                    else:
                        # Cache exists but with wrong embedding model - invalidate it
                        logger.info(f"⚠️ QA cache found but with different embedding model (cached: {cached_embedding_model}, requested: {embedding_model}). Recalculating...")
                        conn.execute("""
                            DELETE FROM qa_similarity_cache
                            WHERE question_text_hash = ?
                        """, (query_hash,))
                        conn.commit()
                
                # No cache, compute similarity using stored embeddings (OPTIMIZED)
                topic_ids = [t["topic_id"] for t in matched_topics]
                if not topic_ids:
                    return []
                
                # OPTIMIZED: Fetch QA pairs with their pre-computed embeddings
                # This is much faster than fetching 50 and calculating embeddings on-the-fly
                # First check if question_embedding column exists
                cursor = conn.execute("PRAGMA table_info(topic_qa_pairs)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                has_question_embedding = 'question_embedding' in columns
                
                placeholders = ','.join(['?' for _ in topic_ids])
                
                if has_question_embedding:
                    # Use optimized query with stored embeddings
                    cursor = conn.execute(f"""
                        SELECT qa_id, topic_id, question, answer, explanation,
                               difficulty_level, question_type, bloom_taxonomy_level,
                               times_asked, average_student_rating,
                               question_embedding, embedding_model, embedding_dim
                        FROM topic_qa_pairs
                        WHERE topic_id IN ({placeholders}) 
                          AND is_active = TRUE
                          AND question_embedding IS NOT NULL
                          AND embedding_model = ?
                        ORDER BY times_asked DESC, average_student_rating DESC
                        LIMIT 50
                    """, topic_ids + [embedding_model])
                else:
                    # Fallback: fetch QA pairs without embedding filter (for backward compatibility)
                    logger.warning("⚠️ question_embedding column not found, fetching QA pairs without embedding filter")
                    cursor = conn.execute(f"""
                        SELECT qa_id, topic_id, question, answer, explanation,
                               difficulty_level, question_type, bloom_taxonomy_level,
                               times_asked, average_student_rating
                        FROM topic_qa_pairs
                        WHERE topic_id IN ({placeholders}) 
                          AND is_active = TRUE
                        ORDER BY times_asked DESC, average_student_rating DESC
                        LIMIT 50
                    """, topic_ids)
                
                qa_pairs_with_embeddings = [dict(row) for row in cursor.fetchall()]
                
                # If no QA pairs with embeddings found and column exists, try without embedding filter
                # (for backward compatibility with old QA pairs)
                if not qa_pairs_with_embeddings and has_question_embedding:
                    logger.info(f"⚠️ No QA pairs with embeddings found for model {embedding_model}, trying without embedding filter...")
                    cursor = conn.execute(f"""
                        SELECT qa_id, topic_id, question, answer, explanation,
                               difficulty_level, question_type, bloom_taxonomy_level,
                               times_asked, average_student_rating,
                               question_embedding, embedding_model, embedding_dim
                        FROM topic_qa_pairs
                        WHERE topic_id IN ({placeholders}) AND is_active = TRUE
                        ORDER BY times_asked DESC, average_student_rating DESC
                        LIMIT 50
                    """, topic_ids)
                    qa_pairs_with_embeddings = [dict(row) for row in cursor.fetchall()]
            
            if not qa_pairs_with_embeddings:
                return []
            
            # OPTIMIZED: Use stored embeddings for fast similarity calculation
            # But fallback to batch method if embeddings are not available
            if has_question_embedding and any(qa.get("question_embedding") for qa in qa_pairs_with_embeddings):
                logger.info(f"🚀 Calculating similarity for {len(qa_pairs_with_embeddings)} QA pairs using stored embeddings...")
                qa_with_similarity = await self._calculate_qa_similarities_with_stored_embeddings(
                    query=query,
                    qa_pairs=qa_pairs_with_embeddings,
                    embedding_model=embedding_model
                )
            else:
                # Fallback to batch method if embeddings are not available
                logger.info(f"⚠️ Stored embeddings not available, using batch method for {len(qa_pairs_with_embeddings)} QA pairs...")
                qa_with_similarity = await self._calculate_qa_similarities_batch(
                    query=query,
                    qa_pairs=qa_pairs_with_embeddings,
                    embedding_model=embedding_model
                )
            
            # Sort by similarity
            qa_with_similarity.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Cache results (top 10)
            if qa_with_similarity:
                cache_data = json.dumps(qa_with_similarity[:10], ensure_ascii=False)
                with self.db.get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO qa_similarity_cache (
                            question_text, question_text_hash, matched_qa_ids,
                            embedding_model, expires_at
                        ) VALUES (?, ?, ?, ?, datetime('now', '+30 days'))
                    """, (
                        query,
                        query_hash,
                        cache_data,
                        embedding_model  # Store the actual embedding model used
                    ))
                    conn.commit()
            
            return qa_with_similarity[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error in QA matching: {e}")
            return []
    
    async def _calculate_qa_similarities_with_stored_embeddings(
        self,
        query: str,
        qa_pairs: List[Dict],
        embedding_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculate similarities using pre-computed QA question embeddings (FASTEST)
        
        This is the fastest approach because:
        1. QA question embeddings are already computed and stored in DB
        2. Only query embedding needs to be calculated (1 API call)
        3. Cosine similarity is calculated in-memory using numpy (very fast)
        
        Args:
            query: Student question
            qa_pairs: List of QA pairs with stored embeddings
            embedding_model: Embedding model to use
            
        Returns:
            List of QA pairs with similarity scores (filtered by threshold > 0.75)
        """
        # Use default embedding model if not provided
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        
        try:
            # Separate QA pairs with and without embeddings
            qa_with_embeddings = []
            qa_without_embeddings = []
            
            for qa in qa_pairs:
                if qa.get("question_embedding") and qa.get("embedding_model") == embedding_model:
                    # Parse stored embedding (JSON array)
                    try:
                        stored_embedding = json.loads(qa["question_embedding"])
                        qa_with_embeddings.append({
                            **qa,
                            "stored_embedding": np.array(stored_embedding, dtype=np.float32)
                        })
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"⚠️ Failed to parse stored embedding for QA {qa.get('qa_id')}: {e}")
                        qa_without_embeddings.append(qa)
                else:
                    qa_without_embeddings.append(qa)
            
            # Calculate query embedding (only once!)
            logger.info(f"📦 Calculating query embedding (1 API call)...")
            response = requests.post(
                f"{MODEL_INFERENCER_URL}/embeddings",
                json={
                    "texts": [query],
                    "model": embedding_model
                },
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"⚠️ Query embedding failed, falling back to batch method")
                return await self._calculate_qa_similarities_batch(query, qa_pairs, embedding_model)
            
            data = response.json()
            query_embedding = np.array(data.get("embeddings", [])[0], dtype=np.float32)
            
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                query_norm = 1.0
            query_embedding = query_embedding / query_norm
            
            # Fast similarity calculation using stored embeddings
            qa_with_similarity = []
            
            if qa_with_embeddings:
                logger.info(f"⚡ Calculating similarity for {len(qa_with_embeddings)} QA pairs with stored embeddings...")
                
                # Batch cosine similarity calculation (vectorized)
                stored_embeddings = np.array([qa["stored_embedding"] for qa in qa_with_embeddings])
                
                # Normalize stored embeddings
                stored_norms = np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
                stored_norms[stored_norms == 0] = 1.0
                stored_embeddings = stored_embeddings / stored_norms
                
                # Batch cosine similarity (very fast!)
                similarities = np.dot(stored_embeddings, query_embedding)
                
                for i, qa in enumerate(qa_with_embeddings):
                    similarity = float(similarities[i])
                    
                    if similarity > 0.75:  # Threshold for relevance
                        qa_with_similarity.append({
                            "type": "qa_pair",
                            "qa_id": qa["qa_id"],
                            "topic_id": qa["topic_id"],
                            "question": qa["question"],
                            "answer": qa["answer"],
                            "explanation": qa.get("explanation"),
                            "difficulty_level": qa["difficulty_level"],
                            "question_type": qa["question_type"],
                            "bloom_level": qa["bloom_taxonomy_level"],
                            "similarity_score": similarity,
                            "times_asked": qa["times_asked"],
                            "rating": qa.get("average_student_rating")
                        })
            
            # Handle QA pairs without embeddings (fallback to batch calculation)
            if qa_without_embeddings:
                logger.info(f"🔄 Calculating similarity for {len(qa_without_embeddings)} QA pairs without stored embeddings (fallback)...")
                fallback_results = await self._calculate_qa_similarities_batch(
                    query=query,
                    qa_pairs=qa_without_embeddings,
                    embedding_model=embedding_model
                )
                qa_with_similarity.extend(fallback_results)
            
            logger.info(f"✅ Similarity calculation completed: {len(qa_with_similarity)} QA pairs above threshold")
            return qa_with_similarity
            
        except Exception as e:
            logger.error(f"❌ Error in stored embedding similarity calculation: {e}, falling back to batch method")
            return await self._calculate_qa_similarities_batch(query, qa_pairs, embedding_model)
    
    async def _calculate_qa_similarities_batch(
        self,
        query: str,
        qa_pairs: List[Dict],
        embedding_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculate similarities for all QA pairs using batch embedding (OPTIMIZED)
        
        This is much faster than calculating similarity one-by-one because:
        1. Query embedding is calculated only once
        2. All QA question embeddings are calculated in a single batch API call
        3. Cosine similarities are calculated in bulk using numpy
        
        Args:
            query: Student question
            qa_pairs: List of QA pairs to match against
            embedding_model: Embedding model to use
            
        Returns:
            List of QA pairs with similarity scores (filtered by threshold > 0.75)
        """
        # Use default embedding model if not provided
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        
        try:
            # Prepare all texts for batch embedding
            all_texts = [query] + [qa["question"] for qa in qa_pairs]
            
            logger.info(f"📦 Batch embedding: {len(all_texts)} texts (1 query + {len(qa_pairs)} QA questions)")
            
            # Single batch API call for all embeddings
            response = requests.post(
                f"{MODEL_INFERENCER_URL}/embeddings",
                json={
                    "texts": all_texts,
                    "model": embedding_model
                },
                timeout=30  # Increased timeout for batch processing
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                
                if len(embeddings) < len(all_texts):
                    logger.warning(f"⚠️ Expected {len(all_texts)} embeddings, got {len(embeddings)}")
                    # Fallback to individual calculations
                    return await self._calculate_qa_similarities_fallback(query, qa_pairs, embedding_model)
                
                # Convert to numpy arrays for efficient computation
                query_embedding = np.array(embeddings[0])
                qa_embeddings = np.array(embeddings[1:])
                
                # Normalize embeddings for cosine similarity
                query_norm = np.linalg.norm(query_embedding)
                if query_norm == 0:
                    query_norm = 1.0
                query_embedding = query_embedding / query_norm
                
                qa_norms = np.linalg.norm(qa_embeddings, axis=1, keepdims=True)
                qa_norms[qa_norms == 0] = 1.0
                qa_embeddings = qa_embeddings / qa_norms
                
                # Batch cosine similarity calculation (vectorized)
                similarities = np.dot(qa_embeddings, query_embedding)
                
                logger.info(f"✅ Batch similarity calculation completed: {len(similarities)} similarities computed")
                
                # Build results with similarity scores
                qa_with_similarity = []
                for i, qa in enumerate(qa_pairs):
                    similarity = float(similarities[i])
                    
                    if similarity > 0.75:  # Threshold for relevance
                        qa_with_similarity.append({
                            "type": "qa_pair",
                            "qa_id": qa["qa_id"],
                            "topic_id": qa["topic_id"],
                            "question": qa["question"],
                            "answer": qa["answer"],
                            "explanation": qa.get("explanation"),
                            "difficulty_level": qa["difficulty_level"],
                            "question_type": qa["question_type"],
                            "bloom_level": qa["bloom_taxonomy_level"],
                            "similarity_score": similarity,
                            "times_asked": qa["times_asked"],
                            "rating": qa.get("average_student_rating")
                        })
                
                logger.info(f"📊 Found {len(qa_with_similarity)} QA pairs above threshold (0.75)")
                return qa_with_similarity
            else:
                logger.warning(f"⚠️ Batch embedding API failed: {response.status_code}, falling back to individual calculations")
                return await self._calculate_qa_similarities_fallback(query, qa_pairs, embedding_model)
                
        except Exception as e:
            logger.error(f"❌ Batch similarity calculation failed: {e}, falling back to individual calculations")
            return await self._calculate_qa_similarities_fallback(query, qa_pairs, embedding_model)
    
    async def _calculate_qa_similarities_fallback(
        self,
        query: str,
        qa_pairs: List[Dict],
        embedding_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback method: Calculate similarities one-by-one (slower but more reliable)
        Used when batch embedding fails
        """
        logger.info(f"🔄 Using fallback: calculating {len(qa_pairs)} similarities individually...")
        qa_with_similarity = []
        
        for qa in qa_pairs:
            try:
                similarity = await self._calculate_similarity(query, qa["question"], embedding_model)
                
                if similarity > 0.75:  # Threshold for relevance
                    qa_with_similarity.append({
                        "type": "qa_pair",
                        "qa_id": qa["qa_id"],
                        "topic_id": qa["topic_id"],
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "explanation": qa.get("explanation"),
                        "difficulty_level": qa["difficulty_level"],
                        "question_type": qa["question_type"],
                        "bloom_level": qa["bloom_taxonomy_level"],
                        "similarity_score": similarity,
                        "times_asked": qa["times_asked"],
                        "rating": qa.get("average_student_rating")
                    })
            except Exception as e:
                logger.warning(f"⚠️ Error calculating similarity for QA {qa.get('qa_id')}: {e}")
                continue
        
        return qa_with_similarity
    
    async def _calculate_similarity(self, text1: str, text2: str, embedding_model: Optional[str] = None) -> float:
        """
        Calculate semantic similarity between two texts
        Uses embedding model
        
        Args:
            text1: First text
            text2: Second text
            embedding_model: Embedding model to use (defaults to text-embedding-v4)
        """
        # Use default embedding model if not provided
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        
        try:
            # Get embeddings from model inference service
            response = requests.post(
                f"{MODEL_INFERENCER_URL}/embeddings",
                json={
                    "texts": [text1, text2],
                    "model": embedding_model
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                
                if len(embeddings) >= 2:
                    # Cosine similarity
                    import numpy as np
                    emb1 = np.array(embeddings[0])
                    emb2 = np.array(embeddings[1])
                    
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    return float(similarity)
            
            # Fallback: simple word overlap
            return self._word_overlap_similarity(text1, text2)
            
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}, using word overlap")
            return self._word_overlap_similarity(text1, text2)
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity (fallback)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _retrieve_knowledge_base(self, matched_topics: List[Dict]) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge base entries for matched topics
        """
        
        kb_results = []
        
        try:
            with self.db.get_connection() as conn:
                for topic in matched_topics:
                    cursor = conn.execute("""
                        SELECT 
                            kb.knowledge_id,
                            kb.topic_id,
                            kb.topic_summary,
                            kb.key_concepts,
                            kb.learning_objectives,
                            kb.examples,
                            kb.content_quality_score,
                            t.topic_title
                        FROM topic_knowledge_base kb
                        JOIN course_topics t ON kb.topic_id = t.topic_id
                        WHERE kb.topic_id = ?
                    """, (topic["topic_id"],))
                    
                    kb_entry = cursor.fetchone()
                    if kb_entry:
                        kb_dict = dict(kb_entry)
                        
                        # Parse JSON fields
                        kb_dict["key_concepts"] = json.loads(kb_dict["key_concepts"]) if kb_dict["key_concepts"] else []
                        kb_dict["learning_objectives"] = json.loads(kb_dict["learning_objectives"]) if kb_dict["learning_objectives"] else []
                        kb_dict["examples"] = json.loads(kb_dict["examples"]) if kb_dict["examples"] else []
                        
                        kb_results.append({
                            "type": "knowledge_base",
                            "topic_id": topic["topic_id"],
                            "topic_title": kb_dict["topic_title"],
                            "content": kb_dict,
                            "relevance_score": topic["confidence"],  # Use topic classification confidence
                            "quality_score": kb_dict["content_quality_score"]
                        })
            
            logger.info(f"Retrieved {len(kb_results)} KB entries")
            return kb_results
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge base: {e}")
            return []
    
    def _merge_results(
        self,
        chunk_results: List[Dict],
        kb_results: List[Dict],
        qa_matches: List[Dict],
        strategy: str = "weighted_fusion"
    ) -> List[Dict[str, Any]]:
        """
        Merge different retrieval sources with intelligent ranking
        
        Strategies:
        - weighted_fusion: Weight by source type (chunks: 40%, KB: 30%, QA: 30%)
        - reciprocal_rank_fusion: RRF algorithm
        - confidence_based: Use classification confidence
        
        Returns:
            Sorted list of merged results
        """
        
        merged = []
        
        if strategy == "weighted_fusion":
            # CHUNKS: 40% weight (traditional RAG baseline)
            for i, chunk in enumerate(chunk_results[:8]):  # Top 8 chunks
                score = chunk.get("score", 0.5)
                crag_score = chunk.get("crag_score", score)
                
                merged.append({
                    "content": chunk.get("content", chunk.get("text", "")),
                    "source": "chunk",
                    "source_type": "vector_search",
                    "rank": i + 1,
                    "original_score": score,
                    "final_score": crag_score * 0.4,  # 40% weight
                    "metadata": chunk.get("metadata", {})
                })
            
            # KNOWLEDGE BASE: 30% weight (structured knowledge)
            for kb in kb_results:
                # Use summary as main content
                summary = kb["content"].get("topic_summary", "")
                
                merged.append({
                    "content": summary,
                    "source": "knowledge_base",
                    "source_type": "structured_kb",
                    "topic_title": kb["topic_title"],
                    "topic_id": kb["topic_id"],
                    "original_score": kb["relevance_score"],
                    "final_score": kb["relevance_score"] * 0.3,  # 30% weight
                    "metadata": {
                        "quality_score": kb["quality_score"],
                        "concepts": kb["content"]["key_concepts"],
                        "objectives": kb["content"]["learning_objectives"],
                        "examples": kb["content"]["examples"]
                    }
                })
            
            # QA PAIRS: 30% weight (direct matches get high priority)
            for qa in qa_matches[:3]:  # Top 3 QA matches
                if qa["similarity_score"] > 0.85:  # Only high similarity
                    content = f"SORU: {qa['question']}\n\nCEVAP: {qa['answer']}"
                    if qa.get("explanation"):
                        content += f"\n\nAÇIKLAMA: {qa['explanation']}"
                    
                    merged.append({
                        "content": content,
                        "source": "qa_pair",
                        "source_type": "direct_qa",
                        "qa_id": qa["qa_id"],
                        "original_score": qa["similarity_score"],
                        "final_score": qa["similarity_score"] * 0.3,  # 30% weight
                        "metadata": {
                            "difficulty": qa["difficulty_level"],
                            "question_type": qa["question_type"],
                            "bloom_level": qa["bloom_level"],
                            "times_asked": qa["times_asked"]
                        }
                    })
        
        elif strategy == "reciprocal_rank_fusion":
            # RRF: 1 / (k + rank) where k=60
            k = 60
            
            for i, chunk in enumerate(chunk_results):
                merged.append({
                    "content": chunk.get("content", ""),
                    "source": "chunk",
                    "final_score": 1.0 / (k + i + 1),
                    "metadata": chunk.get("metadata", {})
                })
            
            for i, kb in enumerate(kb_results):
                merged.append({
                    "content": kb["content"]["topic_summary"],
                    "source": "knowledge_base",
                    "final_score": 1.0 / (k + i + 1),
                    "metadata": kb["content"]
                })
            
            for i, qa in enumerate(qa_matches):
                merged.append({
                    "content": f"{qa['question']}\n{qa['answer']}",
                    "source": "qa_pair",
                    "final_score": 1.0 / (k + i + 1),
                    "metadata": qa
                })
        
        # Sort by final score
        merged.sort(key=lambda x: x["final_score"], reverse=True)
        
        logger.info(
            f"Merged results: {len(merged)} items "
            f"(chunks: {len([m for m in merged if m['source'] == 'chunk'])}, "
            f"KB: {len([m for m in merged if m['source'] == 'knowledge_base'])}, "
            f"QA: {len([m for m in merged if m['source'] == 'qa_pair'])})"
        )
        
        return merged
    
    def get_direct_answer_if_available(self, retrieval_result: Dict) -> Optional[Dict[str, Any]]:
        """
        Check if we have a direct answer from QA pairs
        Returns QA pair if similarity > 0.90 (very high)
        """
        
        qa_matches = retrieval_result.get("results", {}).get("qa_pairs", [])
        
        if qa_matches and len(qa_matches) > 0:
            top_qa = qa_matches[0]
            if top_qa["similarity_score"] > 0.90:
                logger.info(
                    f"🎯 DIRECT ANSWER AVAILABLE! "
                    f"Similarity: {top_qa['similarity_score']:.3f}"
                )
                return top_qa
        
        return None
    
    def build_context_from_merged_results(
        self,
        merged_results: List[Dict],
        max_chars: int = 8000,
        include_sources: bool = True
    ) -> str:
        """
        Build context string from merged results for LLM
        
        Args:
            merged_results: Merged retrieval results
            max_chars: Maximum context length
            include_sources: Whether to label sources
            
        Returns:
            Formatted context string
        """
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(merged_results):
            content = result["content"]
            source = result["source"]
            
            # Format with source label
            if include_sources:
                source_label = {
                    "chunk": "DERS MATERYALİ",
                    "knowledge_base": "BİLGİ TABANI",
                    "qa_pair": "SORU-CEVAP"
                }.get(source, "KAYNAK")
                
                formatted = f"[{source_label} #{i+1}]\n{content}\n"
            else:
                formatted = f"{content}\n"
            
            # Check length limit
            if current_length + len(formatted) > max_chars:
                break
            
            context_parts.append(formatted)
            current_length += len(formatted)
        
        context = "\n---\n\n".join(context_parts)
        
        logger.info(
            f"Built context: {current_length} chars from {len(context_parts)} sources"
        )
        
        return context
    
    async def track_qa_usage(
        self,
        qa_id: int,
        user_id: str,
        session_id: str,
        original_question: str,
        similarity_score: float,
        response_time_ms: int
    ):
        """
        Track QA pair usage for analytics
        """
        
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO student_qa_interactions (
                        qa_id, user_id, session_id, original_question,
                        similarity_score, response_time_ms, response_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    qa_id,
                    user_id,
                    session_id,
                    original_question,
                    similarity_score,
                    response_time_ms,
                    "direct_qa"
                ))
                
                # Increment times_matched in topic_qa_pairs
                conn.execute("""
                    UPDATE topic_qa_pairs
                    SET times_matched = times_matched + 1
                    WHERE qa_id = ?
                """, (qa_id,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error tracking QA usage: {e}")






