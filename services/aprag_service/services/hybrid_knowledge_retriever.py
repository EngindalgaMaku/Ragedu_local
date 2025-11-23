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
        use_qa_pairs: bool = True
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
        chunk_results = await self._retrieve_chunks(query, session_id, top_k)
        
        # 3. QA PAIRS MATCHING (if high topic confidence)
        qa_matches = []
        if use_qa_pairs and matched_topics and classification_confidence > 0.6:
            logger.info(f"❓ Checking QA pairs...")
            qa_matches = await self._match_qa_pairs(query, matched_topics)
        
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
        Classify query to one or more topics using LLM
        
        Returns:
            {
                "matched_topics": [{"topic_id": 1, "topic_title": "...", "confidence": 0.9}],
                "confidence": 0.9  # Overall confidence
            }
        """
        
        try:
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
                    topic_dict["keywords"] = json.loads(topic_dict["keywords"]) if topic_dict["keywords"] else []
                    topics.append(topic_dict)
            
            if not topics:
                return {"matched_topics": [], "confidence": 0.0}
            
            # Prepare topics text for LLM
            topics_text = "\n".join([
                f"ID: {t['topic_id']}, Başlık: {t['topic_title']}, "
                f"Anahtar Kelimeler: {', '.join(t['keywords'])}"
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

            response = requests.post(
                f"{MODEL_INFERENCER_URL}/models/generate",
                json={
                    "prompt": prompt,
                    "model": "llama-3.1-8b-instant",
                    "max_tokens": 512,
                    "temperature": 0.2
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                
                # Parse JSON
                import re
                json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                    return {
                        "matched_topics": classification.get("matched_topics", []),
                        "confidence": classification.get("overall_confidence", 0.5)
                    }
            
            # Fallback: keyword matching
            logger.warning("LLM classification failed, using keyword fallback")
            return self._keyword_based_classification(query, topics)
            
        except Exception as e:
            logger.error(f"Error in topic classification: {e}")
            # Fallback to keyword matching
            return self._keyword_based_classification(query, topics)
    
    def _keyword_based_classification(self, query: str, topics: List[Dict]) -> Dict[str, Any]:
        """Fallback: Simple keyword-based classification"""
        query_lower = query.lower()
        matched = []
        
        for topic in topics:
            keywords = topic.get("keywords", [])
            matches = sum(1 for kw in keywords if kw.lower() in query_lower)
            
            if matches > 0:
                confidence = min(matches / max(len(keywords), 1), 1.0)
                matched.append({
                    "topic_id": topic["topic_id"],
                    "topic_title": topic["topic_title"],
                    "confidence": confidence
                })
        
        matched.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "matched_topics": matched[:3],
            "confidence": matched[0]["confidence"] if matched else 0.0
        }
    
    async def _retrieve_chunks(self, query: str, session_id: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Traditional chunk-based retrieval via document processing service
        """
        
        try:
            response = requests.post(
                f"{DOCUMENT_PROCESSING_URL}/query",
                json={
                    "session_id": session_id,
                    "query": query,
                    "top_k": top_k,
                    "use_rerank": True,  # Use reranking for better quality
                    "min_score": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                chunks = data.get("chunks", [])
                logger.info(f"Retrieved {len(chunks)} chunks")
                return chunks
            else:
                logger.warning(f"Chunk retrieval failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    async def _match_qa_pairs(self, query: str, matched_topics: List[Dict]) -> List[Dict[str, Any]]:
        """
        Match query against stored QA pairs
        Uses similarity cache for performance
        """
        
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        try:
            with self.db.get_connection() as conn:
                # Check cache
                cache_hit = conn.execute("""
                    SELECT matched_qa_ids FROM qa_similarity_cache
                    WHERE question_text_hash = ? AND expires_at > CURRENT_TIMESTAMP
                """, (query_hash,)).fetchone()
                
                if cache_hit:
                    logger.info("✅ QA cache hit!")
                    cached_data = dict(cache_hit)
                    matched_qa_ids = json.loads(cached_data["matched_qa_ids"])
                    
                    # Increment cache hits
                    conn.execute("""
                        UPDATE qa_similarity_cache
                        SET cache_hits = cache_hits + 1
                        WHERE question_text_hash = ?
                    """, (query_hash,))
                    conn.commit()
                    
                    return matched_qa_ids
                
                # No cache, compute similarity
                topic_ids = [t["topic_id"] for t in matched_topics]
                if not topic_ids:
                    return []
                
                placeholders = ','.join(['?' for _ in topic_ids])
                cursor = conn.execute(f"""
                    SELECT qa_id, topic_id, question, answer, explanation,
                           difficulty_level, question_type, bloom_taxonomy_level,
                           times_asked, average_student_rating
                    FROM topic_qa_pairs
                    WHERE topic_id IN ({placeholders}) AND is_active = TRUE
                    ORDER BY times_asked DESC, average_student_rating DESC
                    LIMIT 50
                """, topic_ids)
                
                qa_pairs = [dict(row) for row in cursor.fetchall()]
            
            if not qa_pairs:
                return []
            
            # Calculate semantic similarity
            qa_with_similarity = []
            for qa in qa_pairs:
                similarity = await self._calculate_similarity(query, qa["question"])
                
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
                        "semantic_similarity"
                    ))
                    conn.commit()
            
            return qa_with_similarity[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error in QA matching: {e}")
            return []
    
    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        Uses embedding model
        """
        
        try:
            # Get embeddings from model inference service
            response = requests.post(
                f"{MODEL_INFERENCER_URL}/embeddings",
                json={
                    "texts": [text1, text2],
                    "model": "nomic-embed-text"
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






