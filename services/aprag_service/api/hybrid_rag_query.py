"""
Hybrid RAG Query Endpoint
KB-Enhanced RAG: Chunks + Knowledge Base + QA Pairs
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
from datetime import datetime
import requests
import os

logger = logging.getLogger(__name__)

router = APIRouter()

# Import database manager
from database.database import DatabaseManager

# Import hybrid retriever
import sys
sys.path.append(os.path.dirname(__file__))
from services.hybrid_knowledge_retriever import HybridKnowledgeRetriever

# Environment variables
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", "http://model-inference-service:8002")
DOCUMENT_PROCESSING_URL = os.getenv("DOCUMENT_PROCESSING_URL", "http://document-processing-service:8080")


# ============================================================================
# Request/Response Models
# ============================================================================

class HybridRAGQueryRequest(BaseModel):
    """Request model for KB-Enhanced RAG query"""
    session_id: str
    query: str
    user_id: Optional[str] = "student"
    
    # Retrieval options
    top_k: int = 10
    use_kb: bool = True  # Use knowledge base
    use_qa_pairs: bool = True  # Check QA pairs for direct answers
    use_crag: bool = True  # Use CRAG evaluation
    
    # Generation options
    model: Optional[str] = "llama-3.1-8b-instant"
    max_tokens: int = 1024
    temperature: float = 0.7
    max_context_chars: int = 8000
    
    # Preferences
    include_examples: bool = True  # Include examples from KB
    include_sources: bool = True  # Include source labels in context


class HybridRAGQueryResponse(BaseModel):
    """Response model for KB-Enhanced RAG query"""
    answer: str
    confidence: str  # high, medium, low
    retrieval_strategy: str
    
    # Source breakdown
    sources_used: Dict[str, int]  # {"chunks": 5, "kb": 1, "qa_pairs": 1}
    direct_qa_match: bool  # Was a direct QA match used?
    
    # Topic information
    matched_topics: List[Dict[str, Any]]
    classification_confidence: float
    
    # CRAG information
    crag_action: Optional[str] = None  # accept, filter, reject
    crag_confidence: Optional[float] = None
    
    # Metadata
    processing_time_ms: int
    sources: List[Dict[str, Any]]  # Detailed source information


# ============================================================================
# Helper Functions
# ============================================================================

def get_db() -> DatabaseManager:
    """Get database manager dependency"""
    db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
    return DatabaseManager(db_path)


async def generate_answer_with_llm(
    query: str,
    context: str,
    topic_title: Optional[str] = None,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 768,
    temperature: float = 0.6
) -> str:
    """Generate answer using LLM with KB-enhanced context"""
    
    # Focused, Turkish-only prompt with topic context
    prompt = f"""Sen bir eğitim asistanısın. Aşağıdaki ders materyallerini kullanarak ÖĞRENCİ SORUSUNU kısa, net ve konu dışına çıkmadan yanıtla.

{f"📚 KONU: {topic_title}" if topic_title else ""}

📖 DERS MATERYALLERİ VE BİLGİ TABANI:
{context}

👨‍🎓 ÖĞRENCİ SORUSU:
{query}

YANIT KURALLARI (ÇOK ÖNEMLİ):
1. Yanıt TAMAMEN TÜRKÇE olmalı.
2. Sadece sorulan soruya odaklan; konu dışına çıkma, gereksiz alt başlıklar açma.
3. Yanıtın toplam uzunluğunu en fazla 3 paragraf ve yaklaşık 5–8 cümle ile sınırla.
4. Gerekirse en fazla 1 tane kısa gerçek hayat örneği ver; uzun anlatımlardan kaçın.
5. Bilgiyi mutlaka yukarıdaki ders materyali ve bilgi tabanından al; emin olmadığın şeyleri yazma, uydurma.
6. Önemli kavramları gerektiğinde **kalın** yazarak vurgulayabilirsin ama liste/rapor formatına dönüştürme.

✍️ YANIT (sadece cevabı yaz, başlık veya madde listesi ekleme):"""

    try:
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            logger.error(f"LLM generation failed: {response.status_code}")
            return "Yanıt oluşturulamadı. Lütfen tekrar deneyin."
            
    except Exception as e:
        logger.error(f"Error in LLM generation: {e}")
        return "Bir hata oluştu. Lütfen tekrar deneyin."


async def evaluate_with_crag(query: str, chunks: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate retrieved chunks with CRAG
    """
    
    try:
        response = requests.post(
            f"{DOCUMENT_PROCESSING_URL}/crag/evaluate",
            json={
                "query": query,
                "documents": [c.get("content", c.get("text", "")) for c in chunks]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"CRAG evaluation failed: {response.status_code}")
            # Fallback: accept all
            return {
                "action": "accept",
                "confidence": 0.5,
                "filtered_docs": chunks
            }
            
    except Exception as e:
        logger.error(f"Error in CRAG evaluation: {e}")
        return {
            "action": "accept",
            "confidence": 0.5,
            "filtered_docs": chunks
        }


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/query", response_model=HybridRAGQueryResponse)
async def hybrid_rag_query(request: HybridRAGQueryRequest):
    """
    KB-Enhanced RAG Query
    
    Combines:
    1. Traditional chunk-based retrieval
    2. Knowledge base (structured summaries, concepts)
    3. QA pairs (direct answer matching)
    4. CRAG evaluation (quality filtering)
    
    Workflow:
    1. Classify query to topic(s)
    2. Check QA pairs for direct match (similarity > 0.90)
    3. If direct match: Use QA + KB summary
    4. Else: Retrieve chunks + KB + QA
    5. CRAG evaluation on chunks
    6. Merge results with weighted scoring
    7. Generate answer with LLM
    """
    
    start_time = datetime.now()
    db = get_db()
    
    try:
        # Initialize hybrid retriever
        retriever = HybridKnowledgeRetriever(db)
        
        # HYBRID RETRIEVAL
        retrieval_result = await retriever.retrieve_for_query(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            use_kb=request.use_kb,
            use_qa_pairs=request.use_qa_pairs
        )
        
        # Extract components
        matched_topics = retrieval_result["matched_topics"]
        classification_confidence = retrieval_result["classification_confidence"]
        chunk_results = retrieval_result["results"]["chunks"]
        kb_results = retrieval_result["results"]["knowledge_base"]
        qa_matches = retrieval_result["results"]["qa_pairs"]
        merged_results = retrieval_result["results"]["merged"]
        
        # CHECK FOR DIRECT QA MATCH
        direct_qa = retriever.get_direct_answer_if_available(retrieval_result)
        
        if direct_qa:
            # FAST PATH: Direct answer from QA pair
            logger.info(f"🎯 Using direct QA answer (similarity: {direct_qa['similarity_score']:.3f})")
            
            answer = direct_qa["answer"]
            if direct_qa.get("explanation"):
                answer += f"\n\n💡 {direct_qa['explanation']}"
            
            # Add KB summary for context if available
            if kb_results:
                kb_summary = kb_results[0]["content"]["topic_summary"]
                answer += f"\n\n📚 Ek Bilgi: {kb_summary[:200]}..."
            
            # Track usage
            await retriever.track_qa_usage(
                qa_id=direct_qa["qa_id"],
                user_id=request.user_id,
                session_id=request.session_id,
                original_question=request.query,
                similarity_score=direct_qa["similarity_score"],
                response_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return HybridRAGQueryResponse(
                answer=answer,
                confidence="high",
                retrieval_strategy="direct_qa_match",
                sources_used={"qa_pairs": 1, "kb": 1 if kb_results else 0, "chunks": 0},
                direct_qa_match=True,
                matched_topics=matched_topics,
                classification_confidence=classification_confidence,
                processing_time_ms=processing_time,
                sources=[{
                    "type": "qa_pair",
                    "question": direct_qa["question"],
                    "answer": direct_qa["answer"],
                    "similarity": direct_qa["similarity_score"]
                }]
            )
        
        # NORMAL PATH: Use chunks + KB + QA
        
        # CRAG EVALUATION on chunks
        crag_result = None
        if request.use_crag and chunk_results:
            logger.info(f"🔍 Running CRAG evaluation...")
            crag_result = await evaluate_with_crag(request.query, chunk_results)
            
            if crag_result["action"] == "reject":
                logger.warning("❌ CRAG REJECT - No relevant information found")
                
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                return HybridRAGQueryResponse(
                    answer="Bu bilgi ders materyallerinde bulunmuyor. Lütfen farklı bir soru sorun veya konuyu daha açık belirtin.",
                    confidence="low",
                    retrieval_strategy="crag_reject",
                    sources_used={"chunks": 0, "kb": 0, "qa_pairs": 0},
                    direct_qa_match=False,
                    matched_topics=matched_topics,
                    classification_confidence=classification_confidence,
                    crag_action="reject",
                    crag_confidence=crag_result.get("confidence", 0.0),
                    processing_time_ms=processing_time,
                    sources=[]
                )
        
        # BUILD CONTEXT from merged results
        context = retriever.build_context_from_merged_results(
            merged_results=merged_results,
            max_chars=request.max_context_chars,
            include_sources=request.include_sources
        )
        
        if not context.strip():
            logger.warning("⚠️ No context available after retrieval")
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return HybridRAGQueryResponse(
                answer="Üzgünüm, bu soruyla ilgili yeterli bilgi bulamadım.",
                confidence="low",
                retrieval_strategy="no_context",
                sources_used={"chunks": 0, "kb": 0, "qa_pairs": 0},
                direct_qa_match=False,
                matched_topics=matched_topics,
                classification_confidence=classification_confidence,
                processing_time_ms=processing_time,
                sources=[]
            )
        
        # GENERATE ANSWER with LLM
        logger.info(f"🤖 Generating answer with LLM...")
        topic_title = matched_topics[0]["topic_title"] if matched_topics else None
        
        answer = await generate_answer_with_llm(
            query=request.query,
            context=context,
            topic_title=topic_title,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Determine confidence
        confidence = "high"
        if classification_confidence < 0.6:
            confidence = "low"
        elif classification_confidence < 0.8:
            confidence = "medium"
        
        # Count sources used
        sources_used = {
            "chunks": len([m for m in merged_results if m["source"] == "chunk"]),
            "kb": len([m for m in merged_results if m["source"] == "knowledge_base"]),
            "qa_pairs": len([m for m in merged_results if m["source"] == "qa_pair"])
        }
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Prepare detailed sources
        detailed_sources = []
        for result in merged_results[:5]:  # Top 5 sources
            detailed_sources.append({
                "type": result["source"],
                "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                "score": result["final_score"],
                "metadata": result.get("metadata", {})
            })
        
        return HybridRAGQueryResponse(
            answer=answer,
            confidence=confidence,
            retrieval_strategy="hybrid_kb_rag",
            sources_used=sources_used,
            direct_qa_match=False,
            matched_topics=matched_topics,
            classification_confidence=classification_confidence,
            crag_action=crag_result["action"] if crag_result else None,
            crag_confidence=crag_result.get("confidence") if crag_result else None,
            processing_time_ms=processing_time,
            sources=detailed_sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in hybrid RAG query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Hybrid RAG query failed: {str(e)}")


@router.post("/query-feedback")
async def submit_query_feedback(
    interaction_id: int,
    qa_id: Optional[int] = None,
    was_helpful: bool = True,
    student_rating: Optional[int] = None,
    had_followup: bool = False,
    followup_question: Optional[str] = None
):
    """
    Submit feedback for a hybrid RAG query
    Updates QA pair statistics if applicable
    """
    
    db = get_db()
    
    try:
        with db.get_connection() as conn:
            # Update student_qa_interactions if qa_id provided
            if qa_id:
                conn.execute("""
                    UPDATE student_qa_interactions
                    SET was_helpful = ?, student_rating = ?,
                        had_followup = ?, followup_question = ?
                    WHERE interaction_id = ? AND qa_id = ?
                """, (
                    was_helpful,
                    student_rating,
                    had_followup,
                    followup_question,
                    interaction_id,
                    qa_id
                ))
                
                # Update average rating in topic_qa_pairs
                if student_rating:
                    conn.execute("""
                        UPDATE topic_qa_pairs
                        SET average_student_rating = (
                            SELECT AVG(student_rating)
                            FROM student_qa_interactions
                            WHERE qa_id = ? AND student_rating IS NOT NULL
                        )
                        WHERE qa_id = ?
                    """, (qa_id, qa_id))
            
            conn.commit()
        
        return {
            "success": True,
            "message": "Feedback received, thank you!"
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/qa-analytics/{session_id}")
async def get_qa_analytics(session_id: str):
    """
    Get QA pair analytics for a session
    """
    
    db = get_db()
    
    try:
        with self.db.get_connection() as conn:
            # Get most used QA pairs
            cursor = conn.execute("""
                SELECT 
                    qa.qa_id,
                    qa.question,
                    qa.difficulty_level,
                    qa.times_asked,
                    qa.times_matched,
                    qa.average_student_rating,
                    t.topic_title
                FROM topic_qa_pairs qa
                JOIN course_topics t ON qa.topic_id = t.topic_id
                WHERE t.session_id = ? AND qa.is_active = TRUE
                ORDER BY qa.times_matched DESC, qa.average_student_rating DESC
                LIMIT 20
            """, (session_id,))
            
            popular_qa = [dict(row) for row in cursor.fetchall()]
            
            # Get statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_qa_pairs,
                    AVG(times_matched) as avg_times_matched,
                    AVG(average_student_rating) as avg_rating,
                    SUM(CASE WHEN times_matched > 0 THEN 1 ELSE 0 END) as used_qa_pairs
                FROM topic_qa_pairs qa
                JOIN course_topics t ON qa.topic_id = t.topic_id
                WHERE t.session_id = ? AND qa.is_active = TRUE
            """, (session_id,))
            
            stats = dict(cursor.fetchone())
        
        return {
            "success": True,
            "session_id": session_id,
            "statistics": stats,
            "popular_qa_pairs": popular_qa
        }
        
    except Exception as e:
        logger.error(f"Error fetching QA analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")


