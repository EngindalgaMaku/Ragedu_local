"""
RAG Testing Routes for API Gateway.

Provides endpoints for testing RAG system quality:
- Automatic test generation from chunks
- Manual test execution
- CRAG evaluation metrics
- Test history and analytics
"""

import logging
import random
import time
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from pydantic import BaseModel, Field
import requests

logger = logging.getLogger(__name__)


# ===== LIGHTWEIGHT EVALUATION METRICS =====

def calculate_answer_metrics(answer: str, query: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Hafif ve hızlı değerlendirme metrikleri.
    RAGAS yerine basit ama etkili ölçümler.
    """
    if not answer or not answer.strip():
        return {
            "word_count": 0,
            "sentence_count": 0,
            "has_citation": False,
            "context_relevance": 0.0,
            "completeness_score": 0.0,
            "overall_quality": 0.0
        }
    
    # 1. Temel İstatistikler
    words = answer.split()
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    word_count = len(words)
    sentence_count = len(sentences)
    
    # 2. Kaynak Gösterimi (Citation)
    has_citation = bool(re.search(r'\[.*?\]|\(.*?\)|Kaynak:|Referans:', answer))
    
    # 3. Bağlam Alakası (Context Relevance)
    # Soruda geçen anahtar kelimelerin cevap ve bağlamda bulunma oranı
    query_words = set(re.findall(r'\w+', query.lower()))
    query_words = {w for w in query_words if len(w) > 3}  # Kısa kelimeleri filtrele
    
    answer_words = set(re.findall(r'\w+', answer.lower()))
    
    if query_words:
        keyword_coverage = len(query_words & answer_words) / len(query_words)
    else:
        keyword_coverage = 0.5
    
    # Bağlam metinlerinde sorgu kelimelerinin geçme oranı
    context_text = " ".join(contexts).lower() if contexts else ""
    context_words = set(re.findall(r'\w+', context_text))
    
    if query_words and context_words:
        context_relevance = len(query_words & context_words) / len(query_words)
    else:
        context_relevance = 0.0
    
    # 4. Tamlık Skoru (Completeness)
    # Cevap yeterince detaylı mı?
    ideal_word_range = (50, 300)  # İdeal cevap uzunluğu
    if word_count < ideal_word_range[0]:
        completeness = word_count / ideal_word_range[0]
    elif word_count > ideal_word_range[1]:
        completeness = max(0.7, 1.0 - (word_count - ideal_word_range[1]) / 500)
    else:
        completeness = 1.0
    
    # 5. Genel Kalite Skoru
    overall_quality = (
        keyword_coverage * 0.3 +
        context_relevance * 0.3 +
        completeness * 0.2 +
        (1.0 if has_citation else 0.5) * 0.2
    )
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "has_citation": has_citation,
        "keyword_coverage": round(keyword_coverage, 3),
        "context_relevance": round(context_relevance, 3),
        "completeness_score": round(completeness, 3),
        "overall_quality": round(overall_quality, 3)
    }


def compare_answers(answer1: str, answer2: str) -> Dict[str, Any]:
    """
    İki cevabı karşılaştırır.
    Makale için: "Simple RAG vs Advanced RAG" gibi karşılaştırmalar.
    """
    if not answer1 or not answer2:
        return {"comparison": "invalid", "difference": 0}
    
    words1 = set(re.findall(r'\w+', answer1.lower()))
    words2 = set(re.findall(r'\w+', answer2.lower()))
    
    # Jaccard benzerliği
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    similarity = intersection / union if union > 0 else 0
    
    # Uzunluk farkı
    length_diff = abs(len(answer1) - len(answer2))
    
    return {
        "similarity": round(similarity, 3),
        "length_difference": length_diff,
        "unique_to_answer2": len(words2 - words1)  # Advanced RAG'in eklediği yeni kelimeler
    }

router = APIRouter(prefix="/admin/rag-tests", tags=["Admin RAG Tests"])

# Microservice URLs
DOCUMENT_PROCESSOR_URL = os.getenv('DOCUMENT_PROCESSOR_URL', 'http://document-processing-service:8080')
MODEL_INFERENCE_URL = os.getenv('MODEL_INFERENCE_URL', 'https://model-inferencer-awe3elsvra-ew.a.run.app')


# ===== REQUEST/RESPONSE MODELS =====

class ManualTestRequest(BaseModel):
    """Request model for manual test execution."""
    query: str = Field(..., description="Test query to execute")
    expected_relevant: bool = Field(..., description="Whether query should be relevant")
    category: str = Field(default="manual", description="Test category")
    llm_model: Optional[str] = Field(default=None, description="LLM model to use")
    session_id: str = Field(..., description="Session ID to test against")


class AutoTestConfig(BaseModel):
    """Configuration for automatic test generation."""
    num_tests: int = Field(default=10, ge=1, le=100, description="Number of tests to generate")
    include_relevant: bool = Field(default=True, description="Include relevant queries")
    include_irrelevant: bool = Field(default=True, description="Include irrelevant queries")


# ===== HELPER FUNCTIONS =====

def _require_admin(request: Request) -> Dict[str, Any]:
    """Require admin role"""
    from src.api.main import _get_current_user, _is_admin
    
    # Debug logging
    auth_header = request.headers.get("Authorization")
    logger.info(f"🔐 RAG Test Auth Check - Authorization header present: {bool(auth_header)}")
    if auth_header:
        logger.info(f"🔐 RAG Test Auth Check - Token preview: {auth_header[:20]}...")
    
    user = _get_current_user(request)
    logger.info(f"🔐 RAG Test Auth Check - User retrieved: {user}")
    
    if not user:
        logger.error("🔐 RAG Test Auth Check - No user found")
        raise HTTPException(status_code=403, detail="Admin access required - No user found")
    
    if not _is_admin(user):
        logger.error(f"🔐 RAG Test Auth Check - User is not admin. Role: {user.get('role')}")
        raise HTTPException(status_code=403, detail=f"Admin access required - Current role: {user.get('role')}")
    
    logger.info(f"🔐 RAG Test Auth Check - Admin access granted for user: {user.get('username')}")
    return user


def _extract_context_text(doc: Dict[str, Any], max_chars: int = 1200) -> Optional[str]:
    """
    Extracts human-readable text from a retrieved document structure.
    """
    candidates = [
        doc.get('text'),
        doc.get('content'),
        doc.get('chunk_text'),
        doc.get('page_content'),
        doc.get('metadata', {}).get('text') if isinstance(doc.get('metadata'), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            text = candidate.strip()
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text
    return None


# ===== ENDPOINTS =====

@router.get("/available-models", summary="Get Available Models")
async def get_available_models(request: Request) -> Dict[str, Any]:
    """
    Get available LLM and embedding models.
    
    Returns:
        Available models for testing
    """
    _require_admin(request)
    
    try:
        # Simple hardcoded model list - no heavy imports
        llm_models = [
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "provider": "groq", "description": "Fast inference"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "provider": "groq", "description": "Balanced performance"}
        ]
        
        # Embedding models
        embedding_models = [
            {"id": "nomic-embed-text-latest", "name": "Nomic Embed Latest", "description": "Latest embedding model"},
            {"id": "all-MiniLM-L6-v2", "name": "MiniLM L6 (Hızlı)", "description": "Hızlı ve hafif model"},
            {"id": "all-mpnet-base-v2", "name": "MPNet Base (Dengeli)", "description": "Dengeli performans"},
            {"id": "paraphrase-multilingual-MiniLM-L12-v2", "name": "Multilingual (Türkçe)", "description": "Türkçe desteği"}
        ]
        
        return {
            "success": True,
            "llm_models": llm_models,
            "embedding_models": embedding_models,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/sessions", summary="Get Available Sessions")
async def get_available_sessions(request: Request) -> Dict[str, Any]:
    """
    Get available RAG sessions for testing.
    
    Returns:
        List of available sessions
    """
    _require_admin(request)
    
    try:
        from src.database.database import DatabaseManager
        
        db = DatabaseManager()
        
        # Get all active sessions
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    id,
                    title,
                    description,
                    created_by,
                    created_at,
                    category,
                    status
                FROM sessions
                WHERE status = 'active'
                ORDER BY created_at DESC
                LIMIT 50
            """)
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "created_by": row[3],
                    "created_at": row[4],
                    "category": row[5],
                    "status": row[6]
                })
        
        return {
            "success": True,
            "sessions": sessions,
            "total": len(sessions),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


@router.get("/status", summary="Get RAG Testing System Status")
async def get_rag_test_status(request: Request) -> Dict[str, Any]:
    """
    Get RAG testing system status and configuration.
    
    Returns:
        System status including CRAG evaluator availability
    """
    _require_admin(request)
    
    try:
        # Check Document Processing Service
        try:
            response = requests.get(
                f"{DOCUMENT_PROCESSOR_URL}/health",
                timeout=5
            )
            chroma_available = response.status_code == 200
        except Exception as e:
            logger.error(f"Document processor health check failed: {e}")
            chroma_available = False
        
        # CRAG evaluation moved to Document Processing Service
        crag_enabled = True  # Available via Document Processing Service
        crag_evaluator_available = True
        crag_config = {"status": "available_via_document_service"}
        
        return {
            "success": True,
            "status": {
                "chromadb_available": chroma_available,
                "document_count": 0,  # Would need to query from service
                "chunk_count": 0,     # Would need to query from service
                "crag_enabled": crag_enabled,
                "crag_evaluator_available": crag_evaluator_available,
                "crag_config": crag_config,
                "test_generation_available": chroma_available
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting RAG test status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG test status: {str(e)}")


@router.post("/execute-manual", summary="Execute Manual Test for Comparative Analysis")
async def execute_manual_test(
    test_request: ManualTestRequest,
    request: Request
) -> Dict[str, Any]:
    """
    Executes a multi-faceted test to provide a comparative analysis for academic papers.
    It generates three distinct answers for the same query:
    1. Direct LLM Answer: No RAG context.
    2. Simple RAG Answer: Uses documents from the initial vector search.
    3. Advanced RAG (DYSK) Answer: Uses documents filtered by the DYSK/CRAG layer.
    """
    _require_admin(request)
    
    try:
        start_time = time.time()
        from src.services.session_manager import professional_session_manager

        async def _generate_answer_from_docs_via_api(query: str, docs: List[Dict[str, Any]], model: str, max_context_chars: int = 8000) -> str:
            try:
                payload = {
                    "query": query,
                    "docs": docs,
                    "model": model,
                    "max_context_chars": max_context_chars
                }
                response = requests.post(
                    f"{MODEL_INFERENCE_URL}/generate-answer",
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                return response.json().get("response", "API'den yanıt alınamadı.")
            except Exception as e:
                logger.error(f"API call to /generate-answer failed: {e}")
                return f"Yanıt üretilirken API hatası: {e}"

        # --- 1. Get Authoritative Embedding Model ---
        try:
            rag_settings = professional_session_manager.get_session_rag_settings(test_request.session_id)
            session_embedding_model = rag_settings.get("embedding_model") if rag_settings else None
            if not session_embedding_model:
                raise HTTPException(status_code=400, detail=f"Session '{test_request.session_id}' has no configured embedding model.")
            logger.info(f"✅ Using authoritative embedding model: {session_embedding_model}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not retrieve session settings: {e}")

        # --- 2. Generate Direct LLM Answer (No RAG) ---
        # Use direct generation endpoint WITHOUT RAG context
        try:
            system_prompt = (
                "Sen, öğrencilere yardımcı olan eğitimli bir yapay zeka asistanısın. "
                "KATI KURALLAR:\n"
                "1. KESINLIKLE TÜRKÇE CEVAP VER. Hiçbir durumda İngilizce kelime, cümle veya ifade kullanma.\n"
                "2. Eğer bir kavramın İngilizce ismi varsa bile, Türkçe karşılığını kullan veya Türkçe açıklama yap.\n"
                "3. Teknik terimler için bile Türkçe karşılıkları tercih et.\n"
                "4. Soruları açık, anlaşılır ve eğitici bir şekilde yanıtla.\n"
                "5. Öğrencilerin öğrenme sürecini destekle.\n"
                "6. Eğer bir soruya tam olarak cevap veremiyorsan, dürüst ol ve önerilerde bulun, ama her şeyi TÜRKÇE yap.\n"
                "Bu kurallara kesinlikle uy. İngilizce kullanmak yasaktır."
            )
            
            # Build prompt without RAG context
            full_prompt = f"System: {system_prompt}\n\nUser: {test_request.query}\n\nCevap:"
            
            generation_request = {
                "prompt": full_prompt,
                "model": test_request.llm_model or "llama-3.1-8b-instant",
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            direct_response = requests.post(
                f"{MODEL_INFERENCE_URL}/models/generate",
                json=generation_request,
                timeout=120
            )
            direct_response.raise_for_status()
            direct_llm_answer = direct_response.json().get("response", "Direkt LLM yanıtı alınamadı.")
        except Exception as e:
            logger.error(f"Direct LLM generation failed: {e}")
            direct_llm_answer = f"Direkt LLM yanıtı üretilirken hata: {str(e)}"

        # --- 3. Perform Retrieval ---
        collection_name = f"session_{test_request.session_id}"
        try:
            retrieval_response = requests.post(
                f"{DOCUMENT_PROCESSOR_URL}/retrieve",
                json={"query": test_request.query, "collection_name": collection_name, "top_k": 10, "embedding_model": session_embedding_model},
                timeout=30
            )
            retrieval_response.raise_for_status()
            retrieved_docs = retrieval_response.json().get("results", [])
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Document retrieval failed: {e}")

        # --- 4. Generate Simple RAG Answer (Before DYSK) ---
        simple_rag_answer = ""
        if retrieved_docs:
            simple_rag_answer = await _generate_answer_from_docs_via_api(
                query=test_request.query,
                docs=retrieved_docs[:5],
                model=test_request.llm_model or "llama-3.1-8b-instant"
            )

        # --- 5. Perform DYSK (CRAG) Evaluation ---
        try:
            crag_request = {"query": test_request.query, "retrieved_docs": retrieved_docs}
            crag_response = requests.post(f"{DOCUMENT_PROCESSOR_URL}/crag-evaluate", json=crag_request, timeout=60)
            crag_response.raise_for_status()
            crag_data = crag_response.json()
            crag_evaluation = crag_data.get("evaluation", {})
            final_docs = crag_data.get("filtered_docs", [])
            actual_result = crag_evaluation.get("action", "error")
        except Exception as e:
            logger.error(f"DYSK evaluation failed: {e}")
            crag_evaluation = {"error": str(e), "action": "error"}
            final_docs = []
            actual_result = "error"

        # --- 6. Generate Advanced RAG (DYSK) Answer ---
        advanced_rag_answer = ""
        if actual_result not in ["reject", "error"] and final_docs:
            advanced_rag_answer = await _generate_answer_from_docs_via_api(
                query=test_request.query,
                docs=final_docs,
                model=test_request.llm_model or "llama-3.1-8b-instant"
            )
        elif not final_docs and actual_result != "error":
            advanced_rag_answer = "DYSK katmanı tarafından tüm dokümanlar alakasız bulunduğu için yanıt üretilmedi."
        elif actual_result == "error":
             advanced_rag_answer = f"DYSK katmanında bir hata oluştuğu için yanıt üretilemedi: {crag_evaluation.get('error', 'Bilinmeyen hata')}"


        # --- 7. Calculate Lightweight Evaluation Metrics ---
        evaluation_metrics = {
            "direct_llm": calculate_answer_metrics(direct_llm_answer, test_request.query, []),
            "simple_rag": calculate_answer_metrics(simple_rag_answer, test_request.query, [d.get('text', d.get('content', '')) for d in retrieved_docs[:5]]),
            "advanced_rag": calculate_answer_metrics(advanced_rag_answer, test_request.query, [d.get('text', d.get('content', '')) for d in final_docs])
        }
        
        # Karşılaştırmalı analiz
        comparative_analysis = {
            "direct_vs_simple": compare_answers(direct_llm_answer, simple_rag_answer),
            "simple_vs_advanced": compare_answers(simple_rag_answer, advanced_rag_answer),
            "direct_vs_advanced": compare_answers(direct_llm_answer, advanced_rag_answer)
        }

        # --- 8. Finalize and Return ---
        execution_time = (time.time() - start_time) * 1000
        expected_result = 'accept' if test_request.expected_relevant else 'reject'
        # For this comparative test, 'passed' is based on whether the DYSK layer made the correct final decision.
        passed = (actual_result == expected_result)

        results_preview = []
        for r in final_docs[:5]:
            preview = {
                "text": r.get('text', r.get('content', ''))[:250] + "...",
                "similarity_score": r.get('score', 0),
                "crag_score": r.get('crag_score', 0)
            }
            results_preview.append(preview)

        test_result = {
            "test_id": f"manual_{datetime.utcnow().timestamp()}",
            "query": test_request.query,
            "expected_relevant": test_request.expected_relevant,
            "expected_result": expected_result,
            "actual_result": actual_result,
            "passed": passed,
            "documents_retrieved": len(final_docs),
            "crag_evaluation": crag_evaluation,
            "llm_answers": {
                "direct_llm": direct_llm_answer,
                "simple_rag": simple_rag_answer,
                "advanced_rag": advanced_rag_answer,
            },
            "evaluation_metrics": evaluation_metrics,
            "comparative_analysis": comparative_analysis,
            "execution_time_ms": round(execution_time, 2),
            "llm_model": test_request.llm_model or 'default',
            "embedding_model": session_embedding_model,
            "session_id": test_request.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results_preview": results_preview
        }
        
        return {
            "success": True,
            "test_result": test_result,
            "message": "✅ Karşılaştırmalı Test Tamamlandı"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing manual test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute test: {str(e)}")


@router.post("/generate-auto-tests", summary="Generate Automatic Tests")
async def generate_automatic_tests(
    config: AutoTestConfig,
    request: Request
) -> Dict[str, Any]:
    """
    Generate automatic test queries - simple version without heavy dependencies.
    
    Args:
        config: Test generation configuration
    
    Returns:
        Generated test queries
    """
    _require_admin(request)
    
    try:
        test_queries = []
        
        # Relevant sample queries (domain-specific)
        relevant_sample_queries = [
            "Yapay zeka nedir?",
            "Makine öğrenmesi nasıl çalışır?",
            "Derin öğrenme algoritmaları",
            "Sinir ağları hakkında bilgi ver",
            "Python programlama temelleri",
            "Veri bilimi nedir?",
            "Doğal dil işleme teknikleri",
            "Bilgisayar görüşü uygulamaları",
            "Reinforcement learning kavramı",
            "Transfer learning nedir?"
        ]
        
        # Generate relevant queries
        if config.include_relevant:
            num_relevant = config.num_tests // 2 if config.include_irrelevant else config.num_tests
            selected_relevant = random.sample(relevant_sample_queries, min(num_relevant, len(relevant_sample_queries)))
            
            for query in selected_relevant:
                test_queries.append({
                    "query": query,
                    "expected_relevant": True,
                    "category": "auto_relevant",
                    "source_chunk": None
                })
        
        # Add irrelevant queries
        if config.include_irrelevant:
            irrelevant_queries = [
                "Bugün hava nasıl?",
                "En iyi pizza tarifi nedir?",
                "Futbol maçı ne zaman?",
                "Dolar kuru kaç?",
                "Hangi restoran iyi?",
                "Film önerisi ver",
                "Müzik öner",
                "Yemek tarifi",
                "Seyahat planı yap",
                "Hisse senedi tavsiyesi"
            ]
            
            num_irrelevant = config.num_tests // 2 if config.include_relevant else config.num_tests
            selected_irrelevant = random.sample(irrelevant_queries, min(num_irrelevant, len(irrelevant_queries)))
            
            for query in selected_irrelevant:
                test_queries.append({
                    "query": query,
                    "expected_relevant": False,
                    "category": "auto_irrelevant",
                    "source_chunk": None
                })
        
        return {
            "success": True,
            "test_queries": test_queries[:config.num_tests],
            "total_generated": len(test_queries[:config.num_tests]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating automatic tests: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tests: {str(e)}")


@router.post("/execute-batch", summary="Execute Batch Tests")
async def execute_batch_tests(
    test_queries: List[Dict[str, Any]] = Body(..., description="List of test queries to execute"),
    request: Request = None
) -> Dict[str, Any]:
    """
    Execute multiple test queries in batch using microservice.
    
    Args:
        test_queries: List of test queries with expected results
    
    Returns:
        Batch test results and summary
    """
    _require_admin(request)
    
    try:
        results = []
        total_time = 0
        passed_count = 0
        failed_count = 0
        false_positives = 0
        false_negatives = 0
        
        for i, test_query in enumerate(test_queries):
            try:
                # Determine collection name
                collection_name = f"session_{test_query.get('session_id')}" if test_query.get('session_id') else "rag_documents"
                
                # Execute test via document processor
                start_time = time.time()
                
                response = requests.post(
                    f"{DOCUMENT_PROCESSOR_URL}/retrieve",
                    json={
                        "query": test_query['query'],
                        "collection_name": collection_name,
                        "top_k": 5,
                        "embedding_model": test_query.get('embedding_model')
                    },
                    timeout=30
                )
                
                execution_time = (time.time() - start_time) * 1000
                total_time += execution_time
                
                if response.status_code == 200:
                    retrieval_result = response.json()
                    retrieved_results = retrieval_result.get("results", [])
                    
                    # Determine result
                    actual_result = 'accept' if len(retrieved_results) > 0 else 'reject'
                    expected_result = 'accept' if test_query['expected_relevant'] else 'reject'
                    passed = (actual_result == expected_result)
                    
                    if passed:
                        passed_count += 1
                    else:
                        failed_count += 1
                        
                        # Track false positives/negatives
                        if actual_result == 'accept' and expected_result == 'reject':
                            false_positives += 1
                        elif actual_result == 'reject' and expected_result == 'accept':
                            false_negatives += 1
                    
                    results.append({
                        "test_id": f"batch_{i}_{datetime.utcnow().timestamp()}",
                        "query": test_query['query'],
                        "expected_relevant": test_query['expected_relevant'],
                        "expected_result": expected_result,
                        "actual_result": actual_result,
                        "passed": passed,
                        "documents_retrieved": len(retrieved_results),
                        "execution_time_ms": round(execution_time, 2),
                        "category": test_query.get('category', 'batch')
                    })
                else:
                    # Service error
                    results.append({
                        "test_id": f"batch_{i}_{datetime.utcnow().timestamp()}",
                        "query": test_query['query'],
                        "error": f"Service error: {response.status_code}",
                        "passed": False
                    })
                    failed_count += 1
            
            except Exception as e:
                logger.error(f"Error executing test {i}: {e}")
                results.append({
                    "test_id": f"batch_{i}_{datetime.utcnow().timestamp()}",
                    "query": test_query['query'],
                    "error": str(e),
                    "passed": False
                })
                failed_count += 1
        
        # Calculate summary
        total_tests = len(test_queries)
        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        avg_execution_time = (total_time / total_tests) if total_tests > 0 else 0
        false_positive_rate = (false_positives / total_tests * 100) if total_tests > 0 else 0
        false_negative_rate = (false_negatives / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": round(success_rate, 2),
            "avg_execution_time_ms": round(avg_execution_time, 2),
            "false_positive_rate": round(false_positive_rate, 2),
            "false_negative_rate": round(false_negative_rate, 2),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "summary": summary,
            "results": results,
            "message": f"✅ {success_rate:.1f}% success rate"
        }
    
    except Exception as e:
        logger.error(f"Error executing batch tests: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute batch tests: {str(e)}")


@router.get("/sample-queries", summary="Get Sample Test Queries")
async def get_sample_queries(
    category: str = Query("all", description="Query category: relevant, irrelevant, or all"),
    request: Request = None
) -> Dict[str, Any]:
    """
    Get sample test queries for manual testing.
    
    Returns:
        List of sample queries categorized by relevance
    """
    _require_admin(request)
    
    relevant_queries = [
        "Makine öğrenmesi nedir?",
        "Sinir ağları nasıl çalışır?",
        "Gradient descent algoritması nedir?",
        "Overfitting nasıl önlenir?",
        "Backpropagation nedir?",
        "Deep learning nedir?",
        "CNN ne işe yarar?",
        "RNN ve LSTM farkı nedir?",
        "Transfer learning nedir?",
        "Data augmentation ne demek?"
    ]
    
    irrelevant_queries = [
        "Bugün hava nasıl?",
        "En iyi restoran hangisi?",
        "Futbol maçı ne zaman?",
        "Dolar kuru kaç?",
        "Yemek tarifi ver",
        "Müzik öner",
        "Film önerisi",
        "Hisse senedi tavsiyesi",
        "Seyahat planı yap",
        "Şaka anlat"
    ]
    
    if category == "relevant":
        queries = [{"query": q, "expected_relevant": True} for q in relevant_queries]
    elif category == "irrelevant":
        queries = [{"query": q, "expected_relevant": False} for q in irrelevant_queries]
    else:
        queries = (
            [{"query": q, "expected_relevant": True} for q in relevant_queries] +
            [{"query": q, "expected_relevant": False} for q in irrelevant_queries]
        )
    
    return {
        "success": True,
        "queries": queries,
        "total": len(queries),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/debug-status", summary="Debug CRAG Status (No Auth)")
async def debug_crag_status() -> Dict[str, Any]:
    """
    Debug endpoint to test CRAG status without authentication.
    For development/testing purposes only.
    """
    try:
        # CRAG evaluation moved to Document Processing Service
        crag_enabled = True  # Available via Document Processing Service
        crag_evaluator_available = True
        crag_config = {"status": "available_via_document_service"}
        
        return {
            "success": True,
            "debug": True,
            "message": "CRAG Debug Status - No Authentication Required",
            "crag_enabled": crag_enabled,
            "crag_evaluator_available": crag_evaluator_available,
            "crag_config": crag_config,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in debug CRAG status: {e}")
        return {
            "success": False,
            "debug": True,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
