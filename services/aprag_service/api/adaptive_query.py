"""
Adaptive Query Endpoint (Faz 5)
Full Eğitsel-KBRAG Pipeline Integration

This endpoint integrates all Eğitsel-KBRAG components:
- Faz 2: CACS document scoring
- Faz 3: Pedagogical monitors (ZPD, Bloom, Cognitive Load)
- Faz 4: Emoji feedback preparation
- Full personalized learning experience
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

# Import all Eğitsel-KBRAG components
try:
    from database.database import DatabaseManager
    from config.feature_flags import FeatureFlags
    from business_logic.cacs import get_cacs_scorer
    from business_logic.pedagogical import (
        get_zpd_calculator,
        get_bloom_detector,
        get_cognitive_load_manager
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from database.database import DatabaseManager
    from config.feature_flags import FeatureFlags
    from business_logic.cacs import get_cacs_scorer
    from business_logic.pedagogical import (
        get_zpd_calculator,
        get_bloom_detector,
        get_cognitive_load_manager
    )

# DB manager will be injected via dependency
db_manager = None

logger = logging.getLogger(__name__)
router = APIRouter()


class RAGDocument(BaseModel):
    """RAG document model"""
    doc_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    score: float = Field(0.5, ge=0.0, le=1.0, description="RAG similarity score")
    metadata: Optional[Dict[str, Any]] = None


class AdaptiveQueryRequest(BaseModel):
    """Full Eğitsel-KBRAG pipeline request"""
    user_id: str = Field(..., description="Student user ID")
    session_id: str = Field(..., description="Session ID")
    query: str = Field(..., description="Student's question")
    rag_documents: List[RAGDocument] = Field(..., description="Documents from RAG system")
    rag_response: str = Field(..., description="Original RAG response")


class DocumentScore(BaseModel):
    """Scored document"""
    doc_id: str
    final_score: float
    base_score: float
    personal_score: float
    global_score: float
    context_score: float
    rank: int


class PedagogicalContext(BaseModel):
    """Pedagogical analysis context"""
    zpd_level: str
    zpd_recommended: str
    zpd_success_rate: float
    bloom_level: str
    bloom_level_index: int
    cognitive_load: float
    needs_simplification: bool


class AdaptiveQueryResponse(BaseModel):
    """Full Eğitsel-KBRAG pipeline response"""
    # Main response
    personalized_response: str
    original_response: str
    interaction_id: int
    
    # Document scoring
    top_documents: List[DocumentScore]
    cacs_applied: bool
    
    # Pedagogical context
    pedagogical_context: PedagogicalContext
    
    # Feedback
    feedback_emoji_options: List[str] = ['😊', '👍', '😐', '❌']
    
    # Metadata
    processing_time_ms: Optional[float] = None
    components_active: Dict[str, bool]


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    global db_manager
    if db_manager is None:
        import os
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
        db_manager = DatabaseManager(db_path)
    return db_manager


@router.post("", response_model=AdaptiveQueryResponse)
async def adaptive_query(
    request: AdaptiveQueryRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Full Eğitsel-KBRAG Pipeline
    
    **Complete adaptive learning workflow:**
    
    1. **Student Profile & History**
       - Load student profile
       - Get recent interactions (last 20)
    
    2. **CACS Document Scoring** (Faz 2)
       - Score all RAG documents
       - Personalized ranking
       - Select top documents
    
    3. **Pedagogical Analysis** (Faz 3)
       - ZPD: Optimal difficulty level
       - Bloom: Cognitive level detection
       - Cognitive Load: Complexity management
    
    4. **Personalized Response Generation**
       - Adapt to student's ZPD level
       - Match Bloom taxonomy level
       - Optimize cognitive load
    
    5. **Interaction Recording**
       - Save to database
       - Prepare for emoji feedback (Faz 4)
    
    **Requires:** All Eğitsel-KBRAG components enabled
    """
    
    # Check if Eğitsel-KBRAG is enabled
    if not FeatureFlags.is_egitsel_kbrag_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Eğitsel-KBRAG is not enabled"
        )
    
    start_time = datetime.now()
    
    try:
        logger.info(f"🚀 Adaptive query for user {request.user_id}: {request.query[:60]}...")
        
        # Track which components are active
        components_active = {
            'cacs': FeatureFlags.is_cacs_enabled(),
            'zpd': FeatureFlags.is_zpd_enabled(),
            'bloom': FeatureFlags.is_bloom_enabled(),
            'cognitive_load': FeatureFlags.is_cognitive_load_enabled(),
            'emoji_feedback': FeatureFlags.is_emoji_feedback_enabled()
        }
        
        # === 1. STUDENT PROFILE & HISTORY ===
        logger.info("1️⃣ Loading student profile and history...")
        
        profile = db.execute_query(
            "SELECT * FROM student_profiles WHERE user_id = ? AND session_id = ?",
            (request.user_id, request.session_id)
        )
        student_profile = profile[0] if profile else {}
        
        if not student_profile:
            logger.info("  → New student, creating default profile")
        
        recent_interactions = db.execute_query(
            """
            SELECT * FROM student_interactions 
            WHERE user_id = ? AND session_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            (request.user_id, request.session_id)
        )
        
        logger.info(f"  → Profile loaded, {len(recent_interactions)} past interactions")
        
        # === 2. CACS DOCUMENT SCORING (Faz 2) ===
        top_docs = []
        cacs_applied = False
        
        if components_active['cacs']:
            logger.info("2️⃣ CACS: Scoring documents...")
            
            cacs_scorer = get_cacs_scorer()
            
            if cacs_scorer:
                # Fetch global scores
                global_scores = {}
                for doc in request.rag_documents:
                    score_data = db.execute_query(
                        "SELECT * FROM document_global_scores WHERE doc_id = ?",
                        (doc.doc_id,)
                    )
                    if score_data:
                        global_scores[doc.doc_id] = score_data[0]
                
                # Score each document
                scored_docs = []
                for doc in request.rag_documents:
                    cacs_result = cacs_scorer.calculate_score(
                        doc_id=doc.doc_id,
                        base_score=doc.score,
                        student_profile=student_profile,
                        conversation_history=recent_interactions,
                        global_scores=global_scores,
                        current_query=request.query
                    )
                    
                    scored_docs.append(DocumentScore(
                        doc_id=doc.doc_id,
                        final_score=cacs_result['final_score'],
                        base_score=cacs_result['base_score'],
                        personal_score=cacs_result['personal_score'],
                        global_score=cacs_result['global_score'],
                        context_score=cacs_result['context_score'],
                        rank=0  # Will be set after sorting
                    ))
                
                # Sort and rank
                scored_docs.sort(key=lambda x: x.final_score, reverse=True)
                for rank, doc in enumerate(scored_docs, start=1):
                    doc.rank = rank
                
                top_docs = scored_docs[:3]
                cacs_applied = True
                
                logger.info(f"  → {len(scored_docs)} documents scored, top 3 selected")
                logger.info(f"  → Top: {top_docs[0].doc_id} (score: {top_docs[0].final_score:.3f})")
        else:
            logger.info("2️⃣ CACS: Disabled, using base scores")
            # Use base scores
            docs_with_scores = [(doc, doc.score) for doc in request.rag_documents]
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (doc, score) in enumerate(docs_with_scores[:3], start=1):
                top_docs.append(DocumentScore(
                    doc_id=doc.doc_id,
                    final_score=score,
                    base_score=score,
                    personal_score=0.5,
                    global_score=0.5,
                    context_score=0.5,
                    rank=rank
                ))
        
        # === 3. PEDAGOGICAL ANALYSIS (Faz 3) ===
        logger.info("3️⃣ Pedagogical analysis...")
        
        # ZPD
        zpd_info = {'current_level': 'intermediate', 'recommended_level': 'intermediate', 'success_rate': 0.5}
        if components_active['zpd']:
            zpd_calc = get_zpd_calculator()
            if zpd_calc:
                zpd_info = zpd_calc.calculate_zpd_level(recent_interactions, student_profile)
                logger.info(f"  → ZPD: {zpd_info['current_level']} → {zpd_info['recommended_level']} "
                          f"(success: {zpd_info['success_rate']:.2f})")
        
        # Bloom Taxonomy
        bloom_info = {'level': 'understand', 'level_index': 2}
        if components_active['bloom']:
            bloom_det = get_bloom_detector()
            if bloom_det:
                bloom_info = bloom_det.detect_bloom_level(request.query)
                logger.info(f"  → Bloom: Level {bloom_info['level_index']} ({bloom_info['level']})")
        
        # Cognitive Load
        cognitive_info = {'total_load': 0.5, 'needs_simplification': False}
        if components_active['cognitive_load']:
            cog_load_mgr = get_cognitive_load_manager()
            if cog_load_mgr:
                cognitive_info = cog_load_mgr.calculate_cognitive_load(
                    request.rag_response,
                    request.query
                )
                logger.info(f"  → Cognitive Load: {cognitive_info['total_load']:.2f} "
                          f"(simplify: {cognitive_info['needs_simplification']})")
        
        pedagogical_context = PedagogicalContext(
            zpd_level=zpd_info['current_level'],
            zpd_recommended=zpd_info['recommended_level'],
            zpd_success_rate=zpd_info['success_rate'],
            bloom_level=bloom_info['level'],
            bloom_level_index=bloom_info['level_index'],
            cognitive_load=cognitive_info['total_load'],
            needs_simplification=cognitive_info['needs_simplification']
        )
        
        # === 4. PERSONALIZED RESPONSE GENERATION ===
        logger.info("4️⃣ Generating personalized response...")
        
        personalized_response = _generate_personalized_response(
            request.rag_response,
            request.query,
            pedagogical_context.dict(),
            [doc.doc_id for doc in top_docs]
        )
        
        # Simplify if needed
        if cognitive_info['needs_simplification'] and components_active['cognitive_load']:
            cog_load_mgr = get_cognitive_load_manager()
            if cog_load_mgr:
                chunks = cog_load_mgr.chunk_response(personalized_response)
                if len(chunks) > 1:
                    personalized_response = "\n\n".join([f"**Bölüm {i}:**\n{chunk}" for i, chunk in enumerate(chunks, 1)])
                    logger.info(f"  → Response chunked into {len(chunks)} parts")
        
        # === 5. INTERACTION RECORDING ===
        logger.info("5️⃣ Recording interaction...")
        
        interaction_id = db.execute_insert(
            """
            INSERT INTO student_interactions 
            (user_id, session_id, query, response, personalized_response,
             processing_time_ms, model_used, chain_type, sources, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.user_id,
                request.session_id,
                request.query,
                request.rag_response,
                personalized_response,
                0,  # Will be updated
                'egitsel-kbrag',
                'adaptive',
                json.dumps([{'doc_id': d.doc_id, 'score': d.final_score} for d in top_docs]),
                json.dumps({
                    'zpd_level': zpd_info['recommended_level'],
                    'bloom_level': bloom_info['level'],
                    'cognitive_load': cognitive_info['total_load'],
                    'cacs_applied': cacs_applied
                })
            )
        )
        
        # Calculate processing time
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update processing time
        db.execute_update(
            "UPDATE student_interactions SET processing_time_ms = ? WHERE interaction_id = ?",
            (processing_time_ms, interaction_id)
        )
        
        logger.info(f"✅ Adaptive query completed: interaction_id={interaction_id}, "
                   f"time={processing_time_ms:.0f}ms")
        
        # === 6. RESPONSE ===
        return AdaptiveQueryResponse(
            personalized_response=personalized_response,
            original_response=request.rag_response,
            interaction_id=interaction_id,
            top_documents=top_docs,
            cacs_applied=cacs_applied,
            pedagogical_context=pedagogical_context,
            feedback_emoji_options=['😊', '👍', '😐', '❌'],
            processing_time_ms=processing_time_ms,
            components_active=components_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adaptive query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adaptive query failed: {str(e)}"
        )


def _generate_personalized_response(
    original_response: str,
    query: str,
    pedagogical_context: Dict[str, Any],
    top_doc_ids: List[str]
) -> str:
    """
    Generate personalized response based on pedagogical context
    
    In production, this would call an LLM with a specialized prompt.
    For now, we enhance the response with pedagogical markers.
    """
    
    # Extract context
    zpd_level = pedagogical_context.get('zpd_recommended', 'intermediate')
    bloom_level = pedagogical_context.get('bloom_level', 'understand')
    needs_simplification = pedagogical_context.get('needs_simplification', False)
    
    # Build personalized response
    personalized = f"# {query}\n\n"
    
    # Add pedagogical context (subtle)
    if zpd_level == 'beginner' or zpd_level == 'elementary':
        personalized += "_Bu açıklama senin seviyene göre basitleştirildi._\n\n"
    elif zpd_level == 'advanced' or zpd_level == 'expert':
        personalized += "_Bu açıklama ileri seviye detaylar içeriyor._\n\n"
    
    # Add bloom level specific intro
    bloom_intros = {
        'remember': '📝 **Temel Tanım:**\n',
        'understand': '💡 **Açıklama:**\n',
        'apply': '🔧 **Pratik Uygulama:**\n',
        'analyze': '🔍 **Detaylı Analiz:**\n',
        'evaluate': '⚖️ **Değerlendirme:**\n',
        'create': '🎨 **Yaratıcı Çözüm:**\n'
    }
    
    personalized += bloom_intros.get(bloom_level, '💬 **Yanıt:**\n')
    personalized += original_response
    
    # Add simplification note if needed
    if needs_simplification:
        personalized += "\n\n_💡 Bu yanıt daha kolay anlaşılması için parçalara ayrıldı._"
    
    return personalized


@router.get("/status")
async def get_adaptive_query_status():
    """
    Get Adaptive Query Pipeline status
    
    Returns the status of all Eğitsel-KBRAG components.
    """
    return {
        "pipeline": "Eğitsel-KBRAG Full Pipeline",
        "status": "ready" if FeatureFlags.is_egitsel_kbrag_enabled() else "disabled",
        "components": {
            "cacs": FeatureFlags.is_cacs_enabled(),
            "zpd": FeatureFlags.is_zpd_enabled(),
            "bloom": FeatureFlags.is_bloom_enabled(),
            "cognitive_load": FeatureFlags.is_cognitive_load_enabled(),
            "emoji_feedback": FeatureFlags.is_emoji_feedback_enabled()
        },
        "description": "Full adaptive learning pipeline integrating all Eğitsel-KBRAG components"
    }

