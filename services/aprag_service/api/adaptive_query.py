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
import requests

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
    from api.personalization import PersonalizeRequest, personalize_response
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
    from api.personalization import PersonalizeRequest, personalize_response

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
    
    # Debug/Research data (optional, for research purposes)
    personalization_data: Optional[Dict[str, Any]] = None


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
        
        # Track which components are active (use session-specific settings)
        # First check session-specific settings from database
        session_settings_dict = {}
        try:
            session_settings_result = db.execute_query(
                "SELECT * FROM session_settings WHERE session_id = ?",
                (request.session_id,)
            )
            if session_settings_result and len(session_settings_result) > 0:
                # Convert SQLite Row to dict
                row = session_settings_result[0]
                session_settings_dict = dict(row) if hasattr(row, 'keys') else row
                logger.info(f"📋 Loaded session settings for {request.session_id}")
                logger.info(f"   Session settings keys: {list(session_settings_dict.keys())}")
            else:
                logger.warning(f"⚠️ No session settings found for session {request.session_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load session settings: {e}", exc_info=True)
        
        # First check if Eğitsel-KBRAG is enabled (required for all features)
        egitsel_kbrag_enabled = FeatureFlags.is_egitsel_kbrag_enabled()
        logger.info(f"🔍 Eğitsel-KBRAG enabled: {egitsel_kbrag_enabled}")
        if not egitsel_kbrag_enabled:
            logger.warning("⚠️ Eğitsel-KBRAG is disabled, all features will be disabled")
        
        # Check feature flags with session-specific overrides
        def get_feature_flag(feature_name: str, default_func) -> bool:
            """Get feature flag with session-specific override"""
            # Eğitsel-KBRAG must be enabled first
            if not egitsel_kbrag_enabled:
                logger.debug(f"  → {feature_name}: disabled (Eğitsel-KBRAG off)")
                return False
            
            # First check session-specific setting
            if session_settings_dict:
                session_key = f"enable_{feature_name}"
                # Try multiple ways to access the value
                value = None
                if session_key in session_settings_dict:
                    value = session_settings_dict[session_key]
                elif hasattr(session_settings_dict, session_key):
                    value = getattr(session_settings_dict, session_key)
                
                if value is not None:
                    bool_value = bool(value)
                    logger.info(f"  ✅ {feature_name}: session override = {bool_value} (from session_settings)")
                    return bool_value
            
            # Fallback to default function
            default_value = default_func()
            logger.info(f"  → {feature_name}: default = {default_value} (no session override)")
            return default_value
        
        components_active = {
            'cacs': get_feature_flag('cacs', FeatureFlags.is_cacs_enabled),
            'zpd': get_feature_flag('zpd', FeatureFlags.is_zpd_enabled),
            'bloom': get_feature_flag('bloom', FeatureFlags.is_bloom_enabled),
            'cognitive_load': get_feature_flag('cognitive_load', FeatureFlags.is_cognitive_load_enabled),
            'emoji_feedback': get_feature_flag('emoji_feedback', FeatureFlags.is_emoji_feedback_enabled),
            'progressive_assessment': FeatureFlags.is_progressive_assessment_enabled(request.session_id),
            'personalized_responses': get_feature_flag('personalized_responses', FeatureFlags.is_personalized_responses_enabled)
        }
        
        logger.info(f"🔧 Components active: {components_active}")
        if session_settings_dict:
            logger.info(f"📋 Session settings values:")
            for key in ['enable_cacs', 'enable_zpd', 'enable_bloom', 'enable_cognitive_load', 'enable_emoji_feedback']:
                value = session_settings_dict.get(key, 'N/A')
                logger.info(f"   - {key}: {value}")
        
        # === 1. STUDENT PROFILE & HISTORY ===
        logger.info("1️⃣ Loading student profile and history...")
        
        profile = db.execute_query(
            "SELECT * FROM student_profiles WHERE user_id = ? AND session_id = ?",
            (request.user_id, request.session_id)
        )
        student_profile = profile[0] if profile else {}
        
        if not student_profile:
            logger.info("  → New student, creating default profile")
            # Create default profile for new student
            try:
                db.execute_insert(
                    """
                    INSERT INTO student_profiles
                    (user_id, session_id, average_understanding, average_satisfaction,
                     total_interactions, total_feedback_count, last_updated)
                    VALUES (?, ?, 3.0, 3.0, 0, 0, CURRENT_TIMESTAMP)
                    """,
                    (request.user_id, request.session_id)
                )
                logger.info(f"  → Created default profile for user {request.user_id}")
                student_profile = {
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "average_understanding": 3.0,
                    "average_satisfaction": 3.0,
                    "total_interactions": 0,
                    "total_feedback_count": 0
                }
            except Exception as e:
                logger.warning(f"  → Failed to create default profile: {e}")
                student_profile = {}
        
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
                # Fetch global scores (table may not exist yet, handle gracefully)
                global_scores = {}
                try:
                    for doc in request.rag_documents:
                        score_data = db.execute_query(
                            "SELECT * FROM document_global_scores WHERE doc_id = ?",
                            (doc.doc_id,)
                        )
                        if score_data:
                            global_scores[doc.doc_id] = score_data[0]
                except Exception as e:
                    logger.warning(f"Could not fetch global scores (table may not exist): {e}")
                    global_scores = {}  # Continue with empty global_scores
                
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
        logger.info(f"   Components active check: zpd={components_active['zpd']}, bloom={components_active['bloom']}, cognitive_load={components_active['cognitive_load']}")
        
        # ZPD
        zpd_info = {'current_level': 'intermediate', 'recommended_level': 'intermediate', 'success_rate': 0.5}
        if components_active['zpd']:
            try:
                zpd_calc = get_zpd_calculator()
                if zpd_calc:
                    zpd_info = zpd_calc.calculate_zpd_level(recent_interactions, student_profile)
                    logger.info(f"  → ZPD: {zpd_info['current_level']} → {zpd_info['recommended_level']} "
                              f"(success: {zpd_info['success_rate']:.2f})")
                else:
                    logger.warning("  ⚠️ ZPD calculator not available")
            except Exception as e:
                logger.error(f"  ❌ ZPD calculation failed: {e}", exc_info=True)
        else:
            logger.warning("  ⚠️ ZPD disabled (components_active['zpd'] = False)")
        
        # Bloom Taxonomy
        bloom_info = {'level': 'understand', 'level_index': 2}
        if components_active['bloom']:
            try:
                bloom_det = get_bloom_detector()
                if bloom_det:
                    bloom_info = bloom_det.detect_bloom_level(request.query)
                    logger.info(f"  → Bloom: Level {bloom_info['level_index']} ({bloom_info['level']})")
                else:
                    logger.warning("  ⚠️ Bloom detector not available")
            except Exception as e:
                logger.error(f"  ❌ Bloom detection failed: {e}", exc_info=True)
        else:
            logger.warning("  ⚠️ Bloom disabled (components_active['bloom'] = False)")
        
        # Cognitive Load
        cognitive_info = {'total_load': 0.5, 'needs_simplification': False}
        if components_active['cognitive_load']:
            try:
                cog_load_mgr = get_cognitive_load_manager()
                if cog_load_mgr:
                    cognitive_info = cog_load_mgr.calculate_cognitive_load(
                        request.rag_response,
                        request.query
                    )
                    logger.info(f"  → Cognitive Load: {cognitive_info['total_load']:.2f} "
                              f"(simplify: {cognitive_info['needs_simplification']})")
                else:
                    logger.warning("  ⚠️ Cognitive load manager not available")
            except Exception as e:
                logger.error(f"  ❌ Cognitive load calculation failed: {e}", exc_info=True)
        else:
            logger.warning("  ⚠️ Cognitive Load disabled (components_active['cognitive_load'] = False)")
        
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
        
        # Generate personalized response using LLM-based personalization
        personalized_response = request.rag_response
        personalization_data = {}
        
        if components_active['personalized_responses']:
            try:
                top_doc_ids = [doc.doc_id for doc in top_docs[:3]]
                personalized_result = await _generate_personalized_response(
                    original_response=request.rag_response,
                    query=request.query,
                    pedagogical_context={
                        'zpd_recommended': zpd_info.get('recommended_level', 'intermediate'),
                        'bloom_level': bloom_info.get('level', 'understand'),
                        'cognitive_load': cognitive_info.get('total_load', 0.5),
                        'needs_simplification': cognitive_info.get('needs_simplification', False)
                    },
                    top_doc_ids=top_doc_ids,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    db=db
                )
                
                # Handle both old format (string) and new format (tuple)
                if isinstance(personalized_result, tuple):
                    personalized_response, personalization_data = personalized_result
                else:
                    personalized_response = personalized_result
                    personalization_data = {}
                
                logger.info(f"  ✅ Personalized response generated ({len(personalized_response)} chars)")
            except Exception as e:
                logger.error(f"  ❌ Personalization failed: {e}", exc_info=True)
                # Fallback to original response
                personalized_response = request.rag_response
                personalization_data = {
                    "personalization_factors": {
                        "difficulty_level": zpd_info.get('recommended_level', 'intermediate'),
                        "explanation_style": "balanced",
                        "needs_examples": False
                    },
                    "zpd_info": zpd_info,
                    "bloom_info": bloom_info,
                    "cognitive_load": cognitive_info,
                    "pedagogical_instructions": "",
                    "difficulty_adjustment": zpd_info.get('recommended_level'),
                    "explanation_level": "balanced",
                    "error": str(e)
                }
        else:
            logger.info("  ⚠️ Personalized responses disabled, using original response")
            personalization_data = {
                "personalization_factors": {
                    "difficulty_level": zpd_info.get('recommended_level', 'intermediate'),
                    "explanation_style": "balanced",
                    "needs_examples": False
                },
                "zpd_info": zpd_info,
                "bloom_info": bloom_info,
                "cognitive_load": cognitive_info,
                "pedagogical_instructions": "",
                "difficulty_adjustment": zpd_info.get('recommended_level'),
                "explanation_level": "balanced"
            }
        
        # Ensure personalization_data is a dict
        if personalization_data is None:
            personalization_data = {}
        
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
            (user_id, session_id, query, original_response, personalized_response,
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
        # Build comprehensive debug data for analysis and reporting
        comprehensive_debug_data = {
            # Request parameters
            "request_params": {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "query": request.query,
                "rag_documents_count": len(request.rag_documents),
                "rag_response_length": len(request.rag_response)
            },
            
            # Session settings
            "session_settings": session_settings_dict if session_settings_dict else None,
            
            # Feature flags status
            "feature_flags": {
                "egitsel_kbrag_enabled": egitsel_kbrag_enabled,
                "components_active": components_active,
                "session_settings_loaded": bool(session_settings_dict)
            },
            
            # Student profile
            "student_profile": {
                "exists": bool(student_profile),
                "average_understanding": student_profile.get('average_understanding') if student_profile else None,
                "average_satisfaction": student_profile.get('average_satisfaction') if student_profile else None,
                "total_interactions": student_profile.get('total_interactions', 0) if student_profile else 0,
                "total_feedback_count": student_profile.get('total_feedback_count', 0) if student_profile else 0
            },
            
            # Recent interactions
            "recent_interactions": {
                "count": len(recent_interactions),
                "last_5_interactions": [
                    {
                        "query": i.get('query', '')[:100] if isinstance(i, dict) else str(i)[:100],
                        "timestamp": i.get('timestamp') if isinstance(i, dict) else None,
                        "emoji_feedback": i.get('emoji_feedback') if isinstance(i, dict) else None,
                        "feedback_score": i.get('feedback_score') if isinstance(i, dict) else None
                    }
                    for i in recent_interactions[:5]
                ] if recent_interactions else []
            },
            
            # CACS scoring details
            "cacs_scoring": {
                "applied": cacs_applied,
                "documents_scored": len(top_docs),
                "scoring_details": [
                    {
                        "doc_id": doc.doc_id,
                        "final_score": doc.final_score,
                        "base_score": doc.base_score,
                        "personal_score": doc.personal_score,
                        "global_score": doc.global_score,
                        "context_score": doc.context_score,
                        "rank": doc.rank
                    }
                    for doc in top_docs
                ] if top_docs else [],
                "reason": "CACS applied successfully" if cacs_applied else ("CACS disabled" if not components_active['cacs'] else "CACS scorer not available")
            },
            
            # Pedagogical analysis details
            "pedagogical_analysis": {
                "zpd": {
                    "enabled": components_active['zpd'],
                    "current_level": zpd_info.get('current_level'),
                    "recommended_level": zpd_info.get('recommended_level'),
                    "success_rate": zpd_info.get('success_rate'),
                    "calculation_method": "zpd_calculator" if (components_active['zpd'] and zpd_info.get('current_level') != 'intermediate') else ("not_applied" if not components_active['zpd'] else "zpd_calculator")
                },
                "bloom": {
                    "enabled": components_active['bloom'],
                    "level": bloom_info.get('level'),
                    "level_index": bloom_info.get('level_index'),
                    "confidence": bloom_info.get('confidence'),
                    "detection_method": "bloom_detector" if (components_active['bloom'] and bloom_info.get('level') != 'understand') else ("not_applied" if not components_active['bloom'] else "bloom_detector")
                },
                "cognitive_load": {
                    "enabled": components_active['cognitive_load'],
                    "total_load": cognitive_info.get('total_load'),
                    "needs_simplification": cognitive_info.get('needs_simplification'),
                    "calculation_method": "cognitive_load_manager" if (components_active['cognitive_load'] and cognitive_info.get('total_load') != 0.5) else ("not_applied" if not components_active['cognitive_load'] else "cognitive_load_manager")
                }
            },
            
            # Personalization details
            "personalization": {
                "applied": bool(personalization_data and personalization_data.get('personalization_factors')),
                "personalization_factors": personalization_data.get('personalization_factors') if personalization_data else None,
                "zpd_info": personalization_data.get('zpd_info') if personalization_data else None,
                "bloom_info": personalization_data.get('bloom_info') if personalization_data else None,
                "cognitive_load": personalization_data.get('cognitive_load') if personalization_data else None,
                "pedagogical_instructions": personalization_data.get('pedagogical_instructions') if personalization_data else None,
                "difficulty_adjustment": personalization_data.get('difficulty_adjustment') if personalization_data else None,
                "explanation_level": personalization_data.get('explanation_level') if personalization_data else None,
                "reason": "Personalization applied successfully" if (personalization_data and personalization_data.get('personalization_factors')) else ("Personalization disabled" if not components_active['personalized_responses'] else "Personalization service unavailable")
            },
            
            # Response comparison
            "response_comparison": {
                "original_length": len(request.rag_response),
                "personalized_length": len(personalized_response),
                "length_difference": len(personalized_response) - len(request.rag_response),
                "is_different": personalized_response != request.rag_response,
                "similarity_ratio": len(set(request.rag_response.split()) & set(personalized_response.split())) / max(len(set(request.rag_response.split())), len(set(personalized_response.split())), 1) if request.rag_response and personalized_response else 0
            },
            
            # Timing breakdown
            "timing": {
                "total_processing_ms": processing_time_ms,
                "breakdown": {
                    "profile_loading": 0,  # Could be tracked if needed
                    "cacs_scoring": 0,  # Could be tracked if needed
                    "pedagogical_analysis": 0,  # Could be tracked if needed
                    "personalization": 0,  # Could be tracked if needed
                    "interaction_recording": 0  # Could be tracked if needed
                }
            },
            
            # Interaction metadata
            "interaction_metadata": {
                "interaction_id": interaction_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "chain_type": "adaptive",
                "model_used": "egitsel-kbrag"
            }
        }
        
        # Merge comprehensive_debug into personalization_data
        final_personalization_data = personalization_data.copy() if personalization_data else {}
        final_personalization_data["comprehensive_debug"] = comprehensive_debug_data
        
        logger.info(f"📊 Comprehensive debug data prepared: {len(comprehensive_debug_data)} sections")
        logger.info(f"   - Feature flags: {comprehensive_debug_data.get('feature_flags', {}).get('components_active', {})}")
        logger.info(f"   - Student profile exists: {comprehensive_debug_data.get('student_profile', {}).get('exists', False)}")
        logger.info(f"   - CACS applied: {comprehensive_debug_data.get('cacs_scoring', {}).get('applied', False)}")
        
        return AdaptiveQueryResponse(
            personalized_response=personalized_response,
            original_response=request.rag_response,
            interaction_id=interaction_id,
            top_documents=top_docs,
            cacs_applied=cacs_applied,
            pedagogical_context=pedagogical_context,
            feedback_emoji_options=['😊', '👍', '😐', '❌'],
            processing_time_ms=processing_time_ms,
            components_active=components_active,
            personalization_data=final_personalization_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adaptive query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adaptive query failed: {str(e)}"
        )


async def _generate_personalized_response(
    original_response: str,
    query: str,
    pedagogical_context: Dict[str, Any],
    top_doc_ids: List[str],
    user_id: str,
    session_id: str,
    db: DatabaseManager
) -> str:
    """
    Generate personalized response using the personalization service with real LLM
    
    This now calls the personalization service for proper LLM-based personalization
    instead of using templates.
    """
    
    try:
        logger.info(f"🧠 Generating personalized response using LLM service...")
        
        # Create personalization request
        personalize_req = PersonalizeRequest(
            user_id=user_id,
            session_id=session_id,
            query=query,
            original_response=original_response,
            context={
                'pedagogical_context': pedagogical_context,
                'top_doc_ids': top_doc_ids
            }
        )
        
        # Call the personalization service
        personalization_result = await personalize_response(personalize_req, db)
        
        # Extract personalized response
        personalized_text = personalization_result.personalized_response
        
        # Build personalization data for debug panel
        # Note: zpd_info, bloom_info, cognitive_load are already Dict[str, Any] in PersonalizeResponse
        personalization_data = {
            "personalization_factors": personalization_result.personalization_factors,
            "zpd_info": personalization_result.zpd_info if personalization_result.zpd_info else None,
            "bloom_info": personalization_result.bloom_info if personalization_result.bloom_info else None,
            "cognitive_load": personalization_result.cognitive_load if personalization_result.cognitive_load else None,
            "pedagogical_instructions": personalization_result.pedagogical_instructions,
            "difficulty_adjustment": personalization_result.difficulty_adjustment,
            "explanation_level": personalization_result.explanation_level
        }
        
        # Log successful personalization
        factors = personalization_result.personalization_factors
        logger.info(f"  → LLM personalization applied:")
        logger.info(f"    • Difficulty: {factors.get('difficulty_level', 'N/A')}")
        logger.info(f"    • Style: {factors.get('explanation_style', 'N/A')}")
        logger.info(f"    • Examples needed: {factors.get('needs_examples', False)}")
        
        if personalization_result.zpd_info:
            logger.info(f"    • ZPD: {personalization_result.zpd_info.get('current_level')} → {personalization_result.zpd_info.get('recommended_level')}")
        
        if personalization_result.bloom_info:
            logger.info(f"    • Bloom: {personalization_result.bloom_info.get('level')} (Level {personalization_result.bloom_info.get('level_index')})")
        
        # Return tuple with personalization data
        return (personalized_text, personalization_data)
        
    except Exception as e:
        logger.warning(f"LLM personalization failed, falling back to template: {e}")
        
        # Fallback to template-based approach (original logic)
        fallback_response = _generate_template_fallback(
            original_response,
            query,
            pedagogical_context
        )
        return fallback_response, None


def _generate_template_fallback(
    original_response: str,
    query: str,
    pedagogical_context: Dict[str, Any]
) -> str:
    """
    Fallback template-based personalization (original logic)
    
    Used when LLM personalization service is unavailable.
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
    
    logger.info("  → Template fallback personalization applied")
    
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

