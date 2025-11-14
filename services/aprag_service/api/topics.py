"""
Topic-Based Learning Path Tracking endpoints
Handles topic extraction, classification, and progress tracking
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

# Import feature flags
try:
    import sys
    import os
    # Add parent directory to path to import from config
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.insert(0, parent_dir)
    from config.feature_flags import FeatureFlags
except ImportError:
    # Fallback: Define minimal version if parent config not available
    class FeatureFlags:
        @staticmethod
        def is_aprag_enabled(session_id=None):
            """Fallback implementation when feature flags config is not available"""
            return os.getenv("APRAG_ENABLED", "true").lower() == "true"

# Environment variables - Google Cloud Run compatible
# These should be set via environment variables in Cloud Run
# For Docker: use service names (e.g., http://model-inference-service:8003)
# For Cloud Run: use full URLs (e.g., https://model-inference-xxx.run.app)
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", os.getenv("MODEL_INFERENCE_URL", "http://model-inference-service:8002"))
CHROMA_SERVICE_URL = os.getenv("CHROMA_SERVICE_URL", os.getenv("CHROMADB_URL", "http://chromadb-service:8004"))
DOCUMENT_PROCESSING_URL = os.getenv("DOCUMENT_PROCESSING_URL", "http://document-processing-service:8002")


# ============================================================================
# Request/Response Models
# ============================================================================

class TopicExtractionRequest(BaseModel):
    """Request model for topic extraction"""
    session_id: str
    extraction_method: str = "llm_analysis"  # llm_analysis, manual, hybrid
    options: Optional[Dict[str, Any]] = {
        "include_subtopics": True,
        "min_confidence": 0.7,
        "max_topics": 50
    }


class TopicUpdateRequest(BaseModel):
    """Request model for updating a topic"""
    topic_title: Optional[str] = None
    topic_order: Optional[int] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    estimated_difficulty: Optional[str] = None
    prerequisites: Optional[List[int]] = None
    is_active: Optional[bool] = None


class QuestionClassificationRequest(BaseModel):
    """Request model for question classification"""
    question: str
    session_id: str
    interaction_id: Optional[int] = None


class QuestionGenerationRequest(BaseModel):
    """Request model for question generation"""
    count: int = 10
    difficulty_level: Optional[str] = None  # beginner, intermediate, advanced
    question_types: Optional[List[str]] = None  # factual, conceptual, application, analysis


class TopicResponse(BaseModel):
    """Response model for a topic"""
    topic_id: int
    session_id: str
    topic_title: str
    parent_topic_id: Optional[int]
    topic_order: int
    description: Optional[str]
    keywords: Optional[List[str]]
    estimated_difficulty: Optional[str]
    prerequisites: Optional[List[int]]
    extraction_confidence: Optional[float]
    is_active: bool


# ============================================================================
# Helper Functions
# ============================================================================

def get_db() -> DatabaseManager:
    """Get database manager dependency"""
    db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
    return DatabaseManager(db_path)


def fetch_chunks_for_session(session_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all chunks for a session from ChromaDB
    
    Args:
        session_id: Session ID
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    try:
        # Try to get chunks from document processing service
        # If that doesn't work, we'll need to query ChromaDB directly
        response = requests.get(
            f"{DOCUMENT_PROCESSING_URL}/sessions/{session_id}/chunks",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get("chunks", [])
        else:
            logger.warning(f"Could not fetch chunks from document service: {response.status_code}")
            # Fallback: return empty list, extraction will need manual input
            return []
            
    except Exception as e:
        logger.error(f"Error fetching chunks for session {session_id}: {e}")
        return []


def extract_topics_with_llm(chunks: List[Dict[str, Any]], options: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
    """
    Extract topics from chunks using LLM
    
    Args:
        chunks: List of chunk dictionaries
        options: Extraction options
        session_id: Session ID to get session-specific model configuration
        
    Returns:
        Dictionary with extracted topics
    """
    try:
        # Prepare chunks text for LLM
        chunks_text = "\n\n---\n\n".join([
            f"Chunk {i+1}:\n{chunk.get('chunk_text', chunk.get('content', chunk.get('text', '')))}"
            for i, chunk in enumerate(chunks)
        ])
        
        # Create prompt for topic extraction
        prompt = f"""Sen bir eğitim içeriği analiz uzmanısın. Aşağıdaki ders materyallerini analiz ederek konu yapısını çıkar.

DERS MATERYALLERİ:
{chunks_text[:12000]}  # Limit to 12k chars to stay within Groq 6000 token limit

LÜTFEN ŞUNLARI YAP:
1. Ana konu başlıklarını belirle (5-15 arası)
2. Her ana konu için alt başlıkları belirle
3. Konuları öğrenme sırasına göre sırala
4. Her konu için önkoşul konuları belirle
5. Her konunun zorluk seviyesini belirle (beginner, intermediate, advanced)
6. Her konu için 3-5 anahtar kelime belirle

ÇIKTI FORMATI (JSON):
{{
  "topics": [
    {{
      "topic_title": "Ana Konu Başlığı",
      "order": 1,
      "difficulty": "intermediate",
      "keywords": ["kelime1", "kelime2"],
      "prerequisites": [],
      "subtopics": [
        {{
          "topic_title": "Alt Konu",
          "order": 1,
          "keywords": ["alt1", "alt2"]
        }}
      ],
      "related_chunks": [1, 5, 12]
    }}
  ]
}}

Sadece JSON çıktısı ver, başka açıklama yapma."""

        # Get session-specific model configuration
        model_to_use = get_session_model(session_id) or "llama-3.1-8b-instant"  # Default to Groq model instead of Ollama
        
        # Call model inference service
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model_to_use,
                "max_tokens": 4096,
                "temperature": 0.3
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM service error: {response.text}")
        
        result = response.json()
        llm_output = result.get("response", "")
        
        # Parse JSON from LLM output
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            topics_data = json.loads(json_match.group())
        else:
            # Try to parse entire output as JSON
            topics_data = json.loads(llm_output)
        
        return topics_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON output: {e}")
        logger.error(f"LLM output was: {llm_output[:1000]}")  # Log first 1000 chars
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in topic extraction: {e}")
        logger.error(f"MODEL_INFERENCER_URL: {MODEL_INFERENCER_URL}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to model service: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in topic extraction: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception args: {e.args}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Topic extraction failed: {str(e)}")


def get_session_model(session_id: str) -> str:
    """
    Get the model configured for a specific session.
    
    Args:
        session_id: Session ID
        
    Returns:
        Model name to use for this session
    """
    if not session_id:
        return "llama-3.1-8b-instant"  # Default Groq model
        
    try:
        # Try to get session configuration from auth service or database
        # For now, return a reasonable default that works with the model inference service
        # TODO: Implement actual session-specific model retrieval from database
        return "llama-3.1-8b-instant"  # Use Groq as default instead of Ollama
    except Exception as e:
        logger.warning(f"Could not get session model for {session_id}: {e}")
        return "llama-3.1-8b-instant"  # Default fallback


def classify_question_with_llm(question: str, topics: List[Dict[str, Any]], session_id: str = None) -> Dict[str, Any]:
    """
    Classify a question to a topic using LLM
    
    Args:
        question: Student question
        topics: List of topics for the session
        session_id: Session ID to get session-specific model configuration
        
    Returns:
        Classification result with topic_id, confidence, etc.
    """
    try:
        # Prepare topics list for LLM
        topics_text = "\n".join([
            f"ID: {t['topic_id']}, Başlık: {t['topic_title']}, Anahtar Kelimeler: {', '.join(t.get('keywords', []))}"
            for t in topics
        ])
        
        prompt = f"""Aşağıdaki öğrenci sorusunu, verilen konu listesine göre sınıflandır.

ÖĞRENCİ SORUSU:
{question}

KONU LİSTESİ:
{topics_text}

LÜTFEN ŞUNLARI YAP:
1. Sorunun hangi konuya ait olduğunu belirle
2. Sorunun karmaşıklık seviyesini belirle (basic, intermediate, advanced)
3. Sorunun türünü belirle (factual, conceptual, application, analysis)
4. Güven skoru ver (0.0 - 1.0)

ÇIKTI FORMATI (JSON):
{{
  "topic_id": 5,
  "topic_title": "Kimyasal Bağlar",
  "confidence_score": 0.89,
  "question_complexity": "intermediate",
  "question_type": "conceptual",
  "reasoning": "Soruda kovalent bağların özellikleri soruluyor..."
}}

Sadece JSON çıktısı ver."""

        # Get session-specific model configuration
        model_to_use = get_session_model(session_id) or "llama-3.1-8b-instant"  # Default to Groq model instead of Ollama
        
        # Call model inference service
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model_to_use,
                "max_tokens": 512,
                "temperature": 0.3
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM service error: {response.text}")
        
        result = response.json()
        llm_output = result.get("response", "")
        
        # Parse JSON from LLM output
        import re
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            classification = json.loads(json_match.group())
        else:
            classification = json.loads(llm_output)
        
        return classification
        
    except Exception as e:
        logger.error(f"Error in question classification: {e}")
        raise HTTPException(status_code=500, detail=f"Question classification failed: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/extract")
async def extract_topics(request: TopicExtractionRequest):
    """
    Extract topics from session chunks using LLM analysis
    """
    # Check if APRAG is enabled
    if not FeatureFlags.is_aprag_enabled(request.session_id):
        raise HTTPException(
            status_code=403,
            detail="APRAG module is disabled. Please enable it from admin settings."
        )
    
    db = get_db()
    
    try:
        # Fetch chunks for session
        chunks = fetch_chunks_for_session(request.session_id)
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for session {request.session_id}. Please ensure documents are processed."
            )
        
        # Extract topics using LLM
        start_time = datetime.now()
        topics_data = extract_topics_with_llm(chunks, request.options or {}, request.session_id)
        extraction_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Save topics to database
        saved_topics = []
        
        with db.get_connection() as conn:
            # First, save main topics
            for topic_data in topics_data.get("topics", []):
                # Insert main topic
                cursor = conn.execute("""
                    INSERT INTO course_topics (
                        session_id, topic_title, parent_topic_id, topic_order,
                        description, keywords, estimated_difficulty,
                        prerequisites, related_chunk_ids,
                        extraction_method, extraction_confidence, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request.session_id,
                    topic_data["topic_title"],
                    None,  # parent_topic_id
                    topic_data.get("order", 0),
                    None,  # description
                    json.dumps(topic_data.get("keywords", [])),
                    topic_data.get("difficulty", "intermediate"),
                    json.dumps(topic_data.get("prerequisites", [])),
                    json.dumps(topic_data.get("related_chunks", [])),
                    request.extraction_method,
                    request.options.get("min_confidence", 0.7) if request.options else 0.7,
                    True
                ))
                
                main_topic_id = cursor.lastrowid
                
                # Save subtopics
                for subtopic_data in topic_data.get("subtopics", []):
                    conn.execute("""
                        INSERT INTO course_topics (
                            session_id, topic_title, parent_topic_id, topic_order,
                            keywords, extraction_method, extraction_confidence, is_active
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        request.session_id,
                        subtopic_data["topic_title"],
                        main_topic_id,
                        subtopic_data.get("order", 0),
                        json.dumps(subtopic_data.get("keywords", [])),
                        request.extraction_method,
                        request.options.get("min_confidence", 0.7) if request.options else 0.7,
                        True
                    ))
                
                saved_topics.append({
                    "topic_id": main_topic_id,
                    "topic_title": topic_data["topic_title"],
                    "topic_order": topic_data.get("order", 0),
                    "extraction_confidence": request.options.get("min_confidence", 0.7) if request.options else 0.7
                })
            
            conn.commit()
        
        return {
            "success": True,
            "topics": saved_topics,
            "total_topics": len(saved_topics),
            "extraction_time_ms": int(extraction_time)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        raise HTTPException(status_code=500, detail=f"Topic extraction failed: {str(e)}")


@router.get("/session/{session_id}")
async def get_session_topics(session_id: str):
    """
    Get all topics for a session
    """
    # Check if APRAG is enabled
    if not FeatureFlags.is_aprag_enabled(session_id):
        return {
            "success": False,
            "topics": [],
            "total": 0
        }
    
    db = get_db()
    
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    topic_id, session_id, topic_title, parent_topic_id, topic_order,
                    description, keywords, estimated_difficulty, prerequisites,
                    extraction_confidence, is_active
                FROM course_topics
                WHERE session_id = ? AND is_active = TRUE
                ORDER BY topic_order, topic_id
            """, (session_id,))
            
            topics = []
            for row in cursor.fetchall():
                topic = dict(row)
                # Parse JSON fields
                topic["keywords"] = json.loads(topic["keywords"]) if topic["keywords"] else []
                topic["prerequisites"] = json.loads(topic["prerequisites"]) if topic["prerequisites"] else []
                topics.append(topic)
            
            return {
                "success": True,
                "topics": topics,
                "total": len(topics)
            }
            
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch topics: {str(e)}")


@router.put("/{topic_id}")
async def update_topic(topic_id: int, request: TopicUpdateRequest):
    """
    Update a topic
    """
    # Get session_id from topic first to check APRAG status
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT session_id FROM course_topics WHERE topic_id = ?", (topic_id,))
        topic = cursor.fetchone()
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        
        session_id = dict(topic)["session_id"]
        
        # Check if APRAG is enabled
        if not FeatureFlags.is_aprag_enabled(session_id):
            raise HTTPException(
                status_code=403,
                detail="APRAG module is disabled. Please enable it from admin settings."
            )
    
    try:
        with db.get_connection() as conn:
            # Build update query dynamically
            updates = []
            params = []
            
            if request.topic_title is not None:
                updates.append("topic_title = ?")
                params.append(request.topic_title)
            
            if request.topic_order is not None:
                updates.append("topic_order = ?")
                params.append(request.topic_order)
            
            if request.description is not None:
                updates.append("description = ?")
                params.append(request.description)
            
            if request.keywords is not None:
                updates.append("keywords = ?")
                params.append(json.dumps(request.keywords))
            
            if request.estimated_difficulty is not None:
                updates.append("estimated_difficulty = ?")
                params.append(request.estimated_difficulty)
            
            if request.prerequisites is not None:
                updates.append("prerequisites = ?")
                params.append(json.dumps(request.prerequisites))
            
            if request.is_active is not None:
                updates.append("is_active = ?")
                params.append(request.is_active)
            
            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(topic_id)
            
            conn.execute(f"""
                UPDATE course_topics
                SET {', '.join(updates)}
                WHERE topic_id = ?
            """, params)
            
            conn.commit()
            
            return {"success": True, "message": "Topic updated successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating topic: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update topic: {str(e)}")


@router.post("/classify-question")
async def classify_question(request: QuestionClassificationRequest):
    """
    Classify a question to a topic
    """
    # Check if APRAG is enabled
    if not FeatureFlags.is_aprag_enabled(request.session_id):
        raise HTTPException(
            status_code=403,
            detail="APRAG module is disabled. Please enable it from admin settings."
        )
    
    db = get_db()
    
    try:
        # Get topics for session
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT topic_id, topic_title, keywords
                FROM course_topics
                WHERE session_id = ? AND is_active = TRUE
            """, (request.session_id,))
            
            topics = []
            for row in cursor.fetchall():
                topic = dict(row)
                topic["keywords"] = json.loads(topic["keywords"]) if topic["keywords"] else []
                topics.append(topic)
        
        if not topics:
            raise HTTPException(
                status_code=404,
                detail=f"No topics found for session {request.session_id}. Please extract topics first."
            )
        
        # Classify question using LLM
        classification = classify_question_with_llm(request.question, topics, request.session_id)
        
        # Save mapping if interaction_id is provided
        if request.interaction_id:
            with db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO question_topic_mapping (
                        interaction_id, topic_id, confidence_score,
                        mapping_method, question_complexity, question_type
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    request.interaction_id,
                    classification["topic_id"],
                    classification["confidence_score"],
                    "llm_classification",
                    classification["question_complexity"],
                    classification["question_type"]
                ))
                
                # Update topic progress
                conn.execute("""
                    INSERT OR REPLACE INTO topic_progress (
                        user_id, session_id, topic_id,
                        questions_asked, last_question_timestamp,
                        updated_at
                    ) VALUES (
                        (SELECT user_id FROM student_interactions WHERE interaction_id = ?),
                        ?,
                        ?,
                        COALESCE((SELECT questions_asked FROM topic_progress 
                                 WHERE user_id = (SELECT user_id FROM student_interactions WHERE interaction_id = ?)
                                 AND session_id = ? AND topic_id = ?), 0) + 1,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP
                    )
                """, (
                    request.interaction_id,
                    request.session_id,
                    classification["topic_id"],
                    request.interaction_id,
                    request.session_id,
                    classification["topic_id"]
                ))
                
                conn.commit()
        
        return {
            "success": True,
            "topic_id": classification["topic_id"],
            "topic_title": classification.get("topic_title", ""),
            "confidence_score": classification["confidence_score"],
            "question_complexity": classification["question_complexity"],
            "question_type": classification["question_type"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying question: {e}")
        raise HTTPException(status_code=500, detail=f"Question classification failed: {str(e)}")


@router.get("/progress/{user_id}/{session_id}")
async def get_student_progress(user_id: str, session_id: str):
    """
    Get student progress for all topics in a session
    """
    # Check if APRAG is enabled
    if not FeatureFlags.is_aprag_enabled(session_id):
        return {
            "success": False,
            "progress": [],
            "current_topic": None,
            "next_recommended_topic": None
        }
    
    db = get_db()
    
    try:
        with db.get_connection() as conn:
            # Get all topics with progress
            cursor = conn.execute("""
                SELECT 
                    t.topic_id,
                    t.topic_title,
                    t.topic_order,
                    COALESCE(p.questions_asked, 0) as questions_asked,
                    p.average_understanding,
                    p.mastery_level,
                    p.mastery_score,
                    p.is_ready_for_next,
                    p.readiness_score,
                    p.time_spent_minutes
                FROM course_topics t
                LEFT JOIN topic_progress p ON t.topic_id = p.topic_id 
                    AND p.user_id = ? AND p.session_id = ?
                WHERE t.session_id = ? AND t.is_active = TRUE
                ORDER BY t.topic_order, t.topic_id
            """, (user_id, session_id, session_id))
            
            progress = []
            current_topic = None
            next_recommended = None
            
            for row in cursor.fetchall():
                topic_progress = dict(row)
                
                # Determine current topic (first topic with questions but not mastered)
                if (current_topic is None and 
                    topic_progress["questions_asked"] > 0 and 
                    topic_progress.get("mastery_level") != "mastered"):
                    current_topic = topic_progress
                
                # Find next recommended topic (first topic ready for next)
                if (next_recommended is None and 
                    topic_progress.get("is_ready_for_next")):
                    next_recommended = topic_progress
                
                progress.append(topic_progress)
            
            return {
                "success": True,
                "progress": progress,
                "current_topic": current_topic,
                "next_recommended_topic": next_recommended
            }
            
    except Exception as e:
        logger.error(f"Error fetching progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch progress: {str(e)}")


@router.post("/{topic_id}/generate-questions")
async def generate_questions_for_topic(topic_id: int, request: QuestionGenerationRequest):
    """
    Generate questions for a specific topic using LLM
    """
    db = get_db()
    
    try:
        # Get topic information and session_id
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT session_id, topic_title, description, keywords, estimated_difficulty
                FROM course_topics
                WHERE topic_id = ? AND is_active = TRUE
            """, (topic_id,))
            
            topic = cursor.fetchone()
            if not topic:
                raise HTTPException(status_code=404, detail="Topic not found")
            
            topic_data = dict(topic)
            session_id = topic_data["session_id"]
            
            # Check if APRAG is enabled
            if not FeatureFlags.is_aprag_enabled(session_id):
                raise HTTPException(
                    status_code=403,
                    detail="APRAG module is disabled. Please enable it from admin settings."
                )
        
        # Get chunks related to this topic
        chunks = fetch_chunks_for_session(session_id)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No content chunks found for session {session_id}"
            )
        
        # Filter chunks related to this topic (if we have related_chunk_ids)
        # For now, we'll use all chunks as we don't have chunk-topic mapping fully implemented
        relevant_chunks = chunks[:10]  # Limit to first 10 chunks to stay within token limits
        
        # Prepare content for LLM
        chunks_text = "\n\n---\n\n".join([
            f"Bölüm {i+1}:\n{chunk.get('chunk_text', chunk.get('content', chunk.get('text', '')))}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # Determine difficulty level
        difficulty = request.difficulty_level or topic_data.get("estimated_difficulty", "intermediate")
        
        # Create prompt for question generation
        prompt = f"""Sen bir eğitim uzmanısın. Aşağıdaki ders materyali bağlamında "{topic_data['topic_title']}" konusu için sorular üret.

KONU BAŞLIĞI: {topic_data['topic_title']}
ZORLUK SEVİYESİ: {difficulty}
ANAHTAR KELİMELER: {', '.join(json.loads(topic_data['keywords']) if topic_data['keywords'] else [])}

DERS MATERYALİ:
{chunks_text[:8000]}

LÜTFEN ŞUNLARI YAP:
1. Bu konu ve materyal bağlamında {request.count} adet soru üret
2. Soruları farklı türlerde dağıt: kavramsal sorular, uygulama soruları, analiz soruları
3. Zorluk seviyesi "{difficulty}" seviyesine uygun olsun
4. Sorular doğrudan materyal içeriğine dayalı olsun
5. Açık uçlu ve düşünmeyi teşvik eden sorular olsun

ÇIKTI FORMATI (JSON):
{{
  "questions": [
    "İlk soru burada...",
    "İkinci soru burada...",
    "Üçüncü soru burada..."
  ]
}}

Sadece JSON çıktısı ver, başka açıklama yapma."""

        # Get session-specific model configuration
        model_to_use = get_session_model(session_id) or "llama-3.1-8b-instant"
        
        # Call model inference service
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model_to_use,
                "max_tokens": 2048,
                "temperature": 0.7
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM service error: {response.text}")
        
        result = response.json()
        llm_output = result.get("response", "")
        
        # Parse JSON from LLM output
        import re
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            questions_data = json.loads(json_match.group())
        else:
            # Try to parse entire output as JSON
            questions_data = json.loads(llm_output)
        
        return {
            "success": True,
            "topic_id": topic_id,
            "topic_title": topic_data["topic_title"],
            "questions": questions_data.get("questions", []),
            "count": len(questions_data.get("questions", [])),
            "difficulty_level": difficulty
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON output: {e}")
        logger.error(f"LLM output was: {llm_output[:1000]}")
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON for question generation")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in question generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to model service: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

