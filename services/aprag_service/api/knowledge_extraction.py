"""
Knowledge Extraction Service
Extracts structured knowledge from document chunks for KB-Enhanced RAG
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
import re
import hashlib
from datetime import datetime, timedelta
import requests
import os
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()

# Import database manager
from database.database import DatabaseManager

# Environment variables
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", "http://model-inference-service:8002")
DOCUMENT_PROCESSING_URL = os.getenv("DOCUMENT_PROCESSING_URL", "http://document-processing-service:8002")

# Database manager singleton
def get_db() -> DatabaseManager:
    """Get database manager dependency"""
    db_path = os.getenv("APRAG_DB_PATH", os.getenv("DATABASE_PATH", "/app/data/rag_assistant.db"))
    return DatabaseManager(db_path)

# In-memory job store for batch KB extraction
BATCH_KB_JOBS: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Helper Functions for QA Embeddings
# ============================================================================

async def calculate_and_store_qa_embedding(
    qa_id: int,
    question: str,
    embedding_model: Optional[str] = None,
    db: Optional[DatabaseManager] = None
) -> bool:
    """
    Calculate embedding for a QA pair question and store it in database
    
    Args:
        qa_id: QA pair ID
        question: Question text
        embedding_model: Embedding model to use (defaults to DEFAULT_EMBEDDING_MODEL)
        db: Database manager instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not db:
            db = get_db()
        
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        
        logger.info(f"📦 Calculating embedding for QA {qa_id} using model {embedding_model}...")
        
        # Calculate embedding
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/embeddings",
            json={
                "texts": [question],
                "model": embedding_model
            },
            timeout=15
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Embedding calculation failed for QA {qa_id}: {response.status_code}")
            return False
        
        data = response.json()
        embeddings = data.get("embeddings", [])
        
        if not embeddings or len(embeddings) == 0:
            logger.error(f"❌ No embedding returned for QA {qa_id}")
            return False
        
        embedding = embeddings[0]
        embedding_dim = len(embedding)
        
        # Store in database
        with db.get_connection() as conn:
            # First ensure question_embedding column exists (apply migration if needed)
            cursor_check = conn.execute("PRAGMA table_info(topic_qa_pairs)")
            columns = {row[1]: row[2] for row in cursor_check.fetchall()}
            
            if 'question_embedding' not in columns:
                logger.info("⚠️ question_embedding column not found. Applying migration...")
                try:
                    # Apply migration directly
                    conn.execute("""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN question_embedding TEXT
                    """)
                    conn.execute("""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN embedding_model VARCHAR(100)
                    """)
                    conn.execute("""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN embedding_dim INTEGER
                    """)
                    conn.execute("""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN embedding_updated_at TIMESTAMP
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_qa_pairs_topic_active 
                        ON topic_qa_pairs(topic_id, is_active)
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_qa_pairs_embedding_model 
                        ON topic_qa_pairs(embedding_model)
                    """)
                    conn.commit()
                    logger.info("✅ question_embedding column added successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to add question_embedding column: {e}")
                    conn.rollback()
                    return False
            
            # Now update with the column
            conn.execute("""
                UPDATE topic_qa_pairs
                SET question_embedding = ?,
                    embedding_model = ?,
                    embedding_dim = ?,
                    embedding_updated_at = CURRENT_TIMESTAMP
                WHERE qa_id = ?
            """, (
                json.dumps(embedding, ensure_ascii=False),
                embedding_model,
                embedding_dim,
                qa_id
            ))
            conn.commit()
        
        logger.info(f"✅ Embedding stored for QA {qa_id} (dim={embedding_dim}, model={embedding_model})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error calculating/storing embedding for QA {qa_id}: {e}")
        return False


async def calculate_and_store_qa_embeddings_batch(
    qa_pairs: List[Dict[str, Any]],
    embedding_model: Optional[str] = None,
    db: Optional[DatabaseManager] = None
) -> int:
    """
    Calculate embeddings for multiple QA pairs in batch and store them
    
    Args:
        qa_pairs: List of QA pairs with qa_id and question
        embedding_model: Embedding model to use
        db: Database manager instance
        
    Returns:
        Number of successfully processed embeddings
    """
    try:
        if not db:
            db = get_db()
        
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        
        if not qa_pairs:
            return 0
        
        logger.info(f"📦 Calculating embeddings for {len(qa_pairs)} QA pairs in batch...")
        
        # Prepare questions for batch embedding
        questions = [qa.get("question", "") for qa in qa_pairs if qa.get("question")]
        
        if not questions:
            logger.warning("⚠️ No questions found in QA pairs for embedding")
            return 0
        
        # Calculate embeddings in batch
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/embeddings",
            json={
                "texts": questions,
                "model": embedding_model
            },
            timeout=30  # Increased timeout for batch
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Batch embedding calculation failed: {response.status_code}")
            # Fallback to individual calculations
            success_count = 0
            for qa in qa_pairs:
                if qa.get("qa_id") and qa.get("question"):
                    if await calculate_and_store_qa_embedding(
                        qa["qa_id"], qa["question"], embedding_model, db
                    ):
                        success_count += 1
            return success_count
        
        data = response.json()
        embeddings = data.get("embeddings", [])
        
        if len(embeddings) != len(questions):
            logger.warning(f"⚠️ Expected {len(questions)} embeddings, got {len(embeddings)}")
            # Fallback to individual calculations
            success_count = 0
            for qa in qa_pairs:
                if qa.get("qa_id") and qa.get("question"):
                    if await calculate_and_store_qa_embedding(
                        qa["qa_id"], qa["question"], embedding_model, db
                    ):
                        success_count += 1
            return success_count
        
        # Store all embeddings
        success_count = 0
        embedding_model_name = embedding_model
        with db.get_connection() as conn:
            # First ensure question_embedding column exists (apply migration if needed)
            cursor_check = conn.execute("PRAGMA table_info(topic_qa_pairs)")
            columns = {row[1]: row[2] for row in cursor_check.fetchall()}
            
            if 'question_embedding' not in columns:
                logger.info("⚠️ question_embedding column not found. Applying migration...")
                try:
                    # Apply migration directly
                    conn.execute("""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN question_embedding TEXT
                    """)
                    conn.execute("""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN embedding_model VARCHAR(100)
                    """)
                    conn.execute("""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN embedding_dim INTEGER
                    """)
                    conn.execute("""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN embedding_updated_at TIMESTAMP
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_qa_pairs_topic_active 
                        ON topic_qa_pairs(topic_id, is_active)
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_qa_pairs_embedding_model 
                        ON topic_qa_pairs(embedding_model)
                    """)
                    conn.commit()
                    logger.info("✅ question_embedding column added successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to add question_embedding column: {e}")
                    conn.rollback()
                    # Fallback to individual calculations
                    success_count = 0
                    for qa in qa_pairs:
                        if qa.get("qa_id") and qa.get("question"):
                            if await calculate_and_store_qa_embedding(
                                qa["qa_id"], qa["question"], embedding_model, db
                            ):
                                success_count += 1
                    return success_count
            
            # Now update with the column
            for i, qa in enumerate(qa_pairs):
                if not qa.get("qa_id") or not qa.get("question"):
                    continue
                
                embedding = embeddings[i]
                embedding_dim = len(embedding)
                
                try:
                    conn.execute("""
                        UPDATE topic_qa_pairs
                        SET question_embedding = ?,
                            embedding_model = ?,
                            embedding_dim = ?,
                            embedding_updated_at = CURRENT_TIMESTAMP
                        WHERE qa_id = ?
                    """, (
                        json.dumps(embedding, ensure_ascii=False),
                        embedding_model_name,
                        embedding_dim,
                        qa["qa_id"]
                    ))
                    success_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to store embedding for QA {qa.get('qa_id')}: {e}")
            
            conn.commit()
        
        logger.info(f"✅ Batch embedding completed: {success_count}/{len(qa_pairs)} QA pairs processed")
        return success_count
        
    except Exception as e:
        logger.error(f"❌ Error in batch embedding calculation: {e}")
        # Fallback to individual calculations
        success_count = 0
        for qa in qa_pairs:
            if qa.get("qa_id") and qa.get("question"):
                if await calculate_and_store_qa_embedding(
                    qa["qa_id"], qa["question"], embedding_model, db
                ):
                    success_count += 1
        return success_count


# ============================================================================
# Request/Response Models
# ============================================================================

class KnowledgeExtractionRequest(BaseModel):
    """Request model for knowledge extraction"""
    topic_id: int
    force_refresh: bool = False  # Force re-extraction even if exists
    system_prompt: Optional[str] = None


class BatchKnowledgeExtractionRequest(BaseModel):
    """Request model for batch knowledge extraction"""
    session_id: str
    force_refresh: bool = False
    system_prompt: Optional[str] = None
    extraction_config: Optional[Dict[str, Any]] = {
        "generate_qa_pairs": True,
        "qa_pairs_per_topic": 15,
        "extract_examples": True,
        "extract_misconceptions": True
    }


class QAGenerationRequest(BaseModel):
    """Request model for QA pair generation"""
    topic_id: int
    count: int = 15
    difficulty_distribution: Optional[Dict[str, int]] = {
        "beginner": 5,
        "intermediate": 7,
        "advanced": 3
    }


class KnowledgeValidationRequest(BaseModel):
    """Request model for manual knowledge validation"""
    knowledge_id: int
    validator_user_id: str
    is_approved: bool
    validation_notes: Optional[str] = None
    corrections: Optional[Dict[str, Any]] = None


# ============================================================================
# Helper Functions
# ============================================================================

# Bloom Taxonomy seviye çevirileri
BLOOM_LEVEL_TRANSLATIONS = {
    "remember": "Hatırlama",
    "understand": "Anlama",
    "apply": "Uygulama",
    "analyze": "Analiz",
    "evaluate": "Değerlendirme",
    "create": "Yaratma"
}

def translate_bloom_level(level: str) -> str:
    """Bloom taxonomy seviyesini İngilizce'den Türkçe'ye çevir"""
    return BLOOM_LEVEL_TRANSLATIONS.get(level.lower(), level)

def translate_learning_objectives(objectives: List[Dict]) -> List[Dict]:
    """Öğrenme hedeflerindeki Bloom seviyelerini Türkçeleştir"""
    if not objectives:
        return objectives
    
    translated = []
    for obj in objectives:
        translated_obj = obj.copy()
        if "level" in translated_obj:
            original_level = translated_obj["level"]
            # İngilizce ise Türkçe'ye çevir
            if original_level.lower() in BLOOM_LEVEL_TRANSLATIONS:
                translated_obj["level"] = translate_bloom_level(original_level)
            # level_tr her zaman ekle (geriye dönük uyumluluk için)
            translated_obj["level_tr"] = translate_bloom_level(original_level)
        translated.append(translated_obj)
    return translated

def translate_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
    """QA çiftlerindeki Bloom seviyelerini Türkçeleştir"""
    if not qa_pairs:
        return qa_pairs
    
    translated = []
    for qa in qa_pairs:
        translated_qa = dict(qa)
        # bloom_taxonomy_level veya bloom_level alanını bul ve Türkçeleştir
        bloom_key = None
        if "bloom_taxonomy_level" in translated_qa:
            bloom_key = "bloom_taxonomy_level"
        elif "bloom_level" in translated_qa:
            bloom_key = "bloom_level"
        
        if bloom_key:
            original_level = translated_qa[bloom_key]
            # İngilizce ise Türkçe'ye çevir
            if original_level and original_level.lower() in BLOOM_LEVEL_TRANSLATIONS:
                translated_qa[bloom_key] = translate_bloom_level(original_level)
            # bloom_level_tr her zaman ekle (geriye dönük uyumluluk için)
            translated_qa["bloom_level_tr"] = translate_bloom_level(original_level) if original_level else None
        
        translated.append(translated_qa)
    return translated



def get_session_model(session_id: str) -> str:
    """
    Get the model configured for a specific session from API Gateway
    """
    if not session_id:
        return "llama3:8b"
        
    try:
        api_gateway_url = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
        response = requests.get(f"{api_gateway_url}/sessions/{session_id}", timeout=10)
        
        if response.status_code == 200:
            session_data = response.json()
            rag_settings = session_data.get("rag_settings", {})
            
            if rag_settings and rag_settings.get("model"):
                return rag_settings["model"]
        
        return "llama3:8b"
    except Exception as e:
        logger.warning(f"Could not get session model: {e}")
        return "llama3:8b"


def fetch_chunks_for_session(session_id: str) -> List[Dict[str, Any]]:
    """Fetch all chunks for a session from document processing service"""
    try:
        response = requests.get(
            f"{DOCUMENT_PROCESSING_URL}/sessions/{session_id}/chunks",
            timeout=30
        )
        
        if response.status_code == 200:
            chunks = response.json().get("chunks", [])
            
            # Normalize chunk IDs - ensure every chunk has a valid ID
            for i, chunk in enumerate(chunks):
                # Try multiple possible locations for chunk_id
                chunk_id = (
                    chunk.get("chunk_id") or  # Root level (main.py endpoint)
                    chunk.get("id") or 
                    chunk.get("chunkId") or
                    (chunk.get("chunk_metadata", {}) or {}).get("chunk_id") or  # Nested in metadata (api/routes/sessions.py endpoint)
                    None
                )
                
                if chunk_id is None:
                    # If still no ID, use a stable hash-based ID
                    doc_name = chunk.get("document_name", "unknown")
                    chunk_idx = chunk.get("chunk_index", i + 1)
                    chunk_id = hash(f"{session_id}_{doc_name}_{chunk_idx}") % 1000000
                    if chunk_id < 0:
                        chunk_id = abs(chunk_id)
                    logger.warning(f"⚠️ [FETCH CHUNKS] Chunk {i} has no ID, generated stable ID: {chunk_id}")
                
                # Ensure chunk_id is set in the main dict (keep original type - string UUID or int)
                chunk["chunk_id"] = chunk_id
                
            logger.info(f"✅ [FETCH CHUNKS] Fetched {len(chunks)} chunks, sample IDs (first 5): {[c.get('chunk_id') for c in chunks[:5]]}")
            return chunks
        else:
            logger.warning(f"Could not fetch chunks: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching chunks for session {session_id}: {e}")
        return []


def get_topic_info(topic_id: int, db: DatabaseManager) -> Optional[Dict]:
    """Get topic information from database"""
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT topic_id, session_id, topic_title, description, 
                       keywords, estimated_difficulty, related_chunk_ids
                FROM course_topics
                WHERE topic_id = ? AND is_active = TRUE
            """, (topic_id,))
            
            row = cursor.fetchone()
            if row:
                topic = dict(row)
                # Parse JSON fields
                topic["keywords"] = json.loads(topic["keywords"]) if topic["keywords"] else []
                topic["related_chunk_ids"] = json.loads(topic["related_chunk_ids"]) if topic["related_chunk_ids"] else []
                return topic
            return None
    except Exception as e:
        logger.error(f"Error fetching topic info: {e}")
        return None


def filter_chunks_by_topic(chunks: List[Dict], topic_keywords: List[str], related_chunk_ids: List[int] = None) -> List[Dict]:
    """Filter chunks relevant to a topic with improved matching"""
    logger.info(f"🔍 [CHUNK FILTER] Starting chunk filtering for topic")
    logger.info(f"📊 [CHUNK FILTER] Total chunks: {len(chunks)}")
    logger.info(f"🔑 [CHUNK FILTER] Keywords: {topic_keywords}")
    logger.info(f"🔗 [CHUNK FILTER] Related chunk IDs: {related_chunk_ids}")
    
    if related_chunk_ids and len(related_chunk_ids) > 0:
        # Use explicitly marked chunks if available
        # Try multiple possible ID field names
        explicit_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or chunk.get("chunkId") or (chunk.get("chunk_metadata", {}) or {}).get("chunk_id")
            # Normalize chunk_id to string for comparison (UUIDs are strings, but might be stored as different types)
            chunk_id_str = str(chunk_id) if chunk_id is not None else None
            # Check if chunk_id matches any in related_chunk_ids (handle both string and int comparisons)
            for ref_id in related_chunk_ids:
                ref_id_str = str(ref_id) if ref_id is not None else None
                if chunk_id_str and ref_id_str and chunk_id_str == ref_id_str:
                    explicit_chunks.append(chunk)
                    break
        
        logger.info(f"✅ [CHUNK FILTER] Found {len(explicit_chunks)} explicitly related chunks")
        if len(explicit_chunks) > 0:
            return explicit_chunks
        else:
            logger.warning(f"⚠️ [CHUNK FILTER] No chunks matched related_chunk_ids {related_chunk_ids}. Sample chunk IDs: {[c.get('chunk_id') or c.get('id') or c.get('chunkId') for c in chunks[:5]]}")
            # Fall through to keyword matching
    
    if not topic_keywords:
        logger.warning(f"⚠️ [CHUNK FILTER] No keywords provided, cannot filter chunks properly")
        return []
    
    # Enhanced keyword-based filtering
    relevant_chunks = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("chunk_text", chunk.get("content", "")).lower()
        
        # Count exact keyword matches
        exact_matches = sum(1 for kw in topic_keywords if kw.lower() in content)
        
        # Count partial/fuzzy matches (for Turkish stemming issues)
        partial_matches = 0
        for kw in topic_keywords:
            kw_lower = kw.lower()
            # Remove Turkish suffixes for better matching
            kw_stem = kw_lower.replace("ler", "").replace("lar", "").replace("den", "").replace("dan", "")
            if len(kw_stem) > 3:  # Only do this for longer words
                if kw_stem in content:
                    partial_matches += 0.5
        
        total_score = exact_matches + partial_matches
        
        # Lower threshold: at least 1 exact match OR 2+ partial matches
        if exact_matches >= 1 or total_score >= 2.0:
            chunk["relevance_score"] = total_score / len(topic_keywords)
            relevant_chunks.append(chunk)
            logger.debug(f"📄 [CHUNK FILTER] Chunk {i} matched (score: {total_score:.1f}): {content[:100]}...")
    
    # Sort by relevance
    relevant_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    logger.info(f"🎯 [CHUNK FILTER] Found {len(relevant_chunks)} relevant chunks from keyword matching")
    
    if len(relevant_chunks) == 0:
        logger.error(f"❌ [CHUNK FILTER] No relevant chunks found! Topic-chunk linkage broken!")
        logger.error(f"🔧 [CHUNK FILTER] SOLUTION: Need to regenerate topics with proper chunk relationships")
        # Return empty list instead of all chunks - force proper topic generation
        return []
    
    # Return top 10 most relevant
    return relevant_chunks[:10]


# ============================================================================
# LLM Knowledge Extraction Functions
# ============================================================================

async def extract_topic_summary(topic_title: str, chunks_text: str, model: str = "llama-3.1-8b-instant", system_prompt: Optional[str] = None) -> str:
    """Extract comprehensive topic summary using LLM (always in Turkish)"""
    
    # Use custom system prompt if provided, otherwise use default
    if system_prompt:
        base_instruction = system_prompt.strip()
        logger.info(f"📝 [KB EXTRACTION] Using custom system prompt for summary: {base_instruction[:100]}...")
        prompt = f"""{base_instruction}

"{topic_title}" konusu için kapsamlı ve TAMAMEN TÜRKÇE bir özet oluştur.

KONU: {topic_title}

DERS MATERYALİ:
{chunks_text[:12000]}

ÖZET KURALLARI:
1. 200-300 kelime arası olmalı
2. Konunun ana fikrini net ve anlaşılır şekilde açıkla
3. Önemli terimleri ve kavramları tanımla
4. Öğrenciler için erişilebilir ve sade bir TÜRKÇE kullan
5. Mantıksal akış: tanım → özellikler → önem → uygulamalar
6. Türkçe dilbilgisi ve yazım kurallarına dikkat et
7. Kesinlikle İngilizce cümle kurma, cevap TAMAMEN TÜRKÇE olsun.

YANITINI SADECE TÜRKÇE OLARAK YAZ.

ÖZET:"""

    try:
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model,
                "max_tokens": 600,
                "temperature": 0.3
            },
            timeout=90
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get("response", "").strip()
            # Cleanup: bazı modeller giriş cümlesi ekliyor, bunları kaldır
            # Örnek: "Here is a comprehensive and complete summary in Turkish:"
            summary = re.sub(
                r"^Here\s+is\s+a\s+comprehensive.*?:\s*",
                "",
                summary,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return summary
        else:
            logger.error(f"LLM service error: {response.status_code}")
            return ""
            
    except Exception as e:
        logger.error(f"Error in summary extraction: {e}")
        return ""


async def extract_key_concepts(topic_title: str, chunks_text: str, model: str = "llama3:8b", system_prompt: Optional[str] = None) -> List[Dict]:
    """Extract key concepts with definitions"""
    
    # Use custom system prompt if provided
    if system_prompt:
        base_instruction = system_prompt.strip()
        logger.info(f"💡 [KB EXTRACTION] Using custom system prompt for concepts: {base_instruction[:100]}...")
        prompt = f"""{base_instruction}

"{topic_title}" konusundaki temel kavramları listele ve her birini kısaca tanımla.

MATERYAL:
{chunks_text[:10000]}

KURALLAR:
1. 5-10 temel kavram belirle
2. Her kavram için net tanım yaz (1-2 cümle)
3. Kavramların önem seviyesini belirle (high, medium, low)
4. İlişkili kavramları birlikte grupla
5. TÜM terimler ve tanımlar TÜRKÇE olsun, İngilizce kelime kullanma.

ÇIKTI FORMATI (JSON) - TÜM ALANLAR TÜRKÇE OLMALIDIR:
{{
  "concepts": [
    {{
      "term": "Hücre Zarı",
      "definition": "Hücreyi dış ortamdan ayıran ve seçici geçirgen yapı.",
      "importance": "high",
      "category": "yapı"
    }},
    {{
      "term": "Osmoz",
      "definition": "Suyun yarı geçirgen zardan konsantrasyonu düşük bölgeden yüksek bölgeye geçişi.",
      "importance": "high",
      "category": "süreç"
    }}
  ]
}}

Sadece JSON çıktısı ver, başka açıklama yapma."""

    try:
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model,
                "max_tokens": 1200,
                "temperature": 0.3
            },
            timeout=90
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_output = result.get("response", "")
            
            # Parse JSON
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("concepts", [])
        
        return []
        
    except Exception as e:
        logger.error(f"Error in concept extraction: {e}")
        return []


async def extract_learning_objectives(topic_title: str, chunks_text: str, model: str = "llama3:8b", system_prompt: Optional[str] = None) -> List[Dict]:
    """Extract Bloom's Taxonomy-aligned learning objectives"""
    
    # Use custom system prompt if provided
    if system_prompt:
        base_instruction = system_prompt.strip()
        logger.info(f"🎯 [KB EXTRACTION] Using custom system prompt for objectives: {base_instruction[:100]}...")
        prompt = f"""{base_instruction}

"{topic_title}" konusu için Bloom Taksonomisine uygun öğrenme hedefleri oluştur.

BLOOM TAKSONOMİSİ SEVİYELERİ (TÜRKÇE İSİMLER KULLAN):
1. Hatırlama: Temel bilgileri hatırlar
2. Anlama: Kavramları açıklar
3. Uygulama: Bilgiyi yeni durumlarda kullanır
4. Analiz: İlişkileri inceler ve karşılaştırır
5. Değerlendirme: Eleştirel değerlendirme yapar
6. Yaratma: Yeni fikirler ve çözümler üretir

MATERYAL:
{chunks_text[:10000]}

Her seviyeden en az 1 hedef belirle. Hedefler ölçülebilir ve net olmalı.

ÖNEMLİ: "level" alanında TÜRKÇE seviye isimlerini kullan: "Hatırlama", "Anlama", "Uygulama", "Analiz", "Değerlendirme", "Yaratma"

ÇIKTI FORMATI (JSON) - TÜM METİNLER TÜRKÇE OLMALIDIR:
{{
  "objectives": [
    {{
      "level": "Hatırlama",
      "objective": "Öğrenci hücre zarının temel bileşenlerini sıralayabilmeli",
      "assessment_method": "çoktan seçmeli test"
    }},
    {{
      "level": "Anlama",
      "objective": "Öğrenci hücre zarının seçici geçirgenlik özelliğini açıklayabilmeli",
      "assessment_method": "açık uçlu soru"
    }},
    {{
      "level": "Uygulama",
      "objective": "Öğrenci osmoz olayını günlük hayat örnekleriyle ilişkilendirebilmeli",
      "assessment_method": "senaryo analizi"
    }}
  ]
}}

Sadece JSON çıktısı ver."""

    try:
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model,
                "max_tokens": 1200,
                "temperature": 0.3
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_output = result.get("response", "")
            
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                objectives = data.get("objectives", [])
                
                # İngilizce seviyeleri Türkçe'ye çevir (geriye dönük uyumluluk için)
                for obj in objectives:
                    if "level" in obj:
                        level = obj["level"]
                        # Eğer İngilizce ise Türkçe'ye çevir
                        if level.lower() in BLOOM_LEVEL_TRANSLATIONS:
                            obj["level"] = translate_bloom_level(level)
                
                return objectives
        
        return []
        
    except Exception as e:
        logger.error(f"Error in objectives extraction: {e}")
        return []


async def generate_qa_pairs(
    topic_title: str, 
    chunks_text: str, 
    count: int = 15,
    difficulty_dist: Dict[str, int] = None,
    model: str = "llama3:8b"
) -> List[Dict]:
    """Generate question-answer pairs for a topic"""
    
    if difficulty_dist is None:
        difficulty_dist = {"beginner": 5, "intermediate": 7, "advanced": 3}
    
    # Smart truncation for Groq models
    material_text = chunks_text[:8000] if ("groq" in model.lower() or "instant" in model.lower()) else chunks_text[:15000]
    
    prompt = f"""Sen bir TÜRKÇE eğitim uzmanısın ve "{topic_title}" konusu için TAMAMEN TÜRKÇE sorular üretiyorsun.

ÖNEMLİ: BU TALİMAT MUTLAK ZORUNLUDUR - HER ŞEY TÜRKÇE OLMALIDIR!

MATERYAL:
{material_text}

SORU DAĞILIMI:
- Temel (beginner): {difficulty_dist.get('beginner', 5)} soru - Tanım, hatırlama, basit faktlar
- Orta (intermediate): {difficulty_dist.get('intermediate', 7)} soru - Kavramsal anlama, ilişkilendirme
- İleri (advanced): {difficulty_dist.get('advanced', 3)} soru - Analiz, sentez, problem çözme

SORU TÜRLERİ:
- factual: Doğrudan bilgi soruları
- conceptual: Kavram anlama soruları
- application: Uygulama soruları
- analysis: Analiz ve karşılaştırma soruları

KRİTİK DİL KURALLARI:
1. Her soru MUTLAKA TÜRKÇE olmalıdır - tek bir İngilizce kelime bile kabul edilmez
2. Her cevap MUTLAKA TÜRKÇE olmalıdır - İngilizce terim kullanma
3. Açıklamalar MUTLAKA TÜRKÇE olmalıdır
4. Teknik terimler Türkçe karşılıklarıyla yazılmalıdır
5. Bloom seviyeleri Türkçe olmalıdır: "Hatırlama", "Anlama", "Uygulama", "Analiz", "Değerlendirme", "Yaratma"
6. KESINLIKLE İngilizce cümle kurma - sadece Türkçe üret

ÇIKTI FORMATI (JSON) - TÜM METİNLER TÜRKÇE OLMALIDIR:
{{
  "qa_pairs": [
    {{
      "question": "Hücre zarı hangi temel yapılardan oluşur?",
      "answer": "Hücre zarı fosfolipid çift katman, proteinler ve karbonhidratlardan oluşur.",
      "explanation": "Fosfolipidler zarın temel yapısını oluştururken, proteinler taşıma ve sinyal işlevlerini üstlenir. Karbonhidratlar ise hücre tanıma süreçlerinde rol oynar.",
      "difficulty": "beginner",
      "question_type": "factual",
      "bloom_level": "Hatırlama",
      "related_concepts": ["fosfolipid", "membran proteini", "glikoprotein"]
    }},
    {{
      "question": "Osmoz olayı ile difüzyon arasındaki temel fark nedir?",
      "answer": "Osmoz sadece su moleküllerinin yarı geçirgen zardan geçişidir, difüzyon ise herhangi bir maddenin yüksek konsantrasyondan düşük konsantrasyona geçişidir.",
      "explanation": "Her iki olayda da moleküller konsantrasyon gradyanını takip eder ancak osmoz özellikle su hareketi için kullanılır ve yarı geçirgen zar gerektirir. Difüzyon daha genel bir kavramdır.",
      "difficulty": "intermediate",
      "question_type": "conceptual",
      "bloom_level": "Anlama",
      "related_concepts": ["osmoz", "difüzyon", "konsantrasyon gradyanı"]
    }}
  ]
}}

Toplam {count} soru-cevap çifti oluştur. 

ÖNEMLİ JSON KURALLARI:
1. Her alan arasında virgül (,) olmalı
2. Son alanda virgül OLMAMALI
3. Her obje arasında virgül olmalı
4. Sadece geçerli JSON formatı kullan
5. Tırnak işaretlerini doğru kapat
6. Sadece JSON çıktısı ver, başka metin ekleme

MUTLAK TÜRKÇE FORMAT ÖRNEĞİ:
{{
  "qa_pairs": [
    {{"question": "Türkçe soru...", "answer": "Türkçe cevap...", "difficulty": "beginner", "bloom_level": "Hatırlama"}},
    {{"question": "Başka Türkçe soru...", "answer": "Başka Türkçe cevap...", "difficulty": "intermediate", "bloom_level": "Anlama"}}
  ]
}}

SON UYARI: Hiçbir durumda İngilizce kelime kullanma! Sadece TAMAMEN TÜRKÇE JSON çıktısı ver.
CEVABINI TÜRKÇE KONTROL ET - İngilizce kelime varsa yeniden yaz!

UNUTMA: Bu bir TÜRKÇE eğitim materyali - her kelimen Türkçe olmalı!"""

    try:
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model,
                "max_tokens": 3000,
                "temperature": 0.5,  # Slightly higher for creativity
            },
            timeout=150,
        )

        if response.status_code != 200:
            logger.error(
                f"LLM service error during QA generation: {response.status_code} - {response.text}"
            )
            return []

        result = response.json()
        llm_output = result.get("response", "")

        if not llm_output:
            logger.warning("QA generation returned empty response")
            return []

        # Try to extract JSON block
        json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if not json_match:
            logger.error(
                "QA generation: could not find JSON object in LLM output. "
                "First 300 chars: %s",
                llm_output[:300],
            )
            return []

        json_str = json_match.group()

        # First parse attempt
        try:
            data = json.loads(json_str)
            qa_pairs = data.get("qa_pairs", [])
            
            # İngilizce bloom_level'ları Türkçe'ye çevir (geriye dönük uyumluluk için)
            for qa in qa_pairs:
                if "bloom_level" in qa:
                    level = qa["bloom_level"]
                    # Eğer İngilizce ise Türkçe'ye çevir
                    if level.lower() in BLOOM_LEVEL_TRANSLATIONS:
                        qa["bloom_level"] = translate_bloom_level(level)
            
            return qa_pairs
        except json.JSONDecodeError as e:
            # Second attempt: remove trailing commas before } or ]
            try:
                cleaned = re.sub(r",(\s*[}\]])", r"\1", json_str)  # Fixed: removed extra backslash
                data = json.loads(cleaned)
                logger.warning(
                    "QA generation JSON required cleanup but was parsed successfully: %s",
                    str(e),
                )
                qa_pairs = data.get("qa_pairs", [])
                # İngilizce bloom_level'ları Türkçe'ye çevir
                for qa in qa_pairs:
                    if "bloom_level" in qa and qa["bloom_level"].lower() in BLOOM_LEVEL_TRANSLATIONS:
                        qa["bloom_level"] = translate_bloom_level(qa["bloom_level"])
                return qa_pairs
            except Exception as e2:
                # Third attempt: more aggressive cleanup
                try:
                    # Remove trailing commas, fix incomplete fields
                    cleaned = re.sub(r",(\s*[}\]])", r"\1", json_str)
                    cleaned = re.sub(r':\s*"[^"]*$', ': ""', cleaned, flags=re.MULTILINE)  # Fix incomplete strings
                    cleaned = re.sub(r':\s*\[[^\]]*$', ': []', cleaned, flags=re.MULTILINE)  # Fix incomplete arrays
                    data = json.loads(cleaned)
                    logger.warning("QA generation JSON required aggressive cleanup but was parsed successfully")
                    qa_pairs = data.get("qa_pairs", [])
                    # İngilizce bloom_level'ları Türkçe'ye çevir
                    for qa in qa_pairs:
                        if "bloom_level" in qa and qa["bloom_level"].lower() in BLOOM_LEVEL_TRANSLATIONS:
                            qa["bloom_level"] = translate_bloom_level(qa["bloom_level"])
                    return qa_pairs
                except Exception as e3:
                    # Fourth attempt: ULTRA-AGGRESSIVE JSON repair (similar to topics.py)
                    try:
                        logger.warning("QA generation JSON: Attempting ultra-aggressive repair...")
                        
                        # Extract everything between first { and last }
                        repair_text = json_str.strip()
                        first_brace = repair_text.find('{')
                        last_brace = repair_text.rfind('}')
                        
                        if first_brace >= 0 and last_brace > first_brace:
                            repair_text = repair_text[first_brace:last_brace + 1]
                            
                            # 1. Fix missing commas between fields (more comprehensive)
                            # Pattern: "key": "value" "key2" -> "key": "value", "key2"
                            repair_text = re.sub(r'"\s*"\s*"', '", "', repair_text)  # Fix missing comma between strings
                            repair_text = re.sub(r'}\s*{', '},{', repair_text)  # Fix missing comma between objects
                            repair_text = re.sub(r']\s*{', '],{', repair_text)  # Fix missing comma between array and object
                            repair_text = re.sub(r'}\s*"', '},"', repair_text)  # Fix missing comma between object and string
                            repair_text = re.sub(r']\s*"', '],"', repair_text)  # Fix missing comma between array and string
                            repair_text = re.sub(r'([0-9])\s*"', r'\1,"', repair_text)  # Fix missing comma between number and string
                            repair_text = re.sub(r'"\s*([0-9])', r'", \1', repair_text)  # Fix missing comma between string and number
                            
                            # Fix missing commas after closing quotes before new keys (most common issue)
                            # Pattern: "value" "key" -> "value", "key"
                            repair_text = re.sub(r'"\s+"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', r'", "\1":', repair_text)
                            # Pattern: "value" "key": -> "value", "key":
                            repair_text = re.sub(r'"\s+"([^"]+)"\s*:', r'", "\1":', repair_text)
                            
                            # Fix missing commas after values before closing braces/brackets
                            repair_text = re.sub(r'(["\d\]}])\s*([}\]])', r'\1\2', repair_text)  # Remove space, but keep comma if needed
                            repair_text = re.sub(r'(["\d\]}])\s*"([a-zA-Z_])', r'\1, "\2', repair_text)  # Add comma before new key
                            
                            # 2. Fix double commas
                            repair_text = re.sub(r',\s*,+', ',', repair_text)  # Remove multiple consecutive commas
                            
                            # 3. Fix trailing commas (more comprehensive)
                            repair_text = re.sub(r',(\s*[}\]])', r'\1', repair_text)
                            
                            # 4. Fix incomplete strings (more comprehensive)
                            repair_text = re.sub(r':\s*"[^"]*$', ': ""', repair_text, flags=re.MULTILINE)
                            
                            # 5. Fix incomplete arrays
                            repair_text = re.sub(r':\s*\[[^\]]*$', ': []', repair_text, flags=re.MULTILINE)
                            
                            # 6. Fix incomplete objects
                            repair_text = re.sub(r':\s*\{[^\}]*$', ': {}', repair_text, flags=re.MULTILINE)
                            
                            # 7. Fix malformed arrays (leading/trailing commas)
                            repair_text = re.sub(r'\[\s*,+', '[', repair_text)  # Remove leading commas in arrays
                            repair_text = re.sub(r',+\s*\]', ']', repair_text)  # Remove trailing commas in arrays
                            
                            # 8. Fix missing commas in arrays (between elements)
                            # Pattern: "item1" "item2" -> "item1", "item2"
                            repair_text = re.sub(r'"\s+"([^"]*")', r'", "\1', repair_text)
                            
                            # 9. Fix missing commas after closing braces/brackets in nested structures
                            repair_text = re.sub(r'}\s+"', '}, "', repair_text)
                            repair_text = re.sub(r']\s+"', '], "', repair_text)
                            
                            # 10. Ensure proper structure - try to close incomplete qa_pairs array if needed
                            if '"qa_pairs"' in repair_text and not repair_text.rstrip().endswith(']'):
                                # Find the last complete object in qa_pairs array
                                qa_pairs_start = repair_text.find('"qa_pairs"')
                                if qa_pairs_start >= 0:
                                    # Try to find the last } before the end
                                    last_obj_end = repair_text.rfind('}')
                                    if last_obj_end > qa_pairs_start:
                                        # Check if we need to close the array
                                        after_last_obj = repair_text[last_obj_end:]
                                        if ']' not in after_last_obj:
                                            repair_text = repair_text[:last_obj_end+1] + ']}'
                            
                            logger.debug("Ultra-repaired QA JSON snippet: %s", repair_text[:500])
                            
                            data = json.loads(repair_text)
                            logger.warning("QA generation JSON: Ultra-aggressive repair succeeded!")
                            qa_pairs = data.get("qa_pairs", [])
                            
                            # İngilizce bloom_level'ları Türkçe'ye çevir
                            for qa in qa_pairs:
                                if "bloom_level" in qa and qa["bloom_level"].lower() in BLOOM_LEVEL_TRANSLATIONS:
                                    qa["bloom_level"] = translate_bloom_level(qa["bloom_level"])
                            
                            return qa_pairs
                        else:
                            raise ValueError("Could not find JSON boundaries in repair text")
                            
                    except Exception as e4:
                        # Final attempt: Parse error message to find exact location and fix
                        error_msg = str(e4)
                        logger.warning(f"QA generation JSON: Final repair attempt. Error: {error_msg}")
                        
                        try:
                            # Extract line and column from error message if available
                            line_match = re.search(r'line (\d+)', error_msg)
                            col_match = re.search(r'column (\d+)', error_msg)
                            
                            if line_match and col_match:
                                line_num = int(line_match.group(1))
                                col_num = int(col_match.group(1))
                                
                                # Split into lines and fix the specific line
                                lines = repair_text.split('\n')
                                if line_num <= len(lines):
                                    problem_line = lines[line_num - 1]
                                    logger.debug(f"Problem line {line_num}: {problem_line[:100]}")
                                    
                                    # Try to fix missing comma at the specific column
                                    if col_num < len(problem_line):
                                        # Look for patterns that need comma before this position
                                        before_col = problem_line[:col_num]
                                        after_col = problem_line[col_num:]
                                        
                                        # Common patterns that need comma:
                                        # - "value" "key" -> "value", "key"
                                        # - } "key" -> }, "key"
                                        # - ] "key" -> ], "key"
                                        
                                        if '"' in before_col and '"' in after_col:
                                            # Check if comma is missing between two quoted strings
                                            last_quote_before = before_col.rfind('"')
                                            if last_quote_before >= 0:
                                                between = before_col[last_quote_before+1:].strip()
                                                if not between.endswith(',') and not between.endswith(':'):
                                                    # Add comma before the quote
                                                    fixed_line = before_col[:last_quote_before+1] + ', ' + after_col
                                                    lines[line_num - 1] = fixed_line
                                                    repair_text = '\n'.join(lines)
                                                    logger.debug(f"Fixed line {line_num} by adding comma")
                            
                            # Try one more time with the fixed text
                            data = json.loads(repair_text)
                            logger.warning("QA generation JSON: Final repair attempt succeeded!")
                            qa_pairs = data.get("qa_pairs", [])
                            
                            # İngilizce bloom_level'ları Türkçe'ye çevir
                            for qa in qa_pairs:
                                if "bloom_level" in qa and qa["bloom_level"].lower() in BLOOM_LEVEL_TRANSLATIONS:
                                    qa["bloom_level"] = translate_bloom_level(qa["bloom_level"])
                            
                            return qa_pairs
                            
                        except Exception as e5:
                            logger.error(
                                "Error in QA generation JSON parsing (after all repair attempts): %s", str(e5)
                            )
                            logger.error("Original error: %s", str(e3))
                            logger.error("Second error: %s", str(e4))
                            logger.debug("Problematic QA JSON snippet (first 1000 chars): %s", json_str[:1000])
                            logger.debug("LLM output (first 500 chars): %s", llm_output[:500])
                            return []

    except Exception as e:
        logger.error(f"Error in QA generation: {e}")
        return []


async def extract_examples_and_applications(topic_title: str, chunks_text: str, model: str = "llama3:8b", system_prompt: Optional[str] = None) -> List[Dict]:
    """Extract real-world examples and applications"""
    
    # Use custom system prompt if provided
    if system_prompt:
        base_instruction = system_prompt.strip()
        logger.info(f"📚 [KB EXTRACTION] Using custom system prompt for examples: {base_instruction[:100]}...")
        prompt = f"""{base_instruction}

"{topic_title}" konusu için gerçek hayat örnekleri ve uygulama senaryoları oluştur.

MATERYAL:
{chunks_text[:10000]}

Her örnek şunları içermeli (TÜMÜ TÜRKÇE OLMALIDIR):
1. Başlık
2. Senaryo açıklaması
3. Konuyla ilişki açıklaması
4. Hangi kavramları gösterdiği

ÇIKTI FORMATI (JSON) - TÜM METİNLER TÜRKÇE OLMALIDIR:
{{
  "examples": [
    {{
      "title": "Tuzlu Suda Bitki Solması",
      "scenario": "Bir ev bitkisi yanlışlıkla tuzlu su ile sulandığında yaprakları solarır ve kurur.",
      "explanation": "Bu durum osmoz olayının sonucudur. Tuzlu su hipertonik bir ortam oluşturur ve bitki hücrelerindeki su dışarı çıkar. Hücre zarı bu süreci düzenleyemez çünkü tuz konsantrasyonu çok yüksektir.",
      "concepts_demonstrated": ["osmoz", "hipertonik çözelti", "hücre zarı"],
      "difficulty": "intermediate",
      "real_world_relevance": "high"
    }}
  ]
}}

5-8 örnek oluştur. 

ÖNEMLİ JSON KURALLARI:
1. Her alan arasında virgül (,) olmalı
2. Son alanda virgül OLMAMALI
3. Her obje arasında virgül olmalı
4. Sadece geçerli JSON formatı kullan
5. Tırnak işaretlerini doğru kapat
6. Sadece JSON çıktısı ver, başka metin ekleme

Sadece JSON çıktısı ver."""

    try:
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model,
                "max_tokens": 1500,
                "temperature": 0.6,
            },
            timeout=90,
        )

        if response.status_code != 200:
            logger.error(
                f"LLM service error during examples extraction: {response.status_code} - {response.text}"
            )
            return []

        result = response.json()
        llm_output = result.get("response", "")

        if not llm_output:
            logger.warning("Examples extraction returned empty response")
            return []

        # Try to extract JSON block
        json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if not json_match:
            logger.error(
                "Examples extraction: could not find JSON object in LLM output. "
                "First 300 chars: %s",
                llm_output[:300],
            )
            return []

        json_str = json_match.group()

        # First parse attempt
        try:
            data = json.loads(json_str)
            return data.get("examples", [])
        except json.JSONDecodeError as e:
            # Second attempt: remove trailing commas before } or ]
            try:
                cleaned = re.sub(r",(\s*[}\]])", r"\1", json_str)
                data = json.loads(cleaned)
                logger.warning(
                    "Examples JSON required cleanup but was parsed successfully: %s",
                    str(e),
                )
                return data.get("examples", [])
            except Exception as e2:
                # Third attempt: more aggressive cleanup (same as QA generation)
                try:
                    # Remove trailing commas, fix incomplete fields
                    cleaned = re.sub(r",(\s*[}\]])", r"\1", json_str)
                    cleaned = re.sub(r':\s*"[^"]*$', ': ""', cleaned, flags=re.MULTILINE)  # Fix incomplete strings
                    cleaned = re.sub(r':\s*\[[^\]]*$', ': []', cleaned, flags=re.MULTILINE)  # Fix incomplete arrays
                    cleaned = re.sub(r':\s*\{[^\}]*$', ': {}', cleaned, flags=re.MULTILINE)  # Fix incomplete objects
                    # Remove any remaining trailing commas
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                    data = json.loads(cleaned)
                    logger.warning("Examples JSON required aggressive cleanup but was parsed successfully")
                    return data.get("examples", [])
                except Exception as e3:
                    # Fourth attempt: ULTRA-AGGRESSIVE JSON repair (similar to topics.py)
                    try:
                        logger.warning("Examples JSON: Attempting ultra-aggressive repair...")
                        
                        # Extract everything between first { and last }
                        repair_text = json_str.strip()
                        first_brace = repair_text.find('{')
                        last_brace = repair_text.rfind('}')
                        
                        if first_brace >= 0 and last_brace > first_brace:
                            repair_text = repair_text[first_brace:last_brace + 1]
                            
                            # 1. Fix missing commas between fields (more comprehensive)
                            repair_text = re.sub(r'"\s*"\s*"', '", "', repair_text)
                            repair_text = re.sub(r'}\s*{', '},{', repair_text)
                            repair_text = re.sub(r']\s*{', '],{', repair_text)
                            repair_text = re.sub(r'}\s*"', '},"', repair_text)
                            repair_text = re.sub(r']\s*"', '],"', repair_text)
                            repair_text = re.sub(r'([0-9])\s*"', r'\1,"', repair_text)
                            repair_text = re.sub(r'"\s*([0-9])', r'", \1', repair_text)
                            
                            # Fix missing commas after closing quotes before new keys (most common issue)
                            repair_text = re.sub(r'"\s+"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', r'", "\1":', repair_text)
                            repair_text = re.sub(r'"\s+"([^"]+)"\s*:', r'", "\1":', repair_text)
                            
                            # Fix missing commas after values before closing braces/brackets
                            repair_text = re.sub(r'(["\d\]}])\s*"([a-zA-Z_])', r'\1, "\2', repair_text)
                            
                            # 2. Fix double commas
                            repair_text = re.sub(r',\s*,+', ',', repair_text)
                            
                            # 3. Fix trailing commas
                            repair_text = re.sub(r',(\s*[}\]])', r'\1', repair_text)
                            
                            # 4. Fix incomplete strings
                            repair_text = re.sub(r':\s*"[^"]*$', ': ""', repair_text, flags=re.MULTILINE)
                            
                            # 5. Fix incomplete arrays
                            repair_text = re.sub(r':\s*\[[^\]]*$', ': []', repair_text, flags=re.MULTILINE)
                            
                            # 6. Fix incomplete objects
                            repair_text = re.sub(r':\s*\{[^\}]*$', ': {}', repair_text, flags=re.MULTILINE)
                            
                            # 7. Fix malformed arrays
                            repair_text = re.sub(r'\[\s*,+', '[', repair_text)
                            repair_text = re.sub(r',+\s*\]', ']', repair_text)
                            
                            # 8. Fix missing commas in arrays
                            repair_text = re.sub(r'"\s+"([^"]*")', r'", "\1', repair_text)
                            
                            # 9. Fix missing commas after closing braces/brackets
                            repair_text = re.sub(r'}\s+"', '}, "', repair_text)
                            repair_text = re.sub(r']\s+"', '], "', repair_text)
                            
                            logger.debug("Ultra-repaired examples JSON snippet: %s", repair_text[:500])
                            
                            data = json.loads(repair_text)
                            logger.warning("Examples JSON: Ultra-aggressive repair succeeded!")
                            return data.get("examples", [])
                        else:
                            raise ValueError("Could not find JSON boundaries in repair text")
                            
                    except Exception as e4:
                        # Final attempt: Parse error message to find exact location and fix
                        error_msg = str(e4)
                        logger.warning(f"Examples JSON: Final repair attempt. Error: {error_msg}")
                        
                        try:
                            # Extract line and column from error message if available
                            line_match = re.search(r'line (\d+)', error_msg)
                            col_match = re.search(r'column (\d+)', error_msg)
                            
                            if line_match and col_match:
                                line_num = int(line_match.group(1))
                                col_num = int(col_match.group(1))
                                
                                # Split into lines and fix the specific line
                                lines = repair_text.split('\n')
                                if line_num <= len(lines):
                                    problem_line = lines[line_num - 1]
                                    logger.debug(f"Problem line {line_num}: {problem_line[:100]}")
                                    
                                    # Try to fix missing comma at the specific column
                                    if col_num < len(problem_line):
                                        # Look for patterns that need comma before this position
                                        before_col = problem_line[:col_num]
                                        after_col = problem_line[col_num:]
                                        
                                        # Common patterns that need comma:
                                        # - "value" "key" -> "value", "key"
                                        # - } "key" -> }, "key"
                                        # - ] "key" -> ], "key"
                                        
                                        if '"' in before_col and '"' in after_col:
                                            # Check if comma is missing between two quoted strings
                                            last_quote_before = before_col.rfind('"')
                                            if last_quote_before >= 0:
                                                between = before_col[last_quote_before+1:].strip()
                                                if not between.endswith(',') and not between.endswith(':'):
                                                    # Add comma before the quote
                                                    fixed_line = before_col[:last_quote_before+1] + ', ' + after_col
                                                    lines[line_num - 1] = fixed_line
                                                    repair_text = '\n'.join(lines)
                                                    logger.debug(f"Fixed line {line_num} by adding comma")
                            
                            # Try one more time with the fixed text
                            data = json.loads(repair_text)
                            logger.warning("Examples JSON: Final repair attempt succeeded!")
                            return data.get("examples", [])
                            
                        except Exception as e5:
                            logger.error(
                                "Error in examples JSON parsing (after all repair attempts): %s", str(e5)
                            )
                            logger.error("Original error: %s", str(e3))
                            logger.error("Second error: %s", str(e4))
                            logger.debug("Problematic examples JSON snippet (first 1000 chars): %s", json_str[:1000])
                            logger.debug("LLM output (first 500 chars): %s", llm_output[:500])
                            return []

    except Exception as e:
        logger.error(f"Error in examples extraction: {e}")
        return []


def calculate_quality_score(summary: str, concepts: List, objectives: List, qa_pairs: List) -> float:
    """Calculate overall quality score for extracted knowledge"""
    
    score = 0.0
    
    # Summary quality (30%)
    if summary:
        word_count = len(summary.split())
        if 150 <= word_count <= 400:
            score += 0.30
        elif 100 <= word_count < 150 or 400 < word_count <= 500:
            score += 0.20
        else:
            score += 0.10
    
    # Concepts coverage (25%)
    if len(concepts) >= 5:
        score += 0.25
    elif len(concepts) >= 3:
        score += 0.15
    
    # Learning objectives (20%)
    if len(objectives) >= 4:
        score += 0.20
    elif len(objectives) >= 2:
        score += 0.10
    
    # QA pairs quantity and quality (25%)
    if len(qa_pairs) >= 10:
        score += 0.15
        # Check difficulty distribution
        difficulties = [qa.get("difficulty") for qa in qa_pairs]
        if "beginner" in difficulties and "intermediate" in difficulties and "advanced" in difficulties:
            score += 0.10
    elif len(qa_pairs) >= 5:
        score += 0.10
    
    return min(score, 1.0)  # Cap at 1.0


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/extract/{topic_id}")
async def extract_knowledge_for_topic(topic_id: int, request: KnowledgeExtractionRequest = None, system_prompt: Optional[str] = None):
    """
    Extract comprehensive structured knowledge for a topic
    
    This endpoint:
    1. Gets topic info and related chunks
    2. Extracts summary, concepts, objectives
    3. Generates examples and applications
    4. Stores in topic_knowledge_base table
    
    Args:
        topic_id: Topic ID to extract knowledge for
        request: Optional extraction request with force_refresh flag
        system_prompt: Optional custom system prompt for LLM extraction
    """
    
    logger.info(f"🔄 [KB DEBUG] Starting knowledge extraction for topic_id={topic_id}")
    
    db = get_db()
    
    try:
        # Check if knowledge already exists
        if not (request and request.force_refresh):
            logger.info(f"🔍 [KB DEBUG] Checking existing KB for topic {topic_id}")
            with db.get_connection() as conn:
                existing = conn.execute("""
                    SELECT knowledge_id FROM topic_knowledge_base WHERE topic_id = ?
                """, (topic_id,)).fetchone()
                
                if existing:
                    logger.info(f"✅ [KB DEBUG] Existing KB found for topic {topic_id}, returning cached version")
                    return {
                        "success": True,
                        "message": "Knowledge base already exists. Use force_refresh=true to regenerate.",
                        "knowledge_id": dict(existing)["knowledge_id"],
                        "was_regenerated": False
                    }
        
        logger.info(f"📚 [KB DEBUG] Getting topic information for topic_id={topic_id}")
        # Get topic information
        topic = get_topic_info(topic_id, db)
        if not topic:
            logger.error(f"❌ [KB DEBUG] Topic {topic_id} not found in database")
            raise HTTPException(status_code=404, detail="Topic not found")
        
        logger.info(f"📄 [KB DEBUG] Topic found: {topic['topic_title']} (session: {topic['session_id']})")
        
        # Fetch chunks
        logger.info(f"📦 [KB DEBUG] Fetching chunks for session {topic['session_id']}")
        all_chunks = fetch_chunks_for_session(topic["session_id"])
        if not all_chunks:
            logger.error(f"❌ [KB DEBUG] No chunks found for session {topic['session_id']}")
            raise HTTPException(status_code=404, detail="No chunks found for session")
        
        logger.info(f"📦 [KB DEBUG] Found {len(all_chunks)} chunks for session")
        
        # Filter relevant chunks
        relevant_chunks = filter_chunks_by_topic(
            all_chunks,
            topic["keywords"],
            topic["related_chunk_ids"]
        )
        
        if not relevant_chunks:
            logger.error(f"❌ [KB DEBUG] CRITICAL: No relevant chunks for topic {topic_id}!")
            logger.error(f"🔧 [KB DEBUG] This indicates broken topic-chunk relationships!")
            logger.error(f"💡 [KB DEBUG] SOLUTION: Regenerate topics to establish proper chunk links")
            
            # Instead of using random chunks, return a meaningful error
            raise HTTPException(
                status_code=422,
                detail=f"Konu ile döküman parçaları arasında bağlantı bulunamadı. Konuları yeniden ürettirmeniz gerekiyor. (Topic-chunk relationships are broken. Please regenerate topics.)"
            )
        
        logger.info(f"🎯 [KB DEBUG] Using {len(relevant_chunks)} relevant chunks for extraction")
        
        # Prepare chunks text
        chunks_text = "\n\n---\n\n".join([
            chunk.get("chunk_text", chunk.get("content", ""))
            for chunk in relevant_chunks
        ])
        
        logger.info(f"📝 [KB DEBUG] Prepared {len(chunks_text)} characters of text for LLM")
        
        # Get session's preferred model
        model_to_use = get_session_model(topic["session_id"])
        logger.info(f"🤖 [KB DEBUG] Using model: {model_to_use} for extraction")
        
        # EXTRACTION PIPELINE
        extraction_start = datetime.now()
        logger.info(f"🚀 [KB DEBUG] Starting LLM extraction pipeline...")
        
        # Get system_prompt from request if provided
        system_prompt = request.system_prompt if request else None
        
        # 1. Summary
        logger.info(f"📝 [KB DEBUG] Step 1: Extracting topic summary...")
        summary = await extract_topic_summary(topic["topic_title"], chunks_text, model_to_use, system_prompt)
        logger.info(f"📝 [KB DEBUG] Summary extracted: {len(summary)} characters")
        
        # 2. Key Concepts
        logger.info(f"💡 [KB DEBUG] Step 2: Extracting key concepts...")
        concepts = await extract_key_concepts(topic["topic_title"], chunks_text, model_to_use, system_prompt)
        logger.info(f"💡 [KB DEBUG] Extracted {len(concepts)} concepts")
        
        # 3. Learning Objectives
        logger.info(f"🎯 [KB DEBUG] Step 3: Extracting learning objectives...")
        objectives = await extract_learning_objectives(topic["topic_title"], chunks_text, model_to_use, system_prompt)
        logger.info(f"🎯 [KB DEBUG] Extracted {len(objectives)} learning objectives")
        
        # 4. Examples
        logger.info(f"📚 [KB DEBUG] Step 4: Extracting examples and applications...")
        examples = await extract_examples_and_applications(topic["topic_title"], chunks_text, model_to_use, system_prompt)
        logger.info(f"📚 [KB DEBUG] Extracted {len(examples)} examples")
        
        extraction_time = (datetime.now() - extraction_start).total_seconds()
        logger.info(f"⏱️ [KB DEBUG] Extraction pipeline completed in {extraction_time:.2f} seconds")
        
        # Calculate quality score
        quality_score = calculate_quality_score(summary, concepts, objectives, [])
        logger.info(f"📊 [KB DEBUG] Calculated quality score: {quality_score:.2f}")
        
        # Save to database
        logger.info(f"💾 [KB DEBUG] Saving extracted knowledge to database...")
        with db.get_connection() as conn:
            # Delete existing if force_refresh
            if request and request.force_refresh:
                logger.info(f"🗑️ [KB DEBUG] Deleting existing KB for topic {topic_id} (force_refresh=True)")
                conn.execute(
                    "DELETE FROM topic_knowledge_base WHERE topic_id = ?", (topic_id,)
                )

            cursor = conn.execute(
                """
                INSERT INTO topic_knowledge_base (
                    topic_id, topic_summary, key_concepts, learning_objectives,
                    examples, content_quality_score, extraction_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    topic_id,
                    summary,
                    json.dumps(concepts, ensure_ascii=False),
                    json.dumps(objectives, ensure_ascii=False),
                    json.dumps(examples, ensure_ascii=False),
                    quality_score,
                    model_to_use,
                ),
            )

            knowledge_id = cursor.lastrowid
            conn.commit()
            logger.info(f"✅ [KB DEBUG] Successfully saved KB with knowledge_id={knowledge_id}")

        # Ensure QA pairs exist for this topic (generate once if missing)
        qa_pairs_generated = 0
        try:
            with db.get_connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM topic_qa_pairs WHERE topic_id = ?",
                    (topic_id,),
                ).fetchone()
                existing_qa_count = dict(row)["c"] if row else 0

            if existing_qa_count == 0:
                logger.info(
                    f"No QA pairs found for topic {topic_id}, generating default set"
                )
                qa_pairs = await generate_qa_pairs(
                    topic["topic_title"],
                    chunks_text,
                    15,
                    None,
                    model_to_use,
                )

                if qa_pairs:
                    with db.get_connection() as conn:
                        qa_ids = []
                        for qa in qa_pairs:
                            cursor = conn.execute(
                                """
                                INSERT INTO topic_qa_pairs (
                                    topic_id, question, answer, explanation,
                                    difficulty_level, question_type, bloom_taxonomy_level,
                                    related_concepts, extraction_method, extraction_model
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    topic_id,
                                    qa.get("question", ""),
                                    qa.get("answer", ""),
                                    qa.get("explanation", ""),
                                    qa.get("difficulty", "beginner"),
                                    qa.get("question_type", "factual"),
                                    qa.get("bloom_level", "remember"),
                                    json.dumps(
                                        qa.get("related_concepts", []),
                                        ensure_ascii=False,
                                    ),
                                    "llm_generated",
                                    model_to_use,
                                ),
                            )
                            qa_ids.append({
                                "qa_id": cursor.lastrowid,
                                "question": qa.get("question", "")
                            })
                        conn.commit()
                        qa_pairs_generated = len(qa_pairs)
                        
                        # Calculate and store embeddings for new QA pairs
                        if qa_ids:
                            logger.info(f"📦 Calculating embeddings for {len(qa_ids)} new QA pairs...")
                            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
                            await calculate_and_store_qa_embeddings_batch(
                                qa_pairs=qa_ids,
                                embedding_model=embedding_model,
                                db=db
                            )
        except Exception as e:
            logger.error(
                f"Error generating QA pairs inside knowledge extraction for topic {topic_id}: {e}"
            )
        
        return {
            "success": True,
            "knowledge_id": knowledge_id,
            "topic_id": topic_id,
            "topic_title": topic["topic_title"],
            "extracted_components": {
                "summary_length": len(summary.split()),
                "concepts_count": len(concepts),
                "objectives_count": len(objectives),
                "examples_count": len(examples),
                "qa_pairs_generated": qa_pairs_generated,
            },
            "quality_score": quality_score,
            "extraction_time_seconds": round(extraction_time, 2),
            "was_regenerated": request.force_refresh if request else False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in knowledge extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Knowledge extraction failed: {str(e)}")


@router.post("/extract-batch/{session_id}")
async def extract_knowledge_batch(
    session_id: str,
    request: BatchKnowledgeExtractionRequest = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Start background knowledge extraction job for all topics in a session.
    Returns a job_id immediately; progress can be tracked via
    GET /extract-batch/status/{job_id}
    """

    db = get_db()

    try:
        # Get all topics for session
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT topic_id, topic_title
                FROM course_topics
                WHERE session_id = ? AND is_active = TRUE
                ORDER BY topic_order
            """, (session_id,))

            topics = [dict(row) for row in cursor.fetchall()]

        if not topics:
            raise HTTPException(status_code=404, detail="No topics found for session")

        job_id = str(uuid.uuid4())

        # Initialize job status
        BATCH_KB_JOBS[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "running",  # running | completed | failed
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "total_topics": len(topics),
            "processed_successfully": 0,
            "errors_count": 0,
            "current_topic_id": None,
            "current_topic_title": None,
            # results & errors kept small to avoid huge payloads
            "results": [],
            "errors": [],
        }

        # Convert request model to plain dict for background task
        request_data: Dict[str, Any] = request.model_dump() if request else {}

        # Schedule background job
        if background_tasks is None:
            # Safety fallback: run inline (mainly for tests)
            await _run_batch_extraction_job(session_id, topics, request_data, job_id)
        else:
            background_tasks.add_task(
                _run_batch_extraction_job, session_id, topics, request_data, job_id
            )

        return {
            "success": True,
            "message": "Knowledge extraction started in background",
            "job_id": job_id,
            "total_topics": len(topics),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch extraction start failed: {str(e)}")


async def _run_batch_extraction_job(
    session_id: str,
    topics: List[Dict[str, Any]],
    request_data: Dict[str, Any],
    job_id: str,
):
    """
    Internal background job that performs batch KB extraction.
    Updates BATCH_KB_JOBS[job_id] with progress.
    """
    db = get_db()
    job = BATCH_KB_JOBS.get(job_id)

    if not job:
        # Job was somehow removed; nothing to do
        logger.warning(f"[KB BATCH JOB] Job {job_id} not found at start")
        return

    try:
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        force_refresh = bool(request_data.get("force_refresh", False))
        system_prompt = request_data.get("system_prompt")
        extraction_config = request_data.get("extraction_config") or {}
        generate_qa_pairs = extraction_config.get("generate_qa_pairs", True)
        qa_pairs_per_topic = extraction_config.get("qa_pairs_per_topic", 15)

        for topic in topics:
            try:
                topic_id = topic["topic_id"]
                topic_title = topic["topic_title"]

                logger.info(f"[KB BATCH JOB {job_id}] Processing topic: {topic_title}")

                # Update current topic in job status
                job["current_topic_id"] = topic_id
                job["current_topic_title"] = topic_title

                # Extract knowledge
                extraction_req = KnowledgeExtractionRequest(
                    topic_id=topic_id,
                    force_refresh=force_refresh,
                )
                result = await extract_knowledge_for_topic(topic_id, extraction_req, system_prompt)

                # Generate QA pairs if requested
                if generate_qa_pairs:
                    qa_req = QAGenerationRequest(
                        topic_id=topic_id,
                        count=qa_pairs_per_topic,
                    )
                    qa_result = await generate_qa_pairs_endpoint(topic_id, qa_req)
                    result["qa_pairs_generated"] = qa_result.get("count", 0)

                results.append(
                    {
                        "topic_id": result.get("topic_id"),
                        "topic_title": result.get("topic_title"),
                        "knowledge_id": result.get("knowledge_id"),
                        "qa_pairs_generated": result.get("qa_pairs_generated", 0),
                    }
                )

                # Update progress
                job["processed_successfully"] = len(results)
                job["results"] = results

            except Exception as e:
                import traceback

                error_detail = traceback.format_exc()
                logger.error(
                    f"[KB BATCH JOB {job_id}] Error processing topic {topic.get('topic_id')} "
                    f"({topic.get('topic_title')}): {e}"
                )
                logger.error(f"[KB BATCH JOB {job_id}] Full error trace: {error_detail}")

                error_entry = {
                    "topic_id": topic.get("topic_id"),
                    "topic_title": topic.get("topic_title"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                errors.append(error_entry)
                job["errors"] = errors
                job["errors_count"] = len(errors)

        # Mark job as completed
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow().isoformat()
        job["processed_successfully"] = len(results)
        job["errors_count"] = len(errors)

    except Exception as e:
        import traceback

        logger.error(f"[KB BATCH JOB {job_id}] Fatal error in batch extraction: {e}")
        logger.error(traceback.format_exc())

        job["status"] = "failed"
        job["completed_at"] = datetime.utcnow().isoformat()
        job["error"] = str(e)
        job["error_type"] = type(e).__name__


@router.get("/extract-batch/status/{job_id}")
async def get_knowledge_batch_status(job_id: str):
    """
    Get status of a background KB extraction batch job
    """
    job = BATCH_KB_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/extract-batch-missing/{session_id}")
async def extract_knowledge_batch_missing(
    session_id: str,
    request: BatchKnowledgeExtractionRequest = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Start background knowledge extraction job for topics that don't have KB yet.
    Returns a job_id immediately; progress can be tracked via
    GET /extract-batch/status/{job_id}
    """

    db = get_db()

    try:
        # Get all topics for session that don't have KB
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT ct.topic_id, ct.topic_title
                FROM course_topics ct
                LEFT JOIN topic_knowledge_base tkb ON ct.topic_id = tkb.topic_id
                WHERE ct.session_id = ? 
                  AND ct.is_active = TRUE
                  AND tkb.topic_id IS NULL
                ORDER BY ct.topic_order
            """, (session_id,))

            topics = [dict(row) for row in cursor.fetchall()]

        if not topics:
            return {
                "success": True,
                "message": "All topics already have knowledge base",
                "job_id": None,
                "total_topics": 0,
                "missing_count": 0,
            }

        job_id = str(uuid.uuid4())

        # Initialize job status
        BATCH_KB_JOBS[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "running",  # running | completed | failed
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "total_topics": len(topics),
            "processed_successfully": 0,
            "errors_count": 0,
            "current_topic_id": None,
            "current_topic_title": None,
            # results & errors kept small to avoid huge payloads
            "results": [],
            "errors": [],
        }

        # Convert request model to plain dict for background task
        request_data: Dict[str, Any] = request.model_dump() if request else {}
        request_data["force_refresh"] = False  # Don't overwrite existing KBs

        # Schedule background job
        if background_tasks is None:
            # Safety fallback: run inline (mainly for tests)
            await _run_batch_extraction_job(session_id, topics, request_data, job_id)
        else:
            background_tasks.add_task(
                _run_batch_extraction_job, session_id, topics, request_data, job_id
            )

        return {
            "success": True,
            "message": f"Knowledge extraction started for {len(topics)} missing topics",
            "job_id": job_id,
            "total_topics": len(topics),
            "missing_count": len(topics),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch extraction for missing KBs: {e}")
        raise HTTPException(status_code=500, detail=f"Batch extraction start failed: {str(e)}")


@router.post("/generate-qa/{topic_id}")
async def generate_qa_pairs_endpoint(topic_id: int, request: QAGenerationRequest):
    """
    Generate QA pairs for a topic and store in database
    """
    
    db = get_db()
    
    try:
        # Get topic and chunks
        topic = get_topic_info(topic_id, db)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        
        all_chunks = fetch_chunks_for_session(topic["session_id"])
        relevant_chunks = filter_chunks_by_topic(all_chunks, topic["keywords"], topic["related_chunk_ids"])
        
        chunks_text = "\n\n---\n\n".join([
            chunk.get("chunk_text", chunk.get("content", ""))
            for chunk in relevant_chunks
        ])
        
        # Generate QA pairs
        qa_pairs = await generate_qa_pairs(
            topic["topic_title"],
            chunks_text,
            request.count,
            request.difficulty_distribution
        )
        
        # Store in database
        qa_ids = []
        with db.get_connection() as conn:
            for qa in qa_pairs:
                cursor = conn.execute("""
                    INSERT INTO topic_qa_pairs (
                        topic_id, question, answer, explanation,
                        difficulty_level, question_type, bloom_taxonomy_level,
                        related_concepts, extraction_method, extraction_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    topic_id,
                    qa["question"],
                    qa["answer"],
                    qa.get("explanation", ""),
                    qa["difficulty"],
                    qa["question_type"],
                    qa.get("bloom_level", "Hatırlama"),  # Türkçe default
                    json.dumps(qa.get("related_concepts", []), ensure_ascii=False),
                    "llm_generated",
                    "llama-3.1-8b-instant"
                ))
                qa_ids.append({
                    "qa_id": cursor.lastrowid,
                    "question": qa["question"]
                })
            
            conn.commit()
        
        # Calculate and store embeddings for new QA pairs
        if qa_ids:
            logger.info(f"📦 Calculating embeddings for {len(qa_ids)} new QA pairs...")
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
            await calculate_and_store_qa_embeddings_batch(
                qa_pairs=qa_ids,
                embedding_model=embedding_model,
                db=db
            )
        
        return {
            "success": True,
            "topic_id": topic_id,
            "topic_title": topic["topic_title"],
            "count": len(qa_pairs),
            "qa_pairs": translate_qa_pairs(qa_pairs)  # Türkçeleştirilmiş QA pairs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating QA pairs: {e}")
        raise HTTPException(status_code=500, detail=f"QA generation failed: {str(e)}")


@router.get("/kb/{topic_id}")
async def get_knowledge_base(topic_id: int):
    """
    Get knowledge base for a topic
    """
    
    db = get_db()
    
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM topic_knowledge_base WHERE topic_id = ?
            """, (topic_id,))
            
            kb = cursor.fetchone()
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")
            
            kb_dict = dict(kb)
            
            # Parse JSON fields
            kb_dict["key_concepts"] = json.loads(kb_dict["key_concepts"]) if kb_dict["key_concepts"] else []
            learning_objectives = json.loads(kb_dict["learning_objectives"]) if kb_dict["learning_objectives"] else []
            kb_dict["learning_objectives"] = translate_learning_objectives(learning_objectives)
            kb_dict["examples"] = json.loads(kb_dict["examples"]) if kb_dict["examples"] else []
            
            # Get QA pairs
            cursor = conn.execute("""
                SELECT qa_id, question, answer, difficulty_level, question_type, 
                       bloom_taxonomy_level, times_asked, average_student_rating
                FROM topic_qa_pairs
                WHERE topic_id = ? AND is_active = TRUE
                ORDER BY times_asked DESC, average_student_rating DESC
                LIMIT 20
            """, (topic_id,))
            
            qa_pairs = [dict(row) for row in cursor.fetchall()]
            kb_dict["qa_pairs"] = translate_qa_pairs(qa_pairs)
            
            return {
                "success": True,
                "knowledge_base": kb_dict
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch knowledge base: {str(e)}")


# ============================================================================
# SELECTIVE KB REFRESH ENDPOINTS
# ============================================================================

class SelectiveRefreshRequest(BaseModel):
    """Request model for selective KB component refresh"""
    topic_id: int
    force_refresh: bool = True


@router.post("/refresh-summary/{topic_id}")
async def refresh_topic_summary_only(topic_id: int, request: SelectiveRefreshRequest = None):
    """
    Refresh only the topic summary without touching other KB components
    """
    logger.info(f"🔄 [SELECTIVE REFRESH] Starting summary refresh for topic_id={topic_id}")
    
    db = get_db()
    
    try:
        # Get topic and existing KB
        topic = get_topic_info(topic_id, db)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        # Get existing KB to provide context
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT topic_summary, key_concepts, learning_objectives
                FROM topic_knowledge_base WHERE topic_id = ?
            """, (topic_id,))
            
            existing_kb = cursor.fetchone()
            if not existing_kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found. Create KB first.")
            
            kb_dict = dict(existing_kb)

        # Fetch relevant chunks
        all_chunks = fetch_chunks_for_session(topic["session_id"])
        relevant_chunks = filter_chunks_by_topic(
            all_chunks, topic["keywords"], topic["related_chunk_ids"]
        )
        
        if not relevant_chunks:
            raise HTTPException(
                status_code=422,
                detail="No relevant chunks found for topic"
            )

        chunks_text = "\n\n---\n\n".join([
            chunk.get("chunk_text", chunk.get("content", ""))
            for chunk in relevant_chunks
        ])

        model_to_use = get_session_model(topic["session_id"])
        
        # Enhanced summary prompt with context
        current_concepts = json.loads(kb_dict["key_concepts"]) if kb_dict["key_concepts"] else []
        current_objectives = json.loads(kb_dict["learning_objectives"]) if kb_dict["learning_objectives"] else []
        
        prompt = f"""Sen bir TÜRKÇE eğitim içeriği uzmanısın. Aşağıdaki "{topic['topic_title']}" konusu için mevcut bilgi tabanını analiz ederek SADECE ÖZETİ yenile.

KONU: {topic['topic_title']}

MEVCUT BİLGİ TABANI:
Özet: {kb_dict['topic_summary'] or 'Yok'}
Kavramlar: {[c.get('term', '') for c in current_concepts[:5]]}
Öğrenme Hedefleri: {[o.get('objective', '')[:50] + '...' for o in current_objectives[:3]]}

DERS MATERYALİ:
{chunks_text[:12000]}

ÖZET YENİLEME KURALLARI:
1. 250-350 kelime arası kapsamlı özet
2. Mevcut kavramlar ve hedefleri dikkate al
3. Eksik olan önemli noktaları tamamla
4. Akademik ama anlaşılır TÜRKÇE
5. Konunun güncel ve doğru sunumu
6. Önceki özetteki hataları düzelt
7. Mantıksal akış: tanım → özellikler → önem → uygulamalar
8. Kesinlikle İngilizce cümle kurma, cevap TAMAMEN TÜRKÇE olsun

YENİ ÖZET (SADECE METİN, JSON DEĞİL):"""

        # Generate new summary
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model_to_use,
                "max_tokens": 700,
                "temperature": 0.3
            },
            timeout=90
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM service error")
        
        result = response.json()
        new_summary = result.get("response", "").strip()
        
        # Clean up response
        new_summary = re.sub(r"^Here\s+is\s+.*?:\s*", "", new_summary, flags=re.IGNORECASE)
        new_summary = new_summary.replace("YENİ ÖZET:", "").strip()
        
        if not new_summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary")

        # Update only summary in database
        with db.get_connection() as conn:
            conn.execute("""
                UPDATE topic_knowledge_base
                SET topic_summary = ?
                WHERE topic_id = ?
            """, (new_summary, topic_id))
            conn.commit()

        logger.info(f"✅ [SELECTIVE REFRESH] Summary updated for topic {topic_id}")

        return {
            "success": True,
            "topic_id": topic_id,
            "topic_title": topic["topic_title"],
            "component": "summary",
            "new_summary": new_summary,
            "word_count": len(new_summary.split())
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in selective summary refresh: {e}")
        raise HTTPException(status_code=500, detail=f"Summary refresh failed: {str(e)}")


@router.post("/refresh-concepts/{topic_id}")
async def refresh_topic_concepts_only(topic_id: int, request: SelectiveRefreshRequest = None):
    """
    Refresh only the key concepts without touching other KB components
    """
    logger.info(f"🔄 [SELECTIVE REFRESH] Starting concepts refresh for topic_id={topic_id}")
    
    db = get_db()
    
    try:
        # Get topic and existing KB
        topic = get_topic_info(topic_id, db)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        # Get existing concepts for context
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT key_concepts FROM topic_knowledge_base WHERE topic_id = ?
            """, (topic_id,))
            
            existing_kb = cursor.fetchone()
            if not existing_kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found. Create KB first.")
            
            current_concepts = json.loads(dict(existing_kb)["key_concepts"]) if dict(existing_kb)["key_concepts"] else []

        # Fetch relevant chunks
        all_chunks = fetch_chunks_for_session(topic["session_id"])
        relevant_chunks = filter_chunks_by_topic(
            all_chunks, topic["keywords"], topic["related_chunk_ids"]
        )
        
        chunks_text = "\n\n---\n\n".join([
            chunk.get("chunk_text", chunk.get("content", ""))
            for chunk in relevant_chunks
        ])

        model_to_use = get_session_model(topic["session_id"])
        
        # Enhanced concepts prompt
        current_concepts_text = "\n".join([f"- {c.get('term', 'N/A')}: {c.get('definition', 'N/A')}" for c in current_concepts[:10]]) if current_concepts else "Henüz kavram yok"
        
        prompt = f""""{topic['topic_title']}" konusu için SADECE TEMEL KAVRAMLARI yenile. Mevcut kavramları analiz et ve geliştir.

MEVCUT KAVRAMLAR:
{current_concepts_text}

MATERYAL:
{chunks_text[:10000]}

KAVRAM YENİLEME KURALLARI:
1. Eksik olan kritik kavramları ekle
2. Yanlış tanımları düzelt
3. Önem seviyelerini güncelle
4. 6-12 temel kavram hedefle
5. Her kavram için net TÜRKÇE tanım
6. Kategorileri mantıklı grupla
7. TÜM terimler ve tanımlar TÜRKÇE olsun, İngilizce kelime kullanma

ÇIKTI FORMATI (JSON):
{{
  "concepts": [
    {{
      "term": "Kavram İsmi",
      "definition": "Net Türkçe tanım (1-2 cümle)",
      "importance": "high",
      "category": "yapı"
    }},
    {{
      "term": "İkinci Kavram",
      "definition": "Net Türkçe tanım",
      "importance": "medium",
      "category": "süreç"
    }}
  ]
}}

Sadece JSON çıktısı ver, başka metin ekleme."""

        # Generate new concepts
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model_to_use,
                "max_tokens": 1500,
                "temperature": 0.3
            },
            timeout=90
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM service error")
        
        result = response.json()
        llm_output = result.get("response", "")
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not json_match:
            raise HTTPException(status_code=500, detail="Failed to parse concepts JSON")
            
        try:
            data = json.loads(json_match.group())
            new_concepts = data.get("concepts", [])
        except json.JSONDecodeError:
            # Try cleanup
            cleaned = re.sub(r",(\s*[}\]])", r"\1", json_match.group())
            data = json.loads(cleaned)
            new_concepts = data.get("concepts", [])

        if not new_concepts:
            raise HTTPException(status_code=500, detail="No concepts extracted")

        # Update only concepts in database
        with db.get_connection() as conn:
            conn.execute("""
                UPDATE topic_knowledge_base
                SET key_concepts = ?
                WHERE topic_id = ?
            """, (json.dumps(new_concepts, ensure_ascii=False), topic_id))
            conn.commit()

        logger.info(f"✅ [SELECTIVE REFRESH] Concepts updated for topic {topic_id} ({len(new_concepts)} concepts)")

        return {
            "success": True,
            "topic_id": topic_id,
            "topic_title": topic["topic_title"],
            "component": "concepts",
            "new_concepts": new_concepts,
            "concepts_count": len(new_concepts)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in selective concepts refresh: {e}")
        raise HTTPException(status_code=500, detail=f"Concepts refresh failed: {str(e)}")


@router.post("/refresh-objectives/{topic_id}")
async def refresh_learning_objectives_only(topic_id: int, request: SelectiveRefreshRequest = None):
    """
    Refresh only the learning objectives without touching other KB components
    """
    logger.info(f"🔄 [SELECTIVE REFRESH] Starting objectives refresh for topic_id={topic_id}")
    
    db = get_db()
    
    try:
        # Get topic and existing KB
        topic = get_topic_info(topic_id, db)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        # Get existing objectives for context
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT learning_objectives FROM topic_knowledge_base WHERE topic_id = ?
            """, (topic_id,))
            
            existing_kb = cursor.fetchone()
            if not existing_kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found. Create KB first.")
            
            current_objectives = json.loads(dict(existing_kb)["learning_objectives"]) if dict(existing_kb)["learning_objectives"] else []

        # Fetch relevant chunks
        all_chunks = fetch_chunks_for_session(topic["session_id"])
        relevant_chunks = filter_chunks_by_topic(
            all_chunks, topic["keywords"], topic["related_chunk_ids"]
        )
        
        chunks_text = "\n\n---\n\n".join([
            chunk.get("chunk_text", chunk.get("content", ""))
            for chunk in relevant_chunks
        ])

        model_to_use = get_session_model(topic["session_id"])
        
        # Enhanced objectives prompt
        current_obj_text = "\n".join([f"- {o.get('level', 'N/A')}: {o.get('objective', 'N/A')}" for o in current_objectives]) if current_objectives else "Henüz hedef yok"
        
        prompt = f""""{topic['topic_title']}" konusu için SADECE ÖĞRENİM HEDEFLERİNİ Bloom Taksonomisine göre yenile.

MEVCUT HEDEFLERİN:
{current_obj_text}

MATERYAL:
{chunks_text[:10000]}

HEDEFLERİ YENİLEME KURALLARI:
1. Tüm Bloom seviyelerini kapsa (Hatırlama → Yaratma)
2. Her seviyeden en az 1 hedef
3. Ölçülebilir ve spesifik ifadeler
4. TÜRKÇE Bloom seviyeleri kullan
5. Eksik seviyeleri tamamla
6. Belirsiz hedefleri netleştir

BLOOM SEVİYELERİ (MUTLAKA TÜRKÇE): "Hatırlama", "Anlama", "Uygulama", "Analiz", "Değerlendirme", "Yaratma"

ÇIKTI FORMATI (JSON):
{{
  "objectives": [
    {{
      "level": "Hatırlama",
      "objective": "Öğrenci hücre zarının temel bileşenlerini sıralayabilmeli",
      "assessment_method": "çoktan seçmeli test"
    }},
    {{
      "level": "Anlama",
      "objective": "Öğrenci hücre zarının işlevini açıklayabilmeli",
      "assessment_method": "açık uçlu soru"
    }}
  ]
}}

Sadece JSON çıktısı ver."""

        # Generate new objectives
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model_to_use,
                "max_tokens": 1500,
                "temperature": 0.3
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM service error")
        
        result = response.json()
        llm_output = result.get("response", "")
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not json_match:
            raise HTTPException(status_code=500, detail="Failed to parse objectives JSON")
            
        try:
            data = json.loads(json_match.group())
            new_objectives = data.get("objectives", [])
        except json.JSONDecodeError:
            # Try cleanup
            cleaned = re.sub(r",(\s*[}\]])", r"\1", json_match.group())
            data = json.loads(cleaned)
            new_objectives = data.get("objectives", [])

        if not new_objectives:
            raise HTTPException(status_code=500, detail="No objectives extracted")

        # Translate any English Bloom levels to Turkish
        new_objectives = translate_learning_objectives(new_objectives)

        # Update only objectives in database
        with db.get_connection() as conn:
            conn.execute("""
                UPDATE topic_knowledge_base
                SET learning_objectives = ?
                WHERE topic_id = ?
            """, (json.dumps(new_objectives, ensure_ascii=False), topic_id))
            conn.commit()

        logger.info(f"✅ [SELECTIVE REFRESH] Objectives updated for topic {topic_id} ({len(new_objectives)} objectives)")

        return {
            "success": True,
            "topic_id": topic_id,
            "topic_title": topic["topic_title"],
            "component": "objectives",
            "new_objectives": new_objectives,
            "objectives_count": len(new_objectives)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in selective objectives refresh: {e}")
        raise HTTPException(status_code=500, detail=f"Objectives refresh failed: {str(e)}")


@router.post("/refresh-qa/{topic_id}")
async def refresh_qa_pairs_only(topic_id: int, request: SelectiveRefreshRequest = None):
    """
    Refresh only the QA pairs without touching other KB components
    """
    logger.info(f"🔄 [SELECTIVE REFRESH] Starting QA refresh for topic_id={topic_id}")
    
    db = get_db()
    
    try:
        # Get topic info
        topic = get_topic_info(topic_id, db)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        # Check if KB exists
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT knowledge_id FROM topic_knowledge_base WHERE topic_id = ?
            """, (topic_id,))
            
            existing_kb = cursor.fetchone()
            if not existing_kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found. Create KB first.")

        # Delete existing QA pairs
        with db.get_connection() as conn:
            conn.execute("DELETE FROM topic_qa_pairs WHERE topic_id = ?", (topic_id,))
            conn.commit()

        # Generate new QA pairs
        qa_request = QAGenerationRequest(topic_id=topic_id, count=15)
        qa_result = await generate_qa_pairs_endpoint(topic_id, qa_request)

        logger.info(f"✅ [SELECTIVE REFRESH] QA pairs updated for topic {topic_id} ({qa_result.get('count', 0)} pairs)")

        return {
            "success": True,
            "topic_id": topic_id,
            "topic_title": topic["topic_title"],
            "component": "qa",
            "new_qa_pairs_count": qa_result.get("count", 0),
            "qa_pairs": qa_result.get("qa_pairs", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in selective QA refresh: {e}")
        raise HTTPException(status_code=500, detail=f"QA refresh failed: {str(e)}")


@router.post("/refresh-all/{topic_id}")
async def refresh_all_components(topic_id: int, request: SelectiveRefreshRequest = None):
    """
    Refresh all KB components (summary, concepts, objectives, QA pairs).
    This is equivalent to the existing full KB extraction but with better messaging.
    """
    logger.info(f"🔄 [SELECTIVE REFRESH] Starting FULL refresh for topic_id={topic_id}")
    
    # Use the existing extract_knowledge_for_topic function
    extraction_request = KnowledgeExtractionRequest(
        topic_id=topic_id,
        force_refresh=True
    )
    
    result = await extract_knowledge_for_topic(topic_id, extraction_request)
    
    # Also refresh QA pairs
    qa_request = QAGenerationRequest(topic_id=topic_id, count=15)
    qa_result = await generate_qa_pairs_endpoint(topic_id, qa_request)
    
    # Combine results
    result["component"] = "all"
    result["qa_pairs_refreshed"] = qa_result.get("count", 0)
    
    logger.info(f"✅ [SELECTIVE REFRESH] ALL components updated for topic {topic_id}")
    
    return result


@router.post("/qa-embeddings/calculate-batch/{session_id}")
async def calculate_qa_embeddings_batch(
    session_id: str,
    topic_id: Optional[int] = None,
    embedding_model: Optional[str] = None
):
    """
    Calculate and store embeddings for QA pairs that don't have embeddings yet.
    
    Args:
        session_id: Session ID to filter QA pairs
        topic_id: Optional topic ID to limit to specific topic (if None, processes all topics in session)
        embedding_model: Optional embedding model to use (defaults to DEFAULT_EMBEDDING_MODEL)
    
    Returns:
        Dict with success status and statistics
    """
    try:
        db = get_db()
        
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        
        # Get QA pairs without embeddings
        with db.get_connection() as conn:
            if topic_id:
                # First ensure question_embedding column exists (apply migration if needed)
                cursor_check = conn.execute("PRAGMA table_info(topic_qa_pairs)")
                columns = {row[1]: row[2] for row in cursor_check.fetchall()}
                
                if 'question_embedding' not in columns:
                    logger.info("⚠️ question_embedding column not found. Applying migration...")
                    try:
                        # Apply migration directly
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN question_embedding TEXT
                        """)
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_model VARCHAR(100)
                        """)
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_dim INTEGER
                        """)
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_updated_at TIMESTAMP
                        """)
                        conn.execute("""
                            CREATE INDEX IF NOT EXISTS idx_qa_pairs_topic_active 
                            ON topic_qa_pairs(topic_id, is_active)
                        """)
                        conn.execute("""
                            CREATE INDEX IF NOT EXISTS idx_qa_pairs_embedding_model 
                            ON topic_qa_pairs(embedding_model)
                        """)
                        conn.commit()
                        logger.info("✅ question_embedding column added successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to add question_embedding column: {e}")
                        conn.rollback()
                        raise HTTPException(status_code=500, detail=f"Failed to apply migration: {str(e)}")
                
                # Now query with the column
                cursor = conn.execute("""
                    SELECT qa_id, question
                    FROM topic_qa_pairs
                    WHERE topic_id = ? 
                      AND (question_embedding IS NULL OR question_embedding = '')
                    ORDER BY qa_id
                """, (topic_id,))
            else:
                # Get all QA pairs for topics in this session
                # First ensure question_embedding column exists (apply migration if needed)
                cursor_check = conn.execute("PRAGMA table_info(topic_qa_pairs)")
                columns = {row[1]: row[2] for row in cursor_check.fetchall()}
                
                if 'question_embedding' not in columns:
                    logger.info("⚠️ question_embedding column not found. Applying migration...")
                    try:
                        # Apply migration directly
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN question_embedding TEXT
                        """)
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_model VARCHAR(100)
                        """)
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_dim INTEGER
                        """)
                        conn.execute("""
                            ALTER TABLE topic_qa_pairs 
                            ADD COLUMN embedding_updated_at TIMESTAMP
                        """)
                        conn.execute("""
                            CREATE INDEX IF NOT EXISTS idx_qa_pairs_topic_active 
                            ON topic_qa_pairs(topic_id, is_active)
                        """)
                        conn.execute("""
                            CREATE INDEX IF NOT EXISTS idx_qa_pairs_embedding_model 
                            ON topic_qa_pairs(embedding_model)
                        """)
                        conn.commit()
                        logger.info("✅ question_embedding column added successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to add question_embedding column: {e}")
                        conn.rollback()
                        raise HTTPException(status_code=500, detail=f"Failed to apply migration: {str(e)}")
                
                # Now query with the column
                cursor = conn.execute("""
                    SELECT qa.qa_id, qa.question
                    FROM topic_qa_pairs qa
                    INNER JOIN course_topics t ON qa.topic_id = t.topic_id
                    WHERE t.session_id = ?
                      AND (qa.question_embedding IS NULL OR qa.question_embedding = '')
                    ORDER BY qa.qa_id
                """, (session_id,))
            
            qa_pairs = [{"qa_id": row["qa_id"], "question": row["question"]} for row in cursor.fetchall()]
        
        if not qa_pairs:
            return {
                "success": True,
                "message": "No QA pairs found without embeddings",
                "processed": 0,
                "total": 0
            }
        
        logger.info(f"📦 Found {len(qa_pairs)} QA pairs without embeddings. Calculating...")
        
        # Calculate embeddings in batches (process 50 at a time to avoid timeout)
        batch_size = 50
        total_processed = 0
        
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(qa_pairs) + batch_size - 1)//batch_size} ({len(batch)} QA pairs)...")
            
            processed = await calculate_and_store_qa_embeddings_batch(
                qa_pairs=batch,
                embedding_model=embedding_model,
                db=db
            )
            total_processed += processed
        
        return {
            "success": True,
            "message": f"Embeddings calculated for {total_processed}/{len(qa_pairs)} QA pairs",
            "processed": total_processed,
            "total": len(qa_pairs),
            "embedding_model": embedding_model
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating QA embeddings batch: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to calculate embeddings: {str(e)}")

