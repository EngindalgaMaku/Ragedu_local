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
            chunks = response.json().get("chunks", [])
            
            # Normalize chunk IDs - document processing service may return chunk_id in different places
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
                    # Use document_name + chunk_index for stable ID generation
                    doc_name = chunk.get("document_name", "unknown")
                    chunk_idx = chunk.get("chunk_index", i + 1)
                    chunk_id = hash(f"{session_id}_{doc_name}_{chunk_idx}") % 1000000
                    if chunk_id < 0:
                        chunk_id = abs(chunk_id)
                    logger.warning(f"⚠️ [FETCH CHUNKS] Chunk {i} has no ID, generated stable ID: {chunk_id} for {doc_name} chunk {chunk_idx}")
                else:
                    # Convert string IDs to int if possible (for consistency)
                    try:
                        if isinstance(chunk_id, str) and chunk_id.isdigit():
                            chunk_id = int(chunk_id)
                    except (ValueError, AttributeError):
                        pass
                
                # Ensure chunk_id is set in the main dict as int
                chunk["chunk_id"] = chunk_id
                
            logger.info(f"✅ [FETCH CHUNKS] Fetched {len(chunks)} chunks, sample IDs (first 10): {[c.get('chunk_id') for c in chunks[:10]]}")
            return chunks
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
        # Normalize chunk IDs first - ensure every chunk has a valid ID
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or chunk.get("chunkId") or chunk.get("_id")
            if chunk_id is None:
                chunk_id = i + 1  # Use 1-based index as fallback
                chunk["chunk_id"] = chunk_id
            else:
                chunk["chunk_id"] = chunk_id
        
        # Prepare chunks text for LLM with REAL chunk IDs (NO INDEX to avoid confusion!)
        chunks_text = "\n\n---\n\n".join([
            f"[Chunk ID: {chunk.get('chunk_id', i+1)}]\n{chunk.get('chunk_text', chunk.get('content', chunk.get('text', '')))}"
            for i, chunk in enumerate(chunks)
        ])
        
        # ENHANCED PROMPT: Request keywords and related chunks for proper topic-chunk relationships
        # IMPORTANT: Use Chunk ID (not index) in related_chunks field!
        prompt = f"""Bu metinden Türkçe konuları detaylı olarak aşağıdaki JSON formatında çıkar:

{chunks_text[:25000]}

Zorluk seviyeleri: "başlangıç", "orta", "ileri"
Her konu için mutlaka keywords ve ilgili chunk ID'leri belirtin.

ÇOK ÖNEMLİ: "related_chunks" alanında MUTLAKA köşeli parantez içindeki "Chunk ID" değerini kullanın!
Her chunk'ın başında "[Chunk ID: X]" formatında bilgi var. SADECE bu X değerini kullanın!
Örnek: Eğer bir chunk "[Chunk ID: 42]" ile başlıyorsa, related_chunks'a [42] yazmalısınız!

JSON formatı örneği (Chunk ID'leri kullanın!):
{{"topics":[{{"topic_title":"Hücre Yapısı","keywords":["hücre","organeller","membran"],"related_chunks":[42,15,8],"difficulty":"orta"}},{{"topic_title":"DNA ve RNA","keywords":["DNA","RNA","genetik"],"related_chunks":[23,7,91],"difficulty":"ileri"}}]}}

Sadece JSON çıktısı ver:
{{"topics":["""

        # Get session-specific model configuration - KEEP QWEN FOR QUALITY
        model_to_use = get_session_model(session_id) or "llama3:8b"
        logger.info(f"🧠 [TOPIC EXTRACTION] Using model: {model_to_use} for high-quality Turkish extraction")
        
        # Smart truncation for Groq models
        final_prompt = prompt
        if "groq" in model_to_use.lower() or "instant" in model_to_use.lower():
            # Groq model - limit prompt to 15k chars
            if len(prompt) > 18000:
                # Truncate chunks_text portion
                final_prompt = prompt[:18000] + "\n\nSadece JSON çıktısı ver, başka açıklama yapma."
                logger.warning(f"Prompt truncated for Groq model: {len(prompt)} -> 18000 chars")
        
        # Call model inference service with EXTENDED TIMEOUT for quality models
        timeout_seconds = 600 if "qwen" in model_to_use.lower() else 240  # 10 min for qwen, 4 min for others
        logger.info(f"⏰ [TIMEOUT] Using {timeout_seconds}s timeout for {model_to_use}")
        
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": final_prompt,
                "model": model_to_use,
                "max_tokens": 4096,
                "temperature": 0.3
            },
            timeout=timeout_seconds  # Extended timeout for quality models
        )
        
        if response.status_code != 200:
            # Safely extract error message from response
            error_detail = "Unknown error"
            try:
                if hasattr(response, 'text') and response.text:
                    error_detail = response.text[:500]  # Limit to 500 chars
                elif hasattr(response, 'content') and response.content:
                    error_detail = response.content.decode('utf-8', errors='ignore')[:500]
                else:
                    error_detail = f"HTTP {response.status_code} - No response body"
            except Exception as e:
                error_detail = f"HTTP {response.status_code} - Error reading response: {str(e)}"
            
            logger.error(f"❌ [LLM SERVICE ERROR] Status: {response.status_code}, Model: {model_to_use}, URL: {MODEL_INFERENCER_URL}")
            logger.error(f"❌ [LLM SERVICE ERROR] Response: {error_detail}")
            raise HTTPException(
                status_code=500, 
                detail=f"LLM service error (HTTP {response.status_code}): {error_detail}"
            )
        
        result = response.json()
        llm_output = result.get("response", "")
        
        # DIAGNOSTIC LOGGING - Log the raw LLM output to understand the issue
        logger.info(f"🔍 [TOPIC EXTRACTION DEBUG] Raw LLM output length: {len(llm_output)}")
        logger.info(f"🔍 [TOPIC EXTRACTION DEBUG] First 500 chars: {llm_output[:500]}")
        logger.info(f"🔍 [TOPIC EXTRACTION DEBUG] Last 200 chars: {llm_output[-200:] if len(llm_output) > 200 else llm_output}")
        
        # REMOVE markdown check - focus only on JSON extraction
        # LLM sometimes generates markdown but still includes valid JSON
        logger.info(f"📝 [TOPIC EXTRACTION DEBUG] Attempting JSON extraction from LLM output...")
        
        # Parse JSON from LLM output
        import re
        try:
            # First attempt: Extract JSON block using regex
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                logger.info(f"✅ [TOPIC EXTRACTION DEBUG] Found JSON block, length: {len(json_str)}")
                topics_data = json.loads(json_str)
            else:
                # Second attempt: Try to parse entire output as JSON
                logger.info(f"🔄 [TOPIC EXTRACTION DEBUG] No JSON block found, trying to parse entire output as JSON")
                topics_data = json.loads(llm_output.strip())
            
            # Validate the structure
            if not isinstance(topics_data, dict) or "topics" not in topics_data:
                logger.error(f"❌ [TOPIC EXTRACTION DEBUG] Invalid JSON structure: {list(topics_data.keys()) if isinstance(topics_data, dict) else type(topics_data)}")
                raise HTTPException(status_code=500, detail="LLM returned JSON but with invalid structure (missing 'topics' key)")
            
            logger.info(f"✅ [TOPIC EXTRACTION DEBUG] Successfully parsed {len(topics_data.get('topics', []))} topics")
            return topics_data
            
        except json.JSONDecodeError as json_err:
            logger.error(f"❌ [TOPIC EXTRACTION DEBUG] JSON parsing failed: {json_err}")
            logger.error(f"❌ [TOPIC EXTRACTION DEBUG] Attempting to clean JSON...")
            
            # Third attempt: Clean common JSON issues and retry
            try:
                # Remove trailing commas before } or ]
                cleaned_output = re.sub(r',(\s*[}\]])', r'\1', llm_output)
                # Remove markdown code blocks if present
                cleaned_output = re.sub(r'```json\s*', '', cleaned_output)
                cleaned_output = re.sub(r'```\s*$', '', cleaned_output)
                
                json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
                if json_match:
                    topics_data = json.loads(json_match.group())
                    logger.info(f"✅ [TOPIC EXTRACTION DEBUG] Successfully parsed after cleaning")
                    return topics_data
                else:
                    raise json.JSONDecodeError("No valid JSON found after cleaning", cleaned_output, 0)
                    
            except json.JSONDecodeError as final_err:
                logger.error(f"❌ [TOPIC EXTRACTION DEBUG] Final JSON parsing failed: {final_err}")
                
                # ULTRA-AGGRESSIVE JSON REPAIR
                try:
                    logger.info(f"🔧 [TOPIC EXTRACTION DEBUG] ULTRA-AGGRESSIVE JSON repair starting...")
                    
                    # Extract everything between first { and last }
                    repair_text = llm_output.strip()
                    first_brace = repair_text.find('{')
                    last_brace = repair_text.rfind('}')
                    
                    if first_brace >= 0 and last_brace > first_brace:
                        repair_text = repair_text[first_brace:last_brace + 1]
                        
                        # AGGRESSIVE FIXES
                        # 1. Fix incomplete fields (like "prerequisite" -> "prerequisites": [])
                        repair_text = re.sub(r'"prerequisite[^"]*$', '"prerequisites": []', repair_text)
                        repair_text = re.sub(r'"keyword[^"]*$', '"keywords": []', repair_text)
                        repair_text = re.sub(r'"related_chunk[^"]*$', '"related_chunks": []', repair_text)
                        
                        # 2. Fix incomplete strings
                        repair_text = re.sub(r':\s*"[^"]*$', ': ""', repair_text)
                        
                        # 3. Fix missing commas
                        repair_text = re.sub(r'}\s*{', '},{', repair_text)
                        repair_text = re.sub(r']\s*{', '],{', repair_text)
                        repair_text = re.sub(r'}\s*"', '},"', repair_text)
                        repair_text = re.sub(r']\s*"', '],"', repair_text)
                        repair_text = re.sub(r'([0-9])\s*"', r'\1,"', repair_text)
                        
                        # 4. Fix trailing commas
                        repair_text = re.sub(r',(\s*[}\]])', r'\1', repair_text)
                        
                        # 5. Fix arrays with missing commas
                        repair_text = re.sub(r'"\s*"([^"]*")', r'", "\1', repair_text)
                        
                        # 6. Ensure proper JSON ending
                        if not repair_text.endswith(']}'):
                            repair_text = repair_text.rstrip(', \n\r\t') + ']}'
                        
                        # 7. Fix malformed arrays
                        repair_text = re.sub(r'\[\s*,', '[', repair_text)  # Remove leading commas in arrays
                        repair_text = re.sub(r',\s*\]', ']', repair_text)  # Remove trailing commas in arrays
                        
                        logger.info(f"🔧 [TOPIC EXTRACTION DEBUG] Ultra-repaired JSON: {repair_text[:300]}...")
                        
                        # Try parsing
                        data = json.loads(repair_text)
                        logger.info(f"✅ [TOPIC EXTRACTION DEBUG] SUCCESS! Ultra-aggressive repair worked!")
                        
                        if isinstance(data, dict) and "topics" in data:
                            return data
                        else:
                            logger.error(f"❌ [TOPIC EXTRACTION DEBUG] Invalid structure after ultra-repair")
                            raise ValueError("Invalid structure")
                            
                except Exception as ultra_err:
                    logger.error(f"❌ [TOPIC EXTRACTION DEBUG] Ultra-aggressive repair failed: {ultra_err}")
                    logger.error(f"🔧 [TOPIC EXTRACTION DEBUG] Will use fallback construction...")
                    
                    # ULTIMATE FALLBACK: Never fail, always return something
                    try:
                        logger.info(f"🆘 [TOPIC EXTRACTION DEBUG] Ultimate fallback - extracting any text as topics...")
                        
                        # Extract ANY topic-like text from LLM output
                        potential_topics = []
                        
                        # Method 1: Find anything that looks like a topic title
                        title_patterns = [
                            r'"topic_title"\s*:\s*"([^"]+)"',
                            r'\*\*([^*]+)\*\*',  # **Topic**
                            r'#{1,3}\s*(.+)',     # # Topic
                            r'(\d+)\.\s*([^.\n]+)',  # 1. Topic
                            r'-\s*([^-\n]+)',     # - Topic
                        ]
                        
                        for pattern in title_patterns:
                            matches = re.findall(pattern, llm_output, re.MULTILINE)
                            for match in matches:
                                title = match if isinstance(match, str) else match[-1]  # Get last group if tuple
                                title = title.strip()
                                if len(title) > 5 and len(title) < 100:  # Reasonable length
                                    potential_topics.append(title)
                        
                        # Remove duplicates and clean
                        unique_topics = []
                        seen = set()
                        for topic in potential_topics:
                            clean_topic = re.sub(r'[^\w\s]', '', topic).lower()
                            if clean_topic not in seen and len(topic) > 3:
                                unique_topics.append(topic)
                                seen.add(clean_topic)
                                if len(unique_topics) >= 10:  # Max 10 topics
                                    break
                        
                        # If we found some topics, create JSON
                        if unique_topics:
                            fallback_topics = []
                            for i, title in enumerate(unique_topics):
                                fallback_topics.append({
                                    "topic_title": title,
                                    "order": i + 1,
                                    "difficulty": "orta",
                                    "keywords": [title.split()[0] if title.split() else "genel"],
                                    "prerequisites": [],
                                    "subtopics": [],
                                    "related_chunks": []
                                })
                            
                            fallback_data = {"topics": fallback_topics}
                            logger.info(f"🆘 [TOPIC EXTRACTION DEBUG] Ultimate fallback created {len(fallback_topics)} topics from text patterns")
                            return fallback_data
                        
                        # If no topics found, create generic ones
                        logger.info(f"🆘 [TOPIC EXTRACTION DEBUG] No topics found, creating generic topics...")
                        generic_topics = [
                            {
                                "topic_title": "Genel Biyoloji Konuları",
                                "order": 1,
                                "difficulty": "orta",
                                "keywords": ["biyoloji", "genel"],
                                "prerequisites": [],
                                "subtopics": [],
                                "related_chunks": []
                            }
                        ]
                        
                        fallback_data = {"topics": generic_topics}
                        logger.info(f"🆘 [TOPIC EXTRACTION DEBUG] Created generic fallback with {len(generic_topics)} topics")
                        return fallback_data
                        
                    except Exception as ultimate_err:
                        logger.error(f"❌ [TOPIC EXTRACTION DEBUG] Ultimate fallback failed: {ultimate_err}")
                        # NEVER FAIL - return absolute minimum
                        return {
                            "topics": [{
                                "topic_title": "Ders İçeriği",
                                "order": 1,
                                "difficulty": "orta",
                                "keywords": ["ders"],
                                "prerequisites": [],
                                "subtopics": [],
                                "related_chunks": []
                            }]
                        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON output: {e}")
        logger.error(f"LLM output was: {llm_output[:1000]}")  # Log first 1000 chars
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON")
    except requests.exceptions.Timeout as e:
        logger.error(f"⏰ [TIMEOUT ERROR] Request to model service timed out after {timeout_seconds}s")
        logger.error(f"⏰ [TIMEOUT ERROR] Model: {model_to_use}, URL: {MODEL_INFERENCER_URL}")
        logger.error(f"⏰ [TIMEOUT ERROR] Error: {str(e)}")
        raise HTTPException(
            status_code=504, 
            detail=f"Model service timeout after {timeout_seconds}s. The model ({model_to_use}) may be taking too long to respond. Try using a faster model or reducing the input size."
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"🔌 [CONNECTION ERROR] Failed to connect to model service")
        logger.error(f"🔌 [CONNECTION ERROR] URL: {MODEL_INFERENCER_URL}, Model: {model_to_use}")
        logger.error(f"🔌 [CONNECTION ERROR] Error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to model service at {MODEL_INFERENCER_URL}. Please check if the service is running."
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [REQUEST ERROR] Request error in topic extraction: {e}")
        logger.error(f"❌ [REQUEST ERROR] Model: {model_to_use}, URL: {MODEL_INFERENCER_URL}")
        logger.error(f"❌ [REQUEST ERROR] Error type: {type(e).__name__}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to connect to model service: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in topic extraction: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception args: {e.args}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Topic extraction failed: {str(e)}")


def get_session_model(session_id: str) -> str:
    """
    Get the model configured for a specific session from API Gateway.
    Falls back to llama3:8b (Ollama) for unlimited tokens
    
    Args:
        session_id: Session ID
        
    Returns:
        Model name to use for this session
    """
    if not session_id:
        return "llama3:8b"  # Ollama default - no token limits!
        
    try:
        # Get session config from API Gateway
        import os
        api_gateway_url = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
        
        response = requests.get(
            f"{api_gateway_url}/sessions/{session_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            session_data = response.json()
            rag_settings = session_data.get("rag_settings", {})
            
            if rag_settings and rag_settings.get("model"):
                model = rag_settings["model"]
                logger.info(f"Using session model: {model} for session {session_id}")
                return model
        
        # Fallback to Ollama llama3:8b (no token limits)
        logger.info(f"No model configured for session {session_id}, using Ollama llama3:8b")
        return "llama3:8b"
        
    except Exception as e:
        logger.warning(f"Could not get session model for {session_id}: {e}, using Ollama fallback")
        return "llama3:8b"  # Safe fallback - Ollama has no token limits


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
        # Prepare topics list for LLM with robust title extraction
        topics_text = "\n".join([
            f"ID: {t['topic_id']}, Başlık: {t.get('topic_title', t.get('title', t.get('name', 'Başlıksız')))}, Anahtar Kelimeler: {', '.join(t.get('keywords', []))}"
            for t in topics
        ])
        
        prompt = f"""Sen bir biyoloji uzmanısın. Aşağıdaki öğrenci sorusunu verilen konu listesine göre EN DOĞRU şekilde sınıflandır.

ÖĞRENCİ SORUSU:
{question}

MEVCUT KONULAR:
{topics_text}

SINIFLANDIRMA KURALLARI:
1. Sorunun ANA konusunu belirle (ne hakkında?)
2. Anahtar kelimeleri dikkatlice analiz et
3. En uygun konu ID'sini seç
4. Güven skorunu hesapla

ÖRNEK SINIFLANDIRMALAR:
- "kana rengini ne verir" → Kan ile ilgili sorular "Kan Grupları" konusuna gider
- "hücre bölünmesi nasıl olur" → "Hücre Bölünmesi" konusuna gider
- "DNA nedir" → "Genetik" veya "DNA ve RNA" konusuna gider

ÇIKTI FORMATI (JSON):
{{
  "topic_id": 571,
  "topic_title": "Kan Grupları",
  "confidence_score": 0.95,
  "question_complexity": "basic",
  "question_type": "factual",
  "reasoning": "Soru kan renginden bahsediyor, hemoglobin konusu Kan Grupları kategorisine ait"
}}

Sadece JSON çıktısı ver, başka açıklama yapma."""

        # ULTRA-FAST CLASSIFICATION: Force Groq model, no session model lookup at all
        model_to_use = "llama-3.1-8b-instant"  # Hardcoded fast Groq model
        logger.info(f"⚡ [ULTRA-FAST] Using hardcoded model: {model_to_use} (no session lookup for max speed)")
        
        # Call model inference service with REDUCED TIMEOUT
        response = requests.post(
            f"{MODEL_INFERENCER_URL}/models/generate",
            json={
                "prompt": prompt,
                "model": model_to_use,
                "max_tokens": 512,
                "temperature": 0.1  # Lower temperature for more consistent classification
            },
            timeout=15  # Reduced from 60s to 15s for speed
        )
        
        if response.status_code != 200:
            # Safely extract error message from response
            error_detail = "Unknown error"
            try:
                if hasattr(response, 'text') and response.text:
                    error_detail = response.text[:500]  # Limit to 500 chars
                elif hasattr(response, 'content') and response.content:
                    error_detail = response.content.decode('utf-8', errors='ignore')[:500]
                else:
                    error_detail = f"HTTP {response.status_code} - No response body"
            except Exception as e:
                error_detail = f"HTTP {response.status_code} - Error reading response: {str(e)}"
            
            logger.error(f"❌ [LLM SERVICE ERROR] Status: {response.status_code}, Model: {model_to_use}, URL: {MODEL_INFERENCER_URL}")
            logger.error(f"❌ [LLM SERVICE ERROR] Response: {error_detail}")
            raise HTTPException(
                status_code=500, 
                detail=f"LLM service error (HTTP {response.status_code}): {error_detail}"
            )
        
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
        
    except requests.exceptions.Timeout as e:
        logger.error(f"⏰ [TIMEOUT ERROR] Question classification timed out after 15s")
        logger.error(f"⏰ [TIMEOUT ERROR] Model: {model_to_use}, URL: {MODEL_INFERENCER_URL}")
        logger.error(f"⏰ [TIMEOUT ERROR] Error: {str(e)}")
        raise HTTPException(
            status_code=504, 
            detail=f"Question classification timeout. The model ({model_to_use}) may be taking too long to respond."
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"🔌 [CONNECTION ERROR] Failed to connect to model service for question classification")
        logger.error(f"🔌 [CONNECTION ERROR] URL: {MODEL_INFERENCER_URL}, Model: {model_to_use}")
        logger.error(f"🔌 [CONNECTION ERROR] Error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to model service at {MODEL_INFERENCER_URL}. Please check if the service is running."
        )
    except json.JSONDecodeError as e:
        logger.error(f"❌ [JSON ERROR] Failed to parse classification JSON: {e}")
        logger.error(f"❌ [JSON ERROR] LLM output: {llm_output[:500] if 'llm_output' in locals() else 'N/A'}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse classification response: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ [CLASSIFICATION ERROR] Error in question classification: {e}")
        logger.error(f"❌ [CLASSIFICATION ERROR] Exception type: {type(e).__name__}")
        logger.error(f"❌ [CLASSIFICATION ERROR] Exception args: {e.args}")
        import traceback
        logger.error(f"❌ [CLASSIFICATION ERROR] Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Question classification failed: {str(e)}"
        )


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
        
        # Normalize chunk IDs - try multiple field names
        logger.info(f"📦 [TOPIC EXTRACTION] Normalizing chunk IDs from {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            # Try multiple possible ID field names
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or chunk.get("chunkId") or chunk.get("_id")
            if chunk_id is None:
                # If no ID found, use index as ID (0-based, but we'll use 1-based for consistency)
                chunk_id = i + 1
                chunk["chunk_id"] = chunk_id
                logger.warning(f"⚠️ [TOPIC EXTRACTION] Chunk {i} has no ID, using index {chunk_id}")
            else:
                # Ensure chunk_id is set for consistency
                chunk["chunk_id"] = chunk_id
        logger.info(f"✅ [TOPIC EXTRACTION] Normalized chunk IDs: {[c.get('chunk_id') for c in chunks[:5]]}")
        
        # Extract topics using LLM
        start_time = datetime.now()
        topics_data = extract_topics_with_llm(chunks, request.options or {}, request.session_id)
        extraction_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Save topics to database
        saved_topics = []
        
        with db.get_connection() as conn:
            # First, save main topics
            for topic_data in topics_data.get("topics", []):
                # Map LLM's related_chunks (which might be indices or IDs) to actual chunk IDs
                related_chunk_ids = []
                llm_related = topic_data.get("related_chunks", [])
                
                # If LLM returned chunk indices (0-based or 1-based), map them to actual chunk IDs
                for ref in llm_related:
                    if isinstance(ref, int):
                        # Try as 1-based index first
                        if 1 <= ref <= len(chunks):
                            chunk_id = chunks[ref - 1].get("chunk_id")
                            if chunk_id and chunk_id not in related_chunk_ids:
                                related_chunk_ids.append(chunk_id)
                        # Try as 0-based index
                        elif 0 <= ref < len(chunks):
                            chunk_id = chunks[ref].get("chunk_id")
                            if chunk_id and chunk_id not in related_chunk_ids:
                                related_chunk_ids.append(chunk_id)
                    else:
                        # Already an ID, use it directly
                        if ref not in related_chunk_ids:
                            related_chunk_ids.append(ref)
                
                # If no related chunks found, try to match by keywords
                if not related_chunk_ids and topic_data.get("keywords"):
                    keywords = topic_data.get("keywords", [])
                    for chunk in chunks:
                        chunk_text = (chunk.get("chunk_text") or chunk.get("content") or chunk.get("text", "")).lower()
                        # Check if any keyword appears in chunk
                        if any(kw.lower() in chunk_text for kw in keywords):
                            chunk_id = chunk.get("chunk_id")
                            if chunk_id and chunk_id not in related_chunk_ids:
                                related_chunk_ids.append(chunk_id)
                                if len(related_chunk_ids) >= 5:  # Limit to 5 chunks
                                    break
                
                logger.info(f"📝 [TOPIC EXTRACTION] Topic '{topic_data['topic_title']}' mapped to {len(related_chunk_ids)} chunk IDs: {related_chunk_ids[:5]}")
                
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
                    json.dumps(related_chunk_ids),  # Use mapped chunk IDs
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


@router.delete("/{topic_id}")
async def delete_topic(topic_id: int):
    """
    Delete a topic and its related data (cascading deletes handled by database)
    """
    db = get_db()
    
    # Get session_id from topic first to check APRAG status
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT session_id, topic_title FROM course_topics WHERE topic_id = ?", (topic_id,))
        topic = cursor.fetchone()
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        
        topic_dict = dict(topic)
        session_id = topic_dict["session_id"]
        topic_title = topic_dict.get("topic_title", "Unknown")
        
        # Check if APRAG is enabled
        if not FeatureFlags.is_aprag_enabled(session_id):
            raise HTTPException(
                status_code=403,
                detail="APRAG module is disabled. Please enable it from admin settings."
            )
    
    try:
        with db.get_connection() as conn:
            # Delete topic (cascading deletes will handle related records)
            # This will also delete:
            # - Subtopic relationships (parent_topic_id set to NULL)
            # - Knowledge base entries (ON DELETE CASCADE)
            # - QA pairs (ON DELETE CASCADE)
            # - Topic progress (ON DELETE CASCADE)
            # - Question topic mappings (ON DELETE CASCADE)
            cursor = conn.execute("DELETE FROM course_topics WHERE topic_id = ?", (topic_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Topic not found")
            
            conn.commit()
            
            logger.info(f"Topic {topic_id} ('{topic_title}') deleted successfully")
            
            return {
                "success": True,
                "message": f"Topic '{topic_title}' deleted successfully",
                "topic_id": topic_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting topic: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete topic: {str(e)}")


@router.post("/classify-question")
async def classify_question(request: QuestionClassificationRequest):
    """
    Classify a question to a topic
    """
    # CRITICAL DEBUG: Log the incoming request
    logger.info(f"🚨 [CLASSIFY START] Request received: question='{request.question[:50]}...', session_id={request.session_id}, interaction_id={request.interaction_id}")
    
    # Check if APRAG is enabled
    if not FeatureFlags.is_aprag_enabled(request.session_id):
        raise HTTPException(
            status_code=403,
            detail="APRAG module is disabled. Please enable it from admin settings."
        )
    
    db = get_db()
    
    try:
        # CRITICAL DEBUG: Log before database operations
        logger.info(f"🚨 [CLASSIFY START] Starting database operations...")
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
                # DEBUG: Check if interaction exists
                cursor = conn.execute("SELECT user_id, session_id FROM student_interactions WHERE interaction_id = ?", (request.interaction_id,))
                interaction_row = cursor.fetchone()
                
                if not interaction_row:
                    logger.error(f"❌ [CLASSIFY DEBUG] interaction_id {request.interaction_id} not found in student_interactions table")
                    raise HTTPException(status_code=400, detail=f"Invalid interaction_id: {request.interaction_id}")
                
                interaction_data = dict(interaction_row)
                user_id = interaction_data["user_id"]
                interaction_session_id = interaction_data["session_id"]
                
                # CRITICAL FIX: Ensure user_id is always treated as string to avoid FOREIGN KEY issues
                user_id_str = str(user_id) if user_id is not None else None
                
                logger.info(f"🔍 [CLASSIFY DEBUG] Found interaction: ID={request.interaction_id}, user_id={user_id} (type: {type(user_id)}) -> {user_id_str} (str), session_id={interaction_session_id}")
                logger.info(f"🔍 [CLASSIFY DEBUG] Request session_id={request.session_id}, topic_id={classification['topic_id']}")
                
                # Check if session_id matches
                if interaction_session_id != request.session_id:
                    logger.warning(f"⚠️ [CLASSIFY DEBUG] Session ID mismatch: interaction has {interaction_session_id}, request has {request.session_id}")
                
                # Check if topic exists
                cursor = conn.execute("SELECT topic_id FROM course_topics WHERE topic_id = ? AND session_id = ?", (classification["topic_id"], request.session_id))
                if not cursor.fetchone():
                    logger.error(f"❌ [CLASSIFY DEBUG] topic_id {classification['topic_id']} not found in course_topics for session {request.session_id}")
                    raise HTTPException(status_code=400, detail=f"Invalid topic_id: {classification['topic_id']}")
                
                # Insert question mapping
                logger.info(f"💾 [CLASSIFY DEBUG] Inserting question_topic_mapping...")
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
                
                # Update topic progress - use the INTEGER user_id for FOREIGN KEY constraint
                # TEMPORARY FIX FOR DOCKER DATABASE - Remove last_question_timestamp column
                logger.info(f"💾 [CLASSIFY DEBUG] Updating topic_progress for user_id={user_id} (integer)...")
                conn.execute("""
                    INSERT OR REPLACE INTO topic_progress (
                        user_id, session_id, topic_id,
                        questions_asked, updated_at
                    ) VALUES (
                        ?,
                        ?,
                        ?,
                        COALESCE((SELECT questions_asked FROM topic_progress
                                 WHERE user_id = ? AND session_id = ? AND topic_id = ?), 0) + 1,
                        CURRENT_TIMESTAMP
                    )
                """, (
                    user_id,  # Use integer user_id for FOREIGN KEY constraint to users table
                    request.session_id,
                    classification["topic_id"],
                    user_id,  # Use integer user_id in COALESCE subquery too
                    request.session_id,
                    classification["topic_id"]
                ))
                
                conn.commit()
                logger.info(f"✅ [CLASSIFY DEBUG] Successfully saved classification mapping and updated progress")
        
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
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error classifying question: {e}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Question classification failed: {str(e)}")


import threading
import uuid

# Global dict to track extraction jobs
extraction_jobs = {}

def run_extraction_in_background(job_id: str, session_id: str, method: str):
    """
    Run extraction in background thread
    Updates job status in global dict
    """
    db = get_db()
    
    try:
        extraction_jobs[job_id]["status"] = "processing"
        extraction_jobs[job_id]["message"] = "Chunk'lar alınıyor..."
        
        # Get all chunks for session (NO LIMIT!)
        chunks = fetch_chunks_for_session(session_id)
        
        if not chunks:
            extraction_jobs[job_id]["status"] = "failed"
            extraction_jobs[job_id]["error"] = "No chunks found"
            return
        
        extraction_jobs[job_id]["message"] = f"{len(chunks)} chunk bulundu, batch'lere bölünüyor..."
        
        # Normalize chunk IDs - try multiple field names
        logger.info(f"📦 [TOPIC EXTRACTION] Normalizing chunk IDs from {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            # Try multiple possible ID field names
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or chunk.get("chunkId") or chunk.get("_id")
            if chunk_id is None:
                # If no ID found, use a unique ID based on session and index
                # This ensures we have real IDs, not just indices
                chunk_id = hash(f"{session_id}_{i}") % 1000000  # Generate a unique ID
                if chunk_id < 0:
                    chunk_id = abs(chunk_id)
                chunk["chunk_id"] = chunk_id
                logger.warning(f"⚠️ [TOPIC EXTRACTION] Chunk {i} has no ID, generated unique ID: {chunk_id}")
            else:
                # Ensure chunk_id is set for consistency
                chunk["chunk_id"] = chunk_id
        logger.info(f"✅ [TOPIC EXTRACTION] Normalized chunk IDs (first 10): {[c.get('chunk_id') for c in chunks[:10]]}")
        
        if method == "full":
            # Delete existing topics
            with db.get_connection() as conn:
                deleted_count = conn.execute("DELETE FROM course_topics WHERE session_id = ?", (session_id,)).rowcount
                conn.commit()
                logger.info(f"🗑️ [TOPIC EXTRACTION] Deleted {deleted_count} existing topics for session {session_id}")
            
            extraction_jobs[job_id]["message"] = f"Eski konular silindi ({deleted_count} konu), extraction başlıyor..."
            
            # Split chunks into SMALLER batches for reliability
            batches = split_chunks_to_batches(chunks, max_chars=12000)  # Smaller batches for stability
            extraction_jobs[job_id]["total_batches"] = len(batches)
            
            logger.info(f"Split into {len(batches)} batches")
            
            # Extract topics from each batch with INCREMENTAL SAVES
            all_topics = []
            for i, batch in enumerate(batches):
                extraction_jobs[job_id]["current_batch"] = i + 1
                extraction_jobs[job_id]["message"] = f"Batch {i+1}/{len(batches)} işleniyor..."
                
                logger.info(f"🔄 Processing batch {i+1}/{len(batches)} ({len(batch)} chunks)")
                topics_data = extract_topics_with_llm(batch, {"include_subtopics": True}, session_id)
                batch_topics = topics_data.get("topics", [])
                all_topics.extend(batch_topics)
                
                # INCREMENTAL SAVE - Save topics after each batch
                if batch_topics:
                    # Normalize topics first (handle "name" -> "topic_title" etc.)
                    normalized_batch_topics = []
                    current_topic_count = len(all_topics) - len(batch_topics)
                    
                    for j, topic in enumerate(batch_topics):
                        # Use merge function's logic to normalize the topic
                        title = None
                        possible_title_keys = ["topic_title", "title", "name", "topic_name", "başlık", "konu"]
                        
                        for key in possible_title_keys:
                            if key in topic and topic[key]:
                                title = str(topic[key]).strip()
                                break
                        
                        if title:  # Only save if we found a valid title
                            # Map LLM's related_chunks to actual chunk IDs
                            related_chunk_ids = []
                            llm_related = topic.get("related_chunks", topic.get("ilgili_chunklar", []))
                            
                            # Create a map of chunk_id -> chunk for quick lookup
                            chunk_id_map = {chunk.get("chunk_id"): chunk for chunk in chunks if chunk.get("chunk_id")}
                            
                            # Process LLM's related_chunks
                            for ref in llm_related:
                                if isinstance(ref, int):
                                    # First try: ref is a chunk ID (most likely if LLM followed instructions)
                                    if ref in chunk_id_map:
                                        if ref not in related_chunk_ids:
                                            related_chunk_ids.append(ref)
                                    # Second try: ref is a 1-based index
                                    elif 1 <= ref <= len(chunks):
                                        chunk_id = chunks[ref - 1].get("chunk_id")
                                        if chunk_id and chunk_id not in related_chunk_ids:
                                            related_chunk_ids.append(chunk_id)
                                    # Third try: ref is a 0-based index
                                    elif 0 <= ref < len(chunks):
                                        chunk_id = chunks[ref].get("chunk_id")
                                        if chunk_id and chunk_id not in related_chunk_ids:
                                            related_chunk_ids.append(chunk_id)
                                else:
                                    # Already an ID (string or other type), try to convert and use
                                    try:
                                        ref_id = int(ref) if isinstance(ref, str) and ref.isdigit() else ref
                                        if ref_id in chunk_id_map and ref_id not in related_chunk_ids:
                                            related_chunk_ids.append(ref_id)
                                    except (ValueError, TypeError):
                                        pass
                            
                            # If no related chunks found, try to match by keywords
                            if not related_chunk_ids:
                                keywords = topic.get("keywords", topic.get("anahtar_kelimeler", []))
                                if keywords:
                                    for chunk in chunks:
                                        chunk_text = (chunk.get("chunk_text") or chunk.get("content") or chunk.get("text", "")).lower()
                                        # Check if any keyword appears in chunk
                                        if any(kw.lower() in chunk_text for kw in keywords):
                                            chunk_id = chunk.get("chunk_id")
                                            if chunk_id and chunk_id not in related_chunk_ids:
                                                related_chunk_ids.append(chunk_id)
                                                if len(related_chunk_ids) >= 5:  # Limit to 5 chunks
                                                    break
                            
                            logger.info(f"📝 [TOPIC EXTRACTION] Topic '{title}': LLM returned {llm_related}, mapped to chunk IDs: {related_chunk_ids}")
                            
                            normalized_topic = {
                                "topic_title": title,
                                "order": current_topic_count + j + 1,
                                "difficulty": topic.get("difficulty", topic.get("zorluk", topic.get("estimated_difficulty", "orta"))),
                                "keywords": topic.get("keywords", topic.get("anahtar_kelimeler", [])),
                                "prerequisites": topic.get("prerequisites", topic.get("on_koşullar", [])),
                                "subtopics": topic.get("subtopics", topic.get("alt_konular", [])),
                                "related_chunks": related_chunk_ids  # Use mapped chunk IDs
                            }
                            normalized_batch_topics.append(normalized_topic)
                    
                    # Save normalized batch topics to database immediately
                    if normalized_batch_topics:
                        saved_batch_count = save_topics_to_db(normalized_batch_topics, session_id, db)
                        logger.info(f"💾 [INCREMENTAL SAVE] Batch {i+1}: Saved {saved_batch_count} topics to database")
                        extraction_jobs[job_id]["message"] = f"Batch {i+1}/{len(batches)} tamamlandı, {saved_batch_count} konu kaydedildi"
                    else:
                        logger.warning(f"⚠️ [INCREMENTAL SAVE] Batch {i+1}: No valid topics found to save")
            
            extraction_jobs[job_id]["message"] = "Tüm batch'ler tamamlandı! Son kontrolü yapılıyor..."
            
            # Topics already saved incrementally, just get final count
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) as count FROM course_topics WHERE session_id = ?", (session_id,))
                saved_count = dict(cursor.fetchone())["count"]
            
            logger.info(f"✅ [FINAL] Total {saved_count} topics saved incrementally across {len(batches)} batches")
            
            extraction_jobs[job_id]["status"] = "completed"
            extraction_jobs[job_id]["message"] = "Tamamlandı!"
            extraction_jobs[job_id]["result"] = {
                "batches_processed": len(batches),
                "raw_topics_extracted": len(all_topics),
                "saved_topics_count": saved_count,
                "chunks_analyzed": len(chunks)
            }
            
    except HTTPException as http_err:
        # HTTPException in background thread - extract detail properly
        error_msg = str(http_err.detail) if hasattr(http_err, 'detail') and http_err.detail else str(http_err)
        logger.error(f"Background extraction HTTPException: {error_msg}")
        logger.error(f"HTTPException status: {http_err.status_code if hasattr(http_err, 'status_code') else 'N/A'}")
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = error_msg
    except Exception as e:
        # Log full exception details for debugging
        import traceback
        error_msg = str(e)
        logger.error(f"Background extraction error: {error_msg}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception args: {e.args}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = error_msg


@router.post("/re-extract/{session_id}")
async def re_extract_topics_smart(
    session_id: str,
    method: str = "full",  # full, partial, merge
    force_refresh: bool = True
):
    """
    Smart topic re-extraction - ASYNC with job tracking
    
    Returns immediately with job_id
    Client polls /re-extract/status/{job_id} for progress
    """
    
    try:
        # Create job
        job_id = str(uuid.uuid4())
        extraction_jobs[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "method": method,
            "status": "starting",
            "message": "İşlem başlatılıyor...",
            "current_batch": 0,
            "total_batches": 0,
            "result": None,
            "error": None
        }
        
        # Start background thread
        thread = threading.Thread(
            target=run_extraction_in_background,
            args=(job_id, session_id, method),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started background extraction job: {job_id} for session {session_id}")
        
        # Return immediately
        return {
            "success": True,
            "job_id": job_id,
            "session_id": session_id,
            "method": method,
            "message": "Konu çıkarımı arka planda başlatıldı. Lütfen bekleyin...",
            "status_check_url": f"/api/aprag/topics/re-extract/status/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Error starting extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/re-extract/status/{job_id}")
async def get_extraction_status(job_id: str):
    """
    Get status of background extraction job
    """
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = extraction_jobs[job_id]
    return {
        "job_id": job_id,
        "session_id": job["session_id"],
        "status": job["status"],  # starting, processing, completed, failed
        "message": job["message"],
        "current_batch": job["current_batch"],
        "total_batches": job["total_batches"],
        "result": job["result"],
        "error": job["error"]
    }


# Keep old sync implementation for backward compatibility
@router.post("/re-extract-sync/{session_id}")
async def re_extract_topics_sync(
    session_id: str,
    method: str = "full"
):
    """
    Synchronous re-extraction (blocks until complete)
    USE ASYNC VERSION INSTEAD FOR BETTER UX!
    """
    db = get_db()
    
    try:
        # Get all chunks for session (NO LIMIT!)
        chunks = fetch_chunks_for_session(session_id)
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for session {session_id}"
            )
        
        logger.info(f"Re-extracting topics for session {session_id} with method: {method}")
        logger.info(f"Total chunks available: {len(chunks)}")
        
        if method == "full":
            # Delete existing topics
            with db.get_connection() as conn:
                conn.execute("""
                    DELETE FROM course_topics WHERE session_id = ?
                """, (session_id,))
                conn.commit()
            
            logger.info(f"Deleted old topics for session {session_id}")
            
            # Split chunks into batches (12k chars each to stay within LLM limit)
            batches = split_chunks_to_batches(chunks, max_chars=12000)
            logger.info(f"Split into {len(batches)} batches")
            
            # Extract topics from each batch
            all_topics = []
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} chunks)")
                topics_data = extract_topics_with_llm(batch, {"include_subtopics": True}, session_id)
                all_topics.extend(topics_data.get("topics", []))
            
            # Merge similar topics (remove duplicates)
            merged_topics = merge_similar_topics(all_topics)
            logger.info(f"Merged {len(all_topics)} raw topics into {len(merged_topics)} unique topics")
            
            # Re-order topics
            for i, topic in enumerate(merged_topics):
                topic["order"] = i + 1
            
            # Save to database
            saved_count = save_topics_to_db(merged_topics, session_id, db)
            
            return {
                "success": True,
                "method": "full",
                "session_id": session_id,
                "batches_processed": len(batches),
                "raw_topics_extracted": len(all_topics),
                "merged_topics_count": len(merged_topics),
                "saved_topics_count": saved_count,
                "chunks_analyzed": len(chunks)
            }
        
        elif method == "partial":
            # Get existing topics
            with db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT topic_title FROM course_topics
                    WHERE session_id = ? AND is_active = TRUE
                """, (session_id,))
                existing_titles = [dict(row)["topic_title"] for row in cursor.fetchall()]
            
            # Extract topics from all chunks
            batches = split_chunks_to_batches(chunks, max_chars=12000)
            all_topics = []
            for batch in batches:
                topics_data = extract_topics_with_llm(batch, {"include_subtopics": True}, session_id)
                all_topics.extend(topics_data.get("topics", []))
            
            # Filter out existing topics
            new_topics = [t for t in all_topics if t["topic_title"] not in existing_titles]
            merged_new_topics = merge_similar_topics(new_topics)
            
            # Get max order from existing topics
            with db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT MAX(topic_order) as max_order FROM course_topics
                    WHERE session_id = ?
                """, (session_id,))
                max_order = dict(cursor.fetchone())["max_order"] or 0
            
            # Set order for new topics
            for i, topic in enumerate(merged_new_topics):
                topic["order"] = max_order + i + 1
            
            # Save new topics
            saved_count = save_topics_to_db(merged_new_topics, session_id, db)
            
            return {
                "success": True,
                "method": "partial",
                "session_id": session_id,
                "existing_topics_count": len(existing_titles),
                "new_topics_added": saved_count,
                "chunks_analyzed": len(chunks)
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in smart re-extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Smart re-extraction failed: {str(e)}")


def split_chunks_to_batches(chunks: List[Dict], max_chars: int = 25000) -> List[List[Dict]]:
    """
    Split chunks into batches based on character limit
    
    FLEXIBLE: 25k chars per batch
    - Ollama models: No token limit, can handle large batches
    - Groq models: We'll use smart truncation (first 15k if needed)
    
    This reduces batch count significantly when using Ollama!
    """
    batches = []
    current_batch = []
    current_chars = 0
    
    for chunk in chunks:
        chunk_text = chunk.get('chunk_text', chunk.get('content', chunk.get('text', '')))
        chunk_length = len(chunk_text)
        
        if current_chars + chunk_length > max_chars and current_batch:
            # Start new batch
            batches.append(current_batch)
            current_batch = [chunk]
            current_chars = chunk_length
        else:
            current_batch.append(chunk)
            current_chars += chunk_length
    
    # Add last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


def merge_similar_topics(topics: List[Dict]) -> List[Dict]:
    """
    Merge similar/duplicate topics based on title similarity
    NEVER FAIL: Handle various topic key formats gracefully
    """
    if not topics:
        return []
    
    merged = []
    seen_titles = set()
    
    for topic in topics:
        # ROBUST TITLE EXTRACTION - Handle various key formats
        title = None
        possible_title_keys = ["topic_title", "title", "name", "topic_name", "başlık", "konu"]
        
        for key in possible_title_keys:
            if key in topic and topic[key]:
                title = str(topic[key]).strip()
                break
        
        # If no title found, skip this topic
        if not title:
            logger.warning(f"🔧 [MERGE TOPICS] Skipping topic with no valid title: {list(topic.keys())}")
            continue
        
        title_lower = title.lower()
        
        # Check for exact match
        if title_lower in seen_titles:
            logger.debug(f"🔧 [MERGE TOPICS] Skipping duplicate exact match: {title}")
            continue
        
        # Check for similar titles (simple approach)
        is_duplicate = False
        for existing_title in seen_titles:
            # If 70%+ of words are same, consider duplicate
            words1 = set(title_lower.split())
            words2 = set(existing_title.split())
            if len(words1 & words2) / max(len(words1), len(words2)) > 0.7:
                logger.debug(f"🔧 [MERGE TOPICS] Skipping similar duplicate: {title} (similar to: {existing_title})")
                is_duplicate = True
                break
        
        if not is_duplicate:
            # NORMALIZE THE TOPIC - Ensure consistent format
            normalized_topic = {
                "topic_title": title,
                "order": topic.get("order", topic.get("topic_order", 0)),
                "difficulty": topic.get("difficulty", topic.get("zorluk", topic.get("estimated_difficulty", "orta"))),
                "keywords": topic.get("keywords", topic.get("anahtar_kelimeler", [])),
                "prerequisites": topic.get("prerequisites", topic.get("on_koşullar", [])),
                "subtopics": topic.get("subtopics", topic.get("alt_konular", [])),
                "related_chunks": topic.get("related_chunks", topic.get("ilgili_chunklar", []))
            }
            
            merged.append(normalized_topic)
            seen_titles.add(title_lower)
            logger.debug(f"✅ [MERGE TOPICS] Added unique topic: {title}")
    
    logger.info(f"🔧 [MERGE TOPICS] Merged {len(topics)} input topics into {len(merged)} unique topics")
    return merged


def save_topics_to_db(topics: List[Dict], session_id: str, db: DatabaseManager) -> int:
    """
    Save topics to database with proper Turkish difficulty level handling
    Returns count of saved topics
    """
    
    def normalize_difficulty(difficulty: str) -> str:
        """Convert Turkish difficulty levels to English database values"""
        turkish_to_english = {
            "başlangıç": "beginner",
            "baslangic": "beginner",  # Without Turkish characters
            "temel": "beginner",
            "orta": "intermediate",
            "ileri": "advanced",
            "gelişmiş": "advanced",
            "gelismis": "advanced",  # Without Turkish characters
            "zor": "advanced"
        }
        
        # Normalize input
        normalized_input = difficulty.lower().strip()
        
        # Return mapped value or default to intermediate
        return turkish_to_english.get(normalized_input, difficulty.lower() if difficulty.lower() in ["beginner", "intermediate", "advanced"] else "intermediate")
    
    saved_count = 0
    
    with db.get_connection() as conn:
        for topic_data in topics:
            # ROBUST TITLE EXTRACTION
            title = None
            possible_title_keys = ["topic_title", "title", "name", "topic_name", "başlık", "konu"]
            
            for key in possible_title_keys:
                if key in topic_data and topic_data[key]:
                    title = str(topic_data[key]).strip()
                    break
            
            # Skip if no valid title found
            if not title:
                logger.warning(f"💾 [TOPIC SAVE] Skipping topic with no valid title: {list(topic_data.keys())}")
                continue
            
            # Normalize difficulty level
            original_difficulty = topic_data.get("difficulty", "intermediate")
            normalized_difficulty = normalize_difficulty(original_difficulty)
            
            logger.info(f"💾 [TOPIC SAVE] Saving topic: {title} (difficulty: {original_difficulty} -> {normalized_difficulty})")
            
            # Insert main topic
            cursor = conn.execute("""
                INSERT INTO course_topics (
                    session_id, topic_title, parent_topic_id, topic_order,
                    description, keywords, estimated_difficulty,
                    prerequisites, related_chunk_ids,
                    extraction_method, extraction_confidence, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                title,
                None,  # parent_topic_id
                topic_data.get("order", 0),
                topic_data.get("description", None),
                json.dumps(topic_data.get("keywords", []), ensure_ascii=False),
                normalized_difficulty,  # Use normalized difficulty
                json.dumps(topic_data.get("prerequisites", []), ensure_ascii=False),
                json.dumps(topic_data.get("related_chunks", []), ensure_ascii=False),
                "llm_analysis",
                0.75,
                True
            ))
            
            main_topic_id = cursor.lastrowid
            saved_count += 1
            
            # Save subtopics with robust title extraction
            for subtopic_data in topic_data.get("subtopics", []):
                # ROBUST SUBTOPIC TITLE EXTRACTION
                subtopic_title = None
                possible_title_keys = ["topic_title", "title", "name", "topic_name", "başlık", "konu"]
                
                for key in possible_title_keys:
                    if key in subtopic_data and subtopic_data[key]:
                        subtopic_title = str(subtopic_data[key]).strip()
                        break
                
                # Skip if no valid subtopic title found
                if not subtopic_title:
                    logger.warning(f"💾 [TOPIC SAVE] Skipping subtopic with no valid title: {list(subtopic_data.keys())}")
                    continue
                
                conn.execute("""
                    INSERT INTO course_topics (
                        session_id, topic_title, parent_topic_id, topic_order,
                        keywords, extraction_method, extraction_confidence, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    subtopic_title,
                    main_topic_id,
                    subtopic_data.get("order", 0),
                    json.dumps(subtopic_data.get("keywords", []), ensure_ascii=False),
                    "llm_analysis",
                    0.75,
                    True
                ))
                saved_count += 1
        
        conn.commit()
    
    return saved_count


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
            # Note: Using existing schema columns and mapping to expected response format
            cursor = conn.execute("""
                SELECT 
                    t.topic_id,
                    t.topic_title,
                    t.topic_order,
                    COALESCE(p.questions_asked, 0) as questions_asked,
                    COALESCE(p.understanding_level, 0.0) as average_understanding,
                    CASE 
                        WHEN p.status = 'mastered' THEN 'mastered'
                        WHEN p.status = 'learning' THEN 'learning'
                        WHEN p.status = 'not_started' THEN 'not_started'
                        WHEN p.completion_percentage >= 80 THEN 'mastered'
                        WHEN p.completion_percentage >= 50 THEN 'learning'
                        WHEN p.questions_asked > 0 THEN 'learning'
                        ELSE 'not_started'
                    END as mastery_level,
                    COALESCE(p.completion_percentage / 100.0, 0.0) as mastery_score,
                    CASE 
                        WHEN p.completion_percentage >= 80 THEN 1
                        ELSE 0
                    END as is_ready_for_next,
                    COALESCE(p.completion_percentage / 100.0, 0.0) as readiness_score,
                    COALESCE(CAST(p.time_spent_seconds AS REAL) / 60.0, 0.0) as time_spent_minutes
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
                
                # Determine current topic:
                # 1. First topic with questions but not mastered
                # 2. If no questions asked yet, use first topic in order that has questions_asked > 0
                # 3. If no topics have questions, use first topic in order
                if current_topic is None:
                    questions = topic_progress.get("questions_asked", 0) or 0
                    mastery = topic_progress.get("mastery_level", "not_started")
                    if questions > 0 and mastery != "mastered":
                        current_topic = topic_progress
                    elif questions == 0 and not current_topic:
                        # Use first topic (by order) as current if no progress yet
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
            # Safely extract error message from response
            error_detail = "Unknown error"
            try:
                if hasattr(response, 'text') and response.text:
                    error_detail = response.text[:500]  # Limit to 500 chars
                elif hasattr(response, 'content') and response.content:
                    error_detail = response.content.decode('utf-8', errors='ignore')[:500]
                else:
                    error_detail = f"HTTP {response.status_code} - No response body"
            except Exception as e:
                error_detail = f"HTTP {response.status_code} - Error reading response: {str(e)}"
            
            logger.error(f"❌ [LLM SERVICE ERROR] Status: {response.status_code}, Model: {model_to_use}, URL: {MODEL_INFERENCER_URL}")
            logger.error(f"❌ [LLM SERVICE ERROR] Response: {error_detail}")
            raise HTTPException(
                status_code=500, 
                detail=f"LLM service error (HTTP {response.status_code}): {error_detail}"
            )
        
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

