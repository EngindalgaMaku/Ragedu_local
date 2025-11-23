"""
Session Backup and Restore System
Allows full backup and restore of session data including:
- Session metadata
- Document chunks
- Topics
- Knowledge base
- QA pairs
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["backup-restore"])

# Service URLs
DOCUMENT_PROCESSING_URL = os.getenv(
    'DOCUMENT_PROCESSOR_URL',
    f"http://{os.getenv('DOCUMENT_PROCESSOR_HOST', 'document-processing-service')}:{os.getenv('DOCUMENT_PROCESSOR_PORT', '8080')}"
)
APRAG_SERVICE_URL = os.getenv(
    'APRAG_SERVICE_URL',
    f"http://{os.getenv('APRAG_SERVICE_HOST', 'aprag-service')}:{os.getenv('APRAG_SERVICE_PORT', '8001')}"
)

# Database imports - Use direct SQLite connections
# In Docker, services are separate containers, so we connect to shared database file
import sqlite3
import os
from contextlib import contextmanager

# Get database path from environment or use default
APRAG_DB_PATH = os.getenv("APRAG_DB_PATH", os.getenv("DATABASE_PATH", "data/rag_assistant.db"))

@contextmanager
def get_aprag_db_connection():
    """Get APRAG database connection"""
    conn = sqlite3.connect(APRAG_DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class BackupResponse(BaseModel):
    success: bool
    session_id: str
    backup_id: str
    backup_timestamp: str
    data: Dict[str, Any]
    message: str


class RestoreRequest(BaseModel):
    backup_data: Dict[str, Any]
    new_session_id: Optional[str] = None  # If None, use original session_id
    restore_chunks: bool = True
    restore_topics: bool = True
    restore_kb: bool = True
    restore_qa: bool = True


@router.get("/{session_id}/backup")
async def backup_session(session_id: str) -> BackupResponse:
    """
    Create a complete backup of a session including all related data.
    
    Returns:
    - Session metadata
    - All document chunks
    - All topics
    - All knowledge bases
    - All QA pairs
    """
    try:
        logger.info(f"🔄 Starting backup for session: {session_id}")
        backup_id = f"backup_{session_id}_{int(datetime.now().timestamp())}"
        backup_data = {
            "backup_id": backup_id,
            "backup_timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "version": "1.0",
            "data": {}
        }
        
        # 1. Get session metadata from document processing service
        logger.info("📋 Fetching session metadata...")
        try:
            session_response = requests.get(
                f"{DOCUMENT_PROCESSING_URL}/sessions/{session_id}",
                timeout=30
            )
            if session_response.status_code == 200:
                backup_data["data"]["session_metadata"] = session_response.json()
            else:
                logger.warning(f"Could not fetch session metadata: {session_response.status_code}")
                backup_data["data"]["session_metadata"] = None
        except Exception as e:
            logger.error(f"Error fetching session metadata: {e}")
            backup_data["data"]["session_metadata"] = None
        
        # 2. Get all chunks WITH EMBEDDINGS from document processing service
        logger.info("📄 Fetching document chunks with embeddings...")
        try:
            # First get chunks normally
            chunks_response = requests.get(
                f"{DOCUMENT_PROCESSING_URL}/sessions/{session_id}/chunks",
                timeout=60
            )
            if chunks_response.status_code == 200:
                chunks_data = chunks_response.json()
                chunks = chunks_data.get("chunks", [])
                
                # Now get embeddings from ChromaDB directly via document processing service
                # We'll need to add an endpoint for this, or we can try to get them via the collection
                # For now, we'll store chunks with their metadata and try to restore embeddings later
                # If embeddings are not available, they can be re-generated during restore
                
                backup_data["data"]["chunks"] = chunks
                backup_data["data"]["chunks_with_embeddings"] = []  # Will be populated if available
                
                # Try to get embeddings via a special endpoint (we'll create this)
                try:
                    embeddings_response = requests.get(
                        f"{DOCUMENT_PROCESSING_URL}/sessions/{session_id}/chunks-with-embeddings",
                        timeout=120
                    )
                    if embeddings_response.status_code == 200:
                        embeddings_data = embeddings_response.json()
                        backup_data["data"]["chunks_with_embeddings"] = embeddings_data.get("chunks", [])
                        logger.info(f"✅ Fetched {len(backup_data['data']['chunks_with_embeddings'])} chunks with embeddings")
                    else:
                        logger.warning("Embeddings endpoint not available, will regenerate during restore")
                except Exception as e:
                    logger.warning(f"Could not fetch embeddings (will regenerate): {e}")
                
                logger.info(f"✅ Fetched {len(chunks)} chunks")
            else:
                logger.warning(f"Could not fetch chunks: {chunks_response.status_code}")
                backup_data["data"]["chunks"] = []
                backup_data["data"]["chunks_with_embeddings"] = []
        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
            backup_data["data"]["chunks"] = []
            backup_data["data"]["chunks_with_embeddings"] = []
        
        # 3. Get all topics from APRAG service via API
        logger.info("📚 Fetching topics...")
        try:
            topics_response = requests.get(
                f"{APRAG_SERVICE_URL}/api/aprag/topics?session_id={session_id}",
                timeout=30
            )
            if topics_response.status_code == 200:
                topics_data = topics_response.json()
                topics = topics_data.get("topics", [])
                backup_data["data"]["topics"] = topics
                logger.info(f"✅ Fetched {len(topics)} topics")
            else:
                logger.warning(f"Could not fetch topics: {topics_response.status_code}")
                backup_data["data"]["topics"] = []
        except Exception as e:
            logger.error(f"Error fetching topics: {e}")
            backup_data["data"]["topics"] = []
        
        # 4. Get all knowledge bases for topics via APRAG API
        logger.info("🧠 Fetching knowledge bases...")
        try:
            topic_ids = [t.get("topic_id") for t in backup_data["data"]["topics"]]
            knowledge_bases = []
            
            if topic_ids:
                # Fetch KB for each topic
                for topic_id in topic_ids:
                    try:
                        kb_response = requests.get(
                            f"{APRAG_SERVICE_URL}/api/aprag/knowledge/kb/{topic_id}",
                            timeout=30
                        )
                        if kb_response.status_code == 200:
                            kb_data = kb_response.json()
                            if kb_data.get("success") and kb_data.get("knowledge_base"):
                                knowledge_bases.append(kb_data["knowledge_base"])
                    except Exception as e:
                        logger.warning(f"Could not fetch KB for topic {topic_id}: {e}")
                        continue
                
                backup_data["data"]["knowledge_bases"] = knowledge_bases
                logger.info(f"✅ Fetched {len(knowledge_bases)} knowledge bases")
            else:
                backup_data["data"]["knowledge_bases"] = []
        except Exception as e:
            logger.error(f"Error fetching knowledge bases: {e}")
            backup_data["data"]["knowledge_bases"] = []
        
        # 5. Get all QA pairs for topics via APRAG API
        logger.info("❓ Fetching QA pairs...")
        try:
            topic_ids = [t.get("topic_id") for t in backup_data["data"]["topics"]]
            qa_pairs = []
            
            if topic_ids:
                # Fetch QA pairs for each topic
                for topic_id in topic_ids:
                    try:
                        qa_response = requests.get(
                            f"{APRAG_SERVICE_URL}/api/aprag/knowledge/kb/{topic_id}",
                            timeout=30
                        )
                        if qa_response.status_code == 200:
                            qa_data = qa_response.json()
                            if qa_data.get("success") and qa_data.get("knowledge_base"):
                                kb = qa_data["knowledge_base"]
                                # Extract QA pairs from KB response
                                if isinstance(kb.get("qa_pairs"), list):
                                    for qa in kb["qa_pairs"]:
                                        qa["topic_id"] = topic_id
                                        qa_pairs.append(qa)
                    except Exception as e:
                        logger.warning(f"Could not fetch QA pairs for topic {topic_id}: {e}")
                        continue
                
                backup_data["data"]["qa_pairs"] = qa_pairs
                logger.info(f"✅ Fetched {len(qa_pairs)} QA pairs")
            else:
                backup_data["data"]["qa_pairs"] = []
        except Exception as e:
            logger.error(f"Error fetching QA pairs: {e}")
            backup_data["data"]["qa_pairs"] = []
        
        # Summary
        chunks_count = len(backup_data["data"].get("chunks", []))
        chunks_with_embeddings_count = len(backup_data["data"].get("chunks_with_embeddings", []))
        summary = {
            "chunks_count": chunks_count,
            "chunks_with_embeddings_count": chunks_with_embeddings_count,
            "topics_count": len(backup_data["data"].get("topics", [])),
            "knowledge_bases_count": len(backup_data["data"].get("knowledge_bases", [])),
            "qa_pairs_count": len(backup_data["data"].get("qa_pairs", []))
        }
        backup_data["summary"] = summary
        
        logger.info(f"✅ Backup completed: {summary}")
        
        embedding_status = f" ({chunks_with_embeddings_count} with embeddings)" if chunks_with_embeddings_count > 0 else " (embeddings will be regenerated)"
        return BackupResponse(
            success=True,
            session_id=session_id,
            backup_id=backup_id,
            backup_timestamp=backup_data["backup_timestamp"],
            data=backup_data,
            message=f"Backup completed: {summary['chunks_count']} chunks{embedding_status}, {summary['topics_count']} topics, {summary['knowledge_bases_count']} KBs, {summary['qa_pairs_count']} QA pairs"
        )
        
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")


@router.post("/restore")
async def restore_session(request: RestoreRequest) -> Dict[str, Any]:
    """
    Restore a session from backup data.
    
    Can restore to:
    - Original session_id (if session was deleted)
    - New session_id (if you want to duplicate)
    """
    try:
        backup_data = request.backup_data
        original_session_id = backup_data.get("session_id")
        target_session_id = request.new_session_id or original_session_id
        
        logger.info(f"🔄 Starting restore: {original_session_id} -> {target_session_id}")
        
        restore_summary = {
            "chunks_restored": 0,
            "topics_restored": 0,
            "knowledge_bases_restored": 0,
            "qa_pairs_restored": 0,
            "errors": []
        }
        
        session_data = backup_data.get("data", {})
        
        # 1. Restore chunks WITH EMBEDDINGS to document processing service
        if request.restore_chunks:
            chunks_to_restore = session_data.get("chunks_with_embeddings") or session_data.get("chunks", [])
            if chunks_to_restore:
                logger.info(f"📄 Restoring {len(chunks_to_restore)} chunks with embeddings...")
                try:
                    # Use special restore endpoint in document processing service
                    restore_response = requests.post(
                        f"{DOCUMENT_PROCESSING_URL}/sessions/{target_session_id}/restore-chunks",
                        json={
                            "chunks": chunks_to_restore,
                            "session_id": target_session_id,
                            "original_session_id": original_session_id
                        },
                        timeout=300  # 5 minutes for large sessions
                    )
                    
                    if restore_response.status_code == 200:
                        result = restore_response.json()
                        restore_summary["chunks_restored"] = result.get("chunks_restored", 0)
                        logger.info(f"✅ Restored {restore_summary['chunks_restored']} chunks")
                    else:
                        error_msg = restore_response.text
                        logger.error(f"Chunk restoration failed: {restore_response.status_code} - {error_msg}")
                        restore_summary["errors"].append(f"Chunk restoration failed: {error_msg}")
                except Exception as e:
                    logger.error(f"Error restoring chunks: {e}", exc_info=True)
                    restore_summary["errors"].append(f"Chunk restoration error: {str(e)}")
            else:
                logger.warning("⚠️ No chunks found in backup data")
        
        # 2. Restore topics
        if request.restore_topics and session_data.get("topics"):
            logger.info(f"📚 Restoring {len(session_data['topics'])} topics...")
            try:
                with get_aprag_db_connection() as conn:
                    # Create a mapping from old topic_id to new topic_id
                    topic_id_mapping = {}
                    
                    for topic in session_data["topics"]:
                        old_topic_id = topic.get("topic_id")
                        
                        # Prepare topic data for insertion
                        topic_data = {
                            "session_id": target_session_id,
                            "topic_title": topic.get("topic_title"),
                            "parent_topic_id": None,  # Will update after all topics are inserted
                            "topic_order": topic.get("topic_order"),
                            "description": topic.get("description"),
                            "keywords": json.dumps(topic.get("keywords", [])) if isinstance(topic.get("keywords"), list) else topic.get("keywords"),
                            "estimated_difficulty": topic.get("estimated_difficulty"),
                            "estimated_time_minutes": topic.get("estimated_time_minutes"),
                            "prerequisites": json.dumps(topic.get("prerequisites", [])) if isinstance(topic.get("prerequisites"), list) else topic.get("prerequisites"),
                            "related_chunk_ids": json.dumps(topic.get("related_chunk_ids", [])) if isinstance(topic.get("related_chunk_ids"), list) else topic.get("related_chunk_ids"),
                            "extraction_method": topic.get("extraction_method"),
                            "extraction_confidence": topic.get("extraction_confidence"),
                            "is_active": topic.get("is_active", True)
                        }
                        
                        cursor = conn.execute("""
                            INSERT INTO course_topics (
                                session_id, topic_title, parent_topic_id, topic_order,
                                description, keywords, estimated_difficulty, estimated_time_minutes,
                                prerequisites, related_chunk_ids, extraction_method,
                                extraction_confidence, is_active
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            topic_data["session_id"],
                            topic_data["topic_title"],
                            topic_data["parent_topic_id"],
                            topic_data["topic_order"],
                            topic_data["description"],
                            topic_data["keywords"],
                            topic_data["estimated_difficulty"],
                            topic_data["estimated_time_minutes"],
                            topic_data["prerequisites"],
                            topic_data["related_chunk_ids"],
                            topic_data["extraction_method"],
                            topic_data["extraction_confidence"],
                            topic_data["is_active"]
                        ))
                        
                        new_topic_id = cursor.lastrowid
                        topic_id_mapping[old_topic_id] = new_topic_id
                        restore_summary["topics_restored"] += 1
                    
                    # Update parent_topic_id references
                    for topic in session_data["topics"]:
                        old_topic_id = topic.get("topic_id")
                        old_parent_id = topic.get("parent_topic_id")
                        
                        if old_parent_id and old_parent_id in topic_id_mapping:
                            new_topic_id = topic_id_mapping[old_topic_id]
                            new_parent_id = topic_id_mapping[old_parent_id]
                            
                            conn.execute("""
                                UPDATE course_topics
                                SET parent_topic_id = ?
                                WHERE topic_id = ?
                            """, (new_parent_id, new_topic_id))
                    
                    conn.commit()
                    logger.info(f"✅ Restored {restore_summary['topics_restored']} topics")
            except Exception as e:
                logger.error(f"Error restoring topics: {e}", exc_info=True)
                restore_summary["errors"].append(f"Topic restoration error: {str(e)}")
        
        # 3. Restore knowledge bases
        if request.restore_kb and session_data.get("knowledge_bases") and "topic_id_mapping" in locals():
            logger.info(f"🧠 Restoring {len(session_data['knowledge_bases'])} knowledge bases...")
            try:
                with get_aprag_db_connection() as conn:
                    for kb in session_data["knowledge_bases"]:
                        old_topic_id = kb.get("topic_id")
                        if old_topic_id not in topic_id_mapping:
                            continue
                        
                        new_topic_id = topic_id_mapping[old_topic_id]
                        
                        kb_data = {
                            "topic_id": new_topic_id,
                            "topic_summary": kb.get("topic_summary"),
                            "key_concepts": json.dumps(kb.get("key_concepts", [])) if isinstance(kb.get("key_concepts"), (list, dict)) else kb.get("key_concepts"),
                            "learning_objectives": json.dumps(kb.get("learning_objectives", [])) if isinstance(kb.get("learning_objectives"), (list, dict)) else kb.get("learning_objectives"),
                            "definitions": json.dumps(kb.get("definitions", {})) if isinstance(kb.get("definitions"), dict) else kb.get("definitions"),
                            "formulas": json.dumps(kb.get("formulas", [])) if isinstance(kb.get("formulas"), list) else kb.get("formulas"),
                            "examples": json.dumps(kb.get("examples", [])) if isinstance(kb.get("examples"), list) else kb.get("examples"),
                            "related_topics": json.dumps(kb.get("related_topics", [])) if isinstance(kb.get("related_topics"), list) else kb.get("related_topics"),
                            "prerequisite_concepts": json.dumps(kb.get("prerequisite_concepts", [])) if isinstance(kb.get("prerequisite_concepts"), list) else kb.get("prerequisite_concepts"),
                            "real_world_applications": json.dumps(kb.get("real_world_applications", [])) if isinstance(kb.get("real_world_applications"), list) else kb.get("real_world_applications"),
                            "common_misconceptions": json.dumps(kb.get("common_misconceptions", [])) if isinstance(kb.get("common_misconceptions"), list) else kb.get("common_misconceptions"),
                            "content_quality_score": kb.get("content_quality_score"),
                            "extraction_model": kb.get("extraction_model"),
                            "is_validated": kb.get("is_validated", False),
                            "view_count": kb.get("view_count", 0)
                        }
                        
                        conn.execute("""
                            INSERT INTO topic_knowledge_base (
                                topic_id, topic_summary, key_concepts, learning_objectives,
                                definitions, formulas, examples, related_topics,
                                prerequisite_concepts, real_world_applications, common_misconceptions,
                                content_quality_score, extraction_model, is_validated, view_count
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            kb_data["topic_id"],
                            kb_data["topic_summary"],
                            kb_data["key_concepts"],
                            kb_data["learning_objectives"],
                            kb_data["definitions"],
                            kb_data["formulas"],
                            kb_data["examples"],
                            kb_data["related_topics"],
                            kb_data["prerequisite_concepts"],
                            kb_data["real_world_applications"],
                            kb_data["common_misconceptions"],
                            kb_data["content_quality_score"],
                            kb_data["extraction_model"],
                            kb_data["is_validated"],
                            kb_data["view_count"]
                        ))
                        
                        restore_summary["knowledge_bases_restored"] += 1
                    
                    conn.commit()
                    logger.info(f"✅ Restored {restore_summary['knowledge_bases_restored']} knowledge bases")
            except Exception as e:
                logger.error(f"Error restoring knowledge bases: {e}", exc_info=True)
                restore_summary["errors"].append(f"KB restoration error: {str(e)}")
        
        # 4. Restore QA pairs
        if request.restore_qa and session_data.get("qa_pairs") and "topic_id_mapping" in locals():
            logger.info(f"❓ Restoring {len(session_data['qa_pairs'])} QA pairs...")
            try:
                with get_aprag_db_connection() as conn:
                    for qa in session_data["qa_pairs"]:
                        old_topic_id = qa.get("topic_id")
                        if old_topic_id not in topic_id_mapping:
                            continue
                        
                        new_topic_id = topic_id_mapping[old_topic_id]
                        
                        qa_data = {
                            "topic_id": new_topic_id,
                            "question": qa.get("question"),
                            "answer": qa.get("answer"),
                            "explanation": qa.get("explanation"),
                            "difficulty_level": qa.get("difficulty_level"),
                            "question_type": qa.get("question_type"),
                            "bloom_taxonomy_level": qa.get("bloom_taxonomy_level"),
                            "source_chunk_ids": json.dumps(qa.get("source_chunk_ids", [])) if isinstance(qa.get("source_chunk_ids"), list) else qa.get("source_chunk_ids"),
                            "extraction_method": qa.get("extraction_method", "llm_generated"),
                            "extraction_model": qa.get("extraction_model"),
                            "quality_score": qa.get("quality_score"),
                            "is_active": qa.get("is_active", True),
                            "related_concepts": json.dumps(qa.get("related_concepts", [])) if isinstance(qa.get("related_concepts"), list) else qa.get("related_concepts")
                        }
                        
                        conn.execute("""
                            INSERT INTO topic_qa_pairs (
                                topic_id, question, answer, explanation,
                                difficulty_level, question_type, bloom_taxonomy_level,
                                source_chunk_ids, extraction_method, extraction_model,
                                quality_score, is_active, related_concepts
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            qa_data["topic_id"],
                            qa_data["question"],
                            qa_data["answer"],
                            qa_data["explanation"],
                            qa_data["difficulty_level"],
                            qa_data["question_type"],
                            qa_data["bloom_taxonomy_level"],
                            qa_data["source_chunk_ids"],
                            qa_data["extraction_method"],
                            qa_data["extraction_model"],
                            qa_data["quality_score"],
                            qa_data["is_active"],
                            qa_data["related_concepts"]
                        ))
                        
                        restore_summary["qa_pairs_restored"] += 1
                    
                    conn.commit()
                    logger.info(f"✅ Restored {restore_summary['qa_pairs_restored']} QA pairs")
            except Exception as e:
                logger.error(f"Error restoring QA pairs: {e}", exc_info=True)
                restore_summary["errors"].append(f"QA restoration error: {str(e)}")
        
        logger.info(f"✅ Restore completed: {restore_summary}")
        
        return {
            "success": True,
            "original_session_id": original_session_id,
            "target_session_id": target_session_id,
            "summary": restore_summary,
            "message": f"Restored: {restore_summary['topics_restored']} topics, {restore_summary['knowledge_bases_restored']} KBs, {restore_summary['qa_pairs_restored']} QA pairs"
        }
        
    except Exception as e:
        logger.error(f"❌ Restore failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")


@router.get("/{session_id}/backup/download")
async def download_backup(session_id: str):
    """
    Download backup as JSON file.
    """
    try:
        backup_response = await backup_session(session_id)
        backup_data = backup_response.data
        
        # Create JSON response with proper headers for download
        return JSONResponse(
            content=backup_data,
            headers={
                "Content-Disposition": f'attachment; filename="session_backup_{session_id}_{int(datetime.now().timestamp())}.json"',
                "Content-Type": "application/json"
            }
        )
    except Exception as e:
        logger.error(f"Download backup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

