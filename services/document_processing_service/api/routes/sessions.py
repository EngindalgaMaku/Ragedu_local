"""
Session-related endpoints - retrieve chunks, stats, etc.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from core.chromadb_client import get_chroma_client
from utils.helpers import format_collection_name
from utils.logger import logger

router = APIRouter()


@router.get("/sessions/{session_id}/chunks")
async def get_session_chunks(session_id: str):
    """
    Get all chunks for a specific session
    
    Args:
        session_id: The session ID
        
    Returns:
        JSON with chunks array
        
    Example:
        GET /sessions/abc123/chunks
        Response: {"chunks": [...]}
    """
    try:
        logger.info(f"📦 Fetching chunks for session: {session_id}")
        
        # Get ChromaDB client
        client = get_chroma_client()
        
        # Format collection name WITHOUT timestamp
        # CRITICAL: All files in the same session use the SAME collection (no timestamp)
        collection_name = format_collection_name(session_id, add_timestamp=False)
        
        # CRITICAL FIX: Get ALL collections for this session (including timestamped versions)
        # This is needed because old code might have created timestamped collections
        # We need to find ALL collections that match this session
        all_collections_for_session = []
        
        # First try to get the non-timestamped collection
        try:
            collection = client.get_collection(name=collection_name)
            all_collections_for_session.append(collection)
            logger.info(f"✅ Found primary collection: {collection.name}")
        except Exception as e:
            logger.debug(f"Primary collection '{collection_name}' not found: {e}")
        
        # Then find ALL timestamped versions (for backward compatibility with old code)
        try:
            all_collections = client.list_collections()
            all_collection_names = [c.name for c in all_collections]
            
            # Build base pattern to search for
            base_patterns = [collection_name]
            if len(session_id) == 32:
                uuid_format = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
                base_patterns.append(uuid_format)
                base_patterns.append(f"session_{session_id}")
            
            # Find all collections that match this session (including timestamped)
            for pattern in base_patterns:
                for coll_name in all_collection_names:
                    if coll_name == pattern:
                        # Exact match - already added if found
                        continue
                    elif coll_name.startswith(pattern + "_"):
                        # Timestamped version
                        suffix = coll_name[len(pattern)+1:]
                        if suffix.isdigit():
                            try:
                                alt_collection = client.get_collection(name=coll_name)
                                if alt_collection not in all_collections_for_session:
                                    all_collections_for_session.append(alt_collection)
                                    logger.info(f"✅ Found additional timestamped collection: {coll_name}")
                            except Exception as e:
                                logger.warning(f"Could not load collection {coll_name}: {e}")
        except Exception as e:
            logger.warning(f"Could not list all collections: {e}")
        
        if not all_collections_for_session:
            logger.warning(f"⚠️ No collections found for session {session_id}")
            return {"chunks": []}
        
        # Get all documents from ALL collections
        all_chunks = []
        chunks_by_document = {}  # Track chunks per document for logging
        seen_chunk_ids = set()  # CRITICAL: Track chunk IDs to avoid duplicates
        seen_chunk_signatures = set()  # CRITICAL: Track chunk content signatures to avoid content duplicates
        
        for coll in all_collections_for_session:
            try:
                result = coll.get(
                    include=["documents", "metadatas", "embeddings"]
                )
                
                documents = result.get('documents', [])
                metadatas = result.get('metadatas', [])
                ids = result.get('ids', [])
                
                logger.info(f"📊 Retrieved {len(documents)} chunks from collection: {coll.name}")
                
                # Format chunks - FRONTEND COMPATIBLE FORMAT
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
                    
                    # CRITICAL FIX: Skip duplicate chunks (same chunk_id already seen)
                    if chunk_id in seen_chunk_ids:
                        logger.debug(f"⏭️ Skipping duplicate chunk (by ID): {chunk_id}")
                        continue
                    seen_chunk_ids.add(chunk_id)
                    
                    # CRITICAL FIX: Also check for content duplicates (same document_name + chunk_index + content hash)
                    # This catches cases where same content was stored with different chunk_ids
                    import hashlib
                    doc_name = metadata.get("filename", metadata.get("source_file", "unknown"))
                    chunk_idx = metadata.get("chunk_index", i + 1)
                    content_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()[:16]
                    content_signature = f"{doc_name}:{chunk_idx}:{content_hash}"
                    
                    if content_signature in seen_chunk_signatures:
                        logger.debug(f"⏭️ Skipping duplicate chunk (by content): {doc_name} chunk {chunk_idx}")
                        continue
                    seen_chunk_signatures.add(content_signature)
                    
                    # Extract document name from metadata
                    document_name = metadata.get("filename", metadata.get("source_file", "unknown"))
                    
                    # Track chunks per document
                    if document_name not in chunks_by_document:
                        chunks_by_document[document_name] = 0
                    chunks_by_document[document_name] += 1
                    
                    all_chunks.append({
                        # Frontend expects these exact field names
                        "document_name": document_name,
                        "chunk_index": metadata.get("chunk_index", len(all_chunks) + 1),
                        "chunk_text": doc,  # Frontend expects 'chunk_text' not 'content'
                        "chunk_metadata": {
                            "llm_improved": metadata.get("llm_improved", False),
                            "improvement_timestamp": metadata.get("improvement_timestamp"),
                            "original_length": metadata.get("original_length"),
                            "improved_length": metadata.get("improved_length"),
                            "model_used": metadata.get("model_used"),
                            # Keep additional metadata for backward compatibility
                            "chunk_id": chunk_id,
                            "full_metadata": metadata
                        }
                    })
            except Exception as e:
                logger.error(f"❌ Error fetching documents from collection {coll.name}: {e}")
                continue
        
        # Sort by document_name and chunk_index for consistent display
        all_chunks.sort(key=lambda x: (x.get("document_name", ""), x.get("chunk_index", 0)))
        
        # Log summary
        logger.info(f"📊 Chunks per document: {chunks_by_document}")
        logger.info(f"✅ Retrieved {len(all_chunks)} chunks from {len(all_collections_for_session)} collection(s) for session {session_id}")
        logger.info(f"📁 Found chunks from {len(chunks_by_document)} unique document(s)")
        
        return {"chunks": all_chunks}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in get_session_chunks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


def _find_collection_with_alternatives(client, collection_name: str, session_id: str):
    """Find collection with alternative naming patterns"""
    try:
        return client.get_collection(name=collection_name)
    except:
        # Try alternatives including timestamped
        alternative_names = []
        
        try:
            all_collections = client.list_collections()
            all_collection_names = [c.name for c in all_collections]
            
            # Search for timestamped versions
            for coll_name in all_collection_names:
                if coll_name.startswith(collection_name + "_"):
                    suffix = coll_name[len(collection_name)+1:]
                    if suffix.isdigit():
                        alternative_names.append((coll_name, int(suffix)))
            
            # Sort by timestamp (newest first)
            alternative_names.sort(key=lambda x: x[1], reverse=True)
            
            for alt_name, _ in alternative_names:
                try:
                    return client.get_collection(name=alt_name)
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error finding collection alternatives: {e}")
        
        return None

