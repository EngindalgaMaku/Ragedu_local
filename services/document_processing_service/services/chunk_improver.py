"""
Chunk improvement service using LLM post-processing
Handles single and bulk chunk improvement operations
"""
import time
from datetime import datetime
from typing import Dict, Any, Optional
from utils.logger import logger
from config import MODEL_INFERENCER_URL


def improve_single_chunk(
    chunk_text: str,
    language: str = "tr",
    model_name: str = "llama-3.1-8b-instant",
    session_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    document_name: Optional[str] = None,
    chunk_index: Optional[int] = None,
    update_chromadb: bool = True
) -> Dict[str, Any]:
    """
    Improve a single chunk using LLM post-processing
    
    Args:
        chunk_text: Original chunk text
        language: Language code (e.g., 'tr' for Turkish)
        model_name: LLM model to use
        session_id: Session ID for ChromaDB update
        chunk_id: Specific chunk ID to update
        document_name: Document name (alternative identifier)
        chunk_index: Chunk index (alternative identifier)
        update_chromadb: Whether to update ChromaDB
        
    Returns:
        Dictionary with improvement results
    """
    start_time = time.time()
    
    try:
        # Import post-processor
        try:
            from src.text_processing.chunk_post_processor_grok import GrokChunkPostProcessor, PostProcessingConfig
        except ImportError:
            try:
                from src.text_processing.chunk_post_processor import ChunkPostProcessor as GrokChunkPostProcessor, PostProcessingConfig
            except ImportError:
                return {
                    "success": False,
                    "original_text": chunk_text,
                    "improved_text": None,
                    "message": "LLM post-processing not available",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
        
        # Create post-processor
        config = PostProcessingConfig(
            enabled=True,
            model_name=model_name,
            model_inference_url=MODEL_INFERENCER_URL,
            language=language,
            timeout_seconds=30,
            retry_attempts=2
        )
        post_processor = GrokChunkPostProcessor(config)
        
        # Force processing (disable worth check)
        original_check = post_processor._is_chunk_worth_processing
        post_processor._is_chunk_worth_processing = lambda x: True
        
        # Process chunk
        improved_chunks = post_processor.process_chunks([chunk_text])
        
        # Restore original check
        post_processor._is_chunk_worth_processing = original_check
        
        if improved_chunks and len(improved_chunks) > 0:
            improved_text = improved_chunks[0]
            
            # Check if actually improved
            if improved_text != chunk_text and len(improved_text.strip()) > 10:
                processing_time = (time.time() - start_time) * 1000
                
                # Update ChromaDB if requested
                if update_chromadb and session_id:
                    _update_chunk_in_chromadb(
                        session_id=session_id,
                        chunk_id=chunk_id,
                        document_name=document_name,
                        chunk_index=chunk_index,
                        improved_text=improved_text,
                        model_name=model_name
                    )
                
                return {
                    "success": True,
                    "original_text": chunk_text,
                    "improved_text": improved_text,
                    "message": "Chunk improved successfully",
                    "processing_time_ms": processing_time
                }
        
        return {
            "success": False,
            "original_text": chunk_text,
            "improved_text": None,
            "message": "LLM did not improve the chunk",
            "processing_time_ms": (time.time() - start_time) * 1000
        }
        
    except Exception as e:
        logger.error(f"❌ Error improving single chunk: {e}")
        return {
            "success": False,
            "original_text": chunk_text,
            "improved_text": None,
            "message": f"Error: {str(e)}",
            "processing_time_ms": (time.time() - start_time) * 1000
        }


def improve_all_chunks_in_session(
    session_id: str,
    language: str = "tr",
    model_name: str = "llama-3.1-8b-instant",
    skip_already_improved: bool = True
) -> Dict[str, Any]:
    """
    Improve all chunks in a session using LLM post-processing
    
    Args:
        session_id: Session ID
        language: Language code
        model_name: LLM model to use
        skip_already_improved: Skip chunks already marked as improved
        
    Returns:
        Dictionary with bulk improvement results
    """
    start_time = time.time()
    
    try:
        from core.chromadb_client import get_chroma_client
        from utils.helpers import format_collection_name
        
        # Import post-processor
        try:
            from src.text_processing.chunk_post_processor_grok import GrokChunkPostProcessor, PostProcessingConfig
        except ImportError:
            try:
                from src.text_processing.chunk_post_processor import ChunkPostProcessor as GrokChunkPostProcessor, PostProcessingConfig
            except ImportError:
                return {
                    "success": False,
                    "total_chunks": 0,
                    "processed": 0,
                    "improved": 0,
                    "failed": 0,
                    "skipped": 0,
                    "message": "LLM post-processing not available",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
        
        logger.info(f"🚀 Starting bulk chunk improvement for session {session_id}")
        
        # Get ChromaDB client and collection
        client = get_chroma_client()
        collection_name = format_collection_name(session_id, add_timestamp=False)
        
        # Find collection (try alternatives including timestamped)
        collection = _find_collection(client, collection_name, session_id)
        
        if not collection:
            return {
                "success": False,
                "total_chunks": 0,
                "processed": 0,
                "improved": 0,
                "failed": 0,
                "skipped": 0,
                "message": f"Collection not found for session {session_id}",
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        
        # Get all chunks INCLUDING embeddings
        results = collection.get(include=['documents', 'metadatas', 'embeddings'])
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        embeddings = results.get('embeddings', [])
        
        total_chunks = len(documents)
        logger.info(f"📊 Found {total_chunks} chunks to process")
        
        # Create post-processor
        config = PostProcessingConfig(
            enabled=True,
            model_name=model_name,
            model_inference_url=MODEL_INFERENCER_URL,
            language=language,
            timeout_seconds=30,
            retry_attempts=2
        )
        post_processor = GrokChunkPostProcessor(config)
        
        # Process chunks
        processed = improved = failed = skipped = 0
        
        for i, (doc, metadata, chunk_id, embedding) in enumerate(zip(documents, metadatas, ids, embeddings)):
            # Skip if already improved
            if skip_already_improved and metadata.get('llm_improved'):
                logger.debug(f"⏭️  Skipping chunk {i+1}/{total_chunks} (already improved)")
                skipped += 1
                continue
            
            # Force processing
            original_check = post_processor._is_chunk_worth_processing
            post_processor._is_chunk_worth_processing = lambda x: True
            
            try:
                improved_chunks = post_processor.process_chunks([doc])
                
                if improved_chunks and len(improved_chunks) > 0:
                    improved_text = improved_chunks[0]
                    
                    if improved_text != doc and len(improved_text.strip()) > 10:
                        # Update in ChromaDB with retry logic
                        updated_metadata = metadata.copy()
                        updated_metadata['llm_improved'] = True
                        updated_metadata['llm_model'] = model_name
                        updated_metadata['improvement_timestamp'] = datetime.now().isoformat()
                        
                        _update_chunk_with_retry(
                            collection, chunk_id, improved_text, updated_metadata, embedding
                        )
                        
                        improved += 1
                        logger.info(f"✅ Chunk {i+1}/{total_chunks} improved successfully")
                    else:
                        failed += 1
                else:
                    failed += 1
                
                processed += 1
                
            except Exception as e:
                failed += 1
                processed += 1
                logger.error(f"❌ Chunk {i+1}/{total_chunks} error: {e}")
            finally:
                post_processor._is_chunk_worth_processing = original_check
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "total_chunks": total_chunks,
            "processed": processed,
            "improved": improved,
            "failed": failed,
            "skipped": skipped,
            "message": f"Processed {processed} chunks: {improved} improved, {failed} failed, {skipped} skipped",
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"❌ Error in bulk chunk improvement: {e}")
        return {
            "success": False,
            "total_chunks": 0,
            "processed": 0,
            "improved": 0,
            "failed": 0,
            "skipped": 0,
            "message": f"Error: {str(e)}",
            "processing_time_ms": (time.time() - start_time) * 1000
        }


def _update_chunk_in_chromadb(session_id, chunk_id, document_name, chunk_index, improved_text, model_name):
    """Helper function to update a single chunk in ChromaDB"""
    try:
        from core.chromadb_client import get_chroma_client
        from utils.helpers import format_collection_name
        
        client = get_chroma_client()
        collection_name = format_collection_name(session_id, add_timestamp=False)
        collection = _find_collection(client, collection_name, session_id)
        
        if collection:
            # Find chunk by ID or by document_name + chunk_index
            if chunk_id:
                target_chunk_id = chunk_id
            elif document_name and chunk_index is not None:
                # Search for chunk by metadata
                results = collection.get()
                for i, metadata in enumerate(results.get('metadatas', [])):
                    doc_name = metadata.get('document_name') or metadata.get('filename')
                    idx = metadata.get('chunk_index')
                    if doc_name == document_name and idx == chunk_index:
                        target_chunk_id = results['ids'][i]
                        break
                else:
                    logger.warning(f"Chunk not found for {document_name} index {chunk_index}")
                    return
            else:
                logger.warning("No chunk identifier provided")
                return
            
            # Get existing chunk to preserve embedding
            existing = collection.get(ids=[target_chunk_id], include=['embeddings'])
            if existing and existing['embeddings']:
                updated_metadata = {'llm_improved': True, 'llm_model': model_name, 'improvement_timestamp': datetime.now().isoformat()}
                collection.update(
                    ids=[target_chunk_id],
                    documents=[improved_text],
                    metadatas=[updated_metadata],
                    embeddings=[existing['embeddings'][0]]
                )
                logger.info(f"✅ Chunk updated in ChromaDB (ID: {target_chunk_id})")
    except Exception as e:
        logger.error(f"❌ Failed to update ChromaDB: {e}")


def _find_collection(client, collection_name, session_id):
    """Helper function to find collection with alternative names"""
    try:
        return client.get_collection(name=collection_name)
    except:
        # Try alternatives including timestamped
        alternative_names = [f"session_{session_id}"]
        
        try:
            all_collections = client.list_collections()
            for coll in all_collections:
                if coll.name.startswith(collection_name + "_"):
                    alternative_names.insert(0, coll.name)
        except:
            pass
        
        for alt_name in alternative_names:
            try:
                return client.get_collection(name=alt_name)
            except:
                continue
        
        return None


def _update_chunk_with_retry(collection, chunk_id, improved_text, metadata, embedding):
    """Helper function to update chunk with retry logic"""
    max_retries = 5
    retry_delay = 3.0
    
    for retry in range(max_retries):
        try:
            collection.update(
                ids=[chunk_id],
                documents=[improved_text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            if retry > 0:
                logger.info(f"✅ ChromaDB update succeeded on attempt {retry+1}")
            break
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = retry_delay * (retry + 1)
                logger.warning(f"⚠️ Update failed (attempt {retry+1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise






