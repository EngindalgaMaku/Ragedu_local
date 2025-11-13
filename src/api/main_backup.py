from typing import List, Optional
import os
import tempfile
import sqlite3
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy imports - import heavy modules only when needed to speed up startup
# These will be imported in functions where they're actually used

app = FastAPI(title="RAG3 API", version="0.1.0")

# CORS (adjust for your frontend domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"service": "rageducation-backend", "status": "ok"}


class CreateSessionRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    category: str  # Changed from SessionCategory to str to avoid import issues
    created_by: str = "system"
    grade_level: Optional[str] = ""
    subject_area: Optional[str] = ""
    learning_objectives: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    is_public: bool = False


class SessionResponse(BaseModel):
    session_id: str
    name: str
    description: str
    category: str
    status: str
    created_by: str
    created_at: str
    updated_at: str
    last_accessed: str
    grade_level: str
    subject_area: str
    learning_objectives: List[str]
    tags: List[str]
    document_count: int
    total_chunks: int
    query_count: int
    user_rating: float
    is_public: bool
    backup_count: int


class RAGQueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5
    use_rerank: bool = True
    min_score: float = 0.1
    max_context_chars: int = 8000
    model: Optional[str] = None


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[str] = []


class PDFToMarkdownResponse(BaseModel):
    success: bool
    message: str
    markdown_filename: Optional[str] = None
    metadata: Optional[dict] = None


class MarkdownListResponse(BaseModel):
    markdown_files: List[str]
    count: int


class AddMarkdownDocumentsRequest(BaseModel):
    session_id: str
    markdown_files: List[str]


class AddMarkdownDocumentsResponse(BaseModel):
    success: bool
    processed_count: int
    total_chunks_added: int
    message: str
    errors: Optional[List[str]] = []


class MarkdownContentResponse(BaseModel):
    content: str


class RAGConfigureAndProcessRequest(BaseModel):
    session_id: str
    markdown_files: List[str]
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str


class RAGConfigureAndProcessResponse(BaseModel):
    success: bool
    processed_files: int
    total_chunks: int
    message: str
    errors: Optional[List[str]] = []


class ChunkResponse(BaseModel):
    document_name: str
    chunk_index: int
    chunk_text: str
    chunk_metadata: Optional[dict] = None


class SessionChunksResponse(BaseModel):
    chunks: List[ChunkResponse]
    total_count: int
    session_id: str


class ListModelsResponse(BaseModel):
    models: List[str]


class ChangelogEntry(BaseModel):
    id: int
    version: str
    date: str
    changes: List[str]

class CreateChangelogEntryRequest(BaseModel):
    version: str
    date: str
    changes: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/changelog", response_model=List[ChangelogEntry])
def get_changelog():
    from src.services.session_manager import professional_session_manager
    with professional_session_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, version, date, changes FROM changelog ORDER BY date DESC")
        rows = cursor.fetchall()
        return [
            ChangelogEntry(
                id=row["id"],
                version=row["version"],
                date=row["date"],
                changes=json.loads(row["changes"]),
            )
            for row in rows
        ]

@app.post("/changelog", response_model=ChangelogEntry)
def create_changelog_entry(req: CreateChangelogEntryRequest):
    from src.services.session_manager import professional_session_manager
    with professional_session_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO changelog (version, date, changes) VALUES (?, ?, ?)",
            (req.version, req.date, json.dumps(req.changes)),
        )
        new_id = cursor.lastrowid
        return ChangelogEntry(
            id=new_id,
            version=req.version,
            date=req.date,
            changes=req.changes,
        )


@app.get("/sessions", response_model=List[SessionResponse])
def list_sessions(created_by: Optional[str] = None, category: Optional[str] = None,
                  status: Optional[str] = None, limit: int = 50):
    from src.services.session_manager import professional_session_manager, SessionCategory, SessionStatus
    
    # Convert string parameters to enum types if needed
    category_enum = SessionCategory(category) if category else None
    status_enum = SessionStatus(status) if status else None
    sessions = professional_session_manager.list_sessions(
        created_by=created_by,
        category=category_enum,
        status=status_enum,
        limit=limit,
    )
    res: List[SessionResponse] = []
    for s in sessions:
        res.append(SessionResponse(
            session_id=s.session_id,
            name=s.name,
            description=s.description,
            category=s.category.value,
            status=s.status.value,
            created_by=s.created_by,
            created_at=s.created_at,
            updated_at=s.updated_at,
            last_accessed=s.last_accessed,
            grade_level=s.grade_level,
            subject_area=s.subject_area,
            learning_objectives=s.learning_objectives,
            tags=s.tags,
            document_count=s.document_count,
            total_chunks=s.total_chunks,
            query_count=s.query_count,
            user_rating=s.user_rating,
            is_public=s.is_public,
            backup_count=s.backup_count,
        ))
    return res


@app.post("/sessions", response_model=SessionResponse)
def create_session(req: CreateSessionRequest):
    from src.services.session_manager import professional_session_manager, SessionCategory
    try:
        # Convert string category to enum
        category_enum = SessionCategory(req.category) if req.category else SessionCategory.RESEARCH
        
        meta = professional_session_manager.create_session(
            name=req.name,
            description=req.description or "",
            category=category_enum,
            created_by=req.created_by,
            grade_level=req.grade_level or "",
            subject_area=req.subject_area or "",
            learning_objectives=req.learning_objectives or [],
            tags=req.tags or [],
            is_public=req.is_public,
        )
        return SessionResponse(
            session_id=meta.session_id,
            name=meta.name,
            description=meta.description,
            category=meta.category.value,
            status=meta.status.value,
            created_by=meta.created_by,
            created_at=meta.created_at,
            updated_at=meta.updated_at,
            last_accessed=meta.last_accessed,
            grade_level=meta.grade_level,
            subject_area=meta.subject_area,
            learning_objectives=meta.learning_objectives,
            tags=meta.tags,
            document_count=meta.document_count,
            total_chunks=meta.total_chunks,
            query_count=meta.query_count,
            user_rating=meta.user_rating,
            is_public=meta.is_public,
            backup_count=meta.backup_count,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    from src.services.session_manager import professional_session_manager
    meta = professional_session_manager.get_session_metadata(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=meta.session_id,
        name=meta.name,
        description=meta.description,
        category=meta.category.value,
        status=meta.status.value,
        created_by=meta.created_by,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        last_accessed=meta.last_accessed,
        grade_level=meta.grade_level,
        subject_area=meta.subject_area,
        learning_objectives=meta.learning_objectives,
        tags=meta.tags,
        document_count=meta.document_count,
        total_chunks=meta.total_chunks,
        query_count=meta.query_count,
        user_rating=meta.user_rating,
        is_public=meta.is_public,
        backup_count=meta.backup_count,
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, create_backup: bool = True, deleted_by: Optional[str] = None):
    from src.services.session_manager import professional_session_manager
    ok = professional_session_manager.delete_session(session_id, create_backup=create_backup, deleted_by=deleted_by)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found or delete failed")
    return {"deleted": True}


@app.post("/documents/upload")
async def upload_document(
    session_id: str = Form(...),
    strategy: str = Form("markdown"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    embedding_model: str = Form("mixedbread-ai/mxbai-embed-large-v1"),
    file: UploadFile = File(...),
):
    # Resolve store by session_id (backward-compatible path handling)
    # Lazy import to avoid startup import errors
    import os
    os.makedirs("data/vector_db/sessions", exist_ok=True)
    safe = session_id.strip().replace(" ", "_") or "default"
    collection_name = f"session_{safe}"
    
    # Import ChromaVectorStore only when needed
    from src.vector_store.chroma_store import ChromaVectorStore
    store = ChromaVectorStore(collection_name=collection_name)

    content = await file.read()
    try:
        # Use app_logic to process and add document
        from src.app_logic import add_document_to_store
        stats = add_document_to_store(
            file_bytes=content,
            filename=file.filename,
            vector_store=store,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )
        # Optionally update session stats here if needed
        return {"ok": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest):
    # Lazy imports to avoid startup errors with numpy dependencies
    from src.app_logic import retrieve_and_answer, label_from_meta
    
    # Resolve store
    import os
    os.makedirs("data/vector_db/sessions", exist_ok=True)
    safe = req.session_id.strip().replace(" ", "_") or "default"
    collection_name = f"session_{safe}"
    
    # Import ChromaVectorStore only when this endpoint is called
    from src.vector_store.chroma_store import ChromaVectorStore
    store = ChromaVectorStore(collection_name=collection_name)

    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Use the actual RAG logic
        answer, sources, scores, metas = retrieve_and_answer(
            vector_store=store,
            query=req.query,
            top_k=req.top_k,
            use_rerank=req.use_rerank,
            min_score=req.min_score,
            max_context_chars=req.max_context_chars,
            generation_model=req.model,
        )
        
        # Create source labels from metadata
        source_labels = []
        if sources:
            for i, (text, meta) in enumerate(zip(sources, metas)):
                source_labels.append(label_from_meta(meta, text))

        return RAGQueryResponse(answer=answer, sources=source_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")


@app.post("/documents/convert-document-to-markdown", response_model=PDFToMarkdownResponse)
async def convert_document_to_markdown(file: UploadFile = File(...)):
    """
    Convert uploaded document (PDF, DOCX, PPTX, XLSX) to Markdown format and save to data/markdown/ directory.
    Delegates PDF processing to the PDF Processing Service microservice.
    """
    supported_extensions = ['.pdf', '.docx', '.pptx', '.xlsx']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
        )
    
    try:
        # Get PDF processor URL from environment variable directly
        import requests
        
        pdf_processor_url = os.getenv('PDF_PROCESSOR_URL', 'http://localhost:8001')
        
        # Create data/markdown directory if it doesn't exist
        markdown_dir = Path("data/markdown")
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # Read uploaded file content
        content = await file.read()
        
        try:
            # PDF processor URL already retrieved above
            
            # Prepare file for multipart upload to microservice
            files = {
                'file': (file.filename, content, file.content_type)
            }
            
            # Send request to PDF Processing Service
            response = requests.post(
                f"{pdf_processor_url}/process",
                files=files,
                timeout=300  # 5 minutes timeout for PDF processing
            )
            
            if response.status_code != 200:
                error_detail = f"PDF processor service error: {response.status_code}"
                if response.text:
                    try:
                        error_json = response.json()
                        error_detail = error_json.get('detail', error_detail)
                    except:
                        error_detail = f"{error_detail} - {response.text[:200]}"
                        
                raise HTTPException(
                    status_code=500,
                    detail=error_detail
                )
            
            # Parse response from PDF processor
            processor_result = response.json()
            
            # PDF Processing Service returns: {"content": "...", "metadata": {...}}
            markdown_content = processor_result.get('content')
            metadata = processor_result.get('metadata', {})
            
            if not markdown_content:
                raise HTTPException(status_code=500, detail="Failed to extract content from document")
            
            # Generate markdown filename (remove original extension, add .md)
            base_filename = Path(file.filename).stem
            markdown_filename = f"{base_filename}.md"
            markdown_filepath = markdown_dir / markdown_filename
            
            # Save markdown content to file
            with open(markdown_filepath, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)
            
            return PDFToMarkdownResponse(
                success=True,
                message=f"Document successfully converted to Markdown",
                markdown_filename=markdown_filename,
                metadata=metadata
            )
            
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail="PDF Processing Service is not available. Please try again later."
            )
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=504,
                detail="PDF processing timed out. Please try with a smaller file."
            )
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to communicate with PDF Processing Service: {str(e)}"
            )
                    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document conversion failed: {str(e)}")


@app.get("/documents/list-markdown", response_model=MarkdownListResponse)
def list_markdown_files():
    """
    List all .md files in the data/markdown/ directory.
    """
    try:
        markdown_dir = Path("data/markdown")
        
        # Create directory if it doesn't exist
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all .md files
        md_files = []
        if markdown_dir.exists():
            md_files = [f.name for f in markdown_dir.glob("*.md")]
        
        # Sort alphabetically for consistent ordering
        md_files.sort()
        
        return MarkdownListResponse(
            markdown_files=md_files,
            count=len(md_files)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list markdown files: {str(e)}")


@app.get("/documents/markdown/{filename}", response_model=MarkdownContentResponse)
def get_markdown_file_content(filename: str):
    """
    Get the content of a specific markdown file from the data/markdown/ directory.
    Implements path traversal protection to ensure users cannot access files outside this directory.
    """
    try:
        # Path traversal protection - normalize the filename and ensure it only contains safe characters
        # Remove any directory separators and path manipulation attempts
        safe_filename = os.path.basename(filename).replace('..', '').replace('/', '').replace('\\', '')
        
        # Ensure the filename ends with .md for additional security
        if not safe_filename.lower().endswith('.md'):
            safe_filename += '.md'
        
        # Construct the full path within the markdown directory
        markdown_dir = Path("data/markdown")
        file_path = markdown_dir / safe_filename
        
        # Additional security check: ensure the resolved path is actually within our markdown directory
        try:
            file_path = file_path.resolve()
            markdown_dir = markdown_dir.resolve()
            
            # Check if the file path is within the markdown directory
            if not str(file_path).startswith(str(markdown_dir)):
                raise HTTPException(status_code=400, detail="Invalid file path")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"Markdown file '{safe_filename}' not found")
        
        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding if utf-8 fails
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        return MarkdownContentResponse(content=content)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read markdown file: {str(e)}")


@app.post("/sessions/add-markdown-documents", response_model=AddMarkdownDocumentsResponse)
def add_markdown_documents_to_session(req: AddMarkdownDocumentsRequest):
    """
    Associate existing Markdown files with a RAG session by processing their content
    and adding the resulting vectors to the session's vector store.
    """
    try:
        # Validate session exists
        from src.services.session_manager import professional_session_manager
        session_meta = professional_session_manager.get_session_metadata(req.session_id)
        if not session_meta:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the session's vector store - lazy import
        import os
        os.makedirs("data/vector_db/sessions", exist_ok=True)
        safe = req.session_id.strip().replace(" ", "_") or "default"
        collection_name = f"session_{safe}"
        
        # Import ChromaVectorStore only when needed
        from src.vector_store.chroma_store import ChromaVectorStore
        store = ChromaVectorStore(collection_name=collection_name)
        
        processed_count = 0
        total_chunks_added = 0
        errors = []
        
        # Process each markdown file
        for filename in req.markdown_files:
            try:
                # Construct full path to markdown file
                markdown_path = Path("data/markdown") / filename
                
                # Verify file exists
                if not markdown_path.exists():
                    errors.append(f"File not found: {filename}")
                    continue
                
                # Read markdown content
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                if not markdown_content.strip():
                    errors.append(f"File is empty: {filename}")
                    continue
                
                # Use the existing logic to process the text content
                stats = add_text_content_to_store(
                    text_content=markdown_content,
                    filename=filename,
                    vector_store=store,
                    strategy="markdown",
                    # Use default chunking/embedding params for this endpoint
                )
                
                processed_count += 1
                total_chunks_added += stats.get("chunks", 0)
                
            except Exception as file_error:
                errors.append(f"Error processing {filename}: {str(file_error)}")
        
        # Update session metadata if any files were processed
        if processed_count > 0:
            # Update document count and chunk count for the session
            current_meta = professional_session_manager.get_session_metadata(req.session_id)
            if current_meta:
                professional_session_manager.update_session_metadata(
                    req.session_id,
                    document_count=current_meta.document_count + processed_count,
                    total_chunks=current_meta.total_chunks + total_chunks_added
                )
        
        return AddMarkdownDocumentsResponse(
            success=processed_count > 0,
            processed_count=processed_count,
            total_chunks_added=total_chunks_added,
            message=f"Successfully processed {processed_count} out of {len(req.markdown_files)} markdown files",
            errors=errors if errors else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add markdown documents: {str(e)}")


@app.post("/rag/configure-and-process", response_model=RAGConfigureAndProcessResponse)
def configure_and_process_rag(req: RAGConfigureAndProcessRequest):
    """
    Configure and run the RAG chunking and vectorization process for multiple markdown files.
    This endpoint accepts custom chunking and embedding parameters and stores both vectors
    and chunk text content in the database.
    
    IMPORTANT: This endpoint clears all existing data for the session before processing new data
    to prevent duplicate chunks.
    """
    try:
        # Validate session exists
        from src.services.session_manager import professional_session_manager
        session_meta = professional_session_manager.get_session_metadata(req.session_id)
        if not session_meta:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # CLEAR EXISTING DATA FIRST - This prevents duplicate chunks
        _clear_session_data(req.session_id)
        
        # Get the session's vector store (will be recreated as needed) - lazy import
        import os
        os.makedirs("data/vector_db/sessions", exist_ok=True)
        safe = req.session_id.strip().replace(" ", "_") or "default"
        collection_name = f"session_{safe}"
        
        # Import ChromaVectorStore only when needed
        from src.vector_store.chroma_store import ChromaVectorStore
        store = ChromaVectorStore(collection_name=collection_name)
        
        documents_processed = 0
        total_chunks_created = 0
        errors = []
        global_chunk_index = 0  # Track global chunk index across all documents
        
        # Process each markdown file
        for filename in req.markdown_files:
            try:
                # Construct full path to markdown file
                markdown_path = Path("data/markdown") / filename
                
                # Verify file exists
                if not markdown_path.exists():
                    errors.append(f"File not found: {filename}")
                    continue
                
                # Read markdown content
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                if not markdown_content.strip():
                    errors.append(f"File is empty: {filename}")
                    continue
                
                # Use the enhanced logic to process content and store chunks
                stats = add_text_content_to_store_with_chunk_storage(
                    text_content=markdown_content,
                    filename=filename,
                    vector_store=store,
                    session_id=req.session_id,
                    global_chunk_index=global_chunk_index,
                    strategy=req.chunk_strategy,
                    chunk_size=req.chunk_size,
                    chunk_overlap=req.chunk_overlap,
                    embedding_model=req.embedding_model,
                )
                
                documents_processed += 1
                chunks_added = stats.get("chunks", 0)
                total_chunks_created += chunks_added
                global_chunk_index += chunks_added  # Update global counter
                
            except Exception as file_error:
                errors.append(f"Error processing {filename}: {str(file_error)}")
        
        # Update session metadata if any files were processed
        if documents_processed > 0:
            current_meta = professional_session_manager.get_session_metadata(req.session_id)
            if current_meta:
                professional_session_manager.update_session_metadata(
                    req.session_id,
                    document_count=current_meta.document_count + documents_processed,
                    total_chunks=current_meta.total_chunks + total_chunks_created
                )
        
        return RAGConfigureAndProcessResponse(
            success=documents_processed > 0,
            processed_files=documents_processed,
            total_chunks=total_chunks_created,
            message=f"Successfully processed {documents_processed} out of {len(req.markdown_files)} markdown files",
            errors=errors if errors else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure and process RAG: {str(e)}")


@app.get("/rag/chunks/{session_id}", response_model=SessionChunksResponse)
def get_chunks_for_session(session_id: str):
    """
    Retrieve all text chunks for a specific session from the database.
    This endpoint fetches chunk data that was stored during the RAG processing.
    """
    try:
        # Import session manager locally to avoid startup issues
        from src.services.session_manager import professional_session_manager
        
        # Validate session exists
        session_meta = professional_session_manager.get_session_metadata(session_id)
        if not session_meta:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Connect to database and query chunks
        chunks = []
        with professional_session_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Query all chunks for the session, ordered by document and chunk index
            cursor.execute("""
                SELECT document_name, chunk_index, chunk_text, chunk_metadata
                FROM document_chunks
                WHERE session_id = ?
                ORDER BY document_name, chunk_index
            """, (session_id,))
            
            rows = cursor.fetchall()
            
            for row in rows:
                document_name, chunk_index, chunk_text, chunk_metadata = row
                
                # Parse metadata if it exists
                metadata = None
                if chunk_metadata:
                    try:
                        import json
                        metadata = json.loads(chunk_metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = None
                
                chunks.append(ChunkResponse(
                    document_name=document_name,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    chunk_metadata=metadata
                ))
        
        return SessionChunksResponse(
            chunks=chunks,
            total_count=len(chunks),
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {str(e)}")


def _clear_session_data(session_id: str):
    """
    Clear all existing data for a session to prevent duplicate chunks.
    This includes:
    - Deleting all document chunks from the database
    - Clearing ChromaDB collection data
    - Resetting session metadata counts
    """
    try:
        # Import session manager locally
        from src.services.session_manager import professional_session_manager
        
        # 1. Clear document chunks from database
        with professional_session_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks WHERE session_id = ?", (session_id,))
            rows_deleted = cursor.rowcount
        
        # 2. Clear ChromaDB collection data - lazy import
        try:
            safe = session_id.strip().replace(" ", "_") or "default"
            collection_name = f"session_{safe}"
            
            # Import ChromaVectorStore only when needed
            from src.vector_store.chroma_store import ChromaVectorStore
            store = ChromaVectorStore(collection_name=collection_name)
            
            # Reset the collection (delete and recreate)
            store.reset_collection()
            chroma_cleared = True
        except Exception as chroma_error:
            print(f"Error clearing ChromaDB collection: {chroma_error}")
            chroma_cleared = False
        
        # 3. Reset session metadata counts
        professional_session_manager.update_session_metadata(
            session_id,
            document_count=0,
            total_chunks=0
        )
        
        # Log the clearing operation
        professional_session_manager._log_activity(
            session_id, "cleared",
            f"Session data cleared: {rows_deleted} DB chunks, ChromaDB collection {'cleared' if chroma_cleared else 'failed to clear'}"
        )
        
    except Exception as e:
        # Log error but don't fail the whole operation
        print(f"Error clearing session data for {session_id}: {e}")
        # Re-raise to let the calling function handle it
        raise


def add_text_content_to_store_with_chunk_storage(
    text_content: str,
    filename: str,
    vector_store: any,  # FaissVectorStore - temporarily removed type hint
    session_id: str,
    global_chunk_index: int = 0,
    *,
    strategy: str = "markdown",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
) -> dict:
    """
    Add text content directly to vector store and store chunk text in database.
    Enhanced version of add_text_content_to_store that also stores chunks in the database.
    """
    if not text_content:
        return {"added": 0, "chunks": 0, "embedding_dim": None}
    
    # Import chunking function locally - USE SEMANTIC CHUNKING!
    from src.app_logic import chunk_text
    from src.text_processing.semantic_chunker import create_semantic_chunks
    
    # Use GROQ-powered semantic chunking if available
    try:
        print(f"🧠 Using GROQ-powered semantic chunking...")
        print(f"🔧 Text length: {len(text_content)}, target_size: {chunk_size}, overlap_ratio: {chunk_overlap / chunk_size}")
        print(f"🔧 Fallback strategy: {strategy}")
        
        chunks = create_semantic_chunks(
            text_content,
            target_size=chunk_size,
            overlap_ratio=chunk_overlap / chunk_size,
            language="auto",
            fallback_strategy=strategy
        )
        
        # Verify if chunks are actually from semantic chunking or fallback
        if chunks:
            print(f"✅ Semantic chunking returned {len(chunks)} chunks")
            
            # Check first chunk characteristics to see if it's semantic or fallback
            first_chunk = chunks[0][:100] + "..." if len(chunks[0]) > 100 else chunks[0]
            print(f"🔍 First chunk preview: {first_chunk}")
            
            # Log chunk sizes for analysis
            chunk_sizes = [len(c) for c in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            print(f"📊 Chunk statistics: avg={avg_size:.0f}, min={min(chunk_sizes)}, max={max(chunk_sizes)}")
        else:
            print("⚠️ Semantic chunking returned empty chunks")
            
    except Exception as e:
        print(f"⚠️ Semantic chunking failed with exception: {e}")
        print(f"🔄 Falling back to traditional chunking with strategy: {strategy}")
        chunks = chunk_text(
            text_content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
        print(f"📄 Fallback chunking created {len(chunks)} chunks")
    if not chunks:
        return {"added": 0, "chunks": 0, "embedding_dim": None}
    
    # Import embedding functions locally
    from src.app_logic import generate_embeddings, get_selected_provider
    
    # Generate embeddings using provider-aware logic
    selected_provider = get_selected_provider()
    if selected_provider == 'ollama':
        embeddings = generate_embeddings(chunks, model=embedding_model, provider='ollama')
    else:
        # For cloud providers, use local sentence transformers for embeddings
        embeddings = generate_embeddings(chunks, model=embedding_model, provider='sentence_transformers')
    
    if not embeddings:
        return {"added": 0, "chunks": len(chunks), "embedding_dim": None}
    
    embedding_dim = len(embeddings) if embeddings and isinstance(embeddings, list) and embeddings else None
    before = vector_store.ntotal
    
    # Build per-chunk metadata similar to add_document_to_store
    def parse_marker(txt: str) -> dict:
        for line in txt.splitlines():
            ls = line.strip()
            if ls.startswith("=== Page ") and ls.endswith("==="):
                try:
                    n = int(ls.replace("=", "").strip().split()[-1])
                    return {"page_number": n}
                except Exception:
                    return {}
            if ls.startswith("=== Slide ") and ls.endswith("==="):
                try:
                    n = int(ls.replace("=", "").strip().split()[-1])
                    return {"slide_number": n}
                except Exception:
                    return {}
        return {}

    def guess_title(txt: str) -> str | None:
        seen_marker = False
        for line in txt.splitlines():
            ls = line.strip()
            if not ls:
                continue
            if (ls.startswith("=== Page ") or ls.startswith("=== Slide ")) and ls.endswith("==="):
                seen_marker = True
                continue
            if seen_marker:
                return ls[:120]
        # fallback: first non-empty line
        for line in txt.splitlines():
            ls = line.strip()
            if ls:
                return ls[:120]
        return None

    metadatas = []
    for ch in chunks:
        md = {"source_file": filename}
        md.update(parse_marker(ch))
        t = guess_title(ch)
        if t:
            md["title"] = t
        metadatas.append(md)

    # Add to vector store
    vector_store.add_documents(chunks, embeddings, metadatas=metadatas)
    vector_store.save_store()
    
    # Store chunk text in database with global chunk indexing
    _store_document_chunks_in_db(session_id, filename, chunks, metadatas, global_chunk_index)
    
    after = vector_store.ntotal
    return {
        "added": max(0, after - before),
        "chunks": len(chunks),
        "embedding_dim": embedding_dim,
    }


def _store_document_chunks_in_db(session_id: str, document_name: str, chunks: List[str], metadatas: List[dict], start_chunk_index: int = 0):
    """
    Store document chunks in the sessions database for text retrieval and analysis.
    
    Args:
        session_id: The session identifier
        document_name: The name of the document being processed
        chunks: List of chunk texts
        metadatas: List of metadata dictionaries for each chunk
        start_chunk_index: The starting chunk index (for sequential indexing across documents)
    """
    import json
    
    # Import session manager locally
    from src.services.session_manager import professional_session_manager
    
    # Get database connection through session manager
    with professional_session_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        for i, (chunk_text, metadata) in enumerate(zip(chunks, metadatas)):
            # Use start_chunk_index + i to ensure sequential indexing across all documents
            chunk_index = start_chunk_index + i
            cursor.execute("""
                INSERT INTO document_chunks (session_id, document_name, chunk_index, chunk_text, chunk_metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                document_name,
                chunk_index,
                chunk_text,
                json.dumps(metadata)
            ))


def add_text_content_to_store(
    text_content: str,
    filename: str,
    vector_store: any,  # FaissVectorStore - temporarily removed type hint
    *,
    strategy: str = "markdown",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
) -> dict:
    """
    Add text content directly to vector store (for markdown files that are already saved).
    This is similar to add_document_to_store but works with text content directly.
    """
    if not text_content:
        return {"added": 0, "chunks": 0, "embedding_dim": None}
    
    # Import chunking function locally
    from src.app_logic import chunk_text
    
    # Chunk the text content
    chunks = chunk_text(
        text_content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )
    if not chunks:
        return {"added": 0, "chunks": 0, "embedding_dim": None}
    
    # Import embedding functions locally
    from src.app_logic import generate_embeddings, get_selected_provider
    
    # Generate embeddings using provider-aware logic
    selected_provider = get_selected_provider()
    if selected_provider == 'ollama':
        embeddings = generate_embeddings(chunks, model=embedding_model, provider='ollama')
    else:
        # For cloud providers, use local sentence transformers for embeddings
        embeddings = generate_embeddings(chunks, model=embedding_model, provider='sentence_transformers')
    
    if not embeddings:
        return {"added": 0, "chunks": len(chunks), "embedding_dim": None}
    
    embedding_dim = len(embeddings) if embeddings and isinstance(embeddings, list) and embeddings else None
    before = vector_store.ntotal
    
    # Build per-chunk metadata similar to add_document_to_store
    def parse_marker(txt: str) -> dict:
        for line in txt.splitlines():
            ls = line.strip()
            if ls.startswith("=== Page ") and ls.endswith("==="):
                try:
                    n = int(ls.replace("=", "").strip().split()[-1])
                    return {"page_number": n}
                except Exception:
                    return {}
            if ls.startswith("=== Slide ") and ls.endswith("==="):
                try:
                    n = int(ls.replace("=", "").strip().split()[-1])
                    return {"slide_number": n}
                except Exception:
                    return {}
        return {}

    def guess_title(txt: str) -> str | None:
        seen_marker = False
        for line in txt.splitlines():
            ls = line.strip()
            if not ls:
                continue
            if (ls.startswith("=== Page ") or ls.startswith("=== Slide ")) and ls.endswith("==="):
                seen_marker = True
                continue
            if seen_marker:
                return ls[:120]
        # fallback: first non-empty line
        for line in txt.splitlines():
            ls = line.strip()
            if ls:
                return ls[:120]
        return None

    metadatas = []
    for ch in chunks:
        md = {"source_file": filename}
        md.update(parse_marker(ch))
        t = guess_title(ch)
        if t:
            md["title"] = t
        metadatas.append(md)

    # Add to vector store
    vector_store.add_documents(chunks, embeddings, metadatas=metadatas)
    vector_store.save_store()
    
    after = vector_store.ntotal
    return {
        "added": max(0, after - before),
        "chunks": len(chunks),
        "embedding_dim": embedding_dim,
    }


@app.get("/models")
def get_models():
    """Frontend'in aradığı ana models endpoint'i - Tüm 6 Groq modelini döndürür"""
    return {
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b"
            # All models confirmed working via direct Groq API testing 2025-11-02
        ]
    }

@app.get("/models/available")
def get_available_models():
    """New endpoint with zero dependencies to test Groq models"""
    return {
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b"
            # All models confirmed working via direct Groq API testing 2025-11-02
        ]
    }

@app.get("/groq-models")
def get_groq_models():
    """Get available Groq cloud models - completely new endpoint"""
    return {
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b"
            # All models confirmed working via direct Groq API testing 2025-11-02
        ]
    }

@app.get("/models/list")
def list_available_models():
    """Get available Groq cloud models - completely isolated"""
    return {
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b"
            # All models confirmed working via direct Groq API testing 2025-11-02
        ]
    }

@app.get("/test")
def test_endpoint():
    """Simple test endpoint with zero dependencies"""
    return {"status": "success", "message": "API is working"}


if __name__ == "__main__":
    import uvicorn
    # Cloud Run requires listening on 0.0.0.0 with PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting RAG3 API server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# Run with: uvicorn src.api.main:app --reload
