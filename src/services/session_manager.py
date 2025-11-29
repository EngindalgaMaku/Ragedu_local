"""
Professional Session Management System
Profesyonel Oturum Yönetim Sistemi

Bu modül öğretmenlerin ders oturumlarını veritabanı tabanlı olarak yönetmesine olanak sağlar:
- Oturum oluşturma, metadata ve validasyon ile
- Profesyonel kaydetme/export işlevleri  
- Güvenli silme, onay ve backup ile
- Oturum kategorizasyonu ve arama
- Import/Export yetenekleri
- Oturum analitikleri ve kullanım istatistikleri
"""

import os
import json
import sqlite3
import shutil
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
from contextlib import contextmanager

import logging
# from src.utils.logger import get_logger
from src.config import config, get_storage_config
from src.utils.cloud_storage_manager import cloud_storage_manager


class SessionStatus(Enum):
    """Oturum durumu"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DRAFT = "draft"
    COMPLETED = "completed"
    SUSPENDED = "suspended"


class SessionCategory(Enum):
    """Oturum kategorileri"""
    GENERAL = "general"
    SCIENCE = "science"
    MATHEMATICS = "mathematics"
    LANGUAGE = "language"
    SOCIAL_STUDIES = "social_studies"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    COMPUTER_SCIENCE = "computer_science"
    ART = "art"
    MUSIC = "music"
    PHYSICAL_EDUCATION = "physical_education"
    RESEARCH = "research"
    EXAM_PREP = "exam_prep"


@dataclass
class SessionMetadata:
    """Oturum metadata bilgileri"""
    session_id: str
    name: str
    description: str
    category: SessionCategory
    status: SessionStatus
    created_by: str
    created_at: str
    updated_at: str
    last_accessed: str
    grade_level: str
    subject_area: str
    learning_objectives: List[str]
    tags: List[str]
    document_count: int = 0
    total_chunks: int = 0
    query_count: int = 0
    student_entry_count: int = 0
    avg_response_time: float = 0.0
    user_rating: float = 0.0
    notes: str = ""
    is_public: bool = False
    collaborators: List[str] = None
    backup_count: int = 0
    rag_settings: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.collaborators is None:
            self.collaborators = []


@dataclass
class SessionBackup:
    """Oturum yedekleme bilgileri"""
    backup_id: str
    session_id: str
    backup_path: str
    created_at: str
    backup_size: int
    description: str
    auto_created: bool = False


@dataclass 
class SessionExportData:
    """Oturum export verisi"""
    metadata: SessionMetadata
    vector_data: Dict[str, Any]
    performance_stats: Dict[str, Any]
    export_timestamp: str
    export_version: str = "1.0"


class ProfessionalSessionManager:
    """Profesyonel oturum yönetim sistemi"""
    
    def __init__(self, db_path: str = "data/analytics/sessions.db"):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Cloud-compatible setup
        self.is_cloud = config.is_cloud_environment()
        self.cloud_storage = cloud_storage_manager
        
        if self.is_cloud:
            # Use cloud storage manager for database path
            self.db_path = None  # Will be handled by cloud storage manager
            self.db_name = "sessions.db"
            
            # Cloud storage paths
            storage_config = get_storage_config()
            self.backup_dir = Path("/tmp/rag3/backups")
            self.export_dir = Path("/tmp/rag3/exports")
        else:
            # Local development setup
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_name = self.db_path.name
            
            # Local backup and export directories
            self.backup_dir = Path("data/backups/sessions")
            self.export_dir = Path("data/exports/sessions")
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Database initialization
        self._init_schema()
        
        # Auto-cleanup old backups (30+ days) - only for local
        if not self.is_cloud:
            self._cleanup_old_backups()
    
    @contextmanager
    def get_connection(self):
        """Cloud-compatible database connection context manager"""
        if self.is_cloud:
            # Use cloud storage manager for cloud deployments
            with self.cloud_storage.get_database_connection(self.db_name) as conn:
                yield conn
        else:
            # Use local SQLite for development
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Database operation failed: {e}")
                raise
            finally:
                conn.close()
    
    def _init_schema(self):
        """Veritabanı şemasını oluştur"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_by TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    grade_level TEXT,
                    subject_area TEXT,
                    learning_objectives TEXT,  -- JSON array
                    tags TEXT,                 -- JSON array
                    document_count INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    query_count INTEGER DEFAULT 0,
                    student_entry_count INTEGER DEFAULT 0,
                    avg_response_time REAL DEFAULT 0.0,
                    user_rating REAL DEFAULT 0.0,
                    notes TEXT,
                    is_public BOOLEAN DEFAULT 0,
                    collaborators TEXT,        -- JSON array
                    backup_count INTEGER DEFAULT 0,
                    rag_settings TEXT          -- JSON for teacher-defined RAG config
                )
            """)
            
            # Safe migration: add missing columns if needed
            try:
                cursor.execute("PRAGMA table_info(sessions)")
                cols = {row[1] for row in cursor.fetchall()}
                self.logger.info(f"Existing columns in sessions table: {cols}")
                
                if "rag_settings" not in cols:
                    self.logger.info("Adding missing column: rag_settings")
                    cursor.execute("ALTER TABLE sessions ADD COLUMN rag_settings TEXT")
                    conn.commit()
                    self.logger.info("Successfully added rag_settings column")
                
                if "student_entry_count" not in cols:
                    self.logger.info("Adding missing column: student_entry_count")
                    cursor.execute("ALTER TABLE sessions ADD COLUMN student_entry_count INTEGER DEFAULT 0")
                    conn.commit()
                    self.logger.info("Successfully added student_entry_count column")
                    
                # Verify columns after migration
                cursor.execute("PRAGMA table_info(sessions)")
                cols_after = {row[1] for row in cursor.fetchall()}
                self.logger.info(f"Columns after migration: {cols_after}")
            except Exception as e:
                self.logger.error(f"Failed to ensure missing columns: {e}", exc_info=True)
                conn.rollback()
                raise
            
            # Session backups table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_backups (
                    backup_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    backup_path TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    backup_size INTEGER,
                    description TEXT,
                    auto_created BOOLEAN DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Session activity log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_activity (
                    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    activity_type TEXT NOT NULL,  -- 'created', 'accessed', 'modified', 'backup', 'export'
                    description TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Session shares/collaborations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_shares (
                    share_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    shared_with TEXT NOT NULL,
                    permission_level TEXT DEFAULT 'read',  -- 'read', 'write', 'admin'
                    shared_by TEXT NOT NULL,
                    shared_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Document chunks table - stores text content of each generated chunk
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    chunk_metadata TEXT,  -- JSON format for additional chunk metadata
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Student entries table - track unique student access to sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    student_identifier TEXT NOT NULL,  -- user_id or username or "student"
                    first_entry DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_entry DATETIME DEFAULT CURRENT_TIMESTAMP,
                    entry_count INTEGER DEFAULT 1,
                    UNIQUE(session_id, student_identifier),
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_by ON sessions (created_by)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_category ON sessions (category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions (updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_activity_session_id ON session_activity (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_activity_timestamp ON session_activity (timestamp)")

            # Changelog table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS changelog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    date TEXT NOT NULL,
                    changes TEXT NOT NULL -- JSON array of strings
                )
            """)
            
            self.logger.info("Professional session management database schema initialized")
    
    def create_session(self, name: str, description: str, category: SessionCategory,
                      created_by: str, grade_level: str = "", subject_area: str = "",
                      learning_objectives: List[str] = None, tags: List[str] = None,
                      is_public: bool = False) -> SessionMetadata:
        """
        Profesyonel oturum oluştur - validasyon ve metadata ile
        
        Args:
            name: Oturum adı
            description: Açıklama  
            category: Kategori
            created_by: Oluşturan kişi
            grade_level: Sınıf seviyesi
            subject_area: Konu alanı
            learning_objectives: Öğrenme hedefleri
            tags: Etiketler
            is_public: Herkese açık mı
            
        Returns:
            SessionMetadata: Oluşturulan oturum metadatası
        """
        # Validation
        if not name.strip():
            raise ValueError("Session name cannot be empty")
        
        if len(name) > 100:
            raise ValueError("Session name too long (max 100 characters)")
        
        if self._session_exists(name, created_by):
            raise ValueError(f"Session '{name}' already exists for user {created_by}")
        
        # Generate unique session ID
        session_id = self._generate_session_id(name, created_by)
        
        # Create metadata
        now = datetime.now().isoformat()
        metadata = SessionMetadata(
            session_id=session_id,
            name=name.strip(),
            description=description.strip(),
            category=category,
            status=SessionStatus.DRAFT,  # Start as draft
            created_by=created_by,
            created_at=now,
            updated_at=now,
            last_accessed=now,
            grade_level=grade_level,
            subject_area=subject_area,
            learning_objectives=learning_objectives or [],
            tags=tags or [],
            is_public=is_public
        )
        
        # Save to database
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (
                    session_id, name, description, category, status, created_by,
                    created_at, updated_at, last_accessed, grade_level, subject_area,
                    learning_objectives, tags, is_public, collaborators, rag_settings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, metadata.name, metadata.description, category.value,
                metadata.status.value, created_by, now, now, now,
                grade_level, subject_area,
                json.dumps(learning_objectives or []),
                json.dumps(tags or []),
                is_public, json.dumps(metadata.collaborators or []), json.dumps(metadata.rag_settings or {})
            ))
        
        # Log activity
        self._log_activity(session_id, "created", f"Session '{name}' created", created_by)
        
        self.logger.info(f"Created professional session '{name}' with ID {session_id}")
        return metadata
    
    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Oturum metadatasını getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # First, ensure rag_settings column exists
            try:
                cursor.execute("PRAGMA table_info(sessions)")
                cols = {row[1] for row in cursor.fetchall()}
                if "rag_settings" not in cols:
                    self.logger.warning("rag_settings column missing, adding it now")
                    cursor.execute("ALTER TABLE sessions ADD COLUMN rag_settings TEXT")
                    conn.commit()
                    self.logger.info("Successfully added rag_settings column")
            except Exception as e:
                self.logger.error(f"Failed to ensure rag_settings column: {e}")
                # Continue anyway, we'll handle it in the query
            
            # Use explicit column list to handle missing columns gracefully
            try:
                cursor.execute("""
                    SELECT session_id, name, description, category, status, created_by, 
                           created_at, updated_at, last_accessed, grade_level, subject_area,
                           learning_objectives, tags, document_count, total_chunks, query_count,
                           student_entry_count, avg_response_time, user_rating, notes, 
                           is_public, collaborators, backup_count,
                           COALESCE(rag_settings, '') as rag_settings
                    FROM sessions WHERE session_id = ?
                """, (session_id,))
            except sqlite3.OperationalError as e:
                if "no such column: rag_settings" in str(e):
                    # Column still missing, try to add it again
                    self.logger.warning("rag_settings column still missing, attempting to add again")
                    try:
                        cursor.execute("ALTER TABLE sessions ADD COLUMN rag_settings TEXT")
                        conn.commit()
                        # Retry the query
                        cursor.execute("""
                            SELECT session_id, name, description, category, status, created_by, 
                                   created_at, updated_at, last_accessed, grade_level, subject_area,
                                   learning_objectives, tags, document_count, total_chunks, query_count,
                                   student_entry_count, avg_response_time, user_rating, notes, 
                                   is_public, collaborators, backup_count,
                                   COALESCE(rag_settings, '') as rag_settings
                            FROM sessions WHERE session_id = ?
                        """, (session_id,))
                    except Exception as e2:
                        self.logger.error(f"Failed to add rag_settings column: {e2}")
                        # Fallback: query without rag_settings
                        cursor.execute("""
                            SELECT session_id, name, description, category, status, created_by, 
                                   created_at, updated_at, last_accessed, grade_level, subject_area,
                                   learning_objectives, tags, document_count, total_chunks, query_count,
                                   student_entry_count, avg_response_time, user_rating, notes, 
                                   is_public, collaborators, backup_count
                            FROM sessions WHERE session_id = ?
                        """, (session_id,))
                else:
                    raise
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert to dict with column names
            columns = [desc[0] for desc in cursor.description]
            row_dict = dict(zip(columns, row))
            
            # Ensure rag_settings key exists even if column was missing
            if 'rag_settings' not in row_dict:
                row_dict['rag_settings'] = ''
                
            return self._row_to_metadata(row_dict)
    
    def update_session_metadata(self, session_id: str, **updates) -> bool:
        """Oturum metadatasını güncelle"""
        if not self._session_exists_by_id(session_id):
            return False
        
        allowed_fields = {
            'name', 'description', 'category', 'status', 'grade_level',
            'subject_area', 'learning_objectives', 'tags', 'notes',
            'is_public', 'user_rating', 'query_count', 'student_entry_count'
        }
        
        update_fields = []
        update_values = []
        
        for field, value in updates.items():
            if field in allowed_fields:
                if field in ['learning_objectives', 'tags']:
                    value = json.dumps(value if value else [])
                elif field == 'category' and hasattr(value, 'value'):
                    value = value.value
                elif field == 'status' and hasattr(value, 'value'):
                    value = value.value
                
                update_fields.append(f"{field} = ?")
                update_values.append(value)
        
        if not update_fields:
            return False
        
        update_fields.append("updated_at = ?")
        update_values.append(datetime.now().isoformat())
        update_values.append(session_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = f"UPDATE sessions SET {', '.join(update_fields)} WHERE session_id = ?"
            cursor.execute(query, update_values)
        
        # Log activity
        self._log_activity(session_id, "modified", f"Session metadata updated: {', '.join(updates.keys())}")
        
        return True
    
    def update_session_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status"""
        if not self._session_exists_by_id(session_id):
            return False
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET status = ?, updated_at = ?
                WHERE session_id = ?
            """, (status.value, datetime.now().isoformat(), session_id))
        
        # Log activity
        self._log_activity(
            session_id, "status_updated",
            f"Session status updated to: {status.value}"
        )
        
        return True
    
    def update_session_counts(self, session_id: str, document_count: int, total_chunks: int) -> bool:
        """Update session document and chunk counts"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET document_count = ?, total_chunks = ?, updated_at = ?
                WHERE session_id = ?
            """, (document_count, total_chunks, datetime.now().isoformat(), session_id))
        
        # Log activity
        self._log_activity(
            session_id, "counts_updated",
            f"Updated counts: {document_count} documents, {total_chunks} chunks"
        )
        
        return True
    
    def list_sessions(self, created_by: Optional[str] = None,
                     category: Optional[SessionCategory] = None,
                     status: Optional[SessionStatus] = None,
                     limit: int = 50) -> List[SessionMetadata]:
        """Oturumları listele"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT session_id, name, description, category, status, created_by, 
                       created_at, updated_at, last_accessed, grade_level, subject_area,
                       learning_objectives, tags, document_count, total_chunks, query_count,
                       student_entry_count, avg_response_time, user_rating, notes, 
                       is_public, collaborators, backup_count,
                       COALESCE(rag_settings, '') as rag_settings
                FROM sessions WHERE 1=1
            """
            params = []
            
            if created_by:
                query += " AND created_by = ?"
                params.append(created_by)
            
            if category:
                query += " AND category = ?"
                params.append(category.value)
                
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert rows to dicts
            columns = [desc[0] for desc in cursor.description]
            row_dicts = [dict(zip(columns, row)) for row in rows]
            
            return [self._row_to_metadata(row_dict) for row_dict in row_dicts]
    
    def search_sessions(self, query: str, created_by: Optional[str] = None) -> List[SessionMetadata]:
        """Oturumlarda arama yap"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            search_query = """
                SELECT session_id, name, description, category, status, created_by, 
                       created_at, updated_at, last_accessed, grade_level, subject_area,
                       learning_objectives, tags, document_count, total_chunks, query_count,
                       student_entry_count, avg_response_time, user_rating, notes, 
                       is_public, collaborators, backup_count,
                       COALESCE(rag_settings, '') as rag_settings
                FROM sessions 
                WHERE (name LIKE ? OR description LIKE ? OR tags LIKE ? OR notes LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"]
            
            if created_by:
                search_query += " AND created_by = ?"
                params.append(created_by)
            
            search_query += " ORDER BY updated_at DESC LIMIT 20"
            
            cursor.execute(search_query, params)
            rows = cursor.fetchall()
            
            # Convert rows to dicts
            columns = [desc[0] for desc in cursor.description]
            row_dicts = [dict(zip(columns, row)) for row in rows]
            
            return [self._row_to_metadata(row_dict) for row_dict in row_dicts]
    
    def create_backup(self, session_id: str, description: str = "", 
                     auto_created: bool = False) -> SessionBackup:
        """Oturum yedeği oluştur"""
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            raise ValueError(f"Session {session_id} not found")
        
        # Backup file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{session_id}_{timestamp}.zip"
        backup_path = self.backup_dir / backup_filename
        
        # Create backup ZIP
        vector_base_path = Path(f"data/vector_db/sessions/{session_id}")
        files_to_backup = [
            f"{vector_base_path}.index",
            f"{vector_base_path}.chunks", 
            f"{vector_base_path}.meta.jsonl"
        ]
        
        total_size = 0
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add session metadata (convert enums to serializable format)
            metadata_dict = self._metadata_to_dict(metadata)
            zipf.writestr("session_metadata.json", json.dumps(metadata_dict, indent=2))
            
            # Add vector files if they exist
            for file_path in files_to_backup:
                if Path(file_path).exists():
                    zipf.write(file_path, Path(file_path).name)
                    total_size += Path(file_path).stat().st_size
        
        # Create backup record
        backup_id = hashlib.md5(f"{session_id}_{timestamp}".encode()).hexdigest()
        backup = SessionBackup(
            backup_id=backup_id,
            session_id=session_id,
            backup_path=str(backup_path),
            created_at=datetime.now().isoformat(),
            backup_size=total_size,
            description=description or f"Backup of '{metadata.name}'",
            auto_created=auto_created
        )
        
        # Save backup record to database
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_backups 
                (backup_id, session_id, backup_path, created_at, backup_size, description, auto_created)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                backup_id, session_id, str(backup_path), backup.created_at,
                backup.backup_size, backup.description, auto_created
            ))
            
            # Update backup count
            cursor.execute(
                "UPDATE sessions SET backup_count = backup_count + 1 WHERE session_id = ?",
                (session_id,)
            )
        
        # Log activity
        self._log_activity(
            session_id, "backup",
            f"Backup created: {backup.description}", 
            metadata={'backup_id': backup_id, 'size': total_size}
        )
        
        self.logger.info(f"Created backup for session {session_id}: {backup_path}")
        return backup
    
    def delete_session(self, session_id: str, create_backup: bool = True,
                      deleted_by: Optional[str] = None) -> bool:
        """
        Oturumu güvenli şekilde sil
        
        Args:
            session_id: Silinecek oturum ID'si
            create_backup: Silmeden önce yedek oluştur
            deleted_by: Silen kişi
            
        Returns:
            bool: Silme işlemi başarılı mı
        """
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            return False
        
        try:
            # Create backup before deletion if requested
            if create_backup:
                self.create_backup(
                    session_id, 
                    f"Pre-deletion backup of '{metadata.name}'",
                    auto_created=True
                )
            
            # Delete vector store files
            vector_base_path = Path(f"data/vector_db/sessions/{session_id}")
            files_to_delete = [
                f"{vector_base_path}.index",
                f"{vector_base_path}.chunks",
                f"{vector_base_path}.meta.jsonl"
            ]
            
            for file_path in files_to_delete:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            
            # Delete from database (cascading will handle related records)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            # Log activity (before deletion for record keeping)
            self._log_activity(
                session_id, "deleted",
                f"Session '{metadata.name}' deleted",
                user_id=deleted_by,
                metadata={'backup_created': create_backup}
            )
            
            self.logger.info(f"Deleted session {session_id} ('{metadata.name}')")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def export_session(self, session_id: str, export_format: str = "json") -> str:
        """
        Oturumu export et
        
        Args:
            session_id: Export edilecek oturum ID'si  
            export_format: Export formatı ('json', 'zip')
            
        Returns:
            str: Export dosyası yolu
        """
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            raise ValueError(f"Session {session_id} not found")
        
        # Get performance stats
        performance_stats = self._get_session_performance_stats(session_id)
        
        # Prepare export data
        export_data = SessionExportData(
            metadata=metadata,
            vector_data=self._get_session_vector_data(session_id),
            performance_stats=performance_stats,
            export_timestamp=datetime.now().isoformat()
        )
        
        # Create export file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{metadata.name}_{timestamp}"
        
        if export_format == "json":
            export_path = self.export_dir / f"{export_filename}.json"
            with open(export_path, 'w', encoding='utf-8') as f:
                # Export data'yı JSON serializable hale getir
                export_dict = self._export_data_to_dict(export_data)
                json.dump(export_dict, f, indent=2, ensure_ascii=False)
                
        elif export_format == "zip":
            export_path = self.export_dir / f"{export_filename}.zip"
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add session data as JSON
                export_dict = self._export_data_to_dict(export_data)
                zipf.writestr("session_data.json",
                            json.dumps(export_dict, indent=2, ensure_ascii=False))
                
                # Add vector files if they exist
                vector_base_path = Path(f"data/vector_db/sessions/{session_id}")
                vector_files = [
                    f"{vector_base_path}.index",
                    f"{vector_base_path}.chunks",
                    f"{vector_base_path}.meta.jsonl"
                ]
                
                for file_path in vector_files:
                    if Path(file_path).exists():
                        zipf.write(file_path, f"vector_data/{Path(file_path).name}")
        
        # Log activity
        self._log_activity(
            session_id, "export",
            f"Session exported as {export_format}",
            metadata={'export_path': str(export_path), 'format': export_format}
        )
        
        self.logger.info(f"Exported session {session_id} to {export_path}")
        return str(export_path)
    
    def get_session_analytics(self, session_id: Optional[str] = None,
                            created_by: Optional[str] = None, 
                            days: int = 30) -> Dict[str, Any]:
        """Oturum analitiklerini getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            if session_id:
                # Belirli bir oturum için
                cursor.execute("""
                    SELECT COUNT(*) as activity_count
                    FROM session_activity 
                    WHERE session_id = ? AND timestamp > ?
                """, (session_id, since_date))
                activity_stats = cursor.fetchone()
                
                metadata = self.get_session_metadata(session_id)
                return {
                    "session_metadata": asdict(metadata) if metadata else None,
                    "activity_count": activity_stats[0] if activity_stats else 0,
                    "performance_stats": self._get_session_performance_stats(session_id)
                }
            else:
                # Genel istatistikler
                base_query = "SELECT COUNT(*) as count FROM sessions WHERE created_at > ?"
                base_params = [since_date]
                
                if created_by:
                    base_query += " AND created_by = ?"
                    base_params.append(created_by)
                
                cursor.execute(base_query, base_params)
                total_sessions = cursor.fetchone()[0]
                
                # Status breakdown
                status_query = base_query.replace("COUNT(*) as count", "status, COUNT(*) as count") + " GROUP BY status"
                cursor.execute(status_query, base_params)
                status_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Category breakdown  
                category_query = base_query.replace("COUNT(*) as count", "category, COUNT(*) as count") + " GROUP BY category"
                cursor.execute(category_query, base_params)
                category_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_sessions": total_sessions,
                    "status_breakdown": status_breakdown,
                    "category_breakdown": category_breakdown,
                    "period_days": days
                }
    
    def _session_exists(self, name: str, created_by: str) -> bool:
        """Aynı isimde oturum var mı kontrol et"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM sessions WHERE name = ? AND created_by = ?", 
                (name, created_by)
            )
            return cursor.fetchone() is not None
    
    def _session_exists_by_id(self, session_id: str) -> bool:
        """ID ile oturum var mı kontrol et"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            return cursor.fetchone() is not None
    
    def _generate_session_id(self, name: str, created_by: str) -> str:
        """Benzersiz oturum ID'si oluştur"""
        base = f"{name}_{created_by}_{datetime.now().timestamp()}"
        return hashlib.md5(base.encode()).hexdigest()
    
    def _row_to_metadata(self, row) -> SessionMetadata:
        """Database row'unu SessionMetadata'ya çevir"""
        # row is already a dict from get_session_metadata
        if isinstance(row, dict):
            row_dict = row
        else:
            # Fallback: if it's a tuple or Row object, convert to dict
            row_dict = dict(row) if hasattr(row, 'keys') else {}
        
        # Safely handle rag_settings - use .get() to avoid KeyError
        rag_settings_value = row_dict.get('rag_settings', '')
        if rag_settings_value:
            try:
                if isinstance(rag_settings_value, str):
                    rag_settings = json.loads(rag_settings_value) if rag_settings_value.strip() else None
                else:
                    rag_settings = rag_settings_value
            except (json.JSONDecodeError, TypeError):
                rag_settings = None
        else:
            rag_settings = None
        
        return SessionMetadata(
            session_id=row_dict['session_id'],
            name=row_dict['name'],
            description=row_dict.get('description') or "",
            category=SessionCategory(row_dict['category']),
            status=SessionStatus(row_dict['status']),
            created_by=row_dict['created_by'],
            created_at=row_dict['created_at'],
            updated_at=row_dict['updated_at'],
            last_accessed=row_dict.get('last_accessed') or row_dict.get('updated_at'),
            grade_level=row_dict.get('grade_level') or "",
            subject_area=row_dict.get('subject_area') or "",
            learning_objectives=json.loads(row_dict.get('learning_objectives') or '[]'),
            tags=json.loads(row_dict.get('tags') or '[]'),
            document_count=row_dict.get('document_count', 0),
            total_chunks=row_dict.get('total_chunks', 0),
            query_count=row_dict.get('query_count', 0),
            avg_response_time=row_dict.get('avg_response_time', 0.0),
            user_rating=row_dict.get('user_rating', 0.0),
            notes=row_dict.get('notes') or "",
            is_public=bool(row_dict.get('is_public', 0)),
            collaborators=json.loads(row_dict.get('collaborators') or '[]'),
            backup_count=row_dict.get('backup_count', 0),
            rag_settings=rag_settings,
            student_entry_count=row_dict.get('student_entry_count', 0)
        )
    
    def _log_activity(self, session_id: str, activity_type: str, description: str,
                     user_id: Optional[str] = None, metadata: Dict = None):
        """Oturum etkinliğini logla"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_activity 
                (session_id, activity_type, description, timestamp, user_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id, activity_type, description, datetime.now().isoformat(),
                user_id, json.dumps(metadata or {})
            ))
    
    def _get_session_performance_stats(self, session_id: str) -> Dict[str, Any]:
        """Oturum performans istatistiklerini getir"""
        # Bu fonksiyon vector store ve experiment database ile entegre edilebilir
        return {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "last_activity": None,
            "document_count": 0,
            "chunk_count": 0
        }
    
    def _get_session_vector_data(self, session_id: str) -> Dict[str, Any]:
        """Oturum vektör verilerini getir"""
        vector_base_path = Path(f"data/vector_db/sessions/{session_id}")
        
        return {
            "has_index": (Path(f"{vector_base_path}.index")).exists(),
            "has_chunks": (Path(f"{vector_base_path}.chunks")).exists(), 
            "has_metadata": (Path(f"{vector_base_path}.meta.jsonl")).exists(),
            "vector_path": str(vector_base_path)
        }
    
    def _cleanup_old_backups(self, days: int = 30):
        """Eski yedekleri temizle"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT backup_path FROM session_backups 
                    WHERE created_at < ? AND auto_created = 1
                """, (cutoff_date.isoformat(),))
                
                old_backups = cursor.fetchall()
                
                for backup in old_backups:
                    backup_path = Path(backup[0])
                    if backup_path.exists():
                        backup_path.unlink()
                
                # Remove from database
                cursor.execute("""
                    DELETE FROM session_backups 
                    WHERE created_at < ? AND auto_created = 1
                """, (cutoff_date.isoformat(),))
                
                if old_backups:
                    self.logger.info(f"Cleaned up {len(old_backups)} old backups")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
    
    def _metadata_to_dict(self, metadata: SessionMetadata) -> Dict[str, Any]:
        """SessionMetadata'yı JSON serializable dict'e çevir"""
        metadata_dict = asdict(metadata)
        
        # Enum'ları string'e çevir
        if 'category' in metadata_dict:
            metadata_dict['category'] = metadata.category.value
        if 'status' in metadata_dict:
            metadata_dict['status'] = metadata.status.value
            
        return metadata_dict
    
    def _export_data_to_dict(self, export_data: SessionExportData) -> Dict[str, Any]:
        """SessionExportData'yı JSON serializable dict'e çevir"""
        export_dict = asdict(export_data)
        
        # Metadata içindeki enum'ları çevir
        if 'metadata' in export_dict:
            metadata_dict = export_dict['metadata']
            if isinstance(metadata_dict.get('category'), SessionCategory):
                metadata_dict['category'] = metadata_dict['category'].value
            if isinstance(metadata_dict.get('status'), SessionStatus):
                metadata_dict['status'] = metadata_dict['status'].value
        
        return export_dict

    def save_session_rag_settings(self, session_id: str, settings: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Save teacher-defined RAG settings for a session."""
        if not self._session_exists_by_id(session_id):
            return False
        import logging
        logger = logging.getLogger(__name__)
        print(f"[SESSION MANAGER] Saving RAG settings for session {session_id}: {settings}")
        logger.info(f"[SESSION MANAGER] Saving RAG settings for session {session_id}: {settings}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            settings_json = json.dumps(settings or {})
            print(f"[SESSION MANAGER] JSON to save: {settings_json}")
            logger.info(f"[SESSION MANAGER] JSON to save: {settings_json}")
            cursor.execute(
                "UPDATE sessions SET rag_settings = ?, updated_at = ? WHERE session_id = ?",
                (settings_json, datetime.now().isoformat(), session_id)
            )
            conn.commit()  # Explicit commit
            # Verify the save
            cursor.execute("SELECT rag_settings FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                print(f"[SESSION MANAGER] Verified saved settings: {row[0]}")
                logger.info(f"[SESSION MANAGER] Verified saved settings: {row[0]}")
        self._log_activity(session_id, "rag_settings_updated", "RAG settings saved", user_id, settings)
        return True

    def get_session_rag_settings(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rag_settings FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            try:
                return json.loads(row[0]) if row[0] else None
            except Exception:
                return None
    
    def track_student_entry(self, session_id: str, student_identifier: str) -> bool:
        """Track unique student entry to a session"""
        if not student_identifier or not session_id:
            return False
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Try to insert or update student entry
                cursor.execute("""
                    INSERT INTO student_entries (session_id, student_identifier, first_entry, last_entry, entry_count)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
                    ON CONFLICT(session_id, student_identifier)
                    DO UPDATE SET
                        last_entry = CURRENT_TIMESTAMP,
                        entry_count = entry_count + 1
                """, (session_id, student_identifier))
                
                # Update session's student_entry_count with unique student count
                cursor.execute("""
                    UPDATE sessions
                    SET student_entry_count = (
                        SELECT COUNT(DISTINCT student_identifier)
                        FROM student_entries
                        WHERE session_id = ?
                    ),
                    updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (session_id, session_id))
                
                return True
        except Exception as e:
            self.logger.error(f"Failed to track student entry for session {session_id}: {e}")
            return False
    
    def get_student_entry_stats(self, session_id: str) -> Dict[str, Any]:
        """Get student entry statistics for a session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        COUNT(DISTINCT student_identifier) as unique_students,
                        SUM(entry_count) as total_entries,
                        MAX(last_entry) as last_student_entry
                    FROM student_entries
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "unique_students": row["unique_students"] or 0,
                        "total_entries": row["total_entries"] or 0,
                        "last_student_entry": row["last_student_entry"]
                    }
                return {"unique_students": 0, "total_entries": 0, "last_student_entry": None}
        except Exception as e:
            self.logger.error(f"Failed to get student entry stats for session {session_id}: {e}")
            return {"unique_students": 0, "total_entries": 0, "last_student_entry": None}


# Global instance
professional_session_manager = ProfessionalSessionManager()