"""
Cloud Storage Manager for RAG3
Handles persistent database storage for stateless Cloud Run deployments
"""

import os
import sqlite3
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from contextlib import contextmanager
import threading

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

from src.config import config

class CloudStorageManager:
    """
    Manages persistent database storage for cloud deployments.
    Syncs SQLite database with Google Cloud Storage to handle Cloud Run restarts.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Enhanced cloud detection - check multiple indicators
        google_cloud_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        cloud_run_service = os.getenv('K_SERVICE')  # Cloud Run specific
        environment_prod = os.getenv('ENVIRONMENT') == 'production'
        
        # We're in cloud if any of these are true
        self.is_cloud = bool(google_cloud_project) or bool(cloud_run_service) or environment_prod or config.is_cloud_environment()
        
        if self.is_cloud:
            self.logger.info(f"Detected cloud environment - GCP Project: {google_cloud_project}, Cloud Run: {bool(cloud_run_service)}")
        else:
            self.logger.info("Detected local development environment")
        
        # Local paths
        self.local_db_dir = Path('/tmp/rag3/databases')
        self.local_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Cloud storage client
        self._storage_client = None
        self._bucket = None
        
        # Thread-safe database operations
        self._db_lock = threading.RLock()
        
        # Initialize cloud storage if needed
        if self.is_cloud:
            self._init_cloud_storage()
    
    def _init_cloud_storage(self):
        """Initialize Google Cloud Storage client"""
        if not GCS_AVAILABLE:
            self.logger.error("Google Cloud Storage not available - install google-cloud-storage")
            return
        
        try:
            self._storage_client = storage.Client()
            bucket_name = config.database_config.get('storage_bucket', 'rag3-data')
            self._bucket = self._storage_client.bucket(bucket_name)
            
            # Test bucket access
            if not self._bucket.exists():
                self.logger.warning(f"GCS bucket '{bucket_name}' does not exist - creating it")
                self._bucket = self._storage_client.create_bucket(bucket_name)
            
            self.logger.info(f"Cloud storage initialized: {bucket_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cloud storage: {e}")
            self._storage_client = None
            self._bucket = None
    
    def get_database_path(self, db_name: str = "sessions.db") -> str:
        """
        Get the local database path, downloading from cloud storage if needed.
        
        Args:
            db_name: Name of the database file
            
        Returns:
            str: Local path to database file
        """
        local_db_path = self.local_db_dir / db_name
        
        if self.is_cloud and self._bucket:
            # Try to download latest database from cloud storage
            self._download_database(db_name, str(local_db_path))
        
        return str(local_db_path)
    
    def _download_database(self, db_name: str, local_path: str) -> bool:
        """Download database from cloud storage"""
        if not self._bucket:
            return False
        
        try:
            storage_path = config.database_config.get('storage_path', 'databases/') + db_name
            blob = self._bucket.blob(storage_path)
            
            if blob.exists():
                self.logger.info(f"Downloading database from GCS: {storage_path}")
                blob.download_to_filename(local_path)
                self.logger.info(f"Database downloaded to: {local_path}")
                return True
            else:
                self.logger.info(f"No existing database in GCS: {storage_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to download database: {e}")
            return False
    
    def sync_database(self, db_name: str = "sessions.db") -> bool:
        """
        Sync local database to cloud storage.
        Called after database modifications.
        
        Args:
            db_name: Name of the database file
            
        Returns:
            bool: Success status
        """
        if not self.is_cloud or not self._bucket:
            return True  # No sync needed for local development
        
        local_db_path = self.local_db_dir / db_name
        
        if not local_db_path.exists():
            self.logger.warning(f"Database file does not exist: {local_db_path}")
            return False
        
        try:
            with self._db_lock:
                # Create a backup copy for upload
                temp_path = local_db_path.with_suffix('.tmp')
                shutil.copy2(local_db_path, temp_path)
                
                try:
                    storage_path = config.database_config.get('storage_path', 'databases/') + db_name
                    blob = self._bucket.blob(storage_path)
                    
                    # Upload with metadata
                    blob.metadata = {
                        'uploaded_at': str(int(time.time())),
                        'source': 'rag3-api',
                        'version': '1.0'
                    }
                    
                    blob.upload_from_filename(str(temp_path))
                    self.logger.info(f"Database synced to GCS: {storage_path}")
                    return True
                    
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                        
        except Exception as e:
            self.logger.error(f"Failed to sync database: {e}")
            return False
    
    @contextmanager
    def get_database_connection(self, db_name: str = "sessions.db", timeout: float = 30.0):
        """
        Context manager for database connections with automatic sync.
        
        Args:
            db_name: Name of the database file
            timeout: Connection timeout in seconds
            
        Yields:
            sqlite3.Connection: Database connection
        """
        # Get database path (downloads from cloud if needed)
        db_path = self.get_database_path(db_name)
        
        with self._db_lock:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.row_factory = sqlite3.Row
            
            try:
                yield conn
                conn.commit()
                
                # Sync to cloud after successful transaction
                if conn.in_transaction is False:  # Only sync if transaction completed
                    self.sync_database(db_name)
                    
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Database operation failed: {e}")
                raise
            finally:
                conn.close()
    
    def backup_database(self, db_name: str = "sessions.db", backup_name: Optional[str] = None) -> Optional[str]:
        """
        Create a backup of the database in cloud storage.
        
        Args:
            db_name: Name of the database file
            backup_name: Custom backup name (auto-generated if None)
            
        Returns:
            str: Backup path in cloud storage, or None if failed
        """
        if not self.is_cloud or not self._bucket:
            return None
        
        local_db_path = self.local_db_dir / db_name
        if not local_db_path.exists():
            return None
        
        try:
            if not backup_name:
                timestamp = int(time.time())
                backup_name = f"{db_name}.backup.{timestamp}"
            
            backup_path = f"backups/databases/{backup_name}"
            blob = self._bucket.blob(backup_path)
            
            blob.metadata = {
                'backup_created_at': str(int(time.time())),
                'original_db': db_name,
                'backup_type': 'manual'
            }
            
            blob.upload_from_filename(str(local_db_path))
            self.logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create database backup: {e}")
            return None
    
    def restore_database(self, backup_path: str, db_name: str = "sessions.db") -> bool:
        """
        Restore database from a backup in cloud storage.
        
        Args:
            backup_path: Path to backup file in cloud storage
            db_name: Target database name
            
        Returns:
            bool: Success status
        """
        if not self.is_cloud or not self._bucket:
            return False
        
        try:
            blob = self._bucket.blob(backup_path)
            if not blob.exists():
                self.logger.error(f"Backup not found: {backup_path}")
                return False
            
            local_db_path = self.local_db_dir / db_name
            
            with self._db_lock:
                # Download backup to local path
                blob.download_to_filename(str(local_db_path))
                
                # Sync the restored database back to current location
                self.sync_database(db_name)
            
            self.logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            return False
    
    def list_backups(self, db_name: str = "sessions.db") -> list:
        """List available database backups"""
        if not self.is_cloud or not self._bucket:
            return []
        
        try:
            prefix = f"backups/databases/{db_name}.backup."
            blobs = self._bucket.list_blobs(prefix=prefix)
            
            backups = []
            for blob in blobs:
                backups.append({
                    'name': blob.name,
                    'created': blob.time_created,
                    'size': blob.size,
                    'metadata': blob.metadata or {}
                })
            
            return sorted(backups, key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        stats = {
            'is_cloud': self.is_cloud,
            'gcs_available': GCS_AVAILABLE and self._bucket is not None,
            'local_db_dir': str(self.local_db_dir),
            'local_databases': []
        }
        
        # List local databases
        if self.local_db_dir.exists():
            for db_file in self.local_db_dir.glob('*.db'):
                stats['local_databases'].append({
                    'name': db_file.name,
                    'size': db_file.stat().st_size,
                    'modified': db_file.stat().st_mtime
                })
        
        # Add cloud storage stats if available
        if self.is_cloud and self._bucket:
            try:
                bucket_info = {
                    'name': self._bucket.name,
                    'location': self._bucket.location,
                    'storage_class': self._bucket.storage_class
                }
                stats['cloud_storage'] = bucket_info
            except Exception as e:
                stats['cloud_storage_error'] = str(e)
        
        return stats

    # Markdown File Management
    def save_markdown_file(self, filename: str, content: str) -> bool:
        """
        Save markdown file to cloud storage if in cloud environment.
        
        Args:
            filename: Name of the markdown file
            content: Markdown content to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_cloud or not self._bucket:
            # For local development, save to local directory
            try:
                local_markdown_dir = Path("data/markdown")
                local_markdown_dir.mkdir(parents=True, exist_ok=True)
                
                with open(local_markdown_dir / filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            except Exception as e:
                self.logger.error(f"Failed to save markdown locally: {e}")
                return False
        
        try:
            # Save to Google Cloud Storage
            markdown_path = f"markdown/{filename}"
            blob = self._bucket.blob(markdown_path)
            blob.upload_from_string(content, content_type='text/markdown')
            
            self.logger.info(f"Markdown file saved to GCS: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save markdown to GCS: {e}")
            return False

    def list_markdown_files(self) -> List[str]:
        """
        List all markdown files from cloud storage or local directory.
        
        Returns:
            List[str]: List of markdown filenames
        """
        if not self.is_cloud or not self._bucket:
            # For local development, read from local directory
            try:
                local_markdown_dir = Path("data/markdown")
                local_markdown_dir.mkdir(parents=True, exist_ok=True)
                
                if local_markdown_dir.exists():
                    md_files = [f.name for f in local_markdown_dir.glob("*.md")]
                    return sorted(md_files)
                return []
            except Exception as e:
                self.logger.error(f"Failed to list local markdown files: {e}")
                return []
        
        try:
            # List from Google Cloud Storage
            prefix = "markdown/"
            blobs = self._bucket.list_blobs(prefix=prefix)
            
            md_files = []
            for blob in blobs:
                if blob.name.endswith('.md'):
                    # Extract filename from path
                    filename = blob.name.replace(prefix, '')
                    if filename:  # Skip empty names
                        md_files.append(filename)
            
            return sorted(md_files)
            
        except Exception as e:
            self.logger.error(f"Failed to list markdown files from GCS: {e}")
            return []

    def get_markdown_file_content(self, filename: str) -> Optional[str]:
        """
        Get markdown file content from cloud storage or local directory.
        
        Args:
            filename: Name of the markdown file
            
        Returns:
            Optional[str]: File content or None if not found
        """
        if not self.is_cloud or not self._bucket:
            # For local development, read from local directory
            try:
                local_markdown_dir = Path("data/markdown")
                file_path = local_markdown_dir / filename
                
                if file_path.exists() and file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                return None
            except Exception as e:
                self.logger.error(f"Failed to read local markdown file: {e}")
                return None
        
        try:
            # Read from Google Cloud Storage
            markdown_path = f"markdown/{filename}"
            blob = self._bucket.blob(markdown_path)
            
            if not blob.exists():
                return None
                
            content = blob.download_as_text(encoding='utf-8')
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to read markdown from GCS: {e}")
            return None

    def delete_markdown_file(self, filename: str) -> bool:
        """
        Delete markdown file from cloud storage or local directory.
        
        Args:
            filename: Name of the markdown file to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_cloud or not self._bucket:
            # For local development, delete from local directory
            try:
                local_markdown_dir = Path("data/markdown")
                file_path = local_markdown_dir / filename
                
                if file_path.exists() and file_path.is_file():
                    file_path.unlink()
                    return True
                return False
            except Exception as e:
                self.logger.error(f"Failed to delete local markdown file: {e}")
                return False
        
        try:
            # Delete from Google Cloud Storage
            markdown_path = f"markdown/{filename}"
            blob = self._bucket.blob(markdown_path)
            
            if blob.exists():
                blob.delete()
                self.logger.info(f"Markdown file deleted from GCS: {filename}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete markdown from GCS: {e}")
            return False

    def delete_all_markdown_files(self) -> int:
        """Delete all markdown files. Returns number of files deleted."""
        deleted_count = 0
        try:
            if not self.is_cloud or not self._bucket:
                local_markdown_dir = Path("data/markdown")
                if local_markdown_dir.exists():
                    for f in local_markdown_dir.glob("*.md"):
                        try:
                            f.unlink()
                            deleted_count += 1
                        except Exception:
                            continue
                return deleted_count

            # Cloud: delete all blobs under markdown/
            prefix = "markdown/"
            blobs = list(self._bucket.list_blobs(prefix=prefix))
            for blob in blobs:
                if blob.name.endswith(".md"):
                    try:
                        blob.delete()
                        deleted_count += 1
                    except Exception:
                        continue
            return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to delete all markdown files: {e}")
            return deleted_count

# Global instance
cloud_storage_manager = CloudStorageManager()