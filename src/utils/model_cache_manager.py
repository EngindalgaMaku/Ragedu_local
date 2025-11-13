"""
Model Cache Manager for Marker PDF Processing
Handles persistent caching of large ML models to prevent repeated downloads
"""

import os
import logging
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import psutil
import threading
import time
import zipfile
import tempfile

logger = logging.getLogger(__name__)

# Try to import cloud storage manager for persistent cache in cloud environments
try:
    from .cloud_storage_manager import CloudStorageManager
    CLOUD_STORAGE_AVAILABLE = True
    logger.info("✅ CloudStorageManager imported for persistent model cache")
except ImportError as e:
    logger.warning(f"⚠️ CloudStorageManager not available: {e}")
    CLOUD_STORAGE_AVAILABLE = False

class ModelCacheManager:
    """
    Centralized model cache manager for Marker PDF processing.
    Handles persistent storage and retrieval of ML models to prevent repeated downloads.
    """
    
    def __init__(self):
        self.base_cache_dir = Path(os.getenv("MARKER_CACHE_DIR", "/app/models"))
        self.models_dir = self.base_cache_dir / "marker_models"
        self.cache_info_file = self.models_dir / "cache_info.json"
        self.lock = threading.Lock()
        
        # Initialize cloud storage manager for persistent cache
        self.cloud_storage = None
        self.is_cloud_environment = self._detect_cloud_environment()
        
        if self.is_cloud_environment and CLOUD_STORAGE_AVAILABLE:
            try:
                self.cloud_storage = CloudStorageManager()
                logger.info("☁️ Cloud storage initialized for persistent model cache")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize cloud storage: {e}")
                self.cloud_storage = None
        
        self._ensure_cache_directories()
        self._load_cache_info()
    
    def _ensure_cache_directories(self):
        """Ensure cache directories exist with proper permissions"""
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            # Set permissions for container environments
            if os.name != 'nt':  # Unix-like systems
                os.chmod(self.models_dir, 0o755)
                
            logger.info(f"📁 Model cache directory initialized: {self.models_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to create cache directory: {e}")
            # Fallback to temp directory
            import tempfile
            self.base_cache_dir = Path(tempfile.gettempdir()) / "marker_cache"
            self.models_dir = self.base_cache_dir / "marker_models"
            self.models_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"⚠️ Using fallback cache directory: {self.models_dir}")
    
    def _load_cache_info(self):
        """Load cache metadata information"""
        self.cache_info = {}
        try:
            if self.cache_info_file.exists():
                with open(self.cache_info_file, 'r') as f:
                    self.cache_info = json.load(f)
                logger.info(f"📋 Loaded cache info: {len(self.cache_info)} cached model sets")
        except Exception as e:
            logger.warning(f"⚠️ Could not load cache info: {e}")
            self.cache_info = {}
    
    def _save_cache_info(self):
        """Save cache metadata information"""
        try:
            with open(self.cache_info_file, 'w') as f:
                json.dump(self.cache_info, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"⚠️ Could not save cache info: {e}")
    
    def _get_model_cache_key(self, model_config: Dict[str, Any]) -> str:
        """Generate a unique cache key for model configuration"""
        # Create a hash based on model configuration
        config_str = json.dumps(model_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _detect_cloud_environment(self) -> bool:
        """Detect if running in cloud environment (Cloud Run)"""
        cloud_indicators = [
            os.getenv("GOOGLE_CLOUD_PROJECT"),
            os.getenv("K_SERVICE"),  # Cloud Run specific
            os.getenv("ENVIRONMENT") == "production"
        ]
        is_cloud = any(cloud_indicators)
        logger.info(f"🌐 Cloud environment detected: {is_cloud}")
        return is_cloud
    
    def get_cached_model_dict(self, force_download: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get cached model dictionary or download if not available
        
        Args:
            force_download: Force fresh download even if cache exists
            
        Returns:
            Model dictionary or None if failed
        """
        with self.lock:
            cache_key = "default_marker_models"
            cached_path = self.models_dir / cache_key
            
            # Try to restore from cloud storage first (if in cloud environment)
            if self.is_cloud_environment and self.cloud_storage and not force_download:
                try:
                    if self._restore_cache_from_cloud(cache_key, cached_path):
                        logger.info(f"☁️ Models restored from cloud storage")
                        return self._load_cached_models(cached_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to restore from cloud storage: {e}")
            
            # Check if models are already cached locally and valid
            if not force_download and cached_path.exists() and cache_key in self.cache_info:
                try:
                    logger.info(f"✅ Using locally cached models from: {cached_path}")
                    return self._load_cached_models(cached_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load cached models: {e}")
                    # Continue to download fresh models
            
            # Download and cache models
            model_dict = self._download_and_cache_models(cache_key, cached_path)
            
            # Backup to cloud storage (if in cloud environment and successful)
            if model_dict and self.is_cloud_environment and self.cloud_storage:
                try:
                    self._backup_cache_to_cloud(cache_key, cached_path)
                    logger.info(f"☁️ Models backed up to cloud storage")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to backup to cloud storage: {e}")
            
            return model_dict
    
    def _load_cached_models(self, cached_path: Path) -> Dict[str, Any]:
        """Load models from cache"""
        # CRITICAL: Environment variables should already be set at process start
        # Just verify they point to the right location
        expected_torch = str(cached_path / "torch")
        actual_torch = os.environ.get("TORCH_HOME", "")
        
        if actual_torch != expected_torch:
            logger.warning(f"⚠️ TORCH_HOME mismatch: expected {expected_torch}, got {actual_torch}")
            # Don't override - let the process-level env vars take precedence
        
        memory_before = self._get_memory_usage()
        logger.info(f"🔄 Loading cached models from environment... (Memory: {memory_before:.1f}MB)")
        logger.info(f"🔧 TORCH_HOME: {os.environ.get('TORCH_HOME')}")
        logger.info(f"🔧 TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
        
        # Verify cache exists
        torch_cache = os.environ.get("TORCH_HOME", "")
        if not os.path.exists(torch_cache):
            logger.error(f"❌ Cache directory missing: {torch_cache}")
            raise FileNotFoundError(f"Model cache not found: {torch_cache}")
        
        # Import here to use cached models
        from marker.models import create_model_dict
        
        # Load models with cache paths already set in environment
        model_dict = create_model_dict()
        
        memory_after = self._get_memory_usage()
        logger.info(f"✅ Environment-cached models loaded! (Memory: {memory_after:.1f}MB, +{memory_after-memory_before:.1f}MB)")
        
        return model_dict
    
    def _download_and_cache_models(self, cache_key: str, cached_path: Path) -> Optional[Dict[str, Any]]:
        """Download fresh models and cache them"""
        try:
            logger.info(f"📥 Downloading and caching Marker models...")
            memory_before = self._get_memory_usage()
            
            # Create cache subdirectories
            torch_cache = cached_path / "torch"
            hf_cache = cached_path / "huggingface"
            transformers_cache = cached_path / "transformers"
            hf_home = cached_path / "hf_home"
            
            for cache_dir in [torch_cache, hf_cache, transformers_cache, hf_home]:
                cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Set environment variables for model download locations
            os.environ["TORCH_HOME"] = str(torch_cache)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
            os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
            os.environ["HF_HOME"] = str(hf_home)
            
            # Additional marker-specific cache settings
            os.environ["MARKER_CACHE_DIR"] = str(cached_path)
            os.environ["MARKER_MODELS_DIR"] = str(cached_path / "marker_models")
            
            logger.info(f"🏗️ Model cache directories prepared at: {cached_path}")
            
            # Import and download models
            from marker.models import create_model_dict
            
            start_time = time.time()
            model_dict = create_model_dict()
            download_time = time.time() - start_time
            
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            # Update cache info
            self.cache_info[cache_key] = {
                "created_at": datetime.now().isoformat(),
                "cache_path": str(cached_path),
                "download_time_seconds": download_time,
                "memory_used_mb": memory_used,
                "model_count": len(model_dict) if model_dict else 0
            }
            self._save_cache_info()
            
            logger.info(f"✅ Models cached successfully!")
            logger.info(f"📊 Download time: {download_time:.1f}s, Memory used: {memory_used:.1f}MB")
            logger.info(f"📁 Cache location: {cached_path}")
            
            return model_dict
            
        except Exception as e:
            logger.error(f"❌ Failed to download and cache models: {e}")
            logger.error(f"📋 Error details: {str(e)}")
            
            # Clean up partial cache
            if cached_path.exists():
                try:
                    shutil.rmtree(cached_path)
                except:
                    pass
            
            return None
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cached models"""
        with self.lock:
            if cache_key:
                cached_path = self.models_dir / cache_key
                if cached_path.exists():
                    shutil.rmtree(cached_path)
                    if cache_key in self.cache_info:
                        del self.cache_info[cache_key]
                        self._save_cache_info()
                    logger.info(f"🗑️ Cleared cache for: {cache_key}")
            else:
                # Clear all caches
                if self.models_dir.exists():
                    shutil.rmtree(self.models_dir)
                    self._ensure_cache_directories()
                self.cache_info = {}
                self._save_cache_info()
                logger.info(f"🗑️ Cleared all model caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = 0
        if self.models_dir.exists():
            for path in self.models_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        
        return {
            "cache_directory": str(self.models_dir),
            "total_cache_size_mb": total_size / (1024 * 1024),
            "cached_model_sets": len(self.cache_info),
            "cache_info": self.cache_info,
            "is_cloud_environment": self.is_cloud_environment,
            "cloud_storage_available": self.cloud_storage is not None,
            "disk_usage": {
                "total_gb": shutil.disk_usage(self.models_dir).total / (1024**3),
                "used_gb": shutil.disk_usage(self.models_dir).used / (1024**3),
                "free_gb": shutil.disk_usage(self.models_dir).free / (1024**3)
            }
        }
    
    def _backup_cache_to_cloud(self, cache_key: str, cached_path: Path):
        """Backup model cache to cloud storage as compressed archive"""
        if not self.cloud_storage or not cached_path.exists():
            return
        
        try:
            logger.info(f"☁️ Backing up model cache to cloud storage...")
            
            # Create temporary zip file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                zip_path = tmp_zip.name
            
            # Compress cache directory
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in cached_path.rglob('*'):
                    if file_path.is_file():
                        # Create relative path for archive
                        arc_path = file_path.relative_to(cached_path)
                        zipf.write(file_path, str(arc_path))
            
            # Get compressed size
            zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            logger.info(f"📦 Cache compressed to {zip_size_mb:.1f}MB")
            
            # Upload to cloud storage
            cloud_cache_key = f"model_cache/{cache_key}.zip"
            
            with open(zip_path, 'rb') as zip_file:
                zip_content = zip_file.read()
            
            # Use cloud storage manager's blob upload method
            success = self.cloud_storage.upload_blob(cloud_cache_key, zip_content)
            
            if success:
                logger.info(f"✅ Model cache backed up to cloud: {cloud_cache_key}")
                
                # Update cache info with cloud backup info
                if cache_key in self.cache_info:
                    self.cache_info[cache_key]["cloud_backup"] = {
                        "cloud_key": cloud_cache_key,
                        "backup_time": datetime.now().isoformat(),
                        "compressed_size_mb": zip_size_mb
                    }
                    self._save_cache_info()
            else:
                logger.warning(f"⚠️ Failed to backup model cache to cloud")
            
            # Clean up temp file
            os.unlink(zip_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to backup cache to cloud: {e}")
            # Clean up temp file if it exists
            if 'zip_path' in locals() and os.path.exists(zip_path):
                os.unlink(zip_path)
    
    def _restore_cache_from_cloud(self, cache_key: str, cached_path: Path) -> bool:
        """Restore model cache from cloud storage"""
        if not self.cloud_storage:
            return False
        
        try:
            cloud_cache_key = f"model_cache/{cache_key}.zip"
            logger.info(f"☁️ Checking for cached models in cloud storage: {cloud_cache_key}")
            
            # Download compressed cache from cloud storage
            zip_content = self.cloud_storage.download_blob(cloud_cache_key)
            
            if not zip_content:
                logger.info(f"📭 No cached models found in cloud storage")
                return False
            
            logger.info(f"📥 Downloading cached models from cloud ({len(zip_content)/1024/1024:.1f}MB)")
            
            # Create temporary zip file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                tmp_zip.write(zip_content)
                zip_path = tmp_zip.name
            
            # Extract cache to local directory
            cached_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(cached_path)
            
            # Clean up temp file
            os.unlink(zip_path)
            
            # Set proper permissions
            if os.name != 'nt':  # Unix-like systems
                for file_path in cached_path.rglob('*'):
                    if file_path.is_file():
                        os.chmod(file_path, 0o644)
                    elif file_path.is_dir():
                        os.chmod(file_path, 0o755)
            
            # Update cache info
            self.cache_info[cache_key] = {
                "restored_from_cloud": True,
                "restored_at": datetime.now().isoformat(),
                "cache_path": str(cached_path),
                "cloud_key": cloud_cache_key
            }
            self._save_cache_info()
            
            logger.info(f"✅ Model cache restored from cloud storage to: {cached_path}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to restore cache from cloud: {e}")
            # Clean up partial extraction
            if cached_path.exists():
                try:
                    shutil.rmtree(cached_path)
                except:
                    pass
            # Clean up temp file
            if 'zip_path' in locals() and os.path.exists(zip_path):
                os.unlink(zip_path)
            return False

# Global instance
_cache_manager = None

def get_model_cache_manager() -> ModelCacheManager:
    """Get the global model cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ModelCacheManager()
    return _cache_manager

def get_cached_marker_models(force_download: bool = False) -> Optional[Dict[str, Any]]:
    """
    Get cached Marker models with automatic download if needed
    
    Args:
        force_download: Force fresh download even if cache exists
        
    Returns:
        Model dictionary or None if failed
    """
    return get_model_cache_manager().get_cached_model_dict(force_download)

if __name__ == "__main__":
    # Test the cache manager
    cache_manager = get_model_cache_manager()
    print("📋 Cache Manager Test")
    
    # Get cache stats
    stats = cache_manager.get_cache_stats()
    print(f"📊 Cache Stats: {stats}")
    
    # Test model loading
    print("🔄 Testing model cache...")
    models = get_cached_marker_models()
    
    if models:
        print(f"✅ Models loaded successfully! Count: {len(models)}")
    else:
        print("❌ Failed to load models")
    
    # Final stats
    final_stats = cache_manager.get_cache_stats()
    print(f"📊 Final Stats: {final_stats}")