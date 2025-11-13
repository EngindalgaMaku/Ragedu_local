"""
Caching utilities for improved performance.
"""

import os
import json
import hashlib
import time
from typing import Any, Optional
from pathlib import Path

class SimpleCache:
    """
    Simple file-based cache implementation.
    """
    
    def __init__(self, cache_dir: str = "data/cache", ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, file_path: Path) -> bool:
        """Check if cache file is expired."""
        if not file_path.exists():
            return True
        
        file_age = time.time() - file_path.stat().st_mtime
        return file_age > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        
        if self._is_expired(cache_path):
            # Clean up expired cache
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError:
                    pass
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False)
            return True
        except (json.JSONEncodeError, OSError):
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            return True
        except OSError:
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            return True
        except OSError:
            return False


# Global cache instance
_cache_instance: Optional[SimpleCache] = None

def get_cache(cache_dir: str = "data/cache", ttl: int = 3600) -> SimpleCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SimpleCache(cache_dir, ttl)
    return _cache_instance