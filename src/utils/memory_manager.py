"""
Memory Management Utilities for PDF Processing
Prevents memory crashes and manages resource cleanup
"""

import os
import gc
import logging
import threading
import time
import psutil
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Advanced memory management for PDF processing operations.
    Monitors memory usage and enforces limits to prevent container crashes.
    """
    
    def __init__(self):
        self.max_memory_mb = int(os.getenv("MARKER_MAX_MEMORY_MB", "15000"))  # 15GB out of 16GB Cloud Run limit
        self.warning_threshold = 0.80  # Warn at 80% of limit (12GB)
        self.critical_threshold = 0.90  # Force cleanup at 90% of limit (13.5GB)
        self.monitoring_active = False
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.memory_callbacks = []  # Functions to call when memory gets high
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size (physical memory)
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "system_total_mb": psutil.virtual_memory().total / 1024 / 1024,
                "system_used_percent": psutil.virtual_memory().percent
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0, "available_mb": 0, "system_total_mb": 0, "system_used_percent": 0}
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is at critical levels"""
        memory_info = self.get_memory_usage()
        current_mb = memory_info["rss_mb"]
        
        # Check both absolute limit and system memory pressure
        absolute_critical = current_mb > (self.max_memory_mb * self.critical_threshold)
        system_critical = memory_info["system_used_percent"] > 90
        
        return absolute_critical or system_critical
    
    def is_memory_warning(self) -> bool:
        """Check if memory usage is at warning levels"""
        memory_info = self.get_memory_usage()
        current_mb = memory_info["rss_mb"]
        
        # Check both absolute limit and system memory pressure
        absolute_warning = current_mb > (self.max_memory_mb * self.warning_threshold)
        system_warning = memory_info["system_used_percent"] > 80
        
        return absolute_warning or system_warning
    
    def force_cleanup(self):
        """Force aggressive memory cleanup"""
        logger.warning("🧹 Forcing aggressive memory cleanup...")
        
        # Call registered cleanup callbacks
        for callback in self.memory_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Memory cleanup callback failed: {e}")
        
        # Multiple garbage collection passes
        for i in range(3):
            collected = gc.collect()
            logger.info(f"GC pass {i+1}: collected {collected} objects")
        
        # Clear various caches if available
        try:
            # Clear torch cache if available
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🔥 Cleared CUDA cache")
        except ImportError:
            pass
        
        memory_after = self.get_memory_usage()
        logger.info(f"🧹 Cleanup complete. Memory usage: {memory_after['rss_mb']:.1f}MB")
    
    def add_cleanup_callback(self, callback: Callable):
        """Add a function to call during memory cleanup"""
        self.memory_callbacks.append(callback)
    
    def start_monitoring(self, check_interval: float = 5.0):
        """Start background memory monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.stop_monitoring.clear()
        
        def monitor():
            while not self.stop_monitoring.is_set():
                try:
                    memory_info = self.get_memory_usage()
                    current_mb = memory_info["rss_mb"]
                    
                    if self.is_memory_critical():
                        logger.error(f"🚨 CRITICAL MEMORY USAGE: {current_mb:.1f}MB / {self.max_memory_mb}MB")
                        logger.error(f"🚨 System memory: {memory_info['system_used_percent']:.1f}%")
                        self.force_cleanup()
                        
                        # Check again after cleanup
                        if self.is_memory_critical():
                            logger.error("🚨 Memory still critical after cleanup! Process may crash!")
                            
                    elif self.is_memory_warning():
                        logger.warning(f"⚠️ HIGH MEMORY USAGE: {current_mb:.1f}MB / {self.max_memory_mb}MB")
                        logger.warning(f"⚠️ System memory: {memory_info['system_used_percent']:.1f}%")
                        
                        # Light cleanup
                        gc.collect()
                    
                    self.stop_monitoring.wait(check_interval)
                        
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    self.stop_monitoring.wait(check_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"📊 Memory monitoring started (limit: {self.max_memory_mb}MB)")
    
    def stop_memory_monitoring(self):
        """Stop background memory monitoring"""
        if self.monitoring_active:
            self.stop_monitoring.set()
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2.0)
            logger.info("📊 Memory monitoring stopped")
    
    @contextmanager
    def memory_limit_context(self, operation_name: str = "operation"):
        """Context manager that monitors memory during operation"""
        memory_before = self.get_memory_usage()
        logger.info(f"🔄 Starting {operation_name} - Memory: {memory_before['rss_mb']:.1f}MB")
        
        # Start monitoring if not already active
        was_monitoring = self.monitoring_active
        if not was_monitoring:
            self.start_monitoring()
        
        try:
            yield self
            
        except MemoryError:
            logger.error(f"❌ {operation_name} failed due to memory error!")
            self.force_cleanup()
            raise
            
        except Exception as e:
            logger.error(f"❌ {operation_name} failed: {e}")
            self.force_cleanup()
            raise
            
        finally:
            # Stop monitoring if we started it
            if not was_monitoring:
                self.stop_memory_monitoring()
            
            # Final cleanup
            gc.collect()
            
            memory_after = self.get_memory_usage()
            memory_used = memory_after['rss_mb'] - memory_before['rss_mb']
            logger.info(f"✅ {operation_name} complete - Memory: {memory_after['rss_mb']:.1f}MB (+{memory_used:+.1f}MB)")

# Decorators for automatic memory management
def memory_managed(operation_name: str = None):
    """Decorator that adds memory management to a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__name__}"
            memory_manager = get_memory_manager()
            
            with memory_manager.memory_limit_context(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def cleanup_on_exit(func):
    """Decorator that ensures cleanup happens when function exits"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            gc.collect()
    return wrapper

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def check_memory_available(required_mb: float = 2000) -> bool:  # Increased default requirement for 16GB
    """Check if enough memory is available for an operation"""
    memory_manager = get_memory_manager()
    memory_info = memory_manager.get_memory_usage()
    
    current_usage = memory_info["rss_mb"]
    available = memory_manager.max_memory_mb - current_usage
    
    if available < required_mb:
        logger.warning(f"⚠️ Insufficient memory: {available:.1f}MB available, {required_mb:.1f}MB required")
        return False
    
    return True

def optimize_for_large_processing():
    """Optimize memory settings for large file processing with 16GB RAM"""
    memory_manager = get_memory_manager()
    
    # Force cleanup before starting
    memory_manager.force_cleanup()
    
    # Start monitoring with less frequent checks due to more headroom
    memory_manager.start_monitoring(check_interval=5.0)  # Less frequent with more memory
    
    logger.info(f"🔧 Memory optimized for large file processing - 16GB configuration")
    logger.info(f"📊 Memory limit: {memory_manager.max_memory_mb}MB")
    logger.info(f"⚠️ Warning threshold: {memory_manager.max_memory_mb * memory_manager.warning_threshold:.0f}MB")
    logger.info(f"🚨 Critical threshold: {memory_manager.max_memory_mb * memory_manager.critical_threshold:.0f}MB")
    
    return memory_manager

if __name__ == "__main__":
    # Test memory manager
    memory_manager = get_memory_manager()
    
    print("📋 Memory Manager Test")
    memory_info = memory_manager.get_memory_usage()
    print(f"Current memory usage: {memory_info}")
    
    # Test memory monitoring
    memory_manager.start_monitoring()
    print("Memory monitoring started...")
    
    time.sleep(2)
    
    memory_manager.stop_memory_monitoring()
    print("Memory monitoring stopped")
    
    # Test context manager
    with memory_manager.memory_limit_context("test operation"):
        print("Inside memory managed context")
        time.sleep(1)
    
    print("Test complete")