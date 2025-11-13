"""
Performance monitoring utilities for RAG system.
Tracks GPU utilization, memory usage, and timing metrics.
"""

import time
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from src.utils.logger import get_logger

# Try to import GPU monitoring libraries
try:
    import pynvml
    NVIDIA_GPU_AVAILABLE = True
except ImportError:
    pynvml = None
    NVIDIA_GPU_AVAILABLE = False

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    retrieval_time_ms: Optional[float] = None
    generation_time_ms: Optional[float] = None
    total_response_time_ms: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    cache_hit: Optional[bool] = None
    num_retrieved_docs: Optional[int] = None
    context_length: Optional[int] = None

class PerformanceMonitor:
    """
    Monitors system performance metrics during RAG operations.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__, {})
        self.gpu_available = self._init_gpu_monitoring()
        self.current_metrics = PerformanceMetrics()
        
    def _init_gpu_monitoring(self) -> bool:
        """Initialize GPU monitoring if available."""
        if not NVIDIA_GPU_AVAILABLE:
            self.logger.info("NVIDIA GPU monitoring not available (pynvml not installed)")
            return False
            
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self.logger.info(f"Initialized GPU monitoring for {device_count} NVIDIA GPU(s)")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize GPU monitoring: {e}")
            return False
    
    def get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        if not self.gpu_available:
            return None
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Primary GPU
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except Exception as e:
            self.logger.debug(f"Error getting GPU utilization: {e}")
            return None
    
    def get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
        """Get GPU memory usage information."""
        if not self.gpu_available:
            return None
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'used_mb': mem_info.used / 1024 / 1024,
                'free_mb': mem_info.free / 1024 / 1024,
                'total_mb': mem_info.total / 1024 / 1024,
                'utilization_percent': (mem_info.used / mem_info.total) * 100
            }
        except Exception as e:
            self.logger.debug(f"Error getting GPU memory info: {e}")
            return None
    
    def get_system_memory_usage(self) -> float:
        """Get system memory usage percentage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            self.logger.debug(f"Error getting system memory usage: {e}")
            return 0.0
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            'timestamp': time.time(),
            'memory_usage_percent': self.get_system_memory_usage(),
            'gpu_utilization_percent': self.get_gpu_utilization(),
            'gpu_memory': self.get_gpu_memory_info()
        }
        
        # Add CPU usage if available
        try:
            metrics['cpu_usage_percent'] = psutil.cpu_percent(interval=0.1)
        except Exception:
            pass
            
        return metrics
    
    @contextmanager
    def measure_timing(self, operation: str):
        """Context manager to measure operation timing."""
        start_time = time.perf_counter()
        start_metrics = self.get_system_metrics()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_metrics = self.get_system_metrics()
            
            elapsed_ms = (end_time - start_time) * 1000
            
            # Update current metrics based on operation type
            if operation == 'retrieval':
                self.current_metrics.retrieval_time_ms = elapsed_ms
            elif operation == 'generation':
                self.current_metrics.generation_time_ms = elapsed_ms
            elif operation == 'total':
                self.current_metrics.total_response_time_ms = elapsed_ms
            
            # Store system metrics (average during operation)
            if start_metrics['gpu_utilization_percent'] is not None:
                avg_gpu = (start_metrics['gpu_utilization_percent'] + 
                          (end_metrics['gpu_utilization_percent'] or 0)) / 2
                self.current_metrics.gpu_utilization_percent = avg_gpu
            
            avg_memory = (start_metrics['memory_usage_percent'] + 
                         end_metrics['memory_usage_percent']) / 2
            self.current_metrics.memory_usage_percent = avg_memory
            
            self.logger.debug(f"{operation.capitalize()} completed in {elapsed_ms:.2f}ms")
    
    def set_cache_hit(self, hit: bool):
        """Record cache hit/miss."""
        self.current_metrics.cache_hit = hit
    
    def set_retrieval_info(self, num_docs: int, context_length: int):
        """Record retrieval information."""
        self.current_metrics.num_retrieved_docs = num_docs
        self.current_metrics.context_length = context_length
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.current_metrics
    
    def reset_metrics(self):
        """Reset current metrics for new operation."""
        self.current_metrics = PerformanceMetrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert current metrics to dictionary."""
        return {
            'retrieval_time_ms': self.current_metrics.retrieval_time_ms,
            'generation_time_ms': self.current_metrics.generation_time_ms,
            'total_response_time_ms': self.current_metrics.total_response_time_ms,
            'gpu_utilization_percent': self.current_metrics.gpu_utilization_percent,
            'memory_usage_percent': self.current_metrics.memory_usage_percent,
            'cache_hit': self.current_metrics.cache_hit,
            'num_retrieved_docs': self.current_metrics.num_retrieved_docs,
            'context_length': self.current_metrics.context_length
        }

# Global performance monitor instance
_monitor_instance: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance

@contextmanager
def measure_performance(operation: str):
    """Convenience function to measure performance of an operation."""
    monitor = get_performance_monitor()
    with monitor.measure_timing(operation):
        yield monitor

def get_realtime_metrics() -> Dict[str, Any]:
    """Get real-time system metrics for dashboards."""
    monitor = get_performance_monitor()
    return monitor.get_system_metrics()