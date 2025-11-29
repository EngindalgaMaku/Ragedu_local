"""
Centralized Logging Configuration for Module Extraction Service
Provides consistent logging patterns and error handling utilities
"""

import logging
import os
import sys
from typing import Optional, Dict, Any
from functools import wraps
import traceback
from datetime import datetime


class ModuleExtractionLogger:
    """
    Centralized logger for module extraction operations
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
            
            # Set higher log level for external libraries
            logging.getLogger("requests").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    def log_operation_start(self, operation: str, **context):
        """Log the start of an operation with context"""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.info(f"🚀 [START] {operation} - {context_str}")
    
    def log_operation_success(self, operation: str, duration_ms: Optional[float] = None, **results):
        """Log successful completion of an operation"""
        result_str = ", ".join([f"{k}={v}" for k, v in results.items()])
        duration_info = f" ({duration_ms:.2f}ms)" if duration_ms else ""
        self.logger.info(f"✅ [SUCCESS] {operation}{duration_info} - {result_str}")
    
    def log_operation_warning(self, operation: str, warning: str, **context):
        """Log a warning during an operation"""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.warning(f"⚠️ [WARNING] {operation} - {warning} - {context_str}")
    
    def log_operation_error(self, operation: str, error: Exception, **context):
        """Log an error during an operation"""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_type = type(error).__name__
        self.logger.error(f"❌ [ERROR] {operation} - {error_type}: {str(error)} - {context_str}")
        self.logger.error(f"❌ [TRACEBACK] {traceback.format_exc()}")
    
    def log_validation_issue(self, severity: str, category: str, message: str, **context):
        """Log a validation issue"""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        severity_icon = {
            "error": "🚨",
            "warning": "⚠️",
            "info": "ℹ️"
        }.get(severity.lower(), "📝")
        
        log_method = {
            "error": self.logger.error,
            "warning": self.logger.warning,
            "info": self.logger.info
        }.get(severity.lower(), self.logger.info)
        
        log_method(f"{severity_icon} [VALIDATION-{severity.upper()}] {category}: {message} - {context_str}")
    
    def log_llm_operation(self, operation: str, model: str, tokens: Optional[int] = None, **context):
        """Log LLM operations specifically"""
        token_info = f" (tokens: {tokens})" if tokens else ""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.info(f"🧠 [LLM-{operation.upper()}] {model}{token_info} - {context_str}")
    
    def log_database_operation(self, operation: str, table: str, affected_rows: Optional[int] = None, **context):
        """Log database operations"""
        rows_info = f" (rows: {affected_rows})" if affected_rows is not None else ""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.info(f"💾 [DB-{operation.upper()}] {table}{rows_info} - {context_str}")
    
    def log_curriculum_operation(self, operation: str, subject: str, grade: Optional[str] = None, **context):
        """Log curriculum-related operations"""
        grade_info = f" (grade: {grade})" if grade else ""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.info(f"📚 [CURRICULUM-{operation.upper()}] {subject}{grade_info} - {context_str}")


def get_module_logger(name: str) -> ModuleExtractionLogger:
    """Get a module extraction logger instance"""
    return ModuleExtractionLogger(name)


def log_execution_time(logger: ModuleExtractionLogger, operation: str):
    """Decorator to log execution time of methods"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                logger.log_operation_start(operation, function=func.__name__)
                result = await func(*args, **kwargs)
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.log_operation_success(operation, duration_ms=duration_ms)
                return result
            except Exception as e:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.log_operation_error(f"{operation} (failed after {duration_ms:.2f}ms)", e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                logger.log_operation_start(operation, function=func.__name__)
                result = func(*args, **kwargs)
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.log_operation_success(operation, duration_ms=duration_ms)
                return result
            except Exception as e:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.log_operation_error(f"{operation} (failed after {duration_ms:.2f}ms)", e)
                raise
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize objects to JSON for logging"""
    try:
        import json
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


class ErrorHandler:
    """
    Centralized error handling for module extraction operations
    """
    
    def __init__(self, logger: ModuleExtractionLogger):
        self.logger = logger
    
    def handle_llm_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM-related errors with fallback strategies"""
        self.logger.log_operation_error("LLM_REQUEST", error, **context)
        
        if "timeout" in str(error).lower():
            return {
                "success": False,
                "error_type": "timeout",
                "message": "LLM request timed out. Try reducing input size or using a faster model.",
                "fallback_available": True,
                "suggested_actions": [
                    "Reduce input text size",
                    "Switch to a faster model",
                    "Retry with increased timeout"
                ]
            }
        elif "connection" in str(error).lower():
            return {
                "success": False,
                "error_type": "connection",
                "message": "Could not connect to LLM service. Please check if the service is running.",
                "fallback_available": False,
                "suggested_actions": [
                    "Check LLM service status",
                    "Verify network connectivity",
                    "Check service URL configuration"
                ]
            }
        elif "token" in str(error).lower() or "limit" in str(error).lower():
            return {
                "success": False,
                "error_type": "token_limit",
                "message": "Input exceeds model token limits. Content will be truncated.",
                "fallback_available": True,
                "suggested_actions": [
                    "Split input into smaller batches",
                    "Use content summarization",
                    "Switch to a model with higher token limits"
                ]
            }
        else:
            return {
                "success": False,
                "error_type": "unknown",
                "message": f"LLM operation failed: {str(error)}",
                "fallback_available": False,
                "suggested_actions": [
                    "Check LLM service logs",
                    "Verify request format",
                    "Try with different input"
                ]
            }
    
    def handle_database_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database-related errors"""
        self.logger.log_operation_error(f"DB_{operation.upper()}", error, **context)
        
        if "foreign key" in str(error).lower():
            return {
                "success": False,
                "error_type": "foreign_key_constraint",
                "message": "Database foreign key constraint violation. Check related records exist.",
                "fallback_available": False
            }
        elif "unique" in str(error).lower():
            return {
                "success": False,
                "error_type": "unique_constraint",
                "message": "Duplicate record detected. Record may already exist.",
                "fallback_available": True
            }
        elif "lock" in str(error).lower():
            return {
                "success": False,
                "error_type": "database_lock",
                "message": "Database is locked. Operation will be retried.",
                "fallback_available": True
            }
        else:
            return {
                "success": False,
                "error_type": "database_error",
                "message": f"Database operation failed: {str(error)}",
                "fallback_available": False
            }
    
    def handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation-related errors"""
        self.logger.log_operation_error("VALIDATION", error, **context)
        
        return {
            "success": False,
            "error_type": "validation_error",
            "message": f"Validation failed: {str(error)}",
            "fallback_available": True,
            "suggested_actions": [
                "Check input data format",
                "Enable auto-fix if available",
                "Review validation requirements"
            ]
        }
    
    def create_error_response(self, error_info: Dict[str, Any], status_code: int = 500) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "success": False,
            "error": {
                "type": error_info.get("error_type", "unknown"),
                "message": error_info.get("message", "An unknown error occurred"),
                "timestamp": datetime.now().isoformat(),
                "fallback_available": error_info.get("fallback_available", False),
                "suggested_actions": error_info.get("suggested_actions", [])
            },
            "status_code": status_code
        }


def with_error_handling(logger: ModuleExtractionLogger, operation: str):
    """Decorator to add comprehensive error handling to methods"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = ErrorHandler(logger)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Determine error type and create appropriate response
                if "llm" in operation.lower() or "model" in operation.lower():
                    error_info = error_handler.handle_llm_error(e, {"operation": operation})
                elif "database" in operation.lower() or "db" in operation.lower():
                    error_info = error_handler.handle_database_error(e, operation, {"function": func.__name__})
                elif "validation" in operation.lower():
                    error_info = error_handler.handle_validation_error(e, {"operation": operation})
                else:
                    error_info = {
                        "success": False,
                        "error_type": "general_error",
                        "message": str(e),
                        "fallback_available": False
                    }
                
                # Re-raise the exception with additional context
                error_response = error_handler.create_error_response(error_info)
                raise Exception(f"Operation '{operation}' failed: {error_response['error']['message']}") from e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = ErrorHandler(logger)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Same error handling logic for sync functions
                if "llm" in operation.lower() or "model" in operation.lower():
                    error_info = error_handler.handle_llm_error(e, {"operation": operation})
                elif "database" in operation.lower() or "db" in operation.lower():
                    error_info = error_handler.handle_database_error(e, operation, {"function": func.__name__})
                elif "validation" in operation.lower():
                    error_info = error_handler.handle_validation_error(e, {"operation": operation})
                else:
                    error_info = {
                        "success": False,
                        "error_type": "general_error",
                        "message": str(e),
                        "fallback_available": False
                    }
                
                error_response = error_handler.create_error_response(error_info)
                raise Exception(f"Operation '{operation}' failed: {error_response['error']['message']}") from e
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator


def log_function_call(logger: ModuleExtractionLogger):
    """Decorator to log function calls with parameters (for debugging)"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Only log in debug mode to avoid performance impact
            if logger.logger.isEnabledFor(logging.DEBUG):
                args_str = safe_json_serialize(args)[:200] if args else "None"
                kwargs_str = safe_json_serialize(kwargs)[:200] if kwargs else "None"
                logger.logger.debug(f"📞 [CALL] {func.__name__}(args={args_str}, kwargs={kwargs_str})")
            
            result = await func(*args, **kwargs)
            
            if logger.logger.isEnabledFor(logging.DEBUG):
                result_str = safe_json_serialize(result)[:200] if result else "None"
                logger.logger.debug(f"📤 [RETURN] {func.__name__} -> {result_str}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if logger.logger.isEnabledFor(logging.DEBUG):
                args_str = safe_json_serialize(args)[:200] if args else "None"
                kwargs_str = safe_json_serialize(kwargs)[:200] if kwargs else "None"
                logger.logger.debug(f"📞 [CALL] {func.__name__}(args={args_str}, kwargs={kwargs_str})")
            
            result = func(*args, **kwargs)
            
            if logger.logger.isEnabledFor(logging.DEBUG):
                result_str = safe_json_serialize(result)[:200] if result else "None"
                logger.logger.debug(f"📤 [RETURN] {func.__name__} -> {result_str}")
            
            return result
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator