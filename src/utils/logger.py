"""
Logger utility for RAG system.
Provides structured logging across all modules.
"""

import logging
import colorlog
from typing import Dict, Any

def get_logger(name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        config: Optional configuration dictionary
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if already configured
    if logger.handlers:
        return logger
    
    # Set level from config or default to INFO
    log_level = logging.INFO
    if config:
        # Handle both dictionary-style and object-style config
        if hasattr(config, 'LOG_LEVEL'):
            # RAGConfig object style
            log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
        elif isinstance(config, dict) and 'log_level' in config:
            # Dictionary style
            log_level = getattr(logging, config['log_level'].upper(), logging.INFO)
    
    logger.setLevel(log_level)
    
    # Create console handler with color formatting
    handler = colorlog.StreamHandler()
    handler.setLevel(log_level)
    
    # Color formatter for better readability
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green', 
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger