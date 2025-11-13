"""
Utility functions for the RAG system.

This module provides helper functions for tasks such as logging,
file operations, and other common utilities.
"""

import logging
import os
from ..config import config

def setup_logging():
    """
    Configures the logging for the application.
    
    Sets up a logger that writes to both a file and the console, using
    settings from the config module.
    """
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    log_file = getattr(config, 'LOG_FILE', 'data/logs/rag_system.log')

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create a logger
    logger = logging.getLogger("RAG_System")
    logger.setLevel(log_level)

    # Create handlers if they don't exist
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

if __name__ == '__main__':
    # Example usage of the logger
    logger = setup_logging()
    logger.info("Logging setup complete.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
