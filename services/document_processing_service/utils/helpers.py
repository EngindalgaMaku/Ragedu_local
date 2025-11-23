"""
Helper utility functions
"""
import json
import time
from typing import Dict, Any
from utils.logger import logger


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata to ensure all values are ChromaDB-compliant types (str, int, float, bool)
    Convert lists and dicts to JSON strings for compatibility
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Sanitized metadata dictionary
    """
    if not metadata:
        logger.warning("No metadata received!")
        return {}
    
    sanitized_metadata = {}
    for key, value in metadata.items():
        logger.debug(f"Processing metadata key='{key}', value={repr(value)}, type={type(value)}")
        
        if isinstance(value, (str, int, float, bool)):
            sanitized_metadata[key] = value
            logger.debug(f"Added {key}={value} (primitive type)")
        elif isinstance(value, (list, dict)):
            # Convert lists and dicts to JSON strings
            json_value = json.dumps(value)
            sanitized_metadata[key] = json_value
            logger.debug(f"Converted metadata key '{key}' from {type(value)} to JSON string")
        else:
            logger.warning(f"Excluding non-compliant metadata key '{key}' of type {type(value)}")
    
    logger.debug(f"Final sanitized metadata: {sanitized_metadata}")
    return sanitized_metadata


def format_collection_name(session_id: str, add_timestamp: bool = True) -> str:
    """
    Format collection name for ChromaDB
    
    Features:
    - Convert 32-char hex to UUID format
    - Optionally add timestamp to prevent collisions
    
    Args:
        session_id: Session ID (can be 32-char hex or UUID format)
        add_timestamp: Whether to add timestamp suffix
        
    Returns:
        Formatted collection name
    """
    collection_name = session_id
    
    # If session_id starts with "session_", remove it
    if collection_name.startswith("session_"):
        collection_name = collection_name[8:]
    
    # Convert 32-char hex string to proper UUID format (8-4-4-4-12)
    if len(collection_name) == 32 and collection_name.replace('-', '').isalnum():
        base_uuid = f"{collection_name[:8]}-{collection_name[8:12]}-{collection_name[12:16]}-{collection_name[16:20]}-{collection_name[20:]}"
        collection_name = base_uuid
        logger.debug(f"Transformed to UUID format: '{collection_name}'")
    
    # Add timestamp for collision prevention
    if add_timestamp:
        timestamp = int(time.time())
        collection_name = f"{collection_name}_{timestamp}"
        logger.debug(f"Added timestamp: '{collection_name}'")
    
    return collection_name






