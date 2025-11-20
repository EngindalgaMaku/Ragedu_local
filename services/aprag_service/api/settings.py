"""
Settings and feature flag endpoints
Manages APRAG module settings
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import logging

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Import from local config (APRAG service's own config)
try:
    from config.feature_flags import FeatureFlags
except ImportError:
    # Fallback to parent config if local not available
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
    from config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)

router = APIRouter()


class SettingsResponse(BaseModel):
    """Response model for settings"""
    enabled: bool
    global_enabled: bool
    session_enabled: Optional[bool]
    features: Dict[str, bool]


class ToggleRequest(BaseModel):
    """Request model for toggling features"""
    enabled: bool
    scope: str = "global"  # global, session
    session_id: Optional[str] = None
    flag_key: Optional[str] = None  # Specific flag to toggle


@router.get("/status")
async def get_status(session_id: Optional[str] = None):
    """
    Get APRAG module status and feature flags
    
    Args:
        session_id: Optional session ID for session-level status
    """
    try:
        logger.info(f"[APRAG SETTINGS] get_status called with session_id: {session_id}")
        
        # Get global status
        global_enabled = FeatureFlags.is_aprag_enabled(session_id=None)
        logger.info(f"[APRAG SETTINGS] global_enabled: {global_enabled}")
        
        # Get session-specific status if session_id provided
        enabled = FeatureFlags.is_aprag_enabled(session_id=session_id)
        logger.info(f"[APRAG SETTINGS] is_aprag_enabled(session_id={session_id}): {enabled}")
        
        # Session-level flag (if session_id provided)
        session_enabled = enabled if session_id else None
        logger.info(f"[APRAG SETTINGS] session_enabled: {session_enabled}")
        
        # Get feature flags from environment variables
        feedback_enabled = os.getenv("APRAG_FEEDBACK_COLLECTION", "true").lower() == "true" if global_enabled else False
        personalization_enabled = os.getenv("APRAG_PERSONALIZATION", "true").lower() == "true" if global_enabled else False
        recommendations_enabled = os.getenv("APRAG_RECOMMENDATIONS", "true").lower() == "true" if global_enabled else False
        analytics_enabled = os.getenv("APRAG_ANALYTICS", "true").lower() == "true" if global_enabled else False
        
        logger.info(f"[APRAG SETTINGS] Feature flags: feedback={feedback_enabled}, personalization={personalization_enabled}, recommendations={recommendations_enabled}, analytics={analytics_enabled}")
        
        return {
            "enabled": enabled,
            "global_enabled": global_enabled,
            "session_enabled": session_enabled,
            "features": {
                "feedback_collection": feedback_enabled,
                "personalization": personalization_enabled,
                "recommendations": recommendations_enabled,
                "analytics": analytics_enabled,
            }
        }
    except Exception as e:
        logger.error(f"[APRAG SETTINGS] Error in get_status: {e}")
        logger.error(f"[APRAG SETTINGS] Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"[APRAG SETTINGS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Settings error: {str(e)}")


@router.post("/toggle")
async def toggle_feature(request: ToggleRequest):
    """
    Toggle APRAG features
    
    Args:
        request: Toggle request with enabled status, scope, and flag_key
    """
    logger.info(f"[APRAG SETTINGS] toggle_feature called: enabled={request.enabled}, scope={request.scope}, flag_key={request.flag_key}")
    
    try:
        # Determine the flag key - map frontend keys to environment variable names
        flag_key = request.flag_key or "aprag_enabled"
        
        # Map flag keys to environment variables
        env_var_map = {
            "aprag_enabled": "APRAG_ENABLED",
            "aprag_feedback_collection": "APRAG_FEEDBACK_COLLECTION",
            "aprag_personalization": "APRAG_PERSONALIZATION",
            "aprag_recommendations": "APRAG_RECOMMENDATIONS",
            "aprag_analytics": "APRAG_ANALYTICS"
        }
        
        env_var = env_var_map.get(flag_key, flag_key.upper())
        
        # Set environment variable for runtime changes
        os.environ[env_var] = "true" if request.enabled else "false"
        
        logger.info(f"[APRAG SETTINGS] Successfully toggled {env_var} to {request.enabled} (scope: {request.scope})")
        
        return {
            "message": "Feature flag updated successfully",
            "enabled": request.enabled,
            "scope": request.scope,
            "session_id": request.session_id,
            "flag_key": flag_key,
            "env_var": env_var
        }
    except Exception as e:
        logger.error(f"[APRAG SETTINGS] Error toggling feature: {e}")
        import traceback
        logger.error(f"[APRAG SETTINGS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to toggle feature: {str(e)}")

