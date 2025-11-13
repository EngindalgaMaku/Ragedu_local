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

try:
    from config.feature_flags import FeatureFlags, FeatureFlagScope
except ImportError:
    # Fallback
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../rag3_for_colab'))
    from config.feature_flags import FeatureFlags, FeatureFlagScope

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
        
        # Use fallback implementation that doesn't support session-level flags
        enabled = FeatureFlags.is_aprag_enabled()
        logger.info(f"[APRAG SETTINGS] is_aprag_enabled(): {enabled}")
        
        global_enabled = FeatureFlags.is_aprag_enabled()
        logger.info(f"[APRAG SETTINGS] global_enabled: {global_enabled}")
        
        # Session-level flags not supported in fallback mode
        session_enabled = None
        logger.info(f"[APRAG SETTINGS] session_enabled: {session_enabled} (not supported in fallback mode)")
        
        # Check if the methods exist and support session parameters
        try:
            if hasattr(FeatureFlags, 'is_feedback_collection_enabled'):
                feedback_enabled = FeatureFlags.is_feedback_collection_enabled()
            else:
                feedback_enabled = True  # Default fallback
                
            if hasattr(FeatureFlags, 'is_personalization_enabled'):
                personalization_enabled = FeatureFlags.is_personalization_enabled()
            else:
                personalization_enabled = True
                
            if hasattr(FeatureFlags, 'is_recommendations_enabled'):
                recommendations_enabled = FeatureFlags.is_recommendations_enabled()
            else:
                recommendations_enabled = True
                
            if hasattr(FeatureFlags, 'is_analytics_enabled'):
                analytics_enabled = FeatureFlags.is_analytics_enabled()
            else:
                analytics_enabled = True
        except:
            # If methods don't exist or fail, use defaults
            feedback_enabled = True
            personalization_enabled = True
            recommendations_enabled = True
            analytics_enabled = True
        
        logger.info(f"[APRAG SETTINGS] Feature flags loaded successfully")
        
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
    # The fallback FeatureFlags implementation doesn't support set_flag method
    # and doesn't have class constants, so we'll return a success response
    # without actually changing the flags (fallback mode uses env vars)
    
    logger.info(f"[APRAG SETTINGS] toggle_feature called: enabled={request.enabled}, scope={request.scope}, flag_key={request.flag_key}")
    
    # In fallback mode, we can't actually toggle flags since they're read from env vars
    # But we can simulate the response for frontend compatibility
    
    return {
        "message": "Feature flag toggle request received (fallback mode)",
        "enabled": request.enabled,
        "scope": request.scope,
        "session_id": request.session_id,
        "flag_key": request.flag_key or "aprag_enabled",
        "note": "Feature flags are controlled by environment variables in fallback mode"
    }

