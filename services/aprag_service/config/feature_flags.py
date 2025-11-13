"""
Feature Flag System for APRAG Module
Import from parent config directory
"""

import sys
import os

# Add parent directory to path to import from config
parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
sys.path.insert(0, parent_dir)

try:
    from config.feature_flags import (
        FeatureFlags,
        FeatureFlagScope,
        is_aprag_enabled,
        is_feedback_enabled,
        is_personalization_enabled,
        is_recommendations_enabled
    )
except ImportError:
    # Fallback: Define minimal versions if parent config not available
    from enum import Enum
    
    class FeatureFlagScope(Enum):
        GLOBAL = "global"
        SESSION = "session"
    
    class FeatureFlags:
        @staticmethod
        def is_aprag_enabled(session_id=None):
            return os.getenv("APRAG_ENABLED", "true").lower() == "true"
    
    def is_aprag_enabled(session_id=None):
        return os.getenv("APRAG_ENABLED", "true").lower() == "true"
    
    def is_feedback_enabled():
        return os.getenv("APRAG_FEEDBACK_COLLECTION", "true").lower() == "true"
    
    def is_personalization_enabled():
        return os.getenv("APRAG_PERSONALIZATION", "true").lower() == "true"
    
    def is_recommendations_enabled():
        return os.getenv("APRAG_RECOMMENDATIONS", "true").lower() == "true"

__all__ = [
    'FeatureFlags',
    'FeatureFlagScope',
    'is_aprag_enabled',
    'is_feedback_enabled',
    'is_personalization_enabled',
    'is_recommendations_enabled'
]
