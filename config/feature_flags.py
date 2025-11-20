"""
Feature Flag System for APRAG Module
Allows enabling/disabling APRAG features at global and session level
"""

import os
from typing import Dict, Optional
from enum import Enum


class FeatureFlagScope(Enum):
    """Feature flag scope levels"""
    GLOBAL = "global"
    SESSION = "session"
    USER = "user"


class FeatureFlags:
    """
    Centralized feature flag management for APRAG module.
    Supports environment variables and database-based flags.
    """
    
    # Feature flag keys
    APRAG_ENABLED = "aprag_enabled"
    APRAG_FEEDBACK_COLLECTION = "aprag_feedback_collection"
    APRAG_PERSONALIZATION = "aprag_personalization"
    APRAG_RECOMMENDATIONS = "aprag_recommendations"
    APRAG_ANALYTICS = "aprag_analytics"
    
    # Default values (can be overridden by environment variables)
    _defaults = {
        APRAG_ENABLED: False,
        APRAG_FEEDBACK_COLLECTION: False,
        APRAG_PERSONALIZATION: False,
        APRAG_RECOMMENDATIONS: False,
        APRAG_ANALYTICS: False,
    }
    
    # Cache for flags (will be populated from database)
    _cache: Dict[str, bool] = {}
    _session_cache: Dict[str, Dict[str, bool]] = {}
    
    @classmethod
    def _get_env_value(cls, key: str) -> Optional[bool]:
        """Get feature flag value from environment variable"""
        env_key = key.upper()
        value = os.getenv(env_key)
        if value is None:
            return None
        return value.lower() in ("true", "1", "yes", "on")
    
    @classmethod
    def is_enabled(
        cls, 
        flag: str, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Check if a feature flag is enabled.
        
        Priority:
        1. Session-level override (if session_id provided)
        2. User-level override (if user_id provided)
        3. Global setting from cache (admin panel settings)
        4. Global setting from environment variable (startup config)
        5. Default value
        
        Args:
            flag: Feature flag key
            session_id: Optional session ID for session-level check
            user_id: Optional user ID for user-level check
            
        Returns:
            True if feature is enabled, False otherwise
        """
        # Check session-level override
        if session_id and session_id in cls._session_cache:
            if flag in cls._session_cache[session_id]:
                return cls._session_cache[session_id][flag]
        
        # Check cache first (admin panel settings) - this allows runtime changes
        if flag in cls._cache:
            return cls._cache[flag]
        
        # Check environment variable (startup config)
        env_value = cls._get_env_value(flag)
        if env_value is not None:
            return env_value
        
        # Return default
        return cls._defaults.get(flag, False)
    
    @classmethod
    def is_aprag_enabled(cls, session_id: Optional[str] = None) -> bool:
        """Check if APRAG module is enabled globally or for a session"""
        return cls.is_enabled(cls.APRAG_ENABLED, session_id=session_id)
    
    @classmethod
    def is_feedback_collection_enabled(cls, session_id: Optional[str] = None) -> bool:
        """Check if feedback collection is enabled"""
        if not cls.is_aprag_enabled(session_id):
            return False
        return cls.is_enabled(cls.APRAG_FEEDBACK_COLLECTION, session_id=session_id)
    
    @classmethod
    def is_personalization_enabled(cls, session_id: Optional[str] = None) -> bool:
        """Check if personalization is enabled"""
        if not cls.is_aprag_enabled(session_id):
            return False
        return cls.is_enabled(cls.APRAG_PERSONALIZATION, session_id=session_id)
    
    @classmethod
    def is_recommendations_enabled(cls, session_id: Optional[str] = None) -> bool:
        """Check if recommendations are enabled"""
        if not cls.is_aprag_enabled(session_id):
            return False
        return cls.is_enabled(cls.APRAG_RECOMMENDATIONS, session_id=session_id)
    
    @classmethod
    def is_analytics_enabled(cls, session_id: Optional[str] = None) -> bool:
        """Check if analytics are enabled"""
        if not cls.is_aprag_enabled(session_id):
            return False
        return cls.is_enabled(cls.APRAG_ANALYTICS, session_id=session_id)
    
    @classmethod
    def set_flag(
        cls, 
        flag: str, 
        enabled: bool, 
        scope: FeatureFlagScope = FeatureFlagScope.GLOBAL,
        session_id: Optional[str] = None
    ) -> None:
        """
        Set a feature flag value.
        
        Note: This updates the cache. For persistence, use database methods.
        
        Args:
            flag: Feature flag key
            enabled: Whether the feature is enabled
            scope: Scope of the flag (global, session, user)
            session_id: Session ID if scope is SESSION
        """
        if scope == FeatureFlagScope.GLOBAL:
            cls._cache[flag] = enabled
        elif scope == FeatureFlagScope.SESSION and session_id:
            if session_id not in cls._session_cache:
                cls._session_cache[session_id] = {}
            cls._session_cache[session_id][flag] = enabled
    
    @classmethod
    def get_all_flags(cls, session_id: Optional[str] = None) -> Dict[str, bool]:
        """
        Get all feature flags and their current status.
        
        Args:
            session_id: Optional session ID for session-level flags
            
        Returns:
            Dictionary of flag names to enabled status
        """
        flags = {}
        for flag_key in [
            cls.APRAG_ENABLED,
            cls.APRAG_FEEDBACK_COLLECTION,
            cls.APRAG_PERSONALIZATION,
            cls.APRAG_RECOMMENDATIONS,
            cls.APRAG_ANALYTICS,
        ]:
            flags[flag_key] = cls.is_enabled(flag_key, session_id=session_id)
        return flags
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the feature flag cache"""
        cls._cache.clear()
        cls._session_cache.clear()
    
    @classmethod
    def load_from_database(cls, db_connection) -> None:
        """
        Load feature flags from database.
        
        This method should be called on startup and periodically
        to sync with database settings.
        
        Args:
            db_connection: Database connection object
        """
        # TODO: Implement database loading in Phase 1.2
        # For now, this is a placeholder
        pass


# Convenience functions for common checks
def is_aprag_enabled(session_id: Optional[str] = None) -> bool:
    """Check if APRAG module is enabled"""
    return FeatureFlags.is_aprag_enabled(session_id)


def is_feedback_enabled(session_id: Optional[str] = None) -> bool:
    """Check if feedback collection is enabled"""
    return FeatureFlags.is_feedback_collection_enabled(session_id)


def is_personalization_enabled(session_id: Optional[str] = None) -> bool:
    """Check if personalization is enabled"""
    return FeatureFlags.is_personalization_enabled(session_id)


def is_recommendations_enabled(session_id: Optional[str] = None) -> bool:
    """Check if recommendations are enabled"""
    return FeatureFlags.is_recommendations_enabled(session_id)

