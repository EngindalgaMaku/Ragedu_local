"""
Feature Flags for APRAG Service and Eğitsel-KBRAG
All Eğitsel-KBRAG features are dependent on APRAG being enabled
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    Feature Flags Manager
    
    Hierarchy:
    1. APRAG Service (ana servis)
       └─ 2. Eğitsel-KBRAG (genel özellik seti)
           ├─ CACS Algoritması
           ├─ ZPD Calculator
           ├─ Bloom Detector
           ├─ Cognitive Load Manager
           └─ Emoji Feedback
    
    APRAG pasifse, tüm alt özellikler otomatik devre dışı kalır.
    """
    
    # Ana APRAG Servisi Kontrolü
    _aprag_enabled = None
    _db_manager = None
    
    @staticmethod
    def is_aprag_enabled(session_id=None):
        """
        APRAG servisinin aktif olup olmadığını kontrol et
        
        Args:
            session_id: Optional session ID for session-specific checks
            
        Returns:
            bool: APRAG aktif mi?
        """
        # Environment variable ile kontrol
        env_enabled = os.getenv("APRAG_ENABLED", "true").lower() == "true"
        
        if not env_enabled:
            logger.debug("APRAG disabled via environment variable")
            return False
        
        # Database'den session-specific kontrol (varsa)
        if session_id and FeatureFlags._db_manager:
            try:
                # First ensure table exists (kalıcı çözüm)
                with FeatureFlags._db_manager.get_connection() as conn:
                    cursor = conn.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='feature_flags'
                    """)
                    if not cursor.fetchone():
                        # Table doesn't exist, create it
                        conn.execute("""
                            CREATE TABLE feature_flags (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                feature_name TEXT NOT NULL,
                                session_id TEXT,
                                feature_enabled BOOLEAN NOT NULL DEFAULT 1,
                                config_data TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                UNIQUE(feature_name, session_id)
                            )
                        """)
                        conn.commit()
                        logger.info("✅ Created feature_flags table in feature_flags.py")
                
                # Session-specific feature flag kontrolü
                result = FeatureFlags._db_manager.execute_query(
                    "SELECT feature_enabled FROM feature_flags WHERE feature_name = ? AND session_id = ?",
                    ("aprag", session_id)
                )
                if result:
                    return bool(result[0].get("feature_enabled", True))
            except Exception as e:
                logger.warning(f"Failed to check session-specific APRAG flag: {e}")
        
        return env_enabled
    
    @staticmethod
    def load_from_database(db_manager):
        """Load feature flags from database"""
        FeatureFlags._db_manager = db_manager
        logger.info("Feature flags loaded with database support")
    
    # ==========================================
    # Eğitsel-KBRAG Ana Kontrolü (APRAG'a bağlı)
    # ==========================================
    
    @staticmethod
    def is_egitsel_kbrag_enabled():
        """
        Eğitsel-KBRAG özellik setinin aktif olup olmadığını kontrol et
        
        ÖNEMLİ: APRAG pasifse, bu özellik de otomatik pasif olur
        
        Returns:
            bool: Eğitsel-KBRAG aktif mi?
        """
        # Önce APRAG aktif mi kontrol et
        if not FeatureFlags.is_aprag_enabled():
            logger.debug("Eğitsel-KBRAG disabled: APRAG is not enabled")
            return False
        
        # APRAG aktifse, Eğitsel-KBRAG flag'ini kontrol et
        enabled = os.getenv("ENABLE_EGITSEL_KBRAG", "true").lower() == "true"
        
        if not enabled:
            logger.debug("Eğitsel-KBRAG disabled via environment variable")
        
        return enabled
    
    # ==========================================
    # Bireysel Eğitsel-KBRAG Özellikleri
    # ==========================================
    
    @staticmethod
    def is_cacs_enabled():
        """
        CACS (Conversation-Aware Content Scoring) Algoritması
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("CACS disabled: Eğitsel-KBRAG not enabled")
            return False
        
        return os.getenv("ENABLE_CACS", "true").lower() == "true"
    
    @staticmethod
    def is_zpd_enabled():
        """
        ZPD (Zone of Proximal Development) Calculator
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("ZPD disabled: Eğitsel-KBRAG not enabled")
            return False
        
        return os.getenv("ENABLE_ZPD", "true").lower() == "true"
    
    @staticmethod
    def is_bloom_enabled():
        """
        Bloom Taxonomy Detector
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("Bloom disabled: Eğitsel-KBRAG not enabled")
            return False
        
        return os.getenv("ENABLE_BLOOM", "true").lower() == "true"
    
    @staticmethod
    def is_cognitive_load_enabled():
        """
        Cognitive Load Manager
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("Cognitive Load disabled: Eğitsel-KBRAG not enabled")
            return False
        
        return os.getenv("ENABLE_COGNITIVE_LOAD", "true").lower() == "true"
    
    @staticmethod
    def is_emoji_feedback_enabled():
        """
        Emoji-based Micro Feedback System
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("Emoji Feedback disabled: Eğitsel-KBRAG not enabled")
            return False
        
        return os.getenv("ENABLE_EMOJI_FEEDBACK", "true").lower() == "true"
    
    @staticmethod
    def is_ebars_enabled(session_id=None):
        """
        EBARS (Emoji-Based Adaptive Response System)
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        Adaptive difficulty adjustment based on emoji feedback
        
        Args:
            session_id: Optional session ID for session-specific checks
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("EBARS disabled: Eğitsel-KBRAG not enabled")
            return False
        
        # Check session-specific settings from session_settings table first
        if session_id and FeatureFlags._db_manager:
            try:
                # Check if session_settings table exists and has enable_ebars column
                with FeatureFlags._db_manager.get_connection() as conn:
                    # Check if table exists
                    cursor = conn.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='session_settings'
                    """)
                    if cursor.fetchone():
                        # Check if enable_ebars column exists
                        cursor = conn.execute("PRAGMA table_info(session_settings)")
                        columns = [row[1] for row in cursor.fetchall()]
                        if 'enable_ebars' in columns:
                            # Get EBARS setting from session_settings
                            result = FeatureFlags._db_manager.execute_query(
                                "SELECT enable_ebars FROM session_settings WHERE session_id = ?",
                                (session_id,)
                            )
                            if result:
                                session_setting = bool(result[0].get("enable_ebars", False))
                                logger.debug(f"Session {session_id} EBARS setting from session_settings: {session_setting}")
                                return session_setting
                        else:
                            logger.debug(f"enable_ebars column not found in session_settings table")
                    else:
                        logger.debug(f"session_settings table not found")
            except Exception as e:
                logger.warning(f"Failed to check session EBARS setting from session_settings: {e}")
        
        # Fallback to environment variable (default: false - must be explicitly enabled)
        env_setting = os.getenv("ENABLE_EBARS", "false").lower() == "true"
        logger.debug(f"Using environment EBARS setting: {env_setting}")
        return env_setting
    
    @staticmethod
    def is_progressive_assessment_enabled(session_id=None):
        """
        Progressive Assessment Flow System
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        Progressive, adaptive assessment that provides deeper learning insights
        
        Args:
            session_id: Optional session ID for session-specific checks
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("Progressive Assessment disabled: Eğitsel-KBRAG not enabled")
            return False
        
        # Check session-specific settings first
        if session_id and FeatureFlags._db_manager:
            try:
                result = FeatureFlags._db_manager.execute_query(
                    "SELECT enable_progressive_assessment FROM session_settings WHERE session_id = ?",
                    (session_id,)
                )
                if result:
                    session_setting = bool(result[0].get("enable_progressive_assessment", False))
                    logger.debug(f"Session {session_id} progressive assessment setting: {session_setting}")
                    return session_setting
            except Exception as e:
                logger.warning(f"Failed to check session progressive assessment setting: {e}")
        
        # Fallback to environment variable
        env_setting = os.getenv("ENABLE_PROGRESSIVE_ASSESSMENT", "true").lower() == "true"
        logger.debug(f"Using environment progressive assessment setting: {env_setting}")
        return env_setting
    
    @staticmethod
    def is_personalized_responses_enabled(session_id=None):
        """
        Personalized Response System
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        AI-powered response personalization based on student profile
        
        Args:
            session_id: Optional session ID for session-specific checks
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("Personalized Responses disabled: Eğitsel-KBRAG not enabled")
            return False
        
        # Check session-specific settings first
        if session_id and FeatureFlags._db_manager:
            try:
                result = FeatureFlags._db_manager.execute_query(
                    "SELECT enable_personalized_responses FROM session_settings WHERE session_id = ?",
                    (session_id,)
                )
                if result:
                    session_setting = bool(result[0].get("enable_personalized_responses", False))
                    logger.debug(f"Session {session_id} personalized responses setting: {session_setting}")
                    return session_setting
            except Exception as e:
                logger.warning(f"Failed to check session personalized responses setting: {e}")
        
        # Fallback to environment variable (default: true for Eğitsel-KBRAG)
        env_setting = os.getenv("ENABLE_PERSONALIZED_RESPONSES", "true").lower() == "true"
        logger.debug(f"Using environment personalized responses setting: {env_setting}")
        return env_setting
    
    @staticmethod
    def is_multi_dimensional_feedback_enabled(session_id=None):
        """
        Multi-Dimensional Feedback System
        
        APRAG ve Eğitsel-KBRAG'a bağlı
        Advanced feedback collection and analysis
        
        Args:
            session_id: Optional session ID for session-specific checks
        """
        if not FeatureFlags.is_egitsel_kbrag_enabled():
            logger.debug("Multi-Dimensional Feedback disabled: Eğitsel-KBRAG not enabled")
            return False
        
        # Check session-specific settings first
        if session_id and FeatureFlags._db_manager:
            try:
                result = FeatureFlags._db_manager.execute_query(
                    "SELECT enable_multi_dimensional_feedback FROM session_settings WHERE session_id = ?",
                    (session_id,)
                )
                if result:
                    session_setting = bool(result[0].get("enable_multi_dimensional_feedback", False))
                    logger.debug(f"Session {session_id} multi-dimensional feedback setting: {session_setting}")
                    return session_setting
            except Exception as e:
                logger.warning(f"Failed to check session multi-dimensional feedback setting: {e}")
        
        # Fallback to environment variable
        env_setting = os.getenv("ENABLE_MULTI_DIMENSIONAL_FEEDBACK", "false").lower() == "true"
        logger.debug(f"Using environment multi-dimensional feedback setting: {env_setting}")
        return env_setting
    
    # ==========================================
    # Module Extraction Feature Methods
    # ==========================================
    
    @staticmethod
    def is_module_extraction_enabled(session_id=None):
        """
        Module Extraction System
        
        APRAG'a bağlı
        Extract curriculum-aware modules from session content
        
        Args:
            session_id: Optional session ID for session-specific checks
        """
        # Önce APRAG aktif mi kontrol et
        if not FeatureFlags.is_aprag_enabled(session_id):
            logger.debug("Module extraction disabled: APRAG is not enabled")
            return False
        
        # Check session-specific settings first
        if session_id and FeatureFlags._db_manager:
            try:
                result = FeatureFlags._db_manager.execute_query(
                    "SELECT enable_module_extraction FROM session_settings WHERE session_id = ?",
                    (session_id,)
                )
                if result:
                    session_setting = bool(result[0].get("enable_module_extraction", True))
                    logger.debug(f"Session {session_id} module extraction setting: {session_setting}")
                    return session_setting
            except Exception as e:
                logger.warning(f"Failed to check session module extraction setting: {e}")
        
        # Fallback to environment variable
        env_setting = os.getenv("MODULE_EXTRACTION_ENABLED", "true").lower() == "true"
        logger.debug(f"Using environment module extraction setting: {env_setting}")
        return env_setting
    
    @staticmethod
    def is_module_quality_validation_enabled(session_id=None):
        """
        Module Quality Validation System
        
        APRAG ve Module Extraction'a bağlı
        Validate extracted modules for quality and completeness
        
        Args:
            session_id: Optional session ID for session-specific checks
        """
        if not FeatureFlags.is_module_extraction_enabled(session_id):
            logger.debug("Module quality validation disabled: Module extraction not enabled")
            return False
        
        # Check session-specific settings first
        if session_id and FeatureFlags._db_manager:
            try:
                result = FeatureFlags._db_manager.execute_query(
                    "SELECT enable_module_quality_validation FROM session_settings WHERE session_id = ?",
                    (session_id,)
                )
                if result:
                    session_setting = bool(result[0].get("enable_module_quality_validation", True))
                    logger.debug(f"Session {session_id} module quality validation setting: {session_setting}")
                    return session_setting
            except Exception as e:
                logger.warning(f"Failed to check session module quality validation setting: {e}")
        
        # Fallback to environment variable
        env_setting = os.getenv("MODULE_QUALITY_VALIDATION_ENABLED", "true").lower() == "true"
        logger.debug(f"Using environment module quality validation setting: {env_setting}")
        return env_setting
    
    @staticmethod
    def is_module_curriculum_alignment_enabled(session_id=None):
        """
        Module Curriculum Alignment System
        
        APRAG ve Module Extraction'a bağlı
        Align extracted modules with curriculum standards
        
        Args:
            session_id: Optional session ID for session-specific checks
        """
        if not FeatureFlags.is_module_extraction_enabled(session_id):
            logger.debug("Module curriculum alignment disabled: Module extraction not enabled")
            return False
        
        # Check session-specific settings first
        if session_id and FeatureFlags._db_manager:
            try:
                result = FeatureFlags._db_manager.execute_query(
                    "SELECT enable_module_curriculum_alignment FROM session_settings WHERE session_id = ?",
                    (session_id,)
                )
                if result:
                    session_setting = bool(result[0].get("enable_module_curriculum_alignment", True))
                    logger.debug(f"Session {session_id} module curriculum alignment setting: {session_setting}")
                    return session_setting
            except Exception as e:
                logger.warning(f"Failed to check session module curriculum alignment setting: {e}")
        
        # Fallback to environment variable
        env_setting = os.getenv("MODULE_CURRICULUM_ALIGNMENT_ENABLED", "true").lower() == "true"
        logger.debug(f"Using environment module curriculum alignment setting: {env_setting}")
        return env_setting
    
    # ==========================================
    # Yardımcı Metodlar
    # ==========================================
    
    @staticmethod
    def get_status_report():
        """
        Tüm feature flag'lerin durumunu raporla
        
        Returns:
            dict: Feature flag durumları
        """
        aprag_enabled = FeatureFlags.is_aprag_enabled()
        kbrag_enabled = FeatureFlags.is_egitsel_kbrag_enabled()
        
        return {
            "aprag": {
                "enabled": aprag_enabled,
                "status": "active" if aprag_enabled else "disabled"
            },
            "egitsel_kbrag": {
                "enabled": kbrag_enabled,
                "status": "active" if kbrag_enabled else "disabled (requires APRAG)",
                "features": {
                    "cacs": FeatureFlags.is_cacs_enabled(),
                    "zpd": FeatureFlags.is_zpd_enabled(),
                    "bloom": FeatureFlags.is_bloom_enabled(),
                    "cognitive_load": FeatureFlags.is_cognitive_load_enabled(),
                    "emoji_feedback": FeatureFlags.is_emoji_feedback_enabled(),
                    "progressive_assessment": FeatureFlags.is_progressive_assessment_enabled()
                }
            },
            "module_extraction": {
                "enabled": FeatureFlags.is_module_extraction_enabled(),
                "status": "active" if FeatureFlags.is_module_extraction_enabled() else "disabled (requires APRAG)",
                "features": {
                    "module_extraction": FeatureFlags.is_module_extraction_enabled(),
                    "module_quality_validation": FeatureFlags.is_module_quality_validation_enabled(),
                    "module_curriculum_alignment": FeatureFlags.is_module_curriculum_alignment_enabled()
                }
            }
        }
    
    @staticmethod
    def disable_all():
        """
        Tüm Eğitsel-KBRAG özelliklerini devre dışı bırak
        (Runtime'da environment değişkenlerini değiştir)
        """
        os.environ["ENABLE_EGITSEL_KBRAG"] = "false"
        os.environ["ENABLE_CACS"] = "false"
        os.environ["ENABLE_ZPD"] = "false"
        os.environ["ENABLE_BLOOM"] = "false"
        os.environ["ENABLE_COGNITIVE_LOAD"] = "false"
        os.environ["ENABLE_EMOJI_FEEDBACK"] = "false"
        os.environ["ENABLE_PROGRESSIVE_ASSESSMENT"] = "false"
        
        logger.info("All Eğitsel-KBRAG features disabled")
    
    @staticmethod
    def enable_all():
        """
        Tüm Eğitsel-KBRAG özelliklerini aktif et
        (APRAG aktifse)
        """
        if not FeatureFlags.is_aprag_enabled():
            logger.warning("Cannot enable Eğitsel-KBRAG: APRAG is not enabled")
            return False
        
        os.environ["ENABLE_EGITSEL_KBRAG"] = "true"
        os.environ["ENABLE_CACS"] = "true"
        os.environ["ENABLE_ZPD"] = "true"
        os.environ["ENABLE_BLOOM"] = "true"
        os.environ["ENABLE_COGNITIVE_LOAD"] = "true"
        os.environ["ENABLE_EMOJI_FEEDBACK"] = "true"
        os.environ["ENABLE_PROGRESSIVE_ASSESSMENT"] = "true"
        
        logger.info("All Eğitsel-KBRAG features enabled")
        return True


# Global instance
_feature_flags = FeatureFlags()


def get_feature_flags():
    """Get global feature flags instance"""
    return _feature_flags


def is_feature_enabled(feature_name: str, session_id: Optional[str] = None) -> bool:
    """
    Generic function to check if a feature is enabled.
    Used by EBARS and other modules.
    
    Args:
        feature_name: Name of the feature (e.g., 'ebars', 'aprag')
        session_id: Optional session ID for session-specific checks
        
    Returns:
        bool: Feature enabled?
    """
    if feature_name == "ebars":
        return FeatureFlags.is_ebars_enabled(session_id)
    elif feature_name == "aprag":
        return FeatureFlags.is_aprag_enabled(session_id)
    elif feature_name == "egitsel_kbrag":
        return FeatureFlags.is_egitsel_kbrag_enabled()
    else:
        logger.warning(f"Unknown feature name: {feature_name}")
        return False
