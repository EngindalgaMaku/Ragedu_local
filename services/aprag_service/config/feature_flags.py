"""
Feature Flags for APRAG Service and Eğitsel-KBRAG
All Eğitsel-KBRAG features are dependent on APRAG being enabled
"""

import os
import logging

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
                    "emoji_feedback": FeatureFlags.is_emoji_feedback_enabled()
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
        
        logger.info("All Eğitsel-KBRAG features enabled")
        return True


# Global instance
_feature_flags = FeatureFlags()


def get_feature_flags():
    """Get global feature flags instance"""
    return _feature_flags
