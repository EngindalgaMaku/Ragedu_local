"""
Business Logic Layer for Eğitsel-KBRAG
All features are dependent on APRAG being enabled
"""

from config.feature_flags import FeatureFlags

__all__ = []

# Conditionally import based on feature flags
if FeatureFlags.is_egitsel_kbrag_enabled():
    if FeatureFlags.is_cacs_enabled():
        try:
            from .cacs import get_cacs_scorer, CACSScorer
            __all__.extend(['get_cacs_scorer', 'CACSScorer'])
        except ImportError:
            pass
    
    if FeatureFlags.is_zpd_enabled() or FeatureFlags.is_bloom_enabled() or FeatureFlags.is_cognitive_load_enabled():
        try:
            from .pedagogical import (
                get_zpd_calculator,
                get_bloom_detector,
                get_cognitive_load_manager
            )
            __all__.extend([
                'get_zpd_calculator',
                'get_bloom_detector',
                'get_cognitive_load_manager'
            ])
        except ImportError:
            pass















