#!/usr/bin/env python3
"""Check if Eğitsel-KBRAG modules can be imported"""

import sys
import os

# Change to correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}")

# Check module availability flags
try:
    print("\n=== Checking main.py imports ===")
    
    # Try to import config first
    from config.feature_flags import FeatureFlags
    print("✅ FeatureFlags imported")
    
    # Check scoring
    try:
        from api import scoring
        print(f"✅ SCORING_AVAILABLE: True")
        print(f"   Has router: {hasattr(scoring, 'router')}")
    except Exception as e:
        print(f"❌ SCORING_AVAILABLE: False ({e})")
    
    # Check emoji_feedback
    try:
        from api import emoji_feedback
        print(f"✅ EMOJI_FEEDBACK_AVAILABLE: True")
        print(f"   Has router: {hasattr(emoji_feedback, 'router')}")
    except Exception as e:
        print(f"❌ EMOJI_FEEDBACK_AVAILABLE: False ({e})")
    
    # Check adaptive_query
    try:
        from api import adaptive_query
        print(f"✅ ADAPTIVE_QUERY_AVAILABLE: True")
        print(f"   Has router: {hasattr(adaptive_query, 'router')}")
    except Exception as e:
        print(f"❌ ADAPTIVE_QUERY_AVAILABLE: False ({e})")
    
    # Check feature flags
    print("\n=== Feature Flags ===")
    print(f"EGITSEL_KBRAG_ENABLED: {FeatureFlags.is_egitsel_kbrag_enabled()}")
    print(f"CACS_ENABLED: {FeatureFlags.is_cacs_enabled()}")
    print(f"ZPD_ENABLED: {FeatureFlags.is_zpd_enabled()}")
    print(f"BLOOM_ENABLED: {FeatureFlags.is_bloom_enabled()}")
    print(f"COGNITIVE_LOAD_ENABLED: {FeatureFlags.is_cognitive_load_enabled()}")
    print(f"EMOJI_FEEDBACK_ENABLED: {FeatureFlags.is_emoji_feedback_enabled()}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Check Complete ===")















