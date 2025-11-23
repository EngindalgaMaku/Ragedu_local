#!/usr/bin/env python3
"""Test startup logic"""

import sys
import os

# Change to correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("=== Testing Startup Logic ===\n")

print("1️⃣ Importing feature flags...")
from config.feature_flags import FeatureFlags
print(f"   EGITSEL_KBRAG enabled: {FeatureFlags.is_egitsel_kbrag_enabled()}")
print(f"   CACS enabled: {FeatureFlags.is_cacs_enabled()}")

print("\n2️⃣ Trying to import api.scoring...")
try:
    from api import scoring
    SCORING_AVAILABLE = True
    print(f"   ✅ SCORING_AVAILABLE: {SCORING_AVAILABLE}")
    print(f"   Has router: {hasattr(scoring, 'router')}")
except ImportError as e:
    SCORING_AVAILABLE = False
    print(f"   ❌ SCORING_AVAILABLE: {SCORING_AVAILABLE}")
    print(f"   Error: {e}")

print("\n3️⃣ Trying to import api.emoji_feedback...")
try:
    from api import emoji_feedback
    EMOJI_FEEDBACK_AVAILABLE = True
    print(f"   ✅ EMOJI_FEEDBACK_AVAILABLE: {EMOJI_FEEDBACK_AVAILABLE}")
    print(f"   Has router: {hasattr(emoji_feedback, 'router')}")
except ImportError as e:
    EMOJI_FEEDBACK_AVAILABLE = False
    print(f"   ❌ EMOJI_FEEDBACK_AVAILABLE: {EMOJI_FEEDBACK_AVAILABLE}")
    print(f"   Error: {e}")

print("\n4️⃣ Trying to import api.adaptive_query...")
try:
    from api import adaptive_query
    ADAPTIVE_QUERY_AVAILABLE = True
    print(f"   ✅ ADAPTIVE_QUERY_AVAILABLE: {ADAPTIVE_QUERY_AVAILABLE}")
    print(f"   Has router: {hasattr(adaptive_query, 'router')}")
except ImportError as e:
    ADAPTIVE_QUERY_AVAILABLE = False
    print(f"   ❌ ADAPTIVE_QUERY_AVAILABLE: {ADAPTIVE_QUERY_AVAILABLE}")
    print(f"   Error: {e}")

print("\n5️⃣ Checking router registration logic...")
if SCORING_AVAILABLE and FeatureFlags.is_cacs_enabled():
    print("   ✅ Scoring router WOULD be registered")
else:
    print(f"   ❌ Scoring router WOULD NOT be registered (avail={SCORING_AVAILABLE}, enabled={FeatureFlags.is_cacs_enabled()})")

if EMOJI_FEEDBACK_AVAILABLE and FeatureFlags.is_emoji_feedback_enabled():
    print("   ✅ Emoji feedback router WOULD be registered")
else:
    print(f"   ❌ Emoji feedback router WOULD NOT be registered (avail={EMOJI_FEEDBACK_AVAILABLE}, enabled={FeatureFlags.is_emoji_feedback_enabled()})")

if ADAPTIVE_QUERY_AVAILABLE and FeatureFlags.is_egitsel_kbrag_enabled():
    print("   ✅ Adaptive query router WOULD be registered")
else:
    print(f"   ❌ Adaptive query router WOULD NOT be registered (avail={ADAPTIVE_QUERY_AVAILABLE}, enabled={FeatureFlags.is_egitsel_kbrag_enabled()})")

print("\n=== Test Complete ===")















