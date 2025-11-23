#!/usr/bin/env python3
"""Test imports for Eğitsel-KBRAG modules"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")

try:
    print("\n1️⃣ Testing CACS (Faz 2)...")
    from business_logic.cacs import get_cacs_scorer
    print("   ✅ business_logic.cacs imported")
    
    from api import scoring
    print("   ✅ api.scoring imported")
    
except Exception as e:
    print(f"   ❌ CACS import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2️⃣ Testing Pedagogical (Faz 3)...")
    from business_logic.pedagogical import (
        get_zpd_calculator,
        get_bloom_detector,
        get_cognitive_load_manager
    )
    print("   ✅ business_logic.pedagogical imported")
    
except Exception as e:
    print(f"   ❌ Pedagogical import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3️⃣ Testing Emoji Feedback (Faz 4)...")
    from api import emoji_feedback
    print("   ✅ api.emoji_feedback imported")
    
except Exception as e:
    print(f"   ❌ Emoji feedback import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n4️⃣ Testing Adaptive Query (Faz 5)...")
    from api import adaptive_query
    print("   ✅ api.adaptive_query imported")
    print(f"   ✅ Has router: {hasattr(adaptive_query, 'router')}")
    
except Exception as e:
    print(f"   ❌ Adaptive query import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Import test complete!")















