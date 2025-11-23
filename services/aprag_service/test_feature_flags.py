#!/usr/bin/env python3
"""
Test Feature Flags - Eğitsel-KBRAG
Test that all features are properly dependent on APRAG being enabled
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.feature_flags import FeatureFlags


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_feature_flags():
    """Test all feature flag combinations"""
    
    print_section("🧪 Feature Flags Test - Eğitsel-KBRAG")
    
    # Test 1: APRAG Disabled - All should be disabled
    print("\n📋 Test 1: APRAG Disabled")
    print("-" * 60)
    os.environ["APRAG_ENABLED"] = "false"
    
    print(f"APRAG Enabled: {FeatureFlags.is_aprag_enabled()}")
    print(f"Eğitsel-KBRAG Enabled: {FeatureFlags.is_egitsel_kbrag_enabled()}")
    print(f"  └─ CACS: {FeatureFlags.is_cacs_enabled()}")
    print(f"  └─ ZPD: {FeatureFlags.is_zpd_enabled()}")
    print(f"  └─ Bloom: {FeatureFlags.is_bloom_enabled()}")
    print(f"  └─ Cognitive Load: {FeatureFlags.is_cognitive_load_enabled()}")
    print(f"  └─ Emoji Feedback: {FeatureFlags.is_emoji_feedback_enabled()}")
    
    # Verify all are disabled
    assert not FeatureFlags.is_aprag_enabled(), "❌ APRAG should be disabled"
    assert not FeatureFlags.is_egitsel_kbrag_enabled(), "❌ Eğitsel-KBRAG should be disabled"
    assert not FeatureFlags.is_cacs_enabled(), "❌ CACS should be disabled"
    assert not FeatureFlags.is_zpd_enabled(), "❌ ZPD should be disabled"
    assert not FeatureFlags.is_bloom_enabled(), "❌ Bloom should be disabled"
    assert not FeatureFlags.is_cognitive_load_enabled(), "❌ Cognitive Load should be disabled"
    assert not FeatureFlags.is_emoji_feedback_enabled(), "❌ Emoji Feedback should be disabled"
    
    print("\n✅ Test 1 PASSED: All features disabled when APRAG is disabled")
    
    # Test 2: APRAG Enabled, Eğitsel-KBRAG Disabled
    print("\n📋 Test 2: APRAG Enabled, Eğitsel-KBRAG Disabled")
    print("-" * 60)
    os.environ["APRAG_ENABLED"] = "true"
    os.environ["ENABLE_EGITSEL_KBRAG"] = "false"
    
    print(f"APRAG Enabled: {FeatureFlags.is_aprag_enabled()}")
    print(f"Eğitsel-KBRAG Enabled: {FeatureFlags.is_egitsel_kbrag_enabled()}")
    print(f"  └─ CACS: {FeatureFlags.is_cacs_enabled()}")
    print(f"  └─ ZPD: {FeatureFlags.is_zpd_enabled()}")
    print(f"  └─ Bloom: {FeatureFlags.is_bloom_enabled()}")
    print(f"  └─ Cognitive Load: {FeatureFlags.is_cognitive_load_enabled()}")
    print(f"  └─ Emoji Feedback: {FeatureFlags.is_emoji_feedback_enabled()}")
    
    # Verify APRAG is enabled but Eğitsel-KBRAG features are disabled
    assert FeatureFlags.is_aprag_enabled(), "❌ APRAG should be enabled"
    assert not FeatureFlags.is_egitsel_kbrag_enabled(), "❌ Eğitsel-KBRAG should be disabled"
    assert not FeatureFlags.is_cacs_enabled(), "❌ CACS should be disabled"
    assert not FeatureFlags.is_zpd_enabled(), "❌ ZPD should be disabled"
    assert not FeatureFlags.is_bloom_enabled(), "❌ Bloom should be disabled"
    assert not FeatureFlags.is_cognitive_load_enabled(), "❌ Cognitive Load should be disabled"
    assert not FeatureFlags.is_emoji_feedback_enabled(), "❌ Emoji Feedback should be disabled"
    
    print("\n✅ Test 2 PASSED: Eğitsel-KBRAG features disabled independently")
    
    # Test 3: Both Enabled, All Features Enabled
    print("\n📋 Test 3: APRAG & Eğitsel-KBRAG Enabled (All Features ON)")
    print("-" * 60)
    os.environ["APRAG_ENABLED"] = "true"
    os.environ["ENABLE_EGITSEL_KBRAG"] = "true"
    os.environ["ENABLE_CACS"] = "true"
    os.environ["ENABLE_ZPD"] = "true"
    os.environ["ENABLE_BLOOM"] = "true"
    os.environ["ENABLE_COGNITIVE_LOAD"] = "true"
    os.environ["ENABLE_EMOJI_FEEDBACK"] = "true"
    
    print(f"APRAG Enabled: {FeatureFlags.is_aprag_enabled()}")
    print(f"Eğitsel-KBRAG Enabled: {FeatureFlags.is_egitsel_kbrag_enabled()}")
    print(f"  └─ CACS: {FeatureFlags.is_cacs_enabled()}")
    print(f"  └─ ZPD: {FeatureFlags.is_zpd_enabled()}")
    print(f"  └─ Bloom: {FeatureFlags.is_bloom_enabled()}")
    print(f"  └─ Cognitive Load: {FeatureFlags.is_cognitive_load_enabled()}")
    print(f"  └─ Emoji Feedback: {FeatureFlags.is_emoji_feedback_enabled()}")
    
    # Verify all are enabled
    assert FeatureFlags.is_aprag_enabled(), "❌ APRAG should be enabled"
    assert FeatureFlags.is_egitsel_kbrag_enabled(), "❌ Eğitsel-KBRAG should be enabled"
    assert FeatureFlags.is_cacs_enabled(), "❌ CACS should be enabled"
    assert FeatureFlags.is_zpd_enabled(), "❌ ZPD should be enabled"
    assert FeatureFlags.is_bloom_enabled(), "❌ Bloom should be enabled"
    assert FeatureFlags.is_cognitive_load_enabled(), "❌ Cognitive Load should be enabled"
    assert FeatureFlags.is_emoji_feedback_enabled(), "❌ Emoji Feedback should be enabled"
    
    print("\n✅ Test 3 PASSED: All features enabled correctly")
    
    # Test 4: Selective Feature Enabling
    print("\n📋 Test 4: Selective Feature Enabling (Only CACS & Emoji)")
    print("-" * 60)
    os.environ["APRAG_ENABLED"] = "true"
    os.environ["ENABLE_EGITSEL_KBRAG"] = "true"
    os.environ["ENABLE_CACS"] = "true"
    os.environ["ENABLE_ZPD"] = "false"
    os.environ["ENABLE_BLOOM"] = "false"
    os.environ["ENABLE_COGNITIVE_LOAD"] = "false"
    os.environ["ENABLE_EMOJI_FEEDBACK"] = "true"
    
    print(f"APRAG Enabled: {FeatureFlags.is_aprag_enabled()}")
    print(f"Eğitsel-KBRAG Enabled: {FeatureFlags.is_egitsel_kbrag_enabled()}")
    print(f"  └─ CACS: {FeatureFlags.is_cacs_enabled()}")
    print(f"  └─ ZPD: {FeatureFlags.is_zpd_enabled()}")
    print(f"  └─ Bloom: {FeatureFlags.is_bloom_enabled()}")
    print(f"  └─ Cognitive Load: {FeatureFlags.is_cognitive_load_enabled()}")
    print(f"  └─ Emoji Feedback: {FeatureFlags.is_emoji_feedback_enabled()}")
    
    # Verify selective enabling
    assert FeatureFlags.is_aprag_enabled(), "❌ APRAG should be enabled"
    assert FeatureFlags.is_egitsel_kbrag_enabled(), "❌ Eğitsel-KBRAG should be enabled"
    assert FeatureFlags.is_cacs_enabled(), "❌ CACS should be enabled"
    assert not FeatureFlags.is_zpd_enabled(), "❌ ZPD should be disabled"
    assert not FeatureFlags.is_bloom_enabled(), "❌ Bloom should be disabled"
    assert not FeatureFlags.is_cognitive_load_enabled(), "❌ Cognitive Load should be disabled"
    assert FeatureFlags.is_emoji_feedback_enabled(), "❌ Emoji Feedback should be enabled"
    
    print("\n✅ Test 4 PASSED: Selective feature enabling works correctly")
    
    # Test 5: Status Report
    print("\n📋 Test 5: Status Report")
    print("-" * 60)
    
    status = FeatureFlags.get_status_report()
    
    print("\nStatus Report:")
    print(f"  APRAG: {status['aprag']['status']}")
    print(f"  Eğitsel-KBRAG: {status['egitsel_kbrag']['status']}")
    print(f"    Features:")
    for feature, enabled in status['egitsel_kbrag']['features'].items():
        print(f"      - {feature}: {'✅' if enabled else '❌'}")
    
    print("\n✅ Test 5 PASSED: Status report generated successfully")
    
    # Test 6: Helper Methods
    print("\n📋 Test 6: Helper Methods (disable_all, enable_all)")
    print("-" * 60)
    
    # Enable all
    os.environ["APRAG_ENABLED"] = "true"
    success = FeatureFlags.enable_all()
    assert success, "❌ enable_all should succeed when APRAG is enabled"
    assert FeatureFlags.is_cacs_enabled(), "❌ CACS should be enabled"
    print("✅ enable_all() works correctly")
    
    # Disable all
    FeatureFlags.disable_all()
    assert not FeatureFlags.is_cacs_enabled(), "❌ CACS should be disabled"
    assert not FeatureFlags.is_zpd_enabled(), "❌ ZPD should be disabled"
    print("✅ disable_all() works correctly")
    
    # Try to enable_all when APRAG is disabled
    os.environ["APRAG_ENABLED"] = "false"
    success = FeatureFlags.enable_all()
    assert not success, "❌ enable_all should fail when APRAG is disabled"
    print("✅ enable_all() properly fails when APRAG is disabled")
    
    print("\n✅ Test 6 PASSED: Helper methods work correctly")
    
    # Final Summary
    print_section("✅ ALL TESTS PASSED")
    print("\n🎉 Feature Flag System Working Correctly!")
    print("\nKey Points:")
    print("  ✅ APRAG controls all Eğitsel-KBRAG features")
    print("  ✅ When APRAG is disabled, all features are disabled")
    print("  ✅ Eğitsel-KBRAG can be toggled independently")
    print("  ✅ Individual features can be toggled selectively")
    print("  ✅ Status report and helper methods work correctly")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        test_feature_flags()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)















