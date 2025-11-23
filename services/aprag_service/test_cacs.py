#!/usr/bin/env python3
"""
CACS (Conversation-Aware Content Scoring) Tests
Tests for Faz 2 - Eğitsel-KBRAG integration
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from business_logic.cacs import CACSScorer, get_cacs_scorer, reset_cacs_scorer
from config.feature_flags import FeatureFlags


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


class TestCACSScorer:
    """Unit tests for CACS Scorer"""
    
    def setup_method(self):
        """Setup before each test"""
        os.environ["APRAG_ENABLED"] = "true"
        os.environ["ENABLE_EGITSEL_KBRAG"] = "true"
        os.environ["ENABLE_CACS"] = "true"
        reset_cacs_scorer()
    
    def test_scorer_initialization(self):
        """Test 1: CACS Scorer initialization"""
        print_section("Test 1: CACS Scorer Initialization")
        
        scorer = CACSScorer()
        
        assert scorer is not None
        assert scorer.w_base == 0.30
        assert scorer.w_personal == 0.25
        assert scorer.w_global == 0.25
        assert scorer.w_context == 0.20
        
        # Check weights sum to 1.0
        total = scorer.w_base + scorer.w_personal + scorer.w_global + scorer.w_context
        assert abs(total - 1.0) < 0.01
        
        print("✅ Scorer initialized correctly")
        print(f"   Weights: base={scorer.w_base}, personal={scorer.w_personal}, "
              f"global={scorer.w_global}, context={scorer.w_context}")
    
    def test_weight_normalization(self):
        """Test 2: Weight normalization"""
        print_section("Test 2: Weight Normalization")
        
        # Create scorer with non-normalized weights
        scorer = CACSScorer(
            w_base=0.4,
            w_personal=0.3,
            w_global=0.2,
            w_context=0.2  # Total = 1.1
        )
        
        # Check weights were normalized
        total = scorer.w_base + scorer.w_personal + scorer.w_global + scorer.w_context
        assert abs(total - 1.0) < 0.01
        
        print("✅ Weights normalized correctly")
        print(f"   Normalized: base={scorer.w_base:.3f}, personal={scorer.w_personal:.3f}, "
              f"global={scorer.w_global:.3f}, context={scorer.w_context:.3f}")
    
    def test_basic_scoring(self):
        """Test 3: Basic document scoring"""
        print_section("Test 3: Basic Document Scoring")
        
        scorer = CACSScorer()
        
        # Simple test case
        result = scorer.calculate_score(
            doc_id="doc1",
            base_score=0.8,
            student_profile={},
            conversation_history=[],
            global_scores={},
            current_query="test query"
        )
        
        assert 'final_score' in result
        assert 'base_score' in result
        assert 'personal_score' in result
        assert 'global_score' in result
        assert 'context_score' in result
        assert 'breakdown' in result
        
        assert 0.0 <= result['final_score'] <= 1.0
        
        print("✅ Basic scoring works")
        print(f"   Final score: {result['final_score']:.3f}")
        print(f"   Components: base={result['base_score']:.2f}, "
              f"personal={result['personal_score']:.2f}, "
              f"global={result['global_score']:.2f}, "
              f"context={result['context_score']:.2f}")
    
    def test_personal_score_with_history(self):
        """Test 4: Personal score calculation with history"""
        print_section("Test 4: Personal Score with History")
        
        scorer = CACSScorer()
        
        # History with feedback for the same document
        history = [
            {
                'doc_id': 'doc1',
                'feedback_score': 0.8,
                'query': 'previous question'
            },
            {
                'doc_id': 'doc1',
                'feedback_score': 0.9,
                'query': 'another question'
            }
        ]
        
        result = scorer.calculate_score(
            doc_id="doc1",
            base_score=0.7,
            student_profile={},
            conversation_history=history,
            global_scores={},
            current_query="current question"
        )
        
        # Personal score should be influenced by past positive feedback
        assert result['personal_score'] > 0.5
        
        print("✅ Personal score influenced by history")
        print(f"   Personal score: {result['personal_score']:.3f}")
        print(f"   (Expected > 0.5 due to positive past feedback)")
    
    def test_global_score_calculation(self):
        """Test 5: Global score calculation"""
        print_section("Test 5: Global Score Calculation")
        
        scorer = CACSScorer()
        
        # Global scores with positive feedback
        global_scores = {
            'doc1': {
                'total_feedback_count': 20,
                'positive_feedback_count': 15,
                'negative_feedback_count': 5,
                'avg_emoji_score': 0.75
            }
        }
        
        result = scorer.calculate_score(
            doc_id="doc1",
            base_score=0.7,
            student_profile={},
            conversation_history=[],
            global_scores=global_scores,
            current_query="test"
        )
        
        # Global score should be positive (15 positive / 20 total = 0.75)
        assert result['global_score'] > 0.5
        
        print("✅ Global score calculated from feedback")
        print(f"   Global score: {result['global_score']:.3f}")
        print(f"   (Based on 15 positive / 20 total feedback)")
    
    def test_context_score_calculation(self):
        """Test 6: Context score calculation"""
        print_section("Test 6: Context Score Calculation")
        
        scorer = CACSScorer()
        
        # History with similar queries
        history = [
            {'query': 'machine learning algorithms'},
            {'query': 'machine learning models'},
            {'query': 'learning theory'}
        ]
        
        # Current query similar to history
        result = scorer.calculate_score(
            doc_id="doc1",
            base_score=0.7,
            student_profile={},
            conversation_history=history,
            global_scores={},
            current_query="machine learning basics"
        )
        
        # Context score should be influenced by similar queries
        assert result['context_score'] >= 0.5
        
        print("✅ Context score influenced by query similarity")
        print(f"   Context score: {result['context_score']:.3f}")
    
    def test_feedback_normalization(self):
        """Test 7: Feedback normalization"""
        print_section("Test 7: Feedback Normalization")
        
        scorer = CACSScorer()
        
        # Test different feedback formats
        assert scorer._normalize_feedback(5) == 1.0  # 1-5 scale
        assert scorer._normalize_feedback(1) == 0.0  # 1-5 scale
        assert scorer._normalize_feedback(3) == 0.5  # 1-5 scale
        
        assert scorer._normalize_feedback(0.8) == 0.8  # 0-1 scale
        assert scorer._normalize_feedback(0.0) == 0.0  # 0-1 scale
        
        assert scorer._normalize_feedback('😊') == 0.7  # Emoji
        assert scorer._normalize_feedback('👍') == 1.0  # Emoji
        assert scorer._normalize_feedback('❌') == 0.0  # Emoji
        
        assert scorer._normalize_feedback(None) == 0.5  # Unknown/neutral
        
        print("✅ Feedback normalization works")
        print("   1-5 scale: 1→0.0, 3→0.5, 5→1.0")
        print("   Emojis: 😊→0.7, 👍→1.0, ❌→0.0")
    
    def test_cacs_disabled(self):
        """Test 8: CACS disabled fallback"""
        print_section("Test 8: CACS Disabled Fallback")
        
        # Disable CACS
        os.environ["ENABLE_CACS"] = "false"
        
        scorer = CACSScorer()
        
        result = scorer.calculate_score(
            doc_id="doc1",
            base_score=0.8,
            student_profile={},
            conversation_history=[],
            global_scores={},
            current_query="test"
        )
        
        # Should fallback to base score only
        assert result['final_score'] == 0.8
        assert result['cacs_enabled'] == False
        assert result['breakdown']['base_weight'] == 1.0
        
        print("✅ CACS disabled fallback works")
        print(f"   Returns base score only: {result['final_score']}")
        
        # Re-enable for other tests
        os.environ["ENABLE_CACS"] = "true"
    
    def test_get_cacs_scorer_singleton(self):
        """Test 9: Singleton pattern"""
        print_section("Test 9: Singleton Pattern")
        
        scorer1 = get_cacs_scorer()
        scorer2 = get_cacs_scorer()
        
        assert scorer1 is scorer2
        assert scorer1 is not None
        
        print("✅ Singleton pattern works")
        print("   get_cacs_scorer() returns same instance")


def test_integration():
    """Integration test with all components"""
    print_section("Integration Test: Full CACS Pipeline")
    
    # Ensure CACS is enabled
    os.environ["APRAG_ENABLED"] = "true"
    os.environ["ENABLE_EGITSEL_KBRAG"] = "true"
    os.environ["ENABLE_CACS"] = "true"
    
    reset_cacs_scorer()
    scorer = get_cacs_scorer()
    
    assert scorer is not None
    
    # Realistic test scenario
    student_profile = {
        'user_id': 'student1',
        'session_id': 'session1',
        'preferred_difficulty_level': 'intermediate',
        'success_rate': 0.75
    }
    
    conversation_history = [
        {
            'doc_id': 'doc1',
            'query': 'what is machine learning',
            'feedback_score': 0.8
        },
        {
            'doc_id': 'doc2',
            'query': 'explain neural networks',
            'feedback_score': 0.6
        },
        {
            'doc_id': 'doc1',
            'query': 'machine learning algorithms',
            'feedback_score': 0.9
        }
    ]
    
    global_scores = {
        'doc1': {
            'total_feedback_count': 50,
            'positive_feedback_count': 40,
            'negative_feedback_count': 10,
            'avg_emoji_score': 0.8
        },
        'doc2': {
            'total_feedback_count': 30,
            'positive_feedback_count': 20,
            'negative_feedback_count': 10,
            'avg_emoji_score': 0.67
        }
    }
    
    # Score doc1
    result1 = scorer.calculate_score(
        doc_id="doc1",
        base_score=0.85,
        student_profile=student_profile,
        conversation_history=conversation_history,
        global_scores=global_scores,
        current_query="deep learning basics"
    )
    
    # Score doc2
    result2 = scorer.calculate_score(
        doc_id="doc2",
        base_score=0.75,
        student_profile=student_profile,
        conversation_history=conversation_history,
        global_scores=global_scores,
        current_query="deep learning basics"
    )
    
    print(f"\n📊 Doc1 Scores:")
    print(f"   Final: {result1['final_score']:.3f}")
    print(f"   Base: {result1['base_score']:.2f} (weight: {result1['breakdown']['base_weight']})")
    print(f"   Personal: {result1['personal_score']:.2f} (weight: {result1['breakdown']['personal_weight']})")
    print(f"   Global: {result1['global_score']:.2f} (weight: {result1['breakdown']['global_weight']})")
    print(f"   Context: {result1['context_score']:.2f} (weight: {result1['breakdown']['context_weight']})")
    
    print(f"\n📊 Doc2 Scores:")
    print(f"   Final: {result2['final_score']:.3f}")
    print(f"   Base: {result2['base_score']:.2f}")
    print(f"   Personal: {result2['personal_score']:.2f}")
    print(f"   Global: {result2['global_score']:.2f}")
    print(f"   Context: {result2['context_score']:.2f}")
    
    # Doc1 should score higher (better history, better global feedback)
    print(f"\n🏆 Ranking: doc1 ({result1['final_score']:.3f}) vs doc2 ({result2['final_score']:.3f})")
    
    assert result1['final_score'] > result2['final_score']
    
    print("\n✅ Integration test passed")
    print("   Doc1 ranked higher due to better personal and global scores")


def main():
    """Run all tests"""
    print_section("🧪 CACS Tests - Faz 2: Eğitsel-KBRAG")
    
    test_suite = TestCACSScorer()
    
    try:
        # Run unit tests
        test_suite.setup_method()
        test_suite.test_scorer_initialization()
        
        test_suite.setup_method()
        test_suite.test_weight_normalization()
        
        test_suite.setup_method()
        test_suite.test_basic_scoring()
        
        test_suite.setup_method()
        test_suite.test_personal_score_with_history()
        
        test_suite.setup_method()
        test_suite.test_global_score_calculation()
        
        test_suite.setup_method()
        test_suite.test_context_score_calculation()
        
        test_suite.setup_method()
        test_suite.test_feedback_normalization()
        
        test_suite.setup_method()
        test_suite.test_cacs_disabled()
        
        test_suite.setup_method()
        test_suite.test_get_cacs_scorer_singleton()
        
        # Run integration test
        test_integration()
        
        # Summary
        print_section("✅ ALL TESTS PASSED")
        print("\n🎉 CACS Implementation Working Correctly!")
        print("\nKey Points:")
        print("  ✅ Scorer initialization and normalization work")
        print("  ✅ All 4 score components calculate correctly")
        print("  ✅ Personal score influenced by history")
        print("  ✅ Global score reflects community feedback")
        print("  ✅ Context score considers query similarity")
        print("  ✅ Feedback normalization handles multiple formats")
        print("  ✅ CACS disabled fallback works")
        print("  ✅ Singleton pattern implemented")
        print("  ✅ Integration test successful")
        print("\n" + "=" * 60 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())















