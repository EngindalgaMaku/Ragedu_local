"""
EBARS - Emoji-Based Adaptive Response System
Modular system for adaptive difficulty adjustment based on emoji feedback
"""

from .score_calculator import ComprehensionScoreCalculator
from .prompt_adapter import PromptAdapter
from .feedback_handler import FeedbackHandler

__all__ = [
    "ComprehensionScoreCalculator",
    "PromptAdapter",
    "FeedbackHandler",
]

