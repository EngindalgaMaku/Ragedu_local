"""
Comprehension Score Calculator
Calculates and updates comprehension scores (0-100) based on emoji feedback
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from database.database import DatabaseManager

logger = logging.getLogger(__name__)

# Base emoji to delta mapping (will be adjusted dynamically)
EMOJI_BASE_DELTA = {
    '👍': +5,   # Mükemmel - Öğrenci tam anladı
    '😊': +2,   # Anladım - Öğrenci genel olarak anladı
    '😐': -3,   # Karışık - Öğrenci zorlanıyor
    '❌': -5,   # Anlamadım - Öğrenci anlamadı
}

def calculate_adaptive_delta(base_delta: float, current_score: float) -> float:
    """
    Mevcut puana göre delta'yı ayarla (Dinamik Delta Sistemi).
    - Yüksek puanlarda (70+): Daha küçük delta (daha yavaş değişim)
    - Düşük puanlarda (30-): Daha büyük delta (daha hızlı toparlanma)
    - Orta puanlarda (30-70): Normal delta
    """
    if current_score >= 70:
        # Yüksek seviyede, küçük değişimler (aşırı yükselme engelleme)
        return base_delta * 0.7
    elif current_score <= 30:
        # Düşük seviyede, büyük değişimler (hızlı toparlanma)
        return base_delta * 1.3
    else:
        # Orta seviyede, normal değişim
        return base_delta

# Difficulty level thresholds with hysteresis (Histeresis Mekanizması)
# Histeresis: Eşik geçişlerinde sürekli geçişi önlemek için farklı giriş/çıkış eşikleri
DIFFICULTY_THRESHOLDS_HYSTERESIS = {
    'very_struggling': {
        'enter': 25,  # 25'e düşerse girer
        'exit': 35    # 35'e çıkarsa çıkar
    },
    'struggling': {
        'enter': 40,  # 40'a düşerse girer
        'exit': 50    # 50'ye çıkarsa çıkar
    },
    'normal': {
        'enter': 50,  # 50'ye çıkarsa girer
        'exit': 75    # 75'e çıkarsa çıkar
    },
    'good': {
        'enter': 75,  # 75'e çıkarsa girer
        'exit': 85    # 85'e çıkarsa çıkar
    },
    'excellent': {
        'enter': 85,  # 85'e çıkarsa girer
        'exit': 100   # 100'e çıkarsa çıkar
    }
}

# Fallback thresholds (for initial assignment)
DIFFICULTY_THRESHOLDS = {
    'very_struggling': (0, 30),      # 0-30
    'struggling': (31, 45),          # 31-45
    'normal': (46, 70),              # 46-70
    'good': (71, 80),                # 71-80
    'excellent': (81, 100),          # 81-100
}


class ComprehensionScoreCalculator:
    """Calculate and manage comprehension scores for students"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def get_score(self, user_id: str, session_id: str) -> float:
        """
        Get current comprehension score for a student-session pair.
        Creates default score (50.0) if doesn't exist.
        
        Args:
            user_id: User ID
            session_id: Session ID
            
        Returns:
            Current comprehension score (0-100)
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT comprehension_score
                    FROM student_comprehension_scores
                    WHERE user_id = ? AND session_id = ?
                """, (user_id, session_id))
                
                row = cursor.fetchone()
                
                if row:
                    return float(row["comprehension_score"])
                else:
                    # Create default score
                    return self._create_default_score(conn, user_id, session_id)
                    
        except Exception as e:
            logger.error(f"Error getting comprehension score: {e}")
            return 50.0  # Default fallback
    
    def _create_default_score(self, conn, user_id: str, session_id: str) -> float:
        """Create default comprehension score entry"""
        try:
            conn.execute("""
                INSERT INTO student_comprehension_scores (
                    user_id, session_id, comprehension_score,
                    current_difficulty_level, created_at, last_updated
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (user_id, session_id, 50.0, 'normal'))
            conn.commit()
            return 50.0
        except Exception as e:
            logger.error(f"Error creating default score: {e}")
            return 50.0
    
    def update_score(
        self,
        user_id: str,
        session_id: str,
        emoji: str,
        interaction_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update comprehension score based on emoji feedback.
        
        Args:
            user_id: User ID
            session_id: Session ID
            emoji: Emoji feedback ('👍', '😊', '😐', '❌')
            interaction_id: Optional interaction ID for history tracking
            
        Returns:
            Dict with updated score info:
            {
                'previous_score': float,
                'new_score': float,
                'score_delta': float,
                'previous_difficulty': str,
                'new_difficulty': str,
                'difficulty_changed': bool,
                'adjustment_type': str
            }
        """
        try:
            # Get base delta for emoji
            base_delta = EMOJI_BASE_DELTA.get(emoji, 0)
            
            if base_delta == 0:
                logger.warning(f"Unknown emoji: {emoji}")
                return self._get_current_state(user_id, session_id)
            
            with self.db.get_connection() as conn:
                # Get current score
                cursor = conn.execute("""
                    SELECT comprehension_score, current_difficulty_level,
                           consecutive_positive_count, consecutive_negative_count,
                           total_feedback_count, positive_feedback_count, negative_feedback_count
                    FROM student_comprehension_scores
                    WHERE user_id = ? AND session_id = ?
                """, (user_id, session_id))
                
                row = cursor.fetchone()
                
                if not row:
                    # Create default entry
                    self._create_default_score(conn, user_id, session_id)
                    previous_score = 50.0
                    previous_difficulty = 'normal'
                    consecutive_positive = 0
                    consecutive_negative = 0
                    total_feedback = 0
                    positive_feedback = 0
                    negative_feedback = 0
                else:
                    previous_score = float(row["comprehension_score"])
                    previous_difficulty = row["current_difficulty_level"]
                    consecutive_positive = row["consecutive_positive_count"] or 0
                    consecutive_negative = row["consecutive_negative_count"] or 0
                    total_feedback = row["total_feedback_count"] or 0
                    positive_feedback = row["positive_feedback_count"] or 0
                    negative_feedback = row["negative_feedback_count"] or 0
                
                # Calculate adaptive delta based on current score
                delta = calculate_adaptive_delta(base_delta, previous_score)
                
                # Calculate new score
                new_score = previous_score + delta
                new_score = max(0.0, min(100.0, new_score))  # Clamp to 0-100
                
                # Determine new difficulty level with hysteresis
                new_difficulty = self._score_to_difficulty_with_hysteresis(
                    new_score, previous_difficulty
                )
                difficulty_changed = (previous_difficulty != new_difficulty)
                
                # Update consecutive counts
                is_positive = emoji in ['👍', '😊']
                is_negative = emoji in ['😐', '❌']
                
                if is_positive:
                    consecutive_positive += 1
                    consecutive_negative = 0
                    positive_feedback += 1
                elif is_negative:
                    consecutive_negative += 1
                    consecutive_positive = 0
                    negative_feedback += 1
                
                total_feedback += 1
                
                # Determine adjustment type
                adjustment_type = self._determine_adjustment_type(
                    previous_score, new_score, emoji,
                    consecutive_positive, consecutive_negative
                )
                
                # Update database
                conn.execute("""
                    UPDATE student_comprehension_scores
                    SET comprehension_score = ?,
                        current_difficulty_level = ?,
                        consecutive_positive_count = ?,
                        consecutive_negative_count = ?,
                        total_feedback_count = ?,
                        positive_feedback_count = ?,
                        negative_feedback_count = ?,
                        last_updated = CURRENT_TIMESTAMP,
                        last_feedback_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND session_id = ?
                """, (
                    new_score, new_difficulty,
                    consecutive_positive, consecutive_negative,
                    total_feedback, positive_feedback, negative_feedback,
                    user_id, session_id
                ))
                
                # Record in history
                if interaction_id:
                    conn.execute("""
                        INSERT INTO ebars_feedback_history (
                            user_id, session_id, interaction_id,
                            emoji_feedback, previous_score, score_delta, new_score,
                            previous_difficulty_level, new_difficulty_level, difficulty_changed,
                            adjustment_type, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        user_id, session_id, interaction_id,
                        emoji, previous_score, delta, new_score,
                        previous_difficulty, new_difficulty, difficulty_changed,
                        adjustment_type
                    ))
                
                conn.commit()
                
                logger.info(
                    f"✅ Updated comprehension score: {user_id}/{session_id} "
                    f"{previous_score:.1f} → {new_score:.1f} ({delta:+.1f}) "
                    f"[{previous_difficulty} → {new_difficulty}]"
                )
                
                return {
                    'previous_score': previous_score,
                    'new_score': new_score,
                    'score_delta': delta,
                    'previous_difficulty': previous_difficulty,
                    'new_difficulty': new_difficulty,
                    'difficulty_changed': difficulty_changed,
                    'adjustment_type': adjustment_type,
                    'consecutive_positive': consecutive_positive,
                    'consecutive_negative': consecutive_negative,
                }
                
        except Exception as e:
            logger.error(f"Error updating comprehension score: {e}")
            return self._get_current_state(user_id, session_id)
    
    def _score_to_difficulty(self, score: float) -> str:
        """Convert score to difficulty level (fallback method)"""
        for level, (min_score, max_score) in DIFFICULTY_THRESHOLDS.items():
            if min_score <= score <= max_score:
                return level
        return 'normal'  # Default fallback
    
    def _score_to_difficulty_with_hysteresis(
        self, 
        score: float, 
        current_difficulty: str
    ) -> str:
        """
        Convert score to difficulty level with hysteresis mechanism.
        Prevents rapid switching between difficulty levels.
        """
        # Get current level's exit threshold
        current_thresholds = DIFFICULTY_THRESHOLDS_HYSTERESIS.get(current_difficulty)
        
        if current_thresholds:
            exit_threshold = current_thresholds['exit']
            
            # Check if we should exit current level
            if current_difficulty == 'very_struggling' and score >= exit_threshold:
                # Moving up from very_struggling
                if score >= DIFFICULTY_THRESHOLDS_HYSTERESIS['struggling']['enter']:
                    return 'struggling'
            elif current_difficulty == 'struggling':
                if score >= exit_threshold:
                    return 'normal'
                elif score < DIFFICULTY_THRESHOLDS_HYSTERESIS['very_struggling']['enter']:
                    return 'very_struggling'
            elif current_difficulty == 'normal':
                if score >= exit_threshold:
                    return 'good'
                elif score < DIFFICULTY_THRESHOLDS_HYSTERESIS['struggling']['enter']:
                    return 'struggling'
            elif current_difficulty == 'good':
                if score >= exit_threshold:
                    return 'excellent'
                elif score < DIFFICULTY_THRESHOLDS_HYSTERESIS['normal']['enter']:
                    return 'normal'
            elif current_difficulty == 'excellent':
                if score < DIFFICULTY_THRESHOLDS_HYSTERESIS['good']['enter']:
                    return 'good'
            
            # Stay in current level if within hysteresis range
            return current_difficulty
        else:
            # Fallback to simple threshold if current_difficulty not found
            return self._score_to_difficulty(score)
    
    def _determine_adjustment_type(
        self,
        previous_score: float,
        new_score: float,
        emoji: str,
        consecutive_positive: int,
        consecutive_negative: int
    ) -> str:
        """Determine the type of adjustment made"""
        
        # Immediate drop: 3 consecutive negative feedbacks (2→3, daha dengeli)
        if consecutive_negative >= 3:
            return 'immediate_drop'
        
        # Immediate raise: 4 consecutive positive feedbacks (5→4, daha hızlı)
        if consecutive_positive >= 4:
            return 'immediate_raise'
        
        # Proactive increase: score high and positive feedback
        if new_score >= 70 and emoji in ['👍', '😊']:
            return 'proactive_increase'
        
        # Reactive decrease: score low or negative feedback
        if new_score < 50 or emoji in ['😐', '❌']:
            return 'reactive_decrease'
        
        return 'normal_update'
    
    def _get_current_state(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get current state without updating"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT comprehension_score, current_difficulty_level
                    FROM student_comprehension_scores
                    WHERE user_id = ? AND session_id = ?
                """, (user_id, session_id))
                
                row = cursor.fetchone()
                
                if row:
                    score = float(row["comprehension_score"])
                    difficulty = row["current_difficulty_level"]
                else:
                    score = 50.0
                    difficulty = 'normal'
                
                return {
                    'previous_score': score,
                    'new_score': score,
                    'score_delta': 0.0,
                    'previous_difficulty': difficulty,
                    'new_difficulty': difficulty,
                    'difficulty_changed': False,
                    'adjustment_type': 'no_change',
                }
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {
                'previous_score': 50.0,
                'new_score': 50.0,
                'score_delta': 0.0,
                'previous_difficulty': 'normal',
                'new_difficulty': 'normal',
                'difficulty_changed': False,
                'adjustment_type': 'no_change',
            }
    
    def get_difficulty_level(self, user_id: str, session_id: str) -> str:
        """Get current difficulty level for student"""
        score = self.get_score(user_id, session_id)
        return self._score_to_difficulty(score)

