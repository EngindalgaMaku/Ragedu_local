"""
Feedback Handler
Handles emoji feedback and triggers adaptive response updates
"""

import logging
from typing import Optional, Dict, Any
from database.database import DatabaseManager
from .score_calculator import ComprehensionScoreCalculator
from .prompt_adapter import PromptAdapter

logger = logging.getLogger(__name__)


class FeedbackHandler:
    """Handle emoji feedback and manage adaptive responses"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.score_calculator = ComprehensionScoreCalculator(db)
        self.prompt_adapter = PromptAdapter(db)
    
    def process_feedback(
        self,
        user_id: str,
        session_id: str,
        emoji: str,
        interaction_id: Optional[int] = None,
        query_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process emoji feedback and update comprehension score.
        
        Args:
            user_id: User ID
            session_id: Session ID
            emoji: Emoji feedback ('👍', '😊', '😐', '❌')
            interaction_id: Optional interaction ID
            query_text: Optional query text for history
            
        Returns:
            Dict with updated state:
            {
                'success': bool,
                'previous_score': float,
                'new_score': float,
                'previous_difficulty': str,
                'new_difficulty': str,
                'difficulty_changed': bool,
                'adjustment_type': str,
                'prompt_parameters': dict
            }
        """
        try:
            # Validate emoji
            valid_emojis = ['👍', '😊', '😐', '❌']
            if emoji not in valid_emojis:
                logger.warning(f"Invalid emoji feedback: {emoji}")
                return {
                    'success': False,
                    'error': f'Invalid emoji. Must be one of: {valid_emojis}'
                }
            
            # Update comprehension score
            score_update = self.score_calculator.update_score(
                user_id=user_id,
                session_id=session_id,
                emoji=emoji,
                interaction_id=interaction_id
            )
            
            # Get updated prompt parameters
            prompt_params = self.prompt_adapter.get_prompt_parameters(
                user_id=user_id,
                session_id=session_id,
                difficulty_level=score_update['new_difficulty']
            )
            
            # Update interaction record if interaction_id provided
            if interaction_id:
                self._update_interaction_ebars_data(
                    interaction_id=interaction_id,
                    emoji=emoji,
                    score_update=score_update
                )
            
            logger.info(
                f"✅ Processed feedback: {user_id}/{session_id} "
                f"emoji={emoji} score={score_update['previous_score']:.1f}→{score_update['new_score']:.1f} "
                f"difficulty={score_update['previous_difficulty']}→{score_update['new_difficulty']}"
            )
            
            return {
                'success': True,
                'previous_score': score_update['previous_score'],
                'new_score': score_update['new_score'],
                'score_delta': score_update['score_delta'],
                'previous_difficulty': score_update['previous_difficulty'],
                'new_difficulty': score_update['new_difficulty'],
                'difficulty_changed': score_update['difficulty_changed'],
                'adjustment_type': score_update['adjustment_type'],
                'prompt_parameters': prompt_params,
                'consecutive_positive': score_update.get('consecutive_positive', 0),
                'consecutive_negative': score_update.get('consecutive_negative', 0),
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_interaction_ebars_data(
        self,
        interaction_id: int,
        emoji: str,
        score_update: Dict[str, Any]
    ):
        """Update interaction record with EBARS data"""
        try:
            with self.db.get_connection() as conn:
                # Check if emoji_feedback column exists and update it
                # Also update any EBARS-related metadata
                conn.execute("""
                    UPDATE student_interactions
                    SET emoji_feedback = ?,
                        feedback_score = ?
                    WHERE interaction_id = ?
                """, (
                    emoji,
                    score_update['new_score'] / 100.0,  # Normalize to 0-1
                    interaction_id
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not update interaction EBARS data: {e}")
            # Non-critical, continue
    
    def get_current_state(
        self,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get current EBARS state for a student.
        
        Returns:
            Dict with current state:
            {
                'comprehension_score': float,
                'difficulty_level': str,
                'prompt_parameters': dict,
                'statistics': dict
            }
        """
        try:
            # Get current score
            score = self.score_calculator.get_score(user_id, session_id)
            difficulty = self.score_calculator.get_difficulty_level(user_id, session_id)
            
            # Get prompt parameters
            prompt_params = self.prompt_adapter.get_prompt_parameters(
                user_id=user_id,
                session_id=session_id,
                difficulty_level=difficulty
            )
            
            # Get statistics
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT total_feedback_count, positive_feedback_count,
                           negative_feedback_count, consecutive_positive_count,
                           consecutive_negative_count, last_feedback_at
                    FROM student_comprehension_scores
                    WHERE user_id = ? AND session_id = ?
                """, (user_id, session_id))
                
                row = cursor.fetchone()
                
                if row:
                    stats = {
                        'total_feedback_count': row['total_feedback_count'] or 0,
                        'positive_feedback_count': row['positive_feedback_count'] or 0,
                        'negative_feedback_count': row['negative_feedback_count'] or 0,
                        'consecutive_positive': row['consecutive_positive_count'] or 0,
                        'consecutive_negative': row['consecutive_negative_count'] or 0,
                        'last_feedback_at': row['last_feedback_at'],
                    }
                else:
                    stats = {
                        'total_feedback_count': 0,
                        'positive_feedback_count': 0,
                        'negative_feedback_count': 0,
                        'consecutive_positive': 0,
                        'consecutive_negative': 0,
                        'last_feedback_at': None,
                    }
            
            # Generate sample prompt for display
            sample_prompt = self.generate_adaptive_prompt(
                user_id=user_id,
                session_id=session_id,
                base_prompt="Öğrencinin sorusunu yanıtla."
            )
            
            return {
                'comprehension_score': score,
                'difficulty_level': difficulty,
                'prompt_parameters': prompt_params,
                'statistics': stats,
                'adaptive_prompt': sample_prompt,  # Add generated prompt for display
            }
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {
                'comprehension_score': 50.0,
                'difficulty_level': 'normal',
                'prompt_parameters': {},
                'statistics': {},
            }
    
    def generate_adaptive_prompt(
        self,
        user_id: str,
        session_id: str,
        base_prompt: Optional[str] = None,
        query: Optional[str] = None,
        original_response: Optional[str] = None
    ) -> str:
        """
        Generate adaptive prompt for LLM based on current comprehension score.
        
        Args:
            user_id: User ID
            session_id: Session ID
            base_prompt: Optional base prompt
            query: Optional query text
            original_response: Optional original response to adapt
            
        Returns:
            Full adaptive prompt string
        """
        try:
            # Get current state
            state = self.get_current_state(user_id, session_id)
            score = state['comprehension_score']
            difficulty = state['difficulty_level']
            params = state['prompt_parameters']
            
            # Build adaptive prompt
            if original_response:
                # Full adaptive prompt with original response
                prompt = f"""Sen bir eğitim asistanısın. Aşağıdaki cevabı öğrencinin anlama seviyesine göre kişiselleştir.

🎯 ÖĞRENCİ ALGILAMA PUANI: {score:.1f}/100
📊 ZORLUK SEVİYESİ: {difficulty}

{self.prompt_adapter._build_instructions(params)}

📝 ORİJİNAL SORU:
{query or 'N/A'}

📄 ORİJİNAL CEVAP:
{original_response}

⚠️ ÇOK ÖNEMLİ - DOĞRULUK KURALLARI:
- SADECE orijinal cevapta ve ders materyallerinde bulunan bilgileri kullan
- Orijinal cevapta olmayan yeni bilgiler EKLEME
- Orijinal cevabın içeriğini koru, sadece sunumunu değiştir
- Emin olmadığın bilgileri uydurma veya tahmin etme

✅ ÖNEMLİ: Kişiselleştirilmiş cevabı SADECE TÜRKÇE olarak ver. Orijinal cevabın içeriğini koru, ancak sunumunu, detay seviyesini ve zorluk seviyesini öğrenci algılama puanına göre ayarla.
"""
            else:
                # Simple adaptive prompt
                prompt = self.prompt_adapter.generate_adaptive_prompt(
                    user_id=user_id,
                    session_id=session_id,
                    base_prompt=base_prompt
                )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating adaptive prompt: {e}")
            return base_prompt or ""

