"""
CACS (Conversation-Aware Content Scoring) Algorithm
Eğitsel-KBRAG paper implementation

This module requires APRAG to be enabled.
"""

from typing import Dict, List, Optional, Any
import logging
import json

# Import feature flags
try:
    from config.feature_flags import FeatureFlags
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class CACSScorer:
    """
    CACS Algoritması (Conversation-Aware Content Scoring)
    
    Her dokümanı dört farklı açıdan skorlar:
    1. Base Score (30%): Semantik benzerlik (RAG'dan gelen)
    2. Personal Score (25%): Öğrenci geçmişi ve profil uyumu
    3. Global Score (25%): Genel öğrenci geri bildirimleri
    4. Context Score (20%): Konuşma bağlamı uyumu
    
    Final Score = Σ(weight_i × score_i)
    """
    
    def __init__(
        self,
        w_base: float = 0.30,
        w_personal: float = 0.25,
        w_global: float = 0.25,
        w_context: float = 0.20
    ):
        """
        Initialize CACS Scorer
        
        Args:
            w_base: Base score ağırlığı (semantik benzerlik)
            w_personal: Personal score ağırlığı (öğrenci profili)
            w_global: Global score ağırlığı (toplam geri bildirim)
            w_context: Context score ağırlığı (konuşma bağlamı)
        """
        # Check if CACS is enabled
        if not FeatureFlags.is_cacs_enabled():
            logger.warning("CACS is not enabled. Feature flag check failed.")
        
        self.w_base = w_base
        self.w_personal = w_personal
        self.w_global = w_global
        self.w_context = w_context
        
        # Toplam ağırlık 1.0 olmalı - normalize et
        total = w_base + w_personal + w_global + w_context
        if abs(total - 1.0) > 0.01:
            logger.warning(f"CACS weights don't sum to 1.0 (sum={total:.3f}), normalizing...")
            self.w_base /= total
            self.w_personal /= total
            self.w_global /= total
            self.w_context /= total
            logger.info(f"Normalized weights: base={self.w_base:.3f}, personal={self.w_personal:.3f}, "
                       f"global={self.w_global:.3f}, context={self.w_context:.3f}")
    
    def calculate_score(
        self,
        doc_id: str,
        base_score: float,
        student_profile: Dict,
        conversation_history: List[Dict],
        global_scores: Dict,
        current_query: str
    ) -> Dict:
        """
        Bir doküman için CACS skorunu hesapla
        
        Args:
            doc_id: Doküman ID veya identifier
            base_score: RAG'dan gelen semantik benzerlik skoru (0-1)
            student_profile: Öğrenci profil verisi
            conversation_history: Son N etkileşim listesi
            global_scores: Global skor tablosu (doc_id -> scores dict)
            current_query: Mevcut sorgu metni
            
        Returns:
            {
                'final_score': float,           # Nihai CACS skoru
                'base_score': float,           # Semantik benzerlik
                'personal_score': float,       # Kişisel geçmiş
                'global_score': float,         # Global geri bildirim
                'context_score': float,        # Konuşma bağlamı
                'breakdown': {                 # Ağırlık detayları
                    'base_weight': float,
                    'personal_weight': float,
                    'global_weight': float,
                    'context_weight': float
                }
            }
        """
        
        # Check if CACS is enabled
        if not FeatureFlags.is_cacs_enabled():
            logger.debug(f"CACS disabled, returning base score only for doc {doc_id}")
            return {
                'final_score': base_score,
                'base_score': base_score,
                'personal_score': 0.5,
                'global_score': 0.5,
                'context_score': 0.5,
                'breakdown': {
                    'base_weight': 1.0,
                    'personal_weight': 0.0,
                    'global_weight': 0.0,
                    'context_weight': 0.0
                },
                'cacs_enabled': False
            }
        
        try:
            # 1. Base Score (semantik benzerlik - RAG'dan geliyor)
            base = float(base_score)
            
            # 2. Personal Score (öğrenci geçmişi)
            personal = self._calculate_personal_score(
                doc_id, student_profile, conversation_history
            )
            
            # 3. Global Score (tüm öğrencilerden geri bildirim)
            global_score = self._calculate_global_score(doc_id, global_scores)
            
            # 4. Context Score (konuşma bağlamı)
            context = self._calculate_context_score(
                doc_id, conversation_history, current_query
            )
            
            # Final Score Hesaplama
            final_score = (
                self.w_base * base +
                self.w_personal * personal +
                self.w_global * global_score +
                self.w_context * context
            )
            
            # Ensure score is in [0, 1] range
            final_score = max(0.0, min(1.0, final_score))
            
            result = {
                'final_score': final_score,
                'base_score': base,
                'personal_score': personal,
                'global_score': global_score,
                'context_score': context,
                'breakdown': {
                    'base_weight': self.w_base,
                    'personal_weight': self.w_personal,
                    'global_weight': self.w_global,
                    'context_weight': self.w_context
                },
                'cacs_enabled': True
            }
            
            logger.debug(f"CACS score for doc {doc_id}: {final_score:.3f} "
                        f"(base:{base:.2f}, personal:{personal:.2f}, "
                        f"global:{global_score:.2f}, context:{context:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating CACS score for doc {doc_id}: {e}")
            # Fallback to base score
            return {
                'final_score': base_score,
                'base_score': base_score,
                'personal_score': 0.5,
                'global_score': 0.5,
                'context_score': 0.5,
                'breakdown': {
                    'base_weight': 1.0,
                    'personal_weight': 0.0,
                    'global_weight': 0.0,
                    'context_weight': 0.0
                },
                'cacs_enabled': False,
                'error': str(e)
            }
    
    def _calculate_personal_score(
        self,
        doc_id: str,
        student_profile: Dict,
        conversation_history: List[Dict]
    ) -> float:
        """
        Öğrencinin bu dokümanla geçmiş etkileşimlerini değerlendir
        
        Faktörler:
        - Daha önce benzer dokümanlarla etkileşim olmuş mu?
        - O etkileşimler başarılı mıydı?
        - Öğrencinin profiliyle uyumlu mu?
        
        Returns:
            float: Personal score (0-1)
        """
        score = 0.5  # Neutral başlangıç (veri yoksa)
        
        if not conversation_history:
            return score
        
        try:
            # Geçmişte bu dokümanla etkileşim var mı?
            doc_interactions = []
            for h in conversation_history:
                # Check if doc_id is mentioned in sources
                if h.get('doc_id') == doc_id:
                    doc_interactions.append(h)
                elif doc_id in str(h.get('sources', '')):
                    doc_interactions.append(h)
            
            if doc_interactions:
                # Bu dokümanla önceki etkileşimlerin feedback ortalaması
                feedback_scores = []
                for h in doc_interactions:
                    feedback = h.get('feedback_score')
                    if feedback is not None:
                        normalized = self._normalize_feedback(feedback)
                        feedback_scores.append(normalized)
                
                if feedback_scores:
                    score = sum(feedback_scores) / len(feedback_scores)
                    logger.debug(f"Personal score from {len(feedback_scores)} past interactions: {score:.2f}")
            
            # Öğrencinin tercih ettiği zorluk seviyesiyle uyum
            if student_profile.get('preferred_difficulty_level'):
                # Bu gerçek implementasyonda dokümanın zorluk seviyesiyle
                # öğrencinin tercihi karşılaştırılır
                # Şimdilik basit bir boost uyguluyoruz
                score = min(score * 1.1, 1.0)
            
            # Success rate faktörü
            success_rate = student_profile.get('success_rate')
            if success_rate is not None and success_rate > 0.7:
                # Başarılı öğrenci - biraz daha zorlayıcı içerik tercih et
                score = min(score * 1.05, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating personal score: {e}")
            score = 0.5
        
        return score
    
    def _calculate_global_score(self, doc_id: str, global_scores: Dict) -> float:
        """
        Tüm öğrencilerden bu dokümana gelen genel geri bildirimleri değerlendir
        
        Returns:
            float: Global score (0-1)
        """
        if not global_scores or doc_id not in global_scores:
            return 0.5  # Neutral - henüz geri bildirim yok
        
        try:
            doc_global = global_scores[doc_id]
            
            total_feedback = doc_global.get('total_feedback_count', 0)
            
            if total_feedback == 0:
                return 0.5
            
            # Pozitif/negatif oran
            positive = doc_global.get('positive_feedback_count', 0)
            negative = doc_global.get('negative_feedback_count', 0)
            
            if (positive + negative) == 0:
                # Avg emoji score kullan
                avg_emoji = doc_global.get('avg_emoji_score')
                if avg_emoji is not None:
                    return float(avg_emoji)
                return 0.5
            
            # Basit pozitif oran
            global_score = positive / (positive + negative)
            
            # Güven faktörü: Az geri bildirim varsa neutral'e yaklaştır
            # 10+ feedback = tam güven
            confidence = min(total_feedback / 10.0, 1.0)
            global_score = 0.5 + (global_score - 0.5) * confidence
            
            logger.debug(f"Global score for doc {doc_id}: {global_score:.2f} "
                        f"(positive:{positive}, negative:{negative}, confidence:{confidence:.2f})")
            
            return global_score
            
        except Exception as e:
            logger.warning(f"Error calculating global score: {e}")
            return 0.5
    
    def _calculate_context_score(
        self,
        doc_id: str,
        conversation_history: List[Dict],
        current_query: str
    ) -> float:
        """
        Bu dokümanın mevcut konuşma bağlamıyla alakasını değerlendir
        
        Faktörler:
        - Son sorular benzer konularla ilgili mi?
        - Konuşma akışı bu dokümanı gerektirir mi?
        
        Returns:
            float: Context score (0-1)
        """
        score = 0.5  # Neutral başlangıç
        
        if not conversation_history or not current_query:
            return score
        
        try:
            # Son 5 etkileşime bak
            recent_history = conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history
            
            if not recent_history:
                return score
            
            # Basit keyword overlap (gerçek implementasyonda embedding similarity kullanılır)
            current_keywords = set(current_query.lower().split())
            
            if not current_keywords:
                return score
            
            overlap_scores = []
            for interaction in recent_history:
                prev_query = interaction.get('query', '')
                if not prev_query:
                    continue
                
                prev_keywords = set(prev_query.lower().split())
                
                if prev_keywords and current_keywords:
                    # Jaccard similarity
                    intersection = len(prev_keywords & current_keywords)
                    union = len(prev_keywords | current_keywords)
                    
                    if union > 0:
                        overlap = intersection / union
                        overlap_scores.append(overlap)
            
            if overlap_scores:
                # Ortalama overlap - yüksekse konuşma bağlamı güçlü
                avg_overlap = sum(overlap_scores) / len(overlap_scores)
                # Scale to [0.5, 1.0] range (0.5 = no context, 1.0 = strong context)
                score = 0.5 + avg_overlap * 0.5
                
                logger.debug(f"Context score: {score:.2f} (avg overlap: {avg_overlap:.2f} from {len(overlap_scores)} interactions)")
            
        except Exception as e:
            logger.warning(f"Error calculating context score: {e}")
            score = 0.5
        
        return score
    
    def _normalize_feedback(self, feedback_value: Any) -> float:
        """
        Farklı geri bildirim formatlarını 0-1 arasına normalize et
        
        Args:
            feedback_value: Feedback değeri (int, float, str)
            
        Returns:
            float: Normalized score (0-1)
        """
        if feedback_value is None:
            return 0.5
        
        # Numeric feedback
        if isinstance(feedback_value, (int, float)):
            # 1-5 skala ise (student_feedback tablosundan)
            if 1 <= feedback_value <= 5:
                return (feedback_value - 1) / 4.0
            # Zaten 0-1 arası
            elif 0 <= feedback_value <= 1:
                return float(feedback_value)
            # Out of range - clamp
            else:
                return max(0.0, min(1.0, float(feedback_value)))
        
        # Emoji mapping (Faz 4'te kullanılacak)
        emoji_map = {
            '😊': 0.7,  # Anladım
            '👍': 1.0,  # Mükemmel
            '😐': 0.2,  # Karışık
            '❌': 0.0,  # Anlamadım
        }
        
        if isinstance(feedback_value, str):
            if feedback_value in emoji_map:
                return emoji_map[feedback_value]
            # Try to parse as number
            try:
                num_val = float(feedback_value)
                return self._normalize_feedback(num_val)
            except:
                pass
        
        # Unknown/neutral
        return 0.5


# Singleton instance
_cacs_scorer = None


def get_cacs_scorer(
    w_base: float = 0.30,
    w_personal: float = 0.25,
    w_global: float = 0.25,
    w_context: float = 0.20
) -> Optional[CACSScorer]:
    """
    Get or create CACS scorer instance
    
    Args:
        w_base: Base score weight
        w_personal: Personal score weight
        w_global: Global score weight
        w_context: Context score weight
        
    Returns:
        CACSScorer instance or None if CACS is disabled
    """
    # Check if CACS is enabled
    if not FeatureFlags.is_cacs_enabled():
        logger.warning("CACS is not enabled (Feature flag check)")
        return None
    
    global _cacs_scorer
    if _cacs_scorer is None:
        _cacs_scorer = CACSScorer(
            w_base=w_base,
            w_personal=w_personal,
            w_global=w_global,
            w_context=w_context
        )
        logger.info("CACS Scorer initialized successfully")
    
    return _cacs_scorer


def reset_cacs_scorer():
    """Reset the singleton instance (useful for testing)"""
    global _cacs_scorer
    _cacs_scorer = None
    logger.info("CACS Scorer reset")















