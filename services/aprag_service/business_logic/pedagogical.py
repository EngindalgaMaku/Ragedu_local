"""
Pedagogical Monitors for Eğitsel-KBRAG (Faz 3)

This module implements three pedagogical theories:
1. ZPD (Zone of Proximal Development) - Vygotsky
2. Bloom Taxonomy - Cognitive levels
3. Cognitive Load Theory - John Sweller

Requires APRAG and Eğitsel-KBRAG to be enabled.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import re

# Import feature flags
try:
    from config.feature_flags import FeatureFlags
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class ZPDCalculator:
    """
    Zone of Proximal Development (Yakınsal Gelişim Alanı) Calculator
    
    Based on Vygotsky's theory, determines the optimal learning zone
    between what a student can do independently and with guidance.
    
    ZPD Levels: beginner → elementary → intermediate → advanced → expert
    """
    
    ZPD_LEVELS = ['beginner', 'elementary', 'intermediate', 'advanced', 'expert']
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.ZPDCalculator')
        self.logger.info("ZPD Calculator initialized")
    
    def calculate_zpd_level(
        self,
        recent_interactions: List[Dict],
        student_profile: Dict
    ) -> Dict:
        """
        Calculate student's current ZPD level
        
        Args:
            recent_interactions: Last 20 interactions
            student_profile: Student profile data
            
        Returns:
            {
                'current_level': str,           # Current ZPD level
                'success_rate': float,          # Success rate (0-1)
                'avg_difficulty': float,        # Average difficulty (0-1)
                'recommended_level': str,       # Recommended next level
                'confidence': float,            # Confidence (0-1)
                'level_index': int              # Level as index (0-4)
            }
        """
        
        # Check if ZPD is enabled
        if not FeatureFlags.is_zpd_enabled():
            self.logger.debug("ZPD disabled, returning defaults")
            return {
                'current_level': 'intermediate',
                'success_rate': 0.5,
                'avg_difficulty': 0.5,
                'recommended_level': 'intermediate',
                'confidence': 0.0,
                'level_index': 2
            }
        
        try:
            # Get current level from profile
            current_level = student_profile.get('current_zpd_level', 'intermediate')
            
            # Ensure valid level
            if current_level not in self.ZPD_LEVELS:
                current_level = 'intermediate'
            
            if not recent_interactions:
                self.logger.debug("No interactions, using profile defaults")
                return {
                    'current_level': current_level,
                    'success_rate': 0.5,
                    'avg_difficulty': 0.5,
                    'recommended_level': current_level,
                    'confidence': 0.0,
                    'level_index': self.ZPD_LEVELS.index(current_level)
                }
            
            # Calculate success rate from recent interactions
            success_count = 0
            difficulty_scores = []
            
            for interaction in recent_interactions:
                # Check feedback for success
                feedback = interaction.get('feedback_score') or interaction.get('understanding_level')
                
                if feedback is not None:
                    # Normalize to 0-1 scale
                    if isinstance(feedback, (int, float)):
                        if 1 <= feedback <= 5:
                            # 1-5 scale: 4+ is success
                            normalized = (feedback - 1) / 4.0
                            if feedback >= 4:
                                success_count += 1
                        elif 0 <= feedback <= 1:
                            # Already 0-1 scale
                            normalized = float(feedback)
                            if normalized >= 0.75:
                                success_count += 1
                        else:
                            continue
                
                # Get difficulty level
                difficulty = interaction.get('difficulty_level', 3)
                if isinstance(difficulty, (int, float)):
                    if 1 <= difficulty <= 5:
                        difficulty_scores.append(difficulty / 5.0)
                    elif 0 <= difficulty <= 1:
                        difficulty_scores.append(float(difficulty))
            
            # Calculate metrics
            success_rate = success_count / len(recent_interactions) if recent_interactions else 0.5
            avg_difficulty = sum(difficulty_scores) / len(difficulty_scores) if difficulty_scores else 0.5
            
            # Determine recommended level
            recommended_level = self._determine_recommended_level(
                current_level, success_rate, avg_difficulty
            )
            
            # Confidence: based on amount of data
            confidence = min(len(recent_interactions) / 20.0, 1.0)
            
            result = {
                'current_level': current_level,
                'success_rate': success_rate,
                'avg_difficulty': avg_difficulty,
                'recommended_level': recommended_level,
                'confidence': confidence,
                'level_index': self.ZPD_LEVELS.index(current_level)
            }
            
            self.logger.debug(f"ZPD calculated: {current_level} → {recommended_level} "
                            f"(success: {success_rate:.2f}, confidence: {confidence:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating ZPD: {e}")
            return {
                'current_level': 'intermediate',
                'success_rate': 0.5,
                'avg_difficulty': 0.5,
                'recommended_level': 'intermediate',
                'confidence': 0.0,
                'level_index': 2
            }
    
    def _determine_recommended_level(
        self,
        current_level: str,
        success_rate: float,
        avg_difficulty: float
    ) -> str:
        """
        Determine recommended level based on success rate and difficulty
        
        Rules:
        - Success >0.80 and high difficulty: Level up
        - Success <0.40: Level down
        - Success 0.40-0.80: Stay at current level (optimal ZPD)
        """
        
        try:
            current_idx = self.ZPD_LEVELS.index(current_level)
        except ValueError:
            current_idx = 2  # Default to intermediate
        
        if success_rate > 0.80 and avg_difficulty > 0.6:
            # Very successful with hard content - level up
            new_idx = min(current_idx + 1, len(self.ZPD_LEVELS) - 1)
            return self.ZPD_LEVELS[new_idx]
        
        elif success_rate < 0.40:
            # Too difficult - level down
            new_idx = max(current_idx - 1, 0)
            return self.ZPD_LEVELS[new_idx]
        
        else:
            # In optimal ZPD - stay at current level
            return current_level
    
    def is_in_zpd(self, success_rate: float) -> bool:
        """
        Check if student is in optimal ZPD (40-80% success rate)
        """
        return 0.40 <= success_rate <= 0.80


class BloomTaxonomyDetector:
    """
    Bloom Taxonomy Level Detector
    
    Detects the cognitive level of a question based on Bloom's Taxonomy:
    1. Remember (Hatırlama) - recall facts
    2. Understand (Anlama) - explain ideas
    3. Apply (Uygulama) - use knowledge
    4. Analyze (Analiz) - examine relationships
    5. Evaluate (Değerlendirme) - justify decisions
    6. Create (Yaratma) - produce new work
    """
    
    BLOOM_LEVELS = [
        'remember',      # Level 1
        'understand',    # Level 2
        'apply',         # Level 3
        'analyze',       # Level 4
        'evaluate',      # Level 5
        'create'         # Level 6
    ]
    
    # Turkish keywords for each Bloom level
    BLOOM_KEYWORDS = {
        'remember': [
            'nedir', 'tanımla', 'listele', 'say', 'ezbere', 'hatırla',
            'kim', 'ne zaman', 'nerede', 'hangi', 'tanım', 'neydi',
            'definition', 'define', 'list', 'name', 'what is'
        ],
        'understand': [
            'açıkla', 'özetle', 'yorumla', 'anlat', 'tarif et', 'karşılaştır',
            'neden', 'nasıl', 'fark', 'benzerlik', 'örnek ver', 'anlamı',
            'explain', 'describe', 'summarize', 'compare', 'contrast', 'why', 'how'
        ],
        'apply': [
            'uygula', 'kullan', 'göster', 'çöz', 'hesapla', 'bul',
            'işlem yap', 'kanıtla', 'örnek', 'senaryo', 'pratikte',
            'apply', 'solve', 'calculate', 'demonstrate', 'use', 'show'
        ],
        'analyze': [
            'analiz et', 'ayır', 'incele', 'karşılaştır', 'kategorizle',
            'ilişkilendir', 'sebep', 'sonuç', 'neden', 'etki', 'farklılık',
            'analyze', 'examine', 'investigate', 'compare', 'categorize', 'relationship'
        ],
        'evaluate': [
            'değerlendir', 'eleştir', 'karar ver', 'yargıla', 'savun',
            'önceliklendir', 'en iyi', 'en kötü', 'hangisi daha', 'tercih',
            'evaluate', 'judge', 'critique', 'assess', 'defend', 'prioritize'
        ],
        'create': [
            'oluştur', 'tasarla', 'yarat', 'üret', 'geliştir', 'kur',
            'yeni', 'alternatif', 'model', 'plan yap', 'öner', 'inşa et',
            'create', 'design', 'develop', 'construct', 'propose', 'formulate'
        ]
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.BloomDetector')
        self.logger.info("Bloom Taxonomy Detector initialized")
    
    def detect_bloom_level(self, query: str) -> Dict:
        """
        Detect Bloom taxonomy level of a question
        
        Args:
            query: Student's question
            
        Returns:
            {
                'level': str,                   # Bloom level name
                'level_index': int,             # Level as index (1-6)
                'confidence': float,            # Confidence (0-1)
                'matched_keywords': List[str],  # Keywords that matched
                'description': str              # Level description
            }
        """
        
        # Check if Bloom detection is enabled
        if not FeatureFlags.is_bloom_enabled():
            self.logger.debug("Bloom detection disabled, returning defaults")
            return {
                'level': 'understand',
                'level_index': 2,
                'confidence': 0.0,
                'matched_keywords': [],
                'description': 'Bloom detection disabled'
            }
        
        try:
            query_lower = query.lower()
            
            # Calculate score for each level
            level_scores = {}
            matched_keywords = {}
            
            for level, keywords in self.BLOOM_KEYWORDS.items():
                score = 0
                matches = []
                
                for keyword in keywords:
                    if keyword in query_lower:
                        score += 1
                        matches.append(keyword)
                
                level_scores[level] = score
                matched_keywords[level] = matches
            
            # Determine level with highest score
            if sum(level_scores.values()) == 0:
                # No matches - default to 'understand'
                detected_level = 'understand'
                confidence = 0.0
                matches = []
            else:
                detected_level = max(level_scores, key=level_scores.get)
                total_matches = sum(level_scores.values())
                confidence = level_scores[detected_level] / total_matches if total_matches > 0 else 0.0
                matches = matched_keywords[detected_level]
            
            level_index = self.BLOOM_LEVELS.index(detected_level) + 1
            
            result = {
                'level': detected_level,
                'level_index': level_index,
                'confidence': confidence,
                'matched_keywords': matches,
                'description': self._get_level_description(detected_level)
            }
            
            self.logger.debug(f"Bloom level detected: {detected_level} (L{level_index}, "
                            f"confidence: {confidence:.2f}, keywords: {matches})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting Bloom level: {e}")
            return {
                'level': 'understand',
                'level_index': 2,
                'confidence': 0.0,
                'matched_keywords': [],
                'description': 'Error in detection'
            }
    
    def _get_level_description(self, level: str) -> str:
        """Get Turkish description of Bloom level"""
        descriptions = {
            'remember': 'Hatırlama - Bilgiyi geri çağırma',
            'understand': 'Anlama - Fikirleri açıklama',
            'apply': 'Uygulama - Bilgiyi kullanma',
            'analyze': 'Analiz - İlişkileri inceleme',
            'evaluate': 'Değerlendirme - Kararları savunma',
            'create': 'Yaratma - Yeni eser üretme'
        }
        return descriptions.get(level, 'Unknown level')
    
    def generate_bloom_instructions(
        self,
        detected_level: str,
        student_zpd_level: str
    ) -> str:
        """
        Generate LLM instructions based on Bloom level
        
        Args:
            detected_level: Detected Bloom level
            student_zpd_level: Student's ZPD level
            
        Returns:
            str: Instructions for LLM prompt
        """
        
        bloom_idx = self.BLOOM_LEVELS.index(detected_level) + 1 if detected_level in self.BLOOM_LEVELS else 2
        
        instructions = f"\n\n--- BLOOM SEVİYE TALİMATI ---\n"
        instructions += f"Bu soru Bloom Taksonomisi Seviye {bloom_idx} ({detected_level}) gerektiriyor.\n"
        instructions += f"Öğrencinin mevcut seviyesi: {student_zpd_level}\n\n"
        
        if detected_level == 'remember':
            instructions += "📝 Yanıt Stratejisi (MUTLAKA UYGULA):\n"
            instructions += "1. KISA VE NET: Yanıtı maksimum 2-3 paragraf ile sınırla. Her paragraf 3-4 cümle olsun.\n"
            instructions += "2. DOĞRUDAN TANIM: İlk cümlede doğrudan tanımı ver. Örnek: 'Profaz, hücre bölünmesinin başlangıç evresidir.'\n"
            instructions += "3. HAFIZA İPUÇLARI (ZORUNLU): Mutlaka hafızayı destekleyici ipuçları ekle:\n"
            instructions += "   - Kelime kökeni: 'Profaz = pro (ön) + faz (evre)' gibi\n"
            instructions += "   - Görsel benzetme: 'Kromozomlar iplik gibi görünür' gibi\n"
            instructions += "   - Kısa özet cümle: 'Profaz = hazırlık evresi' gibi\n"
            instructions += "4. ANAHTAR KELİMELERİ VURGULA (ZORUNLU): Önemli terimleri **kalın** yaparak vurgula:\n"
            instructions += "   - Örnek: '**Profaz**', '**kromozomlar**', '**çekirdek zarı**' gibi\n"
            instructions += "   - En az 3-5 anahtar kelimeyi mutlaka vurgula\n"
            instructions += "5. TEKNİK TERİMLERİ AZALT: Öğrenci seviyesi 'elementary' ise, teknik terimleri basit dille açıkla\n"
            instructions += "6. TEKRARLAMA: Önemli kavramları 1-2 kez tekrarla (hafıza için)\n"
        
        elif detected_level == 'understand':
            instructions += "💡 Yanıt Stratejisi:\n"
            instructions += "- Açıklayıcı ve anlaşılır dil kullan\n"
            instructions += "- Örneklerle destekle\n"
            instructions += "- Karşılaştırmalar yap\n"
        
        elif detected_level == 'apply':
            instructions += "🔧 Yanıt Stratejisi:\n"
            instructions += "- Pratik uygulama örnekleri ver\n"
            instructions += "- Adım adım çözüm göster\n"
            instructions += "- Gerçek hayat senaryoları kullan\n"
        
        elif detected_level == 'analyze':
            instructions += "🔍 Yanıt Stratejisi:\n"
            instructions += "- Detaylı analiz yap\n"
            instructions += "- İlişkileri ve sebep-sonuçları açıkla\n"
            instructions += "- Farklı perspektifleri göster\n"
        
        elif detected_level == 'evaluate':
            instructions += "⚖️ Yanıt Stratejisi:\n"
            instructions += "- Farklı bakış açılarını sun\n"
            instructions += "- Karşılaştırma ve değerlendirme yap\n"
            instructions += "- Kriterleri ve gerekçeleri açıkla\n"
        
        elif detected_level == 'create':
            instructions += "🎨 Yanıt Stratejisi:\n"
            instructions += "- Yaratıcı çözümler öner\n"
            instructions += "- Alternatif yaklaşımları tartış\n"
            instructions += "- Yeni fikirler üretmeyi teşvik et\n"
        
        instructions += "\n----------------------------\n"
        
        return instructions


class CognitiveLoadManager:
    """
    Cognitive Load Theory Manager
    
    Based on John Sweller's theory, manages and optimizes cognitive load
    by analyzing response complexity and providing simplification strategies.
    
    Load Types:
    - Intrinsic: Content complexity
    - Extraneous: Presentation complexity
    - Germane: Learning effort
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.CognitiveLoad')
        self.logger.info("Cognitive Load Manager initialized")
    
    def calculate_cognitive_load(self, response: str, query: str) -> Dict:
        """
        Calculate cognitive load of a response
        
        Factors:
        - Length (word count)
        - Complexity (sentence length)
        - Technical density (technical terms)
        - Structural complexity
        
        Returns:
            {
                'total_load': float (0-1),
                'length_load': float,
                'complexity_load': float,
                'technical_load': float,
                'needs_simplification': bool,
                'recommendations': List[str]
            }
        """
        
        # Check if cognitive load management is enabled
        if not FeatureFlags.is_cognitive_load_enabled():
            self.logger.debug("Cognitive load management disabled")
            return {
                'total_load': 0.5,
                'length_load': 0.5,
                'complexity_load': 0.5,
                'technical_load': 0.5,
                'needs_simplification': False,
                'recommendations': []
            }
        
        try:
            # 1. Length Load
            word_count = len(response.split())
            length_load = min(word_count / 500.0, 1.0)  # 500+ words = max load
            
            # 2. Complexity Load (average sentence length)
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                complexity_load = min(avg_sentence_length / 30.0, 1.0)  # 30+ words/sentence = complex
            else:
                complexity_load = 0.0
            
            # 3. Technical Term Density
            # Simple heuristic: long words (8+ chars) as proxy for technical terms
            words = response.split()
            long_words = [w for w in words if len(w) > 8]
            technical_load = min(len(long_words) / word_count, 1.0) if word_count > 0 else 0.0
            
            # Total Load (weighted average)
            total_load = (length_load * 0.4 + complexity_load * 0.3 + technical_load * 0.3)
            
            # Simplification needed?
            needs_simplification = total_load > 0.7
            
            # Generate recommendations
            recommendations = []
            if length_load > 0.7:
                recommendations.append("Yanıtı 3-5 parçaya böl (chunking)")
            if complexity_load > 0.7:
                recommendations.append("Cümleleri kısalt, basit yapılar kullan")
            if technical_load > 0.7:
                recommendations.append("Teknik terimleri açıkla veya basit alternatifleri kullan")
            
            if total_load < 0.3:
                recommendations.append("İçerik çok basit - daha detaylı bilgi eklenebilir")
            
            result = {
                'total_load': total_load,
                'length_load': length_load,
                'complexity_load': complexity_load,
                'technical_load': technical_load,
                'needs_simplification': needs_simplification,
                'recommendations': recommendations
            }
            
            self.logger.debug(f"Cognitive load: {total_load:.2f} "
                            f"(length:{length_load:.2f}, complex:{complexity_load:.2f}, "
                            f"tech:{technical_load:.2f}) - simplify: {needs_simplification}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating cognitive load: {e}")
            return {
                'total_load': 0.5,
                'length_load': 0.5,
                'complexity_load': 0.5,
                'technical_load': 0.5,
                'needs_simplification': False,
                'recommendations': []
            }
    
    def generate_simplification_instructions(self, load_info: Dict) -> str:
        """
        Generate LLM instructions for simplification
        
        Args:
            load_info: Result from calculate_cognitive_load()
            
        Returns:
            str: Instructions for LLM prompt
        """
        
        if not load_info['needs_simplification']:
            return ""
        
        instructions = "\n\n--- BİLİŞSEL YÜK OPTİMİZASYONU ---\n"
        instructions += f"Bilişsel yük yüksek ({load_info['total_load']:.2f}/1.0)\n\n"
        
        instructions += "🧠 Basitleştirme Stratejileri:\n"
        
        for rec in load_info['recommendations']:
            instructions += f"- {rec}\n"
        
        instructions += "\n💡 Uygula:\n"
        instructions += "- Bilgiyi küçük parçalara böl\n"
        instructions += "- Her paragraf tek bir konsepte odaklan\n"
        instructions += "- Görsel organizasyon kullan (başlıklar, listeler)\n"
        instructions += "- Örneklerle destekle (somutlaştır)\n"
        instructions += "- Gereksiz bilgileri çıkar\n"
        
        instructions += "\n----------------------------\n"
        
        return instructions
    
    def chunk_response(self, response: str, max_chunk_size: int = 150) -> List[str]:
        """
        Chunk a long response into smaller pieces
        
        Args:
            response: Response text
            max_chunk_size: Maximum words per chunk
            
        Returns:
            List of chunks
        """
        
        # Split by paragraphs first
        paragraphs = response.split('\n\n')
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_size + para_words > max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = para
                current_size = para_words
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_size += para_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# Singleton instances
_zpd_calculator = None
_bloom_detector = None
_cognitive_load_manager = None


def get_zpd_calculator() -> Optional[ZPDCalculator]:
    """Get or create ZPD calculator instance"""
    if not FeatureFlags.is_zpd_enabled():
        logger.warning("ZPD calculator disabled (feature flag)")
        return None
    
    global _zpd_calculator
    if _zpd_calculator is None:
        _zpd_calculator = ZPDCalculator()
        logger.info("ZPD Calculator singleton created")
    
    return _zpd_calculator


def get_bloom_detector() -> Optional[BloomTaxonomyDetector]:
    """Get or create Bloom detector instance"""
    if not FeatureFlags.is_bloom_enabled():
        logger.warning("Bloom detector disabled (feature flag)")
        return None
    
    global _bloom_detector
    if _bloom_detector is None:
        _bloom_detector = BloomTaxonomyDetector()
        logger.info("Bloom Detector singleton created")
    
    return _bloom_detector


def get_cognitive_load_manager() -> Optional[CognitiveLoadManager]:
    """Get or create Cognitive Load Manager instance"""
    if not FeatureFlags.is_cognitive_load_enabled():
        logger.warning("Cognitive load manager disabled (feature flag)")
        return None
    
    global _cognitive_load_manager
    if _cognitive_load_manager is None:
        _cognitive_load_manager = CognitiveLoadManager()
        logger.info("Cognitive Load Manager singleton created")
    
    return _cognitive_load_manager


def reset_pedagogical_modules():
    """Reset all singleton instances (useful for testing)"""
    global _zpd_calculator, _bloom_detector, _cognitive_load_manager
    _zpd_calculator = None
    _bloom_detector = None
    _cognitive_load_manager = None
    logger.info("All pedagogical modules reset")















