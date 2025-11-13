"""
Language detection utility for the RAG system.
Detects whether a query is in Turkish or English to provide appropriate responses.
"""

import re
from typing import Literal, Dict, Any

class LanguageDetector:
    """
    Simple but effective language detector for Turkish and English.
    Uses keyword patterns and character analysis.
    """
    
    # Turkish-specific keywords and patterns
    TURKISH_KEYWORDS = {
        'common_words': [
            'bir', 'bu', 'şu', 've', 'ile', 'için', 'gibi', 'kadar', 'ancak', 'ama',
            'çok', 'daha', 'en', 'her', 'hiç', 'kendi', 'o', 'biz', 'siz', 'onlar',
            'ne', 'neden', 'nasıl', 'nerede', 'kim', 'hangi', 'hangisi', 'kaç', 'kaçıncı',
            'mi', 'mı', 'mu', 'mü', 'da', 'de', 'ta', 'te',  # question/conjunction particles
            'olan', 'eden', 'iken', 'ise', 'ise', 'dir', 'dır', 'dur', 'dür',  # auxiliary verbs
            'ın', 'in', 'un', 'ün', 'nın', 'nin', 'nun', 'nün'  # possessive suffixes
        ],
        'question_words': [
            'nedir', 'nedir?', 'nasıl', 'nasıl?', 'neden', 'neden?', 'niçin', 'niçin?',
            'ne zaman', 'ne zaman?', 'nerede', 'nerede?', 'kim', 'kim?', 'hangi', 'hangi?',
            'kaç', 'kaç?', 'kaçıncı', 'kaçıncı?'
        ]
    }
    
    # English-specific keywords and patterns
    ENGLISH_KEYWORDS = {
        'common_words': [
            'the', 'and', 'or', 'but', 'with', 'for', 'like', 'as', 'than', 'very',
            'more', 'most', 'each', 'every', 'no', 'not', 'own', 'I', 'you', 'he', 'she',
            'we', 'they', 'what', 'why', 'how', 'where', 'when', 'who', 'which', 'that',
            'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'shall'
        ],
        'question_words': [
            'what', 'what?', 'how', 'how?', 'why', 'why?', 'when', 'when?',
            'where', 'where?', 'who', 'who?', 'which', 'which?', 'whose', 'whose?',
            'what is', 'how to', 'how do', 'how does', 'why is', 'why does'
        ]
    }
    
    # Turkish-specific characters
    TURKISH_CHARS = set('çğıöşüÇĞIİÖŞÜ')
    
    @classmethod
    def detect_language(cls, text: str) -> Literal['tr', 'en']:
        """
        Detect whether the input text is in Turkish or English.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Literal['tr', 'en']: 'tr' for Turkish, 'en' for English
        """
        if not text or not text.strip():
            return 'tr'  # Default to Turkish
            
        text_lower = text.lower().strip()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return 'tr'
        
        # Score calculation
        turkish_score = 0
        english_score = 0
        
        # 1. Check for Turkish-specific characters (strong indicator)
        turkish_chars_count = sum(1 for char in text if char in cls.TURKISH_CHARS)
        if turkish_chars_count > 0:
            turkish_score += turkish_chars_count * 10  # High weight for Turkish chars
        
        # 2. Check for common words
        for word in words:
            if word in cls.TURKISH_KEYWORDS['common_words']:
                turkish_score += 3
            if word in cls.ENGLISH_KEYWORDS['common_words']:
                english_score += 3
        
        # 3. Check for question words (higher weight)
        text_phrases = text_lower
        for phrase in cls.TURKISH_KEYWORDS['question_words']:
            if phrase in text_phrases:
                turkish_score += 5
        
        for phrase in cls.ENGLISH_KEYWORDS['question_words']:
            if phrase in text_phrases:
                english_score += 5
        
        # 4. Turkish suffix patterns (ending analysis)
        turkish_suffixes = [
            'lar', 'ler', 'dan', 'den', 'tan', 'ten', 'ya', 'ye', 'a', 'e',
            'da', 'de', 'ta', 'te', 'ın', 'in', 'un', 'ün', 'nın', 'nin', 'nun', 'nün',
            'dır', 'dir', 'dur', 'dür', 'tır', 'tir', 'tur', 'tür'
        ]
        
        for word in words:
            for suffix in turkish_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix):
                    turkish_score += 1
        
        # 5. English patterns
        english_patterns = [
            r'\b(is|are|was|were)\s+\w+ing\b',  # continuous tenses
            r'\b\w+ed\b',  # past tense
            r'\b\w+ing\b',  # present continuous
            r'\b(a|an)\s+\w+\b',  # articles
            r'\b(the)\s+\w+\b'  # definite article
        ]
        
        for pattern in english_patterns:
            if re.search(pattern, text_lower):
                english_score += 2
        
        # 6. Character frequency analysis (additional check)
        vowel_patterns = {
            'turkish': ['ı', 'ö', 'ü', 'ç', 'ğ', 'ş'],
            'english': ['q', 'x', 'w']  # Less common in Turkish
        }
        
        for char in vowel_patterns['turkish']:
            if char in text_lower:
                turkish_score += 1
                
        for char in vowel_patterns['english']:
            if char in text_lower:
                english_score += 1
        
        # Decision logic
        if turkish_score > english_score:
            return 'tr'
        elif english_score > turkish_score:
            return 'en'
        else:
            # Default to Turkish if scores are equal (since system is Turkish-optimized)
            return 'tr'
    
    @classmethod
    def get_language_info(cls, text: str) -> Dict[str, Any]:
        """
        Get detailed language detection information.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, Any]: Language detection details
        """
        detected_lang = cls.detect_language(text)
        
        return {
            'detected_language': detected_lang,
            'language_name': 'Turkish' if detected_lang == 'tr' else 'English',
            'confidence': 'high' if any(char in cls.TURKISH_CHARS for char in text) else 'medium',
            'text_length': len(text),
            'word_count': len(re.findall(r'\b\w+\b', text.lower()))
        }


def detect_query_language(query: str) -> Literal['tr', 'en']:
    """
    Convenience function to detect query language.
    
    Args:
        query (str): Query text
        
    Returns:
        Literal['tr', 'en']: Detected language code
    """
    return LanguageDetector.detect_language(query)


# Test cases for validation
if __name__ == "__main__":
    test_cases = [
        ("Python programlama dili nedir?", "tr"),
        ("What is Python programming language?", "en"),
        ("Makine öğrenmesi algoritmaları nasıl çalışır?", "tr"),
        ("How do machine learning algorithms work?", "en"),
        ("Bu konu hakkında daha fazla bilgi verir misin?", "tr"),
        ("Can you give me more information about this topic?", "en"),
        ("Derin öğrenme ve yapay zeka arasındaki fark nedir?", "tr"),
        ("What is the difference between deep learning and AI?", "en")
    ]
    
    print("Language Detection Test Results:")
    print("-" * 50)
    for text, expected in test_cases:
        detected = detect_query_language(text)
        status = "✓" if detected == expected else "✗"
        print(f"{status} '{text}' -> {detected} (expected: {expected})")