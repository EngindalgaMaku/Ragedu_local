"""
Turkish language utilities for text processing
Includes stopwords, tokenization, and language-specific processing
"""
import re
from typing import List

# Turkish stopwords for better keyword search
TURKISH_STOPWORDS = {
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü',
    'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için',
    'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mi', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede',
    'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani'
}

def tokenize_turkish(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize Turkish text for BM25 search
    
    Features:
    - Lowercase conversion
    - Remove punctuation
    - Optional stopword removal
    - Keep numbers and special characters (for product codes, dates, etc.)
    
    Args:
        text: Text to tokenize
        remove_stopwords: Whether to remove Turkish stopwords
        
    Returns:
        List of tokens
    """
    # Lowercase
    text = text.lower()
    
    # Split by whitespace and basic punctuation (but keep numbers intact)
    tokens = re.findall(r'\b[\w\d]+\b', text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [t for t in tokens if t not in TURKISH_STOPWORDS and len(t) > 1]
    
    return tokens






