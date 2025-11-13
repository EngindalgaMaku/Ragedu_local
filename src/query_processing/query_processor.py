import re
import time
from typing import Dict, Any, List, Optional, Tuple
from src.utils.logger import get_logger
from src.embedding.embedding_generator import generate_embeddings

class EnhancedQueryProcessor:
    """
    Enhanced query processor with semantic analysis capabilities for Edu-ModRAG system.
    
    Provides advanced query understanding including:
    - Semantic analysis using embeddings
    - Educational context detection
    - Intent classification
    - Query expansion and refinement
    - Integration with modular RAG routing
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enhanced query processor.

        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__, config)
        self.min_query_length = self.config.get("MIN_QUERY_LENGTH", 3)
        
        # Educational context patterns
        self._init_educational_patterns()
        
        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "avg_processing_time": 0.0,
            "semantic_analysis_enabled": True
        }

    def _init_educational_patterns(self):
        """Initialize educational context patterns for Turkish and English."""
        self.educational_patterns = {
            # Academic subjects
            "subjects": {
                "matematik": ["matematik", "geometri", "cebir", "analiz", "türev", "integral"],
                "istatistik": ["istatistik", "olasılık", "dağılım", "hipotez", "test", "analiz"],
                "programlama": ["python", "java", "kod", "algoritma", "program", "yazılım"],
                "fizik": ["fizik", "kuvvet", "enerji", "hareket", "dalga", "elektrik"],
                "kimya": ["kimya", "molekül", "atom", "reaksiyon", "bileşik"],
                "biyoloji": ["biyoloji", "hücre", "dna", "protein", "organizma"]
            },
            
            # Learning objectives (Bloom's Taxonomy)
            "cognitive_levels": {
                "remember": ["nedir", "kimdir", "ne zaman", "hangi", "tanımla"],
                "understand": ["açıkla", "özetle", "yorumla", "karşılaştır", "sınıfla"],
                "apply": ["uygula", "göster", "çöz", "hesapla", "kullan"],
                "analyze": ["analiz et", "ayır", "incele", "karşılaştır", "kategorize et"],
                "evaluate": ["değerlendir", "eleştir", "savun", "karar ver"],
                "create": ["oluştur", "tasarla", "geliştir", "yarat", "kur"]
            },
            
            # Question types in educational context
            "educational_intents": {
                "definition": ["nedir", "ne demek", "tanım", "anlamı"],
                "explanation": ["nasıl", "neden", "niçin", "açıkla", "sebep"],
                "example": ["örnek", "misal", "göster", "demo"],
                "comparison": ["fark", "benzer", "karşılaştır", "arasında"],
                "procedure": ["adım", "süreç", "yöntem", "nasıl yapılır"],
                "evaluation": ["hangi", "en iyi", "değerlendir", "seç"]
            }
        }

    def preprocess(self, query: str) -> str:
        """
        Advanced preprocessing with Turkish language support.

        Args:
            query: Raw user query

        Returns:
            Preprocessed query string
        """
        # Basic cleaning
        query = query.strip()
        
        # Handle Turkish character normalization
        query = self._normalize_turkish_chars(query)
        
        # Remove excessive punctuation but keep meaningful ones
        query = re.sub(r'[!]{2,}', '!', query)
        query = re.sub(r'[?]{2,}', '?', query)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove leading/trailing punctuation except question marks
        query = re.sub(r'^[^\w\s?]+|[^\w\s?]+$', '', query)
        
        return query

    def _normalize_turkish_chars(self, text: str) -> str:
        """Normalize Turkish characters for better processing."""
        # Common Turkish character issues
        replacements = {
            'ı': 'ı', 'İ': 'İ', 'ş': 'ş', 'Ş': 'Ş',
            'ğ': 'ğ', 'Ğ': 'Ğ', 'ü': 'ü', 'Ü': 'Ü',
            'ö': 'ö', 'Ö': 'Ö', 'ç': 'ç', 'Ç': 'Ç'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def validate(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Enhanced validation with detailed feedback.

        Args:
            query: Preprocessed query

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Length check
        if len(query) < self.min_query_length:
            return False, f"Sorgunuz çok kısa. En az {self.min_query_length} karakter olmalı."
        
        # Maximum length check
        max_length = self.config.get("MAX_QUERY_LENGTH", 500)
        if len(query) > max_length:
            return False, f"Sorgunuz çok uzun. En fazla {max_length} karakter olmalı."
        
        # Check for potential spam or inappropriate content
        spam_patterns = [
            r'^(.)\1{10,}',  # Repeated characters
            r'[^\w\s]{10,}',  # Too many special characters
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, query):
                return False, "Sorgunuzda uygunsuz içerik tespit edildi."
        
        # Check if query contains any actual content
        content_words = re.findall(r'\w+', query.lower())
        if len(content_words) == 0:
            return False, "Sorgunuz anlamlı kelimeler içermelidir."
        
        return True, None

    def semantic_analysis(self, query: str) -> Dict[str, Any]:
        """
        Perform semantic analysis using embeddings and pattern matching.

        Args:
            query: Preprocessed query

        Returns:
            Dictionary containing semantic analysis results
        """
        analysis_start = time.time()
        
        try:
            # Generate query embedding for semantic similarity
            query_embedding = None
            if self.config.get("enable_semantic_analysis", True):
                query_embedding = generate_embeddings(
                    [query],
                    model=self.config.get("ollama_embedding_model"),
                    use_cache=True
                )
            
            # Detect educational subject
            detected_subject = self._detect_subject(query)
            
            # Detect cognitive level (Bloom's taxonomy)
            cognitive_level = self._detect_cognitive_level(query)
            
            # Detect educational intent
            educational_intent = self._detect_educational_intent(query)
            
            # Analyze query complexity
            complexity_score = self._analyze_complexity(query)
            
            # Detect language
            language = self._detect_language(query)
            
            # Extract key terms
            key_terms = self._extract_key_terms(query)
            
            analysis_time = time.time() - analysis_start
            
            return {
                "embedding": query_embedding[0] if query_embedding else None,
                "detected_subject": detected_subject,
                "cognitive_level": cognitive_level,
                "educational_intent": educational_intent,
                "complexity_score": complexity_score,
                "language": language,
                "key_terms": key_terms,
                "analysis_time": analysis_time,
                "semantic_features": {
                    "has_question_mark": "?" in query,
                    "word_count": len(query.split()),
                    "has_technical_terms": self._has_technical_terms(query),
                    "formality_level": self._assess_formality(query)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in semantic analysis: {e}")
            return {
                "embedding": None,
                "detected_subject": "unknown",
                "cognitive_level": "understand",
                "educational_intent": "explanation",
                "complexity_score": 0.5,
                "language": "tr",
                "key_terms": query.split(),
                "analysis_time": time.time() - analysis_start,
                "error": str(e)
            }

    def _detect_subject(self, query: str) -> str:
        """Detect the academic subject of the query."""
        query_lower = query.lower()
        subject_scores = {}
        
        for subject, keywords in self.educational_patterns["subjects"].items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                subject_scores[subject] = score
        
        if subject_scores:
            return max(subject_scores, key=subject_scores.get)
        
        return "genel"

    def _detect_cognitive_level(self, query: str) -> str:
        """Detect cognitive level based on Bloom's taxonomy."""
        query_lower = query.lower()
        
        for level, keywords in self.educational_patterns["cognitive_levels"].items():
            if any(keyword in query_lower for keyword in keywords):
                return level
        
        return "understand"  # Default level

    def _detect_educational_intent(self, query: str) -> str:
        """Detect the educational intent of the query."""
        query_lower = query.lower()
        
        for intent, keywords in self.educational_patterns["educational_intents"].items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return "explanation"  # Default intent

    def _analyze_complexity(self, query: str) -> float:
        """Analyze query complexity and return a score between 0 and 1."""
        complexity_indicators = {
            "length": len(query.split()) / 20,  # Normalize by typical query length
            "technical_terms": 0.3 if self._has_technical_terms(query) else 0,
            "multiple_concepts": len(self._extract_key_terms(query)) / 10,
            "question_depth": 0.2 if any(word in query.lower() for word in ["neden", "nasıl", "analiz"]) else 0
        }
        
        total_score = sum(complexity_indicators.values())
        return min(1.0, total_score)  # Cap at 1.0

    def _detect_language(self, query: str) -> str:
        """Simple language detection for Turkish/English."""
        turkish_indicators = ['nedir', 'nasıl', 'neden', 'hangi', 'için', 'ile', 've', 'bir', 'bu', 'şu']
        english_indicators = ['what', 'how', 'why', 'which', 'for', 'with', 'and', 'the', 'is', 'this']
        
        query_lower = query.lower()
        turkish_score = sum(1 for word in turkish_indicators if word in query_lower)
        english_score = sum(1 for word in english_indicators if word in query_lower)
        
        if turkish_score > english_score:
            return "tr"
        elif english_score > turkish_score:
            return "en"
        else:
            return "mixed"

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query."""
        # Remove common stop words
        stop_words = {
            'tr': ['bir', 'bu', 'şu', 've', 'ile', 'için', 'de', 'da', 'ki', 'mi', 'mı', 'ne', 'nə'],
            'en': ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        }
        
        # Extract words
        words = re.findall(r'\w+', query.lower())
        
        # Filter out stop words
        key_terms = []
        for word in words:
            if len(word) > 2 and not any(word in stop_list for stop_list in stop_words.values()):
                key_terms.append(word)
        
        return key_terms

    def _has_technical_terms(self, query: str) -> bool:
        """Check if query contains technical terms."""
        technical_patterns = [
            r'\b(python|java|sql|html|css|javascript)\b',
            r'\b(algoritma|veri|analiz|model|sistem)\b',
            r'\b(fonksiyon|değişken|döngü|koşul)\b',
            r'\b(database|framework|library|api)\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in technical_patterns)

    def _assess_formality(self, query: str) -> str:
        """Assess the formality level of the query."""
        formal_indicators = ['lütfen', 'rica etsem', 'mümkün mü', 'please', 'could you']
        informal_indicators = ['nasıl', 'ne', 'hey', 'what', 'how']
        
        query_lower = query.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in query_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in query_lower)
        
        if formal_count > informal_count:
            return "formal"
        elif informal_count > formal_count:
            return "informal"
        else:
            return "neutral"

    def expand_query(self, query: str, semantic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand and refine query based on semantic analysis.

        Args:
            query: Original query
            semantic_analysis: Semantic analysis results

        Returns:
            Dictionary containing expanded query information
        """
        expanded_terms = []
        
        # Add synonyms based on detected subject
        subject = semantic_analysis.get("detected_subject", "genel")
        if subject in self.educational_patterns["subjects"]:
            related_terms = self.educational_patterns["subjects"][subject]
            expanded_terms.extend(related_terms[:3])  # Add top 3 related terms
        
        # Add terms based on educational intent
        intent = semantic_analysis.get("educational_intent", "explanation")
        if intent == "definition":
            expanded_terms.extend(["tanım", "açıklama", "ne demek"])
        elif intent == "example":
            expanded_terms.extend(["örnek", "uygulama", "demo"])
        
        return {
            "original_query": query,
            "expanded_terms": expanded_terms,
            "suggested_refinements": self._suggest_refinements(query, semantic_analysis),
            "alternative_phrasings": self._generate_alternative_phrasings(query, semantic_analysis)
        }

    def _suggest_refinements(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Suggest query refinements based on analysis."""
        refinements = []
        
        complexity = analysis.get("complexity_score", 0.5)
        if complexity < 0.3:  # Simple query
            refinements.append("Daha spesifik bir soru sormayı deneyebilirsiniz")
        elif complexity > 0.8:  # Complex query
            refinements.append("Sorunuzu daha küçük parçalara bölebilirsiniz")
        
        if not analysis.get("semantic_features", {}).get("has_question_mark"):
            refinements.append("Soru işareti ekleyerek daha net bir soru haline getirebilirsiniz")
        
        return refinements

    def _generate_alternative_phrasings(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate alternative phrasings of the query."""
        alternatives = []
        
        intent = analysis.get("educational_intent", "explanation")
        
        if intent == "definition" and "nedir" not in query.lower():
            alternatives.append(f"{query.rstrip('?')} nedir?")
        
        if intent == "explanation" and "nasıl" not in query.lower():
            alternatives.append(f"{query.rstrip('?')} nasıl açıklanır?")
        
        return alternatives

    def process(self, query: str) -> Dict[str, Any]:
        """
        Complete enhanced processing pipeline.

        Args:
            query: Raw user query

        Returns:
            Comprehensive query analysis results
        """
        start_time = time.time()
        self.processing_stats["total_processed"] += 1
        
        try:
            # Step 1: Preprocessing
            preprocessed_query = self.preprocess(query)
            
            # Step 2: Validation
            is_valid, validation_error = self.validate(preprocessed_query)
            if not is_valid:
                raise ValueError(validation_error)
            
            # Step 3: Semantic Analysis
            semantic_analysis = self.semantic_analysis(preprocessed_query)
            
            # Step 4: Query Expansion
            query_expansion = self.expand_query(preprocessed_query, semantic_analysis)
            
            # Step 5: Generate processing metadata
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)
            
            result = {
                "original_query": query,
                "processed_query": preprocessed_query,
                "is_valid": is_valid,
                "semantic_analysis": semantic_analysis,
                "query_expansion": query_expansion,
                "processing_metadata": {
                    "processing_time": processing_time,
                    "processor_version": "enhanced_v1.0",
                    "features_extracted": len(semantic_analysis),
                }
            }
            
            self.logger.info(
                f"Enhanced query processing completed in {processing_time:.3f}s "
                f"for query type: {semantic_analysis.get('educational_intent', 'unknown')}"
            )
            
            return result
            
        except ValueError as ve:
            self.logger.warning(f"Query validation failed: {ve}")
            return {"error": str(ve), "is_valid": False}
        except Exception as e:
            self.logger.error(f"Error in enhanced query processing: {e}", exc_info=True)
            return {"error": "Sorgu işlenirken beklenmedik bir hata oluştu.", "is_valid": False}

    def _update_processing_stats(self, processing_time: float):
        """Update processing statistics."""
        total = self.processing_stats["total_processed"]
        current_avg = self.processing_stats["avg_processing_time"]
        
        # Update rolling average
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
        return self.processing_stats.copy()


# Backward compatibility
class QueryProcessor(EnhancedQueryProcessor):
    """
    Backward compatibility wrapper for existing code.
    """
    
    def detect_question_type(self, query: str) -> str:
        """Legacy method for backward compatibility."""
        semantic_analysis = self.semantic_analysis(query)
        return semantic_analysis.get("educational_intent", "unknown")