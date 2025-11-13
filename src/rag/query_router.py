"""
Query Analysis and Router Module for Edu-ModRAG System.

This module implements the intelligent query routing system that classifies incoming
queries and routes them to the most appropriate RAG chain based on query complexity,
intent, and context requirements.

Based on the research: "Eğitimde Kişiselleştirilmiş ve Güvenilir Bilgi Erişimi için 
Retrieval-Augmented Generation (RAG) Sistemlerinin Potansiyeli ve Optimizasyonu"
"""

import re
from typing import Dict, Any, List, Optional
from enum import Enum
import ollama
from src.utils.logger import get_logger
from src.embedding.embedding_generator import generate_embeddings

class QueryType(Enum):
    """
    Query classification types based on educational context and complexity.
    """
    SIMPLE_FACTUAL = "simple_factual"        # "t-testi nedir?"
    COMPARATIVE = "comparative"               # "X ve Y arasındaki farklar nelerdir?"
    APPLIED_PROCEDURAL = "applied_procedural" # "Python'da nasıl yaparım?"
    MULTI_DOCUMENT = "multi_document"         # "Birden çok kaynağı özetle"
    CONCEPTUAL = "conceptual"                 # "Neden böyle olur?"
    UNKNOWN = "unknown"                       # Sınıflandırılamayan sorgular

class QueryComplexity(Enum):
    """
    Query complexity levels for pipeline selection.
    """
    LOW = "low"           # Stuff chain - fast and efficient
    MEDIUM = "medium"     # Refine chain - balanced approach
    HIGH = "high"         # Map-Reduce chain - comprehensive analysis

class RAGChainType(Enum):
    """
    Available RAG chain types for different scenarios.
    """
    STUFF = "stuff"           # Fast retrieval + single generation
    REFINE = "refine"         # Iterative refinement
    MAP_REDUCE = "map_reduce" # Multi-document summarization

class QueryRouter:
    """
    Intelligent query router that analyzes queries and selects optimal RAG chains.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Query Router.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__, config)
        
        # Initialize patterns for Turkish language educational queries
        self._init_patterns()
        
        # Performance tracking
        self.routing_stats = {
            "total_queries": 0,
            "chain_usage": {chain.value: 0 for chain in RAGChainType},
            "query_type_distribution": {qtype.value: 0 for qtype in QueryType}
        }
    
    def _init_patterns(self):
        """Initialize regex patterns for query classification."""
        self.patterns = {
            QueryType.SIMPLE_FACTUAL: [
                r'\b(nedir|kimdir|ne demek|tanım|tanımla)\b',
                r'\b(what is|define|definition)\b',
                r'^.{1,50}\s+(nedir|ne)\??\s*$',  # Short "X nedir?" patterns
            ],
            QueryType.COMPARATIVE: [
                r'\b(fark|farklılık|karşılaştır|arasındaki|versus|vs)\b',
                r'\b(difference|compare|comparison|between)\b',
                r'\b(ile|arasında).+(fark|farklı|benzer)\b',
                r'\b(hangisi|hangi).+(daha|en)\b',
            ],
            QueryType.APPLIED_PROCEDURAL: [
                r'\b(nasıl|how to|adım|steps|uygula|implement)\b',
                r'\b(kod|code|python|r|matlab|sql)\b',
                r'\b(örnek|example|demo|göster|show)\b',
                r'\b(yap|yarat|create|build)\b',
            ],
            QueryType.MULTI_DOCUMENT: [
                r'\b(özetle|summarize|summary|genel|overall)\b',
                r'\b(birleştir|combine|merge|hepsi|all)\b',
                r'\b(ders notları|laboratory|kılavuz).+(özetle|kullan)\b',
                r'\b(multiple|several|various).+(source|document)\b',
            ],
            QueryType.CONCEPTUAL: [
                r'\b(neden|niçin|why|açıkla|explain|sebep|reason)\b',
                r'\b(ne için|what for|purpose|amaç)\b',
                r'\b(anlat|describe|detaylı|detailed)\b',
                r'\b(mantık|logic|principle|ilke)\b',
            ]
        }
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify the query type and determine complexity.
        
        Args:
            query: User's query string
            
        Returns:
            Dict containing classification results
        """
        query_clean = query.lower().strip()
        
        # Detect query type using patterns
        query_type = self._detect_query_type(query_clean)
        
        # Determine complexity based on query characteristics
        complexity = self._assess_complexity(query_clean, query_type)
        
        # Select appropriate RAG chain
        recommended_chain = self._select_rag_chain(query_type, complexity)
        
        # Extract additional features
        features = self._extract_features(query_clean)
        
        classification = {
            "query_type": query_type,
            "complexity": complexity,
            "recommended_chain": recommended_chain,
            "features": features,
            "confidence": features.get("confidence", 0.7)
        }
        
        # Update statistics
        self.routing_stats["total_queries"] += 1
        self.routing_stats["query_type_distribution"][query_type.value] += 1
        self.routing_stats["chain_usage"][recommended_chain.value] += 1
        
        self.logger.info(
            f"Query classified: type={query_type.value}, "
            f"complexity={complexity.value}, chain={recommended_chain.value}"
        )
        
        return classification
    
    def _detect_query_type(self, query: str) -> QueryType:
        """
        Detect query type using pattern matching.
        
        Args:
            query: Preprocessed query string
            
        Returns:
            Detected QueryType
        """
        type_scores = {qtype: 0 for qtype in QueryType}
        
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                type_scores[query_type] += len(matches)
        
        # Find the type with highest score
        max_score = max(type_scores.values())
        if max_score > 0:
            for qtype, score in type_scores.items():
                if score == max_score:
                    return qtype
        
        return QueryType.UNKNOWN
    
    def _assess_complexity(self, query: str, query_type: QueryType) -> QueryComplexity:
        """
        Assess query complexity based on multiple factors.
        
        Args:
            query: Preprocessed query string
            query_type: Detected query type
            
        Returns:
            QueryComplexity level
        """
        complexity_score = 0
        
        # Length-based scoring
        word_count = len(query.split())
        if word_count > 15:
            complexity_score += 2
        elif word_count > 8:
            complexity_score += 1
        
        # Type-based scoring
        type_complexity = {
            QueryType.SIMPLE_FACTUAL: 0,
            QueryType.COMPARATIVE: 1,
            QueryType.CONCEPTUAL: 1,
            QueryType.APPLIED_PROCEDURAL: 2,
            QueryType.MULTI_DOCUMENT: 3,
            QueryType.UNKNOWN: 1
        }
        complexity_score += type_complexity.get(query_type, 1)
        
        # Specific complexity indicators
        high_complexity_indicators = [
            r'\b(detaylı|comprehensive|extensive|thorough)\b',
            r'\b(birden.+fazla|multiple|several|various)\b',
            r'\b(analiz|analysis|evaluate|değerlendir)\b',
            r'\b(karşılaştır.+özetle|compare.+summarize)\b'
        ]
        
        for indicator in high_complexity_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 4:
            return QueryComplexity.HIGH
        elif complexity_score >= 2:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.LOW
    
    def _select_rag_chain(self, query_type: QueryType, complexity: QueryComplexity) -> RAGChainType:
        """
        Select the most appropriate RAG chain based on query characteristics.
        
        Args:
            query_type: Detected query type
            complexity: Assessed complexity level
            
        Returns:
            Recommended RAGChainType
        """
        # Chain selection logic based on research findings
        
        # Multi-document queries always use Map-Reduce
        if query_type == QueryType.MULTI_DOCUMENT:
            return RAGChainType.MAP_REDUCE
        
        # High complexity queries use iterative refinement
        if complexity == QueryComplexity.HIGH:
            return RAGChainType.REFINE
        
        # Applied/procedural queries benefit from refinement
        if query_type == QueryType.APPLIED_PROCEDURAL and complexity >= QueryComplexity.MEDIUM:
            return RAGChainType.REFINE
        
        # Comparative queries with medium+ complexity use refinement
        if query_type == QueryType.COMPARATIVE and complexity >= QueryComplexity.MEDIUM:
            return RAGChainType.REFINE
        
        # Simple factual queries use fast Stuff chain
        if query_type == QueryType.SIMPLE_FACTUAL:
            return RAGChainType.STUFF
        
        # Default based on complexity
        if complexity == QueryComplexity.LOW:
            return RAGChainType.STUFF
        elif complexity == QueryComplexity.MEDIUM:
            return RAGChainType.REFINE
        else:
            return RAGChainType.MAP_REDUCE
    
    def _extract_features(self, query: str) -> Dict[str, Any]:
        """
        Extract additional features from the query.
        
        Args:
            query: Preprocessed query string
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            "word_count": len(query.split()),
            "char_count": len(query),
            "has_question_mark": "?" in query,
            "has_numbers": bool(re.search(r'\d', query)),
            "has_code_keywords": bool(re.search(r'\b(python|r|sql|kod|code|function|def)\b', query, re.IGNORECASE)),
            "language": self._detect_language(query),
            "confidence": 0.8  # Default confidence, could be improved with ML models
        }
        
        return features
    
    def _detect_language(self, query: str) -> str:
        """
        Simple language detection.
        
        Args:
            query: Query string
            
        Returns:
            Detected language code
        """
        # Simple Turkish language detection
        turkish_indicators = ['nedir', 'nasıl', 'neden', 'hangi', 'için', 'ile', 've', 'bir', 'bu']
        english_indicators = ['what', 'how', 'why', 'which', 'for', 'with', 'and', 'the', 'is']
        
        turkish_score = sum(1 for word in turkish_indicators if word in query.lower())
        english_score = sum(1 for word in english_indicators if word in query.lower())
        
        if turkish_score > english_score:
            return "tr"
        elif english_score > turkish_score:
            return "en"
        else:
            return "unknown"
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get routing statistics and performance metrics.
        
        Returns:
            Dictionary containing routing statistics
        """
        return {
            "total_queries": self.routing_stats["total_queries"],
            "chain_usage_distribution": self.routing_stats["chain_usage"].copy(),
            "query_type_distribution": self.routing_stats["query_type_distribution"].copy(),
            "efficiency_metrics": self._calculate_efficiency_metrics()
        }
    
    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """
        Calculate efficiency metrics for the routing system.
        
        Returns:
            Dictionary of efficiency metrics
        """
        total = self.routing_stats["total_queries"]
        if total == 0:
            return {"stuff_chain_ratio": 0.0, "efficiency_score": 0.0}
        
        # Stuff chain is most efficient - higher ratio indicates better efficiency
        stuff_usage = self.routing_stats["chain_usage"][RAGChainType.STUFF.value]
        stuff_ratio = stuff_usage / total
        
        # Simple efficiency score based on optimal chain usage
        efficiency_score = (
            stuff_usage * 1.0 +  # Stuff chain: most efficient
            self.routing_stats["chain_usage"][RAGChainType.REFINE.value] * 0.7 +  # Medium efficiency
            self.routing_stats["chain_usage"][RAGChainType.MAP_REDUCE.value] * 0.4   # Least efficient but necessary
        ) / total if total > 0 else 0.0
        
        return {
            "stuff_chain_ratio": stuff_ratio,
            "efficiency_score": efficiency_score
        }
    
    def explain_routing_decision(self, query: str, classification: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of routing decision.
        
        Args:
            query: Original query
            classification: Classification result
            
        Returns:
            Explanation string
        """
        query_type = classification["query_type"]
        complexity = classification["complexity"]
        chain = classification["recommended_chain"]
        
        explanations = {
            RAGChainType.STUFF: "Hızlı ve doğrudan cevap için Stuff chain kullanılıyor.",
            RAGChainType.REFINE: "Derinlemesine analiz için Refine chain kullanılıyor.",
            RAGChainType.MAP_REDUCE: "Çoklu belge analizi için Map-Reduce chain kullanılıyor."
        }
        
        base_explanation = explanations.get(chain, "Varsayılan chain kullanılıyor.")
        
        detail = f"Sorgu tipi: {query_type.value}, Karmaşıklık: {complexity.value}"
        
        return f"{base_explanation} ({detail})"