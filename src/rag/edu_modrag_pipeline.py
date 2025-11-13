"""
Edu-ModRAG (Educational Modular RAG) Pipeline Implementation.

This module implements the complete Edu-ModRAG system that dynamically selects
the most appropriate RAG strategy based on query analysis. It integrates:
- Query Router for intelligent query classification
- Multiple RAG chains (Stuff, Refine, Map-Reduce)
- Performance tracking and optimization
- Educational context-aware features
- Source attribution and transparency

Based on: "Eğitimde Kişiselleştirilmiş ve Güvenilir Bilgi Erişimi için
Retrieval-Augmented Generation (RAG) Sistemlerinin Potansiyeli ve Optimizasyonu"
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from src.rag.query_router import QueryRouter, QueryType, QueryComplexity, RAGChainType
from src.rag.rag_chains import RAGChainFactory, BaseRAGChain
from src.vector_store.chroma_store import ChromaVectorStore
from src.analytics.performance_tracker import PerformanceTracker, PerformanceMetric
from src.utils.logger import get_logger
from src.utils.cache import get_cache

@dataclass
class SourceAttribution:
    """
    Detailed source attribution for transparency.
    """
    source_id: str
    document_name: str
    page_number: Optional[int]
    section: Optional[str]
    relevance_score: float
    content_preview: str
    usage_in_answer: str
    confidence_level: float

@dataclass
class EduModRAGResponse:
    """
    Enhanced structured response from Edu-ModRAG system with full transparency.
    """
    answer: str
    sources: List[Dict[str, Any]]
    source_attributions: List[SourceAttribution]
    query_analysis: Dict[str, Any]
    chain_used: str
    execution_time: float
    performance_metrics: Dict[str, Any]
    explanation: str
    success: bool
    query_id: str
    timestamp: str
    transparency_info: Dict[str, Any]
    error_message: Optional[str] = None

class EduModRAGPipeline:
    """
    Main Edu-ModRAG pipeline that orchestrates intelligent query routing
    and execution with appropriate RAG chains.
    """
    
    def __init__(self, config: Dict[str, Any], chroma_store: ChromaVectorStore):
        """
        Initialize the Edu-ModRAG pipeline.
        
        Args:
            config: System configuration dictionary
            chroma_store: ChromaDB vector store instance
        """
        self.config = config
        self.chroma_store = chroma_store
        self.logger = get_logger(__name__, config)
        
        # Initialize core components
        self.query_router = QueryRouter(config)
        self.chain_instances = {}  # Cache for chain instances
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(config)
        
        # Initialize cache if enabled
        self.cache = get_cache(ttl=config.get("cache_ttl", 3600)) if config.get("enable_cache", True) else None
        
        # Performance and analytics tracking (legacy support)
        self.performance_analytics = {
            "total_queries": 0,
            "chain_performance": {
                "stuff": {"count": 0, "total_time": 0.0, "avg_time": 0.0, "success_rate": 1.0},
                "refine": {"count": 0, "total_time": 0.0, "avg_time": 0.0, "success_rate": 1.0},
                "map_reduce": {"count": 0, "total_time": 0.0, "avg_time": 0.0, "success_rate": 1.0}
            },
            "query_type_performance": {qtype.value: {"count": 0, "avg_time": 0.0} for qtype in QueryType},
            "optimization_opportunities": []
        }
        
        self.logger.info("Edu-ModRAG pipeline initialized successfully")
    
    def execute(self, query: str, top_k: int = None, force_chain: str = None) -> EduModRAGResponse:
        """
        Execute the complete Edu-ModRAG pipeline for a given query.
        
        Args:
            query: User's question/query
            top_k: Number of documents to retrieve (auto-selected if None)
            force_chain: Force specific chain for testing (overrides routing)
            
        Returns:
            EduModRAGResponse containing comprehensive results
        """
        pipeline_start_time = time.time()
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        self.performance_analytics["total_queries"] += 1
        
        self.logger.info(f"Executing Edu-ModRAG pipeline for query_id: {query_id}")
        
        try:
            # Step 1: Query Analysis and Routing
            routing_start = time.time()
            query_analysis = self.query_router.classify_query(query)
            routing_time = time.time() - routing_start
            
            self.logger.info(
                f"Query classified as {query_analysis['query_type'].value} "
                f"with {query_analysis['complexity'].value} complexity "
                f"in {routing_time:.3f}s"
            )
            
            # Step 2: Chain Selection (allow override for testing)
            selected_chain_type = force_chain or query_analysis['recommended_chain'].value
            
            # Step 3: Auto-select top_k based on chain type
            if top_k is None:
                top_k = self._get_optimal_top_k(selected_chain_type, query_analysis)
            
            # Step 4: Execute Selected Chain
            chain_response = self._execute_chain(
                selected_chain_type,
                query,
                top_k,
                query_analysis
            )
            
            # Step 5: Generate Source Attributions
            source_attributions = self._generate_source_attributions(
                chain_response.get("sources", []),
                chain_response.get("answer", "")
            )
            
            # Step 6: Generate Explanation and Transparency Info
            explanation = self._generate_explanation(query, query_analysis, chain_response)
            transparency_info = self._generate_transparency_info(
                query_analysis, chain_response, selected_chain_type
            )
            
            # Step 7: Calculate Performance Metrics
            total_execution_time = time.time() - pipeline_start_time
            performance_metrics = self._calculate_performance_metrics(
                selected_chain_type,
                total_execution_time,
                routing_time,
                chain_response.get("execution_time", 0.0)
            )
            
            # Step 8: Record Performance Data
            perf_metric = PerformanceMetric(
                timestamp=timestamp,
                query_id=query_id,
                query_text=query[:200],  # Truncate for privacy
                chain_type=selected_chain_type,
                query_type=query_analysis['query_type'].value,
                complexity_level=query_analysis['complexity'].value,
                execution_time=total_execution_time,
                tokens_used=self._estimate_tokens_used(query, chain_response),
                success=True,
                retrieval_time=performance_metrics.get("routing_time", 0.0),
                generation_time=chain_response.get("execution_time", 0.0),
                documents_retrieved=len(chain_response.get("sources", [])),
                context_length=chain_response.get("context_length", 0)
            )
            self.performance_tracker.record_performance(perf_metric)
            
            # Step 9: Update Legacy Analytics
            self._update_analytics(selected_chain_type, query_analysis, performance_metrics, True)
            
            # Step 10: Create Enhanced Structured Response
            response = EduModRAGResponse(
                answer=chain_response.get("answer", "Cevap oluşturulamadı."),
                sources=chain_response.get("sources", []),
                source_attributions=source_attributions,
                query_analysis=query_analysis,
                chain_used=selected_chain_type,
                execution_time=total_execution_time,
                performance_metrics=performance_metrics,
                explanation=explanation,
                success=True,
                query_id=query_id,
                timestamp=timestamp,
                transparency_info=transparency_info
            )
            
            self.logger.info(
                f"Edu-ModRAG pipeline completed successfully in {total_execution_time:.2f}s "
                f"using {selected_chain_type} chain (query_id: {query_id})"
            )
            
            return response
            
        except Exception as e:
            total_execution_time = time.time() - pipeline_start_time
            self.logger.error(f"Error in Edu-ModRAG pipeline: {e}", exc_info=True)
            
            # Record failure in performance tracker
            perf_metric = PerformanceMetric(
                timestamp=timestamp,
                query_id=query_id,
                query_text=query[:200],
                chain_type=force_chain or "unknown",
                query_type=query_analysis.get('query_type', QueryType.UNKNOWN).value if 'query_analysis' in locals() else "unknown",
                complexity_level=query_analysis.get('complexity', QueryComplexity.MEDIUM).value if 'query_analysis' in locals() else "medium",
                execution_time=total_execution_time,
                tokens_used=0,
                success=False,
                error_message=str(e)
            )
            self.performance_tracker.record_performance(perf_metric)
            
            # Update analytics for failure
            chain_type = force_chain or query_analysis.get('recommended_chain', RAGChainType.STUFF).value if 'query_analysis' in locals() else "stuff"
            self._update_analytics(chain_type, query_analysis if 'query_analysis' in locals() else {}, {}, False)
            
            return EduModRAGResponse(
                answer="Üzgünüm, sorunuzu işlerken beklenmedik bir hata oluştu. Lütfen daha basit bir şekilde tekrar deneyin.",
                sources=[],
                source_attributions=[],
                query_analysis=query_analysis if 'query_analysis' in locals() else {},
                chain_used=chain_type,
                execution_time=total_execution_time,
                performance_metrics={},
                explanation=f"Sistem hatası: {str(e)}",
                success=False,
                query_id=query_id,
                timestamp=timestamp,
                transparency_info={"error": True, "error_message": str(e)},
                error_message=str(e)
            )
    
    def _execute_chain(self, chain_type: str, query: str, top_k: int, 
                      query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the specified RAG chain.
        
        Args:
            chain_type: Type of RAG chain to execute
            query: User query
            top_k: Number of documents to retrieve
            query_analysis: Query analysis results
            
        Returns:
            Chain execution results
        """
        # Get or create chain instance (caching for performance)
        if chain_type not in self.chain_instances:
            self.chain_instances[chain_type] = RAGChainFactory.create_chain(
                chain_type, self.config, self.chroma_store
            )
        
        chain = self.chain_instances[chain_type]
        
        # Execute the chain
        self.logger.info(f"Executing {chain_type} chain with top_k={top_k}")
        
        result = chain.execute(query, top_k)
        
        # Add query analysis context to the result
        result["query_context"] = query_analysis
        
        return result
    
    def _get_optimal_top_k(self, chain_type: str, query_analysis: Dict[str, Any]) -> int:
        """
        Determine optimal top_k based on chain type and query characteristics.
        
        Args:
            chain_type: Selected RAG chain type
            query_analysis: Query analysis results
            
        Returns:
            Optimal top_k value
        """
        base_top_k = {
            "stuff": 5,       # Fast, focused retrieval
            "refine": 6,      # Slightly more for iterative refinement
            "map_reduce": 8   # More documents for comprehensive analysis
        }
        
        top_k = base_top_k.get(chain_type, 5)
        
        # Adjust based on query complexity
        complexity = query_analysis.get("complexity")
        if complexity == QueryComplexity.HIGH:
            top_k += 2
        elif complexity == QueryComplexity.LOW:
            top_k = max(3, top_k - 1)
        
        # Adjust based on query type
        query_type = query_analysis.get("query_type")
        if query_type == QueryType.MULTI_DOCUMENT:
            top_k += 3
        elif query_type == QueryType.SIMPLE_FACTUAL:
            top_k = max(3, top_k - 2)
        
        # Ensure reasonable bounds
        top_k = max(3, min(top_k, 12))
        
        return top_k
    
    def _generate_source_attributions(self, sources: List[Dict[str, Any]],
                                     answer: str) -> List[SourceAttribution]:
        """
        Generate detailed source attributions for transparency.
        
        Args:
            sources: Retrieved source documents
            answer: Generated answer
            
        Returns:
            List of detailed source attributions
        """
        attributions = []
        
        try:
            for i, source in enumerate(sources):
                # Extract metadata
                metadata = source.get("metadata", {})
                text = source.get("text", "")
                score = source.get("score", 0.0)
                
                # Generate source attribution
                attribution = SourceAttribution(
                    source_id=f"src_{i+1}",
                    document_name=metadata.get("filename", f"Document {i+1}"),
                    page_number=metadata.get("page", None),
                    section=metadata.get("section", None),
                    relevance_score=float(score),
                    content_preview=text[:200] + "..." if len(text) > 200 else text,
                    usage_in_answer=self._detect_usage_in_answer(text, answer),
                    confidence_level=min(1.0, score * 2)  # Convert similarity to confidence
                )
                
                attributions.append(attribution)
                
        except Exception as e:
            self.logger.warning(f"Failed to generate source attributions: {e}")
            
        return attributions
    
    def _detect_usage_in_answer(self, source_text: str, answer: str) -> str:
        """
        Detect how source content was used in the answer.
        
        Args:
            source_text: Original source text
            answer: Generated answer
            
        Returns:
            Description of usage
        """
        # Simple keyword overlap detection
        source_words = set(source_text.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = source_words.intersection(answer_words)
        overlap_ratio = len(overlap) / len(source_words) if source_words else 0
        
        if overlap_ratio > 0.3:
            return "Doğrudan alıntı ve parafraz"
        elif overlap_ratio > 0.1:
            return "Konsept ve terminoloji"
        else:
            return "Genel bağlam"
    
    def _generate_transparency_info(self, query_analysis: Dict[str, Any],
                                  chain_response: Dict[str, Any],
                                  chain_type: str) -> Dict[str, Any]:
        """
        Generate comprehensive transparency information.
        
        Args:
            query_analysis: Query analysis results
            chain_response: Chain execution results
            chain_type: Selected chain type
            
        Returns:
            Transparency information dictionary
        """
        return {
            "reasoning_process": {
                "query_classification": {
                    "type": query_analysis.get("query_type", QueryType.UNKNOWN).value,
                    "complexity": query_analysis.get("complexity", QueryComplexity.MEDIUM).value,
                    "confidence": query_analysis.get("confidence", 0.7)
                },
                "chain_selection": {
                    "selected_chain": chain_type,
                    "reason": self.query_router.explain_routing_decision("", query_analysis),
                    "alternative_chains": self._get_alternative_chains(query_analysis)
                },
                "retrieval_process": {
                    "documents_found": len(chain_response.get("sources", [])),
                    "avg_relevance_score": self._calculate_avg_relevance_score(chain_response.get("sources", [])),
                    "retrieval_strategy": chain_type
                }
            },
            "confidence_indicators": {
                "overall_confidence": self._calculate_overall_confidence(chain_response),
                "source_quality": self._assess_source_quality(chain_response.get("sources", [])),
                "answer_completeness": self._assess_answer_completeness(chain_response.get("answer", ""))
            },
            "limitations": {
                "knowledge_cutoff": "Model bilgileri belirli bir tarihle sınırlıdır",
                "context_limitations": "Sadece sağlanan belgeler kullanılmıştır",
                "potential_biases": "Kaynak belgelerdeki olası önyargılar yanıta yansıyabilir"
            },
            "verification_suggestions": [
                "Kaynak belgeleri doğrudan kontrol edin",
                "Güncel kaynaklarla çapraz kontrol yapın",
                "Farklı bakış açıları için ek kaynaklar araştırın"
            ]
        }
    
    def _get_alternative_chains(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Get alternative chain suggestions."""
        current_chain = query_analysis.get("recommended_chain", RAGChainType.STUFF).value
        all_chains = ["stuff", "refine", "map_reduce"]
        return [chain for chain in all_chains if chain != current_chain]
    
    def _calculate_avg_relevance_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate average relevance score of sources."""
        if not sources:
            return 0.0
        
        scores = [source.get("score", 0.0) for source in sources]
        return sum(scores) / len(scores)
    
    def _calculate_overall_confidence(self, chain_response: Dict[str, Any]) -> float:
        """Calculate overall confidence in the response."""
        # Simple heuristic based on multiple factors
        factors = {
            "source_count": min(1.0, len(chain_response.get("sources", [])) / 5),
            "avg_score": self._calculate_avg_relevance_score(chain_response.get("sources", [])),
            "answer_length": min(1.0, len(chain_response.get("answer", "")) / 500)
        }
        
        return sum(factors.values()) / len(factors)
    
    def _assess_source_quality(self, sources: List[Dict[str, Any]]) -> str:
        """Assess overall quality of retrieved sources."""
        if not sources:
            return "Kaynak bulunamadı"
        
        avg_score = self._calculate_avg_relevance_score(sources)
        
        if avg_score > 0.8:
            return "Yüksek kalite"
        elif avg_score > 0.6:
            return "Orta kalite"
        else:
            return "Düşük kalite"
    
    def _assess_answer_completeness(self, answer: str) -> str:
        """Assess completeness of the generated answer."""
        word_count = len(answer.split())
        
        if word_count < 10:
            return "Çok kısa"
        elif word_count < 50:
            return "Kısa"
        elif word_count < 200:
            return "Orta"
        else:
            return "Kapsamlı"
    
    def _estimate_tokens_used(self, query: str, chain_response: Dict[str, Any]) -> int:
        """Estimate total tokens used in the operation."""
        # Simple estimation based on text length
        query_tokens = len(query.split()) * 1.3  # Rough token multiplier
        
        sources_text = " ".join([
            source.get("text", "") for source in chain_response.get("sources", [])
        ])
        context_tokens = len(sources_text.split()) * 1.3
        
        answer_tokens = len(chain_response.get("answer", "").split()) * 1.3
        
        return int(query_tokens + context_tokens + answer_tokens)

    def _generate_explanation(self, query: str, query_analysis: Dict[str, Any],
                            chain_response: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of the processing decision.
        
        Args:
            query: Original query
            query_analysis: Query analysis results
            chain_response: Chain execution results
            
        Returns:
            Explanation string
        """
        try:
            chain_type = chain_response.get("chain_type", "unknown")
            execution_time = chain_response.get("execution_time", 0.0)
            sources_count = len(chain_response.get("sources", []))
            
            explanations = {
                "stuff": f"Hızlı yanıt için Stuff stratejisi kullanıldı ({execution_time:.1f}s, {sources_count} kaynak)",
                "refine": f"Derinlemesine analiz için Refine stratejisi kullanıldı ({execution_time:.1f}s, {sources_count} kaynak)",
                "map_reduce": f"Kapsamlı analiz için Map-Reduce stratejisi kullanıldı ({execution_time:.1f}s, {sources_count} kaynak)"
            }
            
            base_explanation = explanations.get(chain_type, f"{chain_type} stratejisi kullanıldı")
            
            # Add query analysis context
            query_type = query_analysis.get("query_type", QueryType.UNKNOWN).value
            complexity = query_analysis.get("complexity", QueryComplexity.MEDIUM).value
            
            context = f"Sorgu tipi: {query_type}, Karmaşıklık: {complexity}"
            
            return f"{base_explanation}. {context}."
            
        except Exception as e:
            self.logger.warning(f"Failed to generate explanation: {e}")
            return "İşlem tamamlandı."
    
    def _calculate_performance_metrics(self, chain_type: str, total_time: float,
                                     routing_time: float, chain_time: float) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            chain_type: Used RAG chain type
            total_time: Total pipeline execution time
            routing_time: Time spent on routing
            chain_time: Time spent on chain execution
            
        Returns:
            Performance metrics dictionary
        """
        overhead_time = total_time - chain_time - routing_time
        
        metrics = {
            "total_execution_time": total_time,
            "routing_time": routing_time,
            "chain_execution_time": chain_time,
            "overhead_time": max(0, overhead_time),
            "routing_overhead_percentage": (routing_time / total_time) * 100 if total_time > 0 else 0,
            "chain_efficiency": (chain_time / total_time) * 100 if total_time > 0 else 0,
            "queries_per_second": 1 / total_time if total_time > 0 else 0,
            "selected_chain": chain_type
        }
        
        return metrics
    
    def _update_analytics(self, chain_type: str, query_analysis: Dict[str, Any],
                         performance_metrics: Dict[str, Any], success: bool):
        """
        Update system analytics and performance tracking.
        
        Args:
            chain_type: Used RAG chain type
            query_analysis: Query analysis results
            performance_metrics: Performance metrics
            success: Whether execution was successful
        """
        try:
            # Update chain performance
            if chain_type in self.performance_analytics["chain_performance"]:
                chain_stats = self.performance_analytics["chain_performance"][chain_type]
                chain_stats["count"] += 1
                
                if success and performance_metrics:
                    chain_time = performance_metrics.get("chain_execution_time", 0.0)
                    chain_stats["total_time"] += chain_time
                    chain_stats["avg_time"] = chain_stats["total_time"] / chain_stats["count"]
                    
                    # Update success rate
                    current_successes = chain_stats["count"] * chain_stats["success_rate"]
                    if success:
                        current_successes += 1
                    chain_stats["success_rate"] = current_successes / chain_stats["count"]
            
            # Update query type performance
            query_type = query_analysis.get("query_type")
            if query_type and query_type.value in self.performance_analytics["query_type_performance"]:
                type_stats = self.performance_analytics["query_type_performance"][query_type.value]
                type_stats["count"] += 1
                
                if success and performance_metrics:
                    total_time = performance_metrics.get("total_execution_time", 0.0)
                    if type_stats["avg_time"] == 0:
                        type_stats["avg_time"] = total_time
                    else:
                        type_stats["avg_time"] = (
                            (type_stats["avg_time"] * (type_stats["count"] - 1) + total_time) 
                            / type_stats["count"]
                        )
            
            # Identify optimization opportunities
            self._identify_optimization_opportunities(performance_metrics)
            
        except Exception as e:
            self.logger.warning(f"Failed to update analytics: {e}")
    
    def _identify_optimization_opportunities(self, performance_metrics: Dict[str, Any]):
        """
        Identify potential optimization opportunities based on performance data.
        
        Args:
            performance_metrics: Current execution performance metrics
        """
        opportunities = []
        
        if not performance_metrics:
            return
        
        # High routing overhead
        routing_overhead = performance_metrics.get("routing_overhead_percentage", 0)
        if routing_overhead > 15:  # More than 15% overhead
            opportunities.append({
                "type": "routing_optimization",
                "description": f"Routing overhead is {routing_overhead:.1f}%, consider caching routing decisions",
                "priority": "medium"
            })
        
        # Slow total execution
        total_time = performance_metrics.get("total_execution_time", 0)
        if total_time > 10:  # Slower than 10 seconds
            opportunities.append({
                "type": "performance_optimization",
                "description": f"Query took {total_time:.1f}s, consider optimizing retrieval or generation",
                "priority": "high"
            })
        
        # Add to analytics (keep last 10 opportunities)
        self.performance_analytics["optimization_opportunities"].extend(opportunities)
        self.performance_analytics["optimization_opportunities"] = \
            self.performance_analytics["optimization_opportunities"][-10:]
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary.
        
        Returns:
            Analytics summary dictionary
        """
        return {
            "system_overview": {
                "total_queries_processed": self.performance_analytics["total_queries"],
                "routing_statistics": self.query_router.get_routing_statistics(),
            },
            "chain_performance": self.performance_analytics["chain_performance"].copy(),
            "query_type_performance": self.performance_analytics["query_type_performance"].copy(),
            "optimization_opportunities": self.performance_analytics["optimization_opportunities"].copy(),
            "recommendations": self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """
        Generate performance optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        total_queries = self.performance_analytics["total_queries"]
        if total_queries == 0:
            return ["Henüz yeterli veri yok, daha fazla sorgu işlendikten sonra öneriler sunulacak."]
        
        # Analyze chain usage efficiency
        chain_stats = self.performance_analytics["chain_performance"]
        
        # Check if stuff chain usage is optimal
        stuff_count = chain_stats["stuff"]["count"]
        total_count = sum(stats["count"] for stats in chain_stats.values())
        
        if total_count > 10:  # Only analyze if we have enough data
            stuff_ratio = stuff_count / total_count if total_count > 0 else 0
            
            if stuff_ratio < 0.4:  # Less than 40% stuff chain usage
                recommendations.append(
                    "Stuff chain kullanımı düşük. Basit sorular için routing algoritmalarını optimize edin."
                )
            
            if stuff_ratio > 0.8:  # More than 80% stuff chain usage
                recommendations.append(
                    "Stuff chain kullanımı çok yüksek. Daha karmaşık sorgular için diğer chain'leri teşvik edin."
                )
        
        # Check for slow chains
        for chain_type, stats in chain_stats.items():
            if stats["count"] > 0 and stats["avg_time"] > 8:
                recommendations.append(
                    f"{chain_type} chain ortalaması yavaş ({stats['avg_time']:.1f}s). "
                    f"Optimizasyon gerekebilir."
                )
        
        if not recommendations:
            recommendations.append("Sistem performansı optimal durumda görünüyor.")
        
        return recommendations
    
    def reset_analytics(self):
        """Reset analytics and performance tracking data."""
        self.performance_analytics = {
            "total_queries": 0,
            "chain_performance": {
                "stuff": {"count": 0, "total_time": 0.0, "avg_time": 0.0, "success_rate": 1.0},
                "refine": {"count": 0, "total_time": 0.0, "avg_time": 0.0, "success_rate": 1.0},
                "map_reduce": {"count": 0, "total_time": 0.0, "avg_time": 0.0, "success_rate": 1.0}
            },
            "query_type_performance": {qtype.value: {"count": 0, "avg_time": 0.0} for qtype in QueryType},
            "optimization_opportunities": []
        }
        self.query_router.routing_stats = {
            "total_queries": 0,
            "chain_usage": {chain.value: 0 for chain in RAGChainType},
            "query_type_distribution": {qtype.value: 0 for qtype in QueryType}
        }
        self.logger.info("Analytics data reset successfully")