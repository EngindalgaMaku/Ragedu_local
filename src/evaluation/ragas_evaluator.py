"""
RAGAS-style Evaluation System for Edu-ModRAG.

This module implements comprehensive evaluation metrics for RAG systems,
based on the RAGAS (Retrieval-Augmented Generation Assessment) framework
with educational context enhancements.

Key Metrics:
- Context Precision: How relevant are the retrieved documents?
- Context Recall: How well does retrieval capture all relevant information?
- Faithfulness: How grounded is the answer in the retrieved context?
- Answer Relevancy: How relevant is the answer to the query?
- Educational Effectiveness: Custom metrics for educational scenarios

Based on: Salemi & Zamani (2024) "Evaluating Retrieval Quality in 
Retrieval-Augmented Generation"
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import ollama
from datetime import datetime
from src.utils.logger import get_logger
from src.embedding.embedding_generator import generate_embeddings

@dataclass
class EvaluationResult:
    """Single evaluation result for a query-answer pair."""
    query_id: str
    timestamp: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    educational_effectiveness: float
    overall_score: float
    
    # Detailed breakdown
    precision_details: Dict[str, Any]
    recall_details: Dict[str, Any]
    faithfulness_details: Dict[str, Any]
    relevancy_details: Dict[str, Any]
    educational_details: Dict[str, Any]

class RAGASEvaluator:
    """
    Comprehensive RAG evaluation system with educational enhancements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__, config)
        
        # Initialize Ollama client for LLM-based evaluation
        self.ollama_client = self._init_ollama_client()
        
        # Database setup for evaluation storage
        self.db_path = config.get("evaluation_db_path", "data/analytics/evaluations.db")
        self._init_database()
        
        # Educational evaluation patterns
        self._init_educational_patterns()
        
        self.logger.info("RAGAS Evaluator initialized successfully")
    
    def _init_ollama_client(self):
        """Initialize Ollama client for evaluation."""
        try:
            client = ollama.Client(host=self.config.get("ollama_base_url"))
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client for evaluation: {e}")
            return None
    
    def _init_database(self):
        """Initialize evaluation results database."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context_precision REAL NOT NULL,
                    context_recall REAL NOT NULL,
                    faithfulness REAL NOT NULL,
                    answer_relevancy REAL NOT NULL,
                    educational_effectiveness REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    evaluation_details TEXT,
                    query_text TEXT,
                    answer_text TEXT,
                    chain_type TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluation_results(timestamp);
            """)
    
    def _init_educational_patterns(self):
        """Initialize patterns for educational effectiveness evaluation."""
        self.educational_patterns = {
            "clarity_indicators": [
                r'\b(açık|net|belirgin|anlaşılır)\b',
                r'\b(clear|obvious|evident|apparent)\b'
            ],
            "explanation_quality": [
                r'\b(çünkü|nedeni|sebep|dolayısıyla)\b',
                r'\b(because|since|therefore|thus)\b',
                r'\b(örneğin|mesela|example|instance)\b'
            ],
            "completeness_indicators": [
                r'\b(kapsamlı|detaylı|tam|eksiksiz)\b',
                r'\b(comprehensive|detailed|complete|thorough)\b'
            ],
            "pedagogical_structure": [
                r'\b(önce|sonra|ardından|finally|first|then)\b',
                r'\b(adım|step|aşama|phase)\b',
                r'\b(sonuç|conclusion|result|outcome)\b'
            ]
        }
    
    def evaluate_response(self, 
                         query: str,
                         answer: str,
                         sources: List[Dict[str, Any]],
                         query_id: str,
                         chain_type: str = "unknown",
                         ground_truth: Optional[str] = None) -> EvaluationResult:
        """
        Comprehensive evaluation of a RAG response.
        
        Args:
            query: Original user query
            answer: Generated answer
            sources: Retrieved source documents
            query_id: Unique query identifier
            chain_type: RAG chain type used
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Comprehensive evaluation result
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Starting comprehensive evaluation for query_id: {query_id}")
        
        try:
            # 1. Context Precision Evaluation
            precision_result = self._evaluate_context_precision(query, sources)
            
            # 2. Context Recall Evaluation
            recall_result = self._evaluate_context_recall(query, sources, answer)
            
            # 3. Faithfulness Evaluation
            faithfulness_result = self._evaluate_faithfulness(answer, sources)
            
            # 4. Answer Relevancy Evaluation
            relevancy_result = self._evaluate_answer_relevancy(query, answer)
            
            # 5. Educational Effectiveness Evaluation
            educational_result = self._evaluate_educational_effectiveness(query, answer, sources)
            
            # 6. Calculate Overall Score
            overall_score = self._calculate_overall_score(
                precision_result["score"],
                recall_result["score"],
                faithfulness_result["score"],
                relevancy_result["score"],
                educational_result["score"]
            )
            
            # 7. Create Evaluation Result
            evaluation = EvaluationResult(
                query_id=query_id,
                timestamp=timestamp,
                context_precision=precision_result["score"],
                context_recall=recall_result["score"],
                faithfulness=faithfulness_result["score"],
                answer_relevancy=relevancy_result["score"],
                educational_effectiveness=educational_result["score"],
                overall_score=overall_score,
                precision_details=precision_result,
                recall_details=recall_result,
                faithfulness_details=faithfulness_result,
                relevancy_details=relevancy_result,
                educational_details=educational_result
            )
            
            # 8. Store Evaluation Result
            self._store_evaluation_result(evaluation, query, answer, chain_type)
            
            evaluation_time = time.time() - start_time
            self.logger.info(
                f"Evaluation completed for query_id: {query_id} in {evaluation_time:.2f}s "
                f"(Overall Score: {overall_score:.3f})"
            )
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for query_id: {query_id}: {e}")
            
            # Return minimal evaluation result on error
            return EvaluationResult(
                query_id=query_id,
                timestamp=timestamp,
                context_precision=0.0,
                context_recall=0.0,
                faithfulness=0.0,
                answer_relevancy=0.0,
                educational_effectiveness=0.0,
                overall_score=0.0,
                precision_details={"error": str(e)},
                recall_details={"error": str(e)},
                faithfulness_details={"error": str(e)},
                relevancy_details={"error": str(e)},
                educational_details={"error": str(e)}
            )
    
    def _evaluate_context_precision(self, query: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate context precision: How relevant are the retrieved documents?
        
        Args:
            query: User query
            sources: Retrieved source documents
            
        Returns:
            Precision evaluation details
        """
        if not sources:
            return {"score": 0.0, "reason": "No sources retrieved"}
        
        try:
            relevant_count = 0
            source_relevance_scores = []
            
            for source in sources:
                # Use similarity score as relevance indicator
                relevance_score = source.get("score", 0.0)
                source_relevance_scores.append(relevance_score)
                
                # Consider relevant if above threshold
                if relevance_score > 0.5:  # Configurable threshold
                    relevant_count += 1
            
            precision_score = relevant_count / len(sources)
            avg_relevance = sum(source_relevance_scores) / len(source_relevance_scores)
            
            # Boost precision if using LLM-based relevance assessment
            if self.ollama_client and len(sources) <= 3:  # Limit for efficiency
                llm_precision = self._llm_assess_precision(query, sources)
                precision_score = (precision_score + llm_precision) / 2
            
            return {
                "score": precision_score,
                "relevant_sources": relevant_count,
                "total_sources": len(sources),
                "avg_relevance": avg_relevance,
                "source_scores": source_relevance_scores
            }
            
        except Exception as e:
            self.logger.warning(f"Context precision evaluation failed: {e}")
            return {"score": 0.5, "error": str(e)}
    
    def _llm_assess_precision(self, query: str, sources: List[Dict[str, Any]]) -> float:
        """Use LLM to assess precision of retrieved sources."""
        try:
            sources_text = "\n\n".join([
                f"Kaynak {i+1}: {source.get('text', '')[:300]}"
                for i, source in enumerate(sources[:3])
            ])
            
            prompt = f"""
            Soru: {query}
            
            Getirilen Kaynaklar:
            {sources_text}
            
            Bu kaynakların soruya ne kadar alakalı olduğunu 0-1 arasında değerlendir.
            Sadece sayısal değer ver (örn: 0.8).
            """
            
            response = self.ollama_client.chat(
                model=self.config.get("ollama_generation_model"),
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 10}
            )
            
            # Extract numeric score
            score_text = response['message']['content'].strip()
            score = float(re.findall(r'0\.\d+|1\.0|0|1', score_text)[0])
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.warning(f"LLM precision assessment failed: {e}")
            return 0.5
    
    def _evaluate_context_recall(self, query: str, sources: List[Dict[str, Any]], 
                                answer: str) -> Dict[str, Any]:
        """
        Evaluate context recall: How well does retrieval capture relevant information?
        
        Args:
            query: User query
            sources: Retrieved source documents
            answer: Generated answer
            
        Returns:
            Recall evaluation details
        """
        try:
            if not sources:
                return {"score": 0.0, "reason": "No sources to evaluate"}
            
            # Extract key terms from query and answer
            query_terms = set(re.findall(r'\w+', query.lower()))
            answer_terms = set(re.findall(r'\w+', answer.lower()))
            important_terms = query_terms.union(answer_terms)
            
            # Check coverage of important terms in sources
            source_terms = set()
            for source in sources:
                source_text = source.get("text", "").lower()
                source_terms.update(re.findall(r'\w+', source_text))
            
            # Calculate recall based on term coverage
            covered_terms = important_terms.intersection(source_terms)
            recall_score = len(covered_terms) / len(important_terms) if important_terms else 0.0
            
            # Adjust based on source diversity
            diversity_bonus = min(0.2, len(sources) * 0.05)  # Bonus for multiple sources
            recall_score = min(1.0, recall_score + diversity_bonus)
            
            return {
                "score": recall_score,
                "important_terms": len(important_terms),
                "covered_terms": len(covered_terms),
                "coverage_ratio": recall_score,
                "source_diversity": len(sources)
            }
            
        except Exception as e:
            self.logger.warning(f"Context recall evaluation failed: {e}")
            return {"score": 0.5, "error": str(e)}
    
    def _evaluate_faithfulness(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate faithfulness: How grounded is the answer in the retrieved context?
        
        Args:
            answer: Generated answer
            sources: Retrieved source documents
            
        Returns:
            Faithfulness evaluation details
        """
        try:
            if not sources or not answer:
                return {"score": 0.0, "reason": "No sources or answer to evaluate"}
            
            # Combine all source texts
            context_text = " ".join([source.get("text", "") for source in sources])
            
            # Extract claims from answer (simple sentence splitting)
            answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
            
            supported_claims = 0
            total_claims = len(answer_sentences)
            
            if total_claims == 0:
                return {"score": 0.0, "reason": "No claims to evaluate"}
            
            for sentence in answer_sentences:
                if self._is_claim_supported(sentence, context_text):
                    supported_claims += 1
            
            faithfulness_score = supported_claims / total_claims
            
            # Use LLM for deeper faithfulness assessment if available
            if self.ollama_client and len(answer) < 1000:  # Limit for efficiency
                llm_faithfulness = self._llm_assess_faithfulness(answer, context_text)
                faithfulness_score = (faithfulness_score + llm_faithfulness) / 2
            
            return {
                "score": faithfulness_score,
                "supported_claims": supported_claims,
                "total_claims": total_claims,
                "support_ratio": faithfulness_score
            }
            
        except Exception as e:
            self.logger.warning(f"Faithfulness evaluation failed: {e}")
            return {"score": 0.5, "error": str(e)}
    
    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """Check if a claim is supported by the context using keyword overlap."""
        claim_words = set(re.findall(r'\w+', claim.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))
        
        overlap = claim_words.intersection(context_words)
        overlap_ratio = len(overlap) / len(claim_words) if claim_words else 0
        
        return overlap_ratio > 0.3  # Threshold for support
    
    def _llm_assess_faithfulness(self, answer: str, context: str) -> float:
        """Use LLM to assess faithfulness of answer to context."""
        try:
            prompt = f"""
            Bağlam: {context[:1000]}...
            
            Cevap: {answer}
            
            Bu cevap verilen bağlama ne kadar sadık? (0-1 arası değer ver)
            """
            
            response = self.ollama_client.chat(
                model=self.config.get("ollama_generation_model"),
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 10}
            )
            
            score_text = response['message']['content'].strip()
            score = float(re.findall(r'0\.\d+|1\.0|0|1', score_text)[0])
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.warning(f"LLM faithfulness assessment failed: {e}")
            return 0.5
    
    def _evaluate_answer_relevancy(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Evaluate answer relevancy: How relevant is the answer to the query?
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            Relevancy evaluation details
        """
        try:
            if not query or not answer:
                return {"score": 0.0, "reason": "Missing query or answer"}
            
            # Calculate semantic similarity using embeddings
            similarity_score = 0.0
            if self.config.get("enable_semantic_analysis", True):
                query_embedding = generate_embeddings([query])
                answer_embedding = generate_embeddings([answer])
                
                if query_embedding and answer_embedding:
                    similarity_score = self._cosine_similarity(query_embedding[0], answer_embedding[0])
            
            # Calculate keyword-based relevancy
            query_words = set(re.findall(r'\w+', query.lower()))
            answer_words = set(re.findall(r'\w+', answer.lower()))
            
            common_words = query_words.intersection(answer_words)
            keyword_relevancy = len(common_words) / len(query_words) if query_words else 0
            
            # Combine semantic and keyword-based scores
            relevancy_score = (similarity_score + keyword_relevancy) / 2
            
            return {
                "score": relevancy_score,
                "semantic_similarity": similarity_score,
                "keyword_relevancy": keyword_relevancy,
                "common_keywords": len(common_words),
                "query_keywords": len(query_words)
            }
            
        except Exception as e:
            self.logger.warning(f"Answer relevancy evaluation failed: {e}")
            return {"score": 0.5, "error": str(e)}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _evaluate_educational_effectiveness(self, query: str, answer: str, 
                                          sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate educational effectiveness using custom educational metrics.
        
        Args:
            query: User query
            answer: Generated answer
            sources: Retrieved source documents
            
        Returns:
            Educational effectiveness evaluation details
        """
        try:
            scores = {
                "clarity": self._assess_clarity(answer),
                "explanation_quality": self._assess_explanation_quality(answer),
                "completeness": self._assess_completeness(answer, query),
                "pedagogical_structure": self._assess_pedagogical_structure(answer)
            }
            
            # Calculate weighted average
            weights = {
                "clarity": 0.3,
                "explanation_quality": 0.3,
                "completeness": 0.2,
                "pedagogical_structure": 0.2
            }
            
            effectiveness_score = sum(
                scores[metric] * weights[metric] 
                for metric in scores
            )
            
            return {
                "score": effectiveness_score,
                "breakdown": scores,
                "weights": weights
            }
            
        except Exception as e:
            self.logger.warning(f"Educational effectiveness evaluation failed: {e}")
            return {"score": 0.5, "error": str(e)}
    
    def _assess_clarity(self, answer: str) -> float:
        """Assess clarity of the answer."""
        clarity_score = 0.0
        
        # Check for clarity indicators
        for pattern_group in self.educational_patterns["clarity_indicators"]:
            if re.search(pattern_group, answer.lower()):
                clarity_score += 0.2
        
        # Penalize overly complex sentences
        sentences = re.split(r'[.!?]', answer)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length < 15:  # Ideal range
            clarity_score += 0.3
        elif avg_sentence_length > 25:  # Too complex
            clarity_score -= 0.1
        
        return min(1.0, max(0.0, clarity_score))
    
    def _assess_explanation_quality(self, answer: str) -> float:
        """Assess quality of explanations in the answer."""
        explanation_score = 0.0
        
        # Check for explanatory patterns
        for pattern_group in self.educational_patterns["explanation_quality"]:
            matches = re.findall(pattern_group, answer.lower())
            explanation_score += min(0.3, len(matches) * 0.1)
        
        return min(1.0, explanation_score)
    
    def _assess_completeness(self, answer: str, query: str) -> float:
        """Assess completeness of the answer relative to the query."""
        word_count = len(answer.split())
        
        # Basic completeness based on length
        if word_count < 10:
            return 0.2
        elif word_count < 50:
            return 0.6
        elif word_count < 200:
            return 0.9
        else:
            return 1.0
    
    def _assess_pedagogical_structure(self, answer: str) -> float:
        """Assess pedagogical structure of the answer."""
        structure_score = 0.0
        
        # Check for structural indicators
        for pattern_group in self.educational_patterns["pedagogical_structure"]:
            if re.search(pattern_group, answer.lower()):
                structure_score += 0.25
        
        return min(1.0, structure_score)
    
    def _calculate_overall_score(self, precision: float, recall: float, 
                               faithfulness: float, relevancy: float, 
                               educational: float) -> float:
        """Calculate weighted overall score."""
        weights = {
            "precision": 0.15,
            "recall": 0.15,
            "faithfulness": 0.25,
            "relevancy": 0.25,
            "educational": 0.20
        }
        
        return (
            precision * weights["precision"] +
            recall * weights["recall"] +
            faithfulness * weights["faithfulness"] +
            relevancy * weights["relevancy"] +
            educational * weights["educational"]
        )
    
    def _store_evaluation_result(self, evaluation: EvaluationResult, 
                               query: str, answer: str, chain_type: str):
        """Store evaluation result in database."""
        try:
            import json
            
            evaluation_details = {
                "precision": evaluation.precision_details,
                "recall": evaluation.recall_details,
                "faithfulness": evaluation.faithfulness_details,
                "relevancy": evaluation.relevancy_details,
                "educational": evaluation.educational_details
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO evaluation_results 
                    (query_id, timestamp, context_precision, context_recall, 
                     faithfulness, answer_relevancy, educational_effectiveness, 
                     overall_score, evaluation_details, query_text, answer_text, chain_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evaluation.query_id, evaluation.timestamp,
                    evaluation.context_precision, evaluation.context_recall,
                    evaluation.faithfulness, evaluation.answer_relevancy,
                    evaluation.educational_effectiveness, evaluation.overall_score,
                    json.dumps(evaluation_details, ensure_ascii=False),
                    query[:500], answer[:1000], chain_type
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to store evaluation result: {e}")
    
    def get_evaluation_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive evaluation summary."""
        try:
            from datetime import timedelta
            cutoff_time = (datetime.now() - timedelta(hours=time_range_hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_evaluations,
                        AVG(overall_score) as avg_overall_score,
                        AVG(context_precision) as avg_precision,
                        AVG(context_recall) as avg_recall,
                        AVG(faithfulness) as avg_faithfulness,
                        AVG(answer_relevancy) as avg_relevancy,
                        AVG(educational_effectiveness) as avg_educational,
                        chain_type,
                        COUNT(*) as chain_count
                    FROM evaluation_results 
                    WHERE timestamp >= ?
                    GROUP BY chain_type
                """, (cutoff_time,))
                
                chain_results = cursor.fetchall()
                
                # Overall statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        AVG(overall_score) as avg_score,
                        MIN(overall_score) as min_score,
                        MAX(overall_score) as max_score
                    FROM evaluation_results 
                    WHERE timestamp >= ?
                """, (cutoff_time,))
                
                overall_stats = cursor.fetchone()
            
            return {
                "time_range_hours": time_range_hours,
                "overall_statistics": {
                    "total_evaluations": overall_stats[0],
                    "average_score": overall_stats[1] or 0.0,
                    "min_score": overall_stats[2] or 0.0,
                    "max_score": overall_stats[3] or 0.0
                },
                "chain_performance": {
                    result[7]: {  # chain_type
                        "evaluations": result[8],  # chain_count
                        "avg_overall_score": result[1] or 0.0,
                        "avg_precision": result[2] or 0.0,
                        "avg_recall": result[3] or 0.0,
                        "avg_faithfulness": result[4] or 0.0,
                        "avg_relevancy": result[5] or 0.0,
                        "avg_educational": result[6] or 0.0
                    }
                    for result in chain_results
                },
                "recommendations": self._generate_evaluation_recommendations(chain_results)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation summary: {e}")
            return {"error": str(e)}
    
    def _generate_evaluation_recommendations(self, chain_results: List[Tuple]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        try:
            if not chain_results:
                return ["Henüz yeterli değerlendirme verisi bulunmuyor."]
            
            # Find best and worst performing chains
            chain_scores = {result[7]: result[1] for result in chain_results if result[1] is not None}
            
            if len(chain_scores) > 1:
                best_chain = max(chain_scores, key=chain_scores.get)
                worst_chain = min(chain_scores, key=chain_scores.get)
                
                if chain_scores[best_chain] - chain_scores[worst_chain] > 0.1:
                    recommendations.append(
                        f"{best_chain} chain en yüksek performansı gösteriyor "
                        f"({chain_scores[best_chain]:.3f}). {worst_chain} chain "
                        f"optimizasyona ihtiyaç duyuyor ({chain_scores[worst_chain]:.3f})."
                    )
            
            # Check specific metrics
            for result in chain_results:
                chain_type = result[7]
                if result[4] and result[4] < 0.7:  # Low faithfulness
                    recommendations.append(
                        f"{chain_type} chain sadakat skoru düşük ({result[4]:.3f}). "
                        f"Halüsinasyon kontrolü gerekli."
                    )
                
                if result[6] and result[6] < 0.6:  # Low educational effectiveness
                    recommendations.append(
                        f"{chain_type} chain eğitsel etkinlik skoru düşük ({result[6]:.3f}). "
                        f"Pedagojik kalite iyileştirmesi gerekli."
                    )
            
            if not recommendations:
                recommendations.append("Tüm sistemler kabul edilebilir performans gösteriyor.")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate recommendations: {e}")
            recommendations = ["Öneri oluşturulurken hata oluştu."]
        
        return recommendations