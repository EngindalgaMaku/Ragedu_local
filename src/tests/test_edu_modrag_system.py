"""
Comprehensive Test Suite for Edu-ModRAG System.

This module provides comprehensive testing and validation for the complete
Edu-ModRAG system, including:
- Query Router functionality
- RAG Chain implementations (Stuff, Refine, Map-Reduce)
- Performance tracking and analytics
- RAGAS evaluation metrics
- Source attribution and transparency features
- Integration testing

Usage:
    python -m src.tests.test_edu_modrag_system
"""

import os
import sys
import time
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.query_router import QueryRouter, QueryType, QueryComplexity, RAGChainType
from src.rag.rag_chains import RAGChainFactory, StuffChain, RefineChain, MapReduceChain
from src.rag.edu_modrag_pipeline import EduModRAGPipeline
from src.analytics.performance_tracker import PerformanceTracker, PerformanceMetric
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.vector_store.faiss_store import FaissVectorStore
from src.query_processing.query_processor import EnhancedQueryProcessor
from src import config as app_config
from src.utils.logger import get_logger

class EduModRAGTester:
    """Comprehensive testing system for Edu-ModRAG."""
    
    def __init__(self):
        """Initialize the test system."""
        self.logger = get_logger(__name__, app_config.get_config())
        self.config = app_config.get_config()
        
        # Test results storage
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "test_duration": 0.0
            },
            "component_tests": {},
            "integration_tests": {},
            "performance_benchmarks": {}
        }
        
        # Setup test environment
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Setup test environment with sample data."""
        try:
            # Create temporary test directory
            self.test_dir = tempfile.mkdtemp(prefix="edu_modrag_test_")
            self.logger.info(f"Test environment created: {self.test_dir}")
            
            # Update config for testing
            self.test_config = self.config.copy()
            self.test_config.update({
                "analytics_db_path": os.path.join(self.test_dir, "test_analytics.db"),
                "evaluation_db_path": os.path.join(self.test_dir, "test_evaluations.db"),
                "cache_ttl": 60  # Shorter cache for testing
            })
            
            # Create test vector store with sample data
            self._create_test_vector_store()
            
        except Exception as e:
            self.logger.error(f"Failed to setup test environment: {e}")
            raise
    
    def _create_test_vector_store(self):
        """Create a test vector store with sample educational content."""
        try:
            # Sample educational texts
            sample_texts = [
                # Mathematics content
                "Türev, bir fonksiyonun belirli bir noktadaki anlık değişim hızını gösteren matematiksel kavramdır. "
                "f(x) fonksiyonunun x noktasındaki türevi, f'(x) ile gösterilir ve limit tanımıyla hesaplanır.",
                
                # Programming content  
                "Python programlama dilinde döngüler, belirli işlemleri tekrarlamak için kullanılır. "
                "For döngüsü belirli sayıda tekrar, while döngüsü ise bir koşul sağlandığı sürece tekrar eder.",
                
                # Statistics content
                "Hipotez testi, istatistiksel bir iddiayı test etmek için kullanılan yöntemdir. "
                "Null hipotezi (H0) ve alternatif hipotez (H1) kurularak, p-değeri hesaplanır ve sonuç değerlendirilir.",
                
                # Machine Learning content
                "Derin öğrenme, yapay sinir ağlarının çok katmanlı versiyonlarını kullanan bir makine öğrenmesi yaklaşımıdır. "
                "CNN, RNN, ve Transformer gibi mimariler farklı problem türleri için kullanılır.",
                
                # Computer Science content
                "Algoritma, belirli bir problemi çözmek için izlenecek adımların açık ve kesin bir şekilde tarif edilmesidir. "
                "Zaman karmaşıklığı ve alan karmaşıklığı algoritmanın verimliliğini ölçer."
            ]
            
            # Create FAISS vector store
            index_path = os.path.join(self.test_dir, "test_vector_store")
            self.test_vector_store = FaissVectorStore(index_path=index_path)
            
            # Generate embeddings for sample texts (mock implementation)
            import numpy as np
            mock_embeddings = []
            for i, text in enumerate(sample_texts):
                # Create deterministic embeddings for testing
                np.random.seed(i)
                embedding = np.random.normal(0, 1, 1536).tolist()  # Standard embedding dimension
                mock_embeddings.append(embedding)
            
            # Add documents to vector store
            metadatas = [
                {"source": "math_textbook", "chapter": "derivatives"},
                {"source": "python_guide", "chapter": "control_flow"},
                {"source": "statistics_manual", "chapter": "hypothesis_testing"},
                {"source": "ml_handbook", "chapter": "deep_learning"},
                {"source": "cs_fundamentals", "chapter": "algorithms"}
            ]
            
            self.test_vector_store.add_documents(sample_texts, mock_embeddings, metadatas)
            self.test_vector_store.save_store()
            
            self.logger.info(f"Test vector store created with {len(sample_texts)} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to create test vector store: {e}")
            raise
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results."""
        start_time = time.time()
        
        self.logger.info("🚀 Starting comprehensive Edu-ModRAG system tests...")
        
        try:
            # Component Tests
            self.logger.info("📋 Running component tests...")
            self._test_query_router()
            self._test_rag_chains()
            self._test_enhanced_query_processor()
            self._test_performance_tracker()
            self._test_ragas_evaluator()
            
            # Integration Tests
            self.logger.info("🔗 Running integration tests...")
            self._test_edu_modrag_pipeline()
            self._test_end_to_end_workflow()
            
            # Performance Benchmarks
            self.logger.info("⚡ Running performance benchmarks...")
            self._benchmark_system_performance()
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            self.test_results["test_summary"]["fatal_error"] = str(e)
        
        finally:
            # Calculate final statistics
            total_duration = time.time() - start_time
            self.test_results["test_summary"]["test_duration"] = total_duration
            
            self._generate_test_report()
            self.logger.info(f"✅ Test suite completed in {total_duration:.2f}s")
        
        return self.test_results
    
    def _test_query_router(self):
        """Test Query Router functionality."""
        test_name = "Query Router"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            router = QueryRouter(self.test_config)
            
            test_queries = [
                ("Python nedir?", QueryType.SIMPLE_FACTUAL, RAGChainType.STUFF),
                ("Python ve Java arasındaki farklar nelerdir?", QueryType.COMPARATIVE, RAGChainType.REFINE),
                ("Python'da for döngüsü nasıl kullanılır?", QueryType.APPLIED_PROCEDURAL, RAGChainType.REFINE),
                ("Makine öğrenmesi algoritmalarını özetle", QueryType.MULTI_DOCUMENT, RAGChainType.MAP_REDUCE),
                ("Algoritma verimliliği neden önemlidir?", QueryType.CONCEPTUAL, RAGChainType.REFINE)
            ]
            
            results = {"passed": 0, "failed": 0, "details": []}
            
            for query, expected_type, expected_chain in test_queries:
                classification = router.classify_query(query)
                
                # Validate classification results
                if (classification["query_type"] == expected_type and 
                    classification["recommended_chain"] == expected_chain):
                    results["passed"] += 1
                    results["details"].append(f"✅ '{query}' correctly classified")
                else:
                    results["failed"] += 1
                    results["details"].append(f"❌ '{query}' misclassified: got {classification['query_type']}, expected {expected_type}")
            
            # Test routing statistics
            stats = router.get_routing_statistics()
            if stats["total_queries"] == len(test_queries):
                results["passed"] += 1
                results["details"].append("✅ Statistics tracking working correctly")
            else:
                results["failed"] += 1
                results["details"].append("❌ Statistics tracking failed")
            
            self.test_results["component_tests"][test_name] = results
            self.test_results["test_summary"]["total_tests"] += results["passed"] + results["failed"]
            self.test_results["test_summary"]["passed_tests"] += results["passed"]
            self.test_results["test_summary"]["failed_tests"] += results["failed"]
            
        except Exception as e:
            self.logger.error(f"{test_name} test failed: {e}")
            self.test_results["component_tests"][test_name] = {"error": str(e)}
    
    def _test_rag_chains(self):
        """Test individual RAG chain implementations."""
        test_name = "RAG Chains"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            results = {"passed": 0, "failed": 0, "details": []}
            
            # Test each chain type
            chain_types = ["stuff", "refine", "map_reduce"]
            test_query = "Python döngüleri nasıl çalışır?"
            
            for chain_type in chain_types:
                try:
                    chain = RAGChainFactory.create_chain(chain_type, self.test_config, self.test_vector_store)
                    response = chain.execute(test_query, top_k=3)
                    
                    # Validate response structure
                    if (isinstance(response, dict) and 
                        "answer" in response and 
                        "sources" in response and
                        "chain_type" in response):
                        results["passed"] += 1
                        results["details"].append(f"✅ {chain_type} chain executed successfully")
                    else:
                        results["failed"] += 1
                        results["details"].append(f"❌ {chain_type} chain returned invalid response")
                        
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(f"❌ {chain_type} chain failed: {str(e)}")
            
            self.test_results["component_tests"][test_name] = results
            self.test_results["test_summary"]["total_tests"] += results["passed"] + results["failed"]
            self.test_results["test_summary"]["passed_tests"] += results["passed"]
            self.test_results["test_summary"]["failed_tests"] += results["failed"]
            
        except Exception as e:
            self.logger.error(f"{test_name} test failed: {e}")
            self.test_results["component_tests"][test_name] = {"error": str(e)}
    
    def _test_enhanced_query_processor(self):
        """Test Enhanced Query Processor functionality."""
        test_name = "Enhanced Query Processor"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            processor = EnhancedQueryProcessor(self.test_config)
            
            test_queries = [
                "Python nedir ve nasıl kullanılır?",
                "Makine öğrenmesi algoritmaları",
                "invalid query!@#$%",
                "a"  # Too short
            ]
            
            results = {"passed": 0, "failed": 0, "details": []}
            
            for query in test_queries:
                try:
                    processed = processor.process(query)
                    
                    if "error" not in processed:
                        # Valid processing
                        if ("semantic_analysis" in processed and 
                            "query_expansion" in processed and
                            "processing_metadata" in processed):
                            results["passed"] += 1
                            results["details"].append(f"✅ Query processed successfully: '{query[:30]}...'")
                        else:
                            results["failed"] += 1
                            results["details"].append(f"❌ Incomplete processing for: '{query[:30]}...'")
                    else:
                        # Expected validation failure for invalid queries
                        if len(query) < 3 or not query.replace("!", "").replace("@", "").replace("#", "").replace("$", "").replace("%", "").strip():
                            results["passed"] += 1
                            results["details"].append(f"✅ Invalid query correctly rejected: '{query[:30]}...'")
                        else:
                            results["failed"] += 1
                            results["details"].append(f"❌ Valid query incorrectly rejected: '{query[:30]}...'")
                            
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(f"❌ Processing error for '{query[:30]}...': {str(e)}")
            
            self.test_results["component_tests"][test_name] = results
            self.test_results["test_summary"]["total_tests"] += results["passed"] + results["failed"]
            self.test_results["test_summary"]["passed_tests"] += results["passed"]
            self.test_results["test_summary"]["failed_tests"] += results["failed"]
            
        except Exception as e:
            self.logger.error(f"{test_name} test failed: {e}")
            self.test_results["component_tests"][test_name] = {"error": str(e)}
    
    def _test_performance_tracker(self):
        """Test Performance Tracker functionality."""
        test_name = "Performance Tracker"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            tracker = PerformanceTracker(self.test_config)
            
            # Create sample performance metrics
            sample_metrics = [
                PerformanceMetric(
                    timestamp=datetime.now().isoformat(),
                    query_id="test_001",
                    query_text="Test query 1",
                    chain_type="stuff",
                    query_type="simple_factual",
                    complexity_level="low",
                    execution_time=1.5,
                    tokens_used=150,
                    success=True
                ),
                PerformanceMetric(
                    timestamp=datetime.now().isoformat(),
                    query_id="test_002", 
                    query_text="Test query 2",
                    chain_type="refine",
                    query_type="comparative",
                    complexity_level="medium",
                    execution_time=3.2,
                    tokens_used=300,
                    success=True
                )
            ]
            
            results = {"passed": 0, "failed": 0, "details": []}
            
            # Test recording metrics
            for metric in sample_metrics:
                try:
                    tracker.record_performance(metric)
                    results["passed"] += 1
                    results["details"].append(f"✅ Metric recorded: {metric.query_id}")
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(f"❌ Failed to record metric {metric.query_id}: {str(e)}")
            
            # Test retrieving statistics
            try:
                stats = tracker.get_chain_performance_stats(time_range_hours=1)
                cost_analysis = tracker.get_cost_analysis(time_range_hours=1)
                
                if stats and cost_analysis:
                    results["passed"] += 1
                    results["details"].append("✅ Statistics retrieval successful")
                else:
                    results["failed"] += 1
                    results["details"].append("❌ Statistics retrieval returned empty results")
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Statistics retrieval failed: {str(e)}")
            
            self.test_results["component_tests"][test_name] = results
            self.test_results["test_summary"]["total_tests"] += results["passed"] + results["failed"]
            self.test_results["test_summary"]["passed_tests"] += results["passed"]
            self.test_results["test_summary"]["failed_tests"] += results["failed"]
            
        except Exception as e:
            self.logger.error(f"{test_name} test failed: {e}")
            self.test_results["component_tests"][test_name] = {"error": str(e)}
    
    def _test_ragas_evaluator(self):
        """Test RAGAS Evaluator functionality."""
        test_name = "RAGAS Evaluator"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            evaluator = RAGASEvaluator(self.test_config)
            
            # Sample evaluation data
            query = "Python döngüleri nasıl çalışır?"
            answer = "Python'da for ve while döngüleri kullanarak tekrarlı işlemler yapabilirsiniz."
            sources = [
                {
                    "text": "Python programlama dilinde döngüler, belirli işlemleri tekrarlamak için kullanılır.",
                    "score": 0.85,
                    "metadata": {"source": "python_guide"}
                }
            ]
            
            results = {"passed": 0, "failed": 0, "details": []}
            
            try:
                evaluation = evaluator.evaluate_response(
                    query=query,
                    answer=answer,
                    sources=sources,
                    query_id="eval_test_001",
                    chain_type="stuff"
                )
                
                # Validate evaluation result structure
                required_fields = [
                    "context_precision", "context_recall", "faithfulness",
                    "answer_relevancy", "educational_effectiveness", "overall_score"
                ]
                
                valid_structure = all(hasattr(evaluation, field) for field in required_fields)
                
                if valid_structure:
                    results["passed"] += 1
                    results["details"].append(f"✅ Evaluation completed successfully (Overall: {evaluation.overall_score:.3f})")
                else:
                    results["failed"] += 1
                    results["details"].append("❌ Evaluation result has invalid structure")
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Evaluation failed: {str(e)}")
            
            # Test evaluation summary
            try:
                summary = evaluator.get_evaluation_summary(time_range_hours=1)
                if isinstance(summary, dict) and "overall_statistics" in summary:
                    results["passed"] += 1
                    results["details"].append("✅ Evaluation summary retrieved successfully")
                else:
                    results["failed"] += 1
                    results["details"].append("❌ Evaluation summary has invalid structure")
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Evaluation summary failed: {str(e)}")
            
            self.test_results["component_tests"][test_name] = results
            self.test_results["test_summary"]["total_tests"] += results["passed"] + results["failed"]
            self.test_results["test_summary"]["passed_tests"] += results["passed"]
            self.test_results["test_summary"]["failed_tests"] += results["failed"]
            
        except Exception as e:
            self.logger.error(f"{test_name} test failed: {e}")
            self.test_results["component_tests"][test_name] = {"error": str(e)}
    
    def _test_edu_modrag_pipeline(self):
        """Test the complete Edu-ModRAG pipeline integration."""
        test_name = "Edu-ModRAG Pipeline Integration"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            pipeline = EduModRAGPipeline(self.test_config, self.test_vector_store)
            
            test_queries = [
                "Python nedir?",
                "For döngüsü ve while döngüsü arasındaki farklar nelerdir?",
                "Makine öğrenmesi algoritmalarını kullanarak bir örnek göster"
            ]
            
            results = {"passed": 0, "failed": 0, "details": []}
            
            for query in test_queries:
                try:
                    response = pipeline.execute(query)
                    
                    # Validate response structure
                    required_fields = [
                        "answer", "sources", "source_attributions", "query_analysis",
                        "chain_used", "execution_time", "transparency_info", "success"
                    ]
                    
                    valid_response = all(hasattr(response, field) for field in required_fields)
                    
                    if valid_response and response.success:
                        results["passed"] += 1
                        results["details"].append(f"✅ Pipeline executed successfully for: '{query[:30]}...'")
                        results["details"].append(f"   Chain used: {response.chain_used}, Time: {response.execution_time:.2f}s")
                    else:
                        results["failed"] += 1
                        results["details"].append(f"❌ Pipeline failed for: '{query[:30]}...'")
                        if hasattr(response, 'error_message') and response.error_message:
                            results["details"].append(f"   Error: {response.error_message}")
                            
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(f"❌ Pipeline error for '{query[:30]}...': {str(e)}")
            
            # Test analytics summary
            try:
                analytics = pipeline.get_analytics_summary()
                if isinstance(analytics, dict) and "system_overview" in analytics:
                    results["passed"] += 1
                    results["details"].append("✅ Analytics summary retrieved successfully")
                else:
                    results["failed"] += 1
                    results["details"].append("❌ Analytics summary has invalid structure")
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Analytics summary failed: {str(e)}")
            
            self.test_results["integration_tests"][test_name] = results
            self.test_results["test_summary"]["total_tests"] += results["passed"] + results["failed"]
            self.test_results["test_summary"]["passed_tests"] += results["passed"]
            self.test_results["test_summary"]["failed_tests"] += results["failed"]
            
        except Exception as e:
            self.logger.error(f"{test_name} test failed: {e}")
            self.test_results["integration_tests"][test_name] = {"error": str(e)}
    
    def _test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        test_name = "End-to-End Workflow"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            # Initialize all components
            pipeline = EduModRAGPipeline(self.test_config, self.test_vector_store)
            
            # Test workflow: Query -> Route -> Execute -> Track -> Evaluate
            test_query = "Algoritma karmaşıklığı nasıl hesaplanır?"
            
            results = {"passed": 0, "failed": 0, "details": [], "workflow_steps": []}
            
            # Step 1: Execute query
            start_time = time.time()
            response = pipeline.execute(test_query)
            execution_time = time.time() - start_time
            
            if response.success:
                results["passed"] += 1
                results["details"].append("✅ Step 1: Query execution successful")
                results["workflow_steps"].append({
                    "step": "Query Execution",
                    "duration": execution_time,
                    "chain_used": response.chain_used,
                    "sources_found": len(response.sources)
                })
            else:
                results["failed"] += 1
                results["details"].append("❌ Step 1: Query execution failed")
                results["workflow_steps"].append({
                    "step": "Query Execution",
                    "duration": execution_time,
                    "error": response.error_message
                })
            
            # Step 2: Check transparency features
            if response.transparency_info and response.source_attributions:
                results["passed"] += 1
                results["details"].append("✅ Step 2: Transparency features working")
                results["workflow_steps"].append({
                    "step": "Transparency Analysis",
                    "attributions_count": len(response.source_attributions),
                    "reasoning_provided": bool(response.transparency_info.get("reasoning_process"))
                })
            else:
                results["failed"] += 1
                results["details"].append("❌ Step 2: Transparency features missing")
            
            # Step 3: Verify performance tracking
            try:
                analytics = pipeline.get_analytics_summary()
                if analytics.get("system_overview", {}).get("total_queries_processed", 0) > 0:
                    results["passed"] += 1
                    results["details"].append("✅ Step 3: Performance tracking active")
                    results["workflow_steps"].append({
                        "step": "Performance Tracking",
                        "queries_tracked": analytics["system_overview"]["total_queries_processed"]
                    })
                else:
                    results["failed"] += 1
                    results["details"].append("❌ Step 3: Performance tracking not working")
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Step 3: Performance tracking error: {str(e)}")
            
            # Overall workflow assessment
            success_rate = results["passed"] / (results["passed"] + results["failed"]) if (results["passed"] + results["failed"]) > 0 else 0
            results["workflow_success_rate"] = success_rate
            
            if success_rate >= 0.8:  # 80% success threshold
                results["details"].append(f"✅ End-to-End Workflow: {success_rate:.1%} success rate")
            else:
                results["details"].append(f"❌ End-to-End Workflow: {success_rate:.1%} success rate (below threshold)")
            
            self.test_results["integration_tests"][test_name] = results
            self.test_results["test_summary"]["total_tests"] += results["passed"] + results["failed"]
            self.test_results["test_summary"]["passed_tests"] += results["passed"]
            self.test_results["test_summary"]["failed_tests"] += results["failed"]
            
        except Exception as e:
            self.logger.error(f"{test_name} test failed: {e}")
            self.test_results["integration_tests"][test_name] = {"error": str(e)}
    
    def _benchmark_system_performance(self):
        """Benchmark system performance across different scenarios."""
        test_name = "Performance Benchmarks"
        self.logger.info(f"Running {test_name}...")
        
        try:
            pipeline = EduModRAGPipeline(self.test_config, self.test_vector_store)
            
            benchmark_queries = [
                ("Python nedir?", "simple"),
                ("Python ve Java karşılaştırması yapın", "complex"),
                ("Makine öğrenmesi algoritmalarını açıklayın", "detailed")
            ]
            
            benchmarks = {"chain_performance": {}, "query_complexity_impact": {}}
            
            # Benchmark different chains
            for chain_type in ["stuff", "refine", "map_reduce"]:
                chain_times = []
                
                for query, complexity in benchmark_queries:
                    start_time = time.time()
                    response = pipeline.execute(query, force_chain=chain_type)
                    execution_time = time.time() - start_time
                    
                    chain_times.append({
                        "query": query[:30] + "...",
                        "complexity": complexity,
                        "execution_time": execution_time,
                        "success": response.success,
                        "sources_count": len(response.sources) if response.sources else 0
                    })
                
                benchmarks["chain_performance"][chain_type] = {
                    "tests": chain_times,
                    "avg_time": sum(t["execution_time"] for t in chain_times) / len(chain_times),
                    "success_rate": sum(1 for t in chain_times if t["success"]) / len(chain_times)
                }
            
            # Analyze query complexity impact
            complexity_times = {"simple": [], "complex": [], "detailed": []}
            
            for chain_type, results in benchmarks["chain_performance"].items():
                for test in results["tests"]:
                    complexity_times[test["complexity"]].append(test["execution_time"])
            
            for complexity, times in complexity_times.items():
                if times:
                    benchmarks["query_complexity_impact"][complexity] = {
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times)
                    }
            
            self.test_results["performance_benchmarks"] = benchmarks
            
            # Performance summary
            fastest_chain = min(benchmarks["chain_performance"].items(), 
                              key=lambda x: x[1]["avg_time"])
            most_reliable_chain = max(benchmarks["chain_performance"].items(),
                                    key=lambda x: x[1]["success_rate"])
            
            self.test_results["performance_benchmarks"]["summary"] = {
                "fastest_chain": {"name": fastest_chain[0], "avg_time": fastest_chain[1]["avg_time"]},
                "most_reliable_chain": {"name": most_reliable_chain[0], "success_rate": most_reliable_chain[1]["success_rate"]},
                "recommendations": self._generate_performance_recommendations(benchmarks)
            }
            
        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            self.test_results["performance_benchmarks"]["error"] = str(e)
    
    def _generate_performance_recommendations(self, benchmarks: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        try:
            chain_perf = benchmarks.get("chain_performance", {})
            
            # Identify performance patterns
            if "stuff" in chain_perf and chain_perf["stuff"]["avg_time"] < 2.0:
                recommendations.append("Stuff chain optimized for quick responses - ideal for simple queries")
            
            if "refine" in chain_perf and chain_perf["refine"]["success_rate"] > 0.9:
                recommendations.append("Refine chain shows high reliability - recommended for complex analysis")
            
            if "map_reduce" in chain_perf and chain_perf["map_reduce"]["avg_time"] > 5.0:
                recommendations.append("Map-Reduce chain slower but comprehensive - use for multi-document queries")
            
            # Complexity-based recommendations
            complexity_impact = benchmarks.get("query_complexity_impact", {})
            if complexity_impact:
                simple_avg = complexity_impact.get("simple", {}).get("avg_time", 0)
                complex_avg = complexity_impact.get("complex", {}).get("avg_time", 0)
                
                if complex_avg > simple_avg * 2:
                    recommendations.append("Complex queries take significantly longer - consider query preprocessing")
            
            if not recommendations:
                recommendations.append("System performance within expected parameters")
                
        except Exception as e:
            recommendations.append(f"Could not generate recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        try:
            report_path = os.path.join(self.test_dir, "edu_modrag_test_report.json")
            
            # Add system information
            self.test_results["system_info"] = {
                "python_version": sys.version,
                "test_environment": self.test_dir,
                "configuration": {
                    "embedding_model": self.test_config.get("ollama_embedding_model", "N/A"),
                    "generation_model": self.test_config.get("ollama_generation_model", "N/A"),
                    "cache_enabled": self.test_config.get("enable_cache", False)
                }
            }
            
            # Calculate success rate
            total_tests = self.test_results["test_summary"]["total_tests"]
            passed_tests = self.test_results["test_summary"]["passed_tests"]
            success_rate = (passed_tests / total_tests) if total_tests > 0 else 0
            
            self.test_results["test_summary"]["success_rate"] = success_rate
            
            # Write report
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Test report generated: {report_path}")
            self.logger.info(f"📈 Overall success rate: {success_rate:.1%}")
            
            # Print summary to console
            self._print_test_summary()
            
        except Exception as e:
            self.logger.error(f"Failed to generate test report: {e}")
    
    def _print_test_summary(self):
        """Print test summary to console."""
        summary = self.test_results["test_summary"]
        
        print("\n" + "="*60)
        print("🎯 EDU-MODRAG SYSTEM TEST SUMMARY")
        print("="*60)
        print(f"⏱️  Total Duration: {summary['test_duration']:.2f} seconds")
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary.get('success_rate', 0):.1%}")
        
        if summary['failed_tests'] == 0:
            print("🎉 All tests passed! System is ready for production.")
        elif summary.get('success_rate', 0) >= 0.8:
            print("⚠️  Most tests passed. Minor issues may need attention.")
        else:
            print("❌ Multiple test failures. System needs debugging.")
        
        print("="*60)
    
    def cleanup(self):
        """Clean up test environment."""
        try:
            import shutil
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                self.logger.info(f"Test environment cleaned up: {self.test_dir}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup test environment: {e}")

def main():
    """Main test execution function."""
    tester = EduModRAGTester()
    
    try:
        results = tester.run_all_tests()
        
        # Determine exit code based on test results
        success_rate = results["test_summary"].get("success_rate", 0)
        exit_code = 0 if success_rate >= 0.8 else 1
        
        return exit_code
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1
    
    finally:
        tester.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)