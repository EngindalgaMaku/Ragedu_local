"""
Performance and Cost Tracking System for Edu-ModRAG.

This module provides comprehensive tracking of performance metrics, cost analysis,
and optimization insights for different RAG chain strategies.

Key Features:
- Real-time performance monitoring
- Cost analysis per query and chain type
- Resource utilization tracking
- Optimization recommendations
- Export capabilities for analysis

Based on research findings from Şakar & Emekci (2024) regarding RAG efficiency metrics.
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict
from src.utils.logger import get_logger

@dataclass
class PerformanceMetric:
    """Single performance measurement record."""
    timestamp: str
    query_id: str
    query_text: str
    chain_type: str
    query_type: str
    complexity_level: str
    execution_time: float
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    
    # Chain-specific metrics
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    documents_retrieved: int = 0
    context_length: int = 0
    
    # Cost metrics
    estimated_cost: float = 0.0
    cost_per_token: float = 0.0001  # Default cost estimate

@dataclass
class ChainPerformanceStats:
    """Aggregated performance statistics for a chain type."""
    chain_type: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    avg_execution_time: float
    avg_tokens_used: float
    avg_cost: float
    total_cost: float
    min_execution_time: float
    max_execution_time: float
    percentile_95_time: float

class PerformanceTracker:
    """
    Comprehensive performance and cost tracking system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the performance tracker.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__, config)
        
        # Database setup
        self.db_path = config.get("analytics_db_path", "data/analytics/performance.db")
        self._init_database()
        
        # In-memory cache for real-time metrics
        self.real_time_metrics = {
            "current_session": {
                "start_time": datetime.now(),
                "queries_processed": 0,
                "total_cost": 0.0,
                "avg_response_time": 0.0
            },
            "chain_stats": defaultdict(lambda: {
                "count": 0, "total_time": 0.0, "total_cost": 0.0,
                "success_count": 0, "avg_time": 0.0
            })
        }
        
        # Thread lock for database operations
        self._db_lock = threading.Lock()
        
        # Cost configuration
        self.cost_config = {
            "token_costs": {
                "input": config.get("input_token_cost", 0.0001),
                "output": config.get("output_token_cost", 0.0002),
            },
            "compute_costs": {
                "stuff": config.get("stuff_compute_cost", 0.001),
                "refine": config.get("refine_compute_cost", 0.003),
                "map_reduce": config.get("map_reduce_compute_cost", 0.005),
            }
        }
        
        self.logger.info("Performance tracker initialized successfully")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    chain_type TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    complexity_level TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    retrieval_time REAL DEFAULT 0.0,
                    generation_time REAL DEFAULT 0.0,
                    documents_retrieved INTEGER DEFAULT 0,
                    context_length INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0.0,
                    cost_per_token REAL DEFAULT 0.0001
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chain_type ON performance_metrics(chain_type);
            """)
    
    def record_performance(self, metric: PerformanceMetric):
        """
        Record a performance metric.
        
        Args:
            metric: Performance metric to record
        """
        try:
            # Calculate estimated cost
            metric.estimated_cost = self._calculate_cost(metric)
            
            # Store in database
            with self._db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO performance_metrics 
                        (timestamp, query_id, query_text, chain_type, query_type, 
                         complexity_level, execution_time, tokens_used, success, 
                         error_message, retrieval_time, generation_time, 
                         documents_retrieved, context_length, estimated_cost, cost_per_token)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric.timestamp, metric.query_id, metric.query_text,
                        metric.chain_type, metric.query_type, metric.complexity_level,
                        metric.execution_time, metric.tokens_used, metric.success,
                        metric.error_message, metric.retrieval_time, metric.generation_time,
                        metric.documents_retrieved, metric.context_length,
                        metric.estimated_cost, metric.cost_per_token
                    ))
            
            # Update real-time metrics
            self._update_real_time_metrics(metric)
            
            self.logger.debug(f"Recorded performance metric for query_id: {metric.query_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record performance metric: {e}")
    
    def _calculate_cost(self, metric: PerformanceMetric) -> float:
        """
        Calculate estimated cost for the operation.
        
        Args:
            metric: Performance metric
            
        Returns:
            Estimated cost in USD
        """
        try:
            # Token-based cost
            token_cost = (
                metric.tokens_used * self.cost_config["token_costs"]["input"] +
                metric.tokens_used * 0.3 * self.cost_config["token_costs"]["output"]  # Estimate 30% output tokens
            )
            
            # Compute cost based on chain type
            compute_cost = self.cost_config["compute_costs"].get(metric.chain_type, 0.001)
            
            # Time-based cost factor
            time_cost = metric.execution_time * compute_cost
            
            return token_cost + time_cost
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate cost: {e}")
            return 0.001  # Default minimal cost
    
    def _update_real_time_metrics(self, metric: PerformanceMetric):
        """Update in-memory real-time metrics."""
        try:
            session = self.real_time_metrics["current_session"]
            session["queries_processed"] += 1
            session["total_cost"] += metric.estimated_cost
            
            # Update average response time
            current_avg = session["avg_response_time"]
            count = session["queries_processed"]
            session["avg_response_time"] = (
                (current_avg * (count - 1) + metric.execution_time) / count
            )
            
            # Update chain-specific stats
            chain_stats = self.real_time_metrics["chain_stats"][metric.chain_type]
            chain_stats["count"] += 1
            chain_stats["total_time"] += metric.execution_time
            chain_stats["total_cost"] += metric.estimated_cost
            
            if metric.success:
                chain_stats["success_count"] += 1
            
            chain_stats["avg_time"] = chain_stats["total_time"] / chain_stats["count"]
            
        except Exception as e:
            self.logger.warning(f"Failed to update real-time metrics: {e}")
    
    def get_chain_performance_stats(self, 
                                   chain_type: Optional[str] = None,
                                   time_range_hours: int = 24) -> List[ChainPerformanceStats]:
        """
        Get performance statistics for chain types.
        
        Args:
            chain_type: Specific chain type to analyze (None for all)
            time_range_hours: Time range in hours to analyze
            
        Returns:
            List of chain performance statistics
        """
        try:
            cutoff_time = (datetime.now() - timedelta(hours=time_range_hours)).isoformat()
            
            query = """
                SELECT 
                    chain_type,
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_queries,
                    AVG(CASE WHEN success = 1 THEN execution_time ELSE NULL END) as avg_execution_time,
                    AVG(tokens_used) as avg_tokens_used,
                    AVG(estimated_cost) as avg_cost,
                    SUM(estimated_cost) as total_cost,
                    MIN(execution_time) as min_execution_time,
                    MAX(execution_time) as max_execution_time
                FROM performance_metrics 
                WHERE timestamp >= ?
            """
            
            params = [cutoff_time]
            
            if chain_type:
                query += " AND chain_type = ?"
                params.append(chain_type)
            
            query += " GROUP BY chain_type"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                results = cursor.fetchall()
            
            stats_list = []
            for row in results:
                (ct, total, successful, failed, avg_time, avg_tokens, 
                 avg_cost, total_cost, min_time, max_time) = row
                
                success_rate = successful / total if total > 0 else 0.0
                
                # Get 95th percentile execution time
                percentile_95 = self._get_percentile_time(ct, 95, time_range_hours)
                
                stats = ChainPerformanceStats(
                    chain_type=ct,
                    total_queries=total,
                    successful_queries=successful,
                    failed_queries=failed,
                    success_rate=success_rate,
                    avg_execution_time=avg_time or 0.0,
                    avg_tokens_used=avg_tokens or 0.0,
                    avg_cost=avg_cost or 0.0,
                    total_cost=total_cost or 0.0,
                    min_execution_time=min_time or 0.0,
                    max_execution_time=max_time or 0.0,
                    percentile_95_time=percentile_95
                )
                stats_list.append(stats)
            
            return stats_list
            
        except Exception as e:
            self.logger.error(f"Failed to get chain performance stats: {e}")
            return []
    
    def _get_percentile_time(self, chain_type: str, percentile: int, time_range_hours: int) -> float:
        """Get percentile execution time for a chain type."""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=time_range_hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT execution_time 
                    FROM performance_metrics 
                    WHERE chain_type = ? AND timestamp >= ? AND success = 1
                    ORDER BY execution_time
                """, (chain_type, cutoff_time))
                
                times = [row[0] for row in cursor.fetchall()]
                
                if not times:
                    return 0.0
                
                index = int(len(times) * percentile / 100)
                return times[min(index, len(times) - 1)]
                
        except Exception as e:
            self.logger.warning(f"Failed to get percentile time: {e}")
            return 0.0
    
    def get_cost_analysis(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive cost analysis.
        
        Args:
            time_range_hours: Time range in hours to analyze
            
        Returns:
            Cost analysis dictionary
        """
        try:
            cutoff_time = (datetime.now() - timedelta(hours=time_range_hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Overall cost stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        SUM(estimated_cost) as total_cost,
                        AVG(estimated_cost) as avg_cost_per_query,
                        MIN(estimated_cost) as min_cost,
                        MAX(estimated_cost) as max_cost
                    FROM performance_metrics 
                    WHERE timestamp >= ?
                """, (cutoff_time,))
                
                overall_stats = cursor.fetchone()
                
                # Cost by chain type
                cursor = conn.execute("""
                    SELECT 
                        chain_type,
                        COUNT(*) as queries,
                        SUM(estimated_cost) as total_cost,
                        AVG(estimated_cost) as avg_cost,
                        AVG(execution_time) as avg_time
                    FROM performance_metrics 
                    WHERE timestamp >= ?
                    GROUP BY chain_type
                    ORDER BY total_cost DESC
                """, (cutoff_time,))
                
                chain_costs = {}
                for row in cursor.fetchall():
                    chain_type, queries, total_cost, avg_cost, avg_time = row
                    chain_costs[chain_type] = {
                        "queries": queries,
                        "total_cost": total_cost,
                        "avg_cost": avg_cost,
                        "avg_time": avg_time,
                        "cost_per_second": avg_cost / avg_time if avg_time > 0 else 0
                    }
                
                # Cost by query type
                cursor = conn.execute("""
                    SELECT 
                        query_type,
                        COUNT(*) as queries,
                        SUM(estimated_cost) as total_cost,
                        AVG(estimated_cost) as avg_cost
                    FROM performance_metrics 
                    WHERE timestamp >= ?
                    GROUP BY query_type
                    ORDER BY total_cost DESC
                """, (cutoff_time,))
                
                query_type_costs = {}
                for row in cursor.fetchall():
                    query_type, queries, total_cost, avg_cost = row
                    query_type_costs[query_type] = {
                        "queries": queries,
                        "total_cost": total_cost,
                        "avg_cost": avg_cost
                    }
            
            return {
                "analysis_period_hours": time_range_hours,
                "overall_stats": {
                    "total_queries": overall_stats[0] or 0,
                    "total_cost": overall_stats[1] or 0.0,
                    "avg_cost_per_query": overall_stats[2] or 0.0,
                    "min_cost": overall_stats[3] or 0.0,
                    "max_cost": overall_stats[4] or 0.0,
                    "projected_monthly_cost": (overall_stats[1] or 0.0) * (30 * 24 / time_range_hours)
                },
                "cost_by_chain": chain_costs,
                "cost_by_query_type": query_type_costs,
                "cost_efficiency_recommendations": self._generate_cost_recommendations(chain_costs)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cost analysis: {e}")
            return {"error": str(e)}
    
    def _generate_cost_recommendations(self, chain_costs: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if not chain_costs:
            return recommendations
        
        # Find most expensive chain per query
        most_expensive = max(chain_costs.items(), key=lambda x: x[1]['avg_cost'])
        least_expensive = min(chain_costs.items(), key=lambda x: x[1]['avg_cost'])
        
        if most_expensive[1]['avg_cost'] > least_expensive[1]['avg_cost'] * 2:
            recommendations.append(
                f"{most_expensive[0]} chain maliyeti {least_expensive[0]} chain'den "
                f"{most_expensive[1]['avg_cost'] / least_expensive[1]['avg_cost']:.1f}x daha yüksek. "
                f"Query routing optimizasyonu düşünün."
            )
        
        # Check cost per second efficiency
        for chain_type, stats in chain_costs.items():
            if stats['cost_per_second'] > 0.001:  # High cost per second threshold
                recommendations.append(
                    f"{chain_type} chain saniye başına maliyeti yüksek ({stats['cost_per_second']:.4f}). "
                    f"Performance optimizasyonu gerekli."
                )
        
        return recommendations
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time performance metrics."""
        return {
            "current_session": self.real_time_metrics["current_session"].copy(),
            "chain_stats": dict(self.real_time_metrics["chain_stats"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_optimization_insights(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Generate optimization insights based on performance data.
        
        Args:
            time_range_hours: Time range in hours to analyze
            
        Returns:
            Optimization insights dictionary
        """
        try:
            chain_stats = self.get_chain_performance_stats(time_range_hours=time_range_hours)
            cost_analysis = self.get_cost_analysis(time_range_hours)
            
            insights = {
                "performance_insights": [],
                "cost_insights": [],
                "routing_insights": [],
                "optimization_score": 0.0,
                "recommendations": []
            }
            
            if not chain_stats:
                insights["recommendations"].append("Henüz yeterli veri yok. Daha fazla sorgu işlendikten sonra öneriler sunulacak.")
                return insights
            
            # Performance insights
            for stats in chain_stats:
                if stats.success_rate < 0.95:
                    insights["performance_insights"].append(
                        f"{stats.chain_type} chain başarı oranı düşük (%{stats.success_rate * 100:.1f})"
                    )
                
                if stats.avg_execution_time > 10:
                    insights["performance_insights"].append(
                        f"{stats.chain_type} chain ortalama cevap süresi yüksek ({stats.avg_execution_time:.1f}s)"
                    )
            
            # Cost insights
            if "cost_by_chain" in cost_analysis:
                chain_costs = cost_analysis["cost_by_chain"]
                total_cost = sum(stats["total_cost"] for stats in chain_costs.values())
                
                for chain_type, cost_stats in chain_costs.items():
                    cost_ratio = cost_stats["total_cost"] / total_cost if total_cost > 0 else 0
                    if cost_ratio > 0.6:  # Single chain dominating costs
                        insights["cost_insights"].append(
                            f"{chain_type} chain toplam maliyetin %{cost_ratio * 100:.1f}'ini oluşturuyor"
                        )
            
            # Routing insights
            total_queries = sum(stats.total_queries for stats in chain_stats)
            if total_queries > 10:
                stuff_queries = next((s.total_queries for s in chain_stats if s.chain_type == "stuff"), 0)
                stuff_ratio = stuff_queries / total_queries
                
                if stuff_ratio < 0.4:
                    insights["routing_insights"].append("Stuff chain kullanımı düşük - routing algoritması optimize edilebilir")
                elif stuff_ratio > 0.8:
                    insights["routing_insights"].append("Stuff chain kullanımı çok yüksek - daha karmaşık chain'ler teşvik edilebilir")
            
            # Calculate optimization score (0-100)
            score_factors = {
                "success_rate": sum(s.success_rate for s in chain_stats) / len(chain_stats) if chain_stats else 0,
                "speed": 1.0 - min(1.0, max(s.avg_execution_time for s in chain_stats) / 20) if chain_stats else 0,
                "cost_efficiency": 1.0 - min(1.0, cost_analysis["overall_stats"]["avg_cost_per_query"] / 0.01) if "overall_stats" in cost_analysis else 0
            }
            
            insights["optimization_score"] = sum(score_factors.values()) / len(score_factors) * 100
            
            # Generate recommendations
            if insights["optimization_score"] < 70:
                insights["recommendations"].append("Sistem optimizasyonu gerekli. Detaylı analiz yapın.")
            elif insights["optimization_score"] > 90:
                insights["recommendations"].append("Sistem optimal performansta çalışıyor.")
            else:
                insights["recommendations"].append("Sistem iyi durumda, küçük optimizasyonlar yapılabilir.")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization insights: {e}")
            return {"error": str(e)}
    
    def export_performance_data(self, 
                               time_range_hours: int = 24,
                               format_type: str = "json") -> str:
        """
        Export performance data for analysis.
        
        Args:
            time_range_hours: Time range in hours to export
            format_type: Export format ("json" or "csv")
            
        Returns:
            Exported data as string
        """
        try:
            cutoff_time = (datetime.now() - timedelta(hours=time_range_hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM performance_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_time,))
                
                columns = [description[0] for description in cursor.description]
                data = cursor.fetchall()
            
            if format_type == "json":
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "time_range_hours": time_range_hours,
                    "total_records": len(data),
                    "data": [dict(zip(columns, row)) for row in data]
                }
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            
            elif format_type == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(columns)
                writer.writerows(data)
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to export performance data: {e}")
            return f"Export failed: {str(e)}"
    
    def reset_data(self, older_than_days: int = 30):
        """
        Clean up old performance data.
        
        Args:
            older_than_days: Delete data older than this many days
        """
        try:
            cutoff_time = (datetime.now() - timedelta(days=older_than_days)).isoformat()
            
            with self._db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM performance_metrics WHERE timestamp < ?", 
                        (cutoff_time,)
                    )
                    deleted_count = cursor.rowcount
            
            self.logger.info(f"Cleaned up {deleted_count} performance records older than {older_than_days} days")
            
        except Exception as e:
            self.logger.error(f"Failed to clean up performance data: {e}")