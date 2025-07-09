"""
Comprehensive unit tests for Performance Tracker.

Tests performance metrics collection, analysis, trend detection, and improvement recommendations.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional

from src.core.analytics.performance_tracker import PerformanceTracker, MetricType, PerformanceMetric


class TestPerformanceTracker:
    """Test Performance Tracker functionality."""

    @pytest.fixture
    async def performance_tracker(self):
        """Create performance tracker with mocked dependencies."""
        with patch('src.core.analytics.performance_tracker.PostgreSQLClient') as mock_db:
            tracker = PerformanceTracker()
            tracker.db_client = AsyncMock()
            tracker.cache = AsyncMock()
            tracker._initialized = True
            
            yield tracker

    @pytest.mark.asyncio
    async def test_record_embedding_performance(self, performance_tracker):
        """Test recording embedding generation performance metrics."""
        # Setup
        metric = PerformanceMetric(
            operation_type="embedding_generation",
            latency_ms=45.2,
            model_used="intfloat/e5-large-v2",
            input_size=256,
            success=True,
            metadata={
                "text_length": 256,
                "batch_size": 1,
                "device": "cuda",
                "memory_usage_mb": 150
            }
        )
        
        # Mock database storage
        performance_tracker.db_client.store_metric.return_value = {"metric_id": "metric_123"}
        
        # Execute
        result = await performance_tracker.record_metric(metric)
        
        # Verify
        assert result["metric_id"] == "metric_123"
        performance_tracker.db_client.store_metric.assert_called_once()
        
        # Verify metric data
        call_args = performance_tracker.db_client.store_metric.call_args[0][0]
        assert call_args.operation_type == "embedding_generation"
        assert call_args.latency_ms == 45.2
        assert call_args.model_used == "intfloat/e5-large-v2"

    @pytest.mark.asyncio
    async def test_record_search_performance(self, performance_tracker):
        """Test recording search operation performance metrics."""
        # Setup
        metric = PerformanceMetric(
            operation_type="vector_search",
            latency_ms=28.7,
            success=True,
            metadata={
                "query_length": 128,
                "top_k": 10,
                "search_type": "hybrid",
                "results_found": 8,
                "cache_hit": False,
                "vector_dimension": 1024
            }
        )
        
        # Execute
        await performance_tracker.record_metric(metric)
        
        # Verify search-specific metrics are captured
        call_args = performance_tracker.db_client.store_metric.call_args[0][0]
        assert call_args.metadata["search_type"] == "hybrid"
        assert call_args.metadata["results_found"] == 8

    @pytest.mark.asyncio
    async def test_record_reranking_performance(self, performance_tracker):
        """Test recording reranking operation performance metrics."""
        # Setup
        metric = PerformanceMetric(
            operation_type="reranking",
            latency_ms=156.3,
            model_used="cross_encoder",
            success=True,
            metadata={
                "candidates_count": 20,
                "reranked_count": 10,
                "method": "cross_encoder",
                "batch_processing": True,
                "improvement_score": 0.15
            }
        )
        
        # Execute
        await performance_tracker.record_metric(metric)
        
        # Verify reranking metrics
        call_args = performance_tracker.db_client.store_metric.call_args[0][0]
        assert call_args.metadata["method"] == "cross_encoder"
        assert call_args.metadata["improvement_score"] == 0.15

    @pytest.mark.asyncio
    async def test_record_hallucination_detection_performance(self, performance_tracker):
        """Test recording hallucination detection performance metrics."""
        # Setup
        metric = PerformanceMetric(
            operation_type="hallucination_detection",
            latency_ms=89.4,
            success=True,
            metadata={
                "response_length": 512,
                "context_length": 1024,
                "confidence_score": 0.92,
                "confidence_level": "high",
                "grounding_score": 0.88,
                "trading_approved": False
            }
        )
        
        # Execute
        await performance_tracker.record_metric(metric)
        
        # Verify hallucination detection metrics
        call_args = performance_tracker.db_client.store_metric.call_args[0][0]
        assert call_args.metadata["confidence_level"] == "high"
        assert call_args.metadata["trading_approved"] is False

    @pytest.mark.asyncio
    async def test_analyze_latency_trends(self, performance_tracker):
        """Test latency trend analysis."""
        # Setup mock historical data
        mock_metrics = [
            {"timestamp": datetime.now() - timedelta(hours=i), "latency_ms": 50 + i*2, "operation_type": "embedding_generation"}
            for i in range(24)
        ]
        performance_tracker.db_client.get_metrics.return_value = mock_metrics
        
        # Execute
        analysis = await performance_tracker.analyze_latency_trends("embedding_generation", hours=24)
        
        # Verify trend analysis
        assert "trend_direction" in analysis
        assert "average_latency" in analysis
        assert "percentile_95" in analysis
        assert analysis["data_points"] == 24
        
        # Should detect increasing trend
        assert analysis["trend_direction"] in ["increasing", "stable", "decreasing"]

    @pytest.mark.asyncio
    async def test_detect_performance_degradation(self, performance_tracker):
        """Test performance degradation detection."""
        # Setup baseline and current metrics
        baseline_metrics = [{"latency_ms": 45.0} for _ in range(100)]
        current_metrics = [{"latency_ms": 95.0} for _ in range(20)]  # Significant degradation
        
        performance_tracker.db_client.get_metrics.side_effect = [baseline_metrics, current_metrics]
        
        # Execute
        degradation = await performance_tracker.detect_degradation("embedding_generation")
        
        # Verify degradation detection
        assert degradation["degradation_detected"] is True
        assert degradation["severity"] in ["low", "medium", "high"]
        assert degradation["performance_drop_percent"] > 50
        assert "recommendations" in degradation

    @pytest.mark.asyncio
    async def test_identify_performance_bottlenecks(self, performance_tracker):
        """Test bottleneck identification."""
        # Setup mock metrics for different operations
        mock_metrics = {
            "embedding_generation": [{"latency_ms": 200}, {"latency_ms": 210}],  # Slow
            "vector_search": [{"latency_ms": 30}, {"latency_ms": 35}],  # Fast
            "reranking": [{"latency_ms": 150}, {"latency_ms": 160}],  # Medium
            "hallucination_detection": [{"latency_ms": 80}, {"latency_ms": 90}]  # Medium
        }
        
        performance_tracker.db_client.get_metrics.side_effect = lambda op, **kwargs: mock_metrics.get(op, [])
        
        # Execute
        bottlenecks = await performance_tracker.identify_bottlenecks()
        
        # Verify bottleneck identification
        assert len(bottlenecks) > 0
        assert bottlenecks[0]["operation_type"] == "embedding_generation"  # Slowest
        assert bottlenecks[0]["avg_latency_ms"] > 100
        assert "optimization_suggestions" in bottlenecks[0]

    @pytest.mark.asyncio
    async def test_calculate_success_rates(self, performance_tracker):
        """Test success rate calculation."""
        # Setup mixed success/failure metrics
        mock_metrics = [
            {"success": True} for _ in range(85)
        ] + [
            {"success": False} for _ in range(15)
        ]
        performance_tracker.db_client.get_metrics.return_value = mock_metrics
        
        # Execute
        success_rates = await performance_tracker.calculate_success_rates("vector_search")
        
        # Verify success rate calculation
        assert success_rates["total_operations"] == 100
        assert success_rates["successful_operations"] == 85
        assert success_rates["failed_operations"] == 15
        assert success_rates["success_rate"] == 0.85

    @pytest.mark.asyncio
    async def test_analyze_cache_performance(self, performance_tracker):
        """Test cache performance analysis."""
        # Setup cache metrics
        mock_cache_metrics = [
            {"operation": "embedding_cache", "hit": True} for _ in range(80)
        ] + [
            {"operation": "embedding_cache", "hit": False} for _ in range(20)
        ]
        performance_tracker.db_client.get_cache_metrics.return_value = mock_cache_metrics
        
        # Execute
        cache_analysis = await performance_tracker.analyze_cache_performance()
        
        # Verify cache analysis
        assert cache_analysis["overall_hit_rate"] == 0.8
        assert "cache_types" in cache_analysis
        assert cache_analysis["total_cache_operations"] == 100
        assert "recommendations" in cache_analysis

    @pytest.mark.asyncio
    async def test_generate_optimization_recommendations(self, performance_tracker):
        """Test optimization recommendation generation."""
        # Setup performance data indicating issues
        performance_data = {
            "embedding_generation": {
                "avg_latency_ms": 150,
                "p95_latency_ms": 250,
                "success_rate": 0.95,
                "trending": "increasing"
            },
            "vector_search": {
                "avg_latency_ms": 45,
                "p95_latency_ms": 80,
                "success_rate": 0.98,
                "trending": "stable"
            }
        }
        
        # Execute
        recommendations = await performance_tracker.generate_recommendations(performance_data)
        
        # Verify recommendations
        assert len(recommendations) > 0
        assert any("embedding" in rec["description"].lower() for rec in recommendations)
        assert all("priority" in rec for rec in recommendations)
        assert all("impact" in rec for rec in recommendations)

    @pytest.mark.asyncio
    async def test_track_model_performance_comparison(self, performance_tracker):
        """Test model performance comparison tracking."""
        # Setup metrics for different models
        model_metrics = {
            "intfloat/e5-large-v2": [{"latency_ms": 45, "accuracy": 0.92}] * 50,
            "all-MiniLM-L12-v2": [{"latency_ms": 25, "accuracy": 0.87}] * 50
        }
        
        performance_tracker.db_client.get_metrics_by_model.side_effect = lambda model: model_metrics.get(model, [])
        
        # Execute
        comparison = await performance_tracker.compare_model_performance(
            ["intfloat/e5-large-v2", "all-MiniLM-L12-v2"]
        )
        
        # Verify comparison
        assert len(comparison["models"]) == 2
        assert comparison["models"][0]["model_name"] in model_metrics
        assert comparison["models"][1]["model_name"] in model_metrics
        assert "recommendation" in comparison

    @pytest.mark.asyncio
    async def test_monitor_resource_utilization(self, performance_tracker):
        """Test resource utilization monitoring."""
        # Setup resource metrics
        resource_data = {
            "cpu_usage_percent": 75.5,
            "memory_usage_gb": 8.2,
            "gpu_memory_usage_gb": 6.1,
            "disk_io_mbps": 125.3,
            "network_io_mbps": 45.7
        }
        
        performance_tracker._collect_system_resources = AsyncMock(return_value=resource_data)
        
        # Execute
        resource_metrics = await performance_tracker.monitor_resources()
        
        # Verify resource monitoring
        assert resource_metrics["cpu_usage_percent"] == 75.5
        assert resource_metrics["memory_usage_gb"] == 8.2
        assert "alert_level" in resource_metrics
        assert "recommendations" in resource_metrics

    @pytest.mark.asyncio
    async def test_track_user_feedback_correlation(self, performance_tracker):
        """Test correlation between performance and user feedback."""
        # Setup performance and feedback data
        performance_metrics = [{"latency_ms": 50, "confidence": 0.95, "timestamp": datetime.now()}] * 20
        user_feedback = [{"rating": 4.5, "timestamp": datetime.now()}] * 20
        
        performance_tracker.db_client.get_metrics.return_value = performance_metrics
        performance_tracker.db_client.get_user_feedback.return_value = user_feedback
        
        # Execute
        correlation = await performance_tracker.analyze_feedback_correlation()
        
        # Verify correlation analysis
        assert "latency_feedback_correlation" in correlation
        assert "confidence_feedback_correlation" in correlation
        assert "insights" in correlation

    @pytest.mark.asyncio
    async def test_performance_alerting(self, performance_tracker):
        """Test performance alerting system."""
        # Setup alert thresholds
        alert_config = {
            "latency_threshold_ms": 100,
            "success_rate_threshold": 0.95,
            "degradation_threshold_percent": 20
        }
        performance_tracker.alert_config = alert_config
        
        # Setup metrics that should trigger alerts
        problematic_metrics = [
            {"latency_ms": 150, "success": True},  # High latency
            {"latency_ms": 120, "success": False},  # High latency + failure
        ]
        performance_tracker.db_client.get_recent_metrics.return_value = problematic_metrics
        
        # Execute
        alerts = await performance_tracker.check_for_alerts()
        
        # Verify alerts
        assert len(alerts) > 0
        assert any("latency" in alert["message"].lower() for alert in alerts)
        assert all("severity" in alert for alert in alerts)

    @pytest.mark.asyncio
    async def test_performance_dashboard_data(self, performance_tracker):
        """Test data preparation for performance dashboard."""
        # Setup comprehensive metrics
        mock_dashboard_data = {
            "overview": {
                "total_operations": 10000,
                "avg_latency_ms": 67.5,
                "success_rate": 0.97,
                "operations_per_hour": 850
            },
            "by_operation": {
                "embedding_generation": {"count": 4000, "avg_latency": 45},
                "vector_search": {"count": 3500, "avg_latency": 30},
                "reranking": {"count": 2000, "avg_latency": 120},
                "hallucination_detection": {"count": 500, "avg_latency": 85}
            },
            "trends": {
                "latency_trend": "stable",
                "success_rate_trend": "improving",
                "volume_trend": "increasing"
            }
        }
        
        performance_tracker._generate_dashboard_data = AsyncMock(return_value=mock_dashboard_data)
        
        # Execute
        dashboard_data = await performance_tracker.get_dashboard_data()
        
        # Verify dashboard data
        assert "overview" in dashboard_data
        assert "by_operation" in dashboard_data
        assert "trends" in dashboard_data
        assert dashboard_data["overview"]["total_operations"] == 10000

    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, performance_tracker):
        """Test detection of performance regressions."""
        # Setup historical baseline
        baseline_period = [
            {"latency_ms": 45, "timestamp": datetime.now() - timedelta(days=7)}
            for _ in range(100)
        ]
        
        # Recent period with regression
        recent_period = [
            {"latency_ms": 85, "timestamp": datetime.now() - timedelta(hours=1)}
            for _ in range(50)
        ]
        
        performance_tracker.db_client.get_metrics.side_effect = [baseline_period, recent_period]
        
        # Execute
        regression = await performance_tracker.detect_regression("embedding_generation")
        
        # Verify regression detection
        assert regression["regression_detected"] is True
        assert regression["performance_change_percent"] > 50
        assert regression["statistical_significance"] > 0.95
        assert "contributing_factors" in regression

    @pytest.mark.asyncio
    async def test_custom_metric_tracking(self, performance_tracker):
        """Test tracking of custom business metrics."""
        # Setup custom metric
        custom_metric = PerformanceMetric(
            operation_type="trading_decision",
            success=True,
            metadata={
                "confidence_score": 0.97,
                "decision_type": "buy",
                "portfolio_impact": 0.05,
                "execution_time_ms": 245,
                "safety_checks_passed": True
            }
        )
        
        # Execute
        await performance_tracker.record_metric(custom_metric)
        
        # Verify custom metric handling
        call_args = performance_tracker.db_client.store_metric.call_args[0][0]
        assert call_args.operation_type == "trading_decision"
        assert call_args.metadata["confidence_score"] == 0.97
        assert call_args.metadata["safety_checks_passed"] is True

    @pytest.mark.asyncio
    async def test_performance_export_functionality(self, performance_tracker):
        """Test exporting performance data for analysis."""
        # Setup export request
        export_request = {
            "start_date": datetime.now() - timedelta(days=30),
            "end_date": datetime.now(),
            "operations": ["embedding_generation", "vector_search"],
            "format": "csv",
            "include_metadata": True
        }
        
        # Mock export data
        mock_export_data = [
            {
                "timestamp": datetime.now(),
                "operation_type": "embedding_generation",
                "latency_ms": 45,
                "success": True,
                "model_used": "intfloat/e5-large-v2"
            }
        ] * 1000
        
        performance_tracker.db_client.export_metrics.return_value = mock_export_data
        
        # Execute
        export_result = await performance_tracker.export_performance_data(export_request)
        
        # Verify export
        assert export_result["record_count"] == 1000
        assert export_result["format"] == "csv"
        assert "file_path" in export_result

    @pytest.mark.asyncio
    async def test_health_check(self, performance_tracker):
        """Test performance tracker health check."""
        # Setup health check
        performance_tracker.db_client.ping.return_value = True
        performance_tracker.cache.ping.return_value = True
        
        # Execute
        health = await performance_tracker.health_check()
        
        # Verify
        assert health["status"] == "healthy"
        assert health["components"]["database"] == "healthy"
        assert health["components"]["cache"] == "healthy"
        assert "metrics_count_24h" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])