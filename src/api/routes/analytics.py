"""
Analytics and Performance Dashboard API Routes.

Provides endpoints for performance monitoring, analytics dashboards,
and self-learning system insights.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from src.core.analytics.performance_tracker import PerformanceTracker, MetricType
from src.core.adaptation.memory_health import MemoryHealthManager
from src.core.adaptation.config_optimizer import ConfigOptimizer
from src.core.utils.auth import get_current_user_optional
from src.core.utils.logger import get_logger
from src.core.utils.rate_limiter import RateLimiter

router = APIRouter()
rate_limiter = RateLimiter()
logger = get_logger(__name__)


# Request/Response Models
class DashboardQuery(BaseModel):
    """Query parameters for dashboard data."""
    time_range: str = Field("24h", description="Time range: 1h, 6h, 24h, 7d, 30d")
    operations: Optional[List[str]] = Field(None, description="Filter by operation types")
    agents: Optional[List[str]] = Field(None, description="Filter by agent IDs")
    include_trends: bool = Field(True, description="Include trend analysis")
    include_predictions: bool = Field(False, description="Include performance predictions")


class PerformanceOverview(BaseModel):
    """Performance overview response."""
    timestamp: datetime
    total_operations: int
    avg_latency_ms: float
    success_rate: float
    operations_per_hour: int
    cache_hit_rate: float
    error_count: int
    active_agents: int


class TrendData(BaseModel):
    """Trend analysis data."""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    change_percent: float
    confidence: float
    data_points: List[Dict[str, Any]]


class AlertSummary(BaseModel):
    """Alert summary information."""
    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    recent_alerts: List[Dict[str, Any]]


class DashboardResponse(BaseModel):
    """Main dashboard response."""
    overview: PerformanceOverview
    by_operation: Dict[str, Dict[str, Any]]
    by_agent: Dict[str, Dict[str, Any]]
    trends: Optional[List[TrendData]] = None
    alerts: AlertSummary
    recommendations: List[str]
    last_updated: datetime


@router.get("/dashboard", response_model=DashboardResponse)
async def get_performance_dashboard(
    query: DashboardQuery = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Optional[str] = Depends(get_current_user_optional)
):
    """
    Get comprehensive performance dashboard data.
    
    Provides real-time performance metrics, trends, and recommendations
    for system monitoring and optimization.
    """
    try:
        # Initialize components
        performance_tracker = PerformanceTracker()
        memory_health = MemoryHealthManager()
        
        # Parse time range
        time_ranges = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6), 
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        if query.time_range not in time_ranges:
            raise HTTPException(status_code=400, detail=f"Invalid time range: {query.time_range}")
        
        time_delta = time_ranges[query.time_range]
        start_time = datetime.utcnow() - time_delta
        
        # Get performance overview
        overview_data = await performance_tracker.get_overview_metrics(
            start_time=start_time,
            operations=query.operations,
            agents=query.agents
        )
        
        overview = PerformanceOverview(
            timestamp=datetime.utcnow(),
            total_operations=overview_data["total_operations"],
            avg_latency_ms=overview_data["avg_latency_ms"],
            success_rate=overview_data["success_rate"],
            operations_per_hour=overview_data["operations_per_hour"],
            cache_hit_rate=overview_data["cache_hit_rate"],
            error_count=overview_data["error_count"],
            active_agents=overview_data["active_agents"]
        )
        
        # Get performance by operation type
        by_operation = await performance_tracker.get_metrics_by_operation(
            start_time=start_time,
            operations=query.operations
        )
        
        # Get performance by agent
        by_agent = await performance_tracker.get_metrics_by_agent(
            start_time=start_time,
            agents=query.agents
        )
        
        # Get trends if requested
        trends = None
        if query.include_trends:
            trend_metrics = await performance_tracker.analyze_trends(
                start_time=start_time,
                metrics=["latency", "success_rate", "throughput"]
            )
            trends = [
                TrendData(
                    metric_name=trend["metric"],
                    trend_direction=trend["direction"],
                    change_percent=trend["change_percent"],
                    confidence=trend["confidence"],
                    data_points=trend["data_points"]
                )
                for trend in trend_metrics
            ]
        
        # Get alerts
        alerts_data = await performance_tracker.get_recent_alerts(
            start_time=start_time
        )
        
        alerts = AlertSummary(
            total_alerts=alerts_data["total"],
            critical_alerts=alerts_data["critical"],
            warning_alerts=alerts_data["warning"],
            recent_alerts=alerts_data["recent"][:10]  # Last 10 alerts
        )
        
        # Get recommendations
        recommendations = await performance_tracker.get_optimization_recommendations()
        
        # Schedule background analysis update
        background_tasks.add_task(_update_analytics_cache, query.time_range)
        
        return DashboardResponse(
            overview=overview,
            by_operation=by_operation,
            by_agent=by_agent,
            trends=trends,
            alerts=alerts,
            recommendations=recommendations,
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


@router.get("/metrics/{metric_type}")
async def get_metric_details(
    metric_type: str,
    start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
    end_time: Optional[datetime] = Query(None, description="End time for metrics"),
    granularity: str = Query("hour", description="Data granularity: minute, hour, day"),
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID")
):
    """
    Get detailed metrics for a specific metric type.
    
    Provides granular time-series data for specific performance metrics.
    """
    try:
        performance_tracker = PerformanceTracker()
        
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        # Get detailed metrics
        metrics_data = await performance_tracker.get_detailed_metrics(
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            operation_type=operation_type,
            agent_id=agent_id
        )
        
        return {
            "metric_type": metric_type,
            "start_time": start_time,
            "end_time": end_time,
            "granularity": granularity,
            "data_points": metrics_data["data_points"],
            "statistics": metrics_data["statistics"],
            "anomalies": metrics_data.get("anomalies", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to get metric details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metric details")


@router.get("/health-score")
async def get_system_health_score():
    """
    Get overall system health score.
    
    Provides a comprehensive health assessment based on multiple factors.
    """
    try:
        performance_tracker = PerformanceTracker()
        memory_health = MemoryHealthManager()
        
        # Get component health scores
        performance_score = await performance_tracker.calculate_health_score()
        memory_score = await memory_health.calculate_health_score()
        
        # Calculate overall score
        overall_score = (performance_score + memory_score) / 2
        
        # Determine health status
        if overall_score >= 0.9:
            status = "excellent"
        elif overall_score >= 0.8:
            status = "good"
        elif overall_score >= 0.7:
            status = "fair"
        elif overall_score >= 0.6:
            status = "poor"
        else:
            status = "critical"
        
        return {
            "overall_score": overall_score,
            "status": status,
            "component_scores": {
                "performance": performance_score,
                "memory_health": memory_score
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate health score: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate health score")


@router.get("/recommendations")
async def get_optimization_recommendations(
    category: Optional[str] = Query(None, description="Filter by category"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(10, description="Maximum number of recommendations")
):
    """
    Get system optimization recommendations.
    
    Provides actionable recommendations for improving system performance.
    """
    try:
        performance_tracker = PerformanceTracker()
        config_optimizer = ConfigOptimizer()
        
        # Get performance-based recommendations
        perf_recommendations = await performance_tracker.get_optimization_recommendations()
        
        # Get configuration-based recommendations
        config_recommendations = await config_optimizer.get_recommendations()
        
        # Combine and prioritize recommendations
        all_recommendations = perf_recommendations + config_recommendations
        
        # Filter by category if specified
        if category:
            all_recommendations = [
                r for r in all_recommendations 
                if r.get("category", "").lower() == category.lower()
            ]
        
        # Filter by priority if specified
        if priority:
            all_recommendations = [
                r for r in all_recommendations 
                if r.get("priority", "").lower() == priority.lower()
            ]
        
        # Sort by priority and impact
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_recommendations.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 3),
                -x.get("impact_score", 0)
            )
        )
        
        return {
            "recommendations": all_recommendations[:limit],
            "total_available": len(all_recommendations),
            "categories": list(set(r.get("category") for r in all_recommendations if r.get("category"))),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


@router.get("/trends/{operation_type}")
async def get_operation_trends(
    operation_type: str,
    days: int = Query(7, description="Number of days to analyze"),
    include_predictions: bool = Query(False, description="Include trend predictions")
):
    """
    Get detailed trend analysis for a specific operation type.
    
    Provides trend analysis and optional predictions for operation performance.
    """
    try:
        performance_tracker = PerformanceTracker()
        
        start_time = datetime.utcnow() - timedelta(days=days)
        
        # Get trend analysis
        trends = await performance_tracker.analyze_operation_trends(
            operation_type=operation_type,
            start_time=start_time,
            include_predictions=include_predictions
        )
        
        return {
            "operation_type": operation_type,
            "analysis_period_days": days,
            "trends": trends,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get operation trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get operation trends")


@router.get("/alerts")
async def get_performance_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    start_time: Optional[datetime] = Query(None, description="Start time for alerts"),
    limit: int = Query(50, description="Maximum number of alerts")
):
    """
    Get performance alerts and notifications.
    
    Provides recent alerts about system performance issues.
    """
    try:
        performance_tracker = PerformanceTracker()
        
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        
        alerts = await performance_tracker.get_alerts(
            start_time=start_time,
            severity=severity,
            limit=limit
        )
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "severity_counts": {
                "critical": len([a for a in alerts if a.get("severity") == "critical"]),
                "warning": len([a for a in alerts if a.get("severity") == "warning"]),
                "info": len([a for a in alerts if a.get("severity") == "info"])
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")


@router.post("/analyze")
async def trigger_manual_analysis(
    background_tasks: BackgroundTasks,
    analysis_type: str = Query("full", description="Type of analysis: full, quick, trends"),
    time_range: str = Query("24h", description="Time range for analysis")
):
    """
    Trigger manual performance analysis.
    
    Initiates a comprehensive analysis of system performance.
    """
    try:
        # Schedule analysis in background
        background_tasks.add_task(
            _run_performance_analysis,
            analysis_type,
            time_range
        )
        
        return {
            "message": f"Performance analysis ({analysis_type}) started",
            "analysis_type": analysis_type,
            "time_range": time_range,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger analysis")


@router.get("/export")
async def export_analytics_data(
    start_time: datetime = Query(..., description="Export start time"),
    end_time: datetime = Query(..., description="Export end time"),
    format: str = Query("json", description="Export format: json, csv"),
    include_raw_data: bool = Query(False, description="Include raw metric data")
):
    """
    Export analytics data for external analysis.
    
    Provides data export functionality for reporting and analysis.
    """
    try:
        performance_tracker = PerformanceTracker()
        
        export_data = await performance_tracker.export_data(
            start_time=start_time,
            end_time=end_time,
            format=format,
            include_raw_data=include_raw_data
        )
        
        return {
            "export_id": export_data["export_id"],
            "file_path": export_data["file_path"],
            "format": format,
            "record_count": export_data["record_count"],
            "file_size_bytes": export_data["file_size"],
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export analytics data")


# Background task functions
async def _update_analytics_cache(time_range: str):
    """Update analytics cache in background."""
    try:
        performance_tracker = PerformanceTracker()
        await performance_tracker.update_cache(time_range)
        logger.info(f"Analytics cache updated for {time_range}")
    except Exception as e:
        logger.error(f"Failed to update analytics cache: {e}")


async def _run_performance_analysis(analysis_type: str, time_range: str):
    """Run performance analysis in background."""
    try:
        performance_tracker = PerformanceTracker()
        
        if analysis_type == "full":
            await performance_tracker.run_full_analysis(time_range)
        elif analysis_type == "quick":
            await performance_tracker.run_quick_analysis(time_range)
        elif analysis_type == "trends":
            await performance_tracker.run_trend_analysis(time_range)
        
        logger.info(f"Performance analysis ({analysis_type}) completed")
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")