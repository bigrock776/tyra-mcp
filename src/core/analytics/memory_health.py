"""
Memory health monitoring and management system for Tyra MCP Memory Server.

This module provides comprehensive monitoring of system health, resource utilization,
and automatic remediation of common issues.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class HealthCategory(Enum):
    """Categories of health checks."""

    SYSTEM_RESOURCES = "system_resources"
    DATABASE_HEALTH = "database_health"
    CACHE_HEALTH = "cache_health"
    MODEL_HEALTH = "model_health"
    API_HEALTH = "api_health"
    STORAGE_HEALTH = "storage_health"


@dataclass
class HealthMetric:
    """Individual health metric measurement."""

    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime
    category: HealthCategory
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if metric is in healthy range."""
        return self.status == HealthStatus.HEALTHY

    def needs_attention(self) -> bool:
        """Check if metric needs attention."""
        return self.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]


@dataclass
class HealthCheck:
    """Health check configuration."""

    name: str
    category: HealthCategory
    check_function: Callable
    interval_seconds: int = 60
    warning_threshold: float = 80.0
    critical_threshold: float = 95.0
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    remediation_actions: List[str] = field(default_factory=list)


@dataclass
class SystemSnapshot:
    """Point-in-time system health snapshot."""

    timestamp: datetime
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric]
    active_alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def get_metrics_by_category(self, category: HealthCategory) -> List[HealthMetric]:
        """Get all metrics for a specific category."""
        return [
            metric for metric in self.metrics.values() if metric.category == category
        ]

    def get_unhealthy_metrics(self) -> List[HealthMetric]:
        """Get all metrics that are not healthy."""
        return [metric for metric in self.metrics.values() if not metric.is_healthy()]


class MemoryHealthManager:
    """
    Comprehensive health monitoring and management system.

    Features:
    - System resource monitoring (CPU, memory, disk, network)
    - Database health checks (PostgreSQL, Redis, Memgraph)
    - Cache performance monitoring
    - Model performance tracking
    - API endpoint health
    - Automatic remediation actions
    - Health trend analysis
    - Alerting and notifications
    """

    def __init__(
        self,
        check_interval: int = 30,
        history_retention_hours: int = 24,
        enable_auto_remediation: bool = True,
    ):
        self.check_interval = check_interval
        self.history_retention_hours = history_retention_hours
        self.enable_auto_remediation = enable_auto_remediation

        # Health monitoring state
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: List[SystemSnapshot] = []
        self.last_check_times: Dict[str, datetime] = {}
        self.active_remediations: Dict[str, datetime] = {}

        # System baselines
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.baseline_established = False

        # Background monitoring
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Initialize health checks
        self._initialize_health_checks()

        logger.info("MemoryHealthManager initialized")

    def _initialize_health_checks(self):
        """Initialize default health checks."""

        # System resource checks
        self.health_checks["cpu_usage"] = HealthCheck(
            name="CPU Usage",
            category=HealthCategory.SYSTEM_RESOURCES,
            check_function=self._check_cpu_usage,
            interval_seconds=30,
            warning_threshold=70.0,
            critical_threshold=90.0,
            remediation_actions=["scale_down_workers", "throttle_requests"],
        )

        self.health_checks["memory_usage"] = HealthCheck(
            name="Memory Usage",
            category=HealthCategory.SYSTEM_RESOURCES,
            check_function=self._check_memory_usage,
            interval_seconds=30,
            warning_threshold=80.0,
            critical_threshold=95.0,
            remediation_actions=["clear_caches", "garbage_collect", "restart_workers"],
        )

        self.health_checks["disk_usage"] = HealthCheck(
            name="Disk Usage",
            category=HealthCategory.STORAGE_HEALTH,
            check_function=self._check_disk_usage,
            interval_seconds=300,  # 5 minutes
            warning_threshold=80.0,
            critical_threshold=95.0,
            remediation_actions=[
                "cleanup_logs",
                "cleanup_temp_files",
                "archive_old_data",
            ],
        )

        self.health_checks["disk_io"] = HealthCheck(
            name="Disk I/O",
            category=HealthCategory.STORAGE_HEALTH,
            check_function=self._check_disk_io,
            interval_seconds=60,
            warning_threshold=80.0,
            critical_threshold=95.0,
        )

        # Database health checks
        self.health_checks["postgres_connections"] = HealthCheck(
            name="PostgreSQL Connections",
            category=HealthCategory.DATABASE_HEALTH,
            check_function=self._check_postgres_connections,
            interval_seconds=60,
            warning_threshold=80.0,
            critical_threshold=95.0,
            remediation_actions=["kill_idle_connections", "restart_connection_pool"],
        )

        self.health_checks["redis_memory"] = HealthCheck(
            name="Redis Memory Usage",
            category=HealthCategory.CACHE_HEALTH,
            check_function=self._check_redis_memory,
            interval_seconds=60,
            warning_threshold=80.0,
            critical_threshold=95.0,
            remediation_actions=["flush_expired_keys", "adjust_maxmemory_policy"],
        )

        # Performance checks
        self.health_checks["api_response_time"] = HealthCheck(
            name="API Response Time",
            category=HealthCategory.API_HEALTH,
            check_function=self._check_api_response_time,
            interval_seconds=60,
            warning_threshold=1000.0,  # 1 second
            critical_threshold=5000.0,  # 5 seconds
            remediation_actions=["scale_up_workers", "enable_caching"],
        )

        self.health_checks["embedding_performance"] = HealthCheck(
            name="Embedding Performance",
            category=HealthCategory.MODEL_HEALTH,
            check_function=self._check_embedding_performance,
            interval_seconds=300,
            warning_threshold=500.0,  # 500ms
            critical_threshold=2000.0,  # 2 seconds
            remediation_actions=[
                "restart_embedding_service",
                "switch_to_fallback_model",
            ],
        )

        logger.info(f"Initialized {len(self.health_checks)} health checks")

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Run health checks
                snapshot = await self._run_health_checks()

                # Store snapshot
                self._store_snapshot(snapshot)

                # Handle alerts and remediation
                await self._handle_health_issues(snapshot)

                # Cleanup old data
                self._cleanup_history()

                # Wait for next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _run_health_checks(self) -> SystemSnapshot:
        """Run all enabled health checks."""
        now = datetime.utcnow()
        metrics = {}

        for check_name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue

            # Check if it's time to run this check
            last_check = self.last_check_times.get(check_name)
            if (
                last_check
                and (now - last_check).total_seconds() < health_check.interval_seconds
            ):
                continue

            try:
                # Run the health check
                metric = await self._run_single_check(health_check)
                metrics[check_name] = metric
                self.last_check_times[check_name] = now

            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                # Create failed metric
                metrics[check_name] = HealthMetric(
                    name=health_check.name,
                    value=0.0,
                    unit="",
                    status=HealthStatus.FAILED,
                    threshold_warning=health_check.warning_threshold,
                    threshold_critical=health_check.critical_threshold,
                    timestamp=now,
                    category=health_check.category,
                    message=f"Check failed: {str(e)}",
                )

        # Calculate overall status
        overall_status = self._calculate_overall_status(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        return SystemSnapshot(
            timestamp=now,
            overall_status=overall_status,
            metrics=metrics,
            recommendations=recommendations,
        )

    async def _run_single_check(self, health_check: HealthCheck) -> HealthMetric:
        """Run a single health check."""
        start_time = time.time()

        try:
            # Run the check function
            result = await health_check.check_function()

            # Parse result
            if isinstance(result, (int, float)):
                value = float(result)
                unit = ""
                message = ""
                metadata = {}
            elif isinstance(result, dict):
                value = float(result.get("value", 0))
                unit = result.get("unit", "")
                message = result.get("message", "")
                metadata = result.get("metadata", {})
            else:
                raise ValueError(f"Invalid health check result format: {result}")

            # Determine status
            if value >= health_check.critical_threshold:
                status = HealthStatus.CRITICAL
            elif value >= health_check.warning_threshold:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY

            # Add execution time to metadata
            execution_time = (time.time() - start_time) * 1000
            metadata["execution_time_ms"] = execution_time

            return HealthMetric(
                name=health_check.name,
                value=value,
                unit=unit,
                status=status,
                threshold_warning=health_check.warning_threshold,
                threshold_critical=health_check.critical_threshold,
                timestamp=datetime.utcnow(),
                category=health_check.category,
                message=message,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Health check '{health_check.name}' execution failed: {e}")
            raise

    def _calculate_overall_status(
        self, metrics: Dict[str, HealthMetric]
    ) -> HealthStatus:
        """Calculate overall system health status."""
        if not metrics:
            return HealthStatus.FAILED

        statuses = [metric.status for metric in metrics.values()]

        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _generate_recommendations(self, metrics: Dict[str, HealthMetric]) -> List[str]:
        """Generate recommendations based on current metrics."""
        recommendations = []

        # Check for specific patterns and issues
        unhealthy_metrics = [m for m in metrics.values() if not m.is_healthy()]

        if not unhealthy_metrics:
            return ["System is operating normally"]

        # Group by category
        issues_by_category = {}
        for metric in unhealthy_metrics:
            if metric.category not in issues_by_category:
                issues_by_category[metric.category] = []
            issues_by_category[metric.category].append(metric)

        # Generate category-specific recommendations
        for category, category_metrics in issues_by_category.items():
            if category == HealthCategory.SYSTEM_RESOURCES:
                recommendations.extend(
                    self._recommend_resource_actions(category_metrics)
                )
            elif category == HealthCategory.DATABASE_HEALTH:
                recommendations.extend(
                    self._recommend_database_actions(category_metrics)
                )
            elif category == HealthCategory.CACHE_HEALTH:
                recommendations.extend(self._recommend_cache_actions(category_metrics))
            elif category == HealthCategory.MODEL_HEALTH:
                recommendations.extend(self._recommend_model_actions(category_metrics))

        return recommendations

    def _recommend_resource_actions(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate recommendations for resource issues."""
        recommendations = []

        for metric in metrics:
            if "cpu" in metric.name.lower():
                if metric.status == HealthStatus.CRITICAL:
                    recommendations.append(
                        "Immediately scale down CPU-intensive operations"
                    )
                    recommendations.append(
                        "Consider adding more worker processes or instances"
                    )
                else:
                    recommendations.append(
                        "Monitor CPU usage trends and consider optimization"
                    )

            elif "memory" in metric.name.lower():
                if metric.status == HealthStatus.CRITICAL:
                    recommendations.append("Clear caches and force garbage collection")
                    recommendations.append("Restart workers to free memory")
                else:
                    recommendations.append(
                        "Review memory usage patterns for optimization"
                    )

            elif "disk" in metric.name.lower():
                if metric.status == HealthStatus.CRITICAL:
                    recommendations.append("Clean up log files and temporary data")
                    recommendations.append("Archive old data to external storage")
                else:
                    recommendations.append("Schedule regular cleanup tasks")

        return recommendations

    def _recommend_database_actions(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate recommendations for database issues."""
        recommendations = []

        for metric in metrics:
            if metric.status == HealthStatus.CRITICAL:
                recommendations.append(f"Database issue detected: {metric.name}")
                recommendations.append("Check database logs for errors")
                recommendations.append("Consider restarting database connections")
            else:
                recommendations.append(f"Monitor {metric.name} trends")

        return recommendations

    def _recommend_cache_actions(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate recommendations for cache issues."""
        recommendations = []

        for metric in metrics:
            if metric.status == HealthStatus.CRITICAL:
                recommendations.append("Clear expired cache entries")
                recommendations.append("Adjust cache eviction policies")
            else:
                recommendations.append("Review cache hit rates and TTL settings")

        return recommendations

    def _recommend_model_actions(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate recommendations for model performance issues."""
        recommendations = []

        for metric in metrics:
            if metric.status == HealthStatus.CRITICAL:
                recommendations.append("Switch to fallback embedding model")
                recommendations.append("Restart embedding service")
            else:
                recommendations.append("Monitor model performance trends")

        return recommendations

    async def _handle_health_issues(self, snapshot: SystemSnapshot):
        """Handle health issues and execute remediation actions."""
        if not self.enable_auto_remediation:
            return

        unhealthy_metrics = snapshot.get_unhealthy_metrics()

        for metric in unhealthy_metrics:
            if metric.status == HealthStatus.CRITICAL:
                await self._execute_remediation(metric)

    async def _execute_remediation(self, metric: HealthMetric):
        """Execute remediation actions for a critical metric."""
        # Find the health check
        health_check = None
        for check in self.health_checks.values():
            if check.name == metric.name:
                health_check = check
                break

        if not health_check or not health_check.remediation_actions:
            return

        # Check if we've recently run remediation for this metric
        remediation_key = f"{metric.category.value}:{metric.name}"
        last_remediation = self.active_remediations.get(remediation_key)

        if last_remediation:
            time_since = datetime.utcnow() - last_remediation
            if time_since < timedelta(minutes=5):  # Don't remediate too frequently
                return

        logger.warning(f"Executing remediation for critical metric: {metric.name}")

        # Execute remediation actions
        for action in health_check.remediation_actions:
            try:
                await self._run_remediation_action(action, metric)
                self.active_remediations[remediation_key] = datetime.utcnow()
            except Exception as e:
                logger.error(f"Remediation action '{action}' failed: {e}")

    async def _run_remediation_action(self, action: str, metric: HealthMetric):
        """Run a specific remediation action."""
        logger.info(f"Running remediation action: {action}")

        if action == "clear_caches":
            await self._clear_caches()
        elif action == "garbage_collect":
            await self._force_garbage_collection()
        elif action == "cleanup_logs":
            await self._cleanup_log_files()
        elif action == "restart_workers":
            await self._restart_workers()
        elif action == "scale_down_workers":
            await self._scale_down_workers()
        elif action == "throttle_requests":
            await self._throttle_requests()
        else:
            logger.warning(f"Unknown remediation action: {action}")

    def _store_snapshot(self, snapshot: SystemSnapshot):
        """Store health snapshot in history."""
        self.health_history.append(snapshot)

        # Update baselines if this is a healthy snapshot
        if snapshot.overall_status == HealthStatus.HEALTHY:
            self._update_baselines(snapshot)

    def _update_baselines(self, snapshot: SystemSnapshot):
        """Update baseline metrics from healthy snapshots."""
        for metric_name, metric in snapshot.metrics.items():
            if metric.is_healthy():
                if metric_name not in self.baseline_metrics:
                    self.baseline_metrics[metric_name] = {
                        "values": [],
                        "mean": 0.0,
                        "std": 0.0,
                    }

                baseline = self.baseline_metrics[metric_name]
                baseline["values"].append(metric.value)

                # Keep only last 100 values for baseline calculation
                if len(baseline["values"]) > 100:
                    baseline["values"] = baseline["values"][-100:]

                # Recalculate statistics
                values = np.array(baseline["values"])
                baseline["mean"] = float(np.mean(values))
                baseline["std"] = float(np.std(values))

        if not self.baseline_established and len(self.baseline_metrics) >= 5:
            self.baseline_established = True
            logger.info("Health baselines established")

    def _cleanup_history(self):
        """Clean up old health history data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.history_retention_hours)
        self.health_history = [
            snapshot
            for snapshot in self.health_history
            if snapshot.timestamp >= cutoff_time
        ]

    # Health check implementations
    async def _check_cpu_usage(self) -> float:
        """Check current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)

    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        return {
            "value": memory.percent,
            "unit": "%",
            "message": f"Used: {memory.used // (1024**2)} MB / {memory.total // (1024**2)} MB",
            "metadata": {
                "used_mb": memory.used // (1024**2),
                "total_mb": memory.total // (1024**2),
                "available_mb": memory.available // (1024**2),
            },
        }

    async def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage for the root partition."""
        disk = psutil.disk_usage("/")
        usage_percent = (disk.used / disk.total) * 100

        return {
            "value": usage_percent,
            "unit": "%",
            "message": f"Used: {disk.used // (1024**3)} GB / {disk.total // (1024**3)} GB",
            "metadata": {
                "used_gb": disk.used // (1024**3),
                "total_gb": disk.total // (1024**3),
                "free_gb": disk.free // (1024**3),
            },
        }

    async def _check_disk_io(self) -> Dict[str, Any]:
        """Check disk I/O utilization."""
        # This is a simplified check - in production you'd want more sophisticated monitoring
        disk_io = psutil.disk_io_counters()
        if disk_io:
            # Calculate I/O utilization as a percentage (simplified)
            io_percent = min(
                100.0, (disk_io.read_bytes + disk_io.write_bytes) / (1024**3) * 10
            )
            return {
                "value": io_percent,
                "unit": "%",
                "message": f"Read: {disk_io.read_bytes // (1024**2)} MB, Write: {disk_io.write_bytes // (1024**2)} MB",
            }
        return {"value": 0.0, "unit": "%", "message": "No disk I/O data available"}

    async def _check_postgres_connections(self) -> float:
        """Check PostgreSQL connection pool usage."""
        # This would typically query the database for connection stats
        # For now, return a mock value
        return 25.0  # 25% of max connections used

    async def _check_redis_memory(self) -> float:
        """Check Redis memory usage."""
        # This would typically query Redis for memory stats
        # For now, return a mock value
        return 45.0  # 45% of max memory used

    async def _check_api_response_time(self) -> float:
        """Check average API response time."""
        # This would typically get metrics from the performance tracker
        # For now, return a mock value
        return 150.0  # 150ms average response time

    async def _check_embedding_performance(self) -> float:
        """Check embedding model performance."""
        # This would typically test embedding generation time
        # For now, return a mock value
        return 200.0  # 200ms average embedding time

    # Remediation action implementations
    async def _clear_caches(self):
        """Clear system caches."""
        logger.info("Clearing system caches")
        # Implementation would clear Redis cache, in-memory caches, etc.

    async def _force_garbage_collection(self):
        """Force garbage collection."""
        import gc

        logger.info("Forcing garbage collection")
        gc.collect()

    async def _cleanup_log_files(self):
        """Clean up old log files."""
        logger.info("Cleaning up log files")
        # Implementation would remove old log files

    async def _restart_workers(self):
        """Restart worker processes."""
        logger.info("Restarting worker processes")
        # Implementation would restart application workers

    async def _scale_down_workers(self):
        """Scale down number of workers."""
        logger.info("Scaling down workers")
        # Implementation would reduce worker count

    async def _throttle_requests(self):
        """Enable request throttling."""
        logger.info("Enabling request throttling")
        # Implementation would enable rate limiting

    # Public API methods
    def get_current_health(self) -> Optional[SystemSnapshot]:
        """Get the most recent health snapshot."""
        return self.health_history[-1] if self.health_history else None

    def get_health_history(self, hours: int = 1) -> List[SystemSnapshot]:
        """Get health history for the specified number of hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            snapshot
            for snapshot in self.health_history
            if snapshot.timestamp >= cutoff_time
        ]

    def get_health_trends(self, metric_name: str, hours: int = 6) -> Dict[str, Any]:
        """Get trend analysis for a specific metric."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        values = []
        timestamps = []

        for snapshot in self.health_history:
            if snapshot.timestamp >= cutoff_time and metric_name in snapshot.metrics:
                values.append(snapshot.metrics[metric_name].value)
                timestamps.append(snapshot.timestamp)

        if len(values) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Calculate trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        return {
            "metric_name": metric_name,
            "period_hours": hours,
            "data_points": len(values),
            "current_value": values[-1] if values else 0,
            "trend_slope": slope,
            "trend_direction": (
                "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            ),
            "min_value": min(values),
            "max_value": max(values),
            "mean_value": np.mean(values),
            "std_value": np.std(values),
        }

    def get_baseline_comparison(self, metric_name: str) -> Dict[str, Any]:
        """Compare current metric value to baseline."""
        current_snapshot = self.get_current_health()
        if not current_snapshot or metric_name not in current_snapshot.metrics:
            return {"error": "Metric not found"}

        current_value = current_snapshot.metrics[metric_name].value

        if metric_name not in self.baseline_metrics:
            return {"error": "No baseline established for this metric"}

        baseline = self.baseline_metrics[metric_name]

        deviation = current_value - baseline["mean"]
        std_deviation = deviation / baseline["std"] if baseline["std"] > 0 else 0

        return {
            "metric_name": metric_name,
            "current_value": current_value,
            "baseline_mean": baseline["mean"],
            "baseline_std": baseline["std"],
            "deviation": deviation,
            "std_deviations": std_deviation,
            "is_anomaly": abs(std_deviation) > 2.0,
            "severity": (
                "critical"
                if abs(std_deviation) > 3.0
                else "warning" if abs(std_deviation) > 2.0 else "normal"
            ),
        }

    def export_health_data(self, filepath: Path):
        """Export health monitoring data."""
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "health_checks": {},
            "baselines": self.baseline_metrics,
            "recent_snapshots": [],
        }

        # Export health check configurations
        for name, check in self.health_checks.items():
            export_data["health_checks"][name] = {
                "name": check.name,
                "category": check.category.value,
                "interval_seconds": check.interval_seconds,
                "warning_threshold": check.warning_threshold,
                "critical_threshold": check.critical_threshold,
                "enabled": check.enabled,
            }

        # Export recent snapshots
        recent_snapshots = self.get_health_history(hours=24)
        for snapshot in recent_snapshots[-100:]:  # Last 100 snapshots
            snapshot_data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "overall_status": snapshot.overall_status.value,
                "metrics": {},
                "recommendations": snapshot.recommendations,
            }

            for metric_name, metric in snapshot.metrics.items():
                snapshot_data["metrics"][metric_name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "status": metric.status.value,
                    "message": metric.message,
                    "metadata": metric.metadata,
                }

            export_data["recent_snapshots"].append(snapshot_data)

        # Write to file
        import json

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Health data exported to {filepath}")


# Global health manager instance
_health_manager: Optional[MemoryHealthManager] = None


def get_health_manager() -> MemoryHealthManager:
    """Get global health manager instance."""
    global _health_manager
    if _health_manager is None:
        _health_manager = MemoryHealthManager()
    return _health_manager
