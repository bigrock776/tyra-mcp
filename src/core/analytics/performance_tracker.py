"""
Performance analytics and tracking system for Tyra MCP Memory Server.

This module provides comprehensive performance monitoring, anomaly detection,
and optimization recommendations for the memory system.
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    EMBEDDING_TIME = "embedding_time"
    SEARCH_TIME = "search_time"
    RERANK_TIME = "rerank_time"
    HALLUCINATION_SCORE = "hallucination_score"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceEvent:
    """Single performance measurement event."""

    timestamp: datetime
    metric_type: MetricType
    value: float
    operation: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "value": self.value,
            "operation": self.operation,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


@dataclass
class PerformancePattern:
    """Detected performance pattern."""

    pattern_id: str
    description: str
    metric_types: List[MetricType]
    confidence: float
    impact_score: float
    first_detected: datetime
    last_seen: datetime
    occurrence_count: int = 0
    related_operations: Set[str] = field(default_factory=set)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""

    metric_type: MetricType
    operation: str
    mean: float
    std_dev: float
    p50: float
    p95: float
    p99: float
    sample_count: int
    established_at: datetime
    last_updated: datetime

    def is_anomaly(self, value: float, threshold_std_dev: float = 2.0) -> bool:
        """Check if value is anomalous compared to baseline."""
        return abs(value - self.mean) > (threshold_std_dev * self.std_dev)

    def get_percentile_rank(self, value: float) -> float:
        """Get percentile rank of value compared to baseline."""
        if value <= self.p50:
            return min(50.0, max(0.0, 50.0 * (value / self.p50)))
        elif value <= self.p95:
            return 50.0 + 45.0 * ((value - self.p50) / (self.p95 - self.p50))
        elif value <= self.p99:
            return 95.0 + 4.0 * ((value - self.p95) / (self.p99 - self.p95))
        else:
            return min(100.0, 99.0 + (value - self.p99) / self.p99)


class PerformanceTracker:
    """
    Comprehensive performance tracking and analytics system.

    Features:
    - Real-time performance event collection
    - Baseline establishment and anomaly detection
    - Pattern recognition and trend analysis
    - Automated alerting and recommendations
    - Performance optimization suggestions
    """

    def __init__(
        self,
        baseline_window_size: int = 1000,
        anomaly_threshold: float = 2.0,
        pattern_confidence_threshold: float = 0.7,
        cache_size: int = 10000,
    ):
        self.baseline_window_size = baseline_window_size
        self.anomaly_threshold = anomaly_threshold
        self.pattern_confidence_threshold = pattern_confidence_threshold

        # Event storage
        self.recent_events: deque = deque(maxlen=cache_size)
        self.events_by_operation: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=cache_size // 10)
        )

        # Baselines and patterns
        self.baselines: Dict[Tuple[MetricType, str], PerformanceBaseline] = {}
        self.patterns: Dict[str, PerformancePattern] = {}

        # Analytics state
        self.anomaly_counts: Dict[str, int] = defaultdict(int)
        self.last_baseline_update: Dict[str, datetime] = {}
        self.alert_history: deque = deque(maxlen=1000)

        # Configuration
        self.update_baseline_interval = timedelta(hours=1)
        self.pattern_detection_interval = timedelta(minutes=15)
        self.min_baseline_samples = 100

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

        logger.info("PerformanceTracker initialized")

    async def start(self):
        """Start background analytics tasks."""
        if self._running:
            return

        self._running = True
        self._tasks = [
            asyncio.create_task(self._baseline_updater()),
            asyncio.create_task(self._pattern_detector()),
            asyncio.create_task(self._anomaly_alerter()),
        ]
        logger.info("PerformanceTracker background tasks started")

    async def stop(self):
        """Stop background analytics tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("PerformanceTracker stopped")

    def record_event(
        self,
        metric_type: MetricType,
        value: float,
        operation: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata,
    ):
        """Record a performance event."""
        event = PerformanceEvent(
            timestamp=datetime.utcnow(),
            metric_type=metric_type,
            value=value,
            operation=operation,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
        )

        self.recent_events.append(event)
        self.events_by_operation[operation].append(event)

        # Check for immediate anomalies
        baseline_key = (metric_type, operation)
        if baseline_key in self.baselines:
            baseline = self.baselines[baseline_key]
            if baseline.is_anomaly(value, self.anomaly_threshold):
                self._handle_anomaly(event, baseline)

    def _handle_anomaly(self, event: PerformanceEvent, baseline: PerformanceBaseline):
        """Handle detected anomaly."""
        anomaly_key = f"{event.metric_type.value}:{event.operation}"
        self.anomaly_counts[anomaly_key] += 1

        percentile = baseline.get_percentile_rank(event.value)
        severity = self._calculate_anomaly_severity(event, baseline, percentile)

        alert = {
            "timestamp": event.timestamp.isoformat(),
            "type": "anomaly",
            "severity": severity.value,
            "metric_type": event.metric_type.value,
            "operation": event.operation,
            "value": event.value,
            "baseline_mean": baseline.mean,
            "percentile_rank": percentile,
            "deviation_std": abs(event.value - baseline.mean) / baseline.std_dev,
        }

        self.alert_history.append(alert)
        logger.warning(f"Performance anomaly detected: {alert}")

    def _calculate_anomaly_severity(
        self, event: PerformanceEvent, baseline: PerformanceBaseline, percentile: float
    ) -> AlertLevel:
        """Calculate anomaly severity level."""
        deviation = abs(event.value - baseline.mean) / baseline.std_dev

        if deviation > 4.0 or percentile > 99.5:
            return AlertLevel.CRITICAL
        elif deviation > 3.0 or percentile > 95.0:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO

    async def _baseline_updater(self):
        """Background task to update performance baselines."""
        while self._running:
            try:
                await self._update_baselines()
                await asyncio.sleep(self.update_baseline_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in baseline updater: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _update_baselines(self):
        """Update performance baselines from recent data."""
        operations_to_update = set()

        # Find operations that need baseline updates
        for operation, events in self.events_by_operation.items():
            if len(events) < self.min_baseline_samples:
                continue

            last_update = self.last_baseline_update.get(operation)
            if (
                not last_update
                or datetime.utcnow() - last_update > self.update_baseline_interval
            ):
                operations_to_update.add(operation)

        # Update baselines
        for operation in operations_to_update:
            await self._update_operation_baselines(operation)
            self.last_baseline_update[operation] = datetime.utcnow()

        if operations_to_update:
            logger.info(f"Updated baselines for {len(operations_to_update)} operations")

    async def _update_operation_baselines(self, operation: str):
        """Update baselines for a specific operation."""
        events = list(self.events_by_operation[operation])
        if len(events) < self.min_baseline_samples:
            return

        # Group events by metric type
        metrics_data: Dict[MetricType, List[float]] = defaultdict(list)
        for event in events:
            metrics_data[event.metric_type].append(event.value)

        # Calculate baselines for each metric type
        for metric_type, values in metrics_data.items():
            if len(values) < self.min_baseline_samples:
                continue

            baseline = self._calculate_baseline(metric_type, operation, values)
            self.baselines[(metric_type, operation)] = baseline

    def _calculate_baseline(
        self, metric_type: MetricType, operation: str, values: List[float]
    ) -> PerformanceBaseline:
        """Calculate baseline statistics from values."""
        values_array = np.array(values)

        return PerformanceBaseline(
            metric_type=metric_type,
            operation=operation,
            mean=float(np.mean(values_array)),
            std_dev=float(np.std(values_array)),
            p50=float(np.percentile(values_array, 50)),
            p95=float(np.percentile(values_array, 95)),
            p99=float(np.percentile(values_array, 99)),
            sample_count=len(values),
            established_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
        )

    async def _pattern_detector(self):
        """Background task to detect performance patterns."""
        while self._running:
            try:
                await self._detect_patterns()
                await asyncio.sleep(self.pattern_detection_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in pattern detector: {e}")
                await asyncio.sleep(60)

    async def _detect_patterns(self):
        """Detect performance patterns in recent data."""
        # Correlation patterns
        await self._detect_correlation_patterns()

        # Trend patterns
        await self._detect_trend_patterns()

        # Outlier cluster patterns
        await self._detect_outlier_patterns()

    async def _detect_correlation_patterns(self):
        """Detect correlations between different metrics."""
        # Get recent events grouped by time windows
        time_windows = self._group_events_by_time_window(minutes=5)

        for window_events in time_windows:
            if len(window_events) < 10:
                continue

            # Calculate correlations between metric types
            metric_correlations = self._calculate_metric_correlations(window_events)

            # Identify strong correlations
            for (metric1, metric2), correlation in metric_correlations.items():
                if abs(correlation) > 0.8:  # Strong correlation threshold
                    await self._record_correlation_pattern(
                        metric1, metric2, correlation, window_events
                    )

    def _group_events_by_time_window(
        self, minutes: int = 5
    ) -> List[List[PerformanceEvent]]:
        """Group recent events into time windows."""
        if not self.recent_events:
            return []

        window_size = timedelta(minutes=minutes)
        windows = []
        current_window = []
        current_window_start = None

        for event in self.recent_events:
            if (
                current_window_start is None
                or event.timestamp - current_window_start < window_size
            ):
                if current_window_start is None:
                    current_window_start = event.timestamp
                current_window.append(event)
            else:
                if current_window:
                    windows.append(current_window)
                current_window = [event]
                current_window_start = event.timestamp

        if current_window:
            windows.append(current_window)

        return windows

    def _calculate_metric_correlations(
        self, events: List[PerformanceEvent]
    ) -> Dict[Tuple[MetricType, MetricType], float]:
        """Calculate correlations between metric types in event list."""
        correlations = {}

        # Group events by metric type
        metric_values: Dict[MetricType, List[float]] = defaultdict(list)
        for event in events:
            metric_values[event.metric_type].append(event.value)

        # Calculate correlations between all pairs
        metric_types = list(metric_values.keys())
        for i, metric1 in enumerate(metric_types):
            for metric2 in metric_types[i + 1 :]:
                values1 = metric_values[metric1]
                values2 = metric_values[metric2]

                if len(values1) >= 5 and len(values2) >= 5:
                    # Align values by timestamp (simplified)
                    min_len = min(len(values1), len(values2))
                    corr = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations[(metric1, metric2)] = corr

        return correlations

    async def _record_correlation_pattern(
        self,
        metric1: MetricType,
        metric2: MetricType,
        correlation: float,
        events: List[PerformanceEvent],
    ):
        """Record a detected correlation pattern."""
        pattern_id = f"correlation_{metric1.value}_{metric2.value}"

        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.last_seen = datetime.utcnow()
            pattern.occurrence_count += 1
            pattern.confidence = min(1.0, pattern.confidence + 0.1)
        else:
            operations = {event.operation for event in events}
            recommendations = self._generate_correlation_recommendations(
                metric1, metric2, correlation
            )

            pattern = PerformancePattern(
                pattern_id=pattern_id,
                description=f"Strong correlation ({correlation:.2f}) between {metric1.value} and {metric2.value}",
                metric_types=[metric1, metric2],
                confidence=0.7,
                impact_score=abs(correlation),
                first_detected=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                occurrence_count=1,
                related_operations=operations,
                recommendations=recommendations,
            )

            self.patterns[pattern_id] = pattern
            logger.info(f"New correlation pattern detected: {pattern.description}")

    def _generate_correlation_recommendations(
        self, metric1: MetricType, metric2: MetricType, correlation: float
    ) -> List[str]:
        """Generate recommendations based on metric correlation."""
        recommendations = []

        if metric1 == MetricType.LATENCY and metric2 == MetricType.ERROR_RATE:
            if correlation > 0:
                recommendations.extend(
                    [
                        "High latency correlates with errors - investigate timeout settings",
                        "Consider implementing circuit breakers",
                        "Review database connection pool configuration",
                    ]
                )

        elif metric1 == MetricType.CACHE_HIT_RATE and metric2 == MetricType.LATENCY:
            if correlation < 0:
                recommendations.extend(
                    [
                        "Low cache hit rate increases latency - optimize caching strategy",
                        "Consider warming up caches more aggressively",
                        "Review cache TTL settings",
                    ]
                )

        elif metric1 == MetricType.MEMORY_USAGE and metric2 == MetricType.LATENCY:
            if correlation > 0:
                recommendations.extend(
                    [
                        "High memory usage correlates with latency - check for memory leaks",
                        "Consider implementing memory pressure monitoring",
                        "Review garbage collection settings",
                    ]
                )

        return recommendations

    async def _detect_trend_patterns(self):
        """Detect trending patterns in metrics."""
        for operation, events in self.events_by_operation.items():
            if len(events) < 50:
                continue

            recent_events = list(events)[-50:]  # Last 50 events

            # Group by metric type and detect trends
            metric_trends = self._calculate_trends(recent_events)

            for metric_type, trend_data in metric_trends.items():
                await self._record_trend_pattern(operation, metric_type, trend_data)

    def _calculate_trends(
        self, events: List[PerformanceEvent]
    ) -> Dict[MetricType, Dict[str, Any]]:
        """Calculate trend data for events."""
        trends = {}

        # Group by metric type
        metric_events: Dict[MetricType, List[PerformanceEvent]] = defaultdict(list)
        for event in events:
            metric_events[event.metric_type].append(event)

        for metric_type, metric_event_list in metric_events.items():
            if len(metric_event_list) < 10:
                continue

            # Sort by timestamp
            metric_event_list.sort(key=lambda e: e.timestamp)

            # Calculate trend
            values = [event.value for event in metric_event_list]
            x = np.arange(len(values))

            # Linear regression for trend
            if len(values) >= 3:
                slope, intercept = np.polyfit(x, values, 1)

                trends[metric_type] = {
                    "slope": slope,
                    "intercept": intercept,
                    "direction": (
                        "increasing"
                        if slope > 0
                        else "decreasing" if slope < 0 else "stable"
                    ),
                    "magnitude": abs(slope),
                    "r_squared": self._calculate_r_squared(x, values, slope, intercept),
                }

        return trends

    def _calculate_r_squared(
        self, x: np.ndarray, y: List[float], slope: float, intercept: float
    ) -> float:
        """Calculate R-squared for trend line."""
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    async def _record_trend_pattern(
        self, operation: str, metric_type: MetricType, trend_data: Dict[str, Any]
    ):
        """Record a detected trend pattern."""
        if trend_data["r_squared"] < 0.5:  # Weak trend
            return

        pattern_id = f"trend_{operation}_{metric_type.value}_{trend_data['direction']}"

        description = (
            f"{trend_data['direction'].title()} trend in {metric_type.value} "
            f"for {operation} (R²={trend_data['r_squared']:.3f})"
        )

        recommendations = self._generate_trend_recommendations(
            metric_type, trend_data["direction"], trend_data["magnitude"]
        )

        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.last_seen = datetime.utcnow()
            pattern.occurrence_count += 1
        else:
            pattern = PerformancePattern(
                pattern_id=pattern_id,
                description=description,
                metric_types=[metric_type],
                confidence=trend_data["r_squared"],
                impact_score=trend_data["magnitude"],
                first_detected=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                occurrence_count=1,
                related_operations={operation},
                recommendations=recommendations,
            )

            self.patterns[pattern_id] = pattern

            if trend_data["r_squared"] > 0.8:  # Strong trend
                logger.warning(f"Strong trend pattern detected: {description}")

    def _generate_trend_recommendations(
        self, metric_type: MetricType, direction: str, magnitude: float
    ) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []

        if direction == "increasing":
            if metric_type == MetricType.LATENCY:
                recommendations.extend(
                    [
                        "Latency is trending upward - investigate performance bottlenecks",
                        "Consider scaling up resources or optimizing queries",
                        "Review recent code changes for performance regressions",
                    ]
                )
            elif metric_type == MetricType.ERROR_RATE:
                recommendations.extend(
                    [
                        "Error rate is increasing - investigate root causes",
                        "Check for failing dependencies or services",
                        "Review recent deployments for issues",
                    ]
                )
            elif metric_type == MetricType.MEMORY_USAGE:
                recommendations.extend(
                    [
                        "Memory usage trending up - check for memory leaks",
                        "Consider implementing memory pressure handling",
                        "Review object lifecycles and garbage collection",
                    ]
                )

        elif direction == "decreasing":
            if metric_type == MetricType.THROUGHPUT:
                recommendations.extend(
                    [
                        "Throughput is declining - investigate capacity issues",
                        "Check for resource contention or bottlenecks",
                        "Consider load balancing optimizations",
                    ]
                )
            elif metric_type == MetricType.CACHE_HIT_RATE:
                recommendations.extend(
                    [
                        "Cache hit rate declining - review caching strategy",
                        "Check cache size and eviction policies",
                        "Consider cache warming strategies",
                    ]
                )

        return recommendations

    async def _detect_outlier_patterns(self):
        """Detect outlier cluster patterns."""
        # Implementation for detecting clusters of outliers
        # This could indicate systematic issues rather than random anomalies
        pass

    async def _anomaly_alerter(self):
        """Background task for anomaly alerting."""
        while self._running:
            try:
                await self._process_pending_alerts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in anomaly alerter: {e}")
                await asyncio.sleep(60)

    async def _process_pending_alerts(self):
        """Process and send pending alerts."""
        # Implementation for alert processing and notification
        # This could include sending alerts to external systems
        pass

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [
            event for event in self.recent_events if event.timestamp >= cutoff_time
        ]

        if not recent_events:
            return {"message": "No recent performance data"}

        # Calculate summary statistics
        summary = {
            "period_hours": hours,
            "total_events": len(recent_events),
            "event_rate_per_hour": len(recent_events) / hours,
            "operations": len(set(event.operation for event in recent_events)),
            "metrics": {},
        }

        # Group by metric type
        metric_summaries = defaultdict(list)
        for event in recent_events:
            metric_summaries[event.metric_type].append(event.value)

        for metric_type, values in metric_summaries.items():
            summary["metrics"][metric_type.value] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            }

        # Add pattern and anomaly information
        summary["active_patterns"] = len(self.patterns)
        summary["anomaly_counts"] = dict(self.anomaly_counts)
        summary["recent_alerts"] = len(
            [
                alert
                for alert in self.alert_history
                if datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
            ]
        )

        return summary

    def get_baseline_info(self) -> Dict[str, Any]:
        """Get information about established baselines."""
        baseline_info = {}

        for (metric_type, operation), baseline in self.baselines.items():
            key = f"{metric_type.value}:{operation}"
            baseline_info[key] = {
                "mean": baseline.mean,
                "std_dev": baseline.std_dev,
                "p95": baseline.p95,
                "p99": baseline.p99,
                "sample_count": baseline.sample_count,
                "established_at": baseline.established_at.isoformat(),
                "last_updated": baseline.last_updated.isoformat(),
            }

        return baseline_info

    def get_patterns(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Get detected patterns above confidence threshold."""
        pattern_list = []

        for pattern in self.patterns.values():
            if pattern.confidence >= min_confidence:
                pattern_data = {
                    "pattern_id": pattern.pattern_id,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "impact_score": pattern.impact_score,
                    "occurrence_count": pattern.occurrence_count,
                    "first_detected": pattern.first_detected.isoformat(),
                    "last_seen": pattern.last_seen.isoformat(),
                    "metric_types": [mt.value for mt in pattern.metric_types],
                    "related_operations": list(pattern.related_operations),
                    "recommendations": pattern.recommendations,
                }
                pattern_list.append(pattern_data)

        # Sort by impact score and confidence
        pattern_list.sort(
            key=lambda p: (p["impact_score"] * p["confidence"]), reverse=True
        )

        return pattern_list

    def export_data(self, filepath: Path):
        """Export analytics data to file."""
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "baselines": {},
            "patterns": {},
            "recent_events": [],
            "alert_history": list(self.alert_history),
        }

        # Export baselines
        for (metric_type, operation), baseline in self.baselines.items():
            key = f"{metric_type.value}:{operation}"
            export_data["baselines"][key] = asdict(baseline)
            # Convert datetime fields to ISO strings
            export_data["baselines"][key][
                "established_at"
            ] = baseline.established_at.isoformat()
            export_data["baselines"][key][
                "last_updated"
            ] = baseline.last_updated.isoformat()

        # Export patterns
        for pattern_id, pattern in self.patterns.items():
            pattern_data = asdict(pattern)
            pattern_data["metric_types"] = [mt.value for mt in pattern.metric_types]
            pattern_data["related_operations"] = list(pattern.related_operations)
            pattern_data["first_detected"] = pattern.first_detected.isoformat()
            pattern_data["last_seen"] = pattern.last_seen.isoformat()
            export_data["patterns"][pattern_id] = pattern_data

        # Export recent events
        for event in list(self.recent_events)[-1000:]:  # Last 1000 events
            export_data["recent_events"].append(event.to_dict())

        # Write to file
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Performance data exported to {filepath}")


# Global performance tracker instance
_performance_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker


async def record_operation_latency(
    operation: str, latency_ms: float, session_id: Optional[str] = None, **metadata
):
    """Convenience function to record operation latency."""
    tracker = get_performance_tracker()
    tracker.record_event(
        MetricType.LATENCY, latency_ms, operation, session_id=session_id, **metadata
    )


def track_performance(operation: str, session_id: Optional[str] = None):
    """Decorator for automatic performance tracking."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = datetime.utcnow()
                latency_ms = (end_time - start_time).total_seconds() * 1000

                tracker = get_performance_tracker()
                tracker.record_event(
                    MetricType.LATENCY,
                    latency_ms,
                    operation,
                    session_id=session_id,
                    success=success,
                    error=error,
                )

        def sync_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = datetime.utcnow()
                latency_ms = (end_time - start_time).total_seconds() * 1000

                tracker = get_performance_tracker()
                tracker.record_event(
                    MetricType.LATENCY,
                    latency_ms,
                    operation,
                    session_id=session_id,
                    success=success,
                    error=error,
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
