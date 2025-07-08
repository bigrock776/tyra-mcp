"""
Custom metrics and monitoring for the Tyra memory system.

Provides specialized metrics for memory operations, RAG performance,
and system health monitoring with aggregation and alerting capabilities.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger
from .telemetry import get_telemetry

logger = get_logger(__name__)


@dataclass
class MetricSnapshot:
    """Point-in-time metric value."""

    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricAggregation:
    """Aggregated metric statistics."""

    count: int
    sum: float
    min: float
    max: float
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricAggregation":
        """Create aggregation from list of values."""
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        arr = np.array(values)
        return cls(
            count=len(values),
            sum=float(np.sum(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )


class MemorySystemMetrics:
    """
    Specialized metrics for the memory system.

    Tracks:
    - Memory operation performance
    - RAG pipeline metrics
    - Hallucination detection accuracy
    - Cache performance
    - Resource utilization
    - Business metrics
    """

    def __init__(self):
        self.telemetry = get_telemetry()

        # Time series data storage
        self._metrics_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._window_size = timedelta(minutes=15)  # Default aggregation window

        # Custom counters
        self._counters: Dict[str, int] = defaultdict(int)

        # Rate limiters for high-frequency metrics
        self._last_recorded: Dict[str, float] = {}
        self._min_interval = 0.1  # Minimum seconds between recordings

        # Alerts configuration
        self._alert_thresholds: Dict[str, Dict[str, float]] = {}
        self._alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Initialize custom metrics
        self._initialize_custom_metrics()

    def _initialize_custom_metrics(self) -> None:
        """Initialize custom metric instruments."""
        # Memory operation metrics
        self.memory_store_counter = self.telemetry.create_counter(
            "tyra.memory.store.total",
            unit="1",
            description="Total memory store operations",
        )

        self.memory_search_counter = self.telemetry.create_counter(
            "tyra.memory.search.total",
            unit="1",
            description="Total memory search operations",
        )

        self.memory_store_latency = self.telemetry.create_histogram(
            "tyra.memory.store.latency",
            unit="ms",
            description="Memory store operation latency",
        )

        self.memory_search_latency = self.telemetry.create_histogram(
            "tyra.memory.search.latency",
            unit="ms",
            description="Memory search operation latency",
        )

        # RAG metrics
        self.embedding_latency = self.telemetry.create_histogram(
            "tyra.rag.embedding.latency",
            unit="ms",
            description="Embedding generation latency",
        )

        self.rerank_latency = self.telemetry.create_histogram(
            "tyra.rag.rerank.latency",
            unit="ms",
            description="Reranking operation latency",
        )

        self.hallucination_confidence = self.telemetry.create_histogram(
            "tyra.rag.hallucination.confidence",
            unit="percent",
            description="Hallucination detection confidence scores",
        )

        # Cache metrics
        self.cache_hit_rate = self.telemetry.create_gauge(
            "tyra.cache.hit_rate",
            self._observe_cache_hit_rate,
            unit="ratio",
            description="Cache hit rate",
        )

        # Business metrics
        self.agent_activity = self.telemetry.create_counter(
            "tyra.agent.activity", unit="1", description="Agent activity by type"
        )

        self.confidence_distribution = self.telemetry.create_histogram(
            "tyra.confidence.distribution",
            unit="percent",
            description="Distribution of confidence scores",
        )

    async def record_memory_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a memory operation metric."""
        try:
            labels = {"operation": operation, "success": str(success)}

            if metadata:
                # Add relevant metadata as labels
                if "agent_id" in metadata:
                    labels["agent_id"] = metadata["agent_id"]
                if "chunks" in metadata:
                    labels["chunk_count"] = str(metadata["chunks"])

            # Record in telemetry
            if operation == "store":
                if self.memory_store_counter:
                    self.memory_store_counter.add(1, labels)
                if self.memory_store_latency:
                    self.memory_store_latency.record(duration_ms, labels)
            elif operation == "search":
                if self.memory_search_counter:
                    self.memory_search_counter.add(1, labels)
                if self.memory_search_latency:
                    self.memory_search_latency.record(duration_ms, labels)

            # Record in time series
            await self._record_time_series(
                f"memory.{operation}.latency", duration_ms, labels
            )

            # Check alerts
            await self._check_alerts(f"memory.{operation}.latency", duration_ms)

        except Exception as e:
            logger.error(f"Failed to record memory operation metric: {e}")

    async def record_rag_metric(
        self,
        component: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a RAG pipeline metric."""
        try:
            labels = {"component": component}

            if metadata:
                labels.update(metadata)

            # Record in appropriate histogram
            if component == "embedding" and self.embedding_latency:
                self.embedding_latency.record(value, labels)
            elif component == "rerank" and self.rerank_latency:
                self.rerank_latency.record(value, labels)
            elif component == "hallucination" and self.hallucination_confidence:
                self.hallucination_confidence.record(value, labels)

            # Record in time series
            await self._record_time_series(
                f"rag.{component}.{metric_name}", value, labels
            )

        except Exception as e:
            logger.error(f"Failed to record RAG metric: {e}")

    async def record_confidence_score(
        self, score: float, level: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a confidence score."""
        try:
            labels = {"level": level}

            if context:
                if "agent_id" in context:
                    labels["agent_id"] = context["agent_id"]
                if "operation" in context:
                    labels["operation"] = context["operation"]

            if self.confidence_distribution:
                self.confidence_distribution.record(score, labels)

            # Track confidence levels
            self._counters[f"confidence.{level}"] += 1

            # Record for trending
            await self._record_time_series(f"confidence.{level}", score, labels)

        except Exception as e:
            logger.error(f"Failed to record confidence score: {e}")

    async def record_cache_metrics(
        self, hits: int, misses: int, evictions: int, size_bytes: int
    ) -> None:
        """Record cache performance metrics."""
        try:
            # Update counters
            self._counters["cache.hits"] = hits
            self._counters["cache.misses"] = misses
            self._counters["cache.evictions"] = evictions

            # Calculate hit rate
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0

            # Record metrics
            await self._record_time_series("cache.hit_rate", hit_rate)
            await self._record_time_series("cache.size_bytes", size_bytes)
            await self._record_time_series("cache.evictions", evictions)

            # Check cache performance alerts
            if hit_rate < 0.5:  # Alert if hit rate drops below 50%
                await self._trigger_alert(
                    "cache.hit_rate.low", hit_rate, {"threshold": 0.5}
                )

        except Exception as e:
            logger.error(f"Failed to record cache metrics: {e}")

    async def record_agent_activity(
        self,
        agent_id: str,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record agent activity metrics."""
        try:
            labels = {"agent_id": agent_id, "operation": operation}

            if self.agent_activity:
                self.agent_activity.add(1, labels)

            # Track per-agent metrics
            await self._record_time_series(
                f"agent.{agent_id}.{operation}", duration_ms, labels
            )

            # Update agent activity counters
            self._counters[f"agent.{agent_id}.total"] += 1
            self._counters[f"agent.{agent_id}.{operation}"] += 1

        except Exception as e:
            logger.error(f"Failed to record agent activity: {e}")

    async def _record_time_series(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a time series data point."""
        # Rate limiting
        now = time.time()
        last_recorded = self._last_recorded.get(metric_name, 0)

        if now - last_recorded < self._min_interval:
            return

        self._last_recorded[metric_name] = now

        # Store snapshot
        snapshot = MetricSnapshot(
            timestamp=datetime.utcnow(), value=value, labels=labels or {}
        )

        self._metrics_data[metric_name].append(snapshot)

    def _observe_cache_hit_rate(self, options) -> List:
        """Observable callback for cache hit rate."""
        hits = self._counters.get("cache.hits", 0)
        misses = self._counters.get("cache.misses", 0)
        total = hits + misses

        if total > 0:
            return [{"value": hits / total}]
        return [{"value": 0.0}]

    async def get_metric_aggregation(
        self,
        metric_name: str,
        window: Optional[timedelta] = None,
        labels_filter: Optional[Dict[str, str]] = None,
    ) -> MetricAggregation:
        """Get aggregated statistics for a metric."""
        window = window or self._window_size
        cutoff_time = datetime.utcnow() - window

        # Get relevant snapshots
        snapshots = self._metrics_data.get(metric_name, [])

        # Filter by time window and labels
        values = []
        for snapshot in snapshots:
            if snapshot.timestamp < cutoff_time:
                continue

            if labels_filter:
                # Check if all filter labels match
                if not all(
                    snapshot.labels.get(k) == v for k, v in labels_filter.items()
                ):
                    continue

            values.append(snapshot.value)

        return MetricAggregation.from_values(values)

    async def get_percentiles(
        self,
        metric_name: str,
        percentiles: List[float] = [50, 90, 95, 99],
        window: Optional[timedelta] = None,
    ) -> Dict[float, float]:
        """Get percentile values for a metric."""
        agg = await self.get_metric_aggregation(metric_name, window)

        result = {}
        for p in percentiles:
            if p == 50:
                result[p] = agg.p50
            elif p == 90:
                result[p] = agg.p90
            elif p == 95:
                result[p] = agg.p95
            elif p == 99:
                result[p] = agg.p99
            else:
                # Calculate custom percentile
                values = [s.value for s in self._metrics_data.get(metric_name, [])]
                if values:
                    result[p] = float(np.percentile(values, p))
                else:
                    result[p] = 0.0

        return result

    def set_alert_threshold(
        self,
        metric_name: str,
        threshold_type: str,  # "min", "max", "range"
        value: Union[float, Tuple[float, float]],
        callback: Optional[Callable] = None,
    ) -> None:
        """Set alert threshold for a metric."""
        self._alert_thresholds[metric_name] = {"type": threshold_type, "value": value}

        if callback:
            self._alert_callbacks[metric_name].append(callback)

    async def _check_alerts(self, metric_name: str, value: float) -> None:
        """Check if metric value triggers any alerts."""
        if metric_name not in self._alert_thresholds:
            return

        threshold = self._alert_thresholds[metric_name]
        triggered = False

        if threshold["type"] == "min" and value < threshold["value"]:
            triggered = True
        elif threshold["type"] == "max" and value > threshold["value"]:
            triggered = True
        elif threshold["type"] == "range":
            min_val, max_val = threshold["value"]
            if value < min_val or value > max_val:
                triggered = True

        if triggered:
            await self._trigger_alert(metric_name, value, threshold)

    async def _trigger_alert(
        self, alert_name: str, value: float, context: Dict[str, Any]
    ) -> None:
        """Trigger alert callbacks."""
        logger.warning(f"Alert triggered: {alert_name}", value=value, context=context)

        # Call registered callbacks
        for callback in self._alert_callbacks.get(alert_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_name, value, context)
                else:
                    callback(alert_name, value, context)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    async def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        scores = []

        # Memory operation health
        memory_latency = await self.get_metric_aggregation("memory.search.latency")
        if memory_latency.count > 0:
            # Score based on p95 latency (100ms = 100 score, 1000ms = 0 score)
            latency_score = max(0, 100 - (memory_latency.p95 - 100) / 9)
            scores.append(latency_score)

        # Cache health
        hit_rate = self._counters.get("cache.hits", 0) / max(
            self._counters.get("cache.hits", 0) + self._counters.get("cache.misses", 0),
            1,
        )
        cache_score = hit_rate * 100
        scores.append(cache_score)

        # Error rate
        total_ops = sum(v for k, v in self._counters.items() if k.endswith(".total"))
        error_ops = sum(v for k, v in self._counters.items() if k.endswith(".error"))
        error_rate = error_ops / max(total_ops, 1)
        error_score = (1 - error_rate) * 100
        scores.append(error_score)

        # Average confidence levels
        high_confidence = self._counters.get(
            "confidence.rock_solid", 0
        ) + self._counters.get("confidence.high", 0)
        total_confidence = sum(
            v for k, v in self._counters.items() if k.startswith("confidence.")
        )

        if total_confidence > 0:
            confidence_score = (high_confidence / total_confidence) * 100
            scores.append(confidence_score)

        # Return weighted average
        return sum(scores) / len(scores) if scores else 100.0

    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for dashboard display."""
        return {
            "health_score": await self.get_system_health_score(),
            "memory_operations": {
                "store": await self.get_metric_aggregation("memory.store.latency"),
                "search": await self.get_metric_aggregation("memory.search.latency"),
            },
            "rag_performance": {
                "embedding": await self.get_metric_aggregation("rag.embedding.latency"),
                "reranking": await self.get_metric_aggregation("rag.rerank.latency"),
                "hallucination": await self.get_metric_aggregation(
                    "rag.hallucination.confidence"
                ),
            },
            "cache_metrics": {
                "hit_rate": self._observe_cache_hit_rate(None)[0]["value"],
                "total_hits": self._counters.get("cache.hits", 0),
                "total_misses": self._counters.get("cache.misses", 0),
                "evictions": self._counters.get("cache.evictions", 0),
            },
            "confidence_distribution": {
                "rock_solid": self._counters.get("confidence.rock_solid", 0),
                "high": self._counters.get("confidence.high", 0),
                "fuzzy": self._counters.get("confidence.fuzzy", 0),
                "low": self._counters.get("confidence.low", 0),
            },
            "agent_activity": {
                agent_id: {
                    "total": self._counters.get(f"agent.{agent_id}.total", 0),
                    "operations": {
                        op: self._counters.get(f"agent.{agent_id}.{op}", 0)
                        for op in ["store", "search", "analyze"]
                    },
                }
                for agent_id in ["tyra", "claude", "archon"]
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    def reset_counters(self) -> None:
        """Reset all counters (useful for testing)."""
        self._counters.clear()

    def export_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export all metrics data for persistence or analysis."""
        exported = {}

        for metric_name, snapshots in self._metrics_data.items():
            exported[metric_name] = [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "value": s.value,
                    "labels": s.labels,
                }
                for s in snapshots
            ]

        return exported


# Global metrics instance
_memory_metrics: Optional[MemorySystemMetrics] = None


def get_memory_metrics() -> MemorySystemMetrics:
    """Get the global memory system metrics instance."""
    global _memory_metrics
    if _memory_metrics is None:
        _memory_metrics = MemorySystemMetrics()
    return _memory_metrics
