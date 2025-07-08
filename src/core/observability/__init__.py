"""
Observability module for comprehensive system monitoring.

Provides OpenTelemetry-based tracing, metrics, and logging integration
for complete observability of the Tyra memory system.
"""

from .metrics import (
    MemorySystemMetrics,
    MetricAggregation,
    MetricSnapshot,
    get_memory_metrics,
)
from .telemetry import TelemetryManager, get_telemetry, traced
from .tracing import (
    EnhancedTracer,
    TraceContext,
    get_tracer,
    traced_memory_operation,
    traced_rag_component,
)

__all__ = [
    # Telemetry
    "TelemetryManager",
    "get_telemetry",
    "traced",
    # Metrics
    "MemorySystemMetrics",
    "MetricSnapshot",
    "MetricAggregation",
    "get_memory_metrics",
    # Tracing
    "EnhancedTracer",
    "TraceContext",
    "get_tracer",
    "traced_memory_operation",
    "traced_rag_component",
]
