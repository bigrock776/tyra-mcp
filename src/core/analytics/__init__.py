"""
Analytics module for performance tracking and self-learning capabilities.
"""

from .config_optimizer import (
    ConfigParameter,
    ConfigurationOptimizer,
    OptimizationExperiment,
    OptimizationResult,
    OptimizationStrategy,
    get_config_optimizer,
)
from .memory_health import (
    HealthCategory,
    HealthCheck,
    HealthMetric,
    HealthStatus,
    MemoryHealthManager,
    SystemSnapshot,
    get_health_manager,
)
from .performance_tracker import (
    MetricType,
    PerformanceBaseline,
    PerformanceEvent,
    PerformancePattern,
    PerformanceTracker,
    get_performance_tracker,
    record_operation_latency,
    track_performance,
)

__all__ = [
    # Performance tracking
    "PerformanceTracker",
    "PerformanceEvent",
    "PerformancePattern",
    "PerformanceBaseline",
    "MetricType",
    "get_performance_tracker",
    "record_operation_latency",
    "track_performance",
    # Health monitoring
    "MemoryHealthManager",
    "HealthMetric",
    "HealthCheck",
    "SystemSnapshot",
    "HealthStatus",
    "HealthCategory",
    "get_health_manager",
    # Configuration optimization
    "ConfigurationOptimizer",
    "OptimizationExperiment",
    "OptimizationResult",
    "ConfigParameter",
    "OptimizationStrategy",
    "get_config_optimizer",
]
