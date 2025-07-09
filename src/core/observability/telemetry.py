"""
OpenTelemetry integration for comprehensive observability.

Provides distributed tracing, metrics collection, and structured logging
for complete system observability with support for multiple exporters.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ..utils.config import get_settings
from ..utils.logger import get_logger
from .performance_optimized_telemetry import get_telemetry as get_optimized_telemetry
from .telemetry_optimizer import get_telemetry_optimizer

logger = get_logger(__name__)


class TelemetryManager:
    """
    Centralized telemetry management with OpenTelemetry.

    Features:
    - Distributed tracing with context propagation
    - Metrics collection (counters, histograms, gauges)
    - Resource utilization monitoring
    - Performance tracking
    - Error tracking and alerting
    - Multiple exporter support
    - Automatic instrumentation
    """

    def __init__(self):
        self.settings = get_settings()
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self.meter: Optional[metrics.Meter] = None

        # Metrics instruments
        self.request_counter: Optional[metrics.Counter] = None
        self.error_counter: Optional[metrics.Counter] = None
        self.latency_histogram: Optional[metrics.Histogram] = None
        self.active_requests_gauge: Optional[metrics.UpDownCounter] = None

        # Memory metrics
        self.memory_usage_gauge: Optional[metrics.ObservableGauge] = None
        self.cpu_usage_gauge: Optional[metrics.ObservableGauge] = None

        # Custom metrics
        self._metrics_registry: Dict[str, Any] = {}
        self._active_spans: Dict[str, trace.Span] = {}

        self._initialized = False

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize telemetry with OpenTelemetry providers."""
        try:
            telemetry_config = config or self.settings.observability

            if not telemetry_config.get("enabled", True):
                logger.info("Telemetry is disabled")
                return

            # Create resource
            resource = Resource.create(
                {
                    "service.name": telemetry_config.get(
                        "service_name", "tyra-memory-server"
                    ),
                    "service.version": telemetry_config.get("service_version", "1.0.0"),
                    "service.namespace": telemetry_config.get("namespace", "tyra"),
                    "deployment.environment": telemetry_config.get(
                        "environment", "production"
                    ),
                }
            )

            # Initialize tracing
            await self._initialize_tracing(resource, telemetry_config)

            # Initialize metrics
            await self._initialize_metrics(resource, telemetry_config)

            # Set up context propagation
            set_global_textmap(TraceContextTextMapPropagator())

            # Instrument asyncio if enabled
            if telemetry_config.get("instrument_asyncio", True):
                AsyncioInstrumentor().instrument()

            # Create standard instruments
            await self._create_standard_instruments()

            self._initialized = True

            logger.info(
                "Telemetry initialized",
                tracing_enabled=bool(self.tracer_provider),
                metrics_enabled=bool(self.meter_provider),
                export_target=telemetry_config.get("export_target", "console"),
            )

        except Exception as e:
            logger.error("Failed to initialize telemetry", error=str(e))
            # Don't raise - telemetry should not break the application

    async def _initialize_tracing(
        self, resource: Resource, config: Dict[str, Any]
    ) -> None:
        """Initialize tracing provider and exporters."""
        try:
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)

            # Configure exporter based on target
            export_target = config.get("export_target", "console")

            if export_target == "console":
                exporter = ConsoleSpanExporter()
                processor = SimpleSpanProcessor(exporter)
            elif export_target == "otlp":
                endpoint = config.get("otlp_endpoint", "localhost:4317")
                exporter = OTLPSpanExporter(
                    endpoint=endpoint, insecure=config.get("otlp_insecure", True)
                )
                processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=config.get("max_queue_size", 2048),
                    max_export_batch_size=config.get("max_export_batch_size", 512),
                )
            else:
                # Default to console
                exporter = ConsoleSpanExporter()
                processor = SimpleSpanProcessor(exporter)

            self.tracer_provider.add_span_processor(processor)
            trace.set_tracer_provider(self.tracer_provider)

            # Get tracer
            self.tracer = trace.get_tracer(
                __name__, version=config.get("service_version", "1.0.0")
            )

        except Exception as e:
            logger.error("Failed to initialize tracing", error=str(e))

    async def _initialize_metrics(
        self, resource: Resource, config: Dict[str, Any]
    ) -> None:
        """Initialize metrics provider and exporters."""
        try:
            # Configure exporter based on target
            export_target = config.get("export_target", "console")

            if export_target == "console":
                exporter = ConsoleMetricExporter()
            elif export_target == "otlp":
                endpoint = config.get("otlp_endpoint", "localhost:4317")
                exporter = OTLPMetricExporter(
                    endpoint=endpoint, insecure=config.get("otlp_insecure", True)
                )
            else:
                exporter = ConsoleMetricExporter()

            # Create metric reader
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=config.get("metric_export_interval", 60000),
            )

            # Create meter provider
            self.meter_provider = MeterProvider(
                resource=resource, metric_readers=[reader]
            )

            metrics.set_meter_provider(self.meter_provider)

            # Get meter
            self.meter = metrics.get_meter(
                __name__, version=config.get("service_version", "1.0.0")
            )

        except Exception as e:
            logger.error("Failed to initialize metrics", error=str(e))

    async def _create_standard_instruments(self) -> None:
        """Create standard metric instruments."""
        if not self.meter:
            return

        # Request metrics
        self.request_counter = self.meter.create_counter(
            "tyra.requests.total", unit="1", description="Total number of requests"
        )

        self.error_counter = self.meter.create_counter(
            "tyra.errors.total", unit="1", description="Total number of errors"
        )

        self.latency_histogram = self.meter.create_histogram(
            "tyra.request.duration",
            unit="ms",
            description="Request latency in milliseconds",
        )

        self.active_requests_gauge = self.meter.create_up_down_counter(
            "tyra.requests.active", unit="1", description="Number of active requests"
        )

        # System metrics
        self.memory_usage_gauge = self.meter.create_observable_gauge(
            "tyra.memory.usage",
            [self._observe_memory_usage],
            unit="bytes",
            description="Memory usage in bytes",
        )

        self.cpu_usage_gauge = self.meter.create_observable_gauge(
            "tyra.cpu.usage",
            [self._observe_cpu_usage],
            unit="percent",
            description="CPU usage percentage",
        )

    def _observe_memory_usage(self, options: CallbackOptions) -> List[Observation]:
        """Callback for memory usage observations."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return [
                Observation(memory_info.rss, {"type": "rss"}),
                Observation(memory_info.vms, {"type": "vms"}),
            ]
        except Exception:
            return []

    def _observe_cpu_usage(self, options: CallbackOptions) -> List[Observation]:
        """Callback for CPU usage observations."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return [Observation(cpu_percent)]
        except Exception:
            return []

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Create a traced span context."""
        if not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(
            name, kind=kind, attributes=attributes or {}
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def trace_method(
        self, name: Optional[str] = None, kind: SpanKind = SpanKind.INTERNAL
    ):
        """Decorator for tracing methods."""

        def decorator(func):
            span_name = name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self.trace(span_name, kind=kind):
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.tracer:
                    return func(*args, **kwargs)

                with self.tracer.start_as_current_span(span_name, kind=kind):
                    return func(*args, **kwargs)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    async def record_request(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a request with metrics."""
        if not self._initialized:
            return

        try:
            labels = {"operation": operation, "success": str(success)}
            if attributes:
                labels.update(attributes)

            # Record counter
            if self.request_counter:
                self.request_counter.add(1, labels)

            # Record latency
            if self.latency_histogram:
                self.latency_histogram.record(duration_ms, labels)

            # Record errors
            if not success and self.error_counter:
                self.error_counter.add(1, labels)

        except Exception as e:
            logger.error("Failed to record request metrics", error=str(e))

    async def increment_active_requests(self, operation: str, delta: int = 1) -> None:
        """Increment active requests gauge."""
        if self.active_requests_gauge:
            try:
                self.active_requests_gauge.add(delta, {"operation": operation})
            except Exception as e:
                logger.error("Failed to update active requests", error=str(e))

    def create_counter(
        self, name: str, unit: str = "1", description: str = ""
    ) -> Optional[metrics.Counter]:
        """Create a custom counter metric."""
        if not self.meter:
            return None

        try:
            counter = self.meter.create_counter(
                name, unit=unit, description=description
            )
            self._metrics_registry[name] = counter
            return counter
        except Exception as e:
            logger.error(f"Failed to create counter {name}", error=str(e))
            return None

    def create_histogram(
        self, name: str, unit: str = "1", description: str = ""
    ) -> Optional[metrics.Histogram]:
        """Create a custom histogram metric."""
        if not self.meter:
            return None

        try:
            histogram = self.meter.create_histogram(
                name, unit=unit, description=description
            )
            self._metrics_registry[name] = histogram
            return histogram
        except Exception as e:
            logger.error(f"Failed to create histogram {name}", error=str(e))
            return None

    def create_gauge(
        self, name: str, callback: Callable, unit: str = "1", description: str = ""
    ) -> Optional[metrics.ObservableGauge]:
        """Create a custom gauge metric."""
        if not self.meter:
            return None

        try:
            gauge = self.meter.create_observable_gauge(
                name, [callback], unit=unit, description=description
            )
            self._metrics_registry[name] = gauge
            return gauge
        except Exception as e:
            logger.error(f"Failed to create gauge {name}", error=str(e))
            return None

    def get_current_trace_context(self) -> Optional[Dict[str, str]]:
        """Get current trace context for propagation."""
        if not self.tracer:
            return None

        try:
            from opentelemetry import propagate

            carrier = {}
            propagate.inject(carrier)
            return carrier
        except Exception as e:
            logger.error("Failed to get trace context", error=str(e))
            return None

    def inject_trace_context(self, carrier: Dict[str, str]) -> None:
        """Inject trace context from carrier."""
        if not self.tracer:
            return

        try:
            from opentelemetry import propagate

            ctx = propagate.extract(carrier)
            # Context will be automatically used by the next span
        except Exception as e:
            logger.error("Failed to inject trace context", error=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on telemetry system."""
        health_status = {
            "status": "healthy" if self._initialized else "not_initialized",
            "tracing_enabled": bool(self.tracer_provider),
            "metrics_enabled": bool(self.meter_provider),
            "active_spans": len(self._active_spans),
            "custom_metrics": len(self._metrics_registry),
        }

        # Test tracing
        if self.tracer:
            try:
                async with self.trace("health_check_test"):
                    pass
                health_status["tracing_status"] = "operational"
            except Exception as e:
                health_status["tracing_status"] = f"error: {str(e)}"

        # Test metrics
        if self.request_counter:
            try:
                self.request_counter.add(0, {"operation": "health_check"})
                health_status["metrics_status"] = "operational"
            except Exception as e:
                health_status["metrics_status"] = f"error: {str(e)}"

        return health_status

    async def optimize_telemetry(self) -> Dict[str, Any]:
        """Optimize telemetry performance."""
        try:
            optimizer = get_telemetry_optimizer()
            return await optimizer.optimize()
        except Exception as e:
            logger.error(f"Telemetry optimization failed: {e}")
            return {"status": "failed", "error": str(e)}

    def get_telemetry_performance_stats(self) -> Dict[str, Any]:
        """Get telemetry performance statistics."""
        try:
            optimized_telemetry = get_optimized_telemetry()
            return optimized_telemetry.get_performance_stats()
        except Exception as e:
            logger.error(f"Failed to get telemetry performance stats: {e}")
            return {"error": str(e)}

    def enable_performance_optimization(self) -> None:
        """Enable performance optimization for telemetry."""
        try:
            optimized_telemetry = get_optimized_telemetry()
            optimized_telemetry.enable_telemetry()
            logger.info("Telemetry performance optimization enabled")
        except Exception as e:
            logger.error(f"Failed to enable telemetry optimization: {e}")

    def disable_performance_optimization(self) -> None:
        """Disable performance optimization for telemetry."""
        try:
            optimized_telemetry = get_optimized_telemetry()
            optimized_telemetry.disable_telemetry()
            logger.info("Telemetry performance optimization disabled")
        except Exception as e:
            logger.error(f"Failed to disable telemetry optimization: {e}")

    async def emergency_optimize(self) -> Dict[str, Any]:
        """Emergency optimization for critical performance issues."""
        try:
            optimizer = get_telemetry_optimizer()
            return await optimizer.emergency_optimization()
        except Exception as e:
            logger.error(f"Emergency telemetry optimization failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown telemetry providers gracefully."""
        try:
            if self.tracer_provider:
                # Force flush all pending spans
                for (
                    processor
                ) in self.tracer_provider._active_span_processor._span_processors:
                    if hasattr(processor, "force_flush"):
                        processor.force_flush()
                    if hasattr(processor, "shutdown"):
                        processor.shutdown()

            if self.meter_provider:
                # Shutdown metric readers
                self.meter_provider.shutdown()

            logger.info("Telemetry shutdown completed")

        except Exception as e:
            logger.error("Error during telemetry shutdown", error=str(e))


# Global telemetry instance
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry() -> TelemetryManager:
    """Get the global telemetry manager instance."""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager


# Convenience decorators
def traced(name: Optional[str] = None, kind: SpanKind = SpanKind.INTERNAL):
    """Decorator for adding tracing to functions."""

    def decorator(func):
        return get_telemetry().trace_method(name, kind)(func)

    return decorator
