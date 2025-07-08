"""
Enhanced distributed tracing for the Tyra memory system.

Provides detailed tracing capabilities with custom spans, baggage propagation,
and specialized instrumentations for memory and RAG operations.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from opentelemetry import baggage, context, trace
from opentelemetry.baggage.propagation import set_baggage
from opentelemetry.trace import (
    Link,
    Span,
    SpanKind,
    Status,
    StatusCode,
    get_current_span,
)
from opentelemetry.trace.propagation import set_span_in_context

from ..utils.logger import get_logger
from .telemetry import get_telemetry

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class TraceContext:
    """Enhanced trace context with additional metadata."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    baggage: Dict[str, str]
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {}),
            attributes=data.get("attributes", {}),
        )


class EnhancedTracer:
    """
    Enhanced tracer with custom spans and instrumentation.

    Features:
    - Custom span types for memory operations
    - Automatic error tracking and recovery
    - Performance profiling integration
    - Baggage propagation for request context
    - Span linking for related operations
    """

    def __init__(self):
        self.telemetry = get_telemetry()
        self.tracer = trace.get_tracer(__name__)

        # Span storage for linking
        self._active_spans: Dict[str, Span] = {}
        self._span_links: Dict[str, List[str]] = {}

        # Performance tracking
        self._span_timings: Dict[str, float] = {}

    @asynccontextmanager
    async def trace_memory_operation(
        self,
        operation: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Trace a memory operation with specialized attributes."""
        attributes = {"tyra.memory.operation": operation, "tyra.component": "memory"}

        if agent_id:
            attributes["tyra.agent.id"] = agent_id
            # Add to baggage for propagation
            baggage.set_baggage("agent_id", agent_id)

        if metadata:
            # Add relevant metadata as attributes
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    attributes[f"tyra.memory.{key}"] = value

        async with self.telemetry.trace(
            f"memory.{operation}", kind=SpanKind.INTERNAL, attributes=attributes
        ) as span:
            if span:
                self._active_spans[f"memory.{operation}"] = span
                start_time = time.time()

            try:
                yield span
            finally:
                if span:
                    # Record timing
                    duration = time.time() - start_time
                    span.set_attribute("tyra.duration.ms", duration * 1000)

                    # Clean up
                    self._active_spans.pop(f"memory.{operation}", None)

    @asynccontextmanager
    async def trace_rag_pipeline(self, query: str, pipeline_type: str = "search"):
        """Trace entire RAG pipeline execution."""
        attributes = {
            "tyra.rag.pipeline": pipeline_type,
            "tyra.rag.query_length": len(query),
            "tyra.component": "rag",
        }

        # Create parent span for entire pipeline
        async with self.telemetry.trace(
            f"rag.pipeline.{pipeline_type}",
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        ) as pipeline_span:
            if pipeline_span:
                self._active_spans["rag.pipeline"] = pipeline_span

                # Add query to baggage (truncated)
                baggage.set_baggage("query", query[:100])

            try:
                yield pipeline_span
            finally:
                if pipeline_span:
                    self._active_spans.pop("rag.pipeline", None)

    @asynccontextmanager
    async def trace_embedding_generation(self, text_count: int, model_name: str):
        """Trace embedding generation with model details."""
        attributes = {
            "tyra.embedding.text_count": text_count,
            "tyra.embedding.model": model_name,
            "tyra.component": "embedding",
        }

        # Link to parent pipeline span if exists
        links = []
        if "rag.pipeline" in self._active_spans:
            parent_span = self._active_spans["rag.pipeline"]
            links.append(Link(parent_span.get_span_context()))

        async with self.telemetry.trace(
            "embedding.generate", kind=SpanKind.INTERNAL, attributes=attributes
        ) as span:
            if span and links:
                # Note: Links are set at span creation in OpenTelemetry
                # This is for demonstration - actual implementation would pass links at creation
                span.set_attribute("tyra.linked_spans", len(links))

            yield span

    @asynccontextmanager
    async def trace_reranking(self, candidate_count: int, model_type: str):
        """Trace reranking operation."""
        attributes = {
            "tyra.rerank.candidates": candidate_count,
            "tyra.rerank.model_type": model_type,
            "tyra.component": "reranker",
        }

        async with self.telemetry.trace(
            "rerank.score", kind=SpanKind.INTERNAL, attributes=attributes
        ) as span:
            yield span

    @asynccontextmanager
    async def trace_hallucination_check(self, response_length: int, chunk_count: int):
        """Trace hallucination detection."""
        attributes = {
            "tyra.hallucination.response_length": response_length,
            "tyra.hallucination.chunk_count": chunk_count,
            "tyra.component": "hallucination_detector",
        }

        async with self.telemetry.trace(
            "hallucination.analyze", kind=SpanKind.INTERNAL, attributes=attributes
        ) as span:
            start_time = time.time()

            try:
                yield span
            finally:
                if span:
                    # Add analysis results as attributes
                    duration = time.time() - start_time
                    span.set_attribute("tyra.hallucination.analysis_time", duration)

    @asynccontextmanager
    async def trace_cache_operation(self, operation: str, cache_level: str, key: str):
        """Trace cache operations."""
        attributes = {
            "tyra.cache.operation": operation,
            "tyra.cache.level": cache_level,
            "tyra.cache.key_hash": str(hash(key)),  # Hash for privacy
            "tyra.component": "cache",
        }

        async with self.telemetry.trace(
            f"cache.{operation}", kind=SpanKind.INTERNAL, attributes=attributes
        ) as span:
            yield span

    def trace_function(
        self,
        name: Optional[str] = None,
        component: Optional[str] = None,
        capture_args: bool = False,
    ):
        """Decorator for tracing functions with custom attributes."""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            span_name = name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                attributes = {"tyra.function": func.__name__}

                if component:
                    attributes["tyra.component"] = component

                if capture_args:
                    # Capture safe representations of arguments
                    attributes["tyra.args_count"] = len(args)
                    attributes["tyra.kwargs_keys"] = ",".join(kwargs.keys())

                async with self.telemetry.trace(
                    span_name, kind=SpanKind.INTERNAL, attributes=attributes
                ) as span:
                    try:
                        result = await func(*args, **kwargs)

                        # Add result metadata if available
                        if span and hasattr(result, "__len__"):
                            span.set_attribute("tyra.result_size", len(result))

                        return result
                    except Exception as e:
                        if span:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                attributes = {"tyra.function": func.__name__}

                if component:
                    attributes["tyra.component"] = component

                with self.tracer.start_as_current_span(
                    span_name, kind=SpanKind.INTERNAL, attributes=attributes
                ) as span:
                    try:
                        result = func(*args, **kwargs)

                        if span and hasattr(result, "__len__"):
                            span.set_attribute("tyra.result_size", len(result))

                        return result
                    except Exception as e:
                        if span:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add an event to the current span."""
        span = get_current_span()
        if span and span.is_recording():
            span.add_event(
                name, attributes=attributes or {}, timestamp=timestamp or time.time_ns()
            )

    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute on current span."""
        span = get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, value)

    def record_exception(
        self, exception: Exception, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an exception on the current span."""
        span = get_current_span()
        if span and span.is_recording():
            span.record_exception(exception, attributes=attributes or {})
            span.set_status(Status(StatusCode.ERROR, str(exception)))

    def get_current_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context with baggage."""
        span = get_current_span()
        if not span or not span.is_recording():
            return None

        span_context = span.get_span_context()

        # Get all baggage
        current_baggage = {}
        ctx = context.get_current()
        for key, value in baggage.get_all(ctx).items():
            current_baggage[key] = value

        # Get span attributes
        attributes = {}
        if hasattr(span, "attributes"):
            attributes = dict(span.attributes)

        return TraceContext(
            trace_id=format(span_context.trace_id, "032x"),
            span_id=format(span_context.span_id, "016x"),
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            baggage=current_baggage,
            attributes=attributes,
        )

    def create_span_link(
        self, target_span_name: str, link_type: str = "related"
    ) -> None:
        """Create a link between current span and another span."""
        current_span = get_current_span()
        target_span = self._active_spans.get(target_span_name)

        if current_span and target_span:
            # Store link information
            link_key = f"{current_span.name}:{target_span.name}"
            if link_key not in self._span_links:
                self._span_links[link_key] = []

            self._span_links[link_key].append(
                {"type": link_type, "timestamp": time.time()}
            )

            # Add as span attribute
            current_span.set_attribute(f"tyra.link.{link_type}", target_span.name)

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile an operation and add timing to span."""
        start_time = time.time()
        span = get_current_span()

        try:
            yield
        finally:
            duration = time.time() - start_time

            if span and span.is_recording():
                span.set_attribute(f"tyra.profile.{operation_name}.ms", duration * 1000)

            # Store for analysis
            self._span_timings[operation_name] = duration

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of current trace."""
        ctx = self.get_current_trace_context()
        if not ctx:
            return {}

        return {
            "trace_id": ctx.trace_id,
            "span_id": ctx.span_id,
            "baggage": ctx.baggage,
            "active_spans": list(self._active_spans.keys()),
            "span_links": len(self._span_links),
            "timings": self._span_timings.copy(),
        }


# Global tracer instance
_enhanced_tracer: Optional[EnhancedTracer] = None


def get_tracer() -> EnhancedTracer:
    """Get the global enhanced tracer instance."""
    global _enhanced_tracer
    if _enhanced_tracer is None:
        _enhanced_tracer = EnhancedTracer()
    return _enhanced_tracer


# Convenience decorators
def traced_memory_operation(operation: str):
    """Decorator for tracing memory operations."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()

            # Extract agent_id if available
            agent_id = kwargs.get("agent_id")
            if not agent_id and len(args) > 1:
                # Check if second argument might be agent_id
                if isinstance(args[1], str):
                    agent_id = args[1]

            async with tracer.trace_memory_operation(operation, agent_id):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def traced_rag_component(component: str):
    """Decorator for tracing RAG components."""
    return get_tracer().trace_function(component=component)
