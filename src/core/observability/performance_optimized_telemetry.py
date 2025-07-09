"""Performance-optimized telemetry system with minimal overhead."""

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Optional, Callable, List, Tuple, Union
from functools import wraps
from threading import Lock
from dataclasses import dataclass
from enum import Enum
import weakref

from ...utils.logger import get_logger
from ...utils.config import settings

logger = get_logger(__name__)


class TelemetryLevel(Enum):
    """Telemetry collection levels."""
    DISABLED = 0
    MINIMAL = 1    # Only critical metrics
    STANDARD = 2   # Standard metrics
    DETAILED = 3   # All metrics including debug


@dataclass
class MetricEntry:
    """Lightweight metric entry."""
    timestamp: float
    value: float
    tags: Optional[Dict[str, str]] = None


@dataclass
class SpanData:
    """Minimal span data for performance."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class PerformanceOptimizedTelemetry:
    """High-performance telemetry system with adaptive overhead control."""
    
    def __init__(
        self,
        max_metrics_buffer: int = 10000,
        max_spans_buffer: int = 5000,
        sampling_rate: float = 1.0,
        adaptive_sampling: bool = True,
        telemetry_level: TelemetryLevel = TelemetryLevel.STANDARD
    ):
        """Initialize performance-optimized telemetry.
        
        Args:
            max_metrics_buffer: Maximum metrics to buffer
            max_spans_buffer: Maximum spans to buffer
            sampling_rate: Initial sampling rate (0.0 to 1.0)
            adaptive_sampling: Enable adaptive sampling based on load
            telemetry_level: Telemetry collection level
        """
        self.max_metrics_buffer = max_metrics_buffer
        self.max_spans_buffer = max_spans_buffer
        self.sampling_rate = sampling_rate
        self.adaptive_sampling = adaptive_sampling
        self.telemetry_level = telemetry_level
        
        # Thread-safe collections
        self._metrics_buffer = deque(maxlen=max_metrics_buffer)
        self._spans_buffer = deque(maxlen=max_spans_buffer)
        self._metrics_lock = Lock()
        self._spans_lock = Lock()
        
        # Performance counters
        self._counter_values = defaultdict(int)
        self._gauge_values = defaultdict(float)
        self._histogram_buckets = defaultdict(lambda: deque(maxlen=1000))
        
        # Adaptive sampling state
        self._cpu_usage_samples = deque(maxlen=100)
        self._memory_usage_samples = deque(maxlen=100)
        self._last_adaptation_time = time.time()
        self._adaptation_interval = 30.0  # seconds
        
        # Active spans (weak references to avoid memory leaks)
        self._active_spans = weakref.WeakSet()
        
        # Fast path flags
        self._telemetry_enabled = True
        self._spans_enabled = True
        self._metrics_enabled = True
        
        # Performance tracking
        self._total_overhead_time = 0.0
        self._operation_count = 0
        
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._telemetry_enabled and self.telemetry_level != TelemetryLevel.DISABLED
        
    def should_sample(self) -> bool:
        """Fast sampling decision."""
        if not self.is_enabled():
            return False
            
        # Fast path for full sampling
        if self.sampling_rate >= 1.0:
            return True
            
        # Fast path for no sampling
        if self.sampling_rate <= 0.0:
            return False
            
        # Simple hash-based sampling (faster than random)
        return hash(time.time()) % 1000 < (self.sampling_rate * 1000)
        
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metric_type: str = "gauge"
    ) -> None:
        """Record a metric with minimal overhead."""
        if not self._metrics_enabled or not self.should_sample():
            return
            
        start_time = time.time()
        
        try:
            if metric_type == "counter":
                self._counter_values[name] += value
            elif metric_type == "gauge":
                self._gauge_values[name] = value
            elif metric_type == "histogram":
                self._histogram_buckets[name].append(value)
                
            # Only buffer if level is detailed
            if self.telemetry_level == TelemetryLevel.DETAILED:
                entry = MetricEntry(
                    timestamp=time.time(),
                    value=value,
                    tags=tags
                )
                
                with self._metrics_lock:
                    self._metrics_buffer.append((name, entry))
                    
        finally:
            # Track telemetry overhead
            self._total_overhead_time += time.time() - start_time
            self._operation_count += 1
            
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, tags, "counter")
        
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, tags, "gauge")
        
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        self.record_metric(name, value, tags, "histogram")
        
    @contextmanager
    def span(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Create a performance-optimized span context manager."""
        if not self._spans_enabled or not self.should_sample():
            yield None
            return
            
        span_data = SpanData(
            name=name,
            start_time=time.time(),
            tags=tags
        )
        
        self._active_spans.add(span_data)
        
        try:
            yield span_data
        except Exception as e:
            span_data.error = str(e)
            raise
        finally:
            span_data.end_time = time.time()
            
            # Record span duration
            if span_data.end_time:
                duration = span_data.end_time - span_data.start_time
                self.record_histogram(f"span_duration_{name}", duration)
                
            # Buffer span if detailed level
            if self.telemetry_level == TelemetryLevel.DETAILED:
                with self._spans_lock:
                    self._spans_buffer.append(span_data)
                    
    def time_function(self, name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Decorator for timing functions with minimal overhead."""
        def decorator(func: Callable) -> Callable:
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    if not self.should_sample():
                        return await func(*args, **kwargs)
                        
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        self.record_histogram(f"function_duration_{func_name}", time.time() - start_time)
                        return result
                    except Exception as e:
                        self.increment_counter(f"function_error_{func_name}")
                        raise
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    if not self.should_sample():
                        return func(*args, **kwargs)
                        
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        self.record_histogram(f"function_duration_{func_name}", time.time() - start_time)
                        return result
                    except Exception as e:
                        self.increment_counter(f"function_error_{func_name}")
                        raise
                return sync_wrapper
                
        return decorator
        
    def adapt_sampling_rate(self) -> None:
        """Adapt sampling rate based on system load."""
        if not self.adaptive_sampling:
            return
            
        current_time = time.time()
        if current_time - self._last_adaptation_time < self._adaptation_interval:
            return
            
        self._last_adaptation_time = current_time
        
        try:
            import psutil
            
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            self._cpu_usage_samples.append(cpu_percent)
            self._memory_usage_samples.append(memory_percent)
            
            # Calculate average usage
            avg_cpu = sum(self._cpu_usage_samples) / len(self._cpu_usage_samples)
            avg_memory = sum(self._memory_usage_samples) / len(self._memory_usage_samples)
            
            # Adapt sampling rate based on load
            if avg_cpu > 80 or avg_memory > 80:
                # High load - reduce sampling
                self.sampling_rate = max(0.1, self.sampling_rate * 0.8)
                logger.info(f"Reduced telemetry sampling rate to {self.sampling_rate:.2f} due to high load")
            elif avg_cpu < 50 and avg_memory < 50:
                # Low load - increase sampling
                self.sampling_rate = min(1.0, self.sampling_rate * 1.1)
                logger.debug(f"Increased telemetry sampling rate to {self.sampling_rate:.2f}")
                
        except ImportError:
            # psutil not available, use telemetry overhead as proxy
            overhead_ratio = self._total_overhead_time / max(1, self._operation_count)
            if overhead_ratio > 0.001:  # 1ms average overhead
                self.sampling_rate = max(0.1, self.sampling_rate * 0.9)
                logger.warning(f"Reduced telemetry sampling due to high overhead: {overhead_ratio:.6f}s")
                
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return {
            "counters": dict(self._counter_values),
            "gauges": dict(self._gauge_values),
            "histograms": {
                name: {
                    "count": len(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
                for name, values in self._histogram_buckets.items()
            }
        }
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get telemetry performance statistics."""
        return {
            "enabled": self.is_enabled(),
            "sampling_rate": self.sampling_rate,
            "telemetry_level": self.telemetry_level.name,
            "metrics_buffer_size": len(self._metrics_buffer),
            "spans_buffer_size": len(self._spans_buffer),
            "active_spans": len(self._active_spans),
            "total_overhead_time": self._total_overhead_time,
            "operation_count": self._operation_count,
            "avg_overhead_per_op": self._total_overhead_time / max(1, self._operation_count),
            "overhead_percentage": (self._total_overhead_time / max(1, time.time())) * 100
        }
        
    def flush_buffers(self) -> Tuple[List[Tuple[str, MetricEntry]], List[SpanData]]:
        """Flush and return buffered data."""
        metrics = []
        spans = []
        
        if self._metrics_enabled:
            with self._metrics_lock:
                metrics = list(self._metrics_buffer)
                self._metrics_buffer.clear()
                
        if self._spans_enabled:
            with self._spans_lock:
                spans = list(self._spans_buffer)
                self._spans_buffer.clear()
                
        return metrics, spans
        
    def reset_counters(self) -> None:
        """Reset all counters and gauges."""
        self._counter_values.clear()
        self._gauge_values.clear()
        self._histogram_buckets.clear()
        
    def disable_telemetry(self) -> None:
        """Disable telemetry collection."""
        self._telemetry_enabled = False
        logger.info("Telemetry disabled")
        
    def enable_telemetry(self) -> None:
        """Enable telemetry collection."""
        self._telemetry_enabled = True
        logger.info("Telemetry enabled")
        
    def set_telemetry_level(self, level: TelemetryLevel) -> None:
        """Set telemetry collection level."""
        self.telemetry_level = level
        logger.info(f"Telemetry level set to {level.name}")
        
    def configure_for_production(self) -> None:
        """Configure telemetry for production use."""
        self.sampling_rate = 0.1  # 10% sampling
        self.telemetry_level = TelemetryLevel.MINIMAL
        self.adaptive_sampling = True
        self._spans_enabled = False  # Disable span collection
        logger.info("Telemetry configured for production")
        
    def configure_for_development(self) -> None:
        """Configure telemetry for development use."""
        self.sampling_rate = 1.0  # 100% sampling
        self.telemetry_level = TelemetryLevel.DETAILED
        self.adaptive_sampling = False
        self._spans_enabled = True
        logger.info("Telemetry configured for development")


# Global telemetry instance
_telemetry_instance: Optional[PerformanceOptimizedTelemetry] = None


def get_telemetry() -> PerformanceOptimizedTelemetry:
    """Get the global telemetry instance."""
    global _telemetry_instance
    
    if _telemetry_instance is None:
        # Configure based on environment
        config = getattr(settings, 'observability', {})
        
        telemetry_level = TelemetryLevel.STANDARD
        if config.get('environment') == 'production':
            telemetry_level = TelemetryLevel.MINIMAL
        elif config.get('environment') == 'development':
            telemetry_level = TelemetryLevel.DETAILED
            
        _telemetry_instance = PerformanceOptimizedTelemetry(
            sampling_rate=config.get('sampling_rate', 1.0),
            adaptive_sampling=config.get('adaptive_sampling', True),
            telemetry_level=telemetry_level
        )
        
        # Configure for environment
        if config.get('environment') == 'production':
            _telemetry_instance.configure_for_production()
        else:
            _telemetry_instance.configure_for_development()
            
    return _telemetry_instance


# Convenience functions
def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None, metric_type: str = "gauge") -> None:
    """Record a metric."""
    get_telemetry().record_metric(name, value, tags, metric_type)
    

def increment_counter(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter."""
    get_telemetry().increment_counter(name, value, tags)
    

def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge value."""
    get_telemetry().set_gauge(name, value, tags)
    

def record_histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a histogram value."""
    get_telemetry().record_histogram(name, value, tags)
    

def span(name: str, tags: Optional[Dict[str, str]] = None):
    """Create a telemetry span."""
    return get_telemetry().span(name, tags)
    

def time_function(name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing functions."""
    return get_telemetry().time_function(name, tags)


# Periodic telemetry adaptation
async def telemetry_adaptation_loop():
    """Background task for telemetry adaptation."""
    while True:
        try:
            get_telemetry().adapt_sampling_rate()
            await asyncio.sleep(30)  # Adapt every 30 seconds
        except Exception as e:
            logger.error(f"Telemetry adaptation error: {e}")
            await asyncio.sleep(60)  # Wait longer on error