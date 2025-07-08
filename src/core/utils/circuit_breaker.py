"""
Circuit breaker pattern implementation for resilient database connections.

Provides automatic failure detection, fallback mechanisms, and self-healing
capabilities for database operations.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Operation timeout
    expected_exception: type = Exception  # Exception type that counts as failure


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    open_time: Optional[float] = None
    state_changes: Dict[str, int] = field(default_factory=dict)


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Raised when operation times out."""

    pass


class AsyncCircuitBreaker:
    """
    Async circuit breaker implementation with advanced features.

    Features:
    - Automatic failure detection and recovery
    - Configurable thresholds and timeouts
    - Detailed statistics and monitoring
    - Support for custom fallback functions
    - Thread-safe operation
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_func: Optional[Callable] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_func = fallback_func
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

        logger.info(
            "Circuit breaker initialized",
            name=name,
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout,
        )

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: When circuit is open
            CircuitBreakerTimeoutError: When operation times out
        """
        async with self._lock:
            self.stats.total_requests += 1

            # Check if circuit is open
            if self.stats.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    self.stats.failed_requests += 1
                    logger.warning(
                        "Circuit breaker is open, rejecting request",
                        name=self.name,
                        failure_count=self.stats.failure_count,
                    )

                    if self.fallback_func:
                        try:
                            return await self._execute_fallback(*args, **kwargs)
                        except Exception as e:
                            logger.error(
                                "Fallback function failed", name=self.name, error=str(e)
                            )
                            raise CircuitBreakerOpenError(
                                f"Circuit breaker {self.name} is open and fallback failed: {e}"
                            )

                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is open"
                    )

        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.config.timeout
            )

            await self._on_success()
            return result

        except asyncio.TimeoutError:
            await self._on_failure()
            raise CircuitBreakerTimeoutError(
                f"Operation timed out after {self.config.timeout}s"
            )
        except self.config.expected_exception as e:
            await self._on_failure()

            if self.fallback_func:
                try:
                    return await self._execute_fallback(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        "Fallback function failed",
                        name=self.name,
                        original_error=str(e),
                        fallback_error=str(fallback_error),
                    )
                    raise e

            raise e

    async def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function."""
        if asyncio.iscoroutinefunction(self.fallback_func):
            return await self.fallback_func(*args, **kwargs)
        else:
            return self.fallback_func(*args, **kwargs)

    async def _on_success(self):
        """Handle successful operation."""
        async with self._lock:
            self.stats.successful_requests += 1

            if self.stats.state == CircuitBreakerState.HALF_OPEN:
                self.stats.success_count += 1
                if self.stats.success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            else:
                # Reset failure count on success in closed state
                self.stats.failure_count = 0

    async def _on_failure(self):
        """Handle failed operation."""
        async with self._lock:
            self.stats.failed_requests += 1
            self.stats.failure_count += 1
            self.stats.last_failure_time = time.time()

            if (
                self.stats.state == CircuitBreakerState.CLOSED
                and self.stats.failure_count >= self.config.failure_threshold
            ):
                await self._transition_to_open()
            elif self.stats.state == CircuitBreakerState.HALF_OPEN:
                await self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.stats.last_failure_time is None:
            return False

        return (
            time.time() - self.stats.last_failure_time >= self.config.recovery_timeout
        )

    async def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        previous_state = self.stats.state
        self.stats.state = CircuitBreakerState.OPEN
        self.stats.open_time = time.time()
        self.stats.success_count = 0

        self._record_state_change(previous_state, CircuitBreakerState.OPEN)

        logger.warning(
            "Circuit breaker opened",
            name=self.name,
            failure_count=self.stats.failure_count,
            previous_state=previous_state.value,
        )

    async def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        previous_state = self.stats.state
        self.stats.state = CircuitBreakerState.HALF_OPEN
        self.stats.success_count = 0

        self._record_state_change(previous_state, CircuitBreakerState.HALF_OPEN)

        logger.info(
            "Circuit breaker transitioning to half-open",
            name=self.name,
            previous_state=previous_state.value,
        )

    async def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        previous_state = self.stats.state
        self.stats.state = CircuitBreakerState.CLOSED
        self.stats.failure_count = 0
        self.stats.success_count = 0
        self.stats.open_time = None

        self._record_state_change(previous_state, CircuitBreakerState.CLOSED)

        logger.info(
            "Circuit breaker closed",
            name=self.name,
            previous_state=previous_state.value,
        )

    def _record_state_change(
        self, from_state: CircuitBreakerState, to_state: CircuitBreakerState
    ):
        """Record state change for monitoring."""
        transition = f"{from_state.value}_to_{to_state.value}"
        self.stats.state_changes[transition] = (
            self.stats.state_changes.get(transition, 0) + 1
        )

    @asynccontextmanager
    async def protect(self, func: Callable[..., T], *args, **kwargs):
        """Context manager for protected function execution."""
        try:
            result = await self.call(func, *args, **kwargs)
            yield result
        except Exception as e:
            logger.error(
                "Protected function failed",
                name=self.name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def reset(self):
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            previous_state = self.stats.state
            self.stats.state = CircuitBreakerState.CLOSED
            self.stats.failure_count = 0
            self.stats.success_count = 0
            self.stats.last_failure_time = None
            self.stats.open_time = None

            logger.info(
                "Circuit breaker manually reset",
                name=self.name,
                previous_state=previous_state.value,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        uptime = None
        if self.stats.open_time:
            uptime = time.time() - self.stats.open_time

        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": (
                self.stats.successful_requests / self.stats.total_requests
                if self.stats.total_requests > 0
                else 0
            ),
            "last_failure_time": self.stats.last_failure_time,
            "open_time": self.stats.open_time,
            "uptime_since_open": uptime,
            "state_changes": self.stats.state_changes.copy(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }

    def is_healthy(self) -> bool:
        """Check if circuit breaker is in healthy state."""
        if self.stats.state == CircuitBreakerState.OPEN:
            return False

        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests
            return success_rate > 0.8  # 80% success rate threshold

        return True


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, AsyncCircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_func: Optional[Callable] = None,
    ) -> AsyncCircuitBreaker:
        """Get or create a circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = AsyncCircuitBreaker(name, config, fallback_func)
            return self._breakers[name]

    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        stats = {}
        async with self._lock:
            for name, breaker in self._breakers.items():
                stats[name] = breaker.get_stats()
        return stats

    async def reset_all(self):
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        healthy_count = 0
        total_count = len(self._breakers)

        async with self._lock:
            for breaker in self._breakers.values():
                if breaker.is_healthy():
                    healthy_count += 1

        return {
            "total_breakers": total_count,
            "healthy_breakers": healthy_count,
            "unhealthy_breakers": total_count - healthy_count,
            "overall_health": healthy_count / total_count if total_count > 0 else 1.0,
        }


# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()


async def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    fallback_func: Optional[Callable] = None,
) -> AsyncCircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    return await circuit_breaker_registry.get_breaker(name, config, fallback_func)
