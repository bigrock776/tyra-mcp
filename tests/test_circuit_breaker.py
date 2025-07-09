"""
Comprehensive unit tests for Circuit Breaker.

Tests failure detection, circuit states, recovery mechanisms, and resilience patterns.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional

from src.core.utils.circuit_breaker import CircuitBreaker, CircuitBreakerState, CircuitBreakerConfig


class TestCircuitBreaker:
    """Test Circuit Breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with test configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=5.0,
            expected_exception=Exception,
            half_open_max_calls=2
        )
        return CircuitBreaker(config)

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state_success(self, circuit_breaker):
        """Test circuit breaker in closed state with successful calls."""
        # Setup successful operation
        async def successful_operation():
            return "success"
        
        # Execute multiple successful calls
        for _ in range(5):
            result = await circuit_breaker.call(successful_operation)
            assert result == "success"
        
        # Verify circuit breaker remains closed
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self, circuit_breaker):
        """Test circuit breaker failure counting and threshold detection."""
        # Setup failing operation
        async def failing_operation():
            raise Exception("Operation failed")
        
        # Execute failing calls below threshold
        for i in range(circuit_breaker.config.failure_threshold - 1):
            with pytest.raises(Exception, match="Operation failed"):
                await circuit_breaker.call(failing_operation)
            
            # Should still be closed
            assert circuit_breaker.state == CircuitBreakerState.CLOSED
            assert circuit_breaker.failure_count == i + 1
        
        # Final failure should open circuit
        with pytest.raises(Exception, match="Operation failed"):
            await circuit_breaker.call(failing_operation)
        
        # Verify circuit is now open
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == circuit_breaker.config.failure_threshold

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state_rejection(self, circuit_breaker):
        """Test circuit breaker rejecting calls when open."""
        # Setup failing operation to open circuit
        async def failing_operation():
            raise Exception("Operation failed")
        
        # Trigger circuit to open
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        # Verify circuit is open
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Test rejection of new calls
        async def any_operation():
            return "should not execute"
        
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await circuit_breaker.call(any_operation)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self, circuit_breaker):
        """Test transition from open to half-open state."""
        # Setup failing operation to open circuit
        async def failing_operation():
            raise Exception("Operation failed")
        
        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)
        
        # Next call should transition to half-open
        async def test_operation():
            return "testing recovery"
        
        result = await circuit_breaker.call(test_operation)
        
        # Verify transition to half-open and successful call
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
        assert result == "testing recovery"

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_success_recovery(self, circuit_breaker):
        """Test recovery from half-open to closed state on success."""
        # Open circuit first
        async def failing_operation():
            raise Exception("Operation failed")
        
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)
        
        # Successful operations in half-open state
        async def successful_operation():
            return "success"
        
        # Execute successful calls up to half-open limit
        for i in range(circuit_breaker.config.half_open_max_calls):
            result = await circuit_breaker.call(successful_operation)
            assert result == "success"
            
            if i < circuit_breaker.config.half_open_max_calls - 1:
                assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Should transition back to closed
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure_reopen(self, circuit_breaker):
        """Test circuit reopening on failure in half-open state."""
        # Open circuit first
        async def failing_operation():
            raise Exception("Operation failed")
        
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)
        
        # One successful call to enter half-open
        async def successful_operation():
            return "success"
        
        await circuit_breaker.call(successful_operation)
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Failure in half-open should reopen circuit
        with pytest.raises(Exception, match="Operation failed"):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_database_operations(self, circuit_breaker):
        """Test circuit breaker protecting database operations."""
        # Mock database client
        db_client = AsyncMock()
        
        # Setup database operation that can fail
        async def database_query(query: str):
            result = await db_client.execute(query)
            if result is None:
                raise Exception("Database connection failed")
            return result
        
        # Test successful operations
        db_client.execute.return_value = {"rows": 10}
        
        for _ in range(3):
            result = await circuit_breaker.call(lambda: database_query("SELECT * FROM memories"))
            assert result["rows"] == 10
        
        # Test failure scenario
        db_client.execute.return_value = None
        
        # Should fail and eventually open circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception, match="Database connection failed"):
                await circuit_breaker.call(lambda: database_query("SELECT * FROM memories"))
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_embedding_operations(self, circuit_breaker):
        """Test circuit breaker protecting embedding operations."""
        # Mock embedding provider
        embedding_provider = AsyncMock()
        
        # Setup embedding operation
        async def generate_embedding(text: str):
            result = await embedding_provider.encode(text)
            if result is None:
                raise Exception("Embedding model not available")
            return result
        
        # Test successful operations
        embedding_provider.encode.return_value = [0.1] * 1024
        
        result = await circuit_breaker.call(lambda: generate_embedding("test text"))
        assert len(result) == 1024
        
        # Test failure scenario (model fails)
        embedding_provider.encode.side_effect = Exception("CUDA out of memory")
        
        # Should fail and open circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception, match="CUDA out of memory"):
                await circuit_breaker.call(lambda: generate_embedding("test text"))
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_collection(self, circuit_breaker):
        """Test metrics collection for circuit breaker operations."""
        # Setup operations
        async def successful_operation():
            return "success"
        
        async def failing_operation():
            raise Exception("failure")
        
        # Execute mixed operations
        await circuit_breaker.call(successful_operation)  # Success
        
        try:
            await circuit_breaker.call(failing_operation)  # Failure
        except Exception:
            pass
        
        await circuit_breaker.call(successful_operation)  # Success
        
        # Get metrics
        metrics = circuit_breaker.get_metrics()
        
        # Verify metrics
        assert metrics["total_calls"] == 3
        assert metrics["successful_calls"] == 2
        assert metrics["failed_calls"] == 1
        assert metrics["current_state"] == CircuitBreakerState.CLOSED.value
        assert metrics["failure_rate"] == 1/3

    @pytest.mark.asyncio
    async def test_circuit_breaker_custom_exceptions(self, circuit_breaker):
        """Test circuit breaker with custom exception handling."""
        # Create circuit breaker for specific exception
        db_breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            expected_exception=ConnectionError
        ))
        
        # Setup operations
        async def db_connection_error():
            raise ConnectionError("Database unreachable")
        
        async def other_error():
            raise ValueError("Invalid input")
        
        # ConnectionError should count towards failure
        with pytest.raises(ConnectionError):
            await db_breaker.call(db_connection_error)
        
        assert db_breaker.failure_count == 1
        
        # ValueError should not count (different exception type)
        with pytest.raises(ValueError):
            await db_breaker.call(other_error)
        
        assert db_breaker.failure_count == 1  # Unchanged

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback_mechanism(self, circuit_breaker):
        """Test circuit breaker with fallback mechanism."""
        # Setup primary and fallback operations
        primary_service = AsyncMock()
        fallback_service = AsyncMock()
        
        async def primary_operation():
            result = await primary_service.process()
            if result is None:
                raise Exception("Primary service failed")
            return result
        
        async def fallback_operation():
            return await fallback_service.process()
        
        # Test normal operation
        primary_service.process.return_value = "primary_result"
        result = await circuit_breaker.call(primary_operation)
        assert result == "primary_result"
        
        # Fail primary service to open circuit
        primary_service.process.return_value = None
        fallback_service.process.return_value = "fallback_result"
        
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(primary_operation)
        
        # Circuit is now open, use fallback
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Direct fallback call (bypassing circuit breaker)
        fallback_result = await fallback_operation()
        assert fallback_result == "fallback_result"

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_access(self, circuit_breaker):
        """Test circuit breaker behavior under concurrent access."""
        # Setup operation
        call_counter = 0
        
        async def concurrent_operation(operation_id: int):
            nonlocal call_counter
            call_counter += 1
            await asyncio.sleep(0.01)  # Simulate work
            
            if operation_id % 3 == 0:  # Every 3rd operation fails
                raise Exception(f"Operation {operation_id} failed")
            
            return f"result_{operation_id}"
        
        # Execute concurrent operations
        tasks = [
            circuit_breaker.call(lambda i=i: concurrent_operation(i))
            for i in range(10)
        ]
        
        results = []
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                results.append(f"error: {str(e)}")
        
        # Verify some operations succeeded and some failed
        successes = [r for r in results if r.startswith("result_")]
        failures = [r for r in results if r.startswith("error:")]
        
        assert len(successes) > 0
        assert len(failures) > 0
        assert call_counter == 10

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_usage(self, circuit_breaker):
        """Test circuit breaker used as decorator."""
        # Create decorator version
        @circuit_breaker.protect
        async def protected_function(value: int):
            if value < 0:
                raise ValueError("Negative value not allowed")
            return value * 2
        
        # Test successful calls
        result = await protected_function(5)
        assert result == 10
        
        # Test failing calls
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await protected_function(-1)
        
        # Circuit should be open now
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout_handling(self, circuit_breaker):
        """Test circuit breaker with operation timeouts."""
        # Setup slow operation
        async def slow_operation(delay: float):
            await asyncio.sleep(delay)
            return "completed"
        
        # Configure circuit breaker with timeout
        timeout_breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            call_timeout=0.5  # 500ms timeout
        ))
        
        # Fast operation should succeed
        result = await timeout_breaker.call(lambda: slow_operation(0.1))
        assert result == "completed"
        
        # Slow operation should timeout and count as failure
        with pytest.raises(Exception):  # TimeoutError or similar
            await timeout_breaker.call(lambda: slow_operation(1.0))
        
        assert timeout_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_health_check_integration(self, circuit_breaker):
        """Test circuit breaker integration with health checks."""
        # Setup service with health check
        service = AsyncMock()
        
        async def service_operation():
            if not await service.health_check():
                raise Exception("Service unhealthy")
            return await service.process()
        
        # Test healthy service
        service.health_check.return_value = True
        service.process.return_value = "healthy_result"
        
        result = await circuit_breaker.call(service_operation)
        assert result == "healthy_result"
        
        # Test unhealthy service
        service.health_check.return_value = False
        
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception, match="Service unhealthy"):
                await circuit_breaker.call(service_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_functionality(self, circuit_breaker):
        """Test manual circuit breaker reset."""
        # Open the circuit
        async def failing_operation():
            raise Exception("Failure")
        
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Manual reset
        circuit_breaker.reset()
        
        # Verify circuit is closed and counters reset
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        
        # Should accept calls again
        async def successful_operation():
            return "success after reset"
        
        result = await circuit_breaker.call(successful_operation)
        assert result == "success after reset"

    @pytest.mark.asyncio
    async def test_circuit_breaker_configuration_validation(self):
        """Test circuit breaker configuration validation."""
        # Test valid configuration
        valid_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=10.0,
            half_open_max_calls=3
        )
        cb = CircuitBreaker(valid_config)
        assert cb.config.failure_threshold == 5
        
        # Test invalid configurations
        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)  # Must be positive
        
        with pytest.raises(ValueError):
            CircuitBreakerConfig(recovery_timeout=-1.0)  # Must be positive

    @pytest.mark.asyncio
    async def test_circuit_breaker_monitoring_callbacks(self, circuit_breaker):
        """Test circuit breaker state change callbacks."""
        state_changes = []
        
        # Setup state change callback
        def on_state_change(old_state, new_state, reason):
            state_changes.append({
                "old": old_state,
                "new": new_state,
                "reason": reason
            })
        
        circuit_breaker.on_state_change = on_state_change
        
        # Trigger state changes
        async def failing_operation():
            raise Exception("Failure")
        
        # Open circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        # Wait for recovery
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)
        
        # Trigger half-open
        async def successful_operation():
            return "success"
        
        await circuit_breaker.call(successful_operation)
        
        # Verify state changes were recorded
        assert len(state_changes) >= 2
        assert any(change["new"] == CircuitBreakerState.OPEN for change in state_changes)
        assert any(change["new"] == CircuitBreakerState.HALF_OPEN for change in state_changes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])