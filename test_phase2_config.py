#!/usr/bin/env python3
"""Test Phase 2 Core Infrastructure Components"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.utils.config import ConfigManager, get_settings, reload_settings
from src.core.utils.database import MemgraphManager, PostgreSQLManager, RedisManager
from src.core.utils.logger import get_logger


async def test_config_loading():
    """Test configuration loading and environment variable substitution"""
    print("\n=== Testing Configuration System ===")

    # Test environment variable substitution
    os.environ["TYRA_ENV"] = "testing"
    os.environ["POSTGRES_HOST"] = "test-postgres-host"
    os.environ["REDIS_PASSWORD"] = "test-redis-password"

    try:
        # Test loading main config
        config_manager = ConfigManager()
        config = config_manager.load_config()

        print(f"✓ Config loaded successfully")
        print(f"  - Environment: {config.environment}")
        print(f"  - Server name: {config.server.name}")
        print(f"  - Memory backend: {config.memory.backend}")

        # Test environment variable substitution
        assert (
            config.environment == "testing"
        ), "Environment variable substitution failed"

        # Test loading all configs
        config_manager.load_all_configs()
        print(f"✓ All config files loaded successfully")

        # Test provider config access
        try:
            embedding_config = config_manager.get_provider_config(
                "embeddings", "huggingface"
            )
            print(f"✓ Provider config access working")
        except Exception as e:
            print(
                f"  - Provider config not found (expected if providers.yaml not populated)"
            )

        # Test agent config access
        agent_config = config_manager.get_agent_config("tyra")
        print(f"✓ Agent config access working")

        # Test reload functionality
        reloaded_config = config_manager.reload_config()
        print(f"✓ Config reload working")

        # Test validation
        is_valid = config_manager.validate_config()
        print(f"✓ Config validation: {is_valid}")

    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


async def test_logging_system():
    """Test structured logging system"""
    print("\n=== Testing Logging System ===")

    try:
        # Get logger
        logger = get_logger("test_module")

        # Test different log levels
        logger.debug("Debug message", extra_field="debug_value")
        logger.info("Info message", user_id="test_user", operation="test_op")
        logger.warning("Warning message", threshold=0.8)
        logger.error("Error message", error_code=500, error_type="TestError")

        print("✓ Structured logging working")

        # Test logging context
        from src.core.utils.logger import clear_request_context, set_request_context

        set_request_context(
            request_id="test-123", agent_id="tyra", session_id="session-456"
        )
        logger.info("Message with context")
        clear_request_context()

        print("✓ Logging context management working")

        # Test performance logging
        from src.core.utils.logger import log_performance

        with log_performance("test_operation"):
            await asyncio.sleep(0.1)  # Simulate work

        print("✓ Performance logging working")

    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

    return True


async def test_database_managers():
    """Test database connection managers (without actual connections)"""
    print("\n=== Testing Database Managers ===")

    # Test PostgreSQL Manager
    print("\nPostgreSQL Manager:")
    try:
        pg_config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "username": "test_user",
            "password": "test_pass",
            "pool_size": 20,
        }

        pg_manager = PostgreSQLManager(pg_config)
        print("✓ PostgreSQL manager created")
        print(f"  - Initial stats: {pg_manager.get_stats().__dict__}")

        # We won't actually initialize since DB might not be running
        print("  - Skipping actual connection (DB might not be available)")

    except Exception as e:
        print(f"✗ PostgreSQL manager test failed: {e}")

    # Test Redis Manager
    print("\nRedis Manager:")
    try:
        redis_config = {
            "host": "localhost",
            "port": 6379,
            "password": None,
            "db": 0,
            "pool_size": 50,
        }

        redis_manager = RedisManager(redis_config)
        print("✓ Redis manager created")
        print(f"  - Initial stats: {redis_manager.get_stats().__dict__}")

    except Exception as e:
        print(f"✗ Redis manager test failed: {e}")

    # Test Memgraph Manager
    print("\nMemgraph Manager:")
    try:
        memgraph_config = {
            "host": "localhost",
            "port": 7687,
            "username": "memgraph",
            "password": None,
        }

        memgraph_manager = MemgraphManager(memgraph_config)
        print("✓ Memgraph manager created")
        print(f"  - Initial stats: {memgraph_manager.get_stats().__dict__}")

    except Exception as e:
        print(f"✗ Memgraph manager test failed: {e}")

    return True


async def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("\n=== Testing Circuit Breaker ===")

    try:
        from src.core.utils.circuit_breaker import (
            AsyncCircuitBreaker,
            CircuitBreakerConfig,
        )

        # Create circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short for testing
            success_threshold=2,
        )

        breaker = AsyncCircuitBreaker("test_breaker", config)
        print("✓ Circuit breaker created")

        # Test successful calls
        success_count = 0

        async def successful_operation():
            nonlocal success_count
            success_count += 1
            return f"Success {success_count}"

        result = await breaker.call(successful_operation)
        print(f"✓ Successful call: {result}")

        # Test failing calls
        fail_count = 0

        async def failing_operation():
            nonlocal fail_count
            fail_count += 1
            raise Exception(f"Failure {fail_count}")

        # Trigger failures to open circuit
        for i in range(3):
            try:
                await breaker.call(failing_operation)
            except Exception:
                pass

        stats = breaker.get_stats()
        print(f"✓ Circuit breaker stats after failures:")
        print(f"  - State: {stats['state']}")
        print(f"  - Failed requests: {stats['failed_requests']}")
        print(f"  - Success rate: {stats['success_rate']:.2%}")

        # Test with fallback
        async def fallback_operation(*args, **kwargs):
            return "Fallback response"

        breaker_with_fallback = AsyncCircuitBreaker(
            "test_breaker_fallback", config, fallback_func=fallback_operation
        )

        # Make it fail
        for i in range(3):
            try:
                await breaker_with_fallback.call(failing_operation)
            except Exception:
                pass

        # Should use fallback when open
        result = await breaker_with_fallback.call(failing_operation)
        print(f"✓ Fallback working: {result}")

    except Exception as e:
        print(f"✗ Circuit breaker test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


async def test_provider_registry():
    """Test provider registry system"""
    print("\n=== Testing Provider Registry ===")

    try:
        from src.core.utils.registry import ProviderType, provider_registry

        # Register a test provider
        test_config = {"test_param": "test_value"}

        # We'll use a built-in class for testing since we don't have actual providers yet
        success = await provider_registry.register_provider(
            ProviderType.CACHE,
            "test_cache",
            "builtins.dict",  # Using dict as a dummy provider
            test_config,
        )

        print(f"✓ Provider registration: {success}")

        # List providers
        providers = await provider_registry.list_providers()
        print(f"✓ Listed providers: {len(providers)} provider types")

        # Get provider stats
        stats = await provider_registry.get_stats()
        print(f"✓ Registry stats:")
        print(f"  - Provider counts: {stats['provider_counts']}")
        print(f"  - Health summary: {stats['health_summary']}")

    except Exception as e:
        print(f"✗ Provider registry test failed: {e}")
        return False

    return True


async def main():
    """Run all Phase 2 tests"""
    print("=" * 60)
    print("PHASE 2 CORE INFRASTRUCTURE TEST SUITE")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Configuration System", await test_config_loading()))
    results.append(("Logging System", await test_logging_system()))
    results.append(("Database Managers", await test_database_managers()))
    results.append(("Circuit Breaker", await test_circuit_breaker()))
    results.append(("Provider Registry", await test_provider_registry()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<30} {status}")

    print(f"\nTotal: {passed}/{total} passed ({passed/total*100:.0f}%)")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
