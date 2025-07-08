"""
Advanced database connection management with optimizations and monitoring.

Provides high-performance connection pooling, health monitoring, and
resilience patterns for PostgreSQL, Redis, and Memgraph.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import asyncpg
import redis.asyncio as redis
from gqlalchemy import Memgraph

from .circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker,
)
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for database connections."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_query_time: float = 0.0
    max_query_time: float = 0.0
    min_query_time: float = float("inf")


class DatabaseManager(ABC):
    """Abstract base class for database managers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close all database connections."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the database."""
        pass

    @abstractmethod
    def get_stats(self) -> ConnectionStats:
        """Get connection statistics."""
        pass


class PostgreSQLManager(DatabaseManager):
    """
    Advanced PostgreSQL connection manager with circuit breaker,
    connection pooling, and health monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self.circuit_breaker: Optional[AsyncCircuitBreaker] = None
        self.stats = ConnectionStats()
        self._query_times: List[float] = []
        self._max_query_times = 1000  # Keep last 1000 query times

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool with circuit breaker."""
        try:
            # Create circuit breaker
            cb_config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=3,
                timeout=30.0,
                expected_exception=(
                    asyncpg.PostgresError,
                    asyncpg.ConnectionDoesNotExistError,
                    OSError,
                ),
            )

            self.circuit_breaker = await get_circuit_breaker(
                "postgresql", cb_config, fallback_func=self._fallback_query
            )

            # Create connection pool
            dsn = self._build_dsn()
            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=self.config.get("min_connections", 5),
                max_size=self.config.get("pool_size", 20),
                max_queries=self.config.get("max_queries", 50000),
                max_inactive_connection_lifetime=self.config.get("max_lifetime", 300),
                command_timeout=self.config.get("command_timeout", 10),
                server_settings={
                    "jit": "off",  # Disable JIT for faster startup
                    "application_name": "tyra_mcp_memory_server",
                },
            )

            # Verify connection
            await self._verify_connection()

            logger.info(
                "PostgreSQL connection pool initialized",
                min_size=self.config.get("min_connections", 5),
                max_size=self.config.get("pool_size", 20),
                host=self.config.get("host"),
                database=self.config.get("database"),
            )

        except Exception as e:
            logger.error(
                "Failed to initialize PostgreSQL connection pool",
                error=str(e),
                config=self._safe_config(),
            )
            raise

    def _build_dsn(self) -> str:
        """Build PostgreSQL DSN from configuration."""
        return (
            f"postgresql://{self.config['username']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )

    def _safe_config(self) -> Dict[str, Any]:
        """Get config with sensitive data masked."""
        safe = self.config.copy()
        if "password" in safe:
            safe["password"] = "***"
        return safe

    async def _verify_connection(self) -> None:
        """Verify connection by running a simple query."""
        async with self.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

    async def _fallback_query(self, query: str, *args, **kwargs):
        """Fallback function for failed queries."""
        logger.warning(
            "Using fallback for PostgreSQL query",
            query=query[:100] + "..." if len(query) > 100 else query,
        )
        # Return empty result for fallback
        return []

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a connection from the pool with circuit breaker protection."""
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        start_time = time.time()
        connection = None

        try:
            # Acquire connection through circuit breaker
            connection = await self.circuit_breaker.call(self.pool.acquire)

            self.stats.active_connections += 1
            yield connection

        except Exception as e:
            self.stats.failed_connections += 1
            logger.error(
                "Failed to acquire PostgreSQL connection",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            if connection:
                try:
                    await self.pool.release(connection)
                    self.stats.active_connections -= 1

                    # Update timing stats
                    query_time = time.time() - start_time
                    self._update_timing_stats(query_time)

                except Exception as e:
                    logger.error(
                        "Failed to release PostgreSQL connection", error=str(e)
                    )

    async def execute_query(
        self, query: str, *args, fetch_mode: str = "all"  # "all", "one", "val", "none"
    ) -> Any:
        """Execute a query with circuit breaker protection and monitoring."""
        start_time = time.time()

        try:
            async with self.get_connection() as conn:
                self.stats.total_queries += 1

                if fetch_mode == "all":
                    result = await conn.fetch(query, *args)
                elif fetch_mode == "one":
                    result = await conn.fetchrow(query, *args)
                elif fetch_mode == "val":
                    result = await conn.fetchval(query, *args)
                elif fetch_mode == "none":
                    await conn.execute(query, *args)
                    result = None
                else:
                    raise ValueError(f"Invalid fetch_mode: {fetch_mode}")

                self.stats.successful_queries += 1
                return result

        except Exception as e:
            self.stats.failed_queries += 1
            query_time = time.time() - start_time

            logger.error(
                "PostgreSQL query failed",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e),
                query_time=query_time,
            )
            raise

    async def execute_batch(
        self, query: str, args_list: List[tuple], batch_size: int = 1000
    ) -> None:
        """Execute batch queries efficiently."""
        async with self.get_connection() as conn:
            # Process in batches to avoid memory issues
            for i in range(0, len(args_list), batch_size):
                batch = args_list[i : i + batch_size]
                await conn.executemany(query, batch)

                logger.debug(
                    "Processed batch",
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch),
                    total_batches=(len(args_list) + batch_size - 1) // batch_size,
                )

    def _update_timing_stats(self, query_time: float):
        """Update query timing statistics."""
        self._query_times.append(query_time)

        # Keep only recent query times
        if len(self._query_times) > self._max_query_times:
            self._query_times = self._query_times[-self._max_query_times :]

        # Update stats
        self.stats.max_query_time = max(self.stats.max_query_time, query_time)
        self.stats.min_query_time = min(self.stats.min_query_time, query_time)

        if self._query_times:
            self.stats.avg_query_time = sum(self._query_times) / len(self._query_times)

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            start_time = time.time()

            # Test basic connectivity
            await self.execute_query("SELECT 1", fetch_mode="val")

            # Test pgvector extension
            try:
                await self.execute_query(
                    "SELECT extname FROM pg_extension WHERE extname = 'vector'",
                    fetch_mode="val",
                )
                pgvector_available = True
            except Exception:
                pgvector_available = False

            # Get pool stats
            pool_stats = {
                "size": self.pool.get_size(),
                "max_size": self.pool.get_max_size(),
                "min_size": self.pool.get_min_size(),
                "idle_connections": self.pool.get_idle_size(),
            }

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": response_time,
                "pgvector_available": pgvector_available,
                "pool_stats": pool_stats,
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
                "connection_stats": self.get_stats().__dict__,
            }

        except Exception as e:
            logger.error("PostgreSQL health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
            }

    def get_stats(self) -> ConnectionStats:
        """Get detailed connection statistics."""
        if self.pool:
            self.stats.total_connections = self.pool.get_size()
            self.stats.idle_connections = self.pool.get_idle_size()

        return self.stats

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")


class RedisManager(DatabaseManager):
    """
    Advanced Redis connection manager with circuit breaker and monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[redis.ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.circuit_breaker: Optional[AsyncCircuitBreaker] = None
        self.stats = ConnectionStats()

    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            # Create circuit breaker
            cb_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=15.0,
                success_threshold=2,
                timeout=10.0,
                expected_exception=(
                    redis.RedisError,
                    redis.ConnectionError,
                    redis.TimeoutError,
                ),
            )

            self.circuit_breaker = await get_circuit_breaker(
                "redis", cb_config, fallback_func=self._fallback_operation
            )

            # Create connection pool
            self.pool = redis.ConnectionPool(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6379),
                password=self.config.get("password"),
                db=self.config.get("db", 0),
                max_connections=self.config.get("pool_size", 50),
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)

            # Verify connection
            await self._verify_connection()

            logger.info(
                "Redis connection pool initialized",
                host=self.config.get("host"),
                port=self.config.get("port"),
                db=self.config.get("db", 0),
                pool_size=self.config.get("pool_size", 50),
            )

        except Exception as e:
            logger.error(
                "Failed to initialize Redis connection pool",
                error=str(e),
                config=self._safe_config(),
            )
            raise

    def _safe_config(self) -> Dict[str, Any]:
        """Get config with sensitive data masked."""
        safe = self.config.copy()
        if "password" in safe:
            safe["password"] = "***"
        return safe

    async def _verify_connection(self) -> None:
        """Verify Redis connection."""
        await self.client.ping()

    async def _fallback_operation(self, *args, **kwargs):
        """Fallback for Redis operations."""
        logger.warning("Using fallback for Redis operation")
        return None

    async def execute_command(self, command: str, *args, **kwargs) -> Any:
        """Execute Redis command with circuit breaker protection."""
        if not self.client:
            raise RuntimeError("Redis client not initialized")

        start_time = time.time()

        try:
            self.stats.total_queries += 1

            # Execute command through circuit breaker
            result = await self.circuit_breaker.call(
                getattr(self.client, command.lower()), *args, **kwargs
            )

            self.stats.successful_queries += 1

            # Update timing
            query_time = time.time() - start_time
            self.stats.avg_query_time = (
                self.stats.avg_query_time * (self.stats.successful_queries - 1)
                + query_time
            ) / self.stats.successful_queries

            return result

        except Exception as e:
            self.stats.failed_queries += 1
            logger.error(
                "Redis command failed",
                command=command,
                error=str(e),
                query_time=time.time() - start_time,
            )
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        try:
            start_time = time.time()

            # Test basic connectivity
            await self.client.ping()

            # Get Redis info
            info = await self.client.info()

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": response_time,
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
                "connection_stats": self.get_stats().__dict__,
            }

        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
            }

    def get_stats(self) -> ConnectionStats:
        """Get Redis connection statistics."""
        if self.pool:
            # Redis connection pool doesn't expose detailed stats
            self.stats.total_connections = self.pool.max_connections

        return self.stats

    async def close(self) -> None:
        """Close Redis connections."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("Redis connection pool closed")


class MemgraphManager(DatabaseManager):
    """
    Advanced Memgraph connection manager with circuit breaker.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[Memgraph] = None
        self.circuit_breaker: Optional[AsyncCircuitBreaker] = None
        self.stats = ConnectionStats()

    async def initialize(self) -> None:
        """Initialize Memgraph connection."""
        try:
            # Create circuit breaker
            cb_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=20.0,
                success_threshold=2,
                timeout=15.0,
                expected_exception=(Exception,),  # Memgraph uses generic exceptions
            )

            self.circuit_breaker = await get_circuit_breaker(
                "memgraph", cb_config, fallback_func=self._fallback_query
            )

            # Create Memgraph client
            self.client = Memgraph(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 7687),
                username=self.config.get("username", "memgraph"),
                password=self.config.get("password"),
                encrypted=self.config.get("encrypted", False),
            )

            # Verify connection
            await self._verify_connection()

            logger.info(
                "Memgraph connection initialized",
                host=self.config.get("host"),
                port=self.config.get("port"),
            )

        except Exception as e:
            logger.error(
                "Failed to initialize Memgraph connection",
                error=str(e),
                config=self._safe_config(),
            )
            raise

    def _safe_config(self) -> Dict[str, Any]:
        """Get config with sensitive data masked."""
        safe = self.config.copy()
        if "password" in safe:
            safe["password"] = "***"
        return safe

    async def _verify_connection(self) -> None:
        """Verify Memgraph connection."""
        # Simple query to test connection
        list(self.client.execute("RETURN 1"))

    async def _fallback_query(self, query: str, *args, **kwargs):
        """Fallback for Memgraph queries."""
        logger.warning(
            "Using fallback for Memgraph query",
            query=query[:100] + "..." if len(query) > 100 else query,
        )
        return []

    async def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query with circuit breaker protection."""
        if not self.client:
            raise RuntimeError("Memgraph client not initialized")

        start_time = time.time()

        try:
            self.stats.total_queries += 1

            # Execute query through circuit breaker
            result = await self.circuit_breaker.call(
                self._execute_sync_query, query, parameters
            )

            self.stats.successful_queries += 1

            # Update timing
            query_time = time.time() - start_time
            self.stats.avg_query_time = (
                self.stats.avg_query_time * (self.stats.successful_queries - 1)
                + query_time
            ) / self.stats.successful_queries

            return result

        except Exception as e:
            self.stats.failed_queries += 1
            logger.error(
                "Memgraph query failed",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e),
                query_time=time.time() - start_time,
            )
            raise

    def _execute_sync_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute synchronous Cypher query."""
        result = self.client.execute(query, parameters or {})
        return [dict(record) for record in result]

    async def health_check(self) -> Dict[str, Any]:
        """Perform Memgraph health check."""
        try:
            start_time = time.time()

            # Test basic connectivity
            await self.execute_query("RETURN 1 as test")

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": response_time,
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
                "connection_stats": self.get_stats().__dict__,
            }

        except Exception as e:
            logger.error("Memgraph health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
            }

    def get_stats(self) -> ConnectionStats:
        """Get Memgraph connection statistics."""
        return self.stats

    async def close(self) -> None:
        """Close Memgraph connection."""
        if self.client:
            self.client.close()
        logger.info("Memgraph connection closed")
