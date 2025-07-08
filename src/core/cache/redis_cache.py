"""
Advanced Redis-based caching system with multi-level strategies.

Provides high-performance caching for embeddings, search results, and reranking
with automatic TTL management, compression, and circuit breaker protection.
"""

import asyncio
import json
import pickle
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..utils.config import get_settings
from ..utils.database import RedisManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache levels with different TTLs."""

    L1_EMBEDDINGS = "embeddings"  # 24 hours
    L2_SEARCH = "search"  # 1 hour
    L3_RERANK = "rerank"  # 30 minutes
    L4_HALLUCINATION = "hallucination"  # 15 minutes
    L5_GRAPH = "graph"  # 2 hours


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""

    key: str
    value: Any
    level: CacheLevel
    created_at: float
    expires_at: float
    hit_count: int = 0
    compressed: bool = False
    size_bytes: int = 0


class CacheMetrics:
    """Tracks cache performance metrics."""

    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.errors: int = 0
        self.total_size_bytes: int = 0
        self.compression_ratio: float = 1.0
        self._operation_times: List[float] = []
        self._max_operation_times = 1000

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_operation_time(self) -> float:
        """Calculate average operation time."""
        return (
            sum(self._operation_times) / len(self._operation_times)
            if self._operation_times
            else 0.0
        )

    def record_operation_time(self, operation_time: float):
        """Record operation timing."""
        self._operation_times.append(operation_time)
        if len(self._operation_times) > self._max_operation_times:
            self._operation_times = self._operation_times[-self._max_operation_times :]


class RedisCache:
    """
    Advanced Redis-based caching system.

    Features:
    - Multi-level caching with different TTLs
    - Automatic compression for large values
    - Circuit breaker protection
    - Performance metrics and monitoring
    - Batch operations for efficiency
    - Automatic cache warming
    - Memory-aware eviction
    """

    def __init__(self):
        self.redis_manager: Optional[RedisManager] = None
        self.settings = get_settings()
        self.metrics = CacheMetrics()

        # Cache configuration
        self.ttls = {
            CacheLevel.L1_EMBEDDINGS: timedelta(hours=24),
            CacheLevel.L2_SEARCH: timedelta(hours=1),
            CacheLevel.L3_RERANK: timedelta(minutes=30),
            CacheLevel.L4_HALLUCINATION: timedelta(minutes=15),
            CacheLevel.L5_GRAPH: timedelta(hours=2),
        }

        self.compression_threshold = 1024  # Compress values larger than 1KB
        self.max_cache_size_gb = 4  # Maximum cache size in GB
        self._initialized = False

    async def initialize(self, redis_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Redis cache."""
        try:
            config = redis_config or self.settings.cache.redis

            self.redis_manager = RedisManager(config)
            await self.redis_manager.initialize()

            # Set cache configuration
            self.compression_threshold = config.get("compression_threshold", 1024)
            self.max_cache_size_gb = config.get("max_size_gb", 4)

            # Configure TTLs from settings if available
            if hasattr(self.settings.cache, "ttls"):
                for level, ttl_seconds in self.settings.cache.ttls.items():
                    if hasattr(CacheLevel, level.upper()):
                        cache_level = getattr(CacheLevel, level.upper())
                        self.ttls[cache_level] = timedelta(seconds=ttl_seconds)

            self._initialized = True

            logger.info(
                "Redis cache initialized",
                compression_threshold=self.compression_threshold,
                max_size_gb=self.max_cache_size_gb,
                ttls={level.value: str(ttl) for level, ttl in self.ttls.items()},
            )

        except Exception as e:
            logger.error("Failed to initialize Redis cache", error=str(e))
            raise

    async def get(
        self, key: str, level: CacheLevel = CacheLevel.L2_SEARCH, default: Any = None
    ) -> Any:
        """Get value from cache with automatic decompression."""
        if not self._initialized:
            return default

        start_time = time.time()

        try:
            # Build namespaced key
            cache_key = self._build_key(key, level)

            # Get from Redis
            data = await self.redis_manager.execute_command("get", cache_key)

            if data is None:
                self.metrics.misses += 1
                return default

            # Deserialize and decompress if needed
            value = self._deserialize(data)

            # Update metrics
            self.metrics.hits += 1
            await self._increment_hit_count(cache_key)

            # Record timing
            self.metrics.record_operation_time(time.time() - start_time)

            return value

        except Exception as e:
            self.metrics.errors += 1
            logger.error("Cache get failed", key=key, level=level.value, error=str(e))
            return default

    async def set(
        self,
        key: str,
        value: Any,
        level: CacheLevel = CacheLevel.L2_SEARCH,
        ttl_override: Optional[int] = None,
    ) -> bool:
        """Set value in cache with automatic compression."""
        if not self._initialized:
            return False

        start_time = time.time()

        try:
            # Build namespaced key
            cache_key = self._build_key(key, level)

            # Serialize and compress if needed
            data, compressed, size_bytes = self._serialize(value)

            # Get TTL
            ttl = ttl_override or int(self.ttls[level].total_seconds())

            # Set in Redis with TTL
            await self.redis_manager.execute_command("setex", cache_key, ttl, data)

            # Store metadata
            metadata = {
                "created_at": time.time(),
                "expires_at": time.time() + ttl,
                "compressed": compressed,
                "size_bytes": size_bytes,
                "level": level.value,
            }

            await self.redis_manager.execute_command(
                "hset", f"{cache_key}:meta", mapping=metadata
            )

            # Update metrics
            self.metrics.total_size_bytes += size_bytes
            self.metrics.record_operation_time(time.time() - start_time)

            # Check cache size and evict if necessary
            await self._check_cache_size()

            return True

        except Exception as e:
            self.metrics.errors += 1
            logger.error("Cache set failed", key=key, level=level.value, error=str(e))
            return False

    async def delete(self, key: str, level: CacheLevel = CacheLevel.L2_SEARCH) -> bool:
        """Delete value from cache."""
        if not self._initialized:
            return False

        try:
            cache_key = self._build_key(key, level)

            # Delete key and metadata
            deleted = await self.redis_manager.execute_command("delete", cache_key)
            await self.redis_manager.execute_command("delete", f"{cache_key}:meta")

            if deleted:
                self.metrics.evictions += 1

            return bool(deleted)

        except Exception as e:
            self.metrics.errors += 1
            logger.error(
                "Cache delete failed", key=key, level=level.value, error=str(e)
            )
            return False

    async def batch_get(
        self, keys: List[str], level: CacheLevel = CacheLevel.L2_SEARCH
    ) -> Dict[str, Any]:
        """Get multiple values from cache efficiently."""
        if not self._initialized or not keys:
            return {}

        try:
            # Build namespaced keys
            cache_keys = [self._build_key(key, level) for key in keys]

            # Get all values in one operation
            values = await self.redis_manager.execute_command("mget", *cache_keys)

            # Process results
            results = {}
            for key, cache_key, value in zip(keys, cache_keys, values):
                if value is not None:
                    try:
                        results[key] = self._deserialize(value)
                        self.metrics.hits += 1
                        # Increment hit count asynchronously
                        asyncio.create_task(self._increment_hit_count(cache_key))
                    except Exception as e:
                        logger.error(
                            f"Failed to deserialize cache value for {key}: {e}"
                        )
                        self.metrics.misses += 1
                else:
                    self.metrics.misses += 1

            return results

        except Exception as e:
            self.metrics.errors += 1
            logger.error(
                "Batch cache get failed",
                keys_count=len(keys),
                level=level.value,
                error=str(e),
            )
            return {}

    async def batch_set(
        self,
        items: Dict[str, Any],
        level: CacheLevel = CacheLevel.L2_SEARCH,
        ttl_override: Optional[int] = None,
    ) -> int:
        """Set multiple values in cache efficiently."""
        if not self._initialized or not items:
            return 0

        try:
            ttl = ttl_override or int(self.ttls[level].total_seconds())
            success_count = 0

            # Use pipeline for efficiency
            pipeline_items = []

            for key, value in items.items():
                cache_key = self._build_key(key, level)
                data, compressed, size_bytes = self._serialize(value)

                pipeline_items.append((cache_key, ttl, data))

                # Store metadata
                metadata = {
                    "created_at": time.time(),
                    "expires_at": time.time() + ttl,
                    "compressed": compressed,
                    "size_bytes": size_bytes,
                    "level": level.value,
                }

                await self.redis_manager.execute_command(
                    "hset", f"{cache_key}:meta", mapping=metadata
                )

                self.metrics.total_size_bytes += size_bytes
                success_count += 1

            # Execute batch set
            for cache_key, ttl, data in pipeline_items:
                await self.redis_manager.execute_command("setex", cache_key, ttl, data)

            # Check cache size
            await self._check_cache_size()

            return success_count

        except Exception as e:
            self.metrics.errors += 1
            logger.error(
                "Batch cache set failed",
                items_count=len(items),
                level=level.value,
                error=str(e),
            )
            return 0

    async def clear_level(self, level: CacheLevel) -> int:
        """Clear all entries for a specific cache level."""
        if not self._initialized:
            return 0

        try:
            pattern = f"{level.value}:*"
            cursor = 0
            deleted_count = 0

            # Scan and delete keys
            while True:
                cursor, keys = await self.redis_manager.execute_command(
                    "scan", cursor, match=pattern, count=100
                )

                if keys:
                    deleted_count += await self.redis_manager.execute_command(
                        "delete", *keys
                    )

                    # Delete metadata keys
                    meta_keys = [f"{key}:meta" for key in keys]
                    await self.redis_manager.execute_command("delete", *meta_keys)

                if cursor == 0:
                    break

            self.metrics.evictions += deleted_count

            logger.info(
                "Cleared cache level", level=level.value, deleted_count=deleted_count
            )

            return deleted_count

        except Exception as e:
            self.metrics.errors += 1
            logger.error("Failed to clear cache level", level=level.value, error=str(e))
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            # Get Redis info
            info = await self.redis_manager.execute_command("info", "memory")

            # Get cache size by level
            level_stats = {}
            for level in CacheLevel:
                pattern = f"{level.value}:*"
                count = await self._count_keys(pattern)
                level_stats[level.value] = count

            return {
                "metrics": {
                    "hits": self.metrics.hits,
                    "misses": self.metrics.misses,
                    "hit_rate": self.metrics.hit_rate,
                    "evictions": self.metrics.evictions,
                    "errors": self.metrics.errors,
                    "total_size_bytes": self.metrics.total_size_bytes,
                    "compression_ratio": self.metrics.compression_ratio,
                    "avg_operation_time": self.metrics.avg_operation_time,
                },
                "levels": level_stats,
                "redis_memory": self._parse_redis_memory_info(info),
                "configuration": {
                    "compression_threshold": self.compression_threshold,
                    "max_cache_size_gb": self.max_cache_size_gb,
                    "ttls": {level.value: str(ttl) for level, ttl in self.ttls.items()},
                },
            }

        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"error": str(e)}

    def _build_key(self, key: str, level: CacheLevel) -> str:
        """Build namespaced cache key."""
        return f"{level.value}:{key}"

    def _serialize(self, value: Any) -> tuple[bytes, bool, int]:
        """Serialize and optionally compress value."""
        try:
            # Handle numpy arrays specially
            if isinstance(value, np.ndarray):
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # Try JSON first for better interoperability
                try:
                    data = json.dumps(value).encode("utf-8")
                except (TypeError, ValueError):
                    # Fall back to pickle for complex objects
                    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            original_size = len(data)
            compressed = False

            # Compress if over threshold
            if original_size > self.compression_threshold:
                compressed_data = zlib.compress(data, level=6)
                if (
                    len(compressed_data) < original_size * 0.9
                ):  # Only use if 10% smaller
                    data = compressed_data
                    compressed = True
                    self.metrics.compression_ratio = (
                        len(compressed_data) / original_size
                    )

            return data, compressed, len(data)

        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize and optionally decompress value."""
        try:
            # Check if compressed (zlib magic number)
            if data[:2] == b"\x78\x9c":
                data = zlib.decompress(data)

            # Try JSON first
            try:
                return json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)

        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise

    async def _increment_hit_count(self, cache_key: str) -> None:
        """Increment hit count for a cache entry."""
        try:
            await self.redis_manager.execute_command(
                "hincrby", f"{cache_key}:meta", "hit_count", 1
            )
        except Exception:
            pass  # Non-critical operation

    async def _check_cache_size(self) -> None:
        """Check cache size and evict if necessary."""
        try:
            # Get current memory usage
            info = await self.redis_manager.execute_command("info", "memory")
            used_memory = self._parse_redis_memory_info(info).get("used_memory_gb", 0)

            if used_memory > self.max_cache_size_gb * 0.9:  # 90% threshold
                logger.warning(
                    "Cache size approaching limit",
                    used_gb=used_memory,
                    max_gb=self.max_cache_size_gb,
                )

                # Trigger LRU eviction
                await self._evict_lru_entries(
                    int(self.max_cache_size_gb * 0.1 * 1024 * 1024 * 1024)
                )

        except Exception as e:
            logger.error("Failed to check cache size", error=str(e))

    async def _evict_lru_entries(self, bytes_to_free: int) -> None:
        """Evict least recently used entries."""
        try:
            freed_bytes = 0

            # Get all keys with metadata
            pattern = "*:meta"
            cursor = 0
            candidates = []

            while cursor != 0 or not candidates:
                cursor, meta_keys = await self.redis_manager.execute_command(
                    "scan", cursor, match=pattern, count=100
                )

                for meta_key in meta_keys:
                    metadata = await self.redis_manager.execute_command(
                        "hgetall", meta_key
                    )
                    if metadata:
                        cache_key = meta_key[:-5]  # Remove ":meta"
                        candidates.append(
                            {
                                "key": cache_key,
                                "hit_count": int(metadata.get("hit_count", 0)),
                                "created_at": float(metadata.get("created_at", 0)),
                                "size_bytes": int(metadata.get("size_bytes", 0)),
                            }
                        )

            # Sort by hit count and age
            candidates.sort(key=lambda x: (x["hit_count"], -x["created_at"]))

            # Evict until we free enough space
            for candidate in candidates:
                if freed_bytes >= bytes_to_free:
                    break

                await self.redis_manager.execute_command("delete", candidate["key"])
                await self.redis_manager.execute_command(
                    "delete", f"{candidate['key']}:meta"
                )

                freed_bytes += candidate["size_bytes"]
                self.metrics.evictions += 1

            logger.info(
                "LRU eviction completed",
                freed_bytes=freed_bytes,
                evicted_count=self.metrics.evictions,
            )

        except Exception as e:
            logger.error("LRU eviction failed", error=str(e))

    async def _count_keys(self, pattern: str) -> int:
        """Count keys matching pattern."""
        cursor = 0
        count = 0

        while True:
            cursor, keys = await self.redis_manager.execute_command(
                "scan", cursor, match=pattern, count=100
            )
            count += len(keys)

            if cursor == 0:
                break

        return count

    def _parse_redis_memory_info(self, info: str) -> Dict[str, Any]:
        """Parse Redis INFO memory output."""
        result = {}

        if isinstance(info, dict):
            # Already parsed
            return info

        for line in info.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)

                if key == "used_memory":
                    result["used_memory_bytes"] = int(value)
                    result["used_memory_gb"] = int(value) / (1024**3)
                elif key == "used_memory_human":
                    result["used_memory_human"] = value
                elif key == "maxmemory":
                    result["maxmemory_bytes"] = int(value)
                    result["maxmemory_gb"] = int(value) / (1024**3)

        return result

    async def warm_cache(
        self,
        warmup_func: Callable,
        keys: List[str],
        level: CacheLevel = CacheLevel.L2_SEARCH,
    ) -> int:
        """Warm cache with pre-computed values."""
        if not self._initialized:
            return 0

        warmed_count = 0

        for key in keys:
            try:
                # Check if already cached
                if await self.get(key, level) is None:
                    # Compute value
                    value = await warmup_func(key)

                    if value is not None:
                        # Cache it
                        if await self.set(key, value, level):
                            warmed_count += 1

            except Exception as e:
                logger.error(f"Failed to warm cache for key {key}: {e}")

        logger.info(
            "Cache warming completed",
            level=level.value,
            warmed_count=warmed_count,
            total_keys=len(keys),
        )

        return warmed_count

    async def close(self) -> None:
        """Close the Redis cache."""
        if self.redis_manager:
            await self.redis_manager.close()

        logger.info(
            "Redis cache closed",
            total_operations=self.metrics.hits + self.metrics.misses,
            hit_rate=self.metrics.hit_rate,
        )
