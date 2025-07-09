"""Cache manager for multi-level caching with Redis backend."""

import asyncio
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from ...utils.logger import get_logger
from ...utils.circuit_breaker import circuit_breaker
from ..observability.tracing import trace_method

logger = get_logger(__name__)


class CacheLevel:
    """Cache level definitions with TTLs."""
    EMBEDDINGS = ("embeddings", 86400)  # 24 hours
    SEARCH = ("search", 3600)  # 1 hour  
    RERANK = ("rerank", 1800)  # 30 minutes
    HALLUCINATION = ("hallucination", 900)  # 15 minutes
    GRAPH = ("graph", 7200)  # 2 hours
    

class CacheManager:
    """Manages multi-level caching with Redis backend."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 50,
        default_ttl: int = 3600,
        enable_compression: bool = True,
        compression_threshold: int = 1024  # bytes
    ):
        """Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
            max_connections: Maximum Redis connections
            default_ttl: Default TTL in seconds
            enable_compression: Enable value compression
            compression_threshold: Minimum size for compression
        """
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self._pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=False  # We handle encoding/decoding
            )
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            logger.info(f"Cache manager initialized with Redis at {self.redis_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise
            
    async def close(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
            
    @trace_method("cache_get")
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get(
        self,
        key: str,
        level: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            level: Optional cache level prefix
            
        Returns:
            Cached value or None if not found
        """
        if not self._client:
            logger.warning("Cache not initialized")
            return None
            
        try:
            full_key = self._make_key(key, level)
            value = await self._client.get(full_key)
            
            if value is None:
                self._stats['misses'] += 1
                return None
                
            self._stats['hits'] += 1
            
            # Update access time for LRU-like behavior
            await self._client.expire(full_key, self.default_ttl)
            
            return self._deserialize(value)
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._stats['errors'] += 1
            return None
            
    @trace_method("cache_set")
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[str] = None
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            level: Optional cache level
            
        Returns:
            True if successful
        """
        if not self._client:
            logger.warning("Cache not initialized")
            return False
            
        try:
            full_key = self._make_key(key, level)
            serialized = self._serialize(value)
            
            ttl = ttl or self._get_ttl_for_level(level)
            
            await self._client.setex(
                full_key,
                ttl,
                serialized
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self._stats['errors'] += 1
            return False
            
    async def delete(
        self,
        key: str,
        level: Optional[str] = None
    ) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            level: Optional cache level
            
        Returns:
            True if deleted
        """
        if not self._client:
            return False
            
        try:
            full_key = self._make_key(key, level)
            result = await self._client.delete(full_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
            
    async def clear_level(self, level: str) -> int:
        """Clear all cache entries for a specific level.
        
        Args:
            level: Cache level to clear
            
        Returns:
            Number of deleted keys
        """
        if not self._client:
            return 0
            
        try:
            pattern = f"{level}:*"
            keys = []
            
            # Use SCAN to avoid blocking
            async for key in self._client.scan_iter(pattern):
                keys.append(key)
                
            if keys:
                return await self._client.delete(*keys)
                
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
            
    async def get_many(
        self,
        keys: List[str],
        level: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            level: Optional cache level
            
        Returns:
            Dictionary of key-value pairs
        """
        if not self._client or not keys:
            return {}
            
        try:
            full_keys = [self._make_key(k, level) for k in keys]
            values = await self._client.mget(full_keys)
            
            result = {}
            for key, full_key, value in zip(keys, full_keys, values):
                if value is not None:
                    self._stats['hits'] += 1
                    result[key] = self._deserialize(value)
                else:
                    self._stats['misses'] += 1
                    
            return result
            
        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            self._stats['errors'] += 1
            return {}
            
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
        level: Optional[str] = None
    ) -> bool:
        """Set multiple values in cache.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Optional TTL in seconds
            level: Optional cache level
            
        Returns:
            True if all successful
        """
        if not self._client or not items:
            return False
            
        try:
            ttl = ttl or self._get_ttl_for_level(level)
            
            # Use pipeline for atomic operation
            pipe = self._client.pipeline()
            
            for key, value in items.items():
                full_key = self._make_key(key, level)
                serialized = self._serialize(value)
                pipe.setex(full_key, ttl, serialized)
                
            results = await pipe.execute()
            return all(results)
            
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            return False
            
    def _make_key(self, key: str, level: Optional[str] = None) -> str:
        """Create full cache key with optional level prefix."""
        if level:
            return f"{level}:{key}"
        return key
        
    def _get_ttl_for_level(self, level: Optional[str]) -> int:
        """Get TTL for cache level."""
        level_ttls = {
            "embeddings": 86400,  # 24 hours
            "search": 3600,       # 1 hour
            "rerank": 1800,       # 30 minutes
            "hallucination": 900, # 15 minutes
            "graph": 7200        # 2 hours
        }
        
        if level and level in level_ttls:
            return level_ttls[level]
            
        return self.default_ttl
        
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        # Use pickle for complex objects
        serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compress if enabled and above threshold
        if self.enable_compression and len(serialized) > self.compression_threshold:
            import zlib
            compressed = zlib.compress(serialized)
            # Add compression marker
            return b'Z' + compressed
            
        return b'P' + serialized  # P for pickle
        
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if not data:
            return None
            
        # Check format marker
        marker = data[0:1]
        content = data[1:]
        
        if marker == b'Z':
            # Compressed data
            import zlib
            content = zlib.decompress(content)
            
        return pickle.loads(content)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / max(1, total)
        
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'errors': self._stats['errors'],
            'hit_rate': hit_rate,
            'total_requests': total
        }
        
    async def health_check(self) -> bool:
        """Check cache health."""
        if not self._client:
            return False
            
        try:
            await self._client.ping()
            return True
        except:
            return False
            
    def create_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        # Create stable key from args and kwargs
        key_parts = [str(arg) for arg in args]
        
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}:{v}" for k, v in sorted_kwargs)
            
        key_string = ":".join(key_parts)
        
        # Hash if too long
        if len(key_string) > 200:
            return hashlib.sha256(key_string.encode()).hexdigest()
            
        return key_string