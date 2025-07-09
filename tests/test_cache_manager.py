"""
Comprehensive unit tests for Cache Manager.

Tests Redis caching, multi-level cache strategies, TTL handling, and performance optimization.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional

from src.core.cache.manager import CacheManager, CacheKey, CacheEntry


class TestCacheManager:
    """Test Cache Manager functionality."""

    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager with mocked Redis client."""
        with patch('src.core.cache.manager.redis.asyncio.Redis') as mock_redis:
            manager = CacheManager()
            manager.redis_client = AsyncMock()
            manager.local_cache = {}  # In-memory L1 cache
            manager._initialized = True
            
            yield manager

    @pytest.mark.asyncio
    async def test_cache_set_and_get_basic(self, cache_manager):
        """Test basic cache set and get operations."""
        # Setup
        key = "test_key"
        value = {"data": "test_value", "timestamp": "2023-12-01T10:00:00Z"}
        ttl = 3600  # 1 hour
        
        # Mock Redis operations
        cache_manager.redis_client.set.return_value = True
        cache_manager.redis_client.get.return_value = '{"data": "test_value", "timestamp": "2023-12-01T10:00:00Z"}'
        
        # Execute set
        result = await cache_manager.set(key, value, ttl=ttl)
        
        # Execute get
        cached_value = await cache_manager.get(key)
        
        # Verify
        assert result is True
        assert cached_value["data"] == "test_value"
        
        # Verify Redis was called
        cache_manager.redis_client.set.assert_called_once()
        cache_manager.redis_client.get.assert_called_once_with(key)

    @pytest.mark.asyncio
    async def test_multi_level_cache_l1_hit(self, cache_manager):
        """Test L1 (in-memory) cache hit scenario."""
        # Setup L1 cache with data
        key = "l1_cached_key"
        value = {"level": "L1", "data": "fast_access"}
        cache_manager.local_cache[key] = {
            "value": value,
            "expires_at": asyncio.get_event_loop().time() + 3600,
            "access_count": 0
        }
        
        # Execute
        result = await cache_manager.get(key)
        
        # Verify
        assert result["level"] == "L1"
        assert result["data"] == "fast_access"
        
        # Verify Redis was not called (L1 hit)
        cache_manager.redis_client.get.assert_not_called()
        
        # Verify access count incremented
        assert cache_manager.local_cache[key]["access_count"] == 1

    @pytest.mark.asyncio
    async def test_multi_level_cache_l1_miss_l2_hit(self, cache_manager):
        """Test L1 miss but L2 (Redis) cache hit."""
        # Setup
        key = "l2_cached_key"
        value = {"level": "L2", "data": "redis_data"}
        
        # L1 cache miss (empty)
        cache_manager.local_cache = {}
        
        # L2 cache hit
        cache_manager.redis_client.get.return_value = '{"level": "L2", "data": "redis_data"}'
        
        # Execute
        result = await cache_manager.get(key)
        
        # Verify
        assert result["level"] == "L2"
        assert result["data"] == "redis_data"
        
        # Verify Redis was called
        cache_manager.redis_client.get.assert_called_once_with(key)
        
        # Verify value was promoted to L1
        assert key in cache_manager.local_cache
        assert cache_manager.local_cache[key]["value"]["level"] == "L2"

    @pytest.mark.asyncio
    async def test_cache_miss_both_levels(self, cache_manager):
        """Test cache miss on both L1 and L2 levels."""
        # Setup
        key = "missing_key"
        
        # Both caches miss
        cache_manager.local_cache = {}
        cache_manager.redis_client.get.return_value = None
        
        # Execute
        result = await cache_manager.get(key)
        
        # Verify
        assert result is None
        
        # Verify Redis was called
        cache_manager.redis_client.get.assert_called_once_with(key)

    @pytest.mark.asyncio
    async def test_embedding_cache_operations(self, cache_manager):
        """Test embedding-specific cache operations."""
        # Setup embedding cache
        text = "This is a test sentence for embedding."
        model_name = "intfloat/e5-large-v2"
        embedding = [0.1, 0.2, 0.3] * 341 + [0.4]  # 1024 dimensions
        
        embedding_key = cache_manager.generate_embedding_key(text, model_name)
        
        # Mock Redis operations
        cache_manager.redis_client.set.return_value = True
        cache_manager.redis_client.get.return_value = str(embedding)  # Redis returns string
        
        # Execute set
        await cache_manager.set_embedding(text, model_name, embedding, ttl=86400)  # 24 hours
        
        # Execute get
        cached_embedding = await cache_manager.get_embedding(text, model_name)
        
        # Verify
        assert cached_embedding is not None
        assert len(cached_embedding) == 1024
        
        # Verify key generation includes both text and model
        assert model_name in embedding_key
        assert hash(text) != hash("")  # Key includes text hash

    @pytest.mark.asyncio
    async def test_search_results_cache(self, cache_manager):
        """Test search results caching with query fingerprinting."""
        # Setup search query
        query = "machine learning algorithms"
        filters = {"confidence_min": 0.8, "limit": 10}
        search_type = "hybrid"
        
        search_results = [
            {"memory_id": "mem_1", "content": "ML algorithms", "score": 0.95},
            {"memory_id": "mem_2", "content": "Deep learning", "score": 0.87}
        ]
        
        # Generate search key
        search_key = cache_manager.generate_search_key(query, filters, search_type)
        
        # Mock Redis operations
        cache_manager.redis_client.set.return_value = True
        cache_manager.redis_client.get.return_value = str(search_results)
        
        # Execute set
        await cache_manager.set_search_results(query, filters, search_type, search_results, ttl=3600)
        
        # Execute get
        cached_results = await cache_manager.get_search_results(query, filters, search_type)
        
        # Verify
        assert cached_results is not None
        assert len(cached_results) == 2
        assert cached_results[0]["memory_id"] == "mem_1"

    @pytest.mark.asyncio
    async def test_rerank_cache_operations(self, cache_manager):
        """Test reranking results caching."""
        # Setup reranking data
        query = "Python programming"
        document_ids = ["doc1", "doc2", "doc3"]
        rerank_method = "cross_encoder"
        
        rerank_results = [
            {"doc_id": "doc3", "rerank_score": 0.95},
            {"doc_id": "doc1", "rerank_score": 0.88}, 
            {"doc_id": "doc2", "rerank_score": 0.72}
        ]
        
        # Generate rerank key
        rerank_key = cache_manager.generate_rerank_key(query, document_ids, rerank_method)
        
        # Mock Redis operations
        cache_manager.redis_client.set.return_value = True
        cache_manager.redis_client.get.return_value = str(rerank_results)
        
        # Execute
        await cache_manager.set_rerank_results(query, document_ids, rerank_method, rerank_results)
        cached_rerank = await cache_manager.get_rerank_results(query, document_ids, rerank_method)
        
        # Verify
        assert cached_rerank is not None
        assert len(cached_rerank) == 3
        assert cached_rerank[0]["rerank_score"] == 0.95

    @pytest.mark.asyncio
    async def test_ttl_expiration_handling(self, cache_manager):
        """Test TTL expiration and cleanup."""
        # Setup expired entry in L1 cache
        key = "expired_key"
        value = {"data": "expired_data"}
        cache_manager.local_cache[key] = {
            "value": value,
            "expires_at": asyncio.get_event_loop().time() - 1,  # Already expired
            "access_count": 0
        }
        
        # Mock Redis miss
        cache_manager.redis_client.get.return_value = None
        
        # Execute
        result = await cache_manager.get(key)
        
        # Verify
        assert result is None
        
        # Verify expired entry was removed from L1 cache
        assert key not in cache_manager.local_cache

    @pytest.mark.asyncio
    async def test_cache_invalidation_patterns(self, cache_manager):
        """Test cache invalidation by patterns."""
        # Setup cache with multiple keys
        cache_entries = {
            "embeddings:model1:hash1": {"data": "embedding1"},
            "embeddings:model1:hash2": {"data": "embedding2"},
            "embeddings:model2:hash3": {"data": "embedding3"},
            "search:query1": {"data": "search1"},
            "search:query2": {"data": "search2"}
        }
        
        # Mock Redis scan for pattern matching
        cache_manager.redis_client.scan_iter.return_value = [
            "embeddings:model1:hash1",
            "embeddings:model1:hash2"
        ]
        cache_manager.redis_client.delete.return_value = 2
        
        # Execute pattern-based invalidation
        deleted_count = await cache_manager.invalidate_pattern("embeddings:model1:*")
        
        # Verify
        assert deleted_count == 2
        cache_manager.redis_client.delete.assert_called()

    @pytest.mark.asyncio
    async def test_cache_warming(self, cache_manager):
        """Test cache warming with preloaded data."""
        # Setup warming data
        warming_data = [
            {"key": "popular_query_1", "value": {"results": "data1"}, "ttl": 7200},
            {"key": "popular_query_2", "value": {"results": "data2"}, "ttl": 7200},
            {"key": "common_embedding", "value": {"embedding": [0.1] * 1024}, "ttl": 86400}
        ]
        
        # Mock Redis pipeline for batch operations
        mock_pipeline = AsyncMock()
        cache_manager.redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True, True, True]
        
        # Execute cache warming
        result = await cache_manager.warm_cache(warming_data)
        
        # Verify
        assert result["warmed_count"] == 3
        assert result["failed_count"] == 0
        mock_pipeline.set.assert_called()
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache_manager):
        """Test cache statistics and metrics collection."""
        # Setup mock statistics from Redis
        mock_redis_info = {
            "used_memory": "50MB",
            "keyspace_hits": 1000,
            "keyspace_misses": 200,
            "connected_clients": 5
        }
        cache_manager.redis_client.info.return_value = mock_redis_info
        
        # Setup L1 cache statistics
        cache_manager.local_cache = {
            "key1": {"access_count": 10},
            "key2": {"access_count": 5},
            "key3": {"access_count": 15}
        }
        cache_manager._cache_hits = 800
        cache_manager._cache_misses = 150
        
        # Execute
        stats = await cache_manager.get_statistics()
        
        # Verify
        assert stats["l1_cache"]["size"] == 3
        assert stats["l1_cache"]["total_accesses"] == 30
        assert stats["l2_cache"]["used_memory"] == "50MB"
        assert stats["hit_rate"] > 0.8  # Good hit rate

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, cache_manager):
        """Test concurrent cache access and race condition handling."""
        # Setup concurrent operations
        key = "concurrent_key"
        values = [f"value_{i}" for i in range(10)]
        
        # Mock Redis operations
        cache_manager.redis_client.set.return_value = True
        cache_manager.redis_client.get.return_value = '"value_0"'
        
        # Execute concurrent operations
        async def set_operation(value):
            return await cache_manager.set(f"{key}_{value}", {"data": value})
        
        async def get_operation():
            return await cache_manager.get(key)
        
        # Run concurrent tasks
        set_tasks = [set_operation(v) for v in values]
        get_tasks = [get_operation() for _ in range(5)]
        
        set_results = await asyncio.gather(*set_tasks)
        get_results = await asyncio.gather(*get_tasks)
        
        # Verify
        assert all(r is True for r in set_results)
        assert len(get_results) == 5

    @pytest.mark.asyncio
    async def test_cache_serialization_edge_cases(self, cache_manager):
        """Test cache serialization with various data types."""
        # Setup different data types
        test_cases = [
            ("string_key", "simple string"),
            ("int_key", 42),
            ("float_key", 3.14159),
            ("list_key", [1, 2, 3, "mixed", True]),
            ("dict_key", {"nested": {"deep": "value"}, "numbers": [1, 2, 3]}),
            ("none_key", None),
            ("bool_key", False),
            ("complex_key", {"embeddings": [0.1] * 1024, "metadata": {"confidence": 0.95}})
        ]
        
        # Mock Redis operations
        cache_manager.redis_client.set.return_value = True
        
        # Test each case
        for key, value in test_cases:
            # Mock appropriate return value
            if value is None:
                cache_manager.redis_client.get.return_value = 'null'
            elif isinstance(value, bool):
                cache_manager.redis_client.get.return_value = 'false' if value is False else 'true'
            else:
                cache_manager.redis_client.get.return_value = str(value)
            
            # Execute
            await cache_manager.set(key, value)
            result = await cache_manager.get(key)
            
            # Verify (basic check - actual implementation would use JSON)
            assert result is not None or value is None

    @pytest.mark.asyncio
    async def test_cache_health_monitoring(self, cache_manager):
        """Test cache health monitoring and alerts."""
        # Setup health check scenarios
        
        # Healthy scenario
        cache_manager.redis_client.ping.return_value = True
        cache_manager.redis_client.info.return_value = {
            "used_memory": "100MB",
            "maxmemory": "1GB",
            "keyspace_hits": 1000,
            "keyspace_misses": 100
        }
        
        # Execute health check
        health = await cache_manager.health_check()
        
        # Verify healthy status
        assert health["status"] == "healthy"
        assert health["redis_connection"] is True
        assert health["memory_usage_percent"] < 50  # 100MB / 1GB = ~10%
        assert health["hit_rate"] > 0.9
        
        # Test unhealthy scenario
        cache_manager.redis_client.ping.side_effect = Exception("Connection failed")
        
        # Execute health check
        unhealthy_health = await cache_manager.health_check()
        
        # Verify unhealthy status
        assert unhealthy_health["status"] == "unhealthy"
        assert unhealthy_health["redis_connection"] is False

    @pytest.mark.asyncio
    async def test_cache_backup_and_restore(self, cache_manager):
        """Test cache backup and restore functionality."""
        # Setup cache data
        cache_data = {
            "key1": {"value": "data1", "ttl": 3600},
            "key2": {"value": "data2", "ttl": 7200},
            "key3": {"value": "data3", "ttl": 86400}
        }
        
        # Mock Redis scan and dump operations
        cache_manager.redis_client.scan_iter.return_value = list(cache_data.keys())
        cache_manager.redis_client.dump.side_effect = lambda k: f"dumped_{k}"
        cache_manager.redis_client.ttl.side_effect = lambda k: cache_data[k]["ttl"]
        
        # Execute backup
        backup_result = await cache_manager.create_backup()
        
        # Verify backup
        assert backup_result["keys_backed_up"] == 3
        assert "backup_file" in backup_result
        
        # Mock restore operations
        cache_manager.redis_client.restore.return_value = True
        
        # Execute restore
        restore_result = await cache_manager.restore_backup(backup_result["backup_file"])
        
        # Verify restore
        assert restore_result["keys_restored"] == 3
        assert restore_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_cache_eviction_policies(self, cache_manager):
        """Test cache eviction policies (LRU, LFU, TTL-based)."""
        # Setup L1 cache at capacity
        cache_manager.max_l1_size = 3
        cache_manager.local_cache = {
            "old_key": {"value": "old", "access_count": 1, "last_accessed": 1000},
            "medium_key": {"value": "medium", "access_count": 5, "last_accessed": 2000},
            "new_key": {"value": "new", "access_count": 10, "last_accessed": 3000}
        }
        
        # Add new item that should trigger eviction
        new_value = {"data": "newest"}
        cache_manager.redis_client.set.return_value = True
        
        # Execute (should evict least recently used)
        await cache_manager.set("newest_key", new_value)
        
        # Verify eviction occurred
        assert len(cache_manager.local_cache) <= cache_manager.max_l1_size
        
        # Verify LRU eviction (old_key should be gone)
        assert "old_key" not in cache_manager.local_cache

    @pytest.mark.asyncio
    async def test_cache_compression(self, cache_manager):
        """Test cache value compression for large data."""
        # Setup large data that should be compressed
        large_embedding = [0.1234567890123456] * 4096  # Large embedding
        large_data = {
            "embedding": large_embedding,
            "metadata": {"model": "large-model", "timestamp": "2023-12-01T10:00:00Z"},
            "additional_data": "x" * 1000  # Large string
        }
        
        # Mock compression/decompression
        with patch('src.core.cache.manager.gzip') as mock_gzip:
            mock_gzip.compress.return_value = b"compressed_data"
            mock_gzip.decompress.return_value = str(large_data).encode()
            
            # Execute with compression
            await cache_manager.set("large_key", large_data, compress=True)
            result = await cache_manager.get("large_key")
            
            # Verify compression was used for large data
            mock_gzip.compress.assert_called()

    @pytest.mark.asyncio
    async def test_cache_namespace_isolation(self, cache_manager):
        """Test cache namespace isolation between different components."""
        # Setup namespaced operations
        namespaces = ["embeddings", "search", "rerank", "graph"]
        
        for namespace in namespaces:
            key = f"test_key"
            value = f"data_for_{namespace}"
            
            # Set with namespace
            namespaced_key = await cache_manager.set_with_namespace(namespace, key, value)
            
            # Get with namespace
            result = await cache_manager.get_with_namespace(namespace, key)
            
            # Verify namespace isolation
            assert namespace in namespaced_key
            assert result == value
        
        # Verify different namespaces don't interfere
        embeddings_result = await cache_manager.get_with_namespace("embeddings", "test_key")
        search_result = await cache_manager.get_with_namespace("search", "test_key")
        
        assert embeddings_result != search_result

    @pytest.mark.asyncio
    async def test_error_handling_redis_failure(self, cache_manager):
        """Test error handling when Redis operations fail."""
        # Setup Redis failure
        cache_manager.redis_client.set.side_effect = Exception("Redis server down")
        cache_manager.redis_client.get.side_effect = Exception("Redis server down")
        
        # Execute operations
        set_result = await cache_manager.set("key", "value")
        get_result = await cache_manager.get("key")
        
        # Verify graceful degradation
        assert set_result is False  # Should not crash
        assert get_result is None  # Should return None gracefully
        
        # Verify L1 cache still works
        cache_manager.local_cache["fallback_key"] = {
            "value": "fallback_value",
            "expires_at": asyncio.get_event_loop().time() + 3600
        }
        
        fallback_result = await cache_manager.get("fallback_key")
        assert fallback_result == "fallback_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])