"""
Integration tests for provider system.

Tests the provider registry, hot-swapping, and provider interactions.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from src.core.utils.registry import ProviderRegistry, ProviderType
from src.core.interfaces.embeddings import EmbeddingProvider
from src.core.interfaces.vector_store import VectorStore
from src.core.interfaces.graph_engine import GraphEngine
from src.core.interfaces.reranker import Reranker


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self):
        self.initialized = False
        self.config = {}
        self.embed_calls = 0
    
    async def initialize(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = True
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embed_calls += 1
        return [[0.1] * 384 for _ in texts]
    
    async def embed_query(self, text: str) -> List[float]:
        self.embed_calls += 1
        return [0.1] * 384
    
    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "calls": self.embed_calls}
    
    async def close(self):
        pass


class MockVectorStore(VectorStore):
    """Mock vector store for testing."""
    
    def __init__(self):
        self.initialized = False
        self.config = {}
        self.vectors = {}
        self.search_calls = 0
    
    async def initialize(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = True
    
    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]):
        for i, vector in enumerate(vectors):
            self.vectors[ids[i]] = {"vector": vector, "metadata": metadata[i]}
    
    async def search(self, query_vector: List[float], top_k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        self.search_calls += 1
        return [
            {"id": "mock_result", "score": 0.95, "metadata": {"test": True}}
        ]
    
    async def delete(self, ids: List[str]):
        for id in ids:
            self.vectors.pop(id, None)
    
    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "vector_count": len(self.vectors), "search_calls": self.search_calls}
    
    async def close(self):
        pass


class TestProviderIntegration:
    """Integration tests for the provider system."""
    
    @pytest.fixture
    def provider_registry(self):
        """Create a fresh provider registry for each test."""
        return ProviderRegistry()
    
    @pytest.mark.asyncio
    async def test_provider_registration_and_retrieval(self, provider_registry):
        """Test provider registration and retrieval."""
        # Register a mock embedding provider
        config = {
            "model_name": "test-model",
            "device": "cpu",
            "batch_size": 32
        }
        
        success = await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="test_embedder",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config=config
        )
        
        assert success is True
        
        # Retrieve the provider
        provider = await provider_registry.get_provider(ProviderType.EMBEDDING, "test_embedder")
        
        assert isinstance(provider, MockEmbeddingProvider)
        assert provider.initialized is True
        assert provider.config == config
        
        # Test embedding functionality
        embeddings = await provider.embed_documents(["test document"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert provider.embed_calls == 1
    
    @pytest.mark.asyncio
    async def test_multiple_provider_types(self, provider_registry):
        """Test registration and use of multiple provider types."""
        # Register embedding provider
        embedding_config = {"model_name": "test-embedder"}
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="test_embedder",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config=embedding_config
        )
        
        # Register vector store provider  
        vector_config = {"connection_string": "test://connection"}
        await provider_registry.register_provider(
            provider_type=ProviderType.VECTOR_STORE,
            name="test_vector_store",
            class_path="tests.integration.test_provider_integration.MockVectorStore",
            config=vector_config
        )
        
        # Test both providers work together
        embedder = await provider_registry.get_provider(ProviderType.EMBEDDING, "test_embedder")
        vector_store = await provider_registry.get_provider(ProviderType.VECTOR_STORE, "test_vector_store")
        
        # Generate embedding and store it
        embedding = await embedder.embed_query("test query")
        await vector_store.add_vectors(
            vectors=[embedding],
            metadata=[{"text": "test query"}],
            ids=["test_id"]
        )
        
        # Search using the vector store
        results = await vector_store.search(embedding, top_k=5)
        
        assert len(results) == 1
        assert results[0]["id"] == "mock_result"
        assert vector_store.search_calls == 1
    
    @pytest.mark.asyncio
    async def test_provider_health_monitoring(self, provider_registry):
        """Test provider health monitoring."""
        # Register provider
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="health_test_embedder",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config={"model_name": "health-test"}
        )
        
        # Start health monitoring
        await provider_registry.start_health_monitoring()
        
        # Get provider and use it
        provider = await provider_registry.get_provider(ProviderType.EMBEDDING, "health_test_embedder")
        await provider.embed_query("health test")
        
        # Check health
        health = await provider_registry.health_check(ProviderType.EMBEDDING, "health_test_embedder")
        
        assert health["status"] == "healthy"
        assert health["calls"] == 1
        
        # Stop monitoring
        await provider_registry.stop_health_monitoring()
    
    @pytest.mark.asyncio
    async def test_provider_hot_swapping(self, provider_registry):
        """Test hot-swapping of providers."""
        # Register initial provider
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="swap_test_embedder",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config={"model_name": "original-model"}
        )
        
        # Get provider and use it
        provider1 = await provider_registry.get_provider(ProviderType.EMBEDDING, "swap_test_embedder")
        await provider1.embed_query("test 1")
        
        assert provider1.embed_calls == 1
        assert provider1.config["model_name"] == "original-model"
        
        # Hot-swap the provider
        success = await provider_registry.reload_provider(ProviderType.EMBEDDING, "swap_test_embedder")
        assert success is True
        
        # Get the new provider instance
        provider2 = await provider_registry.get_provider(ProviderType.EMBEDDING, "swap_test_embedder")
        
        # Should be a new instance
        assert provider2 is not provider1
        assert provider2.embed_calls == 0  # New instance, no calls yet
        
        # Use the new provider
        await provider2.embed_query("test 2")
        assert provider2.embed_calls == 1
    
    @pytest.mark.asyncio
    async def test_provider_fallback_mechanism(self, provider_registry):
        """Test provider fallback mechanism."""
        # Register primary provider
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="primary_embedder",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config={"model_name": "primary-model"}
        )
        
        # Register fallback provider
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="fallback_embedder",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config={"model_name": "fallback-model"}
        )
        
        # Set up fallback
        await provider_registry.set_fallback_provider(
            provider_type=ProviderType.EMBEDDING,
            primary_name="primary_embedder",
            fallback_name="fallback_embedder"
        )
        
        # Get primary provider and verify it works
        primary = await provider_registry.get_provider_with_fallback(ProviderType.EMBEDDING, "primary_embedder")
        assert primary.config["model_name"] == "primary-model"
        
        # Test fallback by simulating primary failure
        with patch.object(provider_registry, 'get_provider') as mock_get_provider:
            # First call (primary) fails
            mock_get_provider.side_effect = [
                Exception("Primary provider failed"),
                await provider_registry.get_provider(ProviderType.EMBEDDING, "fallback_embedder")
            ]
            
            # Should get fallback provider
            fallback = await provider_registry.get_provider_with_fallback(ProviderType.EMBEDDING, "primary_embedder")
            assert fallback.config["model_name"] == "fallback-model"
    
    @pytest.mark.asyncio
    async def test_concurrent_provider_access(self, provider_registry):
        """Test concurrent access to providers."""
        # Register provider
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="concurrent_embedder",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config={"model_name": "concurrent-model"}
        )
        
        # Create concurrent access tasks
        async def use_provider(text: str):
            provider = await provider_registry.get_provider(ProviderType.EMBEDDING, "concurrent_embedder")
            return await provider.embed_query(f"concurrent test {text}")
        
        tasks = [use_provider(str(i)) for i in range(10)]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        for result in results:
            assert len(result) == 384
        
        # Provider should have been accessed 10 times
        provider = await provider_registry.get_provider(ProviderType.EMBEDDING, "concurrent_embedder")
        assert provider.embed_calls == 10
    
    @pytest.mark.asyncio
    async def test_provider_statistics_and_monitoring(self, provider_registry):
        """Test provider statistics and monitoring."""
        # Register multiple providers
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="stats_embedder_1",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config={"model_name": "stats-model-1"}
        )
        
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="stats_embedder_2",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config={"model_name": "stats-model-2"}
        )
        
        await provider_registry.register_provider(
            provider_type=ProviderType.VECTOR_STORE,
            name="stats_vector_store",
            class_path="tests.integration.test_provider_integration.MockVectorStore",
            config={"connection_string": "stats://connection"}
        )
        
        # Use providers
        embedder1 = await provider_registry.get_provider(ProviderType.EMBEDDING, "stats_embedder_1")
        embedder2 = await provider_registry.get_provider(ProviderType.EMBEDDING, "stats_embedder_2")
        vector_store = await provider_registry.get_provider(ProviderType.VECTOR_STORE, "stats_vector_store")
        
        await embedder1.embed_query("test 1")
        await embedder2.embed_query("test 2")
        await vector_store.search([0.1] * 384)
        
        # Get statistics
        stats = await provider_registry.get_stats()
        
        # Verify statistics
        assert stats["provider_counts"]["embeddings"] == 2
        assert stats["provider_counts"]["vector_stores"] == 1
        assert stats["usage_stats"]["embeddings"] == 2  # Two embedders accessed
        assert stats["usage_stats"]["vector_stores"] == 1  # One vector store accessed
        
        # List providers
        providers = await provider_registry.list_providers()
        
        assert len(providers["embeddings"]) == 2
        assert len(providers["vector_stores"]) == 1
        
        # Check individual provider info
        embedder_info = providers["embeddings"]["stats_embedder_1"]
        assert embedder_info["name"] == "stats_embedder_1"
        assert embedder_info["initialized"] is True
        assert embedder_info["usage_count"] == 1
    
    @pytest.mark.asyncio
    async def test_provider_error_handling(self, provider_registry):
        """Test provider error handling and recovery."""
        # Test registration with invalid class path
        success = await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="invalid_embedder",
            class_path="invalid.module.InvalidClass",
            config={"model_name": "invalid"}
        )
        
        assert success is False
        
        # Test getting non-existent provider
        with pytest.raises(ValueError, match="Provider .* not found"):
            await provider_registry.get_provider(ProviderType.EMBEDDING, "nonexistent_provider")
        
        # Test provider with initialization error
        class FailingProvider(EmbeddingProvider):
            async def initialize(self, config: Dict[str, Any]):
                raise Exception("Initialization failed")
            
            async def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return []
            
            async def embed_query(self, text: str) -> List[float]:
                return []
        
        # Register provider that will fail during initialization
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.FailingProvider = FailingProvider
            mock_import.return_value = mock_module
            
            success = await provider_registry.register_provider(
                provider_type=ProviderType.EMBEDDING,
                name="failing_embedder",
                class_path="mock.module.FailingProvider",
                config={"model_name": "failing"}
            )
            
            assert success is True
            
            # Getting the provider should fail during initialization
            with pytest.raises(Exception, match="Initialization failed"):
                await provider_registry.get_provider(ProviderType.EMBEDDING, "failing_embedder")
    
    @pytest.mark.asyncio
    async def test_provider_cleanup(self, provider_registry):
        """Test provider cleanup."""
        # Register provider
        await provider_registry.register_provider(
            provider_type=ProviderType.EMBEDDING,
            name="cleanup_embedder",
            class_path="tests.integration.test_provider_integration.MockEmbeddingProvider",
            config={"model_name": "cleanup-model"}
        )
        
        # Get provider to initialize it
        provider = await provider_registry.get_provider(ProviderType.EMBEDDING, "cleanup_embedder")
        assert provider.initialized is True
        
        # Cleanup registry
        await provider_registry.cleanup()
        
        # Provider should still be accessible but cleanup was called
        providers = await provider_registry.list_providers()
        assert len(providers["embeddings"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])