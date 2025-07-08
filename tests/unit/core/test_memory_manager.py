"""
Unit tests for the memory manager.

Tests memory storage, retrieval, RAG pipeline, and integration with all providers.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.interfaces.embeddings import EmbeddingProvider
from src.core.interfaces.graph_engine import GraphEngine
from src.core.interfaces.reranker import Reranker
from src.core.interfaces.vector_store import Memory, VectorStore
from src.core.memory.manager import MemoryManager
from src.core.rag.hallucination_detector import HallucinationDetector


class TestMemoryManager:
    """Test suite for MemoryManager."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "vector_store": {
                "provider": "pgvector",
                "config": {"host": "localhost", "port": 5432},
            },
            "embeddings": {
                "provider": "huggingface",
                "config": {"model": "all-MiniLM-L12-v2"},
            },
            "graph_engine": {
                "provider": "memgraph",
                "config": {"host": "localhost", "port": 7687},
            },
            "reranker": {
                "provider": "cross_encoder",
                "config": {"model": "ms-marco-MiniLM-L-6-v2"},
            },
            "hallucination_detector": {"enabled": True, "threshold": 0.75},
        }

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers for testing."""
        # Mock vector store
        vector_store = Mock(spec=VectorStore)
        vector_store.initialize = AsyncMock()
        vector_store.store_memory = AsyncMock(return_value="memory_id_123")
        vector_store.search_memories = AsyncMock(
            return_value=[
                Memory(
                    id="memory_1",
                    text="test memory 1",
                    embedding=[0.1, 0.2, 0.3],
                    metadata={"score": 0.9},
                    created_at=datetime.utcnow(),
                )
            ]
        )

        # Mock embeddings provider
        embeddings = Mock(spec=EmbeddingProvider)
        embeddings.initialize = AsyncMock()
        embeddings.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embeddings.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        embeddings.get_dimensions = Mock(return_value=384)

        # Mock graph engine
        graph_engine = Mock(spec=GraphEngine)
        graph_engine.initialize = AsyncMock()
        graph_engine.create_entity = AsyncMock(return_value="entity_123")
        graph_engine.create_relationship = AsyncMock(return_value="rel_123")

        # Mock reranker
        reranker = Mock(spec=Reranker)
        reranker.initialize = AsyncMock()
        reranker.rerank = AsyncMock(
            return_value=[{"id": "memory_1", "text": "test memory 1", "score": 0.95}]
        )

        # Mock hallucination detector
        hallucination_detector = Mock(spec=HallucinationDetector)
        hallucination_detector.initialize = AsyncMock()
        hallucination_detector.analyze_response = AsyncMock(
            return_value={
                "is_hallucination": False,
                "confidence": 0.9,
                "grounding_score": 0.85,
            }
        )

        return {
            "vector_store": vector_store,
            "embeddings": embeddings,
            "graph_engine": graph_engine,
            "reranker": reranker,
            "hallucination_detector": hallucination_detector,
        }

    @pytest.fixture
    def memory_manager(self, mock_config, mock_providers):
        """Create memory manager instance for testing."""
        manager = MemoryManager()

        # Inject mock providers
        manager.vector_store = mock_providers["vector_store"]
        manager.embeddings = mock_providers["embeddings"]
        manager.graph_engine = mock_providers["graph_engine"]
        manager.reranker = mock_providers["reranker"]
        manager.hallucination_detector = mock_providers["hallucination_detector"]

        return manager

    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test memory manager initialization."""
        manager = MemoryManager()

        with patch("src.core.memory.manager.get_provider") as mock_get_provider:
            # Mock provider creation
            mock_get_provider.side_effect = [
                Mock(),  # vector_store
                Mock(),  # embeddings
                Mock(),  # graph_engine
                Mock(),  # reranker
                Mock(),  # hallucination_detector
            ]

            # Mock provider initialization
            for provider in mock_get_provider.side_effect:
                provider.initialize = AsyncMock()

            await manager.initialize(mock_config)

            assert manager.is_initialized
            assert mock_get_provider.call_count == 5

    @pytest.mark.asyncio
    async def test_store_memory_basic(self, memory_manager, mock_providers):
        """Test basic memory storage."""
        memory_manager.is_initialized = True

        result = await memory_manager.store_memory(
            text="This is a test memory",
            agent_id="test_agent",
            metadata={"source": "test"},
        )

        # Verify embedding generation
        mock_providers["embeddings"].generate_embedding.assert_called_once_with(
            "This is a test memory"
        )

        # Verify vector store call
        mock_providers["vector_store"].store_memory.assert_called_once()

        # Verify result
        assert result["success"] is True
        assert "memory_id" in result

    @pytest.mark.asyncio
    async def test_store_memory_with_entity_extraction(
        self, memory_manager, mock_providers
    ):
        """Test memory storage with entity extraction."""
        memory_manager.is_initialized = True

        # Mock entity extraction
        with patch.object(memory_manager, "_extract_entities") as mock_extract:
            mock_extract.return_value = [
                {"name": "John", "type": "PERSON"},
                {"name": "Paris", "type": "LOCATION"},
            ]

            result = await memory_manager.store_memory(
                text="John visited Paris", agent_id="test_agent", extract_entities=True
            )

            # Verify entity extraction was called
            mock_extract.assert_called_once_with("John visited Paris")

            # Verify graph operations
            assert mock_providers["graph_engine"].create_entity.call_count == 2
            assert result["entities_created"] == 2

    @pytest.mark.asyncio
    async def test_store_memory_with_chunking(self, memory_manager, mock_providers):
        """Test memory storage with content chunking."""
        memory_manager.is_initialized = True

        # Create long text that should be chunked
        long_text = "This is a very long text. " * 100

        with patch.object(memory_manager, "_chunk_content") as mock_chunk:
            mock_chunk.return_value = ["This is chunk 1", "This is chunk 2"]

            # Mock multiple embedding generation
            mock_providers["embeddings"].generate_embeddings.return_value = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]

            result = await memory_manager.store_memory(
                text=long_text, agent_id="test_agent", chunk_content=True
            )

            # Verify chunking was called
            mock_chunk.assert_called_once_with(long_text)

            # Verify multiple embeddings generated
            mock_providers["embeddings"].generate_embeddings.assert_called_once()

            # Verify multiple memories stored
            assert mock_providers["vector_store"].store_memory.call_count == 2
            assert result["chunk_ids"] is not None
            assert len(result["chunk_ids"]) == 2

    @pytest.mark.asyncio
    async def test_search_memories_basic(self, memory_manager, mock_providers):
        """Test basic memory search."""
        memory_manager.is_initialized = True

        result = await memory_manager.search_memories(
            query="test query", agent_id="test_agent", top_k=5
        )

        # Verify embedding generation for query
        mock_providers["embeddings"].generate_embedding.assert_called_once_with(
            "test query"
        )

        # Verify vector search
        mock_providers["vector_store"].search_memories.assert_called_once()

        # Verify result structure
        assert result["success"] is True
        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_search_memories_with_reranking(self, memory_manager, mock_providers):
        """Test memory search with reranking."""
        memory_manager.is_initialized = True

        result = await memory_manager.search_memories(
            query="test query", agent_id="test_agent", top_k=10, rerank=True
        )

        # Verify reranking was called
        mock_providers["reranker"].rerank.assert_called_once()

        # Verify final results are reranked
        assert result["success"] is True
        assert len(result["results"]) <= 10

    @pytest.mark.asyncio
    async def test_search_memories_with_graph_enrichment(
        self, memory_manager, mock_providers
    ):
        """Test memory search with graph knowledge enrichment."""
        memory_manager.is_initialized = True

        # Mock graph enrichment
        with patch.object(
            memory_manager, "_enrich_with_graph_knowledge"
        ) as mock_enrich:
            mock_enrich.return_value = [
                {
                    "id": "memory_1",
                    "text": "enriched memory",
                    "graph_context": "additional context",
                }
            ]

            result = await memory_manager.search_memories(
                query="test query", agent_id="test_agent", include_graph_context=True
            )

            # Verify graph enrichment was called
            mock_enrich.assert_called_once()

            # Verify enriched results
            assert result["success"] is True
            assert "graph_context" in result["results"][0]

    @pytest.mark.asyncio
    async def test_analyze_response(self, memory_manager, mock_providers):
        """Test response analysis for hallucination detection."""
        memory_manager.is_initialized = True

        result = await memory_manager.analyze_response(
            response="This is a generated response",
            query="original query",
            retrieved_memories=[
                {"text": "relevant memory 1"},
                {"text": "relevant memory 2"},
            ],
        )

        # Verify hallucination detector was called
        mock_providers["hallucination_detector"].analyze_response.assert_called_once()

        # Verify result structure
        assert result["success"] is True
        assert "analysis" in result
        assert "confidence" in result["analysis"]

    @pytest.mark.asyncio
    async def test_get_memory_stats(self, memory_manager, mock_providers):
        """Test memory system statistics retrieval."""
        memory_manager.is_initialized = True

        # Mock stats from vector store
        mock_providers["vector_store"].get_stats = AsyncMock(
            return_value={
                "total_memories": 1000,
                "total_agents": 5,
                "storage_size_mb": 150.5,
            }
        )

        # Mock stats from graph engine
        mock_providers["graph_engine"].get_stats = AsyncMock(
            return_value={"total_entities": 500, "total_relationships": 200}
        )

        result = await memory_manager.get_stats(agent_id="test_agent")

        # Verify stats collection
        mock_providers["vector_store"].get_stats.assert_called_once()
        mock_providers["graph_engine"].get_stats.assert_called_once()

        # Verify result structure
        assert result["success"] is True
        assert "memory_stats" in result
        assert "graph_stats" in result

    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_manager, mock_providers):
        """Test memory deletion."""
        memory_manager.is_initialized = True

        # Mock deletion operations
        mock_providers["vector_store"].delete_memory = AsyncMock(return_value=True)
        mock_providers["graph_engine"].delete_entity = AsyncMock(return_value=True)

        result = await memory_manager.delete_memory("memory_123")

        # Verify deletion calls
        mock_providers["vector_store"].delete_memory.assert_called_once_with(
            "memory_123"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_error_handling_in_store_memory(self, memory_manager, mock_providers):
        """Test error handling during memory storage."""
        memory_manager.is_initialized = True

        # Mock embedding generation failure
        mock_providers["embeddings"].generate_embedding.side_effect = Exception(
            "Embedding failed"
        )

        result = await memory_manager.store_memory(
            text="test text", agent_id="test_agent"
        )

        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_handling_in_search_memories(
        self, memory_manager, mock_providers
    ):
        """Test error handling during memory search."""
        memory_manager.is_initialized = True

        # Mock vector search failure
        mock_providers["vector_store"].search_memories.side_effect = Exception(
            "Search failed"
        )

        result = await memory_manager.search_memories(
            query="test query", agent_id="test_agent"
        )

        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memory_manager, mock_providers):
        """Test handling of concurrent memory operations."""
        memory_manager.is_initialized = True

        # Create multiple concurrent store operations
        store_tasks = [
            memory_manager.store_memory(f"memory {i}", "test_agent") for i in range(5)
        ]

        # Create multiple concurrent search operations
        search_tasks = [
            memory_manager.search_memories(f"query {i}", "test_agent") for i in range(3)
        ]

        # Execute all operations concurrently
        all_results = await asyncio.gather(*(store_tasks + search_tasks))

        # Verify all operations completed
        assert len(all_results) == 8
        assert all(result.get("success", False) for result in all_results)

    @pytest.mark.asyncio
    async def test_agent_isolation(self, memory_manager, mock_providers):
        """Test that memories are properly isolated by agent."""
        memory_manager.is_initialized = True

        # Store memories for different agents
        await memory_manager.store_memory("agent1 memory", "agent1")
        await memory_manager.store_memory("agent2 memory", "agent2")

        # Search should filter by agent
        await memory_manager.search_memories("test query", "agent1")

        # Verify agent filtering was applied
        search_call_args = mock_providers["vector_store"].search_memories.call_args
        assert "agent1" in str(search_call_args)

    @pytest.mark.asyncio
    async def test_session_management(self, memory_manager, mock_providers):
        """Test session-based memory management."""
        memory_manager.is_initialized = True

        # Store memory with session
        result = await memory_manager.store_memory(
            text="session memory", agent_id="test_agent", session_id="session_123"
        )

        # Verify session metadata was included
        store_call_args = mock_providers["vector_store"].store_memory.call_args
        memory_obj = store_call_args[0][0]  # First argument is the Memory object
        assert "session_id" in memory_obj.metadata
        assert memory_obj.metadata["session_id"] == "session_123"

    @pytest.mark.asyncio
    async def test_health_check(self, memory_manager, mock_providers):
        """Test system health check."""
        memory_manager.is_initialized = True

        # Mock health checks for all providers
        for provider in mock_providers.values():
            provider.health_check = AsyncMock(return_value={"status": "healthy"})

        health_status = await memory_manager.health_check()

        # Verify health check calls
        for provider in mock_providers.values():
            provider.health_check.assert_called_once()

        # Verify overall health status
        assert health_status["status"] == "healthy"
        assert "components" in health_status

    @pytest.mark.asyncio
    async def test_cleanup(self, memory_manager, mock_providers):
        """Test proper cleanup of resources."""
        memory_manager.is_initialized = True

        # Mock close methods
        for provider in mock_providers.values():
            provider.close = AsyncMock()

        await memory_manager.close()

        # Verify all providers were closed
        for provider in mock_providers.values():
            provider.close.assert_called_once()

        assert not memory_manager.is_initialized

    @pytest.mark.asyncio
    async def test_performance_tracking(self, memory_manager, mock_providers):
        """Test performance tracking and metrics."""
        memory_manager.is_initialized = True

        # Enable performance tracking
        memory_manager.track_performance = True

        # Perform operations
        await memory_manager.store_memory("test memory", "test_agent")
        await memory_manager.search_memories("test query", "test_agent")

        # Check that performance metrics were recorded
        assert hasattr(memory_manager, "performance_metrics")
        assert len(memory_manager.performance_metrics) > 0

    def test_configuration_validation(self, mock_config):
        """Test configuration validation."""
        manager = MemoryManager()

        # Valid configuration should work
        assert manager._validate_config(mock_config) is True

        # Invalid configuration should raise error
        invalid_config = {"invalid": "config"}

        with pytest.raises(ValueError):
            manager._validate_config(invalid_config)
