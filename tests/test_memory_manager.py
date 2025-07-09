"""
Comprehensive unit tests for Memory Manager.

Tests memory storage, retrieval, search functionality, and error handling.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from src.core.memory.manager import MemoryManager, MemoryStoreRequest, MemorySearchRequest
from src.core.memory.models import Memory, MemoryMetadata


class TestMemoryManager:
    """Test Memory Manager functionality."""

    @pytest.fixture
    async def memory_manager(self):
        """Create memory manager with mocked dependencies."""
        with patch('src.core.memory.manager.PostgreSQLClient') as mock_pg, \
             patch('src.core.memory.manager.EmbeddingProvider') as mock_embedder, \
             patch('src.core.memory.manager.GraphEngine') as mock_graph, \
             patch('src.core.memory.manager.CacheManager') as mock_cache:
            
            manager = MemoryManager()
            manager.db_client = AsyncMock()
            manager.embedder = AsyncMock()
            manager.graph_engine = AsyncMock()
            manager.cache = AsyncMock()
            manager._initialized = True
            
            yield manager

    @pytest.mark.asyncio
    async def test_store_memory_basic(self, memory_manager):
        """Test basic memory storage."""
        # Setup
        memory_id = str(uuid.uuid4())
        request = MemoryStoreRequest(
            text="Python is a programming language",
            agent_id="test_agent",
            metadata={"type": "fact"}
        )
        
        # Mock embedding generation
        memory_manager.embedder.generate_embedding.return_value = [0.1, 0.2, 0.3] * 341  # 1024 dims
        
        # Mock database storage
        memory_manager.db_client.store_memory.return_value = {
            "memory_id": memory_id,
            "status": "stored"
        }
        
        # Mock entity extraction
        memory_manager.graph_engine.extract_entities.return_value = [
            {"name": "Python", "type": "TECHNOLOGY", "confidence": 0.95}
        ]
        
        # Execute
        result = await memory_manager.store_memory(request)
        
        # Verify
        assert result["memory_id"] == memory_id
        assert result["status"] == "stored"
        
        # Verify embedding was generated
        memory_manager.embedder.generate_embedding.assert_called_once_with(request.text)
        
        # Verify database storage
        memory_manager.db_client.store_memory.assert_called_once()
        
        # Verify entity extraction
        memory_manager.graph_engine.extract_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_memory_with_chunking(self, memory_manager):
        """Test memory storage with content chunking."""
        # Setup large content
        large_text = "This is a test sentence. " * 200  # Large content
        request = MemoryStoreRequest(
            text=large_text,
            agent_id="test_agent",
            chunk_content=True
        )
        
        # Mock chunking
        chunks = [large_text[:500], large_text[500:1000], large_text[1000:]]
        memory_manager._chunk_content = MagicMock(return_value=chunks)
        
        # Mock embedding generation for each chunk
        memory_manager.embedder.generate_embedding.return_value = [0.1] * 1024
        
        # Mock database storage for each chunk
        memory_manager.db_client.store_memory.return_value = {"status": "stored"}
        
        # Execute
        result = await memory_manager.store_memory(request)
        
        # Verify
        assert "chunks_created" in result
        assert result["chunks_created"] == 3
        
        # Verify embedding generated for each chunk
        assert memory_manager.embedder.generate_embedding.call_count == 3
        
        # Verify database storage for each chunk
        assert memory_manager.db_client.store_memory.call_count == 3

    @pytest.mark.asyncio
    async def test_search_memory_vector(self, memory_manager):
        """Test vector-based memory search."""
        # Setup
        request = MemorySearchRequest(
            query="Python programming",
            search_type="vector",
            top_k=5,
            agent_id="test_agent"
        )
        
        # Mock query embedding
        query_embedding = [0.1] * 1024
        memory_manager.embedder.generate_embedding.return_value = query_embedding
        
        # Mock database search results
        mock_results = [
            {
                "memory_id": "mem_1",
                "content": "Python programming basics",
                "similarity_score": 0.95,
                "metadata": {"type": "tutorial"},
                "agent_id": "test_agent"
            },
            {
                "memory_id": "mem_2",
                "content": "Advanced Python concepts",
                "similarity_score": 0.87,
                "metadata": {"type": "advanced"},
                "agent_id": "test_agent"
            }
        ]
        memory_manager.db_client.vector_search.return_value = mock_results
        
        # Execute
        results = await memory_manager.search_memories(request)
        
        # Verify
        assert len(results) == 2
        assert results[0]["memory_id"] == "mem_1"
        assert results[0]["similarity_score"] == 0.95
        
        # Verify query embedding was generated
        memory_manager.embedder.generate_embedding.assert_called_once_with(request.query)
        
        # Verify database search
        memory_manager.db_client.vector_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_memory_graph(self, memory_manager):
        """Test graph-based memory search."""
        # Setup
        request = MemorySearchRequest(
            query="Python programming language",
            search_type="graph",
            top_k=10
        )
        
        # Mock entity extraction from query
        memory_manager.graph_engine.extract_entities.return_value = [
            {"name": "Python", "type": "TECHNOLOGY"}
        ]
        
        # Mock graph search results
        mock_graph_results = [
            {
                "memory_id": "graph_mem_1",
                "content": "Python language features",
                "graph_score": 0.92,
                "path": ["Python", "is_a", "Programming Language"],
                "entities": ["Python", "Programming Language"]
            }
        ]
        memory_manager.graph_engine.search_by_entities.return_value = mock_graph_results
        
        # Execute
        results = await memory_manager.search_memories(request)
        
        # Verify
        assert len(results) == 1
        assert results[0]["memory_id"] == "graph_mem_1"
        assert "path" in results[0]
        
        # Verify entity extraction
        memory_manager.graph_engine.extract_entities.assert_called_once()
        
        # Verify graph search
        memory_manager.graph_engine.search_by_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_memory_hybrid(self, memory_manager):
        """Test hybrid search combining vector and graph."""
        # Setup
        request = MemorySearchRequest(
            query="machine learning algorithms",
            search_type="hybrid",
            top_k=8
        )
        
        # Mock vector search results
        vector_results = [
            {"memory_id": "vec_1", "content": "ML algorithms", "similarity_score": 0.94}
        ]
        memory_manager.embedder.generate_embedding.return_value = [0.1] * 1024
        memory_manager.db_client.vector_search.return_value = vector_results
        
        # Mock graph search results
        graph_results = [
            {"memory_id": "graph_1", "content": "Algorithm types", "graph_score": 0.89}
        ]
        memory_manager.graph_engine.extract_entities.return_value = [
            {"name": "machine learning", "type": "CONCEPT"}
        ]
        memory_manager.graph_engine.search_by_entities.return_value = graph_results
        
        # Mock hybrid scoring
        memory_manager._combine_hybrid_scores = MagicMock(return_value=[
            {
                "memory_id": "vec_1",
                "content": "ML algorithms", 
                "hybrid_score": 0.92,
                "vector_score": 0.94,
                "graph_score": 0.0
            },
            {
                "memory_id": "graph_1",
                "content": "Algorithm types",
                "hybrid_score": 0.89,
                "vector_score": 0.0,
                "graph_score": 0.89
            }
        ])
        
        # Execute
        results = await memory_manager.search_memories(request)
        
        # Verify
        assert len(results) == 2
        assert "hybrid_score" in results[0]
        assert "vector_score" in results[0]
        assert "graph_score" in results[0]

    @pytest.mark.asyncio
    async def test_search_with_confidence_filtering(self, memory_manager):
        """Test search with minimum confidence filtering."""
        # Setup
        request = MemorySearchRequest(
            query="test query",
            min_confidence=0.8,
            top_k=10
        )
        
        # Mock results with various confidence levels
        mock_results = [
            {"memory_id": "high_conf", "content": "High confidence", "similarity_score": 0.95},
            {"memory_id": "med_conf", "content": "Medium confidence", "similarity_score": 0.75},
            {"memory_id": "high_conf_2", "content": "Another high", "similarity_score": 0.85}
        ]
        
        memory_manager.embedder.generate_embedding.return_value = [0.1] * 1024
        memory_manager.db_client.vector_search.return_value = mock_results
        
        # Execute
        results = await memory_manager.search_memories(request)
        
        # Verify only high confidence results returned
        assert len(results) == 2  # Only >= 0.8 confidence
        assert all(r["similarity_score"] >= 0.8 for r in results)

    @pytest.mark.asyncio
    async def test_get_memory_by_id(self, memory_manager):
        """Test retrieving specific memory by ID."""
        # Setup
        memory_id = "test_memory_123"
        mock_memory = {
            "memory_id": memory_id,
            "content": "Test memory content",
            "metadata": {"type": "test"},
            "created_at": datetime.utcnow(),
            "agent_id": "test_agent"
        }
        
        memory_manager.db_client.get_memory.return_value = mock_memory
        
        # Execute
        result = await memory_manager.get_memory(memory_id)
        
        # Verify
        assert result["memory_id"] == memory_id
        assert result["content"] == "Test memory content"
        
        # Verify database call
        memory_manager.db_client.get_memory.assert_called_once_with(memory_id)

    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_manager):
        """Test memory deletion."""
        # Setup
        memory_id = "memory_to_delete"
        
        memory_manager.db_client.delete_memory.return_value = {
            "memory_id": memory_id,
            "status": "deleted"
        }
        memory_manager.graph_engine.remove_memory_entities.return_value = {
            "entities_removed": 2
        }
        
        # Execute
        result = await memory_manager.delete_memory(memory_id, hard_delete=True)
        
        # Verify
        assert result["memory_id"] == memory_id
        assert result["status"] == "deleted"
        
        # Verify database deletion
        memory_manager.db_client.delete_memory.assert_called_once_with(memory_id, hard_delete=True)
        
        # Verify graph cleanup
        memory_manager.graph_engine.remove_memory_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_manager):
        """Test memory statistics retrieval."""
        # Setup
        mock_stats = {
            "total_memories": 1000,
            "by_agent": {"tyra": 400, "claude": 300, "archon": 300},
            "by_confidence": {"rock_solid": 200, "high": 400, "fuzzy": 300, "low": 100},
            "performance": {"avg_query_time_ms": 50}
        }
        
        memory_manager.db_client.get_stats.return_value = mock_stats
        
        # Execute
        result = await memory_manager.get_stats(agent_id="tyra")
        
        # Verify
        assert result["total_memories"] == 1000
        assert result["by_agent"]["tyra"] == 400
        assert "performance" in result

    @pytest.mark.asyncio
    async def test_health_check(self, memory_manager):
        """Test memory manager health check."""
        # Setup
        memory_manager.db_client.health_check.return_value = {"status": "healthy"}
        memory_manager.embedder.health_check.return_value = {"status": "healthy"}
        memory_manager.graph_engine.health_check.return_value = {"status": "healthy"}
        memory_manager.cache.health_check.return_value = {"status": "healthy"}
        
        # Execute
        result = await memory_manager.health_check()
        
        # Verify
        assert result["status"] == "healthy"
        assert "components" in result
        assert result["components"]["database"] == "healthy"
        assert result["components"]["embedder"] == "healthy"
        assert result["components"]["graph"] == "healthy"
        assert result["components"]["cache"] == "healthy"

    @pytest.mark.asyncio
    async def test_cache_integration(self, memory_manager):
        """Test cache integration for embeddings and search results."""
        # Setup search request
        request = MemorySearchRequest(query="cached query", top_k=5)
        
        # Mock cache hit for query embedding
        cached_embedding = [0.2] * 1024
        memory_manager.cache.get.return_value = cached_embedding
        
        # Mock search results
        mock_results = [{"memory_id": "cached_result", "similarity_score": 0.9}]
        memory_manager.db_client.vector_search.return_value = mock_results
        
        # Execute
        results = await memory_manager.search_memories(request)
        
        # Verify
        assert len(results) == 1
        
        # Verify cache was checked for embedding
        memory_manager.cache.get.assert_called()
        
        # Verify embedding was not regenerated (cache hit)
        memory_manager.embedder.generate_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_embedding_failure(self, memory_manager):
        """Test error handling when embedding generation fails."""
        # Setup
        request = MemoryStoreRequest(text="Test content", agent_id="test_agent")
        
        # Mock embedding failure
        memory_manager.embedder.generate_embedding.side_effect = Exception("Embedding service unavailable")
        
        # Execute and verify exception handling
        with pytest.raises(Exception, match="Embedding service unavailable"):
            await memory_manager.store_memory(request)

    @pytest.mark.asyncio
    async def test_error_handling_database_failure(self, memory_manager):
        """Test error handling when database operations fail."""
        # Setup
        request = MemorySearchRequest(query="test", top_k=5)
        
        # Mock database failure
        memory_manager.db_client.vector_search.side_effect = Exception("Database connection failed")
        memory_manager.embedder.generate_embedding.return_value = [0.1] * 1024
        
        # Execute and verify exception handling
        with pytest.raises(Exception, match="Database connection failed"):
            await memory_manager.search_memories(request)

    @pytest.mark.asyncio
    async def test_agent_isolation(self, memory_manager):
        """Test that agent data is properly isolated."""
        # Setup search for specific agent
        request = MemorySearchRequest(
            query="test query",
            agent_id="agent_1",
            top_k=10
        )
        
        # Mock results that include agent filtering
        mock_results = [
            {"memory_id": "mem_1", "agent_id": "agent_1", "similarity_score": 0.9},
            # Note: agent_2 results should be filtered out by the database query
        ]
        memory_manager.embedder.generate_embedding.return_value = [0.1] * 1024
        memory_manager.db_client.vector_search.return_value = mock_results
        
        # Execute
        results = await memory_manager.search_memories(request)
        
        # Verify
        assert all(r.get("agent_id") == "agent_1" for r in results)
        
        # Verify database was called with agent filter
        call_args = memory_manager.db_client.vector_search.call_args
        assert "agent_id" in str(call_args)

    @pytest.mark.asyncio
    async def test_memory_versioning(self, memory_manager):
        """Test memory versioning functionality."""
        # Setup
        original_request = MemoryStoreRequest(
            text="Original content",
            agent_id="test_agent",
            memory_id="versioned_memory"
        )
        
        updated_request = MemoryStoreRequest(
            text="Updated content",
            agent_id="test_agent", 
            memory_id="versioned_memory"
        )
        
        # Mock version handling
        memory_manager.db_client.store_memory.return_value = {
            "memory_id": "versioned_memory",
            "version": 2,
            "status": "updated"
        }
        memory_manager.embedder.generate_embedding.return_value = [0.1] * 1024
        
        # Execute
        result = await memory_manager.store_memory(updated_request)
        
        # Verify
        assert result["version"] == 2
        assert result["status"] == "updated"

    @pytest.mark.asyncio
    async def test_bulk_operations(self, memory_manager):
        """Test bulk memory operations."""
        # Setup bulk store request
        bulk_requests = [
            MemoryStoreRequest(text=f"Content {i}", agent_id="bulk_agent")
            for i in range(5)
        ]
        
        # Mock bulk storage
        memory_manager.embedder.generate_embedding.return_value = [0.1] * 1024
        memory_manager.db_client.bulk_store_memories.return_value = {
            "stored_count": 5,
            "failed_count": 0,
            "memory_ids": [f"bulk_mem_{i}" for i in range(5)]
        }
        
        # Execute
        result = await memory_manager.bulk_store_memories(bulk_requests)
        
        # Verify
        assert result["stored_count"] == 5
        assert result["failed_count"] == 0
        assert len(result["memory_ids"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])