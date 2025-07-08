"""
Integration tests for memory API endpoints.

Tests end-to-end memory operations through the FastAPI interface.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.api.app import get_app
from src.core.memory.manager import MemoryManager


class TestMemoryEndpoints:
    """Integration tests for memory API endpoints."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing."""
        return get_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    async def async_client(self, app):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager for testing."""
        manager = AsyncMock(spec=MemoryManager)

        # Mock store_memory method
        manager.store_memory.return_value = {
            "success": True,
            "memory_id": "memory_123",
            "chunk_ids": ["chunk_1", "chunk_2"],
            "entities_created": 2,
            "relationships_created": 1,
            "processing_time": {"embedding": 0.05, "storage": 0.02},
        }

        # Mock search_memories method
        manager.search_memories.return_value = {
            "success": True,
            "results": [
                {
                    "id": "memory_1",
                    "content": "Test memory content",
                    "score": 0.95,
                    "metadata": {"agent_id": "test_agent"},
                }
            ],
            "total_results": 1,
            "hallucination_analysis": {"is_hallucination": False, "confidence": 0.9},
        }

        # Mock analyze_response method
        manager.analyze_response.return_value = {
            "success": True,
            "analysis": {
                "is_hallucination": False,
                "overall_confidence": 85,
                "grounding_score": 0.8,
                "confidence_level": "high",
            },
        }

        # Mock get_stats method
        manager.get_stats.return_value = {
            "success": True,
            "memory_stats": {"total_memories": 100, "recent_activity": 10},
            "performance_stats": {"avg_query_time": 0.05},
            "health_score": 0.95,
            "recommendations": [],
        }

        # Mock delete_memory method
        manager.delete_memory.return_value = True

        # Mock health_check method
        manager.health_check.return_value = {
            "status": "healthy",
            "components": {
                "vector_store": {"healthy": True},
                "embeddings": {"healthy": True},
                "graph_engine": {"healthy": True},
            },
        }

        return manager

    @pytest.mark.asyncio
    async def test_store_memory_endpoint(self, async_client, mock_memory_manager):
        """Test memory storage endpoint."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/store",
                json={
                    "content": "This is a test memory",
                    "agent_id": "test_agent",
                    "metadata": {"source": "test"},
                    "extract_entities": True,
                    "chunk_content": False,
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["memory_id"] == "memory_123"
            assert data["entities_created"] == 2
            assert data["relationships_created"] == 1
            assert "processing_time" in data

            # Verify memory manager was called correctly
            mock_memory_manager.store_memory.assert_called_once()
            call_args = mock_memory_manager.store_memory.call_args
            assert call_args.kwargs["text"] == "This is a test memory"
            assert call_args.kwargs["agent_id"] == "test_agent"

    @pytest.mark.asyncio
    async def test_search_memories_endpoint(self, async_client, mock_memory_manager):
        """Test memory search endpoint."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/search",
                json={
                    "query": "test query",
                    "agent_id": "test_agent",
                    "top_k": 10,
                    "min_confidence": 0.5,
                    "search_type": "hybrid",
                    "include_analysis": True,
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["query"] == "test query"
            assert len(data["results"]) == 1
            assert data["total_results"] == 1
            assert data["search_type"] == "hybrid"
            assert "hallucination_analysis" in data

            # Verify memory manager was called correctly
            mock_memory_manager.search_memories.assert_called_once()
            call_args = mock_memory_manager.search_memories.call_args
            assert call_args.kwargs["query"] == "test query"
            assert call_args.kwargs["agent_id"] == "test_agent"

    @pytest.mark.asyncio
    async def test_analyze_response_endpoint(self, async_client, mock_memory_manager):
        """Test response analysis endpoint."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/analyze",
                json={
                    "response": "This is a generated response",
                    "query": "test query",
                    "retrieved_memories": [
                        {"content": "relevant memory"},
                        {"content": "another relevant memory"},
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "analysis" in data
            assert data["analysis"]["is_hallucination"] is False
            assert data["analysis"]["overall_confidence"] == 85
            assert data["analysis"]["confidence_level"] == "high"

            # Verify memory manager was called correctly
            mock_memory_manager.analyze_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_stats_endpoint(self, async_client, mock_memory_manager):
        """Test memory statistics endpoint."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.get(
                "/v1/memory/stats",
                params={
                    "agent_id": "test_agent",
                    "include_performance": True,
                    "include_recommendations": True,
                },
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "memory_stats" in data
            assert "performance_stats" in data
            assert data["health_score"] == 0.95
            assert "recommendations" in data

            # Verify memory manager was called correctly
            mock_memory_manager.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_memory_endpoint(self, async_client, mock_memory_manager):
        """Test memory deletion endpoint."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.delete("/v1/memory/memory_123")

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "Memory memory_123 deleted successfully" in data["message"]

            # Verify memory manager was called correctly
            mock_memory_manager.delete_memory.assert_called_once_with("memory_123")

    @pytest.mark.asyncio
    async def test_store_memory_validation_error(
        self, async_client, mock_memory_manager
    ):
        """Test validation error in store memory endpoint."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/store",
                json={
                    # Missing required 'content' field
                    "agent_id": "test_agent"
                },
            )

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_store_memory_processing_error(
        self, async_client, mock_memory_manager
    ):
        """Test processing error in store memory endpoint."""
        # Mock memory manager to return error
        mock_memory_manager.store_memory.return_value = {
            "success": False,
            "error": "Processing failed",
        }

        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/store",
                json={"content": "test content", "agent_id": "test_agent"},
            )

            assert (
                response.status_code == 200
            )  # Still returns 200 but with error in response
            data = response.json()

            assert data["success"] is False
            assert data["error"] == "Processing failed"

    @pytest.mark.asyncio
    async def test_search_memories_empty_results(
        self, async_client, mock_memory_manager
    ):
        """Test search with no results."""
        # Mock empty search results
        mock_memory_manager.search_memories.return_value = {
            "success": True,
            "results": [],
            "total_results": 0,
            "search_type": "hybrid",
        }

        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/search",
                json={"query": "nonexistent query", "agent_id": "test_agent"},
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert len(data["results"]) == 0
            assert data["total_results"] == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory(self, async_client, mock_memory_manager):
        """Test deletion of nonexistent memory."""
        # Mock memory manager to return False (memory not found)
        mock_memory_manager.delete_memory.return_value = False

        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.delete("/v1/memory/nonexistent_id")

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_memory_manager_unavailable(self, async_client):
        """Test behavior when memory manager is unavailable."""
        with patch(
            "src.api.app.get_memory_manager",
            side_effect=Exception("Manager unavailable"),
        ):
            response = await async_client.post(
                "/v1/memory/store",
                json={"content": "test content", "agent_id": "test_agent"},
            )

            assert response.status_code == 503  # Service unavailable

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client, mock_memory_manager):
        """Test handling of concurrent requests."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            # Create multiple concurrent requests
            tasks = []

            for i in range(5):
                task = async_client.post(
                    "/v1/memory/store",
                    json={"content": f"test content {i}", "agent_id": "test_agent"},
                )
                tasks.append(task)

            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks)

            # All should succeed
            assert all(response.status_code == 200 for response in responses)
            assert all(response.json()["success"] for response in responses)

    @pytest.mark.asyncio
    async def test_request_tracing_headers(self, async_client, mock_memory_manager):
        """Test that request tracing headers are added."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/store",
                json={"content": "test content", "agent_id": "test_agent"},
            )

            assert response.status_code == 200

            # Should have tracing headers
            assert "X-Request-ID" in response.headers
            assert "X-Process-Time" in response.headers

    @pytest.mark.asyncio
    async def test_large_content_handling(self, async_client, mock_memory_manager):
        """Test handling of large content."""
        # Create large content (1MB)
        large_content = "x" * (1024 * 1024)

        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/store",
                json={
                    "content": large_content,
                    "agent_id": "test_agent",
                    "chunk_content": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_special_characters_handling(self, async_client, mock_memory_manager):
        """Test handling of special characters and unicode."""
        special_content = "Test with émojis 🚀 and special chars: <>&\"'"

        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/store",
                json={"content": special_content, "agent_id": "test_agent"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_agent_isolation(self, async_client, mock_memory_manager):
        """Test that different agents' memories are isolated."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            # Store memory for agent1
            response1 = await async_client.post(
                "/v1/memory/store",
                json={"content": "agent1 memory", "agent_id": "agent1"},
            )

            # Store memory for agent2
            response2 = await async_client.post(
                "/v1/memory/store",
                json={"content": "agent2 memory", "agent_id": "agent2"},
            )

            assert response1.status_code == 200
            assert response2.status_code == 200

            # Verify both calls were made with correct agent IDs
            calls = mock_memory_manager.store_memory.call_args_list
            assert len(calls) == 2
            assert calls[0].kwargs["agent_id"] == "agent1"
            assert calls[1].kwargs["agent_id"] == "agent2"

    @pytest.mark.asyncio
    async def test_session_management(self, async_client, mock_memory_manager):
        """Test session-based memory management."""
        with patch("src.api.app.get_memory_manager", return_value=mock_memory_manager):
            response = await async_client.post(
                "/v1/memory/store",
                json={
                    "content": "session memory",
                    "agent_id": "test_agent",
                    "session_id": "session_123",
                },
            )

            assert response.status_code == 200

            # Verify session ID was passed through
            call_args = mock_memory_manager.store_memory.call_args
            assert call_args.kwargs["session_id"] == "session_123"
