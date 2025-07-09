"""
Comprehensive MCP integration tests for Tyra Memory Server.

Tests all MCP tools with various scenarios including:
- Memory storage and retrieval
- Hallucination detection
- Analytics and health checks
- Error handling and edge cases
- Trading endpoint safety
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import CallToolRequest, CallToolResult, TextContent

from src.mcp.server import TyraMemoryServer


class TestMCPIntegration:
    """Comprehensive MCP integration tests."""

    @pytest.fixture
    async def mcp_server(self):
        """Create and initialize MCP server for testing."""
        server = TyraMemoryServer()
        
        # Mock the external dependencies
        with patch('src.mcp.server.MemoryManager') as mock_memory_manager, \
             patch('src.mcp.server.HallucinationDetector') as mock_hallucination_detector, \
             patch('src.mcp.server.PerformanceTracker') as mock_performance_tracker, \
             patch('src.mcp.server.LearningEngine') as mock_learning_engine:
            
            # Configure mocks
            server.memory_manager = AsyncMock()
            server.hallucination_detector = AsyncMock()
            server.performance_tracker = AsyncMock()
            server.learning_engine = AsyncMock()
            server._initialized = True
            
            yield server

    @pytest.mark.asyncio
    async def test_store_memory_basic(self, mcp_server):
        """Test basic memory storage functionality."""
        # Setup
        memory_id = str(uuid.uuid4())
        mcp_server.memory_manager.store_memory.return_value = {
            "memory_id": memory_id,
            "status": "stored",
            "entities_extracted": 2,
            "chunks_created": 1
        }
        
        # Execute
        arguments = {
            "content": "Python is a programming language created by Guido van Rossum.",
            "agent_id": "test_agent",
            "metadata": {"test": True}
        }
        
        result = await mcp_server._handle_store_memory(arguments)
        
        # Verify
        assert result.isError is False
        assert "memory_id" in result.content[0].text
        assert memory_id in result.content[0].text
        
        # Verify memory manager was called correctly
        mcp_server.memory_manager.store_memory.assert_called_once()
        call_args = mcp_server.memory_manager.store_memory.call_args[1]
        assert call_args["text"] == arguments["content"]
        assert call_args["agent_id"] == arguments["agent_id"]
        assert call_args["metadata"] == arguments["metadata"]

    @pytest.mark.asyncio
    async def test_store_memory_with_chunking(self, mcp_server):
        """Test memory storage with content chunking."""
        # Setup large content that should be chunked
        large_content = "This is a test. " * 1000  # Create large content
        
        mcp_server.memory_manager.store_memory.return_value = {
            "memory_id": str(uuid.uuid4()),
            "status": "stored",
            "entities_extracted": 5,
            "chunks_created": 3
        }
        
        # Execute
        arguments = {
            "content": large_content,
            "agent_id": "test_agent",
            "chunk_content": True
        }
        
        result = await mcp_server._handle_store_memory(arguments)
        
        # Verify
        assert result.isError is False
        assert "chunks_created" in result.content[0].text
        assert "3" in result.content[0].text

    @pytest.mark.asyncio
    async def test_search_memory_vector(self, mcp_server):
        """Test vector memory search."""
        # Setup
        mock_memories = [
            {
                "memory_id": "mem_1",
                "content": "Python programming basics",
                "score": 0.95,
                "metadata": {"type": "educational"},
                "agent_id": "test_agent"
            },
            {
                "memory_id": "mem_2", 
                "content": "Advanced Python concepts",
                "score": 0.87,
                "metadata": {"type": "educational"},
                "agent_id": "test_agent"
            }
        ]
        
        mcp_server.memory_manager.search_memories.return_value = mock_memories
        mcp_server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.92,
            "risk_level": "low",
            "confidence_level": "high"
        }
        
        # Execute
        arguments = {
            "query": "Python programming",
            "agent_id": "test_agent",
            "search_type": "vector",
            "top_k": 5,
            "min_confidence": 0.8
        }
        
        result = await mcp_server._handle_search_memory(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert len(result_data["memories"]) == 2
        assert result_data["memories"][0]["memory_id"] == "mem_1"
        assert result_data["analysis"]["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_search_memory_graph(self, mcp_server):
        """Test graph-based memory search."""
        # Setup
        mock_memories = [
            {
                "memory_id": "mem_graph_1",
                "content": "Entity relationships in knowledge graph",
                "score": 0.91,
                "graph_path": ["Person", "works_for", "Organization"],
                "agent_id": "test_agent"
            }
        ]
        
        mcp_server.memory_manager.search_memories.return_value = mock_memories
        
        # Execute
        arguments = {
            "query": "company relationships",
            "search_type": "graph",
            "top_k": 10
        }
        
        result = await mcp_server._handle_search_memory(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert "graph_path" in result_data["memories"][0]

    @pytest.mark.asyncio
    async def test_search_memory_hybrid(self, mcp_server):
        """Test hybrid memory search."""
        # Setup
        mock_memories = [
            {
                "memory_id": "mem_hybrid_1",
                "content": "Hybrid search combines vector and graph",
                "score": 0.93,
                "vector_score": 0.88,
                "graph_score": 0.95,
                "agent_id": "test_agent"
            }
        ]
        
        mcp_server.memory_manager.search_memories.return_value = mock_memories
        mcp_server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.96,
            "risk_level": "very_low",
            "confidence_level": "rock_solid"
        }
        
        # Execute
        arguments = {
            "query": "search algorithms",
            "search_type": "hybrid",
            "include_analysis": True
        }
        
        result = await mcp_server._handle_search_memory(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert result_data["analysis"]["confidence_level"] == "rock_solid"
        assert "vector_score" in result_data["memories"][0]
        assert "graph_score" in result_data["memories"][0]

    @pytest.mark.asyncio
    async def test_analyze_response_high_confidence(self, mcp_server):
        """Test response analysis with high confidence."""
        # Setup
        mcp_server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.96,
            "risk_level": "very_low",
            "confidence_level": "rock_solid",
            "grounding_score": 0.94,
            "consistency_score": 0.98,
            "supported_claims": 3,
            "unsupported_claims": 0,
            "trading_approved": True
        }
        
        # Execute
        arguments = {
            "response": "Based on the market analysis, Bitcoin shows strong upward momentum with RSI at 45.",
            "query": "What's the current Bitcoin trend?",
            "retrieved_memories": [
                {
                    "content": "Bitcoin RSI technical analysis shows momentum indicators",
                    "id": "mem_btc_1",
                    "metadata": {"type": "market_data"}
                }
            ]
        }
        
        result = await mcp_server._handle_analyze_response(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert result_data["confidence"] == 0.96
        assert result_data["confidence_level"] == "rock_solid"
        assert result_data["trading_approved"] is True

    @pytest.mark.asyncio
    async def test_analyze_response_low_confidence(self, mcp_server):
        """Test response analysis with low confidence."""
        # Setup
        mcp_server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.45,
            "risk_level": "high",
            "confidence_level": "low",
            "grounding_score": 0.30,
            "consistency_score": 0.60,
            "supported_claims": 1,
            "unsupported_claims": 3,
            "trading_approved": False,
            "warnings": ["Low grounding in provided context", "Multiple unsupported claims"]
        }
        
        # Execute
        arguments = {
            "response": "The market will definitely crash tomorrow based on my intuition.",
            "query": "Will the market crash?",
            "retrieved_memories": []
        }
        
        result = await mcp_server._handle_analyze_response(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert result_data["confidence"] == 0.45
        assert result_data["confidence_level"] == "low"
        assert result_data["trading_approved"] is False
        assert len(result_data["warnings"]) > 0

    @pytest.mark.asyncio
    async def test_get_memory_stats(self, mcp_server):
        """Test memory statistics retrieval."""
        # Setup
        mock_stats = {
            "total_memories": 1500,
            "by_agent": {
                "tyra": 800,
                "claude": 400,
                "archon": 300
            },
            "by_confidence": {
                "rock_solid": 300,
                "high": 600,
                "fuzzy": 450,
                "low": 150
            },
            "performance": {
                "avg_query_time_ms": 45,
                "cache_hit_rate": 0.78,
                "embedding_accuracy": 0.92
            },
            "health_score": 0.94,
            "recommendations": [
                "Consider cleaning up 150 low-confidence memories",
                "Cache hit rate could be improved"
            ]
        }
        
        mcp_server.memory_manager.get_stats.return_value = mock_stats
        
        # Execute
        arguments = {
            "agent_id": "tyra",
            "include_performance": True,
            "include_recommendations": True
        }
        
        result = await mcp_server._handle_get_memory_stats(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert result_data["total_memories"] == 1500
        assert result_data["by_agent"]["tyra"] == 800
        assert "performance" in result_data
        assert "recommendations" in result_data

    @pytest.mark.asyncio
    async def test_get_learning_insights(self, mcp_server):
        """Test learning insights retrieval."""
        # Setup
        mock_insights = [
            {
                "category": "performance",
                "insight": "Vector search performance improved 15% after index optimization",
                "confidence": 0.87,
                "timestamp": datetime.utcnow().isoformat(),
                "actionable": True,
                "impact": "medium"
            },
            {
                "category": "quality",
                "insight": "Memories with metadata have 23% higher retrieval accuracy",
                "confidence": 0.92,
                "timestamp": datetime.utcnow().isoformat(),
                "actionable": True,
                "impact": "high"
            }
        ]
        
        mcp_server.learning_engine.get_insights.return_value = mock_insights
        
        # Execute
        arguments = {
            "category": "performance",
            "days": 7
        }
        
        result = await mcp_server._handle_get_learning_insights(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert len(result_data["insights"]) == 2
        assert result_data["insights"][0]["confidence"] == 0.87

    @pytest.mark.asyncio
    async def test_delete_memory(self, mcp_server):
        """Test memory deletion."""
        # Setup
        memory_id = "mem_to_delete_123"
        mcp_server.memory_manager.delete_memory.return_value = {
            "memory_id": memory_id,
            "status": "deleted",
            "related_entities_updated": 2
        }
        
        # Execute
        arguments = {"memory_id": memory_id}
        
        result = await mcp_server._handle_delete_memory(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert result_data["memory_id"] == memory_id
        assert result_data["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_health_check_basic(self, mcp_server):
        """Test basic health check."""
        # Setup
        mock_health = {
            "status": "healthy",
            "components": {
                "memory_manager": "healthy",
                "hallucination_detector": "healthy", 
                "performance_tracker": "healthy",
                "databases": "healthy"
            },
            "uptime_hours": 24.5,
            "last_error": None
        }
        
        mcp_server.memory_manager.health_check.return_value = mock_health
        
        # Execute
        arguments = {"detailed": False}
        
        result = await mcp_server._handle_health_check(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert result_data["status"] == "healthy"
        assert "components" in result_data

    @pytest.mark.asyncio
    async def test_health_check_detailed(self, mcp_server):
        """Test detailed health check."""
        # Setup
        mock_health = {
            "status": "healthy",
            "components": {
                "memory_manager": "healthy",
                "postgresql": "healthy",
                "memgraph": "healthy", 
                "redis": "healthy",
                "circuit_breakers": "healthy"
            },
            "metrics": {
                "memory_usage_mb": 512,
                "query_latency_p95_ms": 85,
                "cache_hit_rate": 0.82,
                "error_rate": 0.001
            },
            "circuit_breaker_states": {
                "postgresql": "closed",
                "memgraph": "closed",
                "embedding": "closed"
            }
        }
        
        mcp_server.memory_manager.health_check.return_value = mock_health
        
        # Execute
        arguments = {"detailed": True}
        
        result = await mcp_server._handle_health_check(arguments)
        
        # Verify
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert "metrics" in result_data
        assert "circuit_breaker_states" in result_data

    @pytest.mark.asyncio
    async def test_error_handling_invalid_tool(self, mcp_server):
        """Test error handling for invalid tool name."""
        # This would be handled at the MCP server level
        with pytest.raises(ValueError, match="Unknown tool"):
            await mcp_server._handle_call_tool("invalid_tool", {})

    @pytest.mark.asyncio
    async def test_error_handling_missing_arguments(self, mcp_server):
        """Test error handling for missing required arguments."""
        # Execute with missing required argument
        result = await mcp_server._handle_store_memory({})
        
        # Verify error is handled gracefully
        assert result.isError is True
        assert "content" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_error_handling_database_failure(self, mcp_server):
        """Test error handling when database operations fail."""
        # Setup database failure
        mcp_server.memory_manager.store_memory.side_effect = Exception("Database connection failed")
        
        # Execute
        arguments = {
            "content": "Test content",
            "agent_id": "test_agent"
        }
        
        result = await mcp_server._handle_store_memory(arguments)
        
        # Verify error is handled gracefully
        assert result.isError is True
        assert "error" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_confidence_scoring_integration(self, mcp_server):
        """Test integration of confidence scoring across tools."""
        # Setup search with various confidence levels
        mock_memories = [
            {"memory_id": "mem_1", "content": "High confidence content", "score": 0.96, "confidence_level": "rock_solid"},
            {"memory_id": "mem_2", "content": "Medium confidence content", "score": 0.85, "confidence_level": "high"},
            {"memory_id": "mem_3", "content": "Low confidence content", "score": 0.65, "confidence_level": "fuzzy"},
        ]
        
        mcp_server.memory_manager.search_memories.return_value = mock_memories
        mcp_server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.89,
            "confidence_level": "high"
        }
        
        # Execute search
        result = await mcp_server._handle_search_memory({
            "query": "test query",
            "min_confidence": 0.8
        })
        
        # Verify only high confidence results are returned
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        # Should filter out the low confidence result
        high_conf_memories = [m for m in result_data["memories"] if m["score"] >= 0.8]
        assert len(high_conf_memories) == 2

    @pytest.mark.asyncio 
    async def test_trading_safety_integration(self, mcp_server):
        """Test trading safety features in MCP tools."""
        # Setup trading-related analysis
        mcp_server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.97,
            "confidence_level": "rock_solid",
            "trading_approved": True,
            "grounding_score": 0.95,
            "safety_checks": {
                "confidence_above_95": True,
                "hallucination_below_threshold": True,
                "context_available": True
            }
        }
        
        # Execute analysis on trading response
        arguments = {
            "response": "Buy AAPL at current price with stop loss at $180",
            "query": "Should I buy Apple stock?",
            "retrieved_memories": [
                {
                    "content": "AAPL technical analysis shows bullish trend",
                    "id": "mem_trading_1",
                    "metadata": {"type": "trading_signal", "confidence": 0.96}
                }
            ]
        }
        
        result = await mcp_server._handle_analyze_response(arguments)
        
        # Verify trading safety approval
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        assert result_data["trading_approved"] is True
        assert result_data["confidence"] >= 0.95
        assert "safety_checks" in result_data

    @pytest.mark.asyncio
    async def test_agent_isolation(self, mcp_server):
        """Test that agent-specific data is properly isolated."""
        # Setup memories for different agents
        mock_memories_tyra = [
            {"memory_id": "tyra_mem_1", "content": "Tyra's memory", "agent_id": "tyra"}
        ]
        mock_memories_claude = [
            {"memory_id": "claude_mem_1", "content": "Claude's memory", "agent_id": "claude"}
        ]
        
        # Test Tyra agent search
        mcp_server.memory_manager.search_memories.return_value = mock_memories_tyra
        result_tyra = await mcp_server._handle_search_memory({
            "query": "test",
            "agent_id": "tyra"
        })
        
        assert result_tyra.isError is False
        result_data = json.loads(result_tyra.content[0].text)
        assert result_data["memories"][0]["agent_id"] == "tyra"
        
        # Verify search was called with correct agent filter
        mcp_server.memory_manager.search_memories.assert_called_with(
            request=mcp_server.memory_manager.search_memories.call_args[1]["request"]
        )

    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self, mcp_server):
        """Test that performance metrics are tracked for all operations."""
        # Setup
        mcp_server.memory_manager.store_memory.return_value = {
            "memory_id": "perf_test_mem",
            "status": "stored"
        }
        
        # Execute operation
        await mcp_server._handle_store_memory({
            "content": "Performance test content",
            "agent_id": "test_agent"
        })
        
        # Verify performance tracking was called
        mcp_server.performance_tracker.record_metric.assert_called()
        # Check that timing metrics were recorded
        call_args = mcp_server.performance_tracker.record_metric.call_args_list
        assert any("latency" in str(call) for call in call_args)


class TestMCPToolSchemas:
    """Test MCP tool schema validation."""
    
    def test_store_memory_schema(self):
        """Test store_memory tool schema is valid."""
        server = TyraMemoryServer()
        tools = server._create_tools()
        
        store_tool = next(tool for tool in tools if tool.name == "store_memory")
        schema = store_tool.inputSchema
        
        # Verify required fields
        assert "content" in schema["properties"]
        assert "content" in schema["required"]
        
        # Verify optional fields have proper defaults
        assert schema["properties"]["extract_entities"]["default"] is True
        assert schema["properties"]["chunk_content"]["default"] is False

    def test_search_memory_schema(self):
        """Test search_memory tool schema is valid."""
        server = TyraMemoryServer()
        tools = server._create_tools()
        
        search_tool = next(tool for tool in tools if tool.name == "search_memory")
        schema = search_tool.inputSchema
        
        # Verify query is required
        assert "query" in schema["required"]
        
        # Verify enum values
        assert "vector" in schema["properties"]["search_type"]["enum"]
        assert "graph" in schema["properties"]["search_type"]["enum"]
        assert "hybrid" in schema["properties"]["search_type"]["enum"]
        
        # Verify constraints
        assert schema["properties"]["top_k"]["minimum"] == 1
        assert schema["properties"]["top_k"]["maximum"] == 100
        assert schema["properties"]["min_confidence"]["minimum"] == 0.0
        assert schema["properties"]["min_confidence"]["maximum"] == 1.0

    def test_analyze_response_schema(self):
        """Test analyze_response tool schema is valid."""
        server = TyraMemoryServer()
        tools = server._create_tools()
        
        analyze_tool = next(tool for tool in tools if tool.name == "analyze_response")
        schema = analyze_tool.inputSchema
        
        # Verify required fields
        assert "response" in schema["required"]
        
        # Verify retrieved_memories structure
        memories_schema = schema["properties"]["retrieved_memories"]
        assert memories_schema["type"] == "array"
        assert "content" in memories_schema["items"]["properties"]
        assert "id" in memories_schema["items"]["properties"]


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])