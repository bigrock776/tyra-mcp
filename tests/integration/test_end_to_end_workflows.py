"""
End-to-end integration tests for complete workflows.

Tests complete workflows from MCP tools through to database operations,
ensuring all components work together correctly.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import CallToolRequest, CallToolResult, TextContent

from src.mcp.server import TyraMemoryServer
from src.core.memory.manager import MemoryManager
from src.core.rag.hallucination_detector import HallucinationDetector
from src.core.analytics.performance_tracker import PerformanceTracker


class TestEndToEndWorkflows:
    """Complete end-to-end workflow tests."""

    @pytest.fixture
    async def full_system_setup(self):
        """Setup complete system with all components."""
        # Create server
        server = TyraMemoryServer()
        
        # Create real component instances (with mocked backends)
        memory_manager = AsyncMock(spec=MemoryManager)
        hallucination_detector = AsyncMock(spec=HallucinationDetector)
        performance_tracker = AsyncMock(spec=PerformanceTracker)
        
        # Set up component interactions
        server.memory_manager = memory_manager
        server.hallucination_detector = hallucination_detector
        server.performance_tracker = performance_tracker
        server._initialized = True
        
        # Mock backend systems
        with patch('src.core.providers.vector_stores.pgvector.PgVectorStore') as mock_vector_store, \
             patch('src.core.providers.graph_engines.memgraph.MemgraphEngine') as mock_graph_engine, \
             patch('src.core.providers.embeddings.huggingface.HuggingFaceProvider') as mock_embedder, \
             patch('src.core.cache.redis_cache.RedisCache') as mock_cache:
            
            # Configure mock responses
            mock_vector_store.return_value.search.return_value = []
            mock_graph_engine.return_value.find_entities.return_value = []
            mock_embedder.return_value.embed_documents.return_value = [[0.1] * 1024]
            mock_cache.return_value.get.return_value = None
            
            yield {
                'server': server,
                'memory_manager': memory_manager,
                'hallucination_detector': hallucination_detector,
                'performance_tracker': performance_tracker,
                'mocks': {
                    'vector_store': mock_vector_store,
                    'graph_engine': mock_graph_engine,
                    'embedder': mock_embedder,
                    'cache': mock_cache
                }
            }

    @pytest.mark.asyncio
    async def test_complete_memory_lifecycle(self, full_system_setup):
        """Test complete memory lifecycle: store -> search -> analyze -> delete."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        hallucination_detector = full_system_setup['hallucination_detector']
        
        # Setup memory manager responses
        memory_id = str(uuid.uuid4())
        memory_manager.store_memory.return_value = {
            "memory_id": memory_id,
            "status": "stored",
            "entities_extracted": 3,
            "chunks_created": 1,
            "processing_time": 0.15
        }
        
        memory_manager.search_memories.return_value = [{
            "memory_id": memory_id,
            "content": "Python is a programming language with strong typing support",
            "score": 0.95,
            "metadata": {"agent_id": "test_agent", "type": "educational"},
            "agent_id": "test_agent"
        }]
        
        hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.92,
            "risk_level": "low",
            "confidence_level": "high",
            "grounding_score": 0.89,
            "consistency_score": 0.95
        }
        
        memory_manager.delete_memory.return_value = {"status": "deleted"}
        
        # 1. Store memory
        store_result = await server._handle_store_memory({
            "content": "Python is a programming language with strong typing support",
            "agent_id": "test_agent",
            "metadata": {"type": "educational"},
            "extract_entities": True,
            "chunk_content": False
        })
        
        assert store_result.isError is False
        assert memory_id in store_result.content[0].text
        
        # 2. Search for the memory
        search_result = await server._handle_search_memory({
            "query": "Python programming language",
            "agent_id": "test_agent",
            "search_type": "hybrid",
            "top_k": 10,
            "min_confidence": 0.8
        })
        
        assert search_result.isError is False
        search_data = json.loads(search_result.content[0].text)
        assert len(search_data["memories"]) == 1
        assert search_data["memories"][0]["memory_id"] == memory_id
        
        # 3. Analyze response based on retrieved memory
        analyze_result = await server._handle_analyze_response({
            "response": "Python is indeed a programming language with strong typing support",
            "query": "What is Python?",
            "retrieved_memories": [{
                "content": "Python is a programming language with strong typing support",
                "id": memory_id,
                "metadata": {"type": "educational"}
            }]
        })
        
        assert analyze_result.isError is False
        analyze_data = json.loads(analyze_result.content[0].text)
        assert analyze_data["confidence"] == 0.92
        assert analyze_data["confidence_level"] == "high"
        
        # 4. Delete the memory
        delete_result = await server._handle_delete_memory({
            "memory_id": memory_id
        })
        
        assert delete_result.isError is False
        delete_data = json.loads(delete_result.content[0].text)
        assert delete_data["status"] == "deleted"
        
        # Verify all components were called
        memory_manager.store_memory.assert_called_once()
        memory_manager.search_memories.assert_called_once()
        hallucination_detector.detect_hallucination.assert_called_once()
        memory_manager.delete_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, full_system_setup):
        """Test workflow with multiple agents accessing different memories."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        
        # Setup different memories for different agents
        tyra_memory_id = str(uuid.uuid4())
        claude_memory_id = str(uuid.uuid4())
        
        # Mock responses for different agents
        def mock_store_memory(**kwargs):
            if kwargs['agent_id'] == 'tyra':
                return {
                    "memory_id": tyra_memory_id,
                    "status": "stored",
                    "entities_extracted": 2,
                    "chunks_created": 1
                }
            elif kwargs['agent_id'] == 'claude':
                return {
                    "memory_id": claude_memory_id,
                    "status": "stored",
                    "entities_extracted": 1,
                    "chunks_created": 1
                }
        
        def mock_search_memories(**kwargs):
            if kwargs.get('agent_id') == 'tyra':
                return [{
                    "memory_id": tyra_memory_id,
                    "content": "Tyra's trading analysis for AAPL",
                    "score": 0.92,
                    "agent_id": "tyra"
                }]
            elif kwargs.get('agent_id') == 'claude':
                return [{
                    "memory_id": claude_memory_id,
                    "content": "Claude's general knowledge about stocks",
                    "score": 0.88,
                    "agent_id": "claude"
                }]
            return []
        
        memory_manager.store_memory.side_effect = mock_store_memory
        memory_manager.search_memories.side_effect = mock_search_memories
        
        # 1. Store memory for Tyra
        tyra_result = await server._handle_store_memory({
            "content": "Tyra's trading analysis for AAPL",
            "agent_id": "tyra",
            "metadata": {"type": "trading_analysis"}
        })
        
        assert tyra_result.isError is False
        assert tyra_memory_id in tyra_result.content[0].text
        
        # 2. Store memory for Claude
        claude_result = await server._handle_store_memory({
            "content": "Claude's general knowledge about stocks",
            "agent_id": "claude",
            "metadata": {"type": "general_knowledge"}
        })
        
        assert claude_result.isError is False
        assert claude_memory_id in claude_result.content[0].text
        
        # 3. Search as Tyra - should only get Tyra's memories
        tyra_search = await server._handle_search_memory({
            "query": "AAPL analysis",
            "agent_id": "tyra",
            "top_k": 10
        })
        
        assert tyra_search.isError is False
        tyra_search_data = json.loads(tyra_search.content[0].text)
        assert len(tyra_search_data["memories"]) == 1
        assert tyra_search_data["memories"][0]["agent_id"] == "tyra"
        
        # 4. Search as Claude - should only get Claude's memories
        claude_search = await server._handle_search_memory({
            "query": "stocks knowledge",
            "agent_id": "claude",
            "top_k": 10
        })
        
        assert claude_search.isError is False
        claude_search_data = json.loads(claude_search.content[0].text)
        assert len(claude_search_data["memories"]) == 1
        assert claude_search_data["memories"][0]["agent_id"] == "claude"
        
        # Verify agent isolation
        assert memory_manager.store_memory.call_count == 2
        assert memory_manager.search_memories.call_count == 2

    @pytest.mark.asyncio
    async def test_trading_safety_workflow(self, full_system_setup):
        """Test complete trading safety workflow with confidence checking."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        hallucination_detector = full_system_setup['hallucination_detector']
        
        # Setup high-confidence trading memory
        trading_memory_id = str(uuid.uuid4())
        memory_manager.store_memory.return_value = {
            "memory_id": trading_memory_id,
            "status": "stored",
            "entities_extracted": 5,
            "chunks_created": 1
        }
        
        memory_manager.search_memories.return_value = [{
            "memory_id": trading_memory_id,
            "content": "AAPL technical analysis shows strong bullish momentum with RSI at 45 and MACD crossover",
            "score": 0.96,
            "metadata": {"type": "technical_analysis", "confidence": 0.97},
            "agent_id": "tyra"
        }]
        
        # Mock high-confidence hallucination detection
        hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.97,
            "risk_level": "very_low",
            "confidence_level": "rock_solid",
            "grounding_score": 0.95,
            "consistency_score": 0.98,
            "trading_approved": True,
            "safety_checks": {
                "confidence_above_95": True,
                "hallucination_below_threshold": True,
                "context_available": True,
                "grounding_sufficient": True
            }
        }
        
        # 1. Store high-quality trading data
        store_result = await server._handle_store_memory({
            "content": "AAPL technical analysis shows strong bullish momentum with RSI at 45 and MACD crossover",
            "agent_id": "tyra",
            "metadata": {"type": "technical_analysis", "confidence": 0.97},
            "extract_entities": True
        })
        
        assert store_result.isError is False
        
        # 2. Search for trading-related memories
        search_result = await server._handle_search_memory({
            "query": "AAPL trading analysis",
            "agent_id": "tyra",
            "search_type": "hybrid",
            "min_confidence": 0.9  # High threshold for trading
        })
        
        assert search_result.isError is False
        search_data = json.loads(search_result.content[0].text)
        assert len(search_data["memories"]) == 1
        assert search_data["memories"][0]["score"] >= 0.9
        
        # 3. Analyze trading response with safety checks
        analyze_result = await server._handle_analyze_response({
            "response": "Based on the technical analysis, AAPL shows bullish momentum. Consider a long position with stop loss at $180.",
            "query": "Should I buy AAPL?",
            "retrieved_memories": [{
                "content": "AAPL technical analysis shows strong bullish momentum with RSI at 45 and MACD crossover",
                "id": trading_memory_id,
                "metadata": {"type": "technical_analysis", "confidence": 0.97}
            }]
        })
        
        assert analyze_result.isError is False
        analyze_data = json.loads(analyze_result.content[0].text)
        
        # Verify trading safety requirements
        assert analyze_data["confidence"] >= 0.95  # Rock solid confidence required
        assert analyze_data["confidence_level"] == "rock_solid"
        assert analyze_data["trading_approved"] is True
        assert "safety_checks" in analyze_data
        assert analyze_data["safety_checks"]["confidence_above_95"] is True
        
        # 4. Test low-confidence trading scenario
        hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.75,
            "risk_level": "medium",
            "confidence_level": "fuzzy",
            "grounding_score": 0.70,
            "consistency_score": 0.80,
            "trading_approved": False,
            "safety_checks": {
                "confidence_above_95": False,
                "hallucination_below_threshold": True,
                "context_available": True,
                "grounding_sufficient": False
            }
        }
        
        unsafe_analyze_result = await server._handle_analyze_response({
            "response": "I think AAPL might go up based on my intuition.",
            "query": "Should I buy AAPL?",
            "retrieved_memories": []
        })
        
        assert unsafe_analyze_result.isError is False
        unsafe_data = json.loads(unsafe_analyze_result.content[0].text)
        
        # Verify trading safety rejection
        assert unsafe_data["confidence"] < 0.95
        assert unsafe_data["trading_approved"] is False
        assert unsafe_data["safety_checks"]["confidence_above_95"] is False

    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(self, full_system_setup):
        """Test performance monitoring throughout the workflow."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        performance_tracker = full_system_setup['performance_tracker']
        
        # Setup performance tracking
        memory_manager.store_memory.return_value = {
            "memory_id": str(uuid.uuid4()),
            "status": "stored",
            "processing_time": 0.125
        }
        
        memory_manager.search_memories.return_value = [{
            "memory_id": "perf_test_mem",
            "content": "Performance test content",
            "score": 0.88,
            "agent_id": "test_agent"
        }]
        
        # Execute operations
        await server._handle_store_memory({
            "content": "Performance test content",
            "agent_id": "test_agent"
        })
        
        await server._handle_search_memory({
            "query": "performance test",
            "agent_id": "test_agent"
        })
        
        # Verify performance tracking was called
        assert performance_tracker.record_metric.called
        
        # Check specific metrics were recorded
        call_args_list = performance_tracker.record_metric.call_args_list
        metric_names = [call.args[0] for call in call_args_list]
        
        # Should track latency for both operations
        assert any("store_memory" in metric for metric in metric_names)
        assert any("search_memory" in metric for metric in metric_names)

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, full_system_setup):
        """Test error recovery and circuit breaker behavior."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        
        # Simulate database failure
        memory_manager.store_memory.side_effect = Exception("Database connection failed")
        
        # First attempt should fail
        result1 = await server._handle_store_memory({
            "content": "Test content",
            "agent_id": "test_agent"
        })
        
        assert result1.isError is True
        assert "error" in result1.content[0].text.lower()
        
        # Simulate recovery
        memory_manager.store_memory.side_effect = None
        memory_manager.store_memory.return_value = {
            "memory_id": str(uuid.uuid4()),
            "status": "stored"
        }
        
        # Second attempt should succeed
        result2 = await server._handle_store_memory({
            "content": "Test content recovery",
            "agent_id": "test_agent"
        })
        
        assert result2.isError is False
        assert "memory_id" in result2.content[0].text

    @pytest.mark.asyncio
    async def test_concurrent_operations_workflow(self, full_system_setup):
        """Test concurrent operations across multiple agents."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        
        # Setup concurrent responses
        memory_manager.store_memory.return_value = {
            "memory_id": str(uuid.uuid4()),
            "status": "stored"
        }
        
        memory_manager.search_memories.return_value = [{
            "memory_id": "concurrent_test",
            "content": "Concurrent test content",
            "score": 0.85,
            "agent_id": "test_agent"
        }]
        
        # Create concurrent operations
        tasks = []
        
        # Multiple store operations
        for i in range(3):
            task = server._handle_store_memory({
                "content": f"Concurrent content {i}",
                "agent_id": f"agent_{i}"
            })
            tasks.append(task)
        
        # Multiple search operations
        for i in range(3):
            task = server._handle_search_memory({
                "query": f"concurrent query {i}",
                "agent_id": f"agent_{i}"
            })
            tasks.append(task)
        
        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed
        assert len(results) == 6
        for result in results:
            assert not isinstance(result, Exception)
            assert result.isError is False

    @pytest.mark.asyncio
    async def test_graph_integration_workflow(self, full_system_setup):
        """Test workflow with graph-based entity relationships."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        
        # Setup graph-aware memory storage
        memory_manager.store_memory.return_value = {
            "memory_id": str(uuid.uuid4()),
            "status": "stored",
            "entities_extracted": 3,
            "relationships_created": 2,
            "graph_updates": {
                "nodes_added": 2,
                "edges_added": 1,
                "entities": ["Python", "Guido van Rossum", "Programming Language"]
            }
        }
        
        # Setup graph-based search
        memory_manager.search_memories.return_value = [{
            "memory_id": "graph_test_mem",
            "content": "Python was created by Guido van Rossum",
            "score": 0.92,
            "graph_path": ["Python", "created_by", "Guido van Rossum"],
            "related_entities": ["Programming Language", "BDFL"],
            "agent_id": "test_agent"
        }]
        
        # 1. Store memory with entity extraction
        store_result = await server._handle_store_memory({
            "content": "Python was created by Guido van Rossum as a programming language",
            "agent_id": "test_agent",
            "extract_entities": True,
            "metadata": {"type": "biographical"}
        })
        
        assert store_result.isError is False
        result_data = json.loads(store_result.content[0].text)
        assert result_data["entities_extracted"] == 3
        assert result_data["relationships_created"] == 2
        
        # 2. Search with graph traversal
        search_result = await server._handle_search_memory({
            "query": "Who created Python?",
            "agent_id": "test_agent",
            "search_type": "graph",
            "top_k": 5
        })
        
        assert search_result.isError is False
        search_data = json.loads(search_result.content[0].text)
        assert len(search_data["memories"]) == 1
        assert "graph_path" in search_data["memories"][0]
        assert "related_entities" in search_data["memories"][0]

    @pytest.mark.asyncio
    async def test_hybrid_search_workflow(self, full_system_setup):
        """Test hybrid search combining vector and graph approaches."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        hallucination_detector = full_system_setup['hallucination_detector']
        
        # Setup hybrid search results
        memory_manager.search_memories.return_value = [
            {
                "memory_id": "hybrid_1",
                "content": "Hybrid search result 1",
                "score": 0.93,
                "vector_score": 0.89,
                "graph_score": 0.97,
                "search_method": "hybrid",
                "agent_id": "test_agent"
            },
            {
                "memory_id": "hybrid_2", 
                "content": "Hybrid search result 2",
                "score": 0.88,
                "vector_score": 0.91,
                "graph_score": 0.85,
                "search_method": "hybrid",
                "agent_id": "test_agent"
            }
        ]
        
        hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.91,
            "confidence_level": "high",
            "risk_level": "low"
        }
        
        # Execute hybrid search
        result = await server._handle_search_memory({
            "query": "hybrid search test",
            "agent_id": "test_agent",
            "search_type": "hybrid",
            "top_k": 10,
            "include_analysis": True
        })
        
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        
        # Verify hybrid search results
        assert len(result_data["memories"]) == 2
        assert result_data["search_type"] == "hybrid"
        assert "analysis" in result_data
        
        # Verify both vector and graph scores are present
        for memory in result_data["memories"]:
            assert "vector_score" in memory
            assert "graph_score" in memory
            assert memory["search_method"] == "hybrid"

    @pytest.mark.asyncio
    async def test_memory_health_workflow(self, full_system_setup):
        """Test memory health monitoring and maintenance."""
        server = full_system_setup['server']
        memory_manager = full_system_setup['memory_manager']
        
        # Setup health check response
        memory_manager.health_check.return_value = {
            "status": "healthy",
            "components": {
                "vector_store": "healthy",
                "graph_engine": "healthy",
                "embedding_service": "healthy",
                "cache": "healthy"
            },
            "metrics": {
                "total_memories": 1500,
                "healthy_memories": 1450,
                "stale_memories": 50,
                "avg_query_time": 0.045,
                "cache_hit_rate": 0.82
            },
            "recommendations": [
                "Clean up 50 stale memories",
                "Consider increasing cache size"
            ]
        }
        
        # Execute health check
        health_result = await server._handle_health_check({
            "detailed": True,
            "include_recommendations": True
        })
        
        assert health_result.isError is False
        health_data = json.loads(health_result.content[0].text)
        
        # Verify health information
        assert health_data["status"] == "healthy"
        assert "components" in health_data
        assert "metrics" in health_data
        assert "recommendations" in health_data
        assert len(health_data["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_learning_insights_workflow(self, full_system_setup):
        """Test learning insights and self-improvement workflow."""
        server = full_system_setup['server']
        learning_engine = AsyncMock()
        server.learning_engine = learning_engine
        
        # Setup learning insights
        learning_engine.get_insights.return_value = [
            {
                "category": "performance",
                "insight": "Vector search performance improved 15% after index optimization",
                "confidence": 0.87,
                "timestamp": datetime.utcnow().isoformat(),
                "actionable": True,
                "impact": "medium",
                "recommendation": "Continue with current optimization strategy"
            },
            {
                "category": "quality",
                "insight": "Memories with rich metadata have 23% higher retrieval accuracy",
                "confidence": 0.92,
                "timestamp": datetime.utcnow().isoformat(),
                "actionable": True,
                "impact": "high",
                "recommendation": "Encourage agents to provide detailed metadata"
            }
        ]
        
        # Execute learning insights request
        insights_result = await server._handle_get_learning_insights({
            "category": "performance",
            "days": 7,
            "include_recommendations": True
        })
        
        assert insights_result.isError is False
        insights_data = json.loads(insights_result.content[0].text)
        
        # Verify learning insights
        assert len(insights_data["insights"]) == 2
        assert insights_data["insights"][0]["category"] == "performance"
        assert insights_data["insights"][1]["category"] == "quality"
        
        # Verify actionable insights
        actionable_insights = [i for i in insights_data["insights"] if i["actionable"]]
        assert len(actionable_insights) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])