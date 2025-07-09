"""
MCP Server initialization and configuration tests.

Tests server setup, component initialization, and basic functionality.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.server import Server
from mcp.types import Tool

from src.mcp.server import TyraMemoryServer


class TestMCPServer:
    """Test MCP server setup and configuration."""

    def test_server_initialization(self):
        """Test that server initializes correctly."""
        server = TyraMemoryServer()
        
        # Verify server instance
        assert isinstance(server.server, Server)
        assert server._initialized is False
        assert server._total_requests == 0

    def test_tools_creation(self):
        """Test that all required tools are created."""
        server = TyraMemoryServer()
        tools = server._create_tools()
        
        # Verify tool count and names
        expected_tools = [
            "store_memory",
            "search_memory", 
            "analyze_response",
            "get_memory_stats",
            "get_learning_insights",
            "delete_memory",
            "health_check"
        ]
        
        tool_names = [tool.name for tool in tools]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
        
        assert len(tools) == len(expected_tools)

    def test_tool_schemas_valid(self):
        """Test that all tool schemas are properly formatted."""
        server = TyraMemoryServer()
        tools = server._create_tools()
        
        for tool in tools:
            # Verify basic structure
            assert isinstance(tool.name, str)
            assert isinstance(tool.description, str)
            assert isinstance(tool.inputSchema, dict)
            
            # Verify schema has required structure
            schema = tool.inputSchema
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema
            
            # If there are required fields, verify they exist in properties
            if "required" in schema:
                for required_field in schema["required"]:
                    assert required_field in schema["properties"]

    @pytest.mark.asyncio
    async def test_component_initialization(self):
        """Test that components are initialized correctly."""
        server = TyraMemoryServer()
        
        with patch('src.mcp.server.MemoryManager') as mock_memory_manager, \
             patch('src.mcp.server.HallucinationDetector') as mock_hallucination_detector, \
             patch('src.mcp.server.PerformanceTracker') as mock_performance_tracker, \
             patch('src.mcp.server.LearningEngine') as mock_learning_engine:
            
            # Mock the constructors
            mock_memory_manager.return_value = AsyncMock()
            mock_hallucination_detector.return_value = AsyncMock()
            mock_performance_tracker.return_value = MagicMock()
            mock_learning_engine.return_value = AsyncMock()
            
            # Initialize components
            await server._initialize_components()
            
            # Verify initialization
            assert server._initialized is True
            assert server.memory_manager is not None
            assert server.hallucination_detector is not None
            assert server.performance_tracker is not None
            assert server.learning_engine is not None
            
            # Verify constructors were called
            mock_memory_manager.assert_called_once()
            mock_hallucination_detector.assert_called_once()
            mock_performance_tracker.assert_called_once()
            mock_learning_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_lazy_initialization(self):
        """Test that components are initialized lazily on first request."""
        server = TyraMemoryServer()
        
        with patch.object(server, '_initialize_components') as mock_init:
            mock_init.return_value = None
            server._initialized = False
            
            # Mock the handle method to avoid actual execution
            with patch.object(server, '_handle_health_check') as mock_handle:
                mock_handle.return_value = MagicMock()
                
                # Make a tool call
                await server._handle_call_tool("health_check", {})
                
                # Verify initialization was called
                mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_during_initialization(self):
        """Test error handling when component initialization fails."""
        server = TyraMemoryServer()
        
        with patch('src.mcp.server.MemoryManager') as mock_memory_manager:
            # Make initialization fail
            mock_memory_manager.side_effect = Exception("Initialization failed")
            
            # Attempt initialization
            try:
                await server._initialize_components()
                assert False, "Should have raised an exception"
            except Exception as e:
                assert "Initialization failed" in str(e)

    def test_tool_routing(self):
        """Test that tool names are correctly routed to handlers."""
        server = TyraMemoryServer()
        
        # Verify handler methods exist
        assert hasattr(server, '_handle_store_memory')
        assert hasattr(server, '_handle_search_memory')
        assert hasattr(server, '_handle_analyze_response')
        assert hasattr(server, '_handle_get_memory_stats')
        assert hasattr(server, '_handle_get_learning_insights')
        assert hasattr(server, '_handle_delete_memory')
        assert hasattr(server, '_handle_health_check')

    @pytest.mark.asyncio
    async def test_request_tracking(self):
        """Test that requests are properly tracked."""
        server = TyraMemoryServer()
        server._initialized = True
        
        # Mock a handler
        with patch.object(server, '_handle_health_check') as mock_handler:
            mock_handler.return_value = MagicMock()
            
            initial_count = server._total_requests
            
            # Make multiple requests
            await server._handle_call_tool("health_check", {})
            await server._handle_call_tool("health_check", {})
            
            # Verify count increased
            assert server._total_requests == initial_count + 2

    @pytest.mark.asyncio
    async def test_performance_metrics_recording(self):
        """Test that performance metrics are recorded for each request."""
        server = TyraMemoryServer()
        server._initialized = True
        server.performance_tracker = MagicMock()
        
        with patch.object(server, '_handle_health_check') as mock_handler:
            mock_handler.return_value = MagicMock()
            
            # Make a request
            await server._handle_call_tool("health_check", {})
            
            # Verify performance tracking was called
            server.performance_tracker.record_metric.assert_called()

    def test_server_configuration(self):
        """Test server configuration settings."""
        server = TyraMemoryServer()
        
        # Verify server has proper configuration
        assert hasattr(server, 'server')
        assert hasattr(server, '_initialized')
        assert hasattr(server, '_total_requests')
        
        # Test that server can be configured with different settings
        # This would be expanded based on actual configuration options
        pass

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        server = TyraMemoryServer()
        server._initialized = True
        
        # Mock components
        server.memory_manager = AsyncMock()
        server.hallucination_detector = AsyncMock()
        server.performance_tracker = MagicMock()
        server.learning_engine = AsyncMock()
        
        # Mock handler to simulate async work
        async def mock_handler(args):
            await asyncio.sleep(0.01)  # Simulate async work
            return MagicMock()
        
        with patch.object(server, '_handle_health_check', side_effect=mock_handler):
            # Make concurrent requests
            tasks = [
                server._handle_call_tool("health_check", {})
                for _ in range(5)
            ]
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all completed successfully
            assert len(results) == 5
            assert all(not isinstance(result, Exception) for result in results)

    @pytest.mark.asyncio 
    async def test_invalid_tool_name(self):
        """Test handling of invalid tool names."""
        server = TyraMemoryServer()
        server._initialized = True
        
        # Test invalid tool name
        with pytest.raises(ValueError, match="Unknown tool"):
            await server._handle_call_tool("invalid_tool_name", {})

    @pytest.mark.asyncio
    async def test_malformed_arguments(self):
        """Test handling of malformed arguments."""
        server = TyraMemoryServer()
        server._initialized = True
        
        # Mock components
        server.memory_manager = AsyncMock()
        server.performance_tracker = MagicMock()
        
        # Test with various malformed argument scenarios
        test_cases = [
            None,  # None arguments
            "invalid",  # String instead of dict
            [],  # List instead of dict
        ]
        
        for invalid_args in test_cases:
            result = await server._handle_call_tool("health_check", invalid_args)
            # Should handle gracefully and return error result
            assert result.isError is True


class TestMCPServerIntegration:
    """Integration tests for MCP server with real components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_memory_flow(self):
        """Test complete memory storage and retrieval flow."""
        server = TyraMemoryServer()
        
        # Mock components but simulate realistic responses
        server.memory_manager = AsyncMock()
        server.hallucination_detector = AsyncMock()
        server.performance_tracker = MagicMock()
        server.learning_engine = AsyncMock()
        server._initialized = True
        
        # Setup realistic responses
        memory_id = "test_memory_123"
        server.memory_manager.store_memory.return_value = {
            "memory_id": memory_id,
            "status": "stored",
            "entities_extracted": 2
        }
        
        server.memory_manager.search_memories.return_value = [
            {
                "memory_id": memory_id,
                "content": "This is test content about Python programming",
                "score": 0.95,
                "metadata": {"source": "test"}
            }
        ]
        
        server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.92,
            "confidence_level": "high"
        }
        
        # Store memory
        store_result = await server._handle_store_memory({
            "content": "This is test content about Python programming",
            "agent_id": "test_agent"
        })
        
        assert store_result.isError is False
        assert memory_id in store_result.content[0].text
        
        # Search for memory
        search_result = await server._handle_search_memory({
            "query": "Python programming",
            "agent_id": "test_agent"
        })
        
        assert search_result.isError is False
        search_data = eval(search_result.content[0].text.replace('null', 'None').replace('true', 'True').replace('false', 'False'))
        assert len(search_data["memories"]) > 0
        assert search_data["memories"][0]["memory_id"] == memory_id

    @pytest.mark.asyncio
    async def test_tool_schema_validation(self):
        """Test that tools properly validate input schemas."""
        server = TyraMemoryServer()
        tools = server._create_tools()
        
        # Test each tool's schema validation would work
        for tool in tools:
            schema = tool.inputSchema
            
            # Basic schema structure validation
            assert isinstance(schema, dict)
            assert "type" in schema
            assert "properties" in schema
            
            # If there are required fields, they should be in properties
            if "required" in schema:
                for field in schema["required"]:
                    assert field in schema["properties"]
            
            # All properties should have types
            for prop_name, prop_schema in schema["properties"].items():
                assert "type" in prop_schema or "enum" in prop_schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])