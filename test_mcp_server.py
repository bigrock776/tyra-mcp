#!/usr/bin/env python3
"""
Test MCP server functionality directly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_mcp_imports():
    """Test MCP package imports."""
    print("🧪 Testing MCP imports...")

    try:
        from mcp.server import Server
        from mcp.server.models import InitializationOptions
        from mcp.server.stdio import stdio_server
        from mcp.types import CallToolRequest, CallToolResult, TextContent, Tool

        print("✅ All MCP imports successful")
        return True
    except ImportError as e:
        print(f"❌ MCP import failed: {e}")
        return False


async def test_basic_server_creation():
    """Test basic MCP server creation."""
    print("\n🧪 Testing basic MCP server creation...")

    try:
        from mcp.server import Server

        # Create a basic server
        server = Server("test-server")
        print("✅ Basic MCP server created successfully")
        return True
    except Exception as e:
        print(f"❌ Server creation failed: {e}")
        return False


async def test_tool_definition():
    """Test MCP tool definition."""
    print("\n🧪 Testing MCP tool definition...")

    try:
        from mcp.types import Tool

        # Create a simple tool definition
        tool = Tool(
            name="test_tool",
            description="A test tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Test message"}
                },
                "required": ["message"],
            },
        )

        print("✅ Tool definition created successfully")
        print(f"   Tool name: {tool.name}")
        return True
    except Exception as e:
        print(f"❌ Tool definition failed: {e}")
        return False


async def test_server_with_tools():
    """Test MCP server with tool registration."""
    print("\n🧪 Testing server with tool registration...")

    try:
        from mcp.server import Server
        from mcp.types import CallToolResult, TextContent, Tool

        # Create server
        server = Server("test-server-with-tools")

        # Register a list_tools handler
        @server.list_tools()
        async def handle_list_tools():
            return [
                Tool(
                    name="echo",
                    description="Echo back a message",
                    inputSchema={
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                )
            ]

        # Register a call_tool handler
        @server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            if name == "echo":
                message = arguments.get("message", "No message")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Echo: {message}")]
                )
            else:
                raise ValueError(f"Unknown tool: {name}")

        print("✅ Server with tools created successfully")
        return True
    except Exception as e:
        print(f"❌ Server with tools failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_mcp_tests():
    """Run all MCP tests."""
    print("🚀 Starting MCP Server Tests\n")

    tests = [
        test_mcp_imports,
        test_basic_server_creation,
        test_tool_definition,
        test_server_with_tools,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print(f"\n📊 MCP Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All MCP tests passed! Server functionality is working.")
        return True
    else:
        print("⚠️  Some MCP tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_mcp_tests())
    sys.exit(0 if success else 1)
