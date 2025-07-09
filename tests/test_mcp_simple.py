#!/usr/bin/env python3
"""
Simple test for MCP functionality without any project imports.
"""


async def test_mcp():
    """Test basic MCP functionality."""

    # Test basic import
    print("🧪 Testing MCP import...")
    from mcp.server import Server
    from mcp.types import Tool

    print("✅ MCP imports successful")

    # Test server creation
    print("🧪 Testing server creation...")
    server = Server("test-server")
    print("✅ Server created successfully")

    # Test tool creation
    print("🧪 Testing tool creation...")
    tool = Tool(
        name="test_tool",
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )
    print("✅ Tool created successfully")

    print("\n🎉 All basic MCP functionality works!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_mcp())
