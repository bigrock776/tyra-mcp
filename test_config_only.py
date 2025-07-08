#!/usr/bin/env python3
"""
Test just the configuration and simple components.
"""

import os
import sys
from pathlib import Path

# Add only the specific module path
sys.path.append(str(Path(__file__).parent / "src" / "core" / "utils"))


async def test_config():
    """Test configuration loading."""
    print("🧪 Testing configuration...")

    try:
        from simple_config import get_setting, get_settings

        config = get_settings()
        print("✅ Configuration loaded")

        # Test specific settings
        env = get_setting("environment", "development")
        print(f"   Environment: {env}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_logger():
    """Test simple logger."""
    print("\n🧪 Testing logger...")

    try:
        from simple_logger import get_logger

        logger = get_logger("test")
        logger.info("Test message", component="test")
        print("✅ Logger working")

        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_mcp_with_config():
    """Test MCP with our configuration."""
    print("\n🧪 Testing MCP with configuration...")

    try:
        from simple_config import get_setting

        from mcp.server import Server
        from mcp.types import CallToolResult, TextContent, Tool

        # Create server with config-driven name
        server_name = get_setting("server.name", "tyra-memory-server")
        server = Server(server_name)

        # Test tool registration
        @server.list_tools()
        async def handle_list_tools():
            return [
                Tool(
                    name="health_check",
                    description="Check server health",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]

        @server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            if name == "health_check":
                return CallToolResult(
                    content=[TextContent(type="text", text="Server is healthy")]
                )

        print(f"✅ MCP server created with config: {server_name}")
        return True
    except Exception as e:
        print(f"❌ MCP with config test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_tests():
    """Run configuration tests."""
    print("🚀 Starting Configuration Tests\n")

    tests = [test_config, test_logger, test_mcp_with_config]
    passed = 0

    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print(f"\n📊 Results: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🎉 All configuration tests passed!")
        print("\n📝 Next: Test individual MCP tools")
        return True
    else:
        print("⚠️  Some tests failed.")
        return False


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_tests())
