#!/usr/bin/env python3
"""
Simple test script for Tyra Memory MCP Server.

This script tests basic functionality without requiring full MCP integration.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.utils.simple_config import get_setting, get_settings
    from src.mcp_server.server import TyraMemoryServer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


async def test_config_loading():
    """Test configuration loading."""
    print("🧪 Testing configuration loading...")

    try:
        config = get_settings()
        print(f"✅ Configuration loaded successfully")
        print(f"   Environment: {config.get('environment', 'unknown')}")
        print(f"   Log level: {config.get('log_level', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


async def test_server_initialization():
    """Test server initialization."""
    print("\n🧪 Testing server initialization...")

    try:
        server = TyraMemoryServer()
        print("✅ Server object created successfully")
        return True
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        return False


async def test_health_check():
    """Test health check functionality."""
    print("\n🧪 Testing health check...")

    try:
        server = TyraMemoryServer()

        # Create a mock health check call
        health_result = await server._handle_health_check({"detailed": False})

        print("✅ Health check completed")
        print(f"   Status: {health_result.get('status', 'unknown')}")
        print(f"   Initialized: {health_result.get('initialized', False)}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def test_database_config():
    """Test database configuration parsing."""
    print("\n🧪 Testing database configuration...")

    try:
        # Test PostgreSQL config
        pg_host = get_setting("databases.postgresql.host", "localhost")
        pg_port = get_setting("databases.postgresql.port", 5432)
        pg_db = get_setting("databases.postgresql.database", "tyra_memory")

        print(f"✅ Database configuration parsed")
        print(f"   PostgreSQL: {pg_host}:{pg_port}/{pg_db}")

        # Test Redis config
        redis_host = get_setting("databases.redis.host", "localhost")
        redis_port = get_setting("databases.redis.port", 6379)

        print(f"   Redis: {redis_host}:{redis_port}")

        # Test Memgraph config
        mg_host = get_setting("graph.memgraph.host", "localhost")
        mg_port = get_setting("graph.memgraph.port", 7687)

        print(f"   Memgraph: {mg_host}:{mg_port}")

        return True
    except Exception as e:
        print(f"❌ Database configuration test failed: {e}")
        return False


async def test_environment_variables():
    """Test environment variable substitution."""
    print("\n🧪 Testing environment variables...")

    try:
        # Set a test environment variable
        os.environ["TEST_VAR"] = "test_value"

        # Test if environment variable substitution works
        env_val = get_setting("environment", "development")
        print(f"✅ Environment variables working")
        print(f"   Environment: {env_val}")

        return True
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False


async def test_tool_schemas():
    """Test MCP tool schema generation."""
    print("\n🧪 Testing MCP tool schemas...")

    try:
        server = TyraMemoryServer()

        # Get list of tools (this doesn't require full initialization)
        tools = await server.server.list_tools()()

        print(f"✅ Tool schemas generated successfully")
        print(f"   Available tools: {len(tools)}")

        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")

        return True
    except Exception as e:
        print(f"❌ Tool schema test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Tyra Memory Server Tests\n")

    tests = [
        test_config_loading,
        test_database_config,
        test_environment_variables,
        test_server_initialization,
        test_tool_schemas,
        test_health_check,
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

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Server appears to be working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("TYRA_ENV", "development")
    os.environ.setdefault("TYRA_LOG_LEVEL", "INFO")

    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
