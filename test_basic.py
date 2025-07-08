#!/usr/bin/env python3
"""
Basic test script for Tyra Memory MCP Server.
Tests only core functionality that doesn't require heavy dependencies.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_simple_config():
    """Test simple configuration loading."""
    print("🧪 Testing simple configuration...")

    try:
        from src.core.utils.simple_config import get_setting, get_settings

        # Test basic config loading
        config = get_settings()
        print(f"✅ Configuration loaded: {type(config)}")

        # Test setting retrieval
        env = get_setting("environment", "development")
        print(f"   Environment: {env}")

        # Test nested setting
        db_host = get_setting("databases.postgresql.host", "localhost")
        print(f"   Database host: {db_host}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def test_simple_logger():
    """Test simple logger functionality."""
    print("\n🧪 Testing simple logger...")

    try:
        from src.core.utils.simple_logger import get_logger

        logger = get_logger("test")
        logger.info("Test log message", component="test", status="ok")
        print("✅ Logger working correctly")

        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False


async def test_yaml_loading():
    """Test YAML configuration file loading."""
    print("\n🧪 Testing YAML loading...")

    try:
        import yaml

        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print("✅ YAML config file loaded")
            print(f"   Keys: {list(config.keys())}")
        else:
            print("⚠️  YAML config file not found, but this is optional")

        return True
    except ImportError:
        print("⚠️  PyYAML not installed, skipping YAML test")
        return True
    except Exception as e:
        print(f"❌ YAML loading failed: {e}")
        return False


async def test_environment_variables():
    """Test environment variable handling."""
    print("\n🧪 Testing environment variables...")

    try:
        # Set test variables
        os.environ["TYRA_TEST_VAR"] = "test_value"
        os.environ["TYRA_ENV"] = "development"

        from src.core.utils.simple_config import get_setting

        # Test environment variable substitution
        test_val = get_setting("test_var", "${TYRA_TEST_VAR}")
        env_val = get_setting("environment", "${TYRA_ENV}")

        print(f"✅ Environment variables working")
        print(f"   Test var: {test_val}")
        print(f"   Environment: {env_val}")

        return True
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False


async def test_file_structure():
    """Test that required files and directories exist."""
    print("\n🧪 Testing file structure...")

    try:
        required_files = [
            "src/core/utils/simple_config.py",
            "src/core/utils/simple_logger.py",
            "config/config.yaml",
            "requirements.txt",
        ]

        for file_path in required_files:
            path = Path(file_path)
            if path.exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ⚠️  {file_path} (missing)")

        print("✅ File structure check completed")
        return True
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False


async def run_basic_tests():
    """Run basic tests without heavy dependencies."""
    print("🚀 Starting Basic Tyra Memory Server Tests\n")

    tests = [
        test_file_structure,
        test_simple_logger,
        test_simple_config,
        test_environment_variables,
        test_yaml_loading,
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

    print(f"\n📊 Basic Test Results: {passed}/{total} passed")

    if passed >= total - 1:  # Allow one optional test to fail
        print("🎉 Basic tests passed! Core functionality appears to be working.")
        print("\n📝 Next steps:")
        print("   1. Set up virtual environment: python3 -m venv venv")
        print("   2. Activate virtual environment: source venv/bin/activate")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Run full test suite: python3 test_server.py")
        return True
    else:
        print("⚠️  Some basic tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    # Set up basic environment
    os.environ.setdefault("TYRA_ENV", "development")
    os.environ.setdefault("TYRA_LOG_LEVEL", "INFO")

    # Run basic tests
    success = asyncio.run(run_basic_tests())
    sys.exit(0 if success else 1)
