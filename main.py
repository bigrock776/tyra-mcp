#!/usr/bin/env python3
"""
Main entry point for Tyra's Advanced Memory MCP Server.

This server provides Model Context Protocol (MCP) tools for:
- Advanced memory storage and retrieval
- Hallucination detection and confidence scoring
- Performance analytics and monitoring
- Adaptive learning and optimization
- Multi-agent memory isolation

Usage:
    python main.py

The server communicates via stdio for MCP protocol compatibility.
"""

import asyncio
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.mcp import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Server failed: {e}", file=sys.stderr)
        sys.exit(1)
