"""
Claude Integration Testing and Utilities.

Provides testing utilities and integration helpers for Claude AI agent.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.clients.memory_client import MemoryClient
from src.core.agents.agent_logger import agent_log_context, get_agent_logger
from src.core.agents.session_manager import AgentSessionManager, create_agent_session
from src.core.memory.manager import MemoryManager

logger = get_agent_logger(__name__)


class ClaudeIntegrationTester:
    """
    Test suite for Claude integration with the memory server.

    Validates MCP tool functionality, memory operations, and agent-specific features.
    """

    def __init__(self):
        self.session_manager: Optional[AgentSessionManager] = None
        self.memory_client: Optional[MemoryClient] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.test_session_id: Optional[str] = None

    async def initialize(self):
        """Initialize test environment."""
        try:
            # Initialize session manager
            from src.core.agents import get_session_manager

            self.session_manager = await get_session_manager()

            # Initialize memory client
            self.memory_client = MemoryClient()
            await self.memory_client.initialize()

            # Initialize memory manager
            self.memory_manager = MemoryManager()
            await self.memory_manager.initialize()

            logger.info("Claude integration tester initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Claude integration tester: {e}")
            raise

    async def create_test_session(self) -> str:
        """Create a test session for Claude."""
        if not self.session_manager:
            raise RuntimeError("Session manager not initialized")

        session = await self.session_manager.create_session(
            agent_id="claude",
            user_id="test_user",
            metadata={
                "test_mode": True,
                "created_by": "integration_test",
                "test_timestamp": datetime.utcnow().isoformat(),
            },
        )

        self.test_session_id = session.session_id
        logger.info(f"Created test session: {self.test_session_id}")
        return self.test_session_id

    async def test_mcp_tools(self) -> Dict[str, Any]:
        """
        Test all MCP tools with Claude-specific scenarios.

        Returns:
            Test results dictionary
        """
        if not self.test_session_id:
            await self.create_test_session()

        results = {
            "save_memory": await self._test_save_memory(),
            "search_memories": await self._test_search_memories(),
            "get_all_memories": await self._test_get_all_memories(),
            "agent_specific_features": await self._test_agent_specific_features(),
        }

        # Calculate overall success rate
        total_tests = sum(
            len(test_results.get("subtests", {})) for test_results in results.values()
        )
        passed_tests = sum(
            sum(
                1
                for result in test_results.get("subtests", {}).values()
                if result.get("passed", False)
            )
            for test_results in results.values()
        )

        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_passed": passed_tests == total_tests,
        }

        return results

    async def _test_save_memory(self) -> Dict[str, Any]:
        """Test save_memory MCP tool."""
        with agent_log_context(
            agent_id="claude",
            session_id=self.test_session_id,
            operation="test_save_memory",
        ):
            subtests = {}

            # Test 1: Basic memory storage
            try:
                result = await self.memory_manager.store_memory(
                    text="Claude is learning about the concept of recursion in programming.",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    metadata={"topic": "programming", "concept": "recursion"},
                )

                subtests["basic_storage"] = {
                    "passed": result.get("success", False),
                    "memory_id": result.get("memory_id"),
                    "details": result,
                }

            except Exception as e:
                subtests["basic_storage"] = {"passed": False, "error": str(e)}

            # Test 2: Memory with entity extraction
            try:
                result = await self.memory_manager.store_memory(
                    text="Einstein developed the theory of relativity in the early 20th century.",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    extract_entities=True,
                    metadata={"topic": "physics", "historical": True},
                )

                subtests["entity_extraction"] = {
                    "passed": result.get("success", False)
                    and result.get("entities_created", 0) > 0,
                    "entities_created": result.get("entities_created", 0),
                    "details": result,
                }

            except Exception as e:
                subtests["entity_extraction"] = {"passed": False, "error": str(e)}

            # Test 3: Large content chunking
            try:
                large_content = (
                    "This is a test of content chunking. " * 100
                )  # Create large content
                result = await self.memory_manager.store_memory(
                    text=large_content,
                    agent_id="claude",
                    session_id=self.test_session_id,
                    chunk_content=True,
                    metadata={"test_type": "chunking"},
                )

                subtests["content_chunking"] = {
                    "passed": result.get("success", False)
                    and len(result.get("chunk_ids", [])) > 1,
                    "chunk_count": len(result.get("chunk_ids", [])),
                    "details": result,
                }

            except Exception as e:
                subtests["content_chunking"] = {"passed": False, "error": str(e)}

            return {
                "tool": "save_memory",
                "subtests": subtests,
                "overall_passed": all(
                    test.get("passed", False) for test in subtests.values()
                ),
            }

    async def _test_search_memories(self) -> Dict[str, Any]:
        """Test search_memories MCP tool."""
        with agent_log_context(
            agent_id="claude",
            session_id=self.test_session_id,
            operation="test_search_memories",
        ):
            subtests = {}

            # Test 1: Basic search
            try:
                result = await self.memory_manager.search_memories(
                    query="programming recursion",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    top_k=5,
                )

                subtests["basic_search"] = {
                    "passed": result.get("success", False),
                    "result_count": len(result.get("results", [])),
                    "details": result,
                }

            except Exception as e:
                subtests["basic_search"] = {"passed": False, "error": str(e)}

            # Test 2: Search with reranking
            try:
                result = await self.memory_manager.search_memories(
                    query="Einstein relativity physics",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    top_k=10,
                    rerank=True,
                    include_analysis=True,
                )

                subtests["reranked_search"] = {
                    "passed": result.get("success", False),
                    "has_analysis": "hallucination_analysis" in result,
                    "result_count": len(result.get("results", [])),
                    "details": result,
                }

            except Exception as e:
                subtests["reranked_search"] = {"passed": False, "error": str(e)}

            # Test 3: Hybrid search
            try:
                result = await self.memory_manager.search_memories(
                    query="theory development",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    search_type="hybrid",
                    min_confidence=0.3,
                )

                subtests["hybrid_search"] = {
                    "passed": result.get("success", False),
                    "search_type": result.get("search_type"),
                    "details": result,
                }

            except Exception as e:
                subtests["hybrid_search"] = {"passed": False, "error": str(e)}

            return {
                "tool": "search_memories",
                "subtests": subtests,
                "overall_passed": all(
                    test.get("passed", False) for test in subtests.values()
                ),
            }

    async def _test_get_all_memories(self) -> Dict[str, Any]:
        """Test get_all_memories MCP tool."""
        with agent_log_context(
            agent_id="claude",
            session_id=self.test_session_id,
            operation="test_get_all_memories",
        ):
            subtests = {}

            # Test 1: Get all memories for Claude
            try:
                result = await self.memory_manager.get_all_memories(
                    agent_id="claude", session_id=self.test_session_id, limit=100
                )

                subtests["get_all_memories"] = {
                    "passed": result.get("success", False),
                    "memory_count": len(result.get("memories", [])),
                    "details": result,
                }

            except Exception as e:
                subtests["get_all_memories"] = {"passed": False, "error": str(e)}

            # Test 2: Paginated retrieval
            try:
                result = await self.memory_manager.get_all_memories(
                    agent_id="claude",
                    session_id=self.test_session_id,
                    limit=5,
                    offset=0,
                )

                subtests["paginated_retrieval"] = {
                    "passed": result.get("success", False),
                    "returned_count": len(result.get("memories", [])),
                    "total_count": result.get("total_count", 0),
                    "details": result,
                }

            except Exception as e:
                subtests["paginated_retrieval"] = {"passed": False, "error": str(e)}

            return {
                "tool": "get_all_memories",
                "subtests": subtests,
                "overall_passed": all(
                    test.get("passed", False) for test in subtests.values()
                ),
            }

    async def _test_agent_specific_features(self) -> Dict[str, Any]:
        """Test Claude-specific features and configurations."""
        with agent_log_context(
            agent_id="claude",
            session_id=self.test_session_id,
            operation="test_agent_features",
        ):
            subtests = {}

            # Test 1: Agent configuration loading
            try:
                config = self.session_manager.get_agent_config("claude")

                subtests["agent_config"] = {
                    "passed": config is not None,
                    "has_confidence_thresholds": "confidence_thresholds" in config,
                    "has_memory_settings": "memory_settings" in config,
                    "details": config,
                }

            except Exception as e:
                subtests["agent_config"] = {"passed": False, "error": str(e)}

            # Test 2: Session management
            try:
                sessions = await self.session_manager.get_agent_sessions("claude")

                subtests["session_management"] = {
                    "passed": len(sessions) > 0,
                    "session_count": len(sessions),
                    "has_test_session": any(
                        s.session_id == self.test_session_id for s in sessions
                    ),
                }

            except Exception as e:
                subtests["session_management"] = {"passed": False, "error": str(e)}

            # Test 3: Confidence threshold enforcement
            try:
                # Store a memory and check if confidence thresholds are applied
                result = await self.memory_manager.store_memory(
                    text="This is a test of confidence thresholds for Claude.",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    metadata={"test_type": "confidence"},
                )

                # Search and check confidence handling
                search_result = await self.memory_manager.search_memories(
                    query="confidence thresholds test",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    include_analysis=True,
                )

                subtests["confidence_handling"] = {
                    "passed": (
                        result.get("success", False)
                        and search_result.get("success", False)
                        and "hallucination_analysis" in search_result
                    ),
                    "store_success": result.get("success", False),
                    "search_success": search_result.get("success", False),
                    "has_analysis": "hallucination_analysis" in search_result,
                }

            except Exception as e:
                subtests["confidence_handling"] = {"passed": False, "error": str(e)}

            return {
                "tool": "agent_specific_features",
                "subtests": subtests,
                "overall_passed": all(
                    test.get("passed", False) for test in subtests.values()
                ),
            }

    async def test_performance(self) -> Dict[str, Any]:
        """
        Test performance characteristics for Claude integration.

        Returns:
            Performance test results
        """
        if not self.test_session_id:
            await self.create_test_session()

        with agent_log_context(
            agent_id="claude",
            session_id=self.test_session_id,
            operation="performance_test",
        ):
            results = {}

            # Test memory storage performance
            storage_times = []
            for i in range(10):
                start_time = asyncio.get_event_loop().time()

                await self.memory_manager.store_memory(
                    text=f"Performance test memory {i} with some additional content to make it realistic.",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    metadata={"test_type": "performance", "iteration": i},
                )

                end_time = asyncio.get_event_loop().time()
                storage_times.append((end_time - start_time) * 1000)  # Convert to ms

            results["storage_performance"] = {
                "avg_time_ms": sum(storage_times) / len(storage_times),
                "min_time_ms": min(storage_times),
                "max_time_ms": max(storage_times),
                "samples": len(storage_times),
            }

            # Test search performance
            search_times = []
            for i in range(10):
                start_time = asyncio.get_event_loop().time()

                await self.memory_manager.search_memories(
                    query=f"performance test {i}",
                    agent_id="claude",
                    session_id=self.test_session_id,
                    top_k=5,
                )

                end_time = asyncio.get_event_loop().time()
                search_times.append((end_time - start_time) * 1000)  # Convert to ms

            results["search_performance"] = {
                "avg_time_ms": sum(search_times) / len(search_times),
                "min_time_ms": min(search_times),
                "max_time_ms": max(search_times),
                "samples": len(search_times),
            }

            return results

    async def cleanup_test_data(self):
        """Clean up test data and sessions."""
        if self.test_session_id and self.session_manager:
            await self.session_manager.end_session(self.test_session_id)
            logger.info("Cleaned up test session")

    async def close(self):
        """Close the integration tester."""
        await self.cleanup_test_data()

        if self.memory_client:
            await self.memory_client.close()

        if self.memory_manager:
            await self.memory_manager.close()


async def run_claude_integration_tests() -> Dict[str, Any]:
    """
    Run complete Claude integration test suite.

    Returns:
        Complete test results
    """
    tester = ClaudeIntegrationTester()

    try:
        await tester.initialize()

        # Run all tests
        mcp_results = await tester.test_mcp_tools()
        performance_results = await tester.test_performance()

        # Combine results
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": "claude",
            "mcp_tools": mcp_results,
            "performance": performance_results,
            "overall_success": mcp_results.get("summary", {}).get(
                "overall_passed", False
            ),
        }

        logger.info(
            f"Claude integration tests completed. Success: {results['overall_success']}"
        )
        return results

    except Exception as e:
        logger.error(f"Claude integration tests failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": "claude",
            "error": str(e),
            "overall_success": False,
        }

    finally:
        await tester.close()


if __name__ == "__main__":
    # Run tests when executed directly
    import asyncio

    async def main():
        results = await run_claude_integration_tests()
        print(json.dumps(results, indent=2))

    asyncio.run(main())
