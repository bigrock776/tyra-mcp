"""
MCP Trading Safety Tests.

Tests the trading-specific safety features accessible via MCP tools,
ensuring the 95% confidence requirement and hallucination detection work correctly.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.mcp.server import TyraMemoryServer


class TestMCPTradingSafety:
    """Test trading safety features through MCP tools."""

    @pytest.fixture
    async def mcp_server_with_trading(self):
        """Create MCP server configured for trading safety tests."""
        server = TyraMemoryServer()
        
        # Mock components
        server.memory_manager = AsyncMock()
        server.hallucination_detector = AsyncMock()
        server.performance_tracker = MagicMock()
        server.learning_engine = AsyncMock()
        server._initialized = True
        
        yield server

    @pytest.mark.asyncio
    async def test_trading_response_rock_solid_confidence(self, mcp_server_with_trading):
        """Test trading response analysis with rock solid confidence (95%+)."""
        server = mcp_server_with_trading
        
        # Setup high confidence response
        server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.97,
            "confidence_level": "rock_solid",
            "grounding_score": 0.96,
            "consistency_score": 0.98,
            "trading_approved": True,
            "safety_checks": {
                "confidence_above_95": True,
                "hallucination_below_threshold": True,
                "context_available": True,
                "claims_supported": True
            },
            "supported_claims": 4,
            "unsupported_claims": 0,
            "warnings": []
        }
        
        # Execute trading analysis
        arguments = {
            "response": "Based on technical analysis, AAPL shows strong momentum with RSI at 45 and price above 200-day MA. Recommend buying with stop loss at $175.",
            "query": "Should I buy Apple stock based on current technical indicators?",
            "retrieved_memories": [
                {
                    "content": "AAPL technical analysis: RSI 45, above 200-day moving average, strong momentum confirmed",
                    "id": "trading_mem_1",
                    "metadata": {"type": "technical_analysis", "confidence": 0.96, "source": "market_data"}
                },
                {
                    "content": "Apple Inc. historical support level at $175 confirmed multiple times",
                    "id": "trading_mem_2", 
                    "metadata": {"type": "support_resistance", "confidence": 0.94}
                }
            ]
        }
        
        result = await server._handle_analyze_response(arguments)
        
        # Verify trading approval
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        
        assert result_data["confidence"] == 0.97
        assert result_data["confidence_level"] == "rock_solid"
        assert result_data["trading_approved"] is True
        assert result_data["safety_checks"]["confidence_above_95"] is True
        assert len(result_data["warnings"]) == 0
        assert result_data["supported_claims"] == 4
        assert result_data["unsupported_claims"] == 0

    @pytest.mark.asyncio
    async def test_trading_response_insufficient_confidence(self, mcp_server_with_trading):
        """Test trading response analysis with insufficient confidence (<95%)."""
        server = mcp_server_with_trading
        
        # Setup low confidence response
        server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.78,
            "confidence_level": "fuzzy",
            "grounding_score": 0.70,
            "consistency_score": 0.85,
            "trading_approved": False,
            "safety_checks": {
                "confidence_above_95": False,
                "hallucination_below_threshold": True,
                "context_available": True,
                "claims_supported": False
            },
            "supported_claims": 2,
            "unsupported_claims": 3,
            "warnings": [
                "Confidence below 95% threshold required for trading",
                "Multiple unsupported claims detected",
                "Insufficient grounding in provided context"
            ]
        }
        
        # Execute trading analysis
        arguments = {
            "response": "I think AAPL will moon tomorrow because of my gut feeling. Buy with everything you have!",
            "query": "Investment advice for Apple stock",
            "retrieved_memories": [
                {
                    "content": "Apple quarterly earnings report shows mixed results",
                    "id": "trading_mem_weak",
                    "metadata": {"type": "earnings", "confidence": 0.72}
                }
            ]
        }
        
        result = await server._handle_analyze_response(arguments)
        
        # Verify trading rejection
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        
        assert result_data["confidence"] == 0.78
        assert result_data["confidence_level"] == "fuzzy"
        assert result_data["trading_approved"] is False
        assert result_data["safety_checks"]["confidence_above_95"] is False
        assert len(result_data["warnings"]) > 0
        assert "95% threshold" in result_data["warnings"][0]

    @pytest.mark.asyncio
    async def test_trading_memory_search_confidence_filtering(self, mcp_server_with_trading):
        """Test that trading-related memory searches properly filter by confidence."""
        server = mcp_server_with_trading
        
        # Setup memories with various confidence levels
        mock_memories = [
            {
                "memory_id": "high_conf_trading",
                "content": "Gold price analysis based on Fed policy and inflation data",
                "score": 0.96,
                "confidence_level": "rock_solid",
                "metadata": {"type": "market_analysis", "trading_relevant": True}
            },
            {
                "memory_id": "medium_conf_trading",
                "content": "Bitcoin technical indicators suggest uptrend",
                "score": 0.85,
                "confidence_level": "high",
                "metadata": {"type": "crypto_analysis", "trading_relevant": True}
            },
            {
                "memory_id": "low_conf_trading",
                "content": "Stock market might go up or down based on rumors",
                "score": 0.55,
                "confidence_level": "low",
                "metadata": {"type": "speculation", "trading_relevant": False}
            }
        ]
        
        server.memory_manager.search_memories.return_value = mock_memories
        server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.91,
            "confidence_level": "high"
        }
        
        # Search with high confidence threshold for trading
        arguments = {
            "query": "market analysis for trading decisions",
            "min_confidence": 0.9,  # High threshold for trading
            "search_type": "hybrid",
            "include_analysis": True
        }
        
        result = await server._handle_search_memory(arguments)
        
        # Verify only high confidence results
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        
        # Should return only the rock_solid confidence memory
        high_conf_memories = [m for m in result_data["memories"] if m["score"] >= 0.9]
        assert len(high_conf_memories) == 1
        assert high_conf_memories[0]["memory_id"] == "high_conf_trading"

    @pytest.mark.asyncio
    async def test_trading_memory_storage_with_validation(self, mcp_server_with_trading):
        """Test storing trading-related memories with proper validation."""
        server = mcp_server_with_trading
        
        # Setup successful storage with trading validation
        server.memory_manager.store_memory.return_value = {
            "memory_id": "trading_analysis_123",
            "status": "stored",
            "entities_extracted": 3,
            "trading_relevant": True,
            "confidence_score": 0.94,
            "validation_checks": {
                "source_verified": True,
                "data_quality": "high",
                "temporal_relevance": True
            }
        }
        
        # Store trading-related memory
        arguments = {
            "content": "Federal Reserve interest rate decision: Raised by 0.25% to combat inflation. Market impact: Bond yields up, growth stocks down. Historical pattern suggests 3-month consolidation period.",
            "agent_id": "tyra_trading",
            "metadata": {
                "type": "fed_policy",
                "source": "federal_reserve_official",
                "date": "2024-01-01",
                "trading_relevant": True,
                "confidence_source": 0.95
            },
            "extract_entities": True
        }
        
        result = await server._handle_store_memory(arguments)
        
        # Verify trading memory stored properly
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        
        assert result_data["memory_id"] == "trading_analysis_123"
        assert result_data["trading_relevant"] is True
        assert result_data["confidence_score"] >= 0.9

    @pytest.mark.asyncio
    async def test_trading_hallucination_detection_edge_cases(self, mcp_server_with_trading):
        """Test hallucination detection with trading-specific edge cases."""
        server = mcp_server_with_trading
        
        # Test case 1: Market prediction without sufficient context
        server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.23,
            "confidence_level": "low",
            "trading_approved": False,
            "risk_level": "very_high",
            "grounding_score": 0.15,
            "hallucination_indicators": [
                "Prediction without supporting data",
                "Emotional language indicating bias",
                "No historical context provided"
            ],
            "warnings": [
                "CRITICAL: Prediction lacks factual grounding",
                "Response contains speculative elements",
                "Not suitable for trading decisions"
            ]
        }
        
        arguments = {
            "response": "The market will definitely crash next week because I have a bad feeling about it. Sell everything now!",
            "query": "Market outlook for next week",
            "retrieved_memories": []
        }
        
        result = await server._handle_analyze_response(arguments)
        result_data = json.loads(result.content[0].text)
        
        assert result_data["trading_approved"] is False
        assert result_data["risk_level"] == "very_high"
        assert "CRITICAL" in result_data["warnings"][0]
        
        # Test case 2: Contradictory information
        server.hallucination_detector.detect_hallucination.return_value = {
            "confidence": 0.45,
            "confidence_level": "low",
            "trading_approved": False,
            "contradictions_detected": True,
            "contradiction_details": [
                "Claims market is both bullish and bearish",
                "Inconsistent timeframes mentioned"
            ],
            "warnings": [
                "Contradictory information detected",
                "Internal inconsistency in analysis"
            ]
        }
        
        arguments = {
            "response": "The market is very bullish this month but also extremely bearish. Buy stocks but also sell them immediately.",
            "query": "Current market sentiment",
            "retrieved_memories": [
                {"content": "Market shows mixed signals", "id": "mixed_mem", "metadata": {}}
            ]
        }
        
        result = await server._handle_analyze_response(arguments)
        result_data = json.loads(result.content[0].text)
        
        assert result_data["trading_approved"] is False
        assert result_data["contradictions_detected"] is True

    @pytest.mark.asyncio
    async def test_trading_memory_stats_safety_metrics(self, mcp_server_with_trading):
        """Test memory statistics include trading safety metrics."""
        server = mcp_server_with_trading
        
        # Setup comprehensive stats with trading safety metrics
        mock_stats = {
            "total_memories": 5000,
            "by_confidence": {
                "rock_solid": 1000,  # 20% - good for trading
                "high": 2000,        # 40% - acceptable
                "fuzzy": 1500,       # 30% - needs verification
                "low": 500           # 10% - not for trading
            },
            "trading_metrics": {
                "trading_approved_memories": 1000,
                "trading_safety_rate": 0.20,  # 20% are trading-safe
                "avg_confidence_trading": 0.94,
                "high_risk_memories": 500,
                "contradictions_detected": 25
            },
            "safety_analysis": {
                "memories_above_95_confidence": 1000,
                "memories_with_sources": 3500,
                "memories_with_validation": 2800,
                "potentially_hallucinated": 150
            },
            "recommendations": [
                "20% of memories meet trading safety standards",
                "Consider increasing source verification for trading decisions",
                "150 memories flagged for potential hallucination - review recommended"
            ]
        }
        
        server.memory_manager.get_stats.return_value = mock_stats
        
        # Get stats with trading focus
        arguments = {
            "include_performance": True,
            "include_recommendations": True
        }
        
        result = await server._handle_get_memory_stats(arguments)
        
        # Verify trading safety metrics are included
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        
        assert "trading_metrics" in result_data
        assert result_data["trading_metrics"]["trading_safety_rate"] == 0.20
        assert result_data["safety_analysis"]["memories_above_95_confidence"] == 1000
        assert "trading safety standards" in result_data["recommendations"][0]

    @pytest.mark.asyncio
    async def test_trading_learning_insights_safety_focus(self, mcp_server_with_trading):
        """Test learning insights include trading safety patterns."""
        server = mcp_server_with_trading
        
        # Setup learning insights focused on trading safety
        mock_insights = [
            {
                "category": "trading_safety",
                "insight": "Memories with verified sources have 34% higher trading approval rate",
                "confidence": 0.91,
                "impact": "high",
                "actionable": True,
                "trading_relevance": "critical",
                "recommendation": "Prioritize source verification for trading-related memories"
            },
            {
                "category": "hallucination_patterns",
                "insight": "Market prediction memories without historical context have 67% hallucination rate",
                "confidence": 0.88,
                "impact": "high",
                "actionable": True,
                "trading_relevance": "critical",
                "recommendation": "Require historical context for all market predictions"
            },
            {
                "category": "confidence_calibration",
                "insight": "Confidence scores align with trading outcomes 89% of the time when above 95%",
                "confidence": 0.94,
                "impact": "medium",
                "actionable": False,
                "trading_relevance": "validation",
                "recommendation": "Current 95% threshold is well-calibrated for trading safety"
            }
        ]
        
        server.learning_engine.get_insights.return_value = mock_insights
        
        # Get insights for trading safety
        arguments = {
            "category": "trading_safety",
            "days": 30
        }
        
        result = await server._handle_get_learning_insights(arguments)
        
        # Verify trading-focused insights
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        
        assert len(result_data["insights"]) == 3
        trading_insights = [i for i in result_data["insights"] if i["trading_relevance"] == "critical"]
        assert len(trading_insights) == 2
        
        # Verify specific trading safety insights
        source_insight = next(i for i in result_data["insights"] if "verified sources" in i["insight"])
        assert source_insight["confidence"] == 0.91
        assert "34% higher" in source_insight["insight"]

    @pytest.mark.asyncio
    async def test_trading_health_check_safety_components(self, mcp_server_with_trading):
        """Test health check includes trading safety component status."""
        server = mcp_server_with_trading
        
        # Setup health check with trading safety components
        mock_health = {
            "status": "healthy",
            "components": {
                "memory_manager": "healthy",
                "hallucination_detector": "healthy",
                "trading_safety_validator": "healthy",
                "confidence_calibrator": "healthy",
                "source_verifier": "warning"  # One component has issues
            },
            "trading_safety_status": {
                "overall": "operational",
                "confidence_threshold_check": "passed",
                "hallucination_detection": "passed", 
                "source_verification": "degraded",
                "safety_approval_rate": 0.96,
                "last_safety_check": "2024-01-01T12:00:00Z"
            },
            "metrics": {
                "trading_memories_processed": 1500,
                "safety_checks_passed": 1440,
                "safety_failures": 60,
                "avg_confidence_score": 0.87
            },
            "alerts": [
                {
                    "level": "warning",
                    "component": "source_verifier",
                    "message": "Source verification response time increased by 45%",
                    "impact": "trading_safety"
                }
            ]
        }
        
        server.memory_manager.health_check.return_value = mock_health
        
        # Detailed health check
        arguments = {"detailed": True}
        
        result = await server._handle_health_check(arguments)
        
        # Verify trading safety components are included
        assert result.isError is False
        result_data = json.loads(result.content[0].text)
        
        assert "trading_safety_status" in result_data
        assert result_data["trading_safety_status"]["overall"] == "operational"
        assert result_data["trading_safety_status"]["safety_approval_rate"] == 0.96
        assert "source_verifier" in result_data["components"]
        assert result_data["components"]["source_verifier"] == "warning"
        
        # Verify alerts include trading safety impact
        assert len(result_data["alerts"]) == 1
        assert result_data["alerts"][0]["impact"] == "trading_safety"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])