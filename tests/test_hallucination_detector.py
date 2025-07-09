"""
Comprehensive unit tests for Hallucination Detector.

Tests confidence scoring, grounding analysis, and trading safety validation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from src.core.rag.hallucination_detector import HallucinationDetector, HallucinationResult


class TestHallucinationDetector:
    """Test Hallucination Detector functionality."""

    @pytest.fixture
    async def hallucination_detector(self):
        """Create hallucination detector with mocked dependencies."""
        with patch('src.core.rag.hallucination_detector.EmbeddingProvider') as mock_embedder:
            detector = HallucinationDetector()
            detector.embedder = AsyncMock()
            detector._initialized = True
            
            yield detector

    @pytest.mark.asyncio
    async def test_detect_high_confidence_response(self, hallucination_detector):
        """Test detection of high-confidence, well-grounded response."""
        # Setup
        query = "What is the capital of France?"
        response = "The capital of France is Paris, a major European city."
        context = "France is a country in Europe. Paris is the capital and largest city of France."
        
        # Mock embedding similarity (high similarity indicates good grounding)
        hallucination_detector.embedder.generate_embedding.side_effect = [
            [0.1, 0.2, 0.3] * 341,  # response embedding
            [0.1, 0.2, 0.3] * 341   # context embedding (high similarity)
        ]
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify
        assert result.confidence >= 0.9
        assert result.confidence_level == "rock_solid"
        assert result.grounding_score >= 0.9
        assert result.risk_level == "very_low"
        assert result.trading_approved is True
        assert len(result.warnings) == 0

    @pytest.mark.asyncio
    async def test_detect_low_confidence_response(self, hallucination_detector):
        """Test detection of low-confidence, poorly grounded response."""
        # Setup
        query = "What will happen to the stock market tomorrow?"
        response = "The stock market will definitely crash tomorrow because I have a feeling."
        context = "Stock market data shows mixed signals."
        
        # Mock embedding similarity (low similarity indicates poor grounding)
        hallucination_detector.embedder.generate_embedding.side_effect = [
            [0.1, 0.2, 0.3] * 341,  # response embedding
            [0.9, 0.8, 0.7] * 341   # context embedding (low similarity)
        ]
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify
        assert result.confidence <= 0.5
        assert result.confidence_level == "low"
        assert result.grounding_score <= 0.5
        assert result.risk_level == "high"
        assert result.trading_approved is False
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_detect_contradictory_response(self, hallucination_detector):
        """Test detection of internally contradictory response."""
        # Setup
        query = "Is the market bullish or bearish?"
        response = "The market is very bullish this week but also extremely bearish. Buy stocks but sell them immediately."
        context = "Market shows mixed signals with both positive and negative indicators."
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify
        assert result.contradictions_detected is True
        assert result.confidence <= 0.6
        assert result.trading_approved is False
        assert any("contradiction" in warning.lower() for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_confidence_scoring_levels(self, hallucination_detector):
        """Test confidence level classification."""
        test_cases = [
            (0.97, "rock_solid"),
            (0.85, "high"),
            (0.65, "fuzzy"),
            (0.35, "low")
        ]
        
        for confidence_score, expected_level in test_cases:
            # Mock to return specific confidence
            with patch.object(hallucination_detector, '_calculate_confidence_score', return_value=confidence_score):
                result = await hallucination_detector.detect_hallucination(
                    query="test query",
                    response="test response",
                    context="test context"
                )
                
                assert result.confidence == confidence_score
                assert result.confidence_level == expected_level

    @pytest.mark.asyncio
    async def test_trading_safety_validation_rock_solid(self, hallucination_detector):
        """Test trading safety validation for rock solid confidence."""
        # Setup high-confidence trading response
        query = "Should I buy AAPL based on technical analysis?"
        response = "Based on technical indicators, AAPL shows strong momentum with RSI at 45 and price above 200-day MA. Consider buying with stop loss at $175."
        context = "AAPL technical analysis: RSI 45, above 200-day moving average, strong momentum confirmed. Historical support at $175."
        
        # Mock high similarity embeddings
        hallucination_detector.embedder.generate_embedding.side_effect = [
            [0.1, 0.2, 0.3] * 341,  # response embedding
            [0.1, 0.2, 0.3] * 341   # context embedding (high similarity)
        ]
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify trading approval
        assert result.confidence >= 0.95
        assert result.confidence_level == "rock_solid"
        assert result.trading_approved is True
        assert result.safety_checks["confidence_above_95"] is True
        assert result.safety_checks["hallucination_below_threshold"] is True
        assert result.safety_checks["context_available"] is True

    @pytest.mark.asyncio
    async def test_trading_safety_validation_insufficient_confidence(self, hallucination_detector):
        """Test trading safety validation for insufficient confidence."""
        # Setup low-confidence trading response
        query = "Investment advice for crypto"
        response = "Crypto will moon tomorrow! Buy everything with leverage!"
        context = "Cryptocurrency market data shows high volatility."
        
        # Mock low similarity embeddings
        hallucination_detector.embedder.generate_embedding.side_effect = [
            [0.1, 0.2, 0.3] * 341,  # response embedding
            [0.9, 0.8, 0.7] * 341   # context embedding (low similarity)
        ]
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify trading rejection
        assert result.confidence < 0.95
        assert result.trading_approved is False
        assert result.safety_checks["confidence_above_95"] is False
        assert len(result.warnings) > 0
        assert any("95%" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_grounding_score_calculation(self, hallucination_detector):
        """Test grounding score calculation based on context overlap."""
        # Setup
        query = "What is machine learning?"
        response = "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
        context = "Artificial intelligence and machine learning are related fields. Machine learning algorithms learn from data."
        
        # Mock embeddings for semantic similarity
        hallucination_detector.embedder.generate_embedding.side_effect = [
            [0.5, 0.6, 0.7] * 341,  # response embedding
            [0.5, 0.6, 0.7] * 341   # context embedding (high similarity)
        ]
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify grounding score is high due to concept overlap
        assert result.grounding_score >= 0.8
        assert result.supported_claims > result.unsupported_claims

    @pytest.mark.asyncio
    async def test_consistency_analysis(self, hallucination_detector):
        """Test internal consistency analysis of response."""
        # Setup consistent response
        consistent_response = "Apple stock has been rising steadily over the past month. Technical indicators show continued upward momentum."
        
        # Setup inconsistent response
        inconsistent_response = "Apple stock is performing very well and very poorly. It's the best investment and worst investment."
        
        # Test consistent response
        result_consistent = await hallucination_detector.detect_hallucination(
            query="How is Apple stock performing?",
            response=consistent_response,
            context="Apple stock market data shows upward trend."
        )
        
        # Test inconsistent response
        result_inconsistent = await hallucination_detector.detect_hallucination(
            query="How is Apple stock performing?",
            response=inconsistent_response,
            context="Apple stock market data shows mixed signals."
        )
        
        # Verify
        assert result_consistent.consistency_score > result_inconsistent.consistency_score
        assert result_inconsistent.contradictions_detected is True
        assert result_consistent.contradictions_detected is False

    @pytest.mark.asyncio
    async def test_temporal_relevance_check(self, hallucination_detector):
        """Test temporal relevance validation."""
        # Setup query asking for current information
        query = "What is the current stock price of Tesla?"
        
        # Response with outdated information
        response = "Tesla stock is trading at $200 per share as of today."
        
        # Context with recent data
        context = "Tesla stock data from last week shows price around $250."
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify temporal inconsistency is detected
        assert result.confidence < 0.8  # Should be lower due to temporal mismatch
        assert result.temporal_relevance_score < 0.7

    @pytest.mark.asyncio
    async def test_claim_validation(self, hallucination_detector):
        """Test individual claim validation against context."""
        # Setup
        query = "Tell me about Python programming"
        response = "Python is a programming language created by Guido van Rossum. It was first released in 1991. Python is known for its simplicity."
        context = "Python programming language was created by Guido van Rossum and first released in 1991. Python emphasizes code readability."
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify claims are properly validated
        assert result.supported_claims >= 3  # Creator, year, simplicity/readability
        assert result.unsupported_claims == 0
        assert result.claim_support_ratio >= 0.9

    @pytest.mark.asyncio
    async def test_confidence_calibration(self, hallucination_detector):
        """Test confidence score calibration."""
        # Test various scenarios to ensure confidence is well-calibrated
        test_scenarios = [
            {
                "query": "What is 2+2?",
                "response": "2+2 equals 4.",
                "context": "Basic arithmetic: 2+2=4",
                "expected_confidence": "high"  # >= 0.9
            },
            {
                "query": "Predict tomorrow's weather",
                "response": "It will definitely be sunny tomorrow.",
                "context": "Weather forecast shows 60% chance of rain.",
                "expected_confidence": "low"  # <= 0.4
            },
            {
                "query": "What is the capital of Japan?",
                "response": "Tokyo is the capital of Japan.",
                "context": "Japan's capital city is Tokyo.",
                "expected_confidence": "high"  # >= 0.9
            }
        ]
        
        for scenario in test_scenarios:
            # Mock appropriate embeddings based on expected confidence
            if scenario["expected_confidence"] == "high":
                hallucination_detector.embedder.generate_embedding.side_effect = [
                    [0.1, 0.2, 0.3] * 341,  # response
                    [0.1, 0.2, 0.3] * 341   # context (high similarity)
                ]
            else:
                hallucination_detector.embedder.generate_embedding.side_effect = [
                    [0.1, 0.2, 0.3] * 341,  # response
                    [0.9, 0.8, 0.7] * 341   # context (low similarity)
                ]
            
            result = await hallucination_detector.detect_hallucination(
                query=scenario["query"],
                response=scenario["response"],
                context=scenario["context"]
            )
            
            if scenario["expected_confidence"] == "high":
                assert result.confidence >= 0.8, f"Failed for scenario: {scenario['query']}"
            else:
                assert result.confidence <= 0.5, f"Failed for scenario: {scenario['query']}"

    @pytest.mark.asyncio
    async def test_no_context_scenario(self, hallucination_detector):
        """Test handling when no context is provided."""
        # Setup
        query = "What is the meaning of life?"
        response = "The meaning of life is 42, according to Douglas Adams."
        context = ""  # No context provided
        
        # Execute
        result = await hallucination_detector.detect_hallucination(
            query=query,
            response=response,
            context=context
        )
        
        # Verify
        assert result.grounding_score <= 0.5  # Should be low without context
        assert result.safety_checks["context_available"] is False
        assert result.trading_approved is False
        assert any("context" in warning.lower() for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_risk_level_classification(self, hallucination_detector):
        """Test risk level classification based on confidence."""
        risk_test_cases = [
            (0.95, "very_low"),
            (0.85, "low"), 
            (0.65, "medium"),
            (0.45, "high"),
            (0.25, "very_high")
        ]
        
        for confidence, expected_risk in risk_test_cases:
            with patch.object(hallucination_detector, '_calculate_confidence_score', return_value=confidence):
                result = await hallucination_detector.detect_hallucination(
                    query="test",
                    response="test",
                    context="test"
                )
                
                assert result.risk_level == expected_risk

    @pytest.mark.asyncio
    async def test_batch_analysis(self, hallucination_detector):
        """Test batch hallucination analysis."""
        # Setup multiple query-response pairs
        batch_data = [
            {
                "query": "What is Python?",
                "response": "Python is a programming language.",
                "context": "Python programming language information."
            },
            {
                "query": "Stock prediction",
                "response": "Stock will go up 1000% tomorrow!",
                "context": "Market data shows volatility."
            }
        ]
        
        # Mock embeddings
        hallucination_detector.embedder.generate_embedding.side_effect = [
            [0.1, 0.2, 0.3] * 341,  # response 1
            [0.1, 0.2, 0.3] * 341,  # context 1 (high similarity)
            [0.1, 0.2, 0.3] * 341,  # response 2  
            [0.9, 0.8, 0.7] * 341   # context 2 (low similarity)
        ]
        
        # Execute batch analysis
        results = []
        for item in batch_data:
            result = await hallucination_detector.detect_hallucination(
                query=item["query"],
                response=item["response"], 
                context=item["context"]
            )
            results.append(result)
        
        # Verify
        assert len(results) == 2
        assert results[0].confidence > results[1].confidence
        assert results[0].trading_approved != results[1].trading_approved

    @pytest.mark.asyncio
    async def test_error_handling_embedding_failure(self, hallucination_detector):
        """Test error handling when embedding generation fails."""
        # Setup
        hallucination_detector.embedder.generate_embedding.side_effect = Exception("Embedding service down")
        
        # Execute and verify graceful error handling
        result = await hallucination_detector.detect_hallucination(
            query="test",
            response="test",
            context="test"
        )
        
        # Should return low confidence result with error indication
        assert result.confidence <= 0.3
        assert result.trading_approved is False
        assert any("error" in warning.lower() for warning in result.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])