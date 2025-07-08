"""
Unit tests for the hallucination detector.

Tests grounding score calculation, confidence levels, and safety thresholds.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.interfaces.embeddings import EmbeddingProvider
from src.core.rag.hallucination_detector import HallucinationDetector


class TestHallucinationDetector:
    """Test suite for HallucinationDetector."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "threshold": 0.75,
            "confidence_levels": {"rock_solid": 95, "high": 80, "fuzzy": 60, "low": 0},
            "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
            "similarity_threshold": 0.7,
            "grounding_weight": 0.6,
            "consistency_weight": 0.4,
        }

    @pytest.fixture
    def mock_embedder(self):
        """Mock embeddings provider for testing."""
        embedder = Mock(spec=EmbeddingProvider)
        embedder.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.generate_embeddings = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        return embedder

    @pytest.fixture
    def detector(self, mock_config, mock_embedder):
        """Create detector instance for testing."""
        detector = HallucinationDetector(mock_config)
        detector.embedder = mock_embedder
        detector.is_initialized = True
        return detector

    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test detector initialization."""
        detector = HallucinationDetector(mock_config)

        with patch.object(detector, "_load_embedder", new_callable=AsyncMock):
            await detector.initialize()
            assert detector.is_initialized

    @pytest.mark.asyncio
    async def test_analyze_response_high_grounding(self, detector):
        """Test response analysis with high grounding score."""
        response = "Paris is the capital of France"
        query = "What is the capital of France?"
        retrieved_chunks = [
            {"text": "Paris is the capital city of France", "score": 0.95},
            {"text": "France's capital is Paris", "score": 0.90},
        ]

        # Mock high similarity scores
        with patch.object(detector, "_calculate_similarity_scores") as mock_similarity:
            mock_similarity.return_value = [0.95, 0.88]

            result = await detector.analyze_response(response, query, retrieved_chunks)

            assert result["is_hallucination"] is False
            assert result["overall_confidence"] > 80  # Should be high confidence
            assert result["grounding_score"] > 0.8
            assert result["confidence_level"] in ["high", "rock_solid"]

    @pytest.mark.asyncio
    async def test_analyze_response_low_grounding(self, detector):
        """Test response analysis with low grounding score."""
        response = "The moon is made of cheese"
        query = "What is the moon made of?"
        retrieved_chunks = [
            {"text": "The moon is composed of rock and dust", "score": 0.7},
            {"text": "Lunar surface contains minerals and regolith", "score": 0.65},
        ]

        # Mock low similarity scores
        with patch.object(detector, "_calculate_similarity_scores") as mock_similarity:
            mock_similarity.return_value = [0.1, 0.05]

            result = await detector.analyze_response(response, query, retrieved_chunks)

            assert result["is_hallucination"] is True
            assert result["overall_confidence"] < 60  # Should be low confidence
            assert result["grounding_score"] < 0.4
            assert result["confidence_level"] in ["low", "fuzzy"]

    @pytest.mark.asyncio
    async def test_grounding_score_calculation(self, detector):
        """Test grounding score calculation logic."""
        response = "Einstein developed the theory of relativity"
        retrieved_chunks = [
            {
                "text": "Albert Einstein formulated the theory of relativity",
                "score": 0.9,
            },
            {
                "text": "The theory of relativity was Einstein's breakthrough",
                "score": 0.85,
            },
            {"text": "Quantum mechanics is a different physics theory", "score": 0.3},
        ]

        # Mock similarity calculation
        with patch.object(detector, "_calculate_similarity_scores") as mock_similarity:
            mock_similarity.return_value = [0.92, 0.88, 0.15]

            grounding_score = await detector._calculate_grounding_score(
                response, retrieved_chunks
            )

            # Should weight high-similarity chunks more heavily
            assert grounding_score > 0.7
            assert grounding_score < 1.0

    @pytest.mark.asyncio
    async def test_consistency_check(self, detector):
        """Test internal consistency checking."""
        # Consistent response parts
        response_parts = [
            "Water boils at 100 degrees Celsius",
            "This happens at sea level pressure",
            "The boiling point is 100°C",
        ]

        # Mock high consistency scores
        with patch.object(detector, "_calculate_similarity_scores") as mock_similarity:
            mock_similarity.return_value = [0.85, 0.82, 0.89]

            consistency_score = await detector._check_internal_consistency(
                response_parts
            )

            assert consistency_score > 0.8

    @pytest.mark.asyncio
    async def test_inconsistent_response(self, detector):
        """Test detection of internally inconsistent responses."""
        # Inconsistent response parts
        response_parts = [
            "Water boils at 100 degrees Celsius",
            "Water freezes at 100 degrees",
            "Ice melts at 0 degrees Celsius",
        ]

        # Mock mixed consistency scores
        with patch.object(detector, "_calculate_similarity_scores") as mock_similarity:
            mock_similarity.return_value = [0.2, 0.1, 0.15]

            consistency_score = await detector._check_internal_consistency(
                response_parts
            )

            assert consistency_score < 0.5

    @pytest.mark.asyncio
    async def test_confidence_level_mapping(self, detector):
        """Test confidence level mapping."""
        # Test different confidence scores
        test_cases = [(98, "rock_solid"), (85, "high"), (65, "fuzzy"), (30, "low")]

        for confidence, expected_level in test_cases:
            level = detector._map_confidence_level(confidence)
            assert level == expected_level

    @pytest.mark.asyncio
    async def test_empty_retrieved_chunks(self, detector):
        """Test handling of empty retrieved chunks."""
        response = "Some response text"
        query = "Some query"
        retrieved_chunks = []

        result = await detector.analyze_response(response, query, retrieved_chunks)

        # Should indicate high risk of hallucination
        assert result["is_hallucination"] is True
        assert result["grounding_score"] == 0.0
        assert result["confidence_level"] == "low"

    @pytest.mark.asyncio
    async def test_similarity_calculation_methods(self, detector):
        """Test different similarity calculation methods."""
        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on the mat"

        # Test cosine similarity
        with patch("numpy.dot", return_value=0.8):
            with patch("numpy.linalg.norm", side_effect=[1.0, 1.0]):
                similarity = await detector._calculate_cosine_similarity(
                    [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]
                )
                assert 0 <= similarity <= 1

    @pytest.mark.asyncio
    async def test_factual_contradiction_detection(self, detector):
        """Test detection of factual contradictions."""
        response = "The Earth is flat and has no curvature"
        retrieved_chunks = [
            {"text": "The Earth is spherical with measurable curvature", "score": 0.9},
            {"text": "Satellite images show Earth's spherical shape", "score": 0.85},
        ]

        # Mock contradiction detection
        with patch.object(detector, "_detect_contradictions") as mock_contradictions:
            mock_contradictions.return_value = True

            result = await detector.analyze_response(
                response, "Earth shape", retrieved_chunks
            )

            assert result["contains_contradictions"] is True
            assert result["is_hallucination"] is True

    @pytest.mark.asyncio
    async def test_confidence_score_weighting(self, detector):
        """Test proper weighting of grounding vs consistency scores."""
        response = "Test response"
        retrieved_chunks = [{"text": "relevant chunk", "score": 0.8}]

        # Mock individual scores
        with patch.object(detector, "_calculate_grounding_score") as mock_grounding:
            with patch.object(
                detector, "_check_internal_consistency"
            ) as mock_consistency:
                mock_grounding.return_value = 0.8
                mock_consistency.return_value = 0.9

                result = await detector.analyze_response(
                    response, "query", retrieved_chunks
                )

                # Should weight grounding more heavily (0.6 vs 0.4)
                expected_score = (0.8 * 0.6 + 0.9 * 0.4) * 100
                assert abs(result["overall_confidence"] - expected_score) < 5

    @pytest.mark.asyncio
    async def test_query_relevance_factor(self, detector):
        """Test query relevance as a factor in analysis."""
        response = "Paris is beautiful"
        query = "What is the weather in Tokyo?"
        retrieved_chunks = [
            {"text": "Paris has many tourist attractions", "score": 0.8}
        ]

        # Mock query-response similarity
        with patch.object(
            detector, "_calculate_query_response_similarity"
        ) as mock_qr_sim:
            mock_qr_sim.return_value = 0.1  # Low relevance

            result = await detector.analyze_response(response, query, retrieved_chunks)

            # Low query relevance should increase hallucination probability
            assert result["query_relevance"] < 0.3
            assert result["is_hallucination"] is True

    @pytest.mark.asyncio
    async def test_batch_analysis(self, detector):
        """Test batch analysis of multiple responses."""
        responses = ["Response 1", "Response 2", "Response 3"]
        query = "Test query"
        retrieved_chunks = [{"text": "relevant chunk", "score": 0.8}]

        results = await detector.analyze_responses_batch(
            responses, query, retrieved_chunks
        )

        assert len(results) == 3
        assert all("is_hallucination" in result for result in results)

    @pytest.mark.asyncio
    async def test_threshold_customization(self, detector):
        """Test customizable hallucination threshold."""
        response = "Borderline response"
        retrieved_chunks = [{"text": "somewhat relevant", "score": 0.6}]

        # Mock moderate confidence score
        with patch.object(detector, "_calculate_grounding_score") as mock_grounding:
            mock_grounding.return_value = 0.7

            # Test with default threshold (0.75)
            result1 = await detector.analyze_response(
                response, "query", retrieved_chunks
            )

            # Test with lower threshold (0.6)
            detector.threshold = 0.6
            result2 = await detector.analyze_response(
                response, "query", retrieved_chunks
            )

            # Results should differ based on threshold
            assert result1["is_hallucination"] != result2["is_hallucination"]

    @pytest.mark.asyncio
    async def test_performance_metrics(self, detector):
        """Test performance metrics collection."""
        detector.collect_metrics = True

        response = "Test response"
        retrieved_chunks = [{"text": "chunk", "score": 0.8}]

        result = await detector.analyze_response(response, "query", retrieved_chunks)

        # Should include performance metrics
        assert "analysis_time_ms" in result
        assert "embedding_time_ms" in result
        assert result["analysis_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, detector):
        """Test error handling in analysis."""
        # Mock embedding failure
        detector.embedder.generate_embedding.side_effect = Exception("Embedding failed")

        response = "Test response"
        retrieved_chunks = [{"text": "chunk", "score": 0.8}]

        result = await detector.analyze_response(response, "query", retrieved_chunks)

        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result

    def test_configuration_validation(self, mock_config):
        """Test configuration validation."""
        # Valid configuration
        detector = HallucinationDetector(mock_config)
        assert detector.threshold == 0.75

        # Invalid threshold
        invalid_config = mock_config.copy()
        invalid_config["threshold"] = 1.5  # Invalid threshold > 1

        with pytest.raises(ValueError):
            HallucinationDetector(invalid_config)

    @pytest.mark.asyncio
    async def test_detailed_analysis_output(self, detector):
        """Test detailed analysis output format."""
        response = "Detailed test response"
        retrieved_chunks = [
            {"text": "chunk 1", "score": 0.9},
            {"text": "chunk 2", "score": 0.7},
        ]

        result = await detector.analyze_response(
            response, "query", retrieved_chunks, detailed=True
        )

        # Should include detailed breakdown
        expected_fields = [
            "is_hallucination",
            "overall_confidence",
            "confidence_level",
            "grounding_score",
            "consistency_score",
            "query_relevance",
            "chunk_similarities",
            "analysis_breakdown",
        ]

        for field in expected_fields:
            assert field in result

    @pytest.mark.asyncio
    async def test_language_detection_and_handling(self, detector):
        """Test handling of different languages."""
        # Test non-English text
        response_spanish = "París es la capital de Francia"
        retrieved_chunks = [{"text": "Paris is the capital of France", "score": 0.8}]

        result = await detector.analyze_response(
            response_spanish, "¿Cuál es la capital de Francia?", retrieved_chunks
        )

        # Should still work with cross-language similarity
        assert "is_hallucination" in result
        assert "overall_confidence" in result
