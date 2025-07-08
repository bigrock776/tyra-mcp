"""
Unit tests for the embedding providers.

Tests HuggingFace embedding generation, caching, and fallback mechanisms.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from src.core.interfaces.embeddings import EmbeddingError, EmbeddingProvider
from src.core.providers.embeddings.huggingface import HuggingFaceEmbedder


class TestHuggingFaceEmbedder:
    """Test suite for HuggingFace embedding provider."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "model": "sentence-transformers/all-MiniLM-L12-v2",
            "device": "cpu",
            "batch_size": 2,
            "max_length": 256,
            "normalize": True,
        }

    @pytest.fixture
    def embedder(self, mock_config):
        """Create embedder instance for testing."""
        return HuggingFaceEmbedder(mock_config)

    @pytest.mark.asyncio
    async def test_initialization(self, embedder):
        """Test embedder initialization."""
        # Should not be initialized yet
        assert not embedder.is_initialized

        # Mock the model loading
        with patch.object(embedder, "_load_model", new_callable=AsyncMock):
            await embedder.initialize()
            assert embedder.is_initialized

    @pytest.mark.asyncio
    async def test_single_embedding_generation(self, embedder):
        """Test generating embedding for single text."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Mock tokenizer output
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        # Mock model output
        mock_model_output = Mock()
        mock_model_output.last_hidden_state = np.array(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]
        )
        mock_model.return_value = mock_model_output

        embedder.model = mock_model
        embedder.tokenizer = mock_tokenizer
        embedder.is_initialized = True

        # Test embedding generation
        with patch("numpy.mean", return_value=np.array([0.3, 0.4])):
            with patch("numpy.linalg.norm", return_value=0.5):
                embedding = await embedder.generate_embedding("test text")

                assert isinstance(embedding, list)
                assert len(embedding) == 2  # Mock embedding dimension
                mock_tokenizer.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, embedder):
        """Test generating embeddings for multiple texts."""
        texts = ["text 1", "text 2", "text 3"]

        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Mock tokenizer output for batch
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2], [2, 3], [3, 4]],
            "attention_mask": [[1, 1], [1, 1], [1, 1]],
        }

        # Mock model output for batch
        mock_model_output = Mock()
        mock_model_output.last_hidden_state = np.array(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
                [[0.9, 1.0], [1.1, 1.2]],
            ]
        )
        mock_model.return_value = mock_model_output

        embedder.model = mock_model
        embedder.tokenizer = mock_tokenizer
        embedder.is_initialized = True

        # Mock pooling and normalization
        with patch(
            "numpy.mean",
            side_effect=[
                np.array([0.2, 0.3]),  # First text
                np.array([0.6, 0.7]),  # Second text
                np.array([1.0, 1.1]),  # Third text
            ],
        ):
            with patch("numpy.linalg.norm", side_effect=[0.36, 0.92, 1.48]):
                embeddings = await embedder.generate_embeddings(texts)

                assert len(embeddings) == 3
                assert all(isinstance(emb, list) for emb in embeddings)
                mock_tokenizer.assert_called_once_with(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt",
                )

    @pytest.mark.asyncio
    async def test_batch_processing_with_large_input(self, embedder):
        """Test that large inputs are properly batched."""
        # Create texts that exceed batch size
        texts = [f"text {i}" for i in range(5)]  # Batch size is 2

        embedder.is_initialized = True

        # Mock the _embed_batch method to track calls
        async def mock_embed_batch(batch_texts):
            return [[0.1, 0.2] for _ in batch_texts]

        embedder._embed_batch = AsyncMock(side_effect=mock_embed_batch)

        embeddings = await embedder.generate_embeddings(texts)

        # Should have made 3 calls: [2, 2, 1] items
        assert embedder._embed_batch.call_count == 3
        assert len(embeddings) == 5

    @pytest.mark.asyncio
    async def test_embedding_dimensions(self, embedder):
        """Test that embedding dimensions are correct."""
        embedder.is_initialized = True
        embedder.dimensions = 384  # Typical for all-MiniLM-L12-v2

        # Mock embedding generation
        mock_embedding = [0.1] * 384
        embedder._embed_batch = AsyncMock(return_value=[mock_embedding])

        embedding = await embedder.generate_embedding("test")

        assert len(embedding) == 384
        assert embedder.get_dimensions() == 384

    @pytest.mark.asyncio
    async def test_error_handling(self, embedder):
        """Test error handling in embedding generation."""
        embedder.is_initialized = True

        # Mock an error in embedding generation
        embedder._embed_batch = AsyncMock(side_effect=Exception("Model error"))

        with pytest.raises(EmbeddingError):
            await embedder.generate_embedding("test")

    @pytest.mark.asyncio
    async def test_empty_input_handling(self, embedder):
        """Test handling of empty inputs."""
        embedder.is_initialized = True

        # Test empty string
        with pytest.raises(EmbeddingError):
            await embedder.generate_embedding("")

        # Test empty list
        embeddings = await embedder.generate_embeddings([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_text_preprocessing(self, embedder):
        """Test text preprocessing and cleaning."""
        embedder.is_initialized = True

        # Mock embedding generation to check input
        async def mock_embed_batch(texts):
            # Check that text was cleaned
            assert texts[0].strip() == "cleaned text"
            return [[0.1, 0.2]]

        embedder._embed_batch = AsyncMock(side_effect=mock_embed_batch)

        # Test with whitespace and special characters
        await embedder.generate_embedding("  cleaned text  \n")

    def test_device_selection(self, mock_config):
        """Test device selection logic."""
        # Test CPU device
        mock_config["device"] = "cpu"
        embedder = HuggingFaceEmbedder(mock_config)
        assert embedder.device == "cpu"

        # Test auto device selection
        mock_config["device"] = "auto"
        with patch("torch.cuda.is_available", return_value=False):
            embedder = HuggingFaceEmbedder(mock_config)
            assert embedder.device == "cpu"

    @pytest.mark.asyncio
    async def test_cache_integration(self, embedder):
        """Test integration with caching system."""
        embedder.is_initialized = True

        # Mock cache
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        embedder.cache = mock_cache

        # Mock embedding generation
        embedder._embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

        await embedder.generate_embedding("test")

        # Should check cache and store result
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, embedder):
        """Test fallback to secondary model."""
        embedder.is_initialized = True

        # Mock primary model failure
        embedder._embed_batch = AsyncMock(side_effect=Exception("Primary model failed"))

        # Mock fallback embedder
        fallback_embedder = Mock()
        fallback_embedder.generate_embedding = AsyncMock(return_value=[0.1, 0.2])

        embedder.fallback_embedder = fallback_embedder

        embedding = await embedder.generate_embedding("test")

        assert embedding == [0.1, 0.2]
        fallback_embedder.generate_embedding.assert_called_once_with("test")

    @pytest.mark.asyncio
    async def test_model_loading_error(self, embedder):
        """Test handling of model loading errors."""
        with patch(
            "transformers.AutoModel.from_pretrained",
            side_effect=Exception("Model not found"),
        ):
            with pytest.raises(EmbeddingError):
                await embedder.initialize()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, embedder):
        """Test handling of concurrent embedding requests."""
        embedder.is_initialized = True

        # Mock embedding generation with delay
        async def mock_embed_batch(texts):
            await asyncio.sleep(0.1)  # Simulate processing time
            return [[0.1, 0.2] for _ in texts]

        embedder._embed_batch = AsyncMock(side_effect=mock_embed_batch)

        # Generate embeddings concurrently
        tasks = [embedder.generate_embedding(f"text {i}") for i in range(5)]

        embeddings = await asyncio.gather(*tasks)

        assert len(embeddings) == 5
        assert all(len(emb) == 2 for emb in embeddings)

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            "model": "sentence-transformers/all-MiniLM-L12-v2",
            "device": "cpu",
            "batch_size": 16,
            "max_length": 512,
        }

        embedder = HuggingFaceEmbedder(valid_config)
        assert embedder.config == valid_config

        # Invalid configuration should raise error
        invalid_config = {
            "model": "",  # Empty model name
            "batch_size": 0,  # Invalid batch size
        }

        with pytest.raises(ValueError):
            HuggingFaceEmbedder(invalid_config)

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, embedder):
        """Test memory usage monitoring during embedding generation."""
        embedder.is_initialized = True

        # Mock memory monitoring
        with patch("psutil.Process") as mock_process:
            mock_process.return_value.memory_info.return_value.rss = (
                1024 * 1024 * 100
            )  # 100MB

            embedder._embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

            await embedder.generate_embedding("test")

            # Should monitor memory usage
            mock_process.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup(self, embedder):
        """Test proper cleanup of resources."""
        embedder.is_initialized = True
        embedder.model = Mock()
        embedder.tokenizer = Mock()

        await embedder.close()

        assert not embedder.is_initialized
        assert embedder.model is None
        assert embedder.tokenizer is None
