"""
Comprehensive unit tests for Embedding System.

Tests embedding generation, fallback mechanisms, caching, and performance.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.core.embeddings.embedder import EmbeddingProvider, EmbeddingRequest, EmbeddingResult


class TestEmbeddingProvider:
    """Test Embedding Provider functionality."""

    @pytest.fixture
    async def embedding_provider(self):
        """Create embedding provider with mocked models."""
        with patch('src.core.embeddings.embedder.SentenceTransformer') as mock_st:
            provider = EmbeddingProvider()
            
            # Mock primary model
            provider.primary_model = MagicMock()
            provider.primary_model.encode.return_value = np.array([[0.1, 0.2, 0.3] * 341 + [0.4]])  # 1024 dims
            
            # Mock fallback model  
            provider.fallback_model = MagicMock()
            provider.fallback_model.encode.return_value = np.array([[0.1, 0.2, 0.3] * 128])  # 384 dims
            
            provider._initialized = True
            yield provider

    @pytest.mark.asyncio
    async def test_generate_embedding_primary_model(self, embedding_provider):
        """Test embedding generation with primary model."""
        # Setup
        text = "This is a test sentence for embedding generation."
        request = EmbeddingRequest(text=text, use_primary=True)
        
        # Execute
        result = await embedding_provider.generate_embedding(request)
        
        # Verify
        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 1024  # Primary model dimensions
        assert result.model_used == "intfloat/e5-large-v2"
        assert result.success is True
        assert result.processing_time > 0
        
        # Verify primary model was called
        embedding_provider.primary_model.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embedding_fallback_model(self, embedding_provider):
        """Test embedding generation with fallback model."""
        # Setup
        text = "Test sentence for fallback embedding."
        request = EmbeddingRequest(text=text, use_primary=False)
        
        # Execute
        result = await embedding_provider.generate_embedding(request)
        
        # Verify
        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 384  # Fallback model dimensions
        assert result.model_used == "sentence-transformers/all-MiniLM-L12-v2"
        assert result.success is True
        
        # Verify fallback model was called
        embedding_provider.fallback_model.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_automatic_fallback_on_primary_failure(self, embedding_provider):
        """Test automatic fallback when primary model fails."""
        # Setup
        text = "Test sentence for fallback scenario."
        request = EmbeddingRequest(text=text, use_primary=True)
        
        # Mock primary model failure
        embedding_provider.primary_model.encode.side_effect = Exception("CUDA out of memory")
        
        # Execute
        result = await embedding_provider.generate_embedding(request)
        
        # Verify fallback was used
        assert result.success is True
        assert len(result.embedding) == 384  # Fallback dimensions
        assert result.model_used == "sentence-transformers/all-MiniLM-L12-v2"
        assert result.fallback_used is True
        assert "CUDA out of memory" in result.error_message
        
        # Verify both models were attempted
        embedding_provider.primary_model.encode.assert_called_once()
        embedding_provider.fallback_model.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, embedding_provider):
        """Test batch embedding generation."""
        # Setup
        texts = [
            "First test sentence.",
            "Second test sentence.", 
            "Third test sentence."
        ]
        request = EmbeddingRequest(texts=texts, batch_size=2)
        
        # Mock batch responses
        embedding_provider.primary_model.encode.side_effect = [
            np.array([[0.1] * 1024, [0.2] * 1024]),  # First batch
            np.array([[0.3] * 1024])                 # Second batch
        ]
        
        # Execute
        results = await embedding_provider.generate_batch_embeddings(request)
        
        # Verify
        assert len(results) == 3
        assert all(len(r.embedding) == 1024 for r in results)
        assert all(r.success for r in results)
        
        # Verify batching was used
        assert embedding_provider.primary_model.encode.call_count == 2

    @pytest.mark.asyncio
    async def test_embedding_caching(self, embedding_provider):
        """Test embedding caching functionality."""
        # Setup cache
        embedding_provider.cache = AsyncMock()
        text = "This sentence should be cached."
        request = EmbeddingRequest(text=text)
        
        # Mock cache miss, then hit
        cached_embedding = [0.5] * 1024
        embedding_provider.cache.get.side_effect = [None, cached_embedding]  # Miss, then hit
        
        # First call - cache miss
        result1 = await embedding_provider.generate_embedding(request)
        
        # Second call - cache hit
        result2 = await embedding_provider.generate_embedding(request)
        
        # Verify
        assert result1.success is True
        assert result2.success is True
        assert result2.cached is True
        
        # Verify cache operations
        assert embedding_provider.cache.get.call_count == 2
        embedding_provider.cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_preprocessing(self, embedding_provider):
        """Test text preprocessing before embedding."""
        # Setup text with various issues
        messy_text = "  This  has   extra    spaces\n\nand\tlines\r\n  "
        request = EmbeddingRequest(text=messy_text, preprocess=True)
        
        # Execute
        result = await embedding_provider.generate_embedding(request)
        
        # Verify
        assert result.success is True
        
        # Verify preprocessing was applied (normalized text passed to model)
        call_args = embedding_provider.primary_model.encode.call_args[0]
        processed_text = call_args[0][0] if isinstance(call_args[0], list) else call_args[0]
        assert processed_text.strip() == "This has extra spaces and lines"

    @pytest.mark.asyncio
    async def test_embedding_normalization(self, embedding_provider):
        """Test embedding vector normalization."""
        # Setup
        text = "Test sentence for normalization."
        request = EmbeddingRequest(text=text, normalize=True)
        
        # Mock unnormalized embedding
        unnormalized = np.array([[1.0, 2.0, 3.0] * 341 + [4.0]])  # 1024 dims
        embedding_provider.primary_model.encode.return_value = unnormalized
        
        # Execute
        result = await embedding_provider.generate_embedding(request)
        
        # Verify normalization (L2 norm should be 1.0)
        embedding_norm = np.linalg.norm(result.embedding)
        assert abs(embedding_norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_multilingual_support(self, embedding_provider):
        """Test multilingual text embedding."""
        # Setup multilingual texts
        multilingual_texts = [
            "Hello, how are you?",  # English
            "Bonjour, comment allez-vous?",  # French
            "Hola, ¿cómo estás?",  # Spanish
            "こんにちは、元気ですか？",  # Japanese
            "你好，你好吗？"  # Chinese
        ]
        
        # Execute
        results = []
        for text in multilingual_texts:
            request = EmbeddingRequest(text=text)
            result = await embedding_provider.generate_embedding(request)
            results.append(result)
        
        # Verify all embeddings generated successfully
        assert all(r.success for r in results)
        assert all(len(r.embedding) == 1024 for r in results)
        
        # Verify embeddings are different (multilingual model should handle different languages)
        embeddings = [r.embedding for r in results]
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j])
                assert similarity < 0.99  # Should not be identical

    @pytest.mark.asyncio
    async def test_long_text_handling(self, embedding_provider):
        """Test handling of very long text inputs."""
        # Setup very long text
        long_text = "This is a very long sentence. " * 1000  # Very long text
        request = EmbeddingRequest(text=long_text, max_length=512)
        
        # Execute
        result = await embedding_provider.generate_embedding(request)
        
        # Verify
        assert result.success is True
        assert result.truncated is True
        assert len(result.embedding) == 1024
        
        # Verify text was truncated before processing
        call_args = embedding_provider.primary_model.encode.call_args[0]
        processed_text = call_args[0][0] if isinstance(call_args[0], list) else call_args[0]
        assert len(processed_text.split()) <= 512

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, embedding_provider):
        """Test handling of empty or whitespace-only text."""
        # Test cases
        empty_cases = ["", "   ", "\n\t\r", None]
        
        for empty_text in empty_cases:
            request = EmbeddingRequest(text=empty_text)
            result = await embedding_provider.generate_embedding(request)
            
            # Should handle gracefully
            if empty_text is None:
                assert result.success is False
                assert "empty" in result.error_message.lower()
            else:
                # Empty strings should get default/zero embeddings or error
                assert result.success is False or all(x == 0 for x in result.embedding)

    @pytest.mark.asyncio
    async def test_performance_metrics(self, embedding_provider):
        """Test performance metrics collection."""
        # Setup
        text = "Performance test sentence."
        request = EmbeddingRequest(text=text, collect_metrics=True)
        
        # Execute
        result = await embedding_provider.generate_embedding(request)
        
        # Verify metrics are collected
        assert result.processing_time > 0
        assert result.model_used is not None
        assert hasattr(result, 'memory_usage')
        assert hasattr(result, 'tokens_processed')

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, embedding_provider):
        """Test concurrent embedding generation."""
        import asyncio
        
        # Setup multiple requests
        texts = [f"Concurrent test sentence {i}" for i in range(10)]
        requests = [EmbeddingRequest(text=text) for text in texts]
        
        # Execute concurrently
        tasks = [embedding_provider.generate_embedding(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert len(results) == 10
        assert all(r.success for r in results)
        assert all(len(r.embedding) == 1024 for r in results)

    @pytest.mark.asyncio
    async def test_model_switching(self, embedding_provider):
        """Test dynamic model switching."""
        # Setup
        text = "Model switching test."
        
        # Test primary model
        request_primary = EmbeddingRequest(text=text, use_primary=True)
        result_primary = await embedding_provider.generate_embedding(request_primary)
        
        # Test fallback model
        request_fallback = EmbeddingRequest(text=text, use_primary=False)
        result_fallback = await embedding_provider.generate_embedding(request_fallback)
        
        # Verify different models were used
        assert result_primary.model_used != result_fallback.model_used
        assert len(result_primary.embedding) != len(result_fallback.embedding)

    @pytest.mark.asyncio
    async def test_embedding_quality_metrics(self, embedding_provider):
        """Test embedding quality assessment."""
        # Setup similar and dissimilar texts
        similar_texts = [
            "The cat sat on the mat.",
            "A cat was sitting on a mat."
        ]
        
        dissimilar_texts = [
            "The cat sat on the mat.",
            "Quantum physics is fascinating."
        ]
        
        # Generate embeddings
        similar_embeddings = []
        for text in similar_texts:
            request = EmbeddingRequest(text=text)
            result = await embedding_provider.generate_embedding(request)
            similar_embeddings.append(result.embedding)
        
        dissimilar_embeddings = []
        for text in dissimilar_texts:
            request = EmbeddingRequest(text=text)
            result = await embedding_provider.generate_embedding(request)
            dissimilar_embeddings.append(result.embedding)
        
        # Calculate similarities
        similar_similarity = np.dot(similar_embeddings[0], similar_embeddings[1])
        dissimilar_similarity = np.dot(dissimilar_embeddings[0], dissimilar_embeddings[1])
        
        # Verify quality - similar texts should have higher similarity
        assert similar_similarity > dissimilar_similarity
        assert similar_similarity > 0.7  # Should be quite similar
        assert dissimilar_similarity < 0.5  # Should be quite different

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, embedding_provider):
        """Test circuit breaker integration for resilience."""
        # Setup circuit breaker
        embedding_provider.circuit_breaker = AsyncMock()
        embedding_provider.circuit_breaker.call.side_effect = Exception("Circuit breaker open")
        
        text = "Circuit breaker test."
        request = EmbeddingRequest(text=text)
        
        # Execute
        result = await embedding_provider.generate_embedding(request)
        
        # Verify graceful handling
        assert result.success is False
        assert "circuit breaker" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, embedding_provider):
        """Test memory efficiency for large batches."""
        # Setup large batch
        large_batch_texts = [f"Memory test sentence {i}" for i in range(100)]
        request = EmbeddingRequest(texts=large_batch_texts, batch_size=10)
        
        # Mock memory monitoring
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
            
            # Execute
            results = await embedding_provider.generate_batch_embeddings(request)
            
            # Verify
            assert len(results) == 100
            assert all(r.success for r in results)
            
            # Verify batching was used (should be 10 batches of 10)
            expected_calls = 10
            assert embedding_provider.primary_model.encode.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_health_check(self, embedding_provider):
        """Test embedding provider health check."""
        # Execute health check
        health = await embedding_provider.health_check()
        
        # Verify
        assert health["status"] == "healthy"
        assert "primary_model" in health["models"]
        assert "fallback_model" in health["models"]
        assert health["models"]["primary_model"]["status"] == "loaded"
        assert health["models"]["fallback_model"]["status"] == "loaded"

    @pytest.mark.asyncio
    async def test_embedding_comparison_utilities(self, embedding_provider):
        """Test embedding comparison and similarity utilities."""
        # Setup
        text1 = "Machine learning is awesome."
        text2 = "Artificial intelligence is great."
        text3 = "I love pizza and pasta."
        
        # Generate embeddings
        embeddings = []
        for text in [text1, text2, text3]:
            request = EmbeddingRequest(text=text)
            result = await embedding_provider.generate_embedding(request)
            embeddings.append(result.embedding)
        
        # Test similarity calculations
        sim_1_2 = embedding_provider.calculate_similarity(embeddings[0], embeddings[1])
        sim_1_3 = embedding_provider.calculate_similarity(embeddings[0], embeddings[2])
        
        # Verify - ML and AI should be more similar than ML and pizza
        assert sim_1_2 > sim_1_3
        assert 0 <= sim_1_2 <= 1
        assert 0 <= sim_1_3 <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])