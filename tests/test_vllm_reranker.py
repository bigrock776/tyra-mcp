"""Tests for vLLM reranker implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
import asyncio
from typing import Dict, Any

from src.core.rag.vllm_reranker import VLLMReranker


class TestVLLMReranker:
    """Test suite for vLLM reranker."""
    
    @pytest.fixture
    def vllm_config(self) -> Dict[str, Any]:
        """Basic vLLM configuration."""
        return {
            "base_url": "http://localhost:8000/v1",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "api_key": "dummy-key",
            "max_tokens": 10,
            "temperature": 0.0,
            "batch_size": 4,
            "timeout": 30
        }
    
    @pytest.fixture
    def sample_documents(self) -> list:
        """Sample documents for testing."""
        return [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "The weather is sunny today with clear skies.",
            "Deep learning uses neural networks with multiple layers.",
            "Cats are popular pets that enjoy sleeping."
        ]
    
    @pytest.fixture
    def mock_vllm_response(self):
        """Mock vLLM API response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": "8"
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_vllm_reranker_initialization(self, vllm_config):
        """Test vLLM reranker initialization."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker()
            
            with patch.object(reranker, '_test_connection', new_callable=AsyncMock):
                await reranker.initialize(vllm_config)
                
                assert reranker.base_url == vllm_config["base_url"]
                assert reranker.model_name == vllm_config["model_name"]
                assert reranker.batch_size == vllm_config["batch_size"]
                assert reranker._session is not None
                
            await reranker.close()
    
    @pytest.mark.asyncio
    async def test_connection_test(self, vllm_config):
        """Test connection test functionality."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker()
            reranker._session = mock_session
            
            # Should not raise an exception
            await reranker._test_connection()
            
            # Verify POST was called
            mock_session.post.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_connection_test_failure(self, vllm_config):
        """Test connection test failure handling."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock failed response
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker()
            reranker._session = mock_session
            
            # Should raise an exception
            with pytest.raises(RuntimeError, match="vLLM connection test failed"):
                await reranker._test_connection()
    
    @pytest.mark.asyncio
    async def test_score_single_document(self, vllm_config, mock_vllm_response):
        """Test scoring a single document."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_vllm_response
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker()
            reranker._session = mock_session
            
            query = "machine learning"
            document = "Machine learning is a subset of artificial intelligence."
            
            score = await reranker._score_single_document(query, document)
            
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert score == 0.8  # 8/10 normalized
            
    @pytest.mark.asyncio
    async def test_score_batch(self, vllm_config, sample_documents, mock_vllm_response):
        """Test scoring a batch of documents."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_vllm_response
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker(batch_size=2)
            reranker._session = mock_session
            
            query = "machine learning"
            documents = sample_documents[:3]  # Test with 3 documents
            
            scores = await reranker._score_batch(query, documents)
            
            assert len(scores) == len(documents)
            assert all(isinstance(score, float) for score in scores)
            assert all(0.0 <= score <= 1.0 for score in scores)
            
    @pytest.mark.asyncio
    async def test_rerank_full_flow(self, vllm_config, sample_documents, mock_vllm_response):
        """Test full reranking flow."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_vllm_response
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker(batch_size=2)
            reranker._session = mock_session
            
            query = "machine learning"
            documents = sample_documents
            
            scores = await reranker.rerank(query, documents)
            
            assert len(scores) == len(documents)
            assert all(isinstance(score, float) for score in scores)
            assert all(0.0 <= score <= 1.0 for score in scores)
            
    @pytest.mark.asyncio
    async def test_rerank_with_scores(self, vllm_config, sample_documents, mock_vllm_response):
        """Test reranking with index-score pairs."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock different scores for different documents
            responses = [
                {"choices": [{"message": {"content": "9"}}]},  # High score
                {"choices": [{"message": {"content": "3"}}]},  # Low score
                {"choices": [{"message": {"content": "7"}}]},  # Medium score
            ]
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.side_effect = responses
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker()
            reranker._session = mock_session
            
            query = "machine learning"
            documents = sample_documents[:3]
            
            indexed_scores = await reranker.rerank_with_scores(query, documents, top_k=2)
            
            assert len(indexed_scores) == 2  # top_k=2
            assert all(isinstance(item, tuple) for item in indexed_scores)
            assert all(len(item) == 2 for item in indexed_scores)
            
            # Check that results are sorted by score (descending)
            scores = [score for _, score in indexed_scores]
            assert scores == sorted(scores, reverse=True)
            
    @pytest.mark.asyncio
    async def test_extract_score_patterns(self):
        """Test score extraction from various response patterns."""
        reranker = VLLMReranker()
        
        test_cases = [
            ("8", 8.0),
            ("8.5", 8.5),
            ("8/10", 8.0),
            ("8.5/10", 8.5),
            ("Score: 7", 7.0),
            ("The relevance is 9 out of 10", 9.0),
            ("invalid response", 5.0),  # Default fallback
            ("", 5.0),  # Empty string fallback
        ]
        
        for content, expected in test_cases:
            score = reranker._extract_score(content)
            assert score == expected, f"Failed for content: '{content}'"
            
    @pytest.mark.asyncio
    async def test_cache_integration(self, vllm_config, sample_documents):
        """Test caching functionality."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True  # Cache set success
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "8"}}]}
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker(cache_manager=mock_cache)
            reranker._session = mock_session
            
            query = "test query"
            documents = sample_documents[:2]
            
            # First call should miss cache and set result
            scores = await reranker.rerank(query, documents)
            
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_called_once()
            
            # Reset mocks
            mock_cache.reset_mock()
            mock_cache.get.return_value = scores  # Cache hit
            
            # Second call should hit cache
            cached_scores = await reranker.rerank(query, documents)
            
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_not_called()  # Should not set again
            assert cached_scores == scores
            
    @pytest.mark.asyncio
    async def test_error_handling(self, vllm_config, sample_documents):
        """Test error handling for various failure scenarios."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Test timeout error
            mock_session.post.side_effect = asyncio.TimeoutError()
            
            reranker = VLLMReranker()
            reranker._session = mock_session
            
            score = await reranker._score_single_document("query", "document")
            assert score == 0.5  # Default fallback score
            
            # Test HTTP error
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Server Error"
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            mock_session.post.side_effect = None  # Reset side effect
            
            score = await reranker._score_single_document("query", "document")
            assert score == 0.5  # Default fallback score
            
    @pytest.mark.asyncio
    async def test_health_check(self, vllm_config):
        """Test health check functionality."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "8"}}]}
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            
            reranker = VLLMReranker()
            reranker._session = mock_session
            
            health = await reranker.health_check()
            
            assert health["status"] == "healthy"
            assert "test_score" in health
            assert "test_latency" in health
            assert health["model"] == reranker.model_name
            
    @pytest.mark.asyncio
    async def test_health_check_failure(self, vllm_config):
        """Test health check failure handling."""
        reranker = VLLMReranker()
        # Don't initialize session to simulate failure
        
        health = await reranker.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
        
    def test_model_info(self, vllm_config):
        """Test model info retrieval."""
        reranker = VLLMReranker(**vllm_config)
        
        info = reranker.get_model_info()
        
        assert info["type"] == "vllm_reranker"
        assert info["model_name"] == vllm_config["model_name"]
        assert info["base_url"] == vllm_config["base_url"]
        assert info["batch_size"] == vllm_config["batch_size"]
        assert "performance" in info
        
    @pytest.mark.asyncio
    async def test_empty_documents(self, vllm_config):
        """Test handling of empty document list."""
        reranker = VLLMReranker()
        
        scores = await reranker.rerank("query", [])
        
        assert scores == []
        
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, vllm_config, sample_documents):
        """Test cache key generation."""
        reranker = VLLMReranker()
        
        query = "test query"
        documents = sample_documents[:3]
        
        key1 = reranker._get_cache_key(query, documents)
        key2 = reranker._get_cache_key(query, documents)
        key3 = reranker._get_cache_key("different query", documents)
        
        # Same query and documents should generate same key
        assert key1 == key2
        
        # Different query should generate different key
        assert key1 != key3
        
        # Key should be string
        assert isinstance(key1, str)
        assert key1.startswith("vllm_rerank:")