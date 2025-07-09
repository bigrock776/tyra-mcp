"""
Comprehensive unit tests for Reranking System.

Tests cross-encoder reranking, vLLM integration, and performance optimization.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from src.core.rag.reranker import Reranker, RerankRequest, RerankResult


class TestReranker:
    """Test Reranker functionality."""

    @pytest.fixture
    async def reranker(self):
        """Create reranker with mocked models."""
        with patch('src.core.rag.reranker.CrossEncoder') as mock_ce:
            reranker = Reranker()
            
            # Mock cross-encoder model
            reranker.cross_encoder = MagicMock()
            reranker.cross_encoder.predict.return_value = [0.8, 0.6, 0.9, 0.4]  # Relevance scores
            
            # Mock vLLM model (if available)
            reranker.vllm_model = AsyncMock()
            
            reranker._initialized = True
            yield reranker

    @pytest.mark.asyncio
    async def test_cross_encoder_reranking(self, reranker):
        """Test cross-encoder based reranking."""
        # Setup
        query = "What is machine learning?"
        documents = [
            {"id": "doc1", "text": "Machine learning is a subset of AI", "score": 0.7},
            {"id": "doc2", "text": "Python is a programming language", "score": 0.8},
            {"id": "doc3", "text": "ML algorithms learn from data", "score": 0.6},
            {"id": "doc4", "text": "Weather forecast for tomorrow", "score": 0.5}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            top_k=3,
            method="cross_encoder"
        )
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify
        assert isinstance(result, RerankResult)
        assert len(result.reranked_documents) == 3
        assert result.reranked_documents[0]["id"] == "doc3"  # Highest score (0.9)
        assert result.reranked_documents[1]["id"] == "doc1"  # Second highest (0.8)
        assert result.reranked_documents[2]["id"] == "doc2"  # Third highest (0.6)
        
        # Verify scores were updated
        assert "rerank_score" in result.reranked_documents[0]
        assert result.method_used == "cross_encoder"

    @pytest.mark.asyncio
    async def test_vllm_reranking(self, reranker):
        """Test vLLM-based reranking."""
        # Setup
        query = "Explain quantum computing"
        documents = [
            {"id": "quantum1", "text": "Quantum computing uses quantum mechanics", "score": 0.7},
            {"id": "quantum2", "text": "Classical computers use binary bits", "score": 0.6},
            {"id": "quantum3", "text": "Qubits can be in superposition", "score": 0.5}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="vllm",
            top_k=2
        )
        
        # Mock vLLM response
        reranker.vllm_model.generate.return_value = [
            {"relevance_score": 0.95, "reasoning": "Directly explains quantum computing"},
            {"relevance_score": 0.3, "reasoning": "About classical computing"},
            {"relevance_score": 0.8, "reasoning": "Explains quantum concept"}
        ]
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify
        assert len(result.reranked_documents) == 2
        assert result.reranked_documents[0]["id"] == "quantum1"  # Highest vLLM score
        assert result.reranked_documents[1]["id"] == "quantum3"  # Second highest
        assert result.method_used == "vllm"
        assert "reasoning" in result.reranked_documents[0]

    @pytest.mark.asyncio
    async def test_hybrid_reranking(self, reranker):
        """Test hybrid reranking combining multiple methods."""
        # Setup
        query = "Best practices for software development"
        documents = [
            {"id": "dev1", "text": "Code review improves quality", "score": 0.6},
            {"id": "dev2", "text": "Testing prevents bugs", "score": 0.7},
            {"id": "dev3", "text": "Documentation is important", "score": 0.5}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="hybrid",
            top_k=3
        )
        
        # Mock both models
        reranker.cross_encoder.predict.return_value = [0.8, 0.9, 0.7]
        reranker.vllm_model.generate.return_value = [
            {"relevance_score": 0.7},
            {"relevance_score": 0.8},
            {"relevance_score": 0.9}
        ]
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify hybrid scoring
        assert result.method_used == "hybrid"
        assert all("hybrid_score" in doc for doc in result.reranked_documents)
        assert all("cross_encoder_score" in doc for doc in result.reranked_documents)
        assert all("vllm_score" in doc for doc in result.reranked_documents)

    @pytest.mark.asyncio
    async def test_relevance_threshold_filtering(self, reranker):
        """Test filtering documents below relevance threshold."""
        # Setup
        query = "Python programming"
        documents = [
            {"id": "py1", "text": "Python is great for data science", "score": 0.8},
            {"id": "py2", "text": "Java is object-oriented", "score": 0.7},
            {"id": "py3", "text": "Weather is nice today", "score": 0.6}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="cross_encoder",
            relevance_threshold=0.7,  # Filter out low relevance
            top_k=10
        )
        
        # Mock scores - only first document meets threshold
        reranker.cross_encoder.predict.return_value = [0.9, 0.5, 0.2]
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify filtering
        assert len(result.reranked_documents) == 1
        assert result.reranked_documents[0]["id"] == "py1"
        assert result.filtered_count == 2

    @pytest.mark.asyncio
    async def test_diversity_penalties(self, reranker):
        """Test diversity penalties to avoid redundant results."""
        # Setup similar documents
        query = "Machine learning algorithms"
        documents = [
            {"id": "ml1", "text": "Machine learning algorithms are powerful", "score": 0.9},
            {"id": "ml2", "text": "ML algorithms are very powerful tools", "score": 0.8},  # Very similar
            {"id": "ml3", "text": "Neural networks are ML algorithms", "score": 0.7},
            {"id": "other", "text": "Database design principles", "score": 0.6}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="cross_encoder",
            apply_diversity_penalty=True,
            diversity_threshold=0.8,  # High similarity threshold
            top_k=3
        )
        
        # Mock high rerank scores
        reranker.cross_encoder.predict.return_value = [0.95, 0.90, 0.85, 0.4]
        
        # Mock similarity calculation (ml1 and ml2 are very similar)
        with patch.object(reranker, '_calculate_similarity') as mock_sim:
            mock_sim.side_effect = lambda a, b: 0.9 if ("ml1" in a and "ml2" in b) else 0.3
            
            # Execute
            result = await reranker.rerank(request)
        
        # Verify diversity - should include ml1, ml3, other (skipping similar ml2)
        assert len(result.reranked_documents) == 3
        doc_ids = [doc["id"] for doc in result.reranked_documents]
        assert "ml1" in doc_ids
        assert "ml3" in doc_ids
        assert "other" in doc_ids
        # ml2 should be filtered out due to similarity with ml1

    @pytest.mark.asyncio
    async def test_batch_reranking(self, reranker):
        """Test efficient batch reranking."""
        # Setup multiple queries
        batch_requests = [
            RerankRequest(
                query="Python programming",
                documents=[
                    {"id": "py1", "text": "Python basics", "score": 0.7},
                    {"id": "py2", "text": "Python advanced", "score": 0.6}
                ],
                top_k=2
            ),
            RerankRequest(
                query="Machine learning",
                documents=[
                    {"id": "ml1", "text": "ML fundamentals", "score": 0.8},
                    {"id": "ml2", "text": "Deep learning", "score": 0.7}
                ],
                top_k=2
            )
        ]
        
        # Mock batch processing
        reranker.cross_encoder.predict.side_effect = [
            [0.9, 0.8],  # Scores for first query
            [0.95, 0.85]  # Scores for second query
        ]
        
        # Execute batch
        results = await reranker.batch_rerank(batch_requests)
        
        # Verify
        assert len(results) == 2
        assert all(isinstance(r, RerankResult) for r in results)
        assert len(results[0].reranked_documents) == 2
        assert len(results[1].reranked_documents) == 2

    @pytest.mark.asyncio
    async def test_caching_rerank_results(self, reranker):
        """Test caching of rerank results for performance."""
        # Setup cache
        reranker.cache = AsyncMock()
        
        query = "Test query for caching"
        documents = [{"id": "doc1", "text": "Test document", "score": 0.7}]
        request = RerankRequest(query=query, documents=documents)
        
        # Mock cache miss, then hit
        cached_result = RerankResult(
            reranked_documents=[{"id": "doc1", "rerank_score": 0.8}],
            method_used="cross_encoder",
            cached=True
        )
        reranker.cache.get.side_effect = [None, cached_result]
        
        # First call - cache miss
        result1 = await reranker.rerank(request)
        
        # Second call - cache hit
        result2 = await reranker.rerank(request)
        
        # Verify caching
        assert result1.cached is False
        assert result2.cached is True
        reranker.cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_model_failure(self, reranker):
        """Test fallback behavior when primary reranking fails."""
        # Setup
        query = "Fallback test query"
        documents = [
            {"id": "doc1", "text": "First document", "score": 0.8},
            {"id": "doc2", "text": "Second document", "score": 0.6}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="cross_encoder"
        )
        
        # Mock cross-encoder failure
        reranker.cross_encoder.predict.side_effect = Exception("Model loading failed")
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify fallback to original scores
        assert result.method_used == "fallback"
        assert len(result.reranked_documents) == 2
        assert result.reranked_documents[0]["score"] == 0.8  # Original order preserved
        assert result.error_message == "Model loading failed"

    @pytest.mark.asyncio
    async def test_performance_optimization(self, reranker):
        """Test performance optimizations for large document sets."""
        # Setup large document set
        query = "Performance test"
        documents = [
            {"id": f"doc{i}", "text": f"Document {i} content", "score": 0.5 + (i % 10) * 0.05}
            for i in range(1000)  # Large set
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="cross_encoder",
            top_k=10,
            max_rerank_candidates=100  # Limit for performance
        )
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify performance optimizations
        assert len(result.reranked_documents) == 10
        assert result.candidates_considered <= 100  # Should limit candidates
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_query_document_pair_scoring(self, reranker):
        """Test detailed query-document pair scoring."""
        # Setup
        query = "Artificial intelligence applications"
        documents = [
            {"id": "ai1", "text": "AI is used in healthcare for diagnosis", "score": 0.7},
            {"id": "ai2", "text": "Machine learning in finance", "score": 0.6},
            {"id": "other", "text": "Cooking recipes for dinner", "score": 0.5}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="cross_encoder",
            include_scores=True
        )
        
        # Mock detailed scoring
        reranker.cross_encoder.predict.return_value = [0.9, 0.8, 0.1]
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify detailed scoring
        assert all("rerank_score" in doc for doc in result.reranked_documents)
        assert all("original_score" in doc for doc in result.reranked_documents)
        assert all("score_improvement" in doc for doc in result.reranked_documents)

    @pytest.mark.asyncio
    async def test_custom_scoring_weights(self, reranker):
        """Test custom scoring weight combinations."""
        # Setup
        query = "Custom scoring test"
        documents = [
            {"id": "doc1", "text": "Relevant content", "score": 0.5, "popularity": 0.9},
            {"id": "doc2", "text": "Less relevant", "score": 0.8, "popularity": 0.3}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="custom",
            custom_weights={
                "rerank_score": 0.7,
                "original_score": 0.2,
                "popularity": 0.1
            }
        )
        
        # Mock rerank scores
        reranker.cross_encoder.predict.return_value = [0.9, 0.4]
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify custom scoring
        assert result.method_used == "custom"
        doc1_final = result.reranked_documents[0]
        assert "final_score" in doc1_final
        # Should combine rerank (0.9*0.7) + original (0.5*0.2) + popularity (0.9*0.1)

    @pytest.mark.asyncio
    async def test_confidence_scoring_integration(self, reranker):
        """Test integration with confidence scoring system."""
        # Setup
        query = "High confidence reranking test"
        documents = [
            {"id": "conf1", "text": "Well-sourced information with citations", "score": 0.7},
            {"id": "conf2", "text": "Speculation without sources", "score": 0.8}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="cross_encoder",
            apply_confidence_boost=True
        )
        
        # Mock confidence scoring
        with patch.object(reranker, '_calculate_confidence') as mock_conf:
            mock_conf.side_effect = [0.95, 0.3]  # High vs low confidence
            
            # Mock rerank scores (initially lower for high-confidence doc)
            reranker.cross_encoder.predict.return_value = [0.7, 0.8]
            
            # Execute
            result = await reranker.rerank(request)
        
        # Verify confidence boost applied
        assert result.reranked_documents[0]["id"] == "conf1"  # Should be boosted to top
        assert "confidence_score" in result.reranked_documents[0]

    @pytest.mark.asyncio
    async def test_multilingual_reranking(self, reranker):
        """Test reranking with multilingual content."""
        # Setup multilingual documents
        query = "machine learning"  # English query
        documents = [
            {"id": "en", "text": "Machine learning algorithms", "score": 0.7, "lang": "en"},
            {"id": "es", "text": "Algoritmos de aprendizaje automático", "score": 0.6, "lang": "es"},
            {"id": "fr", "text": "Algorithmes d'apprentissage automatique", "score": 0.5, "lang": "fr"}
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="cross_encoder",
            prefer_language="en"
        )
        
        # Mock scores with language preference
        reranker.cross_encoder.predict.return_value = [0.9, 0.85, 0.8]
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify language preference applied
        assert result.reranked_documents[0]["lang"] == "en"

    @pytest.mark.asyncio
    async def test_health_check(self, reranker):
        """Test reranker health check."""
        # Execute health check
        health = await reranker.health_check()
        
        # Verify
        assert health["status"] == "healthy"
        assert "cross_encoder" in health["models"]
        assert health["models"]["cross_encoder"]["status"] == "loaded"
        if "vllm" in health["models"]:
            assert health["models"]["vllm"]["status"] in ["loaded", "not_available"]

    @pytest.mark.asyncio
    async def test_explain_reranking_decisions(self, reranker):
        """Test explanation of reranking decisions."""
        # Setup
        query = "Explain machine learning"
        documents = [
            {"id": "exp1", "text": "ML is a subset of AI", "score": 0.6},
            {"id": "exp2", "text": "Cooking recipes", "score": 0.8}  # High original but irrelevant
        ]
        
        request = RerankRequest(
            query=query,
            documents=documents,
            method="cross_encoder",
            explain_decisions=True
        )
        
        # Mock scores that would reorder
        reranker.cross_encoder.predict.return_value = [0.9, 0.1]
        
        # Execute
        result = await reranker.rerank(request)
        
        # Verify explanations
        assert "explanations" in result.__dict__
        assert len(result.explanations) == 2
        assert all("reason" in exp for exp in result.explanations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])