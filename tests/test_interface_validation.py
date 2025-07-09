"""
Interface validation tests for all provider types.

This module contains comprehensive tests to ensure that all provider implementations
properly implement their required interfaces and handle edge cases correctly.
"""

import asyncio
import inspect
import pytest
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from src.core.interfaces.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingInitializationError,
    EmbeddingGenerationError,
    EmbeddingConfigurationError,
)
from src.core.interfaces.vector_store import (
    VectorStore,
    VectorDocument,
    VectorSearchResult,
    VectorStoreError,
    VectorStoreInitializationError,
    VectorStoreOperationError,
)
from src.core.interfaces.graph_engine import (
    GraphEngine,
    Entity,
    Relationship,
    RelationshipType,
    GraphEngineError,
    GraphEngineInitializationError,
    GraphEngineOperationError,
)
from src.core.interfaces.reranker import (
    Reranker,
    RerankingCandidate,
    RerankingResult,
    RerankerType,
    RerankerError,
    RerankerInitializationError,
    RerankerOperationError,
)
from src.core.providers.embeddings.huggingface import HuggingFaceProvider
from src.core.providers.vector_stores.pgvector import PgVectorStore
from src.core.providers.graph_engines.memgraph import MemgraphEngine
from src.core.providers.rerankers.cross_encoder import CrossEncoderReranker
from datetime import datetime


class TestInterfaceValidation:
    """Test suite for validating provider interface implementations."""

    def test_abstract_interface_completeness(self):
        """Test that all abstract interfaces have required methods."""
        
        # Test EmbeddingProvider interface
        embedding_required_methods = [
            "initialize", "embed_texts", "embed_query", "get_dimension", 
            "supports_gpu", "get_model_name", "health_check"
        ]
        
        for method_name in embedding_required_methods:
            assert hasattr(EmbeddingProvider, method_name), f"EmbeddingProvider missing {method_name}"
            method = getattr(EmbeddingProvider, method_name)
            assert getattr(method, '__isabstractmethod__', False), f"{method_name} not abstract"
        
        # Test VectorStore interface
        vector_store_required_methods = [
            "initialize", "store_documents", "store_document", "search_similar",
            "hybrid_search", "get_document", "update_document", "delete_document",
            "delete_documents", "count_documents", "list_documents", "create_index",
            "health_check", "get_stats"
        ]
        
        for method_name in vector_store_required_methods:
            assert hasattr(VectorStore, method_name), f"VectorStore missing {method_name}"
            method = getattr(VectorStore, method_name)
            assert getattr(method, '__isabstractmethod__', False), f"{method_name} not abstract"
        
        # Test GraphEngine interface
        graph_engine_required_methods = [
            "initialize", "create_entity", "create_entities", "get_entity",
            "update_entity", "delete_entity", "create_relationship", "create_relationships",
            "get_relationship", "delete_relationship", "find_entities", "get_entity_relationships",
            "get_connected_entities", "find_path", "execute_cypher", "get_entity_timeline",
            "health_check", "get_stats"
        ]
        
        for method_name in graph_engine_required_methods:
            assert hasattr(GraphEngine, method_name), f"GraphEngine missing {method_name}"
            method = getattr(GraphEngine, method_name)
            assert getattr(method, '__isabstractmethod__', False), f"{method_name} not abstract"
        
        # Test Reranker interface
        reranker_required_methods = [
            "initialize", "rerank", "score_pair", "get_reranker_type",
            "supports_batch_reranking", "get_max_candidates", "health_check"
        ]
        
        for method_name in reranker_required_methods:
            assert hasattr(Reranker, method_name), f"Reranker missing {method_name}"
            method = getattr(Reranker, method_name)
            assert getattr(method, '__isabstractmethod__', False), f"{method_name} not abstract"


class TestEmbeddingProviderValidation:
    """Test validation for embedding provider implementations."""

    def test_interface_implementation_completeness(self):
        """Test that HuggingFaceProvider implements all required methods."""
        provider = HuggingFaceProvider()
        
        # Check all abstract methods are implemented
        required_methods = [
            "initialize", "embed_texts", "embed_query", "get_dimension", 
            "supports_gpu", "get_model_name", "health_check"
        ]
        
        for method_name in required_methods:
            assert hasattr(provider, method_name), f"HuggingFaceProvider missing {method_name}"
            method = getattr(provider, method_name)
            assert callable(method), f"{method_name} is not callable"
        
        # Check method signatures match interface
        assert inspect.signature(provider.initialize).parameters.keys() == {"config"}
        assert inspect.signature(provider.embed_texts).parameters.keys() == {"texts", "batch_size"}
        assert inspect.signature(provider.embed_query).parameters.keys() == {"query"}
        assert len(inspect.signature(provider.get_dimension).parameters) == 0
        assert len(inspect.signature(provider.supports_gpu).parameters) == 0
        assert len(inspect.signature(provider.get_model_name).parameters) == 0
        assert len(inspect.signature(provider.health_check).parameters) == 0

    @pytest.mark.asyncio
    async def test_initialization_validation(self):
        """Test embedding provider initialization with various configurations."""
        provider = HuggingFaceProvider()
        
        # Test with valid configuration
        valid_config = {
            "model_name": "sentence-transformers/all-MiniLM-L12-v2",
            "device": "cpu",
            "batch_size": 16,
            "max_length": 256,
            "normalize_embeddings": True,
        }
        
        # Mock the model loading to avoid actual download
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_model.return_value.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value.max_seq_length = 256
            
            await provider.initialize(valid_config)
            
            assert provider.model_name == valid_config["model_name"]
            assert provider.device == valid_config["device"]
            assert provider.batch_size == valid_config["batch_size"]
            assert provider.max_sequence_length == valid_config["max_length"]
            assert provider.normalize_embeddings == valid_config["normalize_embeddings"]

        # Test with invalid configuration
        invalid_config = {
            "model_name": "non-existent-model",
            "device": "cuda:99",  # Invalid device
        }
        
        provider_invalid = HuggingFaceProvider()
        
        with pytest.raises(EmbeddingInitializationError):
            await provider_invalid.initialize(invalid_config)

    @pytest.mark.asyncio
    async def test_embedding_generation_validation(self):
        """Test embedding generation with various inputs."""
        provider = HuggingFaceProvider()
        
        # Mock initialization
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.random.rand(2, 384)
            mock_model.return_value = mock_instance
            
            await provider.initialize({
                "model_name": "sentence-transformers/all-MiniLM-L12-v2",
                "device": "cpu"
            })
            
            # Test embed_texts with valid input
            texts = ["Hello world", "This is a test"]
            embeddings = await provider.embed_texts(texts)
            
            assert isinstance(embeddings, list)
            assert len(embeddings) == len(texts)
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            assert all(emb.shape == (384,) for emb in embeddings)
            
            # Test embed_query with valid input
            mock_instance.encode.return_value = np.random.rand(1, 384)
            query_embedding = await provider.embed_query("test query")
            
            assert isinstance(query_embedding, np.ndarray)
            assert query_embedding.shape == (384,)
            
            # Test with empty input
            empty_embeddings = await provider.embed_texts([])
            assert empty_embeddings == []

    @pytest.mark.asyncio
    async def test_error_handling_validation(self):
        """Test error handling in embedding provider."""
        provider = HuggingFaceProvider()
        
        # Test embedding without initialization
        with pytest.raises(EmbeddingGenerationError):
            await provider.embed_texts(["test"])
        
        with pytest.raises(EmbeddingGenerationError):
            await provider.embed_query("test")
        
        # Test with initialized provider but model errors
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.side_effect = Exception("Model error")
            mock_model.return_value = mock_instance
            
            await provider.initialize({
                "model_name": "sentence-transformers/all-MiniLM-L12-v2",
                "device": "cpu"
            })
            
            with pytest.raises(EmbeddingGenerationError):
                await provider.embed_texts(["test"])
            
            with pytest.raises(EmbeddingGenerationError):
                await provider.embed_query("test")

    @pytest.mark.asyncio
    async def test_metadata_methods_validation(self):
        """Test metadata methods return correct types."""
        provider = HuggingFaceProvider()
        
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value = mock_instance
            
            await provider.initialize({
                "model_name": "sentence-transformers/all-MiniLM-L12-v2",
                "device": "cpu"
            })
            
            # Test get_dimension
            dimensions = provider.get_dimension()
            assert isinstance(dimensions, int)
            assert dimensions > 0
            
            # Test supports_gpu
            gpu_support = provider.supports_gpu()
            assert isinstance(gpu_support, bool)
            
            # Test get_model_name
            model_name = provider.get_model_name()
            assert isinstance(model_name, str)
            assert len(model_name) > 0
            
            # Test health_check
            health = await provider.health_check()
            assert isinstance(health, dict)
            assert "status" in health


class TestVectorStoreValidation:
    """Test validation for vector store implementations."""

    def test_interface_implementation_completeness(self):
        """Test that PgVectorStore implements all required methods."""
        store = PgVectorStore()
        
        required_methods = [
            "initialize", "store_documents", "store_document", "search_similar",
            "hybrid_search", "get_document", "update_document", "delete_document",
            "delete_documents", "count_documents", "list_documents", "create_index",
            "health_check", "get_stats"
        ]
        
        for method_name in required_methods:
            assert hasattr(store, method_name), f"PgVectorStore missing {method_name}"
            method = getattr(store, method_name)
            assert callable(method), f"{method_name} is not callable"

    @pytest.mark.asyncio
    async def test_initialization_validation(self):
        """Test vector store initialization."""
        store = PgVectorStore()
        
        config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "username": "test_user",
            "password": "test_pass",
            "table_name": "test_vectors",
            "dimensions": 384,
            "index_type": "hnsw",
            "distance_metric": "cosine",
        }
        
        # Mock database manager
        with patch('src.core.utils.database.PostgreSQLManager') as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.initialize.return_value = None
            mock_db_instance.execute_query.return_value = None
            mock_db.return_value = mock_db_instance
            
            await store.initialize(config)
            
            assert store.table_name == config["table_name"]
            assert store.dimensions == config["dimensions"]
            assert store.index_type == config["index_type"]
            assert store.distance_metric == config["distance_metric"]

    @pytest.mark.asyncio
    async def test_document_operations_validation(self):
        """Test document storage and retrieval operations."""
        store = PgVectorStore()
        
        # Mock database manager
        with patch('src.core.utils.database.PostgreSQLManager') as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.initialize.return_value = None
            mock_db_instance.execute_query.return_value = None
            mock_db_instance.execute_batch.return_value = None
            mock_db.return_value = mock_db_instance
            
            await store.initialize({
                "host": "localhost",
                "database": "test_db",
                "username": "test_user",
                "password": "test_pass",
            })
            
            # Test store_documents
            documents = [
                VectorDocument(
                    id="doc1",
                    content="Test document 1",
                    embedding=np.random.rand(384),
                    metadata={"type": "test"},
                ),
                VectorDocument(
                    id="doc2",
                    content="Test document 2",
                    embedding=np.random.rand(384),
                    metadata={"type": "test"},
                ),
            ]
            
            doc_ids = await store.store_documents(documents)
            assert isinstance(doc_ids, list)
            assert len(doc_ids) == len(documents)
            
            # Test store_document
            single_doc = VectorDocument(
                id="doc3",
                content="Test document 3",
                embedding=np.random.rand(384),
                metadata={"type": "single"},
            )
            
            doc_id = await store.store_document(single_doc)
            assert isinstance(doc_id, str)
            assert doc_id == single_doc.id

    @pytest.mark.asyncio
    async def test_search_operations_validation(self):
        """Test search operations return correct types."""
        store = PgVectorStore()
        
        # Mock database manager
        with patch('src.core.utils.database.PostgreSQLManager') as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.initialize.return_value = None
            mock_db_instance.execute_query.return_value = [
                {
                    "id": "doc1",
                    "content": "Test content",
                    "metadata": {"type": "test"},
                    "embedding": np.random.rand(384).tolist(),
                    "distance": 0.1,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
            ]
            mock_db.return_value = mock_db_instance
            
            await store.initialize({
                "host": "localhost",
                "database": "test_db",
                "username": "test_user",
                "password": "test_pass",
            })
            
            # Test search_similar
            query_embedding = np.random.rand(384)
            results = await store.search_similar(query_embedding, top_k=5)
            
            assert isinstance(results, list)
            assert all(isinstance(r, VectorSearchResult) for r in results)
            
            # Test hybrid_search
            hybrid_results = await store.hybrid_search(
                query_embedding, text_query="test query", top_k=5
            )
            
            assert isinstance(hybrid_results, list)
            assert all(isinstance(r, VectorSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_error_handling_validation(self):
        """Test error handling in vector store operations."""
        store = PgVectorStore()
        
        # Test operations without initialization
        with pytest.raises(AttributeError):
            await store.store_documents([])
        
        # Test with database errors
        with patch('src.core.utils.database.PostgreSQLManager') as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.initialize.side_effect = Exception("DB connection failed")
            mock_db.return_value = mock_db_instance
            
            with pytest.raises(VectorStoreInitializationError):
                await store.initialize({
                    "host": "localhost",
                    "database": "test_db",
                    "username": "test_user",
                    "password": "test_pass",
                })


class TestGraphEngineValidation:
    """Test validation for graph engine implementations."""

    def test_interface_implementation_completeness(self):
        """Test that MemgraphEngine implements all required methods."""
        # Import the actual implementation
        try:
            from src.core.providers.graph_engines.memgraph import MemgraphEngine
        except ImportError:
            pytest.skip("MemgraphEngine not available")
        
        engine = MemgraphEngine()
        
        required_methods = [
            "initialize", "create_entity", "create_entities", "get_entity",
            "update_entity", "delete_entity", "create_relationship", "create_relationships",
            "get_relationship", "delete_relationship", "find_entities", "get_entity_relationships",
            "get_connected_entities", "find_path", "execute_cypher", "get_entity_timeline",
            "health_check", "get_stats"
        ]
        
        for method_name in required_methods:
            assert hasattr(engine, method_name), f"MemgraphEngine missing {method_name}"
            method = getattr(engine, method_name)
            assert callable(method), f"{method_name} is not callable"

    @pytest.mark.asyncio
    async def test_entity_operations_validation(self):
        """Test entity operations return correct types."""
        try:
            from src.core.providers.graph_engines.memgraph import MemgraphEngine
        except ImportError:
            pytest.skip("MemgraphEngine not available")
        
        engine = MemgraphEngine()
        
        # Mock the database connection
        with patch('src.core.providers.graph_engines.memgraph.GQLAlchemy') as mock_gql:
            mock_session = AsyncMock()
            mock_gql.return_value = mock_session
            
            await engine.initialize({
                "host": "localhost",
                "port": 7687,
                "username": "test",
                "password": "test",
            })
            
            # Test create_entity
            entity = Entity(
                id="test_entity",
                name="Test Entity",
                entity_type="test",
                properties={"key": "value"},
                confidence=0.95,
            )
            
            # Mock the create response
            mock_session.execute_and_fetch.return_value = [{"id": "test_entity"}]
            
            entity_id = await engine.create_entity(entity)
            assert isinstance(entity_id, str)
            assert entity_id == "test_entity"

    @pytest.mark.asyncio
    async def test_relationship_operations_validation(self):
        """Test relationship operations return correct types."""
        try:
            from src.core.providers.graph_engines.memgraph import MemgraphEngine
        except ImportError:
            pytest.skip("MemgraphEngine not available")
        
        engine = MemgraphEngine()
        
        # Mock the database connection
        with patch('src.core.providers.graph_engines.memgraph.GQLAlchemy') as mock_gql:
            mock_session = AsyncMock()
            mock_gql.return_value = mock_session
            
            await engine.initialize({
                "host": "localhost",
                "port": 7687,
                "username": "test",
                "password": "test",
            })
            
            # Test create_relationship
            relationship = Relationship(
                id="test_rel",
                source_entity_id="entity1",
                target_entity_id="entity2",
                relationship_type="RELATED_TO",
                properties={"strength": 0.8},
                confidence=0.9,
            )
            
            # Mock the create response
            mock_session.execute_and_fetch.return_value = [{"id": "test_rel"}]
            
            rel_id = await engine.create_relationship(relationship)
            assert isinstance(rel_id, str)
            assert rel_id == "test_rel"


class TestRerankerValidation:
    """Test validation for reranker implementations."""

    def test_interface_implementation_completeness(self):
        """Test that CrossEncoderReranker implements all required methods."""
        try:
            from src.core.providers.rerankers.cross_encoder import CrossEncoderReranker
        except ImportError:
            pytest.skip("CrossEncoderReranker not available")
        
        reranker = CrossEncoderReranker()
        
        required_methods = [
            "initialize", "rerank", "score_pair", "get_reranker_type",
            "supports_batch_reranking", "get_max_candidates", "health_check"
        ]
        
        for method_name in required_methods:
            assert hasattr(reranker, method_name), f"CrossEncoderReranker missing {method_name}"
            method = getattr(reranker, method_name)
            assert callable(method), f"{method_name} is not callable"

    @pytest.mark.asyncio
    async def test_reranking_operations_validation(self):
        """Test reranking operations return correct types."""
        try:
            from src.core.providers.rerankers.cross_encoder import CrossEncoderReranker
        except ImportError:
            pytest.skip("CrossEncoderReranker not available")
        
        reranker = CrossEncoderReranker()
        
        # Mock the model
        with patch('sentence_transformers.CrossEncoder') as mock_model:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = np.array([0.8, 0.6, 0.9])
            mock_model.return_value = mock_instance
            
            await reranker.initialize({
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu",
            })
            
            # Test rerank
            candidates = [
                RerankingCandidate(
                    id="doc1",
                    content="Document 1 content",
                    original_score=0.7,
                    metadata={"type": "test"},
                ),
                RerankingCandidate(
                    id="doc2",
                    content="Document 2 content",
                    original_score=0.6,
                    metadata={"type": "test"},
                ),
            ]
            
            results = await reranker.rerank("test query", candidates)
            
            assert isinstance(results, list)
            assert all(isinstance(r, RerankingResult) for r in results)
            assert len(results) == len(candidates)
            
            # Test score_pair
            score = await reranker.score_pair("test query", "test document")
            assert isinstance(score, float)
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_metadata_methods_validation(self):
        """Test metadata methods return correct types."""
        try:
            from src.core.providers.rerankers.cross_encoder import CrossEncoderReranker
        except ImportError:
            pytest.skip("CrossEncoderReranker not available")
        
        reranker = CrossEncoderReranker()
        
        # Mock the model
        with patch('sentence_transformers.CrossEncoder') as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            
            await reranker.initialize({
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu",
            })
            
            # Test get_reranker_type
            reranker_type = reranker.get_reranker_type()
            assert isinstance(reranker_type, RerankerType)
            
            # Test supports_batch_reranking
            batch_support = reranker.supports_batch_reranking()
            assert isinstance(batch_support, bool)
            
            # Test get_max_candidates
            max_candidates = reranker.get_max_candidates()
            assert isinstance(max_candidates, int)
            assert max_candidates > 0


class TestInterfaceErrorHandling:
    """Test error handling across all interfaces."""

    def test_exception_hierarchy(self):
        """Test that all exception types are properly defined."""
        # Test embedding exceptions
        assert issubclass(EmbeddingProviderError, Exception)
        assert issubclass(EmbeddingInitializationError, EmbeddingProviderError)
        assert issubclass(EmbeddingGenerationError, EmbeddingProviderError)
        assert issubclass(EmbeddingConfigurationError, EmbeddingProviderError)
        
        # Test vector store exceptions
        assert issubclass(VectorStoreError, Exception)
        assert issubclass(VectorStoreInitializationError, VectorStoreError)
        assert issubclass(VectorStoreOperationError, VectorStoreError)
        
        # Test graph engine exceptions
        assert issubclass(GraphEngineError, Exception)
        assert issubclass(GraphEngineInitializationError, GraphEngineError)
        assert issubclass(GraphEngineOperationError, GraphEngineError)
        
        # Test reranker exceptions
        assert issubclass(RerankerError, Exception)
        assert issubclass(RerankerInitializationError, RerankerError)
        assert issubclass(RerankerOperationError, RerankerError)

    @pytest.mark.asyncio
    async def test_common_error_scenarios(self):
        """Test common error scenarios across providers."""
        # Test with non-existent provider
        with pytest.raises(ImportError):
            from src.core.providers.embeddings.nonexistent import NonexistentProvider
        
        # Test invalid configuration handling
        provider = HuggingFaceProvider()
        
        with pytest.raises(EmbeddingInitializationError):
            await provider.initialize({})  # Empty config
        
        with pytest.raises(EmbeddingInitializationError):
            await provider.initialize({"model_name": None})  # Invalid model name


class TestDataStructureValidation:
    """Test validation of data structures used in interfaces."""

    def test_vector_document_structure(self):
        """Test VectorDocument data structure."""
        doc = VectorDocument(
            id="test_doc",
            content="Test content",
            embedding=np.random.rand(384),
            metadata={"type": "test"},
        )
        
        assert doc.id == "test_doc"
        assert doc.content == "Test content"
        assert isinstance(doc.embedding, np.ndarray)
        assert doc.embedding.shape == (384,)
        assert doc.metadata == {"type": "test"}

    def test_entity_structure(self):
        """Test Entity data structure."""
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            entity_type="test",
            properties={"key": "value"},
            confidence=0.95,
        )
        
        assert entity.id == "test_entity"
        assert entity.name == "Test Entity"
        assert entity.entity_type == "test"
        assert entity.properties == {"key": "value"}
        assert entity.confidence == 0.95

    def test_relationship_structure(self):
        """Test Relationship data structure."""
        rel = Relationship(
            id="test_rel",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type="RELATED_TO",
            properties={"strength": 0.8},
            confidence=0.9,
        )
        
        assert rel.id == "test_rel"
        assert rel.source_entity_id == "entity1"
        assert rel.target_entity_id == "entity2"
        assert rel.relationship_type == "RELATED_TO"
        assert rel.properties == {"strength": 0.8}
        assert rel.confidence == 0.9

    def test_reranking_candidate_structure(self):
        """Test RerankingCandidate data structure."""
        candidate = RerankingCandidate(
            id="doc1",
            content="Document content",
            original_score=0.7,
            metadata={"type": "test"},
        )
        
        assert candidate.id == "doc1"
        assert candidate.content == "Document content"
        assert candidate.original_score == 0.7
        assert candidate.metadata == {"type": "test"}

    def test_reranking_result_structure(self):
        """Test RerankingResult data structure."""
        result = RerankingResult(
            id="doc1",
            content="Document content",
            original_score=0.7,
            rerank_score=0.8,
            final_score=0.75,
            metadata={"type": "test"},
            explanation="Reranked based on relevance",
        )
        
        assert result.id == "doc1"
        assert result.content == "Document content"
        assert result.original_score == 0.7
        assert result.rerank_score == 0.8
        assert result.final_score == 0.75
        assert result.metadata == {"type": "test"}
        assert result.explanation == "Reranked based on relevance"


class TestInterfaceCompatibility:
    """Test compatibility between different interface implementations."""

    @pytest.mark.asyncio
    async def test_embedding_vector_store_compatibility(self):
        """Test that embedding provider output is compatible with vector store input."""
        # Create embedding provider
        embedding_provider = HuggingFaceProvider()
        
        # Mock the model
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.random.rand(2, 384)
            mock_model.return_value = mock_instance
            
            await embedding_provider.initialize({
                "model_name": "sentence-transformers/all-MiniLM-L12-v2",
                "device": "cpu"
            })
            
            # Generate embeddings
            texts = ["Text 1", "Text 2"]
            embeddings = await embedding_provider.embed_texts(texts)
            
            # Create vector documents
            documents = [
                VectorDocument(
                    id=f"doc_{i}",
                    content=text,
                    embedding=embedding,
                    metadata={"source": "test"},
                )
                for i, (text, embedding) in enumerate(zip(texts, embeddings))
            ]
            
            # Test that documents are valid for vector store
            for doc in documents:
                assert isinstance(doc.embedding, np.ndarray)
                assert doc.embedding.shape == (384,)
                assert doc.embedding.dtype == np.float64 or doc.embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_search_reranker_compatibility(self):
        """Test that search results are compatible with reranker input."""
        # Create mock search results
        search_results = [
            VectorSearchResult(
                id="doc1",
                score=0.8,
                metadata={"type": "test"},
                content="Document 1 content",
            ),
            VectorSearchResult(
                id="doc2",
                score=0.7,
                metadata={"type": "test"},
                content="Document 2 content",
            ),
        ]
        
        # Convert to reranking candidates
        candidates = [
            RerankingCandidate(
                id=result.id,
                content=result.content,
                original_score=result.score,
                metadata=result.metadata,
            )
            for result in search_results
        ]
        
        # Test that candidates are valid
        for candidate in candidates:
            assert isinstance(candidate.id, str)
            assert isinstance(candidate.content, str)
            assert isinstance(candidate.original_score, float)
            assert isinstance(candidate.metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])