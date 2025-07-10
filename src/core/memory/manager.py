"""
Central memory manager orchestrating all memory operations.

High-level memory management interface that coordinates between embedding providers,
vector stores, graph engines, and rerankers to provide unified memory operations.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..interfaces.embeddings import EmbeddingProvider
from ..interfaces.graph_engine import Entity, GraphEngine, Relationship
from ..interfaces.reranker import Reranker, RerankingCandidate, RerankingResult
from ..interfaces.vector_store import VectorDocument, VectorSearchResult, VectorStore
from ..utils.config import get_settings
from ..utils.logger import get_logger
from ..utils.registry import ProviderType, get_provider, get_provider_with_fallback

logger = get_logger(__name__)


@dataclass
class MemorySearchRequest:
    """Request for memory search operations."""

    query: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = 10
    rerank_top_k: int = 5
    min_confidence: float = 0.0
    include_graph: bool = True
    search_type: str = "hybrid"  # "vector", "graph", "hybrid"
    filters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MemorySearchResult:
    """Result from memory search operations."""

    id: str
    content: str
    score: float
    confidence: float
    metadata: Dict[str, Any]
    source_type: str  # "vector", "graph", "hybrid"
    embedding: Optional[np.ndarray] = None
    entities: Optional[List[Entity]] = None
    relationships: Optional[List[Relationship]] = None
    rerank_explanation: Optional[str] = None


@dataclass
class MemoryStoreRequest:
    """Request for storing memories."""

    content: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    extract_entities: bool = True
    create_relationships: bool = True
    chunk_content: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class MemoryStoreResult:
    """Result from memory storage operations."""

    memory_id: str
    chunk_ids: List[str]
    entities_created: List[str]
    relationships_created: List[str]
    embedding_time: float
    storage_time: float
    graph_time: float
    total_time: float


class MemoryManager:
    """
    Central memory manager coordinating all memory operations.

    Features:
    - Unified interface for all memory operations
    - Automatic provider selection and fallbacks
    - Multi-modal search (vector + graph)
    - Advanced reranking and confidence scoring
    - Entity extraction and relationship mapping
    - Performance monitoring and optimization
    - Agent-aware memory isolation
    """

    def __init__(self):
        self.settings = get_settings()

        # Providers (loaded on-demand)
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.vector_store: Optional[VectorStore] = None
        self.graph_engine: Optional[GraphEngine] = None
        self.reranker: Optional[Reranker] = None

        # Performance tracking
        self._total_searches: int = 0
        self._total_stores: int = 0
        self._avg_search_time: float = 0.0
        self._avg_store_time: float = 0.0

        # Cache for provider instances
        self._provider_cache: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the memory manager and load providers."""
        try:
            # Load core providers
            await self._load_providers()

            # Initialize providers
            await self._initialize_providers()

            logger.info(
                "Memory manager initialized",
                embedding_provider=self.settings.embeddings.primary.get("provider"),
                vector_store=self.settings.memory.backend,
                graph_engine=self.settings.graph.backend,
            )

        except Exception as e:
            logger.error("Failed to initialize memory manager", error=str(e))
            raise

    async def _load_providers(self) -> None:
        """Load all required providers."""
        try:
            # Load embedding provider with fallback
            primary_embedding = self.settings.embeddings.primary.get("provider")
            fallback_embedding = self.settings.embeddings.fallback.get("provider")

            self.embedding_provider = await get_provider_with_fallback(
                ProviderType.EMBEDDING, primary_embedding
            )

            # Load vector store
            vector_backend = self.settings.memory.vector.get("backend", "pgvector")
            self.vector_store = await get_provider(
                ProviderType.VECTOR_STORE, vector_backend
            )

            # Load graph engine
            graph_backend = self.settings.graph.backend
            self.graph_engine = await get_provider(
                ProviderType.GRAPH_ENGINE, graph_backend
            )

            # Load reranker (optional)
            try:
                reranker_provider = self.settings.rag.reranking.get(
                    "provider", "cross_encoder"
                )
                self.reranker = await get_provider(
                    ProviderType.RERANKER, reranker_provider
                )
            except Exception as e:
                logger.warning(
                    "Reranker not available, continuing without", error=str(e)
                )
                self.reranker = None

        except Exception as e:
            logger.error("Failed to load providers", error=str(e))
            raise

    async def _initialize_providers(self) -> None:
        """Initialize all loaded providers."""
        # Providers are initialized when loaded by the registry
        # This method can be used for additional setup if needed
        pass

    async def store_memory(self, request: MemoryStoreRequest) -> MemoryStoreResult:
        """Store a memory with full processing pipeline."""
        start_time = time.time()

        try:
            memory_id = str(uuid.uuid4())

            # Chunk content if requested
            chunks = []
            if request.chunk_content and len(request.content) > request.chunk_size:
                chunks = await self._chunk_content(
                    request.content, request.chunk_size, request.chunk_overlap
                )
            else:
                chunks = [request.content]

            # Prepare metadata
            base_metadata = {
                "agent_id": request.agent_id,
                "session_id": request.session_id,
                "stored_at": datetime.utcnow().isoformat(),
                "memory_id": memory_id,
                "chunk_count": len(chunks),
            }

            if request.metadata:
                base_metadata.update(request.metadata)

            # OPTIMIZATION: Run embedding generation and entity extraction in parallel
            embedding_start = time.time()
            
            # Create parallel tasks
            embedding_task = asyncio.create_task(
                self.embedding_provider.embed_texts(chunks)
            )
            
            entity_task = None
            if request.extract_entities and self.graph_engine:
                entity_task = asyncio.create_task(
                    self._extract_entities_only(request.content, memory_id, base_metadata)
                )
            
            # Wait for both to complete
            embeddings = await embedding_task
            embedding_time = time.time() - embedding_start
            
            entities_data = None
            if entity_task:
                entities_data = await entity_task

            # Store in vector database
            storage_start = time.time()
            chunk_ids = []
            documents = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{memory_id}_chunk_{i}"
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_id": chunk_id,
                        "parent_memory_id": memory_id,
                    }
                )

                document = VectorDocument(
                    id=chunk_id,
                    content=chunk,
                    embedding=embedding,
                    metadata=chunk_metadata,
                )
                documents.append(document)
                chunk_ids.append(chunk_id)

            # OPTIMIZATION: Parallel database operations
            vector_store_task = asyncio.create_task(
                self.vector_store.store_documents(documents)
            )
            
            graph_store_task = None
            entities_created = []
            relationships_created = []
            
            if entities_data and self.graph_engine:
                entities, relationships = entities_data
                graph_store_task = asyncio.create_task(
                    self._store_graph_data(entities, relationships)
                )
                entities_created = [e.id for e in entities]
                relationships_created = [r.id for r in relationships]
            
            # Wait for storage operations
            await vector_store_task
            if graph_store_task:
                await graph_store_task
                
            storage_time = time.time() - storage_start

            graph_time = time.time() - start_time

            # Update performance tracking
            total_time = time.time() - start_time
            self._total_stores += 1
            self._avg_store_time = (
                self._avg_store_time * (self._total_stores - 1) + total_time
            ) / self._total_stores

            logger.info(
                "Memory stored successfully",
                memory_id=memory_id,
                chunks=len(chunks),
                entities=len(entities_created),
                relationships=len(relationships_created),
                total_time=total_time,
            )

            return MemoryStoreResult(
                memory_id=memory_id,
                chunk_ids=chunk_ids,
                entities_created=entities_created,
                relationships_created=relationships_created,
                embedding_time=embedding_time,
                storage_time=storage_time,
                graph_time=graph_time,
                total_time=total_time,
            )

        except Exception as e:
            logger.error(
                "Failed to store memory",
                content_length=len(request.content),
                error=str(e),
            )
            raise

    async def search_memory(
        self, request: MemorySearchRequest
    ) -> List[MemorySearchResult]:
        """Search memories using multi-modal approach."""
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = await self.embedding_provider.embed_query(request.query)

            results = []

            if request.search_type in ["vector", "hybrid"]:
                # Vector search
                vector_results = await self._vector_search(query_embedding, request)
                results.extend(vector_results)

            if request.search_type in ["graph", "hybrid"] and self.graph_engine:
                # Graph search
                graph_results = await self._graph_search(request.query, request)
                results.extend(graph_results)

            # Remove duplicates and merge results
            results = await self._merge_and_deduplicate_results(results)

            # Rerank if reranker is available
            if self.reranker and len(results) > 1:
                results = await self._rerank_results(
                    request.query, results, request.rerank_top_k
                )

            # Apply confidence filtering
            if request.min_confidence > 0:
                results = [r for r in results if r.confidence >= request.min_confidence]

            # Apply top_k limit
            results = results[: request.top_k]

            # Update performance tracking
            search_time = time.time() - start_time
            self._total_searches += 1
            self._avg_search_time = (
                self._avg_search_time * (self._total_searches - 1) + search_time
            ) / self._total_searches

            logger.debug(
                "Memory search completed",
                query=(
                    request.query[:100] + "..."
                    if len(request.query) > 100
                    else request.query
                ),
                results=len(results),
                search_type=request.search_type,
                time=search_time,
            )

            return results

        except Exception as e:
            logger.error(
                "Memory search failed",
                query=(
                    request.query[:100] + "..."
                    if len(request.query) > 100
                    else request.query
                ),
                search_type=request.search_type,
                error=str(e),
            )
            raise

    async def _vector_search(
        self, query_embedding: np.ndarray, request: MemorySearchRequest
    ) -> List[MemorySearchResult]:
        """Perform vector similarity search."""
        try:
            # Prepare filters
            filters = request.filters.copy() if request.filters else {}

            if request.agent_id:
                filters["agent_id"] = request.agent_id
            if request.session_id:
                filters["session_id"] = request.session_id

            # Perform search
            if (
                hasattr(self.vector_store, "hybrid_search")
                and request.search_type == "hybrid"
            ):
                vector_results = await self.vector_store.hybrid_search(
                    query_embedding=query_embedding,
                    text_query=request.query,
                    top_k=request.top_k * 2,  # Get more for reranking
                    vector_weight=self.settings.rag.retrieval.get("hybrid_weight", 0.7),
                    filters=filters,
                )
            else:
                vector_results = await self.vector_store.search_similar(
                    query_embedding=query_embedding,
                    top_k=request.top_k * 2,
                    filters=filters,
                    min_score=request.min_confidence,
                )

            # Convert to MemorySearchResult
            results = []
            for result in vector_results:
                memory_result = MemorySearchResult(
                    id=result.id,
                    content=result.content,
                    score=result.score,
                    confidence=result.score,  # Use similarity as confidence
                    metadata=result.metadata,
                    source_type="vector",
                    embedding=result.embedding,
                )
                results.append(memory_result)

            return results

        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            raise

    async def _graph_search(
        self, query: str, request: MemorySearchRequest
    ) -> List[MemorySearchResult]:
        """Perform graph-based search."""
        try:
            # Extract entities from query (simplified approach)
            query_entities = await self._extract_query_entities(query)

            results = []

            # OPTIMIZATION: Batch entity searches instead of sequential
            entity_search_tasks = [
                self.graph_engine.find_entities(
                    properties={"name": entity_name}, limit=10
                )
                for entity_name in query_entities
            ]
            
            if entity_search_tasks:
                entity_search_results = await asyncio.gather(*entity_search_tasks, return_exceptions=True)
            else:
                entity_search_results = []

            # Process results
            for entities_result in entity_search_results:
                if isinstance(entities_result, Exception):
                    logger.error("Entity search failed", error=str(entities_result))
                    continue
                    
                entities = entities_result if entities_result else []
                
                # OPTIMIZATION: Batch connected entity and relationship queries
                entity_data_tasks = []
                for entity in entities:
                    entity_data_tasks.extend([
                        self.graph_engine.get_connected_entities(entity.id, max_depth=2),
                        self.graph_engine.get_entity_relationships(entity.id)
                    ])
                
                if entity_data_tasks:
                    entity_data_results = await asyncio.gather(*entity_data_tasks, return_exceptions=True)
                    
                    # Process entity data in pairs (connected, relationships)
                    for i, entity in enumerate(entities):
                        connected_idx = i * 2
                        relationships_idx = i * 2 + 1
                        
                        connected = []
                        relationships = []
                        
                        if (connected_idx < len(entity_data_results) and 
                            not isinstance(entity_data_results[connected_idx], Exception)):
                            connected = entity_data_results[connected_idx] or []
                            
                        if (relationships_idx < len(entity_data_results) and 
                            not isinstance(entity_data_results[relationships_idx], Exception)):
                            relationships = entity_data_results[relationships_idx] or []

                        # Create result from entity information
                        content = f"Entity: {entity.name} ({entity.entity_type})"
                        if entity.properties:
                            content += f" Properties: {entity.properties}"

                        memory_result = MemorySearchResult(
                            id=entity.id,
                            content=content,
                            score=entity.confidence or 0.8,
                            confidence=entity.confidence or 0.8,
                            metadata=entity.properties or {},
                            source_type="graph",
                            entities=[entity] + connected,
                            relationships=relationships,
                        )
                        results.append(memory_result)

            return results

        except Exception as e:
            logger.error("Graph search failed", error=str(e))
            return []  # Return empty list instead of raising for graceful degradation

    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entity names from query (simplified)."""
        # This is a simplified implementation
        # In production, you'd use NER models or more sophisticated extraction

        # For now, extract capitalized words as potential entities
        import re

        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)

        # Also look for quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        return list(set(entities))  # Remove duplicates

    async def _extract_entities_only(
        self, content: str, memory_id: str, metadata: Dict[str, Any]
    ) -> tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from content without storing."""
        try:
            # Simplified entity extraction
            # In production, use NER models like spaCy, BERT-NER, etc.

            entities = []
            relationships = []

            # Extract potential entities (simplified approach)
            import re

            # Extract proper nouns
            proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)

            # Extract numbers that might be values
            numbers = re.findall(r"\$?\d+(?:,\d{3})*(?:\.\d{2})?[BMK]?", content)

            # Create entities
            for i, noun in enumerate(set(proper_nouns)):
                entity_id = f"{memory_id}_entity_{i}"
                entity = Entity(
                    id=entity_id,
                    name=noun,
                    entity_type="PERSON" if noun.istitle() else "ORGANIZATION",
                    properties={
                        "source_memory": memory_id,
                        "extracted_from": content[:100] + "...",
                        "confidence": 0.7,  # Base confidence for simple extraction
                    },
                    confidence=0.7,
                )
                entities.append(entity)

            # Create some basic relationships
            for i in range(len(entities) - 1):
                rel_id = f"{memory_id}_rel_{i}"
                relationship = Relationship(
                    id=rel_id,
                    source_entity_id=entities[i].id,
                    target_entity_id=entities[i + 1].id,
                    relationship_type="MENTIONED_WITH",
                    properties={
                        "source_memory": memory_id,
                        "context": content[:200] + "...",
                    },
                    confidence=0.6,
                )
                relationships.append(relationship)

            return entities, relationships

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return [], []
    
    async def _store_graph_data(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> None:
        """Store entities and relationships in parallel."""
        try:
            # OPTIMIZATION: Store entities and relationships in parallel
            tasks = []
            
            if entities:
                tasks.append(self.graph_engine.create_entities(entities))
            
            if relationships:
                tasks.append(self.graph_engine.create_relationships(relationships))
            
            if tasks:
                await asyncio.gather(*tasks)
                
        except Exception as e:
            logger.error("Graph data storage failed", error=str(e))
            raise

    async def _extract_and_store_entities(
        self, content: str, memory_id: str, metadata: Dict[str, Any]
    ) -> tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from content."""
        try:
            entities, relationships = await self._extract_entities_only(
                content, memory_id, metadata
            )
            
            # Store entities and relationships
            await self._store_graph_data(entities, relationships)
            
            return entities, relationships

        except Exception as e:
            logger.error("Entity extraction and storage failed", error=str(e))
            return [], []

    async def _merge_and_deduplicate_results(
        self, results: List[MemorySearchResult]
    ) -> List[MemorySearchResult]:
        """Merge and deduplicate search results from different sources."""
        try:
            # Group by content similarity (simplified)
            seen_content = set()
            unique_results = []

            for result in results:
                # Simple deduplication by content hash
                content_hash = hash(result.content[:200])  # First 200 chars

                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)
                else:
                    # Merge with existing result (update score if higher)
                    for existing in unique_results:
                        if hash(existing.content[:200]) == content_hash:
                            if result.score > existing.score:
                                existing.score = result.score
                                existing.confidence = max(
                                    existing.confidence, result.confidence
                                )
                            break

            # Sort by score
            unique_results.sort(key=lambda x: x.score, reverse=True)

            return unique_results

        except Exception as e:
            logger.error("Result merging failed", error=str(e))
            return results

    async def _rerank_results(
        self, query: str, results: List[MemorySearchResult], top_k: int
    ) -> List[MemorySearchResult]:
        """Rerank results using the reranker."""
        try:
            if not self.reranker or len(results) <= 1:
                return results

            # Prepare candidates for reranking
            candidates = []
            for result in results:
                candidate = RerankingCandidate(
                    id=result.id,
                    content=result.content,
                    original_score=result.score,
                    metadata=result.metadata,
                )
                candidates.append(candidate)

            # Rerank
            reranked = await self.reranker.rerank(query, candidates, top_k)

            # Convert back to MemorySearchResult
            reranked_results = []
            for rerank_result in reranked:
                # Find original result
                original_result = next(
                    (r for r in results if r.id == rerank_result.id), None
                )

                if original_result:
                    # Update with reranked scores
                    original_result.score = rerank_result.final_score
                    original_result.confidence = rerank_result.final_score
                    original_result.rerank_explanation = rerank_result.explanation
                    reranked_results.append(original_result)

            return reranked_results

        except Exception as e:
            logger.error("Reranking failed", error=str(e))
            return results[:top_k]  # Return original results if reranking fails

    async def _chunk_content(
        self, content: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Split content into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]

            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > start + chunk_size // 2:  # Don't make chunks too small
                    chunk = content[start : start + break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - chunk_overlap

            if start >= len(content):
                break

        return [chunk for chunk in chunks if chunk.strip()]

    async def get_memory(self, memory_id: str) -> Optional[VectorDocument]:
        """Retrieve a specific memory by ID."""
        try:
            return await self.vector_store.get_document(memory_id)
        except Exception as e:
            logger.error("Failed to get memory", memory_id=memory_id, error=str(e))
            return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory and associated entities/relationships."""
        try:
            # Delete from vector store
            vector_deleted = await self.vector_store.delete_document(memory_id)

            # Delete associated entities and relationships from graph
            if self.graph_engine:
                # Find entities associated with this memory
                entities = await self.graph_engine.find_entities(
                    properties={"source_memory": memory_id}
                )

                for entity in entities:
                    await self.graph_engine.delete_entity(entity.id)

            return vector_deleted

        except Exception as e:
            logger.error("Failed to delete memory", memory_id=memory_id, error=str(e))
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory manager statistics."""
        try:
            stats = {
                "performance": {
                    "total_searches": self._total_searches,
                    "total_stores": self._total_stores,
                    "avg_search_time": self._avg_search_time,
                    "avg_store_time": self._avg_store_time,
                },
                "providers": {},
            }

            # Get provider stats
            if self.embedding_provider and hasattr(
                self.embedding_provider, "get_performance_stats"
            ):
                stats["providers"][
                    "embedding"
                ] = self.embedding_provider.get_performance_stats()

            if self.vector_store and hasattr(self.vector_store, "get_stats"):
                stats["providers"]["vector_store"] = await self.vector_store.get_stats()

            if self.graph_engine and hasattr(self.graph_engine, "get_stats"):
                stats["providers"]["graph_engine"] = await self.graph_engine.get_stats()

            if self.reranker and hasattr(self.reranker, "get_performance_stats"):
                stats["providers"]["reranker"] = self.reranker.get_performance_stats()

            return stats

        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health = {"status": "healthy", "providers": {}}

            # Check all providers
            if self.embedding_provider:
                embedding_health = await self.embedding_provider.health_check()
                health["providers"]["embedding"] = embedding_health
                if embedding_health["status"] != "healthy":
                    health["status"] = "degraded"

            if self.vector_store:
                vector_health = await self.vector_store.health_check()
                health["providers"]["vector_store"] = vector_health
                if vector_health["status"] != "healthy":
                    health["status"] = "degraded"

            if self.graph_engine:
                graph_health = await self.graph_engine.health_check()
                health["providers"]["graph_engine"] = graph_health
                if graph_health["status"] != "healthy":
                    health["status"] = "degraded"

            if self.reranker:
                rerank_health = await self.reranker.health_check()
                health["providers"]["reranker"] = rerank_health
                if rerank_health["status"] != "healthy":
                    health["status"] = "degraded"

            return health

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close(self) -> None:
        """Close the memory manager and all providers."""
        try:
            if self.embedding_provider and hasattr(self.embedding_provider, "close"):
                await self.embedding_provider.close()

            if self.vector_store and hasattr(self.vector_store, "close"):
                await self.vector_store.close()

            if self.graph_engine and hasattr(self.graph_engine, "close"):
                await self.graph_engine.close()

            if self.reranker and hasattr(self.reranker, "close"):
                await self.reranker.close()

            logger.info(
                "Memory manager closed",
                total_searches=self._total_searches,
                total_stores=self._total_stores,
            )

        except Exception as e:
            logger.error("Error closing memory manager", error=str(e))
