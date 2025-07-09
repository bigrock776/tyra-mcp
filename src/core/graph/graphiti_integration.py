"""
Graphiti Integration for Temporal Knowledge Graphs with Memgraph.

Advanced temporal knowledge graph management using Graphiti framework
with local LLM integration and sophisticated entity relationship tracking.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

try:
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.embedder.local import LocalEmbedder
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.utils.maintenance.graph_data_operations import clear_data
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False

from ...interfaces.graph_engine import GraphEngine, GraphEngineError
from ...utils.circuit_breaker import CircuitBreaker
from ...utils.logger import get_logger

logger = get_logger(__name__)


class GraphitiManager:
    """
    Graphiti Manager for temporal knowledge graph operations.
    
    Provides high-level temporal knowledge graph operations using Graphiti
    framework with local LLM integration and Memgraph backend.
    """
    
    def __init__(self):
        self.graphiti_client: Optional[Graphiti] = None
        self.config: Dict[str, Any] = {}
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=300,  # 5 minutes
            expected_exception=GraphEngineError
        )
        
        # Performance tracking
        self._total_episodes: int = 0
        self._total_searches: int = 0
        self._total_entities: int = 0
        self._avg_response_time: float = 0.0
        self._error_count: int = 0
        
        # Circuit breaker state
        self._failure_count: int = 0
        self._last_failure_time: Optional[datetime] = None
        self._circuit_breaker_timeout: int = 300  # 5 minutes
        
        # Temporal configuration
        self.temporal_config: Dict[str, Any] = {}
        self.episode_config: Dict[str, Any] = {}
        self.graph_optimization: Dict[str, Any] = {}
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize Graphiti client with local LLM configuration."""
        if not GRAPHITI_AVAILABLE:
            raise GraphEngineError(
                "Graphiti is not available. Please install graphiti-core package."
            )
        
        try:
            self.config = config
            
            # Load temporal configuration
            self.temporal_config = config.get("temporal_features", {})
            self.episode_config = config.get("episode_config", {})
            self.graph_optimization = config.get("graph_optimization", {})
            
            # Update circuit breaker settings
            circuit_config = config.get("circuit_breaker", {})
            self._circuit_breaker_timeout = circuit_config.get("recovery_timeout", 300)
            
            # Configure local LLM client
            llm_config = LLMConfig(
                base_url=config.get("llm_base_url", "http://localhost:8000/v1"),
                model_name=config.get("llm_model", "meta-llama/Llama-3.1-70B-Instruct"),
                api_key=config.get("llm_api_key", "dummy-key"),  # vLLM doesn't need real key
                temperature=config.get("llm_temperature", 0.1),
                max_tokens=config.get("llm_max_tokens", 1000),
            )
            
            llm_client = OpenAIClient(llm_config)
            
            # Configure local embedder
            embedder_config = {
                "model_name": config.get("embedding_model", "intfloat/e5-large-v2"),
                "device": config.get("embedding_device", "auto"),
                "batch_size": config.get("embedding_batch_size", 32),
                "normalize_embeddings": config.get("normalize_embeddings", True),
            }
            
            local_embedder = LocalEmbedder(embedder_config)
            
            # Configure reranker (optional)
            reranker = None
            if config.get("use_reranker", False):
                reranker_config = LLMConfig(
                    base_url=config.get("reranker_base_url", "http://localhost:8000/v1"),
                    model_name=config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
                    api_key=config.get("reranker_api_key", "dummy-key"),
                )
                reranker = OpenAIRerankerClient(reranker_config)
            
            # Initialize Graphiti client
            self.graphiti_client = Graphiti(
                llm_client=llm_client,
                embedder=local_embedder,
                reranker=reranker,
                graph_config={
                    "uri": config.get("memgraph_uri", "bolt://localhost:7687"),
                    "username": config.get("memgraph_username", ""),
                    "password": config.get("memgraph_password", ""),
                    "database": config.get("memgraph_database", ""),
                }
            )
            
            # Initialize Graphiti
            await self.graphiti_client.build_indices_async()
            
            logger.info(
                "Graphiti manager initialized",
                llm_model=config.get("llm_model"),
                embedding_model=config.get("embedding_model"),
                memgraph_uri=config.get("memgraph_uri"),
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti manager: {e}")
            raise GraphEngineError(f"Graphiti initialization failed: {e}")
    
    @CircuitBreaker(failure_threshold=5, recovery_timeout=300)
    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        episode_type: str = "memory"
    ) -> Dict[str, Any]:
        """
        Add an episode to the temporal knowledge graph.
        
        Args:
            episode_id: Unique identifier for the episode
            content: Text content of the episode
            source: Source of the episode (e.g., "user", "system", "agent")
            timestamp: When the episode occurred
            metadata: Additional metadata for the episode
            episode_type: Type of episode (memory, event, fact, etc.)
            
        Returns:
            Dict containing episode information and extracted entities
        """
        start_time = time.time()
        
        try:
            if not self.graphiti_client:
                raise GraphEngineError("Graphiti client not initialized")
            
            # Apply episode configuration limits
            max_length = self.episode_config.get("max_episode_length", 10000)
            if len(content) > max_length:
                content = content[:max_length]
                logger.warning(f"Episode content truncated to {max_length} characters")
            
            # Use configured default episode type if not specified
            if episode_type == "memory":
                episode_type = self.episode_config.get("default_episode_type", "memory")
            
            # Convert episode type to Graphiti EpisodeType
            episode_type_enum = EpisodeType.MEMORY  # Default
            if episode_type.lower() == "event":
                episode_type_enum = EpisodeType.EVENT
            elif episode_type.lower() == "fact":
                episode_type_enum = EpisodeType.FACT
            
            # Apply temporal configuration
            reference_time = timestamp or datetime.utcnow()
            
            # Add episode to Graphiti with temporal features
            result = await self.graphiti_client.add_episode(
                name=episode_id,
                episode_body=content,
                source=source,
                reference_time=reference_time,
                episode_type=episode_type_enum
            )
            
            # Apply temporal validity if configured
            if self.temporal_config.get("enabled", True) and self.temporal_config.get("auto_expire_facts", True):
                validity_period = self.temporal_config.get("default_validity_period", 8760)  # hours
                expiry_time = reference_time + timedelta(hours=validity_period)
                
                # This would be implemented if Graphiti supports explicit fact expiry
                # For now, we'll track it in metadata
                if not metadata:
                    metadata = {}
                metadata["temporal_validity"] = {
                    "valid_from": reference_time.isoformat(),
                    "valid_to": expiry_time.isoformat(),
                    "auto_expire": True
                }
            
            # Track performance
            self._total_episodes += 1
            response_time = time.time() - start_time
            self._avg_response_time = (
                self._avg_response_time * (self._total_episodes - 1) + response_time
            ) / self._total_episodes
            
            logger.debug(
                "Episode added to Graphiti",
                episode_id=episode_id,
                content_length=len(content),
                source=source,
                response_time=response_time,
            )
            
            return {
                "episode_id": episode_id,
                "content": content,
                "source": source,
                "timestamp": timestamp or datetime.utcnow(),
                "metadata": metadata or {},
                "entities_extracted": result.get("entities_extracted", []),
                "relationships_created": result.get("relationships_created", []),
                "processing_time": response_time,
            }
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to add episode to Graphiti: {e}")
            raise GraphEngineError(f"Episode addition failed: {e}")
    
    @CircuitBreaker(failure_threshold=5, recovery_timeout=300)
    async def search(
        self,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the temporal knowledge graph.
        
        Args:
            query: Search query
            center_node_distance: Distance from center nodes to search
            use_hybrid_search: Whether to use hybrid search
            limit: Maximum number of results
            
        Returns:
            List of search results with temporal context
        """
        start_time = time.time()
        
        try:
            if not self.graphiti_client:
                raise GraphEngineError("Graphiti client not initialized")
            
            # Apply graph optimization settings
            max_results = self.graph_optimization.get("max_search_results", 100)
            search_limit = min(limit, max_results)
            
            # Apply search timeout
            search_timeout = self.graph_optimization.get("search_timeout", 30)
            
            # Perform search with timeout
            try:
                results = await asyncio.wait_for(
                    self.graphiti_client.search(
                        query=query,
                        center_node_distance=center_node_distance,
                        use_hybrid_search=use_hybrid_search
                    ),
                    timeout=search_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Search timed out after {search_timeout} seconds")
                return []
            
            # Process and limit results
            processed_results = []
            for i, result in enumerate(results[:search_limit]):
                # Apply temporal filtering if enabled
                result_data = {
                    "id": result.get("uuid", f"result_{i}"),
                    "content": result.get("fact", ""),
                    "score": result.get("score", 0.0),
                    "source": result.get("source", ""),
                    "valid_from": result.get("valid_at"),
                    "valid_to": result.get("invalid_at"),
                    "source_node_uuid": result.get("source_node_uuid"),
                    "metadata": result.get("metadata", {}),
                }
                
                # Apply temporal validity filtering if configured
                if self.temporal_config.get("enabled", True):
                    current_time = datetime.utcnow()
                    valid_from = result.get("valid_at")
                    valid_to = result.get("invalid_at")
                    
                    # Check if result is temporally valid
                    if valid_from and isinstance(valid_from, datetime) and valid_from > current_time:
                        continue  # Skip future facts
                    if valid_to and isinstance(valid_to, datetime) and valid_to < current_time:
                        continue  # Skip expired facts
                
                processed_results.append(result_data)
            
            # Track performance
            self._total_searches += 1
            response_time = time.time() - start_time
            
            logger.debug(
                "Graphiti search completed",
                query=query[:50] + "..." if len(query) > 50 else query,
                results_count=len(processed_results),
                response_time=response_time,
            )
            
            return processed_results
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to search Graphiti: {e}")
            raise GraphEngineError(f"Search failed: {e}")
    
    @CircuitBreaker(failure_threshold=5, recovery_timeout=300)
    async def get_related_entities(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get entities related to a given entity.
        
        Args:
            entity_name: Name of the entity
            relationship_types: Types of relationships to include
            depth: Depth of relationship traversal
            limit: Maximum number of related entities
            
        Returns:
            Dict containing related entities and relationships
        """
        start_time = time.time()
        
        try:
            if not self.graphiti_client:
                raise GraphEngineError("Graphiti client not initialized")
            
            # Get related entities from Graphiti
            related_data = await self.graphiti_client.get_related_entities(
                entity_name=entity_name,
                relationship_types=relationship_types,
                depth=depth
            )
            
            # Process results
            entities = []
            relationships = []
            
            for entity_data in related_data.get("entities", [])[:limit]:
                entities.append({
                    "id": entity_data.get("uuid", str(uuid.uuid4())),
                    "name": entity_data.get("name", ""),
                    "type": entity_data.get("type", "unknown"),
                    "properties": entity_data.get("properties", {}),
                    "confidence": entity_data.get("confidence", 1.0),
                    "created_at": entity_data.get("created_at"),
                    "last_seen": entity_data.get("last_seen"),
                })
            
            for rel_data in related_data.get("relationships", []):
                relationships.append({
                    "id": rel_data.get("uuid", str(uuid.uuid4())),
                    "source_id": rel_data.get("source_id"),
                    "target_id": rel_data.get("target_id"),
                    "type": rel_data.get("type", "related_to"),
                    "properties": rel_data.get("properties", {}),
                    "confidence": rel_data.get("confidence", 1.0),
                    "created_at": rel_data.get("created_at"),
                    "valid_from": rel_data.get("valid_from"),
                    "valid_to": rel_data.get("valid_to"),
                })
            
            response_time = time.time() - start_time
            
            logger.debug(
                "Related entities retrieved",
                entity_name=entity_name,
                entities_count=len(entities),
                relationships_count=len(relationships),
                response_time=response_time,
            )
            
            return {
                "entity_name": entity_name,
                "entities": entities,
                "relationships": relationships,
                "depth": depth,
                "processing_time": response_time,
            }
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to get related entities: {e}")
            raise GraphEngineError(f"Related entities retrieval failed: {e}")
    
    @CircuitBreaker(failure_threshold=5, recovery_timeout=300)
    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get temporal timeline for an entity.
        
        Args:
            entity_name: Name of the entity
            start_date: Start of time range
            end_date: End of time range
            limit: Maximum number of timeline entries
            
        Returns:
            List of timeline entries with temporal context
        """
        start_time = time.time()
        
        try:
            if not self.graphiti_client:
                raise GraphEngineError("Graphiti client not initialized")
            
            # Get timeline from Graphiti
            timeline_data = await self.graphiti_client.get_entity_timeline(
                entity_name=entity_name,
                start_date=start_date,
                end_date=end_date
            )
            
            # Process timeline entries
            timeline_entries = []
            for entry in timeline_data[:limit]:
                timeline_entries.append({
                    "id": entry.get("uuid", str(uuid.uuid4())),
                    "entity_name": entity_name,
                    "content": entry.get("fact", entry.get("content", "")),
                    "timestamp": entry.get("valid_at", entry.get("created_at")),
                    "end_timestamp": entry.get("invalid_at"),
                    "source": entry.get("source", ""),
                    "episode_id": entry.get("episode_id"),
                    "confidence": entry.get("confidence", 1.0),
                    "metadata": entry.get("metadata", {}),
                })
            
            response_time = time.time() - start_time
            
            logger.debug(
                "Entity timeline retrieved",
                entity_name=entity_name,
                entries_count=len(timeline_entries),
                date_range=f"{start_date} to {end_date}",
                response_time=response_time,
            )
            
            return timeline_entries
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to get entity timeline: {e}")
            raise GraphEngineError(f"Entity timeline retrieval failed: {e}")
    
    @CircuitBreaker(failure_threshold=5, recovery_timeout=300)
    async def validate_facts(
        self,
        entity_name: str,
        new_facts: List[str],
        reference_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Validate facts against existing knowledge.
        
        Args:
            entity_name: Name of the entity
            new_facts: List of facts to validate
            reference_time: Reference time for validation
            
        Returns:
            Dict containing validation results
        """
        start_time = time.time()
        
        try:
            if not self.graphiti_client:
                raise GraphEngineError("Graphiti client not initialized")
            
            # Validate facts using Graphiti
            validation_results = await self.graphiti_client.validate_facts(
                entity_name=entity_name,
                new_facts=new_facts,
                reference_time=reference_time or datetime.utcnow()
            )
            
            # Process validation results
            validated_facts = []
            for i, fact in enumerate(new_facts):
                validation_result = validation_results.get(f"fact_{i}", {})
                validated_facts.append({
                    "fact": fact,
                    "is_valid": validation_result.get("is_valid", True),
                    "confidence": validation_result.get("confidence", 1.0),
                    "contradictions": validation_result.get("contradictions", []),
                    "supporting_evidence": validation_result.get("supporting_evidence", []),
                    "validation_reason": validation_result.get("reason", ""),
                })
            
            response_time = time.time() - start_time
            
            logger.debug(
                "Facts validated",
                entity_name=entity_name,
                facts_count=len(new_facts),
                valid_facts=sum(1 for f in validated_facts if f["is_valid"]),
                response_time=response_time,
            )
            
            return {
                "entity_name": entity_name,
                "validated_facts": validated_facts,
                "overall_validity": all(f["is_valid"] for f in validated_facts),
                "processing_time": response_time,
            }
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to validate facts: {e}")
            raise GraphEngineError(f"Fact validation failed: {e}")
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using Graphiti's NLP capabilities.
        
        Args:
            text: Text to extract entities from
            entity_types: Types of entities to extract
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of extracted entities
        """
        try:
            if not self.graphiti_client:
                raise GraphEngineError("Graphiti client not initialized")
            
            # Create a temporary episode to extract entities
            temp_episode_id = f"temp_extract_{int(time.time())}"
            
            # Add temporary episode
            episode_result = await self.add_episode(
                episode_id=temp_episode_id,
                content=text,
                source="entity_extraction",
                episode_type="fact"
            )
            
            # Extract entities from the episode result
            entities = []
            for entity_data in episode_result.get("entities_extracted", []):
                if entity_data.get("confidence", 0.0) >= min_confidence:
                    if not entity_types or entity_data.get("type") in entity_types:
                        entities.append({
                            "id": entity_data.get("id", str(uuid.uuid4())),
                            "name": entity_data.get("name", ""),
                            "type": entity_data.get("type", "unknown"),
                            "properties": entity_data.get("properties", {}),
                            "confidence": entity_data.get("confidence", 1.0),
                            "first_seen": datetime.utcnow(),
                            "last_seen": datetime.utcnow(),
                            "occurrence_count": 1,
                        })
            
            # Clean up temporary episode (optional)
            # await self.remove_episode(temp_episode_id)
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            raise GraphEngineError(f"Entity extraction failed: {e}")
    
    async def add_memory_to_graph(self, memory_id: str) -> Dict[str, Any]:
        """
        Add a memory from the memory store to the knowledge graph.
        
        Args:
            memory_id: ID of the memory to add
            
        Returns:
            Dict containing results of the addition
        """
        try:
            # This would typically fetch the memory from the memory store
            # For now, we'll return a placeholder
            return {
                "memory_id": memory_id,
                "entities_added": 0,
                "relationships_added": 0,
                "message": "Memory addition to graph not fully implemented yet",
            }
            
        except Exception as e:
            logger.error(f"Failed to add memory to graph: {e}")
            raise GraphEngineError(f"Memory addition failed: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        try:
            if not self.graphiti_client:
                raise GraphEngineError("Graphiti client not initialized")
            
            # Get statistics from Graphiti
            stats = await self.graphiti_client.get_graph_statistics()
            
            return {
                "total_entities": stats.get("total_entities", 0),
                "total_relationships": stats.get("total_relationships", 0),
                "total_episodes": self._total_episodes,
                "total_searches": self._total_searches,
                "entity_types": stats.get("entity_types", {}),
                "relationship_types": stats.get("relationship_types", {}),
                "avg_relationships_per_entity": stats.get("avg_relationships_per_entity", 0.0),
                "most_connected_entities": stats.get("most_connected_entities", []),
                "temporal_range": stats.get("temporal_range", {}),
                "performance": {
                    "avg_response_time": self._avg_response_time,
                    "error_count": self._error_count,
                    "error_rate": self._error_count / max(self._total_searches + self._total_episodes, 1),
                },
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "error": str(e),
                "total_episodes": self._total_episodes,
                "total_searches": self._total_searches,
                "error_count": self._error_count,
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self.graphiti_client:
                return {
                    "status": "unhealthy",
                    "error": "Graphiti client not initialized",
                }
            
            # Test basic operations
            test_start = time.time()
            
            # Simple search test
            test_results = await self.search("test", limit=1)
            response_time = time.time() - test_start
            
            # Get basic stats
            stats = await self.get_statistics()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "graph_stats": {
                    "entities": stats.get("total_entities", 0),
                    "relationships": stats.get("total_relationships", 0),
                    "episodes": stats.get("total_episodes", 0),
                },
                "performance": {
                    "avg_response_time": self._avg_response_time,
                    "error_rate": self._error_count / max(self._total_searches + self._total_episodes, 1),
                },
                "circuit_breaker": {
                    "is_open": self._is_circuit_breaker_open(),
                    "failure_count": self._failure_count,
                    "last_failure": self._last_failure_time,
                },
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_count": self._error_count,
            }
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._failure_count >= 5:
            if self._last_failure_time:
                time_since_failure = datetime.utcnow() - self._last_failure_time
                return time_since_failure.total_seconds() < self._circuit_breaker_timeout
        return False
    
    async def clear_graph(self) -> None:
        """Clear all data from the knowledge graph."""
        try:
            if not self.graphiti_client:
                raise GraphEngineError("Graphiti client not initialized")
            
            # Clear data using Graphiti
            await clear_data(self.graphiti_client)
            
            # Reset counters
            self._total_episodes = 0
            self._total_searches = 0
            self._total_entities = 0
            self._error_count = 0
            self._failure_count = 0
            self._last_failure_time = None
            
            logger.info("Graphiti knowledge graph cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            raise GraphEngineError(f"Graph clearing failed: {e}")
    
    async def close(self) -> None:
        """Close Graphiti client and clean up resources."""
        try:
            if self.graphiti_client:
                await self.graphiti_client.close()
                self.graphiti_client = None
            
            logger.info(
                "Graphiti manager closed",
                total_episodes=self._total_episodes,
                total_searches=self._total_searches,
                error_count=self._error_count,
            )
            
        except Exception as e:
            logger.error(f"Failed to close Graphiti manager: {e}")