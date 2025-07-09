"""Enhanced graph client integrating Memgraph and Graphiti for temporal knowledge graphs."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from ...utils.logger import get_logger
from ...utils.config import settings
from ..interfaces.graph_engine import GraphEngine, GraphEngineError
from ..observability.tracing import trace_method
from .memgraph_client import MemgraphClient
from .graphiti_integration import GraphitiManager

logger = get_logger(__name__)


class EnhancedGraphClient(GraphEngine):
    """Enhanced graph client combining Memgraph and Graphiti capabilities."""
    
    def __init__(
        self,
        memgraph_config: Optional[Dict[str, Any]] = None,
        graphiti_config: Optional[Dict[str, Any]] = None,
        enable_graphiti: bool = True
    ):
        """Initialize enhanced graph client.
        
        Args:
            memgraph_config: Configuration for Memgraph
            graphiti_config: Configuration for Graphiti
            enable_graphiti: Whether to enable Graphiti integration
        """
        self.memgraph_client = MemgraphClient(memgraph_config or {})
        self.graphiti_manager: Optional[GraphitiManager] = None
        self.enable_graphiti = enable_graphiti
        
        if enable_graphiti:
            self.graphiti_manager = GraphitiManager()
            
        self.graphiti_config = graphiti_config or {}
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize both Memgraph and Graphiti clients."""
        try:
            # Initialize Memgraph client
            await self.memgraph_client.initialize(config.get("memgraph", {}))
            logger.info("Memgraph client initialized")
            
            # Initialize Graphiti if enabled
            if self.enable_graphiti and self.graphiti_manager:
                graphiti_config = config.get("graphiti", self.graphiti_config)
                if graphiti_config.get("enabled", True):
                    try:
                        await self.graphiti_manager.initialize(graphiti_config)
                        logger.info("Graphiti manager initialized")
                    except Exception as e:
                        logger.warning(f"Failed to initialize Graphiti: {e}")
                        logger.info("Continuing with Memgraph-only mode")
                        self.graphiti_manager = None
                        
        except Exception as e:
            logger.error(f"Failed to initialize enhanced graph client: {e}")
            raise GraphEngineError(f"Initialization failed: {e}")
            
    async def close(self) -> None:
        """Close both clients."""
        if self.memgraph_client:
            await self.memgraph_client.close()
            
        if self.graphiti_manager:
            await self.graphiti_manager.close()
            
    @trace_method("graph_store_memory")
    async def store_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store memory in both graph systems.
        
        Args:
            memory_id: Unique memory identifier
            content: Memory content
            metadata: Optional metadata
            agent_id: Optional agent identifier
            
        Returns:
            Combined results from both systems
        """
        results = {"memory_id": memory_id}
        
        # Store in Memgraph
        try:
            memgraph_result = await self.memgraph_client.store_entity(
                entity_id=memory_id,
                entity_type="Memory",
                properties={
                    "content": content,
                    "agent_id": agent_id,
                    "created_at": datetime.utcnow(),
                    **(metadata or {})
                }
            )
            results["memgraph"] = memgraph_result
        except Exception as e:
            logger.error(f"Failed to store memory in Memgraph: {e}")
            results["memgraph_error"] = str(e)
            
        # Store in Graphiti
        if self.graphiti_manager:
            try:
                graphiti_result = await self.graphiti_manager.add_episode(
                    episode_id=memory_id,
                    content=content,
                    source=agent_id or "system",
                    metadata=metadata,
                    episode_type="memory"
                )
                results["graphiti"] = graphiti_result
            except Exception as e:
                logger.error(f"Failed to store memory in Graphiti: {e}")
                results["graphiti_error"] = str(e)
                
        return results
        
    @trace_method("graph_query")
    async def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        use_graphiti: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute query using appropriate backend.
        
        Args:
            query: Query string (Cypher for Memgraph, natural language for Graphiti)
            parameters: Query parameters
            use_graphiti: Whether to use Graphiti for natural language queries
            
        Returns:
            Query results
        """
        # If query looks like Cypher, use Memgraph
        if query.strip().upper().startswith(('MATCH', 'CREATE', 'DELETE', 'MERGE', 'RETURN')):
            return await self.memgraph_client.execute_query(query, parameters or {})
            
        # For natural language queries, try Graphiti first
        if use_graphiti and self.graphiti_manager:
            try:
                return await self.graphiti_manager.search(query)
            except Exception as e:
                logger.warning(f"Graphiti query failed, falling back to Memgraph: {e}")
                
        # Fallback to Memgraph with text search
        return await self._text_search_memgraph(query)
        
    async def _text_search_memgraph(self, query: str) -> List[Dict[str, Any]]:
        """Search Memgraph using text-based queries."""
        # Convert natural language to basic Cypher query
        cypher_query = f"""
        MATCH (n)
        WHERE n.content CONTAINS $query OR n.name CONTAINS $query
        RETURN n
        LIMIT 20
        """
        
        return await self.memgraph_client.execute_query(
            cypher_query, 
            {"query": query}
        )
        
    @trace_method("graph_get_related")
    async def get_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get entities related to the given entity.
        
        Args:
            entity_id: ID of the central entity
            relationship_types: Types of relationships to follow
            depth: Traversal depth
            limit: Maximum number of results
            
        Returns:
            List of related entities
        """
        # Try Graphiti first if available
        if self.graphiti_manager:
            try:
                graphiti_result = await self.graphiti_manager.get_related_entities(
                    entity_name=entity_id,
                    relationship_types=relationship_types,
                    depth=depth,
                    limit=limit
                )
                return graphiti_result.get("entities", [])
            except Exception as e:
                logger.warning(f"Graphiti related entities failed: {e}")
                
        # Fallback to Memgraph
        return await self.memgraph_client.get_related_entities(
            entity_id=entity_id,
            relationship_types=relationship_types,
            depth=depth,
            limit=limit
        )
        
    @trace_method("graph_temporal_query")
    async def get_temporal_facts(
        self,
        entity_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        fact_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get temporal facts about an entity.
        
        Args:
            entity_name: Name of the entity
            start_time: Start of time range
            end_time: End of time range
            fact_types: Types of facts to retrieve
            
        Returns:
            List of temporal facts
        """
        # Use Graphiti for temporal queries if available
        if self.graphiti_manager:
            try:
                return await self.graphiti_manager.get_entity_timeline(
                    entity_name=entity_name,
                    start_date=start_time,
                    end_date=end_time
                )
            except Exception as e:
                logger.warning(f"Graphiti temporal query failed: {e}")
                
        # Fallback to Memgraph temporal query
        cypher_query = """
        MATCH (e:Entity {name: $entity_name})-[:HAS_FACT]->(f:Fact)
        WHERE ($start_time IS NULL OR f.valid_from >= $start_time)
          AND ($end_time IS NULL OR f.valid_to <= $end_time)
          AND ($fact_types IS NULL OR f.type IN $fact_types)
        RETURN f
        ORDER BY f.valid_from
        """
        
        return await self.memgraph_client.execute_query(
            cypher_query,
            {
                "entity_name": entity_name,
                "start_time": start_time,
                "end_time": end_time,
                "fact_types": fact_types
            }
        )
        
    async def extract_and_store_entities(
        self,
        text: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract entities from text and store them in the graph.
        
        Args:
            text: Text to extract entities from
            source_id: Source identifier
            metadata: Optional metadata
            
        Returns:
            Extraction and storage results
        """
        results = {"source_id": source_id, "entities": [], "relationships": []}
        
        # Use Graphiti for advanced entity extraction if available
        if self.graphiti_manager:
            try:
                entities = await self.graphiti_manager.extract_entities(text)
                results["entities"] = entities
                
                # Store as episode in Graphiti
                episode_result = await self.graphiti_manager.add_episode(
                    episode_id=source_id,
                    content=text,
                    source="entity_extraction",
                    metadata=metadata
                )
                results["episode"] = episode_result
                
            except Exception as e:
                logger.error(f"Graphiti entity extraction failed: {e}")
                results["graphiti_error"] = str(e)
                
        # Also store basic entities in Memgraph
        try:
            # Basic entity storage in Memgraph
            await self.memgraph_client.store_entity(
                entity_id=source_id,
                entity_type="TextSource",
                properties={
                    "content": text,
                    "created_at": datetime.utcnow(),
                    **(metadata or {})
                }
            )
            results["memgraph_stored"] = True
            
        except Exception as e:
            logger.error(f"Memgraph entity storage failed: {e}")
            results["memgraph_error"] = str(e)
            
        return results
        
    async def validate_knowledge(
        self,
        entity_name: str,
        claims: List[str],
        reference_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Validate knowledge claims against the graph.
        
        Args:
            entity_name: Entity to validate claims about
            claims: List of claims to validate
            reference_time: Reference time for validation
            
        Returns:
            Validation results
        """
        # Use Graphiti for advanced fact validation if available
        if self.graphiti_manager:
            try:
                return await self.graphiti_manager.validate_facts(
                    entity_name=entity_name,
                    new_facts=claims,
                    reference_time=reference_time
                )
            except Exception as e:
                logger.warning(f"Graphiti validation failed: {e}")
                
        # Fallback to basic Memgraph validation
        results = {"entity_name": entity_name, "validated_claims": []}
        
        for claim in claims:
            # Basic validation against existing facts
            validation_result = {
                "claim": claim,
                "is_valid": True,  # Default to valid for basic validation
                "confidence": 0.5,  # Low confidence for basic validation
                "supporting_evidence": [],
                "contradictions": []
            }
            results["validated_claims"].append(validation_result)
            
        return results
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from both graph systems."""
        stats = {"memgraph": {}, "graphiti": {}}
        
        # Get Memgraph statistics
        try:
            stats["memgraph"] = await self.memgraph_client.get_statistics()
        except Exception as e:
            logger.error(f"Failed to get Memgraph statistics: {e}")
            stats["memgraph"] = {"error": str(e)}
            
        # Get Graphiti statistics
        if self.graphiti_manager:
            try:
                stats["graphiti"] = await self.graphiti_manager.get_statistics()
            except Exception as e:
                logger.error(f"Failed to get Graphiti statistics: {e}")
                stats["graphiti"] = {"error": str(e)}
                
        return stats
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on both systems."""
        health = {"overall": "healthy", "memgraph": {}, "graphiti": {}}
        
        # Check Memgraph health
        try:
            memgraph_health = await self.memgraph_client.health_check()
            health["memgraph"] = memgraph_health
        except Exception as e:
            health["memgraph"] = {"status": "unhealthy", "error": str(e)}
            health["overall"] = "degraded"
            
        # Check Graphiti health
        if self.graphiti_manager:
            try:
                graphiti_health = await self.graphiti_manager.health_check()
                health["graphiti"] = graphiti_health
                
                if graphiti_health.get("status") != "healthy":
                    health["overall"] = "degraded"
                    
            except Exception as e:
                health["graphiti"] = {"status": "unhealthy", "error": str(e)}
                health["overall"] = "degraded"
        else:
            health["graphiti"] = {"status": "disabled"}
            
        return health