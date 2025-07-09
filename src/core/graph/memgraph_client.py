"""
Memgraph Client for Knowledge Graph Operations.

Simplified client interface for Memgraph operations used by the API layer.
This wraps the provider-based MemgraphEngine for easier API usage.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..providers.graph_engines.memgraph import MemgraphEngine
from ..utils.logger import get_logger
from ..utils.registry import ProviderType, get_provider

logger = get_logger(__name__)


class MemgraphClient:
    """
    Simplified Memgraph client interface for API operations.
    
    This class provides a simplified interface to the MemgraphEngine
    for use by the API layer, with methods that match the expected
    API contract.
    """
    
    def __init__(self):
        """Initialize the Memgraph client."""
        self.engine: Optional[MemgraphEngine] = None
        self.config: Dict[str, Any] = {}
    
    async def initialize(self, config: Dict[str, Any]):
        """
        Initialize the client with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        engine_provider = config.get("engine_provider", "memgraph")
        
        # Get the underlying engine
        self.engine = await get_provider(ProviderType.GRAPH_ENGINE, engine_provider)
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Entity data dict or None if not found
        """
        try:
            entity = await self.engine.get_entity(entity_id)
            if entity:
                return {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "properties": entity.properties or {},
                    "confidence": entity.confidence or 1.0,
                    "first_seen": entity.created_at,
                    "last_seen": entity.updated_at,
                    "occurrence_count": entity.properties.get("occurrence_count", 1) if entity.properties else 1,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    async def list_entities(
        self,
        entity_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List entities with optional filtering.
        
        Args:
            entity_type: Filter by entity type
            search: Search term for entity names
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of entity data dicts
        """
        try:
            # Build search properties
            properties = {}
            if search:
                # For simplicity, we'll search by name prefix
                # In a real implementation, this would be more sophisticated
                properties["name"] = search
            
            entities = await self.engine.find_entities(
                entity_type=entity_type,
                properties=properties,
                limit=limit
            )
            
            # Convert to API format
            result = []
            for entity in entities:
                result.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "properties": entity.properties or {},
                    "confidence": entity.confidence or 1.0,
                    "first_seen": entity.created_at,
                    "last_seen": entity.updated_at,
                    "occurrence_count": entity.properties.get("occurrence_count", 1) if entity.properties else 1,
                })
            
            # Apply offset (since engine doesn't support offset directly)
            return result[offset:]
            
        except Exception as e:
            logger.error(f"Failed to list entities: {e}")
            return []
    
    async def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: The entity ID
            relationship_type: Filter by relationship type
            direction: Relationship direction ("in", "out", "both")
            
        Returns:
            List of relationship data dicts
        """
        try:
            relationships = await self.engine.get_entity_relationships(
                entity_id=entity_id,
                relationship_type=relationship_type,
                direction=direction
            )
            
            # Convert to API format
            result = []
            for rel in relationships:
                result.append({
                    "id": rel.id,
                    "source_id": rel.source_entity_id,
                    "target_id": rel.target_entity_id,
                    "type": rel.relationship_type,
                    "properties": rel.properties or {},
                    "confidence": rel.confidence or 1.0,
                    "created_at": rel.created_at,
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get relationships for {entity_id}: {e}")
            return []
    
    async def get_neighbors(
        self,
        entity_ids: List[str],
        depth: int = 1,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get neighboring entities up to a specified depth.
        
        Args:
            entity_ids: List of entity IDs to start from
            depth: Maximum depth to traverse
            filters: Optional filters to apply
            
        Returns:
            Graph data with nodes and edges
        """
        try:
            all_entities = set()
            all_relationships = []
            
            # Get connected entities for each starting entity
            for entity_id in entity_ids:
                connected = await self.engine.get_connected_entities(
                    entity_id=entity_id,
                    max_depth=depth
                )
                
                for entity in connected:
                    all_entities.add(entity.id)
                
                # Get relationships for this entity
                relationships = await self.engine.get_entity_relationships(
                    entity_id=entity_id,
                    direction="both"
                )
                all_relationships.extend(relationships)
            
            # Convert to graph format
            nodes = []
            for entity_id in list(all_entities) + entity_ids:
                entity = await self.engine.get_entity(entity_id)
                if entity:
                    nodes.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "properties": entity.properties or {},
                    })
            
            edges = []
            for rel in all_relationships:
                edges.append({
                    "id": rel.id,
                    "source": rel.source_entity_id,
                    "target": rel.target_entity_id,
                    "type": rel.relationship_type,
                    "properties": rel.properties or {},
                })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return {"nodes": [], "edges": []}
    
    async def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Find paths between two entities.
        
        Args:
            start_id: Starting entity ID
            end_id: Ending entity ID
            max_depth: Maximum path length
            
        Returns:
            Graph data representing the path
        """
        try:
            path = await self.engine.find_path(
                source_entity_id=start_id,
                target_entity_id=end_id,
                max_depth=max_depth
            )
            
            if not path:
                return {"nodes": [], "edges": []}
            
            # Get all entities in the path
            entity_ids = {start_id, end_id}
            for rel in path:
                entity_ids.add(rel.source_entity_id)
                entity_ids.add(rel.target_entity_id)
            
            nodes = []
            for entity_id in entity_ids:
                entity = await self.engine.get_entity(entity_id)
                if entity:
                    nodes.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "properties": entity.properties or {},
                    })
            
            edges = []
            for rel in path:
                edges.append({
                    "id": rel.id,
                    "source": rel.source_entity_id,
                    "target": rel.target_entity_id,
                    "type": rel.relationship_type,
                    "properties": rel.properties or {},
                })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Failed to find path from {start_id} to {end_id}: {e}")
            return {"nodes": [], "edges": []}
    
    async def get_subgraph(
        self,
        entity_ids: List[str],
        depth: int = 1,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a subgraph around specified entities.
        
        Args:
            entity_ids: List of entity IDs to center the subgraph on
            depth: Depth of subgraph to extract
            filters: Optional filters to apply
            
        Returns:
            Graph data representing the subgraph
        """
        # For now, this is the same as get_neighbors
        return await self.get_neighbors(entity_ids, depth, filters)
    
    async def temporal_query(
        self,
        start_time: datetime,
        end_time: datetime,
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Query the graph for temporal data.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            entity_types: Filter by entity types
            relationship_types: Filter by relationship types
            limit: Maximum number of results
            
        Returns:
            Graph data within the time range
        """
        try:
            # Build query parameters
            properties = {}
            if entity_types:
                # This is a simplified approach - in practice, we'd need
                # more sophisticated temporal querying
                pass
            
            # Get entities created within the time range
            entities = await self.engine.find_entities(
                entity_type=entity_types[0] if entity_types else None,
                properties=properties,
                limit=limit
            )
            
            # Filter by time range
            filtered_entities = []
            for entity in entities:
                if entity.created_at and start_time <= entity.created_at <= end_time:
                    filtered_entities.append(entity)
            
            # Get relationships between these entities
            all_relationships = []
            entity_ids = [e.id for e in filtered_entities]
            
            for entity_id in entity_ids:
                relationships = await self.engine.get_entity_relationships(
                    entity_id=entity_id,
                    relationship_type=relationship_types[0] if relationship_types else None,
                    direction="both"
                )
                
                # Filter relationships by time range
                for rel in relationships:
                    if rel.created_at and start_time <= rel.created_at <= end_time:
                        all_relationships.append(rel)
            
            # Convert to graph format
            nodes = []
            for entity in filtered_entities:
                nodes.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "properties": entity.properties or {},
                    "created_at": entity.created_at,
                })
            
            edges = []
            for rel in all_relationships:
                edges.append({
                    "id": rel.id,
                    "source": rel.source_entity_id,
                    "target": rel.target_entity_id,
                    "type": rel.relationship_type,
                    "properties": rel.properties or {},
                    "created_at": rel.created_at,
                })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "temporal_range": {
                    "start": start_time,
                    "end": end_time
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to perform temporal query: {e}")
            return {"nodes": [], "edges": []}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            stats = await self.engine.get_stats()
            
            # Convert to API format
            return {
                "total_entities": stats.get("graph_statistics", {}).get("total_entities", 0),
                "total_relationships": stats.get("graph_statistics", {}).get("total_relationships", 0),
                "entity_types": stats.get("schema", {}).get("tracked_entity_types", []),
                "relationship_types": stats.get("schema", {}).get("tracked_relationship_types", []),
                "avg_relationships_per_entity": (
                    stats.get("graph_statistics", {}).get("total_relationships", 0) / 
                    max(stats.get("graph_statistics", {}).get("total_entities", 1), 1)
                ),
                "most_connected_entities": [],  # Would need additional query
                "temporal_range": {
                    "earliest": None,  # Would need additional query
                    "latest": None,    # Would need additional query
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "total_entities": 0,
                "total_relationships": 0,
                "entity_types": [],
                "relationship_types": [],
                "avg_relationships_per_entity": 0.0,
                "most_connected_entities": [],
                "temporal_range": {}
            }