"""
Abstract interface for graph database engines.

This module defines the standard interface that all graph database providers must implement,
enabling easy swapping of graph engines without changing core logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class RelationshipType(Enum):
    """Standard relationship types for knowledge graphs."""

    RELATED_TO = "RELATED_TO"
    MENTIONS = "MENTIONS"
    CONTAINS = "CONTAINS"
    CAUSED_BY = "CAUSED_BY"
    FOLLOWS = "FOLLOWS"
    PRECEDES = "PRECEDES"
    PART_OF = "PART_OF"
    INSTANCE_OF = "INSTANCE_OF"
    SIMILAR_TO = "SIMILAR_TO"


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""

    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    confidence: Optional[float] = None


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    confidence: Optional[float] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None


@dataclass
class GraphSearchResult:
    """Result from graph query operations."""

    entities: List[Entity]
    relationships: List[Relationship]
    metadata: Dict[str, Any]


class GraphEngine(ABC):
    """Abstract base class for graph database engines."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the graph engine with configuration.

        Args:
            config: Engine-specific configuration dictionary
        """
        pass

    @abstractmethod
    async def create_entity(self, entity: Entity) -> str:
        """
        Create a new entity in the graph.

        Args:
            entity: Entity to create

        Returns:
            Entity ID that was created
        """
        pass

    @abstractmethod
    async def create_entities(self, entities: List[Entity]) -> List[str]:
        """
        Create multiple entities in the graph.

        Args:
            entities: List of entities to create

        Returns:
            List of entity IDs that were created
        """
        pass

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: ID of entity to retrieve

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_entity(self, entity: Entity) -> bool:
        """
        Update an existing entity.

        Args:
            entity: Updated entity

        Returns:
            True if entity was updated, False if not found
        """
        pass

    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and all its relationships.

        Args:
            entity_id: ID of entity to delete

        Returns:
            True if entity was deleted, False if not found
        """
        pass

    @abstractmethod
    async def create_relationship(self, relationship: Relationship) -> str:
        """
        Create a new relationship between entities.

        Args:
            relationship: Relationship to create

        Returns:
            Relationship ID that was created
        """
        pass

    @abstractmethod
    async def create_relationships(
        self, relationships: List[Relationship]
    ) -> List[str]:
        """
        Create multiple relationships.

        Args:
            relationships: List of relationships to create

        Returns:
            List of relationship IDs that were created
        """
        pass

    @abstractmethod
    async def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Retrieve a relationship by ID.

        Args:
            relationship_id: ID of relationship to retrieve

        Returns:
            Relationship if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship.

        Args:
            relationship_id: ID of relationship to delete

        Returns:
            True if relationship was deleted, False if not found
        """
        pass

    @abstractmethod
    async def find_entities(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """
        Find entities matching criteria.

        Args:
            entity_type: Optional entity type filter
            properties: Optional property filters
            limit: Maximum number of entities to return

        Returns:
            List of matching entities
        """
        pass

    @abstractmethod
    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> List[Relationship]:
        """
        Get all relationships for an entity.

        Args:
            entity_id: ID of the entity
            relationship_type: Optional relationship type filter
            direction: Direction of relationships to include

        Returns:
            List of relationships
        """
        pass

    @abstractmethod
    async def get_connected_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1,
    ) -> List[Entity]:
        """
        Get entities connected to the given entity.

        Args:
            entity_id: ID of the source entity
            relationship_type: Optional relationship type filter
            max_depth: Maximum depth to traverse

        Returns:
            List of connected entities
        """
        pass

    @abstractmethod
    async def find_path(
        self, source_entity_id: str, target_entity_id: str, max_depth: int = 3
    ) -> Optional[List[Relationship]]:
        """
        Find shortest path between two entities.

        Args:
            source_entity_id: ID of source entity
            target_entity_id: ID of target entity
            max_depth: Maximum path length

        Returns:
            List of relationships forming the path, None if no path found
        """
        pass

    @abstractmethod
    async def execute_cypher(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a raw Cypher query.

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Returns:
            Query results
        """
        pass

    @abstractmethod
    async def get_entity_timeline(
        self,
        entity_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Relationship]:
        """
        Get temporal relationships for an entity.

        Args:
            entity_id: ID of the entity
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of relationships ordered by time
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the graph engine.

        Returns:
            Dictionary with health status information
        """
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph database.

        Returns:
            Dictionary with statistics (node count, edge count, etc.)
        """
        pass

    async def entity_exists(self, entity_id: str) -> bool:
        """
        Check if an entity exists.

        Args:
            entity_id: ID of entity to check

        Returns:
            True if entity exists, False otherwise
        """
        entity = await self.get_entity(entity_id)
        return entity is not None

    async def relationship_exists(self, relationship_id: str) -> bool:
        """
        Check if a relationship exists.

        Args:
            relationship_id: ID of relationship to check

        Returns:
            True if relationship exists, False otherwise
        """
        relationship = await self.get_relationship(relationship_id)
        return relationship is not None


class GraphEngineError(Exception):
    """Base exception for graph engine errors."""

    pass


class GraphEngineInitializationError(GraphEngineError):
    """Raised when graph engine initialization fails."""

    pass


class GraphEngineOperationError(GraphEngineError):
    """Raised when graph engine operation fails."""

    pass


class GraphEngineConfigurationError(GraphEngineError):
    """Raised when graph engine configuration is invalid."""

    pass
