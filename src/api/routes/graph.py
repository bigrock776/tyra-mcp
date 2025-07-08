"""
Knowledge Graph API endpoints.

Provides access to the temporal knowledge graph stored in Memgraph,
including entity extraction, relationship mapping, and graph queries.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.graph.graphiti_integration import GraphitiManager
from ...core.graph.memgraph_client import MemgraphClient
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Enums
class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    TOOL = "tool"
    DOCUMENT = "document"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""

    RELATED_TO = "related_to"
    MENTIONS = "mentions"
    CREATED_BY = "created_by"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    HAPPENED_AT = "happened_at"
    CAUSED_BY = "caused_by"
    SIMILAR_TO = "similar_to"


class GraphQueryType(str, Enum):
    """Types of graph queries."""

    NEIGHBORS = "neighbors"
    PATH = "path"
    SUBGRAPH = "subgraph"
    TEMPORAL = "temporal"
    PATTERN = "pattern"


# Request/Response Models
class Entity(BaseModel):
    """Knowledge graph entity."""

    id: str = Field(..., description="Entity ID")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    properties: Dict[str, Any] = Field(default={}, description="Entity properties")
    confidence: float = Field(..., description="Extraction confidence")
    first_seen: datetime = Field(..., description="First occurrence timestamp")
    last_seen: datetime = Field(..., description="Most recent occurrence timestamp")
    occurrence_count: int = Field(..., description="Number of occurrences")


class Relationship(BaseModel):
    """Relationship between entities."""

    id: str = Field(..., description="Relationship ID")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: RelationshipType = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(
        default={}, description="Relationship properties"
    )
    confidence: float = Field(..., description="Relationship confidence")
    created_at: datetime = Field(..., description="Creation timestamp")


class GraphNode(BaseModel):
    """Node in graph visualization."""

    id: str
    label: str
    type: str
    properties: Dict[str, Any] = {}
    x: Optional[float] = None
    y: Optional[float] = None


class GraphEdge(BaseModel):
    """Edge in graph visualization."""

    id: str
    source: str
    target: str
    label: str
    properties: Dict[str, Any] = {}


class GraphData(BaseModel):
    """Graph data for visualization."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = {}


class EntityExtractionRequest(BaseModel):
    """Request to extract entities from text."""

    text: str = Field(..., description="Text to extract entities from")
    types: Optional[List[EntityType]] = Field(
        None, description="Entity types to extract"
    )
    min_confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )


class GraphQueryRequest(BaseModel):
    """Request for graph queries."""

    query_type: GraphQueryType = Field(..., description="Type of graph query")
    entity_ids: List[str] = Field(..., description="Entity IDs to query")
    depth: int = Field(2, ge=1, le=5, description="Query depth")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class TemporalQueryRequest(BaseModel):
    """Request for temporal graph queries."""

    start_time: datetime = Field(..., description="Start of time range")
    end_time: datetime = Field(..., description="End of time range")
    entity_types: Optional[List[EntityType]] = Field(
        None, description="Filter by entity types"
    )
    relationship_types: Optional[List[RelationshipType]] = Field(
        None, description="Filter by relationship types"
    )
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")


# Dependencies
async def get_memgraph_client() -> MemgraphClient:
    """Get Memgraph client instance."""
    try:
        return get_provider(ProviderType.GRAPH_CLIENT, "memgraph")
    except Exception as e:
        logger.error(f"Failed to get Memgraph client: {e}")
        raise HTTPException(status_code=500, detail="Graph database unavailable")


async def get_graphiti_manager() -> GraphitiManager:
    """Get Graphiti manager instance."""
    try:
        return get_provider(ProviderType.GRAPH_MANAGER, "graphiti")
    except Exception as e:
        logger.error(f"Failed to get Graphiti manager: {e}")
        raise HTTPException(status_code=500, detail="Graph manager unavailable")


@router.post("/entities/extract", response_model=List[Entity])
async def extract_entities(
    request: EntityExtractionRequest,
    graphiti: GraphitiManager = Depends(get_graphiti_manager),
):
    """
    Extract entities from text.

    Uses NLP to identify and extract entities with their types and properties.
    """
    try:
        # Extract entities
        extracted = await graphiti.extract_entities(
            text=request.text,
            entity_types=request.types,
            min_confidence=request.min_confidence,
        )

        # Convert to response format
        entities = []
        for entity_data in extracted:
            entities.append(
                Entity(
                    id=entity_data["id"],
                    name=entity_data["name"],
                    type=EntityType(entity_data["type"]),
                    properties=entity_data.get("properties", {}),
                    confidence=entity_data["confidence"],
                    first_seen=entity_data["first_seen"],
                    last_seen=entity_data["last_seen"],
                    occurrence_count=entity_data["occurrence_count"],
                )
            )

        return entities

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}", response_model=Entity)
async def get_entity(
    entity_id: str, memgraph: MemgraphClient = Depends(get_memgraph_client)
):
    """
    Get details of a specific entity.

    Returns full entity information including properties and metadata.
    """
    try:
        entity_data = await memgraph.get_entity(entity_id)

        if not entity_data:
            raise HTTPException(status_code=404, detail="Entity not found")

        return Entity(
            id=entity_data["id"],
            name=entity_data["name"],
            type=EntityType(entity_data["type"]),
            properties=entity_data.get("properties", {}),
            confidence=entity_data["confidence"],
            first_seen=entity_data["first_seen"],
            last_seen=entity_data["last_seen"],
            occurrence_count=entity_data["occurrence_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities", response_model=List[Entity])
async def list_entities(
    entity_type: Optional[EntityType] = Query(
        None, description="Filter by entity type"
    ),
    search: Optional[str] = Query(None, description="Search in entity names"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    memgraph: MemgraphClient = Depends(get_memgraph_client),
):
    """
    List entities in the knowledge graph.

    Supports filtering by type and searching by name.
    """
    try:
        # Query entities
        entity_list = await memgraph.list_entities(
            entity_type=entity_type, search=search, limit=limit, offset=offset
        )

        # Convert to response format
        entities = []
        for entity_data in entity_list:
            entities.append(
                Entity(
                    id=entity_data["id"],
                    name=entity_data["name"],
                    type=EntityType(entity_data["type"]),
                    properties=entity_data.get("properties", {}),
                    confidence=entity_data["confidence"],
                    first_seen=entity_data["first_seen"],
                    last_seen=entity_data["last_seen"],
                    occurrence_count=entity_data["occurrence_count"],
                )
            )

        return entities

    except Exception as e:
        logger.error(f"Failed to list entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/{entity_id}", response_model=List[Relationship])
async def get_entity_relationships(
    entity_id: str,
    relationship_type: Optional[RelationshipType] = Query(
        None, description="Filter by relationship type"
    ),
    direction: str = Query(
        "both", description="Relationship direction: in, out, or both"
    ),
    memgraph: MemgraphClient = Depends(get_memgraph_client),
):
    """
    Get relationships for an entity.

    Returns all relationships where the entity is either source or target.
    """
    try:
        # Validate direction
        if direction not in ["in", "out", "both"]:
            raise HTTPException(status_code=400, detail="Invalid direction")

        # Query relationships
        relationships_data = await memgraph.get_relationships(
            entity_id=entity_id,
            relationship_type=relationship_type,
            direction=direction,
        )

        # Convert to response format
        relationships = []
        for rel_data in relationships_data:
            relationships.append(
                Relationship(
                    id=rel_data["id"],
                    source_id=rel_data["source_id"],
                    target_id=rel_data["target_id"],
                    type=RelationshipType(rel_data["type"]),
                    properties=rel_data.get("properties", {}),
                    confidence=rel_data["confidence"],
                    created_at=rel_data["created_at"],
                )
            )

        return relationships

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get relationships for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=GraphData)
async def query_graph(
    request: GraphQueryRequest, memgraph: MemgraphClient = Depends(get_memgraph_client)
):
    """
    Execute complex graph queries.

    Supports various query types including neighbor search, path finding,
    and subgraph extraction.
    """
    try:
        result = None

        if request.query_type == GraphQueryType.NEIGHBORS:
            # Get neighbors up to specified depth
            result = await memgraph.get_neighbors(
                entity_ids=request.entity_ids,
                depth=request.depth,
                filters=request.filters,
            )

        elif request.query_type == GraphQueryType.PATH:
            # Find paths between entities
            if len(request.entity_ids) < 2:
                raise HTTPException(
                    status_code=400, detail="Path query requires at least 2 entities"
                )

            result = await memgraph.find_paths(
                start_id=request.entity_ids[0],
                end_id=request.entity_ids[1],
                max_depth=request.depth,
            )

        elif request.query_type == GraphQueryType.SUBGRAPH:
            # Extract subgraph around entities
            result = await memgraph.get_subgraph(
                entity_ids=request.entity_ids,
                depth=request.depth,
                filters=request.filters,
            )

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported query type: {request.query_type}"
            )

        # Convert to graph visualization format
        return _convert_to_graph_data(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/temporal/query", response_model=GraphData)
async def temporal_graph_query(
    request: TemporalQueryRequest,
    memgraph: MemgraphClient = Depends(get_memgraph_client),
):
    """
    Query the temporal knowledge graph.

    Retrieves entities and relationships within a specific time range.
    """
    try:
        # Execute temporal query
        result = await memgraph.temporal_query(
            start_time=request.start_time,
            end_time=request.end_time,
            entity_types=request.entity_types,
            relationship_types=request.relationship_types,
            limit=request.limit,
        )

        # Convert to graph visualization format
        return _convert_to_graph_data(result)

    except Exception as e:
        logger.error(f"Temporal query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memories/{memory_id}/graph")
async def add_memory_to_graph(
    memory_id: str, graphiti: GraphitiManager = Depends(get_graphiti_manager)
):
    """
    Add a memory to the knowledge graph.

    Extracts entities and relationships from the memory and adds them to the graph.
    """
    try:
        # Process memory and add to graph
        result = await graphiti.add_memory_to_graph(memory_id)

        return {
            "memory_id": memory_id,
            "entities_added": result["entities_added"],
            "relationships_added": result["relationships_added"],
            "message": "Memory successfully added to knowledge graph",
        }

    except Exception as e:
        logger.error(f"Failed to add memory {memory_id} to graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_graph_statistics(memgraph: MemgraphClient = Depends(get_memgraph_client)):
    """
    Get knowledge graph statistics.

    Returns information about the size and composition of the graph.
    """
    try:
        stats = await memgraph.get_statistics()

        return {
            "total_entities": stats["total_entities"],
            "total_relationships": stats["total_relationships"],
            "entity_types": stats["entity_types"],
            "relationship_types": stats["relationship_types"],
            "avg_relationships_per_entity": stats["avg_relationships_per_entity"],
            "most_connected_entities": stats["most_connected_entities"],
            "temporal_range": stats["temporal_range"],
        }

    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualize")
async def visualize_graph(
    entity_ids: List[str] = Query(..., description="Entity IDs to visualize"),
    depth: int = Query(2, ge=1, le=3, description="Visualization depth"),
    layout: str = Query(
        "force", description="Layout algorithm: force, hierarchical, circular"
    ),
    memgraph: MemgraphClient = Depends(get_memgraph_client),
):
    """
    Generate graph visualization data.

    Returns graph data with layout coordinates for visualization.
    """
    try:
        # Get subgraph
        subgraph = await memgraph.get_subgraph(entity_ids=entity_ids, depth=depth)

        # Convert to visualization format
        graph_data = _convert_to_graph_data(subgraph)

        # Apply layout algorithm
        if layout == "force":
            graph_data = _apply_force_layout(graph_data)
        elif layout == "hierarchical":
            graph_data = _apply_hierarchical_layout(graph_data)
        elif layout == "circular":
            graph_data = _apply_circular_layout(graph_data)

        return graph_data

    except Exception as e:
        logger.error(f"Graph visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _convert_to_graph_data(raw_data: Dict[str, Any]) -> GraphData:
    """Convert raw graph data to visualization format."""
    nodes = []
    edges = []

    # Process nodes
    for node_data in raw_data.get("nodes", []):
        nodes.append(
            GraphNode(
                id=node_data["id"],
                label=node_data.get("name", node_data["id"]),
                type=node_data.get("type", "unknown"),
                properties=node_data.get("properties", {}),
            )
        )

    # Process edges
    for edge_data in raw_data.get("edges", []):
        edges.append(
            GraphEdge(
                id=edge_data["id"],
                source=edge_data["source"],
                target=edge_data["target"],
                label=edge_data.get("type", "related"),
                properties=edge_data.get("properties", {}),
            )
        )

    return GraphData(
        nodes=nodes,
        edges=edges,
        metadata={"node_count": len(nodes), "edge_count": len(edges)},
    )


def _apply_force_layout(graph_data: GraphData) -> GraphData:
    """Apply force-directed layout to graph."""
    # Placeholder - would use actual layout algorithm
    import random

    for i, node in enumerate(graph_data.nodes):
        node.x = random.uniform(-100, 100)
        node.y = random.uniform(-100, 100)

    return graph_data


def _apply_hierarchical_layout(graph_data: GraphData) -> GraphData:
    """Apply hierarchical layout to graph."""
    # Placeholder - would use actual layout algorithm
    for i, node in enumerate(graph_data.nodes):
        node.x = (i % 5) * 50
        node.y = (i // 5) * 50

    return graph_data


def _apply_circular_layout(graph_data: GraphData) -> GraphData:
    """Apply circular layout to graph."""
    # Placeholder - would use actual layout algorithm
    import math

    n = len(graph_data.nodes)
    for i, node in enumerate(graph_data.nodes):
        angle = (2 * math.pi * i) / n
        node.x = 100 * math.cos(angle)
        node.y = 100 * math.sin(angle)

    return graph_data
