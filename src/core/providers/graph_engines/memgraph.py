"""
Memgraph graph engine implementation with Graphiti integration.

High-performance graph database provider with temporal knowledge graphs,
entity relationship tracking, and advanced Cypher query optimization.
"""

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from gqlalchemy import Memgraph
from gqlalchemy.models import Node, Relationship

from ...interfaces.graph_engine import (
    Entity,
    GraphEngine,
    GraphEngineError,
    GraphEngineInitializationError,
    GraphEngineOperationError,
    GraphSearchResult,
)
from ...interfaces.graph_engine import Relationship as RelationshipInterface
from ...interfaces.graph_engine import (
    RelationshipType,
)
from ...utils.database import MemgraphManager
from ...utils.logger import get_logger

logger = get_logger(__name__)


class MemgraphEngine(GraphEngine):
    """
    Advanced Memgraph graph engine with temporal knowledge graphs.

    Features:
    - Temporal relationship tracking with validity periods
    - High-performance Cypher query execution
    - Entity relationship optimization
    - Batch operations for large datasets
    - Advanced graph analytics and traversal
    - Integration with Graphiti framework
    - Comprehensive monitoring and statistics
    """

    def __init__(self):
        self.db_manager: Optional[MemgraphManager] = None
        self.client: Optional[Memgraph] = None
        self.config: Dict[str, Any] = {}

        # Performance tracking
        self._total_queries: int = 0
        self._total_entities: int = 0
        self._total_relationships: int = 0
        self._avg_query_time: float = 0.0
        self._error_count: int = 0

        # Schema tracking
        self._entity_types: set = set()
        self._relationship_types: set = set()

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Memgraph graph engine."""
        try:
            self.config = config

            # Initialize database manager
            db_config = {
                "host": config.get("host", "localhost"),
                "port": config.get("port", 7687),
                "username": config.get("username", "memgraph"),
                "password": config.get("password", ""),
                "encrypted": config.get("encrypted", False),
                "connection_timeout": config.get("connection_timeout", 30),
            }

            self.db_manager = MemgraphManager(db_config)
            await self.db_manager.initialize()

            # Get direct client for advanced operations
            self.client = Memgraph(
                host=db_config["host"],
                port=db_config["port"],
                username=db_config["username"],
                password=db_config["password"],
                encrypted=db_config["encrypted"],
            )

            # Initialize schema and indexes
            await self._initialize_schema()
            await self._create_indexes()

            logger.info(
                "Memgraph engine initialized",
                host=config.get("host"),
                port=config.get("port"),
            )

        except Exception as e:
            logger.error("Failed to initialize Memgraph engine", error=str(e))
            raise GraphEngineInitializationError(f"Memgraph initialization failed: {e}")

    async def _initialize_schema(self) -> None:
        """Initialize graph schema and constraints."""
        schema_queries = [
            # Create constraints for entity uniqueness
            "CREATE CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE;",
            # Create indexes for performance
            "CREATE INDEX ON :Entity(entity_type);",
            "CREATE INDEX ON :Entity(name);",
            "CREATE INDEX ON :Entity(created_at);",
            "CREATE INDEX ON :Entity(confidence);",
            # Relationship indexes
            "CREATE INDEX ON :RELATIONSHIP(relationship_type);",
            "CREATE INDEX ON :RELATIONSHIP(created_at);",
            "CREATE INDEX ON :RELATIONSHIP(valid_from);",
            "CREATE INDEX ON :RELATIONSHIP(valid_to);",
            "CREATE INDEX ON :RELATIONSHIP(confidence);",
        ]

        for query in schema_queries:
            try:
                await self._execute_query(query)
            except Exception as e:
                # Some constraints/indexes might already exist
                logger.debug(f"Schema query warning: {e}")

    async def _create_indexes(self) -> None:
        """Create optimized indexes for graph operations."""
        index_queries = [
            # Text search indexes
            "CREATE TEXT INDEX entityNameIndex ON (n:Entity) FIELDS (n.name);",
            "CREATE TEXT INDEX entityPropertiesIndex ON (n:Entity) FIELDS (n.properties);",
            # Range indexes for temporal queries
            "CREATE RANGE INDEX entityCreatedIndex ON (n:Entity) FIELDS (n.created_at);",
            "CREATE RANGE INDEX relationshipValidFromIndex ON (r:RELATIONSHIP) FIELDS (r.valid_from);",
            "CREATE RANGE INDEX relationshipValidToIndex ON (r:RELATIONSHIP) FIELDS (r.valid_to);",
        ]

        for query in index_queries:
            try:
                await self._execute_query(query)
            except Exception as e:
                logger.debug(f"Index creation warning: {e}")

    async def _execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query with performance tracking."""
        start_time = time.time()

        try:
            self._total_queries += 1

            # Use database manager for query execution
            result = await self.db_manager.execute_query(query, parameters)

            # Update performance tracking
            query_time = time.time() - start_time
            self._avg_query_time = (
                self._avg_query_time * (self._total_queries - 1) + query_time
            ) / self._total_queries

            logger.debug(
                "Cypher query executed",
                query=query[:100] + "..." if len(query) > 100 else query,
                time=query_time,
                results=len(result) if isinstance(result, list) else 1,
            )

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(
                "Cypher query failed",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Query execution failed: {e}")

    async def create_entity(self, entity: Entity) -> str:
        """Create a new entity in the graph."""
        try:
            # Prepare entity properties
            properties = entity.properties.copy() if entity.properties else {}
            properties.update(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "created_at": (entity.created_at or datetime.utcnow()).isoformat(),
                    "updated_at": (entity.updated_at or datetime.utcnow()).isoformat(),
                    "confidence": entity.confidence or 1.0,
                }
            )

            # Build properties string for Cypher
            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])

            query = f"""
            MERGE (e:Entity {{id: $id}})
            SET e += {{{props_str}}}
            SET e.entity_type = $entity_type
            RETURN e.id as id
            """

            result = await self._execute_query(query, properties)

            # Track entity type
            self._entity_types.add(entity.entity_type)
            self._total_entities += 1

            logger.debug(
                "Entity created",
                entity_id=entity.id,
                entity_type=entity.entity_type,
                name=entity.name,
            )

            return entity.id

        except Exception as e:
            logger.error("Failed to create entity", entity_id=entity.id, error=str(e))
            raise GraphEngineOperationError(f"Entity creation failed: {e}")

    async def create_entities(self, entities: List[Entity]) -> List[str]:
        """Create multiple entities efficiently using batch operations."""
        if not entities:
            return []

        try:
            # Prepare batch data
            entity_data = []
            for entity in entities:
                properties = entity.properties.copy() if entity.properties else {}
                properties.update(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "created_at": (
                            entity.created_at or datetime.utcnow()
                        ).isoformat(),
                        "updated_at": (
                            entity.updated_at or datetime.utcnow()
                        ).isoformat(),
                        "confidence": entity.confidence or 1.0,
                    }
                )
                entity_data.append(properties)
                self._entity_types.add(entity.entity_type)

            # Batch create query
            query = """
            UNWIND $entities as entity_data
            MERGE (e:Entity {id: entity_data.id})
            SET e += entity_data
            RETURN e.id as id
            """

            result = await self._execute_query(query, {"entities": entity_data})

            self._total_entities += len(entities)

            logger.info(
                "Batch entities created",
                count=len(entities),
                unique_types=len(set(e.entity_type for e in entities)),
            )

            return [entity.id for entity in entities]

        except Exception as e:
            logger.error(
                "Failed to create entities in batch", count=len(entities), error=str(e)
            )
            raise GraphEngineOperationError(f"Batch entity creation failed: {e}")

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})
            RETURN e.id as id, e.name as name, e.entity_type as entity_type,
                   e.properties as properties, e.created_at as created_at,
                   e.updated_at as updated_at, e.confidence as confidence
            """

            result = await self._execute_query(query, {"entity_id": entity_id})

            if result:
                row = result[0]
                return Entity(
                    id=row["id"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    properties=row.get("properties", {}),
                    created_at=(
                        datetime.fromisoformat(row["created_at"])
                        if row.get("created_at")
                        else None
                    ),
                    updated_at=(
                        datetime.fromisoformat(row["updated_at"])
                        if row.get("updated_at")
                        else None
                    ),
                    confidence=row.get("confidence"),
                )

            return None

        except Exception as e:
            logger.error("Failed to get entity", entity_id=entity_id, error=str(e))
            raise GraphEngineOperationError(f"Get entity failed: {e}")

    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        try:
            properties = entity.properties.copy() if entity.properties else {}
            properties.update(
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "updated_at": datetime.utcnow().isoformat(),
                    "confidence": entity.confidence or 1.0,
                }
            )

            # Build properties string
            props_str = ", ".join([f"e.{k} = ${k}" for k in properties.keys()])

            query = f"""
            MATCH (e:Entity {{id: $id}})
            SET {props_str}
            RETURN e.id as id
            """

            parameters = {"id": entity.id, **properties}
            result = await self._execute_query(query, parameters)

            return len(result) > 0

        except Exception as e:
            logger.error("Failed to update entity", entity_id=entity.id, error=str(e))
            raise GraphEngineOperationError(f"Entity update failed: {e}")

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})
            DETACH DELETE e
            RETURN count(e) as deleted_count
            """

            result = await self._execute_query(query, {"entity_id": entity_id})

            deleted = result[0]["deleted_count"] > 0 if result else False

            if deleted:
                self._total_entities = max(0, self._total_entities - 1)

            return deleted

        except Exception as e:
            logger.error("Failed to delete entity", entity_id=entity_id, error=str(e))
            raise GraphEngineOperationError(f"Entity deletion failed: {e}")

    async def create_relationship(self, relationship: RelationshipInterface) -> str:
        """Create a new relationship between entities."""
        try:
            # Prepare relationship properties
            properties = (
                relationship.properties.copy() if relationship.properties else {}
            )
            properties.update(
                {
                    "id": relationship.id,
                    "relationship_type": relationship.relationship_type,
                    "created_at": (
                        relationship.created_at or datetime.utcnow()
                    ).isoformat(),
                    "updated_at": (
                        (relationship.updated_at or datetime.updated_at).isoformat()
                        if relationship.updated_at
                        else datetime.utcnow().isoformat()
                    ),
                    "confidence": relationship.confidence or 1.0,
                    "valid_from": (
                        relationship.valid_from.isoformat()
                        if relationship.valid_from
                        else None
                    ),
                    "valid_to": (
                        relationship.valid_to.isoformat()
                        if relationship.valid_to
                        else None
                    ),
                }
            )

            # Build properties string
            props_str = ", ".join(
                [f"{k}: ${k}" for k in properties.keys() if properties[k] is not None]
            )

            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            CREATE (source)-[r:RELATIONSHIP {{{props_str}}}]->(target)
            RETURN r.id as id
            """

            parameters = {
                "source_id": relationship.source_entity_id,
                "target_id": relationship.target_entity_id,
                **{k: v for k, v in properties.items() if v is not None},
            }

            result = await self._execute_query(query, parameters)

            # Track relationship type
            self._relationship_types.add(relationship.relationship_type)
            self._total_relationships += 1

            logger.debug(
                "Relationship created",
                relationship_id=relationship.id,
                type=relationship.relationship_type,
                source=relationship.source_entity_id,
                target=relationship.target_entity_id,
            )

            return relationship.id

        except Exception as e:
            logger.error(
                "Failed to create relationship",
                relationship_id=relationship.id,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Relationship creation failed: {e}")

    async def create_relationships(
        self, relationships: List[RelationshipInterface]
    ) -> List[str]:
        """Create multiple relationships efficiently."""
        if not relationships:
            return []

        try:
            # Prepare batch data
            rel_data = []
            for rel in relationships:
                properties = rel.properties.copy() if rel.properties else {}
                properties.update(
                    {
                        "id": rel.id,
                        "source_id": rel.source_entity_id,
                        "target_id": rel.target_entity_id,
                        "relationship_type": rel.relationship_type,
                        "created_at": (rel.created_at or datetime.utcnow()).isoformat(),
                        "updated_at": (rel.updated_at or datetime.utcnow()).isoformat(),
                        "confidence": rel.confidence or 1.0,
                        "valid_from": (
                            rel.valid_from.isoformat() if rel.valid_from else None
                        ),
                        "valid_to": rel.valid_to.isoformat() if rel.valid_to else None,
                    }
                )
                rel_data.append(properties)
                self._relationship_types.add(rel.relationship_type)

            query = """
            UNWIND $relationships as rel_data
            MATCH (source:Entity {id: rel_data.source_id})
            MATCH (target:Entity {id: rel_data.target_id})
            CREATE (source)-[r:RELATIONSHIP]->(target)
            SET r += rel_data
            RETURN r.id as id
            """

            result = await self._execute_query(query, {"relationships": rel_data})

            self._total_relationships += len(relationships)

            logger.info(
                "Batch relationships created",
                count=len(relationships),
                unique_types=len(set(r.relationship_type for r in relationships)),
            )

            return [rel.id for rel in relationships]

        except Exception as e:
            logger.error(
                "Failed to create relationships in batch",
                count=len(relationships),
                error=str(e),
            )
            raise GraphEngineOperationError(f"Batch relationship creation failed: {e}")

    async def get_relationship(
        self, relationship_id: str
    ) -> Optional[RelationshipInterface]:
        """Retrieve a relationship by ID."""
        try:
            query = """
            MATCH ()-[r:RELATIONSHIP {id: $relationship_id}]->()
            RETURN r.id as id, r.source_id as source_entity_id, r.target_id as target_entity_id,
                   r.relationship_type as relationship_type, r.properties as properties,
                   r.created_at as created_at, r.updated_at as updated_at,
                   r.confidence as confidence, r.valid_from as valid_from, r.valid_to as valid_to
            """

            result = await self._execute_query(
                query, {"relationship_id": relationship_id}
            )

            if result:
                row = result[0]
                return RelationshipInterface(
                    id=row["id"],
                    source_entity_id=row["source_entity_id"],
                    target_entity_id=row["target_entity_id"],
                    relationship_type=row["relationship_type"],
                    properties=row.get("properties", {}),
                    created_at=(
                        datetime.fromisoformat(row["created_at"])
                        if row.get("created_at")
                        else None
                    ),
                    updated_at=(
                        datetime.fromisoformat(row["updated_at"])
                        if row.get("updated_at")
                        else None
                    ),
                    confidence=row.get("confidence"),
                    valid_from=(
                        datetime.fromisoformat(row["valid_from"])
                        if row.get("valid_from")
                        else None
                    ),
                    valid_to=(
                        datetime.fromisoformat(row["valid_to"])
                        if row.get("valid_to")
                        else None
                    ),
                )

            return None

        except Exception as e:
            logger.error(
                "Failed to get relationship",
                relationship_id=relationship_id,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Get relationship failed: {e}")

    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        try:
            query = """
            MATCH ()-[r:RELATIONSHIP {id: $relationship_id}]->()
            DELETE r
            RETURN count(r) as deleted_count
            """

            result = await self._execute_query(
                query, {"relationship_id": relationship_id}
            )

            deleted = result[0]["deleted_count"] > 0 if result else False

            if deleted:
                self._total_relationships = max(0, self._total_relationships - 1)

            return deleted

        except Exception as e:
            logger.error(
                "Failed to delete relationship",
                relationship_id=relationship_id,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Relationship deletion failed: {e}")

    async def find_entities(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """Find entities matching criteria."""
        try:
            where_clauses = []
            parameters = {"limit": limit}

            if entity_type:
                where_clauses.append("e.entity_type = $entity_type")
                parameters["entity_type"] = entity_type

            if properties:
                for key, value in properties.items():
                    param_name = f"prop_{key}"
                    where_clauses.append(f"e.{key} = ${param_name}")
                    parameters[param_name] = value

            where_clause = " AND ".join(where_clauses) if where_clauses else "true"

            query = f"""
            MATCH (e:Entity)
            WHERE {where_clause}
            RETURN e.id as id, e.name as name, e.entity_type as entity_type,
                   e.properties as properties, e.created_at as created_at,
                   e.updated_at as updated_at, e.confidence as confidence
            ORDER BY e.created_at DESC
            LIMIT $limit
            """

            result = await self._execute_query(query, parameters)

            entities = []
            for row in result:
                entities.append(
                    Entity(
                        id=row["id"],
                        name=row["name"],
                        entity_type=row["entity_type"],
                        properties=row.get("properties", {}),
                        created_at=(
                            datetime.fromisoformat(row["created_at"])
                            if row.get("created_at")
                            else None
                        ),
                        updated_at=(
                            datetime.fromisoformat(row["updated_at"])
                            if row.get("updated_at")
                            else None
                        ),
                        confidence=row.get("confidence"),
                    )
                )

            return entities

        except Exception as e:
            logger.error(
                "Failed to find entities", entity_type=entity_type, error=str(e)
            )
            raise GraphEngineOperationError(f"Entity search failed: {e}")

    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[RelationshipInterface]:
        """Get all relationships for an entity."""
        try:
            # Build direction pattern
            if direction == "outgoing":
                pattern = "(e)-[r:RELATIONSHIP]->()"
            elif direction == "incoming":
                pattern = "()-[r:RELATIONSHIP]->(e)"
            else:  # both
                pattern = "(e)-[r:RELATIONSHIP]-()"

            where_clauses = ["e.id = $entity_id"]
            parameters = {"entity_id": entity_id}

            if relationship_type:
                where_clauses.append("r.relationship_type = $relationship_type")
                parameters["relationship_type"] = relationship_type

            where_clause = " AND ".join(where_clauses)

            query = f"""
            MATCH {pattern}
            WHERE {where_clause}
            RETURN r.id as id, r.source_id as source_entity_id, r.target_id as target_entity_id,
                   r.relationship_type as relationship_type, r.properties as properties,
                   r.created_at as created_at, r.updated_at as updated_at,
                   r.confidence as confidence, r.valid_from as valid_from, r.valid_to as valid_to
            ORDER BY r.created_at DESC
            """

            result = await self._execute_query(query, parameters)

            relationships = []
            for row in result:
                relationships.append(
                    RelationshipInterface(
                        id=row["id"],
                        source_entity_id=row["source_entity_id"],
                        target_entity_id=row["target_entity_id"],
                        relationship_type=row["relationship_type"],
                        properties=row.get("properties", {}),
                        created_at=(
                            datetime.fromisoformat(row["created_at"])
                            if row.get("created_at")
                            else None
                        ),
                        updated_at=(
                            datetime.fromisoformat(row["updated_at"])
                            if row.get("updated_at")
                            else None
                        ),
                        confidence=row.get("confidence"),
                        valid_from=(
                            datetime.fromisoformat(row["valid_from"])
                            if row.get("valid_from")
                            else None
                        ),
                        valid_to=(
                            datetime.fromisoformat(row["valid_to"])
                            if row.get("valid_to")
                            else None
                        ),
                    )
                )

            return relationships

        except Exception as e:
            logger.error(
                "Failed to get entity relationships", entity_id=entity_id, error=str(e)
            )
            raise GraphEngineOperationError(f"Get relationships failed: {e}")

    async def get_connected_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1,
    ) -> List[Entity]:
        """Get entities connected to the given entity."""
        try:
            where_clauses = []
            parameters = {"entity_id": entity_id, "max_depth": max_depth}

            if relationship_type:
                where_clauses.append(
                    "ALL(r IN relationships(p) WHERE r.relationship_type = $relationship_type)"
                )
                parameters["relationship_type"] = relationship_type

            where_clause = " AND ".join(where_clauses) if where_clauses else "true"

            query = f"""
            MATCH p = (start:Entity {{id: $entity_id}})-[*1..{max_depth}]-(connected:Entity)
            WHERE {where_clause}
            RETURN DISTINCT connected.id as id, connected.name as name,
                   connected.entity_type as entity_type, connected.properties as properties,
                   connected.created_at as created_at, connected.updated_at as updated_at,
                   connected.confidence as confidence
            ORDER BY connected.name
            """

            result = await self._execute_query(query, parameters)

            entities = []
            for row in result:
                entities.append(
                    Entity(
                        id=row["id"],
                        name=row["name"],
                        entity_type=row["entity_type"],
                        properties=row.get("properties", {}),
                        created_at=(
                            datetime.fromisoformat(row["created_at"])
                            if row.get("created_at")
                            else None
                        ),
                        updated_at=(
                            datetime.fromisoformat(row["updated_at"])
                            if row.get("updated_at")
                            else None
                        ),
                        confidence=row.get("confidence"),
                    )
                )

            return entities

        except Exception as e:
            logger.error(
                "Failed to get connected entities", entity_id=entity_id, error=str(e)
            )
            raise GraphEngineOperationError(f"Get connected entities failed: {e}")

    async def find_path(
        self, source_entity_id: str, target_entity_id: str, max_depth: int = 3
    ) -> Optional[List[RelationshipInterface]]:
        """Find shortest path between two entities."""
        try:
            query = f"""
            MATCH p = shortestPath((source:Entity {{id: $source_id}})-[*1..{max_depth}]-(target:Entity {{id: $target_id}}))
            RETURN [r IN relationships(p) | {{
                id: r.id,
                source_entity_id: r.source_id,
                target_entity_id: r.target_id,
                relationship_type: r.relationship_type,
                properties: r.properties,
                created_at: r.created_at,
                updated_at: r.updated_at,
                confidence: r.confidence,
                valid_from: r.valid_from,
                valid_to: r.valid_to
            }}] as path_relationships
            """

            parameters = {"source_id": source_entity_id, "target_id": target_entity_id}

            result = await self._execute_query(query, parameters)

            if result and result[0]["path_relationships"]:
                relationships = []
                for rel_data in result[0]["path_relationships"]:
                    relationships.append(
                        RelationshipInterface(
                            id=rel_data["id"],
                            source_entity_id=rel_data["source_entity_id"],
                            target_entity_id=rel_data["target_entity_id"],
                            relationship_type=rel_data["relationship_type"],
                            properties=rel_data.get("properties", {}),
                            created_at=(
                                datetime.fromisoformat(rel_data["created_at"])
                                if rel_data.get("created_at")
                                else None
                            ),
                            updated_at=(
                                datetime.fromisoformat(rel_data["updated_at"])
                                if rel_data.get("updated_at")
                                else None
                            ),
                            confidence=rel_data.get("confidence"),
                            valid_from=(
                                datetime.fromisoformat(rel_data["valid_from"])
                                if rel_data.get("valid_from")
                                else None
                            ),
                            valid_to=(
                                datetime.fromisoformat(rel_data["valid_to"])
                                if rel_data.get("valid_to")
                                else None
                            ),
                        )
                    )
                return relationships

            return None

        except Exception as e:
            logger.error(
                "Failed to find path",
                source=source_entity_id,
                target=target_entity_id,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Path finding failed: {e}")

    async def execute_cypher(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a raw Cypher query."""
        return await self._execute_query(query, parameters)

    async def get_entity_timeline(
        self,
        entity_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[RelationshipInterface]:
        """Get temporal relationships for an entity."""
        try:
            where_clauses = [
                "(start_entity.id = $entity_id OR end_entity.id = $entity_id)"
            ]
            parameters = {"entity_id": entity_id}

            if start_time:
                where_clauses.append(
                    "(r.valid_from IS NULL OR r.valid_from >= $start_time)"
                )
                parameters["start_time"] = start_time.isoformat()

            if end_time:
                where_clauses.append("(r.valid_to IS NULL OR r.valid_to <= $end_time)")
                parameters["end_time"] = end_time.isoformat()

            where_clause = " AND ".join(where_clauses)

            query = f"""
            MATCH (start_entity:Entity)-[r:RELATIONSHIP]->(end_entity:Entity)
            WHERE {where_clause}
            RETURN r.id as id, r.source_id as source_entity_id, r.target_id as target_entity_id,
                   r.relationship_type as relationship_type, r.properties as properties,
                   r.created_at as created_at, r.updated_at as updated_at,
                   r.confidence as confidence, r.valid_from as valid_from, r.valid_to as valid_to
            ORDER BY COALESCE(r.valid_from, r.created_at) ASC
            """

            result = await self._execute_query(query, parameters)

            relationships = []
            for row in result:
                relationships.append(
                    RelationshipInterface(
                        id=row["id"],
                        source_entity_id=row["source_entity_id"],
                        target_entity_id=row["target_entity_id"],
                        relationship_type=row["relationship_type"],
                        properties=row.get("properties", {}),
                        created_at=(
                            datetime.fromisoformat(row["created_at"])
                            if row.get("created_at")
                            else None
                        ),
                        updated_at=(
                            datetime.fromisoformat(row["updated_at"])
                            if row.get("updated_at")
                            else None
                        ),
                        confidence=row.get("confidence"),
                        valid_from=(
                            datetime.fromisoformat(row["valid_from"])
                            if row.get("valid_from")
                            else None
                        ),
                        valid_to=(
                            datetime.fromisoformat(row["valid_to"])
                            if row.get("valid_to")
                            else None
                        ),
                    )
                )

            return relationships

        except Exception as e:
            logger.error(
                "Failed to get entity timeline", entity_id=entity_id, error=str(e)
            )
            raise GraphEngineOperationError(f"Timeline query failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            # Check database connection
            db_health = await self.db_manager.health_check()

            # Get graph statistics
            stats_query = """
            MATCH (e:Entity)
            OPTIONAL MATCH ()-[r:RELATIONSHIP]->()
            RETURN
                count(DISTINCT e) as entity_count,
                count(DISTINCT r) as relationship_count,
                count(DISTINCT e.entity_type) as entity_types,
                count(DISTINCT r.relationship_type) as relationship_types
            """

            stats = await self._execute_query(stats_query)
            graph_stats = stats[0] if stats else {}

            # Test basic operations
            test_start = time.time()
            await self._execute_query("RETURN 1 as test")
            response_time = time.time() - test_start

            return {
                "status": (
                    "healthy" if db_health["status"] == "healthy" else "unhealthy"
                ),
                "database": db_health,
                "response_time": response_time,
                "graph_stats": graph_stats,
                "performance": {
                    "total_queries": self._total_queries,
                    "avg_query_time": self._avg_query_time,
                    "error_count": self._error_count,
                    "error_rate": self._error_count / max(self._total_queries, 1),
                },
                "schema": {
                    "entity_types": list(self._entity_types),
                    "relationship_types": list(self._relationship_types),
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_count": self._error_count,
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph database statistics."""
        try:
            # Detailed statistics query
            stats_query = """
            MATCH (e:Entity)
            OPTIONAL MATCH ()-[r:RELATIONSHIP]->()
            RETURN
                count(DISTINCT e) as total_entities,
                count(DISTINCT r) as total_relationships,
                count(DISTINCT e.entity_type) as unique_entity_types,
                count(DISTINCT r.relationship_type) as unique_relationship_types,
                avg(e.confidence) as avg_entity_confidence,
                avg(r.confidence) as avg_relationship_confidence
            """

            stats = await self._execute_query(stats_query)
            graph_stats = stats[0] if stats else {}

            return {
                "graph_statistics": graph_stats,
                "performance": {
                    "total_queries": self._total_queries,
                    "avg_query_time": self._avg_query_time,
                    "queries_per_second": self._total_queries
                    / max(self._avg_query_time * self._total_queries, 1),
                    "error_count": self._error_count,
                    "error_rate": self._error_count / max(self._total_queries, 1),
                },
                "schema": {
                    "tracked_entity_types": list(self._entity_types),
                    "tracked_relationship_types": list(self._relationship_types),
                },
                "configuration": {
                    "host": self.config.get("host"),
                    "port": self.config.get("port"),
                    "encrypted": self.config.get("encrypted", False),
                },
            }

        except Exception as e:
            logger.error("Failed to get graph stats", error=str(e))
            return {"error": str(e)}

    async def close(self) -> None:
        """Close the graph engine connections."""
        if self.client:
            self.client.close()

        if self.db_manager:
            await self.db_manager.close()

        logger.info(
            "Memgraph engine closed",
            total_queries=self._total_queries,
            total_entities=self._total_entities,
            total_relationships=self._total_relationships,
        )
