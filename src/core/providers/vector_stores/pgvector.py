"""
PostgreSQL with pgvector implementation for vector storage.

High-performance vector store using PostgreSQL with pgvector extension,
optimized for similarity search, hybrid queries, and bulk operations.
"""

import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...interfaces.vector_store import (
    VectorDocument,
    VectorSearchResult,
    VectorStore,
    VectorStoreError,
    VectorStoreInitializationError,
    VectorStoreOperationError,
)
from ...utils.database import PostgreSQLManager
from ...utils.logger import get_logger

logger = get_logger(__name__)


class PgVectorStore(VectorStore):
    """
    Advanced PostgreSQL vector store with pgvector optimization.

    Features:
    - HNSW and IVFFlat index support
    - Hybrid vector + text search
    - Optimized bulk operations
    - Connection pooling and circuit breakers
    - Advanced filtering and metadata queries
    - Performance monitoring and statistics
    """

    def __init__(self):
        self.db_manager: Optional[PostgreSQLManager] = None
        self.config: Dict[str, Any] = {}
        self.table_name: str = "memory_embeddings"
        self.vector_column: str = "embedding"
        self.dimensions: int = 1024
        self.index_type: str = "hnsw"
        self.distance_metric: str = "cosine"

        # Performance tracking
        self._total_queries: int = 0
        self._total_inserts: int = 0
        self._avg_query_time: float = 0.0

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the PostgreSQL vector store."""
        try:
            self.config = config
            self.table_name = config.get("table_name", "memory_embeddings")
            self.vector_column = config.get("vector_column", "embedding")
            self.dimensions = config.get("dimensions", 1024)
            self.index_type = config.get("index_type", "hnsw")
            self.distance_metric = config.get("distance_metric", "cosine")

            # Initialize database manager
            db_config = {
                "host": config.get("host", "localhost"),
                "port": config.get("port", 5432),
                "database": config.get("database", "tyra_memory"),
                "username": config.get("username", "tyra"),
                "password": config.get("password", ""),
                "pool_size": config.get("pool_size", 20),
                "min_connections": config.get("min_connections", 5),
                "max_lifetime": config.get("max_lifetime", 300),
            }

            self.db_manager = PostgreSQLManager(db_config)
            await self.db_manager.initialize()

            # Create tables and indexes
            await self._create_tables()
            await self._create_indexes()

            logger.info(
                "PgVector store initialized",
                table=self.table_name,
                dimensions=self.dimensions,
                index_type=self.index_type,
                distance_metric=self.distance_metric,
            )

        except Exception as e:
            logger.error("Failed to initialize PgVector store", error=str(e))
            raise VectorStoreInitializationError(f"PgVector initialization failed: {e}")

    async def _create_tables(self) -> None:
        """Create necessary tables for vector storage."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            {self.vector_column} vector({self.dimensions}),
            metadata JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            content_hash TEXT,
            chunk_index INTEGER DEFAULT 0,
            source_file TEXT,
            agent_id TEXT,
            session_id TEXT,
            content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        );

        -- Create indexes for efficient queries
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata
            ON {self.table_name} USING gin(metadata);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created_at
            ON {self.table_name} (created_at);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_agent_id
            ON {self.table_name} (agent_id);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_session_id
            ON {self.table_name} (session_id);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_content_hash
            ON {self.table_name} (content_hash);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_fts
            ON {self.table_name} USING gin(content_tsvector);

        -- Create function for updating updated_at
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';

        -- Create trigger for updated_at
        DROP TRIGGER IF EXISTS update_{self.table_name}_updated_at ON {self.table_name};
        CREATE TRIGGER update_{self.table_name}_updated_at
            BEFORE UPDATE ON {self.table_name}
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """

        await self.db_manager.execute_query(create_table_sql, fetch_mode="none")

    async def _create_indexes(self) -> None:
        """Create optimized vector indexes."""
        # Check if vector index already exists
        index_exists_sql = f"""
        SELECT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE tablename = '{self.table_name}'
            AND indexname = 'idx_{self.table_name}_{self.vector_column}_{self.index_type}'
        );
        """

        index_exists = await self.db_manager.execute_query(
            index_exists_sql, fetch_mode="val"
        )

        if not index_exists:
            if self.index_type.lower() == "hnsw":
                # HNSW index for better performance
                index_params = self.config.get(
                    "index_params", {"m": 16, "ef_construction": 64}
                )

                create_index_sql = f"""
                CREATE INDEX idx_{self.table_name}_{self.vector_column}_hnsw
                ON {self.table_name}
                USING hnsw ({self.vector_column} vector_{self._get_distance_operator()})
                WITH (m = {index_params.get('m', 16)}, ef_construction = {index_params.get('ef_construction', 64)});
                """
            else:
                # IVFFlat index
                index_params = self.config.get("index_params", {"lists": 100})

                create_index_sql = f"""
                CREATE INDEX idx_{self.table_name}_{self.vector_column}_ivfflat
                ON {self.table_name}
                USING ivfflat ({self.vector_column} vector_{self._get_distance_operator()})
                WITH (lists = {index_params.get('lists', 100)});
                """

            await self.db_manager.execute_query(create_index_sql, fetch_mode="none")
            logger.info(f"Created {self.index_type} vector index")

    def _get_distance_operator(self) -> str:
        """Get the appropriate distance operator for pgvector."""
        if self.distance_metric == "cosine":
            return "cosine_ops"
        elif self.distance_metric == "l2":
            return "l2_ops"
        elif self.distance_metric == "inner_product":
            return "inner_product_ops"
        else:
            return "cosine_ops"  # Default

    def _get_distance_function(self) -> str:
        """Get the distance function for queries."""
        if self.distance_metric == "cosine":
            return "<=>"
        elif self.distance_metric == "l2":
            return "<->"
        elif self.distance_metric == "inner_product":
            return "<#>"
        else:
            return "<=>"  # Default cosine

    async def store_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Store multiple documents efficiently using bulk insert."""
        if not documents:
            return []

        start_time = time.time()

        try:
            # Prepare bulk insert data
            insert_data = []
            for doc in documents:
                metadata = doc.metadata.copy() if doc.metadata else {}

                # Add system metadata
                metadata.update(
                    {
                        "_stored_at": datetime.utcnow().isoformat(),
                        "_dimensions": len(doc.embedding),
                    }
                )

                insert_data.append(
                    (
                        doc.id,
                        doc.content,
                        doc.embedding.tolist(),  # Convert to list for PostgreSQL
                        json.dumps(metadata),
                        doc.created_at or datetime.utcnow(),
                        doc.updated_at or datetime.utcnow(),
                        self._hash_content(doc.content),
                        metadata.get("chunk_index", 0),
                        metadata.get("source_file"),
                        metadata.get("agent_id"),
                        metadata.get("session_id"),
                    )
                )

            # Bulk insert
            insert_sql = f"""
            INSERT INTO {self.table_name} (
                id, content, {self.vector_column}, metadata, created_at, updated_at,
                content_hash, chunk_index, source_file, agent_id, session_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                {self.vector_column} = EXCLUDED.{self.vector_column},
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at,
                content_hash = EXCLUDED.content_hash,
                chunk_index = EXCLUDED.chunk_index,
                source_file = EXCLUDED.source_file,
                agent_id = EXCLUDED.agent_id,
                session_id = EXCLUDED.session_id
            """

            await self.db_manager.execute_batch(insert_sql, insert_data)

            # Update performance tracking
            self._total_inserts += len(documents)
            insert_time = time.time() - start_time

            logger.info(
                "Stored documents in bulk",
                count=len(documents),
                time=insert_time,
                avg_time_per_doc=insert_time / len(documents),
            )

            return [doc.id for doc in documents]

        except Exception as e:
            logger.error(
                "Failed to store documents", count=len(documents), error=str(e)
            )
            raise VectorStoreOperationError(f"Bulk store failed: {e}")

    async def store_document(self, document: VectorDocument) -> str:
        """Store a single document."""
        result = await self.store_documents([document])
        return result[0] if result else ""

    async def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar documents using vector similarity."""
        start_time = time.time()

        try:
            # Build WHERE clause for filters
            where_clauses = []
            params = [query_embedding.tolist(), top_k]
            param_idx = 3

            if filters:
                for key, value in filters.items():
                    if key == "agent_id":
                        where_clauses.append(f"agent_id = ${param_idx}")
                        params.append(value)
                        param_idx += 1
                    elif key == "session_id":
                        where_clauses.append(f"session_id = ${param_idx}")
                        params.append(value)
                        param_idx += 1
                    elif key.startswith("metadata."):
                        # JSONB query
                        json_key = key[9:]  # Remove "metadata." prefix
                        where_clauses.append(
                            f"metadata ->> '{json_key}' = ${param_idx}"
                        )
                        params.append(str(value))
                        param_idx += 1

            # Add minimum score filter
            score_filter = ""
            if min_score is not None:
                if self.distance_metric == "cosine":
                    # For cosine, lower is better (1 - cosine_similarity)
                    score_filter = f"AND ({self.vector_column} {self._get_distance_function()} $1) <= ${param_idx}"
                    params.append(1 - min_score)
                    param_idx += 1
                else:
                    score_filter = f"AND ({self.vector_column} {self._get_distance_function()} $1) <= ${param_idx}"
                    params.append(min_score)
                    param_idx += 1

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Query for similar vectors
            query_sql = f"""
            SELECT
                id,
                content,
                metadata,
                {self.vector_column},
                ({self.vector_column} {self._get_distance_function()} $1) as distance,
                (1 - ({self.vector_column} {self._get_distance_function()} $1)) as similarity_score,
                created_at,
                updated_at
            FROM {self.table_name}
            WHERE {where_clause} {score_filter}
            ORDER BY {self.vector_column} {self._get_distance_function()} $1
            LIMIT $2
            """

            results = await self.db_manager.execute_query(
                query_sql, *params, fetch_mode="all"
            )

            # Convert to VectorSearchResult objects
            search_results = []
            for row in results:
                # Calculate score based on distance metric
                if self.distance_metric == "cosine":
                    score = 1 - row["distance"]  # Convert distance to similarity
                else:
                    score = 1 / (1 + row["distance"])  # Convert distance to similarity

                search_results.append(
                    VectorSearchResult(
                        id=row["id"],
                        score=score,
                        metadata=row["metadata"] or {},
                        content=row["content"],
                        embedding=(
                            np.array(row[self.vector_column])
                            if row[self.vector_column]
                            else None
                        ),
                    )
                )

            # Update performance tracking
            self._total_queries += 1
            query_time = time.time() - start_time
            self._avg_query_time = (
                self._avg_query_time * (self._total_queries - 1) + query_time
            ) / self._total_queries

            logger.debug(
                "Vector similarity search completed",
                results=len(search_results),
                time=query_time,
                top_k=top_k,
            )

            return search_results

        except Exception as e:
            logger.error(
                "Vector similarity search failed",
                error=str(e),
                top_k=top_k,
                filters=filters,
            )
            raise VectorStoreOperationError(f"Similarity search failed: {e}")

    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        text_query: Optional[str] = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Perform hybrid search combining vector and text search."""
        start_time = time.time()

        try:
            # Build WHERE clause for filters
            where_clauses = []
            params = [query_embedding.tolist()]
            param_idx = 2

            if filters:
                for key, value in filters.items():
                    if key == "agent_id":
                        where_clauses.append(f"agent_id = ${param_idx}")
                        params.append(value)
                        param_idx += 1
                    elif key == "session_id":
                        where_clauses.append(f"session_id = ${param_idx}")
                        params.append(value)
                        param_idx += 1
                    elif key.startswith("metadata."):
                        json_key = key[9:]
                        where_clauses.append(
                            f"metadata ->> '{json_key}' = ${param_idx}"
                        )
                        params.append(str(value))
                        param_idx += 1

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Build hybrid query
            if text_query:
                # Add text search parameter
                params.append(text_query)
                text_param_idx = param_idx
                param_idx += 1

                # Hybrid query with both vector and text search
                query_sql = f"""
                SELECT
                    id,
                    content,
                    metadata,
                    {self.vector_column},
                    ({self.vector_column} {self._get_distance_function()} $1) as vector_distance,
                    ts_rank_cd(content_tsvector, plainto_tsquery('english', ${text_param_idx})) as text_rank,
                    (
                        {vector_weight} * (1 - ({self.vector_column} {self._get_distance_function()} $1)) +
                        {1 - vector_weight} * ts_rank_cd(content_tsvector, plainto_tsquery('english', ${text_param_idx}))
                    ) as hybrid_score,
                    created_at,
                    updated_at
                FROM {self.table_name}
                WHERE {where_clause}
                AND (
                    content_tsvector @@ plainto_tsquery('english', ${text_param_idx})
                    OR ({self.vector_column} {self._get_distance_function()} $1) < 0.8
                )
                ORDER BY hybrid_score DESC
                LIMIT {top_k}
                """
            else:
                # Vector-only search if no text query
                query_sql = f"""
                SELECT
                    id,
                    content,
                    metadata,
                    {self.vector_column},
                    ({self.vector_column} {self._get_distance_function()} $1) as vector_distance,
                    0 as text_rank,
                    (1 - ({self.vector_column} {self._get_distance_function()} $1)) as hybrid_score,
                    created_at,
                    updated_at
                FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY {self.vector_column} {self._get_distance_function()} $1
                LIMIT {top_k}
                """

            results = await self.db_manager.execute_query(
                query_sql, *params, fetch_mode="all"
            )

            # Convert to VectorSearchResult objects
            search_results = []
            for row in results:
                search_results.append(
                    VectorSearchResult(
                        id=row["id"],
                        score=float(row["hybrid_score"]),
                        metadata=row["metadata"] or {},
                        content=row["content"],
                        embedding=(
                            np.array(row[self.vector_column])
                            if row[self.vector_column]
                            else None
                        ),
                    )
                )

            # Update performance tracking
            self._total_queries += 1
            query_time = time.time() - start_time
            self._avg_query_time = (
                self._avg_query_time * (self._total_queries - 1) + query_time
            ) / self._total_queries

            logger.debug(
                "Hybrid search completed",
                results=len(search_results),
                time=query_time,
                has_text_query=bool(text_query),
                vector_weight=vector_weight,
            )

            return search_results

        except Exception as e:
            logger.error(
                "Hybrid search failed",
                error=str(e),
                text_query=text_query,
                vector_weight=vector_weight,
            )
            raise VectorStoreOperationError(f"Hybrid search failed: {e}")

    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Retrieve a document by ID."""
        try:
            query_sql = f"""
            SELECT id, content, {self.vector_column}, metadata, created_at, updated_at
            FROM {self.table_name}
            WHERE id = $1
            """

            result = await self.db_manager.execute_query(
                query_sql, document_id, fetch_mode="one"
            )

            if result:
                return VectorDocument(
                    id=result["id"],
                    content=result["content"],
                    embedding=(
                        np.array(result[self.vector_column])
                        if result[self.vector_column]
                        else np.array([])
                    ),
                    metadata=result["metadata"] or {},
                    created_at=result["created_at"],
                    updated_at=result["updated_at"],
                )

            return None

        except Exception as e:
            logger.error(
                "Failed to get document", document_id=document_id, error=str(e)
            )
            raise VectorStoreOperationError(f"Get document failed: {e}")

    async def update_document(self, document: VectorDocument) -> bool:
        """Update an existing document."""
        try:
            update_sql = f"""
            UPDATE {self.table_name}
            SET content = $2, {self.vector_column} = $3, metadata = $4, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """

            await self.db_manager.execute_query(
                update_sql,
                document.id,
                document.content,
                document.embedding.tolist(),
                json.dumps(document.metadata),
                fetch_mode="none",
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to update document", document_id=document.id, error=str(e)
            )
            raise VectorStoreOperationError(f"Update document failed: {e}")

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        try:
            delete_sql = f"DELETE FROM {self.table_name} WHERE id = $1"
            await self.db_manager.execute_query(
                delete_sql, document_id, fetch_mode="none"
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to delete document", document_id=document_id, error=str(e)
            )
            raise VectorStoreOperationError(f"Delete document failed: {e}")

    async def delete_documents(self, document_ids: List[str]) -> int:
        """Delete multiple documents by ID."""
        if not document_ids:
            return 0

        try:
            # Use ANY for efficient bulk delete
            delete_sql = f"DELETE FROM {self.table_name} WHERE id = ANY($1::text[])"
            await self.db_manager.execute_query(
                delete_sql, document_ids, fetch_mode="none"
            )
            return len(document_ids)

        except Exception as e:
            logger.error(
                "Failed to delete documents", count=len(document_ids), error=str(e)
            )
            raise VectorStoreOperationError(f"Bulk delete failed: {e}")

    async def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the store."""
        try:
            where_clauses = []
            params = []
            param_idx = 1

            if filters:
                for key, value in filters.items():
                    if key == "agent_id":
                        where_clauses.append(f"agent_id = ${param_idx}")
                        params.append(value)
                        param_idx += 1
                    elif key == "session_id":
                        where_clauses.append(f"session_id = ${param_idx}")
                        params.append(value)
                        param_idx += 1

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            count_sql = f"SELECT COUNT(*) FROM {self.table_name} WHERE {where_clause}"

            return await self.db_manager.execute_query(
                count_sql, *params, fetch_mode="val"
            )

        except Exception as e:
            logger.error("Failed to count documents", error=str(e))
            raise VectorStoreOperationError(f"Count documents failed: {e}")

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        """List documents with pagination."""
        try:
            where_clauses = []
            params = []
            param_idx = 1

            if filters:
                for key, value in filters.items():
                    if key == "agent_id":
                        where_clauses.append(f"agent_id = ${param_idx}")
                        params.append(value)
                        param_idx += 1
                    elif key == "session_id":
                        where_clauses.append(f"session_id = ${param_idx}")
                        params.append(value)
                        param_idx += 1

            # Add limit and offset
            params.extend([limit, offset])
            limit_param = param_idx
            offset_param = param_idx + 1

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            list_sql = f"""
            SELECT id, content, {self.vector_column}, metadata, created_at, updated_at
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${limit_param} OFFSET ${offset_param}
            """

            results = await self.db_manager.execute_query(
                list_sql, *params, fetch_mode="all"
            )

            documents = []
            for row in results:
                documents.append(
                    VectorDocument(
                        id=row["id"],
                        content=row["content"],
                        embedding=(
                            np.array(row[self.vector_column])
                            if row[self.vector_column]
                            else np.array([])
                        ),
                        metadata=row["metadata"] or {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )

            return documents

        except Exception as e:
            logger.error("Failed to list documents", error=str(e))
            raise VectorStoreOperationError(f"List documents failed: {e}")

    async def create_index(self, index_params: Optional[Dict[str, Any]] = None) -> bool:
        """Create or rebuild vector index."""
        try:
            # Drop existing index
            drop_sql = f"DROP INDEX IF EXISTS idx_{self.table_name}_{self.vector_column}_{self.index_type}"
            await self.db_manager.execute_query(drop_sql, fetch_mode="none")

            # Create new index with updated parameters
            if index_params:
                self.config["index_params"] = index_params

            await self._create_indexes()
            return True

        except Exception as e:
            logger.error("Failed to create index", error=str(e))
            raise VectorStoreOperationError(f"Index creation failed: {e}")

    def _hash_content(self, content: str) -> str:
        """Create a hash of the content for deduplication."""
        import hashlib

        return hashlib.md5(content.encode()).hexdigest()

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            # Check database connection
            db_health = await self.db_manager.health_check()

            # Check table exists
            table_check_sql = f"""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = '{self.table_name}'
            )
            """
            table_exists = await self.db_manager.execute_query(
                table_check_sql, fetch_mode="val"
            )

            # Check vector extension
            vector_check_sql = (
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            vector_exists = await self.db_manager.execute_query(
                vector_check_sql, fetch_mode="val"
            )

            # Get table stats
            stats_sql = f"""
            SELECT
                COUNT(*) as total_documents,
                AVG(length(content)) as avg_content_length,
                MIN(created_at) as oldest_document,
                MAX(created_at) as newest_document
            FROM {self.table_name}
            """
            stats = await self.db_manager.execute_query(stats_sql, fetch_mode="one")

            return {
                "status": (
                    "healthy"
                    if db_health["status"] == "healthy"
                    and table_exists
                    and vector_exists
                    else "unhealthy"
                ),
                "database": db_health,
                "table_exists": table_exists,
                "vector_extension": vector_exists,
                "table_stats": dict(stats) if stats else {},
                "performance": {
                    "total_queries": self._total_queries,
                    "total_inserts": self._total_inserts,
                    "avg_query_time": self._avg_query_time,
                },
                "config": {
                    "table_name": self.table_name,
                    "dimensions": self.dimensions,
                    "index_type": self.index_type,
                    "distance_metric": self.distance_metric,
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "table_name": self.table_name,
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed vector store statistics."""
        try:
            # Get comprehensive stats
            stats_sql = f"""
            SELECT
                COUNT(*) as total_documents,
                COUNT(DISTINCT agent_id) as unique_agents,
                COUNT(DISTINCT session_id) as unique_sessions,
                AVG(length(content)) as avg_content_length,
                MIN(created_at) as oldest_document,
                MAX(created_at) as newest_document,
                COUNT(*) FILTER (WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as documents_last_24h,
                COUNT(*) FILTER (WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days') as documents_last_7d
            FROM {self.table_name}
            """

            stats = await self.db_manager.execute_query(stats_sql, fetch_mode="one")

            return {
                "table_stats": dict(stats) if stats else {},
                "performance": {
                    "total_queries": self._total_queries,
                    "total_inserts": self._total_inserts,
                    "avg_query_time": self._avg_query_time,
                    "queries_per_second": self._total_queries
                    / max(self._avg_query_time * self._total_queries, 1),
                },
                "configuration": {
                    "table_name": self.table_name,
                    "vector_column": self.vector_column,
                    "dimensions": self.dimensions,
                    "index_type": self.index_type,
                    "distance_metric": self.distance_metric,
                },
            }

        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return {"error": str(e)}

    async def close(self) -> None:
        """Close the vector store."""
        if self.db_manager:
            await self.db_manager.close()

        logger.info(
            "PgVector store closed",
            total_queries=self._total_queries,
            total_inserts=self._total_inserts,
        )
