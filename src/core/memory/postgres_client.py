"""PostgreSQL client for memory operations with pgvector support."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator
from uuid import UUID, uuid4

import asyncpg
import numpy as np
from asyncpg.pool import Pool
from pydantic import BaseModel, Field

from ...utils.circuit_breaker import circuit_breaker
from ...utils.logger import get_logger
from ..interfaces.vector_store import VectorStore
from ..observability.tracing import trace_method

logger = get_logger(__name__)


class MemoryDocument(BaseModel):
    """Memory document schema."""
    id: UUID = Field(default_factory=uuid4)
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    expires_at: Optional[datetime] = None


class PostgresClient(VectorStore):
    """PostgreSQL client with pgvector for memory storage and retrieval."""
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 20,
        dimensions: int = 1024,
        table_name: str = "memories"
    ):
        """Initialize PostgreSQL client.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            dimensions: Vector dimensions
            table_name: Table name for memories
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.dimensions = dimensions
        self.table_name = table_name
        self._pool: Optional[Pool] = None
        
    async def initialize(self) -> None:
        """Initialize connection pool and ensure database is ready."""
        try:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=self.pool_size,
                command_timeout=60,
                server_settings={
                    'jit': 'off'  # Disable JIT for better pgvector performance
                }
            )
            
            # Ensure pgvector extension is enabled
            async with self._get_connection() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")  # For text search
                
            logger.info(f"PostgreSQL client initialized with pool size {self.pool_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL client: {e}")
            raise
            
    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL connection pool closed")
            
    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[asyncpg.Connection]:
        """Get connection from pool with context manager."""
        if not self._pool:
            raise RuntimeError("PostgreSQL client not initialized. Call initialize() first.")
            
        conn = await self._pool.acquire()
        try:
            yield conn
        finally:
            await self._pool.release(conn)
            
    @trace_method("postgres_store")
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def store(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        agent_id: Optional[str] = None
    ) -> List[str]:
        """Store memories with embeddings.
        
        Args:
            texts: Text content to store
            embeddings: Embeddings for the texts
            metadata: Optional metadata for each text
            agent_id: Optional agent identifier
            
        Returns:
            List of memory IDs
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
            
        metadata = metadata or [{} for _ in texts]
        memory_ids = []
        
        async with self._get_connection() as conn:
            # Use transaction for atomic operation
            async with conn.transaction():
                for text, embedding, meta in zip(texts, embeddings, metadata):
                    memory_id = str(uuid4())
                    
                    # Convert numpy array to list for storage
                    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    
                    await conn.execute(
                        f"""
                        INSERT INTO {self.table_name} 
                        (id, text, embedding, metadata, agent_id, created_at, updated_at)
                        VALUES ($1, $2, $3::vector({self.dimensions}), $4, $5, $6, $6)
                        """,
                        memory_id,
                        text,
                        embedding_list,
                        json.dumps(meta),
                        agent_id,
                        datetime.now(timezone.utc)
                    )
                    
                    memory_ids.append(memory_id)
                    
        logger.info(f"Stored {len(memory_ids)} memories")
        return memory_ids
        
    @trace_method("postgres_search")
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar memories using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters
            agent_id: Optional agent filter
            
        Returns:
            List of (id, score, metadata) tuples
        """
        # Convert numpy array to list
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Build WHERE clause for filters
        where_clauses = []
        params = [query_list, top_k]
        param_count = 2
        
        if agent_id:
            param_count += 1
            where_clauses.append(f"agent_id = ${param_count}")
            params.append(agent_id)
            
        if filters:
            for key, value in filters.items():
                param_count += 1
                where_clauses.append(f"metadata->>{key!r} = ${param_count}")
                params.append(json.dumps(value))
                
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        query = f"""
        SELECT 
            id,
            1 - (embedding <=> $1::vector({self.dimensions})) as similarity,
            text,
            metadata,
            agent_id,
            created_at
        FROM {self.table_name}
        {where_clause}
        ORDER BY embedding <=> $1::vector({self.dimensions})
        LIMIT $2
        """
        
        async with self._get_connection() as conn:
            rows = await conn.fetch(query, *params)
            
        results = []
        for row in rows:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            metadata.update({
                'text': row['text'],
                'agent_id': row['agent_id'],
                'created_at': row['created_at'].isoformat()
            })
            results.append((row['id'], row['similarity'], metadata))
            
        logger.debug(f"Found {len(results)} similar memories")
        return results
        
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        vector_weight: float = 0.7,
        agent_id: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform hybrid search combining vector and text similarity.
        
        Args:
            query_text: Query text for keyword search
            query_embedding: Query embedding for vector search
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            agent_id: Optional agent filter
            
        Returns:
            List of (id, score, metadata) tuples
        """
        text_weight = 1 - vector_weight
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Build agent filter
        agent_filter = "AND agent_id = $4" if agent_id else ""
        params = [query_list, query_text, top_k]
        if agent_id:
            params.append(agent_id)
            
        query = f"""
        WITH vector_search AS (
            SELECT 
                id,
                1 - (embedding <=> $1::vector({self.dimensions})) as vector_similarity
            FROM {self.table_name}
            WHERE 1=1 {agent_filter}
            ORDER BY embedding <=> $1::vector({self.dimensions})
            LIMIT $3 * 2
        ),
        text_search AS (
            SELECT 
                id,
                ts_rank_cd(to_tsvector('english', text), plainto_tsquery('english', $2)) as text_similarity
            FROM {self.table_name}
            WHERE to_tsvector('english', text) @@ plainto_tsquery('english', $2)
            {agent_filter}
            ORDER BY text_similarity DESC
            LIMIT $3 * 2
        ),
        combined AS (
            SELECT 
                COALESCE(v.id, t.id) as id,
                COALESCE(v.vector_similarity, 0) * {vector_weight} + 
                COALESCE(t.text_similarity, 0) * {text_weight} as combined_score,
                COALESCE(v.vector_similarity, 0) as vector_score,
                COALESCE(t.text_similarity, 0) as text_score
            FROM vector_search v
            FULL OUTER JOIN text_search t ON v.id = t.id
        )
        SELECT 
            c.id,
            c.combined_score,
            c.vector_score,
            c.text_score,
            m.text,
            m.metadata,
            m.agent_id,
            m.created_at
        FROM combined c
        JOIN {self.table_name} m ON c.id = m.id
        ORDER BY c.combined_score DESC
        LIMIT $3
        """
        
        async with self._get_connection() as conn:
            rows = await conn.fetch(query, *params)
            
        results = []
        for row in rows:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            metadata.update({
                'text': row['text'],
                'agent_id': row['agent_id'],
                'created_at': row['created_at'].isoformat(),
                'vector_score': float(row['vector_score']),
                'text_score': float(row['text_score'])
            })
            results.append((row['id'], float(row['combined_score']), metadata))
            
        logger.debug(f"Hybrid search found {len(results)} memories")
        return results
        
    async def get(self, memory_id: str) -> Optional[MemoryDocument]:
        """Get a specific memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory document or None if not found
        """
        query = f"""
        SELECT * FROM {self.table_name} WHERE id = $1
        """
        
        async with self._get_connection() as conn:
            row = await conn.fetchrow(query, memory_id)
            
        if not row:
            return None
            
        return MemoryDocument(
            id=UUID(row['id']),
            text=row['text'],
            embedding=list(row['embedding']) if row['embedding'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            agent_id=row['agent_id'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            version=row.get('version', 1),
            expires_at=row.get('expires_at')
        )
        
    async def update(
        self,
        memory_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory.
        
        Args:
            memory_id: Memory ID to update
            text: Optional new text
            embedding: Optional new embedding
            metadata: Optional new metadata
            
        Returns:
            True if updated, False if not found
        """
        updates = []
        params = [memory_id]
        param_count = 1
        
        if text is not None:
            param_count += 1
            updates.append(f"text = ${param_count}")
            params.append(text)
            
        if embedding is not None:
            param_count += 1
            updates.append(f"embedding = ${param_count}::vector({self.dimensions})")
            params.append(embedding)
            
        if metadata is not None:
            param_count += 1
            updates.append(f"metadata = ${param_count}")
            params.append(json.dumps(metadata))
            
        if not updates:
            return True
            
        param_count += 1
        updates.append(f"updated_at = ${param_count}")
        params.append(datetime.now(timezone.utc))
        
        updates.append("version = version + 1")
        
        query = f"""
        UPDATE {self.table_name}
        SET {', '.join(updates)}
        WHERE id = $1
        """
        
        async with self._get_connection() as conn:
            result = await conn.execute(query, *params)
            
        return result.split()[-1] == '1'
        
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory.
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        query = f"DELETE FROM {self.table_name} WHERE id = $1"
        
        async with self._get_connection() as conn:
            result = await conn.execute(query, memory_id)
            
        return result.split()[-1] == '1'
        
    async def delete_expired(self) -> int:
        """Delete expired memories.
        
        Returns:
            Number of deleted memories
        """
        query = f"""
        DELETE FROM {self.table_name}
        WHERE expires_at IS NOT NULL AND expires_at < $1
        """
        
        async with self._get_connection() as conn:
            result = await conn.execute(query, datetime.now(timezone.utc))
            
        count = int(result.split()[-1])
        if count > 0:
            logger.info(f"Deleted {count} expired memories")
            
        return count
        
    async def get_statistics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics.
        
        Args:
            agent_id: Optional agent filter
            
        Returns:
            Statistics dictionary
        """
        agent_filter = "WHERE agent_id = $1" if agent_id else ""
        params = [agent_id] if agent_id else []
        
        query = f"""
        SELECT 
            COUNT(*) as total_memories,
            COUNT(DISTINCT agent_id) as unique_agents,
            MIN(created_at) as oldest_memory,
            MAX(created_at) as newest_memory,
            AVG(LENGTH(text)) as avg_text_length
        FROM {self.table_name}
        {agent_filter}
        """
        
        async with self._get_connection() as conn:
            row = await conn.fetchrow(query, *params)
            
        return {
            'total_memories': row['total_memories'],
            'unique_agents': row['unique_agents'],
            'oldest_memory': row['oldest_memory'].isoformat() if row['oldest_memory'] else None,
            'newest_memory': row['newest_memory'].isoformat() if row['newest_memory'] else None,
            'avg_text_length': float(row['avg_text_length']) if row['avg_text_length'] else 0
        }