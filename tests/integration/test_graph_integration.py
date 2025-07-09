"""
Integration tests for graph system including Memgraph and Graphiti.

Tests the complete graph pipeline from entity extraction to temporal queries.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.graph.graphiti_integration import GraphitiManager
from src.core.graph.memgraph_client import MemgraphClient
from src.core.providers.graph_engines.memgraph import MemgraphEngine
from src.core.interfaces.graph_engine import Entity, Relationship, RelationshipType


class TestGraphIntegration:
    """Integration tests for the graph system."""
    
    @pytest.fixture
    def mock_memgraph_engine(self):
        """Create mock Memgraph engine for testing."""
        engine = AsyncMock(spec=MemgraphEngine)
        
        # Mock entity operations
        engine.create_entity.return_value = "entity_123"
        engine.get_entity.return_value = Entity(
            id="entity_123",
            name="Test Entity",
            entity_type="concept",
            properties={"test": True},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            confidence=0.95
        )
        
        engine.find_entities.return_value = [
            Entity(
                id="entity_1",
                name="Python",
                entity_type="programming_language",
                properties={"paradigm": "multi-paradigm"},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                confidence=0.98
            ),
            Entity(
                id="entity_2",
                name="Guido van Rossum",
                entity_type="person",
                properties={"role": "creator"},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                confidence=0.97
            )
        ]
        
        # Mock relationship operations
        engine.create_relationship.return_value = "rel_123"
        engine.get_entity_relationships.return_value = [
            Relationship(
                id="rel_1",
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                relationship_type=RelationshipType.CREATED_BY,
                properties={"year": 1991},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                confidence=0.96
            )
        ]
        
        # Mock connected entities
        engine.get_connected_entities.return_value = [
            Entity(
                id="entity_3",
                name="BDFL",
                entity_type="title",
                properties={"meaning": "Benevolent Dictator for Life"},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                confidence=0.90
            )
        ]
        
        # Mock path finding
        engine.find_path.return_value = [
            Relationship(
                id="rel_path_1",
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                relationship_type=RelationshipType.CREATED_BY,
                properties={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                confidence=0.96
            )
        ]
        
        # Mock health check
        engine.health_check.return_value = {
            "status": "healthy",
            "response_time": 0.023,
            "graph_stats": {
                "total_entities": 150,
                "total_relationships": 300
            }
        }
        
        return engine
    
    @pytest.fixture
    def mock_graphiti_manager(self):
        """Create mock Graphiti manager for testing."""
        manager = AsyncMock(spec=GraphitiManager)
        
        # Mock episode addition
        manager.add_episode.return_value = {
            "episode_id": "episode_123",
            "content": "Test episode content",
            "entities_extracted": ["Python", "Guido van Rossum"],
            "relationships_created": ["Python created_by Guido van Rossum"],
            "processing_time": 0.125
        }
        
        # Mock search
        manager.search.return_value = [
            {
                "id": "search_result_1",
                "content": "Python was created by Guido van Rossum",
                "score": 0.94,
                "source": "episode_123",
                "valid_from": datetime.utcnow() - timedelta(days=1),
                "valid_to": None,
                "metadata": {"confidence": 0.94}
            }
        ]
        
        # Mock entity timeline
        manager.get_entity_timeline.return_value = [
            {
                "id": "timeline_1",
                "entity_name": "Python",
                "content": "Python was created in 1991",
                "timestamp": datetime.utcnow() - timedelta(days=365*30),
                "source": "historical_data",
                "confidence": 0.98
            }
        ]
        
        # Mock related entities
        manager.get_related_entities.return_value = {
            "entity_name": "Python",
            "entities": [
                {
                    "id": "entity_guido",
                    "name": "Guido van Rossum",
                    "type": "person",
                    "confidence": 0.97
                }
            ],
            "relationships": [
                {
                    "id": "rel_created_by",
                    "source_id": "entity_python",
                    "target_id": "entity_guido",
                    "type": "created_by",
                    "confidence": 0.96
                }
            ]
        }
        
        # Mock fact validation
        manager.validate_facts.return_value = {
            "entity_name": "Python",
            "validated_facts": [
                {
                    "fact": "Python is a programming language",
                    "is_valid": True,
                    "confidence": 0.98,
                    "contradictions": [],
                    "supporting_evidence": ["Python documentation", "Programming language taxonomy"]
                }
            ],
            "overall_validity": True
        }
        
        # Mock health check
        manager.health_check.return_value = {
            "status": "healthy",
            "response_time": 0.034,
            "graph_stats": {
                "entities": 200,
                "relationships": 450,
                "episodes": 50
            }
        }
        
        return manager
    
    @pytest.mark.asyncio
    async def test_memgraph_client_initialization(self, mock_memgraph_engine):
        """Test Memgraph client initialization."""
        client = MemgraphClient()
        
        config = {
            "engine_provider": "memgraph",
            "host": "localhost",
            "port": 7687
        }
        
        with patch('src.core.utils.registry.get_provider', return_value=mock_memgraph_engine):
            await client.initialize(config)
            
            assert client.engine is mock_memgraph_engine
            assert client.config == config
    
    @pytest.mark.asyncio
    async def test_entity_operations_through_client(self, mock_memgraph_engine):
        """Test entity operations through the Memgraph client."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        # Test get entity
        entity = await client.get_entity("entity_123")
        
        assert entity is not None
        assert entity["id"] == "entity_123"
        assert entity["name"] == "Test Entity"
        assert entity["type"] == "concept"
        assert entity["confidence"] == 0.95
        
        # Test list entities
        entities = await client.list_entities(
            entity_type="programming_language",
            search="Python",
            limit=10
        )
        
        assert len(entities) == 2
        assert entities[0]["name"] == "Python"
        assert entities[1]["name"] == "Guido van Rossum"
        
        # Verify engine was called correctly
        mock_memgraph_engine.get_entity.assert_called_with("entity_123")
        mock_memgraph_engine.find_entities.assert_called_with(
            entity_type="programming_language",
            properties={"name": "Python"},
            limit=10
        )
    
    @pytest.mark.asyncio
    async def test_relationship_operations_through_client(self, mock_memgraph_engine):
        """Test relationship operations through the Memgraph client."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        # Test get relationships
        relationships = await client.get_relationships(
            entity_id="entity_1",
            relationship_type="created_by",
            direction="both"
        )
        
        assert len(relationships) == 1
        assert relationships[0]["source_id"] == "entity_1"
        assert relationships[0]["target_id"] == "entity_2"
        assert relationships[0]["type"] == "created_by"
        
        # Verify engine was called correctly
        mock_memgraph_engine.get_entity_relationships.assert_called_with(
            entity_id="entity_1",
            relationship_type="created_by",
            direction="both"
        )
    
    @pytest.mark.asyncio
    async def test_graph_traversal_operations(self, mock_memgraph_engine):
        """Test graph traversal operations."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        # Test get neighbors
        neighbors = await client.get_neighbors(
            entity_ids=["entity_1"],
            depth=2
        )
        
        assert "nodes" in neighbors
        assert "edges" in neighbors
        
        # Test find paths
        paths = await client.find_paths(
            start_id="entity_1",
            end_id="entity_2",
            max_depth=3
        )
        
        assert "nodes" in paths
        assert "edges" in paths
        
        # Test get subgraph
        subgraph = await client.get_subgraph(
            entity_ids=["entity_1", "entity_2"],
            depth=1
        )
        
        assert "nodes" in subgraph
        assert "edges" in subgraph
        
        # Verify engine calls
        mock_memgraph_engine.get_connected_entities.assert_called()
        mock_memgraph_engine.find_path.assert_called_with(
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            max_depth=3
        )
    
    @pytest.mark.asyncio
    async def test_temporal_graph_queries(self, mock_memgraph_engine):
        """Test temporal graph queries."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        start_time = datetime.utcnow() - timedelta(days=30)
        end_time = datetime.utcnow()
        
        # Test temporal query
        results = await client.temporal_query(
            start_time=start_time,
            end_time=end_time,
            entity_types=["person"],
            relationship_types=["created_by"],
            limit=50
        )
        
        assert "nodes" in results
        assert "edges" in results
        assert "temporal_range" in results
        assert results["temporal_range"]["start"] == start_time
        assert results["temporal_range"]["end"] == end_time
        
        # Verify engine was called
        mock_memgraph_engine.find_entities.assert_called()
    
    @pytest.mark.asyncio
    async def test_graph_statistics(self, mock_memgraph_engine):
        """Test graph statistics retrieval."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        # Mock stats response
        mock_memgraph_engine.get_stats.return_value = {
            "graph_statistics": {
                "total_entities": 150,
                "total_relationships": 300,
                "unique_entity_types": 10,
                "unique_relationship_types": 8
            },
            "schema": {
                "tracked_entity_types": ["person", "concept", "programming_language"],
                "tracked_relationship_types": ["created_by", "related_to", "part_of"]
            }
        }
        
        stats = await client.get_statistics()
        
        assert stats["total_entities"] == 150
        assert stats["total_relationships"] == 300
        assert len(stats["entity_types"]) == 3
        assert len(stats["relationship_types"]) == 3
        assert stats["avg_relationships_per_entity"] == 2.0
        
        # Verify engine was called
        mock_memgraph_engine.get_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graphiti_episode_management(self, mock_graphiti_manager):
        """Test Graphiti episode management."""
        manager = mock_graphiti_manager
        
        # Test add episode
        result = await manager.add_episode(
            episode_id="test_episode",
            content="Python was created by Guido van Rossum in 1991",
            source="test_source",
            timestamp=datetime.utcnow(),
            metadata={"type": "historical_fact"}
        )
        
        assert result["episode_id"] == "test_episode"
        assert "entities_extracted" in result
        assert "relationships_created" in result
        assert result["processing_time"] == 0.125
        
        # Test search
        search_results = await manager.search(
            query="Python creator",
            center_node_distance=2,
            use_hybrid_search=True,
            limit=10
        )
        
        assert len(search_results) == 1
        assert search_results[0]["content"] == "Python was created by Guido van Rossum"
        assert search_results[0]["score"] == 0.94
        
        # Verify calls
        manager.add_episode.assert_called_once()
        manager.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graphiti_entity_timeline(self, mock_graphiti_manager):
        """Test Graphiti entity timeline functionality."""
        manager = mock_graphiti_manager
        
        start_date = datetime.utcnow() - timedelta(days=365*35)
        end_date = datetime.utcnow()
        
        # Test get entity timeline
        timeline = await manager.get_entity_timeline(
            entity_name="Python",
            start_date=start_date,
            end_date=end_date,
            limit=20
        )
        
        assert len(timeline) == 1
        assert timeline[0]["entity_name"] == "Python"
        assert timeline[0]["content"] == "Python was created in 1991"
        assert timeline[0]["confidence"] == 0.98
        
        # Test get related entities
        related = await manager.get_related_entities(
            entity_name="Python",
            relationship_types=["created_by"],
            depth=1
        )
        
        assert related["entity_name"] == "Python"
        assert len(related["entities"]) == 1
        assert related["entities"][0]["name"] == "Guido van Rossum"
        assert len(related["relationships"]) == 1
        
        # Verify calls
        manager.get_entity_timeline.assert_called_once()
        manager.get_related_entities.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graphiti_fact_validation(self, mock_graphiti_manager):
        """Test Graphiti fact validation."""
        manager = mock_graphiti_manager
        
        # Test validate facts
        validation_result = await manager.validate_facts(
            entity_name="Python",
            new_facts=["Python is a programming language"],
            reference_time=datetime.utcnow()
        )
        
        assert validation_result["entity_name"] == "Python"
        assert validation_result["overall_validity"] is True
        assert len(validation_result["validated_facts"]) == 1
        
        fact = validation_result["validated_facts"][0]
        assert fact["fact"] == "Python is a programming language"
        assert fact["is_valid"] is True
        assert fact["confidence"] == 0.98
        assert len(fact["contradictions"]) == 0
        assert len(fact["supporting_evidence"]) == 2
        
        # Verify call
        manager.validate_facts.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graph_health_monitoring(self, mock_memgraph_engine, mock_graphiti_manager):
        """Test graph system health monitoring."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        # Test Memgraph health
        memgraph_health = await mock_memgraph_engine.health_check()
        
        assert memgraph_health["status"] == "healthy"
        assert memgraph_health["response_time"] == 0.023
        assert memgraph_health["graph_stats"]["total_entities"] == 150
        
        # Test Graphiti health
        graphiti_health = await mock_graphiti_manager.health_check()
        
        assert graphiti_health["status"] == "healthy"
        assert graphiti_health["response_time"] == 0.034
        assert graphiti_health["graph_stats"]["entities"] == 200
        
        # Verify calls
        mock_memgraph_engine.health_check.assert_called_once()
        mock_graphiti_manager.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integrated_graph_workflow(self, mock_memgraph_engine, mock_graphiti_manager):
        """Test integrated workflow using both Memgraph and Graphiti."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        # 1. Add episode to Graphiti
        episode_result = await mock_graphiti_manager.add_episode(
            episode_id="workflow_episode",
            content="Python programming language was created by Guido van Rossum",
            source="integration_test",
            timestamp=datetime.utcnow()
        )
        
        assert episode_result["episode_id"] == "workflow_episode"
        
        # 2. Search using Graphiti
        search_results = await mock_graphiti_manager.search(
            query="Python programming language",
            limit=5
        )
        
        assert len(search_results) == 1
        
        # 3. Get entity information from Memgraph
        python_entity = await client.get_entity("entity_1")
        assert python_entity["name"] == "Python"
        
        # 4. Get relationships from Memgraph
        relationships = await client.get_relationships(
            entity_id="entity_1",
            relationship_type="created_by"
        )
        
        assert len(relationships) == 1
        assert relationships[0]["type"] == "created_by"
        
        # 5. Validate facts using Graphiti
        validation_result = await mock_graphiti_manager.validate_facts(
            entity_name="Python",
            new_facts=["Python is a programming language"]
        )
        
        assert validation_result["overall_validity"] is True
        
        # 6. Get entity timeline from Graphiti
        timeline = await mock_graphiti_manager.get_entity_timeline(
            entity_name="Python",
            start_date=datetime.utcnow() - timedelta(days=365*35),
            end_date=datetime.utcnow()
        )
        
        assert len(timeline) == 1
        
        # Verify all components were called
        mock_graphiti_manager.add_episode.assert_called_once()
        mock_graphiti_manager.search.assert_called_once()
        mock_memgraph_engine.get_entity.assert_called_once()
        mock_memgraph_engine.get_entity_relationships.assert_called_once()
        mock_graphiti_manager.validate_facts.assert_called_once()
        mock_graphiti_manager.get_entity_timeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graph_error_handling(self, mock_memgraph_engine, mock_graphiti_manager):
        """Test error handling in graph operations."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        # Test Memgraph error handling
        mock_memgraph_engine.get_entity.side_effect = Exception("Memgraph connection failed")
        
        entity = await client.get_entity("entity_123")
        assert entity is None  # Should return None on error
        
        # Test Graphiti error handling
        mock_graphiti_manager.search.side_effect = Exception("Graphiti search failed")
        
        with pytest.raises(Exception, match="Graphiti search failed"):
            await mock_graphiti_manager.search("test query")
        
        # Reset for health check test
        mock_memgraph_engine.get_entity.side_effect = None
        mock_graphiti_manager.search.side_effect = None
        
        # Test health check with errors
        mock_memgraph_engine.health_check.return_value = {
            "status": "unhealthy",
            "error": "Connection timeout"
        }
        
        health = await mock_memgraph_engine.health_check()
        assert health["status"] == "unhealthy"
        assert "error" in health
    
    @pytest.mark.asyncio
    async def test_graph_performance_monitoring(self, mock_memgraph_engine, mock_graphiti_manager):
        """Test performance monitoring in graph operations."""
        client = MemgraphClient()
        client.engine = mock_memgraph_engine
        
        # Mock performance metrics
        mock_memgraph_engine.get_stats.return_value = {
            "performance": {
                "total_queries": 1000,
                "avg_query_time": 0.045,
                "queries_per_second": 22.2,
                "error_count": 5,
                "error_rate": 0.005
            }
        }
        
        stats = await mock_memgraph_engine.get_stats()
        
        assert stats["performance"]["total_queries"] == 1000
        assert stats["performance"]["avg_query_time"] == 0.045
        assert stats["performance"]["error_rate"] == 0.005
        
        # Test Graphiti performance
        mock_graphiti_manager.get_statistics.return_value = {
            "performance": {
                "avg_response_time": 0.12,
                "error_count": 2,
                "error_rate": 0.002
            }
        }
        
        graphiti_stats = await mock_graphiti_manager.get_statistics()
        
        assert graphiti_stats["performance"]["avg_response_time"] == 0.12
        assert graphiti_stats["performance"]["error_rate"] == 0.002


if __name__ == "__main__":
    pytest.main([__file__, "-v"])