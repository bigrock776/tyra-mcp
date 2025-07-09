"""
Comprehensive unit tests for Graph Engine.

Tests entity extraction, relationship mapping, graph queries, and Memgraph integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from src.core.graph.engine import GraphEngine, EntityExtractionRequest, GraphQueryRequest


class TestGraphEngine:
    """Test Graph Engine functionality."""

    @pytest.fixture
    async def graph_engine(self):
        """Create graph engine with mocked dependencies."""
        with patch('src.core.graph.engine.GQLAlchemy') as mock_gql:
            engine = GraphEngine()
            engine.memgraph_client = AsyncMock()
            engine.entity_extractor = AsyncMock()
            engine._initialized = True
            
            yield engine

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self, graph_engine):
        """Test basic entity extraction from text."""
        # Setup
        text = "Apple Inc. released the iPhone in 2007 in Cupertino, California."
        request = EntityExtractionRequest(text=text, extract_types=["ORGANIZATION", "PRODUCT", "DATE", "LOCATION"])
        
        # Mock entity extraction
        mock_entities = [
            {"name": "Apple Inc.", "type": "ORGANIZATION", "confidence": 0.95, "start": 0, "end": 10},
            {"name": "iPhone", "type": "PRODUCT", "confidence": 0.92, "start": 24, "end": 30},
            {"name": "2007", "type": "DATE", "confidence": 0.88, "start": 34, "end": 38},
            {"name": "Cupertino", "type": "LOCATION", "confidence": 0.90, "start": 42, "end": 51},
            {"name": "California", "type": "LOCATION", "confidence": 0.87, "start": 53, "end": 63}
        ]
        graph_engine.entity_extractor.extract.return_value = mock_entities
        
        # Execute
        result = await graph_engine.extract_entities(request)
        
        # Verify
        assert len(result.entities) == 5
        assert result.entities[0]["name"] == "Apple Inc."
        assert result.entities[0]["type"] == "ORGANIZATION"
        assert result.entities[1]["name"] == "iPhone"
        assert result.entities[1]["type"] == "PRODUCT"
        assert all(e["confidence"] > 0.8 for e in result.entities)

    @pytest.mark.asyncio
    async def test_extract_entities_with_confidence_filtering(self, graph_engine):
        """Test entity extraction with confidence threshold filtering."""
        # Setup
        text = "The quick brown fox jumps over the lazy dog."
        request = EntityExtractionRequest(
            text=text,
            min_confidence=0.9,
            extract_types=["ANIMAL"]
        )
        
        # Mock entities with varying confidence
        mock_entities = [
            {"name": "fox", "type": "ANIMAL", "confidence": 0.95},
            {"name": "dog", "type": "ANIMAL", "confidence": 0.85},  # Below threshold
            {"name": "animal", "type": "ANIMAL", "confidence": 0.92}
        ]
        graph_engine.entity_extractor.extract.return_value = mock_entities
        
        # Execute
        result = await graph_engine.extract_entities(request)
        
        # Verify only high-confidence entities
        assert len(result.entities) == 2
        assert all(e["confidence"] >= 0.9 for e in result.entities)
        assert result.filtered_count == 1

    @pytest.mark.asyncio
    async def test_create_relationships_from_entities(self, graph_engine):
        """Test relationship creation between extracted entities."""
        # Setup
        entities = [
            {"name": "Apple Inc.", "type": "ORGANIZATION"},
            {"name": "iPhone", "type": "PRODUCT"},
            {"name": "Steve Jobs", "type": "PERSON"},
            {"name": "Cupertino", "type": "LOCATION"}
        ]
        
        # Mock relationship extraction
        mock_relationships = [
            {"source": "Apple Inc.", "target": "iPhone", "relation": "CREATED", "confidence": 0.9},
            {"source": "Steve Jobs", "target": "Apple Inc.", "relation": "FOUNDED", "confidence": 0.85},
            {"source": "Apple Inc.", "target": "Cupertino", "relation": "LOCATED_IN", "confidence": 0.8}
        ]
        graph_engine._extract_relationships = AsyncMock(return_value=mock_relationships)
        
        # Execute
        result = await graph_engine.create_relationships(entities)
        
        # Verify
        assert len(result) == 3
        assert result[0]["relation"] == "CREATED"
        assert result[1]["relation"] == "FOUNDED"
        assert result[2]["relation"] == "LOCATED_IN"

    @pytest.mark.asyncio
    async def test_store_entities_in_graph(self, graph_engine):
        """Test storing entities in Memgraph."""
        # Setup
        entities = [
            {"name": "Python", "type": "TECHNOLOGY", "confidence": 0.95},
            {"name": "Programming", "type": "CONCEPT", "confidence": 0.88}
        ]
        relationships = [
            {"source": "Python", "target": "Programming", "relation": "IS_A", "confidence": 0.9}
        ]
        
        # Mock Memgraph operations
        graph_engine.memgraph_client.execute_query.return_value = {"nodes_created": 2, "relationships_created": 1}
        
        # Execute
        result = await graph_engine.store_entities(entities, relationships)
        
        # Verify
        assert result["nodes_created"] == 2
        assert result["relationships_created"] == 1
        graph_engine.memgraph_client.execute_query.assert_called()

    @pytest.mark.asyncio
    async def test_query_by_entity(self, graph_engine):
        """Test querying graph by specific entity."""
        # Setup
        entity_name = "Python"
        max_depth = 2
        
        # Mock query results
        mock_query_result = [
            {
                "path": ["Python", "IS_A", "Programming Language"],
                "nodes": [
                    {"name": "Python", "type": "TECHNOLOGY"},
                    {"name": "Programming Language", "type": "CONCEPT"}
                ],
                "relationships": [{"type": "IS_A", "confidence": 0.9}]
            },
            {
                "path": ["Python", "USED_FOR", "Data Science"],
                "nodes": [
                    {"name": "Python", "type": "TECHNOLOGY"},
                    {"name": "Data Science", "type": "FIELD"}
                ],
                "relationships": [{"type": "USED_FOR", "confidence": 0.85}]
            }
        ]
        graph_engine.memgraph_client.execute_query.return_value = mock_query_result
        
        # Execute
        result = await graph_engine.query_by_entity(entity_name, max_depth=max_depth)
        
        # Verify
        assert len(result) == 2
        assert result[0]["path"][0] == "Python"
        assert result[1]["path"][0] == "Python"
        assert all("relationships" in path for path in result)

    @pytest.mark.asyncio
    async def test_find_connections_between_entities(self, graph_engine):
        """Test finding connections between two entities."""
        # Setup
        source_entity = "Python"
        target_entity = "Machine Learning"
        max_path_length = 3
        
        # Mock connection paths
        mock_paths = [
            {
                "path": ["Python", "USED_FOR", "Data Science", "ENABLES", "Machine Learning"],
                "length": 2,
                "confidence": 0.8
            },
            {
                "path": ["Python", "HAS_LIBRARY", "scikit-learn", "IMPLEMENTS", "Machine Learning"],
                "length": 2,
                "confidence": 0.9
            }
        ]
        graph_engine.memgraph_client.execute_query.return_value = mock_paths
        
        # Execute
        result = await graph_engine.find_connections(source_entity, target_entity, max_path_length)
        
        # Verify
        assert len(result) == 2
        assert result[0]["path"][0] == "Python"
        assert result[0]["path"][-1] == "Machine Learning"
        assert result[1]["confidence"] > result[0]["confidence"]  # Higher confidence first

    @pytest.mark.asyncio
    async def test_temporal_query_support(self, graph_engine):
        """Test temporal queries for time-based relationships."""
        # Setup
        request = GraphQueryRequest(
            query_type="temporal",
            entities=["Apple Inc."],
            time_range={"start": "2000-01-01", "end": "2010-12-31"},
            relation_types=["RELEASED", "ACQUIRED"]
        )
        
        # Mock temporal results
        mock_temporal_results = [
            {
                "entity": "Apple Inc.",
                "event": "RELEASED",
                "target": "iPod",
                "timestamp": "2001-10-23",
                "confidence": 0.95
            },
            {
                "entity": "Apple Inc.",
                "event": "RELEASED", 
                "target": "iPhone",
                "timestamp": "2007-06-29",
                "confidence": 0.98
            }
        ]
        graph_engine.memgraph_client.execute_query.return_value = mock_temporal_results
        
        # Execute
        result = await graph_engine.temporal_query(request)
        
        # Verify
        assert len(result) == 2
        assert result[0]["timestamp"] == "2001-10-23"
        assert result[1]["timestamp"] == "2007-06-29"
        assert all(r["entity"] == "Apple Inc." for r in result)

    @pytest.mark.asyncio
    async def test_graph_traversal_with_filters(self, graph_engine):
        """Test graph traversal with relationship type filters."""
        # Setup
        start_entity = "Python"
        traversal_filters = {
            "relation_types": ["IS_A", "USED_FOR"],
            "node_types": ["TECHNOLOGY", "CONCEPT"],
            "max_depth": 3,
            "min_confidence": 0.8
        }
        
        # Mock filtered traversal results
        mock_traversal = [
            {
                "node": {"name": "Python", "type": "TECHNOLOGY"},
                "depth": 0,
                "path": ["Python"]
            },
            {
                "node": {"name": "Programming Language", "type": "CONCEPT"},
                "depth": 1,
                "path": ["Python", "IS_A", "Programming Language"],
                "confidence": 0.9
            },
            {
                "node": {"name": "Data Science", "type": "FIELD"},
                "depth": 1,
                "path": ["Python", "USED_FOR", "Data Science"],
                "confidence": 0.85
            }
        ]
        graph_engine.memgraph_client.execute_query.return_value = mock_traversal
        
        # Execute
        result = await graph_engine.traverse_graph(start_entity, traversal_filters)
        
        # Verify
        assert len(result) == 3
        assert result[0]["depth"] == 0
        assert result[1]["depth"] == 1
        assert all(r.get("confidence", 1.0) >= 0.8 for r in result)

    @pytest.mark.asyncio
    async def test_entity_similarity_search(self, graph_engine):
        """Test finding similar entities in the graph."""
        # Setup
        query_entity = {"name": "Python", "type": "TECHNOLOGY"}
        similarity_threshold = 0.8
        
        # Mock similarity results
        mock_similar_entities = [
            {"name": "Java", "type": "TECHNOLOGY", "similarity": 0.92},
            {"name": "JavaScript", "type": "TECHNOLOGY", "similarity": 0.85},
            {"name": "C++", "type": "TECHNOLOGY", "similarity": 0.82},
            {"name": "Ruby", "type": "TECHNOLOGY", "similarity": 0.75}  # Below threshold
        ]
        graph_engine._calculate_entity_similarity = AsyncMock(return_value=mock_similar_entities)
        
        # Execute
        result = await graph_engine.find_similar_entities(query_entity, similarity_threshold)
        
        # Verify
        assert len(result) == 3  # Ruby filtered out
        assert all(e["similarity"] >= similarity_threshold for e in result)
        assert result[0]["name"] == "Java"  # Highest similarity first

    @pytest.mark.asyncio
    async def test_graph_statistics(self, graph_engine):
        """Test graph statistics and metrics."""
        # Mock statistics
        mock_stats = {
            "total_nodes": 1000,
            "total_relationships": 2500,
            "node_types": {
                "TECHNOLOGY": 300,
                "PERSON": 200,
                "ORGANIZATION": 150,
                "CONCEPT": 350
            },
            "relationship_types": {
                "IS_A": 800,
                "USED_FOR": 600,
                "CREATED": 400,
                "WORKS_FOR": 300,
                "LOCATED_IN": 400
            },
            "connectivity": {
                "average_degree": 5.0,
                "clustering_coefficient": 0.3,
                "connected_components": 1
            }
        }
        graph_engine.memgraph_client.execute_query.return_value = mock_stats
        
        # Execute
        result = await graph_engine.get_graph_statistics()
        
        # Verify
        assert result["total_nodes"] == 1000
        assert result["total_relationships"] == 2500
        assert "node_types" in result
        assert "relationship_types" in result
        assert "connectivity" in result

    @pytest.mark.asyncio
    async def test_batch_entity_processing(self, graph_engine):
        """Test batch processing of multiple entity extraction requests."""
        # Setup
        texts = [
            "Apple released iPhone in 2007",
            "Google founded by Larry Page and Sergey Brin",
            "Microsoft develops Windows operating system"
        ]
        
        # Mock batch extraction
        batch_results = [
            [{"name": "Apple", "type": "ORGANIZATION"}, {"name": "iPhone", "type": "PRODUCT"}],
            [{"name": "Google", "type": "ORGANIZATION"}, {"name": "Larry Page", "type": "PERSON"}],
            [{"name": "Microsoft", "type": "ORGANIZATION"}, {"name": "Windows", "type": "PRODUCT"}]
        ]
        graph_engine.entity_extractor.batch_extract.return_value = batch_results
        
        # Execute
        result = await graph_engine.batch_extract_entities(texts)
        
        # Verify
        assert len(result) == 3
        assert len(result[0]) == 2
        assert result[0][0]["name"] == "Apple"
        assert result[1][1]["name"] == "Larry Page"

    @pytest.mark.asyncio
    async def test_memory_integration(self, graph_engine):
        """Test integration between graph entities and memory storage."""
        # Setup
        memory_id = "memory_123"
        memory_content = "Python is used for machine learning and data science applications."
        
        # Mock entity extraction and memory linking
        extracted_entities = [
            {"name": "Python", "type": "TECHNOLOGY"},
            {"name": "machine learning", "type": "CONCEPT"},
            {"name": "data science", "type": "FIELD"}
        ]
        graph_engine.entity_extractor.extract.return_value = extracted_entities
        graph_engine.memgraph_client.execute_query.return_value = {"memory_links_created": 3}
        
        # Execute
        result = await graph_engine.link_memory_to_entities(memory_id, memory_content)
        
        # Verify
        assert result["memory_id"] == memory_id
        assert result["entities_linked"] == 3
        assert result["memory_links_created"] == 3

    @pytest.mark.asyncio
    async def test_entity_clustering(self, graph_engine):
        """Test entity clustering for knowledge organization."""
        # Setup
        clustering_request = {
            "entity_types": ["TECHNOLOGY"],
            "clustering_method": "community_detection",
            "min_cluster_size": 3
        }
        
        # Mock clustering results
        mock_clusters = [
            {
                "cluster_id": "prog_languages",
                "entities": ["Python", "Java", "JavaScript", "C++"],
                "centroid": "Programming Languages",
                "cohesion_score": 0.85
            },
            {
                "cluster_id": "databases",
                "entities": ["PostgreSQL", "MySQL", "MongoDB"],
                "centroid": "Database Systems",
                "cohesion_score": 0.78
            }
        ]
        graph_engine._perform_clustering = AsyncMock(return_value=mock_clusters)
        
        # Execute
        result = await graph_engine.cluster_entities(clustering_request)
        
        # Verify
        assert len(result) == 2
        assert result[0]["cluster_id"] == "prog_languages"
        assert len(result[0]["entities"]) == 4
        assert result[0]["cohesion_score"] > result[1]["cohesion_score"]

    @pytest.mark.asyncio
    async def test_graph_validation(self, graph_engine):
        """Test graph consistency and validation."""
        # Mock validation results
        mock_validation = {
            "total_nodes_checked": 1000,
            "total_relationships_checked": 2500,
            "issues_found": [
                {"type": "orphaned_node", "node": "Node_123", "severity": "low"},
                {"type": "invalid_relationship", "relationship": "Rel_456", "severity": "medium"}
            ],
            "consistency_score": 0.92,
            "validation_passed": True
        }
        graph_engine.memgraph_client.execute_query.return_value = mock_validation
        
        # Execute
        result = await graph_engine.validate_graph_consistency()
        
        # Verify
        assert result["validation_passed"] is True
        assert result["consistency_score"] == 0.92
        assert len(result["issues_found"]) == 2

    @pytest.mark.asyncio
    async def test_graph_backup_and_restore(self, graph_engine):
        """Test graph backup and restore functionality."""
        # Mock backup operation
        backup_result = {
            "backup_id": "backup_20231201_123000",
            "nodes_exported": 1000,
            "relationships_exported": 2500,
            "file_size_mb": 15.5,
            "backup_path": "/backups/graph_backup_20231201_123000.cypher"
        }
        graph_engine.memgraph_client.create_backup.return_value = backup_result
        
        # Execute backup
        backup_info = await graph_engine.create_backup()
        
        # Verify backup
        assert backup_info["nodes_exported"] == 1000
        assert backup_info["relationships_exported"] == 2500
        assert "backup_id" in backup_info
        
        # Mock restore operation
        restore_result = {
            "restore_id": "restore_20231201_123000",
            "nodes_imported": 1000,
            "relationships_imported": 2500,
            "status": "success"
        }
        graph_engine.memgraph_client.restore_backup.return_value = restore_result
        
        # Execute restore
        restore_info = await graph_engine.restore_backup(backup_info["backup_id"])
        
        # Verify restore
        assert restore_info["status"] == "success"
        assert restore_info["nodes_imported"] == 1000

    @pytest.mark.asyncio
    async def test_real_time_graph_updates(self, graph_engine):
        """Test real-time graph updates and streaming."""
        # Setup streaming updates
        update_stream = [
            {"operation": "CREATE_NODE", "data": {"name": "NewTech", "type": "TECHNOLOGY"}},
            {"operation": "CREATE_RELATIONSHIP", "data": {"source": "Python", "target": "NewTech", "relation": "INFLUENCED"}},
            {"operation": "UPDATE_NODE", "data": {"name": "Python", "properties": {"version": "3.12"}}}
        ]
        
        # Mock streaming handler
        processed_updates = []
        for update in update_stream:
            result = await graph_engine.process_graph_update(update)
            processed_updates.append(result)
        
        # Verify updates processed
        assert len(processed_updates) == 3
        assert processed_updates[0]["operation"] == "CREATE_NODE"
        assert processed_updates[1]["operation"] == "CREATE_RELATIONSHIP"

    @pytest.mark.asyncio
    async def test_error_handling_connection_failure(self, graph_engine):
        """Test error handling when Memgraph connection fails."""
        # Setup
        entity_name = "Test Entity"
        
        # Mock connection failure
        graph_engine.memgraph_client.execute_query.side_effect = Exception("Connection to Memgraph failed")
        
        # Execute and verify graceful error handling
        with pytest.raises(Exception, match="Connection to Memgraph failed"):
            await graph_engine.query_by_entity(entity_name)

    @pytest.mark.asyncio
    async def test_health_check(self, graph_engine):
        """Test graph engine health check."""
        # Mock health check
        graph_engine.memgraph_client.ping.return_value = True
        graph_engine.entity_extractor.health_check.return_value = {"status": "healthy"}
        
        # Execute
        health = await graph_engine.health_check()
        
        # Verify
        assert health["status"] == "healthy"
        assert health["components"]["memgraph"] == "healthy"
        assert health["components"]["entity_extractor"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])