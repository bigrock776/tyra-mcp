// Memgraph initialization script for Tyra Memory System
// Creates schema, indexes, and initial data for knowledge graph operations

// ==========================================
// SCHEMA DEFINITION
// ==========================================

// Create node labels and properties
CREATE CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE;
CREATE CONSTRAINT ON (m:Memory) ASSERT m.id IS UNIQUE;
CREATE CONSTRAINT ON (a:Agent) ASSERT a.id IS UNIQUE;
CREATE CONSTRAINT ON (s:Session) ASSERT s.id IS UNIQUE;
CREATE CONSTRAINT ON (c:Concept) ASSERT c.id IS UNIQUE;
CREATE CONSTRAINT ON (t:Topic) ASSERT t.id IS UNIQUE;
CREATE CONSTRAINT ON (ev:Event) ASSERT ev.id IS UNIQUE;

// Create indexes for efficient querying
CREATE INDEX ON :Entity(name);
CREATE INDEX ON :Entity(type);
CREATE INDEX ON :Entity(created_at);
CREATE INDEX ON :Memory(content_hash);
CREATE INDEX ON :Memory(created_at);
CREATE INDEX ON :Memory(agent_id);
CREATE INDEX ON :Agent(name);
CREATE INDEX ON :Session(agent_id);
CREATE INDEX ON :Session(created_at);
CREATE INDEX ON :Concept(name);
CREATE INDEX ON :Concept(category);
CREATE INDEX ON :Topic(name);
CREATE INDEX ON :Event(timestamp);
CREATE INDEX ON :Event(type);

// ==========================================
// SYSTEM METADATA
// ==========================================

// Create system metadata node
CREATE (sys:System {
    id: 'tyra_memory_system',
    name: 'Tyra Advanced Memory System',
    version: '1.0',
    initialized_at: datetime(),
    schema_version: '1.0',
    component: 'knowledge_graph'
});

// Create configuration node
CREATE (config:Configuration {
    id: 'graph_config',
    max_depth: 5,
    similarity_threshold: 0.7,
    entity_extraction_confidence: 0.8,
    relationship_confidence: 0.6,
    temporal_window_days: 30,
    max_entities_per_memory: 10,
    max_relationships_per_entity: 20
});

// ==========================================
// ENTITY TYPES AND CATEGORIES
// ==========================================

// Create entity type nodes
CREATE (et1:EntityType {id: 'PERSON', name: 'Person', description: 'Human individuals'});
CREATE (et2:EntityType {id: 'ORGANIZATION', name: 'Organization', description: 'Companies, institutions, groups'});
CREATE (et3:EntityType {id: 'LOCATION', name: 'Location', description: 'Places, addresses, geographical entities'});
CREATE (et4:EntityType {id: 'CONCEPT', name: 'Concept', description: 'Abstract concepts and ideas'});
CREATE (et5:EntityType {id: 'PRODUCT', name: 'Product', description: 'Products, services, items'});
CREATE (et6:EntityType {id: 'EVENT', name: 'Event', description: 'Events, activities, occurrences'});
CREATE (et7:EntityType {id: 'TOPIC', name: 'Topic', description: 'Discussion topics and subjects'});
CREATE (et8:EntityType {id: 'SKILL', name: 'Skill', description: 'Skills, abilities, competencies'});
CREATE (et9:EntityType {id: 'TECHNOLOGY', name: 'Technology', description: 'Technologies, tools, systems'});
CREATE (et10:EntityType {id: 'DOCUMENT', name: 'Document', description: 'Documents, files, resources'});

// Create relationship type nodes
CREATE (rt1:RelationshipType {
    id: 'MENTIONS',
    name: 'Mentions',
    description: 'Entity is mentioned in memory',
    weight: 1.0
});
CREATE (rt2:RelationshipType {
    id: 'RELATED_TO',
    name: 'Related To',
    description: 'Entities are related or associated',
    weight: 0.8
});
CREATE (rt3:RelationshipType {
    id: 'PART_OF',
    name: 'Part Of',
    description: 'Entity is part of another entity',
    weight: 0.9
});
CREATE (rt4:RelationshipType {
    id: 'WORKS_FOR',
    name: 'Works For',
    description: 'Person works for organization',
    weight: 0.9
});
CREATE (rt5:RelationshipType {
    id: 'LOCATED_IN',
    name: 'Located In',
    description: 'Entity is located in location',
    weight: 0.8
});
CREATE (rt6:RelationshipType {
    id: 'SIMILAR_TO',
    name: 'Similar To',
    description: 'Entities are similar',
    weight: 0.7
});
CREATE (rt7:RelationshipType {
    id: 'CAUSED_BY',
    name: 'Caused By',
    description: 'Event or state caused by entity',
    weight: 0.8
});
CREATE (rt8:RelationshipType {
    id: 'DEPENDS_ON',
    name: 'Depends On',
    description: 'Entity depends on another entity',
    weight: 0.8
});
CREATE (rt9:RelationshipType {
    id: 'INTERACTS_WITH',
    name: 'Interacts With',
    description: 'Entities interact with each other',
    weight: 0.7
});
CREATE (rt10:RelationshipType {
    id: 'TEMPORAL_SEQUENCE',
    name: 'Temporal Sequence',
    description: 'Events occur in sequence',
    weight: 0.9
});

// ==========================================
// TEMPORAL GRAPH STRUCTURE
// ==========================================

// Create temporal nodes for time-based organization
CREATE (today:TimeNode {
    id: 'today',
    date: date(),
    type: 'day',
    created_at: datetime()
});

CREATE (this_week:TimeNode {
    id: 'this_week',
    week: date().week,
    year: date().year,
    type: 'week',
    created_at: datetime()
});

CREATE (this_month:TimeNode {
    id: 'this_month',
    month: date().month,
    year: date().year,
    type: 'month',
    created_at: datetime()
});

CREATE (this_year:TimeNode {
    id: 'this_year',
    year: date().year,
    type: 'year',
    created_at: datetime()
});

// Create temporal hierarchy
CREATE (this_year)-[:CONTAINS]->(this_month);
CREATE (this_month)-[:CONTAINS]->(this_week);
CREATE (this_week)-[:CONTAINS]->(today);

// ==========================================
// PERFORMANCE OPTIMIZATION
// ==========================================

// Create performance tracking nodes
CREATE (perf:Performance {
    id: 'graph_performance',
    query_count: 0,
    avg_query_time_ms: 0,
    entity_count: 0,
    relationship_count: 0,
    memory_count: 0,
    last_updated: datetime()
});

// Create cache nodes for frequently accessed data
CREATE (cache:Cache {
    id: 'query_cache',
    size_limit: 1000,
    ttl_seconds: 3600,
    current_size: 0,
    hit_count: 0,
    miss_count: 0,
    last_cleanup: datetime()
});

// ==========================================
// ANALYTICAL VIEWS
// ==========================================

// These would be implemented as stored procedures in a production system
// For now, we'll create metadata nodes to track analytical capabilities

CREATE (analytics:Analytics {
    id: 'graph_analytics',
    centrality_algorithms: ['betweenness', 'closeness', 'pagerank'],
    community_detection: ['louvain', 'label_propagation'],
    path_finding: ['shortest_path', 'all_paths', 'dijkstra'],
    similarity_metrics: ['jaccard', 'overlap', 'cosine'],
    temporal_analysis: ['trend_detection', 'pattern_mining', 'anomaly_detection']
});

// ==========================================
// MEMORY HEALTH TRACKING
// ==========================================

// Create nodes for tracking memory health and quality
CREATE (health:MemoryHealth {
    id: 'memory_health_tracker',
    total_memories: 0,
    stale_memories: 0,
    low_confidence_memories: 0,
    duplicate_memories: 0,
    orphaned_entities: 0,
    last_cleanup: datetime(),
    health_score: 1.0
});

// Create quality metrics
CREATE (quality:QualityMetrics {
    id: 'quality_metrics',
    confidence_threshold: 0.7,
    freshness_threshold_days: 30,
    similarity_threshold: 0.9,
    entity_coverage: 0.0,
    relationship_density: 0.0,
    temporal_coverage: 0.0
});

// ==========================================
// ADAPTATION AND LEARNING
// ==========================================

// Create nodes for tracking adaptive behavior
CREATE (adaptation:Adaptation {
    id: 'graph_adaptation',
    learning_rate: 0.1,
    adaptation_frequency: 'daily',
    auto_optimization: true,
    schema_evolution: true,
    relationship_strength_decay: 0.95,
    entity_importance_boost: 1.05,
    last_adaptation: datetime()
});

// Create learning insights storage
CREATE (insights:LearningInsights {
    id: 'learning_insights',
    pattern_detection: true,
    anomaly_detection: true,
    relationship_evolution: true,
    entity_lifecycle_tracking: true,
    confidence_calibration: true,
    last_analysis: datetime()
});

// ==========================================
// INTEGRATION POINTS
// ==========================================

// Create integration nodes for external systems
CREATE (integration:Integration {
    id: 'external_integration',
    embedding_service: 'enabled',
    vector_store: 'enabled',
    hallucination_detector: 'enabled',
    trading_system: 'enabled',
    audit_logging: 'enabled',
    telemetry: 'enabled'
});

// ==========================================
// SECURITY AND PRIVACY
// ==========================================

// Create privacy tracking nodes
CREATE (privacy:Privacy {
    id: 'privacy_controls',
    data_retention_days: 90,
    anonymization_enabled: true,
    consent_tracking: true,
    audit_trail: true,
    encryption_at_rest: true,
    access_control: true
});

// Create audit log structure
CREATE (audit:AuditLog {
    id: 'graph_audit',
    log_level: 'INFO',
    retention_days: 30,
    log_queries: true,
    log_modifications: true,
    log_access: true,
    last_cleanup: datetime()
});

// ==========================================
// INITIAL SYSTEM CONNECTIONS
// ==========================================

// Connect system components
CREATE (sys)-[:CONFIGURED_BY]->(config);
CREATE (sys)-[:TRACKS_PERFORMANCE]->(perf);
CREATE (sys)-[:USES_CACHE]->(cache);
CREATE (sys)-[:PROVIDES_ANALYTICS]->(analytics);
CREATE (sys)-[:MONITORS_HEALTH]->(health);
CREATE (sys)-[:TRACKS_QUALITY]->(quality);
CREATE (sys)-[:SUPPORTS_ADAPTATION]->(adaptation);
CREATE (sys)-[:GENERATES_INSIGHTS]->(insights);
CREATE (sys)-[:INTEGRATES_WITH]->(integration);
CREATE (sys)-[:ENFORCES_PRIVACY]->(privacy);
CREATE (sys)-[:MAINTAINS_AUDIT]->(audit);

// ==========================================
// GRAPH STATISTICS UPDATE
// ==========================================

// This would typically be done via a stored procedure
// For initialization, we'll set basic counts
MATCH (n) WITH count(n) as node_count
MATCH ()-[r]->() WITH count(r) as rel_count, node_count
MATCH (perf:Performance {id: 'graph_performance'})
SET perf.entity_count = node_count,
    perf.relationship_count = rel_count,
    perf.last_updated = datetime();

// ==========================================
// COMPLETION MESSAGE
// ==========================================

RETURN 'Tyra Memory System knowledge graph initialized successfully' as status,
       datetime() as initialized_at,
       'Schema, indexes, and initial data created' as details;