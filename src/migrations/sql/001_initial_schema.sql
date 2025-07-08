-- =============================================================================
-- Tyra MCP Memory Server - Initial Database Schema
-- =============================================================================
-- This file creates the initial database schema for the Tyra MCP Memory Server
-- including tables for memories, embeddings, entities, relationships, and MCP-specific data.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Core Memory Tables
-- =============================================================================

-- Main memories table
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    agent_id VARCHAR(255) NOT NULL DEFAULT 'tyra',
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'active',
    CONSTRAINT check_status CHECK (status IN ('active', 'archived', 'deleted'))
);

-- Memory chunks for large content
CREATE TABLE IF NOT EXISTS memory_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024), -- Primary embedding dimension
    embedding_fallback vector(384), -- Fallback embedding dimension
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(memory_id, chunk_index)
);

-- Memory embeddings (for backward compatibility)
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES memory_chunks(id) ON DELETE CASCADE,
    embedding vector(1024) NOT NULL,
    embedding_model VARCHAR(255) NOT NULL DEFAULT 'intfloat/e5-large-v2',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =============================================================================
-- Entity and Relationship Tables (Knowledge Graph)
-- =============================================================================

-- Entities extracted from memories
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    description TEXT,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    embedding vector(1024),
    properties JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source_count INTEGER DEFAULT 1,
    UNIQUE(name, entity_type)
);

-- Relationships between entities
CREATE TABLE IF NOT EXISTS relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}'::jsonb,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source_count INTEGER DEFAULT 1,
    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

-- Link entities to memory chunks
CREATE TABLE IF NOT EXISTS memory_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES memory_chunks(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    context_start INTEGER,
    context_end INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(memory_id, entity_id)
);

-- =============================================================================
-- MCP-Specific Tables
-- =============================================================================

-- MCP sessions for agent tracking
CREATE TABLE IF NOT EXISTS mcp_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    session_token VARCHAR(500),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT check_mcp_status CHECK (status IN ('active', 'inactive', 'expired'))
);

-- MCP tool call tracking
CREATE TABLE IF NOT EXISTS mcp_tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES mcp_sessions(id) ON DELETE SET NULL,
    agent_id VARCHAR(255) NOT NULL,
    tool_name VARCHAR(255) NOT NULL,
    parameters JSONB,
    result JSONB,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Performance and Analytics Tables
-- =============================================================================

-- Performance metrics tracking
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    value FLOAT NOT NULL,
    unit VARCHAR(50),
    operation VARCHAR(255),
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- System health snapshots
CREATE TABLE IF NOT EXISTS health_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    overall_status VARCHAR(50) NOT NULL,
    component_health JSONB NOT NULL,
    resource_usage JSONB NOT NULL,
    recommendations JSONB DEFAULT '[]'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Learning experiments and adaptations
CREATE TABLE IF NOT EXISTS learning_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_type VARCHAR(100) NOT NULL,
    description TEXT,
    parameters JSONB NOT NULL,
    baseline_metrics JSONB,
    test_metrics JSONB,
    improvement FLOAT,
    confidence FLOAT,
    status VARCHAR(50) DEFAULT 'running',
    success BOOLEAN,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT check_experiment_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

-- Configuration changes tracking
CREATE TABLE IF NOT EXISTS config_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parameter_name VARCHAR(255) NOT NULL,
    old_value TEXT,
    new_value TEXT NOT NULL,
    change_reason TEXT,
    experiment_id UUID REFERENCES learning_experiments(id) ON DELETE SET NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reverted_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =============================================================================
-- Search and Cache Tables
-- =============================================================================

-- Search query cache
CREATE TABLE IF NOT EXISTS search_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash VARCHAR(255) NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    search_type VARCHAR(50) NOT NULL,
    agent_id VARCHAR(255),
    results JSONB NOT NULL,
    result_count INTEGER NOT NULL,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    hit_count INTEGER DEFAULT 1
);

-- Hallucination analysis cache
CREATE TABLE IF NOT EXISTS hallucination_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_hash VARCHAR(255) NOT NULL UNIQUE,
    response_text TEXT NOT NULL,
    query_text TEXT,
    analysis_result JSONB NOT NULL,
    confidence_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Core memory indexes
CREATE INDEX IF NOT EXISTS idx_memories_agent_created ON memories(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING GIN(metadata);

-- Memory chunk indexes
CREATE INDEX IF NOT EXISTS idx_memory_chunks_memory_id ON memory_chunks(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding ON memory_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding_fallback ON memory_chunks USING hnsw (embedding_fallback vector_cosine_ops);

-- Entity and relationship indexes
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);

-- MCP tracking indexes
CREATE INDEX IF NOT EXISTS idx_mcp_sessions_agent ON mcp_sessions(agent_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_session ON mcp_tool_calls(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_agent_tool ON mcp_tool_calls(agent_id, tool_name, created_at DESC);

-- Performance tracking indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type_time ON performance_metrics(metric_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_operation ON performance_metrics(operation, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_health_snapshots_time ON health_snapshots(timestamp DESC);

-- Learning and adaptation indexes
CREATE INDEX IF NOT EXISTS idx_learning_experiments_type_status ON learning_experiments(experiment_type, status);
CREATE INDEX IF NOT EXISTS idx_config_changes_parameter ON config_changes(parameter_name, applied_at DESC);

-- Cache indexes
CREATE INDEX IF NOT EXISTS idx_search_cache_hash ON search_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_search_cache_expires ON search_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_hallucination_cache_hash ON hallucination_cache(response_hash);
CREATE INDEX IF NOT EXISTS idx_hallucination_cache_expires ON hallucination_cache(expires_at);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_memories_content_fts ON memories USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_memory_chunks_content_fts ON memory_chunks USING gin(to_tsvector('english', content));

-- =============================================================================
-- Functions and Triggers
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_memories_updated_at BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_entities_updated_at BEFORE UPDATE ON entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_relationships_updated_at BEFORE UPDATE ON relationships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to cleanup expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean up expired search cache
    DELETE FROM search_cache WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Clean up expired hallucination cache
    DELETE FROM hallucination_cache WHERE expires_at < CURRENT_TIMESTAMP;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get memory statistics
CREATE OR REPLACE FUNCTION get_memory_stats()
RETURNS JSONB AS $$
DECLARE
    stats JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_memories', (SELECT COUNT(*) FROM memories WHERE status = 'active'),
        'total_chunks', (SELECT COUNT(*) FROM memory_chunks),
        'total_entities', (SELECT COUNT(*) FROM entities),
        'total_relationships', (SELECT COUNT(*) FROM relationships),
        'agents', (
            SELECT jsonb_object_agg(agent_id, memory_count)
            FROM (
                SELECT agent_id, COUNT(*) as memory_count
                FROM memories
                WHERE status = 'active'
                GROUP BY agent_id
            ) agent_stats
        ),
        'recent_activity', (
            SELECT jsonb_build_object(
                'memories_last_24h', (
                    SELECT COUNT(*) FROM memories
                    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                ),
                'tool_calls_last_24h', (
                    SELECT COUNT(*) FROM mcp_tool_calls
                    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                )
            )
        )
    ) INTO stats;

    RETURN stats;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Initial Data and Settings
-- =============================================================================

-- Insert default configuration
INSERT INTO config_changes (parameter_name, old_value, new_value, change_reason)
VALUES
    ('schema_version', NULL, '1.0.0', 'Initial schema creation'),
    ('vector_dimension_primary', NULL, '1024', 'Primary embedding dimension'),
    ('vector_dimension_fallback', NULL, '384', 'Fallback embedding dimension')
ON CONFLICT DO NOTHING;

-- Create a function to check if embeddings are properly set
CREATE OR REPLACE FUNCTION validate_embedding_dimensions()
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if any embeddings have incorrect dimensions
    IF EXISTS (
        SELECT 1 FROM memory_chunks
        WHERE embedding IS NOT NULL AND vector_dims(embedding) != 1024
    ) THEN
        RETURN FALSE;
    END IF;

    IF EXISTS (
        SELECT 1 FROM memory_chunks
        WHERE embedding_fallback IS NOT NULL AND vector_dims(embedding_fallback) != 384
    ) THEN
        RETURN FALSE;
    END IF;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- View for memory with chunk and entity information
CREATE OR REPLACE VIEW memory_details AS
SELECT
    m.id,
    m.content,
    m.agent_id,
    m.session_id,
    m.created_at,
    m.metadata,
    COUNT(DISTINCT mc.id) as chunk_count,
    COUNT(DISTINCT me.entity_id) as entity_count,
    ARRAY_AGG(DISTINCT e.name) FILTER (WHERE e.name IS NOT NULL) as entity_names
FROM memories m
LEFT JOIN memory_chunks mc ON m.id = mc.memory_id
LEFT JOIN memory_entities me ON m.id = me.memory_id
LEFT JOIN entities e ON me.entity_id = e.id
WHERE m.status = 'active'
GROUP BY m.id, m.content, m.agent_id, m.session_id, m.created_at, m.metadata;

-- View for recent performance metrics
CREATE OR REPLACE VIEW recent_performance AS
SELECT
    metric_type,
    metric_name,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    COUNT(*) as sample_count,
    MAX(timestamp) as last_recorded
FROM performance_metrics
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY metric_type, metric_name
ORDER BY last_recorded DESC;

-- =============================================================================
-- Grants and Permissions
-- =============================================================================

-- Grant appropriate permissions to the application user
-- Note: Adjust these based on your specific user setup
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO tyra_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO tyra_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO tyra_app;

-- =============================================================================
-- Schema Version and Completion
-- =============================================================================

-- Record schema creation completion
INSERT INTO health_snapshots (overall_status, component_health, resource_usage)
VALUES (
    'healthy',
    '{"database": {"status": "healthy", "schema_version": "1.0.0", "tables_created": true}}',
    '{"database": {"connections": 0, "size_mb": 0}}'
);

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Tyra MCP Memory Server schema v1.0.0 created successfully';
    RAISE NOTICE 'Total tables created: %', (
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    );
    RAISE NOTICE 'Total indexes created: %', (
        SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public'
    );
END $$;
