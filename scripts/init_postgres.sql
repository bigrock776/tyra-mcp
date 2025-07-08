-- PostgreSQL initialization script for Tyra Advanced Memory System
-- Creates necessary extensions, schema, and initial tables

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create schema
CREATE SCHEMA IF NOT EXISTS memory;

-- Set search path
SET search_path TO memory, public;

-- Create memory embeddings table
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}'::jsonb,
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
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_metadata ON memory_embeddings USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_created_at ON memory_embeddings (created_at);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_agent_id ON memory_embeddings (agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_session_id ON memory_embeddings (session_id);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_content_hash ON memory_embeddings (content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_fts ON memory_embeddings USING gin(content_tsvector);

-- Create vector similarity index (HNSW for better performance)
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_embedding_hnsw
    ON memory_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Create function for updating updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
DROP TRIGGER IF EXISTS update_memory_embeddings_updated_at ON memory_embeddings;
CREATE TRIGGER update_memory_embeddings_updated_at
    BEFORE UPDATE ON memory_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create system logs table for monitoring
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    level VARCHAR(10) NOT NULL,
    component VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for efficient log queries
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    context JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type_time ON performance_metrics(metric_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);

-- Create alerts table
CREATE TABLE IF NOT EXISTS performance_alerts (
    id SERIAL PRIMARY KEY,
    severity VARCHAR(20) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    value FLOAT NOT NULL,
    threshold FLOAT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE,
    context JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for alerts
CREATE INDEX IF NOT EXISTS idx_performance_alerts_severity ON performance_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_resolved ON performance_alerts(resolved);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_timestamp ON performance_alerts(timestamp);

-- Create experiments table for adaptive learning
CREATE TABLE IF NOT EXISTS adaptation_experiments (
    id TEXT PRIMARY KEY,
    adaptation_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    parameters JSONB NOT NULL,
    original_parameters JSONB NOT NULL,
    target_metrics TEXT[] NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    baseline_metrics JSONB DEFAULT '{}'::jsonb,
    experiment_metrics JSONB DEFAULT '{}'::jsonb,
    improvement JSONB DEFAULT '{}'::jsonb,
    confidence FLOAT DEFAULT 0.0,
    success BOOLEAN DEFAULT FALSE,
    notes TEXT
);

-- Create indexes for experiments
CREATE INDEX IF NOT EXISTS idx_adaptation_experiments_type ON adaptation_experiments(adaptation_type);
CREATE INDEX IF NOT EXISTS idx_adaptation_experiments_status ON adaptation_experiments(status);
CREATE INDEX IF NOT EXISTS idx_adaptation_experiments_start_time ON adaptation_experiments(start_time);

-- Create learning insights table
CREATE TABLE IF NOT EXISTS learning_insights (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    insight TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    supporting_experiments TEXT[],
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    actionable BOOLEAN DEFAULT FALSE,
    impact_estimate TEXT
);

-- Create indexes for insights
CREATE INDEX IF NOT EXISTS idx_learning_insights_category ON learning_insights(category);
CREATE INDEX IF NOT EXISTS idx_learning_insights_timestamp ON learning_insights(timestamp);
CREATE INDEX IF NOT EXISTS idx_learning_insights_actionable ON learning_insights(actionable);

-- Grant permissions to tyra user
GRANT USAGE ON SCHEMA memory TO tyra;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA memory TO tyra;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA memory TO tyra;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA memory TO tyra;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA memory GRANT ALL ON TABLES TO tyra;
ALTER DEFAULT PRIVILEGES IN SCHEMA memory GRANT ALL ON SEQUENCES TO tyra;
ALTER DEFAULT PRIVILEGES IN SCHEMA memory GRANT ALL ON FUNCTIONS TO tyra;

-- Create a view for recent performance summary
CREATE OR REPLACE VIEW recent_performance_summary AS
SELECT
    metric_type,
    COUNT(*) as sample_count,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    STDDEV(metric_value) as std_dev,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY metric_value) as p95_value,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY metric_value) as p99_value
FROM performance_metrics
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY metric_type;

-- Create a function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean old performance metrics
    DELETE FROM performance_metrics
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Clean old resolved alerts
    DELETE FROM performance_alerts
    WHERE resolved = TRUE AND timestamp < NOW() - (retention_days || ' days')::INTERVAL;

    -- Clean old completed experiments
    DELETE FROM adaptation_experiments
    WHERE status IN ('completed', 'failed', 'rolled_back')
    AND start_time < NOW() - (retention_days || ' days')::INTERVAL;

    -- Clean old system logs
    DELETE FROM system_logs
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Insert initial system log
INSERT INTO system_logs (level, component, message, metadata)
VALUES ('INFO', 'database', 'Tyra Memory System database initialized',
        jsonb_build_object('schema_version', '1.0', 'init_time', NOW()));

-- Display summary
SELECT 'Tyra Memory System database initialized successfully' as status;
