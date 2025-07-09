-- Analytics Database Schema for Self-Learning System
-- Supports performance tracking, experiment management, and improvement history

-- Performance Metrics Tables
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(12,4) NOT NULL,
    metric_unit VARCHAR(50),
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    model_used VARCHAR(255),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    tags TEXT[],
    
    -- Indexing for fast queries
    INDEX idx_performance_metrics_timestamp (timestamp),
    INDEX idx_performance_metrics_operation (operation_type),
    INDEX idx_performance_metrics_agent (agent_id),
    INDEX idx_performance_metrics_model (model_used),
    INDEX idx_performance_metrics_composite (operation_type, timestamp, agent_id)
);

-- Aggregated performance metrics for faster dashboard queries
CREATE TABLE IF NOT EXISTS performance_metrics_hourly (
    id BIGSERIAL PRIMARY KEY,
    operation_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    hour_bucket TIMESTAMPTZ NOT NULL,
    agent_id VARCHAR(255),
    model_used VARCHAR(255),
    
    -- Aggregated values
    avg_value DECIMAL(12,4),
    min_value DECIMAL(12,4),
    max_value DECIMAL(12,4),
    p50_value DECIMAL(12,4),
    p95_value DECIMAL(12,4),
    p99_value DECIMAL(12,4),
    sample_count INTEGER,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(operation_type, metric_name, hour_bucket, agent_id, model_used),
    INDEX idx_perf_hourly_bucket (hour_bucket),
    INDEX idx_perf_hourly_operation (operation_type)
);

-- Daily performance summaries
CREATE TABLE IF NOT EXISTS performance_metrics_daily (
    id BIGSERIAL PRIMARY KEY,
    operation_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    day_bucket DATE NOT NULL,
    agent_id VARCHAR(255),
    model_used VARCHAR(255),
    
    -- Aggregated values
    avg_value DECIMAL(12,4),
    min_value DECIMAL(12,4),
    max_value DECIMAL(12,4),
    p50_value DECIMAL(12,4),
    p95_value DECIMAL(12,4),
    p99_value DECIMAL(12,4),
    sample_count INTEGER,
    
    -- Trend information
    trend_direction VARCHAR(20), -- 'increasing', 'decreasing', 'stable'
    trend_confidence DECIMAL(3,2),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(operation_type, metric_name, day_bucket, agent_id, model_used),
    INDEX idx_perf_daily_bucket (day_bucket),
    INDEX idx_perf_daily_operation (operation_type)
);

-- Experiment Management Tables
CREATE TABLE IF NOT EXISTS ab_experiments (
    experiment_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    experiment_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    start_date TIMESTAMPTZ,
    end_date TIMESTAMPTZ,
    
    -- Configuration
    randomization_unit VARCHAR(50) DEFAULT 'user',
    sample_size_per_variant INTEGER,
    significance_level DECIMAL(4,3) DEFAULT 0.05,
    statistical_power DECIMAL(4,3) DEFAULT 0.8,
    
    -- Criteria
    inclusion_criteria JSONB DEFAULT '{}',
    exclusion_criteria JSONB DEFAULT '{}',
    
    -- Results
    primary_metric VARCHAR(100),
    winner_variant VARCHAR(255),
    confidence_level DECIMAL(4,3),
    
    metadata JSONB DEFAULT '{}',
    
    INDEX idx_experiments_status (status),
    INDEX idx_experiments_type (experiment_type),
    INDEX idx_experiments_dates (start_date, end_date)
);

CREATE TABLE IF NOT EXISTS ab_experiment_variants (
    variant_id VARCHAR(255) PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL REFERENCES ab_experiments(experiment_id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    weight DECIMAL(4,3) NOT NULL DEFAULT 0.5,
    configuration JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    INDEX idx_variants_experiment (experiment_id)
);

CREATE TABLE IF NOT EXISTS ab_experiment_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(255) NOT NULL REFERENCES ab_experiments(experiment_id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    direction VARCHAR(20) DEFAULT 'higher_is_better', -- 'higher_is_better' or 'lower_is_better'
    target_improvement DECIMAL(6,4),
    
    INDEX idx_exp_metrics_experiment (experiment_id),
    INDEX idx_exp_metrics_primary (is_primary)
);

CREATE TABLE IF NOT EXISTS ab_variant_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(255) NOT NULL REFERENCES ab_experiments(experiment_id) ON DELETE CASCADE,
    randomization_key VARCHAR(255) NOT NULL, -- user_id, session_id, etc.
    variant_id VARCHAR(255) NOT NULL REFERENCES ab_experiment_variants(variant_id) ON DELETE CASCADE,
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(experiment_id, randomization_key),
    INDEX idx_assignments_experiment (experiment_id),
    INDEX idx_assignments_variant (variant_id)
);

CREATE TABLE IF NOT EXISTS ab_experiment_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(255) NOT NULL REFERENCES ab_experiments(experiment_id) ON DELETE CASCADE,
    variant_id VARCHAR(255) NOT NULL REFERENCES ab_experiment_variants(variant_id) ON DELETE CASCADE,
    randomization_key VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Metrics
    metrics JSONB NOT NULL DEFAULT '{}',
    
    -- Context
    metadata JSONB DEFAULT '{}',
    
    INDEX idx_results_experiment (experiment_id),
    INDEX idx_results_variant (variant_id),
    INDEX idx_results_timestamp (timestamp)
);

-- Improvement History Tables
CREATE TABLE IF NOT EXISTS improvement_actions (
    action_id VARCHAR(255) PRIMARY KEY,
    action_type VARCHAR(100) NOT NULL,
    component VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    
    -- Scoring
    confidence DECIMAL(4,3) NOT NULL,
    impact_score DECIMAL(4,3) NOT NULL,
    risk_score DECIMAL(4,3) NOT NULL,
    
    -- Configuration
    auto_apply BOOLEAN DEFAULT FALSE,
    configuration_changes JSONB DEFAULT '{}',
    rollback_plan JSONB DEFAULT '{}',
    validation_checks TEXT[],
    
    -- Status
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMPTZ,
    
    -- Results
    execution_results JSONB DEFAULT '{}',
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_actions_component (component),
    INDEX idx_actions_applied (applied),
    INDEX idx_actions_type (action_type),
    INDEX idx_actions_created (created_at)
);

CREATE TABLE IF NOT EXISTS improvement_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_id VARCHAR(255) NOT NULL REFERENCES improvement_actions(action_id) ON DELETE CASCADE,
    
    -- Execution details
    execution_started TIMESTAMPTZ NOT NULL,
    execution_completed TIMESTAMPTZ,
    execution_status VARCHAR(50) NOT NULL, -- 'success', 'failed', 'rolled_back'
    
    -- Changes
    changes_applied TEXT[],
    validation_results JSONB DEFAULT '{}',
    
    -- Performance impact
    before_metrics JSONB DEFAULT '{}',
    after_metrics JSONB DEFAULT '{}',
    improvement_achieved DECIMAL(6,4),
    
    -- Rollback info
    rollback_info JSONB DEFAULT '{}',
    rollback_completed BOOLEAN DEFAULT FALSE,
    
    error_message TEXT,
    
    INDEX idx_results_action (action_id),
    INDEX idx_results_status (execution_status),
    INDEX idx_results_started (execution_started)
);

-- Configuration Change Audit Trail
CREATE TABLE IF NOT EXISTS configuration_changes (
    change_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component VARCHAR(100) NOT NULL,
    config_key VARCHAR(255) NOT NULL,
    old_value JSONB,
    new_value JSONB,
    change_type VARCHAR(50) NOT NULL, -- 'manual', 'automated', 'experiment'
    change_reason TEXT,
    
    -- Attribution
    changed_by VARCHAR(255), -- user_id or 'system'
    improvement_action_id VARCHAR(255) REFERENCES improvement_actions(action_id),
    experiment_id VARCHAR(255) REFERENCES ab_experiments(experiment_id),
    
    -- Timing
    changed_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_config_changes_component (component),
    INDEX idx_config_changes_key (config_key),
    INDEX idx_config_changes_timestamp (changed_at),
    INDEX idx_config_changes_type (change_type)
);

-- System Health Reports
CREATE TABLE IF NOT EXISTS system_health_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_date DATE NOT NULL,
    report_type VARCHAR(100) NOT NULL, -- 'daily', 'weekly', 'monthly', 'on_demand'
    
    -- Overall scores
    overall_health_score DECIMAL(4,3) NOT NULL,
    memory_health_score DECIMAL(4,3),
    performance_score DECIMAL(4,3),
    prompt_health_score DECIMAL(4,3),
    
    -- Component details
    total_memories INTEGER,
    healthy_memories INTEGER,
    problematic_memories INTEGER,
    
    -- Performance summary
    avg_latency_ms DECIMAL(8,2),
    success_rate DECIMAL(4,3),
    error_rate DECIMAL(4,3),
    
    -- Recommendations
    recommendations TEXT[],
    critical_issues TEXT[],
    
    -- Full report data
    detailed_report JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_health_reports_date (report_date),
    INDEX idx_health_reports_type (report_type),
    INDEX idx_health_reports_score (overall_health_score)
);

-- Prompt Performance Tracking
CREATE TABLE IF NOT EXISTS prompt_templates (
    template_id VARCHAR(255) PRIMARY KEY,
    template_name VARCHAR(500) NOT NULL,
    template_content TEXT NOT NULL,
    template_variables TEXT[],
    version INTEGER DEFAULT 1,
    
    -- Performance tracking
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    total_confidence DECIMAL(10,4) DEFAULT 0,
    total_latency_ms BIGINT DEFAULT 0,
    
    -- Metadata
    component VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    last_used TIMESTAMPTZ,
    
    metadata JSONB DEFAULT '{}',
    
    INDEX idx_prompts_component (component),
    INDEX idx_prompts_usage (usage_count),
    INDEX idx_prompts_success_rate ((success_count::decimal / NULLIF(usage_count, 0))),
    INDEX idx_prompts_last_used (last_used)
);

CREATE TABLE IF NOT EXISTS prompt_usage_history (
    usage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id VARCHAR(255) NOT NULL REFERENCES prompt_templates(template_id) ON DELETE CASCADE,
    
    -- Usage details
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    confidence DECIMAL(4,3),
    latency_ms INTEGER,
    
    -- Context
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    context_data JSONB DEFAULT '{}',
    
    -- Error information
    error_message TEXT,
    error_type VARCHAR(100),
    
    INDEX idx_prompt_usage_template (template_id),
    INDEX idx_prompt_usage_timestamp (timestamp),
    INDEX idx_prompt_usage_success (success),
    INDEX idx_prompt_usage_agent (agent_id)
);

-- Failure Pattern Detection
CREATE TABLE IF NOT EXISTS failure_patterns (
    pattern_id VARCHAR(255) PRIMARY KEY,
    pattern_name VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    
    -- Pattern details
    frequency INTEGER NOT NULL DEFAULT 1,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    
    -- Pattern characteristics
    error_keywords TEXT[],
    affected_components TEXT[],
    common_contexts JSONB DEFAULT '{}',
    
    -- Suggested fixes
    suggested_fixes TEXT[],
    applied_fixes TEXT[],
    
    -- Status
    pattern_status VARCHAR(50) DEFAULT 'active', -- 'active', 'resolved', 'investigating'
    resolution_notes TEXT,
    
    INDEX idx_patterns_frequency (frequency),
    INDEX idx_patterns_last_seen (last_seen),
    INDEX idx_patterns_status (pattern_status)
);

CREATE TABLE IF NOT EXISTS failure_pattern_instances (
    instance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id VARCHAR(255) NOT NULL REFERENCES failure_patterns(pattern_id) ON DELETE CASCADE,
    
    -- Instance details
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    component VARCHAR(100),
    operation_type VARCHAR(100),
    error_message TEXT,
    
    -- Context
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    template_id VARCHAR(255),
    context_data JSONB DEFAULT '{}',
    
    -- Analysis
    similarity_score DECIMAL(4,3),
    
    INDEX idx_pattern_instances_pattern (pattern_id),
    INDEX idx_pattern_instances_timestamp (timestamp),
    INDEX idx_pattern_instances_component (component)
);

-- Dashboard Data Models
CREATE TABLE IF NOT EXISTS dashboard_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    dashboard_type VARCHAR(100) NOT NULL,
    time_range VARCHAR(50) NOT NULL,
    
    -- Cached data
    data JSONB NOT NULL,
    
    -- Cache metadata
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    
    INDEX idx_dashboard_cache_type (dashboard_type),
    INDEX idx_dashboard_cache_expires (expires_at)
);

-- Scheduled Jobs for Analytics
CREATE TABLE IF NOT EXISTS analytics_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    job_name VARCHAR(500) NOT NULL,
    job_type VARCHAR(100) NOT NULL,
    
    -- Scheduling
    schedule_pattern VARCHAR(100) NOT NULL, -- cron-like
    priority VARCHAR(20) DEFAULT 'medium',
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending',
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    
    -- Configuration
    target_component VARCHAR(100),
    job_configuration JSONB DEFAULT '{}',
    timeout_minutes INTEGER DEFAULT 60,
    retry_count INTEGER DEFAULT 3,
    
    -- Results
    last_results JSONB DEFAULT '{}',
    error_history TEXT[],
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_jobs_status (status),
    INDEX idx_jobs_next_run (next_run),
    INDEX idx_jobs_type (job_type)
);

-- Create functions for automatic aggregation
CREATE OR REPLACE FUNCTION aggregate_hourly_metrics()
RETURNS void AS $$
BEGIN
    INSERT INTO performance_metrics_hourly (
        operation_type, metric_name, hour_bucket, agent_id, model_used,
        avg_value, min_value, max_value, p50_value, p95_value, p99_value, sample_count
    )
    SELECT 
        operation_type,
        metric_name,
        date_trunc('hour', timestamp) as hour_bucket,
        agent_id,
        model_used,
        AVG(metric_value) as avg_value,
        MIN(metric_value) as min_value,
        MAX(metric_value) as max_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY metric_value) as p50_value,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY metric_value) as p95_value,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY metric_value) as p99_value,
        COUNT(*) as sample_count
    FROM performance_metrics
    WHERE timestamp >= NOW() - INTERVAL '2 hours'
        AND timestamp < date_trunc('hour', NOW())
    GROUP BY operation_type, metric_name, hour_bucket, agent_id, model_used
    ON CONFLICT (operation_type, metric_name, hour_bucket, agent_id, model_used)
    DO UPDATE SET
        avg_value = EXCLUDED.avg_value,
        min_value = EXCLUDED.min_value,
        max_value = EXCLUDED.max_value,
        p50_value = EXCLUDED.p50_value,
        p95_value = EXCLUDED.p95_value,
        p99_value = EXCLUDED.p99_value,
        sample_count = EXCLUDED.sample_count,
        created_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic prompt template updates
CREATE OR REPLACE FUNCTION update_prompt_template_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE prompt_templates SET
        usage_count = usage_count + 1,
        success_count = success_count + CASE WHEN NEW.success THEN 1 ELSE 0 END,
        total_confidence = total_confidence + COALESCE(NEW.confidence, 0),
        total_latency_ms = total_latency_ms + COALESCE(NEW.latency_ms, 0),
        last_used = NEW.timestamp,
        last_updated = NOW()
    WHERE template_id = NEW.template_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER prompt_usage_stats_trigger
    AFTER INSERT ON prompt_usage_history
    FOR EACH ROW
    EXECUTE FUNCTION update_prompt_template_stats();

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_time_series 
    ON performance_metrics (operation_type, metric_name, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ab_results_analysis 
    ON ab_experiment_results (experiment_id, variant_id, timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_health_reports_trend 
    ON system_health_reports (report_date DESC, overall_health_score);

-- Grant permissions (adjust based on your user setup)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO analytics_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO analytics_user;

-- Comment the tables for documentation
COMMENT ON TABLE performance_metrics IS 'Individual performance metric measurements for all system operations';
COMMENT ON TABLE performance_metrics_hourly IS 'Hourly aggregated performance metrics for dashboard queries';
COMMENT ON TABLE ab_experiments IS 'A/B testing experiments for system optimization';
COMMENT ON TABLE improvement_actions IS 'Automated system improvement actions and their results';
COMMENT ON TABLE system_health_reports IS 'Periodic system health assessment reports';
COMMENT ON TABLE prompt_templates IS 'Prompt templates with performance tracking';
COMMENT ON TABLE failure_patterns IS 'Detected failure patterns for system improvement';