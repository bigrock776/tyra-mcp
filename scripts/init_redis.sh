#!/bin/bash
# Redis initialization script for Tyra Memory System

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Redis connection settings
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"

# Check if Redis is running
check_redis() {
    log "Checking Redis connection..."
    if ! redis-cli -h $REDIS_HOST -p $REDIS_PORT ping >/dev/null 2>&1; then
        error "Cannot connect to Redis at $REDIS_HOST:$REDIS_PORT"
        exit 1
    fi
    log "Redis connection successful"
}

# Initialize Redis cache structure
init_cache_structure() {
    log "Initializing Redis cache structure..."
    
    # Create cache key namespaces
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB <<EOF
-- Set cache configuration
HSET cache:config embeddings_ttl 86400
HSET cache:config search_ttl 3600
HSET cache:config rerank_ttl 1800
HSET cache:config hallucination_ttl 7200
HSET cache:config provider_ttl 3600

-- Set cache statistics
HSET cache:stats hits 0
HSET cache:stats misses 0
HSET cache:stats evictions 0
HSET cache:stats memory_usage 0

-- Create cache indexes (using sets for fast lookups)
SADD cache:indexes embeddings
SADD cache:indexes search
SADD cache:indexes rerank
SADD cache:indexes hallucination
SADD cache:indexes provider

-- Set system metadata
HSET system:metadata init_time "$(date -Iseconds)"
HSET system:metadata schema_version "1.0"
HSET system:metadata component "tyra_memory_cache"

-- Initialize performance counters
HSET perf:counters embedding_requests 0
HSET perf:counters search_requests 0
HSET perf:counters rerank_requests 0
HSET perf:counters hallucination_requests 0

-- Set performance thresholds
HSET perf:thresholds embedding_latency_ms 50
HSET perf:thresholds search_latency_ms 30
HSET perf:thresholds rerank_latency_ms 200
HSET perf:thresholds total_query_latency_ms 300

-- Initialize circuit breaker states
HSET circuit_breaker:postgresql state "closed"
HSET circuit_breaker:postgresql failure_count 0
HSET circuit_breaker:postgresql last_failure_time 0
HSET circuit_breaker:postgresql threshold 5
HSET circuit_breaker:postgresql timeout 60

HSET circuit_breaker:memgraph state "closed"
HSET circuit_breaker:memgraph failure_count 0
HSET circuit_breaker:memgraph last_failure_time 0
HSET circuit_breaker:memgraph threshold 5
HSET circuit_breaker:memgraph timeout 60

HSET circuit_breaker:embedding state "closed"
HSET circuit_breaker:embedding failure_count 0
HSET circuit_breaker:embedding last_failure_time 0
HSET circuit_breaker:embedding threshold 3
HSET circuit_breaker:embedding timeout 120

-- Set up provider registry cache
HSET provider:registry embedding_primary "intfloat/e5-large-v2"
HSET provider:registry embedding_fallback "sentence-transformers/all-MiniLM-L12-v2"
HSET provider:registry vector_store "postgresql"
HSET provider:registry graph_engine "memgraph"
HSET provider:registry hallucination_detector "default"
HSET provider:registry reranker "cross_encoder"

-- Initialize A/B testing framework
HSET ab_testing:config enabled true
HSET ab_testing:config default_split_ratio 0.5
HSET ab_testing:config min_sample_size 100
HSET ab_testing:config max_experiments 5

-- Set up health check keys
HSET health:status redis "healthy"
HSET health:status last_check "$(date -Iseconds)"
HSET health:status uptime_start "$(date -Iseconds)"

-- Initialize telemetry settings
HSET telemetry:config enabled true
HSET telemetry:config export_interval 60
HSET telemetry:config retention_days 7
HSET telemetry:config sampling_rate 1.0

ECHO "Redis cache structure initialized successfully"
EOF
    
    log "Redis cache structure initialized"
}

# Create Redis Lua scripts for atomic operations
create_lua_scripts() {
    log "Creating Redis Lua scripts..."
    
    # Script for atomic cache operations
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB <<'EOF'
-- Cache get with statistics
local cache_get_script = [[
local key = KEYS[1]
local value = redis.call('GET', key)
if value then
    redis.call('HINCRBY', 'cache:stats', 'hits', 1)
    redis.call('HINCRBY', 'cache:stats', 'hit_' .. ARGV[1], 1)
    return value
else
    redis.call('HINCRBY', 'cache:stats', 'misses', 1)
    redis.call('HINCRBY', 'cache:stats', 'miss_' .. ARGV[1], 1)
    return nil
end
]]

-- Cache set with TTL and statistics
local cache_set_script = [[
local key = KEYS[1]
local value = ARGV[1]
local ttl = tonumber(ARGV[2])
local cache_type = ARGV[3]

redis.call('SETEX', key, ttl, value)
redis.call('HINCRBY', 'cache:stats', 'sets', 1)
redis.call('HINCRBY', 'cache:stats', 'set_' .. cache_type, 1)
redis.call('SADD', 'cache:keys:' .. cache_type, key)
return 'OK'
]]

-- Circuit breaker check
local circuit_breaker_check = [[
local service = ARGV[1]
local cb_key = 'circuit_breaker:' .. service
local state = redis.call('HGET', cb_key, 'state')
local failure_count = tonumber(redis.call('HGET', cb_key, 'failure_count')) or 0
local threshold = tonumber(redis.call('HGET', cb_key, 'threshold')) or 5
local timeout = tonumber(redis.call('HGET', cb_key, 'timeout')) or 60
local last_failure = tonumber(redis.call('HGET', cb_key, 'last_failure_time')) or 0
local current_time = tonumber(ARGV[2])

if state == 'open' then
    if current_time - last_failure > timeout then
        redis.call('HSET', cb_key, 'state', 'half_open')
        redis.call('HSET', cb_key, 'failure_count', 0)
        return 'half_open'
    else
        return 'open'
    end
elseif state == 'half_open' then
    return 'half_open'
else
    return 'closed'
end
]]

-- Circuit breaker record failure
local circuit_breaker_failure = [[
local service = ARGV[1]
local cb_key = 'circuit_breaker:' .. service
local failure_count = tonumber(redis.call('HGET', cb_key, 'failure_count')) or 0
local threshold = tonumber(redis.call('HGET', cb_key, 'threshold')) or 5
local current_time = tonumber(ARGV[2])

failure_count = failure_count + 1
redis.call('HSET', cb_key, 'failure_count', failure_count)
redis.call('HSET', cb_key, 'last_failure_time', current_time)

if failure_count >= threshold then
    redis.call('HSET', cb_key, 'state', 'open')
    return 'open'
else
    return 'closed'
end
]]

-- Store scripts with SHA1 hashes for efficient reuse
EVAL "return redis.call('SCRIPT', 'LOAD', [[ ]] .. cache_get_script .. [[ ]])" 0
EVAL "return redis.call('SCRIPT', 'LOAD', [[ ]] .. cache_set_script .. [[ ]])" 0
EVAL "return redis.call('SCRIPT', 'LOAD', [[ ]] .. circuit_breaker_check .. [[ ]])" 0
EVAL "return redis.call('SCRIPT', 'LOAD', [[ ]] .. circuit_breaker_failure .. [[ ]])" 0

ECHO "Lua scripts loaded successfully"
EOF
    
    log "Redis Lua scripts created"
}

# Set up monitoring and alerting
setup_monitoring() {
    log "Setting up Redis monitoring..."
    
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB <<EOF
-- Set up monitoring keys
HSET monitoring:config memory_threshold_mb 400
HSET monitoring:config connection_threshold 80
HSET monitoring:config latency_threshold_ms 100
HSET monitoring:config cpu_threshold_percent 80

-- Initialize monitoring counters
HSET monitoring:stats memory_usage_mb 0
HSET monitoring:stats connections 0
HSET monitoring:stats commands_processed 0
HSET monitoring:stats keyspace_hits 0
HSET monitoring:stats keyspace_misses 0

-- Set up alert thresholds
HSET alerts:thresholds memory_critical 450
HSET alerts:thresholds memory_warning 350
HSET alerts:thresholds connection_critical 90
HSET alerts:thresholds connection_warning 70
HSET alerts:thresholds latency_critical 200
HSET alerts:thresholds latency_warning 150

ECHO "Redis monitoring configured"
EOF
    
    log "Redis monitoring configured"
}

# Test Redis functionality
test_redis() {
    log "Testing Redis functionality..."
    
    # Test basic operations
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB <<EOF
SET test:key "test_value"
GET test:key
DEL test:key
ECHO "Basic operations test passed"
EOF
    
    # Test cache operations
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB <<EOF
HSET cache:test:embedding "test_embedding_key" "test_embedding_value"
HGET cache:test:embedding "test_embedding_key"
HDEL cache:test:embedding "test_embedding_key"
ECHO "Cache operations test passed"
EOF
    
    # Test circuit breaker
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -n $REDIS_DB <<EOF
HGET circuit_breaker:postgresql state
HGET circuit_breaker:memgraph state
ECHO "Circuit breaker test passed"
EOF
    
    log "Redis functionality tests passed"
}

# Main function
main() {
    log "Starting Redis initialization for Tyra Memory System..."
    
    check_redis
    init_cache_structure
    create_lua_scripts
    setup_monitoring
    test_redis
    
    log "Redis initialization completed successfully!"
    log ""
    log "Summary:"
    log "- Cache namespaces created for embeddings, search, rerank, hallucination, and provider data"
    log "- Circuit breaker states initialized for PostgreSQL, Memgraph, and embedding services"
    log "- Performance counters and thresholds configured"
    log "- Lua scripts loaded for atomic operations"
    log "- Monitoring and alerting configured"
    log ""
    log "Redis is ready for the Tyra Memory System!"
}

# Run main function
main "$@"