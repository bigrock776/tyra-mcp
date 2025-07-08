# ⚙️ Configuration Guide - Tyra MCP Memory Server

## 📋 Overview

Tyra MCP Memory Server uses a layered configuration system:

1. **YAML Files**: Primary configuration in `config/`
2. **Environment Variables**: Runtime overrides via `.env`
3. **Command Line Args**: Temporary overrides
4. **Runtime Configs**: Dynamic configuration updates

## 🎯 Quick Configuration

### Minimal Setup

Copy and edit the environment file:
```bash
cp .env.example .env
nano .env
```

**Essential Variables:**
```env
# Database URLs
DATABASE_URL=postgresql://tyra:password@localhost:5432/tyra_memory
REDIS_URL=redis://localhost:6379/0
MEMGRAPH_URL=bolt://localhost:7687

# Embedding Models
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=auto

# Basic Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Quick Start Configurations

**Development (Fast, Less Accurate):**
```env
EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=cpu
CACHE_TTL_EMBEDDINGS=3600
RAG_RERANKING_ENABLED=false
```

**Production (Accurate, Resource Intensive):**
```env
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=cuda
CACHE_TTL_EMBEDDINGS=86400
RAG_RERANKING_ENABLED=true
RAG_HALLUCINATION_THRESHOLD=85
```

## 📁 Configuration Files

### Main Configuration (`config/config.yaml`)

```yaml
# Application Settings
application:
  name: "tyra-mcp-memory-server"
  version: "1.0.0"
  environment: "${ENVIRONMENT:development}"
  debug: "${DEBUG:false}"
  log_level: "${LOG_LEVEL:INFO}"

# Server Configuration
server:
  host: "${HOST:0.0.0.0}"
  port: "${PORT:8000}"
  workers: "${WORKERS:1}"
  reload: "${RELOAD:true}"

# Database Configuration
databases:
  postgres:
    host: "${POSTGRES_HOST:localhost}"
    port: "${POSTGRES_PORT:5432}"
    database: "${POSTGRES_DB:tyra_memory}"
    user: "${POSTGRES_USER:tyra}"
    password: "${POSTGRES_PASSWORD}"
    pool_size: "${POSTGRES_POOL_SIZE:20}"
    max_overflow: "${POSTGRES_MAX_OVERFLOW:30}"
    pool_timeout: "${POSTGRES_POOL_TIMEOUT:30}"
    echo: "${POSTGRES_ECHO:false}"

  redis:
    host: "${REDIS_HOST:localhost}"
    port: "${REDIS_PORT:6379}"
    db: "${REDIS_DB:0}"
    password: "${REDIS_PASSWORD:}"
    max_connections: "${REDIS_MAX_CONNECTIONS:50}"
    decode_responses: true

  memgraph:
    host: "${MEMGRAPH_HOST:localhost}"
    port: "${MEMGRAPH_PORT:7687}"
    user: "${MEMGRAPH_USER:}"
    password: "${MEMGRAPH_PASSWORD:}"
    database: "${MEMGRAPH_DATABASE:memgraph}"

# Embedding Configuration
embeddings:
  primary:
    provider: "huggingface"
    model: "${EMBEDDINGS_PRIMARY_MODEL:intfloat/e5-large-v2}"
    device: "${EMBEDDINGS_DEVICE:auto}"
    batch_size: "${EMBEDDINGS_BATCH_SIZE:32}"
    max_length: "${EMBEDDINGS_MAX_LENGTH:512}"
    normalize: "${EMBEDDINGS_NORMALIZE:true}"

  fallback:
    provider: "huggingface"
    model: "${EMBEDDINGS_FALLBACK_MODEL:sentence-transformers/all-MiniLM-L12-v2}"
    device: "cpu"
    batch_size: 16
    max_length: 256
    normalize: true

# RAG Configuration
rag:
  retrieval:
    default_top_k: "${RAG_DEFAULT_TOP_K:10}"
    max_top_k: "${RAG_MAX_TOP_K:50}"
    min_confidence: "${RAG_MIN_CONFIDENCE:0.3}"
    hybrid_weight: "${RAG_HYBRID_WEIGHT:0.7}"

  reranking:
    enabled: "${RAG_RERANKING_ENABLED:true}"
    provider: "cross_encoder"
    model: "${RAG_RERANKING_MODEL:ms-marco-MiniLM-L-6-v2}"
    top_k: "${RAG_RERANKING_TOP_K:20}"
    batch_size: "${RAG_RERANKING_BATCH_SIZE:8}"

  hallucination:
    enabled: "${RAG_HALLUCINATION_ENABLED:true}"
    threshold: "${RAG_HALLUCINATION_THRESHOLD:75}"
    grounding_weight: "${RAG_GROUNDING_WEIGHT:0.6}"
    consistency_weight: "${RAG_CONSISTENCY_WEIGHT:0.4}"

# Caching Configuration
cache:
  redis:
    enabled: "${CACHE_ENABLED:true}"
    ttl:
      embeddings: "${CACHE_TTL_EMBEDDINGS:86400}"  # 24 hours
      search: "${CACHE_TTL_SEARCH:3600}"           # 1 hour
      rerank: "${CACHE_TTL_RERANK:1800}"           # 30 minutes
      hallucination: "${CACHE_TTL_HALLUCINATION:900}"  # 15 minutes
      graph: "${CACHE_TTL_GRAPH:7200}"             # 2 hours
    compression: "${CACHE_COMPRESSION:true}"
    max_size: "${CACHE_MAX_SIZE:1000000}"  # 1M entries

# Observability Configuration
observability:
  enabled: "${OBSERVABILITY_ENABLED:true}"
  tracing:
    enabled: "${TRACING_ENABLED:true}"
    exporter: "${TRACING_EXPORTER:console}"  # console, jaeger, otlp
    service_name: "${TRACING_SERVICE_NAME:tyra-memory-server}"

  metrics:
    enabled: "${METRICS_ENABLED:true}"
    export_interval: "${METRICS_EXPORT_INTERVAL:60}"

  logging:
    structured: "${LOGGING_STRUCTURED:true}"
    correlation_id: "${LOGGING_CORRELATION_ID:true}"
    request_id: "${LOGGING_REQUEST_ID:true}"
```

### Provider Configuration (`config/providers.yaml`)

```yaml
# Vector Store Providers
vector_stores:
  pgvector:
    class: "src.core.providers.vector_stores.pgvector.PgVectorStore"
    config:
      connection_string: "${DATABASE_URL}"
      table_name: "${VECTOR_TABLE_NAME:memories}"
      vector_column: "${VECTOR_COLUMN:embedding}"
      dimensions: "${VECTOR_DIMENSIONS:1024}"

# Embedding Providers
embedding_providers:
  huggingface:
    class: "src.core.providers.embeddings.huggingface.HuggingFaceEmbedder"
    config:
      cache_dir: "${HF_CACHE_DIR:~/.cache/huggingface}"
      trust_remote_code: false

# Graph Engine Providers
graph_engines:
  memgraph:
    class: "src.core.providers.graph_engines.memgraph.MemgraphEngine"
    config:
      connection_string: "${MEMGRAPH_URL}"
      auto_commit: true

# Reranking Providers
rerankers:
  cross_encoder:
    class: "src.core.providers.rerankers.cross_encoder.CrossEncoderReranker"
    config:
      cache_dir: "${RERANKER_CACHE_DIR:~/.cache/rerankers}"
```

### Agent Configuration (`config/agents.yaml`)

```yaml
# Agent-specific configurations
agents:
  default:
    memory_limit: "${AGENT_MEMORY_LIMIT:10000}"
    session_timeout: "${AGENT_SESSION_TIMEOUT:3600}"
    enable_graph: "${AGENT_ENABLE_GRAPH:true}"
    enable_reranking: "${AGENT_ENABLE_RERANKING:true}"
    confidence_threshold: "${AGENT_CONFIDENCE_THRESHOLD:60}"

  claude:
    memory_limit: 50000
    session_timeout: 7200
    enable_graph: true
    enable_reranking: true
    confidence_threshold: 70

  tyra:
    memory_limit: 100000
    session_timeout: 14400
    enable_graph: true
    enable_reranking: true
    confidence_threshold: 85
    safety_mode: true
```

## 🔧 Environment Variables

### Required Variables

```env
# Database connections
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port/db
MEMGRAPH_URL=bolt://host:port

# Security
SECRET_KEY=your-super-secret-key-here
API_KEY=your-api-key-for-external-access
```

### Optional Variables

```env
# Performance Tuning
WORKERS=4
POSTGRES_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=50
EMBEDDINGS_BATCH_SIZE=32

# Feature Flags
RAG_RERANKING_ENABLED=true
RAG_HALLUCINATION_ENABLED=true
CACHE_ENABLED=true
OBSERVABILITY_ENABLED=true

# Development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
```

### Model Configuration

```env
# Primary embedding model (large, accurate)
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=cuda  # or cpu, auto

# Fallback model (small, fast)
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2

# Reranking model
RAG_RERANKING_MODEL=ms-marco-MiniLM-L-6-v2

# Model cache directory
HF_CACHE_DIR=/data/models/huggingface
```

## 🎛️ Advanced Configuration

### Performance Optimization

**For High Memory Usage:**
```yaml
# Reduce memory footprint
embeddings:
  primary:
    batch_size: 16
    max_length: 256

databases:
  postgres:
    pool_size: 10
    max_overflow: 15

cache:
  redis:
    max_size: 100000
```

**For High Throughput:**
```yaml
# Optimize for speed
server:
  workers: 8

databases:
  postgres:
    pool_size: 50
    max_overflow: 100

embeddings:
  primary:
    batch_size: 64
    device: "cuda"

cache:
  redis:
    enabled: true
    compression: false
```

### Security Configuration

```yaml
# Security settings
security:
  api_key_required: true
  rate_limiting:
    enabled: true
    requests_per_minute: 100

  cors:
    allowed_origins: ["http://localhost:3000"]
    allow_credentials: true

  authentication:
    type: "api_key"  # or bearer_token, oauth
    header: "X-API-Key"
```

### Monitoring Configuration

```yaml
observability:
  tracing:
    enabled: true
    exporter: "jaeger"
    jaeger_endpoint: "http://localhost:14268/api/traces"
    sample_rate: 0.1

  metrics:
    enabled: true
    prometheus_port: 9090
    custom_metrics:
      - memory_operations_total
      - embedding_generation_duration
      - rag_pipeline_duration

  alerts:
    enabled: true
    webhook_url: "https://hooks.slack.com/..."
    thresholds:
      error_rate: 0.05
      response_time_p95: 1000
```

## 🔄 Dynamic Configuration

### Runtime Updates

```python
# Update configuration at runtime
from src.core.utils.config import update_config

# Update cache TTL
await update_config("cache.redis.ttl.embeddings", 7200)

# Update model batch size
await update_config("embeddings.primary.batch_size", 16)

# Enable/disable features
await update_config("rag.reranking.enabled", False)
```

### Configuration Hot Reload

```bash
# Send SIGHUP to reload configuration
kill -HUP $(pidof python)

# Or use API endpoint
curl -X POST http://localhost:8000/admin/reload-config
```

## 🧪 Configuration Validation

### Validate Configuration

```bash
# Validate all configuration files
python -c "
from src.core.utils.config import validate_config
validate_config()
print('✅ Configuration is valid')
"

# Check specific sections
python -c "
from src.core.utils.config import load_config
config = load_config()
print(f'Database URL: {config.databases.postgres.url}')
print(f'Primary model: {config.embeddings.primary.model}')
"
```

### Test Configuration

```python
# Test database connections
from src.core.utils.database import test_all_connections
import asyncio

async def test():
    results = await test_all_connections()
    for service, status in results.items():
        print(f"{service}: {'✅' if status else '❌'}")

asyncio.run(test())
```

## 📊 Configuration Presets

### Development Preset

```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
EMBEDDINGS_DEVICE=cpu
EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
RAG_RERANKING_ENABLED=false
CACHE_TTL_EMBEDDINGS=300
```

### Testing Preset

```env
ENVIRONMENT=testing
DATABASE_URL=postgresql://test:test@localhost:5432/test_db
REDIS_URL=redis://localhost:6379/1
EMBEDDINGS_DEVICE=cpu
CACHE_ENABLED=false
OBSERVABILITY_ENABLED=false
```

### Production Preset

```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
WORKERS=8
EMBEDDINGS_DEVICE=cuda
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
RAG_RERANKING_ENABLED=true
RAG_HALLUCINATION_ENABLED=true
CACHE_ENABLED=true
OBSERVABILITY_ENABLED=true
```

## 🚨 Troubleshooting Configuration

### Common Issues

**1. Model Loading Errors**
```env
# Increase timeout for large models
MODEL_LOAD_TIMEOUT=300

# Specify exact cache directory
HF_CACHE_DIR=/path/to/cache
```

**2. Database Connection Issues**
```yaml
databases:
  postgres:
    pool_timeout: 60
    pool_recycle: 3600
    connect_args:
      sslmode: "prefer"
      connect_timeout: 30
```

**3. Memory Issues**
```yaml
embeddings:
  primary:
    batch_size: 8
    device: "cpu"

cache:
  redis:
    max_size: 50000
```

### Configuration Debugging

```bash
# Show current configuration
make config-show

# Validate configuration
make config-validate

# Show environment variables
make env-show
```

---

🎯 **Configuration Complete!** Your memory server is now optimally configured for your environment.
