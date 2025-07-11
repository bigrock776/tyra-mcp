# ⚙️ Configuration Guide - Tyra MCP Memory Server

## ⚠️ **PREREQUISITES - MANUAL MODEL INSTALLATION REQUIRED**

**CRITICAL**: Before configuring the system, you must manually download required models:

```bash
# Install HuggingFace CLI
pip install huggingface-hub
git lfs install

# Download models to local directories
mkdir -p ./models/embeddings ./models/cross-encoders

huggingface-cli download intfloat/e5-large-v2 \
  --local-dir ./models/embeddings/e5-large-v2 \
  --local-dir-use-symlinks False

huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 \
  --local-dir ./models/embeddings/all-MiniLM-L12-v2 \
  --local-dir-use-symlinks False

huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2 \
  --local-dir-use-symlinks False
```

## 📋 Overview

Tyra MCP Memory Server uses a layered configuration system that provides flexibility and maintainability:

1. **YAML Files**: Primary configuration in `config/` directory
2. **Environment Variables**: Runtime overrides via `.env` file
3. **Command Line Arguments**: Temporary overrides for testing
4. **Runtime Configuration**: Dynamic updates via API endpoints

## 🎯 Quick Configuration

### Essential Setup

1. **Copy Environment Template**
   ```bash
   cp .env.example .env
   ```

2. **Edit Essential Variables**
   ```bash
   nano .env
   ```

3. **Required Configuration**
   ```env
   # Database Connections
   DATABASE_URL=postgresql://tyra:password@localhost:5432/tyra_memory
   REDIS_URL=redis://localhost:6379/0
   MEMGRAPH_URL=bolt://localhost:7687

   # Embedding Models - LOCAL PATHS REQUIRED
   EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
   EMBEDDINGS_PRIMARY_PATH=./models/embeddings/e5-large-v2
   EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
   EMBEDDINGS_FALLBACK_PATH=./models/embeddings/all-MiniLM-L12-v2
   EMBEDDINGS_USE_LOCAL_FILES=true
   EMBEDDINGS_DEVICE=auto

   # Application Settings
   ENVIRONMENT=development
   LOG_LEVEL=INFO
   ```

## 🗂️ Configuration Files

### Main Configuration (`config/config.yaml`)

```yaml
# Application Settings
app:
  name: "Tyra MCP Memory Server"
  version: "1.0.0"
  environment: ${ENVIRONMENT:-development}
  debug: ${DEBUG:-false}
  log_level: ${LOG_LEVEL:-INFO}

# Server Configuration
server:
  mcp:
    host: ${MCP_HOST:-localhost}
    port: ${MCP_PORT:-3000}
    transport: ${MCP_TRANSPORT:-stdio}
    timeout: ${MCP_TIMEOUT:-30}
  
  fastapi:
    host: ${API_HOST:-0.0.0.0}
    port: ${API_PORT:-8000}
    workers: ${API_WORKERS:-4}
    reload: ${API_RELOAD:-false}
    enable_docs: ${API_ENABLE_DOCS:-true}
    cors_origins: ${API_CORS_ORIGINS:-["*"]}

# Database Configuration
database:
  postgresql:
    url: ${DATABASE_URL:-postgresql://tyra:tyra123@localhost:5432/tyra_memory}
    pool_size: ${POSTGRES_POOL_SIZE:-20}
    max_overflow: ${POSTGRES_MAX_OVERFLOW:-10}
    pool_timeout: ${POSTGRES_POOL_TIMEOUT:-30}
    pool_recycle: ${POSTGRES_POOL_RECYCLE:-3600}
    ssl_mode: ${POSTGRES_SSL_MODE:-prefer}
    
  redis:
    url: ${REDIS_URL:-redis://localhost:6379/0}
    max_connections: ${REDIS_MAX_CONNECTIONS:-50}
    connection_timeout: ${REDIS_CONNECTION_TIMEOUT:-5}
    socket_timeout: ${REDIS_SOCKET_TIMEOUT:-5}
    retry_on_timeout: true
    
  memgraph:
    url: ${MEMGRAPH_URL:-bolt://localhost:7687}
    username: ${MEMGRAPH_USERNAME:-}
    password: ${MEMGRAPH_PASSWORD:-}
    connection_timeout: ${MEMGRAPH_CONNECTION_TIMEOUT:-30}
    pool_size: ${MEMGRAPH_POOL_SIZE:-10}

# Memory Configuration
memory:
  backend: postgres
  vector_dimensions: 1024
  similarity_threshold: 0.7
  max_results: 100
  chunk_size: 1000
  chunk_overlap: 200
  
# Security Settings
security:
  secret_key: ${SECRET_KEY:-your-secret-key-here}
  api_key: ${API_KEY:-your-api-key-here}
  jwt_secret: ${JWT_SECRET:-your-jwt-secret-here}
  cors_enabled: ${CORS_ENABLED:-true}
  rate_limiting:
    enabled: ${RATE_LIMIT_ENABLED:-true}
    requests_per_minute: ${RATE_LIMIT_RPM:-100}
    burst: ${RATE_LIMIT_BURST:-20}
```

### Provider Configuration (`config/providers.yaml`)

```yaml
# Embedding Providers
embeddings:
  primary:
    provider: huggingface
    model: ${EMBEDDINGS_PRIMARY_MODEL:-intfloat/e5-large-v2}
    device: ${EMBEDDINGS_DEVICE:-auto}
    batch_size: ${EMBEDDINGS_BATCH_SIZE:-32}
    max_length: ${EMBEDDINGS_MAX_LENGTH:-512}
    normalize: true
    
  fallback:
    provider: huggingface_fallback
    model: ${EMBEDDINGS_FALLBACK_MODEL:-sentence-transformers/all-MiniLM-L12-v2}
    device: cpu
    batch_size: ${EMBEDDINGS_FALLBACK_BATCH_SIZE:-16}
    max_length: ${EMBEDDINGS_FALLBACK_MAX_LENGTH:-256}
    normalize: true

# Vector Store Providers
vector_stores:
  default: pgvector
  pgvector:
    provider: pgvector
    connection_string: ${DATABASE_URL}
    table_name: ${VECTOR_TABLE_NAME:-memory_vectors}
    index_type: ${VECTOR_INDEX_TYPE:-hnsw}
    index_params:
      m: ${HNSW_M:-16}
      ef_construction: ${HNSW_EF_CONSTRUCTION:-64}

# Graph Engine Providers
graph_engines:
  default: memgraph
  memgraph:
    provider: memgraph
    connection_string: ${MEMGRAPH_URL}
    timeout: ${MEMGRAPH_TIMEOUT:-30}
    pool_size: ${MEMGRAPH_POOL_SIZE:-10}

# Reranker Providers
rerankers:
  default: cross_encoder
  cross_encoder:
    provider: cross_encoder
    model: ${RERANKER_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}
    device: ${RERANKER_DEVICE:-auto}
    batch_size: ${RERANKER_BATCH_SIZE:-8}
    top_k: ${RERANKER_TOP_K:-10}
    
  vllm:
    provider: vllm
    model: ${VLLM_MODEL:-microsoft/DialoGPT-medium}
    endpoint: ${VLLM_ENDPOINT:-http://localhost:8001/v1}
    api_key: ${VLLM_API_KEY:-}
    timeout: ${VLLM_TIMEOUT:-30}
```

### Model Configuration (`config/models.yaml`)

```yaml
# Embedding Models
embedding_models:
  intfloat/e5-large-v2:
    dimensions: 1024
    max_length: 512
    device_preference: ["cuda", "cpu"]
    memory_usage: "high"
    accuracy: "high"
    
  sentence-transformers/all-MiniLM-L12-v2:
    dimensions: 384
    max_length: 256
    device_preference: ["cpu"]
    memory_usage: "low"
    accuracy: "medium"

# Reranking Models
reranking_models:
  cross-encoder/ms-marco-MiniLM-L-6-v2:
    type: "cross_encoder"
    device_preference: ["cuda", "cpu"]
    memory_usage: "medium"
    accuracy: "high"
    
  cross-encoder/ms-marco-TinyBERT-L-2-v2:
    type: "cross_encoder"
    device_preference: ["cpu"]
    memory_usage: "low"
    accuracy: "medium"

# Model Download Settings
model_cache:
  directory: ${MODEL_CACHE_DIR:-./data/models}
  max_size_gb: ${MODEL_CACHE_MAX_SIZE:-50}
  auto_download: ${MODEL_AUTO_DOWNLOAD:-true}
  timeout: ${MODEL_DOWNLOAD_TIMEOUT:-300}
```

### RAG Configuration (`config/rag.yaml`)

```yaml
# Retrieval Settings
retrieval:
  top_k: ${RAG_TOP_K:-20}
  min_confidence: ${RAG_MIN_CONFIDENCE:-0.0}
  hybrid_weight: ${RAG_HYBRID_WEIGHT:-0.7}
  diversity_penalty: ${RAG_DIVERSITY_PENALTY:-0.3}
  max_context_length: ${RAG_MAX_CONTEXT_LENGTH:-4000}

# Reranking Settings
reranking:
  enabled: ${RAG_RERANKING_ENABLED:-true}
  provider: ${RAG_RERANKING_PROVIDER:-cross_encoder}
  top_k: ${RAG_RERANKING_TOP_K:-10}
  score_threshold: ${RAG_RERANKING_THRESHOLD:-0.5}
  batch_size: ${RAG_RERANKING_BATCH_SIZE:-8}

# Hallucination Detection
hallucination:
  enabled: ${RAG_HALLUCINATION_ENABLED:-true}
  threshold: ${RAG_HALLUCINATION_THRESHOLD:-75}
  confidence_levels:
    rock_solid: 95
    high: 80
    fuzzy: 60
    low: 0
  grounding_check: ${RAG_GROUNDING_CHECK:-true}
  evidence_collection: ${RAG_EVIDENCE_COLLECTION:-true}

# Response Generation
response:
  max_tokens: ${RAG_MAX_TOKENS:-2000}
  temperature: ${RAG_TEMPERATURE:-0.1}
  include_sources: ${RAG_INCLUDE_SOURCES:-true}
  include_confidence: ${RAG_INCLUDE_CONFIDENCE:-true}
```

### Caching Configuration (`config/cache.yaml`)

```yaml
# Cache Settings
cache:
  enabled: ${CACHE_ENABLED:-true}
  compression: ${CACHE_COMPRESSION:-true}
  compression_threshold: ${CACHE_COMPRESSION_THRESHOLD:-1024}
  default_ttl: ${CACHE_DEFAULT_TTL:-3600}

# Cache Layers
layers:
  l1:
    type: "memory"
    max_size: ${CACHE_L1_MAX_SIZE:-1000}
    ttl: ${CACHE_L1_TTL:-300}
    
  l2:
    type: "redis"
    max_size: ${CACHE_L2_MAX_SIZE:-10000}
    ttl: ${CACHE_L2_TTL:-3600}

# Cache TTL Settings (seconds)
ttl:
  embeddings: ${CACHE_TTL_EMBEDDINGS:-86400}      # 24 hours
  search: ${CACHE_TTL_SEARCH:-3600}               # 1 hour
  rerank: ${CACHE_TTL_RERANK:-1800}               # 30 minutes
  hallucination: ${CACHE_TTL_HALLUCINATION:-900}  # 15 minutes
  graph: ${CACHE_TTL_GRAPH:-7200}                 # 2 hours
  health: ${CACHE_TTL_HEALTH:-60}                 # 1 minute

# Cache Warming
warming:
  enabled: ${CACHE_WARMING_ENABLED:-false}
  strategies: ["popular", "recent", "scheduled"]
  schedule: "0 2 * * *"  # Daily at 2 AM
```

### Document Ingestion Configuration (`config/ingestion.yaml`)

```yaml
# Document Ingestion System
ingestion:
  # File Processing Settings
  file_processing:
    max_file_size: ${INGESTION_MAX_FILE_SIZE:-104857600}  # 100MB
    max_batch_size: ${INGESTION_MAX_BATCH_SIZE:-100}
    concurrent_limit: ${INGESTION_CONCURRENT_LIMIT:-20}
    timeout_seconds: ${INGESTION_TIMEOUT:-300}
    temp_directory: ${INGESTION_TEMP_DIR:-/tmp/tyra_ingestion}
    
  # Supported File Types
  supported_formats:
    pdf:
      enabled: ${INGESTION_PDF_ENABLED:-true}
      max_size: "50MB"
      loader: "PyMuPDF"
      features: ["text_extraction", "metadata_extraction"]
      
    docx:
      enabled: ${INGESTION_DOCX_ENABLED:-true}
      max_size: "25MB"
      loader: "python-docx"
      features: ["paragraph_detection", "table_extraction"]
      
    pptx:
      enabled: ${INGESTION_PPTX_ENABLED:-true}
      max_size: "25MB"
      loader: "python-pptx"
      features: ["slide_extraction", "speaker_notes"]
      
    txt:
      enabled: ${INGESTION_TXT_ENABLED:-true}
      max_size: "10MB"
      encoding_detection: true
      
    markdown:
      enabled: ${INGESTION_MD_ENABLED:-true}
      max_size: "10MB"
      features: ["header_detection", "structure_preservation"]
      
    html:
      enabled: ${INGESTION_HTML_ENABLED:-true}
      max_size: "10MB"
      converter: "html2text"
      
    json:
      enabled: ${INGESTION_JSON_ENABLED:-true}
      max_size: "50MB"
      features: ["nested_object_handling", "array_processing"]
      
    csv:
      enabled: ${INGESTION_CSV_ENABLED:-true}
      max_size: "100MB"
      features: ["header_detection", "streaming_processing"]
      
    epub:
      enabled: ${INGESTION_EPUB_ENABLED:-true}
      max_size: "25MB"
      features: ["chapter_extraction", "metadata_extraction"]

  # Chunking Strategies
  chunking:
    default_strategy: ${INGESTION_DEFAULT_CHUNKING:-auto}
    default_chunk_size: ${INGESTION_DEFAULT_CHUNK_SIZE:-512}
    default_overlap: ${INGESTION_DEFAULT_OVERLAP:-50}
    
    strategies:
      auto:
        enabled: true
        file_type_mapping:
          pdf: "semantic"
          docx: "paragraph"
          pptx: "slide"
          txt: "paragraph"
          md: "paragraph"
          html: "paragraph"
          json: "object"
          csv: "row"
          epub: "semantic"
          
      paragraph:
        enabled: true
        min_chunk_size: 100
        max_chunk_size: 2000
        overlap_ratio: 0.1
        
      semantic:
        enabled: true
        similarity_threshold: 0.7
        min_chunk_size: 200
        max_chunk_size: 1500
        
      slide:
        enabled: true
        group_slides: true
        include_speaker_notes: true
        
      line:
        enabled: true
        lines_per_chunk: 10
        preserve_structure: true
        
      token:
        enabled: true
        tokens_per_chunk: 400
        tokenizer: "cl100k_base"

  # LLM Context Enhancement
  llm_enhancement:
    enabled: ${INGESTION_LLM_ENHANCEMENT:-true}
    default_mode: ${INGESTION_LLM_MODE:-rule_based}  # rule_based, vllm, disabled
    
    rule_based:
      enabled: true
      templates:
        default: "This content is from {file_name} ({file_type}): {description}"
        pdf: "From PDF document '{file_name}': {description}. Page context: {page_info}"
        docx: "From Word document '{file_name}': {description}. Section: {section_info}"
        
    vllm_integration:
      enabled: ${INGESTION_VLLM_ENABLED:-false}
      endpoint: ${VLLM_ENDPOINT:-http://localhost:8000/v1}
      model: ${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
      timeout: ${VLLM_TIMEOUT:-30}
      max_tokens: ${VLLM_MAX_TOKENS:-150}
      temperature: ${VLLM_TEMPERATURE:-0.3}
      
    confidence_scoring:
      enabled: true
      min_confidence: 0.5
      confidence_sources: ["content_match", "structure_analysis", "llm_assessment"]
      
    hallucination_detection:
      enabled: true
      threshold: ${INGESTION_HALLUCINATION_THRESHOLD:-0.8}
      validation_methods: ["grounding_check", "consistency_analysis"]

  # Storage Integration
  storage:
    auto_embed: ${INGESTION_AUTO_EMBED:-true}
    auto_graph: ${INGESTION_AUTO_GRAPH:-true}
    extract_entities: ${INGESTION_EXTRACT_ENTITIES:-true}
    create_relationships: ${INGESTION_CREATE_RELATIONSHIPS:-true}
    
  # Error Handling
  error_handling:
    retry_attempts: ${INGESTION_RETRY_ATTEMPTS:-3}
    retry_delay: ${INGESTION_RETRY_DELAY:-1.0}
    fallback_strategy: "graceful_degradation"  # strict, graceful_degradation, skip
    log_failures: true
    
  # Performance Optimization
  performance:
    streaming_threshold: ${INGESTION_STREAMING_THRESHOLD:-10485760}  # 10MB
    batch_processing: true
    parallel_chunks: ${INGESTION_PARALLEL_CHUNKS:-5}
    cache_parsed_content: ${INGESTION_CACHE_CONTENT:-true}
    cache_ttl: ${INGESTION_CACHE_TTL:-3600}  # 1 hour
```

### Observability Configuration (`config/observability.yaml`)

```yaml
# OpenTelemetry Configuration
otel:
  enabled: ${OTEL_ENABLED:-true}
  service_name: ${OTEL_SERVICE_NAME:-tyra-mcp-memory-server}
  service_version: ${OTEL_SERVICE_VERSION:-1.0.0}
  environment: ${OTEL_ENVIRONMENT:-development}

# Tracing Configuration
tracing:
  enabled: ${OTEL_TRACES_ENABLED:-true}
  exporter: ${OTEL_TRACES_EXPORTER:-console}
  endpoint: ${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4318/v1/traces}
  sampler: ${OTEL_TRACES_SAMPLER:-parentbased_traceidratio}
  sampler_arg: ${OTEL_TRACES_SAMPLER_ARG:-1.0}
  max_spans: ${OTEL_TRACES_MAX_SPANS:-1000}

# Metrics Configuration
metrics:
  enabled: ${OTEL_METRICS_ENABLED:-true}
  exporter: ${OTEL_METRICS_EXPORTER:-console}
  endpoint: ${OTEL_EXPORTER_OTLP_METRICS_ENDPOINT:-http://localhost:4318/v1/metrics}
  export_interval: ${OTEL_METRIC_EXPORT_INTERVAL:-60000}
  export_timeout: ${OTEL_METRIC_EXPORT_TIMEOUT:-30000}

# Logging Configuration
logging:
  enabled: ${OTEL_LOGS_ENABLED:-true}
  exporter: ${OTEL_LOGS_EXPORTER:-console}
  level: ${LOG_LEVEL:-INFO}
  format: ${LOG_FORMAT:-json}
  rotation:
    enabled: ${LOG_ROTATION_ENABLED:-true}
    max_size: ${LOG_ROTATION_MAX_SIZE:-100MB}
    max_files: ${LOG_ROTATION_MAX_FILES:-10}
    max_age: ${LOG_ROTATION_MAX_AGE:-30}
```

### Self-Learning Configuration (`config/self_learning.yaml`)

```yaml
# Self-Learning System
self_learning:
  enabled: ${SELF_LEARNING_ENABLED:-true}
  analysis_interval: ${SELF_LEARNING_ANALYSIS_INTERVAL:-3600}  # 1 hour
  improvement_interval: ${SELF_LEARNING_IMPROVEMENT_INTERVAL:-86400}  # 24 hours
  auto_optimize: ${SELF_LEARNING_AUTO_OPTIMIZE:-true}

# Performance Tracking
performance:
  tracking_enabled: ${PERFORMANCE_TRACKING_ENABLED:-true}
  sample_rate: ${PERFORMANCE_SAMPLE_RATE:-0.1}
  slow_query_threshold: ${PERFORMANCE_SLOW_QUERY_THRESHOLD:-1000}
  metrics_retention_days: ${PERFORMANCE_METRICS_RETENTION:-30}

# Memory Health Management
memory_health:
  enabled: ${MEMORY_HEALTH_ENABLED:-true}
  check_interval: ${MEMORY_HEALTH_CHECK_INTERVAL:-3600}  # 1 hour
  cleanup_interval: ${MEMORY_HEALTH_CLEANUP_INTERVAL:-86400}  # 24 hours
  stale_threshold_days: ${MEMORY_HEALTH_STALE_THRESHOLD:-30}
  redundancy_threshold: ${MEMORY_HEALTH_REDUNDANCY_THRESHOLD:-0.9}

# A/B Testing
ab_testing:
  enabled: ${AB_TESTING_ENABLED:-true}
  default_split: ${AB_TESTING_DEFAULT_SPLIT:-0.5}
  min_sample_size: ${AB_TESTING_MIN_SAMPLE_SIZE:-100}
  significance_level: ${AB_TESTING_SIGNIFICANCE_LEVEL:-0.05}
  max_experiments: ${AB_TESTING_MAX_EXPERIMENTS:-5}

# Adaptation Thresholds
thresholds:
  memory_quality:
    excellent: 0.9
    good: 0.8
    fair: 0.7
    poor: 0.6
  performance:
    response_time_ms: 500
    cache_hit_rate: 0.8
    error_rate: 0.01
  confidence:
    rock_solid: 0.95
    high: 0.8
    fuzzy: 0.6
    low: 0.4
```

## 🌍 Environment Variables

### Database Configuration

```env
# PostgreSQL
DATABASE_URL=postgresql://user:password@host:port/database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tyra_memory
POSTGRES_USER=tyra
POSTGRES_PASSWORD=secure_password
POSTGRES_SSL_MODE=prefer
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=10
POSTGRES_POOL_TIMEOUT=30

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_MAX_CONNECTIONS=50
REDIS_CONNECTION_TIMEOUT=5

# Memgraph
MEMGRAPH_URL=bolt://localhost:7687
MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687
MEMGRAPH_USERNAME=
MEMGRAPH_PASSWORD=
MEMGRAPH_CONNECTION_TIMEOUT=30
MEMGRAPH_POOL_SIZE=10
```

### Application Configuration

```env
# Environment
ENVIRONMENT=development  # development, production, testing
DEBUG=false
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false
API_ENABLE_DOCS=true
API_CORS_ORIGINS=["*"]

# MCP Settings
MCP_HOST=localhost
MCP_PORT=3000
MCP_TRANSPORT=stdio  # stdio, sse
MCP_TIMEOUT=30
MCP_LOG_LEVEL=INFO
```

### Embedding Configuration

```env
# Primary Embedding Model
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=auto  # auto, cpu, cuda
EMBEDDINGS_BATCH_SIZE=32
EMBEDDINGS_MAX_LENGTH=512

# Fallback Embedding Model
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_FALLBACK_DEVICE=cpu
EMBEDDINGS_FALLBACK_BATCH_SIZE=16
EMBEDDINGS_FALLBACK_MAX_LENGTH=256

# Model Cache
MODEL_CACHE_DIR=./data/models
MODEL_CACHE_MAX_SIZE=50
MODEL_AUTO_DOWNLOAD=true
MODEL_DOWNLOAD_TIMEOUT=300
```

### RAG Configuration

```env
# Retrieval Settings
RAG_TOP_K=20
RAG_MIN_CONFIDENCE=0.0
RAG_HYBRID_WEIGHT=0.7
RAG_DIVERSITY_PENALTY=0.3
RAG_MAX_CONTEXT_LENGTH=4000

# Reranking Settings
RAG_RERANKING_ENABLED=true
RAG_RERANKING_PROVIDER=cross_encoder
RAG_RERANKING_TOP_K=10
RAG_RERANKING_THRESHOLD=0.5
RAG_RERANKING_BATCH_SIZE=8

# Hallucination Detection
RAG_HALLUCINATION_ENABLED=true
RAG_HALLUCINATION_THRESHOLD=75
RAG_GROUNDING_CHECK=true
RAG_EVIDENCE_COLLECTION=true

# Response Generation
RAG_MAX_TOKENS=2000
RAG_TEMPERATURE=0.1
RAG_INCLUDE_SOURCES=true
RAG_INCLUDE_CONFIDENCE=true
```

### Cache Configuration

```env
# Cache Settings
CACHE_ENABLED=true
CACHE_COMPRESSION=true
CACHE_COMPRESSION_THRESHOLD=1024
CACHE_DEFAULT_TTL=3600

# Cache TTL Settings (seconds)
CACHE_TTL_EMBEDDINGS=86400      # 24 hours
CACHE_TTL_SEARCH=3600           # 1 hour
CACHE_TTL_RERANK=1800           # 30 minutes
CACHE_TTL_HALLUCINATION=900     # 15 minutes
CACHE_TTL_GRAPH=7200            # 2 hours

# Cache Warming
CACHE_WARMING_ENABLED=false
```

### Security Configuration

```env
# Authentication
SECRET_KEY=your-secret-key-here-generate-with-openssl-rand-hex-32
API_KEY=your-api-key-here
JWT_SECRET=your-jwt-secret-here

# CORS
CORS_ENABLED=true
API_CORS_ORIGINS=["*"]

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_RPM=100
RATE_LIMIT_BURST=20

# SSL/TLS
SSL_ENABLED=false
SSL_CERT_PATH=/etc/ssl/certs/tyra.crt
SSL_KEY_PATH=/etc/ssl/private/tyra.key
```

### Observability Configuration

```env
# OpenTelemetry
OTEL_ENABLED=true
OTEL_SERVICE_NAME=tyra-mcp-memory-server
OTEL_SERVICE_VERSION=1.0.0
OTEL_ENVIRONMENT=development

# Tracing
OTEL_TRACES_ENABLED=true
OTEL_TRACES_EXPORTER=console  # console, jaeger, otlp
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=1.0

# Metrics
OTEL_METRICS_ENABLED=true
OTEL_METRICS_EXPORTER=console  # console, prometheus, otlp
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4318/v1/metrics
OTEL_METRIC_EXPORT_INTERVAL=60000

# Logging
OTEL_LOGS_ENABLED=true
OTEL_LOGS_EXPORTER=console
LOG_FORMAT=json  # json, text
LOG_ROTATION_ENABLED=true
LOG_ROTATION_MAX_SIZE=100MB
LOG_ROTATION_MAX_FILES=10
LOG_ROTATION_MAX_AGE=30
```

### Self-Learning Configuration

```env
# Self-Learning
SELF_LEARNING_ENABLED=true
SELF_LEARNING_ANALYSIS_INTERVAL=3600
SELF_LEARNING_IMPROVEMENT_INTERVAL=86400
SELF_LEARNING_AUTO_OPTIMIZE=true

# Performance Tracking
PERFORMANCE_TRACKING_ENABLED=true
PERFORMANCE_SAMPLE_RATE=0.1
PERFORMANCE_SLOW_QUERY_THRESHOLD=1000
PERFORMANCE_METRICS_RETENTION=30

# Memory Health
MEMORY_HEALTH_ENABLED=true
MEMORY_HEALTH_CHECK_INTERVAL=3600
MEMORY_HEALTH_CLEANUP_INTERVAL=86400
MEMORY_HEALTH_STALE_THRESHOLD=30
MEMORY_HEALTH_REDUNDANCY_THRESHOLD=0.9

# A/B Testing
AB_TESTING_ENABLED=true
AB_TESTING_DEFAULT_SPLIT=0.5
AB_TESTING_MIN_SAMPLE_SIZE=100
AB_TESTING_SIGNIFICANCE_LEVEL=0.05
AB_TESTING_MAX_EXPERIMENTS=5
```

## 🎛️ Configuration Presets

### Development Environment

```env
# Development optimized for speed and debugging
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
API_RELOAD=true
API_ENABLE_DOCS=true

# Lightweight models for faster startup
EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=cpu
EMBEDDINGS_BATCH_SIZE=16

# Reduced caching for development
CACHE_TTL_EMBEDDINGS=3600
CACHE_TTL_SEARCH=300
RAG_RERANKING_ENABLED=false

# Disabled features for development
SELF_LEARNING_ENABLED=false
AB_TESTING_ENABLED=false
OTEL_TRACES_SAMPLER_ARG=0.1
```

### Production Environment

```env
# Production optimized for performance and reliability
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
API_RELOAD=false
API_ENABLE_DOCS=false

# High-quality models for accuracy
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=auto
EMBEDDINGS_BATCH_SIZE=32

# Optimized caching
CACHE_ENABLED=true
CACHE_TTL_EMBEDDINGS=86400
CACHE_TTL_SEARCH=3600
CACHE_WARMING_ENABLED=true

# Full feature set
RAG_RERANKING_ENABLED=true
RAG_HALLUCINATION_ENABLED=true
SELF_LEARNING_ENABLED=true
AB_TESTING_ENABLED=true

# Production monitoring
OTEL_TRACES_ENABLED=true
OTEL_METRICS_ENABLED=true
OTEL_TRACES_SAMPLER_ARG=0.1
PERFORMANCE_TRACKING_ENABLED=true
```

### Testing Environment

```env
# Testing optimized for isolation and speed
ENVIRONMENT=testing
DEBUG=true
LOG_LEVEL=DEBUG

# Separate test databases
DATABASE_URL=postgresql://test_user:test_password@localhost:5432/test_tyra_memory
REDIS_URL=redis://localhost:6379/1
MEMGRAPH_URL=bolt://localhost:7687

# Fast models for testing
EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=cpu
EMBEDDINGS_BATCH_SIZE=8

# Minimal caching for testing
CACHE_ENABLED=false
RAG_RERANKING_ENABLED=false
SELF_LEARNING_ENABLED=false
AB_TESTING_ENABLED=false

# Minimal observability
OTEL_ENABLED=false
PERFORMANCE_TRACKING_ENABLED=false
```

## 🔧 Runtime Configuration

### API Configuration Updates

```bash
# Update configuration via API
curl -X POST http://localhost:8000/v1/admin/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "embeddings": {
      "primary": {
        "model": "intfloat/e5-large-v2",
        "device": "cuda"
      }
    }
  }'

# Reload configuration
curl -X POST http://localhost:8000/v1/admin/config/reload \
  -H "Authorization: Bearer your-api-key"
```

### Provider Swapping

```bash
# Switch to different embedding provider
curl -X POST http://localhost:8000/v1/admin/providers/embedding/switch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L12-v2"
  }'

# Switch reranking provider
curl -X POST http://localhost:8000/v1/admin/providers/reranker/switch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "provider": "vllm",
    "endpoint": "http://localhost:8001/v1"
  }'
```

## 🔍 Configuration Validation

### Validation Script

```bash
# Validate configuration
python scripts/validate_config.py

# Check specific configuration
python scripts/validate_config.py --config embeddings

# Validate environment variables
python scripts/validate_config.py --env
```

### Health Checks

```bash
# Configuration health check
curl http://localhost:8000/v1/admin/config/health

# Provider health check
curl http://localhost:8000/v1/admin/providers/health

# Database configuration check
curl http://localhost:8000/v1/admin/database/health
```

## 🚨 Troubleshooting Configuration

### Common Issues

#### Configuration File Not Found
```bash
# Check if config files exist
ls -la config/

# Create missing config files
cp config/config.yaml.example config/config.yaml
```

#### Environment Variable Issues
```bash
# Check environment variables
printenv | grep TYRA

# Validate .env file
cat .env | grep -v "^#" | grep -v "^$"
```

#### Database Connection Issues
```bash
# Test database URLs
python -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.getenv('DATABASE_URL'))
print('PostgreSQL connection:', engine.execute('SELECT 1').scalar())
"
```

#### Model Loading Issues
```bash
# Check model cache
ls -la data/models/

# Clear model cache
rm -rf data/models/*

# Test model loading
python -c "
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
print('Model loaded successfully')
"
```

### Debug Configuration

```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Check configuration loading
python -c "
from src.core.utils.config import get_config
config = get_config()
print('Configuration loaded successfully')
print('Active providers:', config.get('providers', {}))
"
```

## 📊 Performance Tuning

### Memory Optimization

```env
# Reduce memory usage
EMBEDDINGS_DEVICE=cpu
EMBEDDINGS_BATCH_SIZE=16
CACHE_L1_MAX_SIZE=500
POSTGRES_POOL_SIZE=10
REDIS_MAX_CONNECTIONS=25
```

### Speed Optimization

```env
# Optimize for speed
EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
CACHE_ENABLED=true
CACHE_TTL_EMBEDDINGS=86400
RAG_RERANKING_ENABLED=false
SELF_LEARNING_ENABLED=false
```

### Accuracy Optimization

```env
# Optimize for accuracy
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=cuda
RAG_RERANKING_ENABLED=true
RAG_HALLUCINATION_ENABLED=true
RAG_HALLUCINATION_THRESHOLD=85
```

## 📚 Additional Resources

### Related Documentation
- [Installation Guide](INSTALLATION.md)
- [Container Configuration](docs/CONTAINERS.md)
- [Provider Registry](docs/PROVIDER_REGISTRY.md)
- [API Documentation](API.md)

### Configuration Examples
- [Development Config](config/examples/development.yaml)
- [Production Config](config/examples/production.yaml)
- [Testing Config](config/examples/testing.yaml)

### Best Practices
1. Always validate configuration after changes
2. Use environment variables for sensitive data
3. Test configuration changes in development first
4. Monitor configuration impact on performance
5. Keep configuration files in version control
6. Document custom configuration changes

---

🎉 **Configuration Complete!** Your Tyra MCP Memory Server is now properly configured.

For troubleshooting, see the [Installation Guide](INSTALLATION.md) or check the logs in `logs/memory-server.log`.