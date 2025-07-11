# Tyra Advanced Memory MCP Server

A sophisticated Model Context Protocol (MCP) server providing advanced memory capabilities with RAG (Retrieval-Augmented Generation), hallucination detection, and adaptive learning for AI agents.

## 🌟 Features

### 🧠 Advanced Memory System
- **Multi-Modal Storage**: Vector embeddings + temporal knowledge graphs with Memgraph + Graphiti integration
- **Agent Isolation**: Separate memory spaces for Tyra, Claude, Archon with session management
- **Intelligent Chunking**: 6 dynamic strategies (auto, paragraph, semantic, slide, line, token) with size optimization
- **Entity Extraction**: Automated NER with relationship mapping using temporal knowledge graphs
- **Memory Versioning**: Track memory evolution with temporal validity intervals
- **Memory Health**: Automated stale detection, redundancy removal, and consolidation
- **Multi-Level Caching**: L1 (in-memory), L2 (Redis), L3 (materialized views) for <100ms p95 latency

### 📄 Universal Document Ingestion
- **9 File Formats**: PDF (PyMuPDF), DOCX (python-docx), PPTX (python-pptx), TXT/MD (encoding detection), HTML (html2text), JSON (nested objects), CSV (streaming), EPUB (chapters), and custom format support
- **Smart Processing**: Auto-format detection with specialized loaders and fallback mechanisms
- **Dynamic Chunking**: File-type-aware strategies (semantic for PDFs, paragraph for DOCX, slide for PPTX)
- **LLM Enhancement**: Context injection with rule-based templates + optional vLLM integration
- **Batch Processing**: Concurrent ingestion up to 100 documents with configurable concurrency (default: 20)
- **Streaming Pipeline**: Memory-efficient processing for large files (>10MB) with progress tracking
- **Comprehensive Metadata**: Document properties, chunk metadata, confidence scoring, hallucination detection
- **Error Recovery**: Graceful fallback with retry logic and detailed error reporting

### 🔍 Sophisticated Search & RAG
- **Hybrid Search**: Weighted combination (0.7 vector + 0.3 keyword) with graph traversal enhancement
- **Multi-Strategy Retrieval**: Vector similarity, keyword matching, temporal queries, graph traversal
- **Advanced Reranking**: Cross-encoder models + optional vLLM-based reranking with caching
- **Confidence Scoring**: Multi-level assessment (💪 Rock Solid 95%+, 🧠 High 80%+, 🤔 Fuzzy 60%+, ⚠️ Low <60%)
- **Hallucination Detection**: Real-time grounding analysis with evidence collection and consistency checking
- **Trading Safety**: Unbypassable 95% confidence requirement for financial operations with audit logging
- **Context Enrichment**: Graph-based context expansion with temporal relevance weighting

### 🕸️ Temporal Knowledge Graph
- **Memgraph Integration**: High-performance graph database with Cypher query support
- **Graphiti Framework**: Advanced temporal knowledge management with validity intervals
- **Entity Management**: Automated extraction, typing, merging, and updates with conflict resolution
- **Relationship Tracking**: Temporal relationship extraction with time-based validity and evolution tracking
- **Graph Traversal**: Efficient path finding, subgraph extraction, and entity timeline queries
- **Temporal Queries**: Time-range filtering, relationship evolution, and temporal pattern matching

### 🔀 Modular Provider System
- **Hot-Swappable Components**: Runtime provider switching without restart
- **Embedding Providers**: HuggingFace (intfloat/e5-large-v2 primary, all-MiniLM-L12-v2 fallback), OpenAI fallback
- **Vector Stores**: PostgreSQL + pgvector with HNSW indexing, future support for Weaviate, Qdrant
- **Graph Engines**: Memgraph + Graphiti, extensible to Neo4j, ArangoDB
- **Rerankers**: Cross-encoder models, vLLM integration, custom reranking strategies
- **Cache Providers**: Redis (multi-layer), in-memory LRU, future distributed caching
- **File Loaders**: Extensible loader registry with custom format support
- **Fallback Mechanisms**: Automatic failover with circuit breakers and health monitoring

### 📊 Performance Analytics & Observability
- **Real-Time Monitoring**: Response time, accuracy, memory usage, cache hit rates with configurable dashboards
- **OpenTelemetry Integration**: Complete distributed tracing, metrics collection, and structured logging
- **Trend Analysis**: Automated performance trend detection with statistical significance testing
- **Smart Alerts**: Configurable warning (response time >100ms) and critical thresholds with notification channels
- **Performance Metrics**: Request latency histograms, error rate tracking, resource utilization monitoring
- **Health Checks**: Comprehensive component health monitoring with automatic recovery
- **Audit Logging**: Complete operation audit trail with request correlation and compliance reporting

### 🎯 Adaptive Learning & Self-Optimization
- **Self-Optimization**: Automated parameter tuning based on performance data and user feedback
- **A/B Testing Framework**: Systematic experimentation with statistical significance testing and rollback protection
- **Learning Insights**: Pattern recognition from successful configurations and failure analysis
- **Multi-Strategy Optimization**: Gradient descent, Bayesian optimization, random search with ensemble methods
- **Memory Health Management**: Stale memory detection, redundancy identification, and automated cleanup
- **Prompt Evolution**: Continuous improvement of prompts based on success/failure patterns
- **Configuration Adaptation**: Dynamic configuration updates with safety constraints and rollback capabilities
- **Performance Baselines**: Automatic establishment and tracking of performance benchmarks

### 🌐 Dual Interface Architecture
- **MCP Protocol**: Full Model Context Protocol support for Claude, Tyra, and other MCP-compatible agents
- **REST API**: Comprehensive HTTP API with OpenAPI documentation for web integrations and custom clients
- **WebSocket Support**: Real-time updates and streaming for long-running operations
- **n8n Integration**: Pre-built webhook endpoints and workflow templates for automation
- **SDK Support**: Python and JavaScript client libraries with async support and retry logic

### 🔒 Enterprise Security & Safety
- **Local Operation**: 100% local deployment with no external API dependencies for data privacy
- **Multi-Agent Isolation**: Secure separation of agent memory spaces with access controls
- **Trading Safety**: Unbypassable confidence requirements for financial operations with multiple validation layers
- **Input Validation**: Comprehensive request validation with SQL injection and XSS protection
- **Rate Limiting**: Configurable limits (1000/min default) with burst protection (50/sec)
- **Circuit Breakers**: Automatic failure protection with configurable thresholds and recovery timeouts
- **Audit Trail**: Complete operation logging with compliance reporting and forensic capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- Redis (for caching)
- Memgraph (for knowledge graphs)
- **HuggingFace CLI** (for model downloads)
- **Git LFS** (for large model files)

### Automated Setup

```bash
# Run unified setup script
./setup.sh --env development

# Start the server
source venv/bin/activate
python main.py
```

### Manual Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd tyra-mcp-memory-server
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install Model Prerequisites**
   ```bash
   # Install HuggingFace CLI and Git LFS
   pip install huggingface-hub
   git lfs install
   ```

3. **Download Required Models** ⚠️ **REQUIRED - No Automatic Downloads**
   ```bash
   # Create model directories
   mkdir -p ./models/embeddings ./models/cross-encoders

   # Download primary embedding model (~1.34GB)
   huggingface-cli download intfloat/e5-large-v2 \
     --local-dir ./models/embeddings/e5-large-v2 \
     --local-dir-use-symlinks False

   # Download fallback embedding model (~120MB)
   huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 \
     --local-dir ./models/embeddings/all-MiniLM-L12-v2 \
     --local-dir-use-symlinks False

   # Download cross-encoder for reranking (~120MB)
   huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
     --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2 \
     --local-dir-use-symlinks False
   ```

4. **Verify Model Installation**
   ```bash
   # Test all models are working
   python scripts/test_model_pipeline.py
   ```

5. **Database Setup**
   ```bash
   # Start databases with Docker
   docker-compose -f docker-compose.dev.yml up -d

   # Or configure your own PostgreSQL, Redis, Memgraph instances
   ```

6. **Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

7. **Start Server**
   ```bash
   python main.py
   ```

## 🔧 MCP Integration

### Claude Desktop Configuration

Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "tyra-memory": {
      "command": "python",
      "args": ["/path/to/tyra-mcp-memory-server/main.py"],
      "env": {
        "TYRA_ENV": "production"
      }
    }
  }
}
```

### Available Tools

#### 🧠 Core Memory Tools

##### 📝 `store_memory`
Store information with automatic entity extraction and metadata enrichment.

```json
{
  "tool": "store_memory",
  "content": "User prefers morning trading sessions and uses technical analysis",
  "agent_id": "tyra",
  "session_id": "trading_session_001",
  "extract_entities": true,
  "chunk_content": false,
  "metadata": {"category": "trading_preferences", "confidence": 95}
}
```

##### 🔍 `search_memory`
Advanced hybrid search with confidence scoring and hallucination analysis.

```json
{
  "tool": "search_memory",
  "query": "What are the user's trading preferences?",
  "agent_id": "tyra",
  "search_type": "hybrid",
  "top_k": 10,
  "min_confidence": 0.7,
  "include_analysis": true,
  "rerank": true,
  "temporal_weight": 0.3
}
```

##### 📋 `get_all_memories`
Retrieve all memories for an agent with filtering and pagination.

```json
{
  "tool": "get_all_memories",
  "agent_id": "tyra",
  "limit": 50,
  "offset": 0,
  "include_metadata": true,
  "filter_by_date": "2024-01-01",
  "category": "trading"
}
```

##### 🗑️ `delete_memory`
Remove specific memories with optional cascade deletion.

```json
{
  "tool": "delete_memory",
  "memory_id": "mem_12345",
  "agent_id": "tyra",
  "cascade_delete": false
}
```

#### 📄 Document Processing Tools

##### 📄 `ingest_document`
Ingest documents with automatic format detection and intelligent processing.

```json
{
  "tool": "ingest_document",
  "file_path": "/path/to/document.pdf",
  "file_type": "pdf",
  "chunking_strategy": "auto",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "enable_llm_context": true,
  "extract_entities": true,
  "metadata": {"source": "user_upload", "priority": "high"}
}
```

##### 📊 `batch_ingest`
Process multiple documents concurrently with progress tracking.

```json
{
  "tool": "batch_ingest",
  "documents": [
    {"file_path": "/docs/report1.pdf", "chunking_strategy": "semantic"},
    {"file_path": "/docs/data.csv", "chunking_strategy": "row"}
  ],
  "max_concurrent": 10,
  "progress_callback": true
}
```

##### 📈 `get_ingestion_status`
Monitor document processing status and progress.

```json
{
  "tool": "get_ingestion_status",
  "job_id": "batch_12345",
  "include_details": true
}
```

#### 🛡️ Analysis & Validation Tools

##### 🛡️ `analyze_response`
Analyze any response for hallucinations and confidence scoring.

```json
{
  "tool": "analyze_response",
  "response": "Based on your history, you prefer swing trading",
  "query": "What's my trading style?",
  "retrieved_memories": [...],
  "detailed_analysis": true,
  "include_evidence": true
}
```

##### 🎯 `validate_for_trading`
Special validation for financial operations with 95% confidence requirement.

```json
{
  "tool": "validate_for_trading",
  "query": "Should I buy AAPL stock?",
  "response": "Based on your risk profile, AAPL looks good",
  "context_memories": [...],
  "require_rock_solid": true
}
```

##### 🔄 `rerank_results`
Improve search result relevance with advanced reranking.

```json
{
  "tool": "rerank_results",
  "query": "trading strategies",
  "results": [...],
  "reranker_type": "cross_encoder",
  "top_k": 5
}
```

#### 🕸️ Knowledge Graph Tools

##### 🕸️ `query_graph`
Execute graph traversal queries on the knowledge graph.

```json
{
  "tool": "query_graph",
  "cypher_query": "MATCH (p:PERSON)-[r:TRADES]->(s:STOCK) RETURN p, r, s",
  "agent_id": "tyra",
  "include_temporal": true
}
```

##### 🔗 `get_entity_relationships`
Explore entity connections and relationship paths.

```json
{
  "tool": "get_entity_relationships",
  "entity_name": "AAPL",
  "relationship_types": ["CORRELATES_WITH", "COMPETES_WITH"],
  "max_depth": 3,
  "temporal_filter": "last_30_days"
}
```

##### 📊 `get_entity_timeline`
View entity evolution over time.

```json
{
  "tool": "get_entity_timeline",
  "entity_id": "entity_12345",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "include_relationships": true
}
```

#### 📊 System Monitoring Tools

##### 📊 `get_memory_stats`
Comprehensive system statistics and health metrics.

```json
{
  "tool": "get_memory_stats",
  "agent_id": "tyra",
  "include_performance": true,
  "include_recommendations": true,
  "include_cache_stats": true,
  "time_range": "last_24_hours"
}
```

##### 🏥 `health_check`
Complete system health assessment.

```json
{
  "tool": "health_check",
  "detailed": true,
  "include_components": ["vector_store", "graph_engine", "cache", "embeddings"],
  "run_diagnostics": true
}
```

##### ⚡ `get_performance_metrics`
Real-time performance and resource utilization.

```json
{
  "tool": "get_performance_metrics",
  "metric_types": ["latency", "throughput", "error_rate"],
  "time_window": "5m",
  "include_predictions": true
}
```

#### 🎯 Learning & Optimization Tools

##### 🎯 `get_learning_insights`
Access adaptive learning insights and optimization recommendations.

```json
{
  "tool": "get_learning_insights",
  "category": "parameter_optimization",
  "days": 7,
  "include_experiments": true,
  "confidence_threshold": 0.8
}
```

##### 🔧 `optimize_configuration`
Trigger configuration optimization based on usage patterns.

```json
{
  "tool": "optimize_configuration",
  "components": ["embeddings", "cache", "reranking"],
  "optimization_strategy": "bayesian",
  "safety_constraints": true
}
```

##### 📈 `get_improvement_suggestions`
AI-generated recommendations for system enhancement.

```json
{
  "tool": "get_improvement_suggestions",
  "focus_areas": ["performance", "accuracy", "reliability"],
  "priority": "high",
  "include_implementation": true
}
```

#### 🔧 Administrative Tools

##### ⚙️ `update_configuration`
Dynamically update system configuration without restart.

```json
{
  "tool": "update_configuration",
  "config_section": "cache",
  "updates": {"ttl": 7200, "max_size": "2GB"},
  "validate_before_apply": true
}
```

##### 🧹 `cleanup_memories`
Clean up stale or redundant memories.

```json
{
  "tool": "cleanup_memories",
  "agent_id": "tyra",
  "older_than_days": 90,
  "confidence_threshold": 0.3,
  "dry_run": true
}
```

##### 💾 `backup_memories`
Create backup of agent memories.

```json
{
  "tool": "backup_memories",
  "agent_id": "tyra",
  "include_graph": true,
  "compression": true,
  "backup_location": "/backups/tyra_memories.gz"
}
```

## 🏗️ Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                            │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│     Claude      │      Tyra       │     Archon      │   n8n    │
│   (MCP Client)  │   (MCP Client)  │   (MCP Client)  │ (Webhook)│
└─────────────────┴─────────────────┴─────────────────┴──────────┘
         │                 │                 │            │
         └─────────────────┼─────────────────┘            │
                           │                              │
┌─────────────────────────────────────────────────────────────────┐
│                     INTERFACE LAYER                            │
├─────────────────────────────┬───────────────────────────────────┤
│        MCP Server           │         FastAPI Server           │
│   • Tool Handlers           │   • REST Endpoints               │
│   • Protocol Management     │   • WebSocket Support            │
│   • Agent Isolation         │   • OpenAPI Documentation        │
│   • Session Management      │   • Rate Limiting                │
└─────────────────────────────┴───────────────────────────────────┘
         │                                 │
         └─────────────────┬───────────────┘
                           │
┌─────────────────────────────────────────────────────────────────┐
│                      CORE ENGINE                               │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│   Memory    │ Document    │ Hallucination│   Graph    │ Learn  │
│  Manager    │ Processor   │  Detector    │  Engine    │ Engine │
│ • Storage   │ • 9 Formats │ • Confidence │ • Memgraph │ • A/B  │
│ • Retrieval │ • Chunking  │ • Grounding  │ • Graphiti │ • Opt  │
│ • Caching   │ • LLM Enh   │ • Evidence   │ • Temporal │ • Ins  │
└─────────────┴─────────────┴─────────────┴─────────────┴────────┘
         │           │             │             │         │
         └───────────┼─────────────┼─────────────┼─────────┘
                     │             │             │
┌─────────────────────────────────────────────────────────────────┐
│                    PROVIDER LAYER                              │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤
│Embedding │ Vector   │  Graph   │Reranker  │  Cache   │  File   │
│Providers │  Store   │ Database │ Provider │ Provider │ Loaders │
│• HF E5   │• PG Vec  │• Memgraph│• Cross-E │• Redis   │• PDF    │
│• MiniLM  │• HNSW    │• Cypher  │• vLLM    │• Memory  │• DOCX   │
│• OpenAI  │• Hybrid  │• Graphiti│• Custom  │• L1/L2/L3│• 7 More │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘
         │      │          │          │          │          │
         └──────┼──────────┼──────────┼──────────┼──────────┘
                │          │          │          │
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│   PostgreSQL    │    Memgraph     │      Redis      │   Logs   │
│  + pgvector     │  + Graphiti     │   Multi-Layer   │ + Traces │
│ • Vector Store  │ • Knowledge     │ • Performance   │ • Metrics│
│ • Metadata      │ • Temporal      │ • Session       │ • Events │
│ • HNSW Index    │ • Relationships │ • Embedding     │ • Audit  │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
```

### Core Processing Pipelines

#### 1. Memory Storage Pipeline
```
Input Content → Format Detection → Document Loading → Chunking Strategy Selection
     ↓                ↓               ↓                     ↓
Text Extraction → LLM Enhancement → Entity Extraction → Relationship Mapping
     ↓                ↓               ↓                     ↓
Embedding Generation → Vector Storage → Graph Storage → Cache Update
     ↓                ↓               ↓                     ↓
Metadata Recording → Performance Metrics → Success Response
```

#### 2. Memory Search Pipeline
```
Search Query → Query Enhancement → Multi-Strategy Retrieval
     ↓              ↓                      ↓
Vector Search + Keyword Search + Graph Traversal
     ↓              ↓                      ↓
Result Fusion → Advanced Reranking → Confidence Scoring
     ↓              ↓                      ↓
Hallucination Detection → Evidence Collection → Response Formatting
```

#### 3. Document Ingestion Pipeline
```
Document Input → Format Detection → Specialized Loader Selection
     ↓               ↓                      ↓
Content Extraction → Chunking Strategy → LLM Context Enhancement
     ↓               ↓                      ↓
Batch Processing → Memory Integration → Progress Tracking
```

#### 4. Self-Learning Pipeline
```
Performance Data → Pattern Recognition → Experiment Design
     ↓                   ↓                    ↓
A/B Testing → Statistical Analysis → Configuration Updates
     ↓                   ↓                    ↓
Rollback Protection → Success Validation → Learning Storage
```

### Technology Stack

#### Core Technologies
- **Python 3.8+**: Primary development language
- **FastMCP**: Model Context Protocol implementation
- **FastAPI**: REST API framework with automatic documentation
- **Pydantic**: Data validation and settings management
- **asyncio**: Asynchronous programming for high performance

#### Databases & Storage
- **PostgreSQL 14+**: Primary data store with JSON support
- **pgvector**: Vector similarity search with HNSW indexing
- **Memgraph**: High-performance graph database
- **Redis**: Multi-layer caching and session storage

#### Machine Learning & NLP
- **HuggingFace Transformers**: Embedding models (e5-large-v2, MiniLM)
- **Sentence Transformers**: Optimized embedding inference
- **spaCy**: Named entity recognition and text processing
- **NLTK**: Natural language processing utilities

#### Observability & Monitoring
- **OpenTelemetry**: Distributed tracing and metrics
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Performance dashboards and visualization
- **Jaeger**: Distributed tracing visualization

#### Development & Deployment
- **Docker**: Containerization with multi-stage builds
- **Docker Compose**: Local development environment
- **pytest**: Comprehensive testing framework
- **Black/isort/flake8**: Code formatting and linting

### Performance Characteristics

#### Latency Targets (P95)
- **Memory Storage**: <100ms per operation
- **Vector Search**: <50ms for top-10 results
- **Hybrid Search**: <150ms with reranking
- **Document Ingestion**: <2s per PDF page
- **Graph Queries**: <30ms for simple traversals
- **Hallucination Detection**: <200ms per analysis

#### Throughput Capabilities
- **Concurrent Users**: 50-100 depending on hardware
- **Document Processing**: 10-20 documents/minute
- **Memory Operations**: 1000+ operations/minute
- **Cache Hit Rate**: >85% for frequently accessed data

#### Scalability Metrics
- **Memory Capacity**: Unlimited (PostgreSQL-based)
- **Graph Complexity**: Millions of entities/relationships
- **Vector Dimensions**: 384-1024 (configurable)
- **Cache Size**: 2-8GB recommended for optimal performance

## ⚙️ Configuration

### Environment Variables

```bash
# Core Configuration
TYRA_ENV=development|production
TYRA_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tyra_memory
POSTGRES_USER=tyra
POSTGRES_PASSWORD=secure_password

REDIS_HOST=localhost
REDIS_PORT=6379

MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687

# Optional: OpenAI for fallback embeddings
OPENAI_API_KEY=sk-...
```

### Configuration Files

The system uses a layered configuration approach with multiple YAML files:

#### Main Configuration (`config/config.yaml`)
```yaml
# Core application settings
app:
  name: "Tyra MCP Memory Server"
  version: "1.0.0"
  environment: ${TYRA_ENV:-development}

# Memory system configuration
memory:
  backend: "postgres"
  vector_dimensions: 1024
  chunk_size: 512
  chunk_overlap: 50
  max_memories_per_agent: 1000000

# API server settings
api:
  host: ${API_HOST:-0.0.0.0}
  port: ${API_PORT:-8000}
  enable_docs: ${API_ENABLE_DOCS:-true}
  cors_origins: ["*"]
  rate_limit: 1000  # requests per minute
```

#### Provider Configuration (`config/providers.yaml`)
```yaml
# Embedding providers
embeddings:
  primary: "huggingface"
  fallback: "huggingface_light"
  providers:
    huggingface:
      model: "intfloat/e5-large-v2"
      device: "auto"
      batch_size: 32
```

#### RAG Configuration (`config/rag.yaml`)
```yaml
# Retrieval and reranking settings
retrieval:
  hybrid_weight: 0.7  # vector vs keyword
  max_results: 20
  diversity_penalty: 0.1

reranking:
  enabled: true
  provider: "cross_encoder"
  top_k: 10

hallucination:
  enabled: true
  threshold: 75
  require_evidence: true
```

#### Document Ingestion (`config/ingestion.yaml`)
```yaml
# File processing settings
ingestion:
  max_file_size: 104857600  # 100MB
  max_batch_size: 100
  concurrent_limit: 20
  supported_formats: ["pdf", "docx", "pptx", "txt", "md", "html", "json", "csv", "epub"]
  
  chunking:
    default_strategy: "auto"
    strategies:
      auto:
        file_type_mapping:
          pdf: "semantic"
          docx: "paragraph"
          pptx: "slide"
```

#### Self-Learning Configuration (`config/self_learning.yaml`)
```yaml
# Adaptive learning settings
self_learning:
  enabled: true
  analysis_interval: "1h"
  improvement_interval: "24h"
  auto_optimize: true
  
  quality_thresholds:
    memory_accuracy: 0.85
    performance_degradation: 0.1
    hallucination_rate: 0.05
```

#### Observability Configuration (`config/observability.yaml`)
```yaml
# OpenTelemetry and monitoring
otel:
  enabled: true
  service_name: "tyra-mcp-memory-server"
  
tracing:
  enabled: true
  exporter: "console"  # or "jaeger", "otlp"
  sampler: "parentbased_traceidratio"
  sampler_arg: 1.0

metrics:
  enabled: true
  export_interval: 60000  # 60 seconds
  
logging:
  level: ${LOG_LEVEL:-INFO}
  format: "json"
  rotation:
    max_size: "100MB"
    max_files: 10
```

### Environment Variables Reference

#### Core Settings
```bash
# Application Environment
TYRA_ENV=development|production
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
TYRA_DEBUG=true|false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_ENABLE_DOCS=true
API_RATE_LIMIT=1000

# Database URLs
DATABASE_URL=postgresql://user:pass@localhost:5432/tyra_memory
REDIS_URL=redis://localhost:6379/0
MEMGRAPH_URL=bolt://localhost:7687
```

#### Model Configuration
```bash
# Embedding Models
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=auto|cpu|cuda

# Model Caching
MODEL_CACHE_DIR=/models/cache
EMBEDDING_CACHE_TTL=86400  # 24 hours
```

#### Performance Tuning
```bash
# Memory Management
MEMORY_MAX_CHUNK_SIZE=2048
MEMORY_CHUNK_OVERLAP=100
MEMORY_BATCH_SIZE=50

# Caching Configuration
CACHE_TTL_EMBEDDINGS=86400
CACHE_TTL_SEARCH=3600
CACHE_TTL_RERANK=1800
CACHE_MAX_SIZE=2GB

# Database Pools
POSTGRES_POOL_SIZE=20
REDIS_POOL_SIZE=50
MEMGRAPH_POOL_SIZE=10
```

#### Security Settings
```bash
# Authentication (Optional)
API_KEY_ENABLED=false
API_KEY=your-secure-api-key
JWT_SECRET=your-jwt-secret

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
RATE_LIMIT_BURST=50

# CORS Configuration
CORS_ORIGINS=*
CORS_METHODS=GET,POST,PUT,DELETE
CORS_HEADERS=*
```

#### Document Ingestion
```bash
# File Processing
INGESTION_MAX_FILE_SIZE=104857600
INGESTION_MAX_BATCH_SIZE=100
INGESTION_CONCURRENT_LIMIT=20
INGESTION_TIMEOUT=300

# LLM Enhancement
INGESTION_LLM_ENHANCEMENT=true
INGESTION_LLM_MODE=rule_based
VLLM_ENDPOINT=http://localhost:8000/v1
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

#### Observability
```bash
# OpenTelemetry
OTEL_ENABLED=true
OTEL_SERVICE_NAME=tyra-mcp-memory-server
OTEL_TRACES_ENABLED=true
OTEL_METRICS_ENABLED=true
OTEL_LOGS_ENABLED=true

# Exporters
OTEL_TRACES_EXPORTER=console
OTEL_METRICS_EXPORTER=console
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

## 🔍 Monitoring & Debugging

### Health Checks
```bash
# Check system health
curl -X POST http://localhost:8000/tools/health_check \
  -d '{"detailed": true}'
```

### Performance Analytics
```bash
# Get performance summary
curl -X POST http://localhost:8000/tools/get_memory_stats \
  -d '{"include_performance": true}'
```

### Logs
```bash
# View real-time logs
tail -f logs/tyra-memory.log

# Search for specific events
grep "hallucination" logs/tyra-memory.log
grep "ERROR" logs/tyra-memory.log
```

## 🧪 Development

### Testing
```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run end-to-end tests
python -m pytest tests/e2e/
```

### Development Mode
```bash
# Enable hot reload and debug features
export TYRA_ENV=development
export TYRA_DEBUG=true
export TYRA_HOT_RELOAD=true

python main.py
```

### Model Development
```bash
# Download and test models
python scripts/download_models.py

# Benchmark different models
python scripts/benchmark_models.py
```

## 🚀 Production Deployment

### Docker Deployment

#### Quick Production Start
```bash
# Clone repository
git clone <repository-url>
cd tyra-mcp-memory-server

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Build and start all services
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
curl http://localhost:8000/health
```

#### Multi-Stage Docker Build
```bash
# Build optimized production image
docker build -t tyra-memory-server:latest \
  --target production \
  --build-arg ENVIRONMENT=production .

# Run with custom configuration
docker run -d \
  --name tyra-memory \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  tyra-memory-server:latest
```

#### Container Orchestration
```yaml
# kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tyra-memory-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tyra-memory-server
  template:
    metadata:
      labels:
        app: tyra-memory-server
    spec:
      containers:
      - name: tyra-memory
        image: tyra-memory-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: TYRA_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Systemd Service Deployment

#### Service Installation
```bash
# Install system service
sudo cp scripts/tyra-memory.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable tyra-memory

# Start service
sudo systemctl start tyra-memory

# Check status and logs
sudo systemctl status tyra-memory
sudo journalctl -u tyra-memory -f
```

#### Service Configuration (`scripts/tyra-memory.service`)
```ini
[Unit]
Description=Tyra MCP Memory Server
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=tyra
Group=tyra
WorkingDirectory=/opt/tyra-memory-server
ExecStart=/opt/tyra-memory-server/venv/bin/python main.py
Restart=always
RestartSec=10
Environment=TYRA_ENV=production
Environment=LOG_LEVEL=INFO

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/tyra-memory-server/data /opt/tyra-memory-server/logs

[Install]
WantedBy=multi-user.target
```

### Performance Tuning

#### Hardware Requirements
```bash
# Minimum Requirements
CPU: 4 cores (8 threads recommended)
RAM: 8GB (16GB recommended)
Storage: 100GB SSD (fast I/O critical)
Network: 1Gbps (for high-throughput scenarios)

# Optimal Configuration
CPU: 8+ cores with AVX2 support
RAM: 32GB+ (for large models and caching)
GPU: NVIDIA GPU with 8GB+ VRAM (optional, for GPU acceleration)
Storage: NVMe SSD with 1000+ IOPS
```

#### Database Optimization
```sql
-- PostgreSQL performance tuning
-- Add to postgresql.conf

# Memory settings
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 256MB
maintenance_work_mem = 1GB

# Vector-specific settings
max_connections = 200
shared_preload_libraries = 'vector'

# Performance settings
random_page_cost = 1.1
checkpoint_completion_target = 0.9
wal_buffers = 64MB
```

#### Redis Configuration
```bash
# Redis optimization (redis.conf)
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Persistence settings
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

#### Application Tuning
```yaml
# config/config.yaml - Production optimizations
performance:
  # Database connection pools
  postgres_pool_size: 20
  redis_pool_size: 50
  memgraph_pool_size: 10
  
  # Memory management
  max_chunk_size: 2048
  batch_size: 50
  
  # Caching optimization
  cache_sizes:
    embeddings: "2GB"
    search_results: "1GB"
    rerank_cache: "512MB"
  
  # Async processing
  max_concurrent_requests: 100
  request_timeout: 30
  
  # GPU optimization (if available)
  gpu_enabled: true
  gpu_memory_fraction: 0.8
  cuda_device: 0
```

### Monitoring & Observability

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'tyra-memory-server'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Tyra Memory Server",
    "panels": [
      {
        "title": "Request Latency",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Memory Usage",
        "type": "graph", 
        "targets": [{
          "expr": "process_resident_memory_bytes"
        }]
      }
    ]
  }
}
```

### Load Balancing & High Availability

#### NGINX Configuration
```nginx
upstream tyra_memory_servers {
    least_conn;
    server 127.0.0.1:8001 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name memory.tyra-ai.com;
    
    location / {
        proxy_pass http://tyra_memory_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        proxy_pass http://tyra_memory_servers/health;
        access_log off;
    }
}
```

## 💼 Use Cases & Applications

### 🤖 AI Agent Memory Enhancement
- **Personal Assistants**: Long-term conversation memory and context retention
- **Customer Support**: Historical interaction tracking and personalized responses
- **Research Assistants**: Literature review, citation tracking, and knowledge synthesis
- **Educational Tutors**: Student progress tracking and adaptive learning paths

### 📊 Enterprise Knowledge Management
- **Document Processing**: Automated ingestion of company documents, policies, and procedures
- **Institutional Memory**: Preserve and access organizational knowledge across teams
- **Compliance Tracking**: Audit trails and regulatory requirement monitoring
- **Decision Support**: Historical data analysis and trend identification

### 🏦 Financial Services Applications
- **Trading Support**: Market analysis with confidence-scored recommendations (95% threshold)
- **Risk Assessment**: Historical pattern analysis and risk factor identification
- **Client Profiling**: Investment preferences and trading behavior analysis
- **Regulatory Compliance**: Transaction monitoring and compliance reporting

### 🔬 Research & Analytics
- **Scientific Research**: Literature mining and hypothesis generation
- **Market Research**: Consumer behavior analysis and trend prediction
- **Competitive Intelligence**: Industry analysis and competitor monitoring
- **Data Mining**: Pattern discovery in large document collections

### 🌐 Multi-Agent Orchestration
- **Agent Coordination**: Shared knowledge base for multiple AI agents
- **Workflow Automation**: n8n integration for document processing pipelines
- **Cross-Platform Integration**: Unified memory across different AI systems
- **Session Management**: Context preservation across agent interactions

### 🔒 Secure Deployment Scenarios
- **Air-Gapped Networks**: Completely offline operation with local models
- **HIPAA Compliance**: Healthcare data processing with audit trails
- **Financial Regulations**: Trading compliance with mandatory confidence thresholds
- **Government Applications**: Classified information processing with security controls

## 🔧 Integration Examples

### Claude Desktop Integration
```json
{
  "mcpServers": {
    "tyra-memory": {
      "command": "python",
      "args": ["/path/to/tyra-mcp-memory-server/main.py"],
      "env": {
        "TYRA_ENV": "production",
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/tyra",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### n8n Workflow Integration
```json
{
  "nodes": [
    {
      "name": "Document Upload",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://localhost:8000/v1/ingest/document",
        "method": "POST",
        "body": {
          "source_type": "base64",
          "file_name": "{{ $json.filename }}",
          "file_type": "pdf",
          "content": "{{ $json.base64_content }}"
        }
      }
    },
    {
      "name": "Memory Search",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://localhost:8000/v1/memory/search",
        "method": "POST",
        "body": {
          "query": "{{ $json.search_query }}",
          "agent_id": "n8n_workflow",
          "include_analysis": true,
          "min_confidence": 0.8
        }
      }
    }
  ]
}
```

### Python Client Usage
```python
from tyra_memory_client import MemoryClient
import asyncio

async def main():
    # Initialize client
    client = MemoryClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"  # if authentication enabled
    )
    
    # Store a memory
    result = await client.store_memory(
        content="User prefers technical analysis for stock trading",
        agent_id="trading_bot",
        extract_entities=True,
        metadata={"category": "trading", "confidence": 95}
    )
    print(f"Stored memory: {result.memory_id}")
    
    # Search memories
    results = await client.search_memories(
        query="trading preferences",
        agent_id="trading_bot",
        min_confidence=0.8,
        include_analysis=True
    )
    
    for memory in results.memories:
        print(f"Memory: {memory.content}")
        print(f"Confidence: {memory.confidence_score}")
        print(f"Grounding: {memory.grounding_score}")
    
    # Ingest document
    with open("trading_strategy.pdf", "rb") as f:
        doc_result = await client.ingest_document(
            file_content=f.read(),
            file_name="trading_strategy.pdf",
            file_type="pdf",
            chunking_strategy="semantic",
            enable_llm_context=True
        )
    print(f"Ingested {doc_result.chunks_ingested} chunks")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📈 Performance Benchmarks

### Typical Performance (Local Setup)
- **Memory Storage**: ~100ms per document
- **Vector Search**: ~50ms for top-10 results
- **Hybrid Search**: ~150ms with reranking
- **Hallucination Analysis**: ~200ms per response
- **Memory Usage**: ~500MB base + ~2GB for models

### Scalability
- **Concurrent Requests**: 10-50 depending on hardware
- **Memory Capacity**: Unlimited (PostgreSQL-based)
- **Graph Complexity**: Optimized for millions of entities/relationships

## 🤝 Contributing

### Development Setup
```bash
# Setup development environment
./scripts/setup.sh --env development

# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### Code Standards
- **Type Hints**: All functions must have type annotations
- **Documentation**: Docstrings for all public methods
- **Testing**: Minimum 80% code coverage
- **Formatting**: Black + isort + flake8

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [Architecture Guide](ARCHITECTURE.md)
- [Configuration Reference](CONFIGURATION.md)
- [API Documentation](API.md)
- [Installation Guide](INSTALLATION.md)

### Community
- **Issues**: [GitHub Issues](https://github.com/tyra-ai/memory-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tyra-ai/memory-server/discussions)
- **Discord**: [Tyra AI Community](https://discord.gg/tyra-ai)

### Commercial Support
For enterprise support, custom integrations, and professional services, contact: support@tyra-ai.com

---

**Built with ❤️ by the Tyra AI Team**

*Empowering AI agents with human-like memory capabilities*
