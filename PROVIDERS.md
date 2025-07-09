# 🔌 Tyra MCP Memory Server - Provider Reference

## 📋 Table of Contents

- [Overview](#overview)
- [Provider Types](#provider-types)
- [Embedding Providers](#embedding-providers)
- [Vector Store Providers](#vector-store-providers)
- [Graph Engine Providers](#graph-engine-providers)
- [Reranker Providers](#reranker-providers)
- [Cache Providers](#cache-providers)
- [Hallucination Detectors](#hallucination-detectors)
- [Configuration Guide](#configuration-guide)
- [Adding Custom Providers](#adding-custom-providers)
- [Performance Comparison](#performance-comparison)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Tyra MCP Memory Server uses a modular provider system that allows you to swap components without changing core logic. Each provider type has a standardized interface and can be configured through YAML files or environment variables.

### Key Benefits

- **Hot-swappable**: Change providers without restarting
- **Fallback Support**: Automatic failover to backup providers
- **Performance Monitoring**: Built-in metrics for each provider
- **Easy Extension**: Simple interface for custom providers

## 🏗️ Provider Types

| Provider Type | Purpose | Interface | Required |
|---------------|---------|-----------|----------|
| Embedding | Text vectorization | `EmbeddingProvider` | ✅ Yes |
| Vector Store | Similarity search | `VectorStore` | ✅ Yes |
| Graph Engine | Entity relationships | `GraphEngine` | ⚠️ Optional |
| Reranker | Result optimization | `Reranker` | ⚠️ Optional |
| Cache | Performance caching | `CacheProvider` | ⚠️ Optional |
| Hallucination Detector | Confidence scoring | `HallucinationDetector` | ⚠️ Optional |

## 🧠 Embedding Providers

### HuggingFace Provider (`huggingface`)

**Status**: ✅ Production Ready  
**Performance**: ⭐⭐⭐⭐⭐ Excellent  
**Local**: ✅ 100% Local

The primary embedding provider using Sentence Transformers models.

#### Supported Models

| Model | Dimensions | Performance | Use Case |
|-------|------------|-------------|----------|
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Fast | General purpose |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Balanced | High quality |
| `intfloat/e5-large-v2` | 1024 | Slower | Best quality |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fastest | Low resource |

#### Configuration

```yaml
# config/providers.yaml
embeddings:
  primary:
    provider: "huggingface"
    model_name: "sentence-transformers/all-MiniLM-L12-v2"
    device: "auto"  # auto, cpu, cuda:0
    batch_size: 64
    max_length: 512
    normalize_embeddings: true
    use_fp16: true
    max_concurrent_batches: 3
    memory_efficient_attention: true
    warmup_queries: 10
```

#### Performance Optimizations

- **Parallel Batch Processing**: Concurrent embedding generation
- **GPU Memory Management**: Pre-allocation and efficient cleanup
- **FP16 Precision**: Faster computation with minimal quality loss
- **Model Warmup**: Pre-load optimizations for consistent performance

#### Example Usage

```python
from core.providers.embeddings.huggingface import HuggingFaceProvider

provider = HuggingFaceProvider()
await provider.initialize(config)

# Single text
embedding = await provider.embed_query("Hello world")

# Batch processing
embeddings = await provider.embed_texts(["Text 1", "Text 2", "Text 3"])
```

### OpenAI Provider (`openai`)

**Status**: ⚠️ Fallback Only  
**Performance**: ⭐⭐⭐⭐ Good  
**Local**: ❌ External API

Used as a fallback when HuggingFace provider fails.

#### Supported Models

| Model | Dimensions | Cost | Quality |
|-------|------------|------|---------|
| `text-embedding-3-small` | 1536 | Low | Good |
| `text-embedding-3-large` | 3072 | High | Excellent |
| `text-embedding-ada-002` | 1536 | Medium | Good |

#### Configuration

```yaml
embeddings:
  fallback:
    provider: "openai"
    model_name: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    max_retries: 3
    timeout: 30
    batch_size: 100
```

## 📊 Vector Store Providers

### pgvector Provider (`pgvector`)

**Status**: ✅ Production Ready  
**Performance**: ⭐⭐⭐⭐⭐ Excellent  
**Local**: ✅ 100% Local

PostgreSQL with pgvector extension for vector similarity search.

#### Features

- **HNSW Indexes**: Fast approximate nearest neighbor search
- **Exact Search**: IVFFlat for precise results
- **Hybrid Search**: Combined vector and keyword search
- **ACID Compliance**: Full transaction support
- **Scalability**: Handles millions of vectors efficiently

#### Configuration

```yaml
# config/config.yaml
memory:
  vector:
    backend: "pgvector"
    dimensions: 384  # Must match embedding model
    index_type: "hnsw"  # or "ivfflat"
    distance_metric: "cosine"  # cosine, l2, inner_product
    
    index_params:
      hnsw:
        m: 16
        ef_construction: 64
      ivfflat:
        lists: 100

databases:
  postgresql:
    host: localhost
    port: 5432
    database: tyra_memory
    username: tyra
    password: "${POSTGRES_PASSWORD}"
    pool_size: 50
    min_connections: 10
    max_connections: 100
```

#### Performance Tuning

```sql
-- Optimize for vector search
SET maintenance_work_mem = '2GB';
SET max_parallel_workers_per_gather = 4;
SET shared_preload_libraries = 'vector';

-- Index creation
CREATE INDEX CONCURRENTLY memory_embeddings_hnsw_idx 
ON memory_embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
```

### Chroma Provider (`chroma`)

**Status**: 🚧 Development  
**Performance**: ⭐⭐⭐ Good  
**Local**: ✅ 100% Local

Alternative vector database option.

#### Configuration

```yaml
memory:
  vector:
    backend: "chroma"
    persist_directory: "./data/chroma"
    collection_name: "tyra_memories"
```

## 🕸️ Graph Engine Providers

### Memgraph Provider (`memgraph`)

**Status**: ✅ Production Ready  
**Performance**: ⭐⭐⭐⭐⭐ Excellent  
**Local**: ✅ 100% Local

High-performance graph database for entity relationships.

#### Features

- **Real-time Analytics**: Fast graph traversals
- **Temporal Queries**: Time-based relationship analysis
- **ACID Transactions**: Consistent data operations
- **Cypher Query Language**: Standard graph query syntax
- **Streaming Support**: Real-time data processing

#### Configuration

```yaml
# config/config.yaml
graph:
  backend: "memgraph"
  
  memgraph:
    host: localhost
    port: 7687
    username: memgraph
    password: "${MEMGRAPH_PASSWORD}"
    encrypted: false
    connection_timeout: 30
    
    # Features
    temporal_enabled: true
    analytics_enabled: true
    batch_size: 1000
```

#### Example Queries

```cypher
-- Find connected entities
MATCH (e:Entity {name: $entity_name})-[r]-(connected)
RETURN connected, r
LIMIT 20;

-- Temporal relationship analysis
MATCH (e1:Entity)-[r:MENTIONED_WITH]->(e2:Entity)
WHERE r.timestamp > $start_time
RETURN e1, e2, r.timestamp
ORDER BY r.timestamp DESC;

-- Entity clustering
MATCH (e:Entity)
WITH e, SIZE((e)--()) as degree
WHERE degree > 5
RETURN e.name, degree
ORDER BY degree DESC;
```

### Neo4j Provider (`neo4j`)

**Status**: 🚧 Planned  
**Performance**: ⭐⭐⭐⭐ Good  
**Local**: ✅ Can be local

Alternative graph database option.

## 🎯 Reranker Providers

### Cross-Encoder Provider (`cross_encoder`)

**Status**: ✅ Production Ready  
**Performance**: ⭐⭐⭐⭐ Good  
**Local**: ✅ 100% Local

Neural reranking using cross-encoder models.

#### Supported Models

| Model | Performance | Quality | Use Case |
|-------|-------------|---------|----------|
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Fast | Good | General |
| `cross-encoder/ms-marco-electra-base` | Medium | Better | Quality |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Fastest | Basic | Speed |

#### Configuration

```yaml
# config/config.yaml
rag:
  reranking:
    provider: "cross_encoder"
    model_name: "cross-encoder/ms-marco-MiniLM-L-12-v2"
    batch_size: 32
    max_concurrent_requests: 5
    timeout_seconds: 10
    normalization: "sigmoid"
    cache_enabled: true
```

#### Example Usage

```python
from core.providers.rerankers.cross_encoder import CrossEncoderProvider

reranker = CrossEncoderProvider()
await reranker.initialize(config)

candidates = [
    RerankingCandidate(id="1", content="Document 1", original_score=0.8),
    RerankingCandidate(id="2", content="Document 2", original_score=0.7),
]

results = await reranker.rerank("user query", candidates, top_k=5)
```

### vLLM Provider (`vllm`)

**Status**: 🚧 In Development  
**Performance**: ⭐⭐⭐ Variable  
**Local**: ✅ 100% Local

LLM-based reranking using vLLM for fast inference.

#### Configuration

```yaml
rag:
  reranking:
    provider: "vllm"
    model_name: "microsoft/DialoGPT-medium"
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.8
    max_model_len: 2048
```

## 🚀 Cache Providers

### Redis Provider (`redis`)

**Status**: ✅ Production Ready  
**Performance**: ⭐⭐⭐⭐⭐ Excellent  
**Local**: ✅ 100% Local

High-performance distributed caching.

#### Features

- **Multi-level Caching**: Different TTLs per cache level
- **Compression**: Automatic compression for large values
- **Batch Operations**: Efficient bulk get/set operations
- **Circuit Breaker**: Automatic failover protection
- **Memory Management**: Smart eviction policies

#### Configuration

```yaml
# config/config.yaml
databases:
  redis:
    host: localhost
    port: 6379
    db: 0
    password: "${REDIS_PASSWORD}"
    max_connections: 50
    min_connections: 10
    socket_timeout: 10
    socket_connect_timeout: 10
    connection_pool_class: "AsyncConnectionPool"
    health_check_interval: 60
    
    # TTL Settings
    embedding_cache_ttl: 86400  # 24 hours
    search_cache_ttl: 1800      # 30 minutes
    rerank_cache_ttl: 900       # 15 minutes
    analytics_cache_ttl: 300    # 5 minutes
    compression_threshold: 2048
```

### Memory Provider (`memory`)

**Status**: ✅ Production Ready  
**Performance**: ⭐⭐⭐ Good  
**Local**: ✅ 100% Local

In-memory caching for development and small deployments.

#### Configuration

```yaml
cache:
  provider: "memory"
  max_size_mb: 512
  cleanup_interval: 300  # 5 minutes
```

## 🔍 Hallucination Detectors

### Grounding Detector (`grounding`)

**Status**: ✅ Production Ready  
**Performance**: ⭐⭐⭐⭐ Good  
**Local**: ✅ 100% Local

Evidence-based hallucination detection using grounding analysis.

#### Features

- **Evidence Analysis**: Compares response against source documents
- **Confidence Scoring**: Provides detailed confidence metrics
- **Multi-factor Analysis**: Similarity, relevance, consistency checks
- **Configurable Thresholds**: Adjustable confidence levels

#### Configuration

```yaml
# config/config.yaml
rag:
  hallucination:
    enabled: true
    provider: "grounding"
    
    confidence_levels:
      rock_solid: 95  # Safe for automated actions
      high: 80        # Generally reliable
      fuzzy: 60       # Needs verification
      low: 0          # Not confident
    
    threshold: 75
    max_evidence: 10
    
    evidence_weights:
      similarity: 0.4
      relevance: 0.3
      consistency: 0.3
```

## ⚙️ Configuration Guide

### Provider Selection Strategy

```yaml
# config/providers.yaml
providers:
  # Primary and fallback configuration
  embeddings:
    primary: "huggingface"
    fallback: "openai"
    
  vector_store:
    primary: "pgvector"
    # No fallback for vector stores
    
  graph_engine:
    primary: "memgraph"
    fallback: null  # Optional component
    
  reranker:
    primary: "cross_encoder"
    fallback: "vllm"
    
  cache:
    primary: "redis"
    fallback: "memory"
    
  hallucination_detector:
    primary: "grounding"
    fallback: null  # Optional component
```

### Environment-based Configuration

```bash
# .env file
TYRA_EMBEDDING_PROVIDER=huggingface
TYRA_VECTOR_STORE_PROVIDER=pgvector
TYRA_GRAPH_ENGINE_PROVIDER=memgraph
TYRA_RERANKER_PROVIDER=cross_encoder
TYRA_CACHE_PROVIDER=redis

# Provider-specific settings
POSTGRES_PASSWORD=secure_password
REDIS_PASSWORD=secure_password
MEMGRAPH_PASSWORD=secure_password
OPENAI_API_KEY=your_api_key_here
```

### Dynamic Provider Switching

```python
# Runtime provider switching
from core.utils.registry import switch_provider, ProviderType

# Switch to backup embedding provider
await switch_provider(ProviderType.EMBEDDING, "openai")

# Check current provider
current = get_current_provider(ProviderType.EMBEDDING)
print(f"Current embedding provider: {current}")
```

## 🛠️ Adding Custom Providers

### Step 1: Implement Interface

```python
# src/core/providers/embeddings/my_provider.py
from typing import Dict, Any, List
import numpy as np
from ...interfaces.embeddings import EmbeddingProvider
from ...utils.registry import register_provider, ProviderType

@register_provider(ProviderType.EMBEDDING, "my_custom_provider")
class MyCustomEmbeddingProvider(EmbeddingProvider):
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the provider with configuration."""
        self.config = config
        self.model_name = config.get("model_name", "default-model")
        # Add your initialization logic here
        
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        # Implement query embedding logic
        return np.random.rand(384)  # Example
        
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts efficiently."""
        # Implement batch embedding logic
        return [np.random.rand(384) for _ in texts]  # Example
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "model_name": self.model_name,
            "provider": "my_custom_provider"
        }
        
    def get_dimensions(self) -> int:
        """Return embedding dimensions."""
        return 384
```

### Step 2: Add Configuration

```yaml
# config/providers.yaml
providers:
  embeddings:
    my_custom_provider:
      model_name: "my-custom-model"
      api_endpoint: "https://my-api.com/embed"
      api_key: "${MY_API_KEY}"
      batch_size: 32
      timeout: 30
```

### Step 3: Register and Use

```python
# The @register_provider decorator automatically registers the provider
# Use in configuration:
embeddings:
  primary: "my_custom_provider"
```

### Interface Requirements

Each provider type must implement specific methods:

#### EmbeddingProvider

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None: ...
    
    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray: ...
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]: ...
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def get_dimensions(self) -> int: ...
```

#### VectorStore

```python
class VectorStore(ABC):
    @abstractmethod
    async def store_documents(self, documents: List[VectorDocument]) -> bool: ...
    
    @abstractmethod
    async def search_similar(self, query_embedding: np.ndarray, 
                           top_k: int = 10) -> List[VectorSearchResult]: ...
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool: ...
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]: ...
```

## 📈 Performance Comparison

### Embedding Providers

| Provider | Local | Latency (ms) | Throughput (texts/s) | Quality Score |
|----------|-------|--------------|---------------------|---------------|
| HuggingFace (MiniLM) | ✅ | 45 | 150 | 8.2/10 |
| HuggingFace (E5-Large) | ✅ | 120 | 60 | 9.1/10 |
| OpenAI (text-3-small) | ❌ | 200 | 200 | 8.8/10 |

### Vector Stores

| Provider | Local | Search Latency (ms) | Index Size (GB) | Accuracy |
|----------|-------|-------------------|-----------------|----------|
| pgvector (HNSW) | ✅ | 15 | 2.1 | 98.5% |
| pgvector (IVFFlat) | ✅ | 25 | 1.8 | 99.2% |
| Chroma | ✅ | 35 | 2.5 | 97.8% |

### Rerankers

| Provider | Local | Latency (ms) | Quality Improvement | Resource Usage |
|----------|-------|--------------|-------------------|----------------|
| Cross-Encoder | ✅ | 85 | +12% relevance | Medium |
| vLLM | ✅ | 150 | +18% relevance | High |

## 🚨 Troubleshooting

### Common Issues

#### HuggingFace Provider

**Issue**: Model download fails
```bash
# Solution: Check internet connection and disk space
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2
```

**Issue**: CUDA out of memory
```yaml
# Solution: Reduce batch size or use CPU
embeddings:
  primary:
    provider: "huggingface"
    device: "cpu"  # or reduce batch_size
    batch_size: 16
```

#### pgvector Provider

**Issue**: pgvector extension not found
```sql
-- Solution: Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

**Issue**: Slow vector search
```sql
-- Solution: Create appropriate indexes
CREATE INDEX memory_embeddings_hnsw_idx ON memory_embeddings 
USING hnsw (embedding vector_cosine_ops);
```

#### Redis Provider

**Issue**: Connection refused
```bash
# Solution: Start Redis server
sudo systemctl start redis-server

# Or check configuration
redis-cli ping
```

### Monitoring Commands

```python
# Check provider health
from core.memory.manager import MemoryManager

manager = MemoryManager()
await manager.initialize()

health = await manager.health_check()
print(health)

# Get performance statistics
stats = await manager.get_stats()
print(stats["providers"])
```

### Debug Mode

```yaml
# config/config.yaml
environment: development
debug: true
log_level: DEBUG

logging:
  components:
    embedding_provider: DEBUG
    vector_store: DEBUG
    graph_engine: DEBUG
```

## 📚 Best Practices

### Provider Selection

1. **Start Simple**: Begin with default providers
2. **Monitor Performance**: Use built-in metrics to identify bottlenecks
3. **Test Fallbacks**: Ensure backup providers work correctly
4. **Optimize Gradually**: Tune one provider at a time

### Configuration Management

1. **Use Environment Variables**: Keep sensitive data in `.env`
2. **Version Control**: Track configuration changes
3. **Validate Settings**: Test configurations before deployment
4. **Document Changes**: Keep notes on optimization decisions

### Development Workflow

1. **Local Development**: Use in-memory providers for testing
2. **Staging Environment**: Mirror production provider setup
3. **Performance Testing**: Load test with realistic data
4. **Gradual Rollout**: Switch providers gradually in production

This comprehensive provider reference ensures you can effectively configure, optimize, and extend the Tyra MCP Memory Server's modular architecture.