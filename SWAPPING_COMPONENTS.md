# 🔄 Component Swapping Guide

This guide explains how to swap different components in the Tyra MCP Memory Server, allowing you to change providers, models, and implementations without modifying core code.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Swappable Components](#swappable-components)
3. [Configuration-Based Swapping](#configuration-based-swapping)
4. [Runtime Swapping](#runtime-swapping)
5. [Adding New Providers](#adding-new-providers)
6. [Testing Component Swaps](#testing-component-swaps)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)

## Overview

The Tyra MCP Memory Server uses a provider-based architecture that allows you to swap components through configuration changes. This design enables:

- **Zero-code changes** for switching providers
- **A/B testing** of different implementations
- **Performance optimization** by choosing the best provider
- **Failover capabilities** with automatic fallbacks
- **Easy integration** of new providers

## Swappable Components

### 1. Embedding Providers

Switch between different embedding models for text vectorization.

**Available Providers:**
- `e5-large`: HuggingFace E5 Large v2 (1024 dimensions)
- `all-minilm`: Sentence Transformers MiniLM (384 dimensions)
- `bge-large`: BGE Large English (1024 dimensions)
- `instructor-xl`: Instructor XL (768 dimensions)

**Configuration:** `config/config.yaml`
```yaml
embedding:
  primary: "e5-large"      # Primary embedding provider
  fallback: "all-minilm"   # Fallback if primary fails
  batch_size: 32
```

**Example: Switching to BGE embeddings**
```yaml
embedding:
  primary: "bge-large"
  fallback: "e5-large"
  batch_size: 16
```

### 2. Vector Stores

Change where and how embeddings are stored.

**Available Providers:**
- `pgvector`: PostgreSQL with pgvector extension
- `qdrant`: Qdrant vector database (if configured)
- `weaviate`: Weaviate vector search (if configured)

**Configuration:** `config/config.yaml`
```yaml
vector_store:
  primary: "pgvector"
  search:
    default_top_k: 10
    max_top_k: 100
```

**Example: Using Qdrant instead**
```yaml
vector_store:
  primary: "qdrant"
  search:
    default_top_k: 20
    max_top_k: 200
```

### 3. Reranking Providers

Swap reranking algorithms for better search results.

**Available Providers:**
- `cross_encoder`: Cross-encoder neural reranking
- `vllm`: vLLM-based LLM reranking
- `hybrid`: Combination of multiple rerankers

**Configuration:** `config/config.yaml`
```yaml
rag:
  reranking:
    enabled: true
    provider: "cross_encoder"
    top_k: 10
```

**Example: Using vLLM reranking**
```yaml
rag:
  reranking:
    enabled: true
    provider: "vllm"
    top_k: 5
    vllm_config:
      model: "meta-llama/Llama-3.1-70B-Instruct"
      temperature: 0.1
```

### 4. Graph Engines

Switch between different graph database implementations.

**Available Providers:**
- `memgraph`: Memgraph in-memory graph database (primary)

**Configuration:** `config/config.yaml`
```yaml
graph:
  engine: "memgraph"
  manager: "graphiti"
  enable_temporal: true
```

### 5. Cache Providers

Change caching implementations.

**Available Providers:**
- `redis`: Redis in-memory cache
- `memory`: In-process memory cache

**Configuration:** `config/config.yaml`
```yaml
cache:
  enabled: true
  provider: "redis"  # or "memory"
  redis:
    host: "localhost"
    port: 6379
```

## Configuration-Based Swapping

### Step 1: Update Configuration

Edit the appropriate configuration file:

```bash
# Edit main configuration
vim config/config.yaml

# Or edit provider-specific configuration
vim config/providers.yaml
```

### Step 2: Validate Configuration

```bash
# Validate all configurations
python scripts/validate_config.py

# Check provider references
python scripts/validate_config.py --check-references
```

### Step 3: Restart Services

```bash
# If using Docker
docker-compose restart

# If running directly
# Stop the service (Ctrl+C) and restart
python main.py
```

## Runtime Swapping

### Dynamic Provider Switching

Some components support runtime switching without restart:

```python
# Example: Switch embedding provider at runtime
from src.core.utils.registry import ProviderRegistry

# Get registry instance
registry = ProviderRegistry()

# Switch to different embedding provider
registry.set_active_provider("embeddings", "bge-large")
```

### A/B Testing Configuration

Enable A/B testing for gradual rollout:

```yaml
ab_testing:
  enabled: true
  experiments:
    new_embeddings:
      control: "e5-large"
      treatment: "bge-large"
      traffic_split: 0.2  # 20% to new provider
      metrics:
        - "embedding_latency"
        - "search_accuracy"
```

### Automatic Failover

Configure automatic failover for high availability:

```yaml
embedding:
  primary: "e5-large"
  fallback: "all-minilm"
  failover:
    enabled: true
    max_retries: 3
    timeout_ms: 5000
```

## Adding New Providers

### Quick Method: Using the Provider Wizard

```bash
python scripts/add_provider.py
```

Follow the interactive prompts to:
1. Select provider type
2. Enter provider details
3. Configure settings
4. Generate implementation stub

### Manual Method: Step-by-Step

#### 1. Create Provider Implementation

```python
# src/core/providers/embeddings/my_embedder.py
from src.core.interfaces.embeddings import EmbeddingProvider

class MyEmbedder(EmbeddingProvider):
    def __init__(self, config: dict):
        self.model = config.get("model_name")
        # Initialize your embedder
    
    async def embed_text(self, text: str) -> List[float]:
        # Your embedding logic
        return embeddings
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Batch embedding logic
        return batch_embeddings
    
    @property
    def dimension(self) -> int:
        return 768  # Your model's dimension
```

#### 2. Register Provider

```python
# src/core/providers/embeddings/registry.py
from src.core.utils.registry import ProviderRegistry

registry = ProviderRegistry()
registry.register(
    "embeddings",
    "my_embedder",
    "src.core.providers.embeddings.my_embedder.MyEmbedder"
)
```

#### 3. Add Configuration

```yaml
# config/providers.yaml
embeddings:
  providers:
    my_embedder:
      class: "src.core.providers.embeddings.my_embedder.MyEmbedder"
      config:
        model_name: "my-model/name"
        device: "cuda"
```

#### 4. Update Main Configuration

```yaml
# config/config.yaml
embedding:
  primary: "my_embedder"
  fallback: "e5-large"
```

## Testing Component Swaps

### 1. Unit Testing

```python
# tests/test_provider_swap.py
import pytest
from src.core.utils.registry import ProviderRegistry

def test_provider_swap():
    registry = ProviderRegistry()
    
    # Test swapping embeddings
    registry.set_active_provider("embeddings", "e5-large")
    provider1 = registry.get_provider("embeddings")
    
    registry.set_active_provider("embeddings", "all-minilm")
    provider2 = registry.get_provider("embeddings")
    
    assert provider1 != provider2
    assert provider1.dimension == 1024
    assert provider2.dimension == 384
```

### 2. Integration Testing

```bash
# Run integration tests with different providers
EMBEDDING_PROVIDER=e5-large pytest tests/integration/
EMBEDDING_PROVIDER=all-minilm pytest tests/integration/
```

### 3. Performance Testing

```python
# scripts/benchmark_providers.py
import time
import asyncio
from src.core.utils.registry import ProviderRegistry

async def benchmark_provider(provider_name: str, texts: List[str]):
    registry = ProviderRegistry()
    registry.set_active_provider("embeddings", provider_name)
    
    provider = registry.get_provider("embeddings")
    
    start = time.time()
    embeddings = await provider.embed_batch(texts)
    end = time.time()
    
    return {
        "provider": provider_name,
        "time": end - start,
        "texts_per_second": len(texts) / (end - start)
    }
```

## Performance Considerations

### 1. Embedding Providers

| Provider | Dimensions | Speed | Quality | Memory Usage |
|----------|------------|-------|---------|--------------|
| e5-large | 1024 | Medium | High | High |
| all-minilm | 384 | Fast | Medium | Low |
| bge-large | 1024 | Slow | Highest | High |
| instructor-xl | 768 | Medium | High | Medium |

### 2. Vector Store Providers

| Provider | Write Speed | Query Speed | Scalability | Features |
|----------|------------|-------------|-------------|----------|
| pgvector | Fast | Fast | High | Full SQL |
| qdrant | Medium | Very Fast | Very High | Advanced filtering |
| weaviate | Medium | Fast | High | GraphQL API |

### 3. Reranking Providers

| Provider | Accuracy | Speed | Cost | Best For |
|----------|----------|-------|------|----------|
| cross_encoder | High | Fast | Low | General use |
| vllm | Very High | Slow | High | Critical queries |
| hybrid | Highest | Medium | Medium | Balanced approach |

## Troubleshooting

### Provider Not Found

```bash
Error: Provider 'my_provider' not found in registry
```

**Solution:**
1. Check provider is registered in registry
2. Verify class path is correct
3. Ensure provider module is importable

### Dimension Mismatch

```bash
Error: Embedding dimension mismatch: expected 1024, got 384
```

**Solution:**
1. Check if you're switching between incompatible providers
2. Re-embed all documents with new provider
3. Use migration script to handle dimension changes

### Performance Degradation

**Symptoms:** Slower queries after switching providers

**Solutions:**
1. Check provider configuration (batch size, device)
2. Verify indexes are created for new provider
3. Monitor resource usage (CPU, GPU, memory)
4. Consider using caching more aggressively

### Configuration Not Applied

**Symptoms:** Old provider still being used

**Solutions:**
1. Restart all services
2. Clear provider cache: `redis-cli FLUSHDB`
3. Check for hardcoded provider references
4. Verify configuration file syntax

## Best Practices

1. **Test Before Production**
   - Always test provider swaps in development
   - Run performance benchmarks
   - Verify functionality with integration tests

2. **Gradual Rollout**
   - Use A/B testing for production changes
   - Monitor metrics during rollout
   - Have rollback plan ready

3. **Document Changes**
   - Record why you switched providers
   - Document performance differences
   - Update team knowledge base

4. **Monitor After Swap**
   - Watch error rates
   - Track performance metrics
   - Monitor resource usage
   - Check user feedback

5. **Maintain Compatibility**
   - Keep fallback providers configured
   - Test failover scenarios
   - Ensure data format compatibility

## Advanced Scenarios

### Multi-Provider Setup

Use different providers for different use cases:

```yaml
# config/advanced_providers.yaml
provider_rules:
  - condition: "agent == 'tyra'"
    embedding_provider: "e5-large"
    reranking_provider: "vllm"
  
  - condition: "query_type == 'trading'"
    embedding_provider: "instructor-xl"
    reranking_provider: "vllm"
    confidence_threshold: 0.95
  
  - condition: "default"
    embedding_provider: "all-minilm"
    reranking_provider: "cross_encoder"
```

### Provider Chaining

Chain multiple providers for enhanced functionality:

```yaml
# config/provider_chains.yaml
chains:
  enhanced_search:
    - provider: "e5-large"
      purpose: "initial_embedding"
    - provider: "cross_encoder"
      purpose: "first_rerank"
    - provider: "vllm"
      purpose: "final_rerank"
      config:
        only_top_k: 5
```

### Custom Provider Combinations

Create hybrid providers that combine multiple implementations:

```python
# src/core/providers/embeddings/hybrid.py
class HybridEmbedder(EmbeddingProvider):
    def __init__(self, config: dict):
        self.providers = [
            registry.get_provider("embeddings", name)
            for name in config["providers"]
        ]
    
    async def embed_text(self, text: str) -> List[float]:
        # Combine embeddings from multiple providers
        embeddings = []
        for provider in self.providers:
            emb = await provider.embed_text(text)
            embeddings.extend(emb)
        return embeddings
```

## Summary

The component swapping system in Tyra MCP Memory Server provides:

- **Flexibility** to choose the best provider for your needs
- **Easy testing** of new implementations
- **Zero-downtime** swaps with proper configuration
- **Performance optimization** through provider selection
- **Future-proofing** by allowing new providers

For more details on specific providers, see:
- [PROVIDERS.md](PROVIDERS.md) - List of all available providers
- [ADDING_PROVIDERS.md](ADDING_PROVIDERS.md) - How to add new providers
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview