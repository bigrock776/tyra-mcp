# 🏗️ Provider Registry System Documentation

## Overview

The Provider Registry System is the core component that enables dynamic, hot-swappable providers in the Tyra MCP Memory Server. It provides a centralized mechanism for managing different implementations of embeddings, vector stores, graph engines, rerankers, and other components without requiring code changes or service restarts.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Provider Registry System                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Registry      │  │   Health        │  │   Configuration │ │
│  │   Manager       │  │   Monitor       │  │   Loader        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Dynamic       │  │   Interface     │  │   Fallback      │ │
│  │   Loader        │  │   Validator     │  │   Manager       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Dynamic Loading**: Load providers at runtime from configuration
- **Hot Swapping**: Replace providers without service restart
- **Health Monitoring**: Continuous health checks and automatic failover
- **Interface Validation**: Ensures providers implement required interfaces
- **Fallback Support**: Automatic fallback to secondary providers
- **Usage Tracking**: Monitor provider usage and performance
- **Configuration Management**: Centralized provider configuration

## Provider Types

The system supports the following provider types:

### 1. Embedding Providers (`embeddings`)
- **Purpose**: Generate text embeddings for semantic search
- **Interface**: `EmbeddingProvider`
- **Examples**: HuggingFace, OpenAI, Cohere

### 2. Vector Store Providers (`vector_stores`)
- **Purpose**: Store and search vector embeddings
- **Interface**: `VectorStore`
- **Examples**: PostgreSQL+pgvector, Qdrant, Weaviate

### 3. Graph Engine Providers (`graph_engines`)
- **Purpose**: Manage knowledge graph operations
- **Interface**: `GraphEngine`
- **Examples**: Memgraph (primary)

### 4. Graph Manager Providers (`graph_managers`)
- **Purpose**: High-level graph management and temporal operations
- **Interface**: `GraphManager`
- **Examples**: Graphiti integration

### 5. Reranker Providers (`rerankers`)
- **Purpose**: Rerank search results for better relevance
- **Interface**: `Reranker`
- **Examples**: Cross-encoder, BGE, Cohere

### 6. Cache Providers (`cache`)
- **Purpose**: Caching for performance optimization
- **Interface**: `CacheProvider`
- **Examples**: Redis, Memory cache

## Configuration

### Provider Configuration Structure

Providers are configured in `config/providers.yaml`:

```yaml
# Provider type (embeddings, vector_stores, graph_engines, etc.)
provider_type:
  providers:
    provider_name:
      class: "full.python.path.to.ProviderClass"
      config:
        # Provider-specific configuration
        key: value
        another_key: another_value
```

### Example Configuration

```yaml
embeddings:
  providers:
    e5-large:
      class: "src.core.providers.embeddings.huggingface.HuggingFaceProvider"
      config:
        model_name: "intfloat/e5-large-v2"
        device: "cuda"
        normalize_embeddings: true
        batch_size: 32

vector_stores:
  providers:
    pgvector:
      class: "src.core.providers.vector_stores.pgvector.PgVectorStore"
      config:
        connection_string: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
        table_name: "memory_embeddings"
        vector_column: "embedding"
```

## Usage

### Basic Usage

```python
from src.core.utils.registry import provider_registry, ProviderType

# Get a provider instance
embedder = await provider_registry.get_provider(
    ProviderType.EMBEDDING, 
    "e5-large"
)

# Use the provider
embeddings = await embedder.embed_texts(["Hello world", "This is a test"])
```

### With Fallback

```python
# Set up fallback
await provider_registry.set_fallback_provider(
    ProviderType.EMBEDDING,
    primary_name="e5-large",
    fallback_name="all-minilm"
)

# Get provider with automatic fallback
embedder = await provider_registry.get_provider_with_fallback(
    ProviderType.EMBEDDING,
    "e5-large"
)
```

### Dynamic Registration

```python
# Register a new provider at runtime
success = await provider_registry.register_provider(
    provider_type=ProviderType.EMBEDDING,
    name="new_embedder",
    class_path="my.custom.provider.MyEmbeddingProvider",
    config={"model_name": "my-model", "device": "cpu"}
)
```

## Provider Implementation

### Creating a New Provider

1. **Implement the Required Interface**

```python
from src.core.interfaces.embeddings import EmbeddingProvider
from typing import List, Dict, Any
import numpy as np

class MyEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.config = {}
        self.initialized = False
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the provider with configuration."""
        self.config = config
        # Initialize your provider here
        self.initialized = True
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for texts."""
        if not self.initialized:
            raise RuntimeError("Provider not initialized")
        
        # Your embedding implementation
        return [np.random.rand(384) for _ in texts]
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return 384
    
    def supports_gpu(self) -> bool:
        """Check if GPU is supported."""
        return False
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.config.get("model_name", "unknown")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.initialized else "unhealthy",
            "initialized": self.initialized,
            "model_name": self.get_model_name()
        }
```

2. **Add to Configuration**

```yaml
embeddings:
  providers:
    my_embedder:
      class: "path.to.MyEmbeddingProvider"
      config:
        model_name: "my-custom-model"
        device: "cpu"
```

3. **Use the Provider**

```python
embedder = await provider_registry.get_provider(
    ProviderType.EMBEDDING,
    "my_embedder"
)
```

### Using the Add Provider Script

For easier provider creation, use the interactive script:

```bash
python scripts/add_provider.py
```

This will:
- Guide you through provider creation
- Generate boilerplate code
- Update configuration files
- Create test templates

## Health Monitoring

### Automatic Health Checks

The registry automatically monitors provider health:

```python
# Start health monitoring
await provider_registry.start_health_monitoring()

# Check specific provider health
health = await provider_registry.health_check(
    ProviderType.EMBEDDING,
    "e5-large"
)

print(health["status"])  # "healthy" or "unhealthy"
```

### Health Check Implementation

Providers should implement comprehensive health checks:

```python
async def health_check(self) -> Dict[str, Any]:
    """Comprehensive health check."""
    try:
        # Test basic functionality
        test_result = await self.embed_query("test")
        
        return {
            "status": "healthy",
            "response_time": 0.123,
            "test_successful": True,
            "model_loaded": True,
            "gpu_available": self.supports_gpu(),
            "last_error": None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_error": str(e)
        }
```

## Hot Swapping

### Reloading Providers

```python
# Reload a provider (e.g., after code changes)
success = await provider_registry.reload_provider(
    ProviderType.EMBEDDING,
    "e5-large"
)

if success:
    print("Provider reloaded successfully")
```

### Zero-Downtime Updates

1. **Register New Version**:
```python
await provider_registry.register_provider(
    provider_type=ProviderType.EMBEDDING,
    name="e5-large-v2",
    class_path="src.core.providers.embeddings.huggingface.HuggingFaceProviderV2",
    config=new_config
)
```

2. **Update Configuration**:
```python
# Update active provider in configuration
# This can be done through configuration reload
```

3. **Graceful Transition**:
```python
# Old provider continues serving existing requests
# New requests use the new provider
```

## Statistics and Monitoring

### Usage Statistics

```python
# Get comprehensive registry statistics
stats = await provider_registry.get_stats()

print(f"Total providers: {sum(stats['provider_counts'].values())}")
print(f"Healthy providers: {stats['health_summary']}")
print(f"Total usage: {stats['usage_stats']}")
```

### Provider Information

```python
# List all providers
providers = await provider_registry.list_providers()

# List specific provider type
embedding_providers = await provider_registry.list_providers(
    provider_type=ProviderType.EMBEDDING
)

# Get detailed provider info
for name, info in embedding_providers["embeddings"].items():
    print(f"Provider: {name}")
    print(f"  Status: {info['health_status']}")
    print(f"  Usage: {info['usage_count']}")
    print(f"  Initialized: {info['initialized']}")
```

## Error Handling

### Provider Failures

The registry handles provider failures gracefully:

```python
try:
    provider = await provider_registry.get_provider(
        ProviderType.EMBEDDING,
        "failing_provider"
    )
except Exception as e:
    # Provider failed to load or initialize
    logger.error(f"Provider failed: {e}")
    
    # Try fallback
    provider = await provider_registry.get_provider_with_fallback(
        ProviderType.EMBEDDING,
        "failing_provider"
    )
```

### Circuit Breaker Pattern

Providers automatically implement circuit breaker patterns:

```python
# Provider will be marked as unhealthy after failures
# Health monitoring will attempt recovery
# Fallback providers will be used automatically
```

## Best Practices

### 1. Interface Compliance

- Always implement the full interface
- Use proper type hints
- Handle all error cases
- Provide comprehensive health checks

### 2. Configuration Management

- Use environment variables for secrets
- Provide sensible defaults
- Validate configuration on initialization
- Document all configuration options

### 3. Error Handling

- Implement proper exception handling
- Use structured logging
- Provide meaningful error messages
- Implement graceful degradation

### 4. Performance

- Implement efficient initialization
- Use connection pooling where appropriate
- Cache expensive operations
- Monitor resource usage

### 5. Testing

- Write comprehensive unit tests
- Test with various configurations
- Test error conditions
- Test health check implementations

## Troubleshooting

### Common Issues

1. **Provider Not Found**
   - Check configuration file syntax
   - Verify provider name spelling
   - Ensure provider class is importable

2. **Initialization Failures**
   - Check configuration values
   - Verify dependencies are installed
   - Check network connectivity for external services

3. **Health Check Failures**
   - Check provider-specific requirements
   - Verify resource availability (GPU, memory, etc.)
   - Check external service availability

### Debug Mode

Enable debug logging for detailed provider information:

```python
import logging
logging.getLogger("src.core.utils.registry").setLevel(logging.DEBUG)
```

### Configuration Validation

Validate provider configurations:

```python
# Check if provider config is valid
try:
    provider = await provider_registry.get_provider(
        ProviderType.EMBEDDING,
        "test_provider"
    )
    health = await provider.health_check()
    print(f"Provider health: {health}")
except Exception as e:
    print(f"Configuration error: {e}")
```

## Migration Guide

### From Direct Imports

**Before:**
```python
from src.core.providers.embeddings.huggingface import HuggingFaceProvider

embedder = HuggingFaceProvider()
await embedder.initialize(config)
```

**After:**
```python
from src.core.utils.registry import provider_registry, ProviderType

embedder = await provider_registry.get_provider(
    ProviderType.EMBEDDING,
    "e5-large"
)
```

### Configuration Updates

Update your configuration files to use the new provider registry format. The `add_provider.py` script can help with this migration.

## API Reference

### ProviderRegistry Class

#### Methods

- `register_provider(provider_type, name, class_path, config, metadata=None)` - Register a new provider
- `get_provider(provider_type, name, initialize=True)` - Get provider instance
- `get_provider_with_fallback(provider_type, name)` - Get provider with fallback
- `list_providers(provider_type=None, include_instances=False)` - List providers
- `reload_provider(provider_type, name)` - Reload provider
- `set_fallback_provider(provider_type, primary_name, fallback_name)` - Set fallback
- `health_check(provider_type, name)` - Check provider health
- `start_health_monitoring()` - Start health monitoring
- `stop_health_monitoring()` - Stop health monitoring
- `get_stats()` - Get registry statistics
- `cleanup()` - Clean up resources

### Global Functions

- `get_provider(provider_type, name)` - Get provider from global registry
- `get_provider_with_fallback(provider_type, name)` - Get provider with fallback

### ProviderType Enum

- `EMBEDDING` - Embedding providers
- `VECTOR_STORE` - Vector store providers
- `GRAPH_ENGINE` - Graph engine providers
- `GRAPH_MANAGER` - Graph manager providers
- `GRAPH_CLIENT` - Graph client providers
- `RERANKER` - Reranker providers
- `CACHE` - Cache providers
- `TOOL` - Tool providers

## Contributing

### Adding New Provider Types

1. Add to `ProviderType` enum
2. Update interface validation in `_validate_provider_class`
3. Create corresponding interface
4. Update configuration schema
5. Add to documentation

### Extending Registry Features

1. Follow existing patterns
2. Maintain backward compatibility
3. Add comprehensive tests
4. Update documentation

For more information on creating providers, see [ADDING_PROVIDERS.md](../ADDING_PROVIDERS.md).