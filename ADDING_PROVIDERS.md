# 🔧 Adding New Providers to Tyra MCP Memory Server

This guide walks you through the process of adding new providers to the Tyra MCP Memory Server system. Whether you're adding a new embedding model, vector store, or any other provider type, this guide will help you implement it correctly.

## Quick Start

### Using the Interactive Script (Recommended)

The easiest way to add a new provider is using the interactive script:

```bash
python scripts/add_provider.py
```

This will guide you through:
1. Selecting the provider type
2. Configuring the provider
3. Generating boilerplate code
4. Creating test templates
5. Updating configuration files

### Manual Process

If you prefer to create providers manually, follow the steps in this guide.

## Provider Types

The system supports these provider types:

| Type | Interface | Purpose |
|------|-----------|---------|
| `embeddings` | `EmbeddingProvider` | Generate text embeddings |
| `vector_stores` | `VectorStore` | Store and search vectors |
| `graph_engines` | `GraphEngine` | Manage graph databases |
| `graph_managers` | `GraphManager` | High-level graph operations |
| `rerankers` | `Reranker` | Rerank search results |
| `cache` | `CacheProvider` | Caching operations |

## Step-by-Step Guide

### Step 1: Choose Your Provider Type

Determine which type of provider you're creating:

- **Embedding Provider**: For new embedding models (e.g., OpenAI, Cohere, local models)
- **Vector Store Provider**: For new vector databases (e.g., Pinecone, Chroma, Milvus)
- **Graph Engine Provider**: For graph databases (e.g., Neo4j, ArangoDB)
- **Reranker Provider**: For reranking models (e.g., Cohere Rerank, custom models)

### Step 2: Understand the Interface

Each provider type has a specific interface to implement. Let's use an embedding provider as an example:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class EmbeddingProvider(ABC):
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the provider with configuration."""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    def supports_gpu(self) -> bool:
        """Check if GPU is supported."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass
```

### Step 3: Create the Provider Directory

Create a directory for your provider:

```bash
mkdir -p src/core/providers/embeddings/openai
```

### Step 4: Implement the Provider

Create your provider class:

```python
# src/core/providers/embeddings/openai/openai_provider.py
"""
OpenAI Embedding Provider implementation.
"""

import asyncio
from typing import Any, Dict, List, Optional
import numpy as np
import openai
from openai import AsyncOpenAI

from ....interfaces.embeddings import EmbeddingProvider
from ....utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI Embedding Provider using the OpenAI API."""
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.config: Dict[str, Any] = {}
        self.model_name: str = ""
        self.dimensions: int = 1536
        self.initialized: bool = False
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the OpenAI provider."""
        try:
            self.config = config
            self.model_name = config.get("model_name", "text-embedding-3-small")
            self.dimensions = config.get("dimensions", 1536)
            
            # Initialize OpenAI client
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            self.client = AsyncOpenAI(api_key=api_key)
            
            # Test the connection
            await self._test_connection()
            
            self.initialized = True
            logger.info(
                "OpenAI embedding provider initialized",
                model=self.model_name,
                dimensions=self.dimensions
            )
            
        except Exception as e:
            logger.error("Failed to initialize OpenAI provider", error=str(e))
            raise
    
    async def _test_connection(self) -> None:
        """Test the OpenAI connection."""
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=["test"],
                dimensions=self.dimensions
            )
            if not response.data:
                raise ValueError("OpenAI API test failed")
        except Exception as e:
            raise ConnectionError(f"OpenAI connection test failed: {e}")
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self.initialized:
            raise RuntimeError("Provider not initialized")
        
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    dimensions=self.dimensions
                )
                
                batch_embeddings = [
                    np.array(data.embedding, dtype=np.float32)
                    for data in response.data
                ]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(
                    "Failed to generate embeddings batch",
                    batch_size=len(batch),
                    error=str(e)
                )
                raise
        
        return embeddings
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimensions
    
    def supports_gpu(self) -> bool:
        """OpenAI API doesn't use local GPU."""
        return False
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self.initialized:
                return {
                    "status": "unhealthy",
                    "error": "Provider not initialized"
                }
            
            # Test with a simple embedding
            await self._test_connection()
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "dimensions": self.dimensions,
                "api_available": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name
            }
```

### Step 5: Update Module Imports

Update the `__init__.py` file in the provider directory:

```python
# src/core/providers/embeddings/openai/__init__.py
"""
OpenAI embedding provider.
"""

from .openai_provider import OpenAIEmbeddingProvider

__all__ = ["OpenAIEmbeddingProvider"]
```

### Step 6: Add Configuration

Add your provider to `config/providers.yaml`:

```yaml
embeddings:
  providers:
    openai:
      class: "src.core.providers.embeddings.openai.openai_provider.OpenAIEmbeddingProvider"
      config:
        model_name: "text-embedding-3-small"
        api_key: "${OPENAI_API_KEY}"
        dimensions: 1536
```

### Step 7: Create Tests

Create comprehensive tests for your provider:

```python
# tests/unit/providers/embeddings/test_openai_provider.py
"""
Tests for OpenAI Embedding Provider.
"""

import pytest
from unittest.mock import AsyncMock, patch
import numpy as np

from src.core.providers.embeddings.openai.openai_provider import OpenAIEmbeddingProvider


class TestOpenAIEmbeddingProvider:
    """Test suite for OpenAI Embedding Provider."""
    
    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return OpenAIEmbeddingProvider()
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration."""
        return {
            "model_name": "text-embedding-3-small",
            "api_key": "test-api-key",
            "dimensions": 1536
        }
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider, sample_config):
        """Test provider initialization."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Mock the test connection
            mock_response = AsyncMock()
            mock_response.data = [AsyncMock()]
            mock_client.embeddings.create.return_value = mock_response
            
            await provider.initialize(sample_config)
            
            assert provider.initialized is True
            assert provider.model_name == "text-embedding-3-small"
            assert provider.dimensions == 1536
    
    @pytest.mark.asyncio
    async def test_embed_texts(self, provider, sample_config):
        """Test text embedding generation."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Mock initialization
            mock_response = AsyncMock()
            mock_response.data = [AsyncMock()]
            mock_client.embeddings.create.return_value = mock_response
            
            await provider.initialize(sample_config)
            
            # Mock embedding response
            mock_embed_response = AsyncMock()
            mock_embed_response.data = [
                AsyncMock(embedding=[0.1] * 1536),
                AsyncMock(embedding=[0.2] * 1536)
            ]
            mock_client.embeddings.create.return_value = mock_embed_response
            
            # Test embedding
            texts = ["Hello", "World"]
            embeddings = await provider.embed_texts(texts)
            
            assert len(embeddings) == 2
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            assert all(emb.shape == (1536,) for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_health_check(self, provider, sample_config):
        """Test health check."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            mock_response = AsyncMock()
            mock_response.data = [AsyncMock()]
            mock_client.embeddings.create.return_value = mock_response
            
            await provider.initialize(sample_config)
            
            health = await provider.health_check()
            
            assert health["status"] == "healthy"
            assert health["model_name"] == "text-embedding-3-small"
            assert health["dimensions"] == 1536
```

### Step 8: Validate Your Implementation

1. **Run Interface Validation Tests**:
```bash
python -m pytest tests/test_interface_validation.py -v
```

2. **Run Your Provider Tests**:
```bash
python -m pytest tests/unit/providers/embeddings/test_openai_provider.py -v
```

3. **Test Integration**:
```python
from src.core.utils.registry import provider_registry, ProviderType

# Register and test your provider
provider = await provider_registry.get_provider(
    ProviderType.EMBEDDING,
    "openai"
)

# Test basic functionality
embeddings = await provider.embed_texts(["Hello world"])
print(f"Embedding shape: {embeddings[0].shape}")

# Test health check
health = await provider.health_check()
print(f"Health: {health}")
```

## Provider-Specific Guidelines

### Embedding Providers

**Key Requirements**:
- Return embeddings as numpy arrays
- Support batch processing
- Handle GPU/CPU detection
- Implement proper error handling
- Support query vs document embeddings (if applicable)

**Example Configuration**:
```yaml
embeddings:
  providers:
    my_embedder:
      class: "path.to.MyEmbeddingProvider"
      config:
        model_name: "my-model"
        device: "cuda"
        batch_size: 32
        normalize_embeddings: true
```

### Vector Store Providers

**Key Requirements**:
- Support similarity search
- Handle metadata filtering
- Implement hybrid search (vector + keyword)
- Support CRUD operations
- Implement efficient indexing

**Example Configuration**:
```yaml
vector_stores:
  providers:
    my_store:
      class: "path.to.MyVectorStore"
      config:
        connection_string: "connection://string"
        index_type: "hnsw"
        distance_metric: "cosine"
```

### Graph Engine Providers

**Key Requirements**:
- Support entity and relationship operations
- Handle graph traversal
- Implement temporal queries
- Support transactions
- Provide graph statistics

**Example Configuration**:
```yaml
graph_engines:
  providers:
    my_graph:
      class: "path.to.MyGraphEngine"
      config:
        uri: "bolt://localhost:7687"
        username: "user"
        password: "${GRAPH_PASSWORD}"
```

### Reranker Providers

**Key Requirements**:
- Support query-document relevance scoring
- Handle batch reranking
- Implement score fusion
- Support different reranking strategies

**Example Configuration**:
```yaml
rerankers:
  providers:
    my_reranker:
      class: "path.to.MyReranker"
      config:
        model_name: "rerank-model"
        max_candidates: 100
        device: "cuda"
```

## Best Practices

### 1. Error Handling

```python
try:
    # Your provider logic
    result = await some_operation()
    return result
except SpecificProviderError as e:
    logger.error("Provider-specific error", error=str(e))
    raise
except Exception as e:
    logger.error("Unexpected error in provider", error=str(e))
    raise RuntimeError(f"Provider operation failed: {e}")
```

### 2. Configuration Validation

```python
async def initialize(self, config: Dict[str, Any]) -> None:
    """Initialize with validation."""
    # Validate required fields
    required_fields = ["api_key", "model_name"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration: {field}")
    
    # Validate field types
    if not isinstance(config["batch_size"], int):
        raise ValueError("batch_size must be an integer")
    
    # Set defaults
    self.batch_size = config.get("batch_size", 32)
    self.timeout = config.get("timeout", 30)
```

### 3. Resource Management

```python
async def initialize(self, config: Dict[str, Any]) -> None:
    """Initialize with proper resource management."""
    try:
        # Initialize resources
        self.client = SomeClient(config)
        self.session = await self.client.create_session()
        
        # Store for cleanup
        self._resources = [self.client, self.session]
        
    except Exception as e:
        # Cleanup on failure
        await self._cleanup()
        raise

async def close(self) -> None:
    """Clean up resources."""
    await self._cleanup()

async def _cleanup(self) -> None:
    """Clean up all resources."""
    for resource in getattr(self, '_resources', []):
        try:
            if hasattr(resource, 'close'):
                await resource.close()
        except Exception as e:
            logger.warning(f"Error closing resource: {e}")
```

### 4. Comprehensive Health Checks

```python
async def health_check(self) -> Dict[str, Any]:
    """Comprehensive health check."""
    try:
        # Basic checks
        if not self.initialized:
            return {"status": "unhealthy", "error": "Not initialized"}
        
        # Connectivity check
        await self._test_connection()
        
        # Performance check
        start_time = time.time()
        await self._test_operation()
        response_time = time.time() - start_time
        
        # Resource check
        memory_usage = self._get_memory_usage()
        
        return {
            "status": "healthy",
            "response_time": response_time,
            "memory_usage": memory_usage,
            "model_loaded": True,
            "last_error": None
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
```

### 5. Logging and Monitoring

```python
# Use structured logging
logger.info(
    "Provider operation completed",
    operation="embed_texts",
    input_count=len(texts),
    processing_time=processing_time,
    model=self.model_name
)

# Log performance metrics
logger.debug(
    "Provider performance metrics",
    throughput=len(texts) / processing_time,
    memory_usage=memory_usage,
    gpu_usage=gpu_usage
)
```

## Testing Your Provider

### Unit Tests

1. **Test all interface methods**
2. **Test error conditions**
3. **Test configuration validation**
4. **Test resource cleanup**
5. **Test health checks**

### Integration Tests

1. **Test with real data**
2. **Test with the registry system**
3. **Test fallback scenarios**
4. **Test performance under load**

### Performance Tests

1. **Benchmark throughput**
2. **Test memory usage**
3. **Test concurrent access**
4. **Test resource limits**

## Deployment

### Environment Variables

Set up required environment variables:

```bash
# .env
OPENAI_API_KEY=your-api-key
COHERE_API_KEY=your-cohere-key
CUSTOM_PROVIDER_CONFIG=your-config
```

### Configuration Updates

Update your deployment configuration:

```yaml
# config/providers.yaml
embeddings:
  providers:
    production_embedder:
      class: "path.to.ProductionEmbeddingProvider"
      config:
        model_name: "production-model"
        api_key: "${PRODUCTION_API_KEY}"
        batch_size: 64
        timeout: 60
```

### Monitoring

Set up monitoring for your provider:

```python
# Monitor health
health = await provider_registry.health_check(
    ProviderType.EMBEDDING,
    "production_embedder"
)

# Monitor usage
stats = await provider_registry.get_stats()
```

## Common Issues and Solutions

### 1. Import Errors

**Issue**: Provider class cannot be imported
**Solution**: Check Python path and module structure

### 2. Configuration Errors

**Issue**: Provider fails to initialize
**Solution**: Validate configuration structure and required fields

### 3. Performance Issues

**Issue**: Provider is slow
**Solution**: Implement batch processing, caching, and connection pooling

### 4. Memory Issues

**Issue**: Provider uses too much memory
**Solution**: Implement proper resource management and cleanup

### 5. Connection Issues

**Issue**: Provider cannot connect to external services
**Solution**: Implement retry logic and proper error handling

## Support

For additional help:

1. **Check the logs** - Enable debug logging for detailed information
2. **Review existing providers** - Look at similar implementations
3. **Use the provider registry documentation** - See [PROVIDER_REGISTRY.md](docs/PROVIDER_REGISTRY.md)
4. **Run validation tests** - Use the interface validation test suite

## Contributing

When contributing a new provider:

1. **Follow the interface** - Implement all required methods
2. **Add comprehensive tests** - Cover all functionality
3. **Update documentation** - Document configuration options
4. **Add to CI/CD** - Ensure tests run in the pipeline
5. **Performance benchmarks** - Provide performance characteristics

Your provider will be reviewed for:
- Interface compliance
- Code quality
- Test coverage
- Documentation
- Performance impact

Thank you for contributing to the Tyra MCP Memory Server!