# 🌐 API Reference - Tyra MCP Memory Server

## 📋 Overview

Tyra MCP Memory Server provides both **MCP Protocol** and **REST API** interfaces for memory operations, advanced RAG, and knowledge graph interactions.

## 🔗 Base URLs

- **REST API**: `http://localhost:8000`
- **MCP Protocol**: Uses standard MCP client libraries
- **Admin Interface**: `http://localhost:8000/admin`
- **Health Checks**: `http://localhost:8000/health`

## 🛡️ Authentication

### API Key Authentication

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/v1/memory/store
```

### Bearer Token (Optional)

```bash
curl -H "Authorization: Bearer your-token" http://localhost:8000/v1/memory/store
```

## 🧠 MCP Protocol Interface

### Available Tools

#### `save_memory`
Store memory with advanced processing.

```typescript
// Input
{
  text: string;
  agent_id?: string;
  session_id?: string;
  metadata?: Record<string, any>;
  extract_entities?: boolean;
  chunk_content?: boolean;
}

// Output
{
  success: boolean;
  memory_id: string;
  chunk_ids?: string[];
  entities_created?: number;
  relationships_created?: number;
  processing_time: {
    embedding: number;
    storage: number;
    graph?: number;
  };
  error?: string;
}
```

**Example:**
```javascript
const result = await mcp.call("save_memory", {
  text: "Claude learned that Paris is the capital of France during our geography lesson.",
  agent_id: "claude",
  extract_entities: true,
  metadata: { topic: "geography", confidence: 95 }
});
```

#### `search_memories`
Advanced memory search with RAG pipeline.

```typescript
// Input
{
  query: string;
  agent_id?: string;
  session_id?: string;
  top_k?: number;
  min_confidence?: number;
  search_type?: "vector" | "hybrid" | "keyword";
  include_analysis?: boolean;
  rerank?: boolean;
}

// Output
{
  success: boolean;
  query: string;
  results: Array<{
    id: string;
    content: string;
    score: number;
    metadata: Record<string, any>;
    created_at: string;
  }>;
  total_results: number;
  search_type: string;
  processing_time: {
    embedding: number;
    search: number;
    reranking?: number;
    analysis?: number;
  };
  hallucination_analysis?: {
    is_hallucination: boolean;
    confidence: number;
    grounding_score: number;
  };
  error?: string;
}
```

#### `get_all_memories`
Retrieve all memories for an agent.

```typescript
// Input
{
  agent_id?: string;
  limit?: number;
  offset?: number;
  include_metadata?: boolean;
}

// Output
{
  success: boolean;
  memories: Array<Memory>;
  total_count: number;
  error?: string;
}
```

### MCP Usage Example

```python
import mcp

# Initialize MCP client
client = mcp.Client("tyra-memory-server")

# Store memory
await client.call("save_memory", {
    "text": "The meeting is scheduled for 3 PM tomorrow",
    "agent_id": "tyra",
    "metadata": {"type": "schedule", "priority": "high"}
})

# Search memories
results = await client.call("search_memories", {
    "query": "meeting schedule",
    "agent_id": "tyra",
    "top_k": 5,
    "include_analysis": True
})
```

## 🌐 REST API Endpoints

### Memory Operations

#### `POST /v1/memory/store`
Store a new memory with advanced processing.

**Request:**
```json
{
  "content": "string",
  "agent_id": "string",
  "session_id": "string",
  "metadata": {},
  "extract_entities": true,
  "chunk_content": false
}
```

**Response:**
```json
{
  "success": true,
  "memory_id": "mem_123456",
  "chunk_ids": ["chunk_1", "chunk_2"],
  "entities_created": 3,
  "relationships_created": 2,
  "processing_time": {
    "embedding": 45,
    "storage": 12,
    "graph": 23
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/v1/memory/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "content": "Einstein developed the theory of relativity",
    "agent_id": "claude",
    "extract_entities": true,
    "metadata": {"topic": "physics", "source": "textbook"}
  }'
```

#### `POST /v1/memory/search`
Advanced memory search with RAG pipeline.

**Request:**
```json
{
  "query": "string",
  "agent_id": "string",
  "top_k": 10,
  "min_confidence": 0.5,
  "search_type": "hybrid",
  "include_analysis": true,
  "rerank": true
}
```

**Response:**
```json
{
  "success": true,
  "query": "Einstein theory",
  "results": [
    {
      "id": "mem_123456",
      "content": "Einstein developed the theory of relativity",
      "score": 0.95,
      "metadata": {"topic": "physics"},
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total_results": 1,
  "search_type": "hybrid",
  "processing_time": {
    "embedding": 25,
    "search": 15,
    "reranking": 35,
    "analysis": 20
  },
  "hallucination_analysis": {
    "is_hallucination": false,
    "confidence": 0.92,
    "grounding_score": 0.88
  }
}
```

#### `GET /v1/memory/stats`
Get memory system statistics.

**Parameters:**
- `agent_id` (optional): Filter by agent
- `include_performance` (boolean): Include performance metrics
- `include_recommendations` (boolean): Include optimization recommendations

**Response:**
```json
{
  "success": true,
  "memory_stats": {
    "total_memories": 10000,
    "recent_activity": 150,
    "storage_size_mb": 500.5
  },
  "performance_stats": {
    "avg_query_time": 45.2,
    "cache_hit_rate": 0.85
  },
  "health_score": 0.95,
  "recommendations": [
    "Consider increasing cache TTL for better performance"
  ]
}
```

#### `DELETE /v1/memory/{memory_id}`
Delete a specific memory.

**Response:**
```json
{
  "success": true,
  "message": "Memory mem_123456 deleted successfully"
}
```

### RAG Operations

#### `POST /v1/rag/analyze`
Analyze response for hallucinations and grounding.

**Request:**
```json
{
  "response": "string",
  "query": "string",
  "retrieved_memories": [
    {"content": "string", "score": 0.9}
  ],
  "detailed": true
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "is_hallucination": false,
    "overall_confidence": 85,
    "grounding_score": 0.8,
    "consistency_score": 0.9,
    "confidence_level": "high",
    "query_relevance": 0.88
  },
  "detailed_breakdown": {
    "chunk_similarities": [0.9, 0.8, 0.7],
    "contradictions": [],
    "factual_accuracy": 0.92
  }
}
```

#### `POST /v1/rag/rerank`
Rerank search results for better relevance.

**Request:**
```json
{
  "query": "string",
  "documents": [
    {"id": "doc1", "text": "content1", "score": 0.8},
    {"id": "doc2", "text": "content2", "score": 0.7}
  ],
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "reranked_results": [
    {"id": "doc2", "text": "content2", "score": 0.92},
    {"id": "doc1", "text": "content1", "score": 0.85}
  ],
  "processing_time": 120
}
```

### Graph Operations

#### `GET /v1/graph/entities`
Get entities from knowledge graph.

**Parameters:**
- `agent_id` (optional): Filter by agent
- `entity_type` (optional): Filter by type
- `limit` (optional): Maximum results

**Response:**
```json
{
  "success": true,
  "entities": [
    {
      "id": "entity_123",
      "name": "Einstein",
      "type": "PERSON",
      "properties": {"born": "1879", "field": "physics"},
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total_count": 1
}
```

#### `GET /v1/graph/relationships`
Get relationships between entities.

**Parameters:**
- `from_entity` (optional): Source entity ID
- `to_entity` (optional): Target entity ID
- `relationship_type` (optional): Type of relationship

**Response:**
```json
{
  "success": true,
  "relationships": [
    {
      "id": "rel_123",
      "from_entity": "entity_123",
      "to_entity": "entity_456",
      "type": "DEVELOPED",
      "properties": {"year": "1915"},
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### `POST /v1/graph/query`
Execute graph traversal queries.

**Request:**
```json
{
  "query": "MATCH (p:PERSON)-[r:DEVELOPED]->(t:THEORY) RETURN p, r, t",
  "parameters": {},
  "format": "cypher"
}
```

### Search Operations

#### `POST /v1/search/hybrid`
Hybrid search combining vector and keyword search.

**Request:**
```json
{
  "query": "string",
  "agent_id": "string",
  "vector_weight": 0.7,
  "keyword_weight": 0.3,
  "top_k": 10
}
```

#### `POST /v1/search/vector`
Pure vector similarity search.

#### `POST /v1/search/keyword`
Full-text keyword search.

#### `POST /v1/search/temporal`
Time-based memory search.

**Request:**
```json
{
  "query": "string",
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-12-31T23:59:59Z"
  },
  "temporal_weight": 0.3
}
```

### Chat Operations

#### `POST /v1/chat/completion`
Generate chat completions with memory context.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What did Einstein develop?"}
  ],
  "agent_id": "claude",
  "use_memory": true,
  "memory_top_k": 5,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "response": "Einstein developed the theory of relativity...",
  "memory_context": [
    {"content": "Einstein developed the theory of relativity", "score": 0.95}
  ],
  "confidence_analysis": {
    "overall_confidence": 92,
    "is_hallucination": false
  },
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 75
  }
}
```

### Admin Operations

#### `GET /v1/admin/health`
Comprehensive health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "components": {
    "vector_store": {"healthy": true, "response_time": 15},
    "embeddings": {"healthy": true, "models_loaded": 2},
    "graph_engine": {"healthy": true, "entities": 1000},
    "cache": {"healthy": true, "hit_rate": 0.85}
  },
  "performance": {
    "avg_response_time": 45,
    "requests_per_second": 100,
    "error_rate": 0.01
  }
}
```

#### `POST /v1/admin/reload-config`
Reload configuration without restart.

#### `GET /v1/admin/metrics`
Prometheus-compatible metrics.

#### `POST /v1/admin/backup`
Create system backup.

#### `POST /v1/admin/optimize`
Trigger system optimization.

## 📊 Monitoring & Analytics

### Telemetry Endpoints

#### `GET /v1/telemetry/metrics`
Real-time system metrics.

#### `GET /v1/telemetry/traces`
Distributed tracing data.

#### `GET /v1/telemetry/logs`
Structured log stream.

### Analytics Endpoints

#### `GET /v1/analytics/usage`
Usage analytics and patterns.

#### `GET /v1/analytics/performance`
Performance analytics and recommendations.

#### `GET /v1/analytics/quality`
Memory and response quality metrics.

## 🔧 Utility Endpoints

### System Information

#### `GET /v1/system/info`
System information and capabilities.

**Response:**
```json
{
  "version": "1.0.0",
  "environment": "production",
  "capabilities": {
    "embedding_models": ["intfloat/e5-large-v2", "all-MiniLM-L12-v2"],
    "reranking": true,
    "hallucination_detection": true,
    "knowledge_graph": true
  },
  "limits": {
    "max_memory_size": 1000000,
    "max_batch_size": 100,
    "rate_limit": 1000
  }
}
```

### Configuration

#### `GET /v1/config/schema`
Configuration schema and validation rules.

#### `POST /v1/config/validate`
Validate configuration changes.

## 🚨 Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Required field 'content' is missing",
    "details": {
      "field": "content",
      "value": null,
      "constraint": "required"
    },
    "request_id": "req_123456",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

### Error Codes

- `VALIDATION_ERROR`: Request validation failed
- `AUTHENTICATION_ERROR`: Invalid API key or token
- `RATE_LIMIT_ERROR`: Too many requests
- `MEMORY_NOT_FOUND`: Memory ID not found
- `EMBEDDING_ERROR`: Embedding generation failed
- `DATABASE_ERROR`: Database operation failed
- `INTERNAL_ERROR`: Unexpected server error

## 📈 Rate Limiting

- **Default Limit**: 1000 requests per minute per API key
- **Burst Limit**: 50 requests per second
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## 🔒 Security

### Best Practices

1. **Use HTTPS**: Always use HTTPS in production
2. **Secure API Keys**: Store API keys securely
3. **Input Validation**: All inputs are validated
4. **Rate Limiting**: Implement client-side rate limiting
5. **Error Handling**: Don't expose sensitive information

### Security Headers

- `Content-Security-Policy`
- `X-Content-Type-Options`
- `X-Frame-Options`
- `X-XSS-Protection`

## 📚 SDK Examples

### Python SDK

```python
from tyra_memory import MemoryClient

client = MemoryClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Store memory
memory = await client.store_memory(
    content="Einstein developed relativity theory",
    agent_id="claude",
    extract_entities=True
)

# Search memories
results = await client.search_memories(
    query="Einstein theory",
    agent_id="claude",
    include_analysis=True
)
```

### JavaScript SDK

```javascript
import { MemoryClient } from '@tyra/memory-client';

const client = new MemoryClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Store memory
const memory = await client.storeMemory({
  content: 'Einstein developed relativity theory',
  agentId: 'claude',
  extractEntities: true
});

// Search memories
const results = await client.searchMemories({
  query: 'Einstein theory',
  agentId: 'claude',
  includeAnalysis: true
});
```

---

🎯 **API Reference Complete!** You now have comprehensive documentation for all memory server capabilities.
