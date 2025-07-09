
# 🧠 Claude Memory Reference - Tyra MCP Memory Server Project

## 🎯 Project Mission
Transform Cole's mem0 MCP server into Tyra's advanced memory system by replacing all cloud-based components with local alternatives and integrating state-of-the-art RAG capabilities including hallucination detection, reranking, and temporal knowledge graphs.

## 🏗️ Architecture Quick Reference

### Core Components Map
```
Current (mem0) → Target (Tyra)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mem0ai library → Custom PostgreSQL client
Supabase → PostgreSQL + pgvector
OpenAI embeddings → HuggingFace local models
Basic search → Advanced RAG with reranking
No graph → Memgraph + Graphiti
No hallucination check → Confidence scoring system
No cache → Redis multi-layer cache
```

### Key Technologies
- **Databases**: PostgreSQL (pgvector), Memgraph, Redis
- **Embeddings**: intfloat/e5-large-v2 (primary), all-MiniLM-L12-v2 (fallback)
- **Frameworks**: FastMCP, FastAPI, Pydantic AI
- **RAG**: Cross-encoder reranking, hallucination detection, hybrid search

## 📁 Project Structure Overview

```
/tyra-mcp-memory-server/
├── src/
│   ├── mcp/          # MCP server components
│   ├── api/          # FastAPI routes
│   ├── core/         # Business logic
│   │   ├── memory/   # PostgreSQL operations
│   │   ├── embeddings/
│   │   ├── graph/    # Memgraph integration
│   │   ├── rag/      # Reranking & hallucination
│   │   └── cache/    # Redis caching
│   └── clients/      # Agent integration
├── config/           # YAML configurations
├── migrations/       # Database schemas
└── tests/           # Comprehensive testing
```

## 🔧 Critical Implementation Details

### 1. Database Connections
```python
# PostgreSQL with pgvector
- Pool size: 20 connections
- Vector dimensions: 1024 (primary), 384 (fallback)
- Index type: HNSW for fast similarity search

# Memgraph
- Uses GQLAlchemy for queries
- Stores temporal knowledge graphs
- Integrates with Graphiti framework

# Redis
- Multi-level caching strategy
- TTL: embeddings (24h), search (1h), rerank (30m)
```

### 2. Embedding Strategy
```python
# Primary embedder
model: "intfloat/e5-large-v2"
device: "cuda" if available else "cpu"
dimensions: 1024
batch_size: 32

# Fallback embedder
model: "sentence-transformers/all-MiniLM-L12-v2"  
device: "cpu"
dimensions: 384
batch_size: 16

# Always implement try/except with fallback
```

### 3. MCP Tool Preservation
```python
# Must maintain these interfaces:
@mcp.tool()
async def save_memory(text: str) -> SaveMemoryResult
@mcp.tool()
async def search_memories(query: str, limit: int = 10) -> SearchMemoriesResult
@mcp.tool()
async def get_all_memories() -> GetAllMemoriesResult

# Context injection pattern from original
async with mcp.context(memory_client=enhanced_client):
    # Tools can access via context
```

### 4. RAG Pipeline Flow
```
1. Query → Embedding generation (with fallback)
2. Hybrid search (0.7 vector + 0.3 keyword)
3. Graph enrichment (temporal facts)
4. Reranking (cross-encoder or vLLM)
5. Hallucination detection (grounding scores)
6. Response formatting with confidence
```

### 5. Confidence Scoring Levels
```python
confidence_levels = {
    "rock_solid": 95,     # 💪 Safe for automated actions
    "high": 80,           # 🧠 Generally reliable
    "fuzzy": 60,          # 🤔 Needs verification
    "low": 0              # ⚠️ Not confident
}

# Trading endpoint requires confidence >= 95
```

### 6. API Endpoints Pattern
```
/v1/memory/*      # Memory CRUD operations
/v1/search/*      # Various search strategies
/v1/rag/*         # Reranking and hallucination
/v1/chat/*        # Chat interfaces
/v1/graph/*       # Knowledge graph queries
/v1/admin/*       # Maintenance operations
/v1/telemetry/*   # Observability and metrics
/v1/analytics/*   # Self-learning analytics
```

## 🚨 Critical Requirements

### Must Maintain
1. **100% Local Operation** - No external API calls
2. **MCP Tool Compatibility** - Existing tools must work
3. **Agent Accessibility** - Multi-agent support (Tyra, Claude, Archon)
4. **Performance Targets** - <100ms p95 latency
5. **Safety Features** - Hallucination detection on all responses
6. **Full Observability** - OpenTelemetry tracing on all operations
7. **Self-Learning Capability** - Autonomous performance improvement

### Must Replace
1. **Remove mem0ai** - Replace with custom PostgreSQL client
2. **Remove Supabase** - Use local PostgreSQL
3. **Remove cloud embeddings** - Use HuggingFace models
4. **Remove any Langchain** - Not used in this architecture
5. **Remove cloud dependencies** - Everything runs locally

### Must Add
1. **Memgraph integration** - For knowledge graphs
2. **Hallucination scoring** - Confidence metrics
3. **Reranking system** - Better retrieval accuracy
4. **Redis caching** - Performance optimization
5. **FastAPI layer** - Modular access beyond MCP
6. **OpenTelemetry integration** - Full instrumentation
7. **Self-learning modules** - Performance analytics and auto-optimization
8. **Modular provider system** - Easy swapping of models/engines

## 📋 Implementation Checklist

### Phase Priorities
1. **Foundation First** - Set up databases and configuration
2. **Memory Core** - PostgreSQL client and embeddings
3. **RAG Features** - Reranking and hallucination detection
4. **API Layer** - MCP tools and FastAPI endpoints
5. **Observability** - OpenTelemetry integration for full tracing
6. **Self-Learning** - Autonomous improvement and monitoring
7. **Testing** - Comprehensive test coverage
8. **Optimization** - Performance tuning and caching

### Key Files to Create/Modify
```
Priority 1 (Core):
- src/core/memory/postgres_client.py
- src/core/embeddings/embedder.py
- src/mcp/server.py (replace main.py)
- config/config.yaml

Priority 2 (RAG):
- src/core/rag/reranker.py
- src/core/rag/hallucination.py
- src/core/graph/memgraph_client.py

Priority 3 (API):
- src/api/routes/memory.py
- src/api/routes/search.py
- src/clients/memory_client.py

Priority 4 (Observability):
- src/core/observability/telemetry.py
- src/core/observability/metrics.py
- src/core/observability/tracing.py

Priority 5 (Self-Learning):
- src/core/analytics/performance_tracker.py
- src/core/adaptation/config_optimizer.py
- src/core/adaptation/memory_health.py
```

## 🔍 Testing Strategy

### Unit Tests Focus
- Embedding generation with fallback
- Database connection resilience
- Hallucination scoring accuracy
- Reranking performance
- Cache hit/miss behavior

### Integration Tests Focus
- End-to-end memory storage/retrieval
- MCP tool functionality
- Multi-agent session isolation
- Graph query accuracy
- API endpoint validation

### Performance Benchmarks
- Embedding: <50ms per text
- Vector search: <30ms for top-20
- Reranking: <200ms for 20 docs
- Total query: <300ms p95

## 🛠️ Common Patterns

### Error Handling
```python
# Always use circuit breakers
from circuit_breaker import CircuitBreaker

@CircuitBreaker(failure_threshold=5, recovery_timeout=60)
async def database_operation():
    try:
        # Primary operation
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        # Fallback behavior
```

### Async Everything
```python
# All database operations must be async
async def retrieve_memories(query: str) -> List[Memory]:
    async with get_db_connection() as conn:
        results = await conn.fetch(query)
    return [Memory(**r) for r in results]
```

### Configuration Access
```python
# Use centralized config
from core.utils.config import settings

embedder_model = settings.embeddings.primary.model
hallucination_threshold = settings.rag.hallucination.threshold
```

## 🚀 Performance Optimizations

### Caching Strategy
1. **L1 Cache**: In-memory LRU for hot data
2. **L2 Cache**: Redis for distributed caching
3. **L3 Cache**: PostgreSQL materialized views

### Batch Processing
- Embeddings: Process in batches of 32
- Reranking: Batch up to 20 documents
- Graph queries: Use bulk entity lookups

### Connection Pooling
- PostgreSQL: 20 connections
- Memgraph: 10 connections  
- Redis: 50 connections
- All with retry logic

## 📝 Configuration Reference

### Essential Config Keys
```yaml
# config/config.yaml
memory:
  backend: postgres
  postgres:
    pool_size: 20
    
embeddings:
  primary:
    model: "intfloat/e5-large-v2"
  fallback:
    model: "sentence-transformers/all-MiniLM-L12-v2"
    
rag:
  hallucination:
    threshold: 75
  retrieval:
    hybrid_weight: 0.7  # vector vs keyword balance

observability:
  enabled: true
  export_target: "console"  # or "jaeger", "prometheus"
  trace_all_operations: true
  
self_learning:
  enabled: true
  analysis_interval: "1h"
  improvement_interval: "24h"
  auto_optimize: true
```

## 🎯 Success Metrics

### Technical Success
- [ ] All MCP tools working with new backend
- [ ] Query latency <100ms p95
- [ ] Hallucination detection >90% accurate
- [ ] Zero external API calls
- [ ] 99.9% uptime

### Integration Success
- [ ] Claude can use all memory features
- [ ] Tyra integration seamless
- [ ] Multi-agent support verified
- [ ] n8n webhook endpoints functional

## 🔗 Quick Links

### From Tyra Project
- Hallucination detector: `agentic-rag-knowledge-graph/agent/hallucination_detector.py`
- Reranking system: `agentic-rag-knowledge-graph/agent/reranking.py`
- PostgreSQL schema: `agentic-rag-knowledge-graph/sql/schema.sql`
- Embedder: `agentic-rag-knowledge-graph/ingestion/embedder.py`

### From mem0 Project
- MCP server: `mcp-mem0-main/src/main.py`
- Utils: `mcp-mem0-main/src/utils.py`
- Dependencies: `mcp-mem0-main/pyproject.toml`

## 💡 Pro Tips

1. **Always test fallbacks** - Primary systems will fail
2. **Monitor memory usage** - Embeddings can be large
3. **Use structured logging** - Easier debugging
4. **Validate inputs** - Never trust external data
5. **Cache aggressively** - But invalidate smartly
6. **Document assumptions** - Future you will thank you
7. **Type everything** - Use pydantic models everywhere

## 🚫 Common Pitfalls to Avoid

1. **Don't remove agent functionality** - MCP tools must work
2. **Don't use cloud services** - Everything local
3. **Don't skip fallbacks** - Always have plan B
4. **Don't ignore performance** - Test under load
5. **Don't hardcode configs** - Use YAML files
6. **Don't forget logging** - Observability is key
7. **Don't break compatibility** - Test with existing agents

## 📊 Final Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Claude/Tyra   │────▶│   MCP Server    │────▶│  FastAPI Layer  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                          │
                                ▼                          ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Core Memory   │     │   Advanced RAG  │
                        │     Engine      │     │    Features     │
                        └─────────────────┘     └─────────────────┘
                                │                          │
                        ┌───────┴───────┬─────────────────┤
                        ▼               ▼                 ▼
                ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                │ PostgreSQL  │ │  Memgraph   │ │    Redis    │
                │ + pgvector  │ │  + Graphiti │ │    Cache    │
                └─────────────┘ └─────────────┘ └─────────────┘
```

This architecture provides genius-tier memory capabilities while maintaining 100% local operation and full compatibility with existing agents.