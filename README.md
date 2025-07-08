# Tyra Advanced Memory MCP Server

A sophisticated Model Context Protocol (MCP) server providing advanced memory capabilities with RAG (Retrieval-Augmented Generation), hallucination detection, and adaptive learning for AI agents.

## 🌟 Features

### 🧠 Advanced Memory System
- **Multi-Modal Storage**: Vector embeddings + temporal knowledge graphs
- **Agent Isolation**: Separate memory spaces for Tyra, Claude, Archon
- **Intelligent Chunking**: Automatic content segmentation with overlap
- **Entity Extraction**: Automated relationship mapping and graph building

### 🔍 Sophisticated Search
- **Hybrid Search**: Combines vector similarity and graph traversal
- **Advanced Reranking**: Cross-encoder models for relevance scoring
- **Confidence Scoring**: Multi-level confidence assessment (💪 Rock Solid, 🧠 High, 🤔 Fuzzy, ⚠️ Low)
- **Hallucination Detection**: Real-time grounding analysis with evidence collection

### 📊 Performance Analytics
- **Real-Time Monitoring**: Response time, accuracy, memory usage tracking
- **Trend Analysis**: Automated performance trend detection
- **Smart Alerts**: Configurable warning and critical thresholds
- **Optimization Recommendations**: AI-generated improvement suggestions

### 🎯 Adaptive Learning
- **Self-Optimization**: Automated parameter tuning based on performance
- **A/B Testing**: Systematic experimentation with rollback protection
- **Learning Insights**: Pattern recognition from successful configurations
- **Multi-Strategy Optimization**: Gradient descent, Bayesian optimization, random search

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- Redis (for caching)
- Memgraph (for knowledge graphs)

### Automated Setup

```bash
# Install system dependencies
./scripts/install_dependencies.sh --docker --cuda

# Run complete setup
./scripts/setup.sh --env development

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

2. **Database Setup**
   ```bash
   # Start databases with Docker
   docker-compose -f docker-compose.dev.yml up -d

   # Or configure your own PostgreSQL, Redis, Memgraph instances
   ```

3. **Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

4. **Start Server**
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

#### 📝 `store_memory`
Store information with automatic entity extraction and metadata enrichment.

```json
{
  "tool": "store_memory",
  "content": "User prefers morning trading sessions and uses technical analysis",
  "agent_id": "tyra",
  "extract_entities": true,
  "metadata": {"category": "trading_preferences"}
}
```

#### 🔍 `search_memory`
Advanced search with confidence scoring and hallucination analysis.

```json
{
  "tool": "search_memory",
  "query": "What are the user's trading preferences?",
  "search_type": "hybrid",
  "min_confidence": 0.7,
  "include_analysis": true
}
```

#### 🛡️ `analyze_response`
Analyze any response for hallucinations and confidence scoring.

```json
{
  "tool": "analyze_response",
  "response": "Based on your history, you prefer swing trading",
  "query": "What's my trading style?",
  "retrieved_memories": [...]
}
```

#### 📊 `get_memory_stats`
Comprehensive system statistics and health metrics.

```json
{
  "tool": "get_memory_stats",
  "include_performance": true,
  "include_recommendations": true
}
```

#### 🎯 `get_learning_insights`
Access adaptive learning insights and optimization recommendations.

```json
{
  "tool": "get_learning_insights",
  "category": "parameter_optimization",
  "days": 7
}
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Server    │    │ Memory Manager  │    │ Analytics       │
│                 │    │                 │    │                 │
│ • Tool Handlers │ ── │ • Vector Store  │ ── │ • Performance   │
│ • Error Mgmt    │    │ • Graph Engine  │    │ • Trend Analysis│
│ • Monitoring    │    │ • Reranker      │    │ • Alerts        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Hallucination   │    │ Embedding       │    │ Learning Engine │
│ Detector        │    │ Provider        │    │                 │
│                 │    │                 │    │ • Experiments   │
│ • Confidence    │    │ • HuggingFace   │    │ • Optimization  │
│ • Grounding     │    │ • GPU/CPU Auto  │    │ • Insights      │
│ • Evidence      │    │ • Caching       │    │ • Rollback      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Memory Storage**: Content → Embedding → Vector Store + Graph Engine
2. **Memory Search**: Query → Embedding → Hybrid Search → Reranking → Confidence Analysis
3. **Analytics**: Operations → Metrics → Trend Analysis → Recommendations
4. **Adaptation**: Performance Data → Experiment Planning → Parameter Tuning → Rollback Protection

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

### Advanced Configuration

Detailed configuration is available in `config/config.yaml`:

- **Memory Settings**: Vector dimensions, chunking, indexing
- **RAG Configuration**: Search strategies, reranking, hallucination thresholds
- **Analytics**: Alert thresholds, retention policies, trend analysis
- **Learning**: Optimization strategies, experiment parameters
- **Agent Settings**: Per-agent memory isolation and preferences

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
```bash
# Build production image
docker build -t tyra-memory-server .

# Run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### Systemd Service
```bash
# Install system service
sudo cp scripts/tyra-memory.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tyra-memory
```

### Performance Tuning
- **GPU Optimization**: CUDA-enabled PyTorch for embeddings and reranking
- **Database Tuning**: Optimized PostgreSQL settings for vector operations
- **Caching Strategy**: Multi-level caching with Redis
- **Connection Pooling**: Efficient database connection management

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
- [Architecture Guide](docs/architecture.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### Community
- **Issues**: [GitHub Issues](https://github.com/tyra-ai/memory-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tyra-ai/memory-server/discussions)
- **Discord**: [Tyra AI Community](https://discord.gg/tyra-ai)

### Commercial Support
For enterprise support, custom integrations, and professional services, contact: support@tyra-ai.com

---

**Built with ❤️ by the Tyra AI Team**

*Empowering AI agents with human-like memory capabilities*
