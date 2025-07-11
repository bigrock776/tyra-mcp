# Scripts Directory

This directory contains utility scripts for the Tyra MCP Memory Server project.

## 🧪 Model Testing Scripts

### Required Model Verification

**⚠️ CRITICAL**: These scripts verify that manually downloaded models are working correctly.

#### `test_embedding_model.py`
Tests local embedding models to ensure they're properly installed.
```bash
python scripts/test_embedding_model.py
```

**Tests:**
- Primary embedding model: `intfloat/e5-large-v2`
- Fallback embedding model: `sentence-transformers/all-MiniLM-L12-v2`
- Model loading from local paths
- Embedding generation functionality
- Dimension verification

#### `test_cross_encoder.py`
Tests local cross-encoder models for reranking functionality.
```bash
python scripts/test_cross_encoder.py
```

**Tests:**
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Model loading from local paths
- Reranking functionality
- Score generation and normalization

#### `test_model_pipeline.py`
Tests the complete RAG pipeline with all models integrated.
```bash
python scripts/test_model_pipeline.py
```

**Tests:**
- Complete embedding → similarity search → reranking pipeline
- Model compatibility and integration
- Performance benchmarking
- End-to-end functionality verification

### Expected Model Directory Structure

```
./models/
├── embeddings/
│   ├── e5-large-v2/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer.json
│   └── all-MiniLM-L12-v2/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer.json
└── cross-encoders/
    └── ms-marco-MiniLM-L-6-v2/
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer.json
```

### Download Commands Reminder

```bash
# Install prerequisites
pip install huggingface-hub
git lfs install

# Create directories
mkdir -p ./models/embeddings ./models/cross-encoders

# Download models
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

## 🔧 Other Utility Scripts

## Available Scripts

### `add_provider.py`

Interactive wizard for adding new providers to the system.

**Usage:**
```bash
# Run interactive wizard
python scripts/add_provider.py

# Or with specific project root
python scripts/add_provider.py --project-root /path/to/project
```

**Features:**
- Interactive provider type selection
- Automatic boilerplate code generation
- Configuration file updates
- Test template creation
- Interface validation
- Support for all provider types (embeddings, vector_stores, graph_engines, rerankers, etc.)

**Requirements:**
- `inquirer` package for interactive prompts
- `pyyaml` for configuration file handling

**What it creates:**
1. Provider implementation file with boilerplate code
2. Updates `config/providers.yaml` with new provider configuration
3. Creates test template in `tests/unit/providers/`
4. Updates provider directory `__init__.py` files

### Database Scripts

- `setup_databases.sh` - Set up all required databases
- `test_databases.sh` - Test database connections
- `backup_databases.sh` - Backup all databases

### Installation Scripts

- `install_dependencies.sh` - Install system dependencies
- `install_pgvector.sh` - Install PostgreSQL with pgvector
- `install_memgraph.sh` - Install Memgraph
- `install_redis.sh` - Install Redis

### Testing Scripts

- `run_mcp_tests.py` - Run MCP integration tests

## Development Workflow

1. **Adding a new provider:**
   ```bash
   python scripts/add_provider.py
   ```

2. **Setting up development environment:**
   ```bash
   bash scripts/setup_databases.sh
   bash scripts/install_dependencies.sh
   ```

3. **Running tests:**
   ```bash
   python scripts/run_mcp_tests.py
   ```

4. **Backup before major changes:**
   ```bash
   bash scripts/backup_databases.sh
   ```

## Script Dependencies

Most scripts require the project to be properly set up with dependencies installed. Install dependencies with:

```bash
pip install -r requirements.txt
pip install inquirer  # For add_provider.py
```