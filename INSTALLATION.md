# 📦 Installation Guide - Tyra MCP Memory Server

## 🎯 Quick Start

For most users, this is all you need:

```bash
# Clone or download the project
cd tyra-mcp-memory-server

# Quick setup (creates venv, installs deps, sets up databases)
make quick-start

# Start development server
make dev
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11+ (3.12 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 10GB free space minimum
- **OS**: Linux (Ubuntu 22.04+), macOS 12+, Windows 11 with WSL2

### Required Services
- **PostgreSQL**: 15+ with pgvector extension
- **Redis**: 6.0+ for caching
- **Memgraph**: 2.0+ for knowledge graphs

## 🚀 Detailed Installation

### Step 1: Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install all dependencies including development tools
pip install -e ".[dev,test,docs]"

# OR install production only
pip install -e .
```

### Step 3: Database Installation

#### PostgreSQL with pgvector

**Ubuntu/Debian:**
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
sudo apt install postgresql-15-pgvector

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS (using Homebrew):**
```bash
# Install PostgreSQL
brew install postgresql@15

# Install pgvector
brew install pgvector

# Start PostgreSQL
brew services start postgresql@15
```

**From Source:**
```bash
# If pgvector is not available in your package manager
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### Redis

**Ubuntu/Debian:**
```bash
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Docker (Alternative):**
```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

#### Memgraph

**Ubuntu/Debian:**
```bash
# Add Memgraph repository
curl https://download.memgraph.com/memgraph/v2.15.1/ubuntu-22.04/memgraph_2.15.1-1_amd64.deb \
  --output memgraph.deb
sudo dpkg -i memgraph.deb
sudo apt-get install -f

# Start Memgraph
sudo systemctl start memgraph
sudo systemctl enable memgraph
```

**Docker (Recommended for Development):**
```bash
docker run -d --name memgraph \
  -p 7687:7687 \
  -p 7444:7444 \
  -p 3000:3000 \
  memgraph/memgraph-platform:latest
```

### Step 4: Database Setup

```bash
# Run database setup script
make db-setup

# OR run manually
bash scripts/db/setup_databases.sh

# Initialize schemas
make db-init
```

### Step 5: Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see CONFIGURATION.md for details)
nano .env
```

**Minimum Required Configuration:**
```env
# Database connections
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tyra_memory
POSTGRES_USER=tyra
POSTGRES_PASSWORD=your_secure_password

REDIS_HOST=localhost
REDIS_PORT=6379

MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687

# Embedding models (will be downloaded automatically)
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
```

### Step 6: Verification

```bash
# Run health check
make health-check

# Run tests
make test

# Start development server
make dev
```

If everything is working, you should see:
```
✅ PostgreSQL: Connected
✅ Redis: Connected
✅ Memgraph: Connected
✅ Embeddings: Models loaded
✅ API Server: Running on http://localhost:8000
```

## 🐳 Docker Installation (Alternative)

For a containerized setup:

```bash
# Start all services with docker-compose
make docker-compose-up

# Check logs
make docker-compose-logs

# Stop services
make docker-compose-down
```

## 🔧 Development Setup

```bash
# Install development dependencies
make install-dev

# Install pre-commit hooks
pre-commit install

# Run linting and formatting
make format
make lint

# Run tests in watch mode
make test-watch
```

## 🚨 Troubleshooting

### Common Issues

**1. Permission Denied on PostgreSQL**
```bash
# Fix PostgreSQL permissions
sudo -u postgres createuser --createdb --superuser $USER
sudo -u postgres createdb tyra_memory -O $USER
```

**2. pgvector Extension Not Found**
```bash
# Install pgvector manually
sudo -u postgres psql -c "CREATE EXTENSION vector;"
```

**3. Redis Connection Refused**
```bash
# Check Redis status
sudo systemctl status redis-server

# Start Redis if stopped
sudo systemctl start redis-server
```

**4. Memgraph Connection Failed**
```bash
# Check Memgraph status
sudo systemctl status memgraph

# For Docker users
docker logs memgraph
```

**5. Model Download Issues**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk

# Download models manually
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('intfloat/e5-large-v2')
AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
"
```

### Performance Issues

**1. Slow Embedding Generation**
- Ensure GPU is available and CUDA is installed
- Reduce batch size in configuration
- Use fallback model for faster processing

**2. High Memory Usage**
- Reduce PostgreSQL connection pool size
- Lower embedding model batch size
- Enable Redis memory optimization

**3. Slow Database Queries**
- Ensure database indexes are created
- Check PostgreSQL query performance
- Verify pgvector is properly configured

### Configuration Validation

```bash
# Validate configuration
python -c "
from src.core.utils.config import load_config
config = load_config()
print('✅ Configuration valid')
"

# Test database connections
python -c "
from src.core.utils.database import test_connections
import asyncio
asyncio.run(test_connections())
"
```

## 📚 Next Steps

After successful installation:

1. **Read Configuration Guide**: See `CONFIGURATION.md` for detailed settings
2. **Explore API**: Check `API.md` for endpoint documentation
3. **Run Examples**: Try the examples in `examples/` directory
4. **Development**: See `CONTRIBUTING.md` for development guidelines

## 🆘 Getting Help

- **Documentation**: Full docs at `docs/` or `make docs-serve`
- **Issues**: Report problems in the GitHub issues
- **Examples**: Check `examples/` directory
- **Logs**: View detailed logs with `make logs`

## 📈 Optional Optimizations

### For Production Deployment

```bash
# Install production dependencies
pip install gunicorn uvloop

# Use optimized Redis configuration
echo "maxmemory 2gb" >> /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf

# Optimize PostgreSQL for memory server
echo "shared_preload_libraries = 'vector'" >> /etc/postgresql/15/main/postgresql.conf
echo "max_connections = 200" >> /etc/postgresql/15/main/postgresql.conf
```

### For GPU Acceleration

```bash
# Install CUDA support (NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### For Large Scale Deployment

```bash
# Use external Redis cluster
# Configure multiple PostgreSQL read replicas
# Set up Memgraph cluster mode
# Enable horizontal scaling with load balancer
```

---

🎉 **Installation Complete!** You're ready to start using Tyra's advanced memory system.
