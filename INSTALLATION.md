# 📦 Tyra MCP Memory Server - Complete Installation Guide

**Version**: 1.0.0  
**Priority**: Local Installation → Docker Deployment  
**Estimated Time**: 30-60 minutes  

## 🎯 Quick Start (5 Minutes)

**For experienced users wanting the fastest setup:**

```bash
# Clone repository
git clone <repository-url>
cd tyra-mcp-memory-server

# One-command setup (handles everything)
./setup.sh --env development --quick

# Start server
source venv/bin/activate && python main.py
```

**Verification**: Visit `http://localhost:8000/health` - you should see `{"status": "healthy"}`

---

## 📋 Prerequisites & System Requirements

### 🖥️ **System Requirements**

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **OS** | Ubuntu 20.04, macOS 12, Windows 11+WSL2 | Ubuntu 24.04 LTS, macOS 14+ | Windows requires WSL2 |
| **Python** | 3.11.0 | 3.12.2+ | Required for async features |
| **RAM** | 8GB | 16GB+ | For embedding models |
| **Storage** | 10GB free | 50GB+ | For models and data |
| **CPU** | 4 cores | 8+ cores | Better embedding performance |
| **GPU** | None | NVIDIA RTX 3060+ | 10x faster embeddings |

### 🛠️ **Required Services**

| Service | Version | Purpose | Local Install Priority |
|---------|---------|---------|----------------------|
| **PostgreSQL** | 15+ | Vector storage with pgvector | ⭐ **ESSENTIAL** |
| **Redis** | 6.0+ | Multi-level caching | ⭐ **ESSENTIAL** |
| **Memgraph** | 2.0+ | Knowledge graphs | ⭐ **ESSENTIAL** |
| **Crawl4AI** | Latest | Web scraping for n8n | 🔧 **OPTIONAL** |

### ⚙️ **Development Tools** (Optional but Recommended)

```bash
# Essential tools for development
sudo apt update && sudo apt install -y \
  git curl wget unzip \
  build-essential pkg-config \
  postgresql-client redis-tools \
  python3-dev python3-pip python3-venv \
  nginx certbot
```

---

## 🚀 **Method 1: Local Installation (PRIORITY)**

**This is the recommended method for development and production.**

### **Step 1: Environment Preparation**

#### **1.1 Python Environment Setup**

```bash
# Check Python version (must be 3.11+)
python3 --version
# If not 3.11+, install it:

# Ubuntu 24.04 LTS (has Python 3.12 by default)
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3.12-pip

# Ubuntu 22.04/20.04 (requires deadsnakes PPA)
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# macOS (using Homebrew)
brew install python@3.12

# Verify installation
python3.12 --version
```

#### **1.2 Create Project Directory**

```bash
# Create dedicated directory
sudo mkdir -p /opt/tyra-memory-server
sudo chown $USER:$USER /opt/tyra-memory-server
cd /opt/tyra-memory-server

# Clone repository
git clone <repository-url> .

# Create virtual environment with specific Python version
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify virtual environment
which python
python --version  # Should show 3.12+
```

#### **1.3 Upgrade Python Package Manager**

```bash
# Upgrade pip and install essential tools
pip install --upgrade pip setuptools wheel

# Install Poetry for dependency management (recommended)
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Verify Poetry installation
poetry --version
```

### **Step 2: Database Services Installation**

#### **2.1 PostgreSQL with pgvector Extension**

##### **Ubuntu Installation (24.04/22.04/20.04)**

```bash
# Update package list
sudo apt update

# Install PostgreSQL (Ubuntu 24.04 has PostgreSQL 16 by default)
# For Ubuntu 24.04
sudo apt install -y postgresql postgresql-contrib postgresql-client

# For Ubuntu 22.04/20.04 (install specific version)
sudo apt install -y postgresql-15 postgresql-contrib-15 postgresql-client-15

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Verify PostgreSQL version
sudo -u postgres psql -c "SELECT version();"

# Check PostgreSQL status
sudo systemctl status postgresql
```

##### **pgvector Extension Installation**

```bash
# Method 1: Package installation (Ubuntu 24.04/22.04)
# Ubuntu 24.04 (PostgreSQL 16)
sudo apt install -y postgresql-16-pgvector

# Ubuntu 22.04 (PostgreSQL 15)
sudo apt install -y postgresql-15-pgvector

# Ubuntu 20.04 (requires manual installation)
# Method 2: Build from source (latest version - recommended for all)
sudo apt install -y git build-essential postgresql-server-dev-all

# Clone and build pgvector
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector

# Build and install
make
sudo make install

# Clean up
cd .. && rm -rf pgvector

# Verify installation
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;" template1
```

##### **macOS Installation**

```bash
# Install PostgreSQL via Homebrew
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Install pgvector
brew install pgvector

# Verify installation
brew services list | grep postgresql
```

##### **Database Configuration**

```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE tyra_memory;
CREATE USER tyra WITH ENCRYPTED PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE tyra_memory TO tyra;
ALTER USER tyra CREATEDB;

# Enable pgvector extension
\c tyra_memory
CREATE EXTENSION vector;
CREATE EXTENSION pg_trgm;  -- For text search
CREATE EXTENSION btree_gin;  -- For indexing

# Verify extensions
\dx

# Exit PostgreSQL
\q

# Test connection
psql -h localhost -U tyra -d tyra_memory -c "SELECT version();"
```

##### **PostgreSQL Performance Tuning**

```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/15/main/postgresql.conf

# Add/modify these settings for better performance:
```

```ini
# Memory settings
shared_buffers = 512MB              # 25% of RAM
effective_cache_size = 2GB          # 50-75% of RAM
work_mem = 8MB                      # For sorting operations
maintenance_work_mem = 128MB        # For VACUUM, CREATE INDEX

# Connection settings
max_connections = 200
listen_addresses = 'localhost'

# WAL settings
wal_buffers = 16MB
checkpoint_segments = 32
checkpoint_completion_target = 0.7

# Query planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_statement = 'all'              # For debugging (disable in production)
log_duration = on
log_checkpoints = on
```

```bash
# Restart PostgreSQL to apply changes
sudo systemctl restart postgresql

# Verify settings
sudo -u postgres psql -c "SHOW shared_buffers;"
```

#### **2.2 Redis Installation & Configuration**

##### **Ubuntu/Debian**

```bash
# Install Redis
sudo apt install -y redis-server

# Configure Redis for production
sudo nano /etc/redis/redis.conf
```

```ini
# Essential Redis settings for Tyra
bind 127.0.0.1
port 6379
timeout 300
tcp-keepalive 300

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence (choose one)
save 900 1      # Save if at least 1 key changed in 900 seconds
save 300 10     # Save if at least 10 keys changed in 300 seconds
save 60 10000   # Save if at least 10000 keys changed in 60 seconds

# Performance
tcp-backlog 511
timeout 0
keepalive 300
```

```bash
# Start and enable Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis connection
redis-cli ping  # Should return "PONG"

# Test basic operations
redis-cli set test "Hello Tyra"
redis-cli get test
redis-cli del test
```

##### **macOS Redis Installation**

```bash
# Install Redis
brew install redis

# Start Redis service
brew services start redis

# Test connection
redis-cli ping
```

#### **2.3 Memgraph Installation & Setup**

##### **Ubuntu Installation (24.04/22.04/20.04)**

```bash
# Method 1: Official Repository (Recommended)
# Add Memgraph repository
curl https://download.memgraph.com/memgraph/v2.12.1/ubuntu-22.04/memgraph_2.12.1-1_amd64.deb.gpg.key | sudo apt-key add -
echo "deb https://download.memgraph.com/memgraph/v2.12.1/ubuntu-22.04/ stable main" | sudo tee /etc/apt/sources.list.d/memgraph.list

# Update and install
sudo apt update
sudo apt install -y memgraph

# Method 2: Direct Download (Ubuntu 24.04/22.04)
# Ubuntu 24.04
curl -O https://download.memgraph.com/memgraph/v2.13.0/ubuntu-24.04/memgraph_2.13.0-1_amd64.deb
sudo dpkg -i memgraph_2.13.0-1_amd64.deb

# Ubuntu 22.04
curl -O https://download.memgraph.com/memgraph/v2.12.1/ubuntu-22.04/memgraph_2.12.1-1_amd64.deb
sudo dpkg -i memgraph_2.12.1-1_amd64.deb

# Ubuntu 20.04
curl -O https://download.memgraph.com/memgraph/v2.12.1/ubuntu-20.04/memgraph_2.12.1-1_amd64.deb
sudo dpkg -i memgraph_2.12.1-1_amd64.deb

# Fix any dependency issues
sudo apt-get install -f

# Install Memgraph client tools
sudo apt install -y memgraph-client

# For manual client installation:
wget https://download.memgraph.com/memgraph-lab/v2.12.1/ubuntu-22.04/memgraph-lab_2.12.1-1_amd64.deb
sudo dpkg -i memgraph-lab_2.12.1-1_amd64.deb

# Start and enable Memgraph
sudo systemctl start memgraph
sudo systemctl enable memgraph

# Check status and version
sudo systemctl status memgraph
echo "RETURN 'Memgraph ' + version() AS info;" | mgconsole --host 127.0.0.1 --port 7687
```

##### **macOS Memgraph Installation**

```bash
# Install Memgraph via Homebrew
brew tap memgraph/tap
brew install memgraph

# Start Memgraph
brew services start memgraph

# Install client tools
brew install memgraph-client
```

##### **Memgraph Verification & Initial Setup**

```bash
# Test connection with mgconsole
echo "RETURN 'Hello Memgraph!' AS message;" | mgconsole --host 127.0.0.1 --port 7687

# Alternative: Test with bolt protocol
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687')
with driver.session() as session:
    result = session.run('RETURN \"Hello Memgraph!\" AS message')
    print(result.single()['message'])
driver.close()
"

# Run initial schema setup
./scripts/init_memgraph.sh
```

#### **2.4 Crawl4AI Installation (Optional - for n8n Web Scraping)**

Crawl4AI is required for the n8n web scraping workflows included in the examples.

##### **Ubuntu Installation (24.04/22.04/20.04)**

```bash
# Install system dependencies for Crawl4AI
sudo apt update
sudo apt install -y \
  chromium-browser \
  chromium-chromedriver \
  firefox \
  wget \
  unzip \
  curl \
  ca-certificates \
  fonts-liberation \
  libasound2 \
  libatk-bridge2.0-0 \
  libatk1.0-0 \
  libatspi2.0-0 \
  libdrm2 \
  libgtk-3-0 \
  libnspr4 \
  libnss3 \
  libxcomposite1 \
  libxdamage1 \
  libxrandr2 \
  xdg-utils

# Install Chrome (recommended for better compatibility)
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt update
sudo apt install -y google-chrome-stable

# Install Crawl4AI via pip (after activating virtual environment)
source venv/bin/activate
pip install crawl4ai[all]

# Install additional browser drivers
crawl4ai-setup

# Verify installation
python -c "
import crawl4ai
from crawl4ai import WebCrawler
print('✅ Crawl4AI installed successfully')
print(f'Version: {crawl4ai.__version__}')
"

# Test basic functionality
python -c "
from crawl4ai import WebCrawler
import asyncio

async def test_crawl():
    crawler = WebCrawler()
    await crawler.astart()
    result = await crawler.arun(url='https://httpbin.org/get')
    await crawler.aclose()
    print('✅ Crawl4AI test successful')
    return result.markdown[:100] + '...'

print(asyncio.run(test_crawl()))
"
```

##### **macOS Crawl4AI Installation**

```bash
# Install Homebrew dependencies
brew install --cask google-chrome
brew install chromium

# Install Crawl4AI
source venv/bin/activate
pip install crawl4ai[all]

# Setup browser drivers
crawl4ai-setup

# Verify installation (same as Ubuntu)
python -c "import crawl4ai; print(f'✅ Crawl4AI {crawl4ai.__version__} ready')"
```

##### **Docker Crawl4AI Setup (Alternative)**

```bash
# Use Crawl4AI Docker image for isolated operation
docker pull unclecode/crawl4ai:latest

# Test Docker version
docker run --rm unclecode/crawl4ai:latest \
  python -c "from crawl4ai import WebCrawler; print('✅ Crawl4AI Docker ready')"

# For integration with Tyra, configure n8n to use Docker endpoints
# See examples/n8n-workflows/CRAWL4AI_SETUP.md for details
```

##### **Crawl4AI Configuration for n8n**

```bash
# Create Crawl4AI configuration directory
mkdir -p config/crawl4ai

# Create configuration file
cat > config/crawl4ai/config.yaml << EOF
crawl4ai:
  # Browser settings
  browser: "chrome"  # chrome, firefox, chromium
  headless: true
  
  # Performance settings
  max_concurrent: 5
  timeout: 30
  
  # Cache settings
  cache_enabled: true
  cache_dir: "./cache/crawl4ai"
  
  # API settings (for n8n integration)
  api:
    host: "localhost"
    port: 8080
    
  # Security settings
  allowed_domains: []  # Empty = allow all, or specify domains
  blocked_domains: []
  
  # Content extraction
  extraction:
    remove_scripts: true
    remove_styles: true
    clean_html: true
    markdown_conversion: true
EOF

# Start Crawl4AI API server (optional - for n8n HTTP integration)
source venv/bin/activate
python -m crawl4ai.server --host localhost --port 8080 &

# Verify API server
curl http://localhost:8080/health
```

### **Step 3: Python Dependencies Installation**

#### **3.1 Install Core Dependencies**

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Method A: Using Poetry (Recommended)
poetry install --with dev,test

# Method B: Using pip requirements
pip install -r requirements.txt

# Method C: Install individual critical packages
pip install \
  fastapi[all]==0.104.1 \
  uvicorn[standard]==0.24.0 \
  asyncpg==0.29.0 \
  redis[hiredis]==5.0.1 \
  sentence-transformers==2.2.2 \
  torch>=2.0.0 \
  transformers>=4.35.0 \
  numpy>=1.24.0 \
  pydantic>=2.5.0 \
  opentelemetry-api>=1.21.0

# Optional: Install Crawl4AI for n8n web scraping workflows
pip install crawl4ai[all]

# Verify critical imports
python -c "
import asyncpg, redis, sentence_transformers, torch
print('✅ All critical packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Verify optional Crawl4AI installation
python -c "
try:
    import crawl4ai
    print(f'✅ Crawl4AI {crawl4ai.__version__} available for n8n workflows')
except ImportError:
    print('⚠️  Crawl4AI not installed (optional - for n8n web scraping)')
"
```

#### **3.2 GPU Support Setup (Optional but Recommended)**

```bash
# Check for NVIDIA GPU
nvidia-smi

# If GPU detected, install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU setup
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
"
```

### **Step 4: Configuration Setup**

#### **4.1 Environment Configuration**

```bash
# Create environment file from template
cp .env.example .env

# Generate secure secret key
python -c "import secrets; print(f'SECRET_KEY={secrets.token_hex(32)}')" >> .env

# Edit environment configuration
nano .env
```

**Essential `.env` configuration:**

```env
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database Connections
DATABASE_URL=postgresql://tyra:secure_password_here@localhost:5432/tyra_memory
REDIS_URL=redis://localhost:6379/0
MEMGRAPH_URL=bolt://localhost:7687

# Database Credentials (individual)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tyra_memory
POSTGRES_USER=tyra
POSTGRES_PASSWORD=secure_password_here

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687
MEMGRAPH_USER=
MEMGRAPH_PASSWORD=

# Embedding Configuration
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=auto  # auto, cuda, cpu
EMBEDDINGS_BATCH_SIZE=32

# Server Configuration
API_HOST=localhost
API_PORT=8000
MCP_TRANSPORT=stdio

# Security
SECRET_KEY=your_generated_secret_key_here
API_KEY=optional_api_key_for_authentication

# Cache Settings (seconds)
CACHE_TTL_EMBEDDINGS=86400  # 24 hours
CACHE_TTL_SEARCH=3600       # 1 hour  
CACHE_TTL_RERANK=1800       # 30 minutes

# Observability
TELEMETRY_ENABLED=true
TELEMETRY_ENDPOINT=console
METRICS_ENABLED=true

# Performance
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT=30
MEMORY_LIMIT_MB=4096
```

#### **4.2 YAML Configuration Validation**

```bash
# Validate all configuration files
python scripts/validate_config.py

# Check specific configurations
python scripts/validate_config.py --config-file config/config.yaml
python scripts/validate_config.py --check-references

# Test configuration loading
python -c "
from src.core.utils.config import get_settings
settings = get_settings()
print('✅ Configuration loaded successfully')
print(f'Environment: {settings.environment}')
print(f'Database: {settings.databases.postgresql.host}')
"
```

### **Step 5: Database Initialization**

#### **5.1 Run Database Migrations**

```bash
# Initialize PostgreSQL schema
python -m src.migrations.run_migrations

# Alternative: Use migration scripts
./scripts/deploy/migrate.sh

# Initialize Memgraph schema
./scripts/init_memgraph.sh

# Verify database setup
python -c "
import asyncio
from src.core.memory.postgres_client import PostgresClient

async def test_db():
    client = PostgresClient()
    await client.initialize()
    health = await client.health_check()
    print(f'✅ PostgreSQL health: {health}')
    await client.close()

asyncio.run(test_db())
"
```

#### **5.2 Download and Cache Models**

```bash
# Pre-download embedding models (optional but recommended)
python -c "
from sentence_transformers import SentenceTransformer
import os

models = [
    'intfloat/e5-large-v2',
    'sentence-transformers/all-MiniLM-L12-v2'
]

for model_name in models:
    print(f'Downloading {model_name}...')
    model = SentenceTransformer(model_name)
    print(f'✅ {model_name} ready')

print('✅ All models downloaded and cached')
"
```

### **Step 6: Start Services & Verification**

#### **6.1 Start Tyra MCP Memory Server**

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Start MCP server (primary mode)
python main.py

# Alternative: Start API server only
python -m uvicorn src.api.app:app --host localhost --port 8000 --reload

# Start with debug mode
python main.py --debug

# Start in background
nohup python main.py > logs/server.log 2>&1 &
```

#### **6.2 Health Check & Verification**

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Test MCP tools
python -c "
import asyncio
from src.mcp.server import MCPServer

async def test_mcp():
    server = MCPServer()
    await server.initialize()
    print('✅ MCP Server initialized successfully')

asyncio.run(test_mcp())
"

# Test memory storage
curl -X POST http://localhost:8000/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{"text": "Test memory storage", "agent_id": "test-agent"}'

# Test memory search
curl -X POST http://localhost:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "agent_id": "test-agent"}'
```

#### **6.3 Comprehensive Test Suite**

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v                 # Unit tests
pytest tests/integration/ -v          # Integration tests
pytest tests/test_mcp_integration.py -v  # MCP tests

# Run trading safety tests (critical)
pytest tests/test_mcp_trading_safety.py -v

# Performance tests
pytest tests/performance/ -v

# Test with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

---

## 🐳 **Method 2: Docker Installation (Alternative)**

**Use when you want isolated deployment or easier service management.**

### **2.1 Docker Prerequisites**

```bash
# Install Docker and Docker Compose (Ubuntu)
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
docker compose version
```

### **2.2 Docker Deployment Options**

#### **Development with Docker**

```bash
# Clone repository
git clone <repository-url>
cd tyra-mcp-memory-server

# Create environment file
cp .env.example .env
# Edit .env with your settings

# Start development environment
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# View logs
docker compose logs -f

# Access containers
docker compose exec tyra-memory-server bash
```

#### **Production with Docker**

```bash
# Production deployment
docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d

# With monitoring stack
docker compose --profile monitoring up -d

# Check status
docker compose ps
```

#### **Individual Service Containers**

```bash
# Run only databases via Docker
docker compose -f docker/docker-compose.yml up -d postgres redis memgraph

# Run Tyra locally, databases in Docker
source venv/bin/activate
export DATABASE_URL=postgresql://tyra:password@localhost:5432/tyra_memory
python main.py
```

---

## 🎯 **Method 3: Automated Setup Script (Fastest)**

**For users who want zero configuration.**

### **3.1 Quick Setup Options**

```bash
# Full automated setup
./setup.sh --env development --auto-install

# Custom setup with options
./setup.sh \
  --env production \
  --domain yourdomain.com \
  --ssl \
  --monitoring \
  --gpu-support

# Minimal setup (databases only)
./setup.sh --minimal

# Test environment
./setup.sh --env testing --no-cache
```

### **3.2 Setup Script Options**

| Option | Description | Example |
|--------|-------------|---------|
| `--env` | Environment type | `development`, `production`, `testing` |
| `--domain` | Domain name for SSL | `yourdomain.com` |
| `--ssl` | Enable SSL/TLS | Auto-configured with Let's Encrypt |
| `--monitoring` | Enable monitoring stack | Prometheus + Grafana |
| `--gpu-support` | Install CUDA packages | For faster embeddings |
| `--minimal` | Databases only | Skip Python setup |
| `--auto-install` | No prompts | Fully automated |

---

## 🔧 **Configuration Deep Dive**

### **4.1 Performance Tuning**

#### **Ubuntu 24.04 LTS Optimizations**

```bash
# Ubuntu 24.04 comes with optimized defaults
# Enable performance governor for better CPU performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optimize PostgreSQL 16 for Ubuntu 24.04
sudo nano /etc/postgresql/16/main/postgresql.conf
# Add these optimizations:
# shared_buffers = 512MB
# effective_cache_size = 2GB
# work_mem = 8MB
# maintenance_work_mem = 128MB
# max_wal_size = 2GB
# checkpoint_completion_target = 0.7

# For Ubuntu 24.04 with systemd 255+, enable memory optimization
sudo systemctl edit tyra-memory-server
# Add:
# [Service]
# MemoryHigh=4G
# MemoryMax=6G
# TasksMax=1000
```

#### **Memory Optimization**

```bash
# For systems with limited RAM (8GB)
export EMBEDDINGS_DEVICE=cpu
export EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
export EMBEDDINGS_BATCH_SIZE=16
export MAX_CONCURRENT_REQUESTS=20

# For Ubuntu 24.04 with 16GB+ RAM
export EMBEDDINGS_DEVICE=auto
export EMBEDDINGS_BATCH_SIZE=64
export MAX_CONCURRENT_REQUESTS=100
export POSTGRES_SHARED_BUFFERS=1GB
export REDIS_MAXMEMORY=4gb
```

#### **GPU Acceleration**

```bash
# For systems with NVIDIA GPU
export EMBEDDINGS_DEVICE=cuda
export EMBEDDINGS_BATCH_SIZE=64
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### **Production Tuning**

```bash
# High-performance production settings
export MAX_CONCURRENT_REQUESTS=100
export CONNECTION_POOL_SIZE=50
export CACHE_TTL_EMBEDDINGS=86400
export ENABLE_BATCH_PROCESSING=true
```

### **4.2 Security Configuration**

```bash
# Generate secure secrets
python -c "
import secrets
print(f'SECRET_KEY={secrets.token_hex(32)}')
print(f'API_KEY={secrets.token_urlsafe(32)}')
print(f'JWT_SECRET={secrets.token_hex(32)}')
"

# File permissions
chmod 600 .env
chmod 700 config/local/
chmod 600 config/local/*
```

### **4.3 Monitoring & Observability**

```bash
# Enable comprehensive monitoring
export TELEMETRY_ENABLED=true
export TELEMETRY_ENDPOINT=jaeger
export METRICS_ENABLED=true
export METRICS_ENDPOINT=prometheus
export LOG_LEVEL=INFO
export AUDIT_LOGGING=true
```

---

## 🧪 **Testing & Validation**

### **5.1 Comprehensive Testing**

```bash
# Database connectivity tests
python scripts/test_databases.sh

# Full integration test
python -m pytest tests/integration/ -v --tb=short

# Performance benchmarks
python scripts/benchmark_performance.py

# Trading safety validation (CRITICAL)
python -m pytest tests/test_mcp_trading_safety.py -v
```

### **5.2 Load Testing**

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load/locustfile.py --host http://localhost:8000

# Simple load test
for i in {1..100}; do
  curl -s http://localhost:8000/health > /dev/null &
done
wait
echo "✅ Load test completed"
```

---

## 🛠️ **Troubleshooting Guide**

### **6.1 Common Issues & Solutions**

#### **Issue: Python Version Conflicts**

```bash
# Problem: Wrong Python version
python --version  # Shows 3.9 instead of 3.11+

# Solution: Use specific Python version
sudo apt install python3.12-venv
python3.12 -m venv venv
source venv/bin/activate
python --version  # Should now show 3.12
```

#### **Issue: Database Connection Failures**

```bash
# Problem: PostgreSQL connection refused
# Solution: Check and restart services

sudo systemctl status postgresql
sudo systemctl restart postgresql

# Test connection
pg_isready -h localhost -p 5432
psql -h localhost -U tyra -d tyra_memory -c "SELECT 1;"
```

#### **Issue: Port Already in Use**

```bash
# Problem: Port 8000 already in use
# Solution: Find and kill conflicting process

sudo lsof -i :8000
sudo kill -9 $(sudo lsof -t -i:8000)

# Or use different port
export API_PORT=8001
python main.py
```

#### **Issue: GPU Memory Errors**

```bash
# Problem: CUDA out of memory
# Solution: Reduce batch sizes or use CPU

export EMBEDDINGS_DEVICE=cpu
export EMBEDDINGS_BATCH_SIZE=16
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

#### **Issue: Permission Denied**

```bash
# Problem: Permission errors
# Solution: Fix file permissions

sudo chown -R $USER:$USER .
chmod +x setup.sh
chmod +x scripts/deploy/*.sh
chmod 600 .env
```

#### **Issue: PostgreSQL pgvector Extension Missing**

```bash
# Problem: pgvector extension not available
# Solution: Verify installation and create extension

# Check if pgvector is installed
sudo -u postgres psql -c "SELECT * FROM pg_available_extensions WHERE name='vector';"

# If not found, reinstall pgvector
# Ubuntu 24.04
sudo apt remove postgresql-16-pgvector
sudo apt install postgresql-16-pgvector

# Ubuntu 22.04/20.04 - rebuild from source
sudo apt install -y git build-essential postgresql-server-dev-all
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install && cd .. && rm -rf pgvector

# Create extension in database
sudo -u postgres psql tyra_memory -c "CREATE EXTENSION vector;"
```

#### **Issue: Memgraph Connection Refused**

```bash
# Problem: Cannot connect to Memgraph
# Solution: Check service and configuration

# Check if Memgraph is running
sudo systemctl status memgraph

# Check port availability
sudo netstat -tulpn | grep :7687

# Try restarting Memgraph
sudo systemctl restart memgraph

# Test connection with different methods
mgconsole --host localhost --port 7687
echo "RETURN 1;" | mgconsole --host 127.0.0.1 --port 7687

# Check Memgraph logs
sudo journalctl -u memgraph -f
```

#### **Issue: Crawl4AI Browser Driver Problems**

```bash
# Problem: Browser drivers not working
# Solution: Reinstall browser drivers and dependencies

# Update system packages (Ubuntu 24.04)
sudo apt update && sudo apt upgrade

# Reinstall Chrome
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo apt update
sudo apt install --reinstall google-chrome-stable

# Reinstall Crawl4AI and setup drivers
pip uninstall crawl4ai
pip install crawl4ai[all]
crawl4ai-setup

# Test with specific browser
python -c "
from crawl4ai import WebCrawler
import asyncio

async def test():
    crawler = WebCrawler(browser_type='chrome', headless=True)
    await crawler.astart()
    print('✅ Chrome driver working')
    await crawler.aclose()

asyncio.run(test())
"

# Alternative: Use Docker for Crawl4AI
docker run --rm unclecode/crawl4ai:latest python -c "print('✅ Docker Crawl4AI working')"
```

#### **Issue: Ubuntu 24.04 Package Conflicts**

```bash
# Problem: Package version conflicts on Ubuntu 24.04
# Solution: Use virtual environment and specific versions

# Create isolated environment
python3.12 -m venv --clear venv
source venv/bin/activate

# Install with specific versions for Ubuntu 24.04
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir \
  'asyncpg>=0.29.0' \
  'redis[hiredis]>=5.0.1' \
  'fastapi>=0.104.0' \
  'uvicorn[standard]>=0.24.0'

# If still having issues, use conda instead
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n tyra python=3.12
conda activate tyra
```

### **6.2 Debug Mode**

```bash
# Enable comprehensive debugging
export DEBUG=true
export LOG_LEVEL=DEBUG
export SQLALCHEMY_ECHO=true

# Run with debug
python main.py --debug --verbose

# Monitor logs in real-time
tail -f logs/memory-server.log
tail -f logs/debug.log
tail -f logs/performance.log
```

### **6.3 Health Diagnostics**

```bash
# System health script
python scripts/health_check.py --comprehensive

# Database health
python scripts/test_databases.py --verbose

# Memory usage monitoring
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'RAM: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"
```

---

## 🚀 **Production Deployment**

### **7.1 System Service Setup**

#### **Create systemd service:**

```bash
# Create service file
sudo tee /etc/systemd/system/tyra-memory-server.service > /dev/null <<EOF
[Unit]
Description=Tyra MCP Memory Server
After=network.target postgresql.service redis.service memgraph.service
Requires=postgresql.service redis.service memgraph.service

[Service]
Type=simple
User=tyra
Group=tyra
WorkingDirectory=/opt/tyra-memory-server
Environment=PATH=/opt/tyra-memory-server/venv/bin
Environment=PYTHONPATH=/opt/tyra-memory-server
ExecStart=/opt/tyra-memory-server/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/opt/tyra-memory-server

[Install]
WantedBy=multi-user.target
EOF

# Create user for service
sudo useradd --system --home /opt/tyra-memory-server --shell /bin/false tyra
sudo chown -R tyra:tyra /opt/tyra-memory-server

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable tyra-memory-server
sudo systemctl start tyra-memory-server

# Check status
sudo systemctl status tyra-memory-server
```

### **7.2 SSL/TLS Setup**

#### **Let's Encrypt (Production)**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Test renewal
sudo certbot renew --dry-run
```

### **7.3 Load Balancer (Nginx)**

```bash
# Install Nginx
sudo apt install nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/tyra-memory-server > /dev/null <<EOF
upstream tyra_backend {
    server localhost:8000;
    # Add more servers for load balancing
    # server localhost:8001;
}

server {
    listen 80;
    server_name yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Proxy configuration
    location / {
        proxy_pass http://tyra_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://tyra_backend/health;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/tyra-memory-server/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/tyra-memory-server /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl reload nginx
```

### **7.4 Monitoring Setup**

```bash
# Setup monitoring with automatic installation
./setup.sh --env production --monitoring

# Manual monitoring setup
docker compose --profile monitoring up -d

# Access monitoring dashboards:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - Alertmanager: http://localhost:9093
```

---

## 🔄 **Maintenance & Updates**

### **8.1 Backup Procedures**

```bash
# Automated backup
./scripts/deploy/backup.sh

# Manual backup
pg_dump -h localhost -U tyra tyra_memory > backup_$(date +%Y%m%d_%H%M%S).sql
redis-cli BGSAVE

# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/ .env
```

### **8.2 Update Process**

```bash
# Backup before update
./scripts/deploy/backup.sh

# Update code
git pull origin main

# Update dependencies
source venv/bin/activate
poetry install --with dev,test

# Run migrations
./scripts/deploy/migrate.sh

# Restart services
sudo systemctl restart tyra-memory-server

# Verify update
curl http://localhost:8000/health
```

### **8.3 Performance Monitoring**

```bash
# Monitor system resources
htop
iotop
netstat -tulpn

# Monitor application logs
journalctl -u tyra-memory-server -f

# Check performance metrics
curl http://localhost:8000/v1/admin/metrics
```

---

## 📚 **Additional Resources**

### **Documentation Links**
- [Configuration Guide](CONFIGURATION.md) - Detailed configuration options
- [Container Guide](docs/CONTAINERS.md) - Docker deployment guide
- [Architecture Overview](ARCHITECTURE.md) - System architecture
- [API Documentation](API.md) - Complete API reference
- [Component Swapping](SWAPPING_COMPONENTS.md) - How to swap providers
- [Telemetry Guide](TELEMETRY.md) - Observability setup

### **Useful Commands Reference**

#### **Service Management**
```bash
# Tyra Memory Server
sudo systemctl status tyra-memory-server
sudo systemctl restart tyra-memory-server
sudo systemctl stop tyra-memory-server

# Database services
sudo systemctl status postgresql
sudo systemctl status redis-server  # or redis
sudo systemctl status memgraph

# Check all services at once
sudo systemctl status postgresql redis-server memgraph tyra-memory-server
```

#### **Database Operations**
```bash
# PostgreSQL operations
psql -h localhost -U tyra -d tyra_memory
sudo -u postgres psql -c "SELECT version();"
sudo -u postgres psql tyra_memory -c "\dx"  # List extensions

# Redis operations
redis-cli ping
redis-cli info
redis-cli monitor

# Memgraph operations
mgconsole --host localhost --port 7687
echo "SHOW DATABASES;" | mgconsole
echo "MATCH (n) RETURN count(n);" | mgconsole  # Count all nodes
```

#### **Ubuntu 24.04 Specific Commands**
```bash
# Check Ubuntu version
lsb_release -a
cat /etc/os-release

# PostgreSQL 16 specific (Ubuntu 24.04)
sudo systemctl status postgresql@16-main
sudo -u postgres psql -c "SHOW server_version;"
ls /etc/postgresql/16/main/

# Package management
apt list --installed | grep -E "(postgresql|redis|memgraph)"
apt policy postgresql postgresql-16-pgvector

# Performance monitoring
systemctl status --user  # User services
systemd-analyze blame    # Boot performance
journalctl --since "1 hour ago" | grep -i error
```

#### **Log Monitoring**
```bash
# Application logs
tail -f logs/memory-server.log
journalctl -u tyra-memory-server -f
grep -i error logs/memory-server.log

# System logs
journalctl -u postgresql -f
journalctl -u redis-server -f
journalctl -u memgraph -f

# Performance testing
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
ab -n 1000 -c 10 http://localhost:8000/health

# Crawl4AI testing
python -c "from crawl4ai import WebCrawler; print('✅ Crawl4AI available')"
```

#### **Quick Health Checks**
```bash
# All-in-one health check
echo "=== System Health Check ==="
echo "PostgreSQL: $(pg_isready -h localhost && echo '✅ OK' || echo '❌ FAIL')"
echo "Redis: $(redis-cli ping 2>/dev/null && echo '✅ OK' || echo '❌ FAIL')"
echo "Memgraph: $(timeout 5 mgconsole --host localhost --port 7687 -c 'RETURN 1;' 2>/dev/null && echo '✅ OK' || echo '❌ FAIL')"
echo "Tyra API: $(curl -s http://localhost:8000/health | grep -q healthy && echo '✅ OK' || echo '❌ FAIL')"
echo "Python: $(python --version 2>/dev/null && echo '✅ OK' || echo '❌ FAIL')"
```

---

## 🎉 **Installation Complete!**

### **✅ Verification Checklist**

#### **Core Requirements**
- [ ] Python 3.12+ installed and virtual environment active
- [ ] PostgreSQL 15+ with pgvector extension running and tested
- [ ] Redis server running and accessible (PING returns PONG)
- [ ] Memgraph database running and accepting connections
- [ ] All Python dependencies installed successfully
- [ ] Configuration files properly set up (.env and YAML configs)
- [ ] Database migrations completed without errors
- [ ] Tyra MCP Memory Server starts without errors
- [ ] Health check endpoint returns "healthy"
- [ ] Basic memory storage/retrieval works
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Trading safety tests pass (critical for production)

#### **Optional Components**
- [ ] Crawl4AI installed and configured for n8n workflows
- [ ] GPU support configured (if NVIDIA GPU available)
- [ ] Monitoring stack operational (if enabled)
- [ ] SSL/TLS certificates configured (for production)
- [ ] Load balancer configured (for production)

#### **Platform-Specific Verification**
- [ ] **Ubuntu 24.04**: PostgreSQL 16, latest package versions
- [ ] **Ubuntu 22.04**: PostgreSQL 15, compatible packages
- [ ] **Ubuntu 20.04**: Manual pgvector build successful
- [ ] **macOS**: Homebrew packages installed correctly
- [ ] **Docker**: All containers healthy and communicating

### **🚀 Next Steps**

1. **Configure for your use case**: Edit `config/config.yaml`
2. **Set up monitoring**: Configure observability stack
3. **Integrate with Claude**: Add MCP server to Claude
4. **Load test**: Verify performance under load
5. **Setup backups**: Configure automated backups
6. **Security review**: Review security settings

### **📞 Support**

If you encounter issues:

1. Check the [troubleshooting section](#-troubleshooting-guide)
2. Review logs in `logs/` directory
3. Run `python scripts/health_check.py --comprehensive`
4. Create GitHub issue with diagnostic information

**Congratulations! Your Tyra MCP Memory Server is ready for production use.** 🎊

---

*Installation Guide Version 1.0.0 - Updated January 2025*