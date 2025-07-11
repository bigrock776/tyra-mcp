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
| **HuggingFace CLI** | Latest | - | Required for model downloads |
| **Git LFS** | 2.0+ | - | For large model files |
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

# Alternative: Test with Memgraph python driver
python3 -c "
from gqlalchemy import Memgraph
memgraph = Memgraph(host='localhost', port=7687)
try:
    results = list(memgraph.execute('RETURN \"Hello Memgraph!\" AS message'))
    print(results[0]['message'] if results else 'No result')
finally:
    memgraph.close()
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

### **Step 5: Model Installation (REQUIRED)**

**⚠️ IMPORTANT: Models must be manually downloaded by users - no automatic downloads.**

This section provides instructions for downloading and installing the embedding models, cross-encoders, and reranking models required for Tyra's operation.

#### **5.1 Required Models Overview**

| Model Type | Model Name | Purpose | Size | Location |
|------------|------------|---------|------|----------|
| **Primary Embedding** | `intfloat/e5-large-v2` | Main semantic search | ~1.34GB | `./models/embeddings/e5-large-v2/` |
| **Fallback Embedding** | `sentence-transformers/all-MiniLM-L12-v2` | Backup embedding | ~120MB | `./models/embeddings/all-MiniLM-L12-v2/` |
| **Cross-Encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking | ~120MB | `./models/cross-encoders/ms-marco-MiniLM-L-6-v2/` |

#### **5.2 Prerequisites**

```bash
# Install HuggingFace CLI (required for downloads)
pip install huggingface-hub

# Install Git LFS (required for large files)
git lfs install

# Check available disk space (need at least 2GB free)
df -h ./

# Check available RAM (need at least 8GB)
free -h

# For GPU support, check CUDA availability
nvidia-smi  # Should show your GPU if available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### **5.3 Create Model Directory Structure**

```bash
# Create local model directories
mkdir -p ./models/embeddings
mkdir -p ./models/cross-encoders
mkdir -p ./models/rerankers

# Verify directory structure
tree ./models/
```

#### **5.4 Download Embedding Models**

##### **A. Primary Embedding Model**

```bash
# Download intfloat/e5-large-v2 to local directory
huggingface-cli download intfloat/e5-large-v2 \
  --local-dir ./models/embeddings/e5-large-v2 \
  --local-dir-use-symlinks False

# Verify download
ls -la ./models/embeddings/e5-large-v2/
```

##### **B. Fallback Embedding Model**

```bash
# Download sentence-transformers/all-MiniLM-L12-v2 to local directory
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 \
  --local-dir ./models/embeddings/all-MiniLM-L12-v2 \
  --local-dir-use-symlinks False

# Verify download
ls -la ./models/embeddings/all-MiniLM-L12-v2/
```

#### **5.5 Download Cross-Encoder Models**

##### **A. Primary Cross-Encoder**

```bash
# Download cross-encoder/ms-marco-MiniLM-L-6-v2 to local directory
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2 \
  --local-dir-use-symlinks False

# Verify download
ls -la ./models/cross-encoders/ms-marco-MiniLM-L-6-v2/
```

##### **B. Alternative Cross-Encoder (Optional)**

```bash
# Download alternative cross-encoder
huggingface-cli download cross-encoder/stsb-roberta-base \
  --local-dir ./models/cross-encoders/stsb-roberta-base \
  --local-dir-use-symlinks False
```

#### **5.6 Reranking Configuration**

Update your configuration to use local model paths:

```yaml
# config/config.yaml
embeddings:
  primary:
    model_path: "./models/embeddings/e5-large-v2"
    model_name: "intfloat/e5-large-v2"
  fallback:
    model_path: "./models/embeddings/all-MiniLM-L12-v2"
    model_name: "sentence-transformers/all-MiniLM-L12-v2"

rag:
  reranking:
    provider: "cross_encoder"
    model_path: "./models/cross-encoders/ms-marco-MiniLM-L-6-v2"
    model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_local_files: true
    threshold: 0.7
    top_k: 5
```

#### **5.7 Testing Model Installation**

Create and run model verification scripts:

```bash
# Test embedding models
python scripts/test_embedding_model.py

# Test cross-encoder models  
python scripts/test_cross_encoder.py

# Test complete model pipeline
python scripts/test_model_pipeline.py
```

**Expected output:**
```
✅ Primary embedding model (e5-large-v2): OK
✅ Fallback embedding model (all-MiniLM-L12-v2): OK
✅ Cross-encoder model (ms-marco-MiniLM-L-6-v2): OK
✅ All models loaded successfully from local paths
```

#### **5.4 GPU/CUDA Configuration**

##### **NVIDIA GPU Setup**

```bash
# Check CUDA version compatibility
nvidia-smi | grep "CUDA Version"

# Install appropriate PyTorch version for your CUDA
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU setup
python -c "
import torch
import sentence_transformers

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}GB')
"

# Test GPU with embedding model
python -c "
from sentence_transformers import SentenceTransformer
import torch
import time

# Force GPU usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load model on GPU
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device=device)

# Benchmark
texts = ['Test sentence'] * 100
start = time.time()
embeddings = model.encode(texts, show_progress_bar=True)
elapsed = time.time() - start

print(f'\\nProcessed 100 texts in {elapsed:.2f}s')
print(f'Speed: {100/elapsed:.1f} texts/second')
"
```

##### **Apple Silicon (M1/M2/M3) Setup**

```bash
# Install Metal Performance Shaders (MPS) support
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✅ Apple Silicon GPU acceleration available')
"

# Configure for MPS
export EMBEDDINGS_DEVICE=mps  # Instead of 'cuda'
```

#### **5.5 Configuration Options**

##### **Environment Variables**

```bash
# Core embedding configuration
export EMBEDDINGS_PRIMARY_MODEL="intfloat/e5-large-v2"
export EMBEDDINGS_FALLBACK_MODEL="sentence-transformers/all-MiniLM-L12-v2"
export EMBEDDINGS_DEVICE="auto"  # auto, cuda, cpu, mps
export EMBEDDINGS_BATCH_SIZE=32
export EMBEDDINGS_MAX_LENGTH=512
export EMBEDDINGS_NORMALIZE=true

# Performance tuning
export EMBEDDINGS_NUM_THREADS=4  # CPU threads
export EMBEDDINGS_USE_FP16=true  # Half precision for GPU
export EMBEDDINGS_COMPILE_MODEL=true  # PyTorch 2.0+ optimization

# Cache configuration  
export EMBEDDINGS_CACHE_DIR="~/.cache/tyra/embeddings"
export EMBEDDINGS_CACHE_TTL=86400  # 24 hours
export EMBEDDINGS_PRELOAD_ON_START=true
```

##### **YAML Configuration (config/config.yaml)**

```yaml
embeddings:
  # Primary model configuration
  primary:
    model: "intfloat/e5-large-v2"
    device: "auto"  # auto-detect best device
    dimensions: 1024
    max_length: 512
    batch_size: 32
    normalize: true
    use_fp16: true  # Half precision for GPU
    compile: true   # PyTorch 2.0 optimization
    
  # Fallback model configuration  
  fallback:
    model: "sentence-transformers/all-MiniLM-L12-v2"
    device: "cpu"  # Always use CPU for reliability
    dimensions: 384
    max_length: 256
    batch_size: 16
    normalize: true
    
  # Advanced settings
  pooling_strategy: "mean"  # mean, max, cls
  instruction_prefix: "Represent this for retrieval: "
  query_prefix: "Represent this query for retrieval: "
  
  # Cache settings
  cache:
    enabled: true
    directory: "~/.cache/tyra/embeddings"
    max_size_gb: 10
    ttl_seconds: 86400
    
  # Performance settings
  num_threads: 4  # CPU threads
  warmup_on_start: true
  precompute_popular: true
  
  # Model management
  auto_download: true
  model_cache_dir: "~/.cache/huggingface"
  offline_mode: false
```

#### **5.6 Model Selection Guide**

##### **Choosing the Right Model**

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| **High Accuracy** | intfloat/e5-large-v2 | Best semantic understanding |
| **Low Latency** | all-MiniLM-L12-v2 | 10x faster, good quality |
| **Limited RAM** | all-MiniLM-L6-v2 | Only 80MB, decent quality |
| **Multilingual** | intfloat/multilingual-e5-large | 100+ languages |
| **Domain-Specific** | Custom fine-tuned | Train on your data |

##### **Alternative Models**

```python
# config/embedding_models.yaml
alternative_models:
  # High performance models
  high_performance:
    - model: "BAAI/bge-large-en-v1.5"
      dimensions: 1024
      size: "1.34GB"
      score: 64.23  # MTEB score
      
    - model: "thenlper/gte-large"
      dimensions: 1024
      size: "1.34GB"
      score: 63.13
      
  # Balanced models  
  balanced:
    - model: "BAAI/bge-base-en-v1.5"
      dimensions: 768
      size: "438MB"
      score: 63.55
      
    - model: "thenlper/gte-base"
      dimensions: 768
      size: "438MB"
      score: 62.39
      
  # Lightweight models
  lightweight:
    - model: "BAAI/bge-small-en-v1.5"
      dimensions: 384
      size: "133MB"
      score: 62.17
      
    - model: "thenlper/gte-small"
      dimensions: 384
      size: "133MB"
      score: 61.36
```

#### **5.7 Verification & Testing**

##### **Model Verification Script**

```bash
# Create verification script
cat > verify_embeddings.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def verify_embedding_models():
    """Comprehensive embedding model verification"""
    
    print("🔍 Embedding Model Verification\n")
    
    # Check environment
    print("Environment Information:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Test models
    models_to_test = [
        ("intfloat/e5-large-v2", 1024),
        ("sentence-transformers/all-MiniLM-L12-v2", 384)
    ]
    
    test_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming how we process information",
        "Tyra's memory system uses advanced embedding techniques"
    ]
    
    for model_name, expected_dim in models_to_test:
        print(f"Testing {model_name}...")
        
        try:
            # Load model
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            print(f"  ✅ Model loaded in {load_time:.2f}s")
            
            # Test encoding
            start_time = time.time()
            embeddings = model.encode(test_sentences, show_progress_bar=False)
            encode_time = time.time() - start_time
            
            # Verify dimensions
            assert embeddings.shape == (len(test_sentences), expected_dim), \
                f"Expected shape {(len(test_sentences), expected_dim)}, got {embeddings.shape}"
            print(f"  ✅ Dimensions correct: {embeddings.shape}")
            
            # Test similarity
            similarities = np.dot(embeddings, embeddings.T)
            print(f"  ✅ Similarity matrix computed: {similarities.shape}")
            
            # Performance metrics
            texts_per_second = len(test_sentences) / encode_time
            print(f"  ✅ Performance: {texts_per_second:.1f} texts/second")
            
            # Memory usage
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
            print(f"  ✅ Model size: {model_size:.1f}MB in memory")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
        
        print()
    
    print("✅ All embedding models verified successfully!")
    return True

if __name__ == "__main__":
    sys.exit(0 if verify_embedding_models() else 1)
EOF

chmod +x verify_embeddings.py
python verify_embeddings.py
```

##### **Performance Benchmarking**

```bash
# Create benchmark script
cat > benchmark_embeddings.py << 'EOF'
#!/usr/bin/env python3
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt

def benchmark_embeddings():
    """Benchmark embedding model performance"""
    
    # Test configurations
    batch_sizes = [1, 8, 16, 32, 64, 128]
    text_lengths = [10, 50, 100, 200, 512]  # words
    
    # Generate test data
    test_texts = [
        " ".join(["word"] * length) for length in text_lengths
    ]
    
    models = [
        "intfloat/e5-large-v2",
        "sentence-transformers/all-MiniLM-L12-v2"
    ]
    
    results = {}
    
    for model_name in models:
        print(f"\nBenchmarking {model_name}...")
        model = SentenceTransformer(model_name)
        
        model_results = {}
        
        for batch_size in batch_sizes:
            batch_times = []
            
            for text in test_texts:
                batch = [text] * batch_size
                
                start = time.time()
                model.encode(batch, show_progress_bar=False)
                elapsed = time.time() - start
                
                batch_times.append(elapsed)
            
            model_results[batch_size] = {
                'times': batch_times,
                'throughput': [batch_size / t for t in batch_times]
            }
            
            print(f"  Batch size {batch_size}: "
                  f"{np.mean(model_results[batch_size]['throughput']):.1f} texts/sec")
        
        results[model_name] = model_results
    
    return results

# Run benchmark
results = benchmark_embeddings()
print("\n✅ Benchmark complete!")
EOF

python benchmark_embeddings.py
```

#### **5.8 Troubleshooting Embedding Issues**

##### **Common Issues and Solutions**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Out of Memory** | CUDA OOM errors | Reduce batch_size, use CPU, or use smaller model |
| **Slow Performance** | <10 texts/second | Enable GPU, increase batch_size, use compiled model |
| **Download Failures** | Connection timeouts | Use manual download, check firewall, use mirrors |
| **Wrong Dimensions** | Shape mismatch errors | Verify model name, check configuration |
| **Import Errors** | Module not found | Reinstall sentence-transformers, check virtual env |

##### **Debugging Commands**

```bash
# Check model cache
ls -la ~/.cache/huggingface/hub/
ls -la ~/.cache/sentence-transformers/

# Clear corrupted cache
rm -rf ~/.cache/huggingface/hub/models--intfloat--e5-large-v2
rm -rf ~/.cache/sentence-transformers/intfloat_e5-large-v2

# Test with minimal example
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
print(model.encode('test'))
"

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor during execution
watch -n 1 nvidia-smi
```

##### **Fallback Configuration**

```python
# config/embedding_fallback.yaml
fallback_chain:
  - model: "intfloat/e5-large-v2"
    device: "cuda"
    on_error: "next"
    
  - model: "intfloat/e5-large-v2"
    device: "cpu"
    on_error: "next"
    
  - model: "sentence-transformers/all-MiniLM-L12-v2"
    device: "cuda"
    on_error: "next"
    
  - model: "sentence-transformers/all-MiniLM-L12-v2"
    device: "cpu"
    on_error: "fail"
    
error_handling:
  max_retries: 3
  retry_delay: 1.0
  log_errors: true
  notify_on_fallback: true
```

### **Step 6: Database Initialization**

#### **6.1 Run Database Migrations**

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

#### **6.2 Verify Embedding Models Setup**

```bash
# Verify embedding models are installed (see Step 5 for full installation guide)
python verify_embeddings.py

# If models aren't installed yet, run the comprehensive setup:
# See "Step 5: Embedding Models Installation & Configuration" above
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

## 🔒 **Security Hardening (Local Installation)**

### **Local Security Configuration**

#### **7.1 System Security**

```bash
# Update system packages (Ubuntu)
sudo apt update && sudo apt upgrade -y

# Install security updates automatically
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# Configure firewall for local services only
sudo ufw --force enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow only essential local services
sudo ufw allow from 127.0.0.1 to any port 5432    # PostgreSQL
sudo ufw allow from 127.0.0.1 to any port 6379    # Redis
sudo ufw allow from 127.0.0.1 to any port 7687    # Memgraph
sudo ufw allow from 127.0.0.1 to any port 8000    # Tyra API

# Optional: Allow from local network (adjust subnet as needed)
# sudo ufw allow from 192.168.1.0/24 to any port 8000

# Check firewall status
sudo ufw status verbose
```

#### **7.2 Database Security**

```bash
# PostgreSQL security configuration
sudo -u postgres psql << EOF
-- Create read-only user for monitoring
CREATE USER tyra_readonly WITH PASSWORD 'readonly_secure_password';
GRANT CONNECT ON DATABASE tyra_memory TO tyra_readonly;
GRANT USAGE ON SCHEMA public TO tyra_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO tyra_readonly;

-- Limit connections
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

-- Enable connection logging for security monitoring
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_statement = 'mod';  -- Log modifications only

SELECT pg_reload_conf();
EOF

# Redis security
sudo nano /etc/redis/redis.conf
# Add these security settings:
```

```ini
# Redis security configuration
bind 127.0.0.1 ::1
protected-mode yes
port 6379
requirepass secure_redis_password_here

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG "CONFIG_b8e5c5b8e5c5b8e5c5b8e5c5b8e5c5b8"

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
```

```bash
# Restart Redis with new configuration
sudo systemctl restart redis-server

# Test Redis authentication
redis-cli -a secure_redis_password_here ping
```

#### **7.3 File System Security**

```bash
# Set proper ownership and permissions
sudo chown -R tyra:tyra /opt/tyra-memory-server
find /opt/tyra-memory-server -type f -exec chmod 644 {} \;
find /opt/tyra-memory-server -type d -exec chmod 755 {} \;
chmod +x /opt/tyra-memory-server/setup.sh
chmod +x /opt/tyra-memory-server/scripts/deploy/*.sh

# Secure configuration files
chmod 600 /opt/tyra-memory-server/.env
chmod 600 /opt/tyra-memory-server/config/local/*

# Create secure log directory
sudo mkdir -p /var/log/tyra
sudo chown tyra:tyra /var/log/tyra
sudo chmod 750 /var/log/tyra

# Setup log rotation
sudo tee /etc/logrotate.d/tyra-memory-server > /dev/null << EOF
/var/log/tyra/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 tyra tyra
    postrotate
        systemctl reload tyra-memory-server
    endscript
}
EOF
```

#### **7.4 Network Security**

```bash
# Configure local DNS resolution (optional)
echo "127.0.0.1 tyra-memory-server.local" | sudo tee -a /etc/hosts

# Setup SSL certificates for local development
mkdir -p /opt/tyra-memory-server/ssl

# Generate self-signed certificate for local use
openssl req -x509 -newkey rsa:4096 -keyout /opt/tyra-memory-server/ssl/key.pem \
  -out /opt/tyra-memory-server/ssl/cert.pem -days 365 -nodes \
  -subj "/C=US/ST=Local/L=Local/O=Tyra/OU=Development/CN=localhost"

# Set proper permissions
chmod 600 /opt/tyra-memory-server/ssl/key.pem
chmod 644 /opt/tyra-memory-server/ssl/cert.pem
```

---

## 🌐 **Local Network Configuration**

### **8.1 Local Development Network Setup**

#### **Hosts File Configuration**

```bash
# Add local service aliases for easier access
sudo tee -a /etc/hosts << EOF
# Tyra MCP Memory Server Local Services
127.0.0.1 tyra-api.local
127.0.0.1 tyra-postgres.local
127.0.0.1 tyra-redis.local
127.0.0.1 tyra-memgraph.local
EOF

# Verify configuration
cat /etc/hosts | grep tyra
ping -c 1 tyra-api.local
```

#### **Local Service Discovery**

```bash
# Install and configure Avahi for local service discovery (Ubuntu)
sudo apt install avahi-daemon avahi-utils

# Create service announcement
sudo tee /etc/avahi/services/tyra-memory-server.service << EOF
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">Tyra Memory Server on %h</name>
  <service>
    <type>_http._tcp</type>
    <port>8000</port>
    <txt-record>path=/health</txt-record>
    <txt-record>version=1.0.0</txt-record>
  </service>
</service-group>
EOF

# Restart Avahi
sudo systemctl restart avahi-daemon

# Test service discovery
avahi-browse -t _http._tcp
```

### **8.2 Local Proxy Configuration**

```bash
# Create nginx configuration for local development
sudo tee /etc/nginx/sites-available/tyra-local.conf << EOF
server {
    listen 80;
    server_name tyra-api.local;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # CORS for local development
        add_header Access-Control-Allow-Origin "*";
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type, Authorization";
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/tyra-local.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 🛠️ **Development Environment Setup**

### **9.1 IDE Configuration**

#### **Visual Studio Code Setup**

```bash
# Install VS Code extensions for optimal development
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension ms-python.flake8
code --install-extension bradlc.vscode-tailwindcss
code --install-extension ms-vscode.vscode-json

# Create VS Code workspace configuration
cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.associations": {
        "*.yaml": "yaml",
        "*.yml": "yaml"
    },
    "yaml.schemas": {
        "./config/schema.json": ["config/*.yaml", "config/*.yml"]
    }
}
EOF

# Create launch configuration for debugging
cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Tyra MCP Server",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "args": ["--debug"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "\${workspaceFolder}",
                "DEBUG": "true"
            }
        },
        {
            "name": "Debug Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal"
        }
    ]
}
EOF
```

#### **Development Tools Installation**

```bash
# Install development tools
pip install --upgrade \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pytest-benchmark \
    pre-commit

# Setup pre-commit hooks
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-redis, types-requests]
EOF

# Install pre-commit hooks
pre-commit install

# Test pre-commit setup
pre-commit run --all-files
```

### **9.2 Local Development Scripts**

```bash
# Create development utility scripts
mkdir -p scripts/dev

# Quick development server script
cat > scripts/dev/start-dev.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Tyra MCP Memory Server Development Environment"

# Check services
echo "Checking services..."
systemctl is-active --quiet postgresql || sudo systemctl start postgresql
systemctl is-active --quiet redis-server || sudo systemctl start redis-server
systemctl is-active --quiet memgraph || sudo systemctl start memgraph

# Activate virtual environment
source venv/bin/activate

# Set development environment
export ENVIRONMENT=development
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start server with auto-reload
echo "Starting server with hot reload..."
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload --log-level debug
EOF

chmod +x scripts/dev/start-dev.sh

# Database reset script for development
cat > scripts/dev/reset-db.sh << 'EOF'
#!/bin/bash
set -e

read -p "⚠️  This will DESTROY all data. Are you sure? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo "🗑️  Resetting databases..."

# Reset PostgreSQL
sudo -u postgres psql << SQL
DROP DATABASE IF EXISTS tyra_memory;
CREATE DATABASE tyra_memory;
GRANT ALL PRIVILEGES ON DATABASE tyra_memory TO tyra;
\c tyra_memory
CREATE EXTENSION vector;
CREATE EXTENSION pg_trgm;
CREATE EXTENSION btree_gin;
SQL

# Reset Redis
redis-cli FLUSHALL

# Reset Memgraph
echo "MATCH (n) DETACH DELETE n;" | mgconsole --host localhost --port 7687

# Run migrations
source venv/bin/activate
python -m src.migrations.run_migrations

echo "✅ Databases reset complete"
EOF

chmod +x scripts/dev/reset-db.sh

# Testing script
cat > scripts/dev/run-tests.sh << 'EOF'
#!/bin/bash
set -e

source venv/bin/activate

echo "🧪 Running comprehensive test suite..."

# Unit tests
echo "Running unit tests..."
pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Integration tests  
echo "Running integration tests..."
pytest tests/integration/ -v

# Performance tests
echo "Running performance benchmarks..."
pytest tests/performance/ -v --benchmark-only

# Trading safety tests (critical)
echo "Running trading safety tests..."
pytest tests/test_mcp_trading_safety.py -v

echo "✅ All tests completed"
EOF

chmod +x scripts/dev/run-tests.sh
```

---

## 📊 **Automated Health Monitoring**

### **10.1 System Health Monitoring Setup**

```bash
# Create comprehensive health monitoring script
cat > scripts/monitoring/health-monitor.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import asyncpg
import redis
import psutil
import requests
import json
import time
from datetime import datetime
from pathlib import Path

class TyraHealthMonitor:
    def __init__(self):
        self.checks = []
        self.alerts = []
        
    async def check_postgresql(self):
        """Check PostgreSQL connectivity and performance"""
        try:
            conn = await asyncpg.connect(
                "postgresql://tyra:password@localhost:5432/tyra_memory"
            )
            
            # Test query
            start = time.time()
            result = await conn.fetchval("SELECT 1")
            latency = (time.time() - start) * 1000
            
            # Check extensions
            extensions = await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')"
            )
            
            await conn.close()
            
            return {
                "service": "postgresql",
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "extensions": [row['extname'] for row in extensions]
            }
        except Exception as e:
            return {
                "service": "postgresql", 
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_redis(self):
        """Check Redis connectivity and performance"""
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            start = time.time()
            r.ping()
            latency = (time.time() - start) * 1000
            
            info = r.info()
            
            return {
                "service": "redis",
                "status": "healthy", 
                "latency_ms": round(latency, 2),
                "memory_usage_mb": round(info['used_memory'] / 1024 / 1024, 2),
                "connected_clients": info['connected_clients']
            }
        except Exception as e:
            return {
                "service": "redis",
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_memgraph(self):
        """Check Memgraph connectivity"""
        try:
            from gqlalchemy import Memgraph
            
            memgraph = Memgraph(host="localhost", port=7687)
            
            start = time.time()
            test_result = list(memgraph.execute("RETURN 1 AS test"))
            latency = (time.time() - start) * 1000
            
            node_count_result = list(memgraph.execute("MATCH (n) RETURN count(n) AS count"))
            node_count = node_count_result[0]['count'] if node_count_result else 0
                
            memgraph.close()
            
            return {
                "service": "memgraph",
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "node_count": node_count['count']
            }
        except Exception as e:
            return {
                "service": "memgraph", 
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_tyra_api(self):
        """Check Tyra API health"""
        try:
            start = time.time()
            response = requests.get("http://localhost:8000/health", timeout=5)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return {
                    "service": "tyra_api",
                    "status": "healthy",
                    "latency_ms": round(latency, 2),
                    "response": response.json()
                }
            else:
                return {
                    "service": "tyra_api",
                    "status": "unhealthy", 
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "service": "tyra_api",
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_system_resources(self):
        """Check system resource usage"""
        return {
            "service": "system",
            "status": "info",
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_avg": psutil.getloadavg()
        }
    
    async def run_all_checks(self):
        """Run all health checks"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": []
        }
        
        # Run checks
        results["checks"].append(await self.check_postgresql())
        results["checks"].append(self.check_redis())
        results["checks"].append(self.check_memgraph())
        results["checks"].append(self.check_tyra_api())
        results["checks"].append(self.check_system_resources())
        
        # Overall health
        unhealthy_services = [
            check for check in results["checks"] 
            if check.get("status") == "unhealthy"
        ]
        
        results["overall_status"] = "unhealthy" if unhealthy_services else "healthy"
        results["unhealthy_services"] = len(unhealthy_services)
        
        return results

async def main():
    monitor = TyraHealthMonitor()
    results = await monitor.run_all_checks()
    
    print(json.dumps(results, indent=2))
    
    # Exit with error code if unhealthy
    if results["overall_status"] == "unhealthy":
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x scripts/monitoring/health-monitor.py

# Create systemd service for continuous monitoring
sudo tee /etc/systemd/system/tyra-health-monitor.service << EOF
[Unit]
Description=Tyra Health Monitor
After=tyra-memory-server.service

[Service]
Type=oneshot
User=tyra
WorkingDirectory=/opt/tyra-memory-server
Environment=PATH=/opt/tyra-memory-server/venv/bin
ExecStart=/opt/tyra-memory-server/venv/bin/python scripts/monitoring/health-monitor.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create timer for regular health checks
sudo tee /etc/systemd/system/tyra-health-monitor.timer << EOF
[Unit]
Description=Run Tyra Health Monitor every 5 minutes
Requires=tyra-health-monitor.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF

# Enable monitoring
sudo systemctl daemon-reload
sudo systemctl enable tyra-health-monitor.timer
sudo systemctl start tyra-health-monitor.timer
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

## 💾 **Disaster Recovery & Backup**

### **11.1 Complete Backup Strategy**

#### **Automated Full System Backup**

```bash
# Create comprehensive backup script
cat > scripts/backup/full-backup.sh << 'EOF'
#!/bin/bash
set -e

BACKUP_DIR="/opt/tyra-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/tyra_backup_$TIMESTAMP"

echo "🔄 Starting comprehensive Tyra MCP Memory Server backup..."

# Create backup directory
mkdir -p "$BACKUP_PATH"

# 1. PostgreSQL backup
echo "📊 Backing up PostgreSQL..."
pg_dump -h localhost -U tyra tyra_memory > "$BACKUP_PATH/postgresql_tyra_memory.sql"
sudo -u postgres pg_dumpall --globals-only > "$BACKUP_PATH/postgresql_globals.sql"

# 2. Redis backup
echo "💾 Backing up Redis..."
redis-cli BGSAVE
sleep 5  # Wait for background save to complete
cp /var/lib/redis/dump.rdb "$BACKUP_PATH/redis_dump.rdb"

# 3. Memgraph backup
echo "🕸️  Backing up Memgraph..."
echo "DUMP DATABASE;" | mgconsole --host localhost --port 7687 > "$BACKUP_PATH/memgraph_dump.cypher"

# 4. Configuration files
echo "⚙️  Backing up configuration..."
tar -czf "$BACKUP_PATH/config_backup.tar.gz" \
  config/ \
  .env \
  scripts/ \
  ssl/ \
  logs/

# 5. Python environment
echo "🐍 Backing up Python environment..."
pip freeze > "$BACKUP_PATH/requirements_frozen.txt"
cp pyproject.toml "$BACKUP_PATH/"
cp poetry.lock "$BACKUP_PATH/" 2>/dev/null || true

# 6. System configuration
echo "🔧 Backing up system configuration..."
cp /etc/systemd/system/tyra-memory-server.service "$BACKUP_PATH/" 2>/dev/null || true
cp /etc/nginx/sites-available/tyra-memory-server "$BACKUP_PATH/nginx_config" 2>/dev/null || true

# 7. Embedding models cache
echo "🧠 Backing up embedding models..."
if [ -d "$HOME/.cache/huggingface" ]; then
    tar -czf "$BACKUP_PATH/embedding_models.tar.gz" \
      "$HOME/.cache/huggingface" \
      "$HOME/.cache/sentence-transformers" 2>/dev/null || true
fi

# 8. Create backup manifest
cat > "$BACKUP_PATH/backup_manifest.json" << JSON
{
  "timestamp": "$TIMESTAMP",
  "version": "1.0.0",
  "hostname": "$(hostname)",
  "backup_path": "$BACKUP_PATH",
  "components": [
    "postgresql",
    "redis", 
    "memgraph",
    "configuration",
    "python_environment",
    "system_configuration",
    "embedding_models"
  ],
  "size_mb": $(du -sm "$BACKUP_PATH" | cut -f1)
}
JSON

# 9. Compress entire backup
echo "🗜️  Compressing backup..."
cd "$BACKUP_DIR"
tar -czf "tyra_backup_$TIMESTAMP.tar.gz" "tyra_backup_$TIMESTAMP/"
rm -rf "$BACKUP_PATH"

echo "✅ Backup completed: $BACKUP_DIR/tyra_backup_$TIMESTAMP.tar.gz"
echo "📏 Backup size: $(du -sh $BACKUP_DIR/tyra_backup_$TIMESTAMP.tar.gz | cut -f1)"
EOF

chmod +x scripts/backup/full-backup.sh

# Create backup restoration script
cat > scripts/backup/restore-backup.sh << 'EOF'
#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="/tmp/tyra_restore_$(date +%s)"

echo "🔄 Starting Tyra MCP Memory Server restoration..."
echo "📁 Backup file: $BACKUP_FILE"

# Extract backup
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR" --strip-components=1

# Read backup manifest
MANIFEST="$RESTORE_DIR/backup_manifest.json"
if [ -f "$MANIFEST" ]; then
    echo "📋 Backup manifest found"
    cat "$MANIFEST"
else
    echo "⚠️  No backup manifest found, proceeding with restoration..."
fi

read -p "⚠️  This will OVERWRITE existing data. Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Restoration aborted."
    rm -rf "$RESTORE_DIR"
    exit 1
fi

# Stop services
echo "🛑 Stopping services..."
sudo systemctl stop tyra-memory-server || true
sudo systemctl stop postgresql
sudo systemctl stop redis-server
sudo systemctl stop memgraph

# 1. Restore PostgreSQL
echo "📊 Restoring PostgreSQL..."
sudo systemctl start postgresql
sleep 5

# Drop and recreate database
sudo -u postgres psql << SQL
DROP DATABASE IF EXISTS tyra_memory;
CREATE DATABASE tyra_memory;
GRANT ALL PRIVILEGES ON DATABASE tyra_memory TO tyra;
SQL

# Restore data
sudo -u postgres psql tyra_memory < "$RESTORE_DIR/postgresql_tyra_memory.sql"
sudo -u postgres psql < "$RESTORE_DIR/postgresql_globals.sql" || true

# 2. Restore Redis
echo "💾 Restoring Redis..."
sudo systemctl stop redis-server
sudo cp "$RESTORE_DIR/redis_dump.rdb" /var/lib/redis/dump.rdb
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo systemctl start redis-server

# 3. Restore Memgraph
echo "🕸️  Restoring Memgraph..."
sudo systemctl start memgraph
sleep 5
mgconsole --host localhost --port 7687 < "$RESTORE_DIR/memgraph_dump.cypher"

# 4. Restore configuration
echo "⚙️  Restoring configuration..."
cd /opt/tyra-memory-server
tar -xzf "$RESTORE_DIR/config_backup.tar.gz"

# 5. Restore system configuration
echo "🔧 Restoring system configuration..."
if [ -f "$RESTORE_DIR/tyra-memory-server.service" ]; then
    sudo cp "$RESTORE_DIR/tyra-memory-server.service" /etc/systemd/system/
    sudo systemctl daemon-reload
fi

# 6. Restore embedding models (optional)
if [ -f "$RESTORE_DIR/embedding_models.tar.gz" ]; then
    echo "🧠 Restoring embedding models..."
    cd "$HOME"
    tar -xzf "$RESTORE_DIR/embedding_models.tar.gz"
fi

# Start services
echo "🚀 Starting services..."
sudo systemctl start postgresql
sudo systemctl start redis-server  
sudo systemctl start memgraph
sudo systemctl start tyra-memory-server

# Verify restoration
echo "🔍 Verifying restoration..."
sleep 10
curl -s http://localhost:8000/health || echo "⚠️  Health check failed"

# Cleanup
rm -rf "$RESTORE_DIR"

echo "✅ Restoration completed successfully!"
EOF

chmod +x scripts/backup/restore-backup.sh

# Setup automated backups
sudo tee /etc/systemd/system/tyra-backup.service << EOF
[Unit]
Description=Tyra MCP Memory Server Backup
After=tyra-memory-server.service

[Service]
Type=oneshot
User=tyra
WorkingDirectory=/opt/tyra-memory-server
ExecStart=/opt/tyra-memory-server/scripts/backup/full-backup.sh
StandardOutput=journal
StandardError=journal
EOF

sudo tee /etc/systemd/system/tyra-backup.timer << EOF
[Unit]
Description=Run Tyra backup daily
Requires=tyra-backup.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable automated backups
sudo systemctl daemon-reload
sudo systemctl enable tyra-backup.timer
sudo systemctl start tyra-backup.timer
```

### **11.2 Offline Installation Package Creation**

#### **Create Complete Offline Installation Package**

```bash
# Create offline package creation script
cat > scripts/deployment/create-offline-package.sh << 'EOF'
#!/bin/bash
set -e

PACKAGE_DIR="tyra-offline-$(date +%Y%m%d)"
echo "📦 Creating offline installation package: $PACKAGE_DIR"

mkdir -p "$PACKAGE_DIR"

# 1. Copy source code
echo "📁 Copying source code..."
rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='.pytest_cache' \
  . "$PACKAGE_DIR/source/"

# 2. Download Python packages
echo "🐍 Downloading Python packages..."
mkdir -p "$PACKAGE_DIR/python_packages"
pip download -r requirements.txt -d "$PACKAGE_DIR/python_packages/"

# 3. Download embedding models
echo "🧠 Downloading embedding models..."
mkdir -p "$PACKAGE_DIR/embedding_models"
python -c "
from sentence_transformers import SentenceTransformer
import shutil
import os

models = ['intfloat/e5-large-v2', 'sentence-transformers/all-MiniLM-L12-v2']
for model_name in models:
    print(f'Downloading {model_name}...')
    model = SentenceTransformer(model_name)
    # Copy model files
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    model_cache = model_name.replace('/', '--')
    src = os.path.join(cache_dir, f'models--{model_cache}')
    dst = os.path.join('$PACKAGE_DIR/embedding_models', model_cache)
    if os.path.exists(src):
        shutil.copytree(src, dst)
"

# 4. Download system packages
echo "📦 Downloading system packages..."
mkdir -p "$PACKAGE_DIR/system_packages"

# PostgreSQL packages
apt-get download postgresql-15 postgresql-contrib-15 postgresql-client-15
apt-get download postgresql-15-pgvector
mv *.deb "$PACKAGE_DIR/system_packages/"

# Redis packages  
apt-get download redis-server redis-tools
mv *.deb "$PACKAGE_DIR/system_packages/"

# 5. Create installation script
cat > "$PACKAGE_DIR/install-offline.sh" << 'INSTALL_SCRIPT'
#!/bin/bash
set -e

echo "🚀 Installing Tyra MCP Memory Server (Offline Mode)"

# Install system packages
echo "📦 Installing system packages..."
sudo dpkg -i system_packages/*.deb
sudo apt-get install -f

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3.12 -m venv venv
source venv/bin/activate
pip install --no-index --find-links python_packages/ -r source/requirements.txt

# Install embedding models
echo "🧠 Installing embedding models..."
mkdir -p ~/.cache/huggingface/hub
cp -r embedding_models/* ~/.cache/huggingface/hub/

# Copy source code
echo "📁 Installing source code..."
cp -r source/* .

# Run setup
echo "⚙️  Running setup..."
chmod +x setup.sh
./setup.sh --env production --offline

echo "✅ Offline installation completed!"
INSTALL_SCRIPT

chmod +x "$PACKAGE_DIR/install-offline.sh"

# 6. Create package documentation
cat > "$PACKAGE_DIR/README_OFFLINE.md" << 'README'
# Tyra MCP Memory Server - Offline Installation Package

This package contains everything needed to install Tyra MCP Memory Server on a system without internet access.

## Contents

- `source/` - Complete source code
- `python_packages/` - All Python dependencies
- `embedding_models/` - Pre-downloaded embedding models
- `system_packages/` - System packages (PostgreSQL, Redis, etc.)
- `install-offline.sh` - Automated installation script

## Installation

1. Transfer this entire directory to the target system
2. Run: `sudo ./install-offline.sh`
3. Follow the prompts

## Requirements

- Ubuntu 20.04+ or compatible Linux distribution
- Python 3.11+ 
- At least 8GB RAM and 20GB free disk space
- sudo privileges

## Support

For issues, check the troubleshooting section in the main INSTALLATION.md
README

# 7. Create package
echo "🗜️  Creating package archive..."
tar -czf "$PACKAGE_DIR.tar.gz" "$PACKAGE_DIR/"
rm -rf "$PACKAGE_DIR"

echo "✅ Offline package created: $PACKAGE_DIR.tar.gz"
echo "📏 Package size: $(du -sh $PACKAGE_DIR.tar.gz | cut -f1)"
EOF

chmod +x scripts/deployment/create-offline-package.sh
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

### **✅ Comprehensive Verification Checklist**

#### **Core System Requirements**
- [ ] **Operating System**: Ubuntu 20.04+, macOS 12+, or Windows 11+WSL2
- [ ] **Python Version**: 3.11+ installed (preferably 3.12+)
- [ ] **Memory**: Minimum 8GB RAM (16GB+ recommended)
- [ ] **Storage**: Minimum 20GB free space (50GB+ recommended)
- [ ] **CPU**: Minimum 4 cores (8+ recommended)
- [ ] **Network**: Local network configured for services

#### **Database Services**
- [ ] **PostgreSQL**: 15+ installed and running
  ```bash
  sudo systemctl status postgresql
  sudo -u postgres psql -c "SELECT version();"
  ```
- [ ] **pgvector Extension**: Installed and functional
  ```bash
  sudo -u postgres psql tyra_memory -c "SELECT * FROM pg_extension WHERE extname='vector';"
  ```
- [ ] **Redis**: 6.0+ installed and accessible
  ```bash
  redis-cli ping  # Should return PONG
  ```
- [ ] **Memgraph**: 2.0+ installed and accepting connections
  ```bash
  echo "RETURN 1;" | mgconsole --host localhost --port 7687
  ```

#### **Python Environment**
- [ ] **Virtual Environment**: Created and activated
  ```bash
  python --version  # Should show 3.11+ within venv
  which python     # Should point to venv/bin/python
  ```
- [ ] **Core Dependencies**: All packages installed
  ```bash
  pip list | grep -E "(fastapi|asyncpg|redis|sentence-transformers|torch)"
  ```
- [ ] **Embedding Models**: Downloaded and verified
  ```bash
  python verify_embeddings.py
  ```

#### **Configuration Files**
- [ ] **Environment File**: `.env` created with all required variables
- [ ] **YAML Configuration**: `config/config.yaml` properly structured
- [ ] **Security**: Proper file permissions set (600 for .env)
- [ ] **Secrets**: All passwords and keys generated and secured

#### **Tyra MCP Memory Server**
- [ ] **Service Startup**: Server starts without errors
  ```bash
  python main.py  # Should start without Python errors
  ```
- [ ] **Health Endpoint**: API responds correctly
  ```bash
  curl http://localhost:8000/health  # Should return {"status": "healthy"}
  ```
- [ ] **Database Connectivity**: All databases accessible
  ```bash
  curl http://localhost:8000/health/detailed  # Should show all services healthy
  ```

#### **Core Functionality**
- [ ] **Memory Storage**: Can store memories
  ```bash
  curl -X POST http://localhost:8000/v1/memory/store \
    -H "Content-Type: application/json" \
    -d '{"text": "Test memory", "agent_id": "test"}'
  ```
- [ ] **Memory Retrieval**: Can search memories
  ```bash
  curl -X POST http://localhost:8000/v1/memory/search \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "agent_id": "test"}'
  ```
- [ ] **Embedding Generation**: Models working correctly
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; print(SentenceTransformer('intfloat/e5-large-v2').encode('test').shape)"
  ```

#### **Testing Suite**
- [ ] **Unit Tests**: All pass
  ```bash
  pytest tests/unit/ -v
  ```
- [ ] **Integration Tests**: All pass
  ```bash
  pytest tests/integration/ -v
  ```
- [ ] **MCP Tests**: MCP functionality verified
  ```bash
  pytest tests/test_mcp_integration.py -v
  ```
- [ ] **Trading Safety Tests**: CRITICAL - must pass
  ```bash
  pytest tests/test_mcp_trading_safety.py -v
  ```
- [ ] **Performance Tests**: Meet benchmarks
  ```bash
  pytest tests/performance/ -v --benchmark-only
  ```

#### **Security Configuration**
- [ ] **Firewall**: Configured for local access only
  ```bash
  sudo ufw status  # Should show local ports allowed
  ```
- [ ] **Database Security**: Read-only users created
- [ ] **File Permissions**: Secure permissions on config files
- [ ] **SSL Certificates**: Local certificates generated (if needed)
- [ ] **Password Security**: Strong passwords set for all services

#### **Development Environment** (Optional)
- [ ] **IDE Configuration**: VS Code settings and extensions
- [ ] **Pre-commit Hooks**: Code quality tools configured
- [ ] **Development Scripts**: Helper scripts executable
- [ ] **Local Proxy**: Nginx configuration for development
- [ ] **Service Discovery**: Avahi configuration (Linux)

#### **Production Features** (Optional)
- [ ] **System Service**: systemd service configured and running
  ```bash
  sudo systemctl status tyra-memory-server
  ```
- [ ] **Automated Backups**: Backup timer enabled
  ```bash
  sudo systemctl status tyra-backup.timer
  ```
- [ ] **Health Monitoring**: Monitoring service running
  ```bash
  sudo systemctl status tyra-health-monitor.timer
  ```
- [ ] **Log Rotation**: Log rotation configured
- [ ] **Load Balancer**: Nginx configured (if needed)
- [ ] **SSL/TLS**: Production certificates (if needed)

#### **Optional Components**
- [ ] **Crawl4AI**: Installed and browser drivers working
  ```bash
  python -c "from crawl4ai import WebCrawler; print('✅ Crawl4AI available')"
  ```
- [ ] **GPU Support**: CUDA/MPS acceleration working
  ```bash
  python -c "import torch; print(f'GPU available: {torch.cuda.is_available() or torch.backends.mps.is_available()}')"
  ```
- [ ] **Monitoring Stack**: Prometheus/Grafana operational
- [ ] **Docker**: Alternative deployment method available

#### **Platform-Specific Verification**
- [ ] **Ubuntu 24.04**: PostgreSQL 16, Python 3.12, latest packages
- [ ] **Ubuntu 22.04**: PostgreSQL 15, Python 3.12 via deadsnakes
- [ ] **Ubuntu 20.04**: Manual pgvector build, compatibility packages
- [ ] **macOS**: Homebrew packages, Apple Silicon optimization
- [ ] **Windows WSL2**: Linux compatibility layer functional

#### **Final Integration Tests**

Run this comprehensive verification script:

```bash
# Create final verification script
cat > scripts/verify-installation.sh << 'EOF'
#!/bin/bash
set -e

echo "🔍 Comprehensive Tyra MCP Memory Server Verification"
echo "=================================================="

errors=0

# Function to check and report
check_command() {
    local description="$1"
    local command="$2"
    local expected="$3"
    
    echo -n "Checking $description... "
    
    if output=$(eval "$command" 2>&1); then
        if [[ -z "$expected" ]] || echo "$output" | grep -q "$expected"; then
            echo "✅ PASS"
        else
            echo "❌ FAIL (unexpected output)"
            echo "  Expected: $expected"
            echo "  Got: $output"
            ((errors++))
        fi
    else
        echo "❌ FAIL (command failed)"
        echo "  Command: $command"
        echo "  Error: $output"
        ((errors++))
    fi
}

echo -e "\n📊 Database Services"
check_command "PostgreSQL service" "sudo systemctl is-active postgresql" "active"
check_command "Redis service" "sudo systemctl is-active redis-server" "active"
check_command "Memgraph service" "sudo systemctl is-active memgraph" "active"

echo -e "\n🔌 Database Connectivity"
check_command "PostgreSQL connection" "pg_isready -h localhost -p 5432" "accepting connections"
check_command "Redis connection" "redis-cli ping" "PONG"
check_command "Memgraph connection" "timeout 5 mgconsole --host localhost --port 7687 -c 'RETURN 1;'" ""

echo -e "\n🐍 Python Environment"
check_command "Python version" "python --version" "Python 3.1"
check_command "Virtual environment" "which python" "venv"
check_command "FastAPI installed" "python -c 'import fastapi'" ""
check_command "AsyncPG installed" "python -c 'import asyncpg'" ""
check_command "Redis package" "python -c 'import redis'" ""

echo -e "\n🧠 Embedding Models"
check_command "Sentence Transformers" "python -c 'import sentence_transformers'" ""
check_command "Primary model" "python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"intfloat/e5-large-v2\")'" ""
check_command "Fallback model" "python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"sentence-transformers/all-MiniLM-L12-v2\")'" ""

echo -e "\n🚀 Tyra API Services"
# Start server in background if not running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Starting Tyra server for testing..."
    source venv/bin/activate
    python main.py &
    SERVER_PID=$!
    sleep 10
fi

check_command "Health endpoint" "curl -s http://localhost:8000/health" "healthy"
check_command "Detailed health" "curl -s http://localhost:8000/health/detailed" "postgresql"

# Test memory operations
check_command "Memory storage" "curl -s -X POST http://localhost:8000/v1/memory/store -H 'Content-Type: application/json' -d '{\"text\":\"Test memory\",\"agent_id\":\"test\"}'" ""
check_command "Memory search" "curl -s -X POST http://localhost:8000/v1/memory/search -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"agent_id\":\"test\"}'" ""

# Stop test server if we started it
if [[ -n "$SERVER_PID" ]]; then
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
fi

echo -e "\n🧪 Test Suite"
if command -v pytest > /dev/null; then
    check_command "Unit tests" "pytest tests/unit/ -x --tb=no -q" ""
    check_command "Integration tests" "pytest tests/integration/ -x --tb=no -q" ""
    check_command "Trading safety tests" "pytest tests/test_mcp_trading_safety.py -x --tb=no -q" ""
else
    echo "⚠️  pytest not available, skipping test suite"
fi

echo -e "\n📋 Summary"
echo "========"
if [[ $errors -eq 0 ]]; then
    echo "✅ All checks passed! Installation verified successfully."
    echo ""
    echo "🎉 Your Tyra MCP Memory Server is ready for use!"
    echo ""
    echo "Next steps:"
    echo "1. Configure for your specific use case"
    echo "2. Set up monitoring and backups"
    echo "3. Integrate with Claude or other MCP clients"
    echo "4. Review security settings for production"
    exit 0
else
    echo "❌ $errors check(s) failed. Please review the errors above."
    echo ""
    echo "Common solutions:"
    echo "1. Check service status: sudo systemctl status [service-name]"
    echo "2. Review logs: journalctl -u [service-name] -f"
    echo "3. Verify configuration files"
    echo "4. Check the troubleshooting section in INSTALLATION.md"
    exit 1
fi
EOF

chmod +x scripts/verify-installation.sh

# Run verification
./scripts/verify-installation.sh
```

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