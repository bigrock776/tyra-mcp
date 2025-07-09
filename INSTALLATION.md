# 📦 Installation Guide - Tyra MCP Memory Server

## 🎯 Quick Start

For most users, this is all you need:

```bash
# Clone the repository
git clone <repository-url>
cd tyra-mcp-memory-server

# Run unified setup script
./setup.sh --env development

# Start the server
source venv/bin/activate
python main.py
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11+ (3.12 recommended)
- **RAM**: 8GB minimum, 16GB recommended for production
- **Disk**: 10GB free space minimum
- **OS**: Linux (Ubuntu 22.04+), macOS 12+, Windows 11 with WSL2

### Required Services
- **PostgreSQL**: 15+ with pgvector extension
- **Redis**: 6.0+ for caching
- **Memgraph**: 2.0+ for knowledge graphs

### Optional Dependencies
- **Docker**: For containerized deployment
- **Nginx**: For production load balancing
- **Certbot**: For SSL certificates

## 🚀 Installation Methods

### Method 1: Automated Setup (Recommended)

The unified setup script handles everything automatically:

#### Development Environment
```bash
# Basic development setup
./setup.sh --env development

# With custom settings
POSTGRES_PASSWORD=mypassword ./setup.sh --env development
```

#### Production Environment
```bash
# Production setup with SSL
sudo ./setup.sh --env production --domain yourdomain.com

# Production without SSL
sudo ./setup.sh --env production --skip-ssl
```

#### Testing Environment
```bash
# Isolated testing setup
./setup.sh --env testing
```

### Method 2: Docker Compose

For containerized deployment:

```bash
# Development with Docker
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# Production with Docker
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d

# With monitoring
docker-compose --profile monitoring up -d
```

### Method 3: Manual Installation

For custom setups or when you need full control:

#### Step 1: Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

#### Step 2: Install Dependencies
```bash
# Install Poetry (recommended)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies with Poetry
poetry install

# OR install with pip
pip install -r requirements.txt
```

#### Step 3: Database Setup
```bash
# Option A: Docker databases (recommended for development)
docker-compose -f docker/docker-compose.yml up -d postgres redis memgraph

# Option B: Native installation (see Database Installation section)
```

#### Step 4: Configuration
```bash
# Create environment file
cp .env.example .env

# Edit configuration
nano .env

# Create configuration secrets
mkdir -p config/local
openssl rand -hex 32 > config/local/secret_key
```

#### Step 5: Database Migration
```bash
# Run database migrations
python -m src.migrations.run_migrations

# OR use migration script
./scripts/deploy/migrate.sh
```

#### Step 6: Start Services
```bash
# Start MCP server
python main.py

# OR start API server
python -m uvicorn src.api.app:app --reload
```

## 🗄️ Database Installation

### PostgreSQL with pgvector

#### Ubuntu/Debian
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector
sudo apt install postgresql-15-pgvector

# Create database and user
sudo -u postgres psql
CREATE DATABASE tyra_memory;
CREATE USER tyra WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE tyra_memory TO tyra;
CREATE EXTENSION vector;
\q
```

#### macOS
```bash
# Install PostgreSQL
brew install postgresql@15

# Install pgvector
brew install pgvector

# Start PostgreSQL
brew services start postgresql@15

# Create database
createdb tyra_memory
psql tyra_memory -c "CREATE EXTENSION vector;"
```

#### Docker (Recommended)
```bash
# Use pre-configured PostgreSQL with pgvector
docker run -d \
  --name tyra-postgres \
  -e POSTGRES_DB=tyra_memory \
  -e POSTGRES_USER=tyra \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Redis

#### Ubuntu/Debian
```bash
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### macOS
```bash
brew install redis
brew services start redis
```

#### Docker
```bash
docker run -d \
  --name tyra-redis \
  -p 6379:6379 \
  redis:7-alpine
```

### Memgraph

#### Ubuntu/Debian
```bash
# Install Memgraph
curl -O https://download.memgraph.com/memgraph/v2.11.0/ubuntu-22.04/memgraph_2.11.0-1_amd64.deb
sudo dpkg -i memgraph_2.11.0-1_amd64.deb

# Start Memgraph
sudo systemctl start memgraph
sudo systemctl enable memgraph
```

#### macOS
```bash
brew install memgraph
brew services start memgraph
```

#### Docker
```bash
docker run -d \
  --name tyra-memgraph \
  -p 7687:7687 \
  -p 7444:7444 \
  memgraph/memgraph:latest
```

## 🔧 Configuration

### Environment Variables

Create and edit `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit with your settings
nano .env
```

**Essential Configuration:**
```env
# Database Connections
DATABASE_URL=postgresql://tyra:password@localhost:5432/tyra_memory
REDIS_URL=redis://localhost:6379/0
MEMGRAPH_URL=bolt://localhost:7687

# Embedding Models
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=auto

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=false

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
```

### YAML Configuration

Edit configuration files in `config/`:

```bash
# Main configuration
nano config/config.yaml

# Provider settings
nano config/providers.yaml

# Model configurations
nano config/models.yaml

# Observability settings
nano config/observability.yaml
```

## 🧪 Testing Installation

### Health Check
```bash
# Check if server is running
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

### Database Connections
```bash
# Test PostgreSQL
psql -h localhost -U tyra -d tyra_memory -c "SELECT version();"

# Test Redis
redis-cli ping

# Test Memgraph
echo "RETURN 1;" | mgconsole --host localhost --port 7687
```

### Run Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### MCP Integration Test
```bash
# Test MCP tools
python -m pytest tests/test_mcp_integration.py -v

# Manual MCP test
python main.py --test-mcp
```

## 🚀 Production Deployment

### System Service Setup

Create systemd service:

```bash
# Create service file
sudo nano /etc/systemd/system/tyra-memory-server.service
```

```ini
[Unit]
Description=Tyra MCP Memory Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=tyra
Group=tyra
WorkingDirectory=/opt/tyra-memory-server
Environment=PATH=/opt/tyra-memory-server/venv/bin
ExecStart=/opt/tyra-memory-server/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable tyra-memory-server
sudo systemctl start tyra-memory-server
```

### SSL/TLS Setup

#### Let's Encrypt (Production)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

#### Self-signed (Development)
```bash
# Generate certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/tyra.key \
  -out /etc/ssl/certs/tyra.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

### Load Balancer Setup

Configure Nginx:

```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/tyra-memory-server
```

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/tyra-memory-server /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

### Monitoring Setup

Enable monitoring stack:

```bash
# Setup monitoring
./setup.sh --env production --monitoring

# Access monitoring
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
# Alertmanager: http://localhost:9093
```

## 🔧 Troubleshooting

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python3 --version

# Install specific Python version (Ubuntu)
sudo apt install python3.11 python3.11-venv python3.11-dev

# Use specific Python version
python3.11 -m venv venv
```

#### Database Connection Issues
```bash
# Check if databases are running
sudo systemctl status postgresql
sudo systemctl status redis
sudo systemctl status memgraph

# Test connections
pg_isready -h localhost -p 5432
redis-cli ping
```

#### Permission Issues
```bash
# Fix file permissions
chmod +x setup.sh
chmod +x scripts/deploy/*.sh

# Fix directory permissions
sudo chown -R $USER:$USER .
```

#### Port Conflicts
```bash
# Check if ports are in use
sudo netstat -tulpn | grep :8000
sudo netstat -tulpn | grep :5432
sudo netstat -tulpn | grep :6379
sudo netstat -tulpn | grep :7687

# Kill conflicting processes
sudo kill -9 $(sudo lsof -t -i:8000)
```

#### Memory Issues
```bash
# Check memory usage
free -h
htop

# Optimize for low memory
export EMBEDDINGS_DEVICE=cpu
export EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
```

#### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in /etc/ssl/certs/tyra.crt -text -noout

# Renew Let's Encrypt certificate
sudo certbot renew --dry-run
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Set debug environment
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug
python main.py --debug

# Check logs
tail -f logs/memory-server.log
```

### Log Analysis

```bash
# View recent logs
journalctl -u tyra-memory-server -f

# Check error logs
grep -i error logs/memory-server.log

# Monitor performance
tail -f logs/performance.log
```

## 📊 Performance Optimization

### Memory Optimization
```bash
# Use CPU-only embeddings for low memory
export EMBEDDINGS_DEVICE=cpu

# Reduce batch sizes
export EMBEDDINGS_BATCH_SIZE=16

# Enable memory monitoring
export MEMORY_MONITORING=true
```

### Database Optimization
```bash
# Optimize PostgreSQL
sudo nano /etc/postgresql/15/main/postgresql.conf

# Key settings:
# shared_buffers = 256MB
# effective_cache_size = 1GB
# work_mem = 4MB
# maintenance_work_mem = 64MB
```

### Caching Optimization
```bash
# Enable aggressive caching
export CACHE_ENABLED=true
export CACHE_TTL_EMBEDDINGS=86400
export CACHE_TTL_SEARCH=3600
```

## 🔄 Updates and Maintenance

### Update Process
```bash
# Backup current installation
./scripts/deploy/backup.sh

# Update code
git pull origin main

# Run update script
./scripts/deploy/update.sh

# Verify update
curl http://localhost:8000/health
```

### Rollback Process
```bash
# Rollback to previous version
./scripts/deploy/rollback.sh --target previous

# Rollback to specific version
./scripts/deploy/rollback.sh --target v1.2.3
```

### Maintenance Tasks
```bash
# Database maintenance
./scripts/deploy/migrate.sh --maintenance

# Clear caches
redis-cli flushall

# Rotate logs
sudo logrotate -f /etc/logrotate.d/tyra-memory-server
```

## 🆘 Getting Help

### Documentation
- [Configuration Guide](CONFIGURATION.md)
- [Container Guide](docs/CONTAINERS.md)
- [Architecture Overview](ARCHITECTURE.md)
- [API Documentation](API.md)

### Support Channels
- GitHub Issues: Report bugs and request features
- Documentation: Comprehensive guides and examples
- Community: Share experiences and solutions

### Diagnostic Information

When reporting issues, include:

```bash
# System information
uname -a
python3 --version
docker --version

# Service status
systemctl status tyra-memory-server
docker-compose ps

# Recent logs
tail -50 logs/memory-server.log

# Configuration check
python -c "from src.core.utils.config import get_config; print(get_config())"
```

---

🎉 **Congratulations!** You should now have a fully functional Tyra MCP Memory Server installation. 

For next steps, see the [Configuration Guide](CONFIGURATION.md) to customize your setup.