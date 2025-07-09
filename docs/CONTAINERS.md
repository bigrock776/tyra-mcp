# 🐳 Container Usage Guide - Tyra MCP Memory Server

## 📖 Overview

This guide covers the containerized deployment of the Tyra MCP Memory Server using Docker and Docker Compose. The system provides multiple deployment options optimized for different use cases.

## 🏗️ Architecture

### Multi-Stage Dockerfile

The project uses a sophisticated multi-stage Dockerfile with the following targets:

| Stage | Purpose | Image Size | Use Case |
|-------|---------|------------|----------|
| `python-base` | Base Python environment | ~200MB | Foundation for other stages |
| `builder` | Dependencies builder | ~500MB | Build stage only |
| `development` | Development environment | ~800MB | Local development |
| `production` | Production runtime | ~400MB | Production deployment |
| `mcp-server` | MCP-only lightweight | ~300MB | MCP protocol only |

### Container Features

- **Security**: Non-root user execution, minimal attack surface
- **Health Checks**: Built-in health monitoring for all services
- **Resource Limits**: Configurable CPU/memory constraints
- **Multi-arch Support**: Compatible with x86_64 and ARM64
- **Observability**: Integrated monitoring and tracing

## 🚀 Quick Start

### 1. Basic Deployment

```bash
# Clone repository
git clone <repository-url>
cd tyra-mcp-memory-server

# Start core services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs memory-server
```

### 2. Development Environment

```bash
# Build development image
docker-compose build --target development memory-server

# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access development features
curl http://localhost:8000/docs  # API documentation
curl http://localhost:8000/health  # Health check
```

### 3. Production Deployment

```bash
# Set environment variables
export VERSION=1.0.0
export POSTGRES_PASSWORD=secure_password_here
export SECRET_KEY=your_secret_key_here

# Build production image
docker-compose build --target production

# Deploy to production
docker-compose up -d

# Enable monitoring (optional)
docker-compose --profile monitoring up -d
```

## 📋 Environment Configuration

### Required Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

### Critical Settings

```bash
# Security (Required for production)
POSTGRES_PASSWORD=your_secure_password
SECRET_KEY=your_secret_key_generate_with_openssl_rand_hex_32
API_KEY=your_api_key

# Performance
POSTGRES_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=50
WORKERS=4

# Features
CACHE_ENABLED=true
OBSERVABILITY_ENABLED=true
TRACING_ENABLED=true
```

## 🔧 Build Options

### Standard Builds

```bash
# Development build
docker build --target development -t tyra-memory-server:dev .

# Production build
docker build --target production -t tyra-memory-server:latest .

# MCP-only build
docker build --target mcp-server -t tyra-memory-server:mcp .
```

### Advanced Builds

```bash
# Build with version metadata
docker build \
  --build-arg VERSION=1.2.3 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  --target production \
  -t tyra-memory-server:1.2.3 .

# Multi-architecture build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target production \
  -t tyra-memory-server:latest .
```

## 🎛️ Service Profiles

The Docker Compose configuration includes several profiles for different deployment scenarios:

### Core Services (Default)

```bash
docker-compose up -d
```

Includes:
- `memory-server` - Main application server
- `mcp-server` - MCP protocol server
- `postgres` - PostgreSQL with pgvector
- `redis` - Redis cache
- `memgraph` - Graph database

### Monitoring Profile

```bash
docker-compose --profile monitoring up -d
```

Additional services:
- `prometheus` - Metrics collection
- `grafana` - Dashboards and visualization
- `jaeger` - Distributed tracing

### Backup Profile

```bash
docker-compose --profile backup up backup
```

Services:
- `backup` - Automated backup service

## 📊 Health Monitoring

### Health Check Endpoints

All services include comprehensive health checks:

```bash
# Application health
curl http://localhost:8000/health

# Detailed health status
curl http://localhost:8000/health/detailed

# Readiness check
curl http://localhost:8000/health/ready

# Database connectivity
docker-compose exec postgres pg_isready -U tyra -d tyra_memory
docker-compose exec redis redis-cli ping
```

### Health Check Configuration

Health checks are configured with appropriate timeouts and retry logic:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s      # Check every 30 seconds
  timeout: 10s       # 10 second timeout
  retries: 3         # Retry 3 times
  start_period: 40s  # Wait 40s before first check
```

## 🔒 Security Configuration

### Container Security

- **Non-root execution**: All application containers run as non-root user
- **Read-only filesystems**: Where applicable
- **Minimal base images**: Using slim and alpine variants
- **Secret management**: Environment variables for sensitive data

### Network Security

```yaml
networks:
  tyra-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

## 📈 Performance Optimization

### Database Configuration

PostgreSQL is optimized for vector operations:

```yaml
command:
  - postgres
  - -c shared_preload_libraries=vector
  - -c max_connections=200
  - -c shared_buffers=256MB
  - -c effective_cache_size=1GB
  - -c work_mem=4MB
```

### Redis Configuration

Redis is configured for optimal caching:

```yaml
command:
  - redis-server
  - --maxmemory 1gb
  - --maxmemory-policy allkeys-lru
  - --appendonly yes
```

### Memory Server Optimization

- Multi-worker Uvicorn setup
- Connection pooling
- Async request processing
- Intelligent caching strategies

## 📦 Volume Management

### Data Persistence

```bash
# View all volumes
docker volume ls | grep tyra

# Backup volume data
docker run --rm -v tyra_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_data.tar.gz -C /data .

# Restore volume data
docker run --rm -v tyra_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_data.tar.gz -C /data
```

### Volume Types

| Volume | Purpose | Backup Priority |
|--------|---------|----------------|
| `tyra_postgres_data` | Application data | Critical |
| `tyra_redis_data` | Cache data | Medium |
| `tyra_memgraph_data` | Graph data | High |
| `tyra_memory_logs` | Application logs | Low |
| `tyra_memory_cache` | HuggingFace models | Medium |

## 🔄 Scaling and Load Balancing

### Horizontal Scaling

```bash
# Scale memory server instances
docker-compose up -d --scale memory-server=3

# Load balance with nginx
docker-compose -f docker-compose.yml -f docker-compose.lb.yml up -d
```

### Resource Scaling

```bash
# Increase database resources
docker-compose up -d --scale postgres=1 --force-recreate postgres

# Monitor resource usage
docker stats
```

## 🛠️ Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker-compose logs memory-server

# Check resource usage
docker system df
docker system prune

# Verify configuration
docker-compose config
```

#### Database Connection Issues

```bash
# Test PostgreSQL connection
docker-compose exec postgres psql -U tyra -d tyra_memory -c "SELECT version();"

# Test Redis connection
docker-compose exec redis redis-cli ping

# Check network connectivity
docker-compose exec memory-server nc -zv postgres 5432
```

#### Performance Issues

```bash
# Monitor resource usage
docker stats memory-server

# Check application metrics
curl http://localhost:8000/v1/telemetry/metrics

# Review slow query logs
docker-compose logs postgres | grep "slow query"
```

### Debug Mode

```bash
# Enable debug logging
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up -d

# Access container shell
docker-compose exec memory-server /bin/bash

# Run manual tests
docker-compose exec memory-server python -m pytest tests/
```

## 📚 Additional Resources

### Container Management

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart specific service
docker-compose restart memory-server

# Update service
docker-compose pull memory-server
docker-compose up -d memory-server
```

### Image Management

```bash
# List images
docker images | grep tyra

# Remove old images
docker image prune -f

# Push to registry
docker tag tyra-memory-server:latest your-registry/tyra-memory-server:latest
docker push your-registry/tyra-memory-server:latest
```

### Monitoring Commands

```bash
# Real-time logs
docker-compose logs -f memory-server

# Resource monitoring
docker-compose exec memory-server top

# Network inspection
docker network inspect tyra_tyra-network
```

## 🔐 Production Checklist

### Security

- [ ] Change default passwords
- [ ] Generate secure secret keys
- [ ] Enable TLS/SSL certificates
- [ ] Configure firewall rules
- [ ] Set up log rotation
- [ ] Enable security scanning

### Performance

- [ ] Tune resource limits
- [ ] Configure log levels
- [ ] Set up monitoring alerts
- [ ] Test backup/restore procedures
- [ ] Validate health checks
- [ ] Benchmark performance

### Operations

- [ ] Set up automated backups
- [ ] Configure log aggregation
- [ ] Set up alerting
- [ ] Document rollback procedures
- [ ] Test disaster recovery
- [ ] Monitor resource usage

## 🚨 Emergency Procedures

### Quick Recovery

```bash
# Emergency stop
docker-compose down

# Restore from backup
./scripts/deploy/rollback.sh --target latest-backup --type full

# Restart services
docker-compose up -d
```

### Data Recovery

```bash
# Restore database
docker-compose exec postgres psql -U tyra -d tyra_memory < backup.sql

# Clear corrupted cache
docker-compose exec redis redis-cli flushall

# Restart application
docker-compose restart memory-server
```

---

For additional help, refer to:
- [Installation Guide](../INSTALLATION.md)
- [Configuration Reference](../CONFIGURATION.md)
- [Architecture Overview](../ARCHITECTURE.md)
- [Troubleshooting Guide](README.md#troubleshooting)