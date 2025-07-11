# 🚀 Setup Guide - Tyra MCP Memory Server

This guide covers the unified setup process for the Tyra MCP Memory Server across different environments.

## 📋 Overview

The project now uses a **single unified setup script** (`setup.sh`) that handles all setup scenarios:

- **Development**: Local development with Docker databases
- **Production**: Production deployment with native databases and SSL
- **Testing**: Isolated testing environment

## ⚠️ **PREREQUISITES - MANUAL MODEL INSTALLATION REQUIRED**

**CRITICAL: You must download models manually before setup**

```bash
# 1. Install HuggingFace CLI
pip install huggingface-hub
git lfs install

# 2. Download required models (~1.6GB total)
mkdir -p ./models/embeddings ./models/cross-encoders

# Primary embedding model
huggingface-cli download intfloat/e5-large-v2 \
  --local-dir ./models/embeddings/e5-large-v2 \
  --local-dir-use-symlinks False

# Fallback embedding model  
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 \
  --local-dir ./models/embeddings/all-MiniLM-L12-v2 \
  --local-dir-use-symlinks False

# Cross-encoder for reranking
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2 \
  --local-dir-use-symlinks False

# 3. Verify models
python scripts/test_model_pipeline.py
```

## 🔧 Quick Setup

### Development Environment
```bash
# Basic development setup (after models are downloaded)
./setup.sh

# With custom database settings
POSTGRES_PASSWORD=mypassword ./setup.sh --env development
```

### Production Environment
```bash
# Production setup with SSL
./setup.sh --env production --domain yourdomain.com

# Production without SSL
./setup.sh --env production --skip-ssl
```

### Testing Environment
```bash
# Testing setup (isolated)
./setup.sh --env testing --skip-deps
```

## 🎛️ Setup Options

### Command Line Arguments
```bash
./setup.sh [OPTIONS]

Options:
    --env ENV               Environment (development|production|testing)
    --skip-deps             Skip system dependency installation
    --skip-ssl              Skip SSL/TLS setup
    --skip-monitoring       Skip monitoring setup
    --no-auto-start         Don't auto-start services
    --domain DOMAIN         Domain name for SSL certificates
    --help                  Show help message
```

### Environment Variables
```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=secure_password
REDIS_HOST=localhost
MEMGRAPH_HOST=localhost

# Feature Flags
SSL_ENABLED=true
MONITORING_ENABLED=true
AUTO_START=true
SKIP_DEPS=false
```

## 🏗️ What the Setup Script Does

### 1. System Requirements Check
- Verifies Python 3.11+ is installed
- Checks for required system commands
- Validates operating system compatibility

### 2. System Dependencies Installation
- Installs build tools and development packages
- Sets up database clients (PostgreSQL, Redis)
- Installs web server (Nginx) and SSL tools (Certbot)
- Configures system packages based on OS (apt/yum/brew)

### 3. Python Environment Setup
- Creates virtual environment if needed
- Installs/updates Poetry package manager
- Installs Python dependencies from pyproject.toml
- Configures development tools

### 4. Directory Structure Creation
- Creates required directories (logs, data, monitoring)
- Sets up proper permissions
- Creates model cache and backup directories

### 5. Configuration Management
- Generates `.env` file from template
- Creates security secrets (API keys, JWT tokens)
- Sets up environment-specific configurations

### 6. Database Setup
- **Development**: Starts Docker containers for all databases
- **Production**: Validates connections to existing databases
- **Testing**: Starts isolated test databases
- Runs database migrations

### 7. Security Setup (Production)
- Generates SSL certificates (Let's Encrypt or self-signed)
- Configures Nginx with SSL
- Sets up secure file permissions

### 8. Service Management (Production)
- Installs systemd service files
- Configures auto-start on boot
- Sets up service monitoring

### 9. Monitoring Setup (Optional)
- Deploys Prometheus, Grafana, and Alertmanager
- Configures dashboards and alerts
- Sets up log aggregation

### 10. Validation and Testing
- Tests database connections
- Validates configuration files
- Runs basic functionality tests

## 🔍 Troubleshooting

### Common Issues

#### Permission Errors
```bash
# Fix script permissions
chmod +x setup.sh

# Run with sudo if needed (production)
sudo ./setup.sh --env production
```

#### Database Connection Issues
```bash
# Check database services
docker-compose ps

# Test connections manually
psql -h localhost -U tyra -d tyra_memory
redis-cli ping
```

#### Python Environment Issues
```bash
# Clean virtual environment
rm -rf venv
./setup.sh --env development
```

#### Missing Dependencies
```bash
# Force dependency installation
./setup.sh --env development --skip-deps=false
```

### Debug Mode
```bash
# Run with verbose logging
TYRA_DEBUG=true ./setup.sh --env development
```

## 📊 Environment Comparison

| Feature | Development | Production | Testing |
|---------|-------------|------------|---------|
| **Databases** | Docker containers | Native instances | Docker containers |
| **SSL/TLS** | Self-signed | Let's Encrypt | Disabled |
| **Monitoring** | Optional | Enabled | Disabled |
| **Auto-start** | Manual | systemd service | Manual |
| **Log Level** | DEBUG | INFO | DEBUG |
| **Dependencies** | Full install | Minimal | Minimal |

## 🚨 Migration from Old Setup

If you were using the old separate setup scripts:

### Before (Multiple Scripts)
```bash
# OLD WAY - Don't use anymore
./scripts/setup.sh
./scripts/deploy/setup.sh
./scripts/install_dependencies.sh
```

### After (Unified Script)
```bash
# NEW WAY - Use this
./setup.sh --env development
./setup.sh --env production --domain yourdomain.com
```

### What Changed
- ✅ **Consolidated**: One script handles all environments
- ✅ **Simplified**: Fewer commands to remember
- ✅ **Consistent**: Same interface across environments
- ✅ **Robust**: Better error handling and validation
- ✅ **Comprehensive**: Includes all setup steps

## 🔐 Security Considerations

### Development
- Uses self-signed SSL certificates
- Generates secure random passwords
- Creates proper file permissions

### Production
- Integrates with Let's Encrypt for SSL
- Creates system service accounts
- Configures firewall rules (manual)
- Sets up log rotation

### Best Practices
1. Always run with specific environment flag
2. Review generated `.env` file before use
3. Change default passwords for production
4. Enable monitoring in production
5. Test setup in development first

## 📝 Next Steps After Setup

1. **Review Configuration**
   ```bash
   # Check environment file
   cat .env
   
   # Verify config files
   ls -la config/
   ```

2. **Start Services**
   ```bash
   # Development
   source venv/bin/activate
   python main.py
   
   # Production
   sudo systemctl start tyra-memory-server
   ```

3. **Test Installation**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Run test suite
   pytest tests/
   ```

4. **Access Applications**
   - **API**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs
   - **Monitoring**: http://localhost:3000 (if enabled)

---

For more detailed information, see:
- [Installation Guide](../INSTALLATION.md)
- [Configuration Reference](../CONFIGURATION.md)
- [Container Usage](CONTAINERS.md)