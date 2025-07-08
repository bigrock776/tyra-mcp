#!/bin/bash
# Tyra Advanced Memory System Setup Script
# Automated setup for development and production environments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
CONFIG_DIR="${PROJECT_ROOT}/config"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Default values
ENVIRONMENT=${TYRA_ENV:-development}
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}
POSTGRES_DB=${POSTGRES_DB:-tyra_memory}
POSTGRES_USER=${POSTGRES_USER:-tyra}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-tyra_secure_password}
REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6379}
MEMGRAPH_HOST=${MEMGRAPH_HOST:-localhost}
MEMGRAPH_PORT=${MEMGRAPH_PORT:-7687}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi

    local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    local required_version="3.8"

    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "Python 3.8+ is required, found $python_version"
        exit 1
    fi

    log_success "Python version check passed"

    # Check if pip is available
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi

    # Check if Docker is available (optional)
    if command -v docker &> /dev/null; then
        log_success "Docker is available"
    else
        log_warning "Docker not found - database services must be run separately"
    fi
}

create_directories() {
    log_info "Creating project directories..."

    mkdir -p "${LOGS_DIR}"
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/backups"

    log_success "Directories created"
}

setup_python_environment() {
    log_info "Setting up Python environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "${PROJECT_ROOT}/venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "${PROJECT_ROOT}/venv"
    fi

    # Activate virtual environment
    source "${PROJECT_ROOT}/venv/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
        log_info "Installing Python dependencies..."
        pip install -r "${PROJECT_ROOT}/requirements.txt"
    fi

    # Install additional ML dependencies based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        # Production optimized packages
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        # Development packages with CUDA support if available
        if command -v nvidia-smi &> /dev/null; then
            log_info "NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            log_info "No GPU detected, installing CPU-only PyTorch..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi

    log_success "Python environment setup complete"
}

create_env_file() {
    log_info "Creating environment configuration..."

    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Tyra Advanced Memory System Environment Configuration
# Generated on $(date)

# Environment
TYRA_ENV=${ENVIRONMENT}
TYRA_DEBUG=false
TYRA_LOG_LEVEL=INFO

# Database Configuration
POSTGRES_HOST=${POSTGRES_HOST}
POSTGRES_PORT=${POSTGRES_PORT}
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_SSL_MODE=prefer

# Redis Configuration
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}
REDIS_PASSWORD=

# Memgraph Configuration
MEMGRAPH_HOST=${MEMGRAPH_HOST}
MEMGRAPH_PORT=${MEMGRAPH_PORT}
MEMGRAPH_USER=memgraph
MEMGRAPH_PASSWORD=
MEMGRAPH_ENCRYPTED=false

# OpenAI Configuration (for fallback embedding provider)
OPENAI_API_KEY=

# Security
SECRET_KEY=$(openssl rand -hex 32)

# Development Settings
TYRA_HOT_RELOAD=false
EOF
        log_success "Environment file created at $ENV_FILE"
    else
        log_warning "Environment file already exists at $ENV_FILE"
    fi
}

setup_databases() {
    log_info "Setting up databases..."

    # Ask user preference for database setup
    echo "Choose database setup method:"
    echo "1. Docker (recommended for development)"
    echo "2. Native installation (recommended for production)"
    echo "3. Skip (databases already installed)"
    read -p "Enter choice (1-3): " choice

    case $choice in
        1)
            setup_databases_docker
            ;;
        2)
            setup_databases_native
            ;;
        3)
            log_info "Skipping database setup"
            ;;
        *)
            log_warning "Invalid choice, skipping database setup"
            ;;
    esac
}

setup_databases_docker() {
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        log_info "Using Docker for database setup..."

        # Create docker-compose file for databases
        cat > "${PROJECT_ROOT}/docker-compose.dev.yml" << EOF
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "${REDIS_PORT}:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  memgraph:
    image: memgraph/memgraph:latest
    ports:
      - "${MEMGRAPH_PORT}:7687"
    volumes:
      - memgraph_data:/var/lib/memgraph
    environment:
      - MEMGRAPH=--log-level=TRACE
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "7687"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  postgres_data:
  redis_data:
  memgraph_data:
EOF

        # Start databases
        log_info "Starting database services..."
        docker-compose -f "${PROJECT_ROOT}/docker-compose.dev.yml" up -d

        # Wait for services to be healthy
        log_info "Waiting for databases to be ready..."
        sleep 10

        # Check if services are healthy
        if docker-compose -f "${PROJECT_ROOT}/docker-compose.dev.yml" ps | grep -q "healthy"; then
            log_success "Database services are running and healthy"
        else
            log_warning "Some database services may not be fully ready yet"
        fi

    else
        log_error "Docker not available. Please install Docker and Docker Compose first."
        exit 1
    fi
}

setup_databases_native() {
    log_info "Installing databases natively..."

    # PostgreSQL with pgvector
    if [ -f "${PROJECT_ROOT}/scripts/install_pgvector.sh" ]; then
        log_info "Installing PostgreSQL with pgvector..."
        bash "${PROJECT_ROOT}/scripts/install_pgvector.sh"
    else
        log_error "PostgreSQL installation script not found"
        exit 1
    fi

    # Redis
    if [ -f "${PROJECT_ROOT}/scripts/install_redis.sh" ]; then
        log_info "Installing Redis..."
        bash "${PROJECT_ROOT}/scripts/install_redis.sh"
    else
        log_error "Redis installation script not found"
        exit 1
    fi

    # Memgraph
    if [ -f "${PROJECT_ROOT}/scripts/install_memgraph.sh" ]; then
        log_info "Installing Memgraph..."
        bash "${PROJECT_ROOT}/scripts/install_memgraph.sh"
    else
        log_error "Memgraph installation script not found"
        exit 1
    fi

    log_success "Native database installation complete"
}

create_postgres_init_script() {
    log_info "Creating PostgreSQL initialization script..."

    cat > "${PROJECT_ROOT}/scripts/init_postgres.sql" << 'EOF'
-- PostgreSQL initialization script for Tyra Memory System
-- Creates necessary extensions and initial schema

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable other useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create initial schema
CREATE SCHEMA IF NOT EXISTS memory;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA memory TO tyra;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA memory TO tyra;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA memory TO tyra;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA memory GRANT ALL ON TABLES TO tyra;
ALTER DEFAULT PRIVILEGES IN SCHEMA memory GRANT ALL ON SEQUENCES TO tyra;

-- Create logging table for monitoring
CREATE TABLE IF NOT EXISTS memory.system_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    level VARCHAR(10) NOT NULL,
    component VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- Create index for efficient log queries
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON memory.system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON memory.system_logs(component);
EOF

    log_success "PostgreSQL initialization script created"
}

test_connections() {
    log_info "Testing database connections..."

    # Test PostgreSQL
    if command -v psql &> /dev/null; then
        if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" &> /dev/null; then
            log_success "PostgreSQL connection successful"
        else
            log_error "PostgreSQL connection failed"
        fi
    else
        log_warning "psql not available, skipping PostgreSQL connection test"
    fi

    # Test Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &> /dev/null; then
            log_success "Redis connection successful"
        else
            log_error "Redis connection failed"
        fi
    else
        log_warning "redis-cli not available, skipping Redis connection test"
    fi

    # Test Memgraph (using nc for port check)
    if command -v nc &> /dev/null; then
        if nc -z "$MEMGRAPH_HOST" "$MEMGRAPH_PORT" &> /dev/null; then
            log_success "Memgraph port is accessible"
        else
            log_error "Memgraph connection failed"
        fi
    else
        log_warning "nc not available, skipping Memgraph connection test"
    fi
}

download_models() {
    log_info "Pre-downloading ML models..."

    # Activate virtual environment
    source "${PROJECT_ROOT}/venv/bin/activate"

    # Create model download script
    cat > "${PROJECT_ROOT}/scripts/download_models.py" << 'EOF'
#!/usr/bin/env python3
"""Download and cache ML models for Tyra Memory System."""

import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

def download_embedding_models():
    """Download embedding models."""
    models = [
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]

    for model_name in models:
        print(f"Downloading {model_name}...")
        try:
            model = SentenceTransformer(model_name)
            print(f"✓ {model_name} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")

def download_reranker_models():
    """Download reranker models."""
    models = [
        "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ]

    for model_name in models:
        print(f"Downloading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            print(f"✓ {model_name} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")

if __name__ == "__main__":
    print("Downloading embedding models...")
    download_embedding_models()

    print("\nDownloading reranker models...")
    download_reranker_models()

    print("\nModel download complete!")
EOF

    python3 "${PROJECT_ROOT}/scripts/download_models.py"
    log_success "Model download complete"
}

create_systemd_service() {
    if [ "$ENVIRONMENT" = "production" ]; then
        log_info "Creating systemd service for production..."

        cat > "${PROJECT_ROOT}/scripts/tyra-memory.service" << EOF
[Unit]
Description=Tyra Advanced Memory MCP Server
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=tyra
Group=tyra
WorkingDirectory=${PROJECT_ROOT}
Environment=PATH=${PROJECT_ROOT}/venv/bin
ExecStart=${PROJECT_ROOT}/venv/bin/python ${PROJECT_ROOT}/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tyra-memory

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${PROJECT_ROOT}/logs ${PROJECT_ROOT}/data

[Install]
WantedBy=multi-user.target
EOF

        log_success "Systemd service file created at scripts/tyra-memory.service"
        log_info "To install the service:"
        log_info "  sudo cp scripts/tyra-memory.service /etc/systemd/system/"
        log_info "  sudo systemctl daemon-reload"
        log_info "  sudo systemctl enable tyra-memory"
        log_info "  sudo systemctl start tyra-memory"
    fi
}

print_next_steps() {
    log_success "Setup complete!"
    echo
    log_info "Next steps:"
    echo "1. Review and update the configuration in .env"
    echo "2. Ensure all databases are running and accessible"

    if [ "$ENVIRONMENT" = "development" ]; then
        echo "3. Start the development server:"
        echo "   source venv/bin/activate"
        echo "   python main.py"
    else
        echo "3. Install and start the systemd service (production)"
        echo "   sudo cp scripts/tyra-memory.service /etc/systemd/system/"
        echo "   sudo systemctl enable --now tyra-memory"
    fi

    echo
    log_info "MCP Configuration for Claude Desktop:"
    cat << EOF
{
  "mcpServers": {
    "tyra-memory": {
      "command": "python",
      "args": ["${PROJECT_ROOT}/main.py"],
      "env": {
        "TYRA_ENV": "${ENVIRONMENT}"
      }
    }
  }
}
EOF

    echo
    log_info "For troubleshooting, check logs at: ${LOGS_DIR}/tyra-memory.log"
}

# Main execution
main() {
    log_info "Starting Tyra Advanced Memory System setup..."
    log_info "Environment: $ENVIRONMENT"

    check_requirements
    create_directories
    create_env_file
    setup_python_environment
    create_postgres_init_script
    setup_databases
    test_connections
    download_models
    create_systemd_service
    print_next_steps
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --postgres-host)
            POSTGRES_HOST="$2"
            shift 2
            ;;
        --postgres-password)
            POSTGRES_PASSWORD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --env ENV                 Set environment (development/production)"
            echo "  --postgres-host HOST      PostgreSQL host (default: localhost)"
            echo "  --postgres-password PASS  PostgreSQL password"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main setup
main
