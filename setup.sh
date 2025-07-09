#!/bin/bash

# =============================================================================
# Tyra MCP Memory Server - Unified Setup Script
# =============================================================================
# Comprehensive setup for development, production, and testing environments
# Consolidates functionality from scripts/setup.sh and scripts/deploy/setup.sh

set -euo pipefail

# =============================================================================
# Configuration and Constants
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_FILE="/tmp/tyra-setup-$(date +%Y%m%d-%H%M%S).log"

# Environment configuration
ENVIRONMENT="${TYRA_ENV:-development}"
DOMAIN="${DOMAIN:-localhost}"
SSL_ENABLED="${SSL_ENABLED:-false}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
MONITORING_ENABLED="${MONITORING_ENABLED:-false}"
AUTO_START="${AUTO_START:-true}"
SKIP_DEPS="${SKIP_DEPS:-false}"

# Database configuration
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-tyra_memory}"
POSTGRES_USER="${POSTGRES_USER:-tyra}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-tyra_secure_password}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
MEMGRAPH_HOST="${MEMGRAPH_HOST:-localhost}"
MEMGRAPH_PORT="${MEMGRAPH_PORT:-7687}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Logging and Output Functions
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

# =============================================================================
# System Requirements and Dependencies
# =============================================================================
check_system_requirements() {
    info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check required commands
    local required_commands=("curl" "git" "python3" "pip3")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Python version
    local python_version=$(python3 --version | cut -d' ' -f2)
    local python_major=$(echo $python_version | cut -d'.' -f1)
    local python_minor=$(echo $python_version | cut -d'.' -f2)
    
    if [[ $python_major -lt 3 || ($python_major -eq 3 && $python_minor -lt 11) ]]; then
        error "Python 3.11 or higher required. Found: $python_version"
        exit 1
    fi
    
    success "System requirements check passed"
}

install_system_dependencies() {
    if [[ "$SKIP_DEPS" == "true" ]]; then
        info "Skipping system dependencies installation"
        return
    fi
    
    info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        # Detect Linux distribution
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                curl \
                git \
                python3-dev \
                python3-pip \
                python3-venv \
                postgresql-client \
                redis-tools \
                jq \
                nginx \
                certbot \
                python3-certbot-nginx
        elif command -v yum &> /dev/null; then
            sudo yum update -y
            sudo yum install -y \
                gcc \
                gcc-c++ \
                curl \
                git \
                python3-devel \
                python3-pip \
                postgresql \
                redis \
                jq \
                nginx \
                certbot \
                python3-certbot-nginx
        else
            warning "Unknown Linux distribution, skipping system package installation"
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            brew install curl git python3 postgresql redis jq nginx certbot
        else
            warning "Homebrew not found, skipping system package installation"
        fi
    fi
    
    success "System dependencies installed"
}

# =============================================================================
# Python Environment Setup
# =============================================================================
setup_python_environment() {
    info "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Poetry if not available
    if ! command -v poetry &> /dev/null; then
        info "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Install dependencies
    if [[ -f "pyproject.toml" ]]; then
        info "Installing Python dependencies with Poetry..."
        poetry install
    elif [[ -f "requirements.txt" ]]; then
        info "Installing Python dependencies with pip..."
        pip install -r requirements.txt
    else
        error "No dependency file found (pyproject.toml or requirements.txt)"
        exit 1
    fi
    
    success "Python environment setup completed"
}

# =============================================================================
# Directory Structure
# =============================================================================
create_directories() {
    info "Creating project directories..."
    
    local directories=(
        "logs"
        "data"
        "data/models"
        "data/backups"
        "data/exports"
        "data/fixtures"
        "config/local"
        "scripts/local"
        "monitoring"
        "monitoring/prometheus"
        "monitoring/grafana"
        "monitoring/alertmanager"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            info "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 logs data
    chmod 700 config/local
    
    success "Directory structure created"
}

# =============================================================================
# Configuration Management
# =============================================================================
create_env_file() {
    info "Creating environment configuration..."
    
    local env_file=".env"
    
    if [[ -f "$env_file" ]]; then
        warning "Environment file already exists: $env_file"
        return
    fi
    
    # Copy from example if available
    if [[ -f ".env.example" ]]; then
        cp ".env.example" "$env_file"
        info "Copied .env.example to .env"
    else
        # Create basic .env file
        cat > "$env_file" << EOF
# Tyra MCP Memory Server Configuration
# Environment: $ENVIRONMENT

# Database Configuration
POSTGRES_HOST=$POSTGRES_HOST
POSTGRES_PORT=$POSTGRES_PORT
POSTGRES_DB=$POSTGRES_DB
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD

# Redis Configuration
REDIS_HOST=$REDIS_HOST
REDIS_PORT=$REDIS_PORT

# Memgraph Configuration
MEMGRAPH_HOST=$MEMGRAPH_HOST
MEMGRAPH_PORT=$MEMGRAPH_PORT

# Application Configuration
ENVIRONMENT=$ENVIRONMENT
LOG_LEVEL=INFO
DEBUG=false

# Security
SECRET_KEY=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 16)
EOF
    fi
    
    chmod 600 "$env_file"
    success "Environment configuration created"
}

generate_secrets() {
    info "Generating security secrets..."
    
    local secrets_file="config/local/secrets.yaml"
    
    if [[ -f "$secrets_file" ]]; then
        warning "Secrets file already exists: $secrets_file"
        return
    fi
    
    cat > "$secrets_file" << EOF
# Tyra MCP Memory Server - Security Secrets
# Generated: $(date)
# Environment: $ENVIRONMENT

secrets:
  secret_key: "$(openssl rand -hex 32)"
  api_key: "$(openssl rand -hex 16)"
  jwt_secret: "$(openssl rand -hex 32)"
  
database:
  postgres_password: "$POSTGRES_PASSWORD"
  
ssl:
  enabled: $SSL_ENABLED
  cert_path: "/etc/ssl/certs/tyra-memory-server.crt"
  key_path: "/etc/ssl/private/tyra-memory-server.key"
EOF
    
    chmod 600 "$secrets_file"
    success "Security secrets generated"
}

# =============================================================================
# Database Setup
# =============================================================================
setup_databases() {
    info "Setting up databases..."
    
    case "$ENVIRONMENT" in
        "development")
            setup_databases_docker
            ;;
        "production")
            setup_databases_native
            ;;
        "testing")
            setup_databases_docker
            ;;
        *)
            error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # Run database migrations
    if [[ -f "scripts/db/migrate.sh" ]]; then
        info "Running database migrations..."
        bash scripts/db/migrate.sh
    elif [[ -f "scripts/deploy/migrate.sh" ]]; then
        bash scripts/deploy/migrate.sh
    fi
    
    success "Database setup completed"
}

setup_databases_docker() {
    info "Setting up databases with Docker..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Start database services
    if [[ -f "docker-compose.yml" ]]; then
        info "Starting database services with Docker Compose..."
        docker-compose up -d postgres redis memgraph
        
        # Wait for services to be ready
        info "Waiting for database services to be ready..."
        sleep 10
        
        # Check if services are healthy
        docker-compose ps
    else
        error "Docker Compose file not found"
        exit 1
    fi
    
    success "Docker database setup completed"
}

setup_databases_native() {
    info "Setting up databases natively..."
    
    # This would typically involve connecting to existing database instances
    # For production, databases are usually pre-configured
    warning "Native database setup requires pre-configured database instances"
    
    # Test database connections
    if command -v psql &> /dev/null; then
        info "Testing PostgreSQL connection..."
        if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER"; then
            success "PostgreSQL connection successful"
        else
            error "PostgreSQL connection failed"
        fi
    fi
    
    if command -v redis-cli &> /dev/null; then
        info "Testing Redis connection..."
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping | grep -q "PONG"; then
            success "Redis connection successful"
        else
            error "Redis connection failed"
        fi
    fi
}

# =============================================================================
# SSL/TLS Setup
# =============================================================================
setup_ssl() {
    if [[ "$SSL_ENABLED" != "true" ]]; then
        info "SSL/TLS setup skipped (SSL_ENABLED=false)"
        return
    fi
    
    info "Setting up SSL/TLS certificates..."
    
    local cert_dir="/etc/ssl/certs"
    local key_dir="/etc/ssl/private"
    local cert_file="$cert_dir/tyra-memory-server.crt"
    local key_file="$key_dir/tyra-memory-server.key"
    
    if [[ -f "$cert_file" && -f "$key_file" ]]; then
        info "SSL certificates already exist"
        return
    fi
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        # Use Let's Encrypt for production
        if command -v certbot &> /dev/null; then
            info "Obtaining SSL certificate from Let's Encrypt..."
            sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email "admin@$DOMAIN"
        else
            error "Certbot not found. Please install certbot first."
            exit 1
        fi
    else
        # Generate self-signed certificate for development
        info "Generating self-signed SSL certificate..."
        sudo mkdir -p "$cert_dir" "$key_dir"
        sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$key_file" \
            -out "$cert_file" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"
        
        sudo chmod 600 "$key_file"
        sudo chmod 644 "$cert_file"
    fi
    
    success "SSL/TLS setup completed"
}

# =============================================================================
# Monitoring Setup
# =============================================================================
setup_monitoring() {
    if [[ "$MONITORING_ENABLED" != "true" ]]; then
        info "Monitoring setup skipped (MONITORING_ENABLED=false)"
        return
    fi
    
    info "Setting up monitoring stack..."
    
    # Run monitoring setup script if available
    if [[ -f "scripts/deploy/monitoring-setup.sh" ]]; then
        bash scripts/deploy/monitoring-setup.sh
    else
        warning "Monitoring setup script not found"
    fi
    
    success "Monitoring setup completed"
}

# =============================================================================
# Service Management
# =============================================================================
install_systemd_service() {
    if [[ "$ENVIRONMENT" != "production" ]]; then
        info "Systemd service installation skipped (not production)"
        return
    fi
    
    info "Installing systemd service..."
    
    local service_file="/etc/systemd/system/tyra-memory-server.service"
    
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=Tyra MCP Memory Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=tyra
Group=tyra
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/venv/bin
ExecStart=$PROJECT_ROOT/venv/bin/python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable tyra-memory-server
    
    if [[ "$AUTO_START" == "true" ]]; then
        sudo systemctl start tyra-memory-server
    fi
    
    success "Systemd service installed"
}

# =============================================================================
# Validation and Testing
# =============================================================================
validate_setup() {
    info "Validating setup..."
    
    # Check Python environment
    if [[ -d "venv" ]]; then
        source venv/bin/activate
        python -c "import sys; print(f'Python version: {sys.version}')"
    fi
    
    # Check configuration files
    local config_files=("config/config.yaml" ".env")
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            success "Configuration file found: $file"
        else
            error "Configuration file missing: $file"
        fi
    done
    
    # Test database connections
    if [[ -f "scripts/test_databases.sh" ]]; then
        bash scripts/test_databases.sh
    fi
    
    # Run basic tests
    if [[ -d "tests" ]]; then
        info "Running basic tests..."
        if command -v pytest &> /dev/null; then
            pytest tests/test_basic.py -v || warning "Some tests failed"
        fi
    fi
    
    success "Setup validation completed"
}

# =============================================================================
# Cleanup and Finalization
# =============================================================================
cleanup() {
    info "Cleaning up temporary files..."
    
    # Remove temporary files
    rm -f /tmp/tyra-setup-*.log.old
    
    # Set proper permissions
    find . -name "*.sh" -exec chmod +x {} \;
    
    success "Cleanup completed"
}

show_next_steps() {
    echo
    echo "============================================================================="
    success "Tyra MCP Memory Server Setup Completed!"
    echo "============================================================================="
    echo
    info "Next steps:"
    echo "1. Review configuration files:"
    echo "   - .env (environment variables)"
    echo "   - config/config.yaml (application settings)"
    echo
    echo "2. Start the services:"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "   sudo systemctl start tyra-memory-server"
        echo "   sudo systemctl status tyra-memory-server"
    else
        echo "   docker-compose up -d"
        echo "   # OR"
        echo "   source venv/bin/activate"
        echo "   python -m uvicorn src.api.app:app --reload"
    fi
    echo
    echo "3. Access the application:"
    echo "   - API: http://localhost:8000"
    echo "   - Health check: http://localhost:8000/health"
    echo "   - API docs: http://localhost:8000/docs"
    echo
    if [[ "$MONITORING_ENABLED" == "true" ]]; then
        echo "4. Access monitoring:"
        echo "   - Prometheus: http://localhost:9090"
        echo "   - Grafana: http://localhost:3000"
        echo
    fi
    echo "5. Run tests:"
    echo "   source venv/bin/activate"
    echo "   pytest tests/"
    echo
    echo "Setup log: $LOG_FILE"
    echo "============================================================================="
}

# =============================================================================
# Command Line Interface
# =============================================================================
show_help() {
    cat << EOF
Tyra MCP Memory Server - Unified Setup Script

Usage: $0 [OPTIONS]

Options:
    --env ENV               Environment (development|production|testing)
    --skip-deps             Skip system dependency installation
    --skip-ssl              Skip SSL/TLS setup
    --skip-monitoring       Skip monitoring setup
    --no-auto-start         Don't auto-start services
    --domain DOMAIN         Domain name for SSL certificates
    --help                  Show this help message

Environment Variables:
    TYRA_ENV               Environment setting
    POSTGRES_HOST          PostgreSQL host
    POSTGRES_PASSWORD      PostgreSQL password
    REDIS_HOST             Redis host
    MEMGRAPH_HOST          Memgraph host
    SSL_ENABLED            Enable SSL/TLS
    MONITORING_ENABLED     Enable monitoring stack
    AUTO_START             Auto-start services

Examples:
    $0                                      # Development setup
    $0 --env production --domain example.com  # Production setup
    $0 --env testing --skip-deps             # Testing setup

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS="true"
            shift
            ;;
        --skip-ssl)
            SSL_ENABLED="false"
            shift
            ;;
        --skip-monitoring)
            MONITORING_ENABLED="false"
            shift
            ;;
        --no-auto-start)
            AUTO_START="false"
            shift
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo "============================================================================="
    info "Starting Tyra MCP Memory Server Setup"
    info "Environment: $ENVIRONMENT"
    info "Domain: $DOMAIN"
    info "SSL Enabled: $SSL_ENABLED"
    info "Monitoring Enabled: $MONITORING_ENABLED"
    echo "============================================================================="
    
    # Core setup steps
    check_system_requirements
    install_system_dependencies
    create_directories
    setup_python_environment
    create_env_file
    generate_secrets
    setup_databases
    
    # Environment-specific setup
    if [[ "$ENVIRONMENT" == "production" ]]; then
        setup_ssl
        install_systemd_service
    fi
    
    # Optional components
    setup_monitoring
    
    # Validation and cleanup
    validate_setup
    cleanup
    
    # Show next steps
    show_next_steps
    
    success "Setup completed successfully!"
}

# Run main function
main "$@"