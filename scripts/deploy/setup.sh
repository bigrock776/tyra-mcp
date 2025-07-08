#!/bin/bash
# =============================================================================
# Tyra MCP Memory Server - Production Setup Script
# =============================================================================
# Comprehensive setup for production deployment

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/tmp/tyra-setup-$(date +%Y%m%d-%H%M%S).log"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
DOMAIN="${DOMAIN:-localhost}"
SSL_ENABLED="${SSL_ENABLED:-false}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
MONITORING_ENABLED="${MONITORING_ENABLED:-false}"
AUTO_START="${AUTO_START:-true}"

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
# Utility Functions
# =============================================================================
check_dependencies() {
    info "Checking system dependencies..."

    local missing_deps=()

    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi

    # Check curl
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi

    # Check jq for JSON processing
    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        info "Please install the missing dependencies and run this script again."
        exit 1
    fi

    success "All dependencies are installed"
}

generate_secrets() {
    info "Generating secure secrets..."

    # Generate random secrets
    export SECRET_KEY=$(openssl rand -base64 32)
    export API_KEY=$(openssl rand -base64 24)
    export POSTGRES_PASSWORD=$(openssl rand -base64 16)
    export GRAFANA_PASSWORD=$(openssl rand -base64 12)

    success "Secrets generated"
}

create_directories() {
    info "Creating necessary directories..."

    local dirs=(
        "/opt/tyra"
        "/opt/tyra/data"
        "/opt/tyra/logs"
        "/opt/tyra/backups"
        "/opt/tyra/config"
        "/opt/tyra/ssl"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            sudo mkdir -p "$dir"
            sudo chown $(whoami):$(whoami) "$dir"
            info "Created directory: $dir"
        fi
    done

    success "Directories created"
}

setup_environment() {
    info "Setting up environment configuration..."

    # Create .env file
    cat > "$PROJECT_ROOT/.env" << EOF
# =============================================================================
# Tyra MCP Memory Server - Production Environment Configuration
# =============================================================================
# Generated on $(date)

# Environment
ENVIRONMENT=$ENVIRONMENT
DEBUG=false
LOG_LEVEL=INFO
DOMAIN=$DOMAIN

# Security
SECRET_KEY=$SECRET_KEY
API_KEY=$API_KEY

# Database Passwords
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# Build Information
VERSION=1.0.0
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Performance Settings
WORKERS=4
POSTGRES_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=50

# Model Configuration
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=cpu

# Feature Flags
CACHE_ENABLED=true
OBSERVABILITY_ENABLED=true
TRACING_ENABLED=true
METRICS_ENABLED=true
TRACING_EXPORTER=console

# Backup Configuration
BACKUP_RETENTION_DAYS=7

# Port Configuration (if changed from defaults)
# POSTGRES_PORT=5432
# REDIS_PORT=6379
# MEMGRAPH_PORT=7687
# PROMETHEUS_PORT=9091
# GRAFANA_PORT=3001
# JAEGER_UI_PORT=16686

EOF

    chmod 600 "$PROJECT_ROOT/.env"
    success "Environment configuration created"
}

setup_ssl() {
    if [ "$SSL_ENABLED" = "true" ]; then
        info "Setting up SSL certificates..."

        # Create self-signed certificate for development
        if [ ! -f "/opt/tyra/ssl/cert.pem" ]; then
            openssl req -x509 -newkey rsa:4096 -keyout /opt/tyra/ssl/key.pem -out /opt/tyra/ssl/cert.pem -days 365 -nodes -subj "/CN=$DOMAIN"
            success "Self-signed SSL certificate created"
        fi
    fi
}

build_images() {
    info "Building Docker images..."

    cd "$PROJECT_ROOT"

    # Build production image
    docker-compose build memory-server mcp-server

    success "Docker images built successfully"
}

initialize_databases() {
    info "Initializing databases..."

    cd "$PROJECT_ROOT"

    # Start database services first
    docker-compose up -d postgres redis memgraph

    # Wait for databases to be ready
    info "Waiting for databases to be ready..."
    sleep 30

    # Run database migrations
    if [ -f "scripts/db/migrate.sh" ]; then
        bash scripts/db/migrate.sh
    fi

    success "Databases initialized"
}

setup_monitoring() {
    if [ "$MONITORING_ENABLED" = "true" ]; then
        info "Setting up monitoring stack..."

        # Create monitoring configuration
        mkdir -p "$PROJECT_ROOT/monitoring"

        # Prometheus configuration
        cat > "$PROJECT_ROOT/monitoring/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'tyra-memory-server'
    static_configs:
      - targets: ['memory-server:9090']
    metrics_path: '/v1/telemetry/metrics'
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF

        # Create Grafana dashboards directory
        mkdir -p "$PROJECT_ROOT/monitoring/grafana/dashboards"
        mkdir -p "$PROJECT_ROOT/monitoring/grafana/datasources"

        # Grafana datasource configuration
        cat > "$PROJECT_ROOT/monitoring/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

        success "Monitoring configuration created"
    fi
}

setup_backup() {
    if [ "$BACKUP_ENABLED" = "true" ]; then
        info "Setting up backup procedures..."

        # Create backup script
        cat > "/opt/tyra/backup-cron.sh" << 'EOF'
#!/bin/bash
cd /opt/tyra/memory-server
docker-compose --profile backup up backup
EOF

        chmod +x "/opt/tyra/backup-cron.sh"

        # Add to crontab (daily backups at 2 AM)
        (crontab -l 2>/dev/null; echo "0 2 * * * /opt/tyra/backup-cron.sh") | crontab -

        success "Backup procedures configured"
    fi
}

start_services() {
    if [ "$AUTO_START" = "true" ]; then
        info "Starting Tyra MCP Memory Server..."

        cd "$PROJECT_ROOT"

        # Start core services
        if [ "$MONITORING_ENABLED" = "true" ]; then
            docker-compose --profile monitoring up -d
        else
            docker-compose up -d
        fi

        # Wait for services to be ready
        info "Waiting for services to start..."
        sleep 30

        # Health check
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            success "Memory server is running and healthy"
        else
            warning "Memory server may not be fully ready yet"
        fi

        success "Services started successfully"
    fi
}

create_systemd_service() {
    info "Creating systemd service..."

    sudo tee /etc/systemd/system/tyra-memory-server.service > /dev/null << EOF
[Unit]
Description=Tyra MCP Memory Server
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_ROOT
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable tyra-memory-server.service

    success "Systemd service created and enabled"
}

print_summary() {
    echo
    echo "============================================================================="
    success "Tyra MCP Memory Server Setup Complete!"
    echo "============================================================================="
    echo
    info "Configuration:"
    echo "  - Environment: $ENVIRONMENT"
    echo "  - Domain: $DOMAIN"
    echo "  - SSL Enabled: $SSL_ENABLED"
    echo "  - Monitoring Enabled: $MONITORING_ENABLED"
    echo "  - Backup Enabled: $BACKUP_ENABLED"
    echo
    info "Service URLs:"
    echo "  - Memory Server API: http://$DOMAIN:8000"
    echo "  - MCP Server: http://$DOMAIN:3000"
    echo "  - Health Check: http://$DOMAIN:8000/health"
    echo "  - API Documentation: http://$DOMAIN:8000/docs"

    if [ "$MONITORING_ENABLED" = "true" ]; then
        echo "  - Grafana: http://$DOMAIN:3001 (admin/$GRAFANA_PASSWORD)"
        echo "  - Prometheus: http://$DOMAIN:9091"
        echo "  - Jaeger: http://$DOMAIN:16686"
    fi

    echo
    info "Management Commands:"
    echo "  - Start services: cd $PROJECT_ROOT && docker-compose up -d"
    echo "  - Stop services: cd $PROJECT_ROOT && docker-compose down"
    echo "  - View logs: cd $PROJECT_ROOT && docker-compose logs -f"
    echo "  - Backup: cd $PROJECT_ROOT && docker-compose --profile backup up backup"
    echo
    info "Configuration files:"
    echo "  - Environment: $PROJECT_ROOT/.env"
    echo "  - Docker Compose: $PROJECT_ROOT/docker-compose.yml"
    echo "  - Setup Log: $LOG_FILE"
    echo
    warning "IMPORTANT: Store the following securely:"
    echo "  - API Key: $API_KEY"
    echo "  - Postgres Password: $POSTGRES_PASSWORD"
    if [ "$MONITORING_ENABLED" = "true" ]; then
        echo "  - Grafana Password: $GRAFANA_PASSWORD"
    fi
    echo
    success "Setup completed successfully! 🚀"
}

# =============================================================================
# Main Setup Function
# =============================================================================
main() {
    echo "============================================================================="
    info "Tyra MCP Memory Server - Production Setup"
    echo "============================================================================="

    log "Starting setup process..."

    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        warning "Running as root. Consider using a non-root user with sudo privileges."
    fi

    # Setup steps
    check_dependencies
    create_directories
    generate_secrets
    setup_environment
    setup_ssl
    setup_monitoring
    build_images
    initialize_databases
    setup_backup
    create_systemd_service
    start_services
    print_summary

    log "Setup process completed successfully"
}

# =============================================================================
# Command Line Arguments
# =============================================================================
show_help() {
    cat << EOF
Tyra MCP Memory Server - Production Setup Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV       Set environment (development|production) [default: production]
    -d, --domain DOMAIN         Set domain name [default: localhost]
    -s, --ssl                   Enable SSL/TLS
    -m, --monitoring            Enable monitoring stack (Prometheus, Grafana, Jaeger)
    -b, --no-backup             Disable automatic backup setup
    -n, --no-autostart         Don't start services automatically
    -h, --help                  Show this help message

Examples:
    $0                          # Basic production setup
    $0 -d myserver.com -s -m    # Full setup with SSL and monitoring
    $0 -e development -m        # Development setup with monitoring

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -s|--ssl)
            SSL_ENABLED="true"
            shift
            ;;
        -m|--monitoring)
            MONITORING_ENABLED="true"
            shift
            ;;
        -b|--no-backup)
            BACKUP_ENABLED="false"
            shift
            ;;
        -n|--no-autostart)
            AUTO_START="false"
            shift
            ;;
        -h|--help)
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

# Run main setup
main "$@"
