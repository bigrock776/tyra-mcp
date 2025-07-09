#!/bin/bash

# =============================================================================
# Tyra MCP Memory Server - Blue-Green Deployment Script
# =============================================================================
# Zero-downtime deployment using blue-green strategy

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Blue-Green configuration
BLUE_PORT="${BLUE_PORT:-8000}"
GREEN_PORT="${GREEN_PORT:-8001}"
HEALTH_CHECK_PATH="${HEALTH_CHECK_PATH:-/health}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-60}"
LOAD_BALANCER_CONFIG="${LOAD_BALANCER_CONFIG:-/etc/nginx/sites-available/tyra-memory-server}"

# Deployment configuration
DEPLOYMENT_SOURCE="${DEPLOYMENT_SOURCE:-}"
DEPLOYMENT_BRANCH="${DEPLOYMENT_BRANCH:-main}"
WARM_UP_REQUESTS="${WARM_UP_REQUESTS:-10}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Docker configuration
DOCKER_COMPOSE_BLUE="${DOCKER_COMPOSE_BLUE:-$PROJECT_ROOT/docker-compose.blue.yml}"
DOCKER_COMPOSE_GREEN="${DOCKER_COMPOSE_GREEN:-$PROJECT_ROOT/docker-compose.green.yml}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Logging Functions
# =============================================================================
info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_DIR/blue-green.log"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_DIR/blue-green.log"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_DIR/blue-green.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_DIR/blue-green.log"
}

# =============================================================================
# Utility Functions
# =============================================================================
get_current_environment() {
    # Check which environment is currently serving traffic
    if curl -s -f "http://localhost:$BLUE_PORT$HEALTH_CHECK_PATH" > /dev/null 2>&1; then
        if nginx -t > /dev/null 2>&1 && nginx -T 2>/dev/null | grep -q ":$BLUE_PORT"; then
            echo "blue"
        else
            echo "green"
        fi
    elif curl -s -f "http://localhost:$GREEN_PORT$HEALTH_CHECK_PATH" > /dev/null 2>&1; then
        echo "green"
    else
        echo "none"
    fi
}

get_target_environment() {
    local current="$1"
    
    case "$current" in
        blue)
            echo "green"
            ;;
        green)
            echo "blue"
            ;;
        none)
            echo "blue"
            ;;
        *)
            echo "blue"
            ;;
    esac
}

get_environment_port() {
    local environment="$1"
    
    case "$environment" in
        blue)
            echo "$BLUE_PORT"
            ;;
        green)
            echo "$GREEN_PORT"
            ;;
        *)
            echo "$BLUE_PORT"
            ;;
    esac
}

get_environment_compose_file() {
    local environment="$1"
    
    case "$environment" in
        blue)
            echo "$DOCKER_COMPOSE_BLUE"
            ;;
        green)
            echo "$DOCKER_COMPOSE_GREEN"
            ;;
        *)
            echo "$DOCKER_COMPOSE_BLUE"
            ;;
    esac
}

health_check() {
    local port="$1"
    local timeout="${2:-30}"
    
    info "Performing health check on port $port..."
    
    local attempt=1
    local max_attempts=$((timeout / 5))
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port$HEALTH_CHECK_PATH" > /dev/null 2>&1; then
            success "Health check passed on port $port"
            return 0
        fi
        
        info "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 5
        ((attempt++))
    done
    
    error "Health check failed on port $port after $timeout seconds"
    return 1
}

warm_up_environment() {
    local port="$1"
    
    info "Warming up environment on port $port..."
    
    for i in $(seq 1 $WARM_UP_REQUESTS); do
        curl -s "http://localhost:$port$HEALTH_CHECK_PATH" > /dev/null 2>&1 || true
        curl -s "http://localhost:$port/v1/memory/status" > /dev/null 2>&1 || true
        sleep 1
    done
    
    success "Environment warm-up completed"
}

# =============================================================================
# Docker Management Functions
# =============================================================================
create_docker_compose_files() {
    info "Creating Docker Compose files for blue-green deployment..."
    
    # Blue environment
    cat > "$DOCKER_COMPOSE_BLUE" << EOF
version: '3.8'

services:
  tyra-memory-server:
    build: .
    image: tyra-memory-server:blue
    container_name: tyra-memory-server-blue
    environment:
      - API_PORT=$BLUE_PORT
      - ENVIRONMENT=blue
    ports:
      - "$BLUE_PORT:$BLUE_PORT"
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
      - memgraph

  postgres:
    image: pgvector/pgvector:pg15
    container_name: tyra-postgres-blue
    environment:
      POSTGRES_DB: tyra_memory_blue
      POSTGRES_USER: tyra
      POSTGRES_PASSWORD: tyra_secure_password
    volumes:
      - postgres_blue_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: tyra-redis-blue
    volumes:
      - redis_blue_data:/data
    restart: unless-stopped

  memgraph:
    image: memgraph/memgraph:latest
    container_name: tyra-memgraph-blue
    volumes:
      - memgraph_blue_data:/var/lib/memgraph
    restart: unless-stopped

volumes:
  postgres_blue_data:
  redis_blue_data:
  memgraph_blue_data:

networks:
  default:
    name: tyra-blue
    driver: bridge
EOF
    
    # Green environment
    cat > "$DOCKER_COMPOSE_GREEN" << EOF
version: '3.8'

services:
  tyra-memory-server:
    build: .
    image: tyra-memory-server:green
    container_name: tyra-memory-server-green
    environment:
      - API_PORT=$GREEN_PORT
      - ENVIRONMENT=green
    ports:
      - "$GREEN_PORT:$GREEN_PORT"
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
      - memgraph

  postgres:
    image: pgvector/pgvector:pg15
    container_name: tyra-postgres-green
    environment:
      POSTGRES_DB: tyra_memory_green
      POSTGRES_USER: tyra
      POSTGRES_PASSWORD: tyra_secure_password
    volumes:
      - postgres_green_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: tyra-redis-green
    volumes:
      - redis_green_data:/data
    restart: unless-stopped

  memgraph:
    image: memgraph/memgraph:latest
    container_name: tyra-memgraph-green
    volumes:
      - memgraph_green_data:/var/lib/memgraph
    restart: unless-stopped

volumes:
  postgres_green_data:
  redis_green_data:
  memgraph_green_data:

networks:
  default:
    name: tyra-green
    driver: bridge
EOF
    
    success "Docker Compose files created"
}

build_environment_image() {
    local environment="$1"
    
    info "Building Docker image for $environment environment..."
    
    if [ -n "$DOCKER_REGISTRY" ]; then
        local image_name="$DOCKER_REGISTRY/tyra-memory-server:$environment-$IMAGE_TAG"
    else
        local image_name="tyra-memory-server:$environment"
    fi
    
    docker build -t "$image_name" "$PROJECT_ROOT"
    
    success "Docker image built: $image_name"
}

deploy_environment() {
    local environment="$1"
    local compose_file=$(get_environment_compose_file "$environment")
    
    info "Deploying $environment environment..."
    
    # Stop existing containers
    docker-compose -f "$compose_file" down || true
    
    # Build new image
    build_environment_image "$environment"
    
    # Start new containers
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be ready
    sleep 10
    
    success "$environment environment deployed"
}

stop_environment() {
    local environment="$1"
    local compose_file=$(get_environment_compose_file "$environment")
    
    info "Stopping $environment environment..."
    
    docker-compose -f "$compose_file" down
    
    success "$environment environment stopped"
}

# =============================================================================
# Load Balancer Management
# =============================================================================
create_nginx_config() {
    info "Creating Nginx configuration for load balancing..."
    
    cat > "$LOAD_BALANCER_CONFIG" << EOF
upstream tyra_backend {
    server localhost:$BLUE_PORT;
    # server localhost:$GREEN_PORT backup;
}

server {
    listen 80;
    server_name localhost;
    
    location / {
        proxy_pass http://tyra_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Health check
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    location /health {
        access_log off;
        proxy_pass http://tyra_backend;
    }
}
EOF
    
    success "Nginx configuration created"
}

switch_traffic() {
    local target_environment="$1"
    local target_port=$(get_environment_port "$target_environment")
    
    info "Switching traffic to $target_environment environment (port $target_port)..."
    
    # Update Nginx upstream configuration
    sed -i "s/server localhost:[0-9]*;/server localhost:$target_port;/" "$LOAD_BALANCER_CONFIG"
    
    # Test Nginx configuration
    if ! nginx -t; then
        error "Nginx configuration test failed"
        return 1
    fi
    
    # Reload Nginx
    if ! nginx -s reload; then
        error "Failed to reload Nginx"
        return 1
    fi
    
    success "Traffic switched to $target_environment environment"
}

# =============================================================================
# Deployment Functions
# =============================================================================
prepare_deployment() {
    info "Preparing blue-green deployment..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Create Docker Compose files
    create_docker_compose_files
    
    # Create Nginx configuration
    create_nginx_config
    
    success "Deployment preparation completed"
}

perform_deployment() {
    local current_env=$(get_current_environment)
    local target_env=$(get_target_environment "$current_env")
    local target_port=$(get_environment_port "$target_env")
    
    info "Current environment: $current_env"
    info "Target environment: $target_env"
    
    # Deploy to target environment
    deploy_environment "$target_env"
    
    # Health check target environment
    if ! health_check "$target_port" "$HEALTH_CHECK_TIMEOUT"; then
        error "Health check failed for $target_env environment"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            stop_environment "$target_env"
        fi
        return 1
    fi
    
    # Warm up target environment
    warm_up_environment "$target_port"
    
    # Switch traffic to target environment
    if ! switch_traffic "$target_env"; then
        error "Failed to switch traffic to $target_env environment"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            stop_environment "$target_env"
        fi
        return 1
    fi
    
    # Final health check
    if ! health_check "$target_port" 30; then
        error "Final health check failed after traffic switch"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            # Switch back to current environment
            switch_traffic "$current_env"
            stop_environment "$target_env"
        fi
        return 1
    fi
    
    # Stop previous environment after successful deployment
    if [ "$current_env" != "none" ]; then
        info "Stopping previous environment: $current_env"
        stop_environment "$current_env"
    fi
    
    success "Blue-green deployment completed successfully"
    info "Active environment: $target_env"
}

rollback_deployment() {
    local current_env=$(get_current_environment)
    local previous_env=$(get_target_environment "$current_env")
    
    info "Rolling back from $current_env to $previous_env..."
    
    # Deploy previous environment
    deploy_environment "$previous_env"
    
    # Health check previous environment
    local previous_port=$(get_environment_port "$previous_env")
    if ! health_check "$previous_port" "$HEALTH_CHECK_TIMEOUT"; then
        error "Rollback failed: $previous_env environment is not healthy"
        return 1
    fi
    
    # Switch traffic back
    if ! switch_traffic "$previous_env"; then
        error "Failed to switch traffic back to $previous_env"
        return 1
    fi
    
    # Stop current environment
    stop_environment "$current_env"
    
    success "Rollback completed successfully"
    info "Active environment: $previous_env"
}

status_check() {
    local current_env=$(get_current_environment)
    
    echo "============================================================================="
    info "Blue-Green Deployment Status"
    echo "============================================================================="
    
    info "Current active environment: $current_env"
    
    # Check blue environment
    if docker ps | grep -q "tyra-memory-server-blue"; then
        info "Blue environment: Running"
        health_check "$BLUE_PORT" 5 && echo "  Health: OK" || echo "  Health: FAILED"
    else
        info "Blue environment: Stopped"
    fi
    
    # Check green environment
    if docker ps | grep -q "tyra-memory-server-green"; then
        info "Green environment: Running"
        health_check "$GREEN_PORT" 5 && echo "  Health: OK" || echo "  Health: FAILED"
    else
        info "Green environment: Stopped"
    fi
    
    # Check Nginx
    if nginx -t > /dev/null 2>&1; then
        info "Nginx configuration: OK"
    else
        warning "Nginx configuration: INVALID"
    fi
}

cleanup_deployment() {
    info "Cleaning up deployment resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    success "Cleanup completed"
}

show_help() {
    cat << EOF
Tyra MCP Memory Server - Blue-Green Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy      Perform blue-green deployment (default)
    rollback    Rollback to previous environment
    status      Show deployment status
    cleanup     Clean up deployment resources
    prepare     Prepare deployment environment
    help        Show this help message

Options:
    --source SOURCE          Git source for deployment
    --branch BRANCH          Git branch to deploy (default: main)
    --blue-port PORT         Blue environment port (default: 8000)
    --green-port PORT        Green environment port (default: 8001)
    --timeout SECONDS        Health check timeout (default: 60)
    --no-rollback            Don't rollback on failure
    --registry REGISTRY      Docker registry URL
    --tag TAG                Docker image tag (default: latest)

Examples:
    $0 deploy                     # Deploy using blue-green strategy
    $0 rollback                   # Rollback to previous environment
    $0 status                     # Show current deployment status
    $0 cleanup                    # Clean up unused resources

EOF
}

# =============================================================================
# Command Line Interface
# =============================================================================
COMMAND="${1:-deploy}"

while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|rollback|status|cleanup|prepare)
            COMMAND="$1"
            shift
            ;;
        --source)
            DEPLOYMENT_SOURCE="$2"
            shift 2
            ;;
        --branch)
            DEPLOYMENT_BRANCH="$2"
            shift 2
            ;;
        --blue-port)
            BLUE_PORT="$2"
            shift 2
            ;;
        --green-port)
            GREEN_PORT="$2"
            shift 2
            ;;
        --timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE="false"
            shift
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            if [ "$COMMAND" = "deploy" ]; then
                shift
            else
                error "Unknown argument: $1"
                show_help
                exit 1
            fi
            ;;
    esac
done

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo "============================================================================="
    info "Tyra MCP Memory Server - Blue-Green Deployment"
    echo "============================================================================="
    
    case "$COMMAND" in
        deploy)
            prepare_deployment
            perform_deployment
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            status_check
            ;;
        cleanup)
            cleanup_deployment
            ;;
        prepare)
            prepare_deployment
            ;;
        *)
            error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"