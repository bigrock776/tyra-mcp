#!/bin/bash

# =============================================================================
# Tyra MCP Memory Server - Update Script
# =============================================================================
# Handles safe updates of the Tyra Memory Server system

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/opt/tyra/backups}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Update configuration
UPDATE_SOURCE="${UPDATE_SOURCE:-}"
UPDATE_BRANCH="${UPDATE_BRANCH:-main}"
UPDATE_TYPE="${UPDATE_TYPE:-patch}"  # patch, minor, major
SKIP_BACKUP="${SKIP_BACKUP:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
FORCE_UPDATE="${FORCE_UPDATE:-false}"

# Service configuration
SERVICE_NAME="tyra-memory-server"
USE_DOCKER="${USE_DOCKER:-true}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
SYSTEMD_SERVICE="${SYSTEMD_SERVICE:-tyra-memory-server}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Logging Functions
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/update.log"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_DIR/update.log"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_DIR/update.log"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_DIR/update.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_DIR/update.log"
}

# =============================================================================
# Utility Functions
# =============================================================================
check_prerequisites() {
    info "Checking update prerequisites..."
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        error "Git is required but not installed"
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "Not in a Git repository"
        exit 1
    fi
    
    # Check if backup script exists
    if [ "$SKIP_BACKUP" = "false" ] && [ ! -f "$SCRIPT_DIR/backup.sh" ]; then
        error "Backup script not found: $SCRIPT_DIR/backup.sh"
        exit 1
    fi
    
    # Check if migration script exists
    if [ ! -f "$SCRIPT_DIR/migrate.sh" ]; then
        error "Migration script not found: $SCRIPT_DIR/migrate.sh"
        exit 1
    fi
    
    # Check if service is running
    if [ "$USE_DOCKER" = "true" ]; then
        if ! docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
            warning "Service doesn't appear to be running via Docker Compose"
        fi
    else
        if ! systemctl is-active --quiet "$SYSTEMD_SERVICE"; then
            warning "Service $SYSTEMD_SERVICE is not active"
        fi
    fi
    
    success "Prerequisites check passed"
}

get_current_version() {
    # Try to get version from git tag
    local version
    version=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
    
    # If no tags, use commit hash
    if [ "$version" = "unknown" ]; then
        version=$(git rev-parse --short HEAD)
    fi
    
    echo "$version"
}

get_latest_version() {
    local source="$1"
    
    if [ -n "$source" ]; then
        # Fetch from specific source
        git fetch "$source" "$UPDATE_BRANCH" 2>/dev/null || true
        git describe --tags --abbrev=0 "$(git rev-parse $source/$UPDATE_BRANCH)" 2>/dev/null || git rev-parse --short "$source/$UPDATE_BRANCH"
    else
        # Fetch from origin
        git fetch origin "$UPDATE_BRANCH" 2>/dev/null || true
        git describe --tags --abbrev=0 "$(git rev-parse origin/$UPDATE_BRANCH)" 2>/dev/null || git rev-parse --short "origin/$UPDATE_BRANCH"
    fi
}

compare_versions() {
    local current="$1"
    local latest="$2"
    
    if [ "$current" = "$latest" ]; then
        echo "same"
    else
        echo "different"
    fi
}

create_pre_update_backup() {
    info "Creating pre-update backup..."
    
    local backup_name="pre-update-$(date +%Y%m%d-%H%M%S)"
    
    if [ -f "$SCRIPT_DIR/backup.sh" ]; then
        if "$SCRIPT_DIR/backup.sh" --backup-dir "$BACKUP_DIR" > /dev/null 2>&1; then
            success "Pre-update backup created successfully"
            return 0
        else
            error "Failed to create pre-update backup"
            return 1
        fi
    else
        error "Backup script not found"
        return 1
    fi
}

stop_services() {
    info "Stopping Tyra Memory Server services..."
    
    if [ "$USE_DOCKER" = "true" ]; then
        if [ -f "$DOCKER_COMPOSE_FILE" ]; then
            docker-compose -f "$DOCKER_COMPOSE_FILE" down
            success "Docker services stopped"
        else
            warning "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        fi
    else
        if systemctl is-active --quiet "$SYSTEMD_SERVICE"; then
            systemctl stop "$SYSTEMD_SERVICE"
            success "Systemd service stopped"
        else
            warning "Service $SYSTEMD_SERVICE was not running"
        fi
    fi
}

start_services() {
    info "Starting Tyra Memory Server services..."
    
    if [ "$USE_DOCKER" = "true" ]; then
        if [ -f "$DOCKER_COMPOSE_FILE" ]; then
            docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
            success "Docker services started"
        else
            error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
            return 1
        fi
    else
        systemctl start "$SYSTEMD_SERVICE"
        success "Systemd service started"
    fi
}

update_codebase() {
    info "Updating codebase..."
    
    # Save current state
    local current_branch=$(git branch --show-current)
    local current_commit=$(git rev-parse HEAD)
    
    # Store current state for rollback
    echo "$current_branch:$current_commit" > "$PROJECT_ROOT/.update_rollback_info"
    
    # Stash any local changes
    git stash push -m "Pre-update stash $(date)" || true
    
    # Switch to update branch
    git checkout "$UPDATE_BRANCH"
    
    # Pull latest changes
    if [ -n "$UPDATE_SOURCE" ]; then
        git pull "$UPDATE_SOURCE" "$UPDATE_BRANCH"
    else
        git pull origin "$UPDATE_BRANCH"
    fi
    
    success "Codebase updated successfully"
}

update_dependencies() {
    info "Updating Python dependencies..."
    
    # Activate virtual environment if it exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    # Update pip
    pip install --upgrade pip
    
    # Update requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt" --upgrade
    fi
    
    # Update development requirements if they exist
    if [ -f "$PROJECT_ROOT/requirements-dev.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements-dev.txt" --upgrade
    fi
    
    success "Dependencies updated successfully"
}

run_migrations() {
    info "Running database migrations..."
    
    if [ -f "$SCRIPT_DIR/migrate.sh" ]; then
        if "$SCRIPT_DIR/migrate.sh" --no-backup; then
            success "Migrations completed successfully"
        else
            error "Migration failed"
            return 1
        fi
    else
        error "Migration script not found"
        return 1
    fi
}

run_tests() {
    info "Running test suite..."
    
    # Activate virtual environment if it exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    # Run tests
    if [ -f "$PROJECT_ROOT/pytest.ini" ] || [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        if pytest "$PROJECT_ROOT/tests/" -v; then
            success "All tests passed"
        else
            error "Tests failed"
            return 1
        fi
    else
        warning "No test configuration found, skipping tests"
    fi
}

health_check() {
    info "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
            success "Health check passed"
            return 0
        fi
        
        info "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 5
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
    return 1
}

rollback_update() {
    error "Rolling back update..."
    
    if [ -f "$PROJECT_ROOT/.update_rollback_info" ]; then
        local rollback_info=$(cat "$PROJECT_ROOT/.update_rollback_info")
        local rollback_branch=$(echo "$rollback_info" | cut -d: -f1)
        local rollback_commit=$(echo "$rollback_info" | cut -d: -f2)
        
        # Stop services
        stop_services
        
        # Revert to previous state
        git checkout "$rollback_commit"
        
        # Restore dependencies (if requirements changed)
        if [ -d "$PROJECT_ROOT/venv" ]; then
            source "$PROJECT_ROOT/venv/bin/activate"
            pip install -r "$PROJECT_ROOT/requirements.txt"
        fi
        
        # Start services
        start_services
        
        # Clean up
        rm -f "$PROJECT_ROOT/.update_rollback_info"
        
        success "Rollback completed"
    else
        error "No rollback information found"
    fi
}

cleanup() {
    info "Cleaning up update artifacts..."
    
    # Remove rollback info file
    rm -f "$PROJECT_ROOT/.update_rollback_info"
    
    # Clean up Docker images if using Docker
    if [ "$USE_DOCKER" = "true" ]; then
        docker system prune -f > /dev/null 2>&1 || true
    fi
    
    success "Cleanup completed"
}

# =============================================================================
# Main Update Functions
# =============================================================================
perform_update() {
    local current_version=$(get_current_version)
    local latest_version=$(get_latest_version "$UPDATE_SOURCE")
    
    info "Current version: $current_version"
    info "Latest version: $latest_version"
    
    # Check if update is needed
    if [ "$FORCE_UPDATE" = "false" ] && [ "$(compare_versions "$current_version" "$latest_version")" = "same" ]; then
        info "Already up to date"
        return 0
    fi
    
    # Create pre-update backup
    if [ "$SKIP_BACKUP" = "false" ]; then
        if ! create_pre_update_backup; then
            error "Failed to create backup, aborting update"
            exit 1
        fi
    fi
    
    # Stop services
    stop_services
    
    # Update codebase
    if ! update_codebase; then
        error "Failed to update codebase"
        rollback_update
        exit 1
    fi
    
    # Update dependencies
    if ! update_dependencies; then
        error "Failed to update dependencies"
        rollback_update
        exit 1
    fi
    
    # Run migrations
    if ! run_migrations; then
        error "Failed to run migrations"
        rollback_update
        exit 1
    fi
    
    # Start services
    if ! start_services; then
        error "Failed to start services"
        rollback_update
        exit 1
    fi
    
    # Run tests
    if [ "$SKIP_TESTS" = "false" ]; then
        if ! run_tests; then
            error "Tests failed after update"
            rollback_update
            exit 1
        fi
    fi
    
    # Health check
    if ! health_check; then
        error "Health check failed after update"
        rollback_update
        exit 1
    fi
    
    # Cleanup
    cleanup
    
    success "Update completed successfully!"
    info "Updated from $current_version to $latest_version"
}

show_help() {
    cat << EOF
Tyra MCP Memory Server - Update Script

Usage: $0 [OPTIONS]

Options:
    --source SOURCE          Git remote source (default: origin)
    --branch BRANCH          Git branch to update to (default: main)
    --type TYPE              Update type: patch/minor/major (default: patch)
    --skip-backup            Skip pre-update backup
    --skip-tests             Skip test execution
    --force                  Force update even if versions match
    --docker                 Use Docker Compose for service management
    --systemd                Use systemd for service management
    --backup-dir DIR         Set backup directory
    --help                   Show this help message

Examples:
    $0                       # Standard update
    $0 --force               # Force update
    $0 --skip-tests          # Update without running tests
    $0 --source upstream     # Update from upstream remote

EOF
}

# =============================================================================
# Command Line Interface
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            UPDATE_SOURCE="$2"
            shift 2
            ;;
        --branch)
            UPDATE_BRANCH="$2"
            shift 2
            ;;
        --type)
            UPDATE_TYPE="$2"
            shift 2
            ;;
        --skip-backup)
            SKIP_BACKUP="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --force)
            FORCE_UPDATE="true"
            shift
            ;;
        --docker)
            USE_DOCKER="true"
            shift
            ;;
        --systemd)
            USE_DOCKER="false"
            shift
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
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
    info "Tyra MCP Memory Server Update"
    echo "============================================================================="
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Perform update
    perform_update
    
    echo "============================================================================="
    success "Update completed successfully!"
    echo "============================================================================="
}

# Trap to handle cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"