#!/bin/bash

# =============================================================================
# Tyra MCP Memory Server - Rollback Script
# =============================================================================
# Safely rollback to previous version with data preservation

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/opt/tyra/backups}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Rollback configuration
ROLLBACK_TARGET="${ROLLBACK_TARGET:-}"
ROLLBACK_TYPE="${ROLLBACK_TYPE:-code}"  # code, full, migration
SKIP_BACKUP="${SKIP_BACKUP:-false}"
FORCE_ROLLBACK="${FORCE_ROLLBACK:-false}"
PRESERVE_DATA="${PRESERVE_DATA:-true}"

# Service configuration
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/rollback.log"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_DIR/rollback.log"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_DIR/rollback.log"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_DIR/rollback.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_DIR/rollback.log"
}

# =============================================================================
# Utility Functions
# =============================================================================
check_prerequisites() {
    info "Checking rollback prerequisites..."
    
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
    
    # Check if backup directory exists
    if [ ! -d "$BACKUP_DIR" ]; then
        error "Backup directory not found: $BACKUP_DIR"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

get_current_version() {
    local version
    version=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
    
    if [ "$version" = "unknown" ]; then
        version=$(git rev-parse --short HEAD)
    fi
    
    echo "$version"
}

get_available_versions() {
    info "Available versions for rollback:"
    
    # Show git tags
    echo "Git tags:"
    git tag --sort=-version:refname | head -10
    
    echo
    echo "Recent commits:"
    git log --oneline -n 10
    
    echo
    echo "Available backups:"
    list_backups
}

list_backups() {
    if [ -d "$BACKUP_DIR" ]; then
        find "$BACKUP_DIR" -name "*.tar.gz" -type f | sort -r | head -10 | while read -r backup; do
            local name=$(basename "$backup" .tar.gz)
            local date=$(date -r "$backup" '+%Y-%m-%d %H:%M:%S')
            echo "  $name ($date)"
        done
    else
        echo "  No backups found"
    fi
}

validate_rollback_target() {
    local target="$1"
    
    if [ -z "$target" ]; then
        error "No rollback target specified"
        return 1
    fi
    
    # Check if target is a git reference
    if git rev-parse --verify "$target" > /dev/null 2>&1; then
        info "Rollback target '$target' is a valid git reference"
        return 0
    fi
    
    # Check if target is a backup file
    if [ -f "$BACKUP_DIR/$target.tar.gz" ]; then
        info "Rollback target '$target' is a valid backup"
        return 0
    fi
    
    error "Invalid rollback target: $target"
    return 1
}

create_pre_rollback_backup() {
    info "Creating pre-rollback backup..."
    
    if [ "$SKIP_BACKUP" = "true" ]; then
        info "Skipping pre-rollback backup (--skip-backup specified)"
        return 0
    fi
    
    local backup_name="pre-rollback-$(date +%Y%m%d-%H%M%S)"
    
    if [ -f "$SCRIPT_DIR/backup.sh" ]; then
        if "$SCRIPT_DIR/backup.sh" --backup-dir "$BACKUP_DIR" > /dev/null 2>&1; then
            success "Pre-rollback backup created: $backup_name"
            return 0
        else
            error "Failed to create pre-rollback backup"
            return 1
        fi
    else
        warning "Backup script not found, skipping pre-rollback backup"
        return 0
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

rollback_code() {
    local target="$1"
    
    info "Rolling back code to: $target"
    
    # Stash any uncommitted changes
    git stash push -m "Pre-rollback stash $(date)" || true
    
    # Checkout the target
    if ! git checkout "$target"; then
        error "Failed to checkout $target"
        return 1
    fi
    
    # Update submodules if they exist
    if [ -f "$PROJECT_ROOT/.gitmodules" ]; then
        git submodule update --init --recursive
    fi
    
    success "Code rollback completed"
}

rollback_dependencies() {
    info "Rolling back Python dependencies..."
    
    # Activate virtual environment if it exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    else
        warning "Virtual environment not found, using system Python"
    fi
    
    # Install requirements from the rolled-back version
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
        success "Dependencies rolled back"
    else
        warning "requirements.txt not found, skipping dependency rollback"
    fi
}

rollback_migrations() {
    info "Rolling back database migrations..."
    
    # This is a placeholder - actual migration rollback would depend on
    # the specific migration system and rollback strategy
    warning "Database migration rollback not implemented yet"
    warning "Manual database restoration from backup may be required"
    
    # If we have a migration script with rollback support
    if [ -f "$SCRIPT_DIR/migrate.sh" ]; then
        info "Migration script found, checking for rollback support..."
        # The migrate.sh script has rollback functionality
        # but it requires specific rollback migration files
    fi
}

rollback_from_backup() {
    local backup_name="$1"
    local backup_file="$BACKUP_DIR/$backup_name.tar.gz"
    
    info "Rolling back from backup: $backup_name"
    
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Use the backup restore script
    if [ -f "$SCRIPT_DIR/backup.sh" ]; then
        if "$SCRIPT_DIR/backup.sh" restore "$backup_file"; then
            success "Backup rollback completed"
            return 0
        else
            error "Failed to restore from backup"
            return 1
        fi
    else
        error "Backup script not found"
        return 1
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

verify_rollback() {
    info "Verifying rollback..."
    
    # Check current version
    local current_version=$(get_current_version)
    info "Current version after rollback: $current_version"
    
    # Check if services are running
    if [ "$USE_DOCKER" = "true" ]; then
        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
            success "Services are running"
        else
            error "Services are not running properly"
            return 1
        fi
    else
        if systemctl is-active --quiet "$SYSTEMD_SERVICE"; then
            success "Service is active"
        else
            error "Service is not active"
            return 1
        fi
    fi
    
    # Run health check
    if ! health_check; then
        error "Health check failed after rollback"
        return 1
    fi
    
    success "Rollback verification completed"
}

# =============================================================================
# Main Rollback Functions
# =============================================================================
perform_code_rollback() {
    local target="$1"
    
    info "Performing code rollback to: $target"
    
    # Validate target
    if ! validate_rollback_target "$target"; then
        exit 1
    fi
    
    # Create pre-rollback backup
    create_pre_rollback_backup
    
    # Stop services
    stop_services
    
    # Rollback code
    if ! rollback_code "$target"; then
        error "Code rollback failed"
        exit 1
    fi
    
    # Rollback dependencies
    rollback_dependencies
    
    # Start services
    if ! start_services; then
        error "Failed to start services after rollback"
        exit 1
    fi
    
    # Verify rollback
    if ! verify_rollback; then
        error "Rollback verification failed"
        exit 1
    fi
    
    success "Code rollback completed successfully"
}

perform_full_rollback() {
    local target="$1"
    
    info "Performing full rollback to: $target"
    
    # Check if target is a backup
    if [ -f "$BACKUP_DIR/$target.tar.gz" ]; then
        info "Rolling back from backup: $target"
        
        # Create pre-rollback backup
        create_pre_rollback_backup
        
        # Stop services
        stop_services
        
        # Restore from backup
        if ! rollback_from_backup "$target"; then
            error "Backup rollback failed"
            exit 1
        fi
        
        # Services should be started by the backup restore process
        
        # Verify rollback
        if ! verify_rollback; then
            error "Rollback verification failed"
            exit 1
        fi
        
        success "Full rollback completed successfully"
    else
        # Fallback to code rollback
        perform_code_rollback "$target"
    fi
}

show_rollback_options() {
    echo "============================================================================="
    info "Rollback Options"
    echo "============================================================================="
    
    get_available_versions
    
    echo
    info "Rollback types:"
    echo "  code     - Rollback code only (preserves data)"
    echo "  full     - Rollback from backup (includes data)"
    echo "  migration - Rollback database migrations only"
    
    echo
    info "Examples:"
    echo "  $0 --target v1.2.3 --type code"
    echo "  $0 --target tyra-backup-20240115-120000 --type full"
    echo "  $0 --target HEAD~1 --type code"
}

confirm_rollback() {
    local target="$1"
    local type="$2"
    
    if [ "$FORCE_ROLLBACK" = "true" ]; then
        return 0
    fi
    
    echo "============================================================================="
    warning "ROLLBACK CONFIRMATION"
    echo "============================================================================="
    echo "Target: $target"
    echo "Type: $type"
    echo "Current version: $(get_current_version)"
    echo
    warning "This operation will modify your system and may result in data loss."
    
    read -p "Are you sure you want to proceed? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        info "Rollback cancelled by user"
        exit 0
    fi
}

show_help() {
    cat << EOF
Tyra MCP Memory Server - Rollback Script

Usage: $0 [OPTIONS]

Options:
    --target TARGET          Rollback target (git ref or backup name)
    --type TYPE              Rollback type: code/full/migration (default: code)
    --list                   List available rollback targets
    --skip-backup            Skip pre-rollback backup
    --force                  Skip confirmation prompt
    --preserve-data          Preserve data during rollback (default: true)
    --docker                 Use Docker Compose for service management
    --systemd                Use systemd for service management
    --help                   Show this help message

Types:
    code        Rollback code only, preserve data
    full        Rollback from backup (includes data)
    migration   Rollback database migrations only

Examples:
    $0 --list                                    # List available targets
    $0 --target v1.2.3 --type code              # Rollback to version 1.2.3
    $0 --target HEAD~1 --type code              # Rollback to previous commit
    $0 --target tyra-backup-20240115-120000 --type full  # Restore from backup

EOF
}

# =============================================================================
# Command Line Interface
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            ROLLBACK_TARGET="$2"
            shift 2
            ;;
        --type)
            ROLLBACK_TYPE="$2"
            shift 2
            ;;
        --list)
            show_rollback_options
            exit 0
            ;;
        --skip-backup)
            SKIP_BACKUP="true"
            shift
            ;;
        --force)
            FORCE_ROLLBACK="true"
            shift
            ;;
        --preserve-data)
            PRESERVE_DATA="true"
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
    info "Tyra MCP Memory Server Rollback"
    echo "============================================================================="
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # If no target specified, show options
    if [ -z "$ROLLBACK_TARGET" ]; then
        show_rollback_options
        exit 0
    fi
    
    # Confirm rollback
    confirm_rollback "$ROLLBACK_TARGET" "$ROLLBACK_TYPE"
    
    # Perform rollback based on type
    case "$ROLLBACK_TYPE" in
        code)
            perform_code_rollback "$ROLLBACK_TARGET"
            ;;
        full)
            perform_full_rollback "$ROLLBACK_TARGET"
            ;;
        migration)
            rollback_migrations
            ;;
        *)
            error "Unknown rollback type: $ROLLBACK_TYPE"
            exit 1
            ;;
    esac
    
    echo "============================================================================="
    success "Rollback completed successfully!"
    echo "============================================================================="
}

# Run main function
main "$@"