#!/bin/bash
# =============================================================================
# Tyra MCP Memory Server - Database Migration Script
# =============================================================================
# Handles database schema updates and data migrations

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MIGRATIONS_DIR="$PROJECT_ROOT/migrations"

# Database configuration
DATABASE_URL="${DATABASE_URL:-postgresql://tyra:tyra123@localhost:5432/tyra_memory}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
MEMGRAPH_URL="${MEMGRAPH_URL:-bolt://localhost:7687}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================
info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# =============================================================================
# Database Connection Functions
# =============================================================================
check_postgres_connection() {
    info "Checking PostgreSQL connection..."

    if psql "$DATABASE_URL" -c "SELECT 1;" > /dev/null 2>&1; then
        success "PostgreSQL connection successful"
        return 0
    else
        error "Failed to connect to PostgreSQL"
        return 1
    fi
}

check_redis_connection() {
    info "Checking Redis connection..."

    # Extract Redis host and port from URL
    REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's|redis://\([^:]*\).*|\1|p')
    REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's|redis://[^:]*:\([0-9]*\).*|\1|p')
    REDIS_PORT=${REDIS_PORT:-6379}

    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
        success "Redis connection successful"
        return 0
    else
        error "Failed to connect to Redis"
        return 1
    fi
}

check_memgraph_connection() {
    info "Checking Memgraph connection..."

    # Extract Memgraph host and port from URL
    MEMGRAPH_HOST=$(echo "$MEMGRAPH_URL" | sed -n 's|bolt://\([^:]*\).*|\1|p')
    MEMGRAPH_PORT=$(echo "$MEMGRAPH_URL" | sed -n 's|bolt://[^:]*:\([0-9]*\).*|\1|p')
    MEMGRAPH_PORT=${MEMGRAPH_PORT:-7687}

    if echo "RETURN 1;" | mgconsole --host "$MEMGRAPH_HOST" --port "$MEMGRAPH_PORT" > /dev/null 2>&1; then
        success "Memgraph connection successful"
        return 0
    else
        warning "Memgraph connection failed or mgconsole not available"
        return 1
    fi
}

# =============================================================================
# Migration Functions
# =============================================================================
create_migration_table() {
    info "Creating migration tracking table..."

    psql "$DATABASE_URL" << 'EOF'
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    checksum VARCHAR(255)
);
EOF

    success "Migration tracking table ready"
}

get_applied_migrations() {
    psql "$DATABASE_URL" -t -c "SELECT version FROM schema_migrations ORDER BY version;" | tr -d ' '
}

calculate_checksum() {
    local file="$1"
    sha256sum "$file" | cut -d' ' -f1
}

apply_sql_migration() {
    local migration_file="$1"
    local version="$2"
    local checksum="$3"

    info "Applying migration: $version"

    # Apply the migration
    if psql "$DATABASE_URL" -f "$migration_file"; then
        # Record successful migration
        psql "$DATABASE_URL" -c "INSERT INTO schema_migrations (version, checksum) VALUES ('$version', '$checksum');"
        success "Migration $version applied successfully"
        return 0
    else
        error "Failed to apply migration $version"
        return 1
    fi
}

apply_python_migration() {
    local migration_file="$1"
    local version="$2"
    local checksum="$3"

    info "Applying Python migration: $version"

    # Set environment variables for the migration script
    export DATABASE_URL REDIS_URL MEMGRAPH_URL

    # Run the Python migration
    if python "$migration_file"; then
        # Record successful migration
        psql "$DATABASE_URL" -c "INSERT INTO schema_migrations (version, checksum) VALUES ('$version', '$checksum');"
        success "Python migration $version applied successfully"
        return 0
    else
        error "Failed to apply Python migration $version"
        return 1
    fi
}

run_migrations() {
    info "Running database migrations..."

    # Create migrations directory if it doesn't exist
    mkdir -p "$MIGRATIONS_DIR"

    # Get currently applied migrations
    local applied_migrations
    applied_migrations=$(get_applied_migrations)

    # Find all migration files
    local migration_files
    migration_files=$(find "$MIGRATIONS_DIR" -name "*.sql" -o -name "*.py" | sort)

    if [ -z "$migration_files" ]; then
        info "No migration files found"
        return 0
    fi

    local migration_count=0

    for migration_file in $migration_files; do
        local filename=$(basename "$migration_file")
        local version="${filename%.*}"  # Remove extension

        # Check if migration is already applied
        if echo "$applied_migrations" | grep -q "^$version$"; then
            info "Migration $version already applied, skipping"
            continue
        fi

        # Calculate checksum
        local checksum=$(calculate_checksum "$migration_file")

        # Apply migration based on file type
        if [[ "$migration_file" == *.sql ]]; then
            if apply_sql_migration "$migration_file" "$version" "$checksum"; then
                ((migration_count++))
            else
                error "Migration failed, stopping"
                return 1
            fi
        elif [[ "$migration_file" == *.py ]]; then
            if apply_python_migration "$migration_file" "$version" "$checksum"; then
                ((migration_count++))
            else
                error "Migration failed, stopping"
                return 1
            fi
        fi
    done

    if [ $migration_count -eq 0 ]; then
        info "All migrations are up to date"
    else
        success "Applied $migration_count new migrations"
    fi
}

# =============================================================================
# Rollback Functions
# =============================================================================
rollback_migration() {
    local version="$1"

    info "Rolling back migration: $version"

    # Look for rollback file
    local rollback_file="$MIGRATIONS_DIR/${version}_rollback.sql"

    if [ -f "$rollback_file" ]; then
        if psql "$DATABASE_URL" -f "$rollback_file"; then
            # Remove migration record
            psql "$DATABASE_URL" -c "DELETE FROM schema_migrations WHERE version = '$version';"
            success "Migration $version rolled back successfully"
            return 0
        else
            error "Failed to rollback migration $version"
            return 1
        fi
    else
        error "Rollback file not found: $rollback_file"
        return 1
    fi
}

# =============================================================================
# Backup Functions
# =============================================================================
create_backup_before_migration() {
    info "Creating backup before migration..."

    local backup_dir="/tmp/tyra-migration-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup PostgreSQL
    pg_dump "$DATABASE_URL" > "$backup_dir/postgres_backup.sql"

    # Backup Redis (if possible)
    if command -v redis-cli &> /dev/null; then
        REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's|redis://\([^:]*\).*|\1|p')
        REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's|redis://[^:]*:\([0-9]*\).*|\1|p')
        REDIS_PORT=${REDIS_PORT:-6379}

        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --rdb "$backup_dir/redis_backup.rdb" > /dev/null 2>&1 || true
    fi

    success "Backup created at: $backup_dir"
    echo "$backup_dir"
}

# =============================================================================
# Main Functions
# =============================================================================
migrate() {
    local create_backup="${1:-true}"

    echo "============================================================================="
    info "Tyra MCP Memory Server - Database Migration"
    echo "============================================================================="

    # Check connections
    if ! check_postgres_connection; then
        error "Cannot proceed without PostgreSQL connection"
        exit 1
    fi

    check_redis_connection || warning "Redis connection failed - some features may not work"
    check_memgraph_connection || warning "Memgraph connection failed - graph features may not work"

    # Create backup if requested
    if [ "$create_backup" = "true" ]; then
        BACKUP_DIR=$(create_backup_before_migration)
        info "Backup location: $BACKUP_DIR"
    fi

    # Create migration tracking table
    create_migration_table

    # Run migrations
    run_migrations

    success "Database migration completed successfully"
}

status() {
    echo "============================================================================="
    info "Migration Status"
    echo "============================================================================="

    if ! check_postgres_connection; then
        error "Cannot check status without PostgreSQL connection"
        exit 1
    fi

    # Check if migration table exists
    if ! psql "$DATABASE_URL" -c "SELECT 1 FROM schema_migrations LIMIT 1;" > /dev/null 2>&1; then
        warning "Migration tracking table does not exist"
        info "Run '$0 migrate' to initialize"
        return
    fi

    info "Applied migrations:"
    psql "$DATABASE_URL" -c "SELECT version, applied_at FROM schema_migrations ORDER BY applied_at;"

    info "Pending migrations:"
    local applied_migrations=$(get_applied_migrations)
    local pending_count=0

    if [ -d "$MIGRATIONS_DIR" ]; then
        for migration_file in $(find "$MIGRATIONS_DIR" -name "*.sql" -o -name "*.py" | sort); do
            local filename=$(basename "$migration_file")
            local version="${filename%.*}"

            if ! echo "$applied_migrations" | grep -q "^$version$"; then
                echo "  - $version"
                ((pending_count++))
            fi
        done
    fi

    if [ $pending_count -eq 0 ]; then
        success "No pending migrations"
    else
        info "$pending_count pending migrations found"
    fi
}

show_help() {
    cat << EOF
Tyra MCP Memory Server - Database Migration Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    migrate             Run pending migrations (default)
    status              Show migration status
    rollback VERSION    Rollback specific migration
    help                Show this help message

Options:
    --no-backup         Skip backup creation before migration
    --force             Force migration even if connections fail

Environment Variables:
    DATABASE_URL        PostgreSQL connection string
    REDIS_URL          Redis connection string
    MEMGRAPH_URL       Memgraph connection string

Examples:
    $0                  # Run pending migrations with backup
    $0 migrate --no-backup
    $0 status
    $0 rollback 001_initial_schema

EOF
}

# =============================================================================
# Command Line Interface
# =============================================================================
COMMAND="${1:-migrate}"
CREATE_BACKUP="true"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        migrate)
            COMMAND="migrate"
            shift
            ;;
        status)
            COMMAND="status"
            shift
            ;;
        rollback)
            COMMAND="rollback"
            ROLLBACK_VERSION="$2"
            shift 2
            ;;
        --no-backup)
            CREATE_BACKUP="false"
            shift
            ;;
        --force)
            # Force migration (could add logic for this)
            shift
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            if [ "$COMMAND" = "migrate" ]; then
                shift
            else
                error "Unknown argument: $1"
                show_help
                exit 1
            fi
            ;;
    esac
done

# Execute command
case $COMMAND in
    migrate)
        migrate "$CREATE_BACKUP"
        ;;
    status)
        status
        ;;
    rollback)
        if [ -z "${ROLLBACK_VERSION:-}" ]; then
            error "Rollback version not specified"
            show_help
            exit 1
        fi
        rollback_migration "$ROLLBACK_VERSION"
        ;;
    *)
        error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
