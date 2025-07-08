#!/bin/bash
# =============================================================================
# Database Backup Script for Tyra Memory Server
# =============================================================================

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/tyra-memory}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Database credentials (can be overridden by environment variables)
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-tyra_memory}"
POSTGRES_USER="${POSTGRES_USER:-tyra}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-tyra_secure_password}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-tyra_redis_password}"

MEMGRAPH_HOST="${MEMGRAPH_HOST:-localhost}"
MEMGRAPH_PORT="${MEMGRAPH_PORT:-7687}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

create_backup_dir() {
    log_info "Creating backup directory structure..."

    mkdir -p "$BACKUP_DIR/postgresql"
    mkdir -p "$BACKUP_DIR/redis"
    mkdir -p "$BACKUP_DIR/memgraph"

    log_success "Backup directories created"
}

backup_postgresql() {
    log_info "Backing up PostgreSQL database..."

    local backup_file="$BACKUP_DIR/postgresql/tyra_memory_${TIMESTAMP}.sql"
    local backup_file_gz="${backup_file}.gz"

    # Create PostgreSQL backup
    export PGPASSWORD="$POSTGRES_PASSWORD"
    if pg_dump -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        --verbose --clean --if-exists --create > "$backup_file"; then

        # Compress the backup
        gzip "$backup_file"

        # Create metadata file
        cat > "$BACKUP_DIR/postgresql/tyra_memory_${TIMESTAMP}.meta" << EOF
{
    "timestamp": "$TIMESTAMP",
    "database": "$POSTGRES_DB",
    "host": "$POSTGRES_HOST",
    "port": $POSTGRES_PORT,
    "user": "$POSTGRES_USER",
    "backup_file": "$(basename "$backup_file_gz")",
    "size_bytes": $(stat -c%s "$backup_file_gz"),
    "backup_type": "full"
}
EOF

        log_success "PostgreSQL backup created: $(basename "$backup_file_gz")"
        return 0
    else
        log_error "PostgreSQL backup failed"
        return 1
    fi
}

backup_redis() {
    log_info "Backing up Redis database..."

    local backup_file="$BACKUP_DIR/redis/redis_${TIMESTAMP}.rdb"

    # Create Redis backup using BGSAVE and copy the dump file
    if command -v redis-cli &> /dev/null; then
        # Trigger background save
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" BGSAVE

        # Wait for backup to complete
        log_info "Waiting for Redis background save to complete..."
        while [ "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" LASTSAVE)" = "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" LASTSAVE)" ]; do
            sleep 1
        done

        # Copy the dump file
        if [ -f "/var/lib/redis/dump.rdb" ]; then
            cp "/var/lib/redis/dump.rdb" "$backup_file"

            # Create metadata file
            cat > "$BACKUP_DIR/redis/redis_${TIMESTAMP}.meta" << EOF
{
    "timestamp": "$TIMESTAMP",
    "host": "$REDIS_HOST",
    "port": $REDIS_PORT,
    "backup_file": "$(basename "$backup_file")",
    "size_bytes": $(stat -c%s "$backup_file"),
    "backup_type": "rdb"
}
EOF

            log_success "Redis backup created: $(basename "$backup_file")"
            return 0
        else
            log_error "Redis dump file not found"
            return 1
        fi
    else
        log_error "redis-cli not available"
        return 1
    fi
}

backup_memgraph() {
    log_info "Backing up Memgraph database..."

    local backup_file="$BACKUP_DIR/memgraph/memgraph_${TIMESTAMP}.cypherl"

    # Create Memgraph backup using Cypher export
    if command -v mgconsole &> /dev/null; then
        # Export all data using Cypher queries
        cat > "/tmp/memgraph_export_${TIMESTAMP}.cypher" << 'EOF'
CALL mg.load_all() YIELD *;
MATCH (n) OPTIONAL MATCH (n)-[r]->(m)
RETURN
  'CREATE (' +
  CASE WHEN n.id IS NOT NULL THEN 'n' + toString(n.id) ELSE 'n' + toString(id(n)) END +
  ':' + head(labels(n)) +
  ' {' +
  reduce(props = '', key IN keys(n) |
    props + CASE WHEN props = '' THEN '' ELSE ', ' END + key + ': ' +
    CASE
      WHEN apoc.meta.type(n[key]) = 'STRING' THEN '"' + toString(n[key]) + '"'
      ELSE toString(n[key])
    END
  ) +
  '})' +
  CASE WHEN r IS NOT NULL THEN
    ' CREATE (' +
    CASE WHEN n.id IS NOT NULL THEN 'n' + toString(n.id) ELSE 'n' + toString(id(n)) END +
    ')-[r' + toString(id(r)) + ':' + type(r) +
    CASE WHEN size(keys(r)) > 0 THEN
      ' {' +
      reduce(props = '', key IN keys(r) |
        props + CASE WHEN props = '' THEN '' ELSE ', ' END + key + ': ' +
        CASE
          WHEN apoc.meta.type(r[key]) = 'STRING' THEN '"' + toString(r[key]) + '"'
          ELSE toString(r[key])
        END
      ) +
      '}'
    ELSE ''
    END +
    ']->(' +
    CASE WHEN m.id IS NOT NULL THEN 'n' + toString(m.id) ELSE 'n' + toString(id(m)) END +
    ')'
  ELSE ''
  END AS statement;
EOF

        # Execute export query and save to backup file
        if mgconsole --host="$MEMGRAPH_HOST" --port="$MEMGRAPH_PORT" < "/tmp/memgraph_export_${TIMESTAMP}.cypher" > "$backup_file"; then
            # Clean up temporary file
            rm "/tmp/memgraph_export_${TIMESTAMP}.cypher"

            # Create metadata file
            cat > "$BACKUP_DIR/memgraph/memgraph_${TIMESTAMP}.meta" << EOF
{
    "timestamp": "$TIMESTAMP",
    "host": "$MEMGRAPH_HOST",
    "port": $MEMGRAPH_PORT,
    "backup_file": "$(basename "$backup_file")",
    "size_bytes": $(stat -c%s "$backup_file"),
    "backup_type": "cypher"
}
EOF

            log_success "Memgraph backup created: $(basename "$backup_file")"
            return 0
        else
            log_error "Memgraph backup failed"
            rm "/tmp/memgraph_export_${TIMESTAMP}.cypher"
            return 1
        fi
    else
        log_warning "mgconsole not available, skipping Memgraph backup"
        return 0
    fi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups (keeping last $RETENTION_DAYS days)..."

    # Clean PostgreSQL backups
    find "$BACKUP_DIR/postgresql" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/postgresql" -name "*.meta" -mtime +$RETENTION_DAYS -delete

    # Clean Redis backups
    find "$BACKUP_DIR/redis" -name "*.rdb" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/redis" -name "*.meta" -mtime +$RETENTION_DAYS -delete

    # Clean Memgraph backups
    find "$BACKUP_DIR/memgraph" -name "*.cypherl" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/memgraph" -name "*.meta" -mtime +$RETENTION_DAYS -delete

    log_success "Old backups cleaned up"
}

create_backup_summary() {
    log_info "Creating backup summary..."

    local summary_file="$BACKUP_DIR/backup_summary_${TIMESTAMP}.json"

    cat > "$summary_file" << EOF
{
    "backup_timestamp": "$TIMESTAMP",
    "backup_date": "$(date -Iseconds)",
    "databases": {
        "postgresql": {
            "status": "$([ -f "$BACKUP_DIR/postgresql/tyra_memory_${TIMESTAMP}.sql.gz" ] && echo "success" || echo "failed")",
            "file": "tyra_memory_${TIMESTAMP}.sql.gz"
        },
        "redis": {
            "status": "$([ -f "$BACKUP_DIR/redis/redis_${TIMESTAMP}.rdb" ] && echo "success" || echo "failed")",
            "file": "redis_${TIMESTAMP}.rdb"
        },
        "memgraph": {
            "status": "$([ -f "$BACKUP_DIR/memgraph/memgraph_${TIMESTAMP}.cypherl" ] && echo "success" || echo "failed")",
            "file": "memgraph_${TIMESTAMP}.cypherl"
        }
    },
    "total_size_bytes": $(du -sb "$BACKUP_DIR" | cut -f1),
    "retention_days": $RETENTION_DAYS
}
EOF

    log_success "Backup summary created: $(basename "$summary_file")"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --retention-days)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --postgresql-only)
            POSTGRESQL_ONLY=true
            shift
            ;;
        --redis-only)
            REDIS_ONLY=true
            shift
            ;;
        --memgraph-only)
            MEMGRAPH_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --backup-dir DIR       Backup directory (default: /var/backups/tyra-memory)"
            echo "  --retention-days DAYS  Days to keep backups (default: 30)"
            echo "  --postgresql-only      Backup only PostgreSQL"
            echo "  --redis-only          Backup only Redis"
            echo "  --memgraph-only       Backup only Memgraph"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main backup execution
main() {
    log_info "Starting Tyra Memory Server backup..."
    log_info "Backup directory: $BACKUP_DIR"
    log_info "Retention period: $RETENTION_DAYS days"

    create_backup_dir

    local exit_code=0

    # Perform backups based on options
    if [ "$POSTGRESQL_ONLY" = true ]; then
        backup_postgresql || exit_code=1
    elif [ "$REDIS_ONLY" = true ]; then
        backup_redis || exit_code=1
    elif [ "$MEMGRAPH_ONLY" = true ]; then
        backup_memgraph || exit_code=1
    else
        # Backup all databases
        backup_postgresql || exit_code=1
        backup_redis || exit_code=1
        backup_memgraph || exit_code=1
    fi

    cleanup_old_backups
    create_backup_summary

    if [ $exit_code -eq 0 ]; then
        log_success "Backup completed successfully"
    else
        log_error "Backup completed with errors"
    fi

    return $exit_code
}

# Run main function
main
