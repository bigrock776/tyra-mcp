#!/bin/bash
# Tyra MCP Memory Server - Database Setup Script
# This script sets up PostgreSQL, Memgraph, and Redis for the Tyra Memory System

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
POSTGRES_VERSION="15"
POSTGRES_DB="tyra_memory"
POSTGRES_USER="tyra"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-tyra_secure_password}"
MEMGRAPH_VERSION="2.11.0"
REDIS_VERSION="7.0"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Please run as a regular user with sudo privileges."
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            PACKAGE_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            PACKAGE_MANAGER="yum"
        elif command -v pacman &> /dev/null; then
            PACKAGE_MANAGER="pacman"
        else
            error "Unsupported Linux distribution. Please install PostgreSQL, Memgraph, and Redis manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            PACKAGE_MANAGER="brew"
        else
            error "Homebrew is required on macOS. Please install it first."
            exit 1
        fi
    else
        error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check available memory
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        AVAILABLE_MEMORY=$(free -m | awk '/^Mem:/{print $2}')
        if [[ $AVAILABLE_MEMORY -lt 4096 ]]; then
            warning "Less than 4GB of memory available. Database performance may be affected."
        fi
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 10 ]]; then
        warning "Less than 10GB of disk space available. Consider freeing up space."
    fi
    
    log "System requirements check completed."
}

# Install PostgreSQL with pgvector
install_postgresql() {
    log "Installing PostgreSQL $POSTGRES_VERSION with pgvector extension..."
    
    case $PACKAGE_MANAGER in
        "apt")
            # Add PostgreSQL APT repository
            sudo apt-get update
            sudo apt-get install -y wget ca-certificates
            wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
            echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
            
            sudo apt-get update
            sudo apt-get install -y postgresql-$POSTGRES_VERSION postgresql-contrib-$POSTGRES_VERSION
            sudo apt-get install -y postgresql-$POSTGRES_VERSION-pgvector
            sudo apt-get install -y build-essential postgresql-server-dev-$POSTGRES_VERSION
            ;;
        "yum")
            sudo yum install -y postgresql$POSTGRES_VERSION-server postgresql$POSTGRES_VERSION-contrib
            sudo yum install -y postgresql$POSTGRES_VERSION-devel
            sudo postgresql-setup initdb
            ;;
        "pacman")
            sudo pacman -S postgresql postgresql-contrib
            sudo -u postgres initdb -D /var/lib/postgres/data
            ;;
        "brew")
            brew install postgresql@$POSTGRES_VERSION
            brew install pgvector
            brew services start postgresql@$POSTGRES_VERSION
            ;;
    esac
    
    # Start and enable PostgreSQL
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    fi
    
    log "PostgreSQL installation completed."
}

# Configure PostgreSQL
configure_postgresql() {
    log "Configuring PostgreSQL..."
    
    # Create database and user
    sudo -u postgres psql -c "CREATE DATABASE $POSTGRES_DB;"
    sudo -u postgres psql -c "CREATE USER $POSTGRES_USER WITH ENCRYPTED PASSWORD '$POSTGRES_PASSWORD';"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;"
    sudo -u postgres psql -c "ALTER USER $POSTGRES_USER CREATEDB;"
    
    # Create pgvector extension
    sudo -u postgres psql -d $POSTGRES_DB -c "CREATE EXTENSION IF NOT EXISTS vector;"
    
    # Update PostgreSQL configuration for better performance
    PG_VERSION=$(sudo -u postgres psql -t -c "SELECT version();" | head -1 | awk '{print $2}' | cut -d. -f1)
    PG_CONFIG_DIR="/etc/postgresql/$PG_VERSION/main"
    
    if [[ -d "$PG_CONFIG_DIR" ]]; then
        # Backup original configuration
        sudo cp "$PG_CONFIG_DIR/postgresql.conf" "$PG_CONFIG_DIR/postgresql.conf.backup"
        
        # Performance optimizations
        sudo tee -a "$PG_CONFIG_DIR/postgresql.conf" <<EOF

# Tyra Memory System optimizations
shared_preload_libraries = 'pg_stat_statements'
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4
EOF
        
        # Update pg_hba.conf for local connections
        sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = 'localhost'/" "$PG_CONFIG_DIR/postgresql.conf"
        echo "host    $POSTGRES_DB    $POSTGRES_USER    127.0.0.1/32    md5" | sudo tee -a "$PG_CONFIG_DIR/pg_hba.conf"
        
        # Restart PostgreSQL
        sudo systemctl restart postgresql
    fi
    
    log "PostgreSQL configuration completed."
}

# Install Memgraph
install_memgraph() {
    log "Installing Memgraph $MEMGRAPH_VERSION..."
    
    case $PACKAGE_MANAGER in
        "apt")
            # Add Memgraph repository
            curl -fsSL https://download.memgraph.com/memgraph/key | sudo apt-key add -
            echo "deb https://download.memgraph.com/memgraph/v2.11.0/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/memgraph.list
            
            sudo apt-get update
            sudo apt-get install -y memgraph
            ;;
        "yum")
            sudo yum install -y https://download.memgraph.com/memgraph/v2.11.0/centos-7/memgraph-2.11.0-1.x86_64.rpm
            ;;
        "brew")
            brew install memgraph
            ;;
    esac
    
    # Start and enable Memgraph
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start memgraph
        sudo systemctl enable memgraph
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start memgraph
    fi
    
    log "Memgraph installation completed."
}

# Install Redis
install_redis() {
    log "Installing Redis $REDIS_VERSION..."
    
    case $PACKAGE_MANAGER in
        "apt")
            sudo apt-get update
            sudo apt-get install -y redis-server
            ;;
        "yum")
            sudo yum install -y epel-release
            sudo yum install -y redis
            ;;
        "pacman")
            sudo pacman -S redis
            ;;
        "brew")
            brew install redis
            ;;
    esac
    
    # Configure Redis
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Update Redis configuration
        sudo tee -a /etc/redis/redis.conf <<EOF

# Tyra Memory System optimizations
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF
        
        # Start and enable Redis
        sudo systemctl start redis
        sudo systemctl enable redis
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start redis
    fi
    
    log "Redis installation completed."
}

# Initialize databases
initialize_databases() {
    log "Initializing databases..."
    
    # Run PostgreSQL initialization script
    PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost -U $POSTGRES_USER -d $POSTGRES_DB -f "$(dirname "$0")/init_postgres.sql"
    
    # Test database connections
    log "Testing database connections..."
    
    # Test PostgreSQL
    if PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1;" &>/dev/null; then
        log "PostgreSQL connection successful"
    else
        error "PostgreSQL connection failed"
        exit 1
    fi
    
    # Test Memgraph
    if command -v mgconsole &> /dev/null; then
        if echo "RETURN 1;" | mgconsole --host 127.0.0.1 --port 7687 &>/dev/null; then
            log "Memgraph connection successful"
        else
            warning "Memgraph connection failed - check if service is running"
        fi
    else
        info "mgconsole not available - install memgraph-client for testing"
    fi
    
    # Test Redis
    if redis-cli ping | grep -q "PONG"; then
        log "Redis connection successful"
    else
        error "Redis connection failed"
        exit 1
    fi
    
    log "Database initialization completed."
}

# Create environment configuration
create_env_config() {
    log "Creating environment configuration..."
    
    ENV_FILE="$(dirname "$0")/../.env"
    
    cat > "$ENV_FILE" <<EOF
# Tyra MCP Memory Server Database Configuration
# Generated by setup_databases.sh on $(date)

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=$POSTGRES_DB
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_SSL_MODE=prefer

# Memgraph Configuration  
MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687
MEMGRAPH_USER=
MEMGRAPH_PASSWORD=

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Connection Pool Settings
POSTGRES_POOL_SIZE=20
POSTGRES_POOL_TIMEOUT=30
REDIS_POOL_SIZE=50
MEMGRAPH_POOL_SIZE=10

# Performance Settings
VECTOR_DIMENSIONS=1024
HNSW_M=16
HNSW_EF_CONSTRUCTION=64
EMBEDDING_CACHE_TTL=86400
SEARCH_CACHE_TTL=3600
RERANK_CACHE_TTL=1800
EOF
    
    log "Environment configuration created at $ENV_FILE"
}

# Create backup script
create_backup_script() {
    log "Creating backup script..."
    
    BACKUP_SCRIPT="$(dirname "$0")/backup_databases.sh"
    
    cat > "$BACKUP_SCRIPT" <<'EOF'
#!/bin/bash
# Database backup script for Tyra Memory System

set -euo pipefail

BACKUP_DIR="/var/backups/tyra-memory"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Load environment variables
source "$(dirname "$0")/../.env"

# Backup PostgreSQL
echo "Backing up PostgreSQL..."
PGPASSWORD=$POSTGRES_PASSWORD pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB > "$BACKUP_DIR/postgres_$TIMESTAMP.sql"

# Backup Redis
echo "Backing up Redis..."
redis-cli --rdb "$BACKUP_DIR/redis_$TIMESTAMP.rdb"

# Backup Memgraph (if available)
if command -v mgconsole &> /dev/null; then
    echo "Backing up Memgraph..."
    echo "DUMP DATABASE;" | mgconsole --host $MEMGRAPH_HOST --port $MEMGRAPH_PORT > "$BACKUP_DIR/memgraph_$TIMESTAMP.cypher"
fi

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.sql" -o -name "*.rdb" -o -name "*.cypher" | sort | head -n -21 | xargs -r rm

echo "Backup completed successfully to $BACKUP_DIR"
EOF
    
    chmod +x "$BACKUP_SCRIPT"
    log "Backup script created at $BACKUP_SCRIPT"
}

# Main installation function
main() {
    log "Starting Tyra Memory System database setup..."
    
    check_root
    check_requirements
    
    # Install databases
    install_postgresql
    configure_postgresql
    install_memgraph
    install_redis
    
    # Initialize and configure
    initialize_databases
    create_env_config
    create_backup_script
    
    log "Database setup completed successfully!"
    log ""
    log "Summary:"
    log "- PostgreSQL: $POSTGRES_DB database created with user '$POSTGRES_USER'"
    log "- Memgraph: Graph database ready for knowledge graph operations"
    log "- Redis: Cache server configured with 512MB memory limit"
    log "- Environment file: .env created with connection settings"
    log "- Backup script: scripts/backup_databases.sh created"
    log ""
    log "Next steps:"
    log "1. Review the .env file and adjust settings as needed"
    log "2. Run the backup script to test backup functionality"
    log "3. Start the Tyra Memory System server"
    log ""
    warning "IMPORTANT: Change the default passwords in production!"
}

# Run main function
main "$@"