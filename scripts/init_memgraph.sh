#!/bin/bash
# Memgraph initialization script for Tyra Memory System

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Memgraph connection settings
MEMGRAPH_HOST="${MEMGRAPH_HOST:-localhost}"
MEMGRAPH_PORT="${MEMGRAPH_PORT:-7687}"
MEMGRAPH_USER="${MEMGRAPH_USER:-}"
MEMGRAPH_PASSWORD="${MEMGRAPH_PASSWORD:-}"

# Check if Memgraph is running
check_memgraph() {
    log "Checking Memgraph connection..."
    
    # Try different connection methods
    if command -v mgconsole &> /dev/null; then
        if echo "RETURN 1;" | mgconsole --host $MEMGRAPH_HOST --port $MEMGRAPH_PORT --username="$MEMGRAPH_USER" --password="$MEMGRAPH_PASSWORD" >/dev/null 2>&1; then
            log "Memgraph connection successful using mgconsole"
            CONNECTION_METHOD="mgconsole"
            return 0
        fi
    fi
    
    # Try cypher-shell if available
    if command -v cypher-shell &> /dev/null; then
        if echo "RETURN 1;" | cypher-shell -a bolt://$MEMGRAPH_HOST:$MEMGRAPH_PORT -u "$MEMGRAPH_USER" -p "$MEMGRAPH_PASSWORD" >/dev/null 2>&1; then
            log "Memgraph connection successful using cypher-shell"
            CONNECTION_METHOD="cypher-shell"
            return 0
        fi
    fi
    
    # Try Python connection using mgclient
    if command -v python3 &> /dev/null; then
        if python3 -c "
import sys
try:
    import mgclient
    conn = mgclient.connect(host='$MEMGRAPH_HOST', port=$MEMGRAPH_PORT, username='$MEMGRAPH_USER', password='$MEMGRAPH_PASSWORD')
    cursor = conn.cursor()
    cursor.execute('RETURN 1')
    conn.close()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
            log "Memgraph connection successful using Python mgclient"
            CONNECTION_METHOD="python"
            return 0
        fi
    fi
    
    error "Cannot connect to Memgraph at $MEMGRAPH_HOST:$MEMGRAPH_PORT"
    error "Please ensure Memgraph is running and accessible"
    error "You may need to install mgconsole, cypher-shell, or Python mgclient"
    return 1
}

# Execute Cypher query
execute_cypher() {
    local query="$1"
    local description="$2"
    
    info "Executing: $description"
    
    case $CONNECTION_METHOD in
        "mgconsole")
            echo "$query" | mgconsole --host $MEMGRAPH_HOST --port $MEMGRAPH_PORT --username="$MEMGRAPH_USER" --password="$MEMGRAPH_PASSWORD"
            ;;
        "cypher-shell")
            echo "$query" | cypher-shell -a bolt://$MEMGRAPH_HOST:$MEMGRAPH_PORT -u "$MEMGRAPH_USER" -p "$MEMGRAPH_PASSWORD"
            ;;
        "python")
            python3 -c "
import mgclient
conn = mgclient.connect(host='$MEMGRAPH_HOST', port=$MEMGRAPH_PORT, username='$MEMGRAPH_USER', password='$MEMGRAPH_PASSWORD')
cursor = conn.cursor()
cursor.execute('''$query''')
for row in cursor:
    print(row)
conn.close()
"
            ;;
        *)
            error "No valid connection method available"
            return 1
            ;;
    esac
}

# Clear existing data (optional)
clear_database() {
    if [[ "${1:-}" == "--clear" ]]; then
        warning "Clearing existing Memgraph data..."
        execute_cypher "MATCH (n) DETACH DELETE n;" "Clear all existing data"
        log "Database cleared"
    fi
}

# Initialize Memgraph schema
init_schema() {
    log "Initializing Memgraph schema..."
    
    # Read and execute the Cypher initialization script
    local script_path="$(dirname "$0")/init_memgraph.cypher"
    if [[ -f "$script_path" ]]; then
        case $CONNECTION_METHOD in
            "mgconsole")
                mgconsole --host $MEMGRAPH_HOST --port $MEMGRAPH_PORT --username="$MEMGRAPH_USER" --password="$MEMGRAPH_PASSWORD" < "$script_path"
                ;;
            "cypher-shell")
                cypher-shell -a bolt://$MEMGRAPH_HOST:$MEMGRAPH_PORT -u "$MEMGRAPH_USER" -p "$MEMGRAPH_PASSWORD" < "$script_path"
                ;;
            "python")
                python3 -c "
import mgclient
conn = mgclient.connect(host='$MEMGRAPH_HOST', port=$MEMGRAPH_PORT, username='$MEMGRAPH_USER', password='$MEMGRAPH_PASSWORD')
cursor = conn.cursor()
with open('$script_path', 'r') as f:
    queries = f.read().split(';')
    for query in queries:
        query = query.strip()
        if query and not query.startswith('//'):
            try:
                cursor.execute(query)
                for row in cursor:
                    print(row)
            except Exception as e:
                if 'already exists' not in str(e):
                    print(f'Warning: {e}')
conn.close()
"
                ;;
        esac
    else
        error "Cypher initialization script not found: $script_path"
        return 1
    fi
    
    log "Schema initialization completed"
}

# Create additional indexes for performance
create_performance_indexes() {
    log "Creating performance indexes..."
    
    # Additional indexes for common query patterns
    execute_cypher "
        CREATE INDEX ON :Entity(importance_score);
        CREATE INDEX ON :Entity(last_accessed);
        CREATE INDEX ON :Memory(confidence_score);
        CREATE INDEX ON :Memory(embedding_id);
        CREATE INDEX ON :Concept(frequency);
        CREATE INDEX ON :Topic(relevance_score);
        CREATE INDEX ON :Event(duration);
        CREATE INDEX ON :Session(duration);
    " "Create performance indexes"
    
    log "Performance indexes created"
}

# Create stored procedures for common operations
create_procedures() {
    log "Creating stored procedures..."
    
    # Entity similarity procedure
    execute_cypher "
        CREATE OR REPLACE PROCEDURE entity_similarity(entity_id STRING, limit INTEGER DEFAULT 10)
        RETURNS (similar_entity STRING, similarity_score FLOAT)
        LANGUAGE python
        AS
        \$\$
        # This would implement entity similarity calculation
        # For now, return a placeholder
        import mgp
        
        for i in range(min(limit, 5)):
            yield mgp.Record(similar_entity=f'entity_{i}', similarity_score=0.8 - i*0.1)
        \$\$;
    " "Create entity similarity procedure" || warning "Stored procedure creation may require additional setup"
    
    # Memory retrieval procedure
    execute_cypher "
        CREATE OR REPLACE PROCEDURE retrieve_related_memories(entity_id STRING, limit INTEGER DEFAULT 10)
        RETURNS (memory_id STRING, relevance_score FLOAT)
        LANGUAGE python
        AS
        \$\$
        # This would implement memory retrieval logic
        # For now, return a placeholder
        import mgp
        
        for i in range(min(limit, 5)):
            yield mgp.Record(memory_id=f'memory_{i}', relevance_score=0.9 - i*0.1)
        \$\$;
    " "Create memory retrieval procedure" || warning "Stored procedure creation may require additional setup"
    
    info "Stored procedures created (some may require additional Memgraph configuration)"
}

# Set up graph algorithms
setup_algorithms() {
    log "Setting up graph algorithms..."
    
    # Enable algorithms (if available)
    execute_cypher "
        CALL mg.load_all();
    " "Load graph algorithms" || info "Graph algorithms loading may require MAGE installation"
    
    log "Graph algorithms setup completed"
}

# Test graph functionality
test_graph() {
    log "Testing graph functionality..."
    
    # Test basic operations
    execute_cypher "
        CREATE (test:TestEntity {id: 'test_entity', name: 'Test Entity', created_at: datetime()});
        MATCH (test:TestEntity {id: 'test_entity'}) RETURN test.name;
        MATCH (test:TestEntity {id: 'test_entity'}) DELETE test;
    " "Test basic graph operations"
    
    # Test constraints
    execute_cypher "
        MATCH (e:Entity) RETURN count(e) as entity_count;
        MATCH ()-[r]->() RETURN count(r) as relationship_count;
        MATCH (s:System) RETURN s.name, s.version;
    " "Test constraints and system data"
    
    log "Graph functionality tests passed"
}

# Update performance statistics
update_statistics() {
    log "Updating performance statistics..."
    
    execute_cypher "
        MATCH (n) WITH count(n) as node_count
        MATCH ()-[r]->() WITH count(r) as rel_count, node_count
        MATCH (perf:Performance {id: 'graph_performance'})
        SET perf.entity_count = node_count,
            perf.relationship_count = rel_count,
            perf.last_updated = datetime()
        RETURN perf.entity_count, perf.relationship_count;
    " "Update performance statistics"
    
    log "Performance statistics updated"
}

# Create backup script
create_backup_script() {
    log "Creating graph backup script..."
    
    local backup_script="$(dirname "$0")/backup_memgraph.sh"
    
    cat > "$backup_script" <<EOF
#!/bin/bash
# Memgraph backup script for Tyra Memory System

set -euo pipefail

BACKUP_DIR="/var/backups/tyra-memory/memgraph"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)

mkdir -p "\$BACKUP_DIR"

# Export all data
echo "Backing up Memgraph data..."
echo "DUMP DATABASE;" | mgconsole --host $MEMGRAPH_HOST --port $MEMGRAPH_PORT --username="$MEMGRAPH_USER" --password="$MEMGRAPH_PASSWORD" > "\$BACKUP_DIR/memgraph_\$TIMESTAMP.cypher"

# Compress backup
gzip "\$BACKUP_DIR/memgraph_\$TIMESTAMP.cypher"

# Cleanup old backups (keep last 7 days)
find "\$BACKUP_DIR" -name "*.cypher.gz" -mtime +7 -delete

echo "Memgraph backup completed: memgraph_\$TIMESTAMP.cypher.gz"
EOF
    
    chmod +x "$backup_script"
    log "Backup script created: $backup_script"
}

# Main function
main() {
    log "Starting Memgraph initialization for Tyra Memory System..."
    
    # Check connection
    if ! check_memgraph; then
        error "Failed to connect to Memgraph. Please check your installation and configuration."
        exit 1
    fi
    
    # Clear database if requested
    clear_database "$@"
    
    # Initialize schema and data
    init_schema
    create_performance_indexes
    create_procedures
    setup_algorithms
    
    # Test and finalize
    test_graph
    update_statistics
    create_backup_script
    
    log "Memgraph initialization completed successfully!"
    log ""
    log "Summary:"
    log "- Knowledge graph schema created with entity types and relationship types"
    log "- Temporal graph structure initialized for time-based organization"
    log "- Performance indexes created for efficient querying"
    log "- System metadata and configuration nodes established"
    log "- Memory health tracking and quality metrics configured"
    log "- Adaptation and learning infrastructure set up"
    log "- Integration points for external systems created"
    log "- Privacy and security controls initialized"
    log "- Backup script created for data protection"
    log ""
    log "Memgraph is ready for the Tyra Memory System!"
    log ""
    info "To clear the database in the future, run: $0 --clear"
    info "To backup the database, run: $(dirname "$0")/backup_memgraph.sh"
}

# Run main function
main "$@"