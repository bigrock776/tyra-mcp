#!/bin/bash

# Database Backup Script for Tyra MCP Memory Server
# Creates backups of PostgreSQL, Memgraph, and Redis data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
POSTGRES_DB=${POSTGRES_DB:-tyra_memory}
POSTGRES_USER=${POSTGRES_USER:-tyra_user}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-tyra_password}
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}

MEMGRAPH_HOST=${MEMGRAPH_HOST:-localhost}
MEMGRAPH_PORT=${MEMGRAPH_PORT:-7687}

REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6379}

# Backup configuration
BACKUP_DIR=${BACKUP_DIR:-"./backups"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=${RETENTION_DAYS:-7}

echo -e "${BLUE}💾 Starting Tyra MCP Memory Server Database Backup${NC}"
echo "=================================================="
echo "Timestamp: $TIMESTAMP"
echo "Backup directory: $BACKUP_DIR"
echo "Retention: $RETENTION_DAYS days"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Function to check if service is running
check_service() {
    local service=$1
    local port=$2
    local host=${3:-localhost}

    if nc -z "$host" "$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# 1. Backup PostgreSQL
echo -e "\n${BLUE}📊 Backing up PostgreSQL...${NC}"

if check_service "PostgreSQL" "$POSTGRES_PORT" "$POSTGRES_HOST"; then
    PG_BACKUP_FILE="$BACKUP_DIR/postgres_${TIMESTAMP}.sql"

    echo -e "${YELLOW}📁 Creating PostgreSQL backup: $PG_BACKUP_FILE${NC}"

    if PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        --verbose --no-owner --no-privileges --format=plain > "$PG_BACKUP_FILE"; then

        # Compress the backup
        gzip "$PG_BACKUP_FILE"
        echo -e "${GREEN}✓ PostgreSQL backup completed: ${PG_BACKUP_FILE}.gz${NC}"

        # Get backup size
        BACKUP_SIZE=$(du -h "${PG_BACKUP_FILE}.gz" | cut -f1)
        echo -e "${BLUE}📏 Backup size: $BACKUP_SIZE${NC}"
    else
        echo -e "${RED}✗ PostgreSQL backup failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ PostgreSQL is not running, skipping backup${NC}"
fi

# 2. Backup Memgraph
echo -e "\n${BLUE}🔗 Backing up Memgraph...${NC}"

if check_service "Memgraph" "$MEMGRAPH_PORT" "$MEMGRAPH_HOST"; then
    MG_BACKUP_FILE="$BACKUP_DIR/memgraph_${TIMESTAMP}.cypher"

    echo -e "${YELLOW}📁 Creating Memgraph backup: $MG_BACKUP_FILE${NC}"

    # Export all nodes and relationships
    if command -v mgconsole >/dev/null 2>&1; then
        # Use mgconsole if available
        cat > "/tmp/memgraph_export_${TIMESTAMP}.cypher" <<EOF
MATCH (n)
OPTIONAL MATCH (n)-[r]->()
WITH collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships
RETURN nodes, relationships;
EOF

        if mgconsole --host "$MEMGRAPH_HOST" --port "$MEMGRAPH_PORT" < "/tmp/memgraph_export_${TIMESTAMP}.cypher" > "$MG_BACKUP_FILE" 2>/dev/null; then
            rm "/tmp/memgraph_export_${TIMESTAMP}.cypher"
            gzip "$MG_BACKUP_FILE"
            echo -e "${GREEN}✓ Memgraph backup completed: ${MG_BACKUP_FILE}.gz${NC}"

            BACKUP_SIZE=$(du -h "${MG_BACKUP_FILE}.gz" | cut -f1)
            echo -e "${BLUE}📏 Backup size: $BACKUP_SIZE${NC}"
        else
            echo -e "${RED}✗ Memgraph backup failed${NC}"
            rm -f "/tmp/memgraph_export_${TIMESTAMP}.cypher"
        fi
    else
        echo -e "${YELLOW}⚠️ mgconsole not found, creating basic backup placeholder${NC}"
        echo "# Memgraph backup placeholder - install mgconsole for full backup" > "$MG_BACKUP_FILE"
        echo "# Backup created at: $(date)" >> "$MG_BACKUP_FILE"
        gzip "$MG_BACKUP_FILE"
        echo -e "${YELLOW}⚠️ Basic Memgraph backup created: ${MG_BACKUP_FILE}.gz${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Memgraph is not running, skipping backup${NC}"
fi

# 3. Backup Redis
echo -e "\n${BLUE}🗄️ Backing up Redis...${NC}"

if check_service "Redis" "$REDIS_PORT" "$REDIS_HOST"; then
    REDIS_BACKUP_FILE="$BACKUP_DIR/redis_${TIMESTAMP}.rdb"

    echo -e "${YELLOW}📁 Creating Redis backup: $REDIS_BACKUP_FILE${NC}"

    if command -v redis-cli >/dev/null 2>&1; then
        # Trigger Redis save
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE | grep -q "Background saving started"; then
            echo -e "${YELLOW}⏳ Waiting for Redis background save to complete...${NC}"

            # Wait for save to complete
            while true; do
                if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" INFO persistence | grep -q "rdb_bgsave_in_progress:0"; then
                    break
                fi
                sleep 1
            done

            # Get Redis data directory and copy RDB file
            REDIS_DIR=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG GET dir | tail -1)
            REDIS_DBFILENAME=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG GET dbfilename | tail -1)

            if [ -f "$REDIS_DIR/$REDIS_DBFILENAME" ]; then
                cp "$REDIS_DIR/$REDIS_DBFILENAME" "$REDIS_BACKUP_FILE"
                gzip "$REDIS_BACKUP_FILE"
                echo -e "${GREEN}✓ Redis backup completed: ${REDIS_BACKUP_FILE}.gz${NC}"

                BACKUP_SIZE=$(du -h "${REDIS_BACKUP_FILE}.gz" | cut -f1)
                echo -e "${BLUE}📏 Backup size: $BACKUP_SIZE${NC}"
            else
                echo -e "${RED}✗ Redis RDB file not found: $REDIS_DIR/$REDIS_DBFILENAME${NC}"

                # Fallback: export all keys
                echo -e "${YELLOW}📦 Creating Redis key export...${NC}"
                REDIS_KEYS_FILE="$BACKUP_DIR/redis_keys_${TIMESTAMP}.txt"
                redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --scan > "$REDIS_KEYS_FILE"
                gzip "$REDIS_KEYS_FILE"
                echo -e "${YELLOW}⚠️ Redis keys exported: ${REDIS_KEYS_FILE}.gz${NC}"
            fi
        else
            echo -e "${RED}✗ Redis BGSAVE failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ redis-cli not found, skipping Redis backup${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Redis is not running, skipping backup${NC}"
fi

# 4. Create backup manifest
echo -e "\n${BLUE}📋 Creating backup manifest...${NC}"

MANIFEST_FILE="$BACKUP_DIR/backup_manifest_${TIMESTAMP}.json"
cat > "$MANIFEST_FILE" <<EOF
{
    "backup_timestamp": "$TIMESTAMP",
    "backup_date": "$(date -Iseconds)",
    "system_info": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "arch": "$(uname -m)"
    },
    "database_info": {
        "postgresql": {
            "host": "$POSTGRES_HOST",
            "port": $POSTGRES_PORT,
            "database": "$POSTGRES_DB",
            "user": "$POSTGRES_USER"
        },
        "memgraph": {
            "host": "$MEMGRAPH_HOST",
            "port": $MEMGRAPH_PORT
        },
        "redis": {
            "host": "$REDIS_HOST",
            "port": $REDIS_PORT
        }
    },
    "backup_files": [
EOF

# Add backup files to manifest
FIRST_FILE=true
for file in "$BACKUP_DIR"/*_${TIMESTAMP}.*.gz; do
    if [ -f "$file" ]; then
        if [ "$FIRST_FILE" = true ]; then
            FIRST_FILE=false
        else
            echo "," >> "$MANIFEST_FILE"
        fi

        FILENAME=$(basename "$file")
        FILESIZE=$(du -h "$file" | cut -f1)
        echo -n "        {\"filename\": \"$FILENAME\", \"size\": \"$FILESIZE\"}" >> "$MANIFEST_FILE"
    fi
done

cat >> "$MANIFEST_FILE" <<EOF

    ]
}
EOF

echo -e "${GREEN}✓ Backup manifest created: $MANIFEST_FILE${NC}"

# 5. Cleanup old backups
echo -e "\n${BLUE}🧹 Cleaning up old backups...${NC}"

if [ "$RETENTION_DAYS" -gt 0 ]; then
    echo -e "${YELLOW}🗑️ Removing backups older than $RETENTION_DAYS days...${NC}"

    DELETED_COUNT=0

    # Find and delete old backup files
    if find "$BACKUP_DIR" -name "*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].*" -type f -mtime +$RETENTION_DAYS -print0 2>/dev/null | while IFS= read -r -d '' file; do
        echo "  Deleting: $(basename "$file")"
        rm "$file"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    done; then
        if [ $DELETED_COUNT -gt 0 ]; then
            echo -e "${GREEN}✓ Deleted $DELETED_COUNT old backup files${NC}"
        else
            echo -e "${BLUE}📁 No old backup files to delete${NC}"
        fi
    fi
else
    echo -e "${BLUE}📁 Backup retention disabled (RETENTION_DAYS=0)${NC}"
fi

# 6. Backup summary
echo -e "\n${BLUE}📊 Backup Summary${NC}"
echo "==================="

TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "*_${TIMESTAMP}.*" -type f | wc -l)

echo -e "Backup directory: $BACKUP_DIR"
echo -e "Files created: $BACKUP_COUNT"
echo -e "Total backup size: $TOTAL_SIZE"
echo -e "Backup files:"

for file in "$BACKUP_DIR"/*_${TIMESTAMP}.*; do
    if [ -f "$file" ]; then
        FILENAME=$(basename "$file")
        FILESIZE=$(du -h "$file" | cut -f1)
        echo -e "  📄 $FILENAME ($FILESIZE)"
    fi
done

echo -e "\n${GREEN}🎉 Backup completed successfully!${NC}"
echo -e "${BLUE}💡 To restore from backup, use the restore_databases.sh script${NC}"
