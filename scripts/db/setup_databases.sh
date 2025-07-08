#!/bin/bash

# Database Setup Script for Tyra MCP Memory Server
# Sets up PostgreSQL with pgvector, Memgraph, and Redis

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

echo -e "${BLUE}🚀 Starting Tyra MCP Memory Server Database Setup${NC}"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if service is running
check_service() {
    local service=$1
    local port=$2
    local host=${3:-localhost}

    if nc -z "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}✓ $service is running on $host:$port${NC}"
        return 0
    else
        echo -e "${RED}✗ $service is not running on $host:$port${NC}"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local host=${3:-localhost}
    local max_attempts=${4:-30}
    local attempt=0

    echo -e "${YELLOW}⏳ Waiting for $service to start...${NC}"

    while [ $attempt -lt $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo -e "${GREEN}✓ $service is ready!${NC}"
            return 0
        fi

        attempt=$((attempt + 1))
        echo -e "${YELLOW}   Attempt $attempt/$max_attempts...${NC}"
        sleep 2
    done

    echo -e "${RED}✗ $service failed to start after $max_attempts attempts${NC}"
    return 1
}

# 1. Setup PostgreSQL with pgvector
echo -e "\n${BLUE}📊 Setting up PostgreSQL with pgvector...${NC}"

if command_exists psql; then
    echo -e "${GREEN}✓ PostgreSQL client found${NC}"
else
    echo -e "${RED}✗ PostgreSQL client not found. Please install PostgreSQL.${NC}"
    echo "Ubuntu/Debian: sudo apt-get install postgresql-client"
    echo "macOS: brew install postgresql"
    exit 1
fi

# Check if PostgreSQL is running
if check_service "PostgreSQL" "$POSTGRES_PORT" "$POSTGRES_HOST"; then
    # Test connection
    echo -e "${YELLOW}🔌 Testing PostgreSQL connection...${NC}"

    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -c "\q" 2>/dev/null; then
        echo -e "${GREEN}✓ PostgreSQL connection successful${NC}"
    else
        echo -e "${YELLOW}⚠️ Creating PostgreSQL user and database...${NC}"

        # Try to create user and database (requires superuser access)
        if PGPASSWORD="postgres" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -d postgres <<EOF
CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';
CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;
GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;
\q
EOF
        then
            echo -e "${GREEN}✓ PostgreSQL user and database created${NC}"
        else
            echo -e "${RED}✗ Failed to create PostgreSQL user/database${NC}"
            echo "Please create manually or check your PostgreSQL installation"
        fi
    fi

    # Install pgvector extension
    echo -e "${YELLOW}🔧 Installing pgvector extension...${NC}"
    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<EOF
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
\q
EOF
    then
        echo -e "${GREEN}✓ pgvector extension installed${NC}"
    else
        echo -e "${RED}✗ Failed to install pgvector extension${NC}"
        echo "Please install pgvector: https://github.com/pgvector/pgvector"
    fi
else
    echo -e "${YELLOW}⚠️ PostgreSQL is not running. Please start PostgreSQL first.${NC}"
    echo "Ubuntu/Debian: sudo systemctl start postgresql"
    echo "macOS: brew services start postgresql"
fi

# 2. Setup Memgraph
echo -e "\n${BLUE}🔗 Setting up Memgraph...${NC}"

if check_service "Memgraph" "$MEMGRAPH_PORT" "$MEMGRAPH_HOST"; then
    echo -e "${GREEN}✓ Memgraph is already running${NC}"

    # Test Memgraph connection
    echo -e "${YELLOW}🔌 Testing Memgraph connection...${NC}"
    if command_exists mgconsole; then
        if echo "SHOW CONFIG;" | mgconsole --host "$MEMGRAPH_HOST" --port "$MEMGRAPH_PORT" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Memgraph connection successful${NC}"
        else
            echo -e "${YELLOW}⚠️ Memgraph connection failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ mgconsole not found, skipping connection test${NC}"
        echo "Install mgconsole for testing: https://memgraph.com/docs/getting-started"
    fi
else
    echo -e "${YELLOW}⚠️ Memgraph is not running.${NC}"
    echo "Please install and start Memgraph:"
    echo "Docker: docker run -p 7687:7687 -p 7444:7444 memgraph/memgraph"
    echo "Or follow: https://memgraph.com/docs/getting-started/install-memgraph"
fi

# 3. Setup Redis
echo -e "\n${BLUE}🗄️ Setting up Redis...${NC}"

if check_service "Redis" "$REDIS_PORT" "$REDIS_HOST"; then
    echo -e "${GREEN}✓ Redis is already running${NC}"

    # Test Redis connection
    echo -e "${YELLOW}🔌 Testing Redis connection...${NC}"
    if command_exists redis-cli; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping | grep -q PONG; then
            echo -e "${GREEN}✓ Redis connection successful${NC}"

            # Configure Redis for memory usage
            echo -e "${YELLOW}🔧 Configuring Redis...${NC}"
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG SET maxmemory-policy allkeys-lru
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG SET maxmemory 1gb
            echo -e "${GREEN}✓ Redis configured for LRU eviction${NC}"
        else
            echo -e "${YELLOW}⚠️ Redis connection failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ redis-cli not found, skipping connection test${NC}"
        echo "Install redis-cli: sudo apt-get install redis-tools"
    fi
else
    echo -e "${YELLOW}⚠️ Redis is not running.${NC}"
    echo "Please install and start Redis:"
    echo "Ubuntu/Debian: sudo apt-get install redis-server && sudo systemctl start redis"
    echo "macOS: brew install redis && brew services start redis"
    echo "Docker: docker run -p 6379:6379 redis:alpine"
fi

# 4. Run database migrations
echo -e "\n${BLUE}🔄 Running database migrations...${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS_DIR="$SCRIPT_DIR/../../migrations/sql"

if [ -d "$MIGRATIONS_DIR" ]; then
    echo -e "${YELLOW}📁 Found migrations directory: $MIGRATIONS_DIR${NC}"

    # Run PostgreSQL migrations
    if [ -f "$MIGRATIONS_DIR/001_initial_schema.sql" ]; then
        echo -e "${YELLOW}🔄 Running PostgreSQL migrations...${NC}"
        if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$MIGRATIONS_DIR/001_initial_schema.sql"; then
            echo -e "${GREEN}✓ PostgreSQL migrations completed${NC}"
        else
            echo -e "${RED}✗ PostgreSQL migrations failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ No PostgreSQL migration files found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Migrations directory not found: $MIGRATIONS_DIR${NC}"
fi

# 5. Verify setup
echo -e "\n${BLUE}✅ Verifying database setup...${NC}"

echo -e "${YELLOW}📊 Database Status Summary:${NC}"
echo "================================"

# PostgreSQL status
if check_service "PostgreSQL" "$POSTGRES_PORT" "$POSTGRES_HOST"; then
    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT version();" >/dev/null 2>&1; then
        PG_VERSION=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT version();" 2>/dev/null | head -1 | xargs)
        echo -e "PostgreSQL: ${GREEN}✓ Ready${NC} ($PG_VERSION)"

        # Check pgvector
        if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT extname FROM pg_extension WHERE extname = 'vector';" 2>/dev/null | grep -q vector; then
            echo -e "pgvector:   ${GREEN}✓ Installed${NC}"
        else
            echo -e "pgvector:   ${RED}✗ Not installed${NC}"
        fi
    else
        echo -e "PostgreSQL: ${RED}✗ Connection failed${NC}"
    fi
else
    echo -e "PostgreSQL: ${RED}✗ Not running${NC}"
fi

# Memgraph status
if check_service "Memgraph" "$MEMGRAPH_PORT" "$MEMGRAPH_HOST"; then
    echo -e "Memgraph:   ${GREEN}✓ Running${NC}"
else
    echo -e "Memgraph:   ${RED}✗ Not running${NC}"
fi

# Redis status
if check_service "Redis" "$REDIS_PORT" "$REDIS_HOST"; then
    if command_exists redis-cli && redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping | grep -q PONG 2>/dev/null; then
        REDIS_INFO=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" info server 2>/dev/null | grep redis_version | cut -d: -f2 | tr -d '\r')
        echo -e "Redis:      ${GREEN}✓ Ready${NC} (v$REDIS_INFO)"
    else
        echo -e "Redis:      ${GREEN}✓ Running${NC} (connection test failed)"
    fi
else
    echo -e "Redis:      ${RED}✗ Not running${NC}"
fi

echo -e "\n${GREEN}🎉 Database setup completed!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo "1. Start the Tyra MCP Memory Server"
echo "2. Test the memory operations"
echo "3. Check the health endpoints"

# Create environment file template
ENV_FILE="$SCRIPT_DIR/../../.env.example"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "\n${YELLOW}📝 Creating .env.example file...${NC}"
    cat > "$ENV_FILE" <<EOF
# Database Configuration
POSTGRES_HOST=$POSTGRES_HOST
POSTGRES_PORT=$POSTGRES_PORT
POSTGRES_DB=$POSTGRES_DB
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD

# Memgraph Configuration
MEMGRAPH_HOST=$MEMGRAPH_HOST
MEMGRAPH_PORT=$MEMGRAPH_PORT

# Redis Configuration
REDIS_HOST=$REDIS_HOST
REDIS_PORT=$REDIS_PORT

# Application Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development

# Optional: OpenTelemetry Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=tyra-mcp-memory-server
EOF
    echo -e "${GREEN}✓ Created .env.example file${NC}"
fi

echo -e "\n${BLUE}💡 Copy .env.example to .env and customize as needed${NC}"
