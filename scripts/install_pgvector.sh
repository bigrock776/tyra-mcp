#!/bin/bash
# =============================================================================
# Install PostgreSQL with pgvector extension
# =============================================================================

set -e

echo "🔧 Installing PostgreSQL with pgvector extension..."

# Update package lists
echo "📦 Updating package lists..."
sudo apt update

# Install PostgreSQL and dependencies
echo "📦 Installing PostgreSQL and build dependencies..."
sudo apt install -y \
    postgresql-16 \
    postgresql-server-dev-16 \
    postgresql-contrib-16 \
    build-essential \
    git \
    cmake \
    pkg-config

# Start and enable PostgreSQL service
echo "🚀 Starting PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check if pgvector is already installed
if sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;" postgres 2>/dev/null; then
    echo "✅ pgvector extension is already available!"
else
    echo "📦 Installing pgvector extension..."

    # Clone and build pgvector
    cd /tmp
    git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
    cd pgvector

    # Build and install
    make
    sudo make install

    # Clean up
    cd /
    rm -rf /tmp/pgvector

    echo "✅ pgvector extension installed!"
fi

# Create database user and database
echo "👤 Setting up database user and database..."

# Switch to postgres user and run setup
sudo -u postgres psql <<EOF
-- Create user if not exists
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'tyra') THEN
      CREATE USER tyra WITH PASSWORD 'tyra_secure_password';
   END IF;
END
\$\$;

-- Create database if not exists
SELECT 'CREATE DATABASE tyra_memory OWNER tyra'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'tyra_memory')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE tyra_memory TO tyra;
GRANT CREATE ON SCHEMA public TO tyra;

-- Connect to the database and enable extensions
\c tyra_memory

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;

-- Grant usage on extensions
GRANT USAGE ON SCHEMA public TO tyra;
GRANT ALL ON ALL TABLES IN SCHEMA public TO tyra;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO tyra;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO tyra;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO tyra;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO tyra;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO tyra;

EOF

# Test the installation
echo "🧪 Testing pgvector installation..."
sudo -u postgres psql -d tyra_memory -c "
SELECT vector_in('[1,2,3]'::cstring, 0, 0);
SELECT '[1,2,3]'::vector;
"

# Update PostgreSQL configuration for better performance
echo "⚙️ Optimizing PostgreSQL configuration..."

PG_CONFIG="/etc/postgresql/16/main/postgresql.conf"
PG_HBA="/etc/postgresql/16/main/pg_hba.conf"

# Backup original config
sudo cp "$PG_CONFIG" "$PG_CONFIG.backup"
sudo cp "$PG_HBA" "$PG_HBA.backup"

# Update postgresql.conf
sudo tee -a "$PG_CONFIG" <<EOF

# Tyra Memory Server Optimizations
# =================================

# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
maintenance_work_mem = 64MB

# Connection settings
max_connections = 100
shared_preload_libraries = 'pg_stat_statements'

# Logging
log_statement = 'all'
log_duration = on
log_min_duration_statement = 1000

# Performance monitoring
track_activities = on
track_counts = on
track_io_timing = on
track_functions = pl

EOF

# Update pg_hba.conf for development
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONFIG"

# Add entry for local development
sudo tee -a "$PG_HBA" <<EOF

# Tyra Memory Server Development Access
host    tyra_memory     tyra            127.0.0.1/32           md5
host    tyra_memory     tyra            ::1/128                 md5

EOF

# Restart PostgreSQL to apply changes
echo "🔄 Restarting PostgreSQL..."
sudo systemctl restart postgresql

# Verify installation
echo "✅ Verifying installation..."
sudo -u postgres psql -d tyra_memory -c "
SELECT version();
SELECT * FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'btree_gin', 'uuid-ossp');
"

echo ""
echo "🎉 PostgreSQL with pgvector installation complete!"
echo ""
echo "📋 Connection details:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: tyra_memory"
echo "   User: tyra"
echo "   Password: tyra_secure_password"
echo ""
echo "🔧 To connect manually:"
echo "   psql -h localhost -p 5432 -U tyra -d tyra_memory"
echo ""
echo "📝 Configuration files backed up to:"
echo "   $PG_CONFIG.backup"
echo "   $PG_HBA.backup"
echo ""
