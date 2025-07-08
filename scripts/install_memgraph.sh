#!/bin/bash
# =============================================================================
# Install and configure Memgraph
# =============================================================================

set -e

echo "🔧 Installing Memgraph..."

# Update package lists
echo "📦 Updating package lists..."
sudo apt update

# Install dependencies
echo "📦 Installing dependencies..."
sudo apt install -y curl gnupg lsb-release

# Add Memgraph signing key
echo "🔑 Adding Memgraph signing key..."
curl -fsSL https://download.memgraph.com/memgraph/signing-key.asc | sudo gpg --dearmor -o /usr/share/keyrings/memgraph-archive-keyring.gpg

# Add Memgraph repository
echo "📦 Adding Memgraph repository..."
echo "deb [signed-by=/usr/share/keyrings/memgraph-archive-keyring.gpg] https://download.memgraph.com/memgraph/deb stable main" | sudo tee /etc/apt/sources.list.d/memgraph.list

# Update package lists with new repository
sudo apt update

# Install Memgraph
echo "📦 Installing Memgraph..."
sudo apt install -y memgraph

# Configure Memgraph
echo "⚙️ Configuring Memgraph..."

MEMGRAPH_CONFIG="/etc/memgraph/memgraph.conf"

# Create configuration if it doesn't exist
sudo mkdir -p /etc/memgraph
sudo tee "$MEMGRAPH_CONFIG" <<EOF
# Memgraph Configuration for Tyra Memory Server
# ==============================================

# Network settings
--bolt-port=7687
--bolt-address=0.0.0.0

# Storage settings
--data-directory=/var/lib/memgraph
--log-file=/var/log/memgraph/memgraph.log

# Performance settings
--memory-limit=1024

# Query settings
--query-execution-timeout-sec=600

# Logging
--log-level=INFO
--also-log-to-stderr=false

# Authentication (disabled for development)
--auth-module-executable=
--auth-user-or-role-name-regex=.*

EOF

# Create necessary directories
sudo mkdir -p /var/lib/memgraph
sudo mkdir -p /var/log/memgraph
sudo chown memgraph:memgraph /var/lib/memgraph
sudo chown memgraph:memgraph /var/log/memgraph

# Create systemd service
echo "🚀 Creating systemd service..."
sudo tee /etc/systemd/system/memgraph.service <<EOF
[Unit]
Description=Memgraph database server
Documentation=https://memgraph.com/docs/
After=network.target

[Service]
Type=simple
User=memgraph
Group=memgraph
ExecStart=/usr/lib/memgraph/memgraph --config-file=/etc/memgraph/memgraph.conf
Restart=on-failure
RestartSec=5
TimeoutStopSec=600
KillMode=mixed
KillSignal=SIGTERM

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/memgraph /var/log/memgraph

[Install]
WantedBy=multi-user.target

EOF

# Reload systemd and start Memgraph
echo "🚀 Starting Memgraph service..."
sudo systemctl daemon-reload
sudo systemctl start memgraph
sudo systemctl enable memgraph

# Wait for Memgraph to start
echo "⏳ Waiting for Memgraph to start..."
sleep 5

# Test Memgraph installation
echo "🧪 Testing Memgraph installation..."

# Check if Memgraph is running
if sudo systemctl is-active --quiet memgraph; then
    echo "✅ Memgraph service is running"
else
    echo "❌ Memgraph service failed to start"
    sudo systemctl status memgraph
    exit 1
fi

# Install mgclient for testing (optional)
echo "📦 Installing mgclient for testing..."
sudo apt install -y libssl-dev libcurl4-openssl-dev
pip3 install --user mgclient

# Test connection
echo "🧪 Testing Memgraph connection..."
python3 -c "
try:
    import mgclient
    conn = mgclient.connect(host='localhost', port=7687)
    cursor = conn.cursor()
    cursor.execute('RETURN \"Hello from Memgraph!\" AS message;')
    result = cursor.fetchone()
    print(f'✅ Connection successful: {result[0]}')
    conn.close()
except Exception as e:
    print(f'❌ Connection failed: {e}')
"

echo ""
echo "🎉 Memgraph installation complete!"
echo ""
echo "📋 Connection details:"
echo "   Host: localhost"
echo "   Port: 7687"
echo "   Protocol: Bolt"
echo ""
echo "🔧 To connect manually:"
echo "   You can use mgclient or any Bolt-compatible client"
echo ""
echo "📝 Configuration file:"
echo "   $MEMGRAPH_CONFIG"
echo ""
echo "📊 Service management:"
echo "   sudo systemctl start memgraph"
echo "   sudo systemctl stop memgraph"
echo "   sudo systemctl status memgraph"
echo ""
