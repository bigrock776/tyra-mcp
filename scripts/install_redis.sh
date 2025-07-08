#!/bin/bash
# =============================================================================
# Install and configure Redis
# =============================================================================

set -e

echo "🔧 Installing Redis..."

# Update package lists
echo "📦 Updating package lists..."
sudo apt update

# Install Redis
echo "📦 Installing Redis..."
sudo apt install -y redis-server

# Configure Redis
echo "⚙️ Configuring Redis..."

REDIS_CONFIG="/etc/redis/redis.conf"

# Backup original config
sudo cp "$REDIS_CONFIG" "$REDIS_CONFIG.backup"

# Update Redis configuration
sudo sed -i 's/# requirepass foobared/requirepass tyra_redis_password/' "$REDIS_CONFIG"
sudo sed -i 's/# maxmemory <bytes>/maxmemory 256mb/' "$REDIS_CONFIG"
sudo sed -i 's/# maxmemory-policy noeviction/maxmemory-policy allkeys-lru/' "$REDIS_CONFIG"

# Enable persistence
sudo sed -i 's/save 900 1/save 900 1/' "$REDIS_CONFIG"
sudo sed -i 's/save 300 10/save 300 10/' "$REDIS_CONFIG"
sudo sed -i 's/save 60 10000/save 60 10000/' "$REDIS_CONFIG"

# Performance optimizations
sudo tee -a "$REDIS_CONFIG" <<EOF

# Tyra Memory Server Optimizations
# =================================

# Network optimization
tcp-keepalive 300
timeout 0

# Memory optimization
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

EOF

# Start and enable Redis service
echo "🚀 Starting Redis service..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis installation
echo "🧪 Testing Redis installation..."
redis-cli -a tyra_redis_password ping

# Create Redis CLI alias with password
echo "📝 Creating Redis CLI configuration..."
mkdir -p ~/.redis
echo "tyra_redis_password" > ~/.redis/redis-cli-password
chmod 600 ~/.redis/redis-cli-password

echo ""
echo "🎉 Redis installation complete!"
echo ""
echo "📋 Connection details:"
echo "   Host: localhost"
echo "   Port: 6379"
echo "   Password: tyra_redis_password"
echo ""
echo "🔧 To connect manually:"
echo "   redis-cli -a tyra_redis_password"
echo ""
echo "📝 Configuration file backed up to:"
echo "   $REDIS_CONFIG.backup"
echo ""
