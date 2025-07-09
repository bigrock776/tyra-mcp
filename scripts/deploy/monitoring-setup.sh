#!/bin/bash

# =============================================================================
# Tyra MCP Memory Server - Monitoring Setup Script
# =============================================================================
# Sets up comprehensive monitoring stack with Prometheus, Grafana, and alerting

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MONITORING_DIR="$PROJECT_ROOT/monitoring"

# Default configuration
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
ALERTMANAGER_PORT="${ALERTMANAGER_PORT:-9093}"
NODE_EXPORTER_PORT="${NODE_EXPORTER_PORT:-9100}"
POSTGRES_EXPORTER_PORT="${POSTGRES_EXPORTER_PORT:-9187}"
REDIS_EXPORTER_PORT="${REDIS_EXPORTER_PORT:-9121}"

# Service endpoints
TYRA_API_URL="${TYRA_API_URL:-http://localhost:8000}"
POSTGRES_URL="${POSTGRES_URL:-postgresql://tyra:tyra123@localhost:5432/tyra_memory}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379}"

# Alerting configuration
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
ALERT_EMAIL="${ALERT_EMAIL:-admin@example.com}"
SMTP_SERVER="${SMTP_SERVER:-smtp.gmail.com:587}"
SMTP_USER="${SMTP_USER:-}"
SMTP_PASSWORD="${SMTP_PASSWORD:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Utility Functions
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
# Setup Functions
# =============================================================================
create_monitoring_directories() {
    info "Creating monitoring directories..."
    
    mkdir -p "$MONITORING_DIR"/{prometheus,grafana,alertmanager,exporters}
    mkdir -p "$MONITORING_DIR/grafana"/{dashboards,datasources,provisioning}
    mkdir -p "$MONITORING_DIR/prometheus"/{rules,data}
    mkdir -p "$MONITORING_DIR/alertmanager"/{templates,data}
    
    success "Monitoring directories created"
}

setup_prometheus() {
    info "Setting up Prometheus configuration..."
    
    cat > "$MONITORING_DIR/prometheus/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'tyra-memory-server'
    environment: '${ENVIRONMENT:-production}'

rule_files:
  - "rules/*.yml"

scrape_configs:
  # Tyra Memory Server
  - job_name: 'tyra-memory-server'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/v1/telemetry/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s

  # System metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 30s

  # PostgreSQL metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s

  # Redis metrics
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Grafana metrics
  - job_name: 'grafana'
    static_configs:
      - targets: ['grafana:3000']
    metrics_path: '/metrics'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF
    
    success "Prometheus configuration created"
}

setup_alerting_rules() {
    info "Setting up alerting rules..."
    
    cat > "$MONITORING_DIR/prometheus/rules/tyra-alerts.yml" << EOF
groups:
  - name: tyra-memory-server
    rules:
      # High-level service alerts
      - alert: TyraServiceDown
        expr: up{job="tyra-memory-server"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Tyra Memory Server is down"
          description: "Tyra Memory Server has been down for more than 1 minute"

      - alert: TyraHighLatency
        expr: histogram_quantile(0.95, rate(tyra_request_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ \$value }}s"

      - alert: TyraHighErrorRate
        expr: rate(tyra_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ \$value }} errors per second"

      # Memory and embedding alerts
      - alert: TyraEmbeddingCacheLowHitRate
        expr: rate(tyra_embedding_cache_hits_total[5m]) / rate(tyra_embedding_cache_requests_total[5m]) < 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low embedding cache hit rate"
          description: "Embedding cache hit rate is {{ \$value | humanizePercentage }}"

      - alert: TyraMemoryUsageHigh
        expr: tyra_memory_usage_bytes / tyra_memory_limit_bytes > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ \$value | humanizePercentage }} of limit"

      # Database alerts
      - alert: PostgreSQLDown
        expr: up{job="postgres-exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "PostgreSQL database is not responding"

      - alert: PostgreSQLConnectionsHigh
        expr: pg_stat_activity_count / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High PostgreSQL connections"
          description: "PostgreSQL connection usage is {{ \$value | humanizePercentage }}"

      - alert: RedisDown
        expr: up{job="redis-exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis cache is not responding"

      - alert: RedisMemoryUsageHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High Redis memory usage"
          description: "Redis memory usage is {{ \$value | humanizePercentage }}"

      # System resource alerts
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ \$value }}%"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ \$value }}%"

      - alert: LowDiskSpace
        expr: (1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space"
          description: "Disk usage is {{ \$value }}% on {{ \$labels.mountpoint }}"

      # Hallucination detection alerts
      - alert: TyraHighHallucinationRate
        expr: rate(tyra_hallucination_detected_total[5m]) / rate(tyra_responses_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High hallucination detection rate"
          description: "Hallucination rate is {{ \$value | humanizePercentage }}"

      - alert: TyraLowConfidenceResponses
        expr: rate(tyra_low_confidence_responses_total[5m]) / rate(tyra_responses_total[5m]) > 0.3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of low confidence responses"
          description: "Low confidence response rate is {{ \$value | humanizePercentage }}"
EOF
    
    success "Alerting rules created"
}

setup_alertmanager() {
    info "Setting up Alertmanager configuration..."
    
    cat > "$MONITORING_DIR/alertmanager/alertmanager.yml" << EOF
global:
  smtp_smarthost: '$SMTP_SERVER'
  smtp_from: '$SMTP_USER'
  smtp_auth_username: '$SMTP_USER'
  smtp_auth_password: '$SMTP_PASSWORD'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:5001/'

  - name: 'critical-alerts'
    email_configs:
      - to: '$ALERT_EMAIL'
        subject: '[CRITICAL] Tyra Memory Server Alert'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
EOF

    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        cat >> "$MONITORING_DIR/alertmanager/alertmanager.yml" << EOF
    slack_configs:
      - api_url: '$SLACK_WEBHOOK_URL'
        channel: '#alerts'
        title: 'Tyra Memory Server Alert'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          {{ end }}
EOF
    fi

    cat >> "$MONITORING_DIR/alertmanager/alertmanager.yml" << EOF

  - name: 'warning-alerts'
    email_configs:
      - to: '$ALERT_EMAIL'
        subject: '[WARNING] Tyra Memory Server Alert'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
EOF

    success "Alertmanager configuration created"
}

setup_grafana() {
    info "Setting up Grafana configuration..."
    
    # Grafana datasource configuration
    cat > "$MONITORING_DIR/grafana/datasources/datasources.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Grafana dashboard provisioning
    cat > "$MONITORING_DIR/grafana/dashboards/dashboards.yml" << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    # Create Tyra Memory Server dashboard
    cat > "$MONITORING_DIR/grafana/dashboards/tyra-dashboard.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Tyra Memory Server Dashboard",
    "tags": ["tyra", "memory-server", "mcp"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(tyra_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(tyra_errors_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      },
      {
        "id": 3,
        "title": "Response Time",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(tyra_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 4,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "tyra_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "id": 5,
        "title": "Embedding Cache Hit Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(tyra_embedding_cache_hits_total[5m]) / rate(tyra_embedding_cache_requests_total[5m])",
            "legendFormat": "Cache Hit Rate"
          }
        ]
      },
      {
        "id": 6,
        "title": "Database Connection Pool",
        "type": "graph",
        "targets": [
          {
            "expr": "tyra_db_connections_active",
            "legendFormat": "Active Connections"
          },
          {
            "expr": "tyra_db_connections_idle",
            "legendFormat": "Idle Connections"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    success "Grafana configuration created"
}

setup_docker_compose() {
    info "Setting up monitoring Docker Compose..."
    
    cat > "$MONITORING_DIR/docker-compose.monitoring.yml" << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: tyra-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    ports:
      - "${PROMETHEUS_PORT}:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    depends_on:
      - node-exporter
      - postgres-exporter
      - redis-exporter
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: tyra-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "${GRAFANA_PORT}:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    container_name: tyra-alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    ports:
      - "${ALERTMANAGER_PORT}:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: tyra-node-exporter
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "${NODE_EXPORTER_PORT}:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    restart: unless-stopped

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: tyra-postgres-exporter
    environment:
      - DATA_SOURCE_NAME=${POSTGRES_URL}
    ports:
      - "${POSTGRES_EXPORTER_PORT}:9187"
    restart: unless-stopped

  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: tyra-redis-exporter
    environment:
      - REDIS_ADDR=${REDIS_URL}
    ports:
      - "${REDIS_EXPORTER_PORT}:9121"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:

networks:
  default:
    name: tyra-monitoring
    driver: bridge
EOF

    success "Docker Compose configuration created"
}

create_monitoring_scripts() {
    info "Creating monitoring utility scripts..."
    
    # Start monitoring script
    cat > "$MONITORING_DIR/start-monitoring.sh" << 'EOF'
#!/bin/bash
echo "Starting Tyra Memory Server monitoring stack..."
docker-compose -f docker-compose.monitoring.yml up -d
echo "Monitoring stack started!"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo "- Alertmanager: http://localhost:9093"
EOF

    # Stop monitoring script
    cat > "$MONITORING_DIR/stop-monitoring.sh" << 'EOF'
#!/bin/bash
echo "Stopping Tyra Memory Server monitoring stack..."
docker-compose -f docker-compose.monitoring.yml down
echo "Monitoring stack stopped!"
EOF

    # Monitoring health check script
    cat > "$MONITORING_DIR/health-check.sh" << 'EOF'
#!/bin/bash
echo "Checking monitoring stack health..."

# Check if containers are running
if docker ps | grep -q tyra-prometheus; then
    echo "✓ Prometheus is running"
else
    echo "✗ Prometheus is not running"
fi

if docker ps | grep -q tyra-grafana; then
    echo "✓ Grafana is running"
else
    echo "✗ Grafana is not running"
fi

if docker ps | grep -q tyra-alertmanager; then
    echo "✓ Alertmanager is running"
else
    echo "✗ Alertmanager is not running"
fi

# Check if services are responding
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✓ Prometheus is healthy"
else
    echo "✗ Prometheus is not healthy"
fi

if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✓ Grafana is healthy"
else
    echo "✗ Grafana is not healthy"
fi

echo "Health check completed!"
EOF

    chmod +x "$MONITORING_DIR"/*.sh
    
    success "Monitoring utility scripts created"
}

# =============================================================================
# Main Setup Function
# =============================================================================
main() {
    echo "============================================================================="
    info "Setting up Tyra Memory Server Monitoring Stack"
    echo "============================================================================="
    
    create_monitoring_directories
    setup_prometheus
    setup_alerting_rules
    setup_alertmanager
    setup_grafana
    setup_docker_compose
    create_monitoring_scripts
    
    success "Monitoring setup completed successfully!"
    echo
    info "Next steps:"
    echo "1. Review and customize monitoring configuration files"
    echo "2. Update environment variables in docker-compose.monitoring.yml"
    echo "3. Start the monitoring stack:"
    echo "   cd $MONITORING_DIR && ./start-monitoring.sh"
    echo "4. Access dashboards:"
    echo "   - Prometheus: http://localhost:${PROMETHEUS_PORT}"
    echo "   - Grafana: http://localhost:${GRAFANA_PORT} (admin/admin)"
    echo "   - Alertmanager: http://localhost:${ALERTMANAGER_PORT}"
    echo
    info "For troubleshooting, run: cd $MONITORING_DIR && ./health-check.sh"
}

# =============================================================================
# Command Line Interface
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --prometheus-port)
            PROMETHEUS_PORT="$2"
            shift 2
            ;;
        --grafana-port)
            GRAFANA_PORT="$2"
            shift 2
            ;;
        --alert-email)
            ALERT_EMAIL="$2"
            shift 2
            ;;
        --slack-webhook)
            SLACK_WEBHOOK_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Tyra Memory Server Monitoring Setup"
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --prometheus-port PORT    Set Prometheus port (default: 9090)"
            echo "  --grafana-port PORT       Set Grafana port (default: 3000)"
            echo "  --alert-email EMAIL       Set alert email address"
            echo "  --slack-webhook URL       Set Slack webhook URL"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main "$@"