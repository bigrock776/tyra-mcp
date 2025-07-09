#!/bin/bash

# =============================================================================
# Tyra MCP Memory Server - Zero-Downtime Provider Swapping Script
# =============================================================================
# Safely swap providers (embedding models, rerankers, etc.) without downtime

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
CONFIG_DIR="$PROJECT_ROOT/config"

# Provider swap configuration
PROVIDER_TYPE="${PROVIDER_TYPE:-}"
PROVIDER_NAME="${PROVIDER_NAME:-}"
TARGET_PROVIDER="${TARGET_PROVIDER:-}"
SWAP_STRATEGY="${SWAP_STRATEGY:-gradual}"  # gradual, immediate, canary
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"
VALIDATION_REQUESTS="${VALIDATION_REQUESTS:-10}"
CANARY_PERCENTAGE="${CANARY_PERCENTAGE:-10}"

# API configuration
API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-30}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Logging Functions
# =============================================================================
info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_DIR/provider-swap.log"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_DIR/provider-swap.log"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_DIR/provider-swap.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_DIR/provider-swap.log"
}

# =============================================================================
# Utility Functions
# =============================================================================
check_prerequisites() {
    info "Checking prerequisites for provider swapping..."
    
    # Check if API is running
    if ! curl -s -f "$API_URL/health" > /dev/null 2>&1; then
        error "API is not accessible at $API_URL"
        exit 1
    fi
    
    # Check if config directory exists
    if [ ! -d "$CONFIG_DIR" ]; then
        error "Config directory not found: $CONFIG_DIR"
        exit 1
    fi
    
    # Check if jq is available for JSON processing
    if ! command -v jq &> /dev/null; then
        error "jq is required but not installed"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

get_current_providers() {
    info "Getting current provider configuration..."
    
    curl -s -H "Authorization: Bearer $API_KEY" "$API_URL/v1/admin/config/current" | jq '.providers'
}

get_available_providers() {
    local provider_type="$1"
    
    info "Getting available providers for type: $provider_type"
    
    case "$provider_type" in
        embedding)
            curl -s -H "Authorization: Bearer $API_KEY" "$API_URL/v1/admin/providers/embedding" | jq '.providers[]'
            ;;
        reranker)
            curl -s -H "Authorization: Bearer $API_KEY" "$API_URL/v1/admin/providers/reranker" | jq '.providers[]'
            ;;
        vector_store)
            curl -s -H "Authorization: Bearer $API_KEY" "$API_URL/v1/admin/providers/vector_store" | jq '.providers[]'
            ;;
        graph_engine)
            curl -s -H "Authorization: Bearer $API_KEY" "$API_URL/v1/admin/providers/graph_engine" | jq '.providers[]'
            ;;
        *)
            error "Unknown provider type: $provider_type"
            exit 1
            ;;
    esac
}

validate_provider() {
    local provider_type="$1"
    local provider_name="$2"
    
    info "Validating provider: $provider_type/$provider_name"
    
    # Check if provider exists
    if ! get_available_providers "$provider_type" | grep -q "\"$provider_name\""; then
        error "Provider not found: $provider_type/$provider_name"
        return 1
    fi
    
    # Test provider functionality
    local test_endpoint=""
    case "$provider_type" in
        embedding)
            test_endpoint="/v1/admin/providers/embedding/$provider_name/test"
            ;;
        reranker)
            test_endpoint="/v1/admin/providers/reranker/$provider_name/test"
            ;;
        vector_store)
            test_endpoint="/v1/admin/providers/vector_store/$provider_name/test"
            ;;
        graph_engine)
            test_endpoint="/v1/admin/providers/graph_engine/$provider_name/test"
            ;;
    esac
    
    if curl -s -f -H "Authorization: Bearer $API_KEY" "$API_URL$test_endpoint" > /dev/null 2>&1; then
        success "Provider validation passed: $provider_type/$provider_name"
        return 0
    else
        error "Provider validation failed: $provider_type/$provider_name"
        return 1
    fi
}

backup_current_config() {
    info "Backing up current configuration..."
    
    local backup_file="$CONFIG_DIR/providers_backup_$(date +%Y%m%d_%H%M%S).yaml"
    
    if [ -f "$CONFIG_DIR/providers.yaml" ]; then
        cp "$CONFIG_DIR/providers.yaml" "$backup_file"
        success "Configuration backed up to: $backup_file"
        echo "$backup_file"
    else
        error "Current configuration not found"
        exit 1
    fi
}

update_provider_config() {
    local provider_type="$1"
    local provider_name="$2"
    local new_provider="$3"
    
    info "Updating provider configuration: $provider_type/$provider_name -> $new_provider"
    
    # Create temporary config file
    local temp_config="/tmp/providers_temp_$$.yaml"
    
    # Update the configuration using yq or sed
    if command -v yq &> /dev/null; then
        yq eval ".$provider_type.default = \"$new_provider\"" "$CONFIG_DIR/providers.yaml" > "$temp_config"
    else
        # Fallback to sed for simple replacements
        sed "s/default: $provider_name/default: $new_provider/g" "$CONFIG_DIR/providers.yaml" > "$temp_config"
    fi
    
    # Validate the new configuration
    if python3 -c "import yaml; yaml.safe_load(open('$temp_config'))" 2>/dev/null; then
        mv "$temp_config" "$CONFIG_DIR/providers.yaml"
        success "Provider configuration updated"
    else
        error "Generated configuration is invalid"
        rm -f "$temp_config"
        return 1
    fi
}

reload_configuration() {
    info "Reloading system configuration..."
    
    if curl -s -f -X POST -H "Authorization: Bearer $API_KEY" "$API_URL/v1/admin/config/reload" > /dev/null 2>&1; then
        success "Configuration reloaded successfully"
        return 0
    else
        error "Failed to reload configuration"
        return 1
    fi
}

wait_for_provider_ready() {
    local provider_type="$1"
    local provider_name="$2"
    local timeout="${3:-60}"
    
    info "Waiting for provider to be ready: $provider_type/$provider_name"
    
    local attempt=1
    local max_attempts=$((timeout / 5))
    
    while [ $attempt -le $max_attempts ]; do
        if validate_provider "$provider_type" "$provider_name"; then
            success "Provider is ready: $provider_type/$provider_name"
            return 0
        fi
        
        info "Provider not ready, attempt $attempt/$max_attempts, retrying..."
        sleep 5
        ((attempt++))
    done
    
    error "Provider did not become ready within $timeout seconds"
    return 1
}

test_provider_functionality() {
    local provider_type="$1"
    local provider_name="$2"
    local test_count="${3:-10}"
    
    info "Testing provider functionality: $provider_type/$provider_name ($test_count tests)"
    
    local success_count=0
    local failure_count=0
    
    for i in $(seq 1 $test_count); do
        local test_data=""
        local test_endpoint=""
        
        case "$provider_type" in
            embedding)
                test_data='{"text": "This is a test embedding request"}'
                test_endpoint="/v1/embeddings/generate"
                ;;
            reranker)
                test_data='{"query": "test query", "documents": ["doc1", "doc2"]}'
                test_endpoint="/v1/rerank"
                ;;
            vector_store)
                test_data='{"query": "test search", "top_k": 5}'
                test_endpoint="/v1/memory/search"
                ;;
            graph_engine)
                test_data='{"query": "MATCH (n) RETURN count(n) LIMIT 1"}'
                test_endpoint="/v1/graph/query"
                ;;
        esac
        
        if curl -s -f -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" -d "$test_data" "$API_URL$test_endpoint" > /dev/null 2>&1; then
            ((success_count++))
        else
            ((failure_count++))
        fi
        
        sleep 1
    done
    
    local success_rate=$((success_count * 100 / test_count))
    info "Provider test results: $success_count/$test_count successful ($success_rate%)"
    
    if [ $success_rate -ge 80 ]; then
        success "Provider functionality test passed"
        return 0
    else
        error "Provider functionality test failed"
        return 1
    fi
}

# =============================================================================
# Swap Strategy Functions
# =============================================================================
immediate_swap() {
    local provider_type="$1"
    local current_provider="$2"
    local target_provider="$3"
    
    info "Performing immediate swap: $current_provider -> $target_provider"
    
    # Backup current configuration
    local backup_file=$(backup_current_config)
    
    # Update configuration
    if ! update_provider_config "$provider_type" "$current_provider" "$target_provider"; then
        error "Failed to update configuration"
        return 1
    fi
    
    # Reload configuration
    if ! reload_configuration; then
        error "Failed to reload configuration"
        # Restore backup
        cp "$backup_file" "$CONFIG_DIR/providers.yaml"
        reload_configuration
        return 1
    fi
    
    # Wait for provider to be ready
    if ! wait_for_provider_ready "$provider_type" "$target_provider"; then
        error "Provider not ready after swap"
        # Restore backup
        cp "$backup_file" "$CONFIG_DIR/providers.yaml"
        reload_configuration
        return 1
    fi
    
    # Test functionality
    if ! test_provider_functionality "$provider_type" "$target_provider" "$VALIDATION_REQUESTS"; then
        error "Provider functionality test failed"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            warning "Rolling back to previous provider"
            cp "$backup_file" "$CONFIG_DIR/providers.yaml"
            reload_configuration
        fi
        return 1
    fi
    
    success "Immediate swap completed successfully"
    return 0
}

gradual_swap() {
    local provider_type="$1"
    local current_provider="$2"
    local target_provider="$3"
    
    info "Performing gradual swap: $current_provider -> $target_provider"
    
    # Backup current configuration
    local backup_file=$(backup_current_config)
    
    # Step 1: Pre-load target provider
    info "Step 1: Pre-loading target provider"
    if ! curl -s -f -X POST -H "Authorization: Bearer $API_KEY" "$API_URL/v1/admin/providers/$provider_type/$target_provider/preload" > /dev/null 2>&1; then
        warning "Failed to pre-load target provider"
    fi
    
    # Step 2: Validate target provider
    info "Step 2: Validating target provider"
    if ! validate_provider "$provider_type" "$target_provider"; then
        error "Target provider validation failed"
        return 1
    fi
    
    # Step 3: Gradual traffic shift (if supported)
    info "Step 3: Gradually shifting traffic"
    for percentage in 10 25 50 75 90 100; do
        info "Shifting $percentage% of traffic to $target_provider"
        
        # Update traffic routing (this would require API support)
        if curl -s -f -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
           -d "{\"provider\": \"$target_provider\", \"percentage\": $percentage}" \
           "$API_URL/v1/admin/providers/$provider_type/route" > /dev/null 2>&1; then
            
            # Test at this percentage
            if test_provider_functionality "$provider_type" "$target_provider" 5; then
                info "Traffic shift to $percentage% successful"
                sleep 10
            else
                error "Provider failed at $percentage% traffic"
                if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
                    warning "Rolling back traffic routing"
                    curl -s -f -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
                         -d "{\"provider\": \"$current_provider\", \"percentage\": 100}" \
                         "$API_URL/v1/admin/providers/$provider_type/route" > /dev/null 2>&1
                fi
                return 1
            fi
        else
            warning "Traffic routing not supported, falling back to immediate swap"
            return immediate_swap "$provider_type" "$current_provider" "$target_provider"
        fi
    done
    
    # Step 4: Final configuration update
    info "Step 4: Updating final configuration"
    if ! update_provider_config "$provider_type" "$current_provider" "$target_provider"; then
        error "Failed to update final configuration"
        return 1
    fi
    
    if ! reload_configuration; then
        error "Failed to reload final configuration"
        return 1
    fi
    
    success "Gradual swap completed successfully"
    return 0
}

canary_swap() {
    local provider_type="$1"
    local current_provider="$2"
    local target_provider="$3"
    
    info "Performing canary swap: $current_provider -> $target_provider"
    
    # Backup current configuration
    local backup_file=$(backup_current_config)
    
    # Step 1: Deploy canary with limited traffic
    info "Step 1: Deploying canary with $CANARY_PERCENTAGE% traffic"
    if curl -s -f -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
       -d "{\"provider\": \"$target_provider\", \"percentage\": $CANARY_PERCENTAGE}" \
       "$API_URL/v1/admin/providers/$provider_type/canary" > /dev/null 2>&1; then
        
        # Monitor canary for a period
        info "Monitoring canary deployment..."
        sleep 30
        
        # Test canary functionality
        if test_provider_functionality "$provider_type" "$target_provider" "$VALIDATION_REQUESTS"; then
            info "Canary deployment successful"
            
            # Promote canary to full deployment
            info "Step 2: Promoting canary to full deployment"
            if curl -s -f -X POST -H "Authorization: Bearer $API_KEY" \
               "$API_URL/v1/admin/providers/$provider_type/canary/promote" > /dev/null 2>&1; then
                
                # Update configuration
                update_provider_config "$provider_type" "$current_provider" "$target_provider"
                reload_configuration
                
                success "Canary swap completed successfully"
                return 0
            else
                error "Failed to promote canary"
                return 1
            fi
        else
            error "Canary deployment failed"
            # Rollback canary
            curl -s -f -X DELETE -H "Authorization: Bearer $API_KEY" \
                 "$API_URL/v1/admin/providers/$provider_type/canary" > /dev/null 2>&1
            return 1
        fi
    else
        warning "Canary deployment not supported, falling back to gradual swap"
        return gradual_swap "$provider_type" "$current_provider" "$target_provider"
    fi
}

# =============================================================================
# Main Functions
# =============================================================================
perform_provider_swap() {
    local provider_type="$1"
    local current_provider="$2"
    local target_provider="$3"
    local strategy="$4"
    
    info "Starting provider swap: $provider_type/$current_provider -> $target_provider ($strategy)"
    
    # Validate target provider
    if ! validate_provider "$provider_type" "$target_provider"; then
        error "Target provider validation failed"
        exit 1
    fi
    
    # Perform swap based on strategy
    case "$strategy" in
        immediate)
            immediate_swap "$provider_type" "$current_provider" "$target_provider"
            ;;
        gradual)
            gradual_swap "$provider_type" "$current_provider" "$target_provider"
            ;;
        canary)
            canary_swap "$provider_type" "$current_provider" "$target_provider"
            ;;
        *)
            error "Unknown swap strategy: $strategy"
            exit 1
            ;;
    esac
}

show_provider_status() {
    echo "============================================================================="
    info "Provider Status"
    echo "============================================================================="
    
    # Get current provider configuration
    local current_config=$(get_current_providers)
    
    echo "Current providers:"
    echo "$current_config" | jq '.'
    
    echo
    echo "Available providers:"
    for provider_type in embedding reranker vector_store graph_engine; do
        echo "  $provider_type:"
        get_available_providers "$provider_type" | while read -r provider; do
            echo "    - $provider"
        done
    done
}

list_available_providers() {
    local provider_type="$1"
    
    echo "============================================================================="
    info "Available $provider_type Providers"
    echo "============================================================================="
    
    get_available_providers "$provider_type"
}

show_help() {
    cat << EOF
Tyra MCP Memory Server - Zero-Downtime Provider Swapping

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    swap            Swap provider (default)
    status          Show current provider status
    list TYPE       List available providers for type
    help            Show this help message

Options:
    --type TYPE             Provider type (embedding, reranker, vector_store, graph_engine)
    --current NAME          Current provider name
    --target NAME           Target provider name
    --strategy STRATEGY     Swap strategy (immediate, gradual, canary)
    --canary-percentage N   Canary traffic percentage (default: 10)
    --validation-requests N Number of validation requests (default: 10)
    --no-rollback          Don't rollback on failure
    --api-url URL          API URL (default: http://localhost:8000)
    --api-key KEY          API key for authentication

Provider Types:
    embedding       Embedding model providers
    reranker        Reranking model providers
    vector_store    Vector database providers
    graph_engine    Graph database providers

Swap Strategies:
    immediate       Immediate swap (fastest, higher risk)
    gradual         Gradual traffic shift (safer, slower)
    canary          Canary deployment (safest, requires canary support)

Examples:
    $0 swap --type embedding --current huggingface --target openai --strategy gradual
    $0 swap --type reranker --current cross-encoder --target vllm --strategy canary
    $0 status
    $0 list embedding

EOF
}

# =============================================================================
# Command Line Interface
# =============================================================================
COMMAND="${1:-swap}"

while [[ $# -gt 0 ]]; do
    case $1 in
        swap|status|list)
            COMMAND="$1"
            shift
            ;;
        --type)
            PROVIDER_TYPE="$2"
            shift 2
            ;;
        --current)
            PROVIDER_NAME="$2"
            shift 2
            ;;
        --target)
            TARGET_PROVIDER="$2"
            shift 2
            ;;
        --strategy)
            SWAP_STRATEGY="$2"
            shift 2
            ;;
        --canary-percentage)
            CANARY_PERCENTAGE="$2"
            shift 2
            ;;
        --validation-requests)
            VALIDATION_REQUESTS="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE="false"
            shift
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            if [ "$COMMAND" = "list" ] && [ -z "$PROVIDER_TYPE" ]; then
                PROVIDER_TYPE="$1"
                shift
            else
                error "Unknown argument: $1"
                show_help
                exit 1
            fi
            ;;
    esac
done

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo "============================================================================="
    info "Tyra MCP Memory Server - Provider Swapping"
    echo "============================================================================="
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    case "$COMMAND" in
        swap)
            if [ -z "$PROVIDER_TYPE" ] || [ -z "$PROVIDER_NAME" ] || [ -z "$TARGET_PROVIDER" ]; then
                error "Missing required parameters for swap command"
                show_help
                exit 1
            fi
            perform_provider_swap "$PROVIDER_TYPE" "$PROVIDER_NAME" "$TARGET_PROVIDER" "$SWAP_STRATEGY"
            ;;
        status)
            show_provider_status
            ;;
        list)
            if [ -z "$PROVIDER_TYPE" ]; then
                error "Provider type required for list command"
                show_help
                exit 1
            fi
            list_available_providers "$PROVIDER_TYPE"
            ;;
        *)
            error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"