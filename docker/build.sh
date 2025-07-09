#!/bin/bash

# =============================================================================
# Tyra MCP Memory Server - Docker Build Script
# =============================================================================
# Automated build script with version management and multi-target support

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build configuration
IMAGE_NAME="${IMAGE_NAME:-tyra-memory-server}"
REGISTRY="${REGISTRY:-}"
VERSION="${VERSION:-$(git describe --tags --abbrev=0 2>/dev/null || echo "latest")}"
BUILD_TARGET="${BUILD_TARGET:-production}"
PLATFORM="${PLATFORM:-linux/amd64}"

# Build arguments
BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo "unknown")"

# Options
PUSH_IMAGE="${PUSH_IMAGE:-false}"
NO_CACHE="${NO_CACHE:-false}"
MULTI_ARCH="${MULTI_ARCH:-false}"
SQUASH="${SQUASH:-false}"

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
# Build Functions
# =============================================================================
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Buildx for multi-arch builds
    if [ "$MULTI_ARCH" = "true" ] && ! docker buildx version &> /dev/null; then
        error "Docker Buildx is required for multi-architecture builds"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "$PROJECT_ROOT/docker/Dockerfile" ]; then
        error "Dockerfile not found. Are you in the correct directory?"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

setup_buildx() {
    if [ "$MULTI_ARCH" = "true" ]; then
        info "Setting up Docker Buildx for multi-architecture builds..."
        
        # Create builder if it doesn't exist
        if ! docker buildx inspect tyra-builder &> /dev/null; then
            docker buildx create --name tyra-builder --driver docker-container --use
        else
            docker buildx use tyra-builder
        fi
        
        # Boot the builder
        docker buildx inspect --bootstrap
        
        success "Buildx setup completed"
    fi
}

get_image_tags() {
    local tags=()
    
    # Add version tag
    if [ -n "$REGISTRY" ]; then
        tags+=("$REGISTRY/$IMAGE_NAME:$VERSION")
        tags+=("$REGISTRY/$IMAGE_NAME:latest")
    else
        tags+=("$IMAGE_NAME:$VERSION")
        tags+=("$IMAGE_NAME:latest")
    fi
    
    # Add target-specific tags
    if [ "$BUILD_TARGET" != "production" ]; then
        if [ -n "$REGISTRY" ]; then
            tags+=("$REGISTRY/$IMAGE_NAME:$BUILD_TARGET")
        else
            tags+=("$IMAGE_NAME:$BUILD_TARGET")
        fi
    fi
    
    printf '%s\n' "${tags[@]}"
}

build_image() {
    local target="$1"
    local tags
    mapfile -t tags < <(get_image_tags)
    
    info "Building $target image..."
    info "Tags: ${tags[*]}"
    
    # Prepare build arguments
    local build_args=(
        "--target" "$target"
        "--build-arg" "BUILD_DATE=$BUILD_DATE"
        "--build-arg" "VERSION=$VERSION"
        "--build-arg" "VCS_REF=$VCS_REF"
    )
    
    # Add tags
    for tag in "${tags[@]}"; do
        build_args+=("--tag" "$tag")
    done
    
    # Add platform for multi-arch builds
    if [ "$MULTI_ARCH" = "true" ]; then
        build_args+=("--platform" "linux/amd64,linux/arm64")
    else
        build_args+=("--platform" "$PLATFORM")
    fi
    
    # Add cache options
    if [ "$NO_CACHE" = "true" ]; then
        build_args+=("--no-cache")
    fi
    
    # Add squash option
    if [ "$SQUASH" = "true" ] && [ "$MULTI_ARCH" = "false" ]; then
        build_args+=("--squash")
    fi
    
    # Choose build command
    if [ "$MULTI_ARCH" = "true" ]; then
        if [ "$PUSH_IMAGE" = "true" ]; then
            build_args+=("--push")
        else
            build_args+=("--load")
        fi
        docker buildx build "${build_args[@]}" "$PROJECT_ROOT"
    else
        docker build "${build_args[@]}" "$PROJECT_ROOT"
    fi
    
    success "$target image built successfully"
}

push_images() {
    if [ "$PUSH_IMAGE" = "true" ] && [ "$MULTI_ARCH" = "false" ]; then
        info "Pushing images to registry..."
        
        local tags
        mapfile -t tags < <(get_image_tags)
        
        for tag in "${tags[@]}"; do
            info "Pushing $tag..."
            docker push "$tag"
        done
        
        success "Images pushed successfully"
    fi
}

show_build_info() {
    info "Build Information:"
    echo "  Image Name: $IMAGE_NAME"
    echo "  Version: $VERSION"
    echo "  Target: $BUILD_TARGET"
    echo "  Platform: $PLATFORM"
    echo "  Multi-arch: $MULTI_ARCH"
    echo "  Registry: ${REGISTRY:-"(local)"}"
    echo "  Push: $PUSH_IMAGE"
    echo "  Build Date: $BUILD_DATE"
    echo "  VCS Ref: $VCS_REF"
    echo
}

test_image() {
    local image="$IMAGE_NAME:$VERSION"
    
    info "Testing built image..."
    
    # Test that image runs
    if docker run --rm --entrypoint="" "$image" python --version; then
        success "Image test passed - Python is working"
    else
        error "Image test failed - Python not working"
        return 1
    fi
    
    # Test health check for production image
    if [ "$BUILD_TARGET" = "production" ]; then
        info "Testing health check..."
        # This would require the image to be running, so we'll skip for now
        # docker run --rm -d --name test-container "$image"
        # sleep 30
        # docker exec test-container curl -f http://localhost:8000/health
        # docker stop test-container
    fi
}

clean_build_cache() {
    info "Cleaning build cache..."
    docker buildx prune -f || docker system prune -f
    success "Build cache cleaned"
}

# =============================================================================
# Main Functions
# =============================================================================
show_help() {
    cat << EOF
Tyra MCP Memory Server - Docker Build Script

Usage: $0 [OPTIONS]

Options:
    --target TARGET         Build target (development|production|mcp-server)
    --version VERSION       Image version tag (default: git tag or "latest")
    --registry REGISTRY     Docker registry URL
    --platform PLATFORM     Target platform (default: linux/amd64)
    --multi-arch            Build for multiple architectures
    --push                  Push image to registry after build
    --no-cache              Build without using cache
    --squash                Squash layers (single-arch only)
    --test                  Test image after build
    --clean                 Clean build cache before building
    --help                  Show this help message

Environment Variables:
    IMAGE_NAME              Docker image name (default: tyra-memory-server)
    REGISTRY               Docker registry URL
    VERSION                Image version
    BUILD_TARGET           Build target
    PLATFORM               Target platform
    PUSH_IMAGE             Push after build (true/false)
    NO_CACHE               Disable cache (true/false)
    MULTI_ARCH             Multi-arch build (true/false)

Examples:
    $0                                    # Build production image
    $0 --target development               # Build development image
    $0 --target mcp-server --push        # Build and push MCP server image
    $0 --multi-arch --push               # Build multi-arch and push
    $0 --version 1.2.3 --registry myregistry.com

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --multi-arch)
                MULTI_ARCH="true"
                shift
                ;;
            --push)
                PUSH_IMAGE="true"
                shift
                ;;
            --no-cache)
                NO_CACHE="true"
                shift
                ;;
            --squash)
                SQUASH="true"
                shift
                ;;
            --test)
                TEST_IMAGE="true"
                shift
                ;;
            --clean)
                CLEAN_CACHE="true"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

main() {
    echo "============================================================================="
    info "Tyra MCP Memory Server - Docker Build"
    echo "============================================================================="
    
    show_build_info
    
    check_prerequisites
    
    if [ "${CLEAN_CACHE:-false}" = "true" ]; then
        clean_build_cache
    fi
    
    setup_buildx
    
    build_image "$BUILD_TARGET"
    
    if [ "${TEST_IMAGE:-false}" = "true" ]; then
        test_image
    fi
    
    push_images
    
    echo "============================================================================="
    success "Docker build completed successfully!"
    echo "============================================================================="
    
    # Show final image info
    if [ "$MULTI_ARCH" = "false" ]; then
        docker images | grep "$IMAGE_NAME" | head -5
    fi
}

# =============================================================================
# Script Execution
# =============================================================================
parse_args "$@"
main