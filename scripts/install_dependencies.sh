#!/bin/bash
# Install system dependencies for Tyra Advanced Memory System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
        elif [ -f /etc/arch-release ]; then
            OS="arch"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        OS="unknown"
    fi

    log_info "Detected OS: $OS"
}

# Install system packages
install_system_packages() {
    log_info "Installing system dependencies..."

    case $OS in
        "debian")
            sudo apt-get update
            sudo apt-get install -y \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                build-essential \
                postgresql-client \
                redis-tools \
                curl \
                wget \
                git \
                netcat-openbsd \
                openssl \
                pkg-config \
                libpq-dev \
                libffi-dev \
                libssl-dev
            ;;
        "redhat")
            sudo yum update -y
            sudo yum install -y \
                python3 \
                python3-pip \
                python3-devel \
                gcc \
                gcc-c++ \
                make \
                postgresql \
                redis \
                curl \
                wget \
                git \
                nc \
                openssl \
                pkgconfig \
                postgresql-devel \
                libffi-devel \
                openssl-devel
            ;;
        "arch")
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                python \
                python-pip \
                base-devel \
                postgresql \
                redis \
                curl \
                wget \
                git \
                netcat \
                openssl \
                pkg-config \
                postgresql-libs \
                libffi
            ;;
        "macos")
            if ! command -v brew &> /dev/null; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi

            brew update
            brew install \
                python@3.11 \
                postgresql \
                redis \
                curl \
                wget \
                git \
                netcat \
                openssl \
                pkg-config \
                libffi
            ;;
        *)
            log_error "Unsupported operating system: $OSTYPE"
            exit 1
            ;;
    esac

    log_success "System packages installed"
}

# Install Docker if requested
install_docker() {
    if [ "$INSTALL_DOCKER" = "true" ]; then
        log_info "Installing Docker..."

        case $OS in
            "debian")
                # Install Docker's official GPG key
                sudo apt-get update
                sudo apt-get install -y ca-certificates curl gnupg
                sudo install -m 0755 -d /etc/apt/keyrings
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
                sudo chmod a+r /etc/apt/keyrings/docker.gpg

                # Add Docker repository
                echo \
                  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
                  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
                  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

                # Install Docker
                sudo apt-get update
                sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

                # Add user to docker group
                sudo usermod -aG docker $USER
                ;;
            "redhat")
                sudo yum install -y yum-utils
                sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
                sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
                sudo systemctl start docker
                sudo systemctl enable docker
                sudo usermod -aG docker $USER
                ;;
            "macos")
                log_info "Please install Docker Desktop for Mac from: https://docs.docker.com/desktop/mac/install/"
                ;;
            *)
                log_warning "Docker installation not supported for this OS"
                ;;
        esac

        log_success "Docker installation complete (may require logout/login to take effect)"
    fi
}

# Install CUDA (optional)
install_cuda() {
    if [ "$INSTALL_CUDA" = "true" ]; then
        log_info "Installing NVIDIA CUDA..."

        # Check if NVIDIA GPU is present
        if ! command -v nvidia-smi &> /dev/null; then
            log_warning "No NVIDIA GPU detected or drivers not installed"
            return
        fi

        case $OS in
            "debian")
                # Install CUDA keyring
                wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
                sudo dpkg -i cuda-keyring_1.0-1_all.deb
                sudo apt-get update
                sudo apt-get -y install cuda-toolkit-12-3
                rm cuda-keyring_1.0-1_all.deb
                ;;
            "redhat")
                sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
                sudo yum clean all
                sudo yum -y install cuda-toolkit-12-3
                ;;
            *)
                log_warning "CUDA installation not supported for this OS via script"
                log_info "Please install CUDA manually from: https://developer.nvidia.com/cuda-downloads"
                ;;
        esac

        # Add CUDA to PATH
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

        log_success "CUDA installation complete (restart required)"
    fi
}

# Verify installations
verify_installations() {
    log_info "Verifying installations..."

    # Check Python
    if python3 --version &> /dev/null; then
        log_success "Python: $(python3 --version)"
    else
        log_error "Python installation failed"
    fi

    # Check pip
    if pip3 --version &> /dev/null; then
        log_success "pip: $(pip3 --version | cut -d' ' -f1-2)"
    else
        log_error "pip installation failed"
    fi

    # Check PostgreSQL client
    if psql --version &> /dev/null; then
        log_success "PostgreSQL client: $(psql --version)"
    else
        log_warning "PostgreSQL client not available"
    fi

    # Check Redis client
    if redis-cli --version &> /dev/null; then
        log_success "Redis client: $(redis-cli --version)"
    else
        log_warning "Redis client not available"
    fi

    # Check Docker
    if command -v docker &> /dev/null; then
        log_success "Docker: $(docker --version)"
    else
        log_warning "Docker not available"
    fi

    # Check NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        log_info "No NVIDIA GPU detected (CPU-only mode)"
    fi
}

# Main function
main() {
    log_info "Installing system dependencies for Tyra Advanced Memory System..."

    detect_os
    install_system_packages
    install_docker
    install_cuda
    verify_installations

    log_success "System dependencies installation complete!"

    if [ "$INSTALL_DOCKER" = "true" ] || [ "$INSTALL_CUDA" = "true" ]; then
        log_warning "Some installations may require a system restart or logout/login to take effect"
    fi

    log_info "Next steps:"
    echo "1. Run the main setup script: ./scripts/setup.sh"
    echo "2. Follow the configuration instructions"
}

# Parse command line arguments
INSTALL_DOCKER=false
INSTALL_CUDA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            INSTALL_DOCKER=true
            shift
            ;;
        --cuda)
            INSTALL_CUDA=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --docker    Install Docker and Docker Compose"
            echo "  --cuda      Install NVIDIA CUDA toolkit"
            echo "  --help      Show this help message"
            echo
            echo "This script installs system dependencies for Tyra Advanced Memory System."
            echo "It automatically detects your operating system and installs appropriate packages."
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main installation
main
