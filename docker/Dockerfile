# =============================================================================
# Tyra MCP Memory Server - Optimized Production Dockerfile
# =============================================================================
# Multi-stage build for optimized image size and security

# =============================================================================
# Stage 1: Base Python Environment
# =============================================================================
FROM python:3.12-slim AS python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/opt/poetry_cache

# =============================================================================
# Stage 2: Dependencies Builder
# =============================================================================
FROM python-base AS builder

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# =============================================================================
# Stage 3: Development Environment (Optional)
# =============================================================================
FROM python-base AS development

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install all dependencies (including dev)
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

# Install the package in development mode
RUN poetry install

# Expose ports
EXPOSE 8000 9090

# Set development environment
ENV ENVIRONMENT=development
ENV DEBUG=true
ENV LOG_LEVEL=DEBUG

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command
CMD ["poetry", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Stage 4: Production Runtime
# =============================================================================
FROM python:3.12-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 tyra \
    && useradd --uid 1000 --gid tyra --shell /bin/bash --create-home tyra

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=tyra:tyra . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache \
    && chown -R tyra:tyra /app

# Switch to non-root user
USER tyra

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =============================================================================
# Stage 5: MCP Server Only (Lightweight)
# =============================================================================
FROM python:3.12-slim AS mcp-server

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 tyra \
    && useradd --uid 1000 --gid tyra --shell /bin/bash --create-home tyra

WORKDIR /app

# Copy virtual environment and essential files only
COPY --from=builder /app/.venv /app/.venv
COPY --chown=tyra:tyra src/ ./src/
COPY --chown=tyra:tyra config/ ./config/
COPY --chown=tyra:tyra pyproject.toml ./

# Create directories
RUN mkdir -p /app/logs && chown -R tyra:tyra /app

USER tyra

# Expose MCP port only
EXPOSE 3000

# Health check for MCP server
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 3000), timeout=5).close()" || exit 1

# MCP server command
CMD ["python", "src/mcp_server/server.py"]

# =============================================================================
# Build Arguments and Labels
# =============================================================================

# Build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Labels for metadata
LABEL org.opencontainers.image.title="Tyra MCP Memory Server" \
      org.opencontainers.image.description="Advanced memory system with RAG capabilities and MCP protocol support" \
      org.opencontainers.image.version=${VERSION} \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.revision=${VCS_REF} \
      org.opencontainers.image.vendor="Tyra AI" \
      org.opencontainers.image.source="https://github.com/tyra-ai/memory-server" \
      org.opencontainers.image.documentation="https://docs.tyra-ai.com/memory-server" \
      org.opencontainers.image.licenses="MIT"

# =============================================================================
# Build Instructions
# =============================================================================
#
# Build development image:
#   docker build --target development -t tyra-memory-server:dev .
#
# Build production image:
#   docker build --target production -t tyra-memory-server:latest .
#
# Build MCP-only image:
#   docker build --target mcp-server -t tyra-memory-server:mcp .
#
# Build with version info:
#   docker build --build-arg VERSION=1.0.0 --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') --target production -t tyra-memory-server:1.0.0 .
#
# =============================================================================
