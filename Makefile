# Tyra MCP Memory Server - Makefile
# =============================================================================
# Common development and deployment tasks
# =============================================================================

.PHONY: help install dev test lint format clean build docs docker

# Default target
help: ## Show this help message
	@echo "Tyra MCP Memory Server - Available Commands:"
	@echo "============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

install: ## Install dependencies
	@echo "Installing dependencies..."
	pip install -e .
	@echo "✅ Dependencies installed"

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	pip install -e ".[dev,test,docs]"
	pre-commit install
	@echo "✅ Development environment ready"

venv: ## Create virtual environment
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "✅ Virtual environment created"
	@echo "Activate with: source venv/bin/activate"

# =============================================================================
# DEVELOPMENT
# =============================================================================

dev: ## Start development server with hot reload
	@echo "Starting development server..."
	python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

mcp-server: ## Start MCP server in development mode
	@echo "Starting MCP server..."
	python src/mcp_server/server.py

run-tests: ## Run all tests
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test: run-tests ## Alias for run-tests

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	@echo "Running end-to-end tests..."
	pytest tests/e2e/ -v

test-watch: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	pytest-watch tests/ -- -v

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run all linting tools
	@echo "Running linting checks..."
	flake8 src/
	mypy src/
	bandit -r src/
	@echo "✅ Linting complete"

format: ## Format code with black and isort
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatted"

check: lint test ## Run both linting and tests

pre-commit: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

security: ## Run security checks
	@echo "Running security checks..."
	bandit -r src/
	safety check

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

db-setup: ## Set up databases (PostgreSQL, Redis, Memgraph)
	@echo "Setting up databases..."
	bash scripts/setup.sh
	@echo "✅ Databases setup complete"

db-init: ## Initialize database schemas
	@echo "Initializing database schemas..."
	python -c "from src.core.utils.database import init_databases; import asyncio; asyncio.run(init_databases())"
	@echo "✅ Database schemas initialized"

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	psql -h localhost -p 5432 -U tyra -d tyra_memory -f scripts/init_postgres.sql
	@echo "✅ Database migrations complete"

db-reset: ## Reset all databases (WARNING: Destroys all data)
	@echo "⚠️  WARNING: This will destroy all data!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	@echo "Resetting databases..."
	bash scripts/reset_databases.sh
	@echo "✅ Databases reset"

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Generate documentation
	@echo "Generating documentation..."
	mkdocs build
	@echo "✅ Documentation generated in site/"

docs-serve: ## Serve documentation locally
	@echo "Starting documentation server..."
	mkdocs serve --dev-addr 0.0.0.0:8001

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "Deploying documentation..."
	mkdocs gh-deploy --force

# =============================================================================
# BUILDING & PACKAGING
# =============================================================================

build: ## Build the package
	@echo "Building package..."
	python -m build
	@echo "✅ Package built"

clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned"

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t tyra-memory-server .
	@echo "✅ Docker image built"

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8000:8000 --env-file .env tyra-memory-server

docker-compose-up: ## Start all services with docker-compose
	@echo "Starting services with docker-compose..."
	docker-compose up -d
	@echo "✅ Services started"

docker-compose-down: ## Stop all services
	@echo "Stopping services..."
	docker-compose down
	@echo "✅ Services stopped"

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f

# =============================================================================
# DEPLOYMENT
# =============================================================================

deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	# Add your staging deployment commands here
	@echo "✅ Deployed to staging"

deploy-production: ## Deploy to production environment
	@echo "Deploying to production..."
	# Add your production deployment commands here
	@echo "✅ Deployed to production"

# =============================================================================
# MONITORING & MAINTENANCE
# =============================================================================

health-check: ## Check system health
	@echo "Checking system health..."
	curl -f http://localhost:8000/health || echo "❌ API not responding"
	python -c "from src.core.utils.database import health_check; import asyncio; asyncio.run(health_check())"

logs: ## View application logs
	@echo "Viewing logs..."
	tail -f logs/tyra-memory.log

metrics: ## View system metrics
	@echo "System metrics:"
	python -c "from src.core.analytics.performance_tracker import get_system_metrics; import asyncio; print(asyncio.run(get_system_metrics()))"

backup: ## Create database backup
	@echo "Creating database backup..."
	bash scripts/backup.sh
	@echo "✅ Backup created"

# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

shell: ## Start interactive Python shell with context
	@echo "Starting Python shell..."
	python -c "from src.core.utils.config import load_config; config = load_config(); print('Config loaded. Available: config'); import IPython; IPython.embed()"

jupyter: ## Start Jupyter notebook server
	@echo "Starting Jupyter notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

profile: ## Profile the application
	@echo "Profiling application..."
	python -m cProfile -o profile.stats src/mcp_server/server.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

benchmark: ## Run performance benchmarks
	@echo "Running benchmarks..."
	python scripts/benchmark.py

# =============================================================================
# CI/CD HELPERS
# =============================================================================

ci-test: ## Run CI test suite
	@echo "Running CI test suite..."
	pytest tests/ --cov=src --cov-report=xml --cov-report=term
	flake8 src/
	mypy src/
	bandit -r src/ -f json -o bandit-report.json

ci-build: ## Build for CI
	@echo "Building for CI..."
	python -m build
	docker build -t tyra-memory-server:ci .

# =============================================================================
# QUICK COMMANDS
# =============================================================================

quick-start: install-dev db-setup ## Quick start for new developers
	@echo "🚀 Quick start complete!"
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and configure"
	@echo "2. Run 'make dev' to start development server"
	@echo "3. Run 'make test' to verify everything works"

reset-env: clean venv install-dev ## Reset development environment
	@echo "🔄 Development environment reset"

status: ## Show project status
	@echo "📊 Project Status:"
	@echo "=================="
	@echo "Python version: $(shell python --version)"
	@echo "Virtual env: $(shell which python)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "Last commit: $(shell git log -1 --oneline 2>/dev/null || echo 'No commits')"
	@echo "Tests status: $(shell pytest tests/ -q --tb=no 2>/dev/null && echo '✅ Passing' || echo '❌ Failing')"

# =============================================================================
# HELP INFORMATION
# =============================================================================

info: ## Show detailed project information
	@echo "🧠 Tyra MCP Memory Server"
	@echo "========================"
	@echo "Advanced memory system with MCP protocol support"
	@echo ""
	@echo "Key Features:"
	@echo "• PostgreSQL + pgvector for vector storage"
	@echo "• Redis for caching"
	@echo "• Memgraph for knowledge graphs"
	@echo "• HuggingFace embeddings (local)"
	@echo "• Advanced RAG with hallucination detection"
	@echo "• MCP protocol compatibility"
	@echo "• FastAPI REST endpoints"
	@echo ""
	@echo "Quick Commands:"
	@echo "• make quick-start  - Set up everything for new developers"
	@echo "• make dev         - Start development server"
	@echo "• make test        - Run all tests"
	@echo "• make help        - Show all available commands"
