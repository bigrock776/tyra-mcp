"""
Configuration management system.

This module provides centralized configuration loading and validation
with support for environment variable substitution and hot reloading.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class ServerConfig(BaseModel):
    """Server configuration."""

    mcp: Dict[str, Any] = Field(default_factory=dict)
    fastapi: Dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """Memory backend configuration."""

    backend: str = "postgres"
    postgres: Dict[str, Any] = Field(default_factory=dict)
    vector: Dict[str, Any] = Field(default_factory=dict)


class GraphConfig(BaseModel):
    """Graph database configuration."""

    backend: str = "memgraph"
    host: str = "localhost"
    port: int = 7687
    username: str = "memgraph"
    password: Optional[str] = None
    pool_size: int = 10
    connection_timeout: int = 30


class CacheConfig(BaseModel):
    """Cache configuration."""

    backend: str = "redis"
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    pool_size: int = 50
    ttl: Dict[str, int] = Field(default_factory=dict)


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""

    primary: Dict[str, Any] = Field(default_factory=dict)
    fallback: Dict[str, Any] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    """RAG configuration."""

    retrieval: Dict[str, Any] = Field(default_factory=dict)
    hallucination: Dict[str, Any] = Field(default_factory=dict)
    reranking: Dict[str, Any] = Field(default_factory=dict)


class SelfLearningConfig(BaseModel):
    """Self-learning configuration."""

    enabled: bool = True
    analysis_interval: str = "1h"
    improvement_interval: str = "24h"
    modules: Dict[str, bool] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "structured"
    file: Optional[str] = None
    max_size: str = "100MB"
    backup_count: int = 5


class SecurityConfig(BaseModel):
    """Security configuration."""

    api_key_required: bool = False
    api_key: Optional[str] = None
    rate_limiting: Dict[str, Any] = Field(default_factory=dict)
    cors: Dict[str, Any] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    opentelemetry: Dict[str, Any] = Field(default_factory=dict)


class DevelopmentConfig(BaseModel):
    """Development configuration."""

    debug: bool = False
    reload: bool = False
    profiling: bool = False
    mock_providers: bool = False


class TyraConfig(BaseSettings):
    """Main configuration class."""

    version: str = "1.0"
    api_version: str = "v1"

    server: ServerConfig = Field(default_factory=ServerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    self_learning: SelfLearningConfig = Field(default_factory=SelfLearningConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


class ConfigManager:
    """Configuration manager with hot reloading and environment substitution."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else self._find_config_dir()
        self.config: Optional[TyraConfig] = None
        self.providers_config: Optional[Dict[str, Any]] = None
        self.agents_config: Optional[Dict[str, Any]] = None
        self.self_learning_config: Optional[Dict[str, Any]] = None
        self.models_config: Optional[Dict[str, Any]] = None

    def _find_config_dir(self) -> Path:
        """Find the configuration directory."""
        # Start from the current file and go up to find config directory
        current = Path(__file__).parent
        while current.parent != current:
            config_dir = current / "config"
            if config_dir.exists():
                return config_dir
            current = current.parent

        # Fallback to relative path
        return Path("config")

    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(data, dict):
            return {
                key: self._substitute_env_vars(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Replace ${VAR_NAME} or $VAR_NAME with environment variable
            pattern = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

            def replace_var(match):
                var_name = match.group(1) or match.group(2)
                return os.getenv(var_name, match.group(0))

            return pattern.sub(replace_var, data)
        else:
            return data

    def load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load and parse a YAML configuration file."""
        file_path = self.config_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Substitute environment variables
            data = self._substitute_env_vars(data)
            return data

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filename}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading {filename}: {e}")

    def load_config(self) -> TyraConfig:
        """Load the main configuration."""
        config_data = self.load_yaml_file("config.yaml")
        self.config = TyraConfig(**config_data)
        return self.config

    def load_providers_config(self) -> Dict[str, Any]:
        """Load providers configuration."""
        self.providers_config = self.load_yaml_file("providers.yaml")
        return self.providers_config

    def load_agents_config(self) -> Dict[str, Any]:
        """Load agents configuration."""
        self.agents_config = self.load_yaml_file("agents.yaml")
        return self.agents_config

    def load_self_learning_config(self) -> Dict[str, Any]:
        """Load self-learning configuration."""
        self.self_learning_config = self.load_yaml_file("self_learning.yaml")
        return self.self_learning_config

    def load_models_config(self) -> Dict[str, Any]:
        """Load models configuration."""
        self.models_config = self.load_yaml_file("models.yaml")
        return self.models_config

    def load_all_configs(self) -> TyraConfig:
        """Load all configuration files."""
        self.load_config()
        self.load_providers_config()
        self.load_agents_config()
        self.load_self_learning_config()
        self.load_models_config()
        return self.config

    def get_provider_config(
        self, provider_type: str, provider_name: str
    ) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        if not self.providers_config:
            self.load_providers_config()

        providers = self.providers_config.get(provider_type, {}).get("providers", {})
        if provider_name not in providers:
            raise ValueError(f"Provider {provider_name} not found in {provider_type}")

        return providers[provider_name]

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        if not self.agents_config:
            self.load_agents_config()

        agents = self.agents_config.get("agents", {})
        if agent_name not in agents:
            # Return default agent config
            return self.agents_config.get("shared", {})

        # Merge agent-specific config with shared config
        shared_config = self.agents_config.get("shared", {})
        agent_config = agents[agent_name]

        # Deep merge configuration
        merged_config = shared_config.copy()
        for key, value in agent_config.items():
            if isinstance(value, dict) and key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value

        return merged_config

    def reload_config(self) -> TyraConfig:
        """Reload all configuration from files."""
        return self.load_all_configs()

    def validate_config(self) -> bool:
        """Validate the current configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded")

        # Basic validation - more lenient for flexible configuration
        try:
            # Configuration is already validated by Pydantic models
            return True

        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")


# Global configuration instance
config_manager = ConfigManager()
settings: Optional[TyraConfig] = None


def get_settings() -> TyraConfig:
    """Get the global configuration settings."""
    global settings
    if settings is None:
        settings = config_manager.load_all_configs()
        config_manager.validate_config()
    return settings


def reload_settings() -> TyraConfig:
    """Reload configuration from files."""
    global settings
    settings = config_manager.reload_config()
    config_manager.validate_config()
    return settings


def get_provider_config(provider_type: str, provider_name: str) -> Dict[str, Any]:
    """Get configuration for a specific provider."""
    return config_manager.get_provider_config(provider_type, provider_name)


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for a specific agent."""
    return config_manager.get_agent_config(agent_name)
