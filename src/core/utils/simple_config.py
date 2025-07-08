"""
Simplified configuration management for Tyra Memory Server.

This module provides a straightforward configuration loading system that works
with our YAML configuration files and environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigLoader:
    """Simple configuration loader with environment variable substitution."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = self._find_config_dir(config_dir)
        self._config_cache: Dict[str, Any] = {}

    def _find_config_dir(self, config_dir: Optional[Union[str, Path]]) -> Path:
        """Find the configuration directory."""
        if config_dir:
            return Path(config_dir)

        # Look for config directory relative to this file
        current = Path(__file__).parent.parent.parent.parent  # Go up to project root
        config_path = current / "config"

        if config_path.exists():
            return config_path

        # Fallback
        return Path("config")

    def _substitute_env_vars(self, text: str) -> str:
        """Substitute environment variables in text.

        Supports formats:
        - ${VAR_NAME:-default_value}
        - ${VAR_NAME}
        - $VAR_NAME
        """

        def replace_var(match):
            full_match = match.group(0)

            if full_match.startswith("${") and ":-" in full_match:
                # Handle ${VAR:-default} format
                var_part = full_match[2:-1]  # Remove ${ and }
                var_name, default = var_part.split(":-", 1)
                return os.getenv(var_name, default)
            elif full_match.startswith("${"):
                # Handle ${VAR} format
                var_name = full_match[2:-1]  # Remove ${ and }
                return os.getenv(var_name, full_match)
            else:
                # Handle $VAR format
                var_name = full_match[1:]  # Remove $
                return os.getenv(var_name, full_match)

        # Pattern to match environment variables
        pattern = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
        return pattern.sub(replace_var, text)

    def _process_value(self, value: Any) -> Any:
        """Recursively process configuration values."""
        if isinstance(value, dict):
            return {k: self._process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._process_value(item) for item in value]
        elif isinstance(value, str):
            return self._substitute_env_vars(value)
        else:
            return value

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        file_path = self.config_dir / filename

        if not file_path.exists():
            print(f"Warning: Configuration file not found: {file_path}")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Process environment variables
            return self._process_value(data)

        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {filename}: {e}")
            return {}
        except Exception as e:
            print(f"Error loading configuration file {filename}: {e}")
            return {}

    def get_config(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get the main configuration."""
        if use_cache and "main" in self._config_cache:
            return self._config_cache["main"]

        config = self.load_yaml("config.yaml")

        if use_cache:
            self._config_cache["main"] = config

        return config

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Example: get('database.postgresql.host', 'localhost')
        """
        config = self.get_config()
        keys = key_path.split(".")

        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_settings() -> Dict[str, Any]:
    """Get configuration settings - simplified interface."""
    return get_config_loader().get_config()


def get_setting(key_path: str, default: Any = None) -> Any:
    """Get a specific setting using dot notation."""
    return get_config_loader().get(key_path, default)


# Create a simple object-like interface for easy access
class ConfigProxy:
    """Configuration proxy for easy attribute access."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigProxy(value)
            return value
        raise AttributeError(f"Configuration key '{name}' not found")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with default."""
        value = self._config.get(key, default)
        if isinstance(value, dict):
            return ConfigProxy(value)
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def keys(self):
        return self._config.keys()

    def items(self):
        return self._config.items()

    def __repr__(self) -> str:
        return f"ConfigProxy({self._config})"


def get_config_proxy() -> ConfigProxy:
    """Get configuration as a proxy object for easy attribute access."""
    return ConfigProxy(get_settings())
