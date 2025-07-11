"""
Graph engine provider registry for dynamic loading and configuration.

This module provides a centralized registry for graph engine providers,
enabling easy addition of new graph databases and runtime provider selection.
"""

import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Type

from ...interfaces.graph_engine import GraphEngine

logger = logging.getLogger(__name__)


class GraphEngineProviderRegistry:
    """Registry for graph engine providers."""

    _providers: Dict[str, Type[GraphEngine]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        provider_class: Type[GraphEngine],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a graph engine provider.

        Args:
            name: Provider name
            provider_class: Provider class implementing GraphEngine
            aliases: Optional list of aliases for this provider
        """
        if not issubclass(provider_class, GraphEngine):
            raise ValueError(f"Provider {provider_class} must implement GraphEngine")

        cls._providers[name] = provider_class
        logger.info(f"Registered graph engine provider: {name}")

        # Register aliases
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
                logger.debug(f"Registered alias '{alias}' for provider '{name}'")

    @classmethod
    def get_provider(cls, name: str) -> Type[GraphEngine]:
        """
        Get a provider class by name.

        Args:
            name: Provider name or alias

        Returns:
            Provider class

        Raises:
            KeyError: If provider not found
        """
        # Check aliases first
        if name in cls._aliases:
            name = cls._aliases[name]

        if name not in cls._providers:
            raise KeyError(f"Unknown graph engine provider: {name}")

        return cls._providers[name]

    @classmethod
    def create_provider(cls, name: str, config: Dict[str, Any]) -> GraphEngine:
        """
        Create and configure a provider instance.

        Args:
            name: Provider name or alias
            config: Provider configuration

        Returns:
            Configured provider instance
        """
        provider_class = cls.get_provider(name)
        return provider_class(**config)

    @classmethod
    def list_providers(cls) -> List[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())

    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """Get mapping of aliases to provider names."""
        return cls._aliases.copy()

    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a provider is available."""
        return name in cls._providers or name in cls._aliases


# Auto-register available providers
def _register_default_providers():
    """Register default graph engine providers."""
    try:
        from .memgraph import MemgraphEngine

        GraphEngineProviderRegistry.register(
            "memgraph", MemgraphEngine, aliases=["memgraph-cloud", "mg"]
        )

    except ImportError as e:
        logger.warning(f"Could not register Memgraph provider: {e}")

    # Register additional providers as they become available
    # try:
    #     from .neptune import NeptuneEngine
    #     GraphEngineProviderRegistry.register("neptune", NeptuneEngine)
    # except ImportError:
    #     pass


# Register providers on module import
_register_default_providers()


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available providers.

    Returns:
        Dictionary with provider info
    """
    providers = {}

    for name in GraphEngineProviderRegistry.list_providers():
        try:
            provider_class = GraphEngineProviderRegistry.get_provider(name)
            providers[name] = {
                "class": provider_class.__name__,
                "module": provider_class.__module__,
                "description": getattr(provider_class, "__doc__", "No description"),
            }
        except Exception as e:
            providers[name] = {"error": str(e)}

    return providers


def create_graph_engine(config: Dict[str, Any]) -> GraphEngine:
    """
    Create graph engine from configuration.

    Args:
        config: Configuration dictionary with 'provider' key and provider-specific config

    Returns:
        Configured graph engine

    Example:
        config = {
            "provider": "memgraph",
            "host": "localhost",
            "port": 7687,
            "username": "memgraph",
            "password": "memgraph"
        }
        engine = create_graph_engine(config)
    """
    if "provider" not in config:
        raise ValueError("Configuration must include 'provider' key")

    provider_name = config.pop("provider")
    return GraphEngineProviderRegistry.create_provider(provider_name, config)
