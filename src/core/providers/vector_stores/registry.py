"""
Vector store provider registry for dynamic loading and configuration.

This module provides a centralized registry for vector store providers,
enabling easy addition of new backends and runtime provider selection.
"""

import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Type

from ...interfaces.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VectorStoreProviderRegistry:
    """Registry for vector store providers."""

    _providers: Dict[str, Type[VectorStore]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        provider_class: Type[VectorStore],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a vector store provider.

        Args:
            name: Provider name
            provider_class: Provider class implementing VectorStore
            aliases: Optional list of aliases for this provider
        """
        if not issubclass(provider_class, VectorStore):
            raise ValueError(f"Provider {provider_class} must implement VectorStore")

        cls._providers[name] = provider_class
        logger.info(f"Registered vector store provider: {name}")

        # Register aliases
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
                logger.debug(f"Registered alias '{alias}' for provider '{name}'")

    @classmethod
    def get_provider(cls, name: str) -> Type[VectorStore]:
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
            raise KeyError(f"Unknown vector store provider: {name}")

        return cls._providers[name]

    @classmethod
    def create_provider(cls, name: str, config: Dict[str, Any]) -> VectorStore:
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
    """Register default vector store providers."""
    try:
        from .pgvector import PgVectorStore

        VectorStoreProviderRegistry.register(
            "pgvector", PgVectorStore, aliases=["postgresql", "postgres"]
        )

    except ImportError as e:
        logger.warning(f"Could not register PgVector provider: {e}")

    # Register additional providers as they become available
    # try:
    #     from .qdrant import QdrantStore
    #     VectorStoreProviderRegistry.register("qdrant", QdrantStore)
    # except ImportError:
    #     pass

    # try:
    #     from .weaviate import WeaviateStore
    #     VectorStoreProviderRegistry.register("weaviate", WeaviateStore)
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

    for name in VectorStoreProviderRegistry.list_providers():
        try:
            provider_class = VectorStoreProviderRegistry.get_provider(name)
            providers[name] = {
                "class": provider_class.__name__,
                "module": provider_class.__module__,
                "description": getattr(provider_class, "__doc__", "No description"),
            }
        except Exception as e:
            providers[name] = {"error": str(e)}

    return providers


def create_vector_store(config: Dict[str, Any]) -> VectorStore:
    """
    Create vector store from configuration.

    Args:
        config: Configuration dictionary with 'provider' key and provider-specific config

    Returns:
        Configured vector store

    Example:
        config = {
            "provider": "pgvector",
            "database_url": "postgresql://user:pass@localhost/db",
            "table_name": "embeddings"
        }
        store = create_vector_store(config)
    """
    if "provider" not in config:
        raise ValueError("Configuration must include 'provider' key")

    provider_name = config.pop("provider")
    return VectorStoreProviderRegistry.create_provider(provider_name, config)
