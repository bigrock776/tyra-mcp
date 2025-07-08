"""
Embedding provider registry for dynamic loading and configuration.

This module provides a centralized registry for embedding providers,
enabling easy addition of new models and runtime provider selection.
"""

import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Type

from ...interfaces.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingProviderRegistry:
    """Registry for embedding providers."""

    _providers: Dict[str, Type[EmbeddingProvider]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        provider_class: Type[EmbeddingProvider],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register an embedding provider.

        Args:
            name: Provider name
            provider_class: Provider class implementing EmbeddingProvider
            aliases: Optional list of aliases for this provider
        """
        if not issubclass(provider_class, EmbeddingProvider):
            raise ValueError(
                f"Provider {provider_class} must implement EmbeddingProvider"
            )

        cls._providers[name] = provider_class
        logger.info(f"Registered embedding provider: {name}")

        # Register aliases
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
                logger.debug(f"Registered alias '{alias}' for provider '{name}'")

    @classmethod
    def get_provider(cls, name: str) -> Type[EmbeddingProvider]:
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
            raise KeyError(f"Unknown embedding provider: {name}")

        return cls._providers[name]

    @classmethod
    def create_provider(cls, name: str, config: Dict[str, Any]) -> EmbeddingProvider:
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
    """Register default embedding providers."""
    try:
        from .huggingface import HuggingFaceEmbeddingProvider

        EmbeddingProviderRegistry.register(
            "huggingface", HuggingFaceEmbeddingProvider, aliases=["hf", "transformers"]
        )

        # Model-specific aliases
        EmbeddingProviderRegistry.register(
            "e5-large",
            HuggingFaceEmbeddingProvider,
            aliases=["e5", "intfloat/e5-large-v2"],
        )

        EmbeddingProviderRegistry.register(
            "minilm",
            HuggingFaceEmbeddingProvider,
            aliases=["all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L12-v2"],
        )

    except ImportError as e:
        logger.warning(f"Could not register HuggingFace provider: {e}")


# Register providers on module import
_register_default_providers()


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available providers.

    Returns:
        Dictionary with provider info
    """
    providers = {}

    for name in EmbeddingProviderRegistry.list_providers():
        try:
            provider_class = EmbeddingProviderRegistry.get_provider(name)
            providers[name] = {
                "class": provider_class.__name__,
                "module": provider_class.__module__,
                "description": getattr(provider_class, "__doc__", "No description"),
            }
        except Exception as e:
            providers[name] = {"error": str(e)}

    return providers


def create_embedding_provider(config: Dict[str, Any]) -> EmbeddingProvider:
    """
    Create embedding provider from configuration.

    Args:
        config: Configuration dictionary with 'provider' key and provider-specific config

    Returns:
        Configured embedding provider

    Example:
        config = {
            "provider": "huggingface",
            "model_name": "intfloat/e5-large-v2",
            "device": "cuda"
        }
        provider = create_embedding_provider(config)
    """
    if "provider" not in config:
        raise ValueError("Configuration must include 'provider' key")

    provider_name = config.pop("provider")
    return EmbeddingProviderRegistry.create_provider(provider_name, config)
