"""
Reranker provider registry for dynamic loading and configuration.

This module provides a centralized registry for reranker providers,
enabling easy addition of new reranking models and runtime provider selection.
"""

import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Type

from ...interfaces.reranker import Reranker

logger = logging.getLogger(__name__)


class RerankerProviderRegistry:
    """Registry for reranker providers."""

    _providers: Dict[str, Type[Reranker]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        provider_class: Type[Reranker],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a reranker provider.

        Args:
            name: Provider name
            provider_class: Provider class implementing Reranker
            aliases: Optional list of aliases for this provider
        """
        if not issubclass(provider_class, Reranker):
            raise ValueError(f"Provider {provider_class} must implement Reranker")

        cls._providers[name] = provider_class
        logger.info(f"Registered reranker provider: {name}")

        # Register aliases
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
                logger.debug(f"Registered alias '{alias}' for provider '{name}'")

    @classmethod
    def get_provider(cls, name: str) -> Type[Reranker]:
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
            raise KeyError(f"Unknown reranker provider: {name}")

        return cls._providers[name]

    @classmethod
    def create_provider(cls, name: str, config: Dict[str, Any]) -> Reranker:
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
    """Register default reranker providers."""
    try:
        from .cross_encoder import CrossEncoderReranker

        RerankerProviderRegistry.register(
            "cross_encoder",
            CrossEncoderReranker,
            aliases=["ce", "cross-encoder", "ms-marco"],
        )

        # Model-specific aliases
        RerankerProviderRegistry.register(
            "ms-marco-mini",
            CrossEncoderReranker,
            aliases=["cross-encoder/ms-marco-MiniLM-L-12-v2"],
        )

        RerankerProviderRegistry.register(
            "ms-marco-base",
            CrossEncoderReranker,
            aliases=["cross-encoder/ms-marco-MiniLM-L-6-v2"],
        )

    except ImportError as e:
        logger.warning(f"Could not register CrossEncoder provider: {e}")

    # Register additional providers as they become available
    try:
        from ...rag.vllm_reranker import VLLMReranker
        RerankerProviderRegistry.register(
            "vllm",
            VLLMReranker,
            aliases=["vllm-reranker", "local-llm", "llm"]
        )
        logger.info("Registered vLLM reranker provider")
    except ImportError as e:
        logger.warning(f"Could not register vLLM reranker provider: {e}")

    # try:
    #     from .bge_reranker import BGEReranker
    #     RerankerProviderRegistry.register("bge", BGEReranker)
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

    for name in RerankerProviderRegistry.list_providers():
        try:
            provider_class = RerankerProviderRegistry.get_provider(name)
            providers[name] = {
                "class": provider_class.__name__,
                "module": provider_class.__module__,
                "description": getattr(provider_class, "__doc__", "No description"),
            }
        except Exception as e:
            providers[name] = {"error": str(e)}

    return providers


def create_reranker(config: Dict[str, Any]) -> Reranker:
    """
    Create reranker from configuration.

    Args:
        config: Configuration dictionary with 'provider' key and provider-specific config

    Returns:
        Configured reranker

    Example:
        config = {
            "provider": "cross_encoder",
            "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "device": "cuda"
        }
        reranker = create_reranker(config)
    """
    if "provider" not in config:
        raise ValueError("Configuration must include 'provider' key")

    provider_name = config.pop("provider")
    return RerankerProviderRegistry.create_provider(provider_name, config)
