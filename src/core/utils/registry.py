"""
Provider registry system for dynamic component loading.

This module provides a centralized registry for managing providers,
enabling hot-swapping of components without code changes.
"""

import asyncio
import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .config import get_provider_config
from .simple_logger import get_logger

logger = get_logger(__name__)


class ProviderType(Enum):
    """Types of providers that can be registered."""

    EMBEDDING = "embeddings"
    VECTOR_STORE = "vector_stores"
    GRAPH_ENGINE = "graph_engines"
    GRAPH_MANAGER = "graph_managers"
    GRAPH_CLIENT = "graph_clients"
    RERANKER = "rerankers"
    CACHE = "cache"
    TOOL = "tools"


@dataclass
class ProviderInfo:
    """Information about a registered provider."""

    name: str
    provider_type: ProviderType
    class_path: str
    provider_class: Type
    config: Dict[str, Any]
    instance: Optional[Any] = None
    initialized: bool = False
    last_error: Optional[str] = None
    load_time: Optional[float] = None
    usage_count: int = 0
    health_status: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProviderRegistry:
    """
    Registry for managing providers with dynamic loading and health monitoring.

    Features:
    - Dynamic provider loading from configuration
    - Instance caching and reuse
    - Health monitoring and automatic fallbacks
    - Hot-swapping without service restart
    - Usage statistics and monitoring
    """

    def __init__(self):
        self._providers: Dict[str, Dict[str, ProviderInfo]] = {
            provider_type.value: {} for provider_type in ProviderType
        }
        self._lock = asyncio.Lock()
        self._fallback_providers: Dict[str, str] = {}
        self._health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None

    async def register_provider(
        self,
        provider_type: ProviderType,
        name: str,
        class_path: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a provider with the registry.

        Args:
            provider_type: Type of provider
            name: Provider name/identifier
            class_path: Full class path (e.g., "module.class.ClassName")
            config: Provider configuration
            metadata: Optional metadata

        Returns:
            True if registration successful
        """
        async with self._lock:
            try:
                # Load the provider class
                provider_class = self._load_class(class_path)

                # Validate provider class
                if not self._validate_provider_class(provider_type, provider_class):
                    raise ValueError(
                        f"Invalid provider class for {provider_type.value}"
                    )

                # Create provider info
                provider_info = ProviderInfo(
                    name=name,
                    provider_type=provider_type,
                    class_path=class_path,
                    provider_class=provider_class,
                    config=config,
                    metadata=metadata or {},
                )

                # Register provider
                self._providers[provider_type.value][name] = provider_info

                logger.info(
                    "Provider registered successfully",
                    provider_type=provider_type.value,
                    name=name,
                    class_path=class_path,
                )

                return True

            except Exception as e:
                logger.error(
                    "Failed to register provider",
                    provider_type=provider_type.value,
                    name=name,
                    class_path=class_path,
                    error=str(e),
                )
                return False

    def _load_class(self, class_path: str) -> Type:
        """Load a class from its full path."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            raise ImportError(f"Failed to load class {class_path}: {e}")

    def _validate_provider_class(
        self, provider_type: ProviderType, provider_class: Type
    ) -> bool:
        """Validate that a provider class implements the required interface."""
        # Import interfaces dynamically to avoid circular imports
        interface_map = {
            ProviderType.EMBEDDING: "src.core.interfaces.embeddings.EmbeddingProvider",
            ProviderType.VECTOR_STORE: "src.core.interfaces.vector_store.VectorStore",
            ProviderType.GRAPH_ENGINE: "src.core.interfaces.graph_engine.GraphEngine",
            ProviderType.RERANKER: "src.core.interfaces.reranker.Reranker",
        }

        if provider_type not in interface_map:
            return True  # No validation for other types

        try:
            interface_class = self._load_class(interface_map[provider_type])
            return issubclass(provider_class, interface_class)
        except Exception as e:
            logger.warning(
                "Could not validate provider interface",
                provider_type=provider_type.value,
                error=str(e),
            )
            return True  # Allow if validation fails

    async def get_provider(
        self, provider_type: ProviderType, name: str, initialize: bool = True
    ) -> Any:
        """
        Get a provider instance.

        Args:
            provider_type: Type of provider
            name: Provider name
            initialize: Whether to initialize the provider if not already done

        Returns:
            Provider instance
        """
        async with self._lock:
            providers = self._providers.get(provider_type.value, {})

            if name not in providers:
                # Try to load from configuration
                await self._load_provider_from_config(provider_type, name)
                providers = self._providers.get(provider_type.value, {})

                if name not in providers:
                    raise ValueError(
                        f"Provider {name} not found for {provider_type.value}"
                    )

            provider_info = providers[name]
            provider_info.usage_count += 1

            # Create instance if needed
            if provider_info.instance is None:
                try:
                    provider_info.instance = provider_info.provider_class()
                    provider_info.load_time = asyncio.get_event_loop().time()
                except Exception as e:
                    provider_info.last_error = str(e)
                    logger.error(
                        "Failed to create provider instance",
                        provider_type=provider_type.value,
                        name=name,
                        error=str(e),
                    )
                    raise

            # Initialize if needed
            if initialize and not provider_info.initialized:
                try:
                    if hasattr(provider_info.instance, "initialize"):
                        if inspect.iscoroutinefunction(
                            provider_info.instance.initialize
                        ):
                            await provider_info.instance.initialize(
                                provider_info.config
                            )
                        else:
                            provider_info.instance.initialize(provider_info.config)

                    provider_info.initialized = True
                    provider_info.health_status = "healthy"

                    logger.info(
                        "Provider initialized successfully",
                        provider_type=provider_type.value,
                        name=name,
                    )

                except Exception as e:
                    provider_info.last_error = str(e)
                    provider_info.health_status = "unhealthy"
                    logger.error(
                        "Failed to initialize provider",
                        provider_type=provider_type.value,
                        name=name,
                        error=str(e),
                    )
                    raise

            return provider_info.instance

    async def _load_provider_from_config(self, provider_type: ProviderType, name: str):
        """Load provider from configuration files."""
        try:
            provider_config = get_provider_config(provider_type.value, name)
            class_path = provider_config.get("class")
            config = provider_config.get("config", {})

            if not class_path:
                raise ValueError(f"No class path found for provider {name}")

            await self.register_provider(
                provider_type=provider_type,
                name=name,
                class_path=class_path,
                config=config,
            )

        except Exception as e:
            logger.error(
                "Failed to load provider from config",
                provider_type=provider_type.value,
                name=name,
                error=str(e),
            )
            raise

    async def list_providers(
        self,
        provider_type: Optional[ProviderType] = None,
        include_instances: bool = False,
    ) -> Dict[str, Any]:
        """
        List all registered providers.

        Args:
            provider_type: Filter by provider type
            include_instances: Whether to include instance information

        Returns:
            Dictionary of provider information
        """
        result = {}

        provider_types = [provider_type] if provider_type else list(ProviderType)

        async with self._lock:
            for ptype in provider_types:
                type_name = ptype.value if isinstance(ptype, ProviderType) else ptype
                providers = self._providers.get(type_name, {})

                result[type_name] = {}
                for name, info in providers.items():
                    provider_data = {
                        "name": info.name,
                        "class_path": info.class_path,
                        "initialized": info.initialized,
                        "health_status": info.health_status,
                        "usage_count": info.usage_count,
                        "last_error": info.last_error,
                        "load_time": info.load_time,
                        "metadata": info.metadata,
                    }

                    if include_instances and info.instance:
                        provider_data["has_instance"] = True
                        provider_data["instance_type"] = type(info.instance).__name__

                    result[type_name][name] = provider_data

        return result

    async def reload_provider(self, provider_type: ProviderType, name: str) -> bool:
        """
        Reload a provider (hot-swap).

        Args:
            provider_type: Type of provider
            name: Provider name

        Returns:
            True if reload successful
        """
        async with self._lock:
            providers = self._providers.get(provider_type.value, {})

            if name not in providers:
                logger.warning(
                    "Cannot reload non-existent provider",
                    provider_type=provider_type.value,
                    name=name,
                )
                return False

            provider_info = providers[name]

            try:
                # Close existing instance if it has a close method
                if provider_info.instance and hasattr(provider_info.instance, "close"):
                    if inspect.iscoroutinefunction(provider_info.instance.close):
                        await provider_info.instance.close()
                    else:
                        provider_info.instance.close()

                # Reload the class
                importlib.reload(
                    importlib.import_module(provider_info.class_path.rsplit(".", 1)[0])
                )
                provider_info.provider_class = self._load_class(
                    provider_info.class_path
                )

                # Reset instance
                provider_info.instance = None
                provider_info.initialized = False
                provider_info.last_error = None
                provider_info.health_status = "unknown"

                logger.info(
                    "Provider reloaded successfully",
                    provider_type=provider_type.value,
                    name=name,
                )

                return True

            except Exception as e:
                provider_info.last_error = str(e)
                provider_info.health_status = "error"
                logger.error(
                    "Failed to reload provider",
                    provider_type=provider_type.value,
                    name=name,
                    error=str(e),
                )
                return False

    async def set_fallback_provider(
        self, provider_type: ProviderType, primary_name: str, fallback_name: str
    ):
        """Set a fallback provider for automatic switching."""
        key = f"{provider_type.value}:{primary_name}"
        self._fallback_providers[key] = fallback_name

        logger.info(
            "Fallback provider configured",
            provider_type=provider_type.value,
            primary=primary_name,
            fallback=fallback_name,
        )

    async def get_provider_with_fallback(
        self, provider_type: ProviderType, name: str
    ) -> Any:
        """Get provider with automatic fallback if primary fails."""
        try:
            return await self.get_provider(provider_type, name)
        except Exception as e:
            logger.warning(
                "Primary provider failed, trying fallback",
                provider_type=provider_type.value,
                primary=name,
                error=str(e),
            )

            # Try fallback
            key = f"{provider_type.value}:{name}"
            fallback_name = self._fallback_providers.get(key)

            if fallback_name:
                try:
                    return await self.get_provider(provider_type, fallback_name)
                except Exception as fallback_error:
                    logger.error(
                        "Fallback provider also failed",
                        provider_type=provider_type.value,
                        fallback=fallback_name,
                        error=str(fallback_error),
                    )

            # Re-raise original error if no fallback or fallback failed
            raise e

    async def health_check(
        self, provider_type: ProviderType, name: str
    ) -> Dict[str, Any]:
        """Perform health check on a provider."""
        try:
            provider = await self.get_provider(provider_type, name, initialize=False)

            if hasattr(provider, "health_check"):
                if inspect.iscoroutinefunction(provider.health_check):
                    result = await provider.health_check()
                else:
                    result = provider.health_check()

                # Update health status
                async with self._lock:
                    providers = self._providers.get(provider_type.value, {})
                    if name in providers:
                        providers[name].health_status = result.get("status", "unknown")

                return result
            else:
                return {"status": "healthy", "message": "No health check available"}

        except Exception as e:
            # Update health status
            async with self._lock:
                providers = self._providers.get(provider_type.value, {})
                if name in providers:
                    providers[name].health_status = "unhealthy"
                    providers[name].last_error = str(e)

            return {"status": "unhealthy", "error": str(e)}

    async def start_health_monitoring(self):
        """Start periodic health monitoring of all providers."""
        if self._health_check_task is not None:
            return

        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started provider health monitoring")

    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped provider health monitoring")

    async def _health_check_loop(self):
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)

                # Check all initialized providers
                for provider_type_name, providers in self._providers.items():
                    for name, info in providers.items():
                        if info.initialized and info.instance:
                            try:
                                provider_type = ProviderType(provider_type_name)
                                await self.health_check(provider_type, name)
                            except Exception as e:
                                logger.error(
                                    "Health check failed",
                                    provider_type=provider_type_name,
                                    name=name,
                                    error=str(e),
                                )

                logger.debug("Completed health check cycle")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        stats = {
            "provider_counts": {},
            "health_summary": {},
            "usage_stats": {},
            "error_summary": {},
        }

        async with self._lock:
            for provider_type_name, providers in self._providers.items():
                # Count providers
                stats["provider_counts"][provider_type_name] = len(providers)

                # Health summary
                health_counts = {"healthy": 0, "unhealthy": 0, "unknown": 0}
                total_usage = 0
                error_count = 0

                for info in providers.values():
                    health_counts[info.health_status] = (
                        health_counts.get(info.health_status, 0) + 1
                    )
                    total_usage += info.usage_count
                    if info.last_error:
                        error_count += 1

                stats["health_summary"][provider_type_name] = health_counts
                stats["usage_stats"][provider_type_name] = total_usage
                stats["error_summary"][provider_type_name] = error_count

        return stats

    async def cleanup(self):
        """Cleanup registry and close all providers."""
        await self.stop_health_monitoring()

        async with self._lock:
            for provider_type_name, providers in self._providers.items():
                for name, info in providers.items():
                    if info.instance and hasattr(info.instance, "close"):
                        try:
                            if inspect.iscoroutinefunction(info.instance.close):
                                await info.instance.close()
                            else:
                                info.instance.close()
                        except Exception as e:
                            logger.error(
                                "Error closing provider",
                                provider_type=provider_type_name,
                                name=name,
                                error=str(e),
                            )

        logger.info("Provider registry cleanup completed")


# Global provider registry
provider_registry = ProviderRegistry()


async def get_provider(provider_type: ProviderType, name: str) -> Any:
    """Get a provider instance from the global registry."""
    return await provider_registry.get_provider(provider_type, name)


async def get_provider_with_fallback(provider_type: ProviderType, name: str) -> Any:
    """Get provider with automatic fallback."""
    return await provider_registry.get_provider_with_fallback(provider_type, name)
