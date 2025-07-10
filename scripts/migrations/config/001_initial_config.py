#!/usr/bin/env python3
"""
Migration 001: Initial configuration structure
Version: 1.0.0
Date: 2025-01-10

This migration establishes the base configuration structure for the Tyra MCP Memory Server.
"""

from typing import Dict, Any


class Migration:
    """Initial configuration migration."""
    
    version = "1.0.0"
    description = "Initial configuration structure"
    
    @staticmethod
    def up(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration to configuration."""
        # Ensure base structure exists
        base_structure = {
            "version": "1.0.0",
            "memory": {
                "backend": "postgres",
                "postgres": {
                    "pool_size": 20,
                    "timeout": 30
                }
            },
            "embedding": {
                "primary": "e5-large",
                "fallback": "all-minilm",
                "batch_size": 32
            },
            "vector_store": {
                "primary": "pgvector",
                "search": {
                    "default_top_k": 10,
                    "max_top_k": 100
                }
            },
            "graph": {
                "engine": "memgraph",
                "manager": "graphiti",
                "enable_temporal": True
            },
            "rag": {
                "retrieval": {
                    "strategy": "hybrid",
                    "hybrid_weight": 0.7
                },
                "reranking": {
                    "enabled": True,
                    "provider": "cross_encoder"
                },
                "hallucination": {
                    "enabled": True,
                    "threshold": 0.75
                }
            },
            "cache": {
                "enabled": True,
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0
                },
                "ttl": {
                    "embeddings": 86400,
                    "search": 3600,
                    "rerank": 1800
                }
            }
        }
        
        # Merge with existing config
        for key, value in base_structure.items():
            if key not in config:
                config[key] = value
        
        return config
    
    @staticmethod
    def down(config: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback migration."""
        # Remove version field
        config.pop("version", None)
        return config