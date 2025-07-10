#!/usr/bin/env python3
"""
Migration 002: Add observability configuration
Version: 1.1.0
Date: 2025-01-10

This migration adds comprehensive observability configuration including tracing,
metrics, and telemetry optimization settings.
"""

from typing import Dict, Any


class Migration:
    """Add observability configuration."""
    
    version = "1.1.0"
    description = "Add observability and telemetry configuration"
    
    @staticmethod
    def up(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration to configuration."""
        # Update version
        config["version"] = "1.1.0"
        
        # Add observability configuration
        if "observability" not in config:
            config["observability"] = {
                "enabled": True,
                "tracing": {
                    "enabled": True,
                    "exporter": "console",
                    "sample_rate": 1.0,
                    "service_name": "tyra-mcp-memory-server",
                    "trace_all_operations": True
                },
                "metrics": {
                    "enabled": True,
                    "exporter": "console",
                    "collection_interval": 10,
                    "histograms": {
                        "embedding_latency": [10, 25, 50, 100, 250, 500, 1000],
                        "search_latency": [10, 25, 50, 100, 250, 500, 1000],
                        "rerank_latency": [25, 50, 100, 250, 500, 1000, 2000]
                    }
                },
                "logging": {
                    "level": "INFO",
                    "format": "structured",
                    "include_trace_id": True
                },
                "telemetry_optimization": {
                    "enabled": True,
                    "adaptive_sampling": True,
                    "performance_mode": "balanced",
                    "auto_adjust_interval": 300
                }
            }
        
        # Add telemetry settings to existing sections
        if "memory" in config and "telemetry" not in config["memory"]:
            config["memory"]["telemetry"] = {
                "trace_queries": True,
                "log_slow_queries": True,
                "slow_query_threshold_ms": 100
            }
        
        if "embedding" in config and "telemetry" not in config["embedding"]:
            config["embedding"]["telemetry"] = {
                "trace_generations": True,
                "log_fallback_usage": True
            }
        
        return config
    
    @staticmethod
    def down(config: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback migration."""
        # Revert version
        config["version"] = "1.0.0"
        
        # Remove observability configuration
        config.pop("observability", None)
        
        # Remove telemetry from sections
        if "memory" in config and "telemetry" in config["memory"]:
            config["memory"].pop("telemetry", None)
        
        if "embedding" in config and "telemetry" in config["embedding"]:
            config["embedding"].pop("telemetry", None)
        
        return config