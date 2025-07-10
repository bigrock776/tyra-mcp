#!/usr/bin/env python3
"""
Migration 003: Add self-learning and analytics
Version: 1.2.0
Date: 2025-01-10

This migration adds self-learning, analytics, and A/B testing configuration.
"""

from typing import Dict, Any


class Migration:
    """Add self-learning and analytics configuration."""
    
    version = "1.2.0"
    description = "Add self-learning, analytics, and A/B testing"
    
    @staticmethod
    def up(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration to configuration."""
        # Update version
        config["version"] = "1.2.0"
        
        # Add self-learning configuration
        if "self_learning" not in config:
            config["self_learning"] = {
                "enabled": True,
                "analysis_interval": "1h",
                "improvement_interval": "24h",
                "auto_optimize": True,
                "modules": {
                    "memory_health": True,
                    "performance_optimization": True,
                    "prompt_evolution": True,
                    "ab_testing": True,
                    "config_adaptation": True
                },
                "thresholds": {
                    "memory_staleness_days": 30,
                    "low_confidence_threshold": 0.6,
                    "performance_degradation_percent": 20,
                    "min_samples_for_analysis": 100
                },
                "constraints": {
                    "max_auto_changes_per_day": 5,
                    "require_approval_above": 0.3,
                    "rollback_on_error": True
                }
            }
        
        # Add analytics configuration
        if "analytics" not in config:
            config["analytics"] = {
                "enabled": True,
                "retention_days": 90,
                "aggregation_intervals": ["1h", "1d", "1w"],
                "dashboards": {
                    "performance": True,
                    "usage": True,
                    "quality": True,
                    "self_learning": True
                }
            }
        
        # Add A/B testing configuration
        if "ab_testing" not in config:
            config["ab_testing"] = {
                "enabled": True,
                "default_traffic_split": 0.1,
                "min_experiment_duration": "24h",
                "statistical_significance": 0.95,
                "auto_conclude": True,
                "experiments": {}
            }
        
        return config
    
    @staticmethod
    def down(config: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback migration."""
        # Revert version
        config["version"] = "1.1.0"
        
        # Remove new configurations
        config.pop("self_learning", None)
        config.pop("analytics", None)
        config.pop("ab_testing", None)
        
        return config