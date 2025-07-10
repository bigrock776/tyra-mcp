#!/usr/bin/env python3
"""
Migration 004: Add trading safety configuration
Version: 1.3.0
Date: 2025-01-10

This migration adds comprehensive trading safety configuration for agents
and API endpoints that handle trading operations.
"""

from typing import Dict, Any


class Migration:
    """Add trading safety configuration."""
    
    version = "1.3.0"
    description = "Add trading safety and confidence requirements"
    
    @staticmethod
    def up(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration to configuration."""
        # Update version
        config["version"] = "1.3.0"
        
        # Add trading safety configuration
        if "trading_safety" not in config:
            config["trading_safety"] = {
                "enabled": True,
                "min_confidence_score": 0.95,
                "require_hallucination_check": True,
                "audit_all_decisions": True,
                "double_check_threshold": 0.98,
                "forbidden_patterns": [
                    "maybe",
                    "possibly",
                    "might be",
                    "could be",
                    "not sure",
                    "approximately"
                ],
                "required_confirmations": {
                    "high_value_trades": 2,
                    "new_instruments": 2,
                    "unusual_patterns": 3
                },
                "safety_checks": {
                    "verify_data_freshness": True,
                    "check_market_hours": True,
                    "validate_price_ranges": True,
                    "confirm_liquidity": True
                }
            }
        
        # Add API-specific safety configuration
        if "api" in config:
            if "safety" not in config["api"]:
                config["api"]["safety"] = {
                    "trading_endpoints": {
                        "/v1/chat/trading": {
                            "min_confidence": 0.95,
                            "require_auth": True,
                            "rate_limit": 10,
                            "audit_log": True
                        }
                    },
                    "high_risk_operations": [
                        "execute_trade",
                        "modify_position",
                        "close_position"
                    ]
                }
        
        # Update agent configurations for trading safety
        if "agents" in config:
            config["agents"]["_defaults"] = {
                "trading_safety": {
                    "enabled": False,
                    "min_confidence": 0.90
                }
            }
        
        # Add confidence level definitions
        if "confidence_levels" not in config:
            config["confidence_levels"] = {
                "rock_solid": {
                    "min_score": 0.95,
                    "description": "Safe for automated actions",
                    "emoji": "💪"
                },
                "high": {
                    "min_score": 0.80,
                    "description": "Generally reliable",
                    "emoji": "🧠"
                },
                "fuzzy": {
                    "min_score": 0.60,
                    "description": "Needs verification",
                    "emoji": "🤔"
                },
                "low": {
                    "min_score": 0.0,
                    "description": "Not confident",
                    "emoji": "⚠️"
                }
            }
        
        return config
    
    @staticmethod
    def down(config: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback migration."""
        # Revert version
        config["version"] = "1.2.0"
        
        # Remove trading safety configurations
        config.pop("trading_safety", None)
        config.pop("confidence_levels", None)
        
        # Remove API safety
        if "api" in config and "safety" in config["api"]:
            config["api"].pop("safety", None)
        
        # Remove agent defaults
        if "agents" in config and "_defaults" in config["agents"]:
            config["agents"].pop("_defaults", None)
        
        return config