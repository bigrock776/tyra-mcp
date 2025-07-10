"""Configuration migrations package."""

import importlib
import os
from pathlib import Path
from typing import List, Type

# Migration modules in order
MIGRATIONS = [
    "001_initial_config",
    "002_add_observability",
    "003_add_self_learning",
    "004_add_trading_safety"
]


def get_migration_modules() -> List[Type]:
    """Get all migration modules in order."""
    modules = []
    
    for migration_name in MIGRATIONS:
        try:
            module = importlib.import_module(f"scripts.migrations.config.{migration_name}")
            if hasattr(module, "Migration"):
                modules.append(module.Migration)
        except ImportError as e:
            print(f"Warning: Could not import migration {migration_name}: {e}")
    
    return modules


def get_latest_version() -> str:
    """Get the latest migration version."""
    modules = get_migration_modules()
    if modules:
        return modules[-1].version
    return "0.0.0"