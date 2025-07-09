#!/usr/bin/env python3
"""
Configuration Migration Tool

This script provides utilities to migrate configuration files between different
versions or formats, helping with upgrades and configuration changes.
"""

import argparse
import json
import os
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.utils.simple_config import SimpleConfig
from src.core.utils.simple_logger import get_logger

logger = get_logger(__name__)


class ConfigMigrationTool:
    """Tool for migrating configuration files."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.project_root / "config" / "backups"
        self.migrations_dir = Path(__file__).parent / "migrations"
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_config(self, config_file: Path) -> Path:
        """Create a backup of the configuration file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{config_file.stem}_{timestamp}.backup.yaml"
        
        try:
            shutil.copy2(config_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def load_config(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config {config_file}: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], config_file: Path) -> None:
        """Save configuration to file."""
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() == '.json':
                    json.dump(config, f, indent=2)
                else:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config: {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config {config_file}: {e}")
            raise
    
    def migrate_providers_config(self, source_file: Path, target_version: str = "latest") -> Dict[str, Any]:
        """Migrate providers configuration to latest format."""
        config = self.load_config(source_file)
        
        if target_version == "latest":
            # Apply all migrations in order
            config = self._migrate_providers_v1_to_v2(config)
            config = self._migrate_providers_v2_to_v3(config)
            config = self._add_missing_provider_sections(config)
        
        return config
    
    def _migrate_providers_v1_to_v2(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from v1 to v2 format (flat structure to nested)."""
        migrated = {}
        
        # Check if already in v2+ format
        if any(key in config for key in ["embeddings", "vector_stores", "graph_engines"]):
            return config
        
        # Migrate flat configuration to nested structure
        for key, value in config.items():
            if key.endswith("_providers") or key.endswith("_config"):
                # Extract provider type
                provider_type = key.replace("_providers", "").replace("_config", "")
                
                if provider_type not in migrated:
                    migrated[provider_type] = {"providers": {}}
                
                if key.endswith("_providers"):
                    migrated[provider_type]["providers"] = value
                else:
                    migrated[provider_type]["config"] = value
        
        logger.info("Migrated from v1 to v2 format")
        return migrated if migrated else config
    
    def _migrate_providers_v2_to_v3(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from v2 to v3 format (add new provider types)."""
        # Add new provider types if not present
        new_provider_types = [
            "graph_managers",
            "graph_clients",
            "cache"
        ]
        
        for provider_type in new_provider_types:
            if provider_type not in config:
                config[provider_type] = {"providers": {}}
        
        # Migrate graph_engines to include both engines and managers
        if "graph_engines" in config and "graph_managers" not in config:
            config["graph_managers"] = {"providers": {}}
        
        logger.info("Migrated from v2 to v3 format")
        return config
    
    def _add_missing_provider_sections(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add missing provider sections with defaults."""
        default_providers = {
            "embeddings": {
                "providers": {
                    "e5-large": {
                        "class": "src.core.providers.embeddings.huggingface.HuggingFaceProvider",
                        "config": {
                            "model_name": "intfloat/e5-large-v2",
                            "device": "auto",
                            "normalize_embeddings": True
                        }
                    },
                    "all-minilm": {
                        "class": "src.core.providers.embeddings.huggingface.HuggingFaceProvider",
                        "config": {
                            "model_name": "sentence-transformers/all-MiniLM-L12-v2",
                            "device": "cpu",
                            "normalize_embeddings": True
                        }
                    }
                }
            },
            "vector_stores": {
                "providers": {
                    "pgvector": {
                        "class": "src.core.providers.vector_stores.pgvector.PgVectorStore",
                        "config": {
                            "connection_string": "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}",
                            "table_name": "memory_embeddings"
                        }
                    }
                }
            },
            "graph_engines": {
                "providers": {
                    "memgraph": {
                        "class": "src.core.providers.graph_engines.memgraph.MemgraphEngine",
                        "config": {
                            "host": "localhost",
                            "port": 7687,
                            "username": "memgraph",
                            "password": "${MEMGRAPH_PASSWORD}"
                        }
                    }
                }
            },
            "graph_managers": {
                "providers": {
                    "graphiti": {
                        "class": "src.core.graph.graphiti_integration.GraphitiManager",
                        "config": {
                            "llm_base_url": "http://localhost:8000/v1",
                            "llm_model": "meta-llama/Llama-3.1-70B-Instruct"
                        }
                    }
                }
            },
            "rerankers": {
                "providers": {
                    "cross_encoder": {
                        "class": "src.core.providers.rerankers.cross_encoder.CrossEncoderReranker",
                        "config": {
                            "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                            "device": "auto"
                        }
                    }
                }
            }
        }
        
        for provider_type, default_config in default_providers.items():
            if provider_type not in config:
                config[provider_type] = default_config
                logger.info(f"Added default {provider_type} configuration")
        
        return config
    
    def migrate_main_config(self, source_file: Path, target_version: str = "latest") -> Dict[str, Any]:
        """Migrate main configuration file."""
        config = self.load_config(source_file)
        
        if target_version == "latest":
            config = self._add_missing_main_config_sections(config)
            config = self._migrate_observability_config(config)
            config = self._migrate_self_learning_config(config)
        
        return config
    
    def _add_missing_main_config_sections(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add missing main configuration sections."""
        default_sections = {
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
            },
            "agents": {
                "isolation": True,
                "session_timeout": 3600
            },
            "api": {
                "cors_enabled": True,
                "rate_limit": {
                    "enabled": True,
                    "requests_per_minute": 100
                }
            }
        }
        
        for section, default_config in default_sections.items():
            if section not in config:
                config[section] = default_config
                logger.info(f"Added default {section} configuration")
        
        return config
    
    def _migrate_observability_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate observability configuration."""
        if "observability" not in config:
            config["observability"] = {
                "enabled": True,
                "tracing": {
                    "enabled": True,
                    "exporter": "console",
                    "sample_rate": 1.0
                },
                "metrics": {
                    "enabled": True,
                    "exporter": "console",
                    "collection_interval": 10
                },
                "logging": {
                    "level": "INFO",
                    "format": "structured"
                }
            }
            logger.info("Added observability configuration")
        
        return config
    
    def _migrate_self_learning_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate self-learning configuration."""
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
                    "ab_testing": True
                }
            }
            logger.info("Added self-learning configuration")
        
        return config
    
    def migrate_agents_config(self, source_file: Path) -> Dict[str, Any]:
        """Migrate agents configuration."""
        config = self.load_config(source_file)
        
        # Add default agents if not present
        default_agents = {
            "claude": {
                "name": "Claude",
                "type": "assistant",
                "session_timeout": 3600,
                "memory_isolation": True,
                "features": {
                    "memory_access": True,
                    "graph_queries": True,
                    "advanced_rag": True
                }
            },
            "tyra": {
                "name": "Tyra",
                "type": "trading_assistant",
                "session_timeout": 7200,
                "memory_isolation": True,
                "features": {
                    "memory_access": True,
                    "graph_queries": True,
                    "advanced_rag": True,
                    "trading_safety": True
                },
                "safety": {
                    "min_confidence": 0.95,
                    "hallucination_detection": True,
                    "audit_logging": True
                }
            }
        }
        
        if "agents" not in config:
            config["agents"] = default_agents
        else:
            # Add missing agents
            for agent_name, agent_config in default_agents.items():
                if agent_name not in config["agents"]:
                    config["agents"][agent_name] = agent_config
                    logger.info(f"Added default {agent_name} agent configuration")
        
        return config
    
    def validate_config(self, config: Dict[str, Any], config_type: str) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if config_type == "providers":
            issues.extend(self._validate_providers_config(config))
        elif config_type == "main":
            issues.extend(self._validate_main_config(config))
        elif config_type == "agents":
            issues.extend(self._validate_agents_config(config))
        
        return issues
    
    def _validate_providers_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate providers configuration."""
        issues = []
        
        required_types = ["embeddings", "vector_stores", "graph_engines"]
        for provider_type in required_types:
            if provider_type not in config:
                issues.append(f"Missing required provider type: {provider_type}")
                continue
            
            if "providers" not in config[provider_type]:
                issues.append(f"Missing providers section in {provider_type}")
                continue
            
            providers = config[provider_type]["providers"]
            if not providers:
                issues.append(f"No providers configured for {provider_type}")
            
            for provider_name, provider_config in providers.items():
                if "class" not in provider_config:
                    issues.append(f"Missing class for {provider_type}.{provider_name}")
                
                if "config" not in provider_config:
                    issues.append(f"Missing config for {provider_type}.{provider_name}")
        
        return issues
    
    def _validate_main_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate main configuration."""
        issues = []
        
        required_sections = ["memory", "embedding", "vector_store"]
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Validate embedding configuration
        if "embedding" in config:
            embedding_config = config["embedding"]
            if "primary" not in embedding_config:
                issues.append("Missing primary embedding provider")
            if "fallback" not in embedding_config:
                issues.append("Missing fallback embedding provider")
        
        return issues
    
    def _validate_agents_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate agents configuration."""
        issues = []
        
        if "agents" not in config:
            issues.append("Missing agents section")
            return issues
        
        agents = config["agents"]
        for agent_name, agent_config in agents.items():
            if "type" not in agent_config:
                issues.append(f"Missing type for agent {agent_name}")
            
            if "features" not in agent_config:
                issues.append(f"Missing features for agent {agent_name}")
        
        return issues
    
    def run_migration(self, config_files: List[str], target_version: str = "latest", 
                     backup: bool = True, validate: bool = True) -> None:
        """Run migration for specified configuration files."""
        for config_file_name in config_files:
            config_file = self.config_dir / config_file_name
            
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_file}")
                continue
            
            logger.info(f"Migrating {config_file}")
            
            # Create backup if requested
            if backup:
                self.backup_config(config_file)
            
            # Determine migration type
            if "providers" in config_file_name:
                migrated_config = self.migrate_providers_config(config_file, target_version)
                config_type = "providers"
            elif "agents" in config_file_name:
                migrated_config = self.migrate_agents_config(config_file)
                config_type = "agents"
            else:
                migrated_config = self.migrate_main_config(config_file, target_version)
                config_type = "main"
            
            # Validate if requested
            if validate:
                issues = self.validate_config(migrated_config, config_type)
                if issues:
                    logger.warning(f"Validation issues found in {config_file}:")
                    for issue in issues:
                        logger.warning(f"  - {issue}")
                else:
                    logger.info(f"Configuration validation passed for {config_file}")
            
            # Save migrated configuration
            self.save_config(migrated_config, config_file)
            logger.info(f"Migration completed for {config_file}")
    
    def create_missing_configs(self) -> None:
        """Create missing configuration files with defaults."""
        config_templates = {
            "config.yaml": {
                "memory": {"backend": "postgres"},
                "embedding": {"primary": "e5-large", "fallback": "all-minilm"},
                "vector_store": {"primary": "pgvector"},
                "graph": {"engine": "memgraph", "manager": "graphiti"},
                "rag": {"retrieval": {"strategy": "hybrid"}},
                "cache": {"enabled": True},
                "agents": {"isolation": True},
                "api": {"cors_enabled": True}
            },
            "agents.yaml": {
                "agents": {
                    "claude": {
                        "name": "Claude",
                        "type": "assistant",
                        "features": {"memory_access": True}
                    },
                    "tyra": {
                        "name": "Tyra",
                        "type": "trading_assistant",
                        "features": {"memory_access": True, "trading_safety": True}
                    }
                }
            },
            "models.yaml": {
                "models": {
                    "embedding": {
                        "primary": "intfloat/e5-large-v2",
                        "fallback": "sentence-transformers/all-MiniLM-L12-v2"
                    },
                    "reranking": {
                        "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-12-v2"
                    }
                }
            }
        }
        
        for config_name, config_template in config_templates.items():
            config_file = self.config_dir / config_name
            
            if not config_file.exists():
                self.save_config(config_template, config_file)
                logger.info(f"Created missing config file: {config_file}")
    
    def interactive_migration(self) -> None:
        """Run interactive migration with user prompts."""
        print("\n🔧 Configuration Migration Tool")
        print("=" * 40)
        
        # List available config files
        config_files = list(self.config_dir.glob("*.yaml"))
        print(f"\nFound {len(config_files)} configuration files:")
        for i, config_file in enumerate(config_files, 1):
            print(f"  {i}. {config_file.name}")
        
        # Ask user what to migrate
        print("\nSelect files to migrate (comma-separated numbers, or 'all'):")
        selection = input("> ").strip()
        
        if selection.lower() == "all":
            files_to_migrate = [f.name for f in config_files]
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(",")]
                files_to_migrate = [config_files[i].name for i in indices]
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                return
        
        # Ask for options
        backup = input("Create backups? (Y/n): ").strip().lower() != "n"
        validate = input("Validate after migration? (Y/n): ").strip().lower() != "n"
        
        # Run migration
        self.run_migration(files_to_migrate, backup=backup, validate=validate)
        
        # Ask about creating missing configs
        if input("Create missing configuration files? (Y/n): ").strip().lower() != "n":
            self.create_missing_configs()
        
        print("\n✅ Migration completed successfully!")


def main():
    """Main entry point for the migration tool."""
    parser = argparse.ArgumentParser(
        description="Migrate configuration files for Tyra MCP Memory Server"
    )
    parser.add_argument(
        "--config-files",
        nargs="+",
        help="Configuration files to migrate"
    )
    parser.add_argument(
        "--target-version",
        default="latest",
        help="Target version for migration"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backups"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after migration"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive migration"
    )
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create missing configuration files"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    # Create migration tool
    migration_tool = ConfigMigrationTool(project_root=args.project_root)
    
    try:
        if args.interactive:
            migration_tool.interactive_migration()
        elif args.create_missing:
            migration_tool.create_missing_configs()
        elif args.config_files:
            migration_tool.run_migration(
                config_files=args.config_files,
                target_version=args.target_version,
                backup=not args.no_backup,
                validate=not args.no_validate
            )
        else:
            print("No action specified. Use --interactive or --config-files")
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()