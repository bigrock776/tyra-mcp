#!/usr/bin/env python3
"""
Configuration Validation Tool

This script validates configuration files for the Tyra MCP Memory Server,
ensuring they follow the correct format and contain all required sections.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.utils.simple_logger import get_logger

logger = get_logger(__name__)


class ConfigValidator:
    """Validates configuration files for correctness and completeness."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def load_config(self, config_file: Path) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            self.errors.append(f"Failed to load {config_file}: {e}")
            return None
    
    def validate_providers_config(self, config: Dict[str, Any]) -> bool:
        """Validate providers configuration."""
        valid = True
        
        # Required provider types
        required_types = ["embeddings", "vector_stores", "graph_engines"]
        
        for provider_type in required_types:
            if provider_type not in config:
                self.errors.append(f"Missing required provider type: {provider_type}")
                valid = False
                continue
            
            if not isinstance(config[provider_type], dict):
                self.errors.append(f"Provider type {provider_type} must be a dictionary")
                valid = False
                continue
            
            if "providers" not in config[provider_type]:
                self.errors.append(f"Missing 'providers' section in {provider_type}")
                valid = False
                continue
            
            providers = config[provider_type]["providers"]
            if not isinstance(providers, dict):
                self.errors.append(f"Providers section in {provider_type} must be a dictionary")
                valid = False
                continue
            
            if not providers:
                self.warnings.append(f"No providers configured for {provider_type}")
            
            # Validate each provider
            for provider_name, provider_config in providers.items():
                if not isinstance(provider_config, dict):
                    self.errors.append(f"Provider {provider_type}.{provider_name} must be a dictionary")
                    valid = False
                    continue
                
                if "class" not in provider_config:
                    self.errors.append(f"Missing 'class' for {provider_type}.{provider_name}")
                    valid = False
                
                if "config" not in provider_config:
                    self.errors.append(f"Missing 'config' for {provider_type}.{provider_name}")
                    valid = False
                
                # Validate class path format
                if "class" in provider_config:
                    class_path = provider_config["class"]
                    if not isinstance(class_path, str):
                        self.errors.append(f"Class path for {provider_type}.{provider_name} must be a string")
                        valid = False
                    elif not class_path.startswith("src."):
                        self.warnings.append(f"Class path for {provider_type}.{provider_name} should start with 'src.'")
                    elif class_path.count(".") < 2:
                        self.errors.append(f"Invalid class path format for {provider_type}.{provider_name}")
                        valid = False
        
        # Optional provider types
        optional_types = ["graph_managers", "graph_clients", "rerankers", "cache"]
        for provider_type in optional_types:
            if provider_type in config:
                if not isinstance(config[provider_type], dict):
                    self.errors.append(f"Provider type {provider_type} must be a dictionary")
                    valid = False
                elif "providers" not in config[provider_type]:
                    self.errors.append(f"Missing 'providers' section in {provider_type}")
                    valid = False
        
        return valid
    
    def validate_main_config(self, config: Dict[str, Any]) -> bool:
        """Validate main configuration."""
        valid = True
        
        # Required top-level sections
        required_sections = ["memory", "embedding", "vector_store"]
        
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
                valid = False
        
        # Validate memory section
        if "memory" in config:
            memory_config = config["memory"]
            if "backend" not in memory_config:
                self.errors.append("Missing 'backend' in memory section")
                valid = False
            elif memory_config["backend"] not in ["postgres", "sqlite", "memory"]:
                self.errors.append(f"Invalid memory backend: {memory_config['backend']}")
                valid = False
        
        # Validate embedding section
        if "embedding" in config:
            embedding_config = config["embedding"]
            if "primary" not in embedding_config:
                self.errors.append("Missing 'primary' in embedding section")
                valid = False
            if "fallback" not in embedding_config:
                self.warnings.append("Missing 'fallback' in embedding section")
        
        # Validate vector_store section
        if "vector_store" in config:
            vector_store_config = config["vector_store"]
            if "primary" not in vector_store_config:
                self.errors.append("Missing 'primary' in vector_store section")
                valid = False
        
        # Validate optional sections
        if "rag" in config:
            rag_config = config["rag"]
            if "retrieval" in rag_config:
                retrieval_config = rag_config["retrieval"]
                if "strategy" in retrieval_config:
                    strategy = retrieval_config["strategy"]
                    if strategy not in ["vector", "hybrid", "graph"]:
                        self.warnings.append(f"Unknown retrieval strategy: {strategy}")
        
        if "cache" in config:
            cache_config = config["cache"]
            if "enabled" not in cache_config:
                self.warnings.append("Missing 'enabled' in cache section")
        
        return valid
    
    def validate_agents_config(self, config: Dict[str, Any]) -> bool:
        """Validate agents configuration."""
        valid = True
        
        if "agents" not in config:
            self.errors.append("Missing 'agents' section")
            return False
        
        agents = config["agents"]
        if not isinstance(agents, dict):
            self.errors.append("Agents section must be a dictionary")
            return False
        
        if not agents:
            self.warnings.append("No agents configured")
        
        # Validate each agent
        for agent_name, agent_config in agents.items():
            if not isinstance(agent_config, dict):
                self.errors.append(f"Agent {agent_name} must be a dictionary")
                valid = False
                continue
            
            # Required fields
            if "name" not in agent_config:
                self.errors.append(f"Missing 'name' for agent {agent_name}")
                valid = False
            
            if "type" not in agent_config:
                self.errors.append(f"Missing 'type' for agent {agent_name}")
                valid = False
            elif agent_config["type"] not in ["assistant", "trading_assistant", "admin"]:
                self.warnings.append(f"Unknown agent type for {agent_name}: {agent_config['type']}")
            
            # Validate features
            if "features" not in agent_config:
                self.errors.append(f"Missing 'features' for agent {agent_name}")
                valid = False
            elif not isinstance(agent_config["features"], dict):
                self.errors.append(f"Features for agent {agent_name} must be a dictionary")
                valid = False
            
            # Validate trading assistant specific requirements
            if agent_config.get("type") == "trading_assistant":
                if "safety" not in agent_config:
                    self.warnings.append(f"Trading assistant {agent_name} should have safety configuration")
                else:
                    safety_config = agent_config["safety"]
                    if "min_confidence" not in safety_config:
                        self.errors.append(f"Missing 'min_confidence' in safety config for {agent_name}")
                        valid = False
                    elif safety_config["min_confidence"] < 0.9:
                        self.warnings.append(f"Low min_confidence for trading assistant {agent_name}")
        
        return valid
    
    def validate_models_config(self, config: Dict[str, Any]) -> bool:
        """Validate models configuration."""
        valid = True
        
        if "models" not in config:
            self.errors.append("Missing 'models' section")
            return False
        
        models = config["models"]
        if not isinstance(models, dict):
            self.errors.append("Models section must be a dictionary")
            return False
        
        # Validate embedding models
        if "embedding" in models:
            embedding_models = models["embedding"]
            if not isinstance(embedding_models, dict):
                self.errors.append("Embedding models must be a dictionary")
                valid = False
            elif "primary" not in embedding_models:
                self.errors.append("Missing 'primary' embedding model")
                valid = False
        
        # Validate reranking models
        if "reranking" in models:
            reranking_models = models["reranking"]
            if not isinstance(reranking_models, dict):
                self.errors.append("Reranking models must be a dictionary")
                valid = False
        
        return valid
    
    def validate_observability_config(self, config: Dict[str, Any]) -> bool:
        """Validate observability configuration."""
        valid = True
        
        if "observability" not in config:
            self.warnings.append("Missing observability configuration")
            return True
        
        obs_config = config["observability"]
        if not isinstance(obs_config, dict):
            self.errors.append("Observability configuration must be a dictionary")
            return False
        
        # Validate tracing
        if "tracing" in obs_config:
            tracing_config = obs_config["tracing"]
            if "enabled" not in tracing_config:
                self.warnings.append("Missing 'enabled' in tracing configuration")
            if "exporter" in tracing_config:
                exporter = tracing_config["exporter"]
                if exporter not in ["console", "jaeger", "zipkin", "otlp"]:
                    self.warnings.append(f"Unknown tracing exporter: {exporter}")
        
        # Validate metrics
        if "metrics" in obs_config:
            metrics_config = obs_config["metrics"]
            if "enabled" not in metrics_config:
                self.warnings.append("Missing 'enabled' in metrics configuration")
        
        return valid
    
    def validate_self_learning_config(self, config: Dict[str, Any]) -> bool:
        """Validate self-learning configuration."""
        valid = True
        
        if "self_learning" not in config:
            self.warnings.append("Missing self-learning configuration")
            return True
        
        sl_config = config["self_learning"]
        if not isinstance(sl_config, dict):
            self.errors.append("Self-learning configuration must be a dictionary")
            return False
        
        # Validate intervals
        if "analysis_interval" in sl_config:
            interval = sl_config["analysis_interval"]
            if not isinstance(interval, str):
                self.errors.append("Analysis interval must be a string")
                valid = False
        
        if "improvement_interval" in sl_config:
            interval = sl_config["improvement_interval"]
            if not isinstance(interval, str):
                self.errors.append("Improvement interval must be a string")
                valid = False
        
        # Validate modules
        if "modules" in sl_config:
            modules = sl_config["modules"]
            if not isinstance(modules, dict):
                self.errors.append("Self-learning modules must be a dictionary")
                valid = False
        
        return valid
    
    def validate_environment_variables(self, config: Dict[str, Any]) -> bool:
        """Validate environment variable references."""
        valid = True
        
        # Find all environment variable references
        env_vars = set()
        
        def find_env_vars(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    find_env_vars(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_env_vars(item, f"{path}[{i}]")
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                env_vars.add((env_var, path))
        
        find_env_vars(config)
        
        # Check if .env.example exists and contains the variables
        env_example_file = self.project_root / ".env.example"
        if env_example_file.exists():
            try:
                with open(env_example_file, 'r') as f:
                    env_example_content = f.read()
                
                for env_var, path in env_vars:
                    if env_var not in env_example_content:
                        self.warnings.append(f"Environment variable {env_var} (used in {path}) not found in .env.example")
            except Exception as e:
                self.warnings.append(f"Could not read .env.example: {e}")
        else:
            if env_vars:
                self.warnings.append("No .env.example file found, but environment variables are referenced")
        
        return valid
    
    def validate_config_file(self, config_file: Path) -> bool:
        """Validate a specific configuration file."""
        self.errors.clear()
        self.warnings.clear()
        
        print(f"\n🔍 Validating {config_file.name}...")
        
        # Load configuration
        config = self.load_config(config_file)
        if config is None:
            return False
        
        # Determine validation type based on filename
        valid = True
        
        if "providers" in config_file.name:
            valid &= self.validate_providers_config(config)
        elif "agents" in config_file.name:
            valid &= self.validate_agents_config(config)
        elif "models" in config_file.name:
            valid &= self.validate_models_config(config)
        elif "observability" in config_file.name:
            valid &= self.validate_observability_config(config)
        elif "self_learning" in config_file.name:
            valid &= self.validate_self_learning_config(config)
        elif config_file.name == "config.yaml":
            valid &= self.validate_main_config(config)
        
        # Always validate environment variables
        valid &= self.validate_environment_variables(config)
        
        # Report results
        if self.errors:
            print(f"❌ {len(self.errors)} error(s) found:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"⚠️  {len(self.warnings)} warning(s) found:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ Configuration is valid!")
        elif not self.errors:
            print("✅ Configuration is valid (with warnings)")
        
        return valid
    
    def validate_all_configs(self) -> Tuple[int, int]:
        """Validate all configuration files and return (valid_count, total_count)."""
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        
        if not config_files:
            print("No configuration files found!")
            return 0, 0
        
        valid_count = 0
        total_count = len(config_files)
        
        print(f"Found {total_count} configuration files to validate:")
        
        for config_file in config_files:
            if self.validate_config_file(config_file):
                valid_count += 1
        
        return valid_count, total_count
    
    def check_provider_references(self) -> bool:
        """Check that provider references in main config exist in providers config."""
        valid = True
        
        # Load main config
        main_config_file = self.config_dir / "config.yaml"
        if not main_config_file.exists():
            return True
        
        main_config = self.load_config(main_config_file)
        if not main_config:
            return False
        
        # Load providers config
        providers_config_file = self.config_dir / "providers.yaml"
        if not providers_config_file.exists():
            return True
        
        providers_config = self.load_config(providers_config_file)
        if not providers_config:
            return False
        
        # Check embedding references
        if "embedding" in main_config:
            embedding_config = main_config["embedding"]
            for key in ["primary", "fallback"]:
                if key in embedding_config:
                    provider_name = embedding_config[key]
                    if provider_name not in providers_config.get("embeddings", {}).get("providers", {}):
                        self.errors.append(f"Embedding provider '{provider_name}' referenced in main config not found in providers config")
                        valid = False
        
        # Check vector store references
        if "vector_store" in main_config:
            vector_store_config = main_config["vector_store"]
            if "primary" in vector_store_config:
                provider_name = vector_store_config["primary"]
                if provider_name not in providers_config.get("vector_stores", {}).get("providers", {}):
                    self.errors.append(f"Vector store provider '{provider_name}' referenced in main config not found in providers config")
                    valid = False
        
        # Check reranker references
        if "rag" in main_config and "reranking" in main_config["rag"]:
            reranking_config = main_config["rag"]["reranking"]
            if "provider" in reranking_config:
                provider_name = reranking_config["provider"]
                if provider_name not in providers_config.get("rerankers", {}).get("providers", {}):
                    self.errors.append(f"Reranker provider '{provider_name}' referenced in main config not found in providers config")
                    valid = False
        
        return valid


def main():
    """Main entry point for the validation tool."""
    parser = argparse.ArgumentParser(
        description="Validate configuration files for Tyra MCP Memory Server"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Specific configuration file to validate"
    )
    parser.add_argument(
        "--check-references",
        action="store_true",
        help="Check that provider references exist"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = ConfigValidator(project_root=args.project_root)
    
    try:
        if args.config_file:
            # Validate specific file
            config_file = args.config_file
            if not config_file.is_absolute():
                config_file = validator.config_dir / config_file
            
            if not config_file.exists():
                print(f"❌ Configuration file not found: {config_file}")
                sys.exit(1)
            
            valid = validator.validate_config_file(config_file)
            
            if args.check_references:
                valid &= validator.check_provider_references()
            
            sys.exit(0 if valid else 1)
        else:
            # Validate all configuration files
            valid_count, total_count = validator.validate_all_configs()
            
            if args.check_references:
                print(f"\n🔗 Checking provider references...")
                ref_valid = validator.check_provider_references()
                if not ref_valid:
                    print("❌ Provider reference errors found:")
                    for error in validator.errors:
                        print(f"   • {error}")
                else:
                    print("✅ All provider references are valid")
            
            # Summary
            print(f"\n📊 Validation Summary:")
            print(f"   Valid: {valid_count}/{total_count}")
            print(f"   Status: {'✅ PASS' if valid_count == total_count else '❌ FAIL'}")
            
            sys.exit(0 if valid_count == total_count else 1)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()