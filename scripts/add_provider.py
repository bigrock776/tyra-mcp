#!/usr/bin/env python3
"""
Provider Addition Utility Script

This script provides an interactive and automated way to add new providers
to the Tyra MCP Memory Server system, including updating configurations,
creating boilerplate code, and validating the provider.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import inquirer  # For interactive prompts

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.utils.registry import ProviderType


class ProviderAdditionWizard:
    """Interactive wizard for adding new providers."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.src_dir = self.project_root / "src"
        self.providers_config_file = self.config_dir / "providers.yaml"
        
        # Load existing providers configuration
        self.providers_config = self._load_providers_config()
    
    def _load_providers_config(self) -> Dict[str, Any]:
        """Load the existing providers configuration."""
        try:
            with open(self.providers_config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: {self.providers_config_file} not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing providers.yaml: {e}")
            sys.exit(1)
    
    def _save_providers_config(self) -> None:
        """Save the updated providers configuration."""
        try:
            with open(self.providers_config_file, 'w') as f:
                yaml.dump(self.providers_config, f, default_flow_style=False, sort_keys=False)
            print(f"✅ Updated {self.providers_config_file}")
        except Exception as e:
            print(f"Error saving providers.yaml: {e}")
            sys.exit(1)
    
    def get_provider_type_choices(self) -> List[str]:
        """Get available provider types."""
        return [ptype.value for ptype in ProviderType]
    
    def get_interface_info(self, provider_type: str) -> Dict[str, str]:
        """Get interface information for a provider type."""
        interface_map = {
            "embeddings": {
                "interface": "EmbeddingProvider",
                "module": "src.core.interfaces.embeddings",
                "required_methods": ["initialize", "embed_texts", "embed_query", "get_dimension", "supports_gpu", "get_model_name", "health_check"]
            },
            "vector_stores": {
                "interface": "VectorStore",
                "module": "src.core.interfaces.vector_store",
                "required_methods": ["initialize", "store_documents", "search_similar", "hybrid_search", "get_document", "health_check"]
            },
            "graph_engines": {
                "interface": "GraphEngine",
                "module": "src.core.interfaces.graph_engine",
                "required_methods": ["initialize", "create_entity", "create_relationship", "find_entities", "health_check"]
            },
            "rerankers": {
                "interface": "Reranker",
                "module": "src.core.interfaces.reranker",
                "required_methods": ["initialize", "rerank", "score_pair", "get_reranker_type", "health_check"]
            },
            "graph_managers": {
                "interface": "GraphManager",
                "module": "src.core.interfaces.graph_manager",
                "required_methods": ["initialize", "add_episode", "search", "health_check"]
            },
            "graph_clients": {
                "interface": "GraphClient",
                "module": "src.core.interfaces.graph_client",
                "required_methods": ["initialize", "get_entity", "get_relationships", "health_check"]
            }
        }
        return interface_map.get(provider_type, {"interface": "Unknown", "module": "unknown", "required_methods": []})
    
    def get_provider_directory(self, provider_type: str) -> Path:
        """Get the directory path for a provider type."""
        return self.src_dir / "core" / "providers" / provider_type
    
    def run_interactive_wizard(self) -> Dict[str, Any]:
        """Run the interactive provider addition wizard."""
        print("\n🚀 Tyra MCP Memory Server - Provider Addition Wizard")
        print("=" * 60)
        
        # Step 1: Choose provider type
        provider_type = inquirer.list_input(
            "What type of provider would you like to add?",
            choices=self.get_provider_type_choices()
        )
        
        # Step 2: Get provider name
        provider_name = inquirer.text(
            "Enter a name for your provider",
            validate=lambda _, x: len(x) > 0 and x.isidentifier()
        )
        
        # Step 3: Get class name
        default_class_name = f"{provider_name.title()}Provider"
        class_name = inquirer.text(
            f"Enter the class name (default: {default_class_name})",
            default=default_class_name
        )
        
        # Step 4: Get module name
        default_module_name = provider_name.lower().replace("-", "_")
        module_name = inquirer.text(
            f"Enter the module name (default: {default_module_name})",
            default=default_module_name
        )
        
        # Step 5: Create provider configuration
        print(f"\n📝 Configuring {provider_type} provider...")
        config = self._get_provider_config(provider_type, provider_name)
        
        # Step 6: Confirmation
        print(f"\n📋 Provider Summary:")
        print(f"   Type: {provider_type}")
        print(f"   Name: {provider_name}")
        print(f"   Class: {class_name}")
        print(f"   Module: {module_name}")
        print(f"   Config: {config}")
        
        confirm = inquirer.confirm("Would you like to create this provider?", default=True)
        
        if not confirm:
            print("❌ Provider creation cancelled.")
            return {}
        
        return {
            "provider_type": provider_type,
            "provider_name": provider_name,
            "class_name": class_name,
            "module_name": module_name,
            "config": config
        }
    
    def _get_provider_config(self, provider_type: str, provider_name: str) -> Dict[str, Any]:
        """Get provider-specific configuration through interactive prompts."""
        config = {}
        
        # Common configuration options based on provider type
        if provider_type == "embeddings":
            config["model_name"] = inquirer.text(
                "Enter the model name (e.g., 'sentence-transformers/all-MiniLM-L12-v2')",
                default="sentence-transformers/all-MiniLM-L12-v2"
            )
            config["device"] = inquirer.list_input(
                "Select device",
                choices=["auto", "cpu", "cuda", "mps"]
            )
            config["normalize_embeddings"] = inquirer.confirm(
                "Normalize embeddings?",
                default=True
            )
        
        elif provider_type == "vector_stores":
            config["connection_string"] = inquirer.text(
                "Enter connection string (use ${VAR} for env variables)",
                default="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
            )
            config["table_name"] = inquirer.text(
                "Enter table name",
                default="memory_embeddings"
            )
        
        elif provider_type == "graph_engines":
            config["host"] = inquirer.text("Enter host", default="localhost")
            config["port"] = inquirer.text("Enter port", default="7687")
            config["username"] = inquirer.text("Enter username", default="neo4j")
            config["password"] = inquirer.text("Enter password (use ${VAR} for env)", default="${NEO4J_PASSWORD}")
        
        elif provider_type == "rerankers":
            config["model_name"] = inquirer.text(
                "Enter model name",
                default="cross-encoder/ms-marco-MiniLM-L-12-v2"
            )
            config["device"] = inquirer.list_input(
                "Select device",
                choices=["auto", "cpu", "cuda", "mps"]
            )
            config["max_length"] = inquirer.text("Enter max sequence length", default="512")
        
        # Allow adding custom configuration
        add_custom = inquirer.confirm("Add custom configuration parameters?", default=False)
        
        if add_custom:
            while True:
                key = inquirer.text("Enter config key (empty to finish)")
                if not key:
                    break
                value = inquirer.text(f"Enter value for {key}")
                
                # Try to parse as appropriate type
                if value.lower() in ["true", "false"]:
                    config[key] = value.lower() == "true"
                elif value.isdigit():
                    config[key] = int(value)
                elif value.replace(".", "").isdigit():
                    config[key] = float(value)
                else:
                    config[key] = value
        
        return config
    
    def create_provider_boilerplate(self, provider_info: Dict[str, Any]) -> Path:
        """Create boilerplate code for the new provider."""
        provider_type = provider_info["provider_type"]
        provider_name = provider_info["provider_name"]
        class_name = provider_info["class_name"]
        module_name = provider_info["module_name"]
        
        # Get provider directory
        provider_dir = self.get_provider_directory(provider_type)
        provider_dir.mkdir(parents=True, exist_ok=True)
        
        # Create provider file
        provider_file = provider_dir / f"{module_name}.py"
        
        # Get interface info
        interface_info = self.get_interface_info(provider_type)
        
        # Generate boilerplate code
        boilerplate = self._generate_provider_boilerplate(
            provider_type, class_name, interface_info, provider_info["config"]
        )
        
        # Write the file
        try:
            with open(provider_file, 'w') as f:
                f.write(boilerplate)
            print(f"✅ Created provider boilerplate: {provider_file}")
        except Exception as e:
            print(f"❌ Error creating provider file: {e}")
            return None
        
        # Update __init__.py
        self._update_provider_init(provider_dir, module_name, class_name)
        
        return provider_file
    
    def _generate_provider_boilerplate(self, provider_type: str, class_name: str, interface_info: Dict[str, str], config: Dict[str, Any]) -> str:
        """Generate boilerplate code for a provider."""
        interface_name = interface_info["interface"]
        interface_module = interface_info["module"]
        required_methods = interface_info["required_methods"]
        
        # Generate import statements
        imports = f'''"""
{class_name} implementation for {provider_type}.

Auto-generated provider boilerplate. Implement the required methods
according to the {interface_name} interface.
"""

import asyncio
from typing import Any, Dict, List, Optional
from {interface_module} import {interface_name}
from ...utils.logger import get_logger

logger = get_logger(__name__)


class {class_name}({interface_name}):
    """
    {class_name} implementation.
    
    This provider implements the {interface_name} interface.
    """
    
    def __init__(self):
        """Initialize the {class_name}."""
        self.config: Dict[str, Any] = {{}}
        self.initialized: bool = False
        
        # Add your initialization variables here
        # Example configuration from wizard:
        # {config}
'''
        
        # Generate method stubs
        method_stubs = []
        
        for method in required_methods:
            if method == "initialize":
                method_stubs.append(f'''
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the {class_name} with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        try:
            self.config = config
            
            # TODO: Implement initialization logic
            # Example:
            # self.model_name = config.get("model_name", "default-model")
            # self.device = config.get("device", "cpu")
            
            self.initialized = True
            logger.info("{class_name} initialized successfully", config=config)
            
        except Exception as e:
            logger.error("Failed to initialize {class_name}", error=str(e))
            raise
''')
            
            elif method == "health_check":
                method_stubs.append(f'''
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the {class_name}.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # TODO: Implement health check logic
            # Example:
            # if not self.initialized:
            #     return {{"status": "unhealthy", "error": "Not initialized"}}
            
            return {{
                "status": "healthy",
                "provider": "{class_name}",
                "initialized": self.initialized,
                "config": self.config
            }}
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {{
                "status": "unhealthy",
                "error": str(e)
            }}
''')
            
            else:
                # Generate generic method stub
                method_stubs.append(f'''
    async def {method}(self, *args, **kwargs) -> Any:
        """
        {method.replace("_", " ").title()} method implementation.
        
        TODO: Implement this method according to the {interface_name} interface.
        """
        if not self.initialized:
            raise RuntimeError("{class_name} not initialized")
        
        # TODO: Implement {method} logic
        logger.warning("{method} method not implemented yet")
        raise NotImplementedError("{method} method not implemented")
''')
        
        return imports + "\n".join(method_stubs)
    
    def _update_provider_init(self, provider_dir: Path, module_name: str, class_name: str) -> None:
        """Update the provider directory's __init__.py file."""
        init_file = provider_dir / "__init__.py"
        
        # Read existing content
        existing_content = ""
        if init_file.exists():
            try:
                with open(init_file, 'r') as f:
                    existing_content = f.read()
            except Exception as e:
                logger.warning(f"Could not read {init_file}: {e}")
        
        # Add import if not already present
        import_line = f"from .{module_name} import {class_name}"
        
        if import_line not in existing_content:
            if existing_content:
                existing_content += f"\n{import_line}\n"
            else:
                existing_content = f'"""\nProvider implementations for this category.\n"""\n\n{import_line}\n'
        
        # Write updated content
        try:
            with open(init_file, 'w') as f:
                f.write(existing_content)
            print(f"✅ Updated {init_file}")
        except Exception as e:
            print(f"❌ Error updating {init_file}: {e}")
    
    def add_provider_to_config(self, provider_info: Dict[str, Any]) -> None:
        """Add the provider to the providers.yaml configuration."""
        provider_type = provider_info["provider_type"]
        provider_name = provider_info["provider_name"]
        class_name = provider_info["class_name"]
        module_name = provider_info["module_name"]
        config = provider_info["config"]
        
        # Build class path
        class_path = f"src.core.providers.{provider_type}.{module_name}.{class_name}"
        
        # Ensure provider type section exists
        if provider_type not in self.providers_config:
            self.providers_config[provider_type] = {"providers": {}}
        elif "providers" not in self.providers_config[provider_type]:
            self.providers_config[provider_type]["providers"] = {}
        
        # Add provider configuration
        self.providers_config[provider_type]["providers"][provider_name] = {
            "class": class_path,
            "config": config
        }
        
        # Save updated configuration
        self._save_providers_config()
        print(f"✅ Added {provider_name} to {provider_type} providers")
    
    def create_provider_tests(self, provider_info: Dict[str, Any]) -> Path:
        """Create basic test template for the provider."""
        provider_type = provider_info["provider_type"]
        provider_name = provider_info["provider_name"]
        class_name = provider_info["class_name"]
        module_name = provider_info["module_name"]
        
        # Create test file
        test_dir = self.project_root / "tests" / "unit" / "providers" / provider_type
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = test_dir / f"test_{module_name}.py"
        
        # Generate test boilerplate
        test_boilerplate = f'''"""
Tests for {class_name}.

Auto-generated test template. Implement comprehensive tests
for your provider implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.providers.{provider_type}.{module_name} import {class_name}


class Test{class_name}:
    """Test suite for {class_name}."""
    
    @pytest.fixture
    def provider(self):
        """Create a {class_name} instance for testing."""
        return {class_name}()
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {provider_info["config"]}
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider, sample_config):
        """Test provider initialization."""
        await provider.initialize(sample_config)
        
        assert provider.initialized is True
        assert provider.config == sample_config
    
    @pytest.mark.asyncio
    async def test_health_check_before_init(self, provider):
        """Test health check before initialization."""
        health = await provider.health_check()
        
        assert health["status"] == "healthy"  # or "unhealthy" based on your implementation
        assert health["initialized"] is False
    
    @pytest.mark.asyncio
    async def test_health_check_after_init(self, provider, sample_config):
        """Test health check after initialization."""
        await provider.initialize(sample_config)
        health = await provider.health_check()
        
        assert health["status"] == "healthy"
        assert health["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_initialization_with_invalid_config(self, provider):
        """Test initialization with invalid configuration."""
        invalid_config = {{}}
        
        # TODO: Implement based on your validation logic
        # with pytest.raises(ValueError):
        #     await provider.initialize(invalid_config)
    
    # TODO: Add more specific tests for your provider methods
    # Example:
    # @pytest.mark.asyncio
    # async def test_specific_method(self, provider, sample_config):
    #     await provider.initialize(sample_config)
    #     result = await provider.specific_method()
    #     assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        
        # Write test file
        try:
            with open(test_file, 'w') as f:
                f.write(test_boilerplate)
            print(f"✅ Created test template: {test_file}")
        except Exception as e:
            print(f"❌ Error creating test file: {e}")
            return None
        
        return test_file
    
    def validate_provider(self, provider_info: Dict[str, Any]) -> bool:
        """Validate the created provider."""
        provider_type = provider_info["provider_type"]
        provider_name = provider_info["provider_name"]
        
        try:
            # Try to import the provider
            from src.core.utils.registry import provider_registry
            
            # Try to get the provider (this will validate the configuration)
            provider_type_enum = ProviderType(provider_type)
            provider_instance = provider_registry.get_provider(provider_type_enum, provider_name)
            
            print(f"✅ Provider validation successful")
            return True
            
        except Exception as e:
            print(f"❌ Provider validation failed: {e}")
            return False
    
    def run_complete_wizard(self) -> None:
        """Run the complete provider addition wizard."""
        try:
            # Step 1: Interactive wizard
            provider_info = self.run_interactive_wizard()
            
            if not provider_info:
                return
            
            # Step 2: Create boilerplate code
            print("\n🔧 Creating provider boilerplate...")
            provider_file = self.create_provider_boilerplate(provider_info)
            
            if not provider_file:
                return
            
            # Step 3: Add to configuration
            print("\n📝 Adding provider to configuration...")
            self.add_provider_to_config(provider_info)
            
            # Step 4: Create tests
            print("\n🧪 Creating test template...")
            test_file = self.create_provider_tests(provider_info)
            
            # Step 5: Success message
            print("\n🎉 Provider created successfully!")
            print(f"   Provider file: {provider_file}")
            print(f"   Test file: {test_file}")
            print(f"   Configuration: Updated {self.providers_config_file}")
            
            print("\n📋 Next steps:")
            print("1. Implement the required methods in your provider class")
            print("2. Write comprehensive tests for your provider")
            print("3. Run the tests to ensure everything works")
            print("4. Update documentation as needed")
            
            # Optional validation
            if inquirer.confirm("Would you like to validate the provider configuration?", default=True):
                self.validate_provider(provider_info)
            
        except KeyboardInterrupt:
            print("\n❌ Provider creation cancelled by user.")
        except Exception as e:
            print(f"❌ Error during provider creation: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for the provider addition script."""
    parser = argparse.ArgumentParser(
        description="Add new providers to Tyra MCP Memory Server"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive wizard (default)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    # Create wizard instance
    wizard = ProviderAdditionWizard(project_root=args.project_root)
    
    # Run the wizard
    wizard.run_complete_wizard()


if __name__ == "__main__":
    main()