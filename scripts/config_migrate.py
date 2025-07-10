#!/usr/bin/env python3
"""
Enhanced Configuration Migration Tool with Version Tracking

This script provides versioned migration support for configuration files,
tracking migration history and supporting rollbacks.
"""

import argparse
import json
import os
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.utils.simple_logger import get_logger
from scripts.migrations.version_manager import VersionManager
from scripts.migrations.config import get_migration_modules, get_latest_version

logger = get_logger(__name__)


class EnhancedConfigMigrationTool:
    """Enhanced tool for migrating configuration files with version tracking."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.project_root / "config" / "backups"
        self.version_manager = VersionManager(self.config_dir)
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def create_backup(self, files: List[Path]) -> Path:
        """Create a timestamped backup of configuration files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for file_path in files:
            if file_path.exists():
                dest = backup_subdir / file_path.name
                shutil.copy2(file_path, dest)
                logger.info(f"Backed up {file_path.name}")
        
        # Create manifest
        self.version_manager.create_backup_manifest(backup_subdir)
        
        return backup_subdir
    
    def get_pending_migrations(self) -> List[Tuple[str, str, type]]:
        """Get list of pending migrations."""
        current_version = self.version_manager.get_current_version()
        migration_modules = get_migration_modules()
        pending = []
        
        for module in migration_modules:
            if self._compare_versions(module.version, current_version) > 0:
                pending.append((module.version, module.description, module))
        
        return pending
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings."""
        v1_parts = [int(x) for x in v1.split('.')]
        v2_parts = [int(x) for x in v2.split('.')]
        
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for i in range(max_len):
            if v1_parts[i] < v2_parts[i]:
                return -1
            elif v1_parts[i] > v2_parts[i]:
                return 1
        
        return 0
    
    def apply_migration(self, config: Dict[str, Any], migration_class: type) -> Dict[str, Any]:
        """Apply a single migration to configuration."""
        try:
            migrated = migration_class.up(config.copy())
            logger.info(f"Applied migration {migration_class.version}: {migration_class.description}")
            return migrated
        except Exception as e:
            logger.error(f"Failed to apply migration {migration_class.version}: {e}")
            raise
    
    def rollback_migration(self, config: Dict[str, Any], migration_class: type) -> Dict[str, Any]:
        """Rollback a single migration."""
        try:
            rolled_back = migration_class.down(config.copy())
            logger.info(f"Rolled back migration {migration_class.version}")
            return rolled_back
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration_class.version}: {e}")
            raise
    
    def migrate_to_version(self, target_version: str = "latest") -> bool:
        """Migrate all configuration files to target version."""
        if target_version == "latest":
            target_version = get_latest_version()
        
        current_version = self.version_manager.get_current_version()
        
        if self._compare_versions(target_version, current_version) == 0:
            logger.info(f"Already at version {target_version}")
            return True
        
        # Get all config files
        config_files = list(self.config_dir.glob("*.yaml"))
        
        if not config_files:
            logger.warning("No configuration files found")
            return False
        
        # Create backup
        logger.info("Creating backup before migration...")
        backup_dir = self.create_backup(config_files)
        logger.info(f"Backup created at: {backup_dir}")
        
        # Get pending migrations
        pending = self.get_pending_migrations()
        
        if not pending:
            logger.info("No pending migrations")
            return True
        
        # Apply migrations to main config file
        main_config_file = self.config_dir / "config.yaml"
        if main_config_file.exists():
            config = self.load_config(main_config_file)
            files_affected = [main_config_file.name]
            
            for version, description, migration_class in pending:
                if self._compare_versions(version, target_version) <= 0:
                    try:
                        config = self.apply_migration(config, migration_class)
                        self.save_config(config, main_config_file)
                        self.version_manager.add_migration(
                            version, description, files_affected, success=True
                        )
                    except Exception as e:
                        logger.error(f"Migration failed: {e}")
                        self.version_manager.add_migration(
                            version, description, files_affected, success=False
                        )
                        return False
        
        logger.info(f"Successfully migrated to version {target_version}")
        return True
    
    def show_migration_status(self) -> None:
        """Show current migration status and history."""
        current_version = self.version_manager.get_current_version()
        latest_version = get_latest_version()
        
        print(f"\n📊 Migration Status")
        print("=" * 50)
        print(f"Current Version: {current_version}")
        print(f"Latest Version:  {latest_version}")
        
        if current_version == latest_version:
            print("Status: ✅ Up to date")
        else:
            print("Status: ⚠️  Updates available")
        
        # Show pending migrations
        pending = self.get_pending_migrations()
        if pending:
            print(f"\n📋 Pending Migrations ({len(pending)}):")
            for version, description, _ in pending:
                print(f"  • {version}: {description}")
        
        # Show migration history
        history = self.version_manager.get_migration_history()
        if history:
            print(f"\n📜 Migration History (last 5):")
            for migration in history[-5:]:
                status = "✅" if migration["success"] else "❌"
                timestamp = migration["timestamp"].split("T")[0]
                print(f"  {status} {migration['version']} - {timestamp}")
    
    def interactive_migration(self) -> None:
        """Run interactive migration with user prompts."""
        print("\n🔧 Configuration Migration Tool")
        print("=" * 50)
        
        self.show_migration_status()
        
        pending = self.get_pending_migrations()
        if not pending:
            print("\nNo migrations to apply.")
            return
        
        print("\nOptions:")
        print("1. Migrate to latest version")
        print("2. Migrate to specific version")
        print("3. Show migration details")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            if input("\nCreate backup before migration? (Y/n): ").strip().lower() != "n":
                self.migrate_to_version("latest")
            else:
                logger.warning("Migration cancelled - backup is recommended")
        
        elif choice == "2":
            print("\nAvailable versions:")
            for version, description, _ in pending:
                print(f"  • {version}: {description}")
            
            target = input("\nEnter target version: ").strip()
            if target:
                self.migrate_to_version(target)
        
        elif choice == "3":
            print("\nMigration Details:")
            for version, description, module in pending:
                print(f"\n{version}: {description}")
                if hasattr(module, "__doc__") and module.__doc__:
                    print(f"  {module.__doc__.strip()}")
        
        else:
            print("Exiting...")


def main():
    """Main entry point for the enhanced migration tool."""
    parser = argparse.ArgumentParser(
        description="Enhanced configuration migration tool with version tracking"
    )
    parser.add_argument(
        "--migrate",
        metavar="VERSION",
        help="Migrate to specific version (or 'latest')"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status and history"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive migration"
    )
    parser.add_argument(
        "--rollback",
        metavar="VERSION",
        help="Rollback to specific version"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backups (not recommended)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    # Create migration tool
    migration_tool = EnhancedConfigMigrationTool(project_root=args.project_root)
    
    try:
        if args.status:
            migration_tool.show_migration_status()
        
        elif args.interactive:
            migration_tool.interactive_migration()
        
        elif args.migrate:
            success = migration_tool.migrate_to_version(args.migrate)
            sys.exit(0 if success else 1)
        
        elif args.rollback:
            print("Rollback functionality coming soon...")
            # TODO: Implement rollback
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()