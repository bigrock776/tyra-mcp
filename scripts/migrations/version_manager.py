#!/usr/bin/env python3
"""
Configuration Version Manager

Manages configuration versions and tracks migration history.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class VersionManager:
    """Manages configuration versions and migration history."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.version_file = config_dir / ".version_history.json"
        self.history = self._load_history()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load version history from file."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "current_version": "0.0.0",
            "migrations": [],
            "created_at": datetime.now().isoformat()
        }
    
    def _save_history(self) -> None:
        """Save version history to file."""
        with open(self.version_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_current_version(self) -> str:
        """Get the current configuration version."""
        return self.history.get("current_version", "0.0.0")
    
    def set_current_version(self, version: str) -> None:
        """Set the current configuration version."""
        self.history["current_version"] = version
        self._save_history()
    
    def add_migration(self, version: str, description: str, 
                     files_affected: List[str], success: bool = True) -> None:
        """Record a migration in history."""
        migration_record = {
            "version": version,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "files_affected": files_affected,
            "success": success
        }
        
        self.history["migrations"].append(migration_record)
        
        if success:
            self.set_current_version(version)
        else:
            self._save_history()
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get the migration history."""
        return self.history.get("migrations", [])
    
    def get_pending_migrations(self, available_versions: List[str]) -> List[str]:
        """Get list of pending migrations based on current version."""
        current = self.get_current_version()
        pending = []
        
        for version in available_versions:
            if self._compare_versions(version, current) > 0:
                pending.append(version)
        
        return sorted(pending, key=lambda v: [int(x) for x in v.split('.')])
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings. Returns: -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
        v1_parts = [int(x) for x in v1.split('.')]
        v2_parts = [int(x) for x in v2.split('.')]
        
        # Pad with zeros if needed
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for i in range(max_len):
            if v1_parts[i] < v2_parts[i]:
                return -1
            elif v1_parts[i] > v2_parts[i]:
                return 1
        
        return 0
    
    def create_backup_manifest(self, backup_dir: Path) -> Dict[str, Any]:
        """Create a manifest for configuration backups."""
        manifest = {
            "version": self.get_current_version(),
            "timestamp": datetime.now().isoformat(),
            "files": []
        }
        
        # List all configuration files
        for config_file in self.config_dir.glob("*.yaml"):
            manifest["files"].append({
                "name": config_file.name,
                "size": config_file.stat().st_size,
                "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
            })
        
        # Save manifest
        manifest_file = backup_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest