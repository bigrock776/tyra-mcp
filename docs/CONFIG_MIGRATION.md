# Configuration Migration Guide

This guide explains how to use the configuration migration tools to manage configuration versions and updates in the Tyra MCP Memory Server.

## Overview

The configuration migration system provides:
- **Version Tracking**: Track configuration versions and migration history
- **Automated Migrations**: Apply configuration updates automatically
- **Backup Management**: Create backups before applying changes
- **Rollback Support**: Revert to previous configurations if needed
- **Validation**: Ensure configurations are valid after migration

## Migration Tools

### 1. Enhanced Migration Tool (`config_migrate.py`)

The primary tool for managing configuration versions with full tracking.

```bash
# Show current migration status
python scripts/config_migrate.py --status

# Migrate to latest version
python scripts/config_migrate.py --migrate latest

# Migrate to specific version
python scripts/config_migrate.py --migrate 1.2.0

# Interactive migration
python scripts/config_migrate.py --interactive
```

### 2. Legacy Migration Tool (`migrate_config.py`)

The original migration tool for backward compatibility.

```bash
# Interactive migration
python scripts/migrate_config.py --interactive

# Migrate specific files
python scripts/migrate_config.py --config-files config.yaml providers.yaml

# Create missing config files
python scripts/migrate_config.py --create-missing
```

### 3. Configuration Validator (`validate_config.py`)

Validate configuration files for correctness.

```bash
# Validate all configs
python scripts/validate_config.py

# Validate specific file
python scripts/validate_config.py --config-file config.yaml

# Check provider references
python scripts/validate_config.py --check-references
```

## Version History

### Version 1.0.0 - Initial Configuration
- Base configuration structure
- Memory, embedding, vector store settings
- Graph and RAG configuration
- Cache settings

### Version 1.1.0 - Observability
- Added comprehensive observability configuration
- Tracing, metrics, and logging settings
- Telemetry optimization features
- Performance monitoring

### Version 1.2.0 - Self-Learning
- Self-learning and analytics configuration
- A/B testing framework
- Performance tracking
- Automatic optimization settings

### Version 1.3.0 - Trading Safety
- Trading safety configuration
- Confidence level definitions
- API safety settings
- Agent-specific safety features

## Migration Process

### 1. Check Current Status

```bash
python scripts/config_migrate.py --status
```

Output:
```
📊 Migration Status
==================================================
Current Version: 1.0.0
Latest Version:  1.3.0
Status: ⚠️  Updates available

📋 Pending Migrations (3):
  • 1.1.0: Add observability and telemetry configuration
  • 1.2.0: Add self-learning, analytics, and A/B testing
  • 1.3.0: Add trading safety and confidence requirements
```

### 2. Review Migration Details

Before migrating, review what changes will be applied:

```bash
python scripts/config_migrate.py --interactive
# Select option 3 to view migration details
```

### 3. Create Backup

Backups are created automatically, but you can also create manual backups:

```bash
python scripts/backup.sh
```

### 4. Apply Migrations

```bash
# Migrate to latest
python scripts/config_migrate.py --migrate latest

# Or use interactive mode
python scripts/config_migrate.py --interactive
```

### 5. Validate Configuration

After migration, validate the configuration:

```bash
python scripts/validate_config.py
```

## Migration Files

Migrations are stored in `scripts/migrations/config/`:

```
scripts/migrations/config/
├── __init__.py
├── 001_initial_config.py
├── 002_add_observability.py
├── 003_add_self_learning.py
└── 004_add_trading_safety.py
```

Each migration file contains:
- `version`: Target version number
- `description`: What the migration does
- `up()`: Apply the migration
- `down()`: Rollback the migration

## Creating New Migrations

To add a new migration:

1. Create a new file in `scripts/migrations/config/`:
```python
# 005_add_new_feature.py
class Migration:
    version = "1.4.0"
    description = "Add new feature configuration"
    
    @staticmethod
    def up(config):
        config["version"] = "1.4.0"
        config["new_feature"] = {
            "enabled": True,
            "settings": {}
        }
        return config
    
    @staticmethod
    def down(config):
        config["version"] = "1.3.0"
        config.pop("new_feature", None)
        return config
```

2. Add the migration to `MIGRATIONS` list in `__init__.py`

3. Test the migration:
```bash
python scripts/config_migrate.py --migrate 1.4.0
```

## Version Tracking

The system tracks migration history in `config/.version_history.json`:

```json
{
  "current_version": "1.3.0",
  "migrations": [
    {
      "version": "1.1.0",
      "description": "Add observability and telemetry configuration",
      "timestamp": "2025-01-10T10:00:00",
      "files_affected": ["config.yaml"],
      "success": true
    }
  ],
  "created_at": "2025-01-10T09:00:00"
}
```

## Backup Structure

Backups are stored in `config/backups/`:

```
config/backups/
├── backup_20250110_100000/
│   ├── config.yaml
│   ├── providers.yaml
│   ├── agents.yaml
│   └── manifest.json
└── config_20250110_090000.backup.yaml
```

The manifest.json contains:
- Version at time of backup
- Timestamp
- List of backed up files

## Troubleshooting

### Migration Fails

1. Check the error message in logs
2. Restore from backup:
```bash
cp config/backups/backup_*/config.yaml config/
```
3. Fix the issue and retry

### Invalid Configuration After Migration

1. Run validation:
```bash
python scripts/validate_config.py
```
2. Fix reported issues
3. Re-run migration if needed

### Missing Environment Variables

1. Check `.env.example` for required variables
2. Update your `.env` file
3. Re-run validation

## Best Practices

1. **Always backup before migrating** - Backups are created automatically
2. **Validate after migration** - Ensure configuration is correct
3. **Test in development first** - Don't migrate production directly
4. **Review migration details** - Understand what changes will be applied
5. **Keep migrations small** - Each migration should do one thing
6. **Document migrations** - Add clear descriptions and comments
7. **Test rollback** - Ensure migrations can be reversed

## Integration with CI/CD

Add migration checks to your CI/CD pipeline:

```yaml
# .github/workflows/config-check.yml
- name: Check configuration version
  run: python scripts/config_migrate.py --status

- name: Validate configuration
  run: python scripts/validate_config.py
```

## Advanced Usage

### Partial Migrations

Apply migrations to specific config files:

```bash
python scripts/migrate_config.py --config-files providers.yaml
```

### Dry Run

See what would change without applying:

```bash
python scripts/config_migrate.py --migrate latest --dry-run
```

### Force Migration

Skip version checks (use with caution):

```bash
python scripts/config_migrate.py --migrate 1.3.0 --force
```

## Summary

The configuration migration system ensures smooth updates as the project evolves. By tracking versions and providing automated migrations, it reduces the risk of configuration errors during updates.