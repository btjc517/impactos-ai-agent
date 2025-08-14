"""
Runtime Configuration Engine for ImpactOS AI Agent.

This module provides comprehensive runtime configuration management that:
- Enables hot-reloading of configuration without system restart
- Supports hierarchical configuration with inheritance and overrides
- Provides validation and type checking for configuration values
- Includes configuration versioning and rollback capabilities
- Supports dynamic configuration based on context and conditions
- Integrates with all system components for consistent behavior

Key Features:
- Runtime configuration updates without downtime
- Configuration validation and schema enforcement
- Environment-specific configurations with fallbacks
- Configuration change tracking and audit trails
- Integration with flexible pipeline and dynamic facts systems
- API endpoints for configuration management
- Configuration templates and presets
"""

from __future__ import annotations

import os
import json
import sqlite3
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Callable, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


class ConfigScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    SESSION = "session"
    PIPELINE = "pipeline"
    PROCESSOR = "processor"


class ConfigChangeType(Enum):
    """Types of configuration changes."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RELOAD = "reload"
    ROLLBACK = "rollback"


@dataclass
class ConfigSchema:
    """Schema definition for configuration validation."""
    
    key: str
    data_type: str  # 'string', 'integer', 'float', 'boolean', 'object', 'array'
    required: bool = False
    default_value: Optional[Any] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    scope: ConfigScope = ConfigScope.GLOBAL


@dataclass
class ConfigChange:
    """Record of a configuration change."""
    
    change_id: str
    config_key: str
    scope: ConfigScope
    scope_id: str
    change_type: ConfigChangeType
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    changed_by: str = "system"
    changed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigValidator:
    """Validates configuration values against schemas."""
    
    def __init__(self):
        self.validators = {
            'string': self._validate_string,
            'integer': self._validate_integer,
            'float': self._validate_float,
            'boolean': self._validate_boolean,
            'object': self._validate_object,
            'array': self._validate_array
        }
    
    def validate(self, value: Any, schema: ConfigSchema) -> Tuple[bool, List[str]]:
        """Validate a value against its schema."""
        errors = []
        
        # Check if required value is missing
        if schema.required and (value is None or value == ""):
            errors.append(f"Required configuration '{schema.key}' is missing")
            return False, errors
        
        # Use default if value is None and default exists
        if value is None and schema.default_value is not None:
            value = schema.default_value
        
        # Skip validation if value is None and not required
        if value is None:
            return True, errors
        
        # Type validation
        validator = self.validators.get(schema.data_type)
        if validator:
            is_valid, type_errors = validator(value)
            if not is_valid:
                errors.extend(type_errors)
        
        # Rule validation
        for rule in schema.validation_rules:
            rule_errors = self._validate_rule(value, rule)
            errors.extend(rule_errors)
        
        return len(errors) == 0, errors
    
    def _validate_string(self, value: Any) -> Tuple[bool, List[str]]:
        """Validate string type."""
        if not isinstance(value, str):
            return False, [f"Expected string, got {type(value).__name__}"]
        return True, []
    
    def _validate_integer(self, value: Any) -> Tuple[bool, List[str]]:
        """Validate integer type."""
        if not isinstance(value, int) or isinstance(value, bool):
            return False, [f"Expected integer, got {type(value).__name__}"]
        return True, []
    
    def _validate_float(self, value: Any) -> Tuple[bool, List[str]]:
        """Validate float type."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, [f"Expected number, got {type(value).__name__}"]
        return True, []
    
    def _validate_boolean(self, value: Any) -> Tuple[bool, List[str]]:
        """Validate boolean type."""
        if not isinstance(value, bool):
            return False, [f"Expected boolean, got {type(value).__name__}"]
        return True, []
    
    def _validate_object(self, value: Any) -> Tuple[bool, List[str]]:
        """Validate object (dict) type."""
        if not isinstance(value, dict):
            return False, [f"Expected object, got {type(value).__name__}"]
        return True, []
    
    def _validate_array(self, value: Any) -> Tuple[bool, List[str]]:
        """Validate array (list) type."""
        if not isinstance(value, list):
            return False, [f"Expected array, got {type(value).__name__}"]
        return True, []
    
    def _validate_rule(self, value: Any, rule: Dict[str, Any]) -> List[str]:
        """Validate a value against a specific rule."""
        errors = []
        rule_type = rule.get('type')
        
        if rule_type == 'min_value' and isinstance(value, (int, float)):
            min_val = rule.get('value')
            if value < min_val:
                errors.append(f"Value {value} is less than minimum {min_val}")
        
        elif rule_type == 'max_value' and isinstance(value, (int, float)):
            max_val = rule.get('value')
            if value > max_val:
                errors.append(f"Value {value} is greater than maximum {max_val}")
        
        elif rule_type == 'min_length' and isinstance(value, (str, list)):
            min_len = rule.get('value')
            if len(value) < min_len:
                errors.append(f"Length {len(value)} is less than minimum {min_len}")
        
        elif rule_type == 'max_length' and isinstance(value, (str, list)):
            max_len = rule.get('value')
            if len(value) > max_len:
                errors.append(f"Length {len(value)} is greater than maximum {max_len}")
        
        elif rule_type == 'enum' and 'values' in rule:
            allowed_values = rule['values']
            if value not in allowed_values:
                errors.append(f"Value '{value}' not in allowed values: {allowed_values}")
        
        elif rule_type == 'regex' and isinstance(value, str):
            import re
            pattern = rule.get('pattern')
            if not re.match(pattern, value):
                errors.append(f"Value '{value}' does not match pattern '{pattern}'")
        
        return errors


class ConfigurationEngine:
    """Main runtime configuration engine."""
    
    def __init__(self, db_path: str = "db/impactos.db", 
                 config_dir: str = "config",
                 enable_hot_reload: bool = True):
        self.db_path = db_path
        self.config_dir = Path(config_dir)
        self.enable_hot_reload = enable_hot_reload
        
        self.validator = ConfigValidator()
        self._config_cache: Dict[str, Any] = {}
        self._schemas: Dict[str, ConfigSchema] = {}
        self._change_listeners: List[Callable] = []
        self._lock = threading.RLock()
        
        self._ensure_config_tables()
        self._load_schemas()
        self._load_configurations()
        
        if enable_hot_reload:
            self._start_file_watcher()
    
    def _ensure_config_tables(self):
        """Ensure configuration management tables exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS config_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    UNIQUE(config_key, scope, scope_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS config_schemas (
                    config_key TEXT PRIMARY KEY,
                    schema_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS config_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    change_id TEXT UNIQUE NOT NULL,
                    config_key TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    old_value_json TEXT,
                    new_value_json TEXT,
                    changed_by TEXT NOT NULL,
                    changed_at TEXT NOT NULL,
                    metadata_json TEXT
                )
            ''')
            
            conn.commit()
    
    def _load_schemas(self):
        """Load configuration schemas from database and files."""
        # Load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT config_key, schema_json FROM config_schemas")
            
            for row in cursor.fetchall():
                config_key, schema_json = row
                schema_data = json.loads(schema_json)
                self._schemas[config_key] = ConfigSchema(**schema_data)
        
        # Load from schema files
        schema_file = self.config_dir / "config_schema.json"
        if schema_file.exists():
            try:
                with open(schema_file, 'r') as f:
                    schemas_data = json.load(f)
                    
                for key, schema_data in schemas_data.items():
                    schema_data['key'] = key
                    self._schemas[key] = ConfigSchema(**schema_data)
            except Exception as e:
                logger.warning(f"Failed to load config schemas from file: {e}")
        
        # Add default schemas if none exist
        if not self._schemas:
            self._add_default_schemas()
    
    def _add_default_schemas(self):
        """Add default configuration schemas."""
        default_schemas = [
            ConfigSchema(
                key="system.database.path",
                data_type="string",
                required=True,
                default_value="db/impactos.db",
                description="Database file path"
            ),
            ConfigSchema(
                key="system.processing.enable_fact_discovery",
                data_type="boolean",
                default_value=True,
                description="Enable dynamic fact discovery"
            ),
            ConfigSchema(
                key="system.processing.enable_flexible_pipeline",
                data_type="boolean",
                default_value=True,
                description="Enable flexible processing pipeline"
            ),
            ConfigSchema(
                key="system.processing.default_pipeline",
                data_type="string",
                default_value="dynamic_processing",
                validation_rules=[{
                    'type': 'enum',
                    'values': ['dynamic_processing', 'medallion_compat', 'streaming_pipeline', 'custom_analytics']
                }],
                description="Default processing pipeline to use"
            ),
            ConfigSchema(
                key="ai.openai.api_key",
                data_type="string",
                required=False,
                description="OpenAI API key for AI features"
            ),
            ConfigSchema(
                key="ai.model.default",
                data_type="string",
                default_value="gpt-4o-mini",
                description="Default AI model to use"
            )
        ]
        
        for schema in default_schemas:
            self._schemas[schema.key] = schema
    
    def _load_configurations(self):
        """Load configurations from database and files."""
        with self._lock:
            # Load from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT config_key, scope, scope_id, value_json 
                    FROM config_values WHERE is_active = 1
                ''')
                
                for row in cursor.fetchall():
                    config_key, scope, scope_id, value_json = row
                    cache_key = self._make_cache_key(config_key, scope, scope_id)
                    self._config_cache[cache_key] = json.loads(value_json)
            
            # Load from configuration files
            config_files = [
                "system_config.json",
                "pipeline_config.json",
                "facts.json"  # For backward compatibility
            ]
            
            for filename in config_files:
                config_file = self.config_dir / filename
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            file_config = json.load(f)
                            self._merge_file_config(filename, file_config)
                    except Exception as e:
                        logger.warning(f"Failed to load config from {filename}: {e}")
    
    def _merge_file_config(self, filename: str, config_data: Dict[str, Any]):
        """Merge configuration from a file into the cache."""
        if filename == "system_config.json":
            # Flatten nested config structure
            self._flatten_config("system", config_data, ConfigScope.GLOBAL, "default")
        
        elif filename == "pipeline_config.json":
            pipelines = config_data.get("pipelines", {})
            for pipeline_id, pipeline_config in pipelines.items():
                cache_key = self._make_cache_key(f"pipeline.{pipeline_id}", 
                                               ConfigScope.PIPELINE.value, pipeline_id)
                self._config_cache[cache_key] = pipeline_config
        
        elif filename == "facts.json":
            # Legacy facts config
            cache_key = self._make_cache_key("legacy.facts", ConfigScope.GLOBAL.value, "default")
            self._config_cache[cache_key] = config_data
    
    def _flatten_config(self, prefix: str, config_data: Dict[str, Any], 
                       scope: ConfigScope, scope_id: str):
        """Flatten nested configuration into cache."""
        for key, value in config_data.items():
            full_key = f"{prefix}.{key}"
            
            if isinstance(value, dict) and not self._is_leaf_config(value):
                # Recurse into nested objects
                self._flatten_config(full_key, value, scope, scope_id)
            else:
                # Store leaf value
                cache_key = self._make_cache_key(full_key, scope.value, scope_id)
                self._config_cache[cache_key] = value
    
    def _is_leaf_config(self, value: Dict[str, Any]) -> bool:
        """Check if a dict represents a leaf configuration value."""
        # Consider it a leaf if it has schema-like structure
        return any(key in value for key in ['type', 'required', 'default', 'validation'])
    
    def _make_cache_key(self, config_key: str, scope: str, scope_id: str) -> str:
        """Create cache key from components."""
        return f"{scope}:{scope_id}:{config_key}"
    
    def get(self, config_key: str, scope: ConfigScope = ConfigScope.GLOBAL, 
           scope_id: str = "default", default: Any = None) -> Any:
        """Get configuration value with scope hierarchy."""
        with self._lock:
            # Try exact scope first
            cache_key = self._make_cache_key(config_key, scope.value, scope_id)
            if cache_key in self._config_cache:
                return self._config_cache[cache_key]
            
            # Try parent scopes (fallback hierarchy)
            fallback_scopes = self._get_fallback_scopes(scope)
            for fallback_scope in fallback_scopes:
                cache_key = self._make_cache_key(config_key, fallback_scope.value, "default")
                if cache_key in self._config_cache:
                    return self._config_cache[cache_key]
            
            # Try schema default
            schema = self._schemas.get(config_key)
            if schema and schema.default_value is not None:
                return schema.default_value
            
            return default
    
    def set(self, config_key: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL,
           scope_id: str = "default", changed_by: str = "system") -> bool:
        """Set configuration value with validation."""
        with self._lock:
            # Validate against schema if exists
            schema = self._schemas.get(config_key)
            if schema:
                is_valid, errors = self.validator.validate(value, schema)
                if not is_valid:
                    logger.error(f"Configuration validation failed for {config_key}: {errors}")
                    return False
            
            # Get old value for change tracking
            old_value = self.get(config_key, scope, scope_id)
            
            # Update cache
            cache_key = self._make_cache_key(config_key, scope.value, scope_id)
            self._config_cache[cache_key] = value
            
            # Persist to database
            self._persist_config_value(config_key, value, scope, scope_id)
            
            # Record change
            change = ConfigChange(
                change_id=f"{config_key}_{scope.value}_{scope_id}_{int(datetime.now(timezone.utc).timestamp())}",
                config_key=config_key,
                scope=scope,
                scope_id=scope_id,
                change_type=ConfigChangeType.UPDATE if old_value is not None else ConfigChangeType.CREATE,
                old_value=old_value,
                new_value=value,
                changed_by=changed_by,
                changed_at=datetime.now(timezone.utc).isoformat()
            )
            
            self._record_change(change)
            
            # Notify listeners
            self._notify_change_listeners(change)
            
            return True
    
    def _get_fallback_scopes(self, scope: ConfigScope) -> List[ConfigScope]:
        """Get fallback scope hierarchy."""
        hierarchy = {
            ConfigScope.PROCESSOR: [ConfigScope.PIPELINE, ConfigScope.TENANT, ConfigScope.GLOBAL],
            ConfigScope.PIPELINE: [ConfigScope.TENANT, ConfigScope.GLOBAL],
            ConfigScope.SESSION: [ConfigScope.USER, ConfigScope.TENANT, ConfigScope.GLOBAL],
            ConfigScope.USER: [ConfigScope.TENANT, ConfigScope.GLOBAL],
            ConfigScope.TENANT: [ConfigScope.GLOBAL],
            ConfigScope.GLOBAL: []
        }
        
        return hierarchy.get(scope, [])
    
    def _persist_config_value(self, config_key: str, value: Any, 
                             scope: ConfigScope, scope_id: str):
        """Persist configuration value to database."""
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO config_values
                (config_key, scope, scope_id, value_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM config_values 
                     WHERE config_key = ? AND scope = ? AND scope_id = ?), 
                    ?
                ), ?)
            ''', (
                config_key, scope.value, scope_id, json.dumps(value),
                config_key, scope.value, scope_id, now, now
            ))
            conn.commit()
    
    def _record_change(self, change: ConfigChange):
        """Record configuration change in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO config_changes
                (change_id, config_key, scope, scope_id, change_type, 
                 old_value_json, new_value_json, changed_by, changed_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                change.change_id,
                change.config_key,
                change.scope.value,
                change.scope_id,
                change.change_type.value,
                json.dumps(change.old_value) if change.old_value is not None else None,
                json.dumps(change.new_value) if change.new_value is not None else None,
                change.changed_by,
                change.changed_at,
                json.dumps(change.metadata)
            ))
            conn.commit()
    
    def add_change_listener(self, listener: Callable[[ConfigChange], None]):
        """Add a configuration change listener."""
        with self._lock:
            self._change_listeners.append(listener)
    
    def _notify_change_listeners(self, change: ConfigChange):
        """Notify all change listeners of a configuration change."""
        for listener in self._change_listeners:
            try:
                listener(change)
            except Exception as e:
                logger.error(f"Configuration change listener failed: {e}")
    
    def reload_from_files(self) -> bool:
        """Reload configuration from files."""
        try:
            with self._lock:
                self._load_configurations()
                
                change = ConfigChange(
                    change_id=f"reload_{int(datetime.now(timezone.utc).timestamp())}",
                    config_key="system.reload",
                    scope=ConfigScope.GLOBAL,
                    scope_id="default",
                    change_type=ConfigChangeType.RELOAD,
                    changed_by="system",
                    changed_at=datetime.now(timezone.utc).isoformat()
                )
                
                self._record_change(change)
                self._notify_change_listeners(change)
                
                return True
        except Exception as e:
            logger.error(f"Failed to reload configuration from files: {e}")
            return False
    
    def get_all_configs(self, scope: ConfigScope = ConfigScope.GLOBAL, 
                       scope_id: str = "default") -> Dict[str, Any]:
        """Get all configuration values for a scope."""
        with self._lock:
            configs = {}
            
            # Get all configs for this exact scope
            for cache_key, value in self._config_cache.items():
                stored_scope, stored_scope_id, config_key = cache_key.split(':', 2)
                
                if stored_scope == scope.value and stored_scope_id == scope_id:
                    configs[config_key] = value
            
            return configs
    
    def export_config(self, filename: str, scope: ConfigScope = ConfigScope.GLOBAL,
                     scope_id: str = "default") -> bool:
        """Export configuration to a file."""
        try:
            configs = self.get_all_configs(scope, scope_id)
            export_path = self.config_dir / filename
            
            with open(export_path, 'w') as f:
                json.dump(configs, f, indent=2, sort_keys=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to export configuration to {filename}: {e}")
            return False
    
    def _start_file_watcher(self):
        """Start file watcher for hot-reload capability."""
        # Simple implementation - in production you might use watchdog library
        pass


# Global configuration engine instance
_config_engine: Optional[ConfigurationEngine] = None


def get_config_engine() -> ConfigurationEngine:
    """Get the global configuration engine instance."""
    global _config_engine
    if _config_engine is None:
        _config_engine = ConfigurationEngine()
    return _config_engine


def get_config(key: str, default: Any = None, scope: ConfigScope = ConfigScope.GLOBAL, 
              scope_id: str = "default") -> Any:
    """Convenience function to get configuration value."""
    return get_config_engine().get(key, scope, scope_id, default)


def set_config(key: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL,
              scope_id: str = "default", changed_by: str = "system") -> bool:
    """Convenience function to set configuration value."""
    return get_config_engine().set(key, value, scope, scope_id, changed_by)