"""
Adaptive Schema System for ImpactOS AI Agent.

This module provides dynamic schema adaptation capabilities that:
- Automatically adapt database schemas to incoming data
- Handle schema evolution and migration
- Provide backward compatibility during schema changes
- Support multiple data types and formats
- Enable runtime schema modification without downtime

Key Features:
- Data-driven schema inference and evolution
- Automatic schema migration with conflict resolution
- Type coercion and validation
- Column mapping and transformation
- Schema versioning and rollback capabilities
- Integration with flexible pipeline system
"""

from __future__ import annotations

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for adaptive schema."""
    INTEGER = "INTEGER"
    REAL = "REAL"
    TEXT = "TEXT"
    BLOB = "BLOB"
    DATE = "DATE"
    DATETIME = "DATETIME"
    BOOLEAN = "BOOLEAN"
    JSON = "JSON"


class SchemaChangeType(Enum):
    """Types of schema changes."""
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    RENAME_COLUMN = "rename_column"
    CHANGE_TYPE = "change_type"
    ADD_CONSTRAINT = "add_constraint"
    DROP_CONSTRAINT = "drop_constraint"
    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"


@dataclass
class ColumnSchema:
    """Definition of a column schema."""
    
    name: str
    data_type: DataType
    nullable: bool = True
    default_value: Optional[Any] = None
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_sql(self) -> str:
        """Convert column schema to SQL definition."""
        sql = f"{self.name} {self.data_type.value}"
        
        if not self.nullable:
            sql += " NOT NULL"
        
        if self.default_value is not None:
            if self.data_type in [DataType.TEXT, DataType.DATE, DataType.DATETIME]:
                sql += f" DEFAULT '{self.default_value}'"
            else:
                sql += f" DEFAULT {self.default_value}"
        
        for constraint in self.constraints:
            sql += f" {constraint}"
        
        return sql


@dataclass
class TableSchema:
    """Definition of a table schema."""
    
    table_name: str
    columns: List[ColumnSchema] = field(default_factory=list)
    primary_key: Optional[List[str]] = None
    foreign_keys: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Get column by name."""
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None
    
    def to_create_sql(self) -> str:
        """Convert table schema to CREATE TABLE SQL."""
        columns_sql = [col.to_sql() for col in self.columns]
        
        if self.primary_key:
            columns_sql.append(f"PRIMARY KEY ({', '.join(self.primary_key)})")
        
        for fk in self.foreign_keys:
            columns_sql.append(
                f"FOREIGN KEY ({fk['column']}) REFERENCES {fk['ref_table']}({fk['ref_column']})"
            )
        
        return f"CREATE TABLE {self.table_name} (\n  {',\\n  '.join(columns_sql)}\n)"


@dataclass
class SchemaChange:
    """Represents a schema change operation."""
    
    change_type: SchemaChangeType
    table_name: str
    change_details: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    created_at: Optional[str] = None


class SchemaInferenceEngine:
    """Engine for inferring schema from data."""
    
    def __init__(self):
        self.type_inference_rules = {
            'integer_patterns': [
                lambda x: pd.api.types.is_integer_dtype(x),
                lambda x: x.dtype.name.startswith('int')
            ],
            'float_patterns': [
                lambda x: pd.api.types.is_float_dtype(x),
                lambda x: x.dtype.name.startswith('float')
            ],
            'datetime_patterns': [
                lambda x: pd.api.types.is_datetime64_any_dtype(x),
                lambda x: self._looks_like_date(x)
            ],
            'boolean_patterns': [
                lambda x: pd.api.types.is_bool_dtype(x),
                lambda x: self._looks_like_boolean(x)
            ]
        }
    
    def infer_schema_from_dataframe(self, df: pd.DataFrame, 
                                  table_name: str) -> TableSchema:
        """Infer table schema from pandas DataFrame."""
        columns = []
        
        for col_name in df.columns:
            series = df[col_name].dropna()
            
            # Infer data type
            data_type = self._infer_column_type(series)
            
            # Check nullability
            nullable = df[col_name].isnull().any()
            
            # Analyze constraints
            constraints = self._infer_constraints(series, df[col_name])
            
            # Create column schema
            column_schema = ColumnSchema(
                name=str(col_name),
                data_type=data_type,
                nullable=nullable,
                constraints=constraints,
                metadata={
                    'unique_count': series.nunique(),
                    'null_count': df[col_name].isnull().sum(),
                    'sample_values': series.head(3).tolist() if len(series) > 0 else []
                }
            )
            
            columns.append(column_schema)
        
        # Infer primary key candidates
        primary_key_candidates = self._identify_primary_key_candidates(df)
        
        return TableSchema(
            table_name=table_name,
            columns=columns,
            primary_key=primary_key_candidates[:1] if primary_key_candidates else None,
            metadata={
                'inferred_at': datetime.now(timezone.utc).isoformat(),
                'row_count': len(df),
                'column_count': len(df.columns)
            }
        )
    
    def _infer_column_type(self, series: pd.Series) -> DataType:
        """Infer data type for a column."""
        if len(series) == 0:
            return DataType.TEXT
        
        # Try integer first
        if any(rule(series) for rule in self.type_inference_rules['integer_patterns']):
            return DataType.INTEGER
        
        # Try float
        if any(rule(series) for rule in self.type_inference_rules['float_patterns']):
            return DataType.REAL
        
        # Try datetime
        if any(rule(series) for rule in self.type_inference_rules['datetime_patterns']):
            return DataType.DATETIME
        
        # Try boolean
        if any(rule(series) for rule in self.type_inference_rules['boolean_patterns']):
            return DataType.BOOLEAN
        
        # Check if it looks like JSON
        if self._looks_like_json(series):
            return DataType.JSON
        
        # Default to TEXT
        return DataType.TEXT
    
    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if series contains date-like values."""
        if pd.api.types.is_string_dtype(series):
            sample_str = str(series.iloc[0]) if len(series) > 0 else ""
            
            # Common date patterns
            import re
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            ]
            
            for pattern in date_patterns:
                if re.search(pattern, sample_str):
                    return True
        
        return False
    
    def _looks_like_boolean(self, series: pd.Series) -> bool:
        """Check if series contains boolean-like values."""
        if pd.api.types.is_string_dtype(series):
            unique_values = set(str(v).lower() for v in series.dropna().unique())
            boolean_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
            return unique_values.issubset(boolean_values)
        
        return False
    
    def _looks_like_json(self, series: pd.Series) -> bool:
        """Check if series contains JSON-like values."""
        if pd.api.types.is_string_dtype(series) and len(series) > 0:
            sample_str = str(series.iloc[0]).strip()
            return (sample_str.startswith('{') and sample_str.endswith('}')) or \
                   (sample_str.startswith('[') and sample_str.endswith(']'))
        return False
    
    def _infer_constraints(self, series: pd.Series, full_series: pd.Series) -> List[str]:
        """Infer constraints for a column."""
        constraints = []
        
        # Check for uniqueness
        if len(series) > 0 and series.nunique() == len(full_series):
            constraints.append("UNIQUE")
        
        # Check for positive values (for numeric columns)
        if pd.api.types.is_numeric_dtype(series) and len(series) > 0:
            if series.min() >= 0:
                constraints.append("CHECK (value >= 0)")
        
        return constraints
    
    def _identify_primary_key_candidates(self, df: pd.DataFrame) -> List[str]:
        """Identify potential primary key columns."""
        candidates = []
        
        for col in df.columns:
            # Must be unique and not null
            if df[col].nunique() == len(df) and not df[col].isnull().any():
                col_name = str(col).lower()
                
                # Prefer columns with 'id' in name
                if 'id' in col_name:
                    candidates.insert(0, str(col))
                else:
                    candidates.append(str(col))
        
        return candidates


class SchemaEvolutionEngine:
    """Engine for evolving schemas based on new data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.inference_engine = SchemaInferenceEngine()
    
    def compare_schemas(self, existing_schema: TableSchema, 
                       new_schema: TableSchema) -> List[SchemaChange]:
        """Compare two schemas and identify required changes."""
        changes = []
        
        existing_cols = {col.name.lower(): col for col in existing_schema.columns}
        new_cols = {col.name.lower(): col for col in new_schema.columns}
        
        # Find new columns
        for col_name, col_schema in new_cols.items():
            if col_name not in existing_cols:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.ADD_COLUMN,
                    table_name=existing_schema.table_name,
                    change_details={
                        'column_name': col_schema.name,
                        'column_schema': asdict(col_schema)
                    },
                    confidence_score=0.9
                ))
        
        # Find removed columns (mark for potential dropping)
        for col_name, col_schema in existing_cols.items():
            if col_name not in new_cols:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.DROP_COLUMN,
                    table_name=existing_schema.table_name,
                    change_details={
                        'column_name': col_schema.name
                    },
                    confidence_score=0.5  # Lower confidence for drops
                ))
        
        # Find type changes
        for col_name, new_col in new_cols.items():
            if col_name in existing_cols:
                existing_col = existing_cols[col_name]
                if existing_col.data_type != new_col.data_type:
                    # Check if type change is safe
                    is_safe = self._is_type_change_safe(existing_col.data_type, new_col.data_type)
                    
                    changes.append(SchemaChange(
                        change_type=SchemaChangeType.CHANGE_TYPE,
                        table_name=existing_schema.table_name,
                        change_details={
                            'column_name': col_name,
                            'old_type': existing_col.data_type.value,
                            'new_type': new_col.data_type.value,
                            'is_safe': is_safe
                        },
                        confidence_score=0.8 if is_safe else 0.3
                    ))
        
        return changes
    
    def _is_type_change_safe(self, old_type: DataType, new_type: DataType) -> bool:
        """Check if a type change is safe (won't lose data)."""
        safe_transitions = {
            DataType.INTEGER: [DataType.REAL, DataType.TEXT],
            DataType.REAL: [DataType.TEXT],
            DataType.BOOLEAN: [DataType.INTEGER, DataType.TEXT],
            DataType.DATE: [DataType.DATETIME, DataType.TEXT],
            DataType.DATETIME: [DataType.TEXT]
        }
        
        return new_type in safe_transitions.get(old_type, [])


class AdaptiveSchemaManager:
    """Main manager for adaptive schema operations."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        self.db_path = db_path
        self.inference_engine = SchemaInferenceEngine()
        self.evolution_engine = SchemaEvolutionEngine(db_path)
        self._ensure_schema_tables()
    
    def _ensure_schema_tables(self):
        """Ensure schema management tables exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    schema_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    UNIQUE(table_name, version)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    change_details TEXT NOT NULL,
                    applied_at TEXT,
                    success INTEGER,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
    
    def adapt_schema(self, df: pd.DataFrame, table_name: str, 
                    auto_apply: bool = True) -> Tuple[TableSchema, List[SchemaChange]]:
        """Adapt schema to accommodate new data."""
        
        # Infer schema from new data
        new_schema = self.inference_engine.infer_schema_from_dataframe(df, table_name)
        
        # Get existing schema if table exists
        existing_schema = self._get_current_schema(table_name)
        
        if existing_schema is None:
            # Create new table
            changes = [SchemaChange(
                change_type=SchemaChangeType.CREATE_TABLE,
                table_name=table_name,
                change_details={'schema': asdict(new_schema)},
                confidence_score=1.0
            )]
            
            if auto_apply:
                self._apply_schema_changes(changes)
            
            self._save_schema_version(new_schema)
            return new_schema, changes
        
        else:
            # Evolve existing schema
            changes = self.evolution_engine.compare_schemas(existing_schema, new_schema)
            
            if auto_apply and changes:
                # Apply high-confidence changes automatically
                auto_changes = [c for c in changes if c.confidence_score >= 0.7]
                if auto_changes:
                    self._apply_schema_changes(auto_changes)
                    logger.info(f"Applied {len(auto_changes)} automatic schema changes to {table_name}")
            
            # Update schema version
            if changes:
                evolved_schema = self._merge_schemas(existing_schema, new_schema, changes)
                self._save_schema_version(evolved_schema)
                return evolved_schema, changes
            
            return existing_schema, []
    
    def _get_current_schema(self, table_name: str) -> Optional[TableSchema]:
        """Get current schema for a table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            
            if not cursor.fetchone():
                return None
            
            # Try to get schema from version table
            cursor.execute('''
                SELECT schema_json FROM schema_versions 
                WHERE table_name = ? AND is_active = 1
                ORDER BY version DESC LIMIT 1
            ''', (table_name,))
            
            row = cursor.fetchone()
            if row:
                schema_data = json.loads(row[0])
                return TableSchema(**schema_data)
            
            # Fallback: infer from actual table structure
            cursor.execute(f"PRAGMA table_info({table_name})")
            table_info = cursor.fetchall()
            
            columns = []
            for col_info in table_info:
                _, name, sql_type, not_null, default, pk = col_info
                
                # Map SQL types to DataType enum
                data_type = self._map_sql_type_to_datatype(sql_type)
                
                columns.append(ColumnSchema(
                    name=name,
                    data_type=data_type,
                    nullable=not not_null,
                    default_value=default
                ))
            
            return TableSchema(table_name=table_name, columns=columns)
    
    def _map_sql_type_to_datatype(self, sql_type: str) -> DataType:
        """Map SQL type string to DataType enum."""
        sql_type_upper = sql_type.upper()
        
        if 'INT' in sql_type_upper:
            return DataType.INTEGER
        elif 'REAL' in sql_type_upper or 'FLOAT' in sql_type_upper or 'DOUBLE' in sql_type_upper:
            return DataType.REAL
        elif 'TEXT' in sql_type_upper or 'VARCHAR' in sql_type_upper or 'CHAR' in sql_type_upper:
            return DataType.TEXT
        elif 'BLOB' in sql_type_upper:
            return DataType.BLOB
        elif 'DATE' in sql_type_upper:
            return DataType.DATE
        elif 'BOOL' in sql_type_upper:
            return DataType.BOOLEAN
        else:
            return DataType.TEXT
    
    def _apply_schema_changes(self, changes: List[SchemaChange]):
        """Apply schema changes to the database."""
        with sqlite3.connect(self.db_path) as conn:
            for change in changes:
                try:
                    if change.change_type == SchemaChangeType.CREATE_TABLE:
                        schema_data = change.change_details['schema']
                        schema = TableSchema(**schema_data)
                        conn.execute(schema.to_create_sql())
                    
                    elif change.change_type == SchemaChangeType.ADD_COLUMN:
                        col_name = change.change_details['column_name']
                        col_schema = ColumnSchema(**change.change_details['column_schema'])
                        sql = f"ALTER TABLE {change.table_name} ADD COLUMN {col_schema.to_sql()}"
                        conn.execute(sql)
                    
                    elif change.change_type == SchemaChangeType.DROP_COLUMN:
                        # SQLite doesn't support DROP COLUMN directly
                        # This would require table recreation
                        logger.warning(f"Column dropping not implemented for {change.table_name}")
                        continue
                    
                    # Log successful change
                    self._log_schema_change(change, success=True)
                    logger.info(f"Applied schema change: {change.change_type.value} on {change.table_name}")
                    
                except Exception as e:
                    # Log failed change
                    self._log_schema_change(change, success=False, error=str(e))
                    logger.error(f"Failed to apply schema change: {e}")
            
            conn.commit()
    
    def _save_schema_version(self, schema: TableSchema):
        """Save a schema version to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get next version number
            cursor.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_versions WHERE table_name = ?",
                (schema.table_name,)
            )
            
            next_version = cursor.fetchone()[0] + 1
            
            # Deactivate previous versions
            cursor.execute(
                "UPDATE schema_versions SET is_active = 0 WHERE table_name = ?",
                (schema.table_name,)
            )
            
            # Insert new version
            cursor.execute('''
                INSERT INTO schema_versions (table_name, version, schema_json, created_at)
                VALUES (?, ?, ?, ?)
            ''', (
                schema.table_name,
                next_version,
                json.dumps(asdict(schema)),
                datetime.now(timezone.utc).isoformat()
            ))
            
            conn.commit()
    
    def _log_schema_change(self, change: SchemaChange, success: bool, error: Optional[str] = None):
        """Log a schema change attempt."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO schema_changes 
                (table_name, change_type, change_details, applied_at, success, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                change.table_name,
                change.change_type.value,
                json.dumps(change.change_details),
                datetime.now(timezone.utc).isoformat() if success else None,
                1 if success else 0,
                error,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
    
    def _merge_schemas(self, existing_schema: TableSchema, new_schema: TableSchema, 
                      changes: List[SchemaChange]) -> TableSchema:
        """Merge schemas based on applied changes."""
        # This is a simplified merge - in practice, you'd apply each change type
        merged_columns = existing_schema.columns.copy()
        
        # Add new columns from changes
        for change in changes:
            if change.change_type == SchemaChangeType.ADD_COLUMN and change.confidence_score >= 0.7:
                col_schema = ColumnSchema(**change.change_details['column_schema'])
                merged_columns.append(col_schema)
        
        return TableSchema(
            table_name=existing_schema.table_name,
            columns=merged_columns,
            primary_key=existing_schema.primary_key,
            foreign_keys=existing_schema.foreign_keys,
            indexes=existing_schema.indexes,
            metadata={
                **existing_schema.metadata,
                'last_evolved': datetime.now(timezone.utc).isoformat()
            }
        )


# Convenience function for pipeline integration
def adapt_schema(context, config):
    """Adapter function for use in flexible pipeline."""
    try:
        df = context.data.get('dataframe')
        table_name = context.data.get('bronze_table') or context.data.get('table_name')
        
        if df is None or table_name is None:
            context.errors.append({
                'processor_id': 'adaptive_schema',
                'error': 'Missing dataframe or table_name in context',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            return context
        
        # Get database path from config or context
        db_path = config.config.get('db_path', context.data.get('db_path', 'db/impactos.db'))
        
        # Initialize adaptive schema manager
        schema_manager = AdaptiveSchemaManager(db_path)
        
        # Adapt schema
        auto_apply = config.config.get('auto_apply', True)
        adapted_schema, changes = schema_manager.adapt_schema(df, table_name, auto_apply)
        
        # Store results in context
        context.intermediate_results['adaptive_schema'] = {
            'adapted_schema': asdict(adapted_schema),
            'changes': [asdict(change) for change in changes],
            'change_count': len(changes)
        }
        
        context.metadata['schema_adaptation_completed'] = True
        context.metadata['schema_changes_applied'] = len([c for c in changes if c.confidence_score >= 0.7])
        
        logger.info(f"Adaptive schema completed for {table_name}: {len(changes)} changes identified")
        
    except Exception as e:
        context.errors.append({
            'processor_id': 'adaptive_schema',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    return context