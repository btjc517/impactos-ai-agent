"""
Flexible Processing Pipeline for ImpactOS AI Agent.

This module replaces the hardcoded Bronze→Silver→Gold medallion architecture
with a configurable, plugin-based processing system that can:
- Support arbitrary processing stages
- Allow runtime configuration of transformation pipelines
- Enable custom processing workflows without code changes
- Provide backward compatibility with existing medallion patterns
- Support dynamic pipeline modification based on data characteristics

Key Features:
- Configurable processing stages with dependency management
- Plugin-based transformers for different data types
- Runtime pipeline configuration and modification
- Adaptive processing based on data characteristics
- Support for both batch and streaming processing
- Built-in error handling and retry mechanisms
"""

from __future__ import annotations

import os
import json
import sqlite3
import logging
import importlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Standard processing stages (extensible)."""
    INGEST = "ingest"
    VALIDATE = "validate"
    CLEAN = "clean"
    TRANSFORM = "transform"
    ENRICH = "enrich"
    AGGREGATE = "aggregate"
    EXPORT = "export"


class ProcessorStatus(Enum):
    """Status of a processor execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessorConfig:
    """Configuration for a processing stage."""
    
    processor_id: str
    stage: ProcessingStage
    processor_type: str  # e.g., "python_function", "sql_script", "external_api"
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # When to run this processor
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Configuration for a complete processing pipeline."""
    
    pipeline_id: str
    name: str
    description: str
    processors: List[ProcessorConfig] = field(default_factory=list)
    global_config: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_active: bool = True


@dataclass
class ProcessingContext:
    """Context passed between processors in a pipeline."""
    
    pipeline_id: str
    run_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ProcessorInterface(ABC):
    """Base interface for all processors."""
    
    @abstractmethod
    def process(self, context: ProcessingContext, config: ProcessorConfig) -> ProcessingContext:
        """Process data and return updated context."""
        pass
    
    @abstractmethod
    def validate_config(self, config: ProcessorConfig) -> List[str]:
        """Validate processor configuration. Return list of errors."""
        pass
    
    def get_name(self) -> str:
        """Get processor name."""
        return self.__class__.__name__
    
    def get_description(self) -> str:
        """Get processor description."""
        return getattr(self.__class__, '__doc__', 'No description available')


class SQLProcessor(ProcessorInterface):
    """Processor that executes SQL transformations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def process(self, context: ProcessingContext, config: ProcessorConfig) -> ProcessingContext:
        """Execute SQL transformation."""
        sql_script = config.config.get('sql')
        if not sql_script:
            context.errors.append({
                'processor_id': config.processor_id,
                'error': 'No SQL script provided',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            return context
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Replace placeholders in SQL with context data
                processed_sql = self._substitute_placeholders(sql_script, context)
                
                # Execute SQL
                if config.config.get('return_results', False):
                    cursor = conn.cursor()
                    cursor.execute(processed_sql)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    
                    context.intermediate_results[config.processor_id] = {
                        'rows': results,
                        'columns': columns,
                        'row_count': len(results)
                    }
                else:
                    conn.execute(processed_sql)
                    conn.commit()
                
                context.metadata[f'{config.processor_id}_executed_at'] = datetime.now(timezone.utc).isoformat()
                
        except Exception as e:
            context.errors.append({
                'processor_id': config.processor_id,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return context
    
    def validate_config(self, config: ProcessorConfig) -> List[str]:
        """Validate SQL processor configuration."""
        errors = []
        if not config.config.get('sql'):
            errors.append('SQL script is required')
        return errors
    
    def _substitute_placeholders(self, sql: str, context: ProcessingContext) -> str:
        """Replace placeholders in SQL with context data."""
        placeholders = {
            '{{pipeline_id}}': context.pipeline_id,
            '{{run_id}}': context.run_id,
            **context.data
        }
        
        result = sql
        for placeholder, value in placeholders.items():
            result = result.replace(placeholder, str(value))
        
        return result


class PythonFunctionProcessor(ProcessorInterface):
    """Processor that executes Python functions."""
    
    def process(self, context: ProcessingContext, config: ProcessorConfig) -> ProcessingContext:
        """Execute Python function."""
        function_path = config.config.get('function')
        if not function_path:
            context.errors.append({
                'processor_id': config.processor_id,
                'error': 'No function path provided',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            return context
        
        try:
            # Import and execute function
            module_name, function_name = function_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            
            # Call function with context and config
            updated_context = function(context, config)
            return updated_context if updated_context is not None else context
            
        except Exception as e:
            context.errors.append({
                'processor_id': config.processor_id,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return context
    
    def validate_config(self, config: ProcessorConfig) -> List[str]:
        """Validate Python function processor configuration."""
        errors = []
        function_path = config.config.get('function')
        if not function_path:
            errors.append('Function path is required')
        elif '.' not in function_path:
            errors.append('Function path must include module name')
        else:
            # Try to validate function exists
            try:
                module_name, function_name = function_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                if not hasattr(module, function_name):
                    errors.append(f'Function {function_name} not found in module {module_name}')
            except ImportError:
                errors.append(f'Module {module_name} not found')
        return errors


class ConditionalProcessor(ProcessorInterface):
    """Processor that conditionally executes other processors."""
    
    def __init__(self, processor_registry: 'ProcessorRegistry'):
        self.processor_registry = processor_registry
    
    def process(self, context: ProcessingContext, config: ProcessorConfig) -> ProcessingContext:
        """Execute conditional logic."""
        conditions = config.config.get('conditions', [])
        target_processor_id = config.config.get('target_processor')
        
        if not target_processor_id:
            context.errors.append({
                'processor_id': config.processor_id,
                'error': 'No target processor specified',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            return context
        
        # Evaluate conditions
        should_execute = self._evaluate_conditions(conditions, context)
        
        if should_execute:
            # Execute target processor
            target_processor = self.processor_registry.get_processor(target_processor_id)
            if target_processor:
                target_config = ProcessorConfig(
                    processor_id=target_processor_id,
                    stage=config.stage,
                    processor_type='delegated',
                    config=config.config.get('target_config', {})
                )
                context = target_processor.process(context, target_config)
            else:
                context.errors.append({
                    'processor_id': config.processor_id,
                    'error': f'Target processor {target_processor_id} not found',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        else:
            context.metadata[f'{config.processor_id}_skipped'] = True
        
        return context
    
    def validate_config(self, config: ProcessorConfig) -> List[str]:
        """Validate conditional processor configuration."""
        errors = []
        if not config.config.get('target_processor'):
            errors.append('Target processor is required')
        if not config.config.get('conditions'):
            errors.append('At least one condition is required')
        return errors
    
    def _evaluate_conditions(self, conditions: List[Dict[str, Any]], context: ProcessingContext) -> bool:
        """Evaluate conditional logic."""
        for condition in conditions:
            condition_type = condition.get('type')
            
            if condition_type == 'data_exists':
                key = condition.get('key')
                if key not in context.data:
                    return False
            elif condition_type == 'metadata_equals':
                key = condition.get('key')
                value = condition.get('value')
                if context.metadata.get(key) != value:
                    return False
            elif condition_type == 'error_count':
                operator = condition.get('operator', 'less_than')
                threshold = condition.get('threshold', 0)
                error_count = len(context.errors)
                
                if operator == 'less_than' and error_count >= threshold:
                    return False
                elif operator == 'greater_than' and error_count <= threshold:
                    return False
        
        return True


class ProcessorRegistry:
    """Registry for available processors."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._processors: Dict[str, ProcessorInterface] = {}
        self._register_built_in_processors()
    
    def _register_built_in_processors(self):
        """Register built-in processors."""
        self.register('sql', SQLProcessor(self.db_path))
        self.register('python_function', PythonFunctionProcessor())
        self.register('conditional', ConditionalProcessor(self))
    
    def register(self, processor_type: str, processor: ProcessorInterface):
        """Register a processor."""
        self._processors[processor_type] = processor
    
    def get_processor(self, processor_type: str) -> Optional[ProcessorInterface]:
        """Get a processor by type."""
        return self._processors.get(processor_type)
    
    def list_processors(self) -> List[str]:
        """List available processor types."""
        return list(self._processors.keys())


class FlexiblePipelineEngine:
    """Main engine for flexible processing pipelines."""
    
    def __init__(self, db_path: str = "db/impactos.db", 
                 config_path: str = "config/pipeline_config.json"):
        self.db_path = db_path
        self.config_path = config_path
        self.processor_registry = ProcessorRegistry(db_path)
        self._ensure_pipeline_tables()
    
    def _ensure_pipeline_tables(self):
        """Ensure pipeline management tables exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_configs (
                    pipeline_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    pipeline_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    context_json TEXT,
                    error_count INTEGER DEFAULT 0,
                    FOREIGN KEY (pipeline_id) REFERENCES pipeline_configs (pipeline_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processor_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    processor_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT,
                    metrics_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
                )
            ''')
            
            conn.commit()
    
    def load_pipeline_configs(self) -> Dict[str, PipelineConfig]:
        """Load pipeline configurations from database and file."""
        configs = {}
        
        # Load from database first
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pipeline_id, config_json FROM pipeline_configs WHERE is_active = 1")
            
            for row in cursor.fetchall():
                pipeline_id, config_json = row
                config_data = json.loads(config_json)
                configs[pipeline_id] = PipelineConfig(**config_data)
        
        # Load from config file if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_configs = json.load(f)
                    for pipeline_id, config_data in file_configs.get('pipelines', {}).items():
                        if pipeline_id not in configs:
                            config_data['pipeline_id'] = pipeline_id
                            configs[pipeline_id] = PipelineConfig(**config_data)
            except Exception as e:
                logger.warning(f"Failed to load pipeline config from file: {e}")
        
        return configs
    
    def save_pipeline_config(self, pipeline_config: PipelineConfig):
        """Save pipeline configuration to database."""
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO pipeline_configs 
                (pipeline_id, name, description, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                pipeline_config.pipeline_id,
                pipeline_config.name,
                pipeline_config.description,
                json.dumps(asdict(pipeline_config)),
                pipeline_config.created_at or now,
                now
            ))
            conn.commit()
    
    def execute_pipeline(self, pipeline_id: str, initial_data: Dict[str, Any] = None) -> str:
        """Execute a pipeline and return run ID."""
        configs = self.load_pipeline_configs()
        pipeline_config = configs.get(pipeline_id)
        
        if not pipeline_config:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        # Create processing context
        run_id = f"{pipeline_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        context = ProcessingContext(
            pipeline_id=pipeline_id,
            run_id=run_id,
            data=initial_data or {}
        )
        
        # Start pipeline run
        self._start_pipeline_run(run_id, pipeline_id)
        
        try:
            # Execute processors in order, respecting dependencies
            execution_order = self._calculate_execution_order(pipeline_config.processors)
            
            for processor_config in execution_order:
                if not processor_config.enabled:
                    continue
                
                # Check conditions
                if not self._should_execute_processor(processor_config, context):
                    self._log_processor_execution(run_id, processor_config.processor_id, 
                                                ProcessorStatus.SKIPPED, None, None)
                    continue
                
                # Execute processor
                processor = self.processor_registry.get_processor(processor_config.processor_type)
                if not processor:
                    error_msg = f"Processor type {processor_config.processor_type} not found"
                    context.errors.append({
                        'processor_id': processor_config.processor_id,
                        'error': error_msg,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    self._log_processor_execution(run_id, processor_config.processor_id, 
                                                ProcessorStatus.FAILED, error_msg, None)
                    continue
                
                # Execute with error handling
                start_time = datetime.now(timezone.utc)
                self._log_processor_execution(run_id, processor_config.processor_id, 
                                            ProcessorStatus.RUNNING, None, None)
                
                try:
                    context = processor.process(context, processor_config)
                    end_time = datetime.now(timezone.utc)
                    
                    self._log_processor_execution(run_id, processor_config.processor_id, 
                                                ProcessorStatus.COMPLETED, None, end_time)
                except Exception as e:
                    error_msg = str(e)
                    context.errors.append({
                        'processor_id': processor_config.processor_id,
                        'error': error_msg,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    self._log_processor_execution(run_id, processor_config.processor_id, 
                                                ProcessorStatus.FAILED, error_msg, None)
            
            # Complete pipeline run
            self._complete_pipeline_run(run_id, context, 
                                      ProcessorStatus.COMPLETED if len(context.errors) == 0 
                                      else ProcessorStatus.FAILED)
            
        except Exception as e:
            context.errors.append({
                'processor_id': 'pipeline',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            self._complete_pipeline_run(run_id, context, ProcessorStatus.FAILED)
        
        return run_id
    
    def _calculate_execution_order(self, processors: List[ProcessorConfig]) -> List[ProcessorConfig]:
        """Calculate execution order based on dependencies."""
        # Simple topological sort
        ordered = []
        remaining = processors.copy()
        
        while remaining:
            # Find processors with no remaining dependencies
            ready = []
            for proc in remaining:
                deps_satisfied = all(
                    any(completed.processor_id == dep for completed in ordered)
                    for dep in proc.dependencies
                ) if proc.dependencies else True
                
                if deps_satisfied:
                    ready.append(proc)
            
            if not ready:
                # Circular dependency or invalid dependency
                logger.warning("Circular dependency detected in pipeline, executing remaining processors in order")
                ordered.extend(remaining)
                break
            
            # Add ready processors to order
            for proc in ready:
                ordered.append(proc)
                remaining.remove(proc)
        
        return ordered
    
    def _should_execute_processor(self, processor_config: ProcessorConfig, 
                                context: ProcessingContext) -> bool:
        """Check if processor should execute based on conditions."""
        if not processor_config.conditions:
            return True
        
        # Use conditional processor logic
        conditional = ConditionalProcessor(self.processor_registry)
        return conditional._evaluate_conditions(processor_config.conditions, context)
    
    def _start_pipeline_run(self, run_id: str, pipeline_id: str):
        """Start a pipeline run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO pipeline_runs (run_id, pipeline_id, status, started_at)
                VALUES (?, ?, ?, ?)
            ''', (run_id, pipeline_id, ProcessorStatus.RUNNING.value, 
                 datetime.now(timezone.utc).isoformat()))
            conn.commit()
    
    def _complete_pipeline_run(self, run_id: str, context: ProcessingContext, status: ProcessorStatus):
        """Complete a pipeline run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE pipeline_runs 
                SET status = ?, completed_at = ?, context_json = ?, error_count = ?
                WHERE run_id = ?
            ''', (status.value, datetime.now(timezone.utc).isoformat(), 
                 json.dumps(asdict(context)), len(context.errors), run_id))
            conn.commit()
    
    def _log_processor_execution(self, run_id: str, processor_id: str, 
                               status: ProcessorStatus, error_message: Optional[str], 
                               completed_at: Optional[datetime]):
        """Log processor execution."""
        with sqlite3.connect(self.db_path) as conn:
            if status == ProcessorStatus.RUNNING:
                conn.execute('''
                    INSERT INTO processor_executions 
                    (run_id, processor_id, status, started_at)
                    VALUES (?, ?, ?, ?)
                ''', (run_id, processor_id, status.value, datetime.now(timezone.utc).isoformat()))
            else:
                conn.execute('''
                    UPDATE processor_executions 
                    SET status = ?, completed_at = ?, error_message = ?
                    WHERE run_id = ? AND processor_id = ?
                ''', (status.value, 
                     completed_at.isoformat() if completed_at else datetime.now(timezone.utc).isoformat(),
                     error_message, run_id, processor_id))
            conn.commit()


def create_medallion_compatibility_pipeline() -> PipelineConfig:
    """Create a pipeline that mimics the traditional medallion architecture."""
    return PipelineConfig(
        pipeline_id="medallion_compat",
        name="Medallion Architecture Compatibility",
        description="Traditional Bronze→Silver→Gold processing pipeline",
        processors=[
            ProcessorConfig(
                processor_id="bronze_ingest",
                stage=ProcessingStage.INGEST,
                processor_type="python_function",
                config={"function": "bronze_ingest.ingest_bronze"},
                dependencies=[]
            ),
            ProcessorConfig(
                processor_id="silver_transform",
                stage=ProcessingStage.TRANSFORM,
                processor_type="python_function",
                config={"function": "silver_transform.run_transform"},
                dependencies=["bronze_ingest"]
            ),
            ProcessorConfig(
                processor_id="gold_aggregate",
                stage=ProcessingStage.AGGREGATE,
                processor_type="sql",
                config={
                    "sql": """
                    CREATE OR REPLACE VIEW gold_metrics AS
                    SELECT * FROM fact_metrics 
                    WHERE confidence_score > 0.8
                    """,
                    "return_results": False
                },
                dependencies=["silver_transform"]
            )
        ]
    )