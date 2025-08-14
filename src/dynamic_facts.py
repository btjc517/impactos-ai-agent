"""
Dynamic Fact Discovery System for ImpactOS AI Agent.

This module replaces the static facts.json approach with a dynamic system that:
- Learns fact patterns from incoming data
- Adapts fact definitions based on data characteristics
- Stores learned facts in the database with versioning
- Provides runtime fact discovery and validation
- Maintains backward compatibility with existing FactConfig interface

Key Features:
- Data-driven fact discovery using ML pattern recognition
- Adaptive schema inference and evolution
- Confidence-based fact validation
- Learning from successful/failed extractions
- Runtime configuration without code changes
"""

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FactDefinition:
    """Dynamic fact definition that evolves with data."""
    
    fact_key: str
    description: str
    required_roles: List[str] = field(default_factory=list)
    optional_roles: List[str] = field(default_factory=list)
    role_hints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_unit: Optional[str] = None
    base_currency: Optional[str] = None
    confidence_score: float = 0.0
    sample_count: int = 0
    last_updated: Optional[str] = None
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy facts.json format for backward compatibility."""
        return {
            "required": self.required_roles,
            "optional": self.optional_roles,
            "role_hints": self.role_hints,
            "description": self.description,
            "default_unit": self.default_unit,
            "base_currency": self.base_currency
        }
    
    @classmethod
    def from_legacy_format(cls, fact_key: str, legacy_data: Dict[str, Any]) -> 'FactDefinition':
        """Create from legacy facts.json format."""
        return cls(
            fact_key=fact_key,
            description=legacy_data.get("description", ""),
            required_roles=legacy_data.get("required", []),
            optional_roles=legacy_data.get("optional", []),
            role_hints=legacy_data.get("role_hints", {}),
            default_unit=legacy_data.get("default_unit"),
            base_currency=legacy_data.get("base_currency"),
            confidence_score=0.8,  # Legacy facts start with high confidence
            sample_count=1
        )


class DataPatternAnalyzer:
    """Analyzes data patterns to discover and evolve fact definitions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_confidence_threshold = self.config.get('min_fact_confidence', 0.6)
        self.min_sample_count = self.config.get('min_sample_count', 3)
        
    def analyze_dataframe(self, df: pd.DataFrame, source_context: Dict[str, Any]) -> List[FactDefinition]:
        """Analyze a DataFrame to discover potential fact patterns."""
        discovered_facts = []
        
        # Analyze column patterns and relationships
        column_analysis = self._analyze_columns(df)
        
        # Look for common fact patterns
        fact_patterns = self._identify_fact_patterns(column_analysis, source_context)
        
        # Generate fact definitions
        for pattern in fact_patterns:
            fact_def = self._create_fact_from_pattern(pattern, column_analysis)
            if fact_def.confidence_score >= self.min_confidence_threshold:
                discovered_facts.append(fact_def)
                
        return discovered_facts
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze each column to understand its characteristics."""
        analysis = {}
        
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            analysis[col] = {
                'dtype': str(series.dtype),
                'sample_values': series.head(5).tolist(),
                'unique_count': series.nunique(),
                'null_count': df[col].isnull().sum(),
                'is_numeric': pd.api.types.is_numeric_dtype(series),
                'is_date_like': self._is_date_like(series),
                'is_currency_like': self._is_currency_like(series),
                'is_unit_like': self._is_unit_like(series),
                'common_patterns': self._extract_patterns(series),
                'value_distribution': self._analyze_value_distribution(series)
            }
            
        return analysis
    
    def _is_date_like(self, series: pd.Series) -> bool:
        """Check if a series contains date-like values."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
            
        # Check string patterns for dates
        sample_str = str(series.iloc[0]) if len(series) > 0 else ""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        import re
        for pattern in date_patterns:
            if re.search(pattern, sample_str):
                return True
                
        return False
    
    def _is_currency_like(self, series: pd.Series) -> bool:
        """Check if a series contains currency values."""
        if not pd.api.types.is_numeric_dtype(series):
            # Check for currency symbols in string values
            sample_str = str(series.iloc[0]) if len(series) > 0 else ""
            currency_symbols = ['£', '$', '€', 'GBP', 'USD', 'EUR']
            return any(symbol in sample_str for symbol in currency_symbols)
        return False
    
    def _is_unit_like(self, series: pd.Series) -> bool:
        """Check if a series contains unit values."""
        if pd.api.types.is_string_dtype(series):
            sample_str = str(series.iloc[0]) if len(series) > 0 else ""
            unit_patterns = ['hours', 'hrs', 'h', 'kg', 'tonnes', 't', 'kwh', 'mwh', 'minutes', 'mins']
            return any(unit in sample_str.lower() for unit in unit_patterns)
        return False
    
    def _extract_patterns(self, series: pd.Series) -> List[str]:
        """Extract common patterns from a series."""
        patterns = []
        
        if pd.api.types.is_string_dtype(series):
            # Look for common string patterns
            values = series.astype(str).str.lower()
            common_words = values.value_counts().head(5).index.tolist()
            patterns.extend(common_words)
            
        return patterns
    
    def _analyze_value_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of values in a series."""
        if pd.api.types.is_numeric_dtype(series):
            return {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'std': float(series.std()) if len(series) > 1 else 0.0
            }
        else:
            return {
                'most_common': series.value_counts().head(3).to_dict()
            }
    
    def _identify_fact_patterns(self, column_analysis: Dict[str, Dict[str, Any]], 
                              source_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential fact patterns from column analysis."""
        patterns = []
        
        # Look for value + date combinations (common fact pattern)
        value_cols = [col for col, info in column_analysis.items() 
                     if info['is_numeric'] and not info['is_date_like']]
        date_cols = [col for col, info in column_analysis.items() if info['is_date_like']]
        
        if value_cols and date_cols:
            for value_col in value_cols:
                pattern = {
                    'pattern_type': 'value_date_fact',
                    'value_column': value_col,
                    'date_columns': date_cols,
                    'confidence': 0.8,
                    'context': source_context
                }
                patterns.append(pattern)
        
        # Look for categorical patterns
        categorical_cols = [col for col, info in column_analysis.items() 
                           if not info['is_numeric'] and not info['is_date_like'] 
                           and info['unique_count'] < len(column_analysis) * 0.5]
        
        if categorical_cols:
            pattern = {
                'pattern_type': 'categorical_fact',
                'categorical_columns': categorical_cols,
                'confidence': 0.6,
                'context': source_context
            }
            patterns.append(pattern)
            
        return patterns
    
    def _create_fact_from_pattern(self, pattern: Dict[str, Any], 
                                 column_analysis: Dict[str, Dict[str, Any]]) -> FactDefinition:
        """Create a fact definition from a discovered pattern."""
        if pattern['pattern_type'] == 'value_date_fact':
            value_col = pattern['value_column']
            date_cols = pattern['date_columns']
            
            # Generate fact key from value column
            fact_key = f"fact_{value_col.lower().replace(' ', '_')}"
            
            # Build role hints
            role_hints = {
                'value': {
                    'dtype': 'numeric',
                    'unit': self._infer_unit(column_analysis[value_col])
                }
            }
            
            for date_col in date_cols:
                role_hints['date'] = {'dtype': 'date'}
                break  # Use first date column
            
            return FactDefinition(
                fact_key=fact_key,
                description=f"Discovered fact pattern for {value_col}",
                required_roles=['value', 'date'],
                optional_roles=[],
                role_hints=role_hints,
                confidence_score=pattern['confidence'],
                sample_count=1,
                last_updated=datetime.now(timezone.utc).isoformat(),
                learned_patterns=pattern
            )
        
        # Default fallback
        return FactDefinition(
            fact_key="fact_unknown",
            description="Unknown fact pattern",
            confidence_score=0.3
        )
    
    def _infer_unit(self, column_info: Dict[str, Any]) -> Optional[str]:
        """Infer the unit from column information."""
        if column_info.get('is_currency_like'):
            return 'currency'
        
        patterns = column_info.get('common_patterns', [])
        for pattern in patterns:
            if any(unit in str(pattern).lower() for unit in ['hour', 'hr', 'h']):
                return 'hours'
            if any(unit in str(pattern).lower() for unit in ['kg', 'kilogram']):
                return 'kg'
            if any(unit in str(pattern).lower() for unit in ['tonne', 'ton', 't']):
                return 'tonnes'
        
        return None


class DynamicFactManager:
    """Manages dynamic fact definitions with learning and adaptation."""
    
    def __init__(self, db_path: str = "db/impactos.db", 
                 legacy_facts_path: str = "config/facts.json"):
        self.db_path = db_path
        self.legacy_facts_path = legacy_facts_path
        self.pattern_analyzer = DataPatternAnalyzer()
        self._ensure_fact_tables()
        self._migrate_legacy_facts()
    
    def _ensure_fact_tables(self):
        """Ensure dynamic fact tables exist in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS dynamic_facts (
                    fact_key TEXT PRIMARY KEY,
                    definition_json TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fact_learning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_key TEXT NOT NULL,
                    source_file TEXT,
                    source_context TEXT,
                    success_score REAL,
                    feedback_data TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (fact_key) REFERENCES dynamic_facts (fact_key)
                )
            ''')
            
            conn.commit()
    
    def _migrate_legacy_facts(self):
        """Migrate legacy facts.json to dynamic system."""
        if not os.path.exists(self.legacy_facts_path):
            return
            
        try:
            with open(self.legacy_facts_path, 'r') as f:
                legacy_facts = json.load(f)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for fact_key, fact_data in legacy_facts.items():
                    # Check if fact already exists
                    cursor.execute(
                        "SELECT fact_key FROM dynamic_facts WHERE fact_key = ?",
                        (fact_key,)
                    )
                    
                    if cursor.fetchone() is None:
                        # Migrate legacy fact
                        fact_def = FactDefinition.from_legacy_format(fact_key, fact_data)
                        fact_def.last_updated = datetime.now(timezone.utc).isoformat()
                        
                        cursor.execute('''
                            INSERT INTO dynamic_facts 
                            (fact_key, definition_json, confidence_score, sample_count, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            fact_key,
                            json.dumps(asdict(fact_def)),
                            fact_def.confidence_score,
                            fact_def.sample_count,
                            fact_def.last_updated,
                            fact_def.last_updated
                        ))
                
                conn.commit()
                logger.info(f"Migrated {len(legacy_facts)} legacy facts to dynamic system")
                
        except Exception as e:
            logger.warning(f"Failed to migrate legacy facts: {e}")
    
    def discover_facts_from_data(self, df: pd.DataFrame, 
                                source_context: Dict[str, Any]) -> List[FactDefinition]:
        """Discover new facts from a DataFrame."""
        return self.pattern_analyzer.analyze_dataframe(df, source_context)
    
    def learn_from_extraction(self, fact_key: str, source_file: str, 
                             success_score: float, feedback_data: Dict[str, Any]):
        """Learn from extraction results to improve fact definitions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO fact_learning_history 
                (fact_key, source_file, success_score, feedback_data, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                fact_key,
                source_file,
                success_score,
                json.dumps(feedback_data),
                datetime.now(timezone.utc).isoformat()
            ))
            
            # Update fact confidence based on learning
            self._update_fact_confidence(fact_key, success_score)
            
            conn.commit()
    
    def _update_fact_confidence(self, fact_key: str, success_score: float):
        """Update fact confidence based on learning feedback."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current fact definition
            cursor.execute(
                "SELECT definition_json, sample_count FROM dynamic_facts WHERE fact_key = ?",
                (fact_key,)
            )
            
            row = cursor.fetchone()
            if row:
                definition_json, sample_count = row
                fact_def = FactDefinition(**json.loads(definition_json))
                
                # Update confidence using exponential moving average
                alpha = 0.1  # Learning rate
                new_confidence = alpha * success_score + (1 - alpha) * fact_def.confidence_score
                new_sample_count = sample_count + 1
                
                fact_def.confidence_score = new_confidence
                fact_def.sample_count = new_sample_count
                fact_def.last_updated = datetime.now(timezone.utc).isoformat()
                
                # Update in database
                cursor.execute('''
                    UPDATE dynamic_facts 
                    SET definition_json = ?, confidence_score = ?, sample_count = ?, updated_at = ?
                    WHERE fact_key = ?
                ''', (
                    json.dumps(asdict(fact_def)),
                    new_confidence,
                    new_sample_count,
                    fact_def.last_updated,
                    fact_key
                ))
    
    def get_all_facts(self) -> Dict[str, FactDefinition]:
        """Get all active fact definitions."""
        facts = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fact_key, definition_json FROM dynamic_facts WHERE is_active = 1"
            )
            
            for row in cursor.fetchall():
                fact_key, definition_json = row
                fact_def = FactDefinition(**json.loads(definition_json))
                facts[fact_key] = fact_def
        
        return facts
    
    def get_facts_legacy_format(self) -> Dict[str, Dict[str, Any]]:
        """Get facts in legacy format for backward compatibility."""
        facts = self.get_all_facts()
        return {
            fact_key: fact_def.to_legacy_format()
            for fact_key, fact_def in facts.items()
        }
    
    def add_or_update_fact(self, fact_def: FactDefinition):
        """Add or update a fact definition."""
        now = datetime.now(timezone.utc).isoformat()
        fact_def.last_updated = now
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO dynamic_facts 
                (fact_key, definition_json, confidence_score, sample_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                fact_def.fact_key,
                json.dumps(asdict(fact_def)),
                fact_def.confidence_score,
                fact_def.sample_count,
                now,
                now
            ))
            
            conn.commit()
    
    def deactivate_fact(self, fact_key: str):
        """Deactivate a fact definition without deleting it."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE dynamic_facts SET is_active = 0 WHERE fact_key = ?",
                (fact_key,)
            )
            conn.commit()


class DynamicFactConfig:
    """Drop-in replacement for legacy FactConfig with dynamic capabilities."""
    
    def __init__(self, config_path: str = "config/facts.json", 
                 db_path: str = "db/impactos.db"):
        self.config_path = config_path
        self.db_path = db_path
        self.fact_manager = DynamicFactManager(db_path, config_path)
    
    def load_facts(self) -> Dict[str, Dict[str, Any]]:
        """Load facts in legacy format for backward compatibility."""
        return self.fact_manager.get_facts_legacy_format()
    
    def discover_and_learn(self, df: pd.DataFrame, 
                          source_context: Dict[str, Any]) -> List[FactDefinition]:
        """Discover new facts from data and add to system."""
        discovered_facts = self.fact_manager.discover_facts_from_data(df, source_context)
        
        for fact_def in discovered_facts:
            self.fact_manager.add_or_update_fact(fact_def)
            logger.info(f"Discovered and added new fact: {fact_def.fact_key}")
        
        return discovered_facts
    
    def provide_feedback(self, fact_key: str, source_file: str, 
                        success_score: float, feedback_data: Dict[str, Any]):
        """Provide feedback on fact extraction performance."""
        self.fact_manager.learn_from_extraction(fact_key, source_file, success_score, feedback_data)