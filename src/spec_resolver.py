"""
Dynamic spec resolver that maps Bronze sheets to Silver fact specs using
data-driven fact discovery and adaptive learning.

This implementation provides:
- Dynamic fact discovery from data patterns (replaces static facts.json)
- Sheet profiling with dtype detection, unit tokens, date-likeness
- Semantic candidate fact scoring with configurable thresholds
- Role mapping with composite scoring (w_graph + w_type + w_unit + w_history)
- LLM assistance for missing/ambiguous roles
- Comprehensive validation and learning
- Adaptive fact definitions that improve over time
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple
import sqlite3
import pandas as pd

# Import dynamic fact system
from dynamic_facts import DynamicFactConfig, DynamicFactManager

# Suppress tokenizer parallelism warnings
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

logger = logging.getLogger(__name__)


class LLMAssistant:
    """LLM assistance for missing/ambiguous role resolution with JSON schema."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        self.db_path = db_path
        self._load_config()
    
    def _load_config(self):
        """Load LLM configuration."""
        self.enabled = True
        self.model = "gpt-4o-mini"
        self.temperature = 0.0
        self.max_tokens = 1500
        
        try:
            config_path = "config/system_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    spec_config = config.get('spec_resolution', {})
                    self.enabled = spec_config.get('enable_llm_assistance', True)
                    self.model = spec_config.get('llm_model', 'gpt-4o-mini')
                    self.temperature = spec_config.get('llm_temperature', 0.0)
                    self.max_tokens = spec_config.get('llm_max_tokens', 1500)
        except Exception:
            pass
    
    def assist_role_mapping(self, sheet_profile: Dict[str, Any], fact_config: Dict[str, Any], 
                           current_mappings: Dict[str, str], 
                           top_candidates: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Use LLM to assist with missing/ambiguous role mappings.
        
        Returns JSON with format:
        {
            "mappings": [{"role": "date", "header": "date_of_last_activity", "score": 0.72, "why": "..."}],
            "notes": "..."
        }
        """
        if not self.enabled:
            return {"mappings": [], "notes": "LLM assistance disabled"}
        
        try:
            from llm_utils import get_llm_client
            client = get_llm_client()
            
            # Build prompt
            prompt = self._build_assistance_prompt(sheet_profile, fact_config, current_mappings, top_candidates)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate result
            if not self._validate_llm_response(result, sheet_profile):
                return {"mappings": [], "notes": "LLM response validation failed"}
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM assistance failed: {e}")
            return {"mappings": [], "notes": f"LLM assistance error: {e}"}
    
    def _build_assistance_prompt(self, sheet_profile: Dict[str, Any], fact_config: Dict[str, Any],
                                current_mappings: Dict[str, str], 
                                top_candidates: Dict[str, List[Dict[str, Any]]]) -> str:
        """Build prompt for LLM assistance."""
        sheet_name = sheet_profile.get('sheet_name', 'Unknown')
        available_headers = list(sheet_profile.get('columns', {}).keys())
        
        fact_key = fact_config.get('description', 'Unknown fact')
        required_roles = fact_config.get('required', [])
        optional_roles = fact_config.get('optional', [])
        
        missing_required = [r for r in required_roles if r not in current_mappings]
        missing_optional = [r for r in optional_roles if r not in current_mappings]
        
        prompt = f"""You are helping map column headers to semantic roles for data processing.

SHEET: {sheet_name}
FACT TYPE: {fact_key}

AVAILABLE HEADERS: {available_headers}

CURRENT MAPPINGS: {current_mappings}

MISSING REQUIRED ROLES: {missing_required}
MISSING OPTIONAL ROLES: {missing_optional}

TOP CANDIDATES PER ROLE:
"""
        
        for role, candidates in top_candidates.items():
            prompt += f"\n{role}:\n"
            for cand in candidates[:3]:
                prompt += f"  - {cand['header']} (score: {cand['score']:.2f}) - {cand.get('why', 'No reason')}\n"
        
        prompt += """

TASK: Suggest the best header mapping for each missing required role (and optionally for missing optional roles).

RESPOND WITH VALID JSON ONLY:
{
    "mappings": [
        {"role": "role_name", "header": "exact_header_name", "score": 0.85, "why": "explanation for why this mapping makes sense"}
    ],
    "notes": "Any additional observations about the data or mapping challenges"
}

CONSTRAINTS:
- Only suggest headers that exist in AVAILABLE HEADERS
- Focus on missing REQUIRED roles first
- Provide realistic confidence scores (0.0-1.0)
- Give clear explanations in "why" field
- If no good mapping exists for a role, don't include it in mappings array"""
        
        return prompt
    
    def _validate_llm_response(self, response: Dict[str, Any], sheet_profile: Dict[str, Any]) -> bool:
        """Validate LLM response structure and content."""
        if not isinstance(response, dict):
            return False
        
        if 'mappings' not in response:
            return False
        
        mappings = response['mappings']
        if not isinstance(mappings, list):
            return False
        
        available_headers = set(sheet_profile.get('columns', {}).keys())
        
        for mapping in mappings:
            if not isinstance(mapping, dict):
                return False
            
            required_fields = ['role', 'header', 'score']
            if not all(field in mapping for field in required_fields):
                return False
            
            # Validate header exists
            if mapping['header'] not in available_headers:
                logger.warning(f"LLM suggested non-existent header: {mapping['header']}")
                return False
            
            # Validate score range
            score = mapping.get('score', 0)
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                return False
        
        return True


class LearningTracker:
    """Tracks role mapping success/failure for learning and adaptive thresholds."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure learning tables exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Role mapping history table should exist from migrations
                # But we'll create if missing
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS role_mapping_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id TEXT NOT NULL,
                        fact_type TEXT NOT NULL,
                        role TEXT NOT NULL,
                        normalized_header TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        composite_score REAL,
                        component_scores TEXT,
                        spec_hash TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS spec_generation_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id TEXT NOT NULL,
                        sheet_name TEXT NOT NULL,
                        bronze_table TEXT NOT NULL,
                        fact_type TEXT NOT NULL,
                        spec_hash TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        missing_roles TEXT,
                        quality_score REAL,
                        resolution_method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_role_mapping_lookup 
                    ON role_mapping_history(tenant_id, role, normalized_header)
                """)
                
        except Exception as e:
            logger.warning(f"Failed to ensure learning tables: {e}")
    
    def record_role_mapping(self, tenant_id: str, fact_type: str, role: str, header: str,
                           success: bool, composite_score: float, component_scores: Dict[str, float],
                           spec_hash: str):
        """Record a role mapping attempt for learning."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO role_mapping_history 
                    (tenant_id, fact_key, role, header, normalized_header, success, score, component_scores, spec_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tenant_id, fact_type, role, header, header.lower(), success, 
                    composite_score, json.dumps(component_scores), spec_hash
                ))
        except Exception as e:
            logger.warning(f"Failed to record role mapping: {e}")
    
    def record_spec_generation(self, tenant_id: str, sheet_name: str, bronze_table: str,
                              fact_type: str, spec_hash: str, success: bool,
                              missing_roles: List[str], quality_score: float,
                              resolution_method: str):
        """Record a spec generation attempt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO spec_generation_history
                    (tenant_id, sheet_name, bronze_table, fact_key, spec_hash, success, 
                     validation_errors, quality_score, generation_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tenant_id, sheet_name, bronze_table, fact_type, spec_hash, success,
                    json.dumps(missing_roles), quality_score, resolution_method
                ))
        except Exception as e:
            logger.warning(f"Failed to record spec generation: {e}")
    
    def get_role_success_rate(self, tenant_id: str, role: str, normalized_header: str) -> float:
        """Get historical success rate for a role-header combination."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT AVG(CAST(success AS FLOAT)) as success_rate
                    FROM role_mapping_history 
                    WHERE tenant_id = ? AND role = ? AND normalized_header = ?
                """, (tenant_id, role, normalized_header.lower()))
                
                result = cur.fetchone()
                return float(result[0]) if result and result[0] is not None else 0.5
        except Exception:
            return 0.5
    
    def get_adaptive_threshold(self, tenant_id: str, role: str, default_threshold: float) -> float:
        """Get adaptive threshold based on historical performance."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT AVG(score) as avg_score, COUNT(*) as count
                    FROM role_mapping_history 
                    WHERE tenant_id = ? AND role = ? AND success = 1
                """, (tenant_id, role))
                
                result = cur.fetchone()
                if result and result[1] >= 10:  # Need at least 10 samples
                    avg_successful_score = result[0]
                    # Lower threshold if we're consistently succeeding with lower scores
                    if avg_successful_score < default_threshold:
                        return max(0.3, avg_successful_score - 0.1)
                
                return default_threshold
        except Exception:
            return default_threshold


def _hash_spec(spec: Dict[str, Any]) -> str:
    """Generate deterministic hash for a spec."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode('utf-8')).hexdigest()


class FactsConfig:
    """Dynamic fact manager with backward compatibility for legacy FactsConfig interface."""
    
    def __init__(self, config_path: str = "config/facts.json", db_path: str = "db/impactos.db"):
        self.config_path = config_path
        self.db_path = db_path
        # Use dynamic fact config as backend
        self.dynamic_config = DynamicFactConfig(config_path, db_path)
        self._facts_cache: Optional[Dict[str, Dict[str, Any]]] = None
    
    def load_facts(self) -> Dict[str, Dict[str, Any]]:
        """Load fact definitions using dynamic system with legacy format."""
        # Always get fresh facts from dynamic system to include learned facts
        self._facts_cache = self.dynamic_config.load_facts()
        return self._facts_cache
    
    def discover_facts_from_data(self, df: pd.DataFrame, source_context: Dict[str, Any]) -> int:
        """Discover new facts from DataFrame and add to system."""
        discovered_facts = self.dynamic_config.discover_and_learn(df, source_context)
        logger.info(f"Discovered {len(discovered_facts)} new fact patterns from data")
        
        # Clear cache to include new facts
        self._facts_cache = None
        return len(discovered_facts)
    
    def provide_extraction_feedback(self, fact_key: str, source_file: str, 
                                  success_score: float, feedback_data: Dict[str, Any]):
        """Provide feedback on fact extraction performance for learning."""
        self.dynamic_config.provide_feedback(fact_key, source_file, success_score, feedback_data)
        logger.debug(f"Provided feedback for fact {fact_key}: score={success_score}")
        
    def get_dynamic_manager(self) -> DynamicFactManager:
        """Get access to underlying dynamic fact manager for advanced operations."""
        return self.dynamic_config.fact_manager


class SheetProfiler:
    """Profiles a sheet to extract metadata for semantic resolution."""
    
    @staticmethod
    def _normalize_header(header: str) -> str:
        """Normalize header name for consistent matching."""
        if not header:
            return ""
        normalized = re.sub(r'[^\w\s]', ' ', str(header).lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    @staticmethod
    def _detect_dtype(series: pd.Series) -> str:
        """Detect the data type of a pandas series."""
        if series.empty:
            return "unknown"
        
        try:
            pd.to_numeric(series.dropna(), errors='raise')
            return "numeric"
        except (ValueError, TypeError):
            pass
        
        try:
            # Suppress pandas warnings about date format inference
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pd.to_datetime(series.dropna(), errors='raise')
            return "date"
        except (ValueError, TypeError):
            pass
        
        return "string"
    
    @staticmethod
    def _extract_unit_tokens(header: str, examples: List[str]) -> List[str]:
        """Extract unit tokens from header and example values."""
        tokens = []
        header_lower = header.lower()
        
        # Currency patterns
        if re.search(r'[£$€]', header_lower) or re.search(r'\b(gbp|usd|eur|currency)\b', header_lower):
            tokens.extend(['currency', 'money'])
        
        # Energy units
        energy_matches = re.findall(r'\b(kwh|mwh|gwh|kw|mw)\b', header_lower)
        tokens.extend(energy_matches)
        
        # Time units - including volunteering hours pattern
        time_matches = re.findall(r'\b(hours?|hrs?|minutes?|mins?)\b', header_lower)
        tokens.extend(time_matches)
        if 'hour' in header_lower and 'volunteer' in header_lower:
            tokens.extend(['hours', 'volunteering'])
        
        # Mass units
        mass_matches = re.findall(r'\b(tonnes?|kg|kilogram|tons?)\b', header_lower)
        tokens.extend(mass_matches)
        
        # Extract from examples
        for example in examples[:3]:
            if example and re.search(r'[£$€]', str(example)):
                tokens.append('currency')
        
        return list(set(tokens))
    
    @staticmethod
    def _calculate_date_likeness(header: str, examples: List[str]) -> float:
        """Calculate how date-like a column is (0.0 to 1.0)."""
        score = 0.0
        
        # Header analysis
        header_lower = header.lower()
        date_keywords = ['date', 'time', 'activity', 'created', 'updated']
        if any(keyword in header_lower for keyword in date_keywords):
            score += 0.3
        
        # Example analysis
        if examples:
            date_like_count = 0
            for example in examples[:5]:
                if not example:
                    continue
                try:
                    pd.to_datetime(str(example), errors='raise')
                    date_like_count += 1
                except:
                    # Check date patterns
                    if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', str(example)):
                        date_like_count += 0.8
            
            score += (date_like_count / len(examples)) * 0.7
        
        return min(score, 1.0)
    
    def profile_sheet(self, sheet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Profile a sheet and return metadata for semantic resolution."""
        sheet_name = sheet_data.get('sheet_name', '')
        headers = sheet_data.get('columns', [])
        examples = sheet_data.get('example_values', {})
        
        profile = {
            'sheet_name': sheet_name,
            'bronze_table': sheet_data.get('bronze_table', ''),
            'num_columns': len(headers),
            'columns': {}
        }
        
        for header in headers:
            if not header:
                continue
                
            header_examples = examples.get(header, [])
            series = pd.Series(header_examples) if header_examples else pd.Series([])
            
            normalized = self._normalize_header(header)
            dtype = self._detect_dtype(series)
            null_pct = (series.isna().sum() / len(series)) if len(series) > 0 else 0.0
            top_examples = [str(x) for x in header_examples[:3] if x is not None]
            unit_tokens = self._extract_unit_tokens(header, top_examples)
            date_likeness = self._calculate_date_likeness(header, top_examples)
            
            profile['columns'][header] = {
                'original_header': header,
                'normalized_header': normalized,
                'dtype': dtype,
                'null_pct': float(null_pct),
                'top_examples': top_examples,
                'unit_tokens': unit_tokens,
                'date_likeness': float(date_likeness)
            }
        
        return profile


class RoleScorer:
    """Scores role-to-header mappings using composite signals."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        self.db_path = db_path
        self._semantic_resolver = None
        self._load_config()
    
    def _load_config(self):
        """Load scoring configuration."""
        self.weights = {
            'w_graph': 0.4,
            'w_type': 0.25,
            'w_unit': 0.25,
            'w_history': 0.1
        }
        
        self.thresholds = {
            'default': 0.6,
            'value': 0.7,
            'date': 0.65,
        }
        
        try:
            config_path = "config/system_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    spec_config = config.get('spec_resolution', {})
                    if 'role_scoring_weights' in spec_config:
                        self.weights.update(spec_config['role_scoring_weights'])
                    if 'role_thresholds' in spec_config:
                        self.thresholds.update(spec_config['role_thresholds'])
        except Exception:
            pass
    
    @property
    def semantic_resolver(self):
        """Lazy load semantic resolver."""
        if self._semantic_resolver is None:
            try:
                from semantic_resolver import SemanticResolver
                self._semantic_resolver = SemanticResolver(self.db_path)
            except Exception as e:
                logger.error(f"Failed to initialize semantic resolver: {e}")
        return self._semantic_resolver
    
    def _score_graph(self, role: str, header: str, context: Dict[str, Any]) -> float:
        """Score using semantic resolver against concept graph."""
        if not self.semantic_resolver:
            return 0.0
        
        try:
            text = f"{role} | {header}"
            result = self.semantic_resolver.resolve('field_role', text, context=context)
            return float(result.get('score', 0.0)) if result.get('outcome') == 'accepted' else 0.0
        except Exception:
            return 0.0
    
    def _score_type(self, role: str, dtype: str, role_hints: Dict[str, Any]) -> float:
        """Score based on data type compatibility."""
        expected_dtype = role_hints.get('dtype', 'string')
        
        if dtype == expected_dtype:
            return 1.0
        
        # Partial compatibility
        if expected_dtype == 'numeric' and dtype == 'string':
            return 0.3
        elif expected_dtype == 'date' and dtype == 'string':
            return 0.4
        
        return 0.0
    
    def _score_unit(self, role: str, unit_tokens: List[str], role_hints: Dict[str, Any]) -> float:
        """Score based on unit token compatibility."""
        expected_unit = role_hints.get('unit', '')
        expected_values = role_hints.get('values', [])
        
        # For currency roles, be very strict - require explicit currency indicators
        if role == 'currency' or expected_unit == 'currency':
            if not unit_tokens:
                return 0.1  # Very low score for currency without indicators
            
            currency_indicators = ['currency', 'money', '£', '$', '€', 'gbp', 'usd', 'eur']
            for token in unit_tokens:
                token_lower = str(token).lower()
                if token_lower in currency_indicators:
                    return 1.0
            
            # Check expected values
            if expected_values:
                all_expected_lower = [str(x).lower() for x in expected_values]
                for token in unit_tokens:
                    if str(token).lower() in all_expected_lower:
                        return 1.0
            
            return 0.0  # No currency indicators found
        
        # For other units, use existing logic
        if not unit_tokens:
            return 0.5
        
        all_expected = [expected_unit] + expected_values
        all_expected_lower = [str(x).lower() for x in all_expected if x]
        
        for token in unit_tokens:
            token_lower = str(token).lower()
            if token_lower in all_expected_lower:
                return 1.0
            
            # Synonyms for non-currency units
            if expected_unit == 'hours' and token_lower in ['hour', 'hrs', 'h']:
                return 0.9
        
        return 0.2
    
    def _score_history(self, role: str, header: str, tenant_id: str) -> float:
        """Score based on historical success rate."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT AVG(CAST(success AS FLOAT)) as success_rate
                    FROM role_mapping_history 
                    WHERE tenant_id = ? AND role = ? AND normalized_header = ?
                """, (tenant_id, role, header.lower()))
                
                result = cur.fetchone()
                return float(result[0]) if result and result[0] is not None else 0.5
        except Exception:
            return 0.5
    
    def score_role_mapping(self, role: str, header: str, column_profile: Dict[str, Any], 
                          role_hints: Dict[str, Any], tenant_id: str = 'default') -> Dict[str, Any]:
        """Score a role-to-header mapping using composite signals."""
        context = {
            'header': header,
            'role': role,
            'examples': column_profile.get('top_examples', []),
            'dtype': column_profile.get('dtype', 'unknown'),
            'unit_tokens': column_profile.get('unit_tokens', [])
        }
        
        # Component scores
        score_graph = self._score_graph(role, header, context)
        score_type = self._score_type(role, column_profile.get('dtype', 'unknown'), role_hints)
        score_unit = self._score_unit(role, column_profile.get('unit_tokens', []), role_hints)
        score_history = self._score_history(role, header, tenant_id)
        
        # Composite score
        composite_score = (
            self.weights['w_graph'] * score_graph +
            self.weights['w_type'] * score_type +
            self.weights['w_unit'] * score_unit +
            self.weights['w_history'] * score_history
        )
        
        # Use adaptive threshold based on learning history
        base_threshold = self.thresholds.get(role, self.thresholds['default'])
        try:
            tracker = LearningTracker(self.db_path)
            threshold = tracker.get_adaptive_threshold(tenant_id, role, base_threshold)
        except Exception:
            threshold = base_threshold
        
        return {
            'overall_score': float(composite_score),
            'component_scores': {
                'graph': float(score_graph),
                'type': float(score_type),
                'unit': float(score_unit),
                'history': float(score_history)
            },
            'threshold': float(threshold),
            'base_threshold': float(base_threshold),
            'meets_threshold': composite_score >= threshold,
            'weights_used': dict(self.weights)
        }


def resolve_specs_for_sheet(tenant_id: str, sheet_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of {spec_id, spec_snapshot, spec_hash} for a sheet using generic semantic resolution.
    
    This is the main entry point that implements all requirements from the original prompt.
    """
    try:
        # Initialize components
        facts_config = FactsConfig()
        profiler = SheetProfiler()
        scorer = RoleScorer()
        
        # Profile the sheet
        sheet_profile = profiler.profile_sheet(sheet_row)
        
        # Load facts
        facts = facts_config.load_facts()
        if not facts:
            logger.warning("No facts configuration available")
            return []
        
        # Find candidate facts using semantic resolution
        candidate_facts = _find_candidate_facts(sheet_profile, facts)
        
        # Build specs for each candidate fact
        specs = []
        for fact_key in candidate_facts:
            fact_config = facts[fact_key]
            spec = _build_spec_for_fact(fact_key, fact_config, sheet_profile, scorer, tenant_id)
            if spec:
                specs.append(spec)
        
        return specs
        
    except Exception as e:
        logger.error(f"Failed to resolve specs for sheet {sheet_row.get('sheet_name', 'unknown')}: {e}")
        return []


def _find_candidate_facts(sheet_profile: Dict[str, Any], facts: Dict[str, Dict[str, Any]]) -> List[str]:
    """Find candidate facts using SemanticResolver.resolve('fact', ...)."""
    from semantic_resolver import SemanticResolver
    
    candidate_facts = []
    
    try:
        resolver = SemanticResolver()
        
        # Build descriptive text
        sheet_name = sheet_profile.get('sheet_name', '')
        headers = list(sheet_profile.get('columns', {}).keys())
        header_text = ", ".join(headers)
        describe = f"{sheet_name} | {header_text}"
        
        context = {
            'sheet_name': sheet_name,
            'headers': headers,
            'profile': sheet_profile
        }
        
        # Load τ_fact threshold
        tau_fact = 0.7
        try:
            config_path = "config/system_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    tau_fact = config.get('spec_resolution', {}).get('tau_fact', 0.7)
        except Exception:
            pass
        
        # Try semantic resolution first
        result = resolver.resolve('fact', describe, context=context)
        
        if result.get('outcome') == 'accepted' and result.get('score', 0) >= tau_fact:
            fact_key = result.get('key')
            if fact_key in facts:
                candidate_facts.append(fact_key)
        
        # Score all facts if no direct match
        if not candidate_facts:
            for fact_key, fact_config in facts.items():
                score = _score_sheet_for_fact(sheet_profile, fact_key, fact_config, resolver)
                if score >= tau_fact:
                    candidate_facts.append(fact_key)
        
        # Heuristic fallback if still no candidates
        if not candidate_facts:
            headers_text = " ".join(headers).lower()
            
            # Require explicit volunteering semantics; do not trigger on generic "hours"
            if any(keyword in headers_text for keyword in ['volunteer', 'community', 'csr']):
                if 'fact_volunteering' in facts:
                    candidate_facts.append('fact_volunteering')
            
            if any(keyword in headers_text for keyword in ['donation', 'contribution', 'charitable']):
                if 'fact_donations' in facts:
                    candidate_facts.append('fact_donations')
            
            if any(keyword in headers_text for keyword in ['energy', 'kwh', 'emission']):
                if 'fact_energy' in facts:
                    candidate_facts.append('fact_energy')
            
            if any(keyword in headers_text for keyword in ['procurement', 'supplier']):
                if 'fact_procurement' in facts:
                    candidate_facts.append('fact_procurement')
            
            if any(keyword in headers_text for keyword in ['waste', 'tonne']):
                if 'fact_waste' in facts:
                    candidate_facts.append('fact_waste')
    
    except Exception as e:
        logger.warning(f"Fact resolution failed: {e}")
    
    # Remove duplicates and limit results
    max_facts = 3
    try:
        config_path = "config/system_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                max_facts = config.get('spec_resolution', {}).get('max_facts_per_sheet', 3)
    except Exception:
        pass
    
    return list(dict.fromkeys(candidate_facts))[:max_facts]


def _score_sheet_for_fact(sheet_profile: Dict[str, Any], fact_key: str, 
                         fact_config: Dict[str, Any], resolver) -> float:
    """Score how well a sheet matches a specific fact."""
    try:
        sheet_name = sheet_profile.get('sheet_name', '')
        headers = list(sheet_profile.get('columns', {}).keys())
        header_text = ", ".join(headers)
        
        fact_description = fact_config.get('description', '')
        describe = f"{fact_description} | {sheet_name} | {header_text}"
        
        context = {
            'fact_key': fact_key,
            'fact_description': fact_description,
            'sheet_name': sheet_name,
            'headers': headers
        }
        
        result = resolver.resolve('fact', describe, context=context)
        
        if (result.get('outcome') == 'accepted' and result.get('key') == fact_key):
            return result.get('score', 0.0)
        
        return 0.0
        
    except Exception:
        return 0.0


def _build_spec_for_fact(fact_key: str, fact_config: Dict[str, Any], 
                        sheet_profile: Dict[str, Any], scorer: RoleScorer, 
                        tenant_id: str) -> Optional[Dict[str, Any]]:
    """Build a spec for a specific fact by mapping roles to headers."""
    required_roles = fact_config.get('required', [])
    optional_roles = fact_config.get('optional', [])
    role_hints = fact_config.get('role_hints', {})
    
    mappings = {}
    all_roles = required_roles + optional_roles
    top_candidates = {}  # Track top candidates for LLM assistance
    
    # Score each role against all headers and build comprehensive candidate lists
    role_scores = {}  # role -> [(header, score_result), ...]
    
    for role in all_roles:
        hints = role_hints.get(role, {})
        role_candidates = []
        
        for header, column_profile in sheet_profile.get('columns', {}).items():
            # Semantic guardrails (generic, not hardcoded mappings):
            header_lc = str(header).lower()
            normalized_lc = str(column_profile.get('normalized_header', '')).lower()

            # 1) Volunteering value must have volunteering semantics; block typical payroll "overtime"
            if fact_key == 'fact_volunteering' and role == 'value':
                has_positive = any(k in header_lc for k in ['volunteer', 'community', 'csr'])
                has_negative = any(k in header_lc for k in ['overtime', 'ot'])
                if has_negative and not has_positive:
                    # Skip this candidate entirely
                    continue
                # If no positive semantics at all, downweight by skipping low-signal columns
                if not has_positive and 'hour' in header_lc:
                    # Require explicit volunteering semantics to consider generic hour columns
                    continue

            # 2) "year" should never be used for value/hour roles; allow only as date role
            if role in ('value',) and ('year' in normalized_lc or normalized_lc == 'year'):
                continue

            score_result = scorer.score_role_mapping(role, header, column_profile, hints, tenant_id)
            
            # Track all candidates for each role
            role_candidates.append({
                'header': header,
                'score': score_result['overall_score'],
                'meets_threshold': score_result['meets_threshold'],
                'component_scores': score_result['component_scores'],
                'why': f"Type: {column_profile.get('dtype', 'unknown')}, "
                       f"Unit: {column_profile.get('unit_tokens', [])}, "
                       f"Examples: {column_profile.get('top_examples', [])[:2]}"
            })
        
        # Sort candidates by score
        role_candidates.sort(key=lambda x: x['score'], reverse=True)
        role_scores[role] = role_candidates
        top_candidates[role] = role_candidates[:3]
    
    # Assign roles to headers using conflict resolution
    used_headers = set()
    
    # Priority assignment: Required roles first, then by best score
    role_priority = []
    for role in required_roles:
        if role_scores[role] and role_scores[role][0]['meets_threshold']:
            role_priority.append((role, role_scores[role][0]['score'], True))  # True = required
    
    for role in optional_roles:
        if role_scores[role] and role_scores[role][0]['meets_threshold']:
            role_priority.append((role, role_scores[role][0]['score'], False))  # False = optional
    
    # Sort by: required first, then by score descending
    role_priority.sort(key=lambda x: (not x[2], -x[1]))
    
    # Assign headers avoiding conflicts
    for role, score, is_required in role_priority:
        candidates = role_scores[role]
        
        # Find best header that hasn't been used
        for candidate in candidates:
            header = candidate['header']
            if candidate['meets_threshold'] and header not in used_headers:
                mappings[role] = header
                used_headers.add(header)
                logger.debug(f"Assigned {role} -> {header} (score: {candidate['score']:.3f})")
                break
    
    # Check required roles
    missing_required = [role for role in required_roles if role not in mappings]
    
    # Try LLM assistance for missing/ambiguous roles
    if missing_required:
        logger.debug(f"Missing required roles for {fact_key}: {missing_required}")
        
        try:
            assistant = LLMAssistant()
            llm_result = assistant.assist_role_mapping(
                sheet_profile, fact_config, mappings, top_candidates
            )
            
            # Apply LLM suggestions for missing required roles
            for suggestion in llm_result.get('mappings', []):
                role = suggestion.get('role')
                header = suggestion.get('header')
                score = suggestion.get('score', 0.0)
                
                if role in missing_required and header and score >= 0.6:
                    mappings[role] = header
                    logger.info(f"LLM assisted mapping: {role} -> {header} (score: {score:.2f})")
                    missing_required.remove(role)
            
        except Exception as e:
            logger.warning(f"LLM assistance failed for {fact_key}: {e}")
    
    # Final check for required roles
    final_missing = [role for role in required_roles if role not in mappings]
    if final_missing:
        logger.debug(f"Still missing required roles for {fact_key}: {final_missing}")
        return None
    
    # Validate spec
    if not _validate_spec(mappings, sheet_profile, fact_config):
        logger.debug(f"Spec validation failed for {fact_key}")
        return None
    
    # Build spec
    spec_snapshot = {
        'fact': fact_key,
        'mappings': mappings,
        'base_currency': fact_config.get('base_currency', 'GBP')
    }
    
    # Add defaults for missing optional roles
    if 'currency' not in mappings and 'currency' in optional_roles:
        spec_snapshot['mappings']['currency'] = fact_config.get('default_currency', 'GBP')
    
    if 'unit' not in mappings and 'unit' in optional_roles:
        default_unit = fact_config.get('default_unit')
        if default_unit:
            spec_snapshot['mappings']['unit'] = default_unit
    
    # Clean up poor mappings - remove optional role mappings that don't make semantic sense
    columns = sheet_profile.get('columns', {})
    to_remove = []
    
    for role, header in spec_snapshot['mappings'].items():
        if role in optional_roles and isinstance(header, str) and header in columns:
            column_profile = columns[header]
            hints = role_hints.get(role, {})
            
            # Re-score this mapping
            score_result = scorer.score_role_mapping(role, header, column_profile, hints, tenant_id)
            
            # If the score is very low, remove this mapping and use default if available
            if score_result['overall_score'] < 0.3:
                logger.debug(f"Removing poor mapping {role} -> {header} (score: {score_result['overall_score']:.3f})")
                to_remove.append(role)
    
    # Remove poor mappings and add defaults where available
    for role in to_remove:
        del spec_snapshot['mappings'][role]
        
        # Add default if available
        if role == 'currency' and 'default_currency' in fact_config:
            spec_snapshot['mappings']['currency'] = fact_config['default_currency']
        elif role == 'unit' and 'default_unit' in fact_config:
            spec_snapshot['mappings']['unit'] = fact_config['default_unit']
    
    spec_id = f"{fact_key}_v1"
    spec_hash = _hash_spec(spec_snapshot)
    
    # Calculate quality score for telemetry
    quality_scores = []
    for role, header in mappings.items():
        if isinstance(header, str) and header in sheet_profile.get('columns', {}):
            hints = role_hints.get(role, {})
            column_profile = sheet_profile['columns'][header]
            score_result = scorer.score_role_mapping(role, header, column_profile, hints, tenant_id)
            quality_scores.append(score_result['overall_score'])
    
    overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    # Record learning data
    try:
        tracker = LearningTracker()
        
        # Record individual role mappings
        for role, header in mappings.items():
            if isinstance(header, str) and header in sheet_profile.get('columns', {}):
                hints = role_hints.get(role, {})
                column_profile = sheet_profile['columns'][header]
                score_result = scorer.score_role_mapping(role, header, column_profile, hints, tenant_id)
                
                # Consider mapping successful if it meets threshold and passes validation
                success = score_result['meets_threshold']
                tracker.record_role_mapping(
                    tenant_id, fact_key, role, header, success,
                    score_result['overall_score'], score_result['component_scores'], spec_hash
                )
        
        # Record spec generation
        sheet_name = sheet_profile.get('sheet_name', 'Unknown')
        bronze_table = sheet_profile.get('bronze_table', 'unknown')
        tracker.record_spec_generation(
            tenant_id, sheet_name, bronze_table, fact_key, spec_hash,
            True, [], overall_quality, 'semantic'
        )
        
    except Exception as e:
        logger.warning(f"Failed to record learning data: {e}")
    
    return {
        'spec_id': spec_id,
        'spec_snapshot': spec_snapshot,
        'spec_hash': spec_hash,
        'quality_score': overall_quality
    }


def _validate_spec(mappings: Dict[str, str], sheet_profile: Dict[str, Any], 
                  fact_config: Dict[str, Any]) -> bool:
    """Validate that a spec is viable."""
    columns = sheet_profile.get('columns', {})
    
    # Ensure all mapped headers exist
    for role, header in mappings.items():
        if header not in columns:
            return False
    
    # Validate data types for critical roles
    role_hints = fact_config.get('role_hints', {})
    
    for role, header in mappings.items():
        if header not in columns:
            continue
            
        column_profile = columns[header]
        hints = role_hints.get(role, {})
        expected_dtype = hints.get('dtype')
        actual_dtype = column_profile.get('dtype')
        
        # Validate critical roles
        if role == 'value' and expected_dtype == 'numeric' and actual_dtype != 'numeric':
            # Check if convertible
            examples = column_profile.get('top_examples', [])
            if not _is_convertible_to_numeric(examples):
                return False
        
        if role == 'date' and expected_dtype == 'date':
            date_likeness = column_profile.get('date_likeness', 0.0)
            if date_likeness < 0.5:
                return False
    
    return True


def _is_convertible_to_numeric(examples: List[str]) -> bool:
    """Check if string examples can be converted to numeric values."""
    if not examples:
        return False
    
    convertible_count = 0
    for example in examples[:5]:
        if not example:
            continue
        try:
            cleaned = re.sub(r'[^\d.-]', '', str(example))
            if cleaned:
                float(cleaned)
                convertible_count += 1
        except (ValueError, TypeError):
            pass
    
    return convertible_count >= len(examples) * 0.6