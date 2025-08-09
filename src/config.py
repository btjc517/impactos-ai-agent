"""
Configuration Management for ImpactOS AI Layer MVP.

This module provides centralized configuration management to replace hardcoded 
values, models, and parameters that limit accuracy and scalability. All critical thresholds, limits, 
and settings should be configurable here.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchConfig:
    """Configuration for FAISS vector search system."""
    embedding_dimension: int = 384  # all-MiniLM-L6-v2 default
    default_k: int = 100  # Number of results to retrieve initially
    min_similarity_threshold: float = 0.2  # Lower threshold for inclusivity
    adaptive_similarity: bool = True  # Enable adaptive similarity based on query type
    batch_size: int = 100  # Processing batch size
    index_type: str = "IndexFlatIP"  # FAISS index type
    
    # Adaptive thresholds based on query type
    similarity_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.similarity_thresholds is None:
            self.similarity_thresholds = {
                'aggregation': 0.25,  # Lower threshold for comprehensive aggregation
                'descriptive': 0.3,   # Medium threshold for descriptive queries
                'analytical': 0.35,   # Higher threshold for precise analysis
                'default': 0.3
            }


@dataclass  
class QueryProcessingConfig:
    """Configuration for query processing and result management."""
    
    # Result limits based on query type (much higher for accuracy)
    max_results_for_gpt: int = 25  # Increased from 15
    max_initial_retrieval: int = 500  # Increased from 200
    
    # Query-specific result limits
    aggregation_result_limit: int = 200  # Increased from 100
    descriptive_result_limit: int = 100  # Increased from 50  
    analytical_result_limit: int = 75   # Increased from 30
    
    # LLM answer synthesis defaults (neutral names; old gpt4_* accepted via aliases)
    answer_max_tokens: int = 2000
    # Temperature removed for GPT-5-only usage; retained for legacy parsing only
    # answer_temperature is deprecated
    answer_temperature: Optional[float] = None
    answer_model: str = "gpt-5-mini"
    
    # Result filtering
    enable_intelligent_filtering: bool = True
    duplicate_detection: bool = True
    
    # NEW: GPT context composition for aggregation queries
    aggregation_ctx_max_aggregated: int = 8
    aggregation_ctx_max_individual: int = 10
    aggregation_ctx_max_vector: int = 5
    
    # Performance tuning
    enable_result_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour


@dataclass
class ExtractionConfig:
    """Configuration for data extraction processes."""
    
    # Extraction token budgets (neutral names; old gpt4_* accepted via aliases)
    extraction_max_tokens: int = 40000       # For complex extractions
    # extraction_max_tokens: int = 4000       # For complex extractions
    structure_analysis_max_tokens: int = 30000
    # structure_analysis_max_tokens: int = 3000
    verification_max_tokens: int = 25000
    # verification_max_tokens: int = 2500
    
    # Ingestion models (run infrequently)
    structure_analysis_model: str = "gpt-5"
    query_generation_model: str = "gpt-5"
    
    # Extraction accuracy
    confidence_threshold: float = 0.7  # Minimum confidence for acceptance
    verification_tolerance: float = 0.02  # 2% tolerance (increased from 1%)
    
    # Processing limits
    max_rows_per_batch: int = 1000
    max_file_size_mb: int = 50
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class ScalabilityConfig:
    """Configuration for system scalability and performance."""
    
    # Memory management
    max_memory_usage_mb: int = 2048  # 2GB limit
    result_chunk_size: int = 100
    
    # Database performance
    db_connection_pool_size: int = 5
    query_timeout_seconds: int = 30
    
    # FAISS optimization
    faiss_nprobe: int = 1  # Search precision vs speed tradeoff
    faiss_metric_type: str = "METRIC_INNER_PRODUCT"
    
    # Batch processing
    embedding_batch_size: int = 50
    metric_processing_batch_size: int = 100


@dataclass  
class AnalysisConfig:
    """Configuration for query analysis patterns and keywords."""
    category_keywords: Dict[str, Any] = None
    aggregation_patterns: Dict[str, Any] = None
    time_patterns: Any = None
    category_min_similarity: float = 0.35
    aggregation_min_similarity: float = 0.30
    framework_mapping_min_similarity: float = 0.35
    # Descriptive prototypes for embedding-based intent detection
    category_descriptions: Dict[str, str] = None
    aggregation_descriptions: Dict[str, str] = None
    use_embedding_for_intent: bool = True
    use_embedding_for_framework_mapping: bool = True
    # Optional: use LLM for intent classification
    use_llm_for_intent: bool = False
    llm_intent_model: str = "gpt-5-nano"
    llm_intent_temperature: float = 0.0
    # Default tightened to 150 as per GPT-5 intent budget
    llm_intent_max_tokens: int = 150


@dataclass
class LLMConfig:
    """Configuration for LLM behavior, escalation, and formatting.

    Added in a backward-compatible way – existing code paths remain unchanged if
    these fields are not referenced.
    """
    # Reasoning and verbosity controls per task
    reasoning_effort: Dict[str, str] = None  # intent/answer/extraction
    verbosity: Dict[str, str] = None         # intent/answer/extraction

    # JSON enforcement flags and schemas (opt-in). Defaults safe for extraction
    json_enforce_intent: bool = False
    json_enforce_extraction: bool = True
    json_schema_intent: Optional[str] = None
    json_schema_extraction: Optional[str] = None

    # Dynamic escalation controls
    escalation_enabled: bool = True
    escalation_limits: Dict[str, Any] = None  # {max_retries, p95_latency_ms}
    
    def __post_init__(self):
        if self.reasoning_effort is None:
            self.reasoning_effort = {
                'intent': 'minimal',
                'answer': 'medium',
                'extraction': 'medium'
            }
        if self.verbosity is None:
            self.verbosity = {
                'intent': 'low',
                'answer': 'medium',
                'extraction': 'low'
            }
        if self.escalation_limits is None:
            self.escalation_limits = {
                'max_retries': 2,
                'p95_latency_ms': 6000
            }


@dataclass
class QAConfig:
    """Quality and validation thresholds for Q&A outputs."""
    citation_min_count: int = 3
    source_agreement_tolerance: float = 10.0  # percent
    min_context_items: int = 3


@dataclass  
class SystemConfig:
    """Main system configuration container."""
    
    vector_search: VectorSearchConfig
    query_processing: QueryProcessingConfig  
    extraction: ExtractionConfig
    scalability: ScalabilityConfig
    analysis: AnalysisConfig
    llm: LLMConfig
    qa: QAConfig
    
    # Environment settings
    environment: str = "development"  # development, staging, production
    debug_logging: bool = True
    
    def __init__(self):
        self.vector_search = VectorSearchConfig()
        self.query_processing = QueryProcessingConfig()
        self.extraction = ExtractionConfig()
        self.scalability = ScalabilityConfig()
        self.analysis = AnalysisConfig()
        # New sections (backward-compatible)
        self.llm = LLMConfig()
        self.qa = QAConfig()


class ConfigManager:
    """Manages configuration loading, saving, and environment-specific overrides."""
    
    def __init__(self, config_path: str = "config/system_config.json"):
        self.config_path = config_path
        self.config = SystemConfig()
        self._load_config()
        self._apply_environment_overrides()
    
    def _load_config(self):
        """Load configuration from file if it exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                self._update_config_from_dict(config_dict)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("No config file found, using defaults")
            self.save_config()  # Create default config file
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        # Backward-compatibility mapping for old key names
        legacy_key_map = {
            'query_processing': {
                'gpt4_model': 'answer_model',
                'gpt4_temperature': 'answer_temperature',  # deprecated
                'gpt4_max_tokens': 'answer_max_tokens',
            },
            'extraction': {
                'gpt4_max_tokens_extraction': 'extraction_max_tokens',
                'gpt4_max_tokens_analysis': 'structure_analysis_max_tokens',
                'gpt4_max_tokens_verification': 'verification_max_tokens',
            },
        }
        for section_name, section_config in config_dict.items():
            if hasattr(self.config, section_name):
                section = getattr(self.config, section_name)
                # Translate legacy keys first
                translated = dict(section_config)
                if section_name in legacy_key_map:
                    for old_key, new_key in legacy_key_map[section_name].items():
                        if old_key in section_config and hasattr(section, new_key):
                            translated[new_key] = section_config[old_key]
                for key, value in translated.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        environment = os.getenv('IMPACTOS_ENV', 'development')
        self.config.environment = environment
        
        if environment == 'production':
            # Production optimizations
            self.config.query_processing.max_results_for_gpt = 30
            self.config.query_processing.answer_max_tokens = 1500
            self.config.vector_search.batch_size = 200
            self.config.scalability.max_memory_usage_mb = 4096
            self.config.debug_logging = False
            
        elif environment == 'development':
            # Development settings for accuracy over performance
            self.config.query_processing.max_results_for_gpt = 25
            self.config.query_processing.answer_max_tokens = 2000
            self.config.debug_logging = True
            
        # Apply environment variable overrides
        env_overrides = {
            'IMPACTOS_SIMILARITY_THRESHOLD': ('vector_search', 'min_similarity_threshold', float),
            'IMPACTOS_MAX_TOKENS': ('query_processing', 'answer_max_tokens', int),
            'IMPACTOS_MAX_RESULTS': ('query_processing', 'max_results_for_gpt', int),
            'IMPACTOS_BATCH_SIZE': ('vector_search', 'batch_size', int),
            'IMPACTOS_USE_LLM_INTENT': ('analysis', 'use_llm_for_intent', lambda v: str(v).lower() in ('1','true','yes')),
            'IMPACTOS_INTENT_MODEL': ('analysis', 'llm_intent_model', str),
            # New flags (optional)
            'IMPACTOS_ESCALATION_ENABLED': ('llm', 'escalation_enabled', lambda v: str(v).lower() in ('1','true','yes')),
            'IMPACTOS_P95_LATENCY_MS': ('llm', 'escalation_limits', lambda v: {'p95_latency_ms': int(v), 'max_retries': self.config.llm.escalation_limits.get('max_retries', 2)}),
        }
        
        for env_var, (section, key, type_func) in env_overrides.items():
            if env_var in os.environ:
                try:
                    value = type_func(os.environ[env_var])
                    setattr(getattr(self.config, section), key, value)
                    logger.info(f"Applied environment override: {env_var} = {value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment override {env_var}: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        config_dict = {
            'vector_search': asdict(self.config.vector_search),
            'query_processing': asdict(self.config.query_processing),
            'extraction': asdict(self.config.extraction),  
            'scalability': asdict(self.config.scalability),
            'analysis': asdict(self.config.analysis),
            'llm': asdict(self.config.llm),
            'qa': asdict(self.config.qa)
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_similarity_threshold(self, query_type: str = 'default') -> float:
        """Get adaptive similarity threshold based on query type."""
        if self.config.vector_search.adaptive_similarity:
            return self.config.vector_search.similarity_thresholds.get(
                query_type, 
                self.config.vector_search.similarity_thresholds['default']
            )
        return self.config.vector_search.min_similarity_threshold
    
    def get_result_limit(self, query_type: str) -> int:
        """Get result limit based on query type."""
        limits = {
            'aggregation': self.config.query_processing.aggregation_result_limit,
            'descriptive': self.config.query_processing.descriptive_result_limit,
            'analytical': self.config.query_processing.analytical_result_limit
        }
        return limits.get(query_type, self.config.query_processing.analytical_result_limit)
    
    def validate_config(self) -> bool:
        """Validate configuration values."""
        issues = []
        
        # Check critical thresholds
        if self.config.vector_search.min_similarity_threshold < 0.1:
            issues.append("Similarity threshold too low (< 0.1)")
        if self.config.vector_search.min_similarity_threshold > 0.9:
            issues.append("Similarity threshold too high (> 0.9)")
            
        # Check token limits
        if self.config.query_processing.answer_max_tokens > 4000:
            issues.append("GPT-5 max tokens exceeds recommended limit (4000)")
            
        # Check memory limits
        if self.config.scalability.max_memory_usage_mb < 512:
            issues.append("Memory limit too low (< 512MB)")
            
        if issues:
            for issue in issues:
                logger.warning(f"Config validation issue: {issue}")
            return False
        
        return True


# Global configuration instance
_config_manager = None

def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        if not _config_manager.validate_config():
            logger.warning("Configuration validation failed - some settings may be suboptimal")
    return _config_manager.config

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance.""" 
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager 