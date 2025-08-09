"""
Metrics Collector for ImpactOS AI Layer Testing System.

This module provides comprehensive metrics collection during query execution
to track performance, accuracy, costs, and quality over time.
"""

import time
import psutil
import os
import sys
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from contextlib import contextmanager
import subprocess

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Comprehensive metrics collection during query execution."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.reset_metrics()
        self.process = psutil.Process(os.getpid())
        
        # GPT pricing (per 1K tokens)
        self.gpt_pricing = {
            'gpt-5': {'input': 0.03, 'output': 0.06},
            'gpt-5-mini': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002}
        }
    
    def reset_metrics(self):
        """Reset all metrics for a new test."""
        self.start_time = None
        self.end_time = None
        
        # Performance metrics
        self.timing_metrics = {
            'total_time': 0,
            'sql_search_time': 0,
            'vector_search_time': 0,
            'gpt_processing_time': 0,
            'result_filtering_time': 0
        }
        
        # Result metrics
        self.result_metrics = {
            'sql_results_found': 0,
            'vector_results_found': 0,
            'total_results_retrieved': 0,
            'results_sent_to_gpt': 0
        }
        
        # Memory metrics
        self.memory_metrics = {
            'memory_usage_peak': 0,
            'memory_usage_avg': 0,
            'memory_samples': []
        }
        
        # GPT metrics
        self.gpt_metrics = {
            'gpt_calls_made': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'estimated_cost_usd': 0,
            'response_length_chars': 0,
            'response_time': 0,
            'model_used': '',
            'temperature': 0,
            'max_tokens_requested': 0,
            'response_truncated': False,
            'api_errors': 0,
            'retry_attempts': 0
        }
        
        # System metrics
        self.system_metrics = {
            'db_size_mb': 0,
            'total_metrics_count': 0,
            'total_embeddings_count': 0,
            'faiss_index_size_mb': 0,
            'faiss_vector_count': 0,
            'faiss_search_time_avg': 0,
            'similarity_threshold': 0,
            'max_results_for_gpt': 0,
            'gpt_max_tokens': 0,
            'python_version': sys.version.split()[0],
            'faiss_version': self._get_package_version('faiss-cpu'),
            'torch_version': self._get_package_version('torch')
        }
        
        # Quality tracking
        self.quality_metrics = {
            'answer_length_words': 0,
            'unique_sources_cited': 0,
            'framework_mappings_included': 0,
            'citations_found': 0,
            'response_time_category': 'unknown'
        }
    
    def _get_package_version(self, package_name: str) -> str:
        """Get version of installed package."""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                                 capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
        except:
            pass
        return 'unknown'
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time specific operations."""
        start = time.time()
        yield
        duration = time.time() - start
        if operation_name in self.timing_metrics:
            self.timing_metrics[operation_name] = duration
        logger.debug(f"{operation_name} took {duration:.3f} seconds")
    
    def start_query_timing(self):
        """Start overall query timing."""
        self.start_time = time.time()
        self._sample_memory()
    
    def end_query_timing(self):
        """End overall query timing."""
        self.end_time = time.time()
        self.timing_metrics['total_time'] = self.end_time - self.start_time
        self._sample_memory()
        self._calculate_memory_stats()
        self._categorize_response_time()
    
    def _sample_memory(self):
        """Sample current memory usage."""
        try:
            memory_mb = self.process.memory_info().rss / (1024 * 1024)
            self.memory_metrics['memory_samples'].append(memory_mb)
        except:
            pass
    
    def _calculate_memory_stats(self):
        """Calculate memory statistics from samples."""
        if self.memory_metrics['memory_samples']:
            samples = self.memory_metrics['memory_samples']
            self.memory_metrics['memory_usage_peak'] = max(samples)
            self.memory_metrics['memory_usage_avg'] = sum(samples) / len(samples)
    
    def _categorize_response_time(self):
        """Categorize response time for UX metrics."""
        total_time = self.timing_metrics['total_time']
        if total_time < 3:
            self.quality_metrics['response_time_category'] = 'fast'
        elif total_time < 10:
            self.quality_metrics['response_time_category'] = 'medium'
        else:
            self.quality_metrics['response_time_category'] = 'slow'
    
    def record_sql_results(self, count: int):
        """Record SQL search results count."""
        self.result_metrics['sql_results_found'] = count
        self.result_metrics['total_results_retrieved'] += count
    
    def record_vector_results(self, count: int):
        """Record vector search results count."""
        self.result_metrics['vector_results_found'] = count
        self.result_metrics['total_results_retrieved'] += count
    
    def record_gpt_results(self, count: int):
        """Record results sent to GPT."""
        self.result_metrics['results_sent_to_gpt'] = count
    
    def record_gpt_call(self, model: str, input_tokens: int, output_tokens: int, 
                       temperature: float, max_tokens: int, response_time: float,
                       response_length: int, truncated: bool = False):
        """Record GPT API call metrics."""
        self.gpt_metrics['gpt_calls_made'] += 1
        self.gpt_metrics['total_input_tokens'] += input_tokens
        self.gpt_metrics['total_output_tokens'] += output_tokens
        self.gpt_metrics['response_length_chars'] += response_length
        self.gpt_metrics['response_time'] += response_time
        self.gpt_metrics['model_used'] = model
        self.gpt_metrics['temperature'] = temperature
        self.gpt_metrics['max_tokens_requested'] = max_tokens
        self.gpt_metrics['response_truncated'] = truncated
        
        # Calculate cost
        if model in self.gpt_pricing:
            pricing = self.gpt_pricing[model]
            input_cost = (input_tokens / 1000) * pricing['input']
            output_cost = (output_tokens / 1000) * pricing['output']
            self.gpt_metrics['estimated_cost_usd'] += input_cost + output_cost
    
    def record_gpt_error(self):
        """Record GPT API error."""
        self.gpt_metrics['api_errors'] += 1
    
    def record_gpt_retry(self):
        """Record GPT API retry attempt."""
        self.gpt_metrics['retry_attempts'] += 1
    
    def analyze_answer_quality(self, answer: str, query: str) -> Dict[str, Any]:
        """Analyze answer quality and extract metrics."""
        quality_scores = {}
        
        # Basic metrics
        self.quality_metrics['answer_length_words'] = len(answer.split())
        
        # Extract citations
        citations = self._extract_citations(answer)
        self.quality_metrics['citations_found'] = len(citations)
        
        # Extract unique sources
        sources = self._extract_sources(answer)
        self.quality_metrics['unique_sources_cited'] = len(sources)
        
        # Count framework mappings
        frameworks = self._count_framework_mappings(answer)
        self.quality_metrics['framework_mappings_included'] = frameworks
        
        # Quality scoring (basic heuristics)
        quality_scores.update({
            'comprehensiveness_score': self._score_comprehensiveness(answer, query),
            'clarity_score': self._score_clarity(answer),
            'factual_accuracy_score': self._score_factual_accuracy(answer),
            'source_diversity_score': self._score_source_diversity(sources),
            'answer_usefulness_score': self._score_usefulness(answer, query),
            'citation_format_score': self._score_citation_format(answer)
        })
        
        return quality_scores
    
    def _extract_citations(self, answer: str) -> List[str]:
        """Extract citation numbers from answer."""
        citations = re.findall(r'\[(\d+)\]', answer)
        return list(set(citations))
    
    def _extract_sources(self, answer: str) -> List[str]:
        """Extract unique source filenames from answer."""
        # Look for patterns like "Source: filename.xlsx"
        sources = re.findall(r'Source: ([^||\n]+)', answer)
        # Clean up and deduplicate
        sources = [s.strip() for s in sources]
        return list(set(sources))
    
    def _count_framework_mappings(self, answer: str) -> int:
        """Count framework mappings mentioned in answer."""
        framework_patterns = [
            r'Uk Sv Model:', r'Un Sdgs:', r'Toms:', r'B Corp:', r'MAC:'
        ]
        count = 0
        for pattern in framework_patterns:
            count += len(re.findall(pattern, answer, re.IGNORECASE))
        return count
    
    def _score_comprehensiveness(self, answer: str, query: str) -> float:
        """Score answer comprehensiveness (0-1)."""
        # Heuristic: longer answers with more details are more comprehensive
        word_count = len(answer.split())
        
        # Base score on length
        if word_count > 200:
            base_score = 0.9
        elif word_count > 100:
            base_score = 0.8
        elif word_count > 50:
            base_score = 0.6
        else:
            base_score = 0.4
        
        # Bonus for numbers and specific data
        has_numbers = bool(re.search(r'\d+', answer))
        has_currency = bool(re.search(r'[£$€]\d+', answer))
        
        bonus = 0
        if has_numbers:
            bonus += 0.05
        if has_currency:
            bonus += 0.05
            
        return min(1.0, base_score + bonus)
    
    def _score_clarity(self, answer: str) -> float:
        """Score answer clarity (0-1)."""
        # Heuristic: well-structured answers with clear formatting
        has_bullet_points = '•' in answer or '*' in answer
        has_numbered_list = bool(re.search(r'\d+\.', answer))
        has_clear_structure = 'Sources:' in answer or 'Based on' in answer
        
        score = 0.6  # Base score
        if has_bullet_points or has_numbered_list:
            score += 0.2
        if has_clear_structure:
            score += 0.2
            
        return min(1.0, score)
    
    def _score_factual_accuracy(self, answer: str) -> float:
        """Score factual accuracy based on citations (0-1)."""
        # Heuristic: answers with citations are more likely to be accurate
        citation_count = len(self._extract_citations(answer))
        source_count = len(self._extract_sources(answer))
        
        if citation_count >= 3 and source_count >= 2:
            return 0.9
        elif citation_count >= 2 and source_count >= 1:
            return 0.8
        elif citation_count >= 1:
            return 0.7
        else:
            return 0.5
    
    def _score_source_diversity(self, sources: List[str]) -> float:
        """Score source diversity (0-1)."""
        if len(sources) >= 3:
            return 1.0
        elif len(sources) == 2:
            return 0.8
        elif len(sources) == 1:
            return 0.6
        else:
            return 0.3
    
    def _score_usefulness(self, answer: str, query: str) -> float:
        """Score answer usefulness for the query (0-1)."""
        # Heuristic: answers that directly address the query keywords
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(query_words.intersection(answer_words))
        coverage = overlap / len(query_words) if query_words else 0
        
        return min(1.0, coverage + 0.3)  # Base 30% + coverage
    
    def _score_citation_format(self, answer: str) -> float:
        """Score citation formatting quality (0-1)."""
        has_numbered_citations = bool(re.search(r'\[\d+\]', answer))
        has_source_details = 'Sources with Details:' in answer
        has_structured_sources = bool(re.search(r'Source: .+ \|', answer))
        
        score = 0.5  # Base score
        if has_numbered_citations:
            score += 0.2
        if has_source_details:
            score += 0.2
        if has_structured_sources:
            score += 0.1
            
        return min(1.0, score)
    
    def collect_system_metrics(self, config, vector_search_system):
        """Collect current system metrics."""
        try:
            # Database size
            db_path = "db/impactos.db"
            if os.path.exists(db_path):
                self.system_metrics['db_size_mb'] = os.path.getsize(db_path) / (1024 * 1024)
            
            # FAISS index size
            faiss_path = "db/faiss_index.faiss"
            if os.path.exists(faiss_path):
                self.system_metrics['faiss_index_size_mb'] = os.path.getsize(faiss_path) / (1024 * 1024)
            
            # Vector counts
            if hasattr(vector_search_system, 'index') and vector_search_system.index:
                self.system_metrics['faiss_vector_count'] = vector_search_system.index.ntotal
            
            # Configuration values
            self.system_metrics['similarity_threshold'] = config.vector_search.min_similarity_threshold
            self.system_metrics['max_results_for_gpt'] = config.query_processing.max_results_for_gpt
            self.system_metrics['gpt_max_tokens'] = getattr(config.query_processing, 'answer_max_tokens', getattr(config.query_processing, 'gpt4_max_tokens', 2000))
            
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all collected metrics."""
        return {
            'performance': {**self.timing_metrics, **self.result_metrics, **self.memory_metrics},
            'gpt': self.gpt_metrics,
            'system': self.system_metrics,
            'quality': self.quality_metrics
        } 