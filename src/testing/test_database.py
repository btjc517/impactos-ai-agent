"""
Test Results Database for ImpactOS AI Layer Performance Tracking.

This module handles storing and retrieving comprehensive test metrics over time
including performance, accuracy, cost, and quality measurements.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TestDatabase:
    """Manages test results database with comprehensive metrics tracking."""
    
    def __init__(self, db_path: str = "db/test_results.db"):
        """Initialize test database."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize database schema for test results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main test runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version_hash TEXT,
                    environment TEXT,
                    config_snapshot TEXT,
                    total_tests INTEGER,
                    passed_tests INTEGER,
                    failed_tests INTEGER,
                    overall_score REAL,
                    notes TEXT
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id INTEGER,
                    query_text TEXT,
                    query_type TEXT,
                    
                    -- Timing metrics (seconds)
                    total_time REAL,
                    sql_search_time REAL,
                    vector_search_time REAL,
                    gpt_processing_time REAL,
                    result_filtering_time REAL,
                    
                    -- Result metrics
                    sql_results_found INTEGER,
                    vector_results_found INTEGER,
                    total_results_retrieved INTEGER,
                    results_sent_to_gpt INTEGER,
                    
                    -- Memory metrics (MB)
                    memory_usage_peak REAL,
                    memory_usage_avg REAL,
                    
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                )
            """)
            
            # GPT/LLM metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gpt_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id INTEGER,
                    query_text TEXT,
                    
                    -- API metrics
                    gpt_calls_made INTEGER,
                    total_input_tokens INTEGER,
                    total_output_tokens INTEGER,
                    estimated_cost_usd REAL,
                    
                    -- Response metrics
                    response_length_chars INTEGER,
                    response_time REAL,
                    model_used TEXT,
                    temperature REAL,
                    max_tokens_requested INTEGER,
                    
                    -- Quality indicators
                    response_truncated BOOLEAN,
                    api_errors INTEGER,
                    retry_attempts INTEGER,
                    
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                )
            """)
            
            # Accuracy metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id INTEGER,
                    query_text TEXT,
                    expected_answer TEXT,
                    actual_answer TEXT,
                    
                    -- Accuracy scores (0-1)
                    answer_accuracy_score REAL,
                    citation_accuracy_score REAL,
                    completeness_score REAL,
                    relevance_score REAL,
                    
                    -- Citation analysis
                    citations_found INTEGER,
                    citations_accurate INTEGER,
                    citations_missing INTEGER,
                    source_coverage_score REAL,
                    
                    -- Content analysis
                    key_facts_found INTEGER,
                    key_facts_expected INTEGER,
                    numerical_accuracy_score REAL,
                    framework_mapping_accuracy REAL,
                    
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                )
            """)
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id INTEGER,
                    
                    -- Database metrics
                    db_size_mb REAL,
                    total_metrics_count INTEGER,
                    total_embeddings_count INTEGER,
                    
                    -- FAISS metrics
                    faiss_index_size_mb REAL,
                    faiss_vector_count INTEGER,
                    faiss_search_time_avg REAL,
                    
                    -- Configuration metrics
                    similarity_threshold REAL,
                    max_results_for_gpt INTEGER,
                    gpt_max_tokens INTEGER,
                    
                    -- Environment info
                    python_version TEXT,
                    faiss_version TEXT,
                    torch_version TEXT,
                    
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                )
            """)
            
            # Quality metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id INTEGER,
                    query_text TEXT,
                    query_type TEXT,
                    
                    -- Answer quality scores (0-1)
                    comprehensiveness_score REAL,
                    clarity_score REAL,
                    factual_accuracy_score REAL,
                    source_diversity_score REAL,
                    
                    -- Response characteristics
                    answer_length_words INTEGER,
                    unique_sources_cited INTEGER,
                    framework_mappings_included INTEGER,
                    
                    -- User experience metrics
                    response_time_category TEXT, -- fast/medium/slow
                    answer_usefulness_score REAL,
                    citation_format_score REAL,
                    
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                )
            """)
            
            conn.commit()
            logger.info("Test database schema initialized")
    
    def create_test_run(self, environment: str = "development", 
                       version_hash: str = None, 
                       config_snapshot: Dict = None,
                       notes: str = None) -> int:
        """Create a new test run record and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO test_runs 
                (version_hash, environment, config_snapshot, notes)
                VALUES (?, ?, ?, ?)
            """, (
                version_hash,
                environment,
                json.dumps(config_snapshot) if config_snapshot else None,
                notes
            ))
            
            test_run_id = cursor.lastrowid
            logger.info(f"Created test run {test_run_id}")
            return test_run_id
    
    def store_performance_metrics(self, test_run_id: int, metrics: Dict[str, Any]):
        """Store performance metrics for a test."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_metrics
                (test_run_id, query_text, query_type, total_time, sql_search_time,
                 vector_search_time, gpt_processing_time, result_filtering_time,
                 sql_results_found, vector_results_found, total_results_retrieved,
                 results_sent_to_gpt, memory_usage_peak, memory_usage_avg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_run_id,
                metrics.get('query_text'),
                metrics.get('query_type'),
                metrics.get('total_time'),
                metrics.get('sql_search_time'),
                metrics.get('vector_search_time'),
                metrics.get('gpt_processing_time'),
                metrics.get('result_filtering_time'),
                metrics.get('sql_results_found'),
                metrics.get('vector_results_found'),
                metrics.get('total_results_retrieved'),
                metrics.get('results_sent_to_gpt'),
                metrics.get('memory_usage_peak'),
                metrics.get('memory_usage_avg')
            ))
    
    def store_gpt_metrics(self, test_run_id: int, metrics: Dict[str, Any]):
        """Store GPT/LLM metrics for a test."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO gpt_metrics
                (test_run_id, query_text, gpt_calls_made, total_input_tokens,
                 total_output_tokens, estimated_cost_usd, response_length_chars,
                 response_time, model_used, temperature, max_tokens_requested,
                 response_truncated, api_errors, retry_attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_run_id,
                metrics.get('query_text'),
                metrics.get('gpt_calls_made'),
                metrics.get('total_input_tokens'),
                metrics.get('total_output_tokens'),
                metrics.get('estimated_cost_usd'),
                metrics.get('response_length_chars'),
                metrics.get('response_time'),
                metrics.get('model_used'),
                metrics.get('temperature'),
                metrics.get('max_tokens_requested'),
                metrics.get('response_truncated'),
                metrics.get('api_errors'),
                metrics.get('retry_attempts')
            ))
    
    def store_accuracy_metrics(self, test_run_id: int, metrics: Dict[str, Any]):
        """Store accuracy metrics for a test."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO accuracy_metrics
                (test_run_id, query_text, expected_answer, actual_answer,
                 answer_accuracy_score, citation_accuracy_score, completeness_score,
                 relevance_score, citations_found, citations_accurate,
                 citations_missing, source_coverage_score, key_facts_found,
                 key_facts_expected, numerical_accuracy_score, framework_mapping_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_run_id,
                metrics.get('query_text'),
                metrics.get('expected_answer'),
                metrics.get('actual_answer'),
                metrics.get('answer_accuracy_score'),
                metrics.get('citation_accuracy_score'),
                metrics.get('completeness_score'),
                metrics.get('relevance_score'),
                metrics.get('citations_found'),
                metrics.get('citations_accurate'),
                metrics.get('citations_missing'),
                metrics.get('source_coverage_score'),
                metrics.get('key_facts_found'),
                metrics.get('key_facts_expected'),
                metrics.get('numerical_accuracy_score'),
                metrics.get('framework_mapping_accuracy')
            ))
    
    def store_system_metrics(self, test_run_id: int, metrics: Dict[str, Any]):
        """Store system metrics for a test run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_metrics
                (test_run_id, db_size_mb, total_metrics_count, total_embeddings_count,
                 faiss_index_size_mb, faiss_vector_count, faiss_search_time_avg,
                 similarity_threshold, max_results_for_gpt, gpt_max_tokens,
                 python_version, faiss_version, torch_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_run_id,
                metrics.get('db_size_mb'),
                metrics.get('total_metrics_count'),
                metrics.get('total_embeddings_count'),
                metrics.get('faiss_index_size_mb'),
                metrics.get('faiss_vector_count'),
                metrics.get('faiss_search_time_avg'),
                metrics.get('similarity_threshold'),
                metrics.get('max_results_for_gpt'),
                metrics.get('gpt_max_tokens'),
                metrics.get('python_version'),
                metrics.get('faiss_version'),
                metrics.get('torch_version')
            ))
    
    def store_quality_metrics(self, test_run_id: int, metrics: Dict[str, Any]):
        """Store quality metrics for a test."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO quality_metrics
                (test_run_id, query_text, query_type, comprehensiveness_score,
                 clarity_score, factual_accuracy_score, source_diversity_score,
                 answer_length_words, unique_sources_cited, framework_mappings_included,
                 response_time_category, answer_usefulness_score, citation_format_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_run_id,
                metrics.get('query_text'),
                metrics.get('query_type'),
                metrics.get('comprehensiveness_score'),
                metrics.get('clarity_score'),
                metrics.get('factual_accuracy_score'),
                metrics.get('source_diversity_score'),
                metrics.get('answer_length_words'),
                metrics.get('unique_sources_cited'),
                metrics.get('framework_mappings_included'),
                metrics.get('response_time_category'),
                metrics.get('answer_usefulness_score'),
                metrics.get('citation_format_score')
            ))
    
    def update_test_run_summary(self, test_run_id: int, total_tests: int, 
                               passed_tests: int, failed_tests: int, overall_score: float):
        """Update test run with summary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE test_runs 
                SET total_tests = ?, passed_tests = ?, failed_tests = ?, overall_score = ?
                WHERE id = ?
            """, (total_tests, passed_tests, failed_tests, overall_score, test_run_id))
    
    def get_test_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent test run history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM test_runs 
                ORDER BY run_timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    tr.run_timestamp,
                    AVG(pm.total_time) as avg_response_time,
                    AVG(pm.sql_results_found) as avg_sql_results,
                    AVG(pm.vector_results_found) as avg_vector_results,
                    AVG(gm.total_input_tokens) as avg_input_tokens,
                    AVG(gm.total_output_tokens) as avg_output_tokens,
                    AVG(gm.estimated_cost_usd) as avg_cost,
                    AVG(am.answer_accuracy_score) as avg_accuracy
                FROM test_runs tr
                LEFT JOIN performance_metrics pm ON tr.id = pm.test_run_id
                LEFT JOIN gpt_metrics gm ON tr.id = gm.test_run_id  
                LEFT JOIN accuracy_metrics am ON tr.id = am.test_run_id
                WHERE tr.run_timestamp > datetime('now', '-{} days')
                GROUP BY DATE(tr.run_timestamp)
                ORDER BY tr.run_timestamp DESC
            """.format(days))
            
            return [dict(row) for row in cursor.fetchall()] 