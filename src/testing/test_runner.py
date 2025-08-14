"""
Test Runner for ImpactOS AI Layer Testing System.

This module orchestrates comprehensive testing including performance, accuracy,
cost analysis, and progress tracking over time.
"""

import os
import sys
import time
import subprocess
import git
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from test_database import TestDatabase
from metrics_collector import MetricsCollector
from test_cases import TestCases, TestCase
from query import QuerySystem
from config import get_config, get_config_manager

logger = logging.getLogger(__name__)


class TestRunner:
    """Comprehensive test runner with metrics collection and progress tracking."""
    
    def __init__(self, db_path: str = "db/test_results.db"):
        """Initialize test runner."""
        self.test_db = TestDatabase(db_path)
        self.metrics_collector = MetricsCollector()
        self.query_system = None
        self.config = get_config()
        self.config_manager = get_config_manager()
        
        # Test results tracking
        self.current_test_run_id = None
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'total': 0
        }
    
    def run_comprehensive_test_suite(self, test_types: List[str] = None, 
                                   notes: str = None) -> Dict[str, Any]:
        """
        Run comprehensive test suite and return results summary.
        
        Args:
            test_types: Optional list of test types to run ['performance', 'accuracy', 'all']
            notes: Optional notes for this test run
            
        Returns:
            Test results summary
        """
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        # Initialize test run
        test_run_summary = self._initialize_test_run(notes)
        
        try:
            # Get test cases
            if test_types is None or 'all' in test_types:
                test_cases = TestCases.get_all_test_cases()
            else:
                test_cases = []
                if 'performance' in test_types:
                    test_cases.extend(TestCases.get_performance_test_cases())
                if 'accuracy' in test_types:
                    test_cases.extend(TestCases.get_accuracy_test_cases())
                # Remove duplicates
                seen = set()
                test_cases = [tc for tc in test_cases if tc.id not in seen and not seen.add(tc.id)]
            
            print(f"📊 Running {len(test_cases)} test cases")
            
            # Initialize query system with local testing database
            test_db_path = "db/impactos.db"  # Local database in testing/db/ directory
            main_db_path = "../../db/impactos.db"  # Main database path
            
            # Use local test database if it exists, otherwise main database
            if os.path.exists(test_db_path):
                db_path = test_db_path
                print(f"🔗 Using local test database: {db_path}")
            elif os.path.exists(main_db_path):
                db_path = main_db_path
                print(f"🔗 Using main database: {db_path}")
            else:
                db_path = "db/impactos.db"  # Fallback
                print(f"🔗 Using fallback database: {db_path}")
            
            self.query_system = QuerySystem(db_path)
            
            # Collect system metrics once per run
            self._collect_system_metrics()
            
            # Run all test cases
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n[{i}/{len(test_cases)}] Testing: {test_case.id} - {test_case.description}")
                
                try:
                    self._run_single_test(test_case)
                    self.test_results['passed'] += 1
                    print(f"✅ PASSED: {test_case.id}")
                    
                except Exception as e:
                    self.test_results['failed'] += 1
                    logger.error(f"❌ FAILED: {test_case.id} - {e}")
                    print(f"❌ FAILED: {test_case.id} - {str(e)[:100]}...")
                
                self.test_results['total'] += 1
            
            # Calculate overall score
            overall_score = self.test_results['passed'] / self.test_results['total'] if self.test_results['total'] > 0 else 0
            
            # Update test run summary
            self.test_db.update_test_run_summary(
                self.current_test_run_id,
                self.test_results['total'],
                self.test_results['passed'],
                self.test_results['failed'],
                overall_score
            )
            
            # Generate summary
            test_run_summary.update({
                'test_results': self.test_results,
                'overall_score': overall_score,
                'completion_time': datetime.now(),
                'duration_minutes': (time.time() - test_run_summary['start_time']) / 60
            })
            
            # Print results
            self._print_test_summary(test_run_summary)
            
            return test_run_summary
            
        except Exception as e:
            logger.error(f"Critical error in test suite: {e}")
            print(f"💥 CRITICAL ERROR: {e}")
            return {'error': str(e), 'test_results': self.test_results}
    
    def _initialize_test_run(self, notes: str = None) -> Dict[str, Any]:
        """Initialize a new test run."""
        environment = os.getenv('IMPACTOS_ENV', 'development')
        version_hash = self._get_git_hash()
        config_snapshot = {
            'similarity_threshold': self.config.vector_search.min_similarity_threshold,
            'max_results_for_gpt': self.config.query_processing.max_results_for_gpt,
            'gpt_max_tokens': getattr(self.config.query_processing, 'answer_max_tokens', getattr(self.config.query_processing, 'gpt4_max_tokens', 2000)),
            'environment': environment
        }
        
        self.current_test_run_id = self.test_db.create_test_run(
            environment=environment,
            version_hash=version_hash,
            config_snapshot=config_snapshot,
            notes=notes
        )
        
        return {
            'test_run_id': self.current_test_run_id,
            'environment': environment,
            'version_hash': version_hash,
            'start_time': time.time(),
            'config': config_snapshot
        }
    
    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha[:8]
        except:
            return 'unknown'
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        try:
            self.metrics_collector.collect_system_metrics(self.config, self.query_system.vector_search)
            system_metrics = self.metrics_collector.system_metrics
            self.test_db.store_system_metrics(self.current_test_run_id, system_metrics)
            
            print(f"📈 System Metrics:")
            print(f"   DB Size: {system_metrics['db_size_mb']:.1f} MB")
            print(f"   FAISS Vectors: {system_metrics['faiss_vector_count']}")
            print(f"   Similarity Threshold: {system_metrics['similarity_threshold']}")
            print(f"   Max GPT Tokens: {system_metrics['gpt_max_tokens']}")
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def _run_single_test(self, test_case: TestCase):
        """Run a single test case with comprehensive metrics collection."""
        # Reset metrics for this test
        self.metrics_collector.reset_metrics()
        
        # Start timing and memory monitoring
        self.metrics_collector.start_query_timing()
        
        # Execute query
        answer = self._execute_query_with_instrumentation(test_case.query)
        
        # End timing
        self.metrics_collector.end_query_timing()
        
        # Analyze answer quality
        quality_scores = self.metrics_collector.analyze_answer_quality(answer, test_case.query)
        
        # Analyze accuracy against expected results
        accuracy_scores = self._analyze_accuracy(test_case, answer)
        
        # Store all metrics
        self._store_test_metrics(test_case, answer, quality_scores, accuracy_scores)
        
        # Validate test passed
        self._validate_test_result(test_case, answer, accuracy_scores)
    
    def _execute_query_with_instrumentation(self, query: str) -> str:
        """Execute query with detailed instrumentation."""
        try:
            # Instrument the query system to collect metrics
            answer = self._instrumented_query(query)
            return answer
            
        except Exception as e:
            self.metrics_collector.record_gpt_error()
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _instrumented_query(self, query: str) -> str:
        """Execute query with detailed timing instrumentation."""
        # Analyze query to get type
        analysis = self.query_system._analyze_query(query)
        
        # Time SQL search
        with self.metrics_collector.time_operation('sql_search_time'):
            sql_results = self.query_system._enhanced_sql_search(analysis)
            self.metrics_collector.record_sql_results(len(sql_results))
        
        # Time vector search  
        with self.metrics_collector.time_operation('vector_search_time'):
            vector_results = self.query_system._faiss_vector_search(query, analysis)
            self.metrics_collector.record_vector_results(len(vector_results))
        
        # Combine results
        all_results = sql_results + vector_results
        
        # Time result filtering
        with self.metrics_collector.time_operation('result_filtering_time'):
            relevant_results = self.query_system._filter_by_relevance(all_results)
            final_results = self.query_system._apply_query_specific_limits(relevant_results, analysis)
            filtered_results = self.query_system._intelligent_filter_for_gpt(final_results, analysis)
            self.metrics_collector.record_gpt_results(len(filtered_results))
        
        # Time GPT processing and collect GPT metrics
        with self.metrics_collector.time_operation('gpt_processing_time'):
            if self.query_system.openai_client and filtered_results:
                gpt_start = time.time()
                answer = self.query_system._generate_gpt_answer(query, filtered_results)
                gpt_time = time.time() - gpt_start
                
                # Estimate token usage (rough approximation)
                input_tokens = len(str(filtered_results)) // 4  # Rough token estimate
                output_tokens = len(answer) // 4
                
                self.metrics_collector.record_gpt_call(
                    model=getattr(self.config.query_processing, 'answer_model', getattr(self.config.query_processing, 'gpt4_model', 'gpt-4o-mini')),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    temperature=float(getattr(self.config.query_processing, 'answer_temperature', 0.0) or 0.0),
                    max_tokens=getattr(self.config.query_processing, 'answer_max_tokens', getattr(self.config.query_processing, 'gpt4_max_tokens', 2000)),
                    response_time=gpt_time,
                    response_length=len(answer),
                    truncated=len(answer) > (getattr(self.config.query_processing, 'answer_max_tokens', getattr(self.config.query_processing, 'gpt4_max_tokens', 2000)) * 3)
                )
            else:
                answer = self.query_system._generate_fallback_answer(query, final_results)
        
        return answer
    
    def _analyze_accuracy(self, test_case: TestCase, answer: str) -> Dict[str, float]:
        """Analyze answer accuracy against test case expectations."""
        accuracy_scores = {}
        
        # Check for expected keywords
        answer_lower = answer.lower()
        keyword_matches = sum(1 for keyword in test_case.expected_answer_keywords 
                            if keyword.lower() in answer_lower)
        keyword_score = keyword_matches / len(test_case.expected_answer_keywords) if test_case.expected_answer_keywords else 0
        accuracy_scores['answer_accuracy_score'] = keyword_score
        
        # Check for expected sources
        source_matches = sum(1 for source in test_case.expected_sources 
                           if source in answer)
        source_score = source_matches / len(test_case.expected_sources) if test_case.expected_sources else 1.0
        accuracy_scores['citation_accuracy_score'] = source_score
        
        # Check for expected frameworks
        framework_matches = sum(1 for framework in test_case.expected_frameworks
                              if framework.lower().replace(' ', '').replace('_', '') in answer.lower().replace(' ', '').replace('_', ''))
        framework_score = framework_matches / len(test_case.expected_frameworks) if test_case.expected_frameworks else 1.0
        accuracy_scores['framework_mapping_accuracy'] = framework_score
        
        # Overall completeness score
        completeness_score = (keyword_score + source_score + framework_score) / 3
        accuracy_scores['completeness_score'] = completeness_score
        
        # Relevance score (basic heuristic)
        relevance_score = min(1.0, keyword_score + 0.3)  # Keyword match + base relevance
        accuracy_scores['relevance_score'] = relevance_score
        
        # Citation analysis
        citations = self.metrics_collector._extract_citations(answer)
        sources = self.metrics_collector._extract_sources(answer)
        
        accuracy_scores.update({
            'citations_found': len(citations),
            'citations_accurate': min(len(citations), len(test_case.expected_sources)),
            'citations_missing': max(0, len(test_case.expected_sources) - len(citations)),
            'source_coverage_score': source_score,
            'key_facts_found': keyword_matches,
            'key_facts_expected': len(test_case.expected_answer_keywords),
            'numerical_accuracy_score': 0.8 if any(char.isdigit() for char in answer) else 0.5
        })
        
        return accuracy_scores
    
    def _store_test_metrics(self, test_case: TestCase, answer: str, 
                          quality_scores: Dict[str, float], accuracy_scores: Dict[str, float]):
        """Store all collected metrics in database."""
        try:
            # Get all metrics from collector
            all_metrics = self.metrics_collector.get_all_metrics()
            
            # Store performance metrics
            perf_metrics = all_metrics['performance'].copy()
            perf_metrics.update({
                'query_text': test_case.query,
                'query_type': test_case.query_type
            })
            self.test_db.store_performance_metrics(self.current_test_run_id, perf_metrics)
            
            # Store GPT metrics
            gpt_metrics = all_metrics['gpt'].copy()
            gpt_metrics['query_text'] = test_case.query
            self.test_db.store_gpt_metrics(self.current_test_run_id, gpt_metrics)
            
            # Store accuracy metrics
            accuracy_metrics = accuracy_scores.copy()
            accuracy_metrics.update({
                'query_text': test_case.query,
                'expected_answer': str(test_case.expected_answer_keywords),
                'actual_answer': answer[:1000]  # Truncate for storage
            })
            self.test_db.store_accuracy_metrics(self.current_test_run_id, accuracy_metrics)
            
            # Store quality metrics
            quality_metrics = all_metrics['quality'].copy()
            quality_metrics.update(quality_scores)
            quality_metrics.update({
                'query_text': test_case.query,
                'query_type': test_case.query_type
            })
            self.test_db.store_quality_metrics(self.current_test_run_id, quality_metrics)
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def _validate_test_result(self, test_case: TestCase, answer: str, accuracy_scores: Dict[str, float]):
        """Validate if test result meets minimum quality requirements."""
        overall_score = accuracy_scores['completeness_score']
        
        if overall_score < test_case.min_response_quality:
            raise AssertionError(
                f"Test quality score {overall_score:.2f} below minimum {test_case.min_response_quality}. "
                f"Answer: {answer[:200]}..."
            )
        
        # Check for critical failures
        if accuracy_scores['answer_accuracy_score'] == 0 and test_case.expected_answer_keywords:
            raise AssertionError("No expected keywords found in answer")
        
        if 'no_results' in test_case.expected_metrics and accuracy_scores['citations_found'] > 0:
            raise AssertionError("Expected no results but found citations")
    
    def _print_test_summary(self, summary: Dict[str, Any]):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🎯 TEST SUITE SUMMARY")
        print("=" * 60)
        
        results = summary['test_results']
        print(f"📊 Results: {results['passed']}/{results['total']} tests passed ({summary['overall_score']:.1%})")
        print(f"⏱️  Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"🔧 Environment: {summary['environment']}")
        print(f"📝 Git Hash: {summary.get('version_hash', 'unknown')}")
        
        if results['failed'] > 0:
            print(f"❌ Failed Tests: {results['failed']}")
        
        print(f"\n📈 Performance Tracking:")
        print(f"   Test Run ID: {summary['test_run_id']}")
        print(f"   Database: db/test_results.db")
        
        # Get recent trends
        trends = self.test_db.get_performance_trends(days=7)
        if len(trends) > 1:
            latest = trends[0]
            previous = trends[1]
            
            if latest['avg_response_time'] and previous['avg_response_time']:
                speed_change = ((latest['avg_response_time'] - previous['avg_response_time']) / previous['avg_response_time']) * 100
                speed_emoji = "🚀" if speed_change < 0 else "🐌" if speed_change > 10 else "⚡"
                print(f"   Speed Trend: {speed_change:+.1f}% {speed_emoji}")
            
            if latest['avg_accuracy'] and previous['avg_accuracy']:
                accuracy_change = ((latest['avg_accuracy'] - previous['avg_accuracy']) / previous['avg_accuracy']) * 100
                accuracy_emoji = "📈" if accuracy_change > 0 else "📉" if accuracy_change < -5 else "📊"
                print(f"   Accuracy Trend: {accuracy_change:+.1f}% {accuracy_emoji}")
        
        print("\n🎉 Test suite completed! Check database for detailed metrics and trends.")


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ImpactOS AI Layer Test Suite")
    parser.add_argument('--types', nargs='+', choices=['performance', 'accuracy', 'all'], 
                       default=['all'], help='Types of tests to run')
    parser.add_argument('--notes', type=str, help='Notes for this test run')
    parser.add_argument('--quick', action='store_true', help='Run quick test suite (simple tests only)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    runner = TestRunner()
    
    if args.quick:
        # Quick test with simple cases only
        test_cases = TestCases.get_test_cases_by_complexity('simple')[:5]
        print(f"🏃‍♂️ Running Quick Test Suite: {len(test_cases)} simple tests")
        
        # Initialize QuerySystem for quick tests with local database
        test_db_path = "db/impactos.db"
        if os.path.exists(test_db_path):
            print(f"🔗 Quick test using local database: {test_db_path}")
            runner.query_system = QuerySystem(test_db_path)
        else:
            print("⚠️ No local database found, using default")
            runner.query_system = QuerySystem()
        print("✅ QuerySystem initialized")
        
        # Create test run
        runner.current_test_run_id = runner.test_db.create_test_run(
            environment='quick_test',
            notes=f"Quick test: {args.notes or 'No notes'}"
        )
        
        passed = 0
        for test_case in test_cases:
            try:
                runner._run_single_test(test_case)
                passed += 1
                print(f"✅ {test_case.id}")
            except Exception as e:
                print(f"❌ {test_case.id}: {e}")
        
        print(f"Quick Test Results: {passed}/{len(test_cases)} passed")
    else:
        # Full test suite
        summary = runner.run_comprehensive_test_suite(
            test_types=args.types,
            notes=args.notes
        )


if __name__ == "__main__":
    main() 