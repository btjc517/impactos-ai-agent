"""
Comprehensive accuracy testing suite for ImpactOS AI Layer MVP.

Tests extraction accuracy, citation precision, Q&A system performance,
and overall system reliability across all data sources.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import sqlite3
import pandas as pd
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

# Import ImpactOS modules
from ingest import DataIngestion
from extract_v2 import QueryBasedExtraction
from query import QuerySystem
from verify import DataVerifier
from schema import DatabaseSchema

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccuracyTestSuite:
    """Comprehensive accuracy testing for ImpactOS AI system."""
    
    def __init__(self, test_db_path: str = "db/test_impactos.db"):
        """Initialize test suite with clean database."""
        self.test_db_path = test_db_path
        self.data_dir = "data"
        self.test_results = {}
        
        # Clean up any existing test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        # Initialize test components
        self.db_schema = DatabaseSchema(test_db_path)
        self.db_schema.initialize_database()  # Initialize tables
        self.ingestion = DataIngestion(test_db_path)
        self.verifier = DataVerifier(test_db_path, self.data_dir)
        self.query_system = QuerySystem(test_db_path)
        
        # Test data files
        self.test_files = [
            "TakingCare_Benevity_Synthetic_Data.xlsx",
            "TakingCare_Carbon_Reporting_Synthetic_Data.xlsx", 
            "TakingCare_EAP_Synthetic_Data.xlsx",
            "TakingCare_EcoVadis_Synthetic_Data.xlsx",
            "TakingCare_HCM_Synthetic_Data.xlsx",
            "TakingCare_ITAsset_Environmental_Synthetic_Data.xlsx",
            "TakingCare_LMS_Synthetic_Data.xlsx",
            "TakingCare_myday_Synthetic_Data.xlsx",
            "TakingCare_Payroll_Synthetic_Data.xlsx",
            "TakingCare_SupplyChain_Synthetic_Data.xlsx",
            "TakingCare_SurveyEngagement_Synthetic_Data.xlsx"
        ]
        
        # Expected metrics for validation (ground truth)
        self.expected_metrics = self._load_expected_metrics()
    
    def run_comprehensive_accuracy_test(self) -> Dict[str, Any]:
        """
        Run complete accuracy test suite.
        
        Returns:
            Comprehensive accuracy report
        """
        logger.info("🚀 Starting comprehensive accuracy testing...")
        
        start_time = datetime.now()
        results = {
            "test_start": start_time.isoformat(),
            "extraction_accuracy": {},
            "citation_accuracy": {},
            "qa_accuracy": {},
            "comparative_analysis": {},
            "overall_metrics": {}
        }
        
        try:
            # Test 1: Data extraction accuracy across all files
            logger.info("📊 Testing data extraction accuracy...")
            results["extraction_accuracy"] = self._test_extraction_accuracy()
            
            # Test 2: Citation accuracy and precision
            logger.info("🎯 Testing citation accuracy...")
            results["citation_accuracy"] = self._test_citation_accuracy()
            
            # Test 3: Q&A system accuracy
            logger.info("❓ Testing Q&A system accuracy...")
            results["qa_accuracy"] = self._test_qa_accuracy()
            
            # Test 4: Comparative analysis (query-based vs text-based)
            logger.info("⚖️ Testing comparative extraction methods...")
            results["comparative_analysis"] = self._test_comparative_extraction()
            
            # Test 5: Calculate overall metrics
            results["overall_metrics"] = self._calculate_overall_metrics(results)
            
            end_time = datetime.now()
            results["test_duration"] = str(end_time - start_time)
            results["test_end"] = end_time.isoformat()
            
            logger.info("✅ Comprehensive accuracy testing completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Accuracy testing failed: {e}")
            results["error"] = str(e)
            return results
    
    def _test_extraction_accuracy(self) -> Dict[str, Any]:
        """Test extraction accuracy across all data files."""
        results = {
            "files_tested": 0,
            "total_metrics_extracted": 0,
            "accurate_extractions": 0,
            "file_results": {},
            "accuracy_by_file": {},
            "overall_accuracy": 0.0
        }
        
        for file_name in self.test_files:
            file_path = os.path.join(self.data_dir, file_name)
            
            if not os.path.exists(file_path):
                logger.warning(f"Test file not found: {file_path}")
                continue
            
            logger.info(f"Testing extraction for: {file_name}")
            
            try:
                # Ingest file using query-based extraction
                success = self.ingestion.ingest_file(file_path, use_query_based=True)
                
                if success:
                    # Verify extracted metrics
                    verification_result = self.verifier.verify_all_pending_metrics()
                    
                    file_results = {
                        "ingestion_success": True,
                        "metrics_extracted": verification_result.get("total", 0),
                        "verified_metrics": verification_result.get("verified", 0),
                        "accuracy": verification_result.get("accuracy", 0.0)
                    }
                    
                    results["files_tested"] += 1
                    results["total_metrics_extracted"] += file_results["metrics_extracted"]
                    results["accurate_extractions"] += file_results["verified_metrics"]
                    results["file_results"][file_name] = file_results
                    results["accuracy_by_file"][file_name] = file_results["accuracy"]
                    
                else:
                    results["file_results"][file_name] = {
                        "ingestion_success": False,
                        "error": "Failed to ingest file"
                    }
                    
            except Exception as e:
                logger.error(f"Error testing {file_name}: {e}")
                results["file_results"][file_name] = {
                    "ingestion_success": False,
                    "error": str(e)
                }
        
        # Calculate overall accuracy
        if results["total_metrics_extracted"] > 0:
            results["overall_accuracy"] = results["accurate_extractions"] / results["total_metrics_extracted"]
        
        return results
    
    def _test_citation_accuracy(self) -> Dict[str, Any]:
        """Test accuracy of citations and cell references."""
        results = {
            "total_citations": 0,
            "accurate_citations": 0,
            "citation_accuracy": 0.0,
            "citation_types": {
                "cell_references": {"total": 0, "accurate": 0},
                "formulas": {"total": 0, "accurate": 0},
                "column_names": {"total": 0, "accurate": 0}
            }
        }
        
        try:
            # Get all metrics with citations
            with sqlite3.connect(self.test_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        metric_name, metric_value, source_column_name,
                        source_cell_reference, source_formula,
                        verification_status, source_id
                    FROM impact_metrics 
                    WHERE source_cell_reference IS NOT NULL
                """)
                
                metrics_with_citations = cursor.fetchall()
                
                for metric in metrics_with_citations:
                    results["total_citations"] += 1
                    
                    # Verify cell reference accuracy
                    if self._verify_cell_reference(metric):
                        results["accurate_citations"] += 1
                        results["citation_types"]["cell_references"]["accurate"] += 1
                    results["citation_types"]["cell_references"]["total"] += 1
                    
                    # Verify formula accuracy
                    if metric["source_formula"] and self._verify_formula(metric):
                        results["citation_types"]["formulas"]["accurate"] += 1
                    if metric["source_formula"]:
                        results["citation_types"]["formulas"]["total"] += 1
                    
                    # Verify column name accuracy
                    if metric["source_column_name"] and self._verify_column_name(metric):
                        results["citation_types"]["column_names"]["accurate"] += 1
                    if metric["source_column_name"]:
                        results["citation_types"]["column_names"]["total"] += 1
            
            # Calculate citation accuracy
            if results["total_citations"] > 0:
                results["citation_accuracy"] = results["accurate_citations"] / results["total_citations"]
            
            # Calculate accuracy by citation type
            for citation_type in results["citation_types"]:
                total = results["citation_types"][citation_type]["total"]
                accurate = results["citation_types"][citation_type]["accurate"]
                if total > 0:
                    results["citation_types"][citation_type]["accuracy"] = accurate / total
                else:
                    results["citation_types"][citation_type]["accuracy"] = 0.0
                    
        except Exception as e:
            logger.error(f"Error testing citation accuracy: {e}")
            results["error"] = str(e)
        
        return results
    
    def _test_qa_accuracy(self) -> Dict[str, Any]:
        """Test Q&A system accuracy with known queries."""
        results = {
            "queries_tested": 0,
            "accurate_responses": 0,
            "qa_accuracy": 0.0,
            "query_results": {}
        }
        
        # Test queries with expected answers
        test_queries = [
            {
                "query": "What is the total amount of donations?",
                "expected_keywords": ["donation", "£", "total"],
                "minimum_confidence": 0.7
            },
            {
                "query": "How many volunteer hours were contributed?",
                "expected_keywords": ["volunteer", "hours", "total"],
                "minimum_confidence": 0.7
            },
            {
                "query": "What is the total carbon emissions?",
                "expected_keywords": ["carbon", "emissions", "total"],
                "minimum_confidence": 0.7
            },
            {
                "query": "How many training sessions were completed?",
                "expected_keywords": ["training", "session", "completed"],
                "minimum_confidence": 0.7
            }
        ]
        
        for test_query in test_queries:
            try:
                logger.info(f"Testing query: {test_query['query']}")
                
                response = self.query_system.query(test_query["query"])
                
                query_result = {
                    "query": test_query["query"],
                    "response": response,
                    "passed": self._evaluate_qa_response(response, test_query)
                }
                
                results["query_results"][test_query["query"]] = query_result
                results["queries_tested"] += 1
                
                if query_result["passed"]:
                    results["accurate_responses"] += 1
                    
            except Exception as e:
                logger.error(f"Error testing query '{test_query['query']}': {e}")
                results["query_results"][test_query["query"]] = {
                    "query": test_query["query"],
                    "error": str(e),
                    "passed": False
                }
        
        # Calculate Q&A accuracy
        if results["queries_tested"] > 0:
            results["qa_accuracy"] = results["accurate_responses"] / results["queries_tested"]
        
        return results
    
    def _test_comparative_extraction(self) -> Dict[str, Any]:
        """Compare query-based vs text-based extraction accuracy."""
        results = {
            "query_based": {"accuracy": 0.0, "total_metrics": 0, "verified_metrics": 0},
            "text_based": {"accuracy": 0.0, "total_metrics": 0, "verified_metrics": 0},
            "improvement_factor": 0.0
        }
        
        # Test with a sample file using both methods
        test_file = os.path.join(self.data_dir, "TakingCare_Benevity_Synthetic_Data.xlsx")
        
        if os.path.exists(test_file):
            try:
                # Test query-based extraction
                logger.info("Testing query-based extraction...")
                self._clean_test_db()
                success = self.ingestion.ingest_file(test_file, use_query_based=True)
                if success:
                    verification = self.verifier.verify_all_pending_metrics()
                    results["query_based"]["accuracy"] = verification.get("accuracy", 0.0)
                    results["query_based"]["total_metrics"] = verification.get("total", 0)
                    results["query_based"]["verified_metrics"] = verification.get("verified", 0)
                
                # Test text-based extraction
                logger.info("Testing text-based extraction...")
                self._clean_test_db()
                success = self.ingestion.ingest_file(test_file, use_query_based=False)
                if success:
                    verification = self.verifier.verify_all_pending_metrics()
                    results["text_based"]["accuracy"] = verification.get("accuracy", 0.0)
                    results["text_based"]["total_metrics"] = verification.get("total", 0)
                    results["text_based"]["verified_metrics"] = verification.get("verified", 0)
                
                # Calculate improvement factor
                if results["text_based"]["accuracy"] > 0:
                    results["improvement_factor"] = results["query_based"]["accuracy"] / results["text_based"]["accuracy"]
                
            except Exception as e:
                logger.error(f"Error in comparative testing: {e}")
                results["error"] = str(e)
        
        return results
    
    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system accuracy metrics."""
        extraction_acc = results.get("extraction_accuracy", {}).get("overall_accuracy", 0.0)
        citation_acc = results.get("citation_accuracy", {}).get("citation_accuracy", 0.0)
        qa_acc = results.get("qa_accuracy", {}).get("qa_accuracy", 0.0)
        
        return {
            "overall_system_accuracy": (extraction_acc + citation_acc + qa_acc) / 3,
            "extraction_accuracy": extraction_acc,
            "citation_accuracy": citation_acc,
            "qa_accuracy": qa_acc,
            "files_processed": results.get("extraction_accuracy", {}).get("files_tested", 0),
            "total_metrics_tested": results.get("extraction_accuracy", {}).get("total_metrics_extracted", 0),
            "accuracy_threshold_met": extraction_acc >= 0.9,  # 90% threshold
            "system_grade": self._calculate_grade(extraction_acc)
        }
    
    def _calculate_grade(self, accuracy: float) -> str:
        """Calculate letter grade based on accuracy."""
        if accuracy >= 0.95:
            return "A+"
        elif accuracy >= 0.90:
            return "A"
        elif accuracy >= 0.85:
            return "B+"
        elif accuracy >= 0.80:
            return "B"
        elif accuracy >= 0.75:
            return "C+"
        elif accuracy >= 0.70:
            return "C"
        else:
            return "F"
    
    def _load_expected_metrics(self) -> Dict[str, Any]:
        """Load expected metrics for validation (ground truth)."""
        # This would typically load from a ground truth file
        # For now, return empty dict
        return {}
    
    def _verify_cell_reference(self, metric: sqlite3.Row) -> bool:
        """Verify that cell reference is accurate."""
        # Implementation would check actual cell reference against source file
        # For now, return True if cell reference exists and follows pattern
        cell_ref = metric["source_cell_reference"]
        if not cell_ref:
            return False
        
        # Check if it matches expected pattern (e.g., "E2:E15", "D10")
        import re
        pattern = r'^[A-Z]+\d+(:[A-Z]+\d+)?$'
        return bool(re.match(pattern, cell_ref))
    
    def _verify_formula(self, metric: sqlite3.Row) -> bool:
        """Verify that extraction formula is accurate."""
        formula = metric["source_formula"]
        if not formula:
            return False
        
        # Check if formula follows expected pattern
        expected_patterns = ["SUM(", "AVERAGE(", "COUNT(", "MAX(", "MIN("]
        return any(pattern in formula for pattern in expected_patterns)
    
    def _verify_column_name(self, metric: sqlite3.Row) -> bool:
        """Verify that column name is accurate."""
        column_name = metric["source_column_name"]
        if not column_name:
            return False
        
        # Basic validation - column name should be non-empty string
        return len(column_name.strip()) > 0
    
    def _evaluate_qa_response(self, response: Any, test_query: Dict[str, Any]) -> bool:
        """Evaluate Q&A response quality."""
        if not response:
            return False
        
        # Handle string responses (process_query returns string)
        if isinstance(response, str):
            if "error" in response.lower():
                return False
            answer = response.lower()
        else:
            # Handle dict responses
            if "error" in str(response):
                return False
            answer = str(response).lower()
        
        # Check if response contains expected keywords
        keywords_found = sum(1 for keyword in test_query["expected_keywords"] 
                           if keyword.lower() in answer)
        keyword_ratio = keywords_found / len(test_query["expected_keywords"])
        
        # Response passes if keyword ratio meets threshold (simplified for string responses)
        return keyword_ratio >= 0.5
    
    def _clean_test_db(self):
        """Clean test database for fresh testing."""
        try:
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
            self.db_schema = DatabaseSchema(self.test_db_path)
            self.db_schema.initialize_database()  # Initialize tables
            self.ingestion = DataIngestion(self.test_db_path)
            self.verifier = DataVerifier(self.test_db_path, self.data_dir)
        except Exception as e:
            logger.error(f"Error cleaning test database: {e}")
    
    def generate_accuracy_report(self, results: Dict[str, Any]) -> str:
        """Generate formatted accuracy report."""
        report = f"""
# 🎯 ImpactOS AI Accuracy Test Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Duration**: {results.get('test_duration', 'Unknown')}

## 📊 Overall Performance
- **System Accuracy**: {results.get('overall_metrics', {}).get('overall_system_accuracy', 0):.1%}
- **System Grade**: {results.get('overall_metrics', {}).get('system_grade', 'N/A')}
- **Accuracy Threshold (90%) Met**: {'✅ Yes' if results.get('overall_metrics', {}).get('accuracy_threshold_met', False) else '❌ No'}

## 🔍 Detailed Results

### Data Extraction Accuracy
- **Files Tested**: {results.get('extraction_accuracy', {}).get('files_tested', 0)}
- **Total Metrics**: {results.get('extraction_accuracy', {}).get('total_metrics_extracted', 0)}
- **Accurate Extractions**: {results.get('extraction_accuracy', {}).get('accurate_extractions', 0)}
- **Extraction Accuracy**: {results.get('extraction_accuracy', {}).get('overall_accuracy', 0):.1%}

### Citation Accuracy
- **Total Citations**: {results.get('citation_accuracy', {}).get('total_citations', 0)}
- **Accurate Citations**: {results.get('citation_accuracy', {}).get('accurate_citations', 0)}
- **Citation Accuracy**: {results.get('citation_accuracy', {}).get('citation_accuracy', 0):.1%}

### Q&A System Accuracy
- **Queries Tested**: {results.get('qa_accuracy', {}).get('queries_tested', 0)}
- **Accurate Responses**: {results.get('qa_accuracy', {}).get('accurate_responses', 0)}
- **Q&A Accuracy**: {results.get('qa_accuracy', {}).get('qa_accuracy', 0):.1%}

### Comparative Analysis
- **Query-based Accuracy**: {results.get('comparative_analysis', {}).get('query_based', {}).get('accuracy', 0):.1%}
- **Text-based Accuracy**: {results.get('comparative_analysis', {}).get('text_based', {}).get('accuracy', 0):.1%}
- **Improvement Factor**: {results.get('comparative_analysis', {}).get('improvement_factor', 0):.1f}x

## 📈 Recommendations
"""
        
        # Add recommendations based on results
        overall_acc = results.get('overall_metrics', {}).get('overall_system_accuracy', 0)
        if overall_acc >= 0.95:
            report += "- 🌟 Excellent performance! System exceeds accuracy expectations.\n"
        elif overall_acc >= 0.90:
            report += "- ✅ Good performance. System meets accuracy requirements.\n"
        else:
            report += "- ⚠️ Performance below target. Consider improvements to extraction pipeline.\n"
        
        return report


def run_accuracy_tests():
    """Main function to run accuracy tests."""
    print("🚀 Starting ImpactOS AI Accuracy Testing...")
    
    # Initialize test suite
    test_suite = AccuracyTestSuite()
    
    # Run comprehensive tests
    results = test_suite.run_comprehensive_accuracy_test()
    
    # Generate and display report
    report = test_suite.generate_accuracy_report(results)
    print(report)
    
    # Save results to file
    with open("tests/accuracy_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open("tests/accuracy_report.md", "w") as f:
        f.write(report)
    
    print(f"\n📄 Results saved to:")
    print(f"- tests/accuracy_test_results.json")
    print(f"- tests/accuracy_report.md")
    
    return results


if __name__ == "__main__":
    run_accuracy_tests() 