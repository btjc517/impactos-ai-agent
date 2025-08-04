"""
Test Cases for ImpactOS AI Layer Testing System.

This module defines comprehensive test cases covering different query types,
complexity levels, and expected outcomes for accuracy measurement.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TestCase:
    """Individual test case definition."""
    id: str
    query: str
    query_type: str  # aggregation, descriptive, analytical
    complexity: str  # simple, medium, complex
    expected_answer_keywords: List[str]
    expected_sources: List[str]
    expected_metrics: Dict[str, Any]
    expected_frameworks: List[str]
    description: str
    min_response_quality: float = 0.7


class TestCases:
    """Collection of standardized test cases."""
    
    @staticmethod
    def get_all_test_cases() -> List[TestCase]:
        """Get all test cases for comprehensive testing."""
        return [
            # === AGGREGATION QUERIES ===
            TestCase(
                id="agg_001",
                query="How much was donated to charity?",
                query_type="aggregation",
                complexity="simple",
                expected_answer_keywords=["total", "donated", "charity", "£", "$"],
                expected_sources=["TakingCare_Benevity_Synthetic_Data.xlsx", "TakingCare_Payroll_Synthetic_Data.xlsx"],
                expected_metrics={"numerical_values": True, "currency_format": True},
                expected_frameworks=["UK Social Value Model", "B Corp"],
                description="Simple aggregation of charity donations across sources",
                min_response_quality=0.8
            ),
            
            TestCase(
                id="agg_002", 
                query="What is the total amount of volunteering hours?",
                query_type="aggregation",
                complexity="simple",
                expected_answer_keywords=["total", "volunteering", "hours", "volunteer"],
                expected_sources=["TakingCare_HCM_Synthetic_Data.xlsx"],
                expected_metrics={"numerical_values": True, "units": "hours"},
                expected_frameworks=["UN SDGs", "TOMs"],
                description="Aggregation of volunteering hours from HCM data"
            ),
            
            TestCase(
                id="agg_003",
                query="What are the total carbon emissions across all scopes?",
                query_type="aggregation", 
                complexity="medium",
                expected_answer_keywords=["carbon", "emissions", "scope", "total", "kg", "CO2"],
                expected_sources=["TakingCare_Carbon_Reporting_Synthetic_Data.xlsx"],
                expected_metrics={"numerical_values": True, "scope_breakdown": True},
                expected_frameworks=["UN SDGs", "B Corp"],
                description="Complex aggregation across Scope 1, 2, and 3 emissions"
            ),
            
            TestCase(
                id="agg_004",
                query="How much local spending was there in total?",
                query_type="aggregation",
                complexity="medium", 
                expected_answer_keywords=["local", "spending", "total", "procurement", "£", "$"],
                expected_sources=["TakingCare_SupplyChain_Synthetic_Data.xlsx"],
                expected_metrics={"numerical_values": True, "currency_format": True},
                expected_frameworks=["UK Social Value Model", "TOMs"],
                description="Aggregation of local procurement spending"
            ),
            
            # === DESCRIPTIVE QUERIES ===
            TestCase(
                id="desc_001",
                query="What sustainability initiatives exist?",
                query_type="descriptive",
                complexity="medium",
                expected_answer_keywords=["sustainability", "carbon", "volunteering", "donations", "training"],
                expected_sources=["TakingCare_Carbon_Reporting_Synthetic_Data.xlsx", "TakingCare_HCM_Synthetic_Data.xlsx"],
                expected_metrics={"initiative_count": 3, "diversity": True},
                expected_frameworks=["UN SDGs", "B Corp", "UK Social Value Model"],
                description="Broad overview of sustainability initiatives"
            ),
            
            TestCase(
                id="desc_002",
                query="What employee development programs are available?",
                query_type="descriptive",
                complexity="simple",
                expected_answer_keywords=["training", "development", "education", "hours", "employee"],
                expected_sources=["TakingCare_HCM_Synthetic_Data.xlsx", "TakingCare_LMS_Synthetic_Data.xlsx"],
                expected_metrics={"program_diversity": True},
                expected_frameworks=["UN SDGs", "B Corp"],
                description="Overview of employee development initiatives"
            ),
            
            TestCase(
                id="desc_003",
                query="What diversity and inclusion metrics are tracked?",
                query_type="descriptive", 
                complexity="medium",
                expected_answer_keywords=["diversity", "inclusion", "gender", "ethnicity", "representation"],
                expected_sources=["TakingCare_HCM_Synthetic_Data.xlsx"],
                expected_metrics={"diversity_categories": True},
                expected_frameworks=["UN SDGs", "B Corp"],
                description="Diversity and inclusion metric overview"
            ),
            
            # === ANALYTICAL QUERIES ===
            TestCase(
                id="anal_001",
                query="How does our carbon footprint compare across different scopes?",
                query_type="analytical",
                complexity="complex",
                expected_answer_keywords=["scope", "compare", "carbon", "footprint", "analysis"],
                expected_sources=["TakingCare_Carbon_Reporting_Synthetic_Data.xlsx"],
                expected_metrics={"comparative_analysis": True, "scope_breakdown": True},
                expected_frameworks=["UN SDGs", "B Corp"],
                description="Comparative analysis of carbon emissions by scope"
            ),
            
            TestCase(
                id="anal_002", 
                query="What is the relationship between training hours and employee engagement?",
                query_type="analytical",
                complexity="complex",
                expected_answer_keywords=["training", "engagement", "relationship", "correlation", "employee"],
                expected_sources=["TakingCare_HCM_Synthetic_Data.xlsx", "TakingCare_SurveyEngagement_Synthetic_Data.xlsx"],
                expected_metrics={"correlation_analysis": True},
                expected_frameworks=["UN SDGs", "B Corp"],
                description="Analysis of training-engagement relationship"
            ),
            
            TestCase(
                id="anal_003",
                query="Which social value areas show the strongest performance?",
                query_type="analytical",
                complexity="complex",
                expected_answer_keywords=["social value", "performance", "strongest", "areas", "comparison"],
                expected_sources=["TakingCare_Benevity_Synthetic_Data.xlsx", "TakingCare_HCM_Synthetic_Data.xlsx"],
                expected_metrics={"performance_ranking": True, "multi_area_analysis": True},
                expected_frameworks=["UK Social Value Model", "B Corp", "UN SDGs"],
                description="Cross-domain performance analysis"
            ),
            
            # === FRAMEWORK-SPECIFIC QUERIES ===
            TestCase(
                id="frame_001",
                query="How do our metrics map to the UN SDGs?",
                query_type="descriptive",
                complexity="medium",
                expected_answer_keywords=["UN SDGs", "sustainable development", "goals", "mapping"],
                expected_sources=["TakingCare_HCM_Synthetic_Data.xlsx", "TakingCare_Carbon_Reporting_Synthetic_Data.xlsx"],
                expected_metrics={"framework_mapping": True, "sdg_coverage": True},
                expected_frameworks=["UN SDGs"],
                description="SDG framework mapping overview"
            ),
            
            TestCase(
                id="frame_002",
                query="What B Corp metrics do we track?",
                query_type="descriptive",
                complexity="simple",
                expected_answer_keywords=["B Corp", "metrics", "track", "certification"],
                expected_sources=["TakingCare_HCM_Synthetic_Data.xlsx", "TakingCare_Benevity_Synthetic_Data.xlsx"],
                expected_metrics={"bcorp_categories": True},
                expected_frameworks=["B Corp"],
                description="B Corp framework metric tracking"
            ),
            
            # === COMPLEX MULTI-SOURCE QUERIES ===
            TestCase(
                id="complex_001",
                query="What is our overall social impact across all measured areas?",
                query_type="analytical",
                complexity="complex",
                expected_answer_keywords=["overall", "social impact", "measured", "areas", "comprehensive"],
                expected_sources=[
                    "TakingCare_Benevity_Synthetic_Data.xlsx",
                    "TakingCare_HCM_Synthetic_Data.xlsx", 
                    "TakingCare_Carbon_Reporting_Synthetic_Data.xlsx",
                    "TakingCare_SupplyChain_Synthetic_Data.xlsx"
                ],
                expected_metrics={"multi_source": True, "comprehensive_analysis": True},
                expected_frameworks=["UK Social Value Model", "UN SDGs", "B Corp", "TOMs"],
                description="Comprehensive multi-source impact analysis",
                min_response_quality=0.9
            ),
            
            TestCase(
                id="complex_002",
                query="How do employee assistance programs contribute to social value?",
                query_type="analytical",
                complexity="complex",
                expected_answer_keywords=["employee assistance", "social value", "contribute", "programs"],
                expected_sources=["TakingCare_EAP_Synthetic_Data.xlsx", "TakingCare_HCM_Synthetic_Data.xlsx"],
                expected_metrics={"program_analysis": True, "value_contribution": True},
                expected_frameworks=["UK Social Value Model", "B Corp"],
                description="EAP social value contribution analysis"
            ),
            
            # === EDGE CASES AND ERROR HANDLING ===
            TestCase(
                id="edge_001", 
                query="What is the carbon emission of purple elephants?",
                query_type="analytical",
                complexity="simple",
                expected_answer_keywords=["no data", "not found", "insufficient", "cannot find"],
                expected_sources=[],
                expected_metrics={"no_results": True},
                expected_frameworks=[],
                description="Query with no matching data - should handle gracefully"
            ),
            
            TestCase(
                id="edge_002",
                query="Show me everything about everything",
                query_type="descriptive", 
                complexity="simple",
                expected_answer_keywords=["comprehensive", "overview", "multiple", "areas"],
                expected_sources=["TakingCare_HCM_Synthetic_Data.xlsx"],  # Should get at least some data
                expected_metrics={"overly_broad": True},
                expected_frameworks=["UN SDGs", "B Corp"],
                description="Overly broad query - should provide structured response"
            ),
            
            # === PERFORMANCE TEST QUERIES ===
            TestCase(
                id="perf_001",
                query="Calculate total social value across all frameworks",
                query_type="aggregation",
                complexity="complex",
                expected_answer_keywords=["total", "social value", "frameworks", "calculation"],
                expected_sources=[
                    "TakingCare_Benevity_Synthetic_Data.xlsx",
                    "TakingCare_HCM_Synthetic_Data.xlsx",
                    "TakingCare_Carbon_Reporting_Synthetic_Data.xlsx"
                ],
                expected_metrics={"high_computation": True, "multi_framework": True},
                expected_frameworks=["UK Social Value Model", "UN SDGs", "B Corp", "TOMs"],
                description="Performance test - complex calculation across frameworks"
            )
        ]
    
    @staticmethod
    def get_test_cases_by_type(query_type: str) -> List[TestCase]:
        """Get test cases filtered by query type."""
        return [tc for tc in TestCases.get_all_test_cases() if tc.query_type == query_type]
    
    @staticmethod
    def get_test_cases_by_complexity(complexity: str) -> List[TestCase]:
        """Get test cases filtered by complexity."""
        return [tc for tc in TestCases.get_all_test_cases() if tc.complexity == complexity]
    
    @staticmethod
    def get_performance_test_cases() -> List[TestCase]:
        """Get test cases specifically for performance testing."""
        all_cases = TestCases.get_all_test_cases()
        return [tc for tc in all_cases if tc.id.startswith('perf_') or tc.complexity == 'complex']
    
    @staticmethod
    def get_accuracy_test_cases() -> List[TestCase]:
        """Get test cases with high accuracy requirements."""
        all_cases = TestCases.get_all_test_cases()
        return [tc for tc in all_cases if tc.min_response_quality >= 0.8] 