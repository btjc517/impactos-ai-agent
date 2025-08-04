"""
Performance Tracker and Analysis for ImpactOS AI Layer Testing System.

This module provides analysis and reporting of test results over time,
helping track improvements, regressions, and performance trends.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from test_database import TestDatabase

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Analyzes and reports on test performance trends over time."""
    
    def __init__(self, db_path: str = "db/test_results.db"):
        """Initialize performance tracker."""
        self.test_db = TestDatabase(db_path)
    
    def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("📊 Generating Performance Report")
        print("=" * 50)
        
        # Get basic statistics
        test_history = self.test_db.get_test_history(limit=50)
        trends = self.test_db.get_performance_trends(days=days)
        
        if not test_history:
            return {"error": "No test data available"}
        
        # Analyze trends
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_period_days': days,
            'summary': self._generate_summary(test_history, trends),
            'performance_trends': self._analyze_performance_trends(trends),
            'accuracy_analysis': self._analyze_accuracy_trends(days),
            'cost_analysis': self._analyze_cost_trends(days),
            'recommendations': self._generate_recommendations(trends)
        }
        
        # Print report
        self._print_performance_report(report)
        
        return report
    
    def _generate_summary(self, test_history: List[Dict], trends: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics."""
        recent_tests = test_history[:10]
        
        summary = {
            'total_test_runs': len(test_history),
            'recent_test_runs': len(recent_tests),
            'avg_success_rate': sum(t.get('overall_score', 0) for t in recent_tests) / len(recent_tests) if recent_tests else 0,
            'last_test_date': test_history[0]['run_timestamp'] if test_history else None
        }
        
        if trends:
            latest_trend = trends[0]
            summary.update({
                'avg_response_time': latest_trend.get('avg_response_time'),
                'avg_cost_per_query': latest_trend.get('avg_cost'),
                'avg_accuracy': latest_trend.get('avg_accuracy')
            })
        
        return summary
    
    def _analyze_performance_trends(self, trends: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(trends) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate changes
        latest = trends[0]
        baseline = trends[-1]
        
        performance_analysis = {}
        
        # Response time trend
        if latest.get('avg_response_time') and baseline.get('avg_response_time'):
            time_change = ((latest['avg_response_time'] - baseline['avg_response_time']) 
                          / baseline['avg_response_time']) * 100
            performance_analysis['response_time_change_percent'] = time_change
            performance_analysis['response_time_trend'] = 'improving' if time_change < 0 else 'degrading'
        
        # Token usage trend
        if latest.get('avg_input_tokens') and baseline.get('avg_input_tokens'):
            token_change = ((latest['avg_input_tokens'] - baseline['avg_input_tokens']) 
                           / baseline['avg_input_tokens']) * 100
            performance_analysis['token_usage_change_percent'] = token_change
        
        # Cost trend
        if latest.get('avg_cost') and baseline.get('avg_cost'):
            cost_change = ((latest['avg_cost'] - baseline['avg_cost']) 
                          / baseline['avg_cost']) * 100
            performance_analysis['cost_change_percent'] = cost_change
            performance_analysis['cost_trend'] = 'improving' if cost_change < 0 else 'increasing'
        
        return performance_analysis
    
    def _analyze_accuracy_trends(self, days: int) -> Dict[str, Any]:
        """Analyze accuracy trends by query type."""
        try:
            with sqlite3.connect(self.test_db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        qm.query_type,
                        AVG(am.answer_accuracy_score) as avg_answer_accuracy,
                        AVG(am.citation_accuracy_score) as avg_citation_accuracy,
                        AVG(am.completeness_score) as avg_completeness,
                        COUNT(*) as test_count
                    FROM quality_metrics qm
                    JOIN accuracy_metrics am ON qm.test_run_id = am.test_run_id 
                        AND qm.query_text = am.query_text
                    JOIN test_runs tr ON qm.test_run_id = tr.id
                    WHERE tr.run_timestamp > datetime('now', '-{} days')
                    GROUP BY qm.query_type
                    ORDER BY avg_completeness DESC
                """.format(days))
                
                results = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'accuracy_by_query_type': results,
                    'best_performing_query_type': results[0]['query_type'] if results else None,
                    'total_accuracy_tests': sum(r['test_count'] for r in results)
                }
                
        except Exception as e:
            logger.error(f"Error analyzing accuracy trends: {e}")
            return {"error": str(e)}
    
    def _analyze_cost_trends(self, days: int) -> Dict[str, Any]:
        """Analyze cost trends and efficiency."""
        try:
            with sqlite3.connect(self.test_db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        DATE(tr.run_timestamp) as test_date,
                        SUM(gm.estimated_cost_usd) as daily_cost,
                        SUM(gm.total_input_tokens) as daily_input_tokens,
                        SUM(gm.total_output_tokens) as daily_output_tokens,
                        COUNT(*) as daily_queries
                    FROM gpt_metrics gm
                    JOIN test_runs tr ON gm.test_run_id = tr.id
                    WHERE tr.run_timestamp > datetime('now', '-{} days')
                    GROUP BY DATE(tr.run_timestamp)
                    ORDER BY test_date DESC
                """.format(days))
                
                daily_costs = [dict(row) for row in cursor.fetchall()]
                
                if daily_costs:
                    total_cost = sum(d['daily_cost'] or 0 for d in daily_costs)
                    total_queries = sum(d['daily_queries'] for d in daily_costs)
                    avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0
                    
                    return {
                        'total_cost_usd': total_cost,
                        'total_queries': total_queries,
                        'avg_cost_per_query': avg_cost_per_query,
                        'daily_costs': daily_costs[:7],  # Last 7 days
                        'cost_efficiency_trend': self._calculate_efficiency_trend(daily_costs)
                    }
                else:
                    return {"error": "No cost data available"}
                    
        except Exception as e:
            logger.error(f"Error analyzing cost trends: {e}")
            return {"error": str(e)}
    
    def _calculate_efficiency_trend(self, daily_costs: List[Dict]) -> str:
        """Calculate cost efficiency trend."""
        if len(daily_costs) < 2:
            return "insufficient_data"
        
        # Calculate cost per query for recent vs older data
        recent_data = daily_costs[:len(daily_costs)//2]
        older_data = daily_costs[len(daily_costs)//2:]
        
        recent_efficiency = sum(d['daily_cost'] or 0 for d in recent_data) / sum(d['daily_queries'] for d in recent_data)
        older_efficiency = sum(d['daily_cost'] or 0 for d in older_data) / sum(d['daily_queries'] for d in older_data)
        
        if recent_efficiency < older_efficiency:
            return "improving"
        elif recent_efficiency > older_efficiency * 1.1:  # 10% tolerance
            return "degrading"
        else:
            return "stable"
    
    def _generate_recommendations(self, trends: List[Dict]) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        if not trends or len(trends) < 2:
            return ["Insufficient data for recommendations. Run more tests to establish trends."]
        
        latest = trends[0]
        baseline = trends[-1]
        
        # Response time recommendations
        if latest.get('avg_response_time') and baseline.get('avg_response_time'):
            time_change = ((latest['avg_response_time'] - baseline['avg_response_time']) 
                          / baseline['avg_response_time']) * 100
            
            if time_change > 20:
                recommendations.append(
                    "⚠️ Response time has degraded by {:.1f}%. Consider optimizing FAISS index or reducing result limits.".format(time_change)
                )
            elif time_change < -10:
                recommendations.append(
                    "✅ Response time improved by {:.1f}%. Current optimizations are working well.".format(abs(time_change))
                )
        
        # Cost recommendations
        if latest.get('avg_cost') and baseline.get('avg_cost'):
            cost_change = ((latest['avg_cost'] - baseline['avg_cost']) 
                          / baseline['avg_cost']) * 100
            
            if cost_change > 15:
                recommendations.append(
                    "💰 API costs increased by {:.1f}%. Consider reducing GPT-4 token limits or using more efficient prompts.".format(cost_change)
                )
        
        # Accuracy recommendations
        if latest.get('avg_accuracy'):
            if latest['avg_accuracy'] < 0.8:
                recommendations.append(
                    "🎯 Accuracy is below 80%. Consider tuning similarity thresholds or improving test cases."
                )
            elif latest['avg_accuracy'] > 0.9:
                recommendations.append(
                    "🏆 Excellent accuracy above 90%. System is performing well."
                )
        
        # General recommendations
        if len(trends) < 5:
            recommendations.append(
                "📊 Run more tests to establish better performance baselines and detect trends."
            )
        
        if not recommendations:
            recommendations.append("📈 System performance is stable. Continue monitoring trends.")
        
        return recommendations
    
    def _print_performance_report(self, report: Dict[str, Any]):
        """Print formatted performance report."""
        print("\n📊 PERFORMANCE ANALYSIS REPORT")
        print("=" * 50)
        
        # Summary
        summary = report['summary']
        print(f"🎯 Summary (Last {report['analysis_period_days']} days):")
        print(f"   Total Test Runs: {summary['total_test_runs']}")
        print(f"   Average Success Rate: {summary['avg_success_rate']:.1%}")
        if summary.get('avg_response_time'):
            print(f"   Average Response Time: {summary['avg_response_time']:.2f}s")
        if summary.get('avg_cost_per_query'):
            print(f"   Average Cost per Query: ${summary['avg_cost_per_query']:.4f}")
        
        # Performance trends
        if 'error' not in report['performance_trends']:
            trends = report['performance_trends']
            print(f"\n⚡ Performance Trends:")
            
            if 'response_time_change_percent' in trends:
                change = trends['response_time_change_percent']
                emoji = "🚀" if change < 0 else "🐌"
                print(f"   Response Time: {change:+.1f}% {emoji}")
            
            if 'cost_change_percent' in trends:
                change = trends['cost_change_percent']
                emoji = "💰" if change > 0 else "💚"
                print(f"   Cost: {change:+.1f}% {emoji}")
        
        # Accuracy analysis
        if 'error' not in report['accuracy_analysis']:
            accuracy = report['accuracy_analysis']
            print(f"\n🎯 Accuracy Analysis:")
            print(f"   Total Accuracy Tests: {accuracy['total_accuracy_tests']}")
            if accuracy.get('best_performing_query_type'):
                print(f"   Best Query Type: {accuracy['best_performing_query_type']}")
            
            for query_type in accuracy.get('accuracy_by_query_type', [])[:3]:
                print(f"   {query_type['query_type'].title()}: {query_type['avg_completeness']:.1%} accuracy")
        
        # Cost analysis
        if 'error' not in report['cost_analysis']:
            costs = report['cost_analysis']
            print(f"\n💰 Cost Analysis:")
            print(f"   Total Cost: ${costs['total_cost_usd']:.4f}")
            print(f"   Total Queries: {costs['total_queries']}")
            print(f"   Cost per Query: ${costs['avg_cost_per_query']:.4f}")
            print(f"   Efficiency Trend: {costs['cost_efficiency_trend']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 50)
    
    def compare_configurations(self, config1_env: str, config2_env: str, days: int = 7) -> Dict[str, Any]:
        """Compare performance between different configurations."""
        try:
            with sqlite3.connect(self.test_db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get metrics for each configuration
                cursor.execute("""
                    SELECT 
                        tr.environment,
                        AVG(pm.total_time) as avg_response_time,
                        AVG(gm.estimated_cost_usd) as avg_cost,
                        AVG(am.completeness_score) as avg_accuracy,
                        COUNT(*) as test_count
                    FROM test_runs tr
                    LEFT JOIN performance_metrics pm ON tr.id = pm.test_run_id
                    LEFT JOIN gpt_metrics gm ON tr.id = gm.test_run_id
                    LEFT JOIN accuracy_metrics am ON tr.id = am.test_run_id
                    WHERE tr.environment IN (?, ?) 
                      AND tr.run_timestamp > datetime('now', '-{} days')
                    GROUP BY tr.environment
                """.format(days), (config1_env, config2_env))
                
                results = [dict(row) for row in cursor.fetchall()]
                
                if len(results) == 2:
                    config1 = next(r for r in results if r['environment'] == config1_env)
                    config2 = next(r for r in results if r['environment'] == config2_env)
                    
                    comparison = {
                        'config1': config1,
                        'config2': config2,
                        'comparison': {
                            'response_time_diff_percent': ((config2['avg_response_time'] - config1['avg_response_time']) 
                                                         / config1['avg_response_time'] * 100) if config1['avg_response_time'] else 0,
                            'cost_diff_percent': ((config2['avg_cost'] - config1['avg_cost']) 
                                                / config1['avg_cost'] * 100) if config1['avg_cost'] else 0,
                            'accuracy_diff_percent': ((config2['avg_accuracy'] - config1['avg_accuracy']) 
                                                    / config1['avg_accuracy'] * 100) if config1['avg_accuracy'] else 0
                        }
                    }
                    
                    return comparison
                else:
                    return {"error": f"Insufficient data for comparison between {config1_env} and {config2_env}"}
                    
        except Exception as e:
            logger.error(f"Error comparing configurations: {e}")
            return {"error": str(e)}


def main():
    """Main entry point for performance tracking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ImpactOS AI Performance Tracker")
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--compare', nargs=2, help='Compare two environments (e.g., development production)')
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker()
    
    if args.compare:
        env1, env2 = args.compare
        print(f"🔍 Comparing {env1} vs {env2}")
        comparison = tracker.compare_configurations(env1, env2, args.days)
        
        if 'error' not in comparison:
            print(f"\nConfiguration Comparison ({args.days} days):")
            print(f"Response Time: {comparison['comparison']['response_time_diff_percent']:+.1f}%")
            print(f"Cost: {comparison['comparison']['cost_diff_percent']:+.1f}%") 
            print(f"Accuracy: {comparison['comparison']['accuracy_diff_percent']:+.1f}%")
        else:
            print(f"❌ {comparison['error']}")
    else:
        report = tracker.generate_performance_report(args.days)


if __name__ == "__main__":
    main() 