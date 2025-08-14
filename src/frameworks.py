"""
Framework Mapping System for ImpactOS AI Layer MVP.

Maps extracted social value metrics to standard frameworks:
- UK Social Value Model (MAC - Measurement and Accounting Criteria)
- UN Sustainable Development Goals (SDGs)
- Themes, Outcomes and Measures (TOMs) 
- B Corporation Assessment framework

This enables standardized social value reporting and benchmarking.
"""

import sqlite3
import os
from typing import Dict, List, Any, Optional
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameworkMapper:
    """Maps social value metrics to standard reporting frameworks."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        """Initialize framework mapper with database connection."""
        self.db_path = db_path
        
        # Framework definitions are loaded from the concept graph (no hardcoding)
        try:
            from semantic_resolver import ConceptGraph
            g = ConceptGraph(self.db_path)
            # Build a lightweight view of frameworks for reporting
            frameworks = {}
            for f in g.list_concepts('framework'):
                frameworks[f['key'].upper()] = {
                    'name': f['name'],
                    'categories': {},
                }
            self.frameworks = frameworks
        except Exception:
            self.frameworks = {}
    
    def _load_framework_definitions(self) -> Dict[str, Any]:
        # Deprecated; retained for backward compatibility
        return self.frameworks
    
    def map_metric_to_frameworks(self, metric_name: str, metric_category: str, 
                                metric_context: str = "") -> Dict[str, List[str]]:
        """
        Map a metric to relevant framework categories.
        
        Args:
            metric_name: Name of the metric
            metric_category: Category (e.g., 'community_engagement')
            metric_context: Additional context description
            
        Returns:
            Dictionary mapping framework names to relevant categories
        """
        mappings = {}
        
        # Semantic mapping via concept graph (no hardcoded rules)
        try:
            from semantic_resolver import SemanticResolver
            resolver = SemanticResolver(self.db_path)
            full_text = f"{metric_name} | {metric_category} | {metric_context}".strip()
            # Resolve framework first
            results: Dict[str, List[str]] = {}
            for framework in ['uk_sv_model', 'un_sdgs', 'toms', 'b_corp']:
                fr = resolver.resolve('framework', framework)
                if fr.get('outcome') == 'accepted':
                    # Resolve categories within that framework by similarity against the metric text
                    # For now, store framework key with a placeholder category resolution to be enriched later
                    results[fr['key'].upper()] = []
            return results
        except Exception:
            # Fallback to empty when resolver unavailable
            return {}
        
        return mappings
    
    def _map_to_uk_sv_model(self, context: str) -> List[str]:
        """Map metric to UK Social Value Model categories."""
        mappings = []
        
        # Employment mappings
        if any(term in context for term in ['employment', 'job', 'salary', 'apprentice', 'skills']):
            mappings.append("3.0")  # Employment
            if 'apprentice' in context:
                mappings.append("3.2")  # Apprenticeships
            if 'skills' in context or 'training' in context:
                mappings.append("3.3")  # Skills development
        
        # Environment mappings
        if any(term in context for term in ['carbon', 'emission', 'co2', 'environment', 'waste', 'energy']):
            mappings.append("4.0")  # Environment
            if 'carbon' in context or 'co2' in context or 'emission' in context:
                mappings.append("4.1")  # Carbon emissions reduction
            if 'waste' in context:
                mappings.append("4.2")  # Waste reduction
        
        # Community engagement mappings
        if any(term in context for term in ['volunteer', 'community', 'engagement', 'charity', 'donation']):
            mappings.append("8.0")  # Community Engagement
            if 'volunteer' in context:
                mappings.append("8.1")  # Community volunteering
            if 'charity' in context or 'donation' in context:
                mappings.append("8.2")  # Charitable giving
        
        # Education mappings
        if any(term in context for term in ['education', 'learning', 'training', 'course']):
            mappings.append("2.0")  # Education
        
        # Health mappings
        if any(term in context for term in ['health', 'wellbeing', 'wellness', 'mental', 'eap']):
            mappings.append("5.0")  # Health
        
        return list(set(mappings))  # Remove duplicates
    
    def _map_to_sdgs(self, context: str) -> List[str]:
        """Map metric to UN SDG goals."""
        mappings = []
        
        if any(term in context for term in ['education', 'learning', 'training']):
            mappings.append("4")  # Quality Education
        
        if any(term in context for term in ['employment', 'job', 'salary', 'economic']):
            mappings.append("8")  # Decent Work and Economic Growth
        
        if any(term in context for term in ['gender', 'equality', 'diversity']):
            mappings.append("5")  # Gender Equality
            mappings.append("10")  # Reduced Inequalities
        
        if any(term in context for term in ['health', 'wellbeing', 'wellness']):
            mappings.append("3")  # Good Health and Well-being
        
        if any(term in context for term in ['carbon', 'climate', 'emission', 'environment']):
            mappings.append("13")  # Climate Action
        
        if any(term in context for term in ['community', 'local', 'engagement']):
            mappings.append("11")  # Sustainable Cities and Communities
        
        if any(term in context for term in ['procurement', 'supply', 'responsible']):
            mappings.append("12")  # Responsible Consumption and Production
        
        return mappings
    
    def _map_to_toms(self, context: str) -> List[str]:
        """Map metric to TOMs framework."""
        mappings = []
        
        if any(term in context for term in ['employment', 'job', 'local']):
            mappings.append("NT1")  # Local employment
        
        if any(term in context for term in ['skills', 'training', 'apprentice']):
            mappings.append("NT2")  # Local skills and employment
        
        if any(term in context for term in ['procurement', 'supply', 'local', 'ethical']):
            mappings.append("NT3")  # Responsible procurement
        
        if any(term in context for term in ['environment', 'carbon', 'waste', 'energy']):
            mappings.append("NT4")  # Environmental management
        
        if 'volunteer' in context:
            mappings.append("NT90")  # Volunteering time
        
        return mappings
    
    def _map_to_bcorp(self, context: str) -> List[str]:
        """Map metric to B Corp impact areas."""
        mappings = []
        
        if any(term in context for term in ['employment', 'salary', 'skills', 'wellbeing', 'diversity']):
            mappings.append("workers")  # Workers
        
        if any(term in context for term in ['community', 'volunteer', 'charity', 'donation', 'local']):
            mappings.append("community")  # Community
        
        if any(term in context for term in ['environment', 'carbon', 'waste', 'energy', 'sustainability']):
            mappings.append("environment")  # Environment
        
        if any(term in context for term in ['governance', 'audit', 'ethics', 'compliance']):
            mappings.append("governance")  # Governance
        
        return mappings
    
    def apply_mappings_to_database(self) -> Dict[str, int]:
        """Apply framework mappings to all metrics in database."""
        try:
            # Feature flag: allow disabling framework mappings entirely
            enabled_env = os.getenv('IMPACTOS_FRAMEWORKS_ENABLED', 'true').strip().lower()
            if enabled_env not in ('1', 'true', 'yes', 'on'):
                logger.info("Framework mappings disabled via IMPACTOS_FRAMEWORKS_ENABLED")
                return {}

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all metrics
                cursor.execute("""
                    SELECT id, metric_name, metric_category, context_description
                    FROM impact_metrics
                """)
                metrics = cursor.fetchall()
                
                mapping_counts: Dict[str, int] = {}
                
                for metric in metrics:
                    # Generate framework mappings
                    mappings = self.map_metric_to_frameworks(
                        metric['metric_name'],
                        metric['metric_category'] or "",
                        metric['context_description'] or ""
                    )
                    
                    # Store mappings
                    for framework, categories in mappings.items():
                        # Ensure at least a general category is recorded
                        if not categories:
                            categories = ['GENERAL']
                        for category in categories:
                            try:
                                cursor.execute(
                                    """
                                    INSERT OR REPLACE INTO framework_mappings 
                                    (impact_metric_id, framework_name, framework_category, mapping_confidence)
                                    VALUES (?, ?, ?, ?)
                                    """,
                                    (metric['id'], framework, category, 0.7),
                                )
                                mapping_counts[framework] = mapping_counts.get(framework, 0) + 1
                            except Exception as e:
                                logger.error(f"Error storing mapping for metric {metric['id']}: {e}")
                
                conn.commit()
                logger.info(f"Applied framework mappings: {mapping_counts}")
                return mapping_counts
                
        except Exception as e:
            logger.error(f"Error applying framework mappings: {e}")
            return {}
    
    def get_framework_summary(self) -> Dict[str, Any]:
        """Get summary of framework mappings."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        framework_code,
                        framework_description,
                        COUNT(*) as metric_count
                    FROM framework_mappings
                    GROUP BY framework_code, framework_description
                    ORDER BY framework_code
                """)
                
                mappings = cursor.fetchall()
                
                summary = {}
                for mapping in mappings:
                    # Parse framework:category from framework_code
                    framework_code = mapping['framework_code']
                    if ':' in framework_code:
                        framework, category = framework_code.split(':', 1)
                        if framework not in summary:
                            summary[framework] = {}
                        summary[framework][category] = mapping['metric_count']
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting framework summary: {e}")
            return {}


def apply_framework_mappings(db_path: str = "db/impactos.db") -> Dict[str, int]:
    """Convenience function to apply framework mappings."""
    mapper = FrameworkMapper(db_path)
    return mapper.apply_mappings_to_database()


def get_framework_report(db_path: str = "db/impactos.db") -> str:
    """Generate a framework mapping report."""
    mapper = FrameworkMapper(db_path)
    summary = mapper.get_framework_summary()
    
    if not summary:
        return "No framework mappings found."
    
    report = ["📊 Framework Mapping Report", "=" * 40, ""]
    
    for framework, categories in summary.items():
        framework_def = mapper.frameworks.get(framework, {})
        framework_name = framework_def.get('name', framework)
        
        report.append(f"🎯 {framework_name}")
        report.append("-" * (len(framework_name) + 3))
        
        for category, count in categories.items():
            # Get category description
            if framework == "UK_SV_MODEL":
                desc = framework_def.get('subcategories', {}).get(category) or \
                       framework_def.get('categories', {}).get(category)
            elif framework == "UN_SDGS":
                desc = framework_def.get('goals', {}).get(category)
            elif framework == "TOMS":
                desc = framework_def.get('themes', {}).get(category)
            elif framework == "B_CORP":
                desc = framework_def.get('impact_areas', {}).get(category)
            else:
                desc = category
            
            report.append(f"  {category}: {desc} ({count} metrics)")
        
        report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test framework mapping
    mapping_counts = apply_framework_mappings()
    print(f"Applied mappings: {mapping_counts}")
    
    print("\n" + get_framework_report()) 