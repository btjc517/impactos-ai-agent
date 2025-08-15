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
        
        # Framework definitions
        self.frameworks = self._load_framework_definitions()
    
    def _load_framework_definitions(self) -> Dict[str, Any]:
        """Load framework mapping definitions."""
        return {
            "UK_SV_MODEL": {
                "name": "UK Social Value Model (MAC)",
                "categories": {
                    "1.0": "Crime",
                    "2.0": "Education", 
                    "3.0": "Employment",
                    "4.0": "Environment",
                    "5.0": "Health",
                    "6.0": "Social Cohesion",
                    "7.0": "Housing",
                    "8.0": "Community Engagement"
                },
                "subcategories": {
                    "3.1": "Jobs for local people",
                    "3.2": "Apprenticeships",
                    "3.3": "Skills development",
                    "4.1": "Carbon emissions reduction",
                    "4.2": "Waste reduction",
                    "4.3": "Biodiversity improvement",
                    "8.1": "Community volunteering",
                    "8.2": "Charitable giving"
                }
            },
            "UN_SDGS": {
                "name": "UN Sustainable Development Goals",
                "goals": {
                    "1": "No Poverty",
                    "3": "Good Health and Well-being", 
                    "4": "Quality Education",
                    "5": "Gender Equality",
                    "8": "Decent Work and Economic Growth",
                    "10": "Reduced Inequalities",
                    "11": "Sustainable Cities and Communities",
                    "12": "Responsible Consumption and Production",
                    "13": "Climate Action",
                    "16": "Peace, Justice and Strong Institutions"
                }
            },
            "TOMS": {
                "name": "Themes, Outcomes and Measures",
                "themes": {
                    "NT1": "Local employment",
                    "NT2": "Local skills and employment", 
                    "NT3": "Responsible procurement",
                    "NT4": "Environmental management",
                    "NT90": "Volunteering time"
                }
            },
            "B_CORP": {
                "name": "B Corporation Assessment",
                "impact_areas": {
                    "governance": "Governance",
                    "workers": "Workers", 
                    "community": "Community",
                    "environment": "Environment",
                    "customers": "Customers"
                }
            }
        }
    
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
        
        # Combine all text for analysis
        full_context = f"{metric_name} {metric_category} {metric_context}".lower()
        
        # Map to UK Social Value Model
        uk_mappings = self._map_to_uk_sv_model(full_context)
        if uk_mappings:
            mappings["UK_SV_MODEL"] = uk_mappings
        
        # Map to UN SDGs
        sdg_mappings = self._map_to_sdgs(full_context)
        if sdg_mappings:
            mappings["UN_SDGS"] = sdg_mappings
        
        # Map to TOMs
        toms_mappings = self._map_to_toms(full_context)
        if toms_mappings:
            mappings["TOMS"] = toms_mappings
        
        # Map to B Corp
        bcorp_mappings = self._map_to_bcorp(full_context)
        if bcorp_mappings:
            mappings["B_CORP"] = bcorp_mappings
        
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all metrics
                cursor.execute("""
                    SELECT id, metric_name, metric_category, context_description
                    FROM impact_metrics
                """)
                metrics = cursor.fetchall()
                
                mapping_counts = {"UK_SV_MODEL": 0, "UN_SDGS": 0, "TOMS": 0, "B_CORP": 0}
                
                for metric in metrics:
                    # Generate framework mappings
                    mappings = self.map_metric_to_frameworks(
                        metric['metric_name'],
                        metric['metric_category'] or "",
                        metric['context_description'] or ""
                    )
                    
                    # Store mappings
                    for framework, categories in mappings.items():
                        for category in categories:
                            try:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO framework_mappings 
                                    (metric_id, framework_id, framework_code, framework_description, mapping_confidence)
                                    VALUES (?, 1, ?, ?, ?)
                                """, (metric['id'], f"{framework}:{category}", 
                                      f"{framework} - {category}", 0.8))
                                
                                mapping_counts[framework] += 1
                                
                            except Exception as e:
                                logger.error(f"Error storing mapping for metric {metric['id']}: {e}")
                
                conn.commit()
                logger.info(f"Applied framework mappings: {mapping_counts}")
                return mapping_counts
                
        except Exception as e:
            logger.error(f"Error applying framework mappings: {e}")
            return {}
    
    def get_metric_frameworks(self, metric_id: int) -> Dict[str, List[Dict[str, str]]]:
        """Get frameworks mapped to a specific metric with compact display info."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT framework_code, framework_description
                    FROM framework_mappings
                    WHERE metric_id = ?
                    ORDER BY framework_code
                """, (metric_id,))
                
                mappings = cursor.fetchall()
                result = {}
                
                for mapping in mappings:
                    framework_code = mapping['framework_code']
                    if ':' in framework_code:
                        framework, category = framework_code.split(':', 1)
                        
                        if framework not in result:
                            result[framework] = []
                        
                        # Get short display info
                        display_info = self._get_framework_display_info(framework, category)
                        result[framework].append(display_info)
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting metric frameworks: {e}")
            return {}
    
    def _get_framework_display_info(self, framework: str, category: str) -> Dict[str, str]:
        """Get compact display information for framework category."""
        framework_def = self.frameworks.get(framework, {})
        
        if framework == "UK_SV_MODEL":
            desc = framework_def.get('subcategories', {}).get(category) or \
                   framework_def.get('categories', {}).get(category, category)
            return {"code": category, "label": desc, "color": "blue"}
        
        elif framework == "UN_SDGS":
            desc = framework_def.get('goals', {}).get(category, f"SDG {category}")
            return {"code": f"SDG{category}", "label": desc, "color": "green"}
        
        elif framework == "TOMS":
            desc = framework_def.get('themes', {}).get(category, category)
            return {"code": category, "label": desc, "color": "orange"}
        
        elif framework == "B_CORP":
            desc = framework_def.get('impact_areas', {}).get(category, category)
            return {"code": category.upper(), "label": desc, "color": "purple"}
        
        return {"code": category, "label": category, "color": "gray"}
    
    def format_framework_badges(self, metric_id: int, compact: bool = True) -> str:
        """Format framework mappings as compact badges/tags."""
        frameworks = self.get_metric_frameworks(metric_id)
        
        if not frameworks:
            return ""
        
        badges = []
        for framework, categories in frameworks.items():
            if compact:
                # Show only most relevant category per framework
                if categories:
                    category = categories[0]  # Take first/primary mapping
                    badges.append(f"[{category['code']}]")
            else:
                # Show all categories
                for category in categories:
                    badges.append(f"[{category['code']}: {category['label']}]")
        
        return " ".join(badges)
    
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