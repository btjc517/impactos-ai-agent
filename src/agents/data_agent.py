"""
Data Agent using Claude Sonnet 3.5.

Responsible for data ingestion, processing, and validation.
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base_agent import BaseAgent
from gpt_tools import FileManager, AssistantManager

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """Data processing agent for ingestion and extraction."""
    
    def __init__(self, api_key: Optional[str] = None, openai_key: Optional[str] = None):
        """Initialize Data Agent with Sonnet 3.5."""
        super().__init__(
            name="DataAgent",
            model="claude-3-5-sonnet-20241022",  # Using latest Sonnet
            role="Data specialist responsible for ingestion, extraction, and validation",
            api_key=api_key
        )
        
        # Initialize GPT tools
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        if self.openai_key:
            self.file_manager = FileManager(self.openai_key)
            self.assistant_manager = AssistantManager(self.openai_key)
        else:
            logger.warning("OpenAI key not available, GPT tools disabled")
            self.file_manager = None
            self.assistant_manager = None
        
        self.system_prompt = """You are the Data Agent for ImpactOS AI system.

Your responsibilities:
1. Ingest data from various file formats (CSV, Excel, PDF)
2. Extract structured impact metrics with high accuracy
3. Validate data quality and completeness
4. Map metrics to appropriate frameworks (UN SDGs, MAC, TOMs, B Corp)
5. Ensure proper citations and source tracking

When processing data:
- Extract metric name, value, unit, time period, and category
- Identify framework alignments
- Validate numerical values and units
- Track source file and location within file
- Flag any data quality issues

Always maintain high accuracy standards and provide detailed extraction results."""
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data-related tasks.
        
        Args:
            task: Task dictionary
            
        Returns:
            Processing result
        """
        task_type = task.get('type')
        
        if task_type == 'ingest':
            return self._ingest_file(task)
        elif task_type == 'extract':
            return self._extract_metrics(task)
        elif task_type == 'validate':
            return self._validate_data(task)
        elif task_type == 'map_frameworks':
            return self._map_frameworks(task)
        else:
            return self._general_data_task(task)
    
    def _ingest_file(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest and process a data file."""
        try:
            file_path = task.get('data', {}).get('file_path', '')
            
            if not os.path.exists(file_path):
                return {
                    'status': 'error',
                    'error': f"File not found: {file_path}",
                    'agent': self.name
                }
            
            # Use GPT tools if available
            if self.file_manager and self.assistant_manager:
                # Upload file to OpenAI
                file_id = self.file_manager.upload_file(file_path)
                
                # Process with GPT-4
                instructions = """Extract all impact metrics from this file.
                For each metric, identify:
                - Metric name
                - Numerical value
                - Unit of measurement
                - Time period
                - Category/theme
                - Related frameworks (SDGs, MAC, TOMs, B Corp)
                
                Return results as structured JSON."""
                
                gpt_result = self.assistant_manager.process_data([file_id], instructions)
                
                # Parse GPT results
                metrics = self._parse_gpt_extraction(gpt_result)
            else:
                # Fallback to Claude-based extraction
                metrics = self._extract_with_claude(file_path)
            
            # Validate extracted metrics
            validation_result = self._validate_metrics(metrics)
            
            # Update context with results
            self.write_context(f"""## Data Ingestion Results

File: {file_path}
Metrics Extracted: {len(metrics)}
Validation Score: {validation_result.get('score', 0)}%

Sample Metrics:
{json.dumps(metrics[:3] if metrics else [], indent=2)}
""")
            
            return {
                'status': 'success',
                'file': file_path,
                'metrics_count': len(metrics),
                'metrics': metrics,
                'validation': validation_result,
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _extract_metrics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specific metrics from data."""
        try:
            data = task.get('data', {})
            content = data.get('content', '')
            metric_types = data.get('metric_types', [])
            
            prompt = f"""Extract impact metrics from this content:

Content:
{content}

Focus on these metric types:
{json.dumps(metric_types, indent=2)}

For each metric, provide:
- name: Metric name
- value: Numerical value
- unit: Unit of measurement
- period: Time period
- category: Impact category
- confidence: Extraction confidence (0-1)
- source_text: Exact text where metric was found

Return as JSON array."""
            
            response = self.call_claude(prompt, self.system_prompt)
            metrics = self._parse_json_response(response)
            
            return {
                'status': 'success',
                'metrics': metrics.get('metrics', metrics) if isinstance(metrics, dict) else metrics,
                'count': len(metrics.get('metrics', metrics) if isinstance(metrics, dict) else metrics),
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _validate_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and completeness."""
        try:
            metrics = task.get('data', {}).get('metrics', [])
            
            validation_results = []
            for metric in metrics:
                result = self._validate_single_metric(metric)
                validation_results.append(result)
            
            # Calculate overall score
            valid_count = sum(1 for r in validation_results if r['is_valid'])
            score = (valid_count / len(validation_results) * 100) if validation_results else 0
            
            return {
                'status': 'success',
                'total_metrics': len(metrics),
                'valid_metrics': valid_count,
                'validation_score': score,
                'issues': [r['issues'] for r in validation_results if r['issues']],
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _map_frameworks(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Map metrics to sustainability frameworks."""
        try:
            metrics = task.get('data', {}).get('metrics', [])
            
            prompt = f"""Map these impact metrics to relevant frameworks:

Metrics:
{json.dumps(metrics, indent=2)}

Frameworks to consider:
1. UN Sustainable Development Goals (SDGs)
2. Measuring and Communicating (MAC) framework
3. The Outcomes Matrix (TOMs)
4. B Corporation standards

For each metric, provide:
- metric_id: Original metric identifier
- framework_mappings: Array of {{framework, alignment, confidence}}

Return as JSON."""
            
            response = self.call_claude(prompt, self.system_prompt)
            mappings = self._parse_json_response(response)
            
            return {
                'status': 'success',
                'mappings': mappings,
                'metrics_mapped': len(metrics),
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Framework mapping failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _general_data_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general data processing tasks."""
        try:
            prompt = f"""Execute this data processing task:

Task: {json.dumps(task, indent=2)}

Apply your expertise in data extraction, validation, and framework mapping."""
            
            response = self.call_claude(prompt, self.system_prompt)
            
            return {
                'status': 'success',
                'response': response,
                'agent': self.name,
                'task_type': task.get('type')
            }
            
        except Exception as e:
            logger.error(f"General task failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _extract_with_claude(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract metrics using Claude when GPT tools unavailable."""
        try:
            # Read file content (simplified for now)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()[:10000]  # Limit content size
            
            prompt = f"""Extract all impact metrics from this file content:

File: {os.path.basename(file_path)}
Content:
{content}

Extract and structure each metric with:
- name, value, unit, period, category
- Source location in file
- Confidence score

Return as JSON array."""
            
            response = self.call_claude(prompt, self.system_prompt)
            metrics = self._parse_json_response(response)
            
            if isinstance(metrics, dict) and 'metrics' in metrics:
                return metrics['metrics']
            elif isinstance(metrics, list):
                return metrics
            else:
                return []
                
        except Exception as e:
            logger.error(f"Claude extraction failed: {e}")
            return []
    
    def _parse_gpt_extraction(self, gpt_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse GPT extraction results into standardized format."""
        try:
            if 'structured_data' in gpt_result:
                data = gpt_result['structured_data']
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'metrics' in data:
                    return data['metrics']
            
            # Try to parse from content
            content = gpt_result.get('content', '')
            if content:
                return self._parse_json_response(content)
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to parse GPT results: {e}")
            return []
    
    def _validate_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate extracted metrics."""
        issues = []
        valid_count = 0
        
        for metric in metrics:
            validation = self._validate_single_metric(metric)
            if validation['is_valid']:
                valid_count += 1
            else:
                issues.extend(validation['issues'])
        
        score = (valid_count / len(metrics) * 100) if metrics else 0
        
        return {
            'score': score,
            'valid_count': valid_count,
            'total_count': len(metrics),
            'issues': issues
        }
    
    def _validate_single_metric(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single metric."""
        issues = []
        
        # Check required fields
        required = ['name', 'value']
        for field in required:
            if field not in metric or not metric[field]:
                issues.append(f"Missing required field: {field}")
        
        # Validate value is numeric
        if 'value' in metric:
            try:
                float(metric['value'])
            except (ValueError, TypeError):
                issues.append(f"Non-numeric value: {metric['value']}")
        
        # Check unit consistency
        if 'unit' in metric and metric['unit']:
            # Could add unit validation logic here
            pass
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'metric': metric.get('name', 'unknown')
        }
    
    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from response."""
        try:
            if "```json" in response:
                json_start = response.index("```json") + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            elif "[" in response or "{" in response:
                # Find JSON structure
                if "[" in response:
                    json_start = response.index("[")
                    json_end = response.rindex("]") + 1
                else:
                    json_start = response.index("{")
                    json_end = response.rindex("}") + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return []
                
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Could not parse JSON: {e}")
            return []