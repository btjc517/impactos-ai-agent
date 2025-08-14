"""
Architect Agent using Claude Opus 4.1.

Responsible for system design, task orchestration, and quality review.
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ArchitectAgent(BaseAgent):
    """Architect agent for high-level orchestration and review."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Architect Agent with Opus 4.1."""
        super().__init__(
            name="Architect",
            model="claude-3-opus-20240229",  # Using Opus for architect role
            role="System architect responsible for task planning, delegation, and quality review",
            api_key=api_key
        )
        
        self.system_prompt = """You are the Architect Agent for ImpactOS AI system.

Your responsibilities:
1. Analyze user requests and break them into actionable tasks
2. Delegate tasks to appropriate specialist agents (Data Agent, Query Agent)
3. Review results from other agents for quality and completeness
4. Ensure system coherence and optimal performance
5. Make strategic decisions about data processing and retrieval

When creating tasks:
- Be specific about requirements and expected outputs
- Consider dependencies between tasks
- Optimize for parallel execution when possible
- Include validation criteria

When reviewing results:
- Verify accuracy and completeness
- Check for consistency across different data sources
- Ensure proper citations and evidence
- Request revisions if quality standards aren't met

Always think strategically about the best approach to solve problems."""
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process architect-level tasks.
        
        Args:
            task: Task dictionary
            
        Returns:
            Processing result
        """
        task_type = task.get('type')
        
        if task_type == 'plan':
            return self._plan_execution(task)
        elif task_type == 'review':
            return self._review_results(task)
        elif task_type == 'optimize':
            return self._optimize_system(task)
        elif task_type == 'analyze_request':
            return self._analyze_user_request(task)
        else:
            return self._general_architect_task(task)
    
    def _analyze_user_request(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user request and create execution plan."""
        try:
            request = task.get('data', {}).get('request', '')
            context = self.read_context()
            
            prompt = f"""Analyze this user request and create an execution plan:

User Request: {request}

Current System Context:
{context}

Please provide:
1. Request interpretation and requirements
2. Required data sources and processing steps
3. Task breakdown with dependencies
4. Delegation strategy (which agents handle what)
5. Success criteria and validation steps

Format your response as JSON with these sections:
- interpretation: Brief summary of what user wants
- requirements: List of specific requirements
- tasks: Array of task objects with type, agent, description, dependencies
- validation: How to verify success
- risks: Potential issues or challenges"""
            
            response = self.call_claude(prompt, self.system_prompt)
            
            # Parse response and create tasks
            plan = self._parse_json_response(response)
            
            # Create tasks for other agents
            if 'tasks' in plan:
                for task_def in plan['tasks']:
                    self.write_task({
                        'type': task_def.get('type', 'process'),
                        'agent': task_def.get('agent', 'Data'),
                        'description': task_def.get('description'),
                        'data': task_def,
                        'status': 'pending',
                        'priority': task_def.get('priority', 'normal')
                    })
            
            # Update context with plan
            self.write_context(f"## Execution Plan\n\n{json.dumps(plan, indent=2)}")
            
            return {
                'status': 'success',
                'plan': plan,
                'tasks_created': len(plan.get('tasks', [])),
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze request: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _plan_execution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan for complex tasks."""
        try:
            objective = task.get('data', {}).get('objective', '')
            constraints = task.get('data', {}).get('constraints', [])
            
            prompt = f"""Create a detailed execution plan for this objective:

Objective: {objective}

Constraints:
{json.dumps(constraints, indent=2)}

Consider:
1. Data ingestion requirements
2. Processing pipeline design
3. Query optimization strategies
4. Framework mapping needs
5. Validation and verification steps

Provide a structured plan with:
- Phase breakdown
- Task dependencies
- Resource requirements
- Timeline estimates
- Risk mitigation strategies"""
            
            response = self.call_claude(prompt, self.system_prompt)
            
            return {
                'status': 'success',
                'plan': response,
                'agent': self.name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _review_results(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review results from other agents."""
        try:
            results = task.get('data', {}).get('results', {})
            criteria = task.get('data', {}).get('criteria', [])
            
            prompt = f"""Review these results against quality criteria:

Results:
{json.dumps(results, indent=2)}

Quality Criteria:
{json.dumps(criteria, indent=2)}

Evaluate:
1. Accuracy and completeness
2. Data quality and consistency
3. Citation validity
4. Framework mapping correctness
5. Overall reliability

Provide:
- Quality score (0-100)
- Issues found
- Recommendations for improvement
- Approval status (approved/needs_revision/rejected)"""
            
            response = self.call_claude(prompt, self.system_prompt)
            review = self._parse_json_response(response)
            
            # If needs revision, create follow-up tasks
            if review.get('approval_status') == 'needs_revision':
                for issue in review.get('issues', []):
                    self.write_task({
                        'type': 'revision',
                        'description': f"Address issue: {issue}",
                        'original_task': task.get('id'),
                        'status': 'pending'
                    })
            
            return {
                'status': 'success',
                'review': review,
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Review failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _optimize_system(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system performance and configuration."""
        try:
            metrics = task.get('data', {}).get('metrics', {})
            
            prompt = f"""Analyze system metrics and suggest optimizations:

Current Metrics:
{json.dumps(metrics, indent=2)}

Consider:
1. Query performance bottlenecks
2. Data processing efficiency
3. Agent workload distribution
4. Cache utilization
5. API usage optimization

Provide optimization recommendations with:
- Priority level
- Expected impact
- Implementation complexity
- Resource requirements"""
            
            response = self.call_claude(prompt, self.system_prompt)
            
            return {
                'status': 'success',
                'optimizations': response,
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _general_architect_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general architect tasks."""
        try:
            prompt = f"""Execute this architect-level task:

Task: {json.dumps(task, indent=2)}

Current Context:
{self.read_context()}

Provide a comprehensive response addressing all aspects of the task."""
            
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
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from Claude response."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.index("```json") + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            elif "{" in response:
                # Try to find JSON object
                json_start = response.index("{")
                json_end = response.rindex("}") + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Return as text if no JSON found
                return {"content": response}
                
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Could not parse JSON: {e}")
            return {"content": response}
    
    def delegate_task(self, agent_name: str, task_type: str, data: Any):
        """
        Delegate a task to another agent.
        
        Args:
            agent_name: Target agent name
            task_type: Type of task
            data: Task data
        """
        task = {
            'type': task_type,
            'agent': agent_name,
            'delegated_by': self.name,
            'data': data,
            'status': 'pending'
        }
        self.write_task(task)
        logger.info(f"Architect delegated {task_type} to {agent_name}")
    
    def orchestrate_data_ingestion(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Orchestrate data ingestion across agents.
        
        Args:
            file_paths: List of files to ingest
            
        Returns:
            Orchestration result
        """
        try:
            # Create ingestion plan
            plan_task = {
                'type': 'plan',
                'data': {
                    'objective': f"Ingest and process {len(file_paths)} data files",
                    'constraints': ['Maintain data quality', 'Proper framework mapping', 'Efficient processing']
                }
            }
            
            plan_result = self.execute(plan_task)
            
            # Delegate to Data Agent
            for file_path in file_paths:
                self.delegate_task('Data', 'ingest', {'file_path': file_path})
            
            # Schedule review
            self.write_task({
                'type': 'review',
                'agent': 'Architect',
                'description': 'Review ingestion results',
                'data': {'file_count': len(file_paths)},
                'status': 'pending',
                'priority': 'high'
            })
            
            return {
                'status': 'success',
                'plan': plan_result,
                'files_queued': len(file_paths),
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}