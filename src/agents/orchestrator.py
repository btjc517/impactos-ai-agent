"""
Agent Orchestrator for coordinating multi-agent system.

Manages agent lifecycle, task distribution, and system coordination.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os

from .architect_agent import ArchitectAgent
from .data_agent import DataAgent
from .query_agent import QueryAgent
from .communication import AgentCommunication

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates the multi-agent system."""
    
    def __init__(self, anthropic_key: Optional[str] = None, 
                 openai_key: Optional[str] = None):
        """
        Initialize the orchestrator.
        
        Args:
            anthropic_key: Anthropic API key
            openai_key: OpenAI API key
        """
        self.anthropic_key = anthropic_key or os.getenv('ANTHROPIC_API_KEY')
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        
        if not self.anthropic_key:
            raise ValueError("Anthropic API key required for agents")
        
        # Initialize communication system
        self.communication = AgentCommunication()
        
        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()
        
        # Task processing
        self.running = False
        self.processing_thread = None
        
        logger.info("Orchestrator initialized with all agents")
    
    def _initialize_agents(self):
        """Initialize all agents in the system."""
        try:
            # Architect Agent (Opus)
            self.agents['Architect'] = ArchitectAgent(self.anthropic_key)
            self.communication.register_agent('Architect', {
                'model': 'claude-3-opus-20240229',
                'role': 'System architect and orchestrator'
            })
            
            # Data Agent (Sonnet)
            self.agents['DataAgent'] = DataAgent(self.anthropic_key, self.openai_key)
            self.communication.register_agent('DataAgent', {
                'model': 'claude-3-5-sonnet-20241022',
                'role': 'Data processing specialist'
            })
            
            # Query Agent (Sonnet)
            self.agents['QueryAgent'] = QueryAgent(self.anthropic_key, self.openai_key)
            self.communication.register_agent('QueryAgent', {
                'model': 'claude-3-5-sonnet-20241022',
                'role': 'Query processing specialist'
            })
            
            logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def start(self):
        """Start the orchestrator and begin processing tasks."""
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.communication.update_system_state({
            'system_status': 'running',
            'orchestrator_started': datetime.now().isoformat()
        })
        
        logger.info("Orchestrator started")
    
    def stop(self):
        """Stop the orchestrator."""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        self.communication.update_system_state({
            'system_status': 'stopped',
            'orchestrator_stopped': datetime.now().isoformat()
        })
        
        # Unregister agents
        for agent_name in self.agents:
            self.communication.unregister_agent(agent_name)
        
        logger.info("Orchestrator stopped")
    
    def _process_loop(self):
        """Main processing loop for task distribution."""
        while self.running:
            try:
                # Process tasks for each agent
                for agent_name, agent in self.agents.items():
                    # Get pending tasks for this agent
                    tasks = self.communication.get_pending_tasks(agent_name)
                    
                    for task in tasks[:1]:  # Process one task at a time per agent
                        logger.info(f"Assigning task {task['id']} to {agent_name}")
                        
                        # Update task status
                        self.communication.update_task_status(task['id'], 'in_progress')
                        
                        # Execute task
                        result = agent.execute(task)
                        
                        # Update task with result
                        self.communication.update_task_status(
                            task['id'], 
                            'completed' if result.get('status') == 'success' else 'failed',
                            result
                        )
                
                # Process inter-agent messages
                self._process_messages()
                
                # Small delay to prevent CPU spinning
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5)  # Longer delay on error
    
    def _process_messages(self):
        """Process inter-agent messages."""
        try:
            for agent_name, agent in self.agents.items():
                messages = self.communication.receive_messages(agent_name)
                
                for message in messages:
                    logger.debug(f"Processing message for {agent_name}: {message['type']}")
                    
                    # Handle different message types
                    if message['type'] == 'collaboration':
                        # Create a task from collaboration request
                        task = {
                            'type': 'collaboration_response',
                            'agent': agent_name,
                            'data': message['content'],
                            'from_agent': message['from']
                        }
                        self.communication.add_task(task)
                        
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
    
    def process_user_request(self, request: str) -> Dict[str, Any]:
        """
        Process a user request through the multi-agent system.
        
        Args:
            request: User's request/question
            
        Returns:
            Processing result
        """
        try:
            logger.info(f"Processing user request: {request[:100]}...")
            
            # Create initial task for Architect to analyze
            task_id = self.communication.add_task({
                'type': 'analyze_request',
                'agent': 'Architect',
                'data': {'request': request},
                'priority': 'high'
            })
            
            # Wait for architect to create execution plan
            max_wait = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                tasks = self.communication._read_tasks()
                task = next((t for t in tasks if t['id'] == task_id), None)
                
                if task and task.get('status') == 'completed':
                    # Architect has created plan, wait for execution
                    return self._wait_for_execution(task.get('result', {}))
                
                time.sleep(1)
            
            return {
                'status': 'timeout',
                'error': 'Request processing timed out',
                'request': request
            }
            
        except Exception as e:
            logger.error(f"Failed to process user request: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'request': request
            }
    
    def _wait_for_execution(self, plan: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """Wait for plan execution to complete."""
        try:
            created_tasks = plan.get('tasks_created', 0)
            if created_tasks == 0:
                return plan
            
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check task completion
                tasks = self.communication._read_tasks()
                plan_tasks = [t for t in tasks if t.get('created_by') == 'Architect']
                
                completed = sum(1 for t in plan_tasks if t.get('status') == 'completed')
                
                if completed >= created_tasks:
                    # All tasks completed, compile results
                    results = [t.get('result', {}) for t in plan_tasks if t.get('status') == 'completed']
                    
                    return {
                        'status': 'success',
                        'plan': plan,
                        'results': results,
                        'tasks_completed': completed
                    }
                
                time.sleep(2)
            
            return {
                'status': 'partial',
                'plan': plan,
                'message': 'Some tasks still processing'
            }
            
        except Exception as e:
            logger.error(f"Execution wait failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def ingest_data(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest data files through the multi-agent system.
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            Ingestion result
        """
        try:
            # Architect orchestrates ingestion
            architect = self.agents['Architect']
            result = architect.orchestrate_data_ingestion(file_paths)
            
            # Wait for data agent to process
            if result.get('status') == 'success':
                return self._wait_for_execution(result, timeout=120)
            
            return result
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def query_data(self, question: str) -> Dict[str, Any]:
        """
        Query data through the multi-agent system.
        
        Args:
            question: Natural language question
            
        Returns:
            Query result with answer
        """
        try:
            # Direct query to Query Agent
            query_agent = self.agents['QueryAgent']
            result = query_agent.search_and_answer(question)
            
            # Have Architect review if needed
            if result.get('status') == 'success':
                review_task = {
                    'type': 'review',
                    'agent': 'Architect',
                    'data': {
                        'results': result,
                        'criteria': ['accuracy', 'completeness', 'citations']
                    }
                }
                
                review_id = self.communication.add_task(review_task)
                
                # Don't wait for review, return result immediately
                result['review_pending'] = True
                result['review_task_id'] = review_id
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'question': question
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            state = self.communication.get_system_state()
            
            # Add agent statuses
            agent_statuses = {}
            for name, agent in self.agents.items():
                agent_statuses[name] = agent.get_agent_summary()
            
            # Get task statistics
            tasks = self.communication._read_tasks()
            task_stats = {
                'total': len(tasks),
                'pending': len([t for t in tasks if t.get('status') == 'pending']),
                'in_progress': len([t for t in tasks if t.get('status') == 'in_progress']),
                'completed': len([t for t in tasks if t.get('status') == 'completed']),
                'failed': len([t for t in tasks if t.get('status') == 'failed'])
            }
            
            return {
                'orchestrator_running': self.running,
                'system_state': state,
                'agents': agent_statuses,
                'task_statistics': task_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop orchestrator
            self.stop()
            
            # Clear old tasks
            self.communication.clear_completed_tasks(older_than_hours=24)
            
            logger.info("Orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")