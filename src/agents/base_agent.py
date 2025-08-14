"""
Base Agent class for multi-agent system.

Provides common functionality for all agents.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import anthropic
import sys

# Add utils to path for configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name: str, model: str, role: str, 
                 api_key: Optional[str] = None):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            model: Claude model to use
            role: Agent's role description
            api_key: Anthropic API key (uses env var if not provided)
        """
        self.name = name
        self.model = model
        self.role = role
        
        # Get API key from config or parameter
        config = get_config()
        self.api_key = api_key or config.get_anthropic_key()
        
        if not self.api_key:
            raise ValueError("Anthropic API key required. Please set ANTHROPIC_API_KEY in environment or .env file")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Communication paths
        self.context_path = Path("agents/context.md")
        self.tasks_path = Path("agents/tasks.json")
        self.results_dir = Path("agents/results")
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.context_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Agent state
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {name} agent with model {model}")
    
    @abstractmethod
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a specific task.
        
        Args:
            task: Task dictionary with type, data, and metadata
            
        Returns:
            Result dictionary
        """
        pass
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with logging and error handling.
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        try:
            logger.info(f"{self.name} executing task: {task.get('type')}")
            
            # Update current task
            self.current_task = task
            task['start_time'] = datetime.now().isoformat()
            
            # Process task
            result = self.process_task(task)
            
            # Update task history
            task['end_time'] = datetime.now().isoformat()
            task['result'] = result
            self.task_history.append(task)
            
            # Save result
            self._save_result(result)
            
            logger.info(f"{self.name} completed task: {task.get('type')}")
            return result
            
        except Exception as e:
            logger.error(f"{self.name} task failed: {e}")
            error_result = {
                "status": "error",
                "error": str(e),
                "agent": self.name,
                "task": task
            }
            self.task_history.append({**task, "error": str(e)})
            return error_result
    
    def call_claude(self, prompt: str, system_prompt: Optional[str] = None,
                   max_tokens: int = 4096) -> str:
        """
        Call Claude API with the specified prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (uses agent role if not provided)
            max_tokens: Maximum response tokens
            
        Returns:
            Claude's response
        """
        try:
            if system_prompt is None:
                system_prompt = f"You are {self.name}, {self.role}"
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise
    
    def read_context(self) -> str:
        """Read shared context from markdown file."""
        try:
            if self.context_path.exists():
                return self.context_path.read_text()
            return ""
        except Exception as e:
            logger.error(f"Failed to read context: {e}")
            return ""
    
    def write_context(self, content: str, append: bool = True):
        """
        Write to shared context.
        
        Args:
            content: Content to write
            append: Whether to append or overwrite
        """
        try:
            if append and self.context_path.exists():
                existing = self.read_context()
                content = f"{existing}\n\n---\n\n{content}"
            
            self.context_path.write_text(content)
            logger.debug(f"{self.name} updated context")
            
        except Exception as e:
            logger.error(f"Failed to write context: {e}")
    
    def read_tasks(self) -> List[Dict[str, Any]]:
        """Read task queue from JSON file."""
        try:
            if self.tasks_path.exists():
                with open(self.tasks_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Failed to read tasks: {e}")
            return []
    
    def write_task(self, task: Dict[str, Any]):
        """Add task to queue."""
        try:
            tasks = self.read_tasks()
            task['id'] = len(tasks) + 1
            task['created_by'] = self.name
            task['created_at'] = datetime.now().isoformat()
            tasks.append(task)
            
            with open(self.tasks_path, 'w') as f:
                json.dump(tasks, f, indent=2)
            
            logger.debug(f"{self.name} added task to queue")
            
        except Exception as e:
            logger.error(f"Failed to write task: {e}")
    
    def get_next_task(self, task_types: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get next pending task from queue.
        
        Args:
            task_types: Filter by specific task types
            
        Returns:
            Next task or None
        """
        try:
            tasks = self.read_tasks()
            
            for task in tasks:
                if task.get('status') == 'pending':
                    if task_types is None or task.get('type') in task_types:
                        # Mark as in progress
                        task['status'] = 'in_progress'
                        task['assigned_to'] = self.name
                        self._update_task(task)
                        return task
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get next task: {e}")
            return None
    
    def _update_task(self, updated_task: Dict[str, Any]):
        """Update task in queue."""
        try:
            tasks = self.read_tasks()
            
            for i, task in enumerate(tasks):
                if task.get('id') == updated_task.get('id'):
                    tasks[i] = updated_task
                    break
            
            with open(self.tasks_path, 'w') as f:
                json.dump(tasks, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update task: {e}")
    
    def complete_task(self, task_id: int, result: Dict[str, Any]):
        """Mark task as completed with result."""
        try:
            tasks = self.read_tasks()
            
            for task in tasks:
                if task.get('id') == task_id:
                    task['status'] = 'completed'
                    task['completed_by'] = self.name
                    task['completed_at'] = datetime.now().isoformat()
                    task['result'] = result
                    self._update_task(task)
                    break
                    
        except Exception as e:
            logger.error(f"Failed to complete task: {e}")
    
    def _save_result(self, result: Dict[str, Any]):
        """Save result to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.debug(f"Saved result to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
    
    def collaborate(self, other_agent: str, message: str):
        """
        Send collaboration message to another agent.
        
        Args:
            other_agent: Name of target agent
            message: Message content
        """
        task = {
            "type": "collaboration",
            "from": self.name,
            "to": other_agent,
            "message": message,
            "status": "pending"
        }
        self.write_task(task)
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of agent's current state."""
        return {
            "name": self.name,
            "model": self.model,
            "role": self.role,
            "current_task": self.current_task,
            "tasks_completed": len([t for t in self.task_history if 'error' not in t]),
            "tasks_failed": len([t for t in self.task_history if 'error' in t]),
            "last_activity": self.task_history[-1]['end_time'] if self.task_history else None
        }