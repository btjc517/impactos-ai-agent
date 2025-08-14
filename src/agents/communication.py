"""
Agent communication protocol for multi-agent coordination.

Manages inter-agent communication through shared files and task queues.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


class AgentCommunication:
    """Manages communication between agents."""
    
    def __init__(self, base_path: str = "agents"):
        """
        Initialize communication system.
        
        Args:
            base_path: Base directory for agent communication files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Communication channels
        self.context_file = self.base_path / "context.md"
        self.tasks_file = self.base_path / "tasks.json"
        self.messages_file = self.base_path / "messages.json"
        self.state_file = self.base_path / "system_state.json"
        
        # Initialize files if they don't exist
        self._initialize_files()
        
        # Message queue
        self.message_queue: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        logger.info(f"Initialized agent communication at {base_path}")
    
    def _initialize_files(self):
        """Initialize communication files if they don't exist."""
        if not self.context_file.exists():
            self.context_file.write_text("# Agent Context\n\nShared context for all agents.\n")
        
        if not self.tasks_file.exists():
            with open(self.tasks_file, 'w') as f:
                json.dump([], f)
        
        if not self.messages_file.exists():
            with open(self.messages_file, 'w') as f:
                json.dump([], f)
        
        if not self.state_file.exists():
            with open(self.state_file, 'w') as f:
                json.dump({
                    "system_status": "initialized",
                    "active_agents": [],
                    "last_update": datetime.now().isoformat()
                }, f)
    
    def send_message(self, from_agent: str, to_agent: str, 
                    message_type: str, content: Any) -> bool:
        """
        Send a message from one agent to another.
        
        Args:
            from_agent: Sender agent name
            to_agent: Receiver agent name
            message_type: Type of message
            content: Message content
            
        Returns:
            Success status
        """
        try:
            message = {
                "id": len(self.message_queue) + 1,
                "from": from_agent,
                "to": to_agent,
                "type": message_type,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            
            with self.lock:
                # Add to queue
                self.message_queue.append(message)
                
                # Persist to file
                messages = self._read_messages()
                messages.append(message)
                self._write_messages(messages)
            
            logger.debug(f"Message sent: {from_agent} -> {to_agent} ({message_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def receive_messages(self, agent_name: str, 
                        mark_read: bool = True) -> List[Dict[str, Any]]:
        """
        Receive messages for an agent.
        
        Args:
            agent_name: Receiving agent name
            mark_read: Whether to mark messages as read
            
        Returns:
            List of messages
        """
        try:
            with self.lock:
                messages = self._read_messages()
                agent_messages = []
                
                for msg in messages:
                    if msg.get('to') == agent_name and msg.get('status') == 'pending':
                        agent_messages.append(msg)
                        if mark_read:
                            msg['status'] = 'read'
                            msg['read_at'] = datetime.now().isoformat()
                
                if mark_read and agent_messages:
                    self._write_messages(messages)
            
            return agent_messages
            
        except Exception as e:
            logger.error(f"Failed to receive messages: {e}")
            return []
    
    def broadcast_message(self, from_agent: str, message_type: str, content: Any):
        """
        Broadcast a message to all agents.
        
        Args:
            from_agent: Sender agent name
            message_type: Type of message
            content: Message content
        """
        try:
            # Get active agents from state
            state = self.get_system_state()
            active_agents = state.get('active_agents', [])
            
            for agent in active_agents:
                if agent != from_agent:
                    self.send_message(from_agent, agent, message_type, content)
            
            logger.info(f"Broadcast sent from {from_agent} to {len(active_agents)-1} agents")
            
        except Exception as e:
            logger.error(f"Failed to broadcast: {e}")
    
    def update_context(self, agent_name: str, content: str, 
                      section: Optional[str] = None):
        """
        Update shared context.
        
        Args:
            agent_name: Agent updating context
            content: Content to add
            section: Optional section name
        """
        try:
            current_context = self.context_file.read_text()
            
            # Add timestamp and agent attribution
            update = f"\n\n---\n_Updated by {agent_name} at {datetime.now().isoformat()}_\n\n"
            
            if section:
                update += f"## {section}\n\n"
            
            update += content
            
            # Append to context
            new_context = current_context + update
            self.context_file.write_text(new_context)
            
            logger.debug(f"{agent_name} updated context")
            
        except Exception as e:
            logger.error(f"Failed to update context: {e}")
    
    def get_context(self, last_n_sections: Optional[int] = None) -> str:
        """
        Get shared context.
        
        Args:
            last_n_sections: Return only last N sections
            
        Returns:
            Context content
        """
        try:
            context = self.context_file.read_text()
            
            if last_n_sections:
                sections = context.split('\n---\n')
                if len(sections) > last_n_sections:
                    sections = sections[-last_n_sections:]
                    context = '\n---\n'.join(sections)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return ""
    
    def add_task(self, task: Dict[str, Any]) -> int:
        """
        Add a task to the queue.
        
        Args:
            task: Task dictionary
            
        Returns:
            Task ID
        """
        try:
            with self.lock:
                tasks = self._read_tasks()
                
                # Assign ID
                task_id = max([t.get('id', 0) for t in tasks], default=0) + 1
                task['id'] = task_id
                task['created_at'] = datetime.now().isoformat()
                task['status'] = task.get('status', 'pending')
                
                tasks.append(task)
                self._write_tasks(tasks)
            
            logger.debug(f"Added task {task_id}: {task.get('type')}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            return -1
    
    def get_pending_tasks(self, agent_name: Optional[str] = None,
                         task_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get pending tasks from queue.
        
        Args:
            agent_name: Filter by assigned agent
            task_types: Filter by task types
            
        Returns:
            List of pending tasks
        """
        try:
            tasks = self._read_tasks()
            pending = []
            
            for task in tasks:
                if task.get('status') != 'pending':
                    continue
                
                if agent_name and task.get('agent') != agent_name:
                    continue
                
                if task_types and task.get('type') not in task_types:
                    continue
                
                pending.append(task)
            
            return pending
            
        except Exception as e:
            logger.error(f"Failed to get pending tasks: {e}")
            return []
    
    def update_task_status(self, task_id: int, status: str, 
                          result: Optional[Dict[str, Any]] = None):
        """
        Update task status.
        
        Args:
            task_id: Task ID
            status: New status
            result: Optional task result
        """
        try:
            with self.lock:
                tasks = self._read_tasks()
                
                for task in tasks:
                    if task.get('id') == task_id:
                        task['status'] = status
                        task['updated_at'] = datetime.now().isoformat()
                        if result:
                            task['result'] = result
                        break
                
                self._write_tasks(tasks)
            
            logger.debug(f"Updated task {task_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
    
    def register_agent(self, agent_name: str, agent_info: Dict[str, Any]):
        """
        Register an agent with the system.
        
        Args:
            agent_name: Agent name
            agent_info: Agent information
        """
        try:
            state = self.get_system_state()
            
            # Update active agents
            if agent_name not in state.get('active_agents', []):
                state.setdefault('active_agents', []).append(agent_name)
            
            # Store agent info
            state.setdefault('agent_info', {})[agent_name] = {
                **agent_info,
                'registered_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self._write_state(state)
            logger.info(f"Registered agent: {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
    
    def unregister_agent(self, agent_name: str):
        """
        Unregister an agent from the system.
        
        Args:
            agent_name: Agent name
        """
        try:
            state = self.get_system_state()
            
            # Remove from active agents
            if agent_name in state.get('active_agents', []):
                state['active_agents'].remove(agent_name)
            
            # Update agent info
            if 'agent_info' in state and agent_name in state['agent_info']:
                state['agent_info'][agent_name]['status'] = 'inactive'
                state['agent_info'][agent_name]['unregistered_at'] = datetime.now().isoformat()
            
            self._write_state(state)
            logger.info(f"Unregistered agent: {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to unregister agent: {e}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to get system state: {e}")
            return {}
    
    def update_system_state(self, updates: Dict[str, Any]):
        """
        Update system state.
        
        Args:
            updates: State updates to apply
        """
        try:
            state = self.get_system_state()
            state.update(updates)
            state['last_update'] = datetime.now().isoformat()
            self._write_state(state)
            
        except Exception as e:
            logger.error(f"Failed to update system state: {e}")
    
    def _read_messages(self) -> List[Dict[str, Any]]:
        """Read messages from file."""
        try:
            with open(self.messages_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _write_messages(self, messages: List[Dict[str, Any]]):
        """Write messages to file."""
        with open(self.messages_file, 'w') as f:
            json.dump(messages, f, indent=2, default=str)
    
    def _read_tasks(self) -> List[Dict[str, Any]]:
        """Read tasks from file."""
        try:
            with open(self.tasks_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _write_tasks(self, tasks: List[Dict[str, Any]]):
        """Write tasks to file."""
        with open(self.tasks_file, 'w') as f:
            json.dump(tasks, f, indent=2, default=str)
    
    def _write_state(self, state: Dict[str, Any]):
        """Write system state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def clear_completed_tasks(self, older_than_hours: int = 24):
        """
        Clear completed tasks older than specified hours.
        
        Args:
            older_than_hours: Clear tasks older than this many hours
        """
        try:
            cutoff = datetime.now().timestamp() - (older_than_hours * 3600)
            
            with self.lock:
                tasks = self._read_tasks()
                filtered = []
                
                for task in tasks:
                    if task.get('status') == 'completed':
                        if 'updated_at' in task:
                            task_time = datetime.fromisoformat(task['updated_at']).timestamp()
                            if task_time < cutoff:
                                continue
                    filtered.append(task)
                
                self._write_tasks(filtered)
            
            removed = len(tasks) - len(filtered)
            logger.info(f"Cleared {removed} completed tasks")
            
        except Exception as e:
            logger.error(f"Failed to clear tasks: {e}")