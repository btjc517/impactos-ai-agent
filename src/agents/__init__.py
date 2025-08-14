"""
Multi-Agent System for ImpactOS AI.

This module implements a Claude-based multi-agent architecture with:
- Architect Agent (Opus 4.1) for orchestration and review
- Data Agent (Sonnet 3.5) for data processing
- Query Agent (Sonnet 3.5) for natural language Q&A
"""

from .base_agent import BaseAgent
from .architect_agent import ArchitectAgent
from .data_agent import DataAgent
from .query_agent import QueryAgent
from .orchestrator import AgentOrchestrator
from .communication import AgentCommunication

__all__ = [
    'BaseAgent',
    'ArchitectAgent',
    'DataAgent',
    'QueryAgent',
    'AgentOrchestrator',
    'AgentCommunication'
]