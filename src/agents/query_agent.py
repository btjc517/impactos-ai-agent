"""
Query Agent using Claude Sonnet 3.5.

Responsible for natural language query processing and answer generation.
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base_agent import BaseAgent
from gpt_tools import AssistantManager, RetrievalService

logger = logging.getLogger(__name__)


class QueryAgent(BaseAgent):
    """Query processing agent for natural language Q&A."""
    
    def __init__(self, api_key: Optional[str] = None, openai_key: Optional[str] = None):
        """Initialize Query Agent with Sonnet 3.5."""
        super().__init__(
            name="QueryAgent",
            model="claude-3-5-sonnet-20241022",  # Using latest Sonnet
            role="Query specialist responsible for understanding questions and generating accurate answers",
            api_key=api_key
        )
        
        # Initialize GPT tools
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        if self.openai_key:
            self.assistant_manager = AssistantManager(self.openai_key)
            self.retrieval_service = RetrievalService(self.openai_key)
        else:
            logger.warning("OpenAI key not available, GPT tools disabled")
            self.assistant_manager = None
            self.retrieval_service = None
        
        self.system_prompt = """You are the Query Agent for ImpactOS AI system.

Your responsibilities:
1. Understand natural language questions about impact data
2. Search and retrieve relevant information
3. Generate accurate, well-cited answers
4. Perform calculations and aggregations
5. Create visualization recommendations when helpful

When answering queries:
- Always cite sources using [1], [2] notation
- Provide specific values and units
- Include time periods when relevant
- Suggest charts/visualizations for complex data
- Acknowledge limitations or missing data

Maintain high accuracy and never fabricate information."""
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query-related tasks.
        
        Args:
            task: Task dictionary
            
        Returns:
            Processing result
        """
        task_type = task.get('type')
        
        if task_type == 'query':
            return self._process_query(task)
        elif task_type == 'analyze_intent':
            return self._analyze_intent(task)
        elif task_type == 'aggregate':
            return self._aggregate_data(task)
        elif task_type == 'visualize':
            return self._create_visualization(task)
        else:
            return self._general_query_task(task)
    
    def _process_query(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a natural language query."""
        try:
            question = task.get('data', {}).get('question', '')
            context = self.read_context()
            
            # Use GPT Assistant if available
            if self.assistant_manager:
                gpt_result = self.assistant_manager.query_data(question)
                answer = gpt_result.get('content', '')
                citations = gpt_result.get('annotations', [])
                
                # Enhance with Claude analysis
                enhanced = self._enhance_answer(answer, citations, question)
                final_answer = enhanced.get('answer', answer)
            else:
                # Pure Claude-based answer
                final_answer = self._answer_with_claude(question, context)
                citations = []
            
            # Check if visualization would help
            viz_recommendation = self._recommend_visualization(question, final_answer)
            
            result = {
                'status': 'success',
                'question': question,
                'answer': final_answer,
                'citations': citations,
                'visualization': viz_recommendation,
                'agent': self.name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update context with Q&A
            self.write_context(f"""## Query Result

Question: {question}
Answer: {final_answer}
""")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _analyze_intent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query intent and requirements."""
        try:
            question = task.get('data', {}).get('question', '')
            
            prompt = f"""Analyze this query to understand user intent:

Query: {question}

Determine:
1. Query type (factual, aggregation, comparison, trend, explanation)
2. Required data sources
3. Metrics or entities involved
4. Time period if specified
5. Desired output format
6. Calculation requirements

Return as JSON with:
- intent_type: Main query type
- data_needs: Required data elements
- metrics: Specific metrics mentioned
- time_range: Time period if any
- output_format: Best format for answer
- complexity: low/medium/high"""
            
            response = self.call_claude(prompt, self.system_prompt)
            intent = self._parse_json_response(response)
            
            return {
                'status': 'success',
                'intent': intent,
                'question': question,
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _aggregate_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data aggregation for complex queries."""
        try:
            metrics = task.get('data', {}).get('metrics', [])
            operation = task.get('data', {}).get('operation', 'sum')
            group_by = task.get('data', {}).get('group_by', None)
            
            prompt = f"""Perform this aggregation operation:

Metrics:
{json.dumps(metrics[:20], indent=2)}  # Limit for context

Operation: {operation}
Group By: {group_by}

Calculate the result and provide:
- aggregated_value: The calculated result
- units: Unit of measurement
- calculation_method: How you calculated it
- confidence: Confidence in the result
- breakdown: Component values if grouped

Return as JSON."""
            
            response = self.call_claude(prompt, self.system_prompt)
            aggregation = self._parse_json_response(response)
            
            return {
                'status': 'success',
                'aggregation': aggregation,
                'operation': operation,
                'metrics_count': len(metrics),
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _create_visualization(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization recommendations."""
        try:
            data = task.get('data', {})
            query_type = data.get('query_type', '')
            metrics = data.get('metrics', [])
            
            prompt = f"""Recommend visualization for this data:

Query Type: {query_type}
Sample Data:
{json.dumps(metrics[:5], indent=2)}

Provide visualization spec:
- chart_type: bar/line/pie/scatter/heatmap
- x_axis: Field for x-axis
- y_axis: Field for y-axis
- series: Data series if multiple
- title: Chart title
- description: What the chart shows
- config: Additional chart configuration

Return as JSON compatible with Recharts/shadcn."""
            
            response = self.call_claude(prompt, self.system_prompt)
            viz_spec = self._parse_json_response(response)
            
            return {
                'status': 'success',
                'visualization': viz_spec,
                'data_points': len(metrics),
                'agent': self.name
            }
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return {'status': 'error', 'error': str(e), 'agent': self.name}
    
    def _general_query_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general query tasks."""
        try:
            prompt = f"""Execute this query-related task:

Task: {json.dumps(task, indent=2)}

Context:
{self.read_context()}

Apply your expertise in natural language understanding and answer generation."""
            
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
    
    def _answer_with_claude(self, question: str, context: str) -> str:
        """Generate answer using Claude when GPT unavailable."""
        prompt = f"""Answer this question based on available context:

Question: {question}

Available Context:
{context}

Provide a clear, accurate answer with:
- Specific values and units
- Source citations
- Any relevant caveats

If information is not available, say so clearly."""
        
        return self.call_claude(prompt, self.system_prompt)
    
    def _enhance_answer(self, gpt_answer: str, citations: List[Dict], 
                       question: str) -> Dict[str, Any]:
        """Enhance GPT answer with Claude's analysis."""
        try:
            prompt = f"""Review and enhance this answer:

Question: {question}
Initial Answer: {gpt_answer}
Citations: {json.dumps(citations, indent=2)}

Improve the answer by:
1. Ensuring accuracy and completeness
2. Adding context or explanations
3. Formatting citations properly
4. Highlighting key insights
5. Suggesting follow-up questions

Return JSON with:
- answer: Enhanced answer text
- key_insights: List of main points
- follow_up_questions: Suggested follow-ups"""
            
            response = self.call_claude(prompt, self.system_prompt)
            enhancement = self._parse_json_response(response)
            
            if isinstance(enhancement, dict) and 'answer' in enhancement:
                return enhancement
            else:
                return {'answer': gpt_answer}
                
        except Exception as e:
            logger.error(f"Answer enhancement failed: {e}")
            return {'answer': gpt_answer}
    
    def _recommend_visualization(self, question: str, answer: str) -> Optional[Dict[str, Any]]:
        """Recommend visualization if helpful."""
        try:
            prompt = f"""Determine if visualization would help for this Q&A:

Question: {question}
Answer: {answer}

If visualization would help, provide:
- should_visualize: true/false
- chart_type: Recommended chart type
- reason: Why this visualization helps

Return as JSON."""
            
            response = self.call_claude(prompt, self.system_prompt)
            recommendation = self._parse_json_response(response)
            
            if recommendation.get('should_visualize'):
                return recommendation
            return None
            
        except Exception as e:
            logger.debug(f"Visualization recommendation failed: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from response."""
        try:
            if "```json" in response:
                json_start = response.index("```json") + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            elif "{" in response or "[" in response:
                # Find JSON structure
                if "[" in response and ("]" not in response or response.index("[") < response.index("{")):
                    json_start = response.index("[")
                    json_end = response.rindex("]") + 1
                else:
                    json_start = response.index("{")
                    json_end = response.rindex("}") + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"content": response}
                
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Could not parse JSON: {e}")
            return {"content": response}
    
    def search_and_answer(self, question: str) -> Dict[str, Any]:
        """
        High-level method to search and answer questions.
        
        Args:
            question: User's question
            
        Returns:
            Complete answer with citations
        """
        try:
            # Analyze intent
            intent_task = {'type': 'analyze_intent', 'data': {'question': question}}
            intent_result = self.execute(intent_task)
            
            # Process query
            query_task = {'type': 'query', 'data': {'question': question}}
            query_result = self.execute(query_task)
            
            # Add intent analysis to result
            if intent_result.get('status') == 'success':
                query_result['intent'] = intent_result.get('intent')
            
            return query_result
            
        except Exception as e:
            logger.error(f"Search and answer failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'question': question,
                'agent': self.name
            }