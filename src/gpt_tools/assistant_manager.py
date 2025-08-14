"""
OpenAI Assistants API Manager for ImpactOS.

Manages GPT assistants with file search and code interpreter capabilities.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI
import time
import sqlite3
from .data_store import DataStore

logger = logging.getLogger(__name__)


class AssistantManager:
    """Manages OpenAI Assistants for data processing and querying."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Assistant Manager.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.assistants: Dict[str, str] = {}  # name -> assistant_id
        self.threads: Dict[str, str] = {}  # session_id -> thread_id
        self.data_store = DataStore()
        
        self._initialize_assistants()
    
    def _initialize_assistants(self):
        """Initialize or retrieve existing assistants."""
        try:
            # Data Processing Assistant
            self.assistants['data_processor'] = self._create_or_get_assistant(
                name="ImpactOS Data Processor",
                instructions="""You are a data processing assistant for social impact metrics.
                Your role is to:
                1. Extract structured data from uploaded files (CSV, Excel, PDF, text)
                2. Identify impact metrics with their actual values, units, and time periods
                3. Map metrics to frameworks (UN SDGs, MAC, TOMs, B Corp) 
                4. Validate data quality and completeness
                5. Return data in a structured JSON format with proper metric extraction
                
                When processing data:
                - Look for numeric values and their associated metric names
                - Preserve actual data values (don't return "None" for real values)
                - Identify currency amounts, percentages, counts, and measurements
                - Group related metrics by category (Environmental, Social, Governance)
                - Include time periods when available (quarters, years, dates)
                
                Always return results as structured JSON with this format:
                {
                  "metrics": [
                    {
                      "name": "metric name",
                      "value": "actual numeric value",
                      "unit": "currency/percentage/count/etc",
                      "category": "Environmental/Social/Governance", 
                      "period": "time period if available"
                    }
                  ],
                  "summary": "brief summary of data processed"
                }
                
                Always provide citations with specific file names and locations.""",
                tools=[
                    {"type": "file_search"},
                    {"type": "code_interpreter"}
                ],
                model="gpt-4-turbo"
            )
            
            # Query Assistant
            self.assistants['query_handler'] = self._create_or_get_assistant(
                name="ImpactOS Query Handler",
                instructions="""You are a comprehensive query assistant for social impact data analysis with access to detailed organizational data files.

                PRIORITY: Always search uploaded files first using your file_search capability to find specific, granular information before providing any response.

                Your capabilities:
                1. **Granular Data Access**: Search through uploaded Excel/CSV files for specific employee records, individual transactions, and detailed data points
                2. **Row-Level Analysis**: Access individual employee data, specific time periods, and exact values from source files  
                3. **Cross-File Search**: Find related information across multiple uploaded files (payroll, environmental, HR data)
                4. **Summary Analytics**: Perform calculations, aggregations, and trend analysis when requested
                5. **Pattern Recognition**: Identify insights, correlations, and patterns in the detailed data

                RESPONSE GUIDELINES:
                
                **For Specific/Granular Queries** (e.g., "Show me Person 1's data", "What did Employee X earn?"):
                - ALWAYS search the uploaded files first using file_search
                - Provide exact values, names, dates, and detailed breakdowns from original data
                - Include specific employee IDs, individual records, and row-level details
                - Quote exact figures from the source files
                - Show data for specific time periods when requested

                **For Summary/Analytical Queries** (e.g., "What's the average salary?", "Carbon trends?"):
                - Search uploaded files for raw data first, then analyze
                - Perform calculations on the detailed data
                - Provide aggregated insights with supporting detail
                - Reference specific data points that support your analysis

                **Always Include**:
                - Specific citations from uploaded files using [1], [2] notation
                - Exact numerical values when available
                - Source file names and relevant sections
                - Clear indication of data availability/limitations

                **Data Types Available**: Payroll data, environmental metrics, employee demographics, wellness data, carbon reporting, compensation details, and more.

                Remember: Your strength is accessing the complete granular dataset - use it to provide detailed, accurate, and specific answers.""",
                tools=[
                    {"type": "file_search"},
                    {"type": "code_interpreter"}
                ],
                model="gpt-4-turbo"
            )
            
            logger.info(f"Initialized assistants: {list(self.assistants.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize assistants: {e}")
            raise
    
    def _create_or_get_assistant(self, name: str, instructions: str, 
                                 tools: List[Dict], model: str) -> str:
        """Create new assistant or get existing one by name."""
        try:
            # Check for existing assistant
            existing = self.client.beta.assistants.list()
            for assistant in existing.data:
                if assistant.name == name:
                    logger.info(f"Using existing assistant: {name}")
                    return assistant.id
            
            # Create new assistant
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=tools,
                model=model
            )
            logger.info(f"Created new assistant: {name}")
            return assistant.id
            
        except Exception as e:
            logger.error(f"Error creating/getting assistant {name}: {e}")
            raise
    
    
    def process_data(self, file_ids: List[str], instructions: str) -> Dict[str, Any]:
        """
        Process data files using data processor assistant.
        
        Args:
            file_ids: List of uploaded file IDs
            instructions: Specific processing instructions
            
        Returns:
            Processed data results
        """
        try:
            # Create thread
            thread = self.client.beta.threads.create()
            
            # Note: Modern OpenAI Assistants API handles file search automatically
            # Vector stores are managed internally, no explicit creation needed
            logger.info("Using built-in file search - no manual vector store management needed")
            
            # Create message with file attachments for file search
            # Files are attached directly to enable the assistant's file_search tool
            attachments = [{"file_id": fid, "tools": [{"type": "file_search"}]} for fid in file_ids]
            
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=instructions,
                attachments=attachments
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistants['data_processor']
            )
            
            # Wait for completion
            result = self._wait_for_run(thread.id, run.id)
            
            # Parse result
            parsed_result = self._parse_result(result)
            
            # Store metrics if extracted
            if "structured_data" in parsed_result and isinstance(parsed_result["structured_data"], dict):
                metrics = parsed_result["structured_data"].get("metrics", [])
                if metrics:
                    # Try to store metrics for each file
                    for file_id in file_ids:
                        try:
                            # Get file info from data store - check all files, not just processed ones
                            existing_file = None
                            
                            # First check all files
                            with sqlite3.connect(self.data_store.db_path) as conn:
                                conn.row_factory = sqlite3.Row
                                cursor = conn.execute("""
                                    SELECT * FROM files WHERE openai_file_id = ?
                                """, (file_id,))
                                row = cursor.fetchone()
                                if row:
                                    existing_file = dict(row)
                            
                            if existing_file:
                                self.data_store.store_metrics(existing_file['id'], metrics)
                                logger.info(f"Stored {len(metrics)} metrics for file {file_id}")
                            else:
                                logger.warning(f"File not found in database for storing metrics: {file_id}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to store metrics for file {file_id}: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise
    
    def query_data(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query data using query handler assistant.
        
        Args:
            question: Natural language question
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Query response with answer and metadata
        """
        try:
            # Get or create thread - try to use recent thread for context
            if session_id and session_id in self.threads:
                thread_id = self.threads[session_id]
            else:
                # Try to get recent thread for data continuity
                recent_thread = self.data_store.get_recent_thread()
                if recent_thread:
                    thread_id = recent_thread
                    logger.info("Using recent thread for data continuity")
                else:
                    thread = self.client.beta.threads.create()
                    thread_id = thread.id
                
                if session_id:
                    self.threads[session_id] = thread_id
                    self.data_store.store_session_info(session_id, thread_id)
            
            # Enhance question with context from stored data
            enhanced_question = self._enhance_query_with_context(question)
            
            # Get all uploaded file IDs from database for context
            uploaded_files = self.data_store.get_all_files()
            file_attachments = []
            if uploaded_files:
                file_attachments = [
                    {"file_id": file_info["openai_file_id"], "tools": [{"type": "file_search"}]}
                    for file_info in uploaded_files
                    if file_info.get("openai_file_id")
                ]
                logger.info(f"Attaching {len(file_attachments)} files to query for data access")
            
            # Create message with file attachments for comprehensive data access
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=enhanced_question,
                attachments=file_attachments if file_attachments else None
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistants['query_handler']
            )
            
            # Wait for completion
            result = self._wait_for_run(thread_id, run.id)
            
            return self._parse_result(result)
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def _wait_for_run(self, thread_id: str, run_id: str, timeout: int = 60) -> Any:
        """Wait for assistant run to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status == 'completed':
                # Get messages
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread_id,
                    order='desc',
                    limit=1
                )
                return messages.data[0] if messages.data else None
            
            elif run.status in ['failed', 'cancelled', 'expired']:
                raise Exception(f"Run {run.status}: {run.last_error}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Run timed out after {timeout} seconds")
    
    def _parse_result(self, message) -> Dict[str, Any]:
        """Parse assistant message into structured result."""
        if not message:
            return {"error": "No response from assistant"}
        
        result = {
            "content": "",
            "annotations": [],
            "files": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract text content
        for content in message.content:
            if content.type == 'text':
                result["content"] = content.text.value
                
                # Extract annotations (citations)
                if hasattr(content.text, 'annotations'):
                    for ann in content.text.annotations:
                        if ann.type == 'file_citation':
                            citation_data = {
                                "type": "citation",
                                "file_id": ann.file_citation.file_id
                            }
                            # Quote might not always be available
                            if hasattr(ann.file_citation, 'quote'):
                                citation_data["quote"] = ann.file_citation.quote
                            result["annotations"].append(citation_data)
            
            elif content.type == 'image_file':
                result["files"].append({
                    "type": "image",
                    "file_id": content.image_file.file_id
                })
        
        # Try to extract JSON if present
        content_str = result["content"]
        if "```json" in content_str:
            try:
                json_start = content_str.index("```json") + 7
                json_end = content_str.index("```", json_start)
                json_str = content_str[json_start:json_end].strip()
                result["structured_data"] = json.loads(json_str)
            except (ValueError, json.JSONDecodeError) as e:
                logger.debug(f"No valid JSON found in response: {e}")
        
        return result
    
    def _detect_query_type(self, question: str) -> str:
        """
        Detect whether query is asking for granular/specific data or summary/analytical data.
        
        Returns: 'granular' or 'summary'
        """
        question_lower = question.lower()
        
        # Granular query indicators - user wants specific records/individual data
        granular_indicators = [
            # Specific individuals/records
            'person', 'employee', 'emp', 'individual', 'worker', 'staff member',
            'who', 'which employee', 'which person',
            
            # Specific identifiers
            'emp001', 'emp002', 'emp003', 'employee id', 'id', 'name',
            
            # Specific time periods
            'in 2021', 'in 2022', 'in 2023', 'january', 'february', 'march',
            'q1', 'q2', 'q3', 'q4', 'quarter',
            
            # Row-level data requests  
            'show me all', 'list all', 'give me details', 'breakdown',
            'each', 'every', 'individual', 'specific', 'detailed',
            'row by row', 'line by line',
            
            # Comparative individual queries
            'compare employees', 'employee comparison', 'who earned',
            'highest paid', 'lowest paid', 'top performer', 'bottom performer'
        ]
        
        # Summary query indicators - user wants aggregated/analytical data
        summary_indicators = [
            # Aggregate functions
            'average', 'mean', 'total', 'sum', 'overall', 'combined',
            'median', 'maximum', 'minimum', 'range',
            
            # Trend analysis
            'trend', 'pattern', 'over time', 'growth', 'decline', 'change',
            'increasing', 'decreasing', 'progression', 'evolution',
            
            # Statistical analysis
            'distribution', 'correlation', 'analysis', 'statistics',
            'percentage', 'proportion', 'ratio',
            
            # High-level questions
            'overview', 'summary', 'generally', 'typically', 'on average',
            'across all', 'organization', 'company wide', 'enterprise'
        ]
        
        # Count indicators
        granular_score = sum(1 for indicator in granular_indicators if indicator in question_lower)
        summary_score = sum(1 for indicator in summary_indicators if indicator in question_lower)
        
        # Special case: if question mentions specific names/IDs, it's definitely granular
        if any(term in question_lower for term in ['person 1', 'person 2', 'person 3', 'emp001', 'emp002', 'emp003']):
            return 'granular'
            
        # If more granular indicators, it's granular
        if granular_score > summary_score:
            return 'granular'
        elif summary_score > 0:
            return 'summary'
        else:
            # Default: if no clear indicators, assume granular for detailed responses
            return 'granular'

    def _enhance_query_with_context(self, question: str) -> str:
        """Enhance query with context based on query type."""
        try:
            query_type = self._detect_query_type(question)
            logger.info(f"Query type detected: {query_type}")
            
            if query_type == 'granular':
                # For granular queries, direct GPT to search files with minimal context
                return self._enhance_granular_query(question)
            else:
                # For summary queries, use rich context with aggregated metrics
                return self._enhance_summary_query(question)
            
        except Exception as e:
            logger.warning(f"Failed to enhance query with context: {e}")
            return question
    
    def _enhance_granular_query(self, question: str) -> str:
        """Enhance granular queries to direct GPT to search uploaded files."""
        # Get basic file info
        summary = self.data_store.get_data_summary()
        
        enhanced_parts = [
            question,
            "",
            "Please search through the uploaded data files to find specific, detailed information to answer this question.",
            "Look for individual records, specific employees, transactions, or data points.",
            "Provide exact values, names, dates, and detailed breakdowns from the original data.",
        ]
        
        if summary.get('files_processed', 0) > 0:
            enhanced_parts.append(f"Available data: {summary['files_processed']} files with detailed records.")
        
        return '\n'.join(enhanced_parts)
    
    def _enhance_summary_query(self, question: str) -> str:
        """Enhance summary queries with aggregated context."""
        # Original logic for summary/analytical queries
        summary = self.data_store.get_data_summary()
        
        # Search for relevant metrics based on question keywords
        keywords = question.lower().split()
        relevant_metrics = []
        
        # Look for specific metric types mentioned
        metric_keywords = ['pay', 'salary', 'gender', 'bonus', 'donation', 'carbon', 'emissions', 'energy', 'waste']
        for keyword in metric_keywords:
            if keyword in question.lower():
                metrics = self.data_store.search_metrics(query=keyword, limit=10)
                relevant_metrics.extend(metrics)
        
        if not relevant_metrics and summary.get('metrics_extracted', 0) > 0:
            # Get some sample metrics if no specific matches
            relevant_metrics = self.data_store.search_metrics(limit=5)
        
        context_parts = [question]
        
        if summary:
            context_parts.append(f"\nData Context: {summary['files_processed']} files processed, {summary['metrics_extracted']} metrics available.")
            
            if summary.get('categories'):
                categories = ', '.join(summary['categories'].keys())
                context_parts.append(f"Available categories: {categories}")
        
        if relevant_metrics:
            context_parts.append("\nRelevant aggregated metrics:")
            for metric in relevant_metrics[:3]:  # Show top 3
                context_parts.append(f"- {metric['name']}: {metric['value']} {metric['unit']} ({metric['category']})")
            
            if len(relevant_metrics) > 3:
                context_parts.append(f"... and {len(relevant_metrics) - 3} more metrics")
        
        enhanced_query = '\n'.join(context_parts)
        
        if len(relevant_metrics) > 0:
            logger.debug(f"Enhanced summary query with context from {len(relevant_metrics)} relevant metrics")
        
        return enhanced_query
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear thread references
            self.threads.clear()
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")