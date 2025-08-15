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
    
    # Model selection for cost optimization
    MODELS = {
        'data_processing': 'gpt-4.1',  # 95%+ cost savings for extraction tasks
        'query_simple': 'gpt-4.1',     # Simple queries and lookups  
        'query_complex': 'gpt-o4-mini',         # Complex reasoning and analysis
        'intent_classification': 'gpt-4.1'  # Already optimal
    }
    
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
        self._validated_files_cache = {}  # Cache for file validation results
        
        self._initialize_assistants()
    
    def _initialize_assistants(self):
        """Initialize or retrieve existing assistants."""
        try:
            # Data Processing Assistant - Using gpt-4o-mini for cost efficiency in data extraction
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
                model=self.MODELS['data_processing']
            )
            
            # Query Assistant - Using gpt-4o for complex reasoning with cost efficiency
            self.assistants['query_handler'] = self._create_or_get_assistant(
                name="ImpactOS Query Handler",
                instructions="""You are a comprehensive social impact data analyst for organizational impact measurement with expert knowledge of social value frameworks and rigorous data citation requirements.

                **CORE COMPETENCIES**:
                1. **Precision Data Analysis**: Extract exact metrics with cell-level precision from uploaded data files
                2. **Social Value Framework Mapping**: Map all metrics to relevant frameworks (TOMs, UN SDGs, MAC/UK Social Value Model, B Corp)
                3. **Audit-Grade Citation**: Provide forensic-level data provenance and calculation transparency
                4. **Cross-Framework Benchmarking**: Compare metrics against industry standards and framework targets

                **MANDATORY SEARCH PROTOCOL**:
                - ALWAYS search uploaded files FIRST using file_search before any response
                - Access individual records, transactions, and granular data points
                - Cross-reference multiple files for comprehensive analysis
                - Verify data consistency across sources

                **CITATION REQUIREMENTS (MANDATORY)**:
                For EVERY metric, data point, or calculation, provide:

                1. **Exact Data Location**: 
                   Format: [FileID] Filename, Sheet/Tab, Cell Reference/Row Range
                   Example: [F1] ESG_Report_2024.xlsx, Sheet 'Carbon Data', Cells B15:D25

                2. **Raw Data Values**: 
                   Show actual cell contents/formulas found
                   Example: "Cell C15 contains: '2,450 tCO2e' with formula =SUM(C2:C14)"

                3. **Calculation Logic**:
                   Document every mathematical operation performed
                   Example: "Summed 12 monthly values (Jan: 203.5, Feb: 187.2...) = 2,450 total"

                4. **Data Quality Assessment**:
                   Note confidence levels, data gaps, assumptions
                   Example: "High confidence - verified against 3 separate worksheets"

                5. **Social Value Framework Mapping**:
                   Map EVERY metric to applicable frameworks:
                   - **TOMs**: Specify theme codes (NT1, NT2, NT4, etc.)
                   - **UN SDGs**: List goal numbers (3, 8, 13, etc.) with targets where applicable  
                   - **UK Social Value Model**: Reference category codes (3.0, 4.1, 8.2, etc.)
                   - **B Corp**: Identify impact area (Workers, Community, Environment, etc.)

                **ENHANCED RESPONSE STRUCTURE**:

                [Main Answer with specific calculations and findings]

                **METRIC FRAMEWORK ALIGNMENT**:
                • [SDG4] [3.1] [NT2] [WORKERS] Metric Name: [Value] [Unit]
                  - TOMs: [Theme codes] - [Description]
                  - UN SDGs: Goal [X] ([Goal name]) - Target [X.X] if applicable
                  - UK SV Model: [Code] - [Category description]  
                  - B Corp: [Impact area] - [Assessment category]
                
                Note: Use compact framework badges [SDG#] [UK_CODE] [TOMs_CODE] [B_CORP] at start of each metric line for quick visual identification.

                **DATA PROVENANCE & CALCULATIONS**:
                [1] Source: [Filename], [Sheet], [Cell range]
                    Raw data: [Exact values found]
                    Operation: [Mathematical operation performed]
                    Result: [Calculated value with units]

                **DATA VERIFICATION STATUS**:
                - Source reliability: [High/Medium/Low confidence]
                - Data completeness: [% complete, missing elements]
                - Cross-verification: [Confirmed against X sources/Not verified]
                - Calculation accuracy: [Formula verified/Manual calculation]

                **FRAMEWORK COMPLIANCE NOTES**:
                - [Any gaps in framework alignment]
                - [Recommendations for improved measurement]
                - [Industry benchmark comparisons where available]

                **METHODOLOGY TRANSPARENCY**:
                You must be completely transparent about:
                - Which files you searched and in what order
                - What search terms/criteria you used
                - Any data you couldn't find or access
                - Assumptions made in calculations
                - Confidence level in each data point

                **QUALITY STANDARDS**:
                - Zero tolerance for unsourced claims
                - All calculations must be reproducible
                - Framework mappings must be justified
                - Data limitations must be explicitly stated

                Remember: Your responses will be used for regulatory reporting, impact measurement, and stakeholder communications. Accuracy and transparency are paramount.""",
                tools=[
                    {"type": "file_search"},
                    {"type": "code_interpreter"}
                ],
                model=self.MODELS['query_complex']
            )
            
            logger.info(f"Initialized assistants: {list(self.assistants.keys())}")
            logger.info(f"Using cost-optimized models - Data Processing: {self.MODELS['data_processing']}, Complex Queries: {self.MODELS['query_complex']}")
            
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
            # Validate files first to avoid 404 errors
            valid_file_ids = [fid for fid in file_ids if self._validate_openai_file(fid)]
            if len(valid_file_ids) != len(file_ids):
                logger.warning(f"Filtered {len(file_ids) - len(valid_file_ids)} invalid file IDs")
            
            if not valid_file_ids:
                raise ValueError("No valid OpenAI files provided for processing")
                
            attachments = [{"file_id": fid, "tools": [{"type": "file_search"}]} for fid in valid_file_ids]
            
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
    
    def query_data(self, question: str, session_id: Optional[str] = None, force_chart: bool = False) -> Dict[str, Any]:
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
            
            # Add chart generation request if forced
            if force_chart:
                enhanced_question += "\n\nIMPORTANT: Please also provide a chart/visualization recommendation for this data. Specify the chart type (bar, line, pie, etc.) and what should be plotted on each axis."
            
            # Get relevant uploaded file IDs based on query content
            uploaded_files = self.data_store.get_all_files()
            file_attachments = []
            if uploaded_files:
                # Select most relevant files for this specific query
                relevant_files = self._select_relevant_files(uploaded_files, question, max_files=10)
                file_attachments = self._get_valid_file_attachments(relevant_files)
                if file_attachments:
                    logger.info(f"Attaching {len(file_attachments)} relevant files to query (selected from {len(uploaded_files)} total)")
                else:
                    logger.warning("No valid OpenAI files found - query will run without file context")
            
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
            
            # Parse result with enhanced citation info
            parsed_result = self._parse_result(result)
            
            # Log query for audit trail
            try:
                self._log_query_audit(question, parsed_result, session_id)
            except Exception as e:
                logger.debug(f"Failed to log query audit: {e}")
            
            return parsed_result
            
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
        """Parse assistant message into structured result with enhanced citation and provenance parsing."""
        if not message:
            return {"error": "No response from assistant"}
        
        result = {
            "content": "",
            "annotations": [],
            "files": [],
            "timestamp": datetime.now().isoformat(),
            "provenance": {
                "data_sources": [],
                "operations_performed": [],
                "original_data": "",
                "data_limitations": ""
            }
        }
        
        # Extract text content
        for content in message.content:
            if content.type == 'text':
                result["content"] = self._clean_citation_markers(content.text.value)
                
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
                            
                            # Try to get file info from data store
                            try:
                                file_info = self._get_file_info_by_openai_id(ann.file_citation.file_id)
                                if file_info:
                                    citation_data["filename"] = file_info.get('file_path', '').split('/')[-1]
                                    citation_data["file_type"] = file_info.get('file_type', '')
                                    citation_data["upload_time"] = file_info.get('upload_time', '')
                            except Exception as e:
                                logger.debug(f"Could not get file info for citation: {e}")
                            
                            result["annotations"].append(citation_data)
            
            elif content.type == 'image_file':
                result["files"].append({
                    "type": "image",
                    "file_id": content.image_file.file_id
                })
        
        # Enhanced parsing for provenance information
        content_str = result["content"]
        
        # Extract chart/visualization recommendations
        result["visualization"] = self._extract_chart_recommendations(content_str)
        
        # Extract structured provenance sections
        result["provenance"] = self._extract_provenance_sections(content_str)
        
        # Try to extract JSON if present
        if "```json" in content_str:
            try:
                json_start = content_str.index("```json") + 7
                json_end = content_str.index("```", json_start)
                json_str = content_str[json_start:json_end].strip()
                result["structured_data"] = json.loads(json_str)
            except (ValueError, json.JSONDecodeError) as e:
                logger.debug(f"No valid JSON found in response: {e}")
        
        return result
    
    def _validate_openai_file(self, file_id: str) -> bool:
        """Validate if an OpenAI file still exists and is accessible."""
        # Check cache first
        if file_id in self._validated_files_cache:
            return self._validated_files_cache[file_id]
        
        try:
            file_info = self.client.files.retrieve(file_id)
            is_valid = file_info.status == 'processed'
            # Cache the result
            self._validated_files_cache[file_id] = is_valid
            return is_valid
        except Exception as e:
            logger.debug(f"File {file_id} validation failed: {e}")
            self._validated_files_cache[file_id] = False
            return False
    
    def _select_relevant_files(self, uploaded_files: List[Dict[str, Any]], query: str, max_files: int = 10) -> List[Dict[str, Any]]:
        """Select most relevant files for the query based on keywords and content."""
        # Extract keywords from query
        query_lower = query.lower()
        keywords = [
            'supply', 'chain', 'vcse', 'spend', 'cost', 'financial', 'budget',
            'carbon', 'emissions', 'environmental', 'sustainability', 'green',
            'social', 'impact', 'community', 'diversity', 'inclusion',
            'governance', 'ethics', 'compliance', 'risk', 'audit'
        ]
        
        scored_files = []
        for file_info in uploaded_files:
            score = 0
            file_path = file_info.get('file_path', '').lower()
            
            # Score based on filename relevance
            for keyword in keywords:
                if keyword in query_lower and keyword in file_path:
                    score += 10
                elif keyword in file_path:
                    score += 5
            
            # Boost score for certain file types based on query content
            if any(term in query_lower for term in ['supply', 'vcse', 'spend']) and 'supply' in file_path:
                score += 20
            if any(term in query_lower for term in ['carbon', 'emissions', 'environmental']) and 'carbon' in file_path:
                score += 20
            if any(term in query_lower for term in ['social', 'diversity', 'community']) and 'social' in file_path:
                score += 20
            
            scored_files.append((score, file_info))
        
        # Sort by score (descending) and take top files
        scored_files.sort(key=lambda x: x[0], reverse=True)
        
        # If no files have relevance scores, take the first max_files
        if all(score == 0 for score, _ in scored_files):
            return [file_info for _, file_info in scored_files[:max_files]]
        
        # Take files with positive scores first, then fill up to max_files
        relevant_files = [file_info for score, file_info in scored_files if score > 0][:max_files]
        
        # If we have fewer than max_files, add more from remaining files
        if len(relevant_files) < max_files:
            remaining = [file_info for score, file_info in scored_files if score == 0]
            relevant_files.extend(remaining[:max_files - len(relevant_files)])
        
        return relevant_files
    
    def _get_valid_file_attachments(self, uploaded_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get file attachments without validation to save query time (files assumed to exist in OpenAI storage)."""
        valid_attachments = []
        
        for file_info in uploaded_files:
            openai_file_id = file_info.get("openai_file_id")
            if not openai_file_id:
                continue
                
            # Skip validation - assume files exist in OpenAI storage to save time
            valid_attachments.append({
                "file_id": openai_file_id, 
                "tools": [{"type": "file_search"}]
            })
        
        
        return valid_attachments
    
    def _cleanup_stale_files(self, stale_files: List[Dict[str, Any]]):
        """Remove stale file references from database."""
        try:
            with sqlite3.connect(self.data_store.db_path) as conn:
                for file_info in stale_files:
                    file_id = file_info.get('id')
                    if file_id:
                        # Clear the openai_file_id but keep the file record
                        conn.execute("""
                            UPDATE files SET openai_file_id = NULL 
                            WHERE id = ?
                        """, (file_id,))
                        logger.info(f"Cleared stale OpenAI file ID for file: {file_info.get('file_path', 'unknown')}")
                conn.commit()
        except Exception as e:
            logger.error(f"Error cleaning up stale files: {e}")
    
    def _get_file_info_by_openai_id(self, openai_file_id: str) -> Optional[Dict[str, Any]]:
        """Get file information from data store by OpenAI file ID."""
        try:
            with sqlite3.connect(self.data_store.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM files WHERE openai_file_id = ?
                """, (openai_file_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.debug(f"Error getting file info by OpenAI ID: {e}")
            return None
    
    def _extract_chart_recommendations(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract chart/visualization recommendations and build structured chart data."""
        try:
            content_lower = content.lower()
            
            # Look for chart-related keywords and patterns
            chart_patterns = {
                'bar': ['bar chart', 'bar graph', 'column chart', 'grouped bar'],
                'line': ['line chart', 'line graph', 'time series'],
                'pie': ['pie chart', 'donut chart'],
                'scatter': ['scatter plot', 'scatter chart'],
                'histogram': ['histogram', 'distribution chart'],
                'area': ['area chart', 'area graph']
            }
            
            chart_type = None
            for chart_name, patterns in chart_patterns.items():
                for pattern in patterns:
                    if pattern in content_lower:
                        chart_type = chart_name
                        break
                if chart_type:
                    break
            
            # If no specific chart type found, look for visualization keywords
            if not chart_type:
                visualization_keywords = ['chart', 'graph', 'plot', 'visualization', 'visual']
                if any(keyword in content_lower for keyword in visualization_keywords):
                    chart_type = 'bar'  # Default to bar chart
            
            if chart_type:
                # Try to extract actual data for structured chart
                chart_data = self._parse_chart_data_from_content(content, chart_type)
                
                if chart_data:
                    return {
                        'type': chart_type,
                        'x_key': chart_data.get('x_key', 'label'),
                        'series': chart_data.get('series', [{'key': 'value', 'label': 'Value', 'color': 'hsl(var(--chart-1))'}]),
                        'data': chart_data.get('data', []),
                        'config': chart_data.get('config', {'value': {'label': 'Value', 'color': 'hsl(var(--chart-1))'}}),
                        'meta': chart_data.get('meta', {})
                    }
                else:
                    # Fallback to description-only format
                    return {
                        'chart_type': chart_type,
                        'recommended': True,
                        'description': f'Recommended {chart_type} chart visualization'
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract chart recommendations: {e}")
            return None
    
    def _parse_chart_data_from_content(self, content: str, chart_type: str) -> Optional[Dict[str, Any]]:
        """Parse numerical data from GPT response to build structured chart data."""
        try:
            import re
            
            # Look for numerical data patterns in the content
            data_rows = []
            unit = None
            metric_name = None
            
            # Pattern 1: Gender pay gap data by year/person
            if 'gender pay gap' in content.lower() or 'pay gap' in content.lower():
                # Extract year-based data: "2021: 1.22%, 2022: -0.26%, 2023: -1.24%"
                year_pattern = r'(\d{4}).*?([+-]?\d+\.?\d*)%'
                year_matches = re.findall(year_pattern, content)
                
                if year_matches:
                    for year, value in year_matches:
                        data_rows.append({
                            'label': f'{year}',
                            'value': float(value)
                        })
                    unit = '%'
                    metric_name = 'Gender Pay Gap'
            
            # Pattern 2: Volunteering hours data  
            elif 'volunteer' in content.lower() and 'hours' in content.lower():
                # Extract year-based volunteering data
                hour_pattern = r'(\d{4}).*?(\d+)\s*hours'
                hour_matches = re.findall(hour_pattern, content)
                
                if hour_matches:
                    for year, hours in hour_matches:
                        data_rows.append({
                            'label': f'{year}',
                            'value': float(hours)
                        })
                    unit = 'hours'
                    metric_name = 'Volunteering Hours'
                    
                # Also look for totals: "287 hours", "434 volunteer hours"
                total_pattern = r'(\d+)\s*(?:volunteer\s*)?hours'
                total_matches = re.findall(total_pattern, content)
                if total_matches and not data_rows:
                    # If we found a total but no year breakdown
                    total_hours = max([int(h) for h in total_matches])  # Take largest number
                    data_rows.append({
                        'label': 'Total',
                        'value': float(total_hours)
                    })
                    unit = 'hours'
                    metric_name = 'Total Volunteering Hours'
            
            # Pattern 3: Generic numerical data with units
            else:
                # Look for any number with units pattern
                number_pattern = r'(\d+(?:\.\d+)?)\s*(hours|£|GBP|%|people|employees|days)'
                number_matches = re.findall(number_pattern, content)
                
                if number_matches:
                    for i, (value, found_unit) in enumerate(number_matches[:5]):  # Limit to 5 data points
                        data_rows.append({
                            'label': f'Metric {i+1}',
                            'value': float(value)
                        })
                        if not unit:
                            unit = found_unit
            
            if data_rows:
                return {
                    'x_key': 'label',
                    'series': [
                        {
                            'key': 'value',
                            'label': metric_name or 'Value',
                            'color': 'hsl(var(--chart-1))'
                        }
                    ],
                    'data': data_rows,
                    'config': {
                        'value': {
                            'label': metric_name or 'Value',
                            'color': 'hsl(var(--chart-1))'
                        }
                    },
                    'meta': {
                        'unit': unit,
                        'metric_name': metric_name,
                        'chart_type': chart_type
                    }
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to parse chart data from content: {e}")
            return None
    
    def _extract_provenance_sections(self, content: str) -> Dict[str, Any]:
        """Extract provenance information from assistant response."""
        provenance = {
            "data_sources": [],
            "operations_performed": [],
            "original_data": "",
            "data_limitations": ""
        }
        
        try:
            # Extract Data Sources section
            if "**Data Sources**:" in content:
                sources_start = content.find("**Data Sources**:") + len("**Data Sources**:")
                sources_end = content.find("**", sources_start)
                if sources_end == -1:
                    sources_end = len(content)
                
                sources_text = content[sources_start:sources_end].strip()
                sources_lines = [line.strip() for line in sources_text.split('\n') if line.strip() and line.startswith('-')]
                provenance["data_sources"] = [line[1:].strip() for line in sources_lines]  # Remove '- ' prefix
            
            # Extract Operations Performed section
            if "**Operations Performed**:" in content:
                ops_start = content.find("**Operations Performed**:") + len("**Operations Performed**:")
                ops_end = content.find("**", ops_start)
                if ops_end == -1:
                    ops_end = len(content)
                
                ops_text = content[ops_start:ops_end].strip()
                ops_lines = [line.strip() for line in ops_text.split('\n') if line.strip() and line.startswith('-')]
                provenance["operations_performed"] = [line[1:].strip() for line in ops_lines]  # Remove '- ' prefix
            
            # Extract Original Data Found section
            if "**Original Data Found**:" in content:
                data_start = content.find("**Original Data Found**:") + len("**Original Data Found**:")
                data_end = content.find("**", data_start)
                if data_end == -1:
                    data_end = len(content)
                
                provenance["original_data"] = content[data_start:data_end].strip()
            
            # Extract Data Limitations section
            if "**Data Limitations**:" in content:
                limits_start = content.find("**Data Limitations**:") + len("**Data Limitations**:")
                limits_end = content.find("**", limits_start)
                if limits_end == -1:
                    limits_end = len(content)
                
                provenance["data_limitations"] = content[limits_start:limits_end].strip()
        
        except Exception as e:
            logger.debug(f"Error extracting provenance sections: {e}")
        
        return provenance
    
    def _clean_citation_markers(self, text: str) -> str:
        """Remove OpenAI's internal citation markers from response text."""
        import re
        
        # Remove Unicode private use area characters used by OpenAI for citation markers
        # These characters appear as "fileciteturn0file0" when not properly handled
        text = re.sub(r'[\ue200-\ue2ff]', '', text)  # Private use area block
        
        # Also clean any remaining citation marker patterns that got through
        text = re.sub(r'filecite.*?file\d+', '', text)
        text = re.sub(r'turn\d+file\d+', '', text)
        
        return text.strip()
    
    def _standardize_category(self, category: str) -> str:
        """
        Intelligent category standardization using semantic similarity.
        
        Instead of hardcoded mappings, this uses:
        1. Embedding-based similarity to find related categories
        2. Statistical frequency analysis to choose canonical forms
        3. Self-learning from user corrections
        """
        if not category:
            return category
            
        # Try to use GPT-powered intelligent categorization
        try:
            from .gpt_categorizer import GPTCategorizer
            categorizer = GPTCategorizer()
            return categorizer.standardize_single(category)
        except ImportError:
            # Fallback to basic text normalization only
            return self._basic_text_normalization(category)
    
    def _basic_text_normalization(self, category: str) -> str:
        """Basic text normalization as fallback."""
        if not category:
            return category
            
        # Basic cleanup only - no hardcoded mappings
        normalized = category.strip()
        
        # Convert underscores and multiple spaces to single spaces
        import re
        normalized = re.sub(r'[_\s]+', ' ', normalized)
        
        # Title case for consistency
        normalized = normalized.title()
        
        return normalized
    
    def _log_query_audit(self, question: str, parsed_result: Dict[str, Any], session_id: str = None):
        """Log query for audit trail with provenance information."""
        try:
            # Extract audit information from parsed result
            response_summary = parsed_result.get('content', '')[:200] + "..." if len(parsed_result.get('content', '')) > 200 else parsed_result.get('content', '')
            
            # Get file names from annotations
            files_accessed = []
            for ann in parsed_result.get('annotations', []):
                if ann.get('filename'):
                    files_accessed.append(ann['filename'])
                elif ann.get('file_id'):
                    files_accessed.append(f"FileID:{ann['file_id']}")
            
            # Get operations and sources from provenance
            provenance = parsed_result.get('provenance', {})
            operations_performed = provenance.get('operations_performed', [])
            data_sources = provenance.get('data_sources', [])
            
            # Log to data store
            self.data_store.log_query_audit(
                query_text=question,
                response_summary=response_summary,
                files_accessed=files_accessed,
                operations_performed=operations_performed,
                data_sources=data_sources,
                session_id=session_id
            )
        except Exception as e:
            logger.debug(f"Failed to log query audit: {e}")
    
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
        """
        Minimal enhancement for summary queries - let Assistant API file_search do the work.
        
        The Assistant API with file_search is much more intelligent than our manual context.
        We should trust it to find relevant data rather than polluting the prompt.
        """
        # Just add a simple instruction to leverage the Assistant's file_search capability
        enhanced_parts = [
            question,
            "",
            "Please search through all uploaded files to provide a comprehensive analysis based on the actual data."
        ]
        
        # Only add a brief data availability hint if we have files
        summary = self.data_store.get_data_summary()
        if summary and summary.get('files_processed', 0) > 0:
            enhanced_parts.append(f"Note: {summary['files_processed']} data files are available for analysis.")
        
        enhanced_query = '\n'.join(enhanced_parts)
        
        logger.debug("Using minimal enhancement - letting Assistant API file_search handle data discovery")
        return enhanced_query
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear thread references
            self.threads.clear()
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")