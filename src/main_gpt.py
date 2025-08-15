"""
Enhanced CLI interface with GPT tools and multi-agent system.

This module provides the new architecture combining OpenAI's GPT tools
with Claude-based multi-agent orchestration.
"""

import argparse
import sys
import os
import time
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from agents import AgentOrchestrator
from gpt_tools import AssistantManager, FileManager, EmbeddingService
from utils import get_config, setup_logging
from database_adapter import DatabaseAdapter, get_database_connection

# Setup logging and configuration
setup_logging()
logger = logging.getLogger(__name__)


class ImpactOSGPTCLI:
    """Enhanced CLI with GPT tools and multi-agent system."""
    
    def __init__(self):
        """Initialize enhanced CLI."""
        # Load configuration
        self.config = get_config()
        self.openai_key = self.config.get_openai_key()
        self.anthropic_key = self.config.get_anthropic_key()
        
        # Initialize systems
        self.orchestrator = None
        self.assistant_manager = None
        self.file_manager = None
        self.db_adapter = None  # Will be initialized when needed
        
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize GPT tools and agent orchestrator."""
        try:
            # Initialize multi-agent orchestrator
            if self.anthropic_key:
                try:
                    self.orchestrator = AgentOrchestrator(
                        anthropic_key=self.anthropic_key,
                        openai_key=self.openai_key
                    )
                    self.orchestrator.start()
                    logger.info("Multi-agent orchestrator initialized")
                except Exception as e:
                    logger.error(f"Multi-agent system initialization failed: {e}")
                    self.orchestrator = None
            
            # Initialize GPT tools
            if self.openai_key:
                try:
                    self.assistant_manager = AssistantManager(self.openai_key)
                    self.file_manager = FileManager(self.openai_key)
                    logger.info("GPT tools initialized")
                except Exception as e:
                    logger.error(f"GPT tools initialization failed: {e}")
                    self.assistant_manager = None
                    self.file_manager = None
            
            # Validate that at least one system is available
            if not self.orchestrator and not self.assistant_manager:
                logger.error("No processing systems available. Please check API keys.")
                raise RuntimeError("System initialization failed - no available processing engines")
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def _get_db_adapter(self) -> DatabaseAdapter:
        """Get or create database adapter instance."""
        if self.db_adapter is None:
            connection_string = get_database_connection()
            self.db_adapter = DatabaseAdapter(connection_string)
            self.db_adapter.connect()
        return self.db_adapter
    
    def ingest_data(self, file_path: str, use_agents: bool = True, skip_upload: bool = False) -> bool:
        """
        Ingest data using either multi-agent system or direct GPT tools.
        
        Args:
            file_path: Path to file to ingest
            use_agents: Whether to use multi-agent system
            skip_upload: Skip uploading to OpenAI (assumes file already uploaded)
            
        Returns:
            Success status
        """
        try:
            # Validate input
            if not file_path:
                logger.error("File path is required")
                return False
                
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Check file is supported
            if not self.config.is_file_supported(file_path):
                logger.error(f"File format not supported: {file_path}")
                return False
            
            # Check file size
            if not self.config.validate_file_size(file_path):
                max_size = self.config.get('max_file_size_mb', 50)
                logger.error(f"File too large (max {max_size}MB): {file_path}")
                return False
            
            if use_agents and self.orchestrator:
                # Use multi-agent system
                logger.info(f"Ingesting {file_path} via multi-agent system")
                try:
                    result = self.orchestrator.ingest_data([file_path])
                    
                    if result.get('status') == 'success':
                        logger.info("Ingestion completed successfully")
                        self._display_ingestion_results(result)
                        return True
                    else:
                        logger.error(f"Multi-agent ingestion failed: {result.get('error')}")
                        # Fall back to GPT tools if available
                        if self.assistant_manager and self.file_manager:
                            logger.info("Falling back to direct GPT tools")
                        else:
                            return False
                except Exception as e:
                    logger.error(f"Multi-agent system error: {e}")
                    # Fall back to GPT tools if available
                    if self.assistant_manager and self.file_manager:
                        logger.info("Falling back to direct GPT tools due to agent error")
                    else:
                        return False
            
            # Use direct GPT tools (either by choice or fallback)
            if self.assistant_manager and self.file_manager:
                logger.info(f"Ingesting {file_path} via GPT tools")
                
                try:
                    file_id = None
                    
                    if skip_upload:
                        logger.info("Skipping file upload as requested")
                        # Try to find existing file ID in database
                        from pathlib import Path
                        filename = Path(file_path).name
                        existing_files = self.assistant_manager.data_store.get_all_files()
                        for file_info in existing_files:
                            if file_info.get('file_path', '').endswith(filename):
                                file_id = file_info.get('openai_file_id')
                                logger.info(f"Found existing file ID: {file_id}")
                                break
                        
                        if not file_id:
                            logger.error(f"No existing file ID found for {filename}. Remove --skip-upload to upload.")
                            return False
                    else:
                        # Upload file with retry logic
                        max_retries = 3
                        
                        for attempt in range(max_retries):
                            try:
                                file_id = self.file_manager.upload_file(file_path)
                                break
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    raise
                                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                                time.sleep(2)  # Wait before retry
                        
                        if not file_id:
                            logger.error("Failed to upload file after all retries")
                            return False
                    
                    # Process with assistant
                    instructions = """Please analyze this file and extract any salary, compensation, payroll, or impact data you find.

Look for:
- Salary amounts, bonuses, compensation data
- Employee counts and demographic information  
- Gender pay gap or diversity metrics
- Environmental metrics (carbon, energy, waste)
- Social impact data (community, donations, volunteering)
- Governance metrics (training, compliance)

Return your findings in this exact JSON format:
```json
{
  "metrics": [
    {
      "name": "metric name",
      "value": "actual number",
      "unit": "currency/percentage/count/etc", 
      "category": "category name",
      "period": "time period if available"
    }
  ],
  "summary": "Brief description of data found"
}
```

Be sure to include the actual numbers and values you see in the data."""
                    
                    result = self.assistant_manager.process_data([file_id], instructions)
                    
                    if result:
                        logger.info("GPT processing completed")
                        self._display_gpt_results(result)
                        return True
                    else:
                        logger.error("GPT processing returned no results")
                        return False
                        
                except Exception as e:
                    logger.error(f"GPT tools processing failed: {e}")
                    return False
                    
            else:
                logger.error("No processing system available")
                return False
                
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return False
    
    def query_data(self, question: str, use_agents: bool = True, force_chart: bool = False) -> str:
        """
        Query data using either multi-agent system or direct GPT tools.
        
        Args:
            question: Natural language question
            use_agents: Whether to use multi-agent system
            
        Returns:
            Answer string
        """
        start_time = time.time()
        
        try:
                
            if use_agents and self.orchestrator:
                # Use multi-agent system
                logger.info("Processing query via multi-agent system")
                result = self.orchestrator.query_data(question, force_chart=force_chart)
                
                if result.get('status') == 'success':
                    answer = result.get('answer', 'No answer generated')
                    if result.get('visualization'):
                        answer += f"\n\n📊 Visualization recommended: {result['visualization'].get('chart_type')}"
                    
                    # Calculate processing time and log comprehensive event
                    total_ms = int((time.time() - start_time) * 1000)
                    
                    try:
                        db_adapter = self._get_db_adapter()
                        db_adapter.log_ai_query_event(
                            source="cli_gpt",
                            question=question,
                            answer=answer,
                            status="ok",
                            model="multi-agent",
                            total_ms=total_ms,
                            timings={"total_ms": total_ms, "multi_agent_processing_ms": total_ms},
                            chart=result.get('visualization'),
                            logs=f"Multi-agent processing completed successfully in {total_ms}ms",
                            metadata={"interface": "cli_gpt", "system": "impactos-ai", "method": "multi_agent"}
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log AI query event: {e}")
                    
                    return answer
                else:
                    error_msg = f"Query failed: {result.get('error')}"
                    total_ms = int((time.time() - start_time) * 1000)
                    
                    # Log error event
                    try:
                        db_adapter = self._get_db_adapter()
                        db_adapter.log_ai_query_event(
                            source="cli_gpt",
                            question=question,
                            status="error",
                            total_ms=total_ms,
                            logs=f"Multi-agent processing failed after {total_ms}ms: {result.get('error', 'Unknown error')}",
                            error=result.get('error'),
                            metadata={"interface": "cli_gpt", "system": "impactos-ai", "method": "multi_agent"}
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log error event: {e}")
                    
                    return error_msg
                    
            elif self.assistant_manager:
                # Direct GPT assistant
                logger.info("Processing query via GPT assistant")
                result = self.assistant_manager.query_data(question, force_chart=force_chart)
                
                answer = result.get('content', 'No answer generated')
                
                # Add enhanced citations and provenance if available
                provenance_info = self._format_provenance_info(result)
                if provenance_info:
                    answer += f"\n\n{provenance_info}"
                
                # Add framework badges to response
                answer = self._add_framework_badges(answer)
                
                # Add chart visualization info if available (for CLI display)
                if result.get('visualization') and not force_chart:
                    # Only show text description if not forcing structured chart output
                    chart_info = result['visualization']
                    if chart_info.get('description'):
                        answer += f"\n\n📊 Visualization: {chart_info.get('description', 'Chart recommended')}"
                
                # Calculate processing time and log comprehensive event
                total_ms = int((time.time() - start_time) * 1000)
                
                try:
                    db_adapter = self._get_db_adapter()
                    db_adapter.log_ai_query_event(
                        source="cli_gpt",
                        question=question,
                        answer=answer,
                        status="ok",
                        model="gpt-assistant",
                        total_ms=total_ms,
                        timings={"total_ms": total_ms, "gpt_assistant_processing_ms": total_ms},
                        chart=result.get('visualization'),
                        logs=f"GPT assistant processing completed successfully in {total_ms}ms",
                        metadata={"interface": "cli_gpt", "system": "impactos-ai", "method": "gpt_assistant"}
                    )
                except Exception as e:
                    logger.debug(f"Failed to log AI query event: {e}")
                
                return answer
                
            else:
                error_msg = "No query system available. Please check API keys."
                total_ms = int((time.time() - start_time) * 1000)
                
                # Log error event
                try:
                    db_adapter = self._get_db_adapter()
                    db_adapter.log_ai_query_event(
                        source="cli_gpt",
                        question=question,
                        status="error",
                        total_ms=total_ms,
                        logs=f"No query system available for processing after {total_ms}ms",
                        error="No query system available",
                        metadata={"interface": "cli_gpt", "system": "impactos-ai"}
                    )
                except Exception as e:
                    logger.debug(f"Failed to log error event: {e}")
                
                return error_msg
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            error_msg = f"Error processing query: {e}"
            total_ms = int((time.time() - start_time) * 1000)
            
            # Log error event
            try:
                db_adapter = self._get_db_adapter()
                db_adapter.log_ai_query_event(
                    source="cli_gpt",
                    question=question,
                    status="error",
                    total_ms=total_ms,
                    logs=f"Exception occurred during processing after {total_ms}ms: {str(e)}",
                    error=str(e),
                    metadata={"interface": "cli_gpt", "system": "impactos-ai"}
                )
            except Exception as log_e:
                logger.debug(f"Failed to log error event: {log_e}")
            
            return error_msg
    
    def show_status(self):
        """Display system status."""
        print("\n=== ImpactOS GPT System Status ===")
        
        # API Keys
        print("\nAPI Keys:")
        print(f"  OpenAI: {'✓ Configured' if self.openai_key else '✗ Missing'}")
        print(f"  Anthropic: {'✓ Configured' if self.anthropic_key else '✗ Missing'}")
        
        # Systems
        print("\nSystems:")
        print(f"  GPT Tools: {'✓ Active' if self.assistant_manager else '✗ Inactive'}")
        print(f"  Multi-Agent: {'✓ Active' if self.orchestrator else '✗ Inactive'}")
        
        # Agent Status
        if self.orchestrator:
            status = self.orchestrator.get_system_status()
            
            print("\nAgents:")
            for agent_name, agent_info in status.get('agents', {}).items():
                tasks_done = agent_info.get('tasks_completed', 0)
                print(f"  {agent_name}: {tasks_done} tasks completed")
            
            print("\nTask Queue:")
            task_stats = status.get('task_statistics', {})
            print(f"  Pending: {task_stats.get('pending', 0)}")
            print(f"  In Progress: {task_stats.get('in_progress', 0)}")
            print(f"  Completed: {task_stats.get('completed', 0)}")
        
        # GPT Resources
        if self.file_manager:
            files = self.file_manager.list_files()
            print(f"\nGPT Files: {len(files)} uploaded")
    
    def _display_ingestion_results(self, result: Dict):
        """Display ingestion results from multi-agent system."""
        print("\n=== Ingestion Results ===")
        
        if 'results' in result:
            for r in result['results']:
                if r.get('metrics_count'):
                    print(f"✓ Extracted {r['metrics_count']} metrics")
                if r.get('validation'):
                    score = r['validation'].get('score', 0)
                    print(f"✓ Validation score: {score:.1f}%")
        
        print(f"\nTasks completed: {result.get('tasks_completed', 0)}")
    
    def _display_gpt_results(self, result: Dict):
        """Display results from GPT processing."""
        print("\n=== GPT Processing Results ===")
        
        if 'structured_data' in result:
            data = result['structured_data']
            if isinstance(data, dict):
                metrics = data.get('metrics', [])
                print(f"✓ Extracted {len(metrics)} metrics")
                
                # Show all metrics with better formatting
                if metrics:
                    print("\nExtracted metrics:")
                    for i, metric in enumerate(metrics, 1):
                        name = metric.get('name', 'Unknown')
                        value = metric.get('value', 'N/A')
                        unit = metric.get('unit', '')
                        category = metric.get('category', 'General')
                        period = metric.get('period', '')
                        
                        print(f"  {i}. {name}: {value} {unit}")
                        print(f"     Category: {category}")
                        if period:
                            print(f"     Period: {period}")
                        print()
                
                # Show summary if available
                summary = data.get('summary')
                if summary:
                    print(f"Summary: {summary}")
        else:
            print("✓ Extracted 0 metrics (no structured data returned)")
        
        if 'annotations' in result:
            print(f"✓ {len(result['annotations'])} citations found")
        else:
            print("✓ 0 citations found")
    
    def batch_ingest(self, directory: str):
        """Ingest all data files from a directory."""
        try:
            data_path = Path(directory)
            if not data_path.exists():
                logger.error(f"Directory not found: {directory}")
                return
            
            # Find all data files
            patterns = ['*.csv', '*.xlsx', '*.pdf', '*.json']
            files = []
            for pattern in patterns:
                files.extend(data_path.glob(pattern))
            
            if not files:
                print(f"No data files found in {directory}")
                return
            
            print(f"\nFound {len(files)} files to ingest")
            
            if self.orchestrator:
                # Use orchestrator for batch processing
                file_paths = [str(f) for f in files]
                result = self.orchestrator.ingest_data(file_paths)
                self._display_ingestion_results(result)
            else:
                # Process individually with GPT
                for file_path in files:
                    print(f"\nProcessing {file_path.name}...")
                    self.ingest_data(str(file_path), use_agents=False)
                    
        except Exception as e:
            logger.error(f"Batch ingestion failed: {e}")
    
    def interactive_mode(self):
        """Run interactive query mode."""
        print("\n=== Interactive Query Mode ===")
        print("Type 'exit' to quit, 'status' for system status")
        print("-" * 40)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() == 'exit':
                    break
                elif question.lower() == 'status':
                    self.show_status()
                elif question:
                    answer = self.query_data(question)
                    print(f"\nAnswer: {answer}")
                    
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode.")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _format_provenance_info(self, result: Dict[str, Any]) -> str:
        """Format provenance and citation information for display."""
        provenance_parts = []
        
        # Enhanced Citations
        if result.get('annotations'):
            provenance_parts.append("=== DATA CITATIONS ===")
            for i, ann in enumerate(result['annotations'], 1):
                citation_info = f"[{i}] File ID: {ann.get('file_id', 'Unknown')}"
                
                if ann.get('filename'):
                    citation_info += f" ({ann['filename']})"
                if ann.get('file_type'):
                    citation_info += f" - Type: {ann['file_type'].upper()}"
                if ann.get('upload_time'):
                    citation_info += f" - Uploaded: {ann['upload_time'][:10]}"
                
                provenance_parts.append(citation_info)
                
                if ann.get('quote'):
                    quote_preview = ann['quote'][:200] + "..." if len(ann['quote']) > 200 else ann['quote']
                    provenance_parts.append(f"    Quote: \"{quote_preview}\"")
                
                provenance_parts.append("")  # Empty line
        
        # Provenance Information
        if result.get('provenance'):
            prov = result['provenance']
            
            if prov.get('data_sources'):
                provenance_parts.append("=== DATA SOURCES ===")
                for source in prov['data_sources']:
                    provenance_parts.append(f"• {source}")
                provenance_parts.append("")
            
            if prov.get('operations_performed'):
                provenance_parts.append("=== OPERATIONS PERFORMED ===")
                for operation in prov['operations_performed']:
                    provenance_parts.append(f"• {operation}")
                provenance_parts.append("")
            
            if prov.get('original_data'):
                provenance_parts.append("=== ORIGINAL DATA FOUND ===")
                provenance_parts.append(prov['original_data'])
                provenance_parts.append("")
            
            if prov.get('data_limitations'):
                provenance_parts.append("=== DATA LIMITATIONS ===")
                provenance_parts.append(prov['data_limitations'])
        
        return "\n".join(provenance_parts) if provenance_parts else ""
    
    def _add_framework_badges(self, answer: str) -> str:
        """Add compact framework badges to GPT Assistant responses."""
        try:
            # Import here to avoid circular imports
            from frameworks import FrameworkMapper
            
            mapper = FrameworkMapper()
            
            # Define metric patterns and their likely framework mappings
            badge_patterns = {
                # Gender/diversity metrics
                'gender pay gap': '[SDG5] [5.0] [WORKERS]',
                'pay gap': '[SDG5] [5.0] [WORKERS]',
                'diversity': '[SDG5] [10] [WORKERS]',
                'gender equality': '[SDG5] [5.0] [WORKERS]',
                
                # Employment metrics
                'employment': '[SDG8] [3.0] [NT1] [WORKERS]',
                'jobs': '[SDG8] [3.0] [NT1] [WORKERS]',
                'salary': '[SDG8] [3.0] [WORKERS]',
                'training': '[SDG4] [2.0] [NT2] [WORKERS]',
                'skills': '[SDG4] [3.3] [NT2] [WORKERS]',
                
                # Community engagement
                'volunteering': '[SDG11] [8.1] [NT90] [COMMUNITY]',
                'volunteer': '[SDG11] [8.1] [NT90] [COMMUNITY]',
                'donation': '[SDG11] [8.2] [COMMUNITY]',
                'charitable': '[SDG11] [8.2] [COMMUNITY]',
                'community': '[SDG11] [8.0] [COMMUNITY]',
                
                # Environmental metrics  
                'carbon': '[SDG13] [4.1] [NT4] [ENVIRONMENT]',
                'emissions': '[SDG13] [4.1] [NT4] [ENVIRONMENT]',
                'energy': '[SDG13] [4.1] [NT4] [ENVIRONMENT]',
                'waste': '[SDG12] [4.2] [NT4] [ENVIRONMENT]',
                'environmental': '[SDG13] [4.0] [NT4] [ENVIRONMENT]'
            }
            
            # Find relevant badges for this answer
            answer_lower = answer.lower()
            relevant_badges = set()
            
            for pattern, badge in badge_patterns.items():
                if pattern in answer_lower:
                    relevant_badges.add(badge)
            
            # Add badges to the beginning of the answer if any found
            if relevant_badges:
                badges_text = " ".join(sorted(relevant_badges))
                # Add badges after "Answer:" if present, or at the beginning
                if answer.startswith("• Answer:"):
                    answer = answer.replace("• Answer:", f"• Answer {badges_text}:", 1)
                elif answer.startswith("Answer:"):
                    answer = answer.replace("Answer:", f"Answer {badges_text}:", 1)
                else:
                    answer = f"{badges_text}\n\n{answer}"
            
            return answer
            
        except Exception as e:
            logger.debug(f"Failed to add framework badges: {e}")
            return answer  # Return original answer if badge addition fails
    
    def query_structured(self, question: str, force_chart: bool = False) -> Dict[str, Any]:
        """Query data and return structured response compatible with web API.
        Returns: {"answer": str, "show_chart": bool, "chart": dict or None}
        """
        try:
            if self.assistant_manager:
                # Get GPT Assistant response with force_chart
                result = self.assistant_manager.query_data(question, force_chart=force_chart)
                answer = result.get('content', 'No answer generated')
                
                # Add framework badges to response
                answer = self._add_framework_badges(answer)
                
                # Check for structured chart data
                visualization = result.get('visualization')
                show_chart = False
                chart_data = None
                
                if visualization:
                    # Check if we have structured chart data (new format)
                    if visualization.get('type') and visualization.get('data'):
                        show_chart = True
                        chart_data = visualization
                    # Or if we have old format with description
                    elif visualization.get('recommended'):
                        show_chart = bool(force_chart)  # Only show if explicitly requested
                
                return {
                    "answer": answer,
                    "show_chart": show_chart,
                    "chart": chart_data
                }
            else:
                return {
                    "answer": "GPT Assistant system not available",
                    "show_chart": False,
                    "chart": None
                }
                
        except Exception as e:
            logger.error(f"Structured query error: {e}")
            return {
                "answer": f"Error processing query: {e}",
                "show_chart": False,
                "chart": None
            }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.orchestrator:
                self.orchestrator.cleanup()
            
            if self.assistant_manager:
                self.assistant_manager.cleanup()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ImpactOS AI with GPT Tools and Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ingest data/sample.xlsx
  %(prog)s query "What are our carbon emissions?"
  %(prog)s batch-ingest data/
  %(prog)s interactive
  %(prog)s status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data file')
    ingest_parser.add_argument('file_path', help='Path to file')
    ingest_parser.add_argument('--no-agents', action='store_true',
                              help='Use direct GPT tools instead of agents')
    ingest_parser.add_argument('--skip-upload', action='store_true',
                              help='Skip uploading file to OpenAI (assumes already uploaded)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query data')
    query_parser.add_argument('question', help='Natural language question')
    query_parser.add_argument('--no-agents', action='store_true',
                             help='Use direct GPT tools instead of agents')
    query_parser.add_argument('--force-chart', action='store_true',
                             help='Force chart generation if possible')
    query_parser.add_argument('--json', action='store_true',
                             help='Return structured JSON response for API compatibility')
    
    # Batch ingest
    batch_parser = subparsers.add_parser('batch-ingest', help='Ingest directory')
    batch_parser.add_argument('directory', help='Directory path')
    
    # Interactive mode
    subparsers.add_parser('interactive', help='Interactive query mode')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = ImpactOSGPTCLI()
    
    try:
        if args.command == 'ingest':
            success = cli.ingest_data(
                args.file_path,
                use_agents=not args.no_agents,
                skip_upload=args.skip_upload
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'query':
            if args.json:
                # Return structured JSON response
                structured = cli.query_structured(
                    args.question,
                    force_chart=args.force_chart
                )
                import json
                print(json.dumps(structured, indent=2))
            else:
                # Return normal text response
                answer = cli.query_data(
                    args.question,
                    use_agents=not args.no_agents,
                    force_chart=args.force_chart
                )
                print(f"\n{answer}")
            
        elif args.command == 'batch-ingest':
            cli.batch_ingest(args.directory)
            
        elif args.command == 'interactive':
            cli.interactive_mode()
            
        elif args.command == 'status':
            cli.show_status()
            
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        cli.cleanup()


if __name__ == "__main__":
    main()