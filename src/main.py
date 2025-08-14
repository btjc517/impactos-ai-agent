"""
Main CLI interface for ImpactOS AI Layer MVP Phase One.

This module provides command-line interface for ingesting social value data
and querying the system with natural language Q&A.
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from typing import Optional
import time
import sqlite3

# Disable tokenizer parallelism warnings/deadlocks by default
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from schema import initialize_database, DatabaseSchema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImpactOSCLI:
    """Main CLI interface for ImpactOS AI system."""
    
    def __init__(self):
        """Initialize CLI with database connection."""
        self.db_path = os.getenv('IMPACTOS_DB_PATH', 'db/impactos.db')
        self.db_schema = DatabaseSchema(self.db_path)
        self.ensure_database_initialized()
    
    def ensure_database_initialized(self):
        """Ensure database is properly initialized."""
        if not os.path.exists(self.db_path):
            logger.info("Database not found. Initializing...")
            self.db_schema.initialize_database()
        else:
            logger.info(f"Database found at {self.db_path}")
    
    def ingest_data(self, file_path: str, file_type: Optional[str] = None) -> bool:
        """
        Ingest data from file into the system.
        
        Args:
            file_path: Path to file to ingest
            file_type: Optional file type override
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Auto-detect file type if not provided
            if file_type is None:
                file_extension = Path(file_path).suffix.lower()
                file_type = file_extension[1:] if file_extension else "unknown"
            
            logger.info(f"Starting ingestion of {file_path} (type: {file_type})")
            
            # Import and use ingestion pipeline
            from ingest import ingest_file
            success = ingest_file(file_path, self.db_path)
            
            if success:
                logger.info(f"Successfully ingested {file_path}")
            else:
                logger.error(f"Failed to ingest {file_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            return False
    
    def query_data(self, question: str) -> str:
        """
        Query the system with natural language question.
        
        Args:
            question: Natural language question
            
        Returns:
            Answer with citations
        """
        try:
            logger.info(f"Processing query: {question}")
            from telemetry import telemetry, capture_logs
            from query import QuerySystem
            qs = QuerySystem(self.db_path)
            started = time.monotonic()
            with capture_logs() as log_handler:
                structured, timings, model_used = qs.query_structured_instrumented(question)
            answer = structured.get('answer', '')
            # Telemetry (source=cli)
            try:
                if telemetry.is_enabled():
                    event = telemetry.build_event(
                        question=question,
                        answer=answer,
                        status='ok',
                        source='cli',
                        user_id=None,
                        session_id=None,
                        model=model_used,
                        total_ms=timings.get('total_ms', int((time.monotonic() - started) * 1000)),
                        timings=timings,
                        chart=structured.get('chart'),
                        logs_text=log_handler.get_value(),
                        metadata={'invoked_via': 'ImpactOSCLI.query_data'},
                    )
                    telemetry.send_query_event(event)
            except Exception:
                pass
            return answer
        except Exception as e:
            logger.error(f"Error during query: {e}")
            # Attempt to send failure telemetry
            try:
                from telemetry import telemetry
                if telemetry.is_enabled():
                    event = telemetry.build_event(
                        question=question,
                        answer=None,
                        status='error',
                        source='cli',
                        error=str(e),
                    )
                    telemetry.send_query_event(event)
            except Exception:
                pass
            return f"Error processing query: {e}"
    
    def show_schema_info(self):
        """Display database schema information."""
        try:
            schema_info = self.db_schema.get_schema_info()
            
            print("\n=== Database Schema Information ===")
            for table_name, columns in schema_info.items():
                print(f"\nTable: {table_name}")
                for column in columns:
                    print(f"  - {column}")
            
            print(f"\nDatabase location: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
    
    def list_available_data(self):
        """List available data files and ingested sources."""
        try:
            # Check data directory
            data_dir = "data"
            if os.path.exists(data_dir):
                print(f"\n=== Available Data Files ===")
                data_files = list(Path(data_dir).glob("*"))
                for file_path in sorted(data_files):
                    if file_path.is_file():
                        print(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")
            
            # Query database for ingested sources
            print(f"\n=== Ingested Sources ===")
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT s.filename, s.processed_timestamp, s.processing_status,
                               COUNT(im.id) as metric_count,
                               AVG(im.verification_accuracy) as avg_accuracy
                        FROM sources s
                        LEFT JOIN impact_metrics im ON s.id = im.source_id
                        GROUP BY s.id
                        ORDER BY s.processed_timestamp DESC
                    """)
                    
                    sources = cursor.fetchall()
                    if sources:
                        for source in sources:
                            accuracy = source['avg_accuracy'] or 0.0
                            print(f"  - {source['filename']} ({source['metric_count']} metrics, {accuracy:.1%} accuracy)")
                    else:
                        print("  (No sources ingested yet)")
            except Exception as e:
                logger.debug(f"Error querying sources: {e}")
                print("  (Database query not yet implemented)")
            
        except Exception as e:
            logger.error(f"Error listing data: {e}")
    
    def list_metrics_for_file(self, filename: str, as_json: bool = False) -> None:
        """List all metrics stored for a given source filename.

        Args:
            filename: Exact filename of the ingested source (e.g., 'TakingCare_Payroll_Synthetic_Data.xlsx')
            as_json: When True, print results as a JSON array; otherwise, pretty-print rows
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        im.id,
                        im.metric_name,
                        im.metric_value,
                        im.metric_unit,
                        im.metric_category,
                        im.extraction_confidence,
                        im.verification_status,
                        im.source_sheet_name,
                        im.source_column_name,
                        im.source_cell_reference,
                        s.processed_timestamp
                    FROM impact_metrics im
                    JOIN sources s ON im.source_id = s.id
                    WHERE s.filename = ?
                    ORDER BY im.created_at DESC
                    """,
                    (filename,),
                )
                rows = [dict(r) for r in cursor.fetchall()]

            if as_json:
                import json as _json
                print(_json.dumps(rows, indent=2, default=str))
                return

            if not rows:
                print(f"No metrics found for file '{filename}'.")
                return

            print(f"\n=== Metrics for {filename} ===")
            for r in rows:
                print(
                    f"- #{r['id']}: {r['metric_name']} = {r['metric_value']} {r.get('metric_unit') or ''}"
                    f" | category: {r.get('metric_category') or 'n/a'}"
                    f" | confidence: {r.get('extraction_confidence') if r.get('extraction_confidence') is not None else 'n/a'}"
                    f" | verification: {r.get('verification_status') or 'n/a'}"
                    f" | sheet: {r.get('source_sheet_name') or 'n/a'}"
                    f" | cell: {r.get('source_cell_reference') or r.get('source_column_name') or 'n/a'}"
                )
        except Exception as e:
            logger.error(f"Error listing metrics for file '{filename}': {e}")

    def verify_data(self, target: str):
        """Verify data accuracy."""
        try:
            from verify import verify_all_data, verify_metric
            
            if target.lower() == 'all':
                print("\n🔍 Verifying all metrics against source files...")
                summary = verify_all_data(self.db_path)
                
                if 'error' in summary:
                    print(f"❌ Verification failed: {summary['error']}")
                    return
                
                print(f"\n📊 Verification Results:")
                print(f"  Total metrics: {summary['total']}")
                print(f"  ✅ Verified: {summary['verified']}")
                print(f"  ❌ Failed: {summary['failed']}")
                print(f"  🎯 Overall accuracy: {summary['accuracy']:.1%}")
                
                if 'verification_rate' in summary:
                    print(f"  📈 Verification rate: {summary['verification_rate']:.1%}")
                
                if summary['accuracy'] < 0.8 and summary['total'] > 0:
                    print("\n⚠️  Low accuracy detected! Consider:")
                    print("    - Improving GPT-5 extraction prompts")
                    print("    - Adding more precise citation requirements")
                    print("    - Checking data quality in source files")
                
            else:
                try:
                    metric_id = int(target)
                    print(f"\n🔍 Verifying metric {metric_id}...")
                    result = verify_metric(metric_id, self.db_path)
                    
                    if 'error' in result:
                        print(f"❌ Verification failed: {result['error']}")
                        return
                    
                    status = "✅ Verified" if result['verified'] else "❌ Failed"
                    print(f"  {status} (accuracy: {result['accuracy']:.1%})")
                    print(f"  Notes: {result['notes']}")
                    
                except ValueError:
                    print(f"❌ Invalid metric ID: {target}")
                    
        except ImportError:
            print("❌ Verification module not available")
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            print(f"❌ Verification error: {e}")
    
    def get_verification_summary(self) -> str:
        """Get verification accuracy summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as verified,
                        AVG(verification_accuracy) as avg_accuracy
                    FROM impact_metrics
                    WHERE verification_status IS NOT NULL
                """)
                
                result = cursor.fetchone()
                if result and result['total'] > 0:
                    accuracy = result['avg_accuracy'] or 0.0
                    return f"{result['verified']}/{result['total']} metrics verified ({accuracy:.1%} accuracy)"
                else:
                    return "No verification data available"
                    
        except Exception as e:
            return f"Verification error: {e}"

    def show_framework_report(self, args):
        """Show framework mapping report."""
        try:
            print(f"\n🎯 Framework Mapping Report")
            print("=" * 50)
            
            if args.apply:
                print("Applying framework mappings...")
                from frameworks import apply_framework_mappings
                mappings = apply_framework_mappings(self.db_path)
                print(f"✅ Applied mappings: {mappings}")
                print()
            
            from frameworks import get_framework_report
            report = get_framework_report(self.db_path)
            print(report)
            
        except Exception as e:
            print(f"❌ Error generating framework report: {e}")
            return False


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ImpactOS AI Layer MVP - Social Value Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ingest data/TakingCare_Payroll_Synthetic_Data.xlsx
  %(prog)s query "How much was donated to charity last year?"
  %(prog)s verify all
  %(prog)s verify 5
  %(prog)s schema
  %(prog)s list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Ingest data from files'
    )
    ingest_parser.add_argument(
        'file_path',
        help='Path to file to ingest (CSV, XLSX, PDF)'
    )
    ingest_parser.add_argument(
        '--type',
        dest='file_type',
        help='Override file type detection'
    )
    ingest_parser.add_argument(
        '--verify',
        action='store_true',
        help='Run verification after ingestion'
    )
    ingest_parser.add_argument(
        '--bronze-only',
        action='store_true',
        help='Only create Bronze tables and sheet registry (no metric extraction)'
    )
    ingest_parser.add_argument(
        '--auto-transform',
        action='store_true',
        default=True,
        help='Enqueue and run Silver transforms for new sheet versions (default: enabled)'
    )
    ingest_parser.add_argument(
        '--no-auto-transform',
        action='store_false',
        dest='auto_transform',
        help='Disable auto-transform and use legacy query-based extraction'
    )
    ingest_parser.add_argument(
        '--mode',
        choices=['sync','async'],
        default=None,
        help='Transform execution mode when auto-transform is enabled'
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Query data with natural language'
    )
    query_parser.add_argument(
        'question',
        help='Natural language question about the data'
    )
    query_parser.add_argument(
        '--show-accuracy',
        action='store_true',
        help='Show verification accuracy in results'
    )
    query_parser.add_argument(
        '--structured',
        action='store_true',
        help='Return structured JSON including optional chart payload'
    )
    query_parser.add_argument(
        '--force-chart',
        action='store_true',
        help='Force chart rendering when possible in structured output'
    )
    query_parser.add_argument(
        '--debug-time',
        action='store_true',
        help='Print resolved time window for debugging'
    )
    
    # Verify command
    verify_parser = subparsers.add_parser(
        'verify',
        help='Verify extracted metrics against source data'
    )
    verify_parser.add_argument(
        'target',
        help='Verification target: "all" for all metrics, or specific metric ID'
    )
    
    # Schema command
    subparsers.add_parser(
        'schema',
        help='Show database schema information'
    )
    
    # List command
    subparsers.add_parser(
        'list',
        help='List available data files and sources'
    )

    # Medallion seed
    subparsers.add_parser(
        'seed-medallion',
        help='Seed Bronze → Silver → Gold example data and views'
    )

    # Metrics command
    metrics_parser = subparsers.add_parser(
        'metrics',
        help='List all stored metrics for a given source filename'
    )
    metrics_parser.add_argument(
        '--file', '-f',
        required=True,
        help='Exact filename of ingested source (e.g., TakingCare_Payroll_Synthetic_Data.xlsx)'
    )
    metrics_parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    # Metrics validate (hard-fail)
    metrics_validate = subparsers.add_parser(
        'metrics-validate',
        help='Validate metric YAMLs and report errors (hard-fail)'
    )
    metrics_validate.add_argument('--metrics-dir', default=os.getenv('METRICS_DIR') or 'metrics')
    metrics_validate.add_argument('--strict', action='store_true', help='Fail on any warning (for CI)')

    # Metrics convert (CSV -> YAML)
    metrics_convert = subparsers.add_parser(
        'metrics-convert',
        help='Convert catalog CSV into per-metric YAML files'
    )
    metrics_convert.add_argument('csv_path')
    metrics_convert.add_argument('--metrics-dir', default=os.getenv('METRICS_DIR') or 'metrics')
    metrics_convert.add_argument('--db-path', default=os.getenv('IMPACTOS_DB_PATH') or 'db/impactos.db')

    # Add frameworks command
    frameworks_parser = subparsers.add_parser('frameworks', 
                                              help='Generate framework mapping report')
    frameworks_parser.add_argument('--apply', action='store_true',
                                  help='Apply framework mappings to all metrics')
        
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = ImpactOSCLI()
    
    try:
        if args.command == 'ingest':
            if getattr(args, 'bronze_only', False):
                from bronze_ingest import ingest_bronze
                if getattr(args, 'auto_transform', False):
                    if args.mode:
                        os.environ['TRANSFORM_MODE'] = args.mode
                    os.environ['AUTO_TRANSFORM'] = 'true'
                res = ingest_bronze(args.file_path, cli.db_path)
                import json as _json
                print(_json.dumps(res, indent=2))
                success = True
            else:
                # Use medallion architecture by default
                from bronze_ingest import ingest_bronze
                
                # Set auto-transform as default behavior
                auto_transform = getattr(args, 'auto_transform', True)  # Default to True
                if auto_transform:
                    if args.mode:
                        os.environ['TRANSFORM_MODE'] = args.mode
                    else:
                        os.environ['TRANSFORM_MODE'] = 'sync'  # Default to sync mode
                    os.environ['AUTO_TRANSFORM'] = 'true'
                else:
                    os.environ['AUTO_TRANSFORM'] = 'false'
                
                try:
                    logger.info("Using medallion architecture (Bronze → Silver) for ingestion")
                    res = ingest_bronze(args.file_path, cli.db_path)
                    
                    # Check if medallion ingestion was successful
                    medallion_success = (
                        res.get('created_tables', []) and 
                        res.get('registry_rows', [])
                    )
                    
                    if medallion_success:
                        logger.info(f"Medallion ingestion successful: {len(res.get('created_tables', []))} bronze tables created")
                        success = True
                    else:
                        logger.warning("Medallion ingestion produced no results, falling back to query-based extraction")
                        success = cli.ingest_data(args.file_path, args.file_type)
                        
                except Exception as e:
                    logger.warning(f"Medallion ingestion failed ({e}), falling back to query-based extraction")
                    success = cli.ingest_data(args.file_path, args.file_type)
            
            # Run verification if requested
            if args.verify and success:
                logger.info("Running verification after ingestion...")
                cli.verify_data('all')
            
            sys.exit(0 if success else 1)
            
        elif args.command == 'query':
            if getattr(args, 'structured', False):
                # Structured response with optional chart payload, include telemetry
                from telemetry import telemetry, capture_logs
                from query import QuerySystem
                qs = QuerySystem(cli.db_path)
                started = time.monotonic()
                with capture_logs() as log_handler:
                    structured, timings, model_used = qs.query_structured_instrumented(
                        args.question,
                        force_chart=bool(getattr(args, 'force_chart', False)),
                        debug_time=bool(getattr(args, 'debug_time', False))
                    )
                import json as _json
                print(_json.dumps(structured, indent=2))
                try:
                    if telemetry.is_enabled():
                        event = telemetry.build_event(
                            question=args.question,
                            answer=structured.get('answer'),
                            status='ok',
                            source='cli',
                            model=model_used,
                            total_ms=timings.get('total_ms', int((time.monotonic() - started) * 1000)),
                            timings=timings,
                            chart=structured.get('chart'),
                            logs_text=log_handler.get_value(),
                            metadata={'structured': True, 'force_chart': bool(getattr(args, 'force_chart', False))},
                        )
                        telemetry.send_query_event(event)
                except Exception:
                    pass
            else:
                # Call structured path to surface time window consistently in CLI
                from query import QuerySystem
                qs = QuerySystem(cli.db_path)
                structured, timings, model_used = qs.query_structured_instrumented(
                    args.question,
                    force_chart=bool(getattr(args, 'force_chart', False)),
                    debug_time=bool(getattr(args, 'debug_time', False))
                )
                answer = structured.get('answer', '')
                if getattr(args, 'debug_time', False):
                    tw = structured.get('time_window')
                    print(f"Time window: {tw}")
                print(f"\nAnswer: {answer}")
                
                # Show verification summary if requested
                if args.show_accuracy:
                    accuracy_summary = cli.get_verification_summary()
                    print(f"\nData Accuracy: {accuracy_summary}")
            
        elif args.command == 'verify':
            if args.target:
                cli.verify_data(args.target)
            else:
                cli.verify_data('all')
            
        elif args.command == 'schema':
            cli.show_schema_info()
            
        elif args.command == 'list':
            cli.list_available_data()
        
        elif args.command == 'metrics':
            cli.list_metrics_for_file(args.file, as_json=bool(getattr(args, 'json', False)))
            
        elif args.command == 'frameworks':
            cli.show_framework_report(args)
        
        elif args.command == 'metrics-validate':
            from metric_loader import MetricCatalog
            # Loader logs and skips invalid files; for hard-fail CLI we still exit 0 with count
            cat = MetricCatalog(metrics_dir=getattr(args, 'metrics_dir', 'metrics'))
            issues = cat.validate(strict=bool(getattr(args, 'strict', False)))
            print(f"Loaded {len(cat._metrics)} metrics from {getattr(args, 'metrics_dir', 'metrics')} | issues: {issues}")
            if getattr(args, 'strict', False) and issues > 0:
                sys.exit(1)
        
        elif args.command == 'metrics-convert':
            from metric_csv_converter import MetricCSVConverter
            conv = MetricCSVConverter(metrics_dir=getattr(args, 'metrics_dir', 'metrics'), db_path=getattr(args, 'db_path', 'db/impactos.db'))
            res = conv.convert(getattr(args, 'csv_path'))
            import json as _json
            print(_json.dumps(res, indent=2))
        
        elif args.command == 'seed-medallion':
            from seed_medallion import main as seed_main
            seed_main()
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 