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
            
            # Import and use query system
            from query import query_data
            answer = query_data(question, self.db_path)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
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
            success = cli.ingest_data(args.file_path, args.file_type)
            
            # Run verification if requested
            if args.verify and success:
                logger.info("Running verification after ingestion...")
                cli.verify_data('all')
            
            sys.exit(0 if success else 1)
            
        elif args.command == 'query':
            if getattr(args, 'structured', False):
                # Structured response with optional chart payload
                from query import QuerySystem
                qs = QuerySystem(cli.db_path)
                structured = qs.query_structured(args.question, force_chart=bool(getattr(args, 'force_chart', False)))
                import json as _json
                print(_json.dumps(structured, indent=2))
            else:
                answer = cli.query_data(args.question)
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
            
        elif args.command == 'frameworks':
            cli.show_framework_report(args)
            
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