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
        self.db_path = "db/impactos.db"
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
            
            # TODO: Query database for ingested sources
            print(f"\n=== Ingested Sources ===")
            print("  (Database query not yet implemented)")
            
        except Exception as e:
            logger.error(f"Error listing data: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ImpactOS AI Layer MVP - Social Value Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ingest data/TakingCare_Payroll_Synthetic_Data.xlsx
  %(prog)s query "How much was donated to charity last year?"
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
    
    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Query data with natural language'
    )
    query_parser.add_argument(
        'question',
        help='Natural language question about the data'
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
            sys.exit(0 if success else 1)
            
        elif args.command == 'query':
            answer = cli.query_data(args.question)
            print(f"\nAnswer: {answer}")
            
        elif args.command == 'schema':
            cli.show_schema_info()
            
        elif args.command == 'list':
            cli.list_available_data()
            
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