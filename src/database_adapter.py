"""
Database adapter for ImpactOS AI system.

This module provides a unified interface for both SQLite (local development)
and PostgreSQL (production with Supabase) databases.
"""

import os
import sqlite3
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

try:
    # Try psycopg3 first (modern, Python 3.13 compatible)
    import psycopg
    import psycopg.rows
    POSTGRES_AVAILABLE = True
    PSYCOPG_VERSION = 3
except ImportError:
    try:
        # Fall back to psycopg2 for older environments
        import psycopg2
        import psycopg2.extras
        POSTGRES_AVAILABLE = True
        PSYCOPG_VERSION = 2
    except ImportError:
        POSTGRES_AVAILABLE = False
        PSYCOPG_VERSION = None

logger = logging.getLogger(__name__)

class DatabaseAdapter:
    """
    Unified database adapter supporting both SQLite and PostgreSQL.
    
    Automatically detects database type from connection string:
    - SQLite: file path (e.g., "db/impactos.db")
    - PostgreSQL: URL (e.g., "postgresql://user:pass@host:port/db")
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database adapter.
        
        Args:
            connection_string: Database connection string or file path
        """
        self.connection_string = connection_string
        self.db_type = self._detect_database_type(connection_string)
        self.connection = None
        
        logger.info(f"Database adapter initialized for {self.db_type}")
    
    def _detect_database_type(self, connection_string: str) -> str:
        """Detect database type from connection string."""
        if connection_string.startswith(('postgresql://', 'postgres://')):
            return 'postgresql'
        else:
            return 'sqlite'
    
    def connect(self):
        """Establish database connection."""
        try:
            if self.db_type == 'postgresql':
                if not POSTGRES_AVAILABLE:
                    raise ImportError("PostgreSQL dependencies not available. Install psycopg[binary] or psycopg2-binary.")
                
                if PSYCOPG_VERSION == 3:
                    # psycopg3 connection
                    self.connection = psycopg.connect(
                        self.connection_string,
                        row_factory=psycopg.rows.dict_row
                    )
                else:
                    # psycopg2 connection
                    self.connection = psycopg2.connect(
                        self.connection_string,
                        cursor_factory=psycopg2.extras.RealDictCursor
                    )
                logger.info(f"Connected to PostgreSQL database (psycopg v{PSYCOPG_VERSION})")
                
            else:  # SQLite
                self.connection = sqlite3.connect(self.connection_string)
                self.connection.row_factory = sqlite3.Row
                logger.info(f"Connected to SQLite database: {self.connection_string}")
                
        except Exception as e:
            logger.error(f"Failed to connect to {self.db_type} database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries representing rows
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Convert results to list of dictionaries
            if self.db_type == 'postgresql':
                results = [dict(row) for row in cursor.fetchall()]
            else:  # SQLite
                results = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            affected_rows = cursor.rowcount
            self.connection.commit()
            cursor.close()
            
            return affected_rows
            
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            self.connection.rollback()
            raise
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            if self.db_type == 'postgresql':
                if PSYCOPG_VERSION == 3:
                    # psycopg3 uses cursor.executemany for batch operations
                    cursor.executemany(query, params_list)
                else:
                    # psycopg2 batch execution
                    psycopg2.extras.execute_batch(cursor, query, params_list)
            else:  # SQLite
                cursor.executemany(query, params_list)
            
            affected_rows = cursor.rowcount
            self.connection.commit()
            cursor.close()
            
            return affected_rows
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            self.connection.rollback()
            raise
    
    def get_schema_query(self, table_name: str) -> str:
        """Get database-specific schema query."""
        if self.db_type == 'postgresql':
            return """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
        else:  # SQLite
            return f"PRAGMA table_info({table_name})"
    
    def create_tables_if_not_exist(self):
        """Create database tables if they don't exist."""
        try:
            if self.db_type == 'postgresql':
                self._create_postgresql_tables()
            else:
                self._create_sqlite_tables()
            
            logger.info("Database tables verified/created")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def log_query(self, source: str, question: str) -> bool:
        """
        Log a query to the query_logs table.
        
        Args:
            source: Query source ('cli', 'render', 'query_system', etc.)
            question: The natural language question asked
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
                INSERT INTO query_logs (source, question, timestamp)
                VALUES (?, ?, ?)
            """ if self.db_type == 'sqlite' else """
                INSERT INTO query_logs (source, question, timestamp)
                VALUES (%s, %s, %s)
            """
            
            from datetime import datetime
            params = (source, question, datetime.now().isoformat())
            
            self.execute_update(query, params)
            logger.debug(f"Logged query from {source}: {question[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            return False
    
    def log_ai_query_event(self, source: str, question: str, answer: str = None, 
                          status: str = 'ok', model: str = None, total_ms: int = None,
                          timings: dict = None, chart: dict = None, logs: str = None,
                          error: str = None, user_id: str = None, session_id: str = None,
                          metadata: dict = None) -> bool:
        """
        Log a comprehensive AI query event to the ai_query_events table.
        
        Args:
            source: Query source ('cli', 'render', 'query_system', etc.)
            question: The natural language question asked
            answer: The AI-generated answer
            status: Query status ('ok', 'error', etc.)
            model: Model used for generation
            total_ms: Total processing time in milliseconds
            timings: Detailed timing breakdown
            chart: Chart data if generated
            logs: Processing logs
            error: Error message if failed
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Only log to ai_query_events if we're using PostgreSQL/Supabase
            if self.db_type != 'postgresql':
                logger.debug(f"Skipping ai_query_events logging for {self.db_type}")
                return True
                
            import uuid
            import json
            from datetime import datetime
            
            # Generate UUID for the event
            event_id = str(uuid.uuid4())
            
            # Convert dict parameters to JSON strings
            timings_json = json.dumps(timings) if timings else None
            chart_json = json.dumps(chart) if chart else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            query = """
                INSERT INTO ai_query_events (
                    id, created_at, source, user_id, session_id, question, answer,
                    status, model, total_ms, timings, chart, logs, error, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (
                event_id,
                datetime.now().isoformat(),
                source,
                user_id,
                session_id,
                question,
                answer,
                status,
                model,
                total_ms,
                timings_json,
                chart_json,
                logs,
                error,
                metadata_json
            )
            
            self.execute_update(query, params)
            logger.debug(f"Logged AI query event from {source}: {question[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log AI query event: {e}")
            return False
    
    def _create_postgresql_tables(self):
        """Create PostgreSQL tables with appropriate data types."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS sources (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(50),
                file_size INTEGER,
                processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_status VARCHAR(50) DEFAULT 'pending',
                error_message TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS impact_metrics (
                id SERIAL PRIMARY KEY,
                source_id INTEGER REFERENCES sources(id),
                metric_name VARCHAR(255) NOT NULL,
                metric_value DECIMAL(15,2),
                metric_unit VARCHAR(100),
                metric_category VARCHAR(100),
                context_description TEXT,
                extraction_confidence DECIMAL(3,2),
                verification_status VARCHAR(50),
                verification_accuracy DECIMAL(3,2),
                verification_notes TEXT,
                source_sheet_name VARCHAR(255),
                source_column_name VARCHAR(255),
                source_cell_reference VARCHAR(50),
                source_formula TEXT,
                extracted_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS framework_mappings (
                id SERIAL PRIMARY KEY,
                metric_id INTEGER REFERENCES impact_metrics(id),
                framework_name VARCHAR(100) NOT NULL,
                framework_code VARCHAR(100),
                mapping_confidence DECIMAL(3,2),
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS commitments (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                target_value DECIMAL(15,2),
                target_unit VARCHAR(100),
                target_date DATE,
                status VARCHAR(50) DEFAULT 'active',
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id SERIAL PRIMARY KEY,
                source VARCHAR(50) NOT NULL,
                question TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            self.execute_update(table_sql)
    
    def _create_sqlite_tables(self):
        """Create SQLite tables."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT,
                file_size INTEGER,
                processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_status TEXT DEFAULT 'pending',
                error_message TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS impact_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER REFERENCES sources(id),
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_unit TEXT,
                metric_category TEXT,
                context_description TEXT,
                extraction_confidence REAL,
                verification_status TEXT,
                verification_accuracy REAL,
                verification_notes TEXT,
                source_sheet_name TEXT,
                source_column_name TEXT,
                source_cell_reference TEXT,
                source_formula TEXT,
                extracted_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS framework_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id INTEGER REFERENCES impact_metrics(id),
                framework_name TEXT NOT NULL,
                framework_code TEXT,
                mapping_confidence REAL,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS commitments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                target_value REAL,
                target_unit TEXT,
                target_date DATE,
                status TEXT DEFAULT 'active',
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                question TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            self.execute_update(table_sql)

def get_database_connection() -> str:
    """
    Get database connection string from environment variables.
    
    Returns:
        Database connection string (PostgreSQL URL or SQLite file path)
    """
    # Check for Supabase/PostgreSQL connection
    supabase_url = os.getenv('DATABASE_URL') or os.getenv('SUPABASE_URL')
    
    if supabase_url:
        logger.info("Using PostgreSQL/Supabase database")
        return supabase_url
    
    # Fall back to SQLite
    sqlite_path = os.getenv('IMPACTOS_DB_PATH', 'db/impactos.db')
    logger.info(f"Using SQLite database: {sqlite_path}")
    
    # Ensure directory exists for SQLite
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    
    return sqlite_path

def create_database_adapter() -> DatabaseAdapter:
    """Create and return a configured database adapter."""
    connection_string = get_database_connection()
    adapter = DatabaseAdapter(connection_string)
    adapter.connect()
    adapter.create_tables_if_not_exist()
    return adapter 