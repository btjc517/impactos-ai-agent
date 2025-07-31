"""
Database schema for ImpactOS AI Layer MVP Phase One.

This module defines the SQLite database structure for storing social value metrics,
commitments, sources, and framework mappings. Also handles FAISS vector index setup.
"""

import sqlite3
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseSchema:
    """Manages SQLite database schema and FAISS vector index setup."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        """
        Initialize database schema manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.ensure_db_directory()
    
    def ensure_db_directory(self):
        """Ensure database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def initialize_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables in dependency order
                self._create_sources_table(cursor)
                self._create_frameworks_table(cursor)
                self._create_impact_metrics_table(cursor)
                self._create_commitments_table(cursor)
                self._create_framework_mappings_table(cursor)
                self._create_embeddings_table(cursor)
                
                conn.commit()
                logger.info(f"Database initialized successfully at {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _create_sources_table(self, cursor: sqlite3.Cursor) -> None:
        """Create sources table for tracking data provenance."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size_bytes INTEGER,
                processed_timestamp TIMESTAMP,
                processing_status TEXT DEFAULT 'pending',
                confidence_score REAL,
                metadata TEXT,
                UNIQUE(filename, upload_timestamp)
            )
        """)
    
    def _create_frameworks_table(self, cursor: sqlite3.Cursor) -> None:
        """Create frameworks table for social value measurement frameworks."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frameworks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                version TEXT,
                description TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert default frameworks
        default_frameworks = [
            ('UK Social Value Model', '2.0', 'UK Social Value Model for measuring community impact', 'government'),
            ('UN Sustainable Development Goals', '2015', 'United Nations 17 Sustainable Development Goals', 'international'),
            ('TOMs (Themes, Outcomes, Measures)', '3.0', 'National TOMs framework for social value measurement', 'government'),
            ('B Corp Assessment', '6.0', 'B Corporation impact assessment framework', 'certification'),
            ('MAC (Measurement Advisory Council)', '1.0', 'Social value measurement advisory framework', 'advisory')
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO frameworks (name, version, description, category)
            VALUES (?, ?, ?, ?)
        """, default_frameworks)
    
    def _create_impact_metrics_table(self, cursor: sqlite3.Cursor) -> None:
        """Create impact_metrics table for storing extracted social value metrics."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS impact_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                metric_category TEXT,
                timestamp TIMESTAMP,
                extraction_confidence REAL,
                context_description TEXT,
                raw_text TEXT,
                
                -- Enhanced citation fields for precise source tracking
                source_sheet_name TEXT,
                source_column_name TEXT,
                source_column_index INTEGER,
                source_row_index INTEGER,
                source_cell_reference TEXT,
                source_formula TEXT,
                
                -- Verification and accuracy fields
                verification_status TEXT DEFAULT 'pending',  -- pending, verified, failed, skipped
                verification_timestamp TIMESTAMP,
                verified_value REAL,
                verification_accuracy REAL,  -- 0.0 to 1.0
                verification_notes TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_impact_metrics_category 
            ON impact_metrics (metric_category)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_impact_metrics_name 
            ON impact_metrics (metric_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_impact_metrics_verification 
            ON impact_metrics (verification_status)
        """)
    
    def _create_commitments_table(self, cursor: sqlite3.Cursor) -> None:
        """Create commitments table for storing organizational commitments."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commitments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                commitment_text TEXT NOT NULL,
                commitment_type TEXT,
                target_value REAL,
                target_unit TEXT,
                target_date DATE,
                status TEXT DEFAULT 'active',
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE
            )
        """)
    
    def _create_framework_mappings_table(self, cursor: sqlite3.Cursor) -> None:
        """Create framework_mappings table for mapping metrics to frameworks."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS framework_mappings (
                impact_metric_id INTEGER,
                framework_id INTEGER,
                category TEXT,
                
                -- Enhanced framework mapping fields
                framework_name TEXT NOT NULL,
                framework_category TEXT NOT NULL,
                mapping_confidence REAL DEFAULT 0.8,
                mapping_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (impact_metric_id, framework_id),
                FOREIGN KEY (impact_metric_id) REFERENCES impact_metrics(id),
                FOREIGN KEY (framework_id) REFERENCES frameworks(id)
            );
        """)
    
    def _create_embeddings_table(self, cursor: sqlite3.Cursor) -> None:
        """Create embeddings table for vector storage metadata."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id INTEGER,
                commitment_id INTEGER,
                embedding_vector_id TEXT NOT NULL,
                text_chunk TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (metric_id) REFERENCES impact_metrics (id) ON DELETE CASCADE,
                FOREIGN KEY (commitment_id) REFERENCES commitments (id) ON DELETE CASCADE,
                CHECK ((metric_id IS NOT NULL AND commitment_id IS NULL) OR 
                       (metric_id IS NULL AND commitment_id IS NOT NULL))
            )
        """)
    
    def get_schema_info(self) -> Dict[str, List[str]]:
        """Get database schema information for debugging."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                schema_info = {}
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
                    schema_info[table] = columns
                
                return schema_info
                
        except sqlite3.Error as e:
            logger.error(f"Error getting schema info: {e}")
            return {}


def initialize_database(db_path: str = "db/impactos.db") -> None:
    """Convenience function to initialize database."""
    schema = DatabaseSchema(db_path)
    schema.initialize_database()


if __name__ == "__main__":
    # Initialize database when run directly
    initialize_database()
    print("Database schema initialized successfully!") 