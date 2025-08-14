"""
Data persistence and indexing for ImpactOS.

Manages storage and retrieval of processed data and metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sqlite3
import hashlib

logger = logging.getLogger(__name__)


class DataStore:
    """Manages persistent storage of processed impact data."""
    
    def __init__(self, data_dir: str = "storage"):
        """
        Initialize data store.
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # SQLite database for metadata and indexing
        self.db_path = self.data_dir / "impactos.db"
        self.init_database()
        
        # JSON storage for processed data
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        logger.info(f"DataStore initialized at {self.data_dir}")
    
    def init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        file_hash TEXT NOT NULL,
                        openai_file_id TEXT,
                        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        file_type TEXT,
                        processed BOOLEAN DEFAULT FALSE
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id INTEGER,
                        name TEXT NOT NULL,
                        value TEXT,
                        unit TEXT,
                        category TEXT,
                        period TEXT,
                        extracted_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (file_id) REFERENCES files (id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        thread_id TEXT,
                        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def store_file_metadata(self, file_path: str, openai_file_id: str, 
                           file_hash: str) -> int:
        """
        Store file metadata in database.
        
        Args:
            file_path: Local file path
            openai_file_id: OpenAI file ID
            file_hash: File content hash
            
        Returns:
            Database file ID
        """
        try:
            file_path_obj = Path(file_path)
            file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
            file_type = file_path_obj.suffix.lower().lstrip('.')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO files 
                    (file_path, file_hash, openai_file_id, file_size, file_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (str(file_path), file_hash, openai_file_id, file_size, file_type))
                
                file_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Stored file metadata: {file_path} -> ID {file_id}")
                return file_id
                
        except Exception as e:
            logger.error(f"Failed to store file metadata: {e}")
            raise
    
    def store_metrics(self, file_id: int, metrics: List[Dict[str, Any]]):
        """
        Store extracted metrics in database.
        
        Args:
            file_id: Database file ID
            metrics: List of metric dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                for metric in metrics:
                    conn.execute("""
                        INSERT INTO metrics (file_id, name, value, unit, category, period)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        file_id,
                        metric.get('name', ''),
                        str(metric.get('value', '')),
                        metric.get('unit', ''),
                        metric.get('category', ''),
                        metric.get('period', '')
                    ))
                
                # Mark file as processed
                conn.execute("""
                    UPDATE files SET processed = TRUE WHERE id = ?
                """, (file_id,))
                
                conn.commit()
                
                logger.info(f"Stored {len(metrics)} metrics for file ID {file_id}")
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
            raise
    
    def get_file_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get file metadata by hash."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM files WHERE file_hash = ?
                """, (file_hash,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Failed to get file by hash: {e}")
            return None
    
    def get_processed_files(self) -> List[Dict[str, Any]]:
        """Get all processed files."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM files WHERE processed = TRUE
                    ORDER BY upload_time DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get processed files: {e}")
            return []
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        """Get all uploaded files."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM files 
                    ORDER BY upload_time DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get all files: {e}")
            return []
    
    def search_metrics(self, query: str = None, category: str = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search metrics by name or category.
        
        Args:
            query: Search query for metric names
            category: Filter by category
            limit: Maximum results
            
        Returns:
            List of matching metrics with file info
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = """
                    SELECT m.*, f.file_path, f.file_type
                    FROM metrics m
                    JOIN files f ON m.file_id = f.id
                    WHERE 1=1
                """
                params = []
                
                if query:
                    sql += " AND m.name LIKE ?"
                    params.append(f"%{query}%")
                
                if category:
                    sql += " AND m.category = ?"
                    params.append(category)
                
                sql += " ORDER BY m.extracted_time DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to search metrics: {e}")
            return []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of stored data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # File counts
                file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                processed_count = conn.execute("SELECT COUNT(*) FROM files WHERE processed = TRUE").fetchone()[0]
                
                # Metric counts
                metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
                
                # Category breakdown
                cursor = conn.execute("""
                    SELECT category, COUNT(*) 
                    FROM metrics 
                    WHERE category != '' 
                    GROUP BY category
                """)
                categories = dict(cursor.fetchall())
                
                return {
                    'files_total': file_count,
                    'files_processed': processed_count,
                    'metrics_extracted': metric_count,
                    'categories': categories,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {}
    
    def store_session_info(self, session_id: str, thread_id: str):
        """Store session information for query context."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, thread_id, last_activity)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (session_id, thread_id))
                
                conn.commit()
                logger.debug(f"Stored session info: {session_id} -> {thread_id}")
                
        except Exception as e:
            logger.error(f"Failed to store session info: {e}")
    
    def get_recent_thread(self) -> Optional[str]:
        """Get most recent thread ID for context."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT thread_id FROM sessions 
                    ORDER BY last_activity DESC 
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Failed to get recent thread: {e}")
            return None
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data beyond specified days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean old sessions
                conn.execute("""
                    DELETE FROM sessions 
                    WHERE last_activity < datetime('now', '-{} days')
                """.format(days))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days} days")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def export_data(self, output_path: str):
        """Export all data to JSON file."""
        try:
            data = {
                'files': self.get_processed_files(),
                'summary': self.get_data_summary(),
                'export_time': datetime.now().isoformat()
            }
            
            # Add all metrics
            all_metrics = self.search_metrics(limit=10000)
            data['metrics'] = all_metrics
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Data exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise