"""
Data ingestion pipeline for ImpactOS AI Layer MVP Phase One.

This module handles loading XLSX/CSV files, extracting social value metrics 
using GPT-4, creating embeddings, and storing in SQLite + FAISS.
"""

import pandas as pd
import sqlite3
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

# ML and AI imports
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI

# Local imports
from schema import DatabaseSchema

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data ingestion from XLSX/CSV files into the system."""
    
    def __init__(self, db_path: str = "db/impactos.db"):
        """
        Initialize ingestion pipeline.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_schema = DatabaseSchema(db_path)
        
        # Initialize AI components
        self.openai_client = self._initialize_openai()
        self.embedding_model = self._initialize_embedding_model()
        
        # Framework mapping configuration
        self.framework_mappings = {
            'UK Social Value Model': {
                '8.1': 'Community - Local community connections',
                '8.2': 'Community - Community cohesion',
                '8.3': 'Community - Arts, heritage, culture'
            },
            'UN SDGs': {
                '1': 'No Poverty',
                '4': 'Quality Education', 
                '8': 'Decent Work and Economic Growth',
                '11': 'Sustainable Cities and Communities',
                '13': 'Climate Action'
            },
            'TOMs': {
                'NT90': 'Number of volunteering hours donated to VCSEs',
                'NT91': 'Number of donations or in-kind contributions to VCSEs'
            }
        }
    
    def _initialize_openai(self) -> Optional[OpenAI]:
        """Initialize OpenAI client with API key."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment. GPT-4 extraction disabled.")
            return None
        
        try:
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    def _initialize_embedding_model(self) -> Optional[SentenceTransformer]:
        """Initialize sentence transformer for embeddings."""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return None
    
    def ingest_file(self, file_path: str) -> bool:
        """
        Main ingestion method for processing files.
        
        Args:
            file_path: Path to file to ingest
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting ingestion of {file_path}")
            
            # 1. Validate file
            if not self._validate_file(file_path):
                return False
            
            # 2. Register source in database
            source_id = self._register_source(file_path)
            if not source_id:
                return False
            
            # 3. Load and parse file
            data = self._load_file(file_path)
            if data is None:
                return False
            
            # 4. Extract metrics using GPT-4 (if available)
            metrics = self._extract_metrics(data, source_id)
            
            # 5. Store metrics and create embeddings
            success = self._store_metrics(metrics, source_id)
            
            # 6. Update source processing status
            self._update_source_status(source_id, 'completed' if success else 'failed')
            
            logger.info(f"Ingestion completed for {file_path}")
            return success
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            return False
    
    def _validate_file(self, file_path: str) -> bool:
        """Validate file exists and is supported format."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        supported_extensions = ['.xlsx', '.csv', '.pdf']
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in supported_extensions:
            logger.error(f"Unsupported file format: {file_ext}")
            return False
        
        return True
    
    def _register_source(self, file_path: str) -> Optional[int]:
        """Register file as source in database."""
        try:
            file_stat = os.stat(file_path)
            filename = Path(file_path).name
            file_type = Path(file_path).suffix[1:]  # Remove the dot
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sources (filename, file_type, file_size_bytes, processing_status)
                    VALUES (?, ?, ?, 'processing')
                """, (filename, file_type, file_stat.st_size))
                
                source_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Source registered with ID: {source_id}")
                return source_id
                
        except sqlite3.Error as e:
            logger.error(f"Error registering source: {e}")
            return None
    
    def _load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load file into pandas DataFrame."""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.xlsx':
                # Try to read XLSX file
                df = pd.read_excel(file_path)
                logger.info(f"Loaded XLSX file with {len(df)} rows, {len(df.columns)} columns")
                
            elif file_ext == '.csv':
                # Try to read CSV file
                df = pd.read_csv(file_path)
                logger.info(f"Loaded CSV file with {len(df)} rows, {len(df.columns)} columns")
                
            elif file_ext == '.pdf':
                # PDF handling would go here - for now, placeholder
                logger.warning("PDF ingestion not yet implemented")
                return None
                
            else:
                logger.error(f"Unsupported file type: {file_ext}")
                return None
            
            # Basic data validation
            if df.empty:
                logger.warning(f"File {file_path} is empty")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def _extract_metrics(self, data: pd.DataFrame, source_id: int) -> List[Dict[str, Any]]:
        """Extract social value metrics from data using GPT-4."""
        try:
            metrics = []
            
            # Convert DataFrame to text for GPT-4 analysis
            data_sample = data.head(10).to_string()  # Use first 10 rows as sample
            column_info = f"Columns: {list(data.columns)}"
            
            if self.openai_client:
                # Use GPT-4 for intelligent extraction
                metrics = self._extract_with_gpt4(data_sample, column_info, source_id)
            else:
                # Fallback to simple heuristic extraction
                metrics = self._extract_with_heuristics(data, source_id)
            
            logger.info(f"Extracted {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return []
    
    def _extract_with_gpt4(self, data_sample: str, column_info: str, source_id: int) -> List[Dict[str, Any]]:
        """Extract metrics using GPT-4."""
        try:
            prompt = f"""
            You are analyzing social value data. Extract measurable impact metrics from this data.
            
            Data preview:
            {data_sample}
            
            {column_info}
            
            Extract social value metrics in this exact JSON format:
            [
                {{
                    "metric_name": "volunteering_hours",
                    "metric_value": 150.5,
                    "metric_unit": "hours",
                    "metric_category": "community_engagement",
                    "context_description": "Total volunteer hours donated to local charities",
                    "confidence_score": 0.95
                }}
            ]
            
            Focus on:
            - Volunteering hours, donations, charitable giving
            - Environmental metrics (carbon, waste, energy)
            - Community engagement, local procurement
            - Employment and skills development
            - Health and wellbeing initiatives
            
            Return only valid JSON array. If no clear metrics found, return empty array [].
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                metrics = json.loads(result_text)
                if isinstance(metrics, list):
                    # Add source_id to each metric
                    for metric in metrics:
                        metric['source_id'] = source_id
                    return metrics
                else:
                    logger.warning("GPT-4 returned non-list response")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT-4 JSON response: {e}")
                logger.debug(f"Raw response: {result_text}")
                return []
                
        except Exception as e:
            logger.error(f"Error with GPT-4 extraction: {e}")
            return []
    
    def _extract_with_heuristics(self, data: pd.DataFrame, source_id: int) -> List[Dict[str, Any]]:
        """Fallback heuristic extraction when GPT-4 is not available."""
        metrics = []
        
        try:
            # Look for common social value column patterns
            value_patterns = {
                'volunteering': ['volunteer', 'hours', 'community'],
                'donations': ['donation', 'charity', 'giving', 'amount'],
                'carbon': ['carbon', 'co2', 'emission', 'environmental'],
                'procurement': ['local', 'procurement', 'supplier', 'spend']
            }
            
            for category, keywords in value_patterns.items():
                for column in data.columns:
                    column_lower = column.lower()
                    
                    # Check if column name matches patterns
                    if any(keyword in column_lower for keyword in keywords):
                        try:
                            # Try to extract numeric values
                            if pd.api.types.is_numeric_dtype(data[column]):
                                total_value = data[column].sum()
                                
                                if total_value > 0:
                                    metrics.append({
                                        'source_id': source_id,
                                        'metric_name': f"{category}_{column.lower().replace(' ', '_')}",
                                        'metric_value': float(total_value),
                                        'metric_unit': self._guess_unit(column_lower),
                                        'metric_category': category,
                                        'context_description': f"Extracted from column: {column}",
                                        'confidence_score': 0.7  # Lower confidence for heuristics
                                    })
                        except Exception as e:
                            logger.debug(f"Error processing column {column}: {e}")
                            continue
            
            logger.info(f"Heuristic extraction found {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in heuristic extraction: {e}")
            return []
    
    def _guess_unit(self, column_name: str) -> str:
        """Guess appropriate unit based on column name."""
        if 'hour' in column_name:
            return 'hours'
        elif 'pound' in column_name or '£' in column_name or 'cost' in column_name:
            return 'GBP'
        elif 'dollar' in column_name or '$' in column_name:
            return 'USD'
        elif 'carbon' in column_name or 'co2' in column_name:
            return 'kg_co2'
        elif 'percent' in column_name or '%' in column_name:
            return 'percentage'
        else:
            return 'count'
    
    def _store_metrics(self, metrics: List[Dict[str, Any]], source_id: int) -> bool:
        """Store extracted metrics in database and create embeddings."""
        if not metrics:
            logger.warning("No metrics to store")
            return True
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for metric in metrics:
                    # Insert metric
                    cursor.execute("""
                        INSERT INTO impact_metrics 
                        (source_id, metric_name, metric_value, metric_unit, metric_category,
                         extraction_confidence, context_description)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric['source_id'],
                        metric['metric_name'],
                        metric['metric_value'],
                        metric['metric_unit'],
                        metric['metric_category'],
                        metric.get('confidence_score', 0.7),
                        metric.get('context_description', '')
                    ))
                    
                    metric_id = cursor.lastrowid
                    
                    # Create embedding if model is available
                    if self.embedding_model:
                        self._create_embedding(metric, metric_id, cursor)
                
                conn.commit()
                logger.info(f"Stored {len(metrics)} metrics successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
            return False
    
    def _create_embedding(self, metric: Dict[str, Any], metric_id: int, cursor: sqlite3.Cursor):
        """Create and store embedding for metric."""
        try:
            # Create text representation for embedding
            text_chunk = (
                f"Metric: {metric['metric_name']} "
                f"Value: {metric['metric_value']} {metric['metric_unit']} "
                f"Category: {metric['metric_category']} "
                f"Context: {metric.get('context_description', '')}"
            )
            
            # Generate embedding
            embedding = self.embedding_model.encode(text_chunk)
            embedding_id = f"metric_{metric_id}"
            
            # Store embedding metadata (actual vector would go to FAISS)
            cursor.execute("""
                INSERT INTO embeddings
                (metric_id, embedding_vector_id, text_chunk, chunk_type)
                VALUES (?, ?, ?, ?)
            """, (metric_id, embedding_id, text_chunk, 'metric'))
            
            # TODO: Store actual embedding vector in FAISS index
            logger.debug(f"Created embedding for metric {metric_id}")
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
    
    def _update_source_status(self, source_id: int, status: str):
        """Update source processing status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sources 
                    SET processing_status = ?, processed_timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, source_id))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating source status: {e}")


def ingest_file(file_path: str, db_path: str = "db/impactos.db") -> bool:
    """Convenience function for ingesting a single file."""
    ingestion = DataIngestion(db_path)
    return ingestion.ingest_file(file_path)


if __name__ == "__main__":
    # Test ingestion with sample data
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        success = ingest_file(file_path)
        print(f"Ingestion {'successful' if success else 'failed'}")
    else:
        print("Usage: python ingest.py <file_path>")
