"""
Data ingestion pipeline for ImpactOS AI Layer MVP Phase One.

This module handles loading XLSX/CSV files, extracting social value metrics 
using GPT-5, creating embeddings, and storing in SQLite + FAISS.
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
from vector_search import FAISSVectorSearch
from enhanced_loader import EnhancedFileLoader, load_file_enhanced

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data ingestion from XLSX/CSV files into the system."""
    
    def __init__(self, db_path: str = "db/impactos.db", use_enhanced_loader: bool = True, use_polars: bool = True):
        """
        Initialize ingestion pipeline.
        
        Args:
            db_path: Path to SQLite database
            use_enhanced_loader: Whether to use enhanced file loading with bulletproof accuracy
            use_polars: Whether to use Polars for type safety (when enhanced loader is enabled)
        """
        self.db_path = db_path
        self.db_schema = DatabaseSchema(db_path)
        
        # Initialize AI components
        self.openai_client = self._initialize_openai()
        self.embedding_model = self._initialize_embedding_model()
        self.vector_search = FAISSVectorSearch(db_path)  # Proper FAISS integration
        
        # Initialize enhanced loader for bulletproof accuracy
        self.use_enhanced_loader = use_enhanced_loader
        self.enhanced_loader = EnhancedFileLoader(use_polars=use_polars) if use_enhanced_loader else None
        self.file_metadata = {}  # Store enhanced loading metadata
        
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
            logger.warning("OPENAI_API_KEY not found in environment. GPT-5 extraction disabled.")
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
    
    def ingest_file(self, file_path: str, use_query_based: bool = True) -> bool:
        """
        Main ingestion method for processing files.
        
        Args:
            file_path: Path to file to ingest
            use_query_based: Whether to use the new query-based extraction (default: True)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting ingestion of {file_path}")
            
            # Set extraction method and file context
            self.use_query_based = use_query_based
            self.current_file_path = file_path
            
            # 1. Validate file
            if not self._validate_file(file_path):
                return False
            
            # 2. Capture filename and any existing sources for de-duplication
            filename = Path(file_path).name
            existing_source_ids = self._get_existing_source_ids_by_filename(filename)

            # 3. Register NEW source in database (marked processing)
            source_id = self._register_source(file_path)
            if not source_id:
                return False
            
            # 4. Load and parse file
            data = self._load_file(file_path)
            if data is None:
                # Roll back the newly created source if we couldn't load data
                self._safe_delete_source(source_id)
                return False
            
            # 5. Extract metrics using chosen method
            if use_query_based:
                logger.info("Using advanced query-based extraction for zero mistakes")
                from extract_v2 import QueryBasedExtraction
                extractor = QueryBasedExtraction(self.db_path)
                metrics = extractor.extract_metrics_v2(file_path, source_id)
            else:
                logger.info("Using legacy text-based extraction")
                metrics = self._extract_metrics(data, source_id)
            
            # 6. Store metrics and create embeddings
            success = self._store_metrics(metrics, source_id)
            
            # 7. Update source processing status and de-duplicate
            if success:
                self._update_source_status(source_id, 'completed')
                # Remove any prior sources with same filename (keep the new one)
                self._deduplicate_sources_by_filename(filename, keep_source_id=source_id, prior_source_ids=existing_source_ids)
            else:
                # Mark failed and remove the newly created source to preserve previous state
                self._update_source_status(source_id, 'failed')
                self._safe_delete_source(source_id)
            
            extraction_method = "query-based" if use_query_based else "text-based"
            logger.info(f"Ingestion completed for {file_path} using {extraction_method} extraction")
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

    def _get_existing_source_ids_by_filename(self, filename: str) -> list:
        """Return list of existing source IDs for the given filename."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id FROM sources WHERE filename = ?
                """, (filename,))
                rows = cursor.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            logger.warning(f"Failed to look up existing sources for {filename}: {e}")
            return []

    def _safe_delete_source(self, source_id: int) -> None:
        """Delete a source by id with cascading cleanup; ignore errors."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("PRAGMA foreign_keys = ON")
                except Exception:
                    pass
                cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
                conn.commit()
                logger.info(f"Deleted source {source_id} after failed ingestion")
        except Exception as e:
            logger.warning(f"Failed to delete source {source_id}: {e}")

    def _deduplicate_sources_by_filename(self, filename: str, keep_source_id: int, prior_source_ids: Optional[list] = None) -> None:
        """Remove old sources with the same filename and rebuild FAISS to avoid duplicates.

        Args:
            filename: The filename used as de-duplication key.
            keep_source_id: The newly created successful source to keep.
            prior_source_ids: Optional list of existing source IDs captured before ingest.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("PRAGMA foreign_keys = ON")
                except Exception:
                    pass
                # Determine candidates to delete: all with same filename except the one to keep
                if prior_source_ids is None:
                    cursor.execute("SELECT id FROM sources WHERE filename = ? AND id != ?", (filename, keep_source_id))
                    to_delete = [row[0] for row in cursor.fetchall()]
                else:
                    to_delete = [sid for sid in prior_source_ids if sid != keep_source_id]

                if not to_delete:
                    logger.info(f"No prior sources to remove for filename {filename}")
                else:
                    # Clean dependent rows in a safe order for SQLite without reliable cascades
                    # 1) Collect impact_metric ids for those sources
                    q_marks = ",".join(["?"] * len(to_delete))
                    cursor.execute(f"SELECT id FROM impact_metrics WHERE source_id IN ({q_marks})", tuple(to_delete))
                    metric_rows = cursor.fetchall()
                    metric_ids = [row[0] for row in metric_rows]

                    if metric_ids:
                        # 2) Delete framework mappings referencing those metrics
                        q_marks_mid = ",".join(["?"] * len(metric_ids))
                        cursor.execute(f"DELETE FROM framework_mappings WHERE impact_metric_id IN ({q_marks_mid})", tuple(metric_ids))
                        # 3) Delete embeddings for those metrics (in case cascade is off)
                        cursor.execute(f"DELETE FROM embeddings WHERE metric_id IN ({q_marks_mid})", tuple(metric_ids))
                        # 4) Delete the metrics themselves (in case cascade from sources is off)
                        cursor.execute(f"DELETE FROM impact_metrics WHERE id IN ({q_marks_mid})", tuple(metric_ids))

                    # 5) Finally delete the old sources
                    cursor.executemany("DELETE FROM sources WHERE id = ?", [(sid,) for sid in to_delete])
                    conn.commit()
                    logger.info(f"Removed {len(to_delete)} old source(s) for {filename}: {to_delete}")

            # Rebuild FAISS index to ensure removed embeddings are purged
            try:
                self.vector_search.rebuild_index_from_database()
            except Exception as e:
                logger.warning(f"Failed to rebuild FAISS index after de-duplication: {e}")

        except Exception as e:
            logger.warning(f"Error during de-duplication for {filename}: {e}")
    
    def _load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load file with enhanced accuracy and bulletproof data extraction."""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.xlsx', '.csv']:
                
                if self.use_enhanced_loader and self.enhanced_loader:
                    # Use enhanced loader for bulletproof accuracy
                    logger.info(f"Loading {file_path} with enhanced accuracy (openpyxl pre-scanning + type inference)")
                    
                    try:
                        df, metadata = self.enhanced_loader.load_file(file_path)
                        
                        # Store metadata for later use in extraction
                        self.file_metadata[file_path] = metadata
                        
                        # Convert Polars to pandas only when necessary.
                        # For query-based extraction, keep Polars and avoid optional pyarrow dependency.
                        if hasattr(df, 'to_pandas'):  # It's a Polars DataFrame
                            if not (hasattr(self, 'use_query_based') and self.use_query_based):
                                df_pandas = df.to_pandas()
                                logger.info("Converted Polars DataFrame to pandas for compatibility")
                            else:
                                df_pandas = df
                                logger.info("Keeping Polars DataFrame (query-based extraction); skipping pandas conversion")
                        else:
                            df_pandas = df
                        
                        # Log enhanced loading results
                        logger.info(f"Enhanced loading completed:")
                        logger.info(f"  - Rows: {metadata['total_rows']}, Columns: {metadata['total_columns']}")
                        logger.info(f"  - Load method: {metadata['load_method']}")
                        logger.info(f"  - Sheet: {metadata.get('sheet_name', 'N/A')}")
                        
                        # Log type inference results
                        type_issues = []
                        for col, inference in metadata['type_inference'].items():
                            if inference['confidence'] < 0.8:
                                type_issues.append(f"{col} ({inference['confidence']:.2f})")
                        
                        if type_issues:
                            logger.warning(f"Low confidence type inference for columns: {', '.join(type_issues)}")
                        else:
                            logger.info("All columns have high confidence type inference (>0.8)")
                        
                        return df_pandas
                        
                    except Exception as e:
                        logger.warning(f"Enhanced loading failed: {e}. Falling back to standard loading.")
                        # Fall through to standard loading
                
                # Standard loading (fallback)
                if file_ext == '.xlsx':
                    df = pd.read_excel(file_path)
                    logger.info(f"Loaded XLSX file (standard method) with {len(df)} rows, {len(df.columns)} columns")
                elif file_ext == '.csv':
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded CSV file (standard method) with {len(df)} rows, {len(df.columns)} columns")
                    
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
        """Extract social value metrics from data using GPT-5."""
        try:
            metrics = []
            
            # NEW: Try query-based extraction first (if enabled)
            if hasattr(self, 'use_query_based') and self.use_query_based:
                logger.info("Using advanced query-based extraction")
                from extract_v2 import QueryBasedExtraction
                extractor = QueryBasedExtraction(self.db_path)
                # We need the file path, so we'll add this to the extraction context
                if hasattr(self, 'current_file_path'):
                    return extractor.extract_metrics_v2(self.current_file_path, source_id)
            
            # FALLBACK: Original text-based approach
            # Convert DataFrame to text for GPT-5 analysis
            data_sample = data.head(10).to_string()  # Use first 10 rows as sample
            column_info = f"Columns: {list(data.columns)}"
            
            if self.openai_client:
                # Use GPT-5 for intelligent extraction
                metrics = self._extract_with_gpt5(data_sample, column_info, source_id)
            else:
                # Fallback to simple heuristic extraction
                metrics = self._extract_with_heuristics(data, source_id)
            
            logger.info(f"Extracted {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return []
    
    def _extract_with_gpt5(self, data_sample: str, column_info: str, source_id: int) -> List[Dict[str, Any]]:
        """Extract metrics using GPT-5 with enhanced cell reference tracking."""
        try:
            # Check if we have enhanced metadata available
            enhanced_info = ""
            if hasattr(self, 'current_file_path') and self.current_file_path in self.file_metadata:
                metadata = self.file_metadata[self.current_file_path]
                enhanced_info = f"""
            
            ENHANCED LOADING METADATA (BULLETPROOF ACCURACY):
            - File loaded with openpyxl pre-scanning and type inference
            - Sheet: {metadata.get('sheet_name', 'Unknown')}
            - Load method: {metadata.get('load_method', 'Unknown')}
            - Type inference confidence levels available for precise citation
            
            TYPE INFERENCE RESULTS:
            {json.dumps({col: {'type': info['inferred_type'], 'confidence': info['confidence']} 
                        for col, info in metadata.get('type_inference', {}).items()}, indent=2)}
            """
            
            prompt = f"""
            You are analyzing social value data from a spreadsheet. Extract measurable impact metrics with PRECISE CITATIONS.

            Data preview (first 10 rows):
            {data_sample}

            {column_info}
            {enhanced_info}

            CRITICAL: For each metric, you MUST provide exact source location details for verification.

            Extract social value metrics in this exact JSON format:
            [
                {{
                    "metric_name": "volunteering_hours",
                    "metric_value": 150.5,
                    "metric_unit": "hours",
                    "metric_category": "community_engagement",
                    "context_description": "Total volunteer hours donated to local charities",
                    "confidence_score": 0.95,
                    "source_sheet_name": "Sheet1",
                    "source_column_name": "Volunteer Hours",
                    "source_column_index": 3,
                    "source_row_index": 12,
                    "source_cell_reference": "D13",
                    "source_formula": "SUM(D2:D12)"
                }}
            ]

            EXTRACTION GUIDELINES:
            1. Focus on these social value categories:
               - Volunteering hours, donations, charitable giving
               - Environmental metrics (carbon, waste, energy)
               - Community engagement, local procurement
               - Employment and skills development
               - Health and wellbeing initiatives

            2. CITATION REQUIREMENTS (MANDATORY):
               - source_column_name: Exact column header name
               - source_column_index: Zero-based column index (A=0, B=1, C=2, etc.)
               - source_row_index: Zero-based row index (header=0, first data=1, etc.)
               - source_cell_reference: Excel-style reference (A1, B5, etc.)
               - source_formula: If aggregated (SUM, AVERAGE), provide formula

            3. VALUE EXTRACTION:
               - Extract exact numeric values as shown in cells
               - If summing multiple cells, provide the SUM formula
               - If calculating average, provide AVERAGE formula
               - Round to appropriate precision (2 decimal places max)

            4. CONFIDENCE SCORING:
               - 0.9-1.0: Exact cell reference with clear numeric value
               - 0.7-0.8: Calculated from multiple cells with formula
               - 0.5-0.6: Inferred from context or partial data
               - Below 0.5: Don't include the metric

            EXAMPLE SCENARIOS:
            - Single cell value: "source_cell_reference": "B5", "source_formula": null
            - Sum of column: "source_cell_reference": "B15", "source_formula": "SUM(B2:B14)"
            - Average: "source_cell_reference": "C10", "source_formula": "AVERAGE(C2:C9)"

            Return only valid JSON array. If no clear metrics with precise citations found, return empty array [].
            NEVER make up cell references - only include metrics you can precisely locate.
            """

            from config import get_config
            cfg = get_config()
            # Log GPT call parameters for ingestion (legacy text-based extraction)
            try:
                _model = getattr(cfg.extraction, 'query_generation_model', 'gpt-5')
                _max_tokens = getattr(cfg.extraction, 'extraction_max_tokens', getattr(cfg.extraction, 'gpt4_max_tokens_extraction', 4000))
                logger.info(f"LLM call start | context=ingestion_legacy_extraction params={{'model': '{_model}', 'max_tokens': {_max_tokens}, 'enforce_json': False, 'reasoning_effort': None, 'text_verbosity': None, 'messages_count': 1}}")
            except Exception:
                pass
            response = self.openai_client.chat.completions.create(
                model=getattr(cfg.extraction, 'query_generation_model', 'gpt-5'),
                messages=[{"role": "user", "content": prompt}],
                # Temperature unused for GPT-5-only setup
                max_completion_tokens=getattr(cfg.extraction, 'extraction_max_tokens', getattr(cfg.extraction, 'gpt4_max_tokens_extraction', 4000))
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                # Clean the response - sometimes GPT-5 adds markdown formatting
                cleaned_response = result_text.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response.split('```json')[1]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response.split('```')[0]
                cleaned_response = cleaned_response.strip()
                
                logger.debug(f"GPT-5 raw response: {result_text[:500]}...")
                logger.debug(f"Cleaned response: {cleaned_response[:500]}...")
                
                metrics = json.loads(cleaned_response)
                if isinstance(metrics, list):
                    # Add source_id to each metric and validate citations
                    validated_metrics = []
                    for metric in metrics:
                        metric['source_id'] = source_id
                        
                        # Validate that required citation fields are present
                        required_fields = ['source_column_name', 'source_column_index', 
                                         'source_row_index', 'source_cell_reference']
                        
                        if all(field in metric and metric[field] is not None for field in required_fields):
                            validated_metrics.append(metric)
                        else:
                            logger.warning(f"Metric '{metric.get('metric_name')}' missing citation fields - skipped")
                    
                    logger.info(f"GPT-5 extracted {len(validated_metrics)} metrics with citations (from {len(metrics)} total)")
                    return validated_metrics
                else:
                    logger.warning("GPT-5 returned non-list response")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT-5 JSON response: {e}")
                logger.error(f"Raw response: {result_text}")
                return []

        except Exception as e:
            logger.error(f"Error with GPT-5 extraction: {e}")
            return []
    
    def _extract_with_heuristics(self, data: pd.DataFrame, source_id: int) -> List[Dict[str, Any]]:
        """Fallback heuristic extraction when GPT-5 is not available."""
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
                    # Insert metric with enhanced citation fields
                    cursor.execute("""
                        INSERT INTO impact_metrics 
                        (source_id, metric_name, metric_value, metric_unit, metric_category,
                         extraction_confidence, context_description,
                         source_sheet_name, source_column_name, source_column_index,
                         source_row_index, source_cell_reference, source_formula)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric['source_id'],
                        metric['metric_name'],
                        metric['metric_value'],
                        metric['metric_unit'],
                        metric['metric_category'],
                        metric.get('confidence_score', 0.7),
                        metric.get('context_description', ''),
                        metric.get('source_sheet_name'),
                        metric.get('source_column_name'),
                        metric.get('source_column_index'),
                        metric.get('source_row_index'),
                        metric.get('source_cell_reference'),
                        metric.get('source_formula')
                    ))

                    metric_id = cursor.lastrowid

                    # Create embedding if model is available
                    if self.embedding_model:
                        self._create_embedding(metric, metric_id, cursor)

                conn.commit()
                logger.info(f"Stored {len(metrics)} metrics with citations successfully")
                return True

        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
            return False
    
    def _create_embedding(self, metric: Dict[str, Any], metric_id: int, cursor: sqlite3.Cursor):
        """Create and store embedding for metric."""
        try:
            # Determine source filename and vendor token for richer embedding context
            file_name = os.path.basename(self.current_file_path) if hasattr(self, 'current_file_path') else 'unknown'
            vendor_name = None
            try:
                base_no_ext = os.path.splitext(file_name)[0]
                tokens = [t for t in base_no_ext.split('_') if t]
                if len(tokens) >= 2:
                    vendor_name = tokens[1]
                elif tokens:
                    vendor_name = tokens[0]
            except Exception:
                vendor_name = None
            vendor_part = f"Vendor: {vendor_name}; " if vendor_name else ""
            source_part = f"Source: {file_name}; " if file_name else ""

            # Create text representation for embedding
            text_chunk = (
                f"{source_part}{vendor_part}"
                f"Metric: {metric['metric_name']} "
                f"Value: {metric['metric_value']} {metric['metric_unit']} "
                f"Category: {metric['metric_category']} "
                f"Context: {metric.get('context_description', '')}"
            )
            
            # Generate embedding vector
            embedding_vector = self.embedding_model.encode(text_chunk)
            embedding_id = f"metric_{metric_id}"
            
            # Store embedding metadata in SQLite
            cursor.execute("""
                INSERT INTO embeddings
                (metric_id, embedding_vector_id, text_chunk, chunk_type)
                VALUES (?, ?, ?, ?)
            """, (metric_id, embedding_id, text_chunk, 'metric'))
            
            # Enhance with enhanced loader metadata if available
            enhanced_metadata = {}
            if hasattr(self, 'current_file_path') and self.current_file_path in self.file_metadata:
                file_meta = self.file_metadata[self.current_file_path]
                enhanced_metadata = {
                    'enhanced_loading': True,
                    'load_method': file_meta.get('load_method'),
                    'sheet_name': file_meta.get('sheet_name'),
                    'type_inference_available': True,
                    'cell_references_available': 'cell_references' in file_meta
                }
                
                # Add column type inference info if available
                column_name = metric.get('source_column_name')
                if column_name and column_name in file_meta.get('type_inference', {}):
                    type_info = file_meta['type_inference'][column_name]
                    enhanced_metadata['column_type_info'] = {
                        'inferred_type': type_info['inferred_type'],
                        'confidence': type_info['confidence'],
                        'type_issues': type_info.get('issues', [])
                    }
            
            # Store actual embedding vector in FAISS index
            embedding_data = {
                'vector': embedding_vector,
                'text_chunk': text_chunk,
                'metric_id': metric_id,
                'chunk_type': 'metric',
                'metric_name': metric['metric_name'],
                'metric_category': metric['metric_category'],
                'filename': file_name,
                'enhanced_metadata': enhanced_metadata,
                'source_info': {
                    'metric_value': metric['metric_value'],
                    'metric_unit': metric['metric_unit'],
                    'context_description': metric.get('context_description', ''),
                    'source_sheet_name': metric.get('source_sheet_name'),
                    'source_column_name': metric.get('source_column_name'),
                    'source_cell_reference': metric.get('source_cell_reference'),
                    'source_formula': metric.get('source_formula'),
                    'verification_status': 'pending'
                }
            }
            
            # Add to FAISS index
            self.vector_search.add_embeddings([embedding_data])
            
            logger.debug(f"Created and stored embedding for metric {metric_id} in FAISS")
            
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


def ingest_file(file_path: str, db_path: str = "db/impactos.db", use_enhanced_loader: bool = True, use_polars: bool = True) -> bool:
    """
    Convenience function for ingesting a single file with enhanced accuracy.
    
    Args:
        file_path: Path to file to ingest
        db_path: Path to database
        use_enhanced_loader: Whether to use bulletproof enhanced loading (default: True)
        use_polars: Whether to use Polars for type safety (default: True)
    """
    ingestion = DataIngestion(db_path, use_enhanced_loader=use_enhanced_loader, use_polars=use_polars)
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
