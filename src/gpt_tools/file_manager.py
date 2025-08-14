"""
File management for OpenAI API integration.

Handles file uploads, storage, and retrieval for Assistants API.
"""

import os
import logging
from typing import List, Dict, Any, Optional, BinaryIO
from pathlib import Path
from openai import OpenAI
import hashlib
import json
import pandas as pd
import tempfile
from .data_store import DataStore

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize File Manager.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.uploaded_files: Dict[str, Dict[str, Any]] = {}  # local_path -> metadata
        self.file_cache: Dict[str, str] = {}  # content_hash -> file_id
        self.data_store = DataStore()
    
    def preprocess_excel(self, file_path: str) -> str:
        """
        Convert Excel file to text format for OpenAI processing.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Path to converted text file
        """
        try:
            # Read Excel file with better handling for different data types
            df_dict = pd.read_excel(file_path, sheet_name=None, dtype=str, keep_default_na=False)
            
            # Build text representation
            content_lines = [f"Excel File: {Path(file_path).name}\n"]
            content_lines.append("=" * 50 + "\n\n")
            
            for sheet_name, df in df_dict.items():
                content_lines.append(f"Sheet: {sheet_name}\n")
                content_lines.append("-" * 30 + "\n")
                
                # Add summary stats
                content_lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}\n")
                content_lines.append(f"Columns: {', '.join(df.columns)}\n\n")
                
                # Clean and format data for better processing
                df_clean = self._clean_dataframe(df)
                
                # Convert DataFrame to readable format with better formatting
                content_lines.append("Data (formatted for analysis):\n")
                content_lines.append(df_clean.to_string(index=False, na_rep=''))
                content_lines.append("\n\n")
                
                # Add structured data section for GPT processing
                content_lines.append("Structured Data Analysis:\n")
                content_lines.append(self._analyze_dataframe_structure(df_clean, sheet_name))
                content_lines.append("\n\n")
                
                # Add basic statistics for numeric-like columns
                numeric_cols = self._identify_numeric_columns(df_clean)
                if numeric_cols:
                    content_lines.append("Numeric Analysis:\n")
                    for col in numeric_cols:
                        values = pd.to_numeric(df_clean[col].str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                        stats = values.describe()
                        content_lines.append(f"{col}: Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}\n")
                    content_lines.append("\n")
            
            # Save to temporary text file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp.write(''.join(content_lines))
                return tmp.name
                
        except Exception as e:
            logger.error(f"Failed to preprocess Excel file {file_path}: {e}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame for better text processing."""
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        
        # Replace empty strings and None with empty
        df_clean = df_clean.fillna('').replace('', '')
        
        # Clean up common formatting issues
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Strip whitespace
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # Replace multiple spaces with single space
                df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        return df_clean
    
    def _identify_numeric_columns(self, df: pd.DataFrame) -> list:
        """Identify columns that likely contain numeric data."""
        numeric_cols = []
        
        for col in df.columns:
            # Try to convert to numeric after cleaning
            sample_values = df[col].str.replace(r'[^\d.-]', '', regex=True)
            numeric_count = pd.to_numeric(sample_values, errors='coerce').notna().sum()
            
            # If more than 50% of values can be converted to numeric, consider it numeric
            if len(df) > 0 and numeric_count / len(df) > 0.5:
                numeric_cols.append(col)
        
        return numeric_cols
    
    def _analyze_dataframe_structure(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Analyze DataFrame structure for GPT processing."""
        analysis_lines = []
        
        # Column analysis
        analysis_lines.append(f"Sheet '{sheet_name}' contains the following data:\n")
        
        for col in df.columns:
            unique_count = df[col].nunique()
            non_empty_count = (df[col] != '').sum()
            
            analysis_lines.append(f"- Column '{col}': {non_empty_count} non-empty values, {unique_count} unique values")
            
            # Sample values for better understanding
            sample_values = df[col][df[col] != ''].head(3).tolist()
            if sample_values:
                analysis_lines.append(f"  Sample values: {sample_values}")
            analysis_lines.append("\n")
        
        # Look for potential metrics patterns
        potential_metrics = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['amount', 'total', 'value', 'cost', 'revenue', 'salary', 'pay', 'bonus', 'donation']):
                potential_metrics.append(col)
        
        if potential_metrics:
            analysis_lines.append(f"Potential metric columns identified: {potential_metrics}\n")
        
        return ''.join(analysis_lines)
    
    def upload_file(self, file_path: str, purpose: str = "assistants") -> str:
        """
        Upload a file to OpenAI.
        
        Args:
            file_path: Path to local file
            purpose: Purpose of upload ("assistants" or "fine-tune")
            
        Returns:
            OpenAI file ID
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Handle Excel files by converting to text
            actual_file_path = file_path
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                logger.info(f"Converting Excel file {file_path.name} to text format")
                actual_file_path = Path(self.preprocess_excel(str(file_path)))
            
            # Check cache and database to avoid duplicate uploads
            content_hash = self._hash_file(actual_file_path)
            
            # Check database first
            existing_file = self.data_store.get_file_by_hash(content_hash)
            if existing_file and existing_file['openai_file_id']:
                logger.info(f"Using existing file from database for {file_path.name}")
                return existing_file['openai_file_id']
            
            # Check local cache
            if content_hash in self.file_cache:
                logger.info(f"Using cached file for {file_path.name}")
                return self.file_cache[content_hash]
            
            # Upload file
            with open(actual_file_path, 'rb') as file:
                response = self.client.files.create(
                    file=file,
                    purpose=purpose
                )
            
            file_id = response.id
            
            # Store metadata
            self.uploaded_files[str(file_path)] = {
                "file_id": file_id,
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "purpose": purpose,
                "content_hash": content_hash
            }
            
            self.file_cache[content_hash] = file_id
            
            # Store in database
            try:
                self.data_store.store_file_metadata(
                    str(file_path), file_id, content_hash
                )
            except Exception as e:
                logger.warning(f"Failed to store file metadata in database: {e}")
            
            logger.info(f"Uploaded {file_path.name} -> {file_id}")
            
            # Clean up temp file if it was created for Excel
            if actual_file_path != file_path:
                try:
                    os.unlink(actual_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
            
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            raise
    
    def upload_batch(self, file_paths: List[str], purpose: str = "assistants") -> List[str]:
        """
        Upload multiple files.
        
        Args:
            file_paths: List of file paths
            purpose: Purpose of upload
            
        Returns:
            List of file IDs
        """
        file_ids = []
        for file_path in file_paths:
            try:
                file_id = self.upload_file(file_path, purpose)
                file_ids.append(file_id)
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")
                # Continue with other files
        
        return file_ids
    
    def upload_content(self, content: str, filename: str, purpose: str = "assistants") -> str:
        """
        Upload text content as a file.
        
        Args:
            content: Text content to upload
            filename: Name for the file
            purpose: Purpose of upload
            
        Returns:
            OpenAI file ID
        """
        try:
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Upload
            file_id = self.upload_file(tmp_path, purpose)
            
            # Clean up
            os.unlink(tmp_path)
            
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to upload content as {filename}: {e}")
            raise
    
    def list_files(self, purpose: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List uploaded files.
        
        Args:
            purpose: Filter by purpose (optional)
            
        Returns:
            List of file metadata
        """
        try:
            response = self.client.files.list(purpose=purpose) if purpose else self.client.files.list()
            
            files = []
            for file in response.data:
                files.append({
                    "id": file.id,
                    "filename": file.filename,
                    "bytes": file.bytes,
                    "created_at": file.created_at,
                    "purpose": file.purpose
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from OpenAI.
        
        Args:
            file_id: OpenAI file ID
            
        Returns:
            True if successful
        """
        try:
            self.client.files.delete(file_id)
            
            # Remove from cache
            for path, metadata in list(self.uploaded_files.items()):
                if metadata.get("file_id") == file_id:
                    del self.uploaded_files[path]
                    if "content_hash" in metadata:
                        self.file_cache.pop(metadata["content_hash"], None)
            
            logger.info(f"Deleted file: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    def get_file_content(self, file_id: str) -> bytes:
        """
        Retrieve file content from OpenAI.
        
        Args:
            file_id: OpenAI file ID
            
        Returns:
            File content as bytes
        """
        try:
            response = self.client.files.content(file_id)
            return response.read()
            
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {e}")
            raise
    
    def process_data_directory(self, data_dir: str = "data") -> Dict[str, List[str]]:
        """
        Upload all data files from a directory.
        
        Args:
            data_dir: Directory containing data files
            
        Returns:
            Dict mapping file types to lists of file IDs
        """
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                logger.warning(f"Data directory not found: {data_dir}")
                return {}
            
            file_ids_by_type = {
                "csv": [],
                "xlsx": [],
                "pdf": [],
                "json": [],
                "other": []
            }
            
            # Process files by type
            for file_path in data_path.iterdir():
                if not file_path.is_file():
                    continue
                
                suffix = file_path.suffix.lower()[1:]  # Remove dot
                file_type = suffix if suffix in file_ids_by_type else "other"
                
                try:
                    file_id = self.upload_file(str(file_path))
                    file_ids_by_type[file_type].append(file_id)
                    logger.info(f"Uploaded {file_path.name} as {file_type}")
                except Exception as e:
                    logger.error(f"Failed to upload {file_path.name}: {e}")
            
            # Log summary
            total_files = sum(len(ids) for ids in file_ids_by_type.values())
            logger.info(f"Uploaded {total_files} files from {data_dir}")
            
            return file_ids_by_type
            
        except Exception as e:
            logger.error(f"Failed to process data directory: {e}")
            return {}
    
    def _hash_file(self, file_path: Path) -> str:
        """Calculate hash of file content for deduplication."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_metadata(self, output_path: str = "file_metadata.json"):
        """Save uploaded files metadata to JSON."""
        try:
            metadata = {
                "uploaded_files": self.uploaded_files,
                "file_cache": self.file_cache
            }
            
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved metadata to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def load_metadata(self, input_path: str = "file_metadata.json"):
        """Load uploaded files metadata from JSON."""
        try:
            if not os.path.exists(input_path):
                logger.warning(f"Metadata file not found: {input_path}")
                return
            
            with open(input_path, 'r') as f:
                metadata = json.load(f)
            
            self.uploaded_files = metadata.get("uploaded_files", {})
            self.file_cache = metadata.get("file_cache", {})
            
            logger.info(f"Loaded metadata from {input_path}")
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    
    def cleanup_all(self):
        """Delete all uploaded files."""
        try:
            files = self.list_files()
            for file in files:
                self.delete_file(file["id"])
            
            self.uploaded_files.clear()
            self.file_cache.clear()
            
            logger.info("Cleaned up all uploaded files")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")