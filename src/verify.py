"""
Data verification module for ImpactOS AI Layer MVP Phase One.

This module validates AI-extracted metrics against source files to ensure
no hallucinated data is reported. Provides precise citations and accuracy scoring.
"""

import pandas as pd
import sqlite3
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataVerifier:
    """Handles verification of AI-extracted metrics against source files."""
    
    def __init__(self, db_path: str = "db/impactos.db", data_dir: str = "data"):
        """
        Initialize data verifier.
        
        Args:
            db_path: Path to SQLite database
            data_dir: Directory containing source data files
        """
        self.db_path = db_path
        self.data_dir = data_dir
        self.verification_tolerance = 0.01  # 1% tolerance for floating point comparisons
    
    def verify_all_pending_metrics(self) -> Dict[str, Any]:
        """
        Verify all metrics with 'pending' verification status.
        
        Returns:
            Verification summary with accuracy statistics
        """
        try:
            # Get all pending metrics
            pending_metrics = self._get_pending_metrics()
            
            if not pending_metrics:
                logger.info("No pending metrics to verify")
                return {"total": 0, "verified": 0, "failed": 0, "accuracy": 1.0}
            
            logger.info(f"Starting verification of {len(pending_metrics)} metrics")
            
            verification_results = []
            for metric in pending_metrics:
                result = self._verify_metric(metric)
                verification_results.append(result)
            
            # Calculate summary statistics
            summary = self._calculate_verification_summary(verification_results)
            
            logger.info(f"Verification completed: {summary['accuracy']:.1%} accuracy")
            return summary
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return {"error": str(e)}
    
    def verify_metric_by_id(self, metric_id: int) -> Dict[str, Any]:
        """
        Verify a specific metric by ID.
        
        Args:
            metric_id: Database ID of metric to verify
            
        Returns:
            Verification result for the metric
        """
        try:
            metric = self._get_metric_by_id(metric_id)
            if not metric:
                return {"error": f"Metric {metric_id} not found"}
            
            return self._verify_metric(metric)
            
        except Exception as e:
            logger.error(f"Error verifying metric {metric_id}: {e}")
            return {"error": str(e)}
    
    def _get_pending_metrics(self) -> List[Dict[str, Any]]:
        """Get all metrics with pending verification status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        im.id, im.metric_name, im.metric_value, im.metric_unit,
                        im.source_sheet_name, im.source_column_name, 
                        im.source_column_index, im.source_row_index,
                        im.source_cell_reference, im.context_description,
                        s.filename, s.file_type
                    FROM impact_metrics im
                    JOIN sources s ON im.source_id = s.id
                    WHERE im.verification_status = 'pending'
                    ORDER BY im.created_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting pending metrics: {e}")
            return []
    
    def _get_metric_by_id(self, metric_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific metric by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        im.id, im.metric_name, im.metric_value, im.metric_unit,
                        im.source_sheet_name, im.source_column_name, 
                        im.source_column_index, im.source_row_index,
                        im.source_cell_reference, im.context_description,
                        s.filename, s.file_type
                    FROM impact_metrics im
                    JOIN sources s ON im.source_id = s.id
                    WHERE im.id = ?
                """, (metric_id,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error getting metric {metric_id}: {e}")
            return None
    
    def _verify_metric(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a single metric against its source file.
        
        Args:
            metric: Metric dictionary from database
            
        Returns:
            Verification result with accuracy and details
        """
        try:
            metric_id = metric['id']
            filename = metric['filename']
            file_type = metric['file_type']
            
            # Load the source file
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                return self._create_verification_result(
                    metric_id, False, 0.0, f"Source file not found: {filename}"
                )
            
            # Verify based on file type
            if file_type in ['xlsx', 'csv']:
                return self._verify_spreadsheet_metric(metric, file_path)
            else:
                return self._create_verification_result(
                    metric_id, False, 0.0, f"Unsupported file type: {file_type}"
                )
                
        except Exception as e:
            logger.error(f"Error verifying metric {metric['id']}: {e}")
            return self._create_verification_result(
                metric['id'], False, 0.0, f"Verification error: {e}"
            )
    
    def _verify_spreadsheet_metric(self, metric: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Verify a metric extracted from a spreadsheet file."""
        try:
            metric_id = metric['id']
            reported_value = metric['metric_value']
            
            # Load the spreadsheet
            if file_path.endswith('.xlsx'):
                # Load specific sheet if specified
                sheet_name = metric['source_sheet_name'] or 0
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_csv(file_path)
            
            # Method 1: Verify using specific cell reference (most precise)
            if metric['source_cell_reference']:
                return self._verify_by_cell_reference(metric, df, file_path)
            
            # Method 2: Verify using column and row indices
            elif metric['source_column_index'] is not None and metric['source_row_index'] is not None:
                return self._verify_by_indices(metric, df)
            
            # Method 3: Verify using column name and heuristics
            elif metric['source_column_name']:
                return self._verify_by_column_name(metric, df)
            
            # Method 4: Fallback - search for value in context area
            else:
                return self._verify_by_value_search(metric, df)
                
        except Exception as e:
            logger.error(f"Error in spreadsheet verification: {e}")
            return self._create_verification_result(
                metric_id, False, 0.0, f"Spreadsheet verification error: {e}"
            )
    
    def _verify_by_cell_reference(self, metric: Dict[str, Any], df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        """Verify using exact cell reference (e.g., 'B5', 'Sheet1!C10')."""
        try:
            cell_ref = metric['source_cell_reference']
            reported_value = metric['metric_value']
            
            # Parse cell reference (e.g., 'B5' -> column=1, row=4)
            col_idx, row_idx = self._parse_cell_reference(cell_ref)
            
            if col_idx >= len(df.columns) or row_idx >= len(df):
                return self._create_verification_result(
                    metric['id'], False, 0.0, f"Cell reference {cell_ref} out of bounds"
                )
            
            # Get actual value from the cell
            actual_value = df.iloc[row_idx, col_idx]
            
            # Compare values
            if pd.isna(actual_value):
                return self._create_verification_result(
                    metric['id'], False, 0.0, f"Cell {cell_ref} is empty"
                )
            
            # Try to convert to numeric if needed
            if isinstance(actual_value, str):
                actual_value = self._extract_numeric_value(actual_value)
            
            accuracy = self._calculate_value_accuracy(reported_value, actual_value)
            is_accurate = accuracy >= (1.0 - self.verification_tolerance)
            
            notes = f"Cell {cell_ref}: reported={reported_value}, actual={actual_value}, accuracy={accuracy:.1%}"
            
            return self._create_verification_result(
                metric['id'], is_accurate, accuracy, notes, actual_value
            )
            
        except Exception as e:
            return self._create_verification_result(
                metric['id'], False, 0.0, f"Cell reference verification error: {e}"
            )
    
    def _verify_by_indices(self, metric: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Verify using column and row indices."""
        try:
            col_idx = metric['source_column_index']
            row_idx = metric['source_row_index']
            reported_value = metric['metric_value']
            
            if col_idx >= len(df.columns) or row_idx >= len(df):
                return self._create_verification_result(
                    metric['id'], False, 0.0, f"Indices ({row_idx}, {col_idx}) out of bounds"
                )
            
            actual_value = df.iloc[row_idx, col_idx]
            
            if pd.isna(actual_value):
                return self._create_verification_result(
                    metric['id'], False, 0.0, f"Cell at ({row_idx}, {col_idx}) is empty"
                )
            
            if isinstance(actual_value, str):
                actual_value = self._extract_numeric_value(actual_value)
            
            accuracy = self._calculate_value_accuracy(reported_value, actual_value)
            is_accurate = accuracy >= (1.0 - self.verification_tolerance)
            
            notes = f"Cell ({row_idx}, {col_idx}): reported={reported_value}, actual={actual_value}"
            
            return self._create_verification_result(
                metric['id'], is_accurate, accuracy, notes, actual_value
            )
            
        except Exception as e:
            return self._create_verification_result(
                metric['id'], False, 0.0, f"Index verification error: {e}"
            )
    
    def _verify_by_column_name(self, metric: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Verify by searching in the specified column."""
        try:
            column_name = metric['source_column_name']
            reported_value = metric['metric_value']
            
            # Find column by exact or partial match
            matching_columns = [col for col in df.columns if column_name.lower() in str(col).lower()]
            
            if not matching_columns:
                return self._create_verification_result(
                    metric['id'], False, 0.0, f"Column '{column_name}' not found"
                )
            
            # Use the first matching column
            target_column = matching_columns[0]
            column_data = df[target_column].dropna()
            
            # Try to find the reported value in the column
            for idx, value in column_data.items():
                if isinstance(value, str):
                    numeric_value = self._extract_numeric_value(value)
                else:
                    numeric_value = value
                
                accuracy = self._calculate_value_accuracy(reported_value, numeric_value)
                if accuracy >= (1.0 - self.verification_tolerance):
                    notes = f"Found in column '{target_column}' row {idx}: reported={reported_value}, actual={numeric_value}"
                    return self._create_verification_result(
                        metric['id'], True, accuracy, notes, numeric_value
                    )
            
            # If exact match not found, check if it's an aggregation
            return self._verify_column_aggregation(metric, column_data, target_column)
            
        except Exception as e:
            return self._create_verification_result(
                metric['id'], False, 0.0, f"Column verification error: {e}"
            )
    
    def _verify_column_aggregation(self, metric: Dict[str, Any], column_data: pd.Series, column_name: str) -> Dict[str, Any]:
        """Verify if the reported value is an aggregation of column data."""
        try:
            reported_value = metric['metric_value']
            
            # Try common aggregations
            aggregations = {
                'sum': column_data.sum(),
                'mean': column_data.mean(),
                'count': len(column_data),
                'max': column_data.max(),
                'min': column_data.min()
            }
            
            for agg_type, agg_value in aggregations.items():
                if isinstance(agg_value, str):
                    agg_value = self._extract_numeric_value(agg_value)
                
                accuracy = self._calculate_value_accuracy(reported_value, agg_value)
                if accuracy >= (1.0 - self.verification_tolerance):
                    notes = f"Matches {agg_type} of column '{column_name}': reported={reported_value}, calculated={agg_value}"
                    return self._create_verification_result(
                        metric['id'], True, accuracy, notes, agg_value
                    )
            
            return self._create_verification_result(
                metric['id'], False, 0.0, f"Value {reported_value} not found in column '{column_name}' or its aggregations"
            )
            
        except Exception as e:
            return self._create_verification_result(
                metric['id'], False, 0.0, f"Aggregation verification error: {e}"
            )
    
    def _verify_by_value_search(self, metric: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback: search for the value anywhere in the dataframe."""
        try:
            reported_value = metric['metric_value']
            
            # Search through all numeric cells
            for col in df.columns:
                for idx, value in df[col].items():
                    if pd.isna(value):
                        continue
                    
                    if isinstance(value, str):
                        numeric_value = self._extract_numeric_value(value)
                    else:
                        numeric_value = value
                    
                    accuracy = self._calculate_value_accuracy(reported_value, numeric_value)
                    if accuracy >= (1.0 - self.verification_tolerance):
                        notes = f"Found at ({idx}, '{col}'): reported={reported_value}, actual={numeric_value}"
                        return self._create_verification_result(
                            metric['id'], True, accuracy, notes, numeric_value
                        )
            
            return self._create_verification_result(
                metric['id'], False, 0.0, f"Value {reported_value} not found anywhere in the file"
            )
            
        except Exception as e:
            return self._create_verification_result(
                metric['id'], False, 0.0, f"Value search error: {e}"
            )
    
    def _parse_cell_reference(self, cell_ref: str) -> Tuple[int, int]:
        """Parse Excel-style cell reference to column and row indices."""
        # Handle sheet references (e.g., 'Sheet1!B5')
        if '!' in cell_ref:
            cell_ref = cell_ref.split('!')[1]
        
        # Extract column letters and row number
        match = re.match(r'^([A-Z]+)(\d+)$', cell_ref.upper())
        if not match:
            raise ValueError(f"Invalid cell reference: {cell_ref}")
        
        col_letters, row_num = match.groups()
        
        # Convert column letters to index (A=0, B=1, ..., Z=25, AA=26, etc.)
        col_idx = 0
        for i, letter in enumerate(reversed(col_letters)):
            col_idx += (ord(letter) - ord('A') + 1) * (26 ** i)
        col_idx -= 1  # Convert to 0-based index
        
        row_idx = int(row_num) - 1  # Convert to 0-based index
        
        return col_idx, row_idx
    
    def _extract_numeric_value(self, text: str) -> float:
        """Extract numeric value from text string."""
        # Remove common currency symbols and formatting
        cleaned = re.sub(r'[£$€,\s]', '', str(text))
        
        # Try to extract number
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        if numbers:
            return float(numbers[0])
        
        return float('nan')
    
    def _calculate_value_accuracy(self, reported: float, actual: float) -> float:
        """Calculate accuracy between reported and actual values."""
        if pd.isna(actual) or pd.isna(reported):
            return 0.0
        
        if actual == 0:
            return 1.0 if reported == 0 else 0.0
        
        # Calculate relative accuracy
        relative_error = abs(reported - actual) / abs(actual)
        accuracy = max(0.0, 1.0 - relative_error)
        
        return accuracy
    
    def _create_verification_result(self, metric_id: int, is_verified: bool, 
                                   accuracy: float, notes: str, 
                                   actual_value: Optional[float] = None) -> Dict[str, Any]:
        """Create a verification result and update the database."""
        try:
            status = 'verified' if is_verified else 'failed'
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE impact_metrics 
                    SET verification_status = ?,
                        verification_timestamp = CURRENT_TIMESTAMP,
                        verified_value = ?,
                        verification_accuracy = ?,
                        verification_notes = ?
                    WHERE id = ?
                """, (status, actual_value, accuracy, notes, metric_id))
                conn.commit()
            
            result = {
                'metric_id': metric_id,
                'verified': is_verified,
                'accuracy': accuracy,
                'notes': notes,
                'status': status
            }
            
            if actual_value is not None:
                result['actual_value'] = actual_value
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating verification result: {e}")
            return {
                'metric_id': metric_id,
                'verified': False,
                'accuracy': 0.0,
                'notes': f"Database update error: {e}",
                'status': 'failed'
            }
    
    def _calculate_verification_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall verification statistics."""
        total = len(results)
        if total == 0:
            return {"total": 0, "verified": 0, "failed": 0, "accuracy": 1.0, "verification_rate": 1.0}
        
        verified = sum(1 for r in results if r['verified'])
        failed = total - verified
        
        # Calculate weighted accuracy
        total_accuracy = sum(r['accuracy'] for r in results)
        average_accuracy = total_accuracy / total
        
        return {
            "total": total,
            "verified": verified,
            "failed": failed,
            "accuracy": average_accuracy,
            "verification_rate": verified / total
        }


def verify_all_data(db_path: str = "db/impactos.db") -> Dict[str, Any]:
    """Convenience function to verify all pending metrics."""
    verifier = DataVerifier(db_path)
    return verifier.verify_all_pending_metrics()


def verify_metric(metric_id: int, db_path: str = "db/impactos.db") -> Dict[str, Any]:
    """Convenience function to verify a specific metric."""
    verifier = DataVerifier(db_path)
    return verifier.verify_metric_by_id(metric_id)


if __name__ == "__main__":
    # Test verification system
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        summary = verify_all_data()
        print(f"Verification Summary: {summary}")
    elif len(sys.argv) > 1:
        metric_id = int(sys.argv[1])
        result = verify_metric(metric_id)
        print(f"Verification Result: {result}")
    else:
        print("Usage: python verify.py [all|metric_id]")
        print("Example: python verify.py all")
        print("Example: python verify.py 5") 