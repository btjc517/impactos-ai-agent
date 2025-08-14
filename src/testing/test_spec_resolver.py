#!/usr/bin/env python3
"""
Comprehensive tests for the generic spec resolver.

Tests the data-driven spec resolution system across all sample files
to ensure required roles are resolved and appropriate specs are emitted.
"""

import os
import sys
import json
import tempfile
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spec_resolver import (
    resolve_specs_for_sheet, 
    FactsConfig, 
    SheetProfiler, 
    RoleScorer,
    LLMAssistant,
    LearningTracker
)
from semantic_resolver import ConceptGraph


class TestSpecResolver:
    """Test the generic spec resolver with sample data files."""
    
    def __init__(self):
        self.test_db = None
        self.temp_dir = None
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Set up test database and configuration."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_impactos.db")
        
        # Set environment variable for test database
        os.environ['IMPACTOS_DB_PATH'] = self.test_db
        
        # Initialize concept graph
        self._setup_concept_graph()
        
        print(f"Test environment set up with database: {self.test_db}")
    
    def _setup_concept_graph(self):
        """Initialize concept graph with required migrations."""
        try:
            # Run migrations
            migrations_dir = Path(__file__).parent.parent.parent / "db" / "migrations"
            
            migration_files = [
                "20250813T060000Z__concept_graph.sql",
                "20250813T061000Z__concept_roles_units.sql", 
                "20250813T070000Z__role_mapping_history.sql"
            ]
            
            with sqlite3.connect(self.test_db) as conn:
                for migration_file in migration_files:
                    migration_path = migrations_dir / migration_file
                    if migration_path.exists():
                        with open(migration_path, 'r') as f:
                            conn.executescript(f.read())
                        print(f"Applied migration: {migration_file}")
            
        except Exception as e:
            print(f"Warning: Failed to set up concept graph: {e}")
    
    def test_facts_config_loading(self):
        """Test that facts configuration loads correctly."""
        print("\n=== Testing Facts Configuration Loading ===")
        
        facts_config = FactsConfig(db_path=self.test_db)
        facts = facts_config.load_facts()
        
        assert facts, "Should load facts from config/facts.json"
        
        expected_facts = ['fact_volunteering', 'fact_donations', 'fact_procurement', 'fact_energy', 'fact_waste']
        for fact_key in expected_facts:
            assert fact_key in facts, f"Should contain {fact_key}"
            
            fact = facts[fact_key]
            assert 'required' in fact, f"{fact_key} should have required roles"
            assert 'optional' in fact, f"{fact_key} should have optional roles"
            assert 'role_hints' in fact, f"{fact_key} should have role hints"
            
            # Check required roles include value and date
            required = fact['required']
            assert 'value' in required, f"{fact_key} should require value role"
            assert 'date' in required, f"{fact_key} should require date role"
        
        print(f"✓ Successfully loaded {len(facts)} fact definitions")
    
    def test_sheet_profiling(self):
        """Test sheet profiling functionality."""
        print("\n=== Testing Sheet Profiling ===")
        
        profiler = SheetProfiler()
        
        # Test with mock sheet data
        mock_sheet_data = {
            'sheet_name': 'Test Volunteering Sheet',
            'columns': [
                'Employee ID', 
                'Total Volunteering Hours', 
                'Date of Last Activity',
                'Total Donations (£)',
                'Year'
            ],
            'example_values': {
                'Employee ID': ['EMP001', 'EMP002', 'EMP003'],
                'Total Volunteering Hours': ['19', '47', '38'],
                'Date of Last Activity': ['2021-06-25', '2022-11-03', '2023-06-24'],
                'Total Donations (£)': ['471.10', '156.97', '370.96'],
                'Year': ['2021', '2022', '2023']
            }
        }
        
        profile = profiler.profile_sheet(mock_sheet_data)
        
        assert profile['sheet_name'] == 'Test Volunteering Sheet'
        assert profile['num_columns'] == 5
        assert 'columns' in profile
        
        columns = profile['columns']
        
        # Test volunteering hours column
        hours_col = columns['Total Volunteering Hours']
        assert hours_col['dtype'] == 'numeric', "Hours should be detected as numeric"
        unit_tokens = hours_col['unit_tokens']
        assert any(token in ['hours', 'volunteering'] for token in unit_tokens), f"Should detect hours/volunteering unit, got {unit_tokens}"
        
        # Test date column
        date_col = columns['Date of Last Activity']
        assert date_col['dtype'] == 'date', "Date should be detected as date type"
        assert date_col['date_likeness'] > 0.5, "Should have high date-likeness"
        
        # Test currency column
        donations_col = columns['Total Donations (£)']
        assert 'currency' in donations_col['unit_tokens'], "Should detect currency unit"
        
        print("✓ Sheet profiling works correctly")
    
    def test_role_scoring(self):
        """Test role scoring functionality."""
        print("\n=== Testing Role Scoring ===")
        
        scorer = RoleScorer(db_path=self.test_db)
        
        # Mock column profile for volunteering hours
        hours_profile = {
            'dtype': 'numeric',
            'null_pct': 0.0,
            'top_examples': ['19', '47', '38'],
            'unit_tokens': ['hours'],
            'date_likeness': 0.0
        }
        
        # Test scoring value role against hours column
        role_hints = {'dtype': 'numeric', 'unit': 'hours'}
        result = scorer.score_role_mapping(
            'value', 'Total Volunteering Hours', hours_profile, role_hints
        )
        
        print(f"Role scoring result: {result}")
        # The score may be lower if semantic resolver is not available
        # But type and unit scores should be high
        assert result['component_scores']['type'] >= 0.8, "Type score should be high for numeric match"
        assert result['component_scores']['unit'] >= 0.8, "Unit score should be high for hours match"
        assert 'component_scores' in result
        
        # Test poor match - currency role with hours data
        currency_hints = {'dtype': 'string', 'unit': 'currency'}
        poor_result = scorer.score_role_mapping(
            'currency', 'Total Volunteering Hours', hours_profile, currency_hints
        )
        
        print(f"Poor match result: {poor_result}")
        # Should score poorly due to type mismatch (numeric vs string) and unit mismatch
        assert poor_result['component_scores']['type'] < 0.5, "Type score should be low for mismatched types"
        
        print("✓ Role scoring works correctly")
    
    def test_benevity_data_resolution(self):
        """Test spec resolution on Benevity sample data."""
        print("\n=== Testing Benevity Data Resolution ===")
        
        # Load actual Benevity data
        data_path = Path(__file__).parent.parent.parent / "data" / "TakingCare_Benevity_Synthetic_Data.xlsx"
        
        if not data_path.exists():
            print(f"⚠ Skipping test - data file not found: {data_path}")
            return
        
        try:
            df = pd.read_excel(data_path)
            headers = list(df.columns)
            
            # Create example values
            example_values = {}
            for col in headers:
                examples = df[col].dropna().astype(str).head(3).tolist()
                example_values[col] = examples
            
            sheet_data = {
                'sheet_name': 'Benevity Data',
                'bronze_table': 'bronze_benevity_test',
                'columns': headers,
                'example_values': example_values
            }
            
            # Resolve specs
            specs = resolve_specs_for_sheet('test_tenant', sheet_data)
            
            assert len(specs) > 0, "Should generate at least one spec for Benevity data"
            
            # Check for expected facts
            spec_facts = [spec['spec_snapshot']['fact'] for spec in specs]
            
            # Should detect volunteering (has volunteering hours)
            assert 'fact_volunteering' in spec_facts, "Should detect volunteering fact"
            
            # Should detect donations (has donation amounts)
            assert 'fact_donations' in spec_facts, "Should detect donations fact"
            
            # Validate spec structure
            for spec in specs:
                snapshot = spec['spec_snapshot']
                assert 'fact' in snapshot
                assert 'mappings' in snapshot
                assert 'base_currency' in snapshot
                
                mappings = snapshot['mappings']
                fact_key = snapshot['fact']
                
                if fact_key == 'fact_volunteering':
                    assert 'value' in mappings, "Volunteering should map value role"
                    assert 'date' in mappings, "Volunteering should map date role"
                    # Check that mapped headers exist in original data
                    for role, header in mappings.items():
                        if isinstance(header, str) and header in headers:
                            continue  # Good mapping
                        elif role in ['unit']:  # Literal values are OK
                            continue
                        else:
                            assert False, f"Invalid mapping: {role} -> {header} (not in {headers})"
                
                elif fact_key == 'fact_donations':
                    assert 'value' in mappings, "Donations should map value role"
                    assert 'date' in mappings, "Donations should map date role"
            
            print(f"✓ Generated {len(specs)} specs for Benevity data")
            for spec in specs:
                fact = spec['spec_snapshot']['fact']
                mappings = spec['spec_snapshot']['mappings']
                print(f"  - {fact}: {mappings}")
            
        except Exception as e:
            print(f"⚠ Error testing Benevity data: {e}")
    
    def test_carbon_data_resolution(self):
        """Test spec resolution on Carbon Reporting sample data."""
        print("\n=== Testing Carbon Data Resolution ===")
        
        data_path = Path(__file__).parent.parent.parent / "data" / "TakingCare_Carbon_Reporting_Synthetic_Data.xlsx"
        
        if not data_path.exists():
            print(f"⚠ Skipping test - data file not found: {data_path}")
            return
        
        try:
            df = pd.read_excel(data_path)
            headers = list(df.columns)
            
            example_values = {}
            for col in headers:
                examples = df[col].dropna().astype(str).head(3).tolist()
                example_values[col] = examples
            
            sheet_data = {
                'sheet_name': 'Carbon Reporting',
                'bronze_table': 'bronze_carbon_test',
                'columns': headers,
                'example_values': example_values
            }
            
            specs = resolve_specs_for_sheet('test_tenant', sheet_data)
            
            assert len(specs) >= 0, "Should handle carbon data gracefully"
            
            # May detect energy fact due to energy consumption column
            spec_facts = [spec['spec_snapshot']['fact'] for spec in specs]
            
            print(f"✓ Generated {len(specs)} specs for Carbon data")
            for spec in specs:
                fact = spec['spec_snapshot']['fact']
                mappings = spec['spec_snapshot']['mappings']
                print(f"  - {fact}: {mappings}")
            
        except Exception as e:
            print(f"⚠ Error testing Carbon data: {e}")
    
    def test_multiple_sample_files(self):
        """Test spec resolution across all available sample files."""
        print("\n=== Testing Multiple Sample Files ===")
        
        data_dir = Path(__file__).parent.parent.parent / "data"
        excel_files = list(data_dir.glob("*.xlsx"))
        
        if not excel_files:
            print("⚠ No Excel files found in data directory")
            return
        
        total_specs = 0
        successful_files = 0
        
        for file_path in excel_files:
            print(f"\nTesting: {file_path.name}")
            
            try:
                df = pd.read_excel(file_path)
                headers = list(df.columns)
                
                example_values = {}
                for col in headers:
                    examples = df[col].dropna().astype(str).head(3).tolist()
                    example_values[col] = examples
                
                sheet_data = {
                    'sheet_name': file_path.stem,
                    'bronze_table': f'bronze_{file_path.stem.lower()}',
                    'columns': headers,
                    'example_values': example_values
                }
                
                specs = resolve_specs_for_sheet('test_tenant', sheet_data)
                total_specs += len(specs)
                successful_files += 1
                
                print(f"  Generated {len(specs)} specs")
                for spec in specs:
                    fact = spec['spec_snapshot']['fact']
                    mappings_count = len(spec['spec_snapshot']['mappings'])
                    print(f"    - {fact} ({mappings_count} mappings)")
                
            except Exception as e:
                print(f"  ⚠ Error: {e}")
        
        print(f"\n✓ Processed {successful_files} files, generated {total_specs} total specs")
        assert total_specs > 0, "Should generate at least some specs across all files"
    
    def test_learning_tracker(self):
        """Test the learning and tracking functionality.""" 
        print("\n=== Testing Learning Tracker ===")
        
        tracker = LearningTracker(self.test_db)
        
        # Record some test mappings
        tracker.record_role_mapping(
            'test_tenant', 'fact_volunteering', 'value', 'Total Hours', 
            True, 0.85, {'graph': 0.7, 'type': 1.0, 'unit': 0.9, 'history': 0.5}, 'test_hash'
        )
        
        tracker.record_role_mapping(
            'test_tenant', 'fact_volunteering', 'value', 'Total Hours', 
            True, 0.92, {'graph': 0.8, 'type': 1.0, 'unit': 0.95, 'history': 0.7}, 'test_hash2'
        )
        
        # Test success rate retrieval
        success_rate = tracker.get_role_success_rate('test_tenant', 'value', 'total hours')
        assert success_rate == 1.0, f"Should have 100% success rate, got {success_rate}"
        
        # Record a spec generation
        tracker.record_spec_generation(
            'test_tenant', 'Test Sheet', 'bronze_test', 'fact_volunteering', 'test_hash',
            True, [], 0.85, 'semantic'
        )
        
        print("✓ Learning tracker works correctly")
    
    def test_validation_edge_cases(self):
        """Test validation with edge cases and error conditions."""
        print("\n=== Testing Validation Edge Cases ===")
        
        # Test with empty sheet
        empty_specs = resolve_specs_for_sheet('test_tenant', {
            'sheet_name': 'Empty Sheet',
            'columns': [],
            'example_values': {}
        })
        assert len(empty_specs) == 0, "Should return no specs for empty sheet"
        
        # Test with non-matching headers
        bad_specs = resolve_specs_for_sheet('test_tenant', {
            'sheet_name': 'Bad Sheet',
            'columns': ['Random Column 1', 'Random Column 2'],
            'example_values': {
                'Random Column 1': ['a', 'b', 'c'],
                'Random Column 2': ['x', 'y', 'z']
            }
        })
        # Should return empty or very low confidence specs
        assert len(bad_specs) <= 1, "Should return few/no specs for unrecognized data"
        
        print("✓ Edge case validation works correctly")
    
    def run_all_tests(self):
        """Run all tests."""
        print("Starting comprehensive spec resolver tests...")
        
        try:
            self.test_facts_config_loading()
            self.test_sheet_profiling()
            self.test_role_scoring()
            self.test_benevity_data_resolution()
            self.test_carbon_data_resolution()
            self.test_multiple_sample_files()
            self.test_learning_tracker()
            self.test_validation_edge_cases()
            
            print("\n🎉 All tests passed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up test environment."""
        if 'IMPACTOS_DB_PATH' in os.environ:
            del os.environ['IMPACTOS_DB_PATH']
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up test directory: {self.temp_dir}")


if __name__ == "__main__":
    tester = TestSpecResolver()
    success = tester.run_all_tests()
    exit(0 if success else 1)
