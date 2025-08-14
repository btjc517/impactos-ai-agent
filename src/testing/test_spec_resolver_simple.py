#!/usr/bin/env python3
"""
Simple test of the spec resolver to validate core functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spec_resolver import resolve_specs_for_sheet, FactsConfig


def test_basic_spec_resolution():
    """Test basic spec resolution functionality."""
    print("=== Testing Basic Spec Resolution ===")
    
    # Test with Benevity-like data
    benevity_data = {
        'sheet_name': 'Benevity Data',
        'bronze_table': 'bronze_benevity_test',
        'columns': [
            'Employee ID', 'Full Name', 'Year', 'Total Donations (£)', 
            'Total Volunteering Hours', 'Number of Campaigns Joined', 
            'Most Supported Cause', 'Matching Contributions (£)', 
            'Region', 'Date of Last Activity'
        ],
        'example_values': {
            'Employee ID': ['EMP001', 'EMP002', 'EMP003'],
            'Total Donations (£)': ['471.10', '156.97', '370.96'],
            'Total Volunteering Hours': ['19', '47', '38'],
            'Date of Last Activity': ['2021-06-25', '2022-11-03', '2023-06-24'],
            'Year': ['2021', '2022', '2023']
        }
    }
    
    specs = resolve_specs_for_sheet('test_tenant', benevity_data)
    
    print(f"Generated {len(specs)} specs:")
    for spec in specs:
        fact = spec['spec_snapshot']['fact']
        mappings = spec['spec_snapshot']['mappings']
        print(f"  - {fact}: {len(mappings)} mappings")
        for role, header in mappings.items():
            print(f"    {role} -> {header}")
    
    # Should generate at least one spec
    assert len(specs) > 0, f"Should generate at least one spec, got {len(specs)}"
    
    # Check that specs have required structure
    for spec in specs:
        assert 'spec_id' in spec
        assert 'spec_snapshot' in spec
        assert 'spec_hash' in spec
        
        snapshot = spec['spec_snapshot']
        assert 'fact' in snapshot
        assert 'mappings' in snapshot
        assert 'base_currency' in snapshot
        
        # Ensure all mapped headers exist in original data
        mappings = snapshot['mappings']
        available_headers = set(benevity_data['columns'])
        
        for role, header in mappings.items():
            if isinstance(header, str) and header not in ['GBP', 'hours', 'USD', 'EUR']:
                assert header in available_headers, f"Header '{header}' for role '{role}' not found in available headers: {available_headers}"
    
    print("✓ Basic spec resolution works correctly")
    return True


def test_facts_config():
    """Test facts configuration loading."""
    print("=== Testing Facts Config ===")
    
    facts_config = FactsConfig()
    facts = facts_config.load_facts()
    
    assert len(facts) > 0, "Should load some facts"
    
    # Check basic structure
    for fact_key, fact_def in facts.items():
        assert 'required' in fact_def
        assert 'optional' in fact_def
        assert 'role_hints' in fact_def
        assert 'description' in fact_def
    
    print(f"✓ Loaded {len(facts)} fact definitions")
    return True


def test_empty_data():
    """Test with empty/invalid data."""
    print("=== Testing Edge Cases ===")
    
    # Empty data
    empty_specs = resolve_specs_for_sheet('test_tenant', {
        'sheet_name': 'Empty',
        'columns': [],
        'example_values': {}
    })
    assert len(empty_specs) == 0, "Should return no specs for empty data"
    
    # Unrecognized data
    random_specs = resolve_specs_for_sheet('test_tenant', {
        'sheet_name': 'Random',
        'columns': ['Random1', 'Random2'],
        'example_values': {
            'Random1': ['a', 'b', 'c'],
            'Random2': ['x', 'y', 'z']
        }
    })
    # Should return few or no specs
    assert len(random_specs) <= 1, "Should return few/no specs for unrecognized data"
    
    print("✓ Edge cases handled correctly")
    return True


def main():
    """Run all simple tests."""
    print("Starting simple spec resolver tests...\n")
    
    try:
        test_facts_config()
        test_basic_spec_resolution() 
        test_empty_data()
        
        print("\n🎉 All simple tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
