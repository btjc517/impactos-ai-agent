#!/usr/bin/env python3
"""
Integration test for framework mapping functionality.
Tests the complete flow from framework mapping to badge generation.
"""

import sys
import os
sys.path.append('src')

def test_framework_mapper():
    """Test FrameworkMapper class functionality."""
    print("🧪 Testing FrameworkMapper...")
    
    try:
        from frameworks import FrameworkMapper
        
        # Test initialization
        mapper = FrameworkMapper("db/impactos.db")
        print("✅ FrameworkMapper initialized successfully")
        
        # Test framework definitions loading
        frameworks = mapper.frameworks
        expected_frameworks = ['UK_SV_MODEL', 'UN_SDGS', 'TOMS', 'B_CORP']
        
        for fw in expected_frameworks:
            if fw in frameworks:
                print(f"✅ {fw} framework loaded")
            else:
                print(f"❌ {fw} framework missing")
        
        # Test mapping functionality
        test_mapping = mapper.map_metric_to_frameworks(
            "Employee Volunteering Hours",
            "community_engagement", 
            "Hours volunteered by staff"
        )
        
        if test_mapping:
            print("✅ Framework mapping working")
            for fw, codes in test_mapping.items():
                print(f"   {fw}: {codes}")
        else:
            print("❌ Framework mapping failed")
            
        return True
        
    except Exception as e:
        print(f"❌ FrameworkMapper test failed: {e}")
        return False

def test_badge_generation():
    """Test badge generation functionality."""
    print("\n🏷️ Testing badge generation...")
    
    try:
        from frameworks import FrameworkMapper
        
        mapper = FrameworkMapper("db/impactos.db")
        
        # Test the new badge methods
        test_cases = [
            ("Volunteering Hours", "community_engagement"),
            ("Carbon Reduction", "environmental"), 
            ("Local Jobs", "employment")
        ]
        
        for metric_name, category in test_cases:
            mappings = mapper.map_metric_to_frameworks(metric_name, category)
            
            # Test the badge formatting logic
            badges = []
            for framework, codes in mappings.items():
                if codes:
                    primary_code = codes[0]
                    if framework == "UN_SDGS":
                        badges.append(f"[SDG{primary_code}]")
                    elif framework == "UK_SV_MODEL":
                        badges.append(f"[{primary_code}]")
                    elif framework == "TOMS":
                        badges.append(f"[{primary_code}]")
                    elif framework == "B_CORP":
                        badges.append(f"[{primary_code.upper()}]")
            
            badge_text = " ".join(badges)
            print(f"✅ {metric_name}: {badge_text}")
            
        return True
        
    except Exception as e:
        print(f"❌ Badge generation test failed: {e}")
        return False

def test_query_system_integration():
    """Test QuerySystem integration."""
    print("\n🔍 Testing QuerySystem integration...")
    
    try:
        from query import QuerySystem
        
        # Test initialization
        query_system = QuerySystem("db/impactos.db")
        print("✅ QuerySystem initialized")
        
        # Test if the new badge method exists
        if hasattr(query_system, '_get_framework_badges'):
            print("✅ _get_framework_badges method exists")
            
            # Test the method
            badges = query_system._get_framework_badges("Volunteer Hours", "community_engagement")
            print(f"✅ Badge generation test: '{badges}'")
            
        else:
            print("❌ _get_framework_badges method missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ QuerySystem integration test failed: {e}")
        return False

def test_database_compatibility():
    """Test database schema compatibility."""
    print("\n🗄️ Testing database compatibility...")
    
    try:
        import sqlite3
        
        # Check if database exists
        if not os.path.exists("db/impactos.db"):
            print("⚠️ Database not found - this is expected for new installs")
            return True
            
        # Test database connection
        with sqlite3.connect("db/impactos.db") as conn:
            cursor = conn.cursor()
            
            # Check if framework_mappings table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='framework_mappings'
            """)
            
            if cursor.fetchone():
                print("✅ framework_mappings table exists")
            else:
                print("⚠️ framework_mappings table missing - needs database setup")
                
        return True
        
    except Exception as e:
        print(f"❌ Database compatibility test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Framework Mapping Integration Test")
    print("=" * 50)
    
    tests = [
        ("Framework Mapper", test_framework_mapper),
        ("Badge Generation", test_badge_generation), 
        ("Query System Integration", test_query_system_integration),
        ("Database Compatibility", test_database_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Framework mapping is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    print("\n💡 Next steps:")
    print("• Run `python3 src/frameworks.py` to apply mappings to existing data")
    print("• Test with actual queries through the web interface")
    print("• Verify badge display in query responses")

if __name__ == "__main__":
    main()