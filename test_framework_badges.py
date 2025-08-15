#!/usr/bin/env python3
"""
Quick test script for framework badge display functionality.
"""

import sys
import os
sys.path.append('src')

from frameworks import FrameworkMapper

def test_framework_badges():
    """Test the new framework badge functionality."""
    print("🏷️  Framework Badge Display Test")
    print("=" * 40)
    
    mapper = FrameworkMapper()
    
    # Test cases with different types of metrics
    test_cases = [
        {
            "name": "Employee Volunteering Hours",
            "category": "community_engagement",
            "context": "Total hours volunteered by employees in local community projects"
        },
        {
            "name": "Carbon Emissions Reduced",
            "category": "environmental",
            "context": "CO2 emissions reduction through energy efficiency measures"
        },
        {
            "name": "Local Jobs Created",
            "category": "employment",
            "context": "Number of jobs for local residents in community development"
        },
        {
            "name": "Skills Training Programs",
            "category": "education",
            "context": "Professional development and skills training for workforce"
        },
        {
            "name": "Charitable Donations",
            "category": "community_engagement",
            "context": "Corporate charitable giving to local causes"
        }
    ]
    
    for case in test_cases:
        print(f"\n📊 Testing: {case['name']}")
        print(f"   Category: {case['category']}")
        
        # Get framework mappings
        mappings = mapper.map_metric_to_frameworks(
            case['name'], 
            case['category'], 
            case['context']
        )
        
        if not mappings:
            print("   ❌ No framework mappings found")
            continue
        
        print("   🎯 Framework Mappings:")
        for framework, codes in mappings.items():
            framework_def = mapper.frameworks.get(framework, {})
            framework_name = framework_def.get('name', framework)
            print(f"      • {framework_name}: {codes}")
        
        # Test compact badge display
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
        
        badge_display = " ".join(badges) if badges else ""
        print(f"   🏷️  Compact badges: {badge_display}")
        
        # Example query response format
        if badge_display:
            print(f"   📝 Response format: {badge_display} {case['category'].title()}: {case['name']} = [VALUE] [UNIT]")
    
    print(f"\n✅ Framework badge testing completed!")
    print("\n💡 Quick Win Implementation:")
    print("• Compact badges automatically show primary framework alignment")
    print("• Only shows most relevant mapping per framework (no clutter)")
    print("• Easy to spot framework compliance at a glance")
    print("• Works with existing system instructions for detailed framework info")

if __name__ == "__main__":
    test_framework_badges()