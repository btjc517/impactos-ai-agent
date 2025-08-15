#!/usr/bin/env python3
"""
Test script to debug chart data parsing.
"""

import re

# Sample response from GPT
sample_response = """
Person 1 in 2023 donated exactly £370.96, logged 38 volunteering hours
Person 2 has multiple records: in 2021 they donated £178.09; in 2022 they donated £362.66; 
and in 2023, the record shows £392.10 donated with 25 volunteering hours
Person 3's 2023 entry indicates a donation of £105.59, 9 volunteering hours
Person 4's 2023 record cites a donation of £133.40
Person 5 in 2023 is recorded with £429.91 in donations, 3 volunteering hours
"""

def test_donation_parsing():
    print("Testing donation data parsing...")
    
    # Test donation pattern
    donation_pattern = r'£(\d+(?:\.\d+)?)'
    donation_matches = re.findall(donation_pattern, sample_response)
    print(f"Donation matches: {donation_matches}")
    
    # Test year + donation pattern  
    year_donation_pattern = r'(\d{4}).*?£(\d+(?:\.\d+)?)'
    year_matches = re.findall(year_donation_pattern, sample_response)
    print(f"Year+donation matches: {year_matches}")
    
    # Test the full logic
    if year_matches:
        # Group by year and sum donations
        year_totals = {}
        for year, amount in year_matches:
            year_totals[year] = year_totals.get(year, 0) + float(amount)
        
        print(f"Year totals: {year_totals}")
        
        data_rows = []
        for year, total in sorted(year_totals.items()):
            data_rows.append({
                'label': year,
                'value': total
            })
        
        print(f"Data rows: {data_rows}")
    
    # Test detection conditions
    content = sample_response.lower()
    donation_detected = ('donation' in content or 'contribution' in content or 
                        '£' in sample_response or 'giving' in content)
    print(f"Donation detected: {donation_detected}")

if __name__ == "__main__":
    test_donation_parsing()