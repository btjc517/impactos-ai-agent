import pandas as pd

# Create test data
data = {
    'Metric': [
        'Carbon Emissions Reduced',
        'Water Conservation',
        'Waste Diverted',
        'Renewable Energy',
        'Volunteer Hours',
        'Community Members Reached',
        'Employee Training Hours',
        'Charitable Donations'
    ],
    'Value': [1500, 50000, 85, 35, 2500, 12000, 450, 75000],
    'Unit': ['tons CO2', 'gallons', 'percentage', 'percentage', 'hours', 'people', 'hours', 'USD'],
    'Category': ['Environmental', 'Environmental', 'Environmental', 'Environmental', 
                 'Social', 'Social', 'Governance', 'Social'],
    'Quarter': ['Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024', 
                'Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024']
}

# Create DataFrame
df = pd.DataFrame(data)

# Create Excel file with multiple sheets
with pd.ExcelWriter('test_impact_data.xlsx') as writer:
    # Main data sheet
    df.to_excel(writer, sheet_name='Impact Metrics', index=False)
    
    # Summary by category
    summary = df.groupby('Category')['Value'].count().reset_index()
    summary.columns = ['Category', 'Metric Count']
    summary.to_excel(writer, sheet_name='Summary', index=False)
    
    # Environmental metrics only
    env_df = df[df['Category'] == 'Environmental']
    env_df.to_excel(writer, sheet_name='Environmental', index=False)

print("Created test_impact_data.xlsx with 3 sheets")