#!/usr/bin/env python3
"""Debug script to see what GPT is actually returning."""

import os
import sys
sys.path.append('src')

from gpt_tools import AssistantManager, FileManager
from utils import get_config, setup_logging

setup_logging()

def debug_gpt_extraction():
    """Test GPT extraction and see the raw response."""
    config = get_config()
    
    # Initialize managers
    assistant_manager = AssistantManager(config.get_openai_key())
    file_manager = FileManager(config.get_openai_key())
    
    # Upload a simple test file first
    file_path = "data/TakingCare_Payroll_Synthetic_Data.xlsx"
    print(f"Uploading {file_path}...")
    
    try:
        file_id = file_manager.upload_file(file_path)
        print(f"File uploaded with ID: {file_id}")
        
        # Simple extraction instruction
        instructions = """Please analyze this file and extract any salary, compensation, or payroll data you find.

Look for:
- Salary amounts
- Bonus payments  
- Employee counts
- Gender information
- Department data

Return your findings in this exact JSON format:
```json
{
  "metrics": [
    {
      "name": "example metric name",
      "value": "123.45",
      "unit": "USD", 
      "category": "Social",
      "period": "2024"
    }
  ],
  "summary": "Brief description of data found"
}
```

Be sure to include the actual numbers and values you see in the data."""

        print("Processing with assistant...")
        result = assistant_manager.process_data([file_id], instructions)
        
        print("\n=== RAW RESULT ===")
        import json
        print(json.dumps(result, indent=2))
        
        print("\n=== CONTENT ===")
        print(result.get('content', 'No content'))
        
        print("\n=== STRUCTURED DATA ===")
        print(result.get('structured_data', 'No structured data'))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gpt_extraction()