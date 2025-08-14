#!/usr/bin/env python3
"""Debug script to check OpenAI API capabilities."""

import os
import sys
sys.path.append('src')

from utils import get_config
from openai import OpenAI

def check_openai_capabilities():
    """Check what OpenAI API features are available."""
    config = get_config()
    client = OpenAI(api_key=config.get_openai_key())
    
    print("=== OpenAI API Capability Check ===\n")
    
    # Check available models
    try:
        models = client.models.list()
        gpt4_models = [m for m in models.data if 'gpt-4' in m.id]
        print(f"✓ GPT-4 models available: {len(gpt4_models)}")
        for model in gpt4_models[:3]:
            print(f"  - {model.id}")
        if len(gpt4_models) > 3:
            print(f"  ... and {len(gpt4_models) - 3} more")
    except Exception as e:
        print(f"✗ Error checking models: {e}")
    
    print()
    
    # Check Assistants API
    try:
        assistants = client.beta.assistants.list(limit=1)
        print(f"✓ Assistants API available: {len(assistants.data)} assistants found")
    except Exception as e:
        print(f"✗ Assistants API error: {e}")
    
    # Check Vector Stores API
    try:
        if hasattr(client.beta, 'vector_stores'):
            vector_stores = client.beta.vector_stores.list(limit=1)
            print(f"✓ Vector Stores API available: {len(vector_stores.data)} stores found")
        else:
            print("✗ Vector Stores API not available (attribute missing)")
    except Exception as e:
        print(f"✗ Vector Stores API error: {e}")
    
    # Check File API
    try:
        files = client.files.list()
        print(f"✓ Files API available: {len(files.data)} files found")
    except Exception as e:
        print(f"✗ Files API error: {e}")
    
    print("\n=== API Access Analysis ===")
    
    # Check if we're on a free/limited tier
    try:
        # Try to create a simple completion to check rate limits
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1
        )
        print("✓ Chat completions working")
    except Exception as e:
        print(f"✗ Chat completions error: {e}")
    
    # Check if vector stores are actually accessible
    print("\n=== Vector Stores Detailed Check ===")
    try:
        # Try to access the vector_stores attribute directly
        vs_client = client.beta.vector_stores
        print(f"✓ Vector stores client object: {type(vs_client)}")
        
        # Try to list vector stores
        stores = vs_client.list()
        print(f"✓ Vector stores list successful: {len(stores.data)} stores")
        
    except AttributeError as e:
        print(f"✗ Vector stores not available - attribute error: {e}")
        print("  This suggests the OpenAI library version may not support vector stores")
        print("  or the API account doesn't have access to this feature")
    except Exception as e:
        print(f"✗ Vector stores error: {e}")
        print("  This could indicate API access limitations or billing issues")
    
    # Check OpenAI library version
    try:
        import openai
        print(f"\nOpenAI library version: {openai.__version__}")
    except Exception as e:
        print(f"Error getting OpenAI version: {e}")

if __name__ == "__main__":
    check_openai_capabilities()