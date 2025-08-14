#!/usr/bin/env python3
"""Check vector stores implementation based on OpenAI docs."""

import os
import sys
sys.path.append('src')

from utils import get_config
from openai import OpenAI

def explore_beta_apis():
    """Explore what's available in the beta APIs."""
    config = get_config()
    client = OpenAI(api_key=config.get_openai_key())
    
    print("=== Beta API Exploration ===\n")
    
    # Check what's in beta
    beta_client = client.beta
    print(f"Beta client type: {type(beta_client)}")
    
    # List all attributes of beta client
    beta_attrs = [attr for attr in dir(beta_client) if not attr.startswith('_')]
    print(f"Beta attributes: {beta_attrs}")
    
    print("\n=== Detailed Attribute Check ===")
    for attr in beta_attrs:
        try:
            obj = getattr(beta_client, attr)
            print(f"✓ {attr}: {type(obj)}")
        except Exception as e:
            print(f"✗ {attr}: Error - {e}")
    
    # Try different ways to access vector stores
    print("\n=== Vector Stores Access Attempts ===")
    
    # Method 1: Direct attribute
    try:
        vs = beta_client.vector_stores
        print(f"✓ Method 1 (direct): {type(vs)}")
    except AttributeError as e:
        print(f"✗ Method 1 (direct): {e}")
    
    # Method 2: Check if assistants have vector store methods
    try:
        assistants = beta_client.assistants
        assistant_attrs = [attr for attr in dir(assistants) if not attr.startswith('_')]
        print(f"Assistant methods: {assistant_attrs}")
        
        # Try to create a simple assistant to see available tools
        test_assistant = beta_client.assistants.list(limit=1)
        if test_assistant.data:
            assistant = test_assistant.data[0]
            print(f"Sample assistant tools: {assistant.tools}")
            print(f"Assistant model: {assistant.model}")
    except Exception as e:
        print(f"✗ Assistant exploration error: {e}")
    
    # Method 3: Try to create an assistant with file_search
    print("\n=== File Search Tool Test ===")
    try:
        # Try to create an assistant with file_search tool
        test_assistant = beta_client.assistants.create(
            name="Test File Search Assistant",
            instructions="You are a test assistant with file search.",
            model="gpt-4-turbo",
            tools=[{"type": "file_search"}]
        )
        print(f"✓ Created assistant with file_search: {test_assistant.id}")
        
        # Clean up
        beta_client.assistants.delete(test_assistant.id)
        print("✓ Cleaned up test assistant")
        
    except Exception as e:
        print(f"✗ File search assistant creation error: {e}")
    
    # Method 4: Check thread creation with attachments
    print("\n=== Thread Attachments Test ===")
    try:
        # Try to create a thread with file attachments
        thread = beta_client.threads.create()
        print(f"✓ Created thread: {thread.id}")
        
        # Try to add a message with file attachment
        # First check if we have any files
        files = client.files.list(limit=1)
        if files.data:
            file_id = files.data[0].id
            try:
                message = beta_client.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content="Test message",
                    attachments=[{"file_id": file_id, "tools": [{"type": "file_search"}]}]
                )
                print(f"✓ Created message with file attachment: {message.id}")
            except Exception as e:
                print(f"✗ Message with attachment error: {e}")
        
        # Clean up thread
        beta_client.threads.delete(thread.id)
        print("✓ Cleaned up test thread")
        
    except Exception as e:
        print(f"✗ Thread test error: {e}")

if __name__ == "__main__":
    explore_beta_apis()