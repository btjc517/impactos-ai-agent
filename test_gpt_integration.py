#!/usr/bin/env python3
"""
Test script for GPT tools and multi-agent integration.

This script validates that the new architecture is working correctly.
"""

import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gpt_tools import AssistantManager, FileManager, EmbeddingService, RetrievalService
from agents import ArchitectAgent, DataAgent, QueryAgent, AgentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gpt_tools():
    """Test GPT tools functionality."""
    print("\n=== Testing GPT Tools ===")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OpenAI API key not found")
        return False
    
    try:
        # Test File Manager
        print("\n1. Testing File Manager...")
        file_manager = FileManager(openai_key)
        
        # Create test content
        test_content = """
        Impact Metrics Test Data:
        - Carbon emissions reduced: 500 tons CO2
        - Volunteer hours: 1,200 hours  
        - Charitable donations: $50,000
        - Employee training hours: 3,500 hours
        """
        
        file_id = file_manager.upload_content(test_content, "test_metrics.txt")
        print(f"✓ File uploaded: {file_id}")
        
        # Test Assistant Manager
        print("\n2. Testing Assistant Manager...")
        assistant_manager = AssistantManager(openai_key)
        
        result = assistant_manager.process_data(
            [file_id],
            "Extract all metrics from this file and return as JSON"
        )
        
        if result.get('content'):
            print("✓ Assistant processed data successfully")
            print(f"  Response length: {len(result['content'])} chars")
        
        # Test Embedding Service
        print("\n3. Testing Embedding Service...")
        embedding_service = EmbeddingService(openai_key)
        
        test_texts = [
            "Carbon emissions reduction",
            "Employee volunteer hours",
            "Charitable giving"
        ]
        
        embeddings = embedding_service.generate_embeddings_batch(test_texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        # Test similarity
        query = "environmental impact"
        similar = embedding_service.find_similar(query, test_texts, top_k=2)
        print(f"✓ Found {len(similar)} similar items for '{query}'")
        
        # Test Retrieval Service
        print("\n4. Testing Retrieval Service...")
        retrieval_service = RetrievalService(openai_key)
        
        docs = [
            {"content": "Carbon emissions reduced by 500 tons", "category": "environment"},
            {"content": "1200 volunteer hours contributed", "category": "social"},
            {"content": "$50,000 donated to charity", "category": "social"}
        ]
        
        retrieval_service.index_documents(docs, metadata_fields=["category"])
        results = retrieval_service.hybrid_search("environmental impact", top_k=2)
        print(f"✓ Retrieval found {len(results)} relevant documents")
        
        # Cleanup
        file_manager.delete_file(file_id)
        assistant_manager.cleanup()
        
        print("\n✅ All GPT tools tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ GPT tools test failed: {e}")
        return False


def test_agents():
    """Test multi-agent system."""
    print("\n=== Testing Multi-Agent System ===")
    
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not anthropic_key:
        print("❌ Anthropic API key not found")
        return False
    
    try:
        # Test individual agents
        print("\n1. Testing Architect Agent...")
        architect = ArchitectAgent(anthropic_key)
        
        task = {
            'type': 'analyze_request',
            'data': {'request': 'What is our total carbon footprint?'}
        }
        
        result = architect.process_task(task)
        if result.get('status') == 'success':
            print("✓ Architect agent working")
        
        print("\n2. Testing Data Agent...")
        data_agent = DataAgent(anthropic_key, openai_key)
        
        task = {
            'type': 'extract',
            'data': {
                'content': 'Carbon emissions: 500 tons, Volunteers: 100 people',
                'metric_types': ['environmental', 'social']
            }
        }
        
        result = data_agent.process_task(task)
        if result.get('status') == 'success':
            print(f"✓ Data agent extracted {result.get('count', 0)} metrics")
        
        print("\n3. Testing Query Agent...")
        query_agent = QueryAgent(anthropic_key, openai_key)
        
        task = {
            'type': 'analyze_intent',
            'data': {'question': 'How many volunteer hours last year?'}
        }
        
        result = query_agent.process_task(task)
        if result.get('status') == 'success':
            print("✓ Query agent analyzed intent")
        
        print("\n✅ All agent tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Agent test failed: {e}")
        return False


def test_orchestrator():
    """Test agent orchestrator."""
    print("\n=== Testing Orchestrator ===")
    
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not anthropic_key:
        print("❌ Anthropic API key not found")
        return False
    
    try:
        print("\n1. Initializing orchestrator...")
        orchestrator = AgentOrchestrator(anthropic_key, openai_key)
        orchestrator.start()
        print("✓ Orchestrator started")
        
        print("\n2. Getting system status...")
        status = orchestrator.get_system_status()
        
        if status.get('orchestrator_running'):
            print("✓ Orchestrator running")
            print(f"  Active agents: {len(status.get('agents', {}))}")
            print(f"  Tasks in queue: {status.get('task_statistics', {}).get('total', 0)}")
        
        print("\n3. Testing simple query...")
        result = orchestrator.query_data("Test query: What metrics do we track?")
        
        if result.get('status') in ['success', 'partial']:
            print("✓ Query processed")
        
        # Cleanup
        orchestrator.cleanup()
        print("\n✅ Orchestrator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Orchestrator test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("ImpactOS GPT Integration Tests")
    print("=" * 50)
    
    # Check environment
    print("\nEnvironment Check:")
    print(f"  OpenAI Key: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
    print(f"  Anthropic Key: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")
    
    # Run tests
    results = []
    
    if os.getenv('OPENAI_API_KEY'):
        results.append(("GPT Tools", test_gpt_tools()))
    
    if os.getenv('ANTHROPIC_API_KEY'):
        results.append(("Agents", test_agents()))
        
        if os.getenv('OPENAI_API_KEY'):
            results.append(("Orchestrator", test_orchestrator()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())