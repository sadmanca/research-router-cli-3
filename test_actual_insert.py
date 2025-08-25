#!/usr/bin/env python3
"""Test actual content insertion to identify the real errors"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.insert import InsertCommand

async def test_actual_insertion():
    """Test actual content insertion to identify errors"""
    
    print("=== Testing Actual Content Insertion ===")
    
    # Simple test content
    test_content = """
    Machine Learning Overview
    
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn from data, 
    identify patterns and make decisions with minimal human intervention.
    
    Deep learning uses neural networks with three or more layers. These neural networks attempt 
    to simulate the behavior of the human brain allowing it to learn from large amounts of data.
    """
    
    # Create session manager and insert command
    session_manager = SessionManager()
    if not session_manager.create_session("test_insert"):
        session_manager.switch_session("test_insert")
    
    insert_cmd = InsertCommand(session_manager)
    
    try:
        print("Creating GraphRAG instance...")
        graphrag = await insert_cmd._get_graphrag_instance()
        
        if not graphrag:
            print("ERROR: Failed to create GraphRAG instance")
            return False
            
        print("GraphRAG instance created successfully")
        print("Starting content insertion...")
        
        # Try to insert content and catch specific errors
        await graphrag.ainsert(test_content)
        
        print("SUCCESS: Content inserted without errors")
        return True
        
    except Exception as e:
        print(f"INSERTION ERROR: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        # Print more detailed error info
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Caused by: {type(e.__cause__).__name__}: {str(e.__cause__)}")
            
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_actual_insertion())
    if success:
        print("\nRESULT: Insertion is working properly")
    else:
        print("\nRESULT: There are real insertion errors that need to be fixed")
    sys.exit(0 if success else 1)