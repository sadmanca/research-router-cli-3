#!/usr/bin/env python3
"""
Debug script for local query NoneType subscriptable error
"""
import asyncio
import sys
import os
import traceback
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_debug_local_query():
    """Debug local query step by step"""
    print("Starting debug test for local query...")
    
    try:
        # Import nano-graphrag components
        from nano_graphrag import GraphRAG, QueryParam
        from research_router_cli.utils.config import Config
        
        config = Config()
        if not config.validate_config():
            print("[ERROR] Config validation failed")
            return
            
        # Set working directory to the session with data
        working_dir = "sessions/0923PM"
        print(f"Using working directory: {working_dir}")
        
        # Create GraphRAG instance
        print("Creating GraphRAG instance...")
        llm_config = config.get_llm_config()
        
        graphrag = GraphRAG(
            working_dir=working_dir,
            enable_llm_cache=True,
            **llm_config
        )
        print("GraphRAG instance created successfully")
        
        # Test query
        query = "what is graphrag-r1?"
        print(f"Testing query: '{query}'")
        
        # Try local query with detailed error tracking
        print("Attempting local query...")
        try:
            result = await graphrag.aquery(
                query, 
                param=QueryParam(mode="local")
            )
            print(f"[SUCCESS] Local query completed!")
            print(f"Result length: {len(result) if result else 0} characters")
            if result and len(result) > 100:
                print(f"Result preview: {result[:100]}...")
            else:
                print(f"Full result: {result}")
        except Exception as e:
            print(f"[ERROR] Local query failed: {e}")
            print("Full traceback:")
            traceback.print_exc()
            
        # Try global query for comparison  
        print(f"\nTesting global query for comparison...")
        try:
            result = await graphrag.aquery(
                query, 
                param=QueryParam(mode="global")
            )
            print(f"[SUCCESS] Global query completed!")
            print(f"Result length: {len(result) if result else 0} characters")
        except Exception as e:
            print(f"[ERROR] Global query failed: {e}")
            
    except Exception as e:
        print(f"[ERROR] Test setup failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_debug_local_query())