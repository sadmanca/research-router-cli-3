#!/usr/bin/env python3
"""Test if query works after insert using existing nano-graphrag files"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from research_router_cli.commands.session import SessionManager
from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam

async def test_query():
    """Test querying with existing nano-graphrag files"""
    
    # Use session test1 which already has nano-graphrag files
    session_manager = SessionManager()
    session_manager.switch_session("test1")
    working_dir = session_manager.get_current_working_dir()
    print(f"Testing with session: test1")
    print(f"Working directory: {working_dir}")
    
    # Check if required files exist
    required_files = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_full_docs.json", 
        "kv_store_text_chunks.json",
        "vdb_entities.json"
    ]
    
    print("Checking required files:")
    all_exist = True
    for filename in required_files:
        filepath = working_dir / filename
        if filepath.exists():
            print(f"  [+] {filename} - {filepath.stat().st_size} bytes")
        else:
            print(f"  [-] {filename} - MISSING")
            all_exist = False
    
    if not all_exist:
        print("Some required files are missing. Cannot test query.")
        return
    
    # Create GraphRAG instance
    try:
        graphrag = GraphRAG(
            working_dir=str(working_dir),
            enable_llm_cache=True,
        )
        
        print("\nTesting queries...")
        
        # Test global query
        print("1. Testing global query...")
        try:
            response = await graphrag.aquery("What is the main topic?", QueryParam(mode="global"))
            print(f"Global query response: {response[:200]}...")
        except Exception as e:
            print(f"Global query failed: {e}")
        
        # Test local query  
        print("2. Testing local query...")
        try:
            response = await graphrag.aquery("What are the key concepts?", QueryParam(mode="local"))
            print(f"Local query response: {response[:200]}...")
        except Exception as e:
            print(f"Local query failed: {e}")
            
    except Exception as e:
        print(f"Error creating GraphRAG instance: {e}")

if __name__ == "__main__":
    asyncio.run(test_query())