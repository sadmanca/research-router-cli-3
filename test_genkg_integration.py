#!/usr/bin/env python3
"""
Test script to verify GenKG integration with nano-graphrag.
This script tests that:
1. GenKG entity extraction works as a drop-in replacement
2. All nano-graphrag storage files are created 
3. HTML and JSON visualizations are generated
4. Queries work after insertion with genkg
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from research_router_cli.commands.session import SessionManager
from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam

async def test_genkg_integration():
    """Test complete GenKG integration with nano-graphrag"""
    
    # Create a test session
    session_manager = SessionManager()
    test_session = "genkg_test"
    
    if not session_manager.create_session(test_session):
        session_manager.switch_session(test_session)
        
    working_dir = session_manager.get_current_working_dir()
    print(f"Testing with session: {test_session}")
    print(f"Working directory: {working_dir}")
    
    # Test text content (simulating PDF content)
    test_content = """
    GraphRAG-R1 is a novel approach to Graph Retrieval-Augmented Generation that combines 
    knowledge graphs with large language models. The method uses reinforcement learning 
    to improve the retrieval process. Key components include entity extraction, 
    relationship modeling, and graph-based reasoning. The approach demonstrates 
    significant improvements in question answering tasks compared to traditional RAG methods.
    
    The system employs advanced techniques like potential-based reward shaping and 
    multi-agent reinforcement learning. Community detection algorithms are used to 
    organize the knowledge graph into meaningful clusters. The evaluation shows 
    superior performance on complex reasoning tasks.
    """
    
    try:
        # Create GraphRAG instance with GenKG enabled
        print("Creating GraphRAG instance with GenKG integration...")
        graphrag = GraphRAG(
            working_dir=str(working_dir),
            enable_llm_cache=True,
            
            # Enable GenKG integration
            use_genkg_extraction=True,
            genkg_node_limit=15,  # Smaller for testing
            genkg_llm_provider="gemini", 
            genkg_model_name="gemini-2.5-flash",
            genkg_create_visualization=True,
        )
        
        print("Inserting test content with GenKG extraction...")
        await graphrag.ainsert(test_content)
        
        # Check that required nano-graphrag files exist
        required_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_full_docs.json", 
            "kv_store_text_chunks.json",
            "vdb_entities.json",
            "kv_store_community_reports.json"
        ]
        
        print("\nChecking nano-graphrag storage files:")
        all_nano_files_exist = True
        for filename in required_files:
            filepath = working_dir / filename
            if filepath.exists():
                print(f"  [+] {filename} - {filepath.stat().st_size} bytes")
            else:
                print(f"  [-] {filename} - MISSING")
                all_nano_files_exist = False
                
        # Check GenKG output files
        print("\nChecking GenKG visualization files:")
        genkg_files = ["output.html", "output.dashkg.json"]
        all_genkg_files_exist = True
        for filename in genkg_files:
            filepath = working_dir / filename
            if filepath.exists():
                print(f"  [+] {filename} - {filepath.stat().st_size} bytes")
            else:
                print(f"  [-] {filename} - MISSING")
                all_genkg_files_exist = False
        
        # Test queries
        if all_nano_files_exist:
            print("\nTesting queries...")
            
            # Test global query
            try:
                print("1. Testing global query...")
                response = await graphrag.aquery(
                    "What is GraphRAG-R1 and how does it work?", 
                    QueryParam(mode="global")
                )
                print(f"Global query response: {response[:200]}...")
                global_success = True
            except Exception as e:
                print(f"Global query failed: {e}")
                global_success = False
            
            # Test local query
            try:
                print("2. Testing local query...")
                response = await graphrag.aquery(
                    "What techniques are used in the system?", 
                    QueryParam(mode="local")
                )
                print(f"Local query response: {response[:200]}...")
                local_success = True
            except Exception as e:
                print(f"Local query failed: {e}")
                local_success = False
                
            # Summary
            print("\n" + "="*50)
            print("INTEGRATION TEST RESULTS:")
            print(f"Nano-graphrag files created: {'✓' if all_nano_files_exist else '✗'}")
            print(f"GenKG visualization files created: {'✓' if all_genkg_files_exist else '✗'}")
            print(f"Global query works: {'✓' if global_success else '✗'}")
            print(f"Local query works: {'✓' if local_success else '✗'}")
            
            if all_nano_files_exist and all_genkg_files_exist and global_success and local_success:
                print("\n🎉 ALL TESTS PASSED! GenKG integration successful!")
                return True
            else:
                print("\n❌ Some tests failed. Check the issues above.")
                return False
        else:
            print("❌ Essential nano-graphrag files missing. Cannot test queries.")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_genkg_integration())
    sys.exit(0 if result else 1)