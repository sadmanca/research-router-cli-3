#!/usr/bin/env python3
"""Test the insert command with GenKG configuration"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.insert import InsertCommand

async def test_insert_with_genkg():
    """Test insert command with GenKG configuration"""
    
    # Create a test session
    session_manager = SessionManager()
    test_session = "genkg_insert_test"
    
    if not session_manager.create_session(test_session):
        session_manager.switch_session(test_session)
    
    print(f"Testing insert with session: {test_session}")
    
    # Create test content file
    test_pdf_content = """
    Machine Learning and Artificial Intelligence
    
    Machine learning is a subset of artificial intelligence that focuses on developing 
    algorithms that can learn and improve from experience. Deep learning, a branch of 
    machine learning, uses neural networks with multiple layers to process data.
    
    Natural language processing enables computers to understand and generate human language. 
    Computer vision allows machines to interpret visual information from images and videos.
    These technologies are transforming industries like healthcare, finance, and transportation.
    """
    
    # Create InsertCommand instance
    insert_cmd = InsertCommand(session_manager)
    
    # Test the GraphRAG configuration
    try:
        graphrag = await insert_cmd._get_graphrag_instance()
        if graphrag:
            print("[+] GraphRAG instance created successfully")
            print(f"  - using_gemini: {getattr(graphrag, 'using_gemini', 'Not set')}")
            print(f"  - use_genkg_extraction: {getattr(graphrag, 'use_genkg_extraction', 'Not set')}")
            print(f"  - entity_extraction_func: {graphrag.entity_extraction_func.__name__}")
            
            # Test insertion
            print("\nTesting content insertion...")
            await graphrag.ainsert(test_pdf_content)
            print("[+] Content inserted successfully with GenKG")
            
            # Check working directory for created files
            working_dir = session_manager.get_current_working_dir()
            files = list(working_dir.glob("*.json")) + list(working_dir.glob("*.graphml")) + list(working_dir.glob("*.html"))
            
            print(f"\nCreated files in {working_dir}:")
            for file in files:
                size = file.stat().st_size
                print(f"  [+] {file.name} ({size} bytes)")
            
            return True
        else:
            print("[-] Failed to create GraphRAG instance")
            return False
    except Exception as e:
        print(f"[-] Error testing insert: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_insert_with_genkg())
    sys.exit(0 if success else 1)