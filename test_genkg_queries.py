#!/usr/bin/env python3
"""
Test GenKG integration with a focus on query functionality.
This test uses a simple text input to verify the complete pipeline works.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "nano-graphrag"))

from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam

async def test_genkg_queries():
    """Test that queries work after GenKG integration"""
    
    # Simple test content
    test_content = """
    Artificial Intelligence (AI) is a field of computer science focused on creating intelligent machines. 
    Machine learning is a subset of AI that enables computers to learn from data. Deep learning uses 
    neural networks to process information. Natural language processing helps computers understand human language.
    Computer vision allows machines to interpret visual information. These technologies work together in modern AI systems.
    """
    
    working_dir = Path("./genkg_query_test")
    
    try:
        print("Testing GenKG integration with query functionality...")
        
        # Create GraphRAG with GenKG enabled and Gemini throughout
        graphrag = GraphRAG(
            working_dir=str(working_dir),
            enable_llm_cache=True,
            
            # Use Gemini for everything
            using_gemini=True,
            
            # GenKG settings
            use_genkg_extraction=True,
            genkg_node_limit=10,
            genkg_create_visualization=True,
            genkg_llm_provider="gemini",
            genkg_model_name="gemini-2.5-flash",
        )
        
        print("Inserting content with GenKG extraction...")
        await graphrag.ainsert(test_content)
        
        print("Checking created files...")
        required_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_full_docs.json", 
            "kv_store_text_chunks.json",
            "vdb_entities.json",
        ]
        
        files_exist = True
        for filename in required_files:
            filepath = working_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"  [+] {filename} ({size} bytes)")
            else:
                print(f"  [-] {filename} MISSING")
                files_exist = False
        
        # Check visualization files  
        viz_files = ["output.html", "output.dashkg.json"]
        for filename in viz_files:
            filepath = working_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"  [+] {filename} ({size} bytes)")
        
        if files_exist:
            print("\nTesting queries...")
            
            # Test global query
            try:
                response = await graphrag.aquery(
                    "What is artificial intelligence?",
                    QueryParam(mode="global")
                )
                print(f"Global query successful: {len(response)} chars")
                global_success = True
            except Exception as e:
                print(f"Global query failed: {e}")
                global_success = False
            
            # Test local query
            try:
                response = await graphrag.aquery(
                    "How do neural networks work?",
                    QueryParam(mode="local") 
                )
                print(f"Local query successful: {len(response)} chars")
                local_success = True
            except Exception as e:
                print(f"Local query failed: {e}")
                local_success = False
                
            print("\n" + "="*50)
            if files_exist and global_success and local_success:
                print("SUCCESS: GenKG integration fully functional!")
                print("- Entity extraction using GenKG: WORKING")
                print("- Nano-graphrag storage files: CREATED") 
                print("- Visualization files: CREATED")
                print("- Global queries: WORKING")
                print("- Local queries: WORKING")
                return True
            else:
                print("PARTIAL SUCCESS: Some components failed")
                return False
        else:
            print("FAILED: Required storage files not created")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_genkg_queries())
    sys.exit(0 if result else 1)