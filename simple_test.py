#!/usr/bin/env python3
"""Simple test for nano-graphrag creation"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from research_router_cli.commands.session import SessionManager
from nano_graphrag import GraphRAG

async def simple_nano_test():
    """Test nano-graphrag directly"""
    
    # Create session manager
    session_manager = SessionManager()
    
    # Create test session
    if not session_manager.create_session("simple_test"):
        session_manager.switch_session("simple_test")
    
    working_dir = session_manager.get_current_working_dir()
    print(f"Working in: {working_dir}")
    
    # Read test PDF
    pdf_path = Path("sessions/test4/downloads/2507.23581v1_GraphRAG-R1 Graph Retrieval-Augmented Generation w.pdf")
    if pdf_path.exists():
        print(f"Reading {pdf_path.name}...")
        from research_router_cli.utils.pdf_processor import PDFProcessor
        pdf_processor = PDFProcessor()
        text = pdf_processor.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters")
        
        # Create GraphRAG with genkg
        graphrag = GraphRAG(
            working_dir=str(working_dir),
            enable_llm_cache=True,
            use_gemini_extraction=True,  
            gemini_node_limit=25,
            gemini_model_name="models/gemini-2.5-flash",
            gemini_combine_with_llm_extraction=False,
        )
        
        print("Inserting into nano-graphrag...")
        await graphrag.ainsert(text)
        
        print("Checking created files...")
        files_to_check = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_full_docs.json", 
            "kv_store_text_chunks.json",
            "vdb_entities.json"
        ]
        
        for filename in files_to_check:
            filepath = working_dir / filename
            if filepath.exists():
                print(f"  [+] {filename} - {filepath.stat().st_size} bytes")
            else:
                print(f"  [-] {filename} - MISSING")
        
        print("Testing query...")
        try:
            from nano_graphrag.base import QueryParam
            response = await graphrag.aquery("What is GraphRAG-R1?", QueryParam(mode="global"))
            print("Query response:", response[:200] + "..." if len(response) > 200 else response)
        except Exception as e:
            print(f"Query failed: {e}")
    else:
        print(f"PDF not found: {pdf_path}")

if __name__ == "__main__":
    asyncio.run(simple_nano_test())