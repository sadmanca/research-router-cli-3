#!/usr/bin/env python3
"""Test script for enhanced-insert functionality"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.enhanced_insert import EnhancedInsertCommand

async def test_enhanced_insert():
    """Test the enhanced insert functionality"""
    
    # Create session manager
    session_manager = SessionManager()
    
    # Create test session 
    if not session_manager.create_session("test_enhanced"):
        # Session already exists, switch to it
        session_manager.switch_session("test_enhanced")
    
    # Create enhanced insert command
    enhanced_insert = EnhancedInsertCommand(session_manager)
    
    # Test with existing PDF (use one from test4 session)
    pdf_path = Path("sessions/test4/downloads/2507.23581v1_GraphRAG-R1 Graph Retrieval-Augmented Generation w.pdf")
    
    if pdf_path.exists():
        print(f"Testing enhanced-insert with {pdf_path.name}")
        try:
            await enhanced_insert.enhanced_insert_pdf(str(pdf_path))
            print("Enhanced insert completed successfully!")
            
            # Check if nano-graphrag files were created
            working_dir = session_manager.get_current_working_dir()
            required_files = [
                "graph_chunk_entity_relation.graphml",
                "kv_store_full_docs.json", 
                "kv_store_text_chunks.json",
                "vdb_entities.json"
            ]
            
            print("\nChecking for nano-graphrag files:")
            for filename in required_files:
                filepath = working_dir / filename
                if filepath.exists():
                    print(f"  [+] {filename} - {filepath.stat().st_size} bytes")
                else:
                    print(f"  [-] {filename} - MISSING")
            
            # Check visualization files
            viz_files = [
                "enhanced_knowledge_graph.html",
                "enhanced_knowledge_graph.dashkg.json"
            ]
            
            print("\nChecking for visualization files:")
            for filename in viz_files:
                filepath = working_dir / filename
                if filepath.exists():
                    print(f"  [+] {filename} - {filepath.stat().st_size} bytes")
                else:
                    print(f"  [-] {filename} - MISSING")
            
        except Exception as e:
            print(f"Error during enhanced insert: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Test PDF not found: {pdf_path}")
        print("Available PDF files:")
        for pdf in Path("sessions").rglob("*.pdf"):
            print(f"  - {pdf}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_insert())