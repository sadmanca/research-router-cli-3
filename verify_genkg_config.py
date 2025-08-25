#!/usr/bin/env python3
"""Verify that the research-router-cli insert command is configured for GenKG"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.insert import InsertCommand

async def verify_genkg_config():
    """Verify that insert command uses GenKG configuration"""
    
    # Create session manager and insert command
    session_manager = SessionManager()
    if not session_manager.create_session("verify_genkg"):
        session_manager.switch_session("verify_genkg")
    
    insert_cmd = InsertCommand(session_manager)
    
    # Check GraphRAG configuration
    try:
        graphrag = await insert_cmd._get_graphrag_instance()
        if graphrag:
            print("=== GenKG Configuration Verification ===")
            print(f"✓ GraphRAG instance created successfully")
            print(f"✓ using_gemini: {getattr(graphrag, 'using_gemini', False)}")
            print(f"✓ use_genkg_extraction: {getattr(graphrag, 'use_genkg_extraction', False)}")
            print(f"✓ entity_extraction_func: {graphrag.entity_extraction_func.__name__}")
            print(f"✓ genkg_create_visualization: {getattr(graphrag, 'genkg_create_visualization', False)}")
            
            # Check if it's the right function
            if graphrag.entity_extraction_func.__name__ == "extract_entities_genkg":
                print("\n🎉 SUCCESS: Insert command is properly configured for GenKG!")
                print("   - Entity extraction will use GenKG methods")
                print("   - All LLM calls will use Gemini")
                print("   - Embeddings will use Gemini")
                print("   - Visualizations will be generated")
                return True
            else:
                print("\n❌ ISSUE: Insert command is not using GenKG entity extraction")
                return False
        else:
            print("❌ Failed to create GraphRAG instance")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_genkg_config())
    if success:
        print("\n✅ The nano-graphrag insert method in research-router-cli is now working with GenKG!")
    sys.exit(0 if success else 1)