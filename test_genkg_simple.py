#!/usr/bin/env python3
"""Simple test without Unicode characters to verify GenKG configuration"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.insert import InsertCommand

async def test_genkg_simple():
    """Test GenKG configuration without Unicode issues"""
    
    print("=== Testing GenKG Configuration ===")
    
    # Create session manager and insert command
    session_manager = SessionManager()
    if not session_manager.create_session("test_simple"):
        session_manager.switch_session("test_simple")
    
    insert_cmd = InsertCommand(session_manager)
    
    # Check GraphRAG configuration
    try:
        print("Creating GraphRAG instance...")
        graphrag = await insert_cmd._get_graphrag_instance()
        
        if graphrag:
            print("SUCCESS: GraphRAG instance created")
            
            # Check configuration
            using_gemini = getattr(graphrag, 'using_gemini', False)
            use_genkg = getattr(graphrag, 'use_genkg_extraction', False)
            func_name = graphrag.entity_extraction_func.__name__
            
            print(f"using_gemini: {using_gemini}")
            print(f"use_genkg_extraction: {use_genkg}")
            print(f"entity_extraction_func: {func_name}")
            
            if func_name == "extract_entities_genkg":
                print("PASS: GenKG entity extraction is configured")
                return True
            else:
                print("FAIL: Not using GenKG entity extraction")
                return False
        else:
            print("FAIL: Could not create GraphRAG instance")
            return False
            
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_genkg_simple())
    if success:
        print("\nFINAL RESULT: GenKG configuration is working")
    else:
        print("\nFINAL RESULT: GenKG configuration has issues")
    sys.exit(0 if success else 1)