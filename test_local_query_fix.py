#!/usr/bin/env python3
"""
Test script for local query functionality after fixing the NoneType subscriptable error
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.query import QueryCommand


async def test_local_query():
    """Test local query functionality"""
    print("Testing local query functionality...")
    
    # Initialize components
    session_manager = SessionManager()
    query_command = QueryCommand(session_manager)
    
    # Try to switch to the 0923PM session which should have data
    session_name = "0923PM"
    if session_manager.switch_session(session_name):
        print(f"Successfully switched to session: {session_name}")
        
        # Test a local query
        test_query = "what is graphrag-r1?"
        print(f"\nTesting local query: '{test_query}'")
        
        try:
            result = await query_command.query(test_query, mode="local")
            if result:
                print("[SUCCESS] Local query executed successfully!")
                print("The fix for 'NoneType' object is not subscriptable error appears to be working.")
            else:
                print("[FAIL] Local query returned no result")
        except Exception as e:
            print(f"[FAIL] Local query failed with error: {e}")
            print("The error might still exist or there could be other issues.")
        
        # Also test global query for comparison
        print(f"\nTesting global query: '{test_query}' for comparison")
        try:
            result = await query_command.query(test_query, mode="global")
            if result:
                print("[SUCCESS] Global query executed successfully!")
            else:
                print("[FAIL] Global query returned no result")
        except Exception as e:
            print(f"[FAIL] Global query failed with error: {e}")
            
    else:
        print(f"[FAIL] Could not switch to session: {session_name}")
        print("Available sessions:")
        session_manager.list_sessions()


if __name__ == "__main__":
    asyncio.run(test_local_query())