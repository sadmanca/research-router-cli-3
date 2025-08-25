#!/usr/bin/env python3
"""
Test what happens when we run an actual query through the research-router system
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_actual_query_behavior():
    """Test actual query behavior to see what's happening"""
    print("Testing actual query behavior...")
    
    try:
        from research_router_cli.commands.query import QueryCommand
        from research_router_cli.commands.session import SessionManager
        
        # Test the exact same query workflow as the CLI
        session_name = "1008PM"
        session_path = Path(f"sessions/{session_name}")
        
        if not session_path.exists():
            print(f"[ERROR] Session {session_name} not found")
            return False
        
        print(f"Testing session: {session_name}")
        
        # Create session manager and query command like the CLI does
        session_manager = SessionManager()
        session_manager.current_session = session_name
        query_command = QueryCommand(session_manager)
        
        # Test queries that should include both documents
        test_queries = [
            "what are the different RAG methods discussed?",
            "compare GraphRAG and airport conversational AI", 
            "what documents are in this knowledge base?",
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: '{query}'")
            print(f"{'='*60}")
            
            try:
                # Call the actual query method used by the CLI
                result = await query_command.query_with_context(
                    question=query,
                    mode="local"
                )
                
                if result:
                    print(f"[SUCCESS] Query returned result")
                    print(f"Result length: {len(result)} characters")
                    
                    # Check if result mentions both documents
                    result_lower = result.lower()
                    
                    # GraphRAG paper indicators
                    graphrag_keywords = ["graphrag", "reinforcement learning", "leiden", "grpo", "retrieval attenuation"]
                    graphrag_mentions = sum(1 for kw in graphrag_keywords if kw in result_lower)
                    
                    # Airport paper indicators  
                    airport_keywords = ["airport", "conversational ai", "flight", "aviation", "schiphol", "sql rag"]
                    airport_mentions = sum(1 for kw in airport_keywords if kw in result_lower)
                    
                    print(f"GraphRAG paper indicators: {graphrag_mentions}")
                    print(f"Airport paper indicators: {airport_mentions}")
                    
                    if graphrag_mentions > 0 and airport_mentions > 0:
                        print(f"[EXCELLENT] Result includes content from BOTH documents!")
                    elif graphrag_mentions > 0:
                        print(f"[ISSUE] Result only from GraphRAG document")
                    elif airport_mentions > 0:
                        print(f"[ISSUE] Result only from Airport document") 
                    else:
                        print(f"[UNCLEAR] Could not identify document sources")
                    
                    # Show a preview of the actual result
                    preview = result[:500].replace('\n', ' ')
                    print(f"Preview: {preview}...")
                    
                else:
                    print(f"[ERROR] Query returned no result")
                    
            except Exception as e:
                print(f"[ERROR] Query failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_actual_query_behavior())
    print(f"\nTest completed: {'SUCCESS' if success else 'FAILED'}")