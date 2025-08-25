#!/usr/bin/env python3
"""
Examine the actual content of query results to verify they truly contain information from both documents
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def examine_actual_query_content():
    """Examine actual query content to see if it truly contains info from both docs"""
    print("Examining actual query content...")
    
    try:
        from research_router_cli.commands.query import QueryCommand
        from research_router_cli.commands.session import SessionManager
        
        # Set up the query system
        session_name = "1008PM"
        session_manager = SessionManager()
        session_manager.current_session = session_name
        query_command = QueryCommand(session_manager)
        
        # Test with a very specific query about airport paper content
        airport_query = "What is the accuracy rate of Graph RAG for airport domain questions and how does it compare to traditional RAG?"
        
        print(f"Testing airport-specific query: '{airport_query}'")
        print("="*80)
        
        result = await query_command.query_with_context(
            question=airport_query,
            mode="local"
        )
        
        if result:
            print(f"Full result ({len(result)} chars):")
            # Handle Unicode characters for Windows console
            try:
                print(result.encode('ascii', errors='replace').decode('ascii'))
            except:
                print("Result contains special characters, analyzing content...")
            print("\n" + "="*80)
            
            # Look for specific details that should only be in the airport paper
            airport_specific_details = [
                "91.49%",  # Graph RAG accuracy mentioned in airport paper
                "84.84%",  # Traditional RAG accuracy in airport paper
                "Amsterdam Airport Schiphol",
                "BM25 + GPT-4",
                "SQL RAG",
                "flight data",
                "aviation safety"
            ]
            
            graphrag_specific_details = [
                "GraphRAG-R1",
                "process-constrained reinforcement learning", 
                "38.03% on HotpotQA",
                "Progressive Retrieval Attenuation",
                "Cost-Aware F1",
                "hierarchical Leiden",
                "GRPO"
            ]
            
            print("DETAILED CONTENT ANALYSIS:")
            print("-" * 40)
            
            found_airport_details = []
            for detail in airport_specific_details:
                if detail.lower() in result.lower():
                    found_airport_details.append(detail)
                    print(f"✅ Found airport detail: '{detail}'")
                else:
                    print(f"❌ Missing airport detail: '{detail}'")
            
            print()
            found_graphrag_details = []
            for detail in graphrag_specific_details:
                if detail.lower() in result.lower():
                    found_graphrag_details.append(detail)
                    print(f"✅ Found GraphRAG detail: '{detail}'")
                else:
                    print(f"❌ Missing GraphRAG detail: '{detail}'")
            
            print(f"\nSUMMARY:")
            print(f"Airport-specific details found: {len(found_airport_details)}/{len(airport_specific_details)}")
            print(f"GraphRAG-specific details found: {len(found_graphrag_details)}/{len(graphrag_specific_details)}")
            
            if len(found_airport_details) > 0 and len(found_graphrag_details) > 0:
                print(f"✅ CONFIRMED: Query result contains ACTUAL details from BOTH documents!")
            elif len(found_airport_details) > 0:
                print(f"⚠️ PARTIAL: Query result only contains details from AIRPORT document")
            elif len(found_graphrag_details) > 0:
                print(f"⚠️ PARTIAL: Query result only contains details from GRAPHRAG document")
            else:
                print(f"❌ ISSUE: Query result contains no specific details from either document")
                
        else:
            print("❌ Query returned no result")
            
        # Test with a GraphRAG-specific query
        print(f"\n" + "="*80)
        graphrag_query = "What is the F1 score improvement achieved by GraphRAG-R1 on HotpotQA?"
        print(f"Testing GraphRAG-specific query: '{graphrag_query}'")
        print("="*80)
        
        result2 = await query_command.query_with_context(
            question=graphrag_query,
            mode="local"
        )
        
        if result2:
            print(f"Result preview (first 500 chars):")
            try:
                print(result2[:500].encode('ascii', errors='replace').decode('ascii') + "...")
            except:
                print("Result contains special characters...")
            
            # Check if this GraphRAG query mentions airport content (it shouldn't unless truly integrated)
            airport_mentions_in_graphrag_query = sum(1 for detail in airport_specific_details if detail.lower() in result2.lower())
            print(f"\nAirport details mentioned in GraphRAG query: {airport_mentions_in_graphrag_query}")
            
            if airport_mentions_in_graphrag_query > 0:
                print("✅ INTEGRATION CONFIRMED: GraphRAG query also includes airport context")
            else:
                print("❌ NO INTEGRATION: GraphRAG query doesn't include airport content")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Examination failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(examine_actual_query_content())
    print(f"\nExamination completed: {'SUCCESS' if success else 'FAILED'}")