#!/usr/bin/env python3
"""
Test the clustering fix by running a direct query
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_fix_with_direct_query():
    """Test if the clustering fix works by running queries"""
    print("Testing clustering fix with direct queries...")
    
    try:
        from nano_graphrag import GraphRAG, QueryParam
        from research_router_cli.utils.config import Config
        
        config = Config()
        if not config.validate_config():
            print("[ERROR] Config validation failed")
            return False
        
        working_dir = "sessions/1008PM"
        llm_config = config.get_llm_config()
        
        # Create GraphRAG instance
        graphrag = GraphRAG(
            working_dir=working_dir,
            enable_llm_cache=True,
            **llm_config
        )
        
        print("Created GraphRAG instance")
        
        # Test different types of queries to see if they include both documents
        test_queries = [
            ("GraphRAG R1 methodology", "Should find content about GraphRAG-R1 paper"),
            ("airport conversational AI systems", "Should find content about airport paper"),
            ("RAG methods comparison", "Should find content from both papers"),
            ("what documents are available", "Should find both documents"),
        ]
        
        for query, description in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: '{query}'")
            print(f"Expected: {description}")
            print(f"{'='*60}")
            
            try:
                # Test local mode (context only to avoid API costs)
                result = await graphrag.aquery(
                    query, 
                    param=QueryParam(mode="local", only_need_context=True)
                )
                
                if result:
                    print(f"[SUCCESS] Query returned {len(result)} characters")
                    
                    # Check if result mentions both documents
                    graphrag_mentions = 0
                    airport_mentions = 0
                    
                    result_lower = result.lower()
                    
                    # GraphRAG paper indicators
                    graphrag_keywords = ["graphrag", "reinforcement learning", "leiden", "grpo", "retrieval attenuation"]
                    graphrag_mentions = sum(1 for kw in graphrag_keywords if kw in result_lower)
                    
                    # Airport paper indicators  
                    airport_keywords = ["airport", "conversational ai", "flight", "aviation", "schiphol", "sql rag"]
                    airport_mentions = sum(1 for kw in airport_keywords if kw in result_lower)
                    
                    print(f"  GraphRAG paper indicators: {graphrag_mentions}")
                    print(f"  Airport paper indicators: {airport_mentions}")
                    
                    if graphrag_mentions > 0 and airport_mentions > 0:
                        print(f"  [EXCELLENT] Context includes content from BOTH documents!")
                    elif graphrag_mentions > 0:
                        print(f"  [PARTIAL] Context only from GraphRAG document")
                    elif airport_mentions > 0:
                        print(f"  [PARTIAL] Context only from Airport document")
                    else:
                        print(f"  [UNCLEAR] Could not identify document sources")
                    
                    # Show a preview
                    preview = result[:300].replace('\n', ' ')
                    print(f"  Preview: {preview}...")
                    
                else:
                    print(f"[ERROR] Query returned no result")
                    
            except Exception as e:
                print(f"[ERROR] Query failed: {e}")
                continue
        
        # Final assessment
        print(f"\n{'='*60}")
        print(f"OVERALL ASSESSMENT:")
        print(f"{'='*60}")
        
        # Check if clustering data exists in the graph
        graph_storage = graphrag.chunk_entity_relation_graph
        community_schema = await graph_storage.community_schema()
        
        print(f"Communities generated: {len(community_schema)}")
        
        # Check which document chunks are in communities
        all_chunk_ids = set()
        for community_data in community_schema.values():
            chunk_ids = community_data.get("chunk_ids", [])
            all_chunk_ids.update(chunk_ids)
        
        doc1_chunks = [c for c in all_chunk_ids if "doc-e0b88d32f06db02af102ff4143604dd2" in c]
        doc2_chunks = [c for c in all_chunk_ids if "doc-792ee78d563dd146ab2e518df79ba1bf" in c]
        
        print(f"Document chunks in communities:")
        print(f"  Doc1 (GraphRAG): {len(doc1_chunks)} chunks")
        print(f"  Doc2 (Airport): {len(doc2_chunks)} chunks")
        
        if len(doc1_chunks) > 0 and len(doc2_chunks) > 0:
            print(f"\n[SUCCESS] CLUSTERING FIX WORKED!")
            print(f"Both documents are now represented in communities!")
            return True
        elif len(doc1_chunks) > 0:
            print(f"\n[PARTIAL] Only first document in communities")
            return False
        elif len(doc2_chunks) > 0:
            print(f"\n[PARTIAL] Only second document in communities")
            return False
        else:
            print(f"\n[ERROR] No documents found in communities")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fix_with_direct_query())
    if success:
        print(f"\n[SUCCESS] Clustering fix verified successfully!")
        print(f"Local queries should now include content from both documents.")
    else:
        print(f"\n[ISSUE] Clustering fix needs further investigation.")