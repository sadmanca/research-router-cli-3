#!/usr/bin/env python3
"""
Debug the chunk lookup issue
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def debug_chunk_lookup():
    """Debug why chunks are not being found during lookup"""
    print("Debugging chunk lookup issue...")
    
    try:
        from nano_graphrag import GraphRAG, QueryParam
        from research_router_cli.utils.config import Config
        
        config = Config()
        working_dir = "sessions/1008PM"
        
        llm_config = config.get_llm_config()
        graphrag = GraphRAG(working_dir=working_dir, enable_llm_cache=True, **llm_config)
        
        print("Created GraphRAG instance")
        
        # Load the text chunks storage directly
        session_path = Path(working_dir)
        text_chunks_file = session_path / "kv_store_text_chunks.json"
        
        with open(text_chunks_file, 'r', encoding='utf-8') as f:
            stored_chunks = json.load(f)
            
        print(f"Storage contains {len(stored_chunks)} chunks")
        print("Sample stored chunk IDs:")
        for i, chunk_id in enumerate(list(stored_chunks.keys())[:5]):
            print(f"  {i+1}: {chunk_id}")
        
        # Now let's debug the lookup process step by step
        query = "what is GraphRAG-R1?"
        
        # Get entities from vector DB
        results = await graphrag.entities_vdb.query(query, top_k=20)
        print(f"\nFound {len(results)} entities from vector search")
        
        if not results:
            print("No entities found - this might be the issue!")
            return
        
        # Get node data
        entity_names = [r["entity_name"] for r in results]
        node_datas = await asyncio.gather(
            *[graphrag.chunk_entity_relation_graph.get_node(name) for name in entity_names]
        )
        
        # Filter out None nodes
        valid_nodes = [(name, node) for name, node in zip(entity_names, node_datas) if node is not None]
        print(f"Got {len(valid_nodes)} valid nodes")
        
        # Now check what source_ids we get
        from nano_graphrag.prompt import GRAPH_FIELD_SEP
        from nano_graphrag._utils import split_string_by_multi_markers
        
        all_text_unit_ids = set()
        for entity_name, node_data in valid_nodes[:3]:  # Check first 3
            if "source_id" in node_data:
                source_id = node_data["source_id"]
                text_units = split_string_by_multi_markers(source_id, [GRAPH_FIELD_SEP])
                all_text_unit_ids.update(text_units)
                print(f"\nEntity: {entity_name}")
                print(f"  Source ID: {source_id}")
                print(f"  Split into text units: {text_units}")
                
                # Check if these text units exist in storage
                for text_unit_id in text_units:
                    if text_unit_id in stored_chunks:
                        print(f"    [FOUND] {text_unit_id} found in storage")
                    else:
                        print(f"    [MISSING] {text_unit_id} NOT found in storage")
        
        print(f"\nTotal unique text unit IDs from entities: {len(all_text_unit_ids)}")
        
        # Check how many of these actually exist in storage
        found_in_storage = sum(1 for tuid in all_text_unit_ids if tuid in stored_chunks)
        print(f"Found in storage: {found_in_storage}")
        print(f"Missing from storage: {len(all_text_unit_ids) - found_in_storage}")
        
        # Show some missing ones
        missing_ids = [tuid for tuid in all_text_unit_ids if tuid not in stored_chunks]
        if missing_ids:
            print(f"\nSome missing text unit IDs:")
            for mid in missing_ids[:5]:
                print(f"  {mid}")
                
                # Check if this matches any of the problematic doc IDs
                problematic_docs = ["doc-e0b88d32f06db02af102ff4143604dd2", "doc-792ee78d563dd146ab2e518df79ba1bf"]
                if mid in problematic_docs:
                    print(f"    ^ This is one of the problematic document IDs!")
        
        # Now test direct retrieval from text_chunks_db
        print(f"\nTesting direct retrieval from text_chunks_db...")
        sample_ids = list(all_text_unit_ids)[:3]
        for sample_id in sample_ids:
            try:
                result = await graphrag.text_chunks.get_by_id(sample_id)
                if result is None:
                    print(f"  {sample_id}: get_by_id returned None")
                else:
                    print(f"  {sample_id}: get_by_id returned data with keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            except Exception as e:
                print(f"  {sample_id}: get_by_id failed: {e}")
                
    except Exception as e:
        print(f"[ERROR] Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_chunk_lookup())