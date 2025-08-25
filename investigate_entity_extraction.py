#!/usr/bin/env python3
"""
Investigate why nano-graphrag only extracted entities from the first document
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def investigate_entity_extraction():
    """Investigate the entity extraction process step by step"""
    print("Investigating entity extraction process...")
    
    session_path = Path("sessions/1008PM")
    
    # 1. Check the text chunks - this is the input to entity extraction
    print("1. ANALYZING TEXT CHUNKS (Input to entity extraction)")
    print("="*60)
    
    text_chunks_file = session_path / "kv_store_text_chunks.json"
    with open(text_chunks_file, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)
    
    chunks_by_doc = {}
    for chunk_id, chunk_data in text_chunks.items():
        if isinstance(chunk_data, dict) and "data" in chunk_data and chunk_data["data"]:
            full_doc_id = chunk_data["data"].get("full_doc_id")
            content = chunk_data["data"].get("content", "")
            if full_doc_id:
                if full_doc_id not in chunks_by_doc:
                    chunks_by_doc[full_doc_id] = []
                chunks_by_doc[full_doc_id].append({
                    "chunk_id": chunk_id,
                    "content_length": len(content),
                    "chunk_order": chunk_data["data"].get("chunk_order_index", -1)
                })
    
    for doc_id, chunks in chunks_by_doc.items():
        chunks.sort(key=lambda x: x["chunk_order"])  # Sort by order
        print(f"\nDocument: {doc_id}")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total content length: {sum(c['content_length'] for c in chunks)}")
        print(f"  Chunk order range: {chunks[0]['chunk_order']} to {chunks[-1]['chunk_order']}")
        
        # Show first chunk preview for each doc
        first_chunk_id = chunks[0]["chunk_id"]
        first_chunk_content = text_chunks[first_chunk_id]["data"]["content"][:200]
        preview = first_chunk_content.replace('\n', ' ')[:150]
        print(f"  First chunk preview: {preview}...")
    
    # 2. Check the entity extraction results
    print(f"\n2. ANALYZING ENTITY EXTRACTION RESULTS")
    print("="*60)
    
    entities_file = session_path / "vdb_entities.json"
    try:
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
        
        print(f"Total entities extracted: {len(entities_data)}")
        
        # Group entities by source document
        entities_by_doc = {}
        entity_details = []
        
        for entity in entities_data:
            if isinstance(entity, dict):
                # Check different possible locations for source document info
                source_doc = None
                metadata = entity.get("metadata", {})
                
                if "source_doc" in metadata:
                    source_doc = metadata["source_doc"]
                elif "source_id" in metadata:
                    source_doc = metadata["source_id"]
                elif "source_id" in entity:
                    source_doc = entity["source_id"]
                
                entity_name = entity.get("entity_name", entity.get("content", "unknown"))
                
                entity_details.append({
                    "entity_name": entity_name,
                    "source_doc": source_doc,
                    "metadata_keys": list(metadata.keys()),
                    "entity_keys": list(entity.keys())
                })
                
                if source_doc:
                    if source_doc not in entities_by_doc:
                        entities_by_doc[source_doc] = []
                    entities_by_doc[source_doc].append(entity_name)
        
        print(f"\nEntities by document:")
        for doc_id, entity_names in entities_by_doc.items():
            print(f"  {doc_id}: {len(entity_names)} entities")
        
        print(f"\nSample entity details:")
        for i, detail in enumerate(entity_details[:3]):
            print(f"  Entity {i+1}:")
            print(f"    Name: {detail['entity_name']}")
            print(f"    Source doc: {detail['source_doc']}")
            print(f"    Metadata keys: {detail['metadata_keys']}")
            print(f"    Entity keys: {detail['entity_keys']}")
    
    except Exception as e:
        print(f"Error reading entities: {e}")
        entities_by_doc = {}
    
    # 3. Check the graph data
    print(f"\n3. ANALYZING KNOWLEDGE GRAPH")
    print("="*60)
    
    try:
        # Check GraphML file for entity source IDs
        from xml.etree import ElementTree as ET
        
        graph_file = session_path / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            tree = ET.parse(graph_file)
            root = tree.getroot()
            
            # Find all nodes (entities) and their source_id attributes
            nodes_with_sources = []
            ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}
            
            for node in root.findall(".//graphml:node", ns):
                node_id = node.get("id")
                source_id = None
                
                for data in node.findall("graphml:data", ns):
                    if data.get("key") == "d2":  # source_id key
                        source_id = data.text
                        break
                
                if source_id:
                    nodes_with_sources.append({
                        "node_id": node_id,
                        "source_id": source_id
                    })
            
            print(f"Graph nodes with source IDs: {len(nodes_with_sources)}")
            
            # Group by document
            graph_nodes_by_doc = {}
            for node in nodes_with_sources:
                # Extract document ID from source_id (it might be a chunk ID or doc ID)
                source_id = node["source_id"]
                
                # Try to match to known document IDs
                for doc_id in chunks_by_doc.keys():
                    if doc_id in source_id:
                        if doc_id not in graph_nodes_by_doc:
                            graph_nodes_by_doc[doc_id] = []
                        graph_nodes_by_doc[doc_id].append(node)
                        break
            
            print(f"\nGraph nodes by document:")
            for doc_id, nodes in graph_nodes_by_doc.items():
                print(f"  {doc_id}: {len(nodes)} nodes")
                # Show sample source IDs
                sample_sources = list(set(node["source_id"] for node in nodes[:3]))
                print(f"    Sample source IDs: {sample_sources}")
        
        else:
            print("Graph file not found")
    
    except Exception as e:
        print(f"Error reading graph: {e}")
    
    # 4. Check LLM cache to see what was actually processed
    print(f"\n4. ANALYZING LLM CACHE (What was actually processed)")
    print("="*60)
    
    try:
        llm_cache_file = session_path / "kv_store_llm_response_cache.json"
        if llm_cache_file.exists():
            with open(llm_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            print(f"LLM cache entries: {len(cache_data)}")
            
            # Look for entity extraction related cache entries
            entity_extraction_calls = []
            for key, value in cache_data.items():
                if isinstance(value, dict) and "prompt" in value:
                    prompt = value["prompt"]
                    # Look for entity extraction keywords in prompts
                    if any(keyword in prompt.lower() for keyword in ["entity", "extract", "relationship", "knowledge"]):
                        # Try to identify which document this relates to
                        doc_refs = []
                        for doc_id in chunks_by_doc.keys():
                            if doc_id in prompt:
                                doc_refs.append(doc_id)
                        
                        entity_extraction_calls.append({
                            "cache_key": key[:100] + "..." if len(key) > 100 else key,
                            "prompt_length": len(prompt),
                            "doc_refs": doc_refs,
                            "response_length": len(str(value.get("response", "")))
                        })
            
            print(f"Entity extraction related LLM calls: {len(entity_extraction_calls)}")
            for i, call in enumerate(entity_extraction_calls[:5]):
                print(f"  Call {i+1}:")
                print(f"    Cache key: {call['cache_key']}")
                print(f"    Prompt length: {call['prompt_length']}")
                print(f"    Doc refs: {call['doc_refs']}")
                print(f"    Response length: {call['response_length']}")
        
        else:
            print("LLM cache file not found")
    
    except Exception as e:
        print(f"Error reading LLM cache: {e}")
    
    # 5. Analysis and conclusions
    print(f"\n5. ANALYSIS AND CONCLUSIONS")
    print("="*60)
    
    doc1 = "doc-e0b88d32f06db02af102ff4143604dd2"  # GraphRAG R1 paper
    doc2 = "doc-792ee78d563dd146ab2e518df79ba1bf"  # Airport RAG paper
    
    print(f"\nDocument processing status:")
    print(f"  {doc1} (GraphRAG paper):")
    print(f"    Chunks: {len(chunks_by_doc.get(doc1, []))}")
    print(f"    Entities: {len(entities_by_doc.get(doc1, []))}")
    
    print(f"  {doc2} (Airport paper):")
    print(f"    Chunks: {len(chunks_by_doc.get(doc2, []))}")  
    print(f"    Entities: {len(entities_by_doc.get(doc2, []))}")
    
    # Identify the likely issue
    if len(chunks_by_doc.get(doc2, [])) > 0 and len(entities_by_doc.get(doc2, [])) == 0:
        print(f"\n[ISSUE IDENTIFIED] Entity extraction failed for second document:")
        print(f"  - Second document has {len(chunks_by_doc.get(doc2, []))} text chunks")
        print(f"  - Second document has 0 entities extracted")
        print(f"  - This suggests entity extraction was interrupted or failed for the second document")
        
        print(f"\nPossible causes:")
        print(f"  1. INCREMENTAL PROCESSING BUG: nano-graphrag may not properly handle incremental document addition")
        print(f"  2. API LIMITS/ERRORS: Entity extraction may have hit API limits after processing first document") 
        print(f"  3. PROCESSING ORDER: Documents may be processed in order, and second failed")
        print(f"  4. CHUNK DEPENDENCY: Entity extraction may depend on successful processing of all chunks from first doc")
        print(f"  5. CACHE ISSUES: Stale cache may have prevented processing of second document")
        
        return {
            "issue": "incomplete_entity_extraction",
            "doc1_chunks": len(chunks_by_doc.get(doc1, [])),
            "doc1_entities": len(entities_by_doc.get(doc1, [])),
            "doc2_chunks": len(chunks_by_doc.get(doc2, [])),
            "doc2_entities": len(entities_by_doc.get(doc2, []))
        }
    else:
        print(f"\n[UNCLEAR] Issue pattern not as expected")
        return {"issue": "unclear"}

if __name__ == "__main__":
    result = asyncio.run(investigate_entity_extraction())
    print(f"\nInvestigation complete. Result: {result}")