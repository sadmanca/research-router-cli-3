#!/usr/bin/env python3
"""
Analyze the disconnect between knowledge graph entities and vector database entities
"""
import asyncio
import json
import sys
import os
from pathlib import Path
from xml.etree import ElementTree as ET

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def analyze_graph_vs_vdb_disconnect():
    """Analyze why the graph has entities but VDB is empty"""
    print("Analyzing disconnect between knowledge graph and vector database...")
    
    session_path = Path("sessions/1008PM")
    
    # 1. Parse the GraphML file to extract entities
    print("1. ANALYZING KNOWLEDGE GRAPH ENTITIES")
    print("="*60)
    
    graph_file = session_path / "graph_chunk_entity_relation.graphml"
    graph_entities = []
    
    try:
        tree = ET.parse(graph_file)
        root = tree.getroot()
        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}
        
        # Parse all nodes (entities)
        for node in root.findall(".//graphml:node", ns):
            node_id = node.get("id")
            entity_data = {"id": node_id}
            
            # Extract all data attributes
            for data in node.findall("graphml:data", ns):
                key = data.get("key")
                value = data.text
                
                # Map keys to readable names (you'll need to check the key definitions)
                if key == "d0":
                    entity_data["entity_type"] = value
                elif key == "d1":
                    entity_data["entity_name"] = value
                elif key == "d2":
                    entity_data["source_id"] = value
                elif key == "d3":
                    entity_data["description"] = value
                # Add more mappings as needed
                else:
                    entity_data[f"attr_{key}"] = value
            
            graph_entities.append(entity_data)
        
        print(f"Total entities in knowledge graph: {len(graph_entities)}")
        
        # Group by document
        entities_by_doc = {}
        for entity in graph_entities:
            source_id = entity.get("source_id", "unknown")
            # Try to extract document ID from source_id
            doc_id = None
            if "doc-e0b88d32f06db02af102ff4143604dd2" in str(source_id):
                doc_id = "doc-e0b88d32f06db02af102ff4143604dd2"
            elif "doc-792ee78d563dd146ab2e518df79ba1bf" in str(source_id):
                doc_id = "doc-792ee78d563dd146ab2e518df79ba1bf"
            
            if doc_id:
                if doc_id not in entities_by_doc:
                    entities_by_doc[doc_id] = []
                entities_by_doc[doc_id].append(entity)
        
        print(f"\nEntities by document in graph:")
        for doc_id, entities in entities_by_doc.items():
            print(f"  {doc_id}: {len(entities)} entities")
            # Show sample entities
            for i, entity in enumerate(entities[:3]):
                entity_name = entity.get("entity_name", entity.get("id", "unknown"))
                print(f"    {i+1}. {entity_name}")
    
    except Exception as e:
        print(f"Error parsing GraphML: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Compare with vector database entities
    print(f"\n2. COMPARING WITH VECTOR DATABASE")
    print("="*60)
    
    entities_file = session_path / "vdb_entities.json"
    try:
        with open(entities_file, 'r', encoding='utf-8') as f:
            vdb_entities = json.load(f)
        
        print(f"Total entities in vector database: {len(vdb_entities)}")
        
        if len(vdb_entities) > 0:
            print(f"\nSample VDB entities:")
            for i, entity in enumerate(vdb_entities[:3]):
                print(f"  {i+1}. {json.dumps(entity, indent=2)}")
        else:
            print("VDB entities file is empty!")
    
    except Exception as e:
        print(f"Error reading VDB entities: {e}")
        vdb_entities = []
    
    # 3. Check community reports to see which entities they reference
    print(f"\n3. CHECKING COMMUNITY REPORTS")
    print("="*60)
    
    community_reports_file = session_path / "kv_store_community_reports.json"
    try:
        with open(community_reports_file, 'r', encoding='utf-8') as f:
            community_reports = json.load(f)
        
        print(f"Total community reports: {len(community_reports)}")
        
        # Check which entities are mentioned in community reports
        entities_in_reports = set()
        for report_id, report in community_reports.items():
            if "nodes" in report:
                for node in report["nodes"]:
                    entities_in_reports.add(node)
        
        print(f"Unique entities mentioned in community reports: {len(entities_in_reports)}")
        print(f"Sample entities from reports: {list(entities_in_reports)[:5]}")
        
        # Check which documents the community reports reference
        docs_in_reports = set()
        for report_id, report in community_reports.items():
            if "chunk_ids" in report:
                for chunk_id in report["chunk_ids"]:
                    docs_in_reports.add(chunk_id)
        
        print(f"\nDocuments referenced in community reports:")
        for doc_id in docs_in_reports:
            print(f"  - {doc_id}")
    
    except Exception as e:
        print(f"Error reading community reports: {e}")
    
    # 4. Check what's in the LLM cache to understand what happened
    print(f"\n4. CHECKING LLM CACHE FOR PROCESSING DETAILS")
    print("="*60)
    
    llm_cache_file = session_path / "kv_store_llm_response_cache.json" 
    try:
        with open(llm_cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        print(f"LLM cache entries: {len(cache_data)}")
        
        # Look for different types of processing
        processing_types = {
            "entity_extraction": 0,
            "community_detection": 0,
            "relationship_extraction": 0,
            "other": 0
        }
        
        cache_details = []
        for key, value in cache_data.items():
            if isinstance(value, dict) and "prompt" in value:
                prompt = value["prompt"].lower()
                response = str(value.get("response", ""))
                
                cache_entry = {
                    "key_preview": key[:100],
                    "prompt_length": len(value["prompt"]),
                    "response_length": len(response),
                    "prompt_preview": value["prompt"][:200].replace('\n', ' ')
                }
                
                if any(word in prompt for word in ["entity", "extract", "entities"]):
                    processing_types["entity_extraction"] += 1
                    cache_entry["type"] = "entity_extraction"
                elif any(word in prompt for word in ["community", "cluster", "group"]):
                    processing_types["community_detection"] += 1
                    cache_entry["type"] = "community_detection"
                elif any(word in prompt for word in ["relationship", "relation", "edge"]):
                    processing_types["relationship_extraction"] += 1
                    cache_entry["type"] = "relationship_extraction"
                else:
                    processing_types["other"] += 1
                    cache_entry["type"] = "other"
                
                cache_details.append(cache_entry)
        
        print(f"\nProcessing types in cache:")
        for proc_type, count in processing_types.items():
            print(f"  {proc_type}: {count}")
        
        print(f"\nSample cache entries:")
        for i, entry in enumerate(cache_details[:3]):
            print(f"  Entry {i+1} ({entry['type']}):")
            print(f"    Key: {entry['key_preview']}...")
            print(f"    Prompt length: {entry['prompt_length']}")
            print(f"    Response length: {entry['response_length']}")
            print(f"    Prompt preview: {entry['prompt_preview']}...")
    
    except Exception as e:
        print(f"Error reading LLM cache: {e}")
    
    # 5. Analysis and diagnosis
    print(f"\n5. DIAGNOSIS")
    print("="*60)
    
    print(f"FINDINGS:")
    print(f"  Knowledge graph: {len(graph_entities)} entities")
    print(f"  Vector database: {len(vdb_entities)} entities")
    print(f"  Community reports: {len(community_reports)} reports")
    
    if len(graph_entities) > 0 and len(vdb_entities) == 0:
        print(f"\n[ISSUE IDENTIFIED] Vector database is not populated despite having entities in knowledge graph")
        print(f"POSSIBLE CAUSES:")
        print(f"  1. VECTOR EMBEDDING FAILURE: Entity embeddings were not generated or saved")
        print(f"  2. VDB SYNC FAILURE: Entities exist in graph but weren't copied to vector database")
        print(f"  3. PROCESSING PIPELINE BREAK: Entity extraction completed but vector indexing failed")
        print(f"  4. FILE CORRUPTION: VDB entities file was truncated or corrupted")
        print(f"  5. API ERRORS: Embedding generation failed due to API limits/errors")
        
        print(f"\nIMPACT:")
        print(f"  - Knowledge graph queries work (can traverse graph)")
        print(f"  - Vector similarity search fails (no entities to search)")
        print(f"  - Local queries fail (depend on entity vector search)")
        print(f"  - Community reports only use first document (entity search failed for second)")
        
        return "vdb_sync_failure"
    
    elif len(graph_entities) == 0 and len(vdb_entities) == 0:
        print(f"\n[ISSUE IDENTIFIED] No entities extracted at all")
        print(f"  - This is a complete entity extraction failure")
        return "complete_extraction_failure"
    
    else:
        print(f"\n[UNCLEAR] Unexpected pattern")
        return "unclear"

if __name__ == "__main__":
    result = asyncio.run(analyze_graph_vs_vdb_disconnect())
    print(f"\nAnalysis result: {result}")