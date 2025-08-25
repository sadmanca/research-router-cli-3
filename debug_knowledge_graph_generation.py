#!/usr/bin/env python3
"""
Debug why the second document didn't get entity extraction and knowledge graph generation
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def debug_knowledge_graph_generation():
    """Debug the knowledge graph generation process"""
    print("Debugging knowledge graph generation for multiple documents...")
    
    session_path = Path("sessions/1008PM")
    
    # Check what documents exist
    full_docs_file = session_path / "kv_store_full_docs.json"
    text_chunks_file = session_path / "kv_store_text_chunks.json"
    entities_file = session_path / "vdb_entities.json"
    
    with open(full_docs_file, 'r', encoding='utf-8') as f:
        full_docs = json.load(f)
    
    with open(text_chunks_file, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)
        
    print(f"Found {len(full_docs)} documents:")
    for doc_id, doc_data in full_docs.items():
        content_preview = doc_data.get('content', '')[:200].replace('\n', ' ').encode('ascii', errors='replace').decode('ascii')
        print(f"  {doc_id}: {content_preview}...")
    
    print(f"\nFound {len(text_chunks)} text chunks:")
    
    # Group chunks by document
    chunks_by_doc = {}
    for chunk_id, chunk_data in text_chunks.items():
        if isinstance(chunk_data, dict) and "data" in chunk_data and chunk_data["data"]:
            full_doc_id = chunk_data["data"].get("full_doc_id")
            if full_doc_id:
                if full_doc_id not in chunks_by_doc:
                    chunks_by_doc[full_doc_id] = []
                chunks_by_doc[full_doc_id].append(chunk_id)
    
    for doc_id, chunk_ids in chunks_by_doc.items():
        print(f"  {doc_id}: {len(chunk_ids)} chunks")
    
    # Check entities
    try:
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
        print(f"\nFound {len(entities_data)} entities")
        
        # Check which documents the entities reference
        entities_by_doc = {}
        for entity_data in entities_data:
            if isinstance(entity_data, dict) and "metadata" in entity_data:
                source_doc = entity_data["metadata"].get("source_doc", "unknown")
                if source_doc not in entities_by_doc:
                    entities_by_doc[source_doc] = []
                entities_by_doc[source_doc].append(entity_data.get("entity_name", "unknown"))
        
        for doc_id, entity_names in entities_by_doc.items():
            print(f"  Entities from {doc_id}: {len(entity_names)}")
            if len(entity_names) <= 5:
                print(f"    Examples: {entity_names}")
            else:
                print(f"    Examples: {entity_names[:5]}...")
                
    except Exception as e:
        print(f"Error reading entities file: {e}")
    
    print("\n" + "="*50)
    print("DIAGNOSIS:")
    print("="*50)
    
    # Check if both documents have chunks
    doc1 = "doc-e0b88d32f06db02af102ff4143604dd2"  # GraphRAG R1 paper
    doc2 = "doc-792ee78d563dd146ab2e518df79ba1bf"  # Airport RAG paper
    
    if doc1 in chunks_by_doc and doc2 in chunks_by_doc:
        print(f"[SUCCESS] Both documents have text chunks:")
        print(f"  {doc1}: {len(chunks_by_doc[doc1])} chunks")  
        print(f"  {doc2}: {len(chunks_by_doc[doc2])} chunks")
    else:
        print(f"[ERROR] Missing chunks for one or both documents:")
        print(f"  {doc1}: {'SUCCESS' if doc1 in chunks_by_doc else 'MISSING'}")
        print(f"  {doc2}: {'SUCCESS' if doc2 in chunks_by_doc else 'MISSING'}")
        return
    
    # The issue is likely in the entity extraction phase
    print(f"\nThe issue appears to be in the ENTITY EXTRACTION phase:")
    print(f"- Both documents have text chunks")
    print(f"- Only the first document has entities extracted")
    print(f"- This prevents knowledge graph and community report generation for the second document")
    
    print(f"\nPossible causes:")
    print(f"1. Entity extraction was interrupted or failed after the first document")
    print(f"2. The entity extraction process only processed the first document due to a bug")  
    print(f"3. The second document content didn't trigger entity extraction (unlikely)")
    print(f"4. API limits or errors during entity extraction for the second document")
    
    print(f"\nTo fix this, you need to:")
    print(f"1. Re-run the entity extraction process for the second document")
    print(f"2. Re-generate the knowledge graph to include entities from both documents")
    print(f"3. Re-generate community reports to include both documents")

if __name__ == "__main__":
    asyncio.run(debug_knowledge_graph_generation())