#!/usr/bin/env python3
"""
Debug the text chunks structure issue more carefully
"""
import json
from pathlib import Path

def debug_text_chunks():
    """Debug text chunks in the 1008PM session"""
    print("Debugging text chunks structure...")
    
    session_path = Path("sessions/1008PM")
    text_chunks_file = session_path / "kv_store_text_chunks.json"
    
    if not text_chunks_file.exists():
        print("[ERROR] Text chunks file not found")
        return
        
    try:
        with open(text_chunks_file, 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)
            
        print(f"Found {len(text_chunks)} text chunks")
        
        # Analyze structure of each chunk
        structures = {}
        problematic_chunks = []
        
        for chunk_id, chunk_data in text_chunks.items():
            if not isinstance(chunk_data, dict):
                problematic_chunks.append((chunk_id, f"Not a dict: {type(chunk_data)}"))
                continue
                
            # Get the structure signature
            keys = tuple(sorted(chunk_data.keys()))
            if keys not in structures:
                structures[keys] = []
            structures[keys].append(chunk_id)
            
            # Check for data field issues
            if "data" in chunk_data:
                if chunk_data["data"] is None:
                    problematic_chunks.append((chunk_id, "data field is None"))
                elif not isinstance(chunk_data["data"], dict):
                    problematic_chunks.append((chunk_id, f"data field is not dict: {type(chunk_data['data'])}"))
                else:
                    # Check data content
                    data_keys = list(chunk_data["data"].keys())
                    if "content" not in chunk_data["data"]:
                        problematic_chunks.append((chunk_id, f"data missing content field, has: {data_keys}"))
            else:
                # Check if it's the old direct format
                if "content" in chunk_data and "tokens" in chunk_data:
                    problematic_chunks.append((chunk_id, "Old direct format (content at root level)"))
                else:
                    problematic_chunks.append((chunk_id, f"No data field and not old format. Keys: {list(chunk_data.keys())}"))
        
        # Report structures found
        print("\nStructures found:")
        for keys, chunk_ids in structures.items():
            print(f"  {keys}: {len(chunk_ids)} chunks")
            if len(chunk_ids) <= 3:
                print(f"    Examples: {chunk_ids}")
            else:
                print(f"    Examples: {chunk_ids[:3]} ...")
        
        # Report problematic chunks
        if problematic_chunks:
            print(f"\nFound {len(problematic_chunks)} problematic chunks:")
            for chunk_id, issue in problematic_chunks[:10]:  # Show first 10
                print(f"  {chunk_id}: {issue}")
                
                # Show the actual data for first few
                if len([x for x in problematic_chunks if x == (chunk_id, issue)]) <= 3:
                    chunk = text_chunks[chunk_id]
                    if isinstance(chunk, dict):
                        print(f"    Data: {dict(list(chunk.items())[:3])}...")  # First 3 keys
                    else:
                        print(f"    Data: {chunk}")
        else:
            print("\nNo problematic chunks found in storage")
            
        # Now check if the issue might be in lookup
        print("\nChecking if issue might be in the lookup process...")
        
        # Look for the specific IDs mentioned in warnings
        warning_ids = ["doc-e0b88d32f06db02af102ff4143604dd2", "doc-792ee78d563dd146ab2e518df79ba1bf"]
        
        for doc_id in warning_ids:
            matching_chunks = [cid for cid in text_chunks.keys() if doc_id in str(text_chunks[cid])]
            print(f"  Chunks referencing {doc_id}: {len(matching_chunks)}")
            
            if matching_chunks:
                sample_chunk = text_chunks[matching_chunks[0]]
                print(f"    Sample chunk keys: {list(sample_chunk.keys()) if isinstance(sample_chunk, dict) else 'Not dict'}")
                
                if isinstance(sample_chunk, dict) and "data" in sample_chunk:
                    data = sample_chunk["data"]
                    if data is None:
                        print(f"    Data is None!")
                    elif isinstance(data, dict):
                        print(f"    Data keys: {list(data.keys())}")
                        if "full_doc_id" in data:
                            print(f"    Data full_doc_id: {data['full_doc_id']}")
                    else:
                        print(f"    Data type: {type(data)}")
                        
    except Exception as e:
        print(f"[ERROR] Failed to debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_text_chunks()