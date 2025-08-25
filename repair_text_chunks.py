#!/usr/bin/env python3
"""
Repair script to fix text chunks data structure mismatch
"""
import json
import os
from pathlib import Path


def repair_text_chunks(session_name):
    """Repair text chunks storage for a given session"""
    session_path = Path(f"sessions/{session_name}")
    
    if not session_path.exists():
        print(f"[ERROR] Session {session_name} not found")
        return False
        
    full_docs_file = session_path / "kv_store_full_docs.json"
    text_chunks_file = session_path / "kv_store_text_chunks.json"
    
    if not full_docs_file.exists():
        print(f"[ERROR] Full docs file not found: {full_docs_file}")
        return False
        
    if not text_chunks_file.exists():
        print(f"[ERROR] Text chunks file not found: {text_chunks_file}")
        return False
        
    print(f"Repairing text chunks for session: {session_name}")
    
    # Load current files
    try:
        with open(full_docs_file, 'r', encoding='utf-8') as f:
            full_docs = json.load(f)
        
        with open(text_chunks_file, 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)
            
        print(f"Loaded {len(full_docs)} full docs and {len(text_chunks)} text chunks")
        
    except Exception as e:
        print(f"[ERROR] Failed to load files: {e}")
        return False
    
    # Create backup
    backup_file = text_chunks_file.with_suffix('.json.backup')
    try:
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, indent=2, ensure_ascii=False)
        print(f"Created backup: {backup_file}")
    except Exception as e:
        print(f"[WARNING] Failed to create backup: {e}")
    
    # Repair text chunks
    repaired_count = 0
    for chunk_id, chunk_data in text_chunks.items():
        if isinstance(chunk_data, dict):
            # Check if it has the wrong structure (content directly instead of data.content)
            if "content" in chunk_data and "data" not in chunk_data:
                # This is the problematic structure - fix it
                repaired_chunk = {
                    "data": {
                        "content": chunk_data["content"],
                        "tokens": chunk_data.get("tokens", len(chunk_data["content"].split())),
                        "chunk_order_index": chunk_data.get("chunk_order_index", 0),
                        "full_doc_id": chunk_data.get("full_doc_id", "unknown")
                    }
                }
                
                # Copy any other fields
                for key, value in chunk_data.items():
                    if key not in ["content", "tokens", "chunk_order_index", "full_doc_id"]:
                        repaired_chunk[key] = value
                        
                text_chunks[chunk_id] = repaired_chunk
                repaired_count += 1
                
            elif "data" in chunk_data and chunk_data["data"] is None:
                # This is the null data case - try to reconstruct from full docs
                doc_id = chunk_data.get("full_doc_id")
                if doc_id and doc_id in full_docs:
                    full_doc = full_docs[doc_id]
                    if isinstance(full_doc, dict) and "content" in full_doc:
                        text_chunks[chunk_id]["data"] = {
                            "content": full_doc["content"][:2000],  # First 2000 chars as chunk
                            "tokens": len(full_doc["content"].split()),
                            "chunk_order_index": 0,
                            "full_doc_id": doc_id
                        }
                        repaired_count += 1
                        
    print(f"Repaired {repaired_count} text chunks")
    
    # Save repaired file
    try:
        with open(text_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Repaired text chunks saved to: {text_chunks_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save repaired file: {e}")
        return False


def main():
    """Main function"""
    print("Text Chunks Repair Tool")
    print("="*50)
    
    # Get session name
    session_name = input("Enter session name to repair (e.g., 1008PM): ").strip()
    
    if not session_name:
        print("[ERROR] No session name provided")
        return
        
    success = repair_text_chunks(session_name)
    
    if success:
        print("\n[SUCCESS] Repair completed!")
        print("You can now try your local query again.")
        print("The warning should be gone and you should see text units being used.")
    else:
        print("\n[FAILED] Repair failed. Check the error messages above.")


if __name__ == "__main__":
    main()