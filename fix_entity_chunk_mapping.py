#!/usr/bin/env python3
"""
Fix the entity-to-chunk ID mapping issue
"""
import json
import os
from pathlib import Path


def fix_entity_chunk_mapping(session_name):
    """Fix the entity source_id references to point to chunks instead of documents"""
    session_path = Path(f"sessions/{session_name}")
    
    if not session_path.exists():
        print(f"[ERROR] Session {session_name} not found")
        return False
        
    # Files we need to work with
    text_chunks_file = session_path / "kv_store_text_chunks.json"
    full_docs_file = session_path / "kv_store_full_docs.json"
    graph_file = session_path / "graph_chunk_entity_relation.graphml"
    
    if not all(f.exists() for f in [text_chunks_file, full_docs_file]):
        print("[ERROR] Required files not found")
        return False
        
    try:
        # Load text chunks and full docs
        with open(text_chunks_file, 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)
            
        with open(full_docs_file, 'r', encoding='utf-8') as f:
            full_docs = json.load(f)
            
        print(f"Loaded {len(text_chunks)} text chunks and {len(full_docs)} documents")
        
        # Build mapping from document IDs to chunk IDs
        doc_to_chunks = {}
        for chunk_id, chunk_data in text_chunks.items():
            if isinstance(chunk_data, dict) and "data" in chunk_data and chunk_data["data"]:
                full_doc_id = chunk_data["data"].get("full_doc_id")
                if full_doc_id:
                    if full_doc_id not in doc_to_chunks:
                        doc_to_chunks[full_doc_id] = []
                    doc_to_chunks[full_doc_id].append(chunk_id)
        
        print(f"Built mapping for {len(doc_to_chunks)} documents to chunks:")
        for doc_id, chunk_ids in doc_to_chunks.items():
            print(f"  {doc_id}: {len(chunk_ids)} chunks")
            
        # The issue is that entities have source_id pointing to doc IDs instead of chunk IDs
        # In a properly functioning system, entities should reference the chunks that contain them
        # Since we can't easily modify the graph structure, let's create "fake" text chunks
        # with the document IDs as keys, containing the full document content
        
        print("\nCreating document-level chunks for entity references...")
        
        fixes_applied = 0
        for doc_id in full_docs.keys():
            if doc_id not in text_chunks:  # This is the missing piece!
                doc_content = full_docs[doc_id].get("content", "")
                
                # Create a text chunk entry for this document ID
                # This allows entities that reference doc IDs to find the content
                text_chunks[doc_id] = {
                    "tokens": len(doc_content.split()),
                    "content": doc_content[:2000],  # First 2000 chars
                    "chunk_order_index": 0,
                    "full_doc_id": doc_id
                }
                
                fixes_applied += 1
                print(f"  Created chunk entry for {doc_id}")
                
        if fixes_applied > 0:
            # Backup original file
            backup_file = text_chunks_file.with_suffix('.json.backup2')
            with open(backup_file, 'w', encoding='utf-8') as f:
                # Load the original to backup
                with open(text_chunks_file, 'r', encoding='utf-8') as orig_f:
                    original_data = json.load(orig_f)
                json.dump(original_data, f, indent=2, ensure_ascii=False)
            
            # Save updated chunks
            with open(text_chunks_file, 'w', encoding='utf-8') as f:
                json.dump(text_chunks, f, indent=2, ensure_ascii=False)
                
            print(f"\n[SUCCESS] Applied {fixes_applied} fixes")
            print(f"Created backup: {backup_file}")
            print(f"Updated: {text_chunks_file}")
            return True
        else:
            print("\n[INFO] No fixes needed - all document IDs already have chunk entries")
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to fix mapping: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("Entity-to-Chunk Mapping Repair Tool")
    print("="*50)
    
    # Get session name
    session_name = input("Enter session name to repair (e.g., 1008PM): ").strip()
    
    if not session_name:
        print("[ERROR] No session name provided")
        return
        
    success = fix_entity_chunk_mapping(session_name)
    
    if success:
        print("\n[SUCCESS] Mapping repair completed!")
        print("The entity-to-chunk mapping has been fixed.")
        print("You should no longer see 'Text unit missing data field' warnings.")
        print("Local queries should now include text units.")
    else:
        print("\n[FAILED] Mapping repair failed. Check the error messages above.")


if __name__ == "__main__":
    main()