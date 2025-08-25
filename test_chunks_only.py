#!/usr/bin/env python3
"""
Test only the chunk creation and storage part (no entity extraction)
"""
import asyncio
import sys
import os
import json
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_chunks_only():
    """Test just the chunk creation and storage without entity extraction"""
    print("Testing chunk creation and storage...")
    
    temp_session = "test_chunks_only"
    session_path = Path(f"sessions/{temp_session}")
    
    try:
        # Clean up any existing test session
        if session_path.exists():
            import shutil
            shutil.rmtree(session_path)
            
        session_path.mkdir(parents=True, exist_ok=True)
        print(f"Created test session: {temp_session}")
        
        # Create storage instances directly
        from nano_graphrag._storage import JsonKVStorage
        from nano_graphrag._op import get_chunks
        from nano_graphrag._utils import compute_mdhash_id
        
        # Create text chunks storage
        text_chunks_storage = JsonKVStorage(
            namespace="text_chunks",
            global_config={"working_dir": str(session_path)}
        )
        
        # Create test documents
        new_docs = {
            "doc-test1": {"content": "GraphRAG-R1 is an advanced framework for Large Language Models that enhances retrieval capabilities. It uses sophisticated algorithms."},
            "doc-test2": {"content": "The framework implements hybrid graph-textual retrieval mechanisms. This enables better performance on complex queries."}
        }
        
        print(f"Processing {len(new_docs)} test documents...")
        
        # Create chunks using the fixed get_chunks function
        inserting_chunks = get_chunks(
            new_docs=new_docs,
            overlap_token_size=10,
            max_token_size=50
        )
        
        print(f"Generated {len(inserting_chunks)} chunks")
        
        # Store chunks in the KV storage
        await text_chunks_storage.upsert(inserting_chunks)
        print("Chunks stored in KV storage")
        
        # Commit the data to file
        await text_chunks_storage.index_done_callback()
        print("Data committed to file")
        
        # Verify chunks were stored correctly by reading them back
        all_chunk_ids = list(inserting_chunks.keys())
        retrieved_chunks = await text_chunks_storage.get_by_ids(all_chunk_ids)
        
        print(f"Retrieved {len(retrieved_chunks)} chunks from storage")
        
        # Validate structure of retrieved chunks
        required_fields = ["tokens", "content", "full_doc_id", "chunk_order_index"]
        structure_errors = []
        
        for i, (chunk_id, stored_chunk, retrieved_chunk) in enumerate(zip(all_chunk_ids, inserting_chunks.values(), retrieved_chunks)):
            print(f"\nChunk {i+1}: {chunk_id[:20]}...")
            
            # Check retrieved chunk
            if retrieved_chunk is None:
                structure_errors.append(f"Retrieved chunk {chunk_id} is None")
                continue
                
            if not isinstance(retrieved_chunk, dict):
                structure_errors.append(f"Retrieved chunk {chunk_id} is not a dict: {type(retrieved_chunk)}")
                continue
                
            # Check for required fields
            missing_fields = [field for field in required_fields if field not in retrieved_chunk]
            if missing_fields:
                structure_errors.append(f"Retrieved chunk {chunk_id} missing fields: {missing_fields}")
            else:
                print(f"  [SUCCESS] All required fields present: {list(retrieved_chunk.keys())}")
                
            # Check field types
            if "tokens" in retrieved_chunk and not isinstance(retrieved_chunk["tokens"], int):
                structure_errors.append(f"Retrieved chunk {chunk_id} 'tokens' should be int, got {type(retrieved_chunk['tokens'])}")
                
            if "content" in retrieved_chunk and not isinstance(retrieved_chunk["content"], str):
                structure_errors.append(f"Retrieved chunk {chunk_id} 'content' should be str, got {type(retrieved_chunk['content'])}")
                
            # Show content preview
            content = retrieved_chunk.get("content", "")
            print(f"  Content: '{content[:60]}{'...' if len(content) > 60 else ''}'")
        
        # Check if storage file was created correctly
        text_chunks_file = session_path / "kv_store_text_chunks.json"
        if text_chunks_file.exists():
            try:
                with open(text_chunks_file, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                print(f"\nStorage file contains {len(stored_data)} chunks")
                
                # Verify file structure
                for chunk_id, chunk_data in stored_data.items():
                    if not isinstance(chunk_data, dict):
                        structure_errors.append(f"File chunk {chunk_id} is not a dict: {type(chunk_data)}")
                        continue
                        
                    # Check if it has the wrong nested structure
                    if "data" in chunk_data and isinstance(chunk_data["data"], dict):
                        structure_errors.append(f"File chunk {chunk_id} has incorrect nested 'data' structure")
                        
            except Exception as e:
                structure_errors.append(f"Failed to read storage file: {e}")
        else:
            structure_errors.append("Storage file was not created")
        
        if structure_errors:
            print("\n[ERROR] Structure validation failed:")
            for error in structure_errors:
                print(f"  - {error}")
            return False
        else:
            print("\n[SUCCESS] All chunks have correct structure!")
            print("Chunk creation and storage is working correctly.")
            return True
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up test session
        try:
            if session_path.exists():
                import shutil
                shutil.rmtree(session_path)
                print(f"Cleaned up test session: {temp_session}")
        except Exception as e:
            print(f"Warning: Failed to clean up test session: {e}")

if __name__ == "__main__":
    success = asyncio.run(test_chunks_only())
    if success:
        print("\n[SUCCESS] Chunks-only test passed!")
        print("The chunk structure fix is verified to work correctly.")
    else:
        print("\n[FAIL] Chunks-only test failed!")
        print("There are still structural issues that need to be fixed.")