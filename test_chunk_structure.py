#!/usr/bin/env python3
"""
Test that the chunk structure fix works correctly
"""
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_chunk_structure():
    """Test that chunks are created with the correct structure"""
    print("Testing chunk structure creation...")
    
    try:
        from nano_graphrag._op import get_chunks, chunking_by_token_size
        
        # Create test documents
        test_docs = {
            "doc-test1": {"content": "This is a test document with some content for chunking."},
            "doc-test2": {"content": "Another test document with different content for verification."}
        }
        
        print(f"Testing with {len(test_docs)} test documents")
        
        # Test chunk creation
        chunks = get_chunks(
            new_docs=test_docs,
            chunk_func=chunking_by_token_size,
            overlap_token_size=10,
            max_token_size=50
        )
        
        print(f"Generated {len(chunks)} chunks")
        
        # Validate structure
        all_valid = True
        required_fields = ["tokens", "content", "full_doc_id", "chunk_order_index"]
        
        for chunk_id, chunk_data in chunks.items():
            print(f"\nChunk ID: {chunk_id}")
            print(f"Chunk structure: {list(chunk_data.keys())}")
            
            # Check if it's a dictionary
            if not isinstance(chunk_data, dict):
                print(f"[ERROR] Chunk is not a dictionary: {type(chunk_data)}")
                all_valid = False
                continue
                
            # Check for required fields
            missing_fields = [field for field in required_fields if field not in chunk_data]
            if missing_fields:
                print(f"[ERROR] Missing required fields: {missing_fields}")
                all_valid = False
            else:
                print("[SUCCESS] All required fields present")
                
            # Check field types
            if "tokens" in chunk_data and not isinstance(chunk_data["tokens"], int):
                print(f"[ERROR] 'tokens' should be int, got {type(chunk_data['tokens'])}")
                all_valid = False
                
            if "content" in chunk_data and not isinstance(chunk_data["content"], str):
                print(f"[ERROR] 'content' should be str, got {type(chunk_data['content'])}")
                all_valid = False
                
            # Show a preview
            content_preview = chunk_data.get("content", "")[:50]
            print(f"Content preview: '{content_preview}{'...' if len(content_preview) == 50 else ''}'")
        
        if all_valid:
            print(f"\n[SUCCESS] All {len(chunks)} chunks have correct TextChunkSchema structure!")
            print("The data structure mismatch issue has been fixed.")
        else:
            print(f"\n[FAIL] Some chunks have incorrect structure.")
            
        return all_valid
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunk_structure()
    if success:
        print("\n[SUCCESS] Chunk structure test passed!")
    else:
        print("\n[FAIL] Chunk structure test failed!")