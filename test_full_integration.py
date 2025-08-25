#!/usr/bin/env python3
"""
Full integration test to verify the chunk structure fix works end-to-end
"""
import asyncio
import sys
import os
import tempfile
import json
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_full_integration():
    """Test complete document insertion and local query with chunk structure fix"""
    print("Starting full integration test...")
    
    # Create a temporary test session
    temp_session = "test_integration"
    session_path = Path(f"sessions/{temp_session}")
    
    try:
        # Clean up any existing test session
        if session_path.exists():
            import shutil
            shutil.rmtree(session_path)
            
        # Create session directory
        session_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Created test session: {temp_session}")
        
        # Test document insertion
        from nano_graphrag import GraphRAG, QueryParam
        from research_router_cli.utils.config import Config
        
        config = Config()
        if not config.validate_config():
            print("[ERROR] Config validation failed")
            return False
            
        llm_config = config.get_llm_config()
        
        # Create GraphRAG instance
        graphrag = GraphRAG(
            working_dir=str(session_path),
            enable_llm_cache=True,
            enable_local=True,
            enable_naive_rag=True,
            **llm_config
        )
        
        print("Created GraphRAG instance")
        
        # Insert test documents
        test_docs = [
            "GraphRAG-R1 is an advanced framework for Large Language Models that enhances retrieval capabilities.",
            "The framework uses graph-based retrieval to improve performance on complex queries requiring multi-hop reasoning.",
            "GraphRAG-R1 implements sophisticated methodologies including phase-dependent training and hybrid retrieval."
        ]
        
        print(f"Inserting {len(test_docs)} test documents...")
        
        try:
            await graphrag.ainsert(test_docs)
            print("[SUCCESS] Documents inserted successfully")
        except Exception as e:
            print(f"[ERROR] Document insertion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check that chunk files were created with correct structure
        text_chunks_file = session_path / "kv_store_text_chunks.json"
        if not text_chunks_file.exists():
            print("[ERROR] Text chunks file was not created")
            return False
            
        try:
            with open(text_chunks_file, 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
                
            print(f"Found {len(text_chunks)} text chunks in storage")
            
            # Validate structure
            required_fields = ["tokens", "content", "full_doc_id", "chunk_order_index"]
            structure_errors = []
            
            for chunk_id, chunk_data in text_chunks.items():
                if not isinstance(chunk_data, dict):
                    structure_errors.append(f"Chunk {chunk_id} is not a dict: {type(chunk_data)}")
                    continue
                    
                missing_fields = [field for field in required_fields if field not in chunk_data]
                if missing_fields:
                    structure_errors.append(f"Chunk {chunk_id} missing fields: {missing_fields}")
                    
            if structure_errors:
                print("[ERROR] Chunk structure validation failed:")
                for error in structure_errors:
                    print(f"  - {error}")
                return False
            else:
                print("[SUCCESS] All chunks have correct structure")
                
        except Exception as e:
            print(f"[ERROR] Failed to validate chunk structure: {e}")
            return False
        
        # Test local query with context only (to avoid API quota issues)
        print("Testing local query context retrieval...")
        try:
            context = await graphrag.aquery(
                "what is GraphRAG-R1?", 
                param=QueryParam(mode="local", only_need_context=True)
            )
            
            if context and len(context) > 100:
                print(f"[SUCCESS] Local query returned {len(context)} characters of context")
                print("Context preview:")
                print("-" * 50)
                print(context[:200] + "..." if len(context) > 200 else context)
                print("-" * 50)
                return True
            else:
                print(f"[ERROR] Local query returned insufficient context: {len(context) if context else 0} chars")
                return False
                
        except Exception as e:
            print(f"[ERROR] Local query failed: {e}")
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
    success = asyncio.run(test_full_integration())
    if success:
        print("\n[SUCCESS] Full integration test passed!")
        print("The chunk structure fix is working correctly.")
    else:
        print("\n[FAIL] Full integration test failed!")
        print("There are still issues that need to be fixed.")