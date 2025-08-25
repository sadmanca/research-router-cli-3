#!/usr/bin/env python3
"""
Fix incomplete indexing by forcing a complete re-indexing of all documents
This will extract entities from all documents and regenerate the knowledge graph
"""
import asyncio
import json
import sys
import os
import shutil
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def fix_incomplete_indexing():
    """Force a complete re-indexing to extract entities from all documents"""
    print("Fixing incomplete indexing issue...")
    
    session_path = Path("sessions/1008PM")
    
    if not session_path.exists():
        print("[ERROR] Session not found")
        return False
    
    try:
        from nano_graphrag import GraphRAG
        from research_router_cli.utils.config import Config
        
        config = Config()
        if not config.validate_config():
            print("[ERROR] Config validation failed")
            return False
        
        # Check what we have before fixing
        full_docs_file = session_path / "kv_store_full_docs.json"
        community_reports_file = session_path / "kv_store_community_reports.json"
        
        with open(full_docs_file, 'r', encoding='utf-8') as f:
            full_docs = json.load(f)
            
        with open(community_reports_file, 'r', encoding='utf-8') as f:
            community_reports = json.load(f)
        
        print(f"Before fix:")
        print(f"  Documents: {len(full_docs)}")
        print(f"  Community reports: {len(community_reports)}")
        
        # Check which documents the community reports reference
        doc_refs = set()
        for report in community_reports.values():
            if "chunk_ids" in report:
                for chunk_id in report["chunk_ids"]:
                    doc_refs.add(chunk_id)
        print(f"  Documents referenced in community reports: {len(doc_refs)}")
        for doc_ref in doc_refs:
            print(f"    - {doc_ref}")
        
        print(f"\n[INFO] The issue is that entity extraction was incomplete.")
        print(f"[INFO] Only the first document has entities and community reports.")
        print(f"[INFO] We need to force a complete re-indexing.")
        
        # The nuclear option: Clear the knowledge graph related files to force re-indexing
        # BUT preserve the original documents and text chunks
        files_to_backup_and_clear = [
            "vdb_entities.json",
            "kv_store_community_reports.json", 
            "graph_chunk_entity_relation.graphml",
            "output.dashkg.json",
            "_genkg_viz_data.json",
            "output.html",
            # Don't clear these - they contain the source data:
            # "kv_store_full_docs.json", 
            # "kv_store_text_chunks.json",
            # "vdb_chunks.json"
        ]
        
        print(f"\n[INFO] Backing up and clearing knowledge graph files to force re-indexing...")
        backup_dir = session_path / "backup_before_reindex"
        backup_dir.mkdir(exist_ok=True)
        
        for filename in files_to_backup_and_clear:
            file_path = session_path / filename
            if file_path.exists():
                backup_path = backup_dir / filename
                shutil.copy2(file_path, backup_path)
                print(f"  Backed up: {filename}")
                file_path.unlink()  # Delete the original
                print(f"  Cleared: {filename}")
        
        # Also clear the LLM cache to avoid stale entity extractions
        llm_cache_file = session_path / "kv_store_llm_response_cache.json"
        if llm_cache_file.exists():
            backup_cache = backup_dir / "kv_store_llm_response_cache.json"
            shutil.copy2(llm_cache_file, backup_cache)
            llm_cache_file.unlink()
            print(f"  Cleared LLM cache to force re-extraction")
        
        print(f"\n[INFO] Now triggering complete re-indexing...")
        
        # Create GraphRAG instance - this should trigger re-indexing
        working_dir = str(session_path)
        llm_config = config.get_llm_config()
        
        # Force disable cache to ensure fresh processing
        graphrag = GraphRAG(
            working_dir=working_dir,
            enable_llm_cache=False,  # Disable cache to force fresh processing
            **llm_config
        )
        
        print(f"[INFO] Re-indexing in progress...")
        print(f"[INFO] This will extract entities from all documents and rebuild the knowledge graph...")
        
        # The indexing should happen automatically when we create the GraphRAG instance
        # But we can also manually trigger it by doing a query (which requires the index)
        
        try:
            # Test query to ensure indexing is complete
            from nano_graphrag import QueryParam
            
            print(f"[INFO] Testing indexing with a simple query...")
            result = await graphrag.aquery(
                "what documents are in this knowledge base?", 
                param=QueryParam(mode="local", only_need_context=True)
            )
            
            if result and len(result) > 100:
                print(f"[SUCCESS] Re-indexing completed successfully!")
                print(f"[SUCCESS] Query returned {len(result)} characters of context")
                
                # Check the results
                with open(community_reports_file, 'r', encoding='utf-8') as f:
                    new_community_reports = json.load(f)
                
                print(f"\nAfter fix:")
                print(f"  Community reports: {len(new_community_reports)}")
                
                # Check which documents the new community reports reference
                new_doc_refs = set()
                for report in new_community_reports.values():
                    if "chunk_ids" in report:
                        for chunk_id in report["chunk_ids"]:
                            new_doc_refs.add(chunk_id)
                
                print(f"  Documents referenced in community reports: {len(new_doc_refs)}")
                for doc_ref in sorted(new_doc_refs):
                    print(f"    - {doc_ref}")
                
                if len(new_doc_refs) > 1:
                    print(f"\n[SUCCESS] Fix successful! Now have community reports from multiple documents!")
                    return True
                else:
                    print(f"\n[WARNING] Still only have community reports from one document")
                    return False
            else:
                print(f"[ERROR] Re-indexing may have failed - query returned insufficient context")
                return False
                
        except Exception as e:
            print(f"[ERROR] Re-indexing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"[ERROR] Fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(fix_incomplete_indexing())
    if success:
        print("\n[SUCCESS] Incomplete indexing fix completed!")
        print("Community reports should now include content from all documents.")
    else:
        print("\n[FAILED] Fix failed. Check the error messages above.")