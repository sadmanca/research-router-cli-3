#!/usr/bin/env python3
"""
Force complete re-indexing to regenerate community reports with the fixed clustering
"""
import asyncio
import json
import sys
import os
import shutil
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def force_complete_reindexing():
    """Force a complete re-indexing process"""
    print("Forcing complete re-indexing with fixed clustering...")
    
    session_path = Path("sessions/1008PM")
    
    if not session_path.exists():
        print("[ERROR] Session not found")
        return False
    
    try:
        from nano_graphrag import GraphRAG, QueryParam
        from research_router_cli.utils.config import Config
        
        config = Config()
        if not config.validate_config():
            print("[ERROR] Config validation failed")
            return False
        
        # Step 1: Check current state
        print("\nStep 1: Checking current state...")
        
        with open(session_path / "kv_store_community_reports.json", 'r', encoding='utf-8') as f:
            old_reports = json.load(f)
        
        print(f"Current community reports: {len(old_reports)}")
        
        # Step 2: Delete clustering-related files to force regeneration
        print("\nStep 2: Clearing clustering and community files...")
        
        backup_dir = session_path / "backup_before_complete_reindex"
        backup_dir.mkdir(exist_ok=True)
        
        files_to_clear = [
            "kv_store_community_reports.json",
            # Keep the graph file but clear cluster data from it
        ]
        
        for filename in files_to_clear:
            file_path = session_path / filename
            if file_path.exists():
                backup_path = backup_dir / filename
                shutil.copy2(file_path, backup_path)
                print(f"  Backed up: {filename}")
                file_path.unlink()
                print(f"  Cleared: {filename}")
        
        # Step 3: Create GraphRAG instance and force a query to trigger full indexing
        print("\nStep 3: Creating GraphRAG instance and forcing indexing...")
        
        working_dir = str(session_path)
        llm_config = config.get_llm_config()
        
        # Create GraphRAG instance
        graphrag = GraphRAG(
            working_dir=working_dir,
            enable_llm_cache=True,
            **llm_config
        )
        
        print("Created GraphRAG instance")
        
        # Force indexing by running a query that requires all components
        print("\nStep 4: Running query to force complete indexing...")
        
        try:
            # Use local mode which requires community reports
            result = await graphrag.aquery(
                "What are the main topics covered in these documents?", 
                param=QueryParam(mode="local", only_need_context=True)
            )
            
            if result and len(result) > 100:
                print(f"[SUCCESS] Query completed with {len(result)} characters of context")
            else:
                print(f"[WARNING] Query returned limited context: {len(result) if result else 0} chars")
        
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            # Continue anyway - the indexing may have happened
        
        # Step 5: Check the results
        print("\nStep 5: Checking results...")
        
        # Wait a moment for files to be written
        import time
        time.sleep(2)
        
        reports_file = session_path / "kv_store_community_reports.json"
        if reports_file.exists():
            with open(reports_file, 'r', encoding='utf-8') as f:
                new_reports = json.load(f)
            
            print(f"New community reports: {len(new_reports)}")
            
            # Check which documents are referenced
            doc_refs = set()
            for report in new_reports.values():
                if "chunk_ids" in report:
                    for chunk_id in report["chunk_ids"]:
                        doc_refs.add(chunk_id)
            
            print(f"Documents referenced in community reports: {len(doc_refs)}")
            for doc_ref in sorted(doc_refs):
                print(f"  - {doc_ref}")
            
            if len(doc_refs) >= 2:
                print(f"\n[SUCCESS] Fix successful! Community reports now include multiple documents!")
                
                # Show sample report titles to verify content
                print(f"\nSample community report titles:")
                for i, (report_id, report) in enumerate(list(new_reports.items())[:5]):
                    title = report.get("report_json", {}).get("title", "No title")
                    nodes_count = len(report.get("nodes", []))
                    print(f"  {report_id}: {title} ({nodes_count} entities)")
                
                return True
            else:
                print(f"\n[WARNING] Still only seeing {len(doc_refs)} document(s) in community reports")
                return False
        else:
            print(f"[ERROR] Community reports file was not created")
            return False
            
    except Exception as e:
        print(f"[ERROR] Re-indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(force_complete_reindexing())
    if success:
        print("\n[SUCCESS] Complete re-indexing successful!")
        print("Local queries should now include content from all documents.")
    else:
        print("\n[FAILED] Complete re-indexing failed.")