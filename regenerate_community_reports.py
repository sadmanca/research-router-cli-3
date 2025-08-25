#!/usr/bin/env python3
"""
Regenerate community reports using the fixed clustering algorithm
"""
import asyncio
import json
import sys
import os
import shutil
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def regenerate_community_reports():
    """Regenerate community reports with the fixed clustering"""
    print("Regenerating community reports with fixed clustering...")
    
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
        
        # Step 1: Backup and clear community-related files
        print("\nStep 1: Backing up and clearing existing community reports...")
        
        backup_dir = session_path / "backup_before_cluster_fix"
        backup_dir.mkdir(exist_ok=True)
        
        files_to_clear = [
            "kv_store_community_reports.json",
        ]
        
        for filename in files_to_clear:
            file_path = session_path / filename
            if file_path.exists():
                backup_path = backup_dir / filename
                shutil.copy2(file_path, backup_path)
                print(f"  Backed up: {filename}")
                file_path.unlink()
                print(f"  Cleared: {filename}")
        
        # Step 2: Force re-clustering by creating GraphRAG instance
        print("\nStep 2: Re-running clustering with fixed algorithm...")
        
        working_dir = str(session_path)
        llm_config = config.get_llm_config()
        
        # Create GraphRAG instance to trigger clustering
        graphrag = GraphRAG(
            working_dir=working_dir,
            enable_llm_cache=True,  # Keep cache for efficiency
            **llm_config
        )
        
        print("Created GraphRAG instance - clustering should be triggered automatically")
        
        # Step 3: Force the clustering process by accessing graph storage
        print("\nStep 3: Explicitly running clustering...")
        
        # Access the graph storage to ensure clustering runs
        graph_storage = graphrag.chunk_entity_relation_graph
        
        # Run clustering explicitly 
        from dataclasses import asdict
        await graph_storage.clustering(graphrag.graph_cluster_algorithm)
        print("Clustering completed successfully")
        
        # Get community schema to trigger community report generation
        community_schema = await graph_storage.community_schema()
        print(f"Generated community schema with {len(community_schema)} communities")
        
        # Step 4: Generate community reports
        print("\nStep 4: Generating community reports...")
        
        # This should trigger the community report generation process
        from nano_graphrag._op import generate_community_report
        
        community_reports = {}
        for community_id, community_data in community_schema.items():
            try:
                print(f"Generating report for community {community_id}...")
                
                # Generate the community report
                report_result = await generate_community_report(
                    community_schema={community_id: community_data},
                    global_config=asdict(graphrag)
                )
                
                if report_result and len(report_result) > 0:
                    community_reports[community_id] = report_result[0]
                    print(f"  Generated report for community {community_id}")
                else:
                    print(f"  No report generated for community {community_id}")
                    
            except Exception as e:
                print(f"  Failed to generate report for community {community_id}: {e}")
                continue
        
        # Step 5: Save the community reports
        print(f"\nStep 5: Saving {len(community_reports)} community reports...")
        
        reports_file = session_path / "kv_store_community_reports.json"
        with open(reports_file, 'w', encoding='utf-8') as f:
            json.dump(community_reports, f, indent=2, ensure_ascii=False)
        
        print(f"Saved community reports to {reports_file}")
        
        # Step 6: Verify the results
        print("\nStep 6: Verifying results...")
        
        # Check which documents are now referenced in community reports
        doc_refs = set()
        for report in community_reports.values():
            if "chunk_ids" in report:
                for chunk_id in report["chunk_ids"]:
                    doc_refs.add(chunk_id)
        
        print(f"Community reports now reference {len(doc_refs)} documents:")
        for doc_ref in sorted(doc_refs):
            print(f"  - {doc_ref}")
        
        if len(doc_refs) >= 2:
            print(f"\n[SUCCESS] Fix successful! Community reports now include multiple documents!")
            return True
        else:
            print(f"\n[WARNING] Still only seeing {len(doc_refs)} document(s) in community reports")
            return False
            
    except Exception as e:
        print(f"[ERROR] Regeneration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(regenerate_community_reports())
    if success:
        print("\n[SUCCESS] Community reports regenerated successfully!")
        print("Local queries should now include content from all documents.")
    else:
        print("\n[FAILED] Community reports regeneration failed.")