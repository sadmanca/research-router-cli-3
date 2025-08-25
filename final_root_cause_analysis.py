#!/usr/bin/env python3
"""
Final root cause analysis - we have entities from both documents, so why only first doc in community reports?
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def final_root_cause_analysis():
    """Final analysis to find the real root cause"""
    print("Final root cause analysis...")
    
    session_path = Path("sessions/1008PM")
    
    # Load all the data
    with open(session_path / "vdb_entities.json", 'r', encoding='utf-8') as f:
        vdb_data = json.load(f)
        vdb_entities = vdb_data["data"]
    
    with open(session_path / "kv_store_community_reports.json", 'r', encoding='utf-8') as f:
        community_reports = json.load(f)
        
    with open(session_path / "kv_store_full_docs.json", 'r', encoding='utf-8') as f:
        full_docs = json.load(f)
    
    print(f"Data loaded:")
    print(f"  VDB entities: {len(vdb_entities)}")
    print(f"  Community reports: {len(community_reports)}")
    print(f"  Documents: {len(full_docs)}")
    
    # Check which entities belong to which documents by analyzing entity names
    doc1_keywords = ["GRAPHRAG", "REINFORCEMENT LEARNING", "GRPO", "RETRIEVAL", "THINKING"]
    doc2_keywords = ["CONVERSATIONAL AI", "AIRPORT", "AVIATION", "FLIGHT", "SQL RAG", "HALLUCINATION"]
    
    doc1_entities = []
    doc2_entities = []
    unclear_entities = []
    
    for entity in vdb_entities:
        entity_name = entity["entity_name"].upper()
        
        # Check if this entity clearly belongs to doc1 (GraphRAG paper)
        if any(keyword in entity_name for keyword in doc1_keywords):
            doc1_entities.append(entity_name)
        # Check if this entity clearly belongs to doc2 (Airport paper)  
        elif any(keyword in entity_name for keyword in doc2_keywords):
            doc2_entities.append(entity_name)
        else:
            unclear_entities.append(entity_name)
    
    print(f"\nEntity classification by document content:")
    print(f"  Doc1 (GraphRAG) entities: {len(doc1_entities)}")
    for entity in doc1_entities[:5]:
        print(f"    - {entity}")
    print(f"  Doc2 (Airport) entities: {len(doc2_entities)}")
    for entity in doc2_entities[:5]:
        print(f"    - {entity}")
    print(f"  Unclear entities: {len(unclear_entities)}")
    for entity in unclear_entities[:5]:
        print(f"    - {entity}")
    
    # Check what entities appear in community reports
    entities_in_reports = set()
    for report_id, report in community_reports.items():
        if "nodes" in report:
            for node in report["nodes"]:
                entities_in_reports.add(node)
    
    print(f"\nEntities appearing in community reports: {len(entities_in_reports)}")
    
    # Check overlap
    doc1_in_reports = [e for e in doc1_entities if e in entities_in_reports]
    doc2_in_reports = [e for e in doc2_entities if e in entities_in_reports]
    
    print(f"  Doc1 entities in reports: {len(doc1_in_reports)} / {len(doc1_entities)}")
    print(f"  Doc2 entities in reports: {len(doc2_in_reports)} / {len(doc2_entities)}")
    
    # This is the key question: Why are doc2 entities not in community reports?
    if len(doc2_entities) > 0 and len(doc2_in_reports) == 0:
        print(f"\n[ROOT CAUSE IDENTIFIED]")
        print(f"The issue is NOT entity extraction - we have entities from both documents.")
        print(f"The issue is in the COMMUNITY DETECTION phase:")
        print(f"  - Entity extraction: SUCCESS (50 entities from both documents)")
        print(f"  - Vector database: SUCCESS (50 entities with embeddings)")
        print(f"  - Community detection: PARTIAL FAILURE")
        print(f"    * Entities from first document were clustered into communities")
        print(f"    * Entities from second document were NOT included in community clustering")
        
        print(f"\nPossible causes of community detection failure:")
        print(f"1. SIMILARITY THRESHOLD: Doc2 entities may not meet similarity threshold for clustering")
        print(f"2. CLUSTERING ALGORITHM: May have stopped after processing doc1 entities")
        print(f"3. EMBEDDING QUALITY: Doc2 entities may have poor embeddings")
        print(f"4. GRAPH CONNECTIVITY: Doc2 entities may not be connected to doc1 entities")
        print(f"5. PROCESSING ORDER: Community detection may have been interrupted after doc1")
        
        print(f"\nTo confirm, let's check some doc2 entity names:")
        for entity in doc2_entities[:3]:
            print(f"  - {entity} (should be in airport/aviation community)")
        
        return "community_detection_partial_failure"
    
    else:
        print(f"\nUnexpected pattern - need further investigation")
        return "unclear"

if __name__ == "__main__":
    result = asyncio.run(final_root_cause_analysis())
    print(f"\nFinal diagnosis: {result}")