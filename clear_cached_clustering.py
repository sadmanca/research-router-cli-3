#!/usr/bin/env python3
"""
Clear cached clustering data to force re-clustering with the fixed algorithm
"""
import networkx as nx
from pathlib import Path
import shutil

def clear_cached_clustering():
    """Clear clustering data from the GraphML file"""
    print("Clearing cached clustering data...")
    
    session_path = Path("sessions/1008PM")
    graph_file = session_path / "graph_chunk_entity_relation.graphml"
    
    if not graph_file.exists():
        print(f"[ERROR] Graph file not found: {graph_file}")
        return False
    
    try:
        # Backup the original file
        backup_file = graph_file.with_suffix('.graphml.backup_before_cluster_clear')
        shutil.copy2(graph_file, backup_file)
        print(f"Created backup: {backup_file}")
        
        # Load the graph
        G = nx.read_graphml(graph_file)
        print(f"Loaded graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Remove clustering data from all nodes
        nodes_with_clusters = 0
        for node_id in G.nodes():
            if "clusters" in G.nodes[node_id]:
                del G.nodes[node_id]["clusters"]
                nodes_with_clusters += 1
        
        print(f"Removed cluster data from {nodes_with_clusters} nodes")
        
        # Save the cleaned graph
        nx.write_graphml(G, graph_file)
        print(f"Saved cleaned graph to {graph_file}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to clear clustering data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = clear_cached_clustering()
    if success:
        print(f"\n[SUCCESS] Cached clustering data cleared!")
        print(f"Next GraphRAG operation will trigger re-clustering with the fixed algorithm.")
    else:
        print(f"\n[FAILED] Could not clear cached clustering data.")