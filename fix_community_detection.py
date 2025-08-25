#!/usr/bin/env python3
"""
Fix community detection to include all connected components, not just the largest one
"""
import shutil
import os
from pathlib import Path

def fix_community_detection():
    """Fix the _leiden_clustering method to process all connected components"""
    print("Fixing community detection to include all connected components...")
    
    networkx_file = Path("nano-graphrag/nano_graphrag/_storage/gdb_networkx.py")
    
    if not networkx_file.exists():
        print(f"[ERROR] NetworkX storage file not found: {networkx_file}")
        return False
    
    # Create backup
    backup_file = networkx_file.with_suffix('.py.backup')
    shutil.copy2(networkx_file, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Read the original file
    with open(networkx_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the _leiden_clustering method
    old_method = '''async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)'''
    
    # New method that processes all connected components
    new_method = '''async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        # FIXED: Process all connected components, not just the largest one
        # This ensures entities from all documents are included in clustering
        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        cluster_offset = 0  # Offset to ensure unique cluster IDs across components
        
        # Get all connected components
        connected_components = list(nx.connected_components(self._graph))
        logger.info(f"Processing {len(connected_components)} connected components for clustering")
        
        for comp_idx, component_nodes in enumerate(connected_components):
            # Extract subgraph for this component
            component_subgraph = self._graph.subgraph(component_nodes).copy()
            
            # Apply stabilization to this component
            component_graph = NetworkXStorage._stabilize_graph(component_subgraph)
            
            # Apply node mapping similar to stable_largest_connected_component
            node_mapping = {node: html.unescape(node.upper().strip()) for node in component_graph.nodes()}
            component_graph = nx.relabel_nodes(component_graph, node_mapping)
            
            logger.info(f"Component {comp_idx}: {component_graph.number_of_nodes()} nodes, {component_graph.number_of_edges()} edges")
            
            # Skip very small components (single nodes with no edges)
            if component_graph.number_of_nodes() < 2:
                logger.info(f"Skipping component {comp_idx}: too small for clustering")
                continue
            
            try:
                # Run hierarchical Leiden clustering on this component
                community_mapping = hierarchical_leiden(
                    component_graph,
                    max_cluster_size=self.global_config["max_graph_cluster_size"],
                    random_seed=self.global_config["graph_cluster_seed"],
                )
                
                # Process the clustering results with offset cluster IDs
                for partition in community_mapping:
                    level_key = partition.level
                    cluster_id = f"{partition.cluster + cluster_offset}"  # Add offset for unique IDs
                    
                    # Map back to original node IDs (reverse the node_mapping)
                    original_node_id = None
                    for orig_id, mapped_id in node_mapping.items():
                        if mapped_id == partition.node:
                            original_node_id = orig_id
                            break
                    
                    if original_node_id:
                        node_communities[original_node_id].append(
                            {"level": level_key, "cluster": cluster_id}
                        )
                        __levels[level_key].add(cluster_id)
                
                # Update cluster offset for next component
                if community_mapping:
                    max_cluster_in_component = max(p.cluster for p in community_mapping)
                    cluster_offset += max_cluster_in_component + 1
                    
            except Exception as e:
                logger.warning(f"Failed to cluster component {comp_idx}: {e}")
                continue
        
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        logger.info(f"Total nodes with community assignments: {len(node_communities)}")
        self._cluster_data_to_subgraphs(node_communities)'''
    
    # Replace the method
    if old_method in content:
        new_content = content.replace(old_method, new_method)
        
        # Write the fixed content
        with open(networkx_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"[SUCCESS] Fixed community detection in {networkx_file}")
        print(f"[INFO] The fix ensures all connected components are processed for clustering")
        print(f"[INFO] This will include entities from all documents in community reports")
        return True
    else:
        print(f"[ERROR] Could not find the expected _leiden_clustering method to replace")
        print(f"[ERROR] The method signature may have changed")
        return False

if __name__ == "__main__":
    success = fix_community_detection()
    if success:
        print("\n[SUCCESS] Community detection fix applied!")
        print("Now you need to regenerate the session to apply the fix:")
        print("1. Delete the existing community reports")
        print("2. Re-run the clustering process")
    else:
        print("\n[FAILED] Could not apply the fix")