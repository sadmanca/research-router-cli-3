#!/usr/bin/env python3
"""
Diagnose the community detection issue by examining the graph connectivity
"""
import asyncio
import json
import sys
import os
import networkx as nx
from pathlib import Path
from xml.etree import ElementTree as ET

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def diagnose_community_detection_issue():
    """Diagnose why community detection is failing for second document entities"""
    print("Diagnosing community detection issue...")
    
    session_path = Path("sessions/1008PM")
    graph_file = session_path / "graph_chunk_entity_relation.graphml"
    
    # Load and analyze the GraphML file
    try:
        tree = ET.parse(graph_file)
        root = tree.getroot()
        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}
        
        # Parse nodes and edges
        nodes = {}
        edges = []
        
        # Get key mappings first
        key_mappings = {}
        for key in root.findall(".//graphml:key", ns):
            key_id = key.get("id")
            key_name = key.get("attr.name")
            key_mappings[key_id] = key_name
        
        print(f"Key mappings: {key_mappings}")
        
        # Parse nodes
        for node in root.findall(".//graphml:node", ns):
            node_id = node.get("id")
            node_data = {"id": node_id}
            
            for data in node.findall("graphml:data", ns):
                key = data.get("key")
                value = data.text
                if key in key_mappings:
                    node_data[key_mappings[key]] = value
            
            nodes[node_id] = node_data
        
        # Parse edges
        for edge in root.findall(".//graphml:edge", ns):
            source = edge.get("source")
            target = edge.get("target")
            edge_data = {"source": source, "target": target}
            
            for data in edge.findall("graphml:data", ns):
                key = data.get("key")
                value = data.text
                if key in key_mappings:
                    edge_data[key_mappings[key]] = value
            
            edges.append(edge_data)
        
        print(f"Loaded {len(nodes)} nodes and {len(edges)} edges from GraphML")
        
        # Classify nodes by document
        doc1_nodes = []
        doc2_nodes = []
        
        for node_id, node_data in nodes.items():
            source_id = node_data.get("source_id", "")
            if "doc-e0b88d32f06db02af102ff4143604dd2" in str(source_id):
                doc1_nodes.append(node_id)
            elif "doc-792ee78d563dd146ab2e518df79ba1bf" in str(source_id):
                doc2_nodes.append(node_id)
        
        print(f"Node classification:")
        print(f"  Doc1 nodes: {len(doc1_nodes)}")
        print(f"  Doc2 nodes: {len(doc2_nodes)}")
        
        # Create NetworkX graph to analyze connectivity
        G = nx.Graph()
        
        # Add nodes
        for node_id, node_data in nodes.items():
            G.add_node(node_id, **node_data)
        
        # Add edges
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
        
        print(f"\nGraph connectivity analysis:")
        print(f"  Total nodes: {G.number_of_nodes()}")
        print(f"  Total edges: {G.number_of_edges()}")
        print(f"  Is connected: {nx.is_connected(G)}")
        
        # Find connected components
        connected_components = list(nx.connected_components(G))
        print(f"  Connected components: {len(connected_components)}")
        
        for i, component in enumerate(connected_components):
            doc1_in_component = len([n for n in component if n in doc1_nodes])
            doc2_in_component = len([n for n in component if n in doc2_nodes])
            print(f"    Component {i}: {len(component)} nodes (Doc1: {doc1_in_component}, Doc2: {doc2_in_component})")
        
        # Find the largest connected component
        largest_component = max(connected_components, key=len)
        doc1_in_largest = len([n for n in largest_component if n in doc1_nodes])
        doc2_in_largest = len([n for n in largest_component if n in doc2_nodes])
        
        print(f"\nLargest connected component analysis:")
        print(f"  Size: {len(largest_component)} nodes")
        print(f"  Doc1 nodes in largest: {doc1_in_largest}")
        print(f"  Doc2 nodes in largest: {doc2_in_largest}")
        
        if doc2_in_largest == 0:
            print(f"\n[ROOT CAUSE IDENTIFIED]: Connectivity Issue!")
            print(f"  The largest connected component contains NO nodes from the second document.")
            print(f"  The stable_largest_connected_component() function in nano-graphrag")
            print(f"  only processes the largest connected component for clustering.")
            print(f"  This means entities from the second document are excluded from clustering!")
            
            # Check what components the doc2 nodes are in
            doc2_components = []
            for i, component in enumerate(connected_components):
                doc2_in_component = len([n for n in component if n in doc2_nodes])
                if doc2_in_component > 0:
                    doc2_components.append((i, len(component), doc2_in_component))
            
            print(f"\nSecond document nodes are in these components:")
            for comp_idx, comp_size, doc2_count in doc2_components:
                print(f"  Component {comp_idx}: {comp_size} total nodes, {doc2_count} from doc2")
            
            return "largest_connected_component_excludes_doc2"
        
        elif doc2_in_largest < len(doc2_nodes):
            print(f"\n[ISSUE IDENTIFIED]: Partial Connectivity Issue!")
            print(f"  Only {doc2_in_largest}/{len(doc2_nodes)} nodes from doc2 are in largest component")
            return "partial_connectivity_issue"
        
        else:
            print(f"\n[UNEXPECTED]: Graph connectivity looks good")
            print(f"  Both documents are well-represented in largest connected component")
            print(f"  The issue may be elsewhere in the community detection process")
            return "connectivity_ok_issue_elsewhere"
    
    except Exception as e:
        print(f"Error analyzing graph: {e}")
        import traceback
        traceback.print_exc()
        return "analysis_failed"

if __name__ == "__main__":
    result = asyncio.run(diagnose_community_detection_issue())
    print(f"\nDiagnosis result: {result}")