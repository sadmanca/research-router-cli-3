# Query Processing Mechanisms: Ultra-Deep Technical Analysis

## Overview

This document provides microscopic-level detail on how nano-graphrag processes queries, what those spinning wheel messages mean, and exactly how information flows from your knowledge graph to the LLM to generate responses.

## Part 1: Query Types and Their Processing Patterns

### Local Query Processing - Step by Step

#### Phase 1: Entity Discovery
```
⠇ Searching knowledge graph (local mode)...
```

**What's Actually Happening:**

```python
# 1. Vector similarity search on entities
results = await entities_vdb.query(query, top_k=20)
# Each entity was embedded during graph building as: entity_name + description
# Results: [{"entity_name": "TENSORFLOW", "id": "ent-a1b2c3", "distance": 0.15}, ...]

# 2. Retrieve full entity data from graph
node_datas = await asyncio.gather(*[
    knowledge_graph_inst.get_node(r["entity_name"]) for r in results
])
# Returns: [{"entity_type": "ORGANIZATION", "description": "...", "source_id": "chunk-1<SEP>chunk-2", "clusters": "[{\"cluster\": 0, \"level\": 0}, ...]"}, ...]
```

**Entity Scoring and Ranking:**
```python
# Calculate node importance (degree centrality)
node_degrees = await asyncio.gather(*[
    knowledge_graph_inst.node_degree(r["entity_name"]) for r in results
])

# Combine vector similarity with graph centrality
node_datas = [{
    **node_data,
    "entity_name": result["entity_name"],
    "rank": degree,  # Number of connections in graph
    "similarity": result["distance"]  # Vector similarity to query
} for result, node_data, degree in zip(results, node_datas, node_degrees)]
```

#### Phase 2: Context Expansion

**Finding Related Communities:**
```python
async def _find_most_related_community_from_entities(node_datas, query_param, community_reports):
    related_communities = []
    for node_d in node_datas:
        # Each entity stores its community memberships
        community_memberships = json.loads(node_d["clusters"])  
        # Format: [{"cluster": 0, "level": 0}, {"cluster": 1, "level": 1}]
        related_communities.extend(community_memberships)
    
    # Count which communities appear most frequently
    community_counts = Counter([
        str(membership["cluster"]) 
        for membership in related_communities 
        if membership["level"] <= query_param.level  # Default: level 1
    ])
    
    # Rank communities by: (frequency, community_rating)
    sorted_community_keys = sorted(
        community_counts.keys(),
        key=lambda k: (
            community_counts[k],  # How many query entities are in this community
            community_data[k]["report_json"].get("rating", -1)  # LLM-assigned importance
        ),
        reverse=True
    )
```

**Finding Related Text Chunks:**
```python
async def _find_most_related_text_unit_from_entities(node_datas, query_param, text_chunks_db, knowledge_graph_inst):
    # Extract source chunk IDs from entities
    text_units = [
        node_data["source_id"].split(GRAPH_FIELD_SEP)  # Split "chunk-1<SEP>chunk-2<SEP>chunk-3"
        for node_data in node_datas
    ]
    
    # Find one-hop neighbors for additional context
    edges = await asyncio.gather(*[
        knowledge_graph_inst.get_node_edges(node_data["entity_name"]) 
        for node_data in node_datas
    ])
    
    # Calculate chunk relevance by relationship density
    for chunk_id in all_chunk_ids:
        relation_counts = 0
        for edge in entity_edges:
            neighbor_entity = edge[1]  # (source, target, edge_data)
            neighbor_chunks = neighbor_entity_chunks.get(neighbor_entity, set())
            if chunk_id in neighbor_chunks:
                relation_counts += 1  # This chunk contains entities related to our query entities
        
        chunk_scores[chunk_id] = {
            "data": await text_chunks_db.get_by_id(chunk_id),
            "relation_counts": relation_counts,  # Higher = more relevant
            "order": entity_index  # Original entity ranking
        }
```

#### Phase 3: Context Assembly

**The CSV Format Construction:**

```python
# Entities Table
entities_section_list = [["id", "entity", "type", "description", "rank"]]
for i, node_data in enumerate(node_datas):
    entities_section_list.append([
        i,
        node_data["entity_name"],         # "TENSORFLOW"
        node_data.get("entity_type", "UNKNOWN"),  # "ORGANIZATION"
        node_data.get("description", "UNKNOWN"),  # "Open-source ML framework..."
        node_data["rank"]                 # 15 (number of connections)
    ])

# Relationships Table  
relations_section_list = [["id", "source", "target", "description", "weight", "rank"]]
for i, edge_data in enumerate(use_relations):
    relations_section_list.append([
        i,
        edge_data["src_tgt"][0],         # "TENSORFLOW" 
        edge_data["src_tgt"][1],         # "MACHINE_LEARNING"
        edge_data.get("description", "UNKNOWN"),  # "TensorFlow implements ML algorithms"
        edge_data.get("weight", 0.0),    # 2.5 (relationship strength)
        edge_data.get("rank", 0)         # 8 (edge centrality)
    ])
```

**Log Message Explanation:**
```
INFO:nano-graphrag:Using 12 entites, 3 communities, 8 relations, 15 text units
```
- **12 entities**: Top-ranked entities from vector search
- **3 communities**: Communities containing these entities
- **8 relations**: Edges connecting these entities 
- **15 text units**: Original text chunks that mentioned these entities

#### Phase 4: LLM Response Generation

**System Prompt Construction:**
```python
sys_prompt = f"""---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format.

---Target response length and format---
{query_param.response_type}  # e.g., "Multiple paragraphs"

---Data tables---

-----Reports-----
```csv
id,content
0,"Community: Machine Learning Frameworks - This community encompasses..."
1,"Community: Deep Learning Applications - Contains entities related to..."
```

-----Entities-----
```csv
id,entity,type,description,rank
0,TENSORFLOW,ORGANIZATION,"Open-source machine learning framework developed by Google",15
1,PYTORCH,ORGANIZATION,"Deep learning library developed by Facebook",12
2,SCIKIT_LEARN,ORGANIZATION,"Machine learning library for Python",10
```

-----Relationships-----
```csv
id,source,target,description,weight,rank
0,TENSORFLOW,MACHINE_LEARNING,"TensorFlow is a framework for implementing ML algorithms",2.5,8
1,PYTORCH,DEEP_LEARNING,"PyTorch specializes in deep learning applications",3.0,7
```

-----Sources-----
```csv
id,content
0,"TensorFlow is an end-to-end open source platform for machine learning..."
1,"PyTorch is an optimized tensor library for deep learning using GPUs..."
```
"""

response = await use_model_func(user_query, system_prompt=sys_prompt)
```

### Global Query Processing - The Map-Reduce Pipeline

#### Phase 1: Community Retrieval
```
⠋ Searching knowledge graph (global mode)...
INFO:nano-graphrag:Retrieved 6 communities
INFO:nano-graphrag:Grouping to 1 groups for global search
```

**Community Selection Algorithm:**
```python
async def global_query(query, ...):
    # 1. Get all communities at specified level
    community_schema = await knowledge_graph_inst.community_schema()
    communities = {k: v for k, v in community_schema.items() if v["level"] <= query_param.level}
    
    # 2. Sort by occurrence (entity count) - most important communities first
    sorted_communities = sorted(
        communities.items(),
        key=lambda x: x[1]["occurrence"],  # Number of entities in community
        reverse=True
    )[:query_param.global_max_consider_community]  # Default: 512 communities
    
    # 3. Filter by minimum quality rating
    community_datas = [c for c in community_datas 
                      if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating]
    
    logger.info(f"Retrieved {len(community_datas)} communities")
```

**Grouping Strategy:**
```python
async def _map_global_communities(query, communities_data, query_param, global_config):
    # Group communities to fit within token limits
    community_groups = []
    while len(communities_data):
        # Calculate token size for this group
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],  # Full community report text
            max_token_size=12000  # Maximum tokens per group
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group):]  # Remove processed communities
    
    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    return community_groups
```

#### Phase 2: Map Phase - Parallel Analysis

**Each Group Processed Independently:**
```python
async def _process(community_truncated_datas):
    # Format communities as structured data for LLM
    communities_section_list = [["id", "content", "rating", "importance"]]
    for i, community in enumerate(community_truncated_datas):
        communities_section_list.append([
            i,
            community["report_string"],  # Full generated report
            community["report_json"].get("rating", 0),  # 0-10 importance score
            community["occurrence"]  # Number of entities in community
        ])
    
    community_context = list_of_list_to_csv(communities_section_list)
    
    # Map prompt - extract key points relevant to query
    sys_prompt = PROMPTS["global_map_rag_points"].format(context_data=community_context)
    response = await use_model_func(query, system_prompt=sys_prompt)
    
    # Expected JSON response format:
    return {
        "points": [
            {"description": "TensorFlow dominates enterprise ML deployment", "score": 89},
            {"description": "PyTorch preferred for research and experimentation", "score": 76},
            {"description": "Open source frameworks enable widespread adoption", "score": 72}
        ]
    }
```

**Map Phase System Prompt:**
```
---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---  
Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

Each key point should have:
- Description: A comprehensive description of the point
- Importance Score: An integer score between 0-100 indicating how important the point is in answering the user's question

The response should be JSON formatted as follows:
{
    "points": [
        {"description": "Description of point 1", "score": score_value},
        {"description": "Description of point 2", "score": score_value}
    ]
}

---Data tables---
```csv
id,content,rating,importance
0,"Community Report: Machine Learning Frameworks\n\nThis community represents...",8.5,25
1,"Community Report: Deep Learning Applications\n\nThis community focuses on...",7.8,18
```
```

#### Phase 3: Reduce Phase - Synthesis

**Combining Analyst Insights:**
```python
# Collect all points from all analysts
final_support_points = []
for analyst_index, analyst_results in enumerate(map_communities_points):
    for point in analyst_results:
        if "description" not in point:
            continue
        final_support_points.append({
            "analyst": analyst_index,      # Which group/analyst provided this insight
            "answer": point["description"], # The actual insight text
            "score": point.get("score", 1) # Importance score from map phase
        })

# Filter out low-scoring points and rank by importance
final_support_points = [p for p in final_support_points if p["score"] > 0]
final_support_points = sorted(final_support_points, key=lambda x: x["score"], reverse=True)

# Truncate to fit final context window
final_support_points = truncate_list_by_token_size(
    final_support_points,
    key=lambda x: x["answer"],
    max_token_size=query_param.global_max_token_for_community_report
)
```

**Final Reduce Context:**
```
----Analyst 0----
Importance Score: 89
TensorFlow dominates enterprise ML deployment due to its comprehensive ecosystem, extensive documentation, and Google's backing, making it the preferred choice for production systems.

----Analyst 0----  
Importance Score: 76
PyTorch is preferred for research and experimentation because of its dynamic computational graphs and pythonic interface, leading to faster prototyping cycles.

----Analyst 1----
Importance Score: 72  
Open source frameworks have enabled widespread adoption of machine learning by removing cost barriers and fostering collaborative development.
```

**Global Reduce System Prompt:**
```
---Role---
You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

---Goal---
Generate a response that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports are ranked in **descending order of importance**.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer.

---Target response length and format---
{response_type}

---Analyst Reports---
{analyst_reports_context}
```

### Naive Query Processing - Direct Chunk Retrieval

#### The Simplest Approach
```
⠇ Searching knowledge graph (naive mode)...
INFO:nano-graphrag:Truncate 19 to 10 chunks
```

**Processing Steps:**
```python
async def naive_query(query, chunks_vdb, text_chunks_db, query_param, global_config):
    # 1. Vector search directly on text chunks
    results = await chunks_vdb.query(query, top_k=20)  # Top 20 most similar chunks
    
    # 2. Retrieve full chunk content
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
    # Each chunk: {"content": "text content", "tokens": 856, "full_doc_id": "doc-1", "chunk_order_index": 3}
    
    # 3. Truncate to fit model context
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=8000  # Naive query limit
    )
    
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    
    # 4. Simple concatenation
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
```

**Naive System Prompt:**
```
You're a helpful assistant
Below are the knowledge you know:
--New Chunk--
TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources...

--New Chunk--
PyTorch is an optimized tensor library for deep learning using GPUs and CPUs. It provides maximum flexibility and speed for research and production...

--New Chunk--
Scikit-learn is a machine learning library for the Python programming language. It features various classification, regression and clustering algorithms...

---
If you don't know the answer, just say so. Do not make anything up.

---Target response length and format---
{response_type}
```

## Part 2: Community Structure Deep Dive

### Understanding "Each level has communities: {0: 4, 1: 2}"

This message reveals the hierarchical clustering structure:

```python
async def clustering(self, algorithm: str = "leiden"):
    # Apply Leiden algorithm to detect communities
    import networkx.algorithms.community as nx_community
    
    # Level 0: Base communities (fine-grained clusters)
    level_0_communities = nx_community.greedy_modularity_communities(self._graph)
    # Example: [{TENSORFLOW, KERAS, PYTORCH}, {SCIKIT_LEARN, PANDAS}, {NUMPY, SCIPY}, {JUPYTER, MATPLOTLIB}]
    
    # Level 1: Meta-communities (coarser clusters)  
    # Create meta-graph where each Level 0 community becomes a node
    meta_graph = nx.Graph()
    for i, comm_i in enumerate(level_0_communities):
        for j, comm_j in enumerate(level_0_communities):
            if i >= j:
                continue
            # Calculate inter-community edge density
            inter_edges = sum(1 for u in comm_i for v in comm_j if self._graph.has_edge(u, v))
            if inter_edges > threshold:
                meta_graph.add_edge(i, j, weight=inter_edges)
    
    level_1_communities = nx_community.greedy_modularity_communities(meta_graph)
    # Example: [{0, 1}, {2, 3}] meaning communities 0,1 are grouped and 2,3 are grouped
    
    # Store hierarchical structure
    for level_0_idx, community in enumerate(level_0_communities):
        # Find which Level 1 community this belongs to
        level_1_parent = next(i for i, l1_comm in enumerate(level_1_communities) if level_0_idx in l1_comm)
        
        community_schema[f"0-{level_0_idx}"] = {
            "level": 0,
            "nodes": list(community),
            "edges": [(u, v) for u in community for v in community if self._graph.has_edge(u, v)],
            "sub_communities": [],  # Level 0 has no sub-communities
            "parent_community": f"1-{level_1_parent}"
        }
    
    for level_1_idx, meta_community in enumerate(level_1_communities):
        sub_communities = [f"0-{l0_idx}" for l0_idx in meta_community]
        all_nodes = [node for l0_idx in meta_community for node in level_0_communities[l0_idx]]
        
        community_schema[f"1-{level_1_idx}"] = {
            "level": 1,  
            "nodes": all_nodes,
            "edges": [(u, v) for u in all_nodes for v in all_nodes if self._graph.has_edge(u, v)],
            "sub_communities": sub_communities,
            "parent_community": None  # Top level
        }
```

**Concrete Example:**
```
Level 0 Communities (4):
- Community 0-0: [TENSORFLOW, KERAS, PYTORCH] (Deep Learning Frameworks)
- Community 0-1: [SCIKIT_LEARN, PANDAS, NUMPY] (Data Science Tools)  
- Community 0-2: [JUPYTER, MATPLOTLIB, SEABORN] (Analysis & Visualization)
- Community 0-3: [DOCKER, KUBERNETES, AWS] (Infrastructure)

Level 1 Communities (2):  
- Community 1-0: [0-0, 0-1] (ML Development Stack)
- Community 1-1: [0-2, 0-3] (Development Infrastructure)
```

### Community Report Generation Process

#### "Generating by levels: [1, 0]"

Reports are generated from top-level (most abstract) to bottom-level (most specific):

```python
levels = sorted(set([c["level"] for c in community_values]), reverse=True)
logger.info(f"Generating by levels: {levels}")  # [1, 0]

for level in levels:  # Start with Level 1, then Level 0
    this_level_communities = [c for c in communities if c["level"] == level]
    
    for community in this_level_communities:
        # Generate report for this specific community
        report_data = await _pack_single_community_describe(community)
        report_json = await generate_community_report_llm(report_data)
        
        # Store for use in lower-level community reports
        community_reports[community_id] = {
            "report_string": format_report_as_text(report_json),
            "report_json": report_json,
            **community
        }
```

**Why Top-Down Generation?**
1. **Context Inheritance**: Lower-level communities can reference higher-level reports
2. **Hierarchical Coherence**: Ensures consistent themes across levels
3. **Computational Efficiency**: Higher-level contexts inform lower-level analysis

#### Community Report LLM Prompt

**Input Data Structure:**
```python
async def _pack_single_community_describe(community):
    # Get all entities in this community
    nodes_data = await asyncio.gather(*[
        knowledge_graph_inst.get_node(node_name) for node_name in community["nodes"]
    ])
    
    # Get all relationships within this community
    edges_data = await asyncio.gather(*[
        knowledge_graph_inst.get_edge(src, tgt) for src, tgt in community["edges"]
    ])
    
    # Rank by importance (degree centrality)
    nodes_list_data = [
        [i, node_name, node_data.get("entity_type", "UNKNOWN"), 
         node_data.get("description", "UNKNOWN"), 
         await knowledge_graph_inst.node_degree(node_name)]
        for i, (node_name, node_data) in enumerate(zip(community["nodes"], nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)  # Sort by degree
    
    # Include sub-community reports if available
    if community["sub_communities"] and level > 0:
        sub_reports = [already_reports[sub_id]["report_string"] 
                      for sub_id in community["sub_communities"] 
                      if sub_id in already_reports]
        reports_section = "\n".join(sub_reports)
    else:
        reports_section = ""
```

**Community Report System Prompt:**
```
You are an AI assistant that helps a human analyst perform general information discovery.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims.

# Report Structure
The report should include the following sections:
- TITLE: community's name that represents its key entities - title should be short but specific
- SUMMARY: An executive summary of the community's overall structure and significant information
- IMPACT SEVERITY RATING: a float score between 0-10 representing the importance of this community
- RATING EXPLANATION: Single sentence explanation of the rating
- DETAILED FINDINGS: A list of 5-10 key insights about the community with explanatory text

Return output as JSON:
{
    "title": <report_title>,
    "summary": <executive_summary>, 
    "rating": <impact_severity_rating>,
    "rating_explanation": <rating_explanation>,
    "findings": [
        {"summary": <insight_1_summary>, "explanation": <insight_1_explanation>},
        {"summary": <insight_2_summary>, "explanation": <insight_2_explanation>}
    ]
}

Text:
```
-----Reports-----  
```csv
id,content
0,"Sub-community report: Deep Learning Frameworks community contains TensorFlow, PyTorch..."
```

-----Entities-----
```csv
id,entity,type,description,degree
0,TENSORFLOW,ORGANIZATION,"Open-source ML framework by Google",15
1,PYTORCH,ORGANIZATION,"Deep learning library by Facebook",12
2,KERAS,ORGANIZATION,"High-level neural networks API",8
```

-----Relationships-----  
```csv
id,source,target,description,rank
0,TENSORFLOW,KERAS,"Keras is integrated into TensorFlow as high-level API",10
1,PYTORCH,TENSORFLOW,"PyTorch and TensorFlow are competing ML frameworks",8
```
```
```

## Part 3: Token Management and Optimization

### Truncation Strategies

**Smart Truncation Algorithm:**
```python
def truncate_list_by_token_size(data_list, key, max_token_size):
    total_tokens = 0
    truncated_list = []
    
    for item in data_list:
        # Extract text content using key function
        content = key(item) if callable(key) else item[key]
        
        # Count tokens using tiktoken
        tokens = encode_string_by_tiktoken(content, model_name="gpt-4o")
        token_count = len(tokens)
        
        # Include if within limit
        if total_tokens + token_count <= max_token_size:
            truncated_list.append(item)
            total_tokens += token_count
        else:
            break  # Stop at first item that would exceed limit
            
    return truncated_list
```

**Context Window Management:**

| Query Type | Component | Token Limit | Purpose |
|------------|-----------|-------------|---------|
| Local | Community Reports | 6,000 | Contextual background |
| Local | Text Units | 8,000 | Original source content |
| Local | Relationships | 8,000 | Entity connections |  
| Local | Entities | 4,000 | Core entities info |
| Global | Community Groups | 12,000 | Analysis batch size |
| Global | Final Context | 12,000 | Synthesis input |
| Naive | Text Chunks | 8,000 | Direct content |

**Total Context Assembly:**
```python
# Local query total context calculation
total_context_tokens = (
    len(encode_communities_csv) +    # ~6,000 tokens
    len(encode_entities_csv) +       # ~4,000 tokens  
    len(encode_relations_csv) +      # ~8,000 tokens
    len(encode_text_units_csv) +     # ~8,000 tokens
    len(system_prompt_template)      # ~1,000 tokens
)  # Total: ~27,000 tokens (well within GPT-4's 128k context)
```

## Part 4: Context Limits and Response Formatting

### LLM Context Window Management

Nano-graphrag carefully manages token budgets across the entire pipeline:

```python
class ContextManager:
    def __init__(self, model="gpt-4o"):
        self.model_limits = {
            "gpt-4o": 128000,      # 128K tokens
            "claude-3": 200000,    # 200K tokens  
            "gemini-pro": 1048576  # 1M tokens
        }
        
        self.total_limit = self.model_limits[model]
        self.output_reserve = 4000      # Reserve for LLM response
        self.system_prompt = 2000       # System prompt overhead
        self.query_overhead = 500       # User query tokens
        
        # Available for context data
        self.context_budget = self.total_limit - self.output_reserve - self.system_prompt - self.query_overhead
        # For GPT-4o: 128,000 - 4,000 - 2,000 - 500 = 121,500 tokens available
```

### Token Allocation by Query Type

#### Local Query Allocation
```python
local_allocation = {
    "community_reports": 30375,    # 25% of context budget
    "entities_data": 24300,        # 20% of context budget  
    "relationships": 30375,        # 25% of context budget
    "text_sources": 36450,         # 30% of context budget
    "total": 121500               # Full context budget
}

# Real example breakdown:
context_usage = {
    "system_prompt": 1842,         # "You are a helpful assistant..."
    "user_query": 18,              # "How does TensorFlow work?"
    "community_reports": 8500,     # CSV data about ML communities
    "entities": 3200,              # TensorFlow, Keras, PyTorch entities
    "relationships": 5100,         # TF→Keras, TF→ML edges
    "text_sources": 12400,         # Original paper excerpts
    "total_input": 31060,          # Well within 128K limit
    "remaining_for_response": 96940 # Plenty of room for detailed answer
}
```

#### Global Query Map-Reduce Allocation
```python
global_allocation = {
    "per_analyst_group": 12000,    # Each analyst gets 12K tokens max
    "synthesis_context": 18225,    # 15% for final synthesis
    "metadata_overhead": 6075      # 5% for system prompts
}

# Example with 3 analysts:
map_phase_usage = [
    {"analyst_1": 11250, "communities": "ML Frameworks, Computer Vision"},
    {"analyst_2": 10890, "communities": "NLP, Robotics, Ethics"},  
    {"analyst_3": 9760,  "communities": "Healthcare AI, Quantum AI"}
]

reduce_phase_usage = {
    "analyst_insights": 15200,     # Combined insights from all analysts
    "synthesis_prompt": 2800,      # Research Director instructions
    "total": 18000                 # Final synthesis input
}
```

### Response Format Control

The `response_type` parameter controls output formatting:

```python
RESPONSE_FORMATS = {
    "single_paragraph": {
        "instruction": "Provide a single, comprehensive paragraph (150-300 words)",
        "example": "Machine learning encompasses various algorithmic approaches including supervised learning for prediction tasks, unsupervised learning for pattern discovery, and reinforcement learning for decision optimization, with applications spanning computer vision, natural language processing, and robotics across industries from healthcare to finance."
    },
    
    "bullet_points": {
        "instruction": "Present as organized bullet points with hierarchical structure",
        "example": """
• **Core ML Algorithms**
  - Supervised learning (classification, regression)
  - Unsupervised learning (clustering, dimensionality reduction)
  - Reinforcement learning (policy optimization)

• **Key Applications**  
  - Computer vision and image processing
  - Natural language processing and understanding
  - Robotics and autonomous systems
        """
    },
    
    "detailed_analysis": {
        "instruction": "Comprehensive analysis with sections and subsections (500-1000 words)",
        "example": """
## Machine Learning Overview

### Algorithmic Foundations
Machine learning algorithms fall into three primary categories based on their learning paradigms...

### Application Domains  
The practical applications of ML span numerous industries and use cases...

### Technical Implementation
Modern ML systems typically employ frameworks like TensorFlow and PyTorch...
        """
    }
}
```

### Format-Specific System Prompt Templates

```python
def build_system_prompt(query_type, response_format, context_data):
    base_prompt = f"""---Role---
You are a helpful assistant responding to questions about data in the provided tables.

---Goal---
Generate a response that answers the user's question using information from the data tables.
If you don't know the answer, say so. Do not make anything up.
"""

    format_instructions = {
        "bullet_points": """
---Target response length and format---
Present information as organized bullet points:
• Use clear, descriptive bullet points
• Include sub-bullets with indentation where appropriate
• Group related information logically
• Aim for 5-10 main points maximum
        """,
        
        "multiple_paragraphs": """
---Target response length and format---  
Structure as 3-5 well-organized paragraphs:
• Each paragraph should focus on a distinct theme
• Use topic sentences and smooth transitions
• Maintain logical flow from general to specific
• Target 300-600 words total
        """,
        
        "executive_summary": """
---Target response length and format---
Format as professional executive summary:
• Brief overview paragraph (2-3 sentences)
• Key findings as bullet points (3-5 points)  
• Implications or recommendations paragraph
• Professional, concise tone throughout
• Target 200-400 words total
        """
    }
    
    return f"{base_prompt}\n{format_instructions[response_format]}\n\n---Data tables---\n{context_data}"
```

### Real Response Examples by Format

**Same Query, Different Formats:**
**Query:** "What are the relationships between TensorFlow, Keras, and PyTorch?"

#### Single Paragraph Format:
```
TensorFlow and PyTorch represent the two dominant deep learning frameworks, with TensorFlow being Google's comprehensive machine learning platform that originally used static computational graphs and PyTorch being Facebook's research-focused framework emphasizing dynamic graphs and pythonic interfaces. Keras serves as a high-level neural networks API that was initially independent but became tightly integrated into TensorFlow as its official high-level interface, providing simplified model building capabilities while TensorFlow handles the low-level computations. The relationship between these frameworks reflects the evolution of deep learning infrastructure, where TensorFlow targets production deployment and scalability, PyTorch excels in research flexibility and rapid prototyping, and Keras bridges the gap by making TensorFlow more accessible to practitioners who need powerful capabilities without complex implementations.
```

#### Bullet Points Format:
```
• **TensorFlow Ecosystem**
  - Google's comprehensive ML platform for production deployment
  - Originally static graph-based, now supports eager execution
  - Integrates tightly with Keras as its high-level API
  - Strong ecosystem with TensorFlow Serving, TensorFlow Lite

• **PyTorch Framework**
  - Facebook's research-oriented deep learning library
  - Dynamic computational graphs enable flexible model architectures  
  - Pythonic interface preferred by researchers and academics
  - Growing production support through TorchServe and TorchScript

• **Keras Integration**
  - High-level neural networks API originally framework-agnostic
  - Now officially integrated into TensorFlow as tf.keras
  - Provides simplified interface for building and training models
  - Supports both TensorFlow backend and limited PyTorch compatibility

• **Competitive Relationship**
  - TensorFlow vs PyTorch represents production vs research priorities
  - Both frameworks increasingly adding each other's strengths
  - Keras reduces TensorFlow complexity to compete with PyTorch's ease of use
```

#### Executive Summary Format:
```
**Executive Summary: Deep Learning Framework Relationships**

TensorFlow, PyTorch, and Keras form the core ecosystem of modern deep learning development, each serving distinct but interconnected roles in the machine learning pipeline.

**Key Findings:**
• TensorFlow dominates production ML deployment with comprehensive tooling and Google's enterprise support
• PyTorch leads in research environments due to its dynamic graphs and intuitive Python-first design philosophy  
• Keras integration into TensorFlow creates a unified high-level interface that competes directly with PyTorch's usability
• Framework competition drives rapid innovation, with TensorFlow adding dynamic capabilities and PyTorch improving production readiness

**Strategic Implications:**
Organizations should evaluate framework choice based on use case priorities: TensorFlow for production systems requiring scale and deployment infrastructure, PyTorch for research and rapid prototyping, with Keras providing an accessible entry point to TensorFlow's capabilities.
```

### Context-Aware Response Optimization

```python
def optimize_response_for_context(context_quality, query_complexity, response_format):
    """
    Adjust response strategy based on available context and query complexity
    """
    optimization_strategy = {
        "high_context_simple_query": {
            "approach": "concise_focused",
            "length_modifier": 0.8,  # Shorter response
            "detail_level": "essential_only"
        },
        
        "high_context_complex_query": {
            "approach": "comprehensive_analysis", 
            "length_modifier": 1.3,  # Longer response
            "detail_level": "full_depth"
        },
        
        "low_context_simple_query": {
            "approach": "direct_answer",
            "length_modifier": 0.6,  # Very short response
            "detail_level": "basic_only"  
        },
        
        "low_context_complex_query": {
            "approach": "acknowledge_limitations",
            "length_modifier": 0.9,  # Standard length
            "detail_level": "qualified_response"
        }
    }
    
    # Determine context-complexity combination
    context_level = "high" if context_quality > 0.7 else "low"
    complexity_level = "complex" if query_complexity > 5 else "simple"
    strategy_key = f"{context_level}_context_{complexity_level}_query"
    
    return optimization_strategy[strategy_key]
```

## Summary

This ultra-detailed analysis reveals that nano-graphrag implements a sophisticated multi-stage information retrieval and synthesis system:

1. **Local Queries**: Entity-centric retrieval with multi-hop context expansion
2. **Global Queries**: Map-reduce community analysis for comprehensive insights  
3. **Naive Queries**: Direct chunk-based retrieval for simple cases
4. **Context Management**: Intelligent token budgeting and overflow handling
5. **Response Formatting**: Dynamic format control with quality optimization

The spinning wheel messages indicate complex parallel processing of vector searches, graph traversals, and hierarchical community analysis, all carefully orchestrated within strict token limits to provide the most relevant context to the LLM for generating accurate, well-formatted responses optimized for the user's specified output format.