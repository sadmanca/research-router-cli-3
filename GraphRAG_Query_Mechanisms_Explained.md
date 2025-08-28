# GraphRAG Query Mechanisms: From Document Insertion to Knowledge Retrieval

This document explains how documents are inserted into the knowledge graph and how the different query types (local, global, naive) work in nano-graphrag.

## Table of Contents

1. [Document Insertion Process](#document-insertion-process)
2. [Global Query Mechanism](#global-query-mechanism)
3. [Local Query Mechanism](#local-query-mechanism)
4. [Naive Query Mechanism](#naive-query-mechanism)
5. [Community Detection and Hierarchical Levels](#community-detection-and-hierarchical-levels)
6. [Code References](#code-references)

---

## Document Insertion Process

### Step 1: Document Chunking
When documents are inserted, they are first broken down into smaller chunks:

**Code Reference**: `nano-graphrag/nano_graphrag/_op.py` (lines 400-450)

**What happens**:
- Documents are split into manageable text chunks (typically 200-1200 tokens)
- Each chunk gets a unique ID (`chunk-xxxxx`) and metadata including:
  - `content`: The actual text content
  - `tokens`: Number of tokens in the chunk  
  - `chunk_order_index`: Position in the original document
  - `full_doc_id`: Reference to the parent document ID

### Step 2: Entity Extraction
From each document, scientific concepts and entities are extracted using GenKG:

**Code Reference**: `nano-graphrag/nano_graphrag/_op.py` (lines 504-539)

**What happens**:
```python
# Extract nodes using GenKG
node_limit = global_config.get("genkg_node_limit", 25)
nodes_with_source = genkg.gemini_create_nodes(summary, node_limit, doc_id)
```

- Documents are first summarized to focus on key scientific concepts
- LLM (Gemini) extracts ~25 high-level concepts per document
- Each entity gets normalized (uppercase, cleaned) for compatibility
- Entities are stored with metadata: `entity_name`, `entity_type`, `description`, `source_id`

### Step 3: Relationship Extraction
Relationships between entities are identified:

**Code Reference**: `nano-graphrag/nano_graphrag/_op.py` (lines 507-571)

**What happens**:
```python
# Extract edges using GenKG 
edges = genkg.create_edges_by_gemini(nodes_with_source, {doc_id: summary})
```

- LLM analyzes semantic relationships between extracted entities
- Creates weighted edges with relationship types (`related_to`, `enables`, `depends_on`, etc.)
- Edges are normalized and stored as: `src_id`, `tgt_id`, `weight`, `description`

### Step 4: Graph Connectivity Enhancement
**NEW**: After merging all documents, connectivity is ensured across components:

**Code Reference**: `nano-graphrag/nano_graphrag/_op.py` (lines 606-680)

**What happens**:
- Analyzes connected components in the knowledge graph
- Uses sentence transformers to find semantic similarities between disconnected components
- Adds "connectivity edges" with lower weights to bridge isolated document clusters
- Ensures queries can access content from ALL documents, not just the largest component

### Step 5: Community Detection
Entities are clustered into communities using the Leiden algorithm:

**Code Reference**: `nano-graphrag/nano_graphrag/_storage/gdb_networkx.py` (`_leiden_clustering` method)

**What happens**:
- **CRITICAL FIX APPLIED**: Previously only processed the largest connected component
- **The Problem**: Multi-document knowledge graphs often have disconnected components (separate entity clusters per document)
- **The Original Bug**: `stable_largest_connected_component()` would discard all but the largest component
- **The Fix**: Now processes ALL connected components:
  ```python
  connected_components = list(nx.connected_components(self._graph))
  logger.info(f"Processing {len(connected_components)} connected components for clustering")
  
  for comp_idx, component_nodes in enumerate(connected_components):
      # Extract subgraph for this component
      component_subgraph = self._graph.subgraph(component_nodes).copy()
      # ... apply Leiden clustering to each component
  ```
- **Result**: All documents now contribute entities to community detection, not just the largest document
- Uses hierarchical Leiden clustering to identify entity communities at multiple granularities
- Creates multiple levels of communities (level 0 = fine-grained, higher levels = coarser)
- Each entity gets assigned community IDs at different hierarchical levels

### Step 6: Community Report Generation
High-level summaries are created for each community:

**Code Reference**: Community reports are generated using LLM summarization

**What happens**:
- For each community, related entities and their relationships are analyzed
- LLM creates a comprehensive report describing the community's main themes
- Reports are stored and indexed for fast retrieval during global queries

---

## Global Query Mechanism

### Log Example:
```
⠋ Searching knowledge graph (global mode)...
INFO:nano-graphrag:Revtrieved 10 communities
INFO:nano-graphrag:Grouping to 1 groups for global search
```

### How It Works:

**Code Reference**: `nano-graphrag/nano_graphrag/_op.py` (lines 1426-1513)

**Step 1: Community Schema Retrieval**
```python
community_schema = await knowledge_graph_inst.community_schema()
```
**What happens**: Gets metadata about all communities including their hierarchical levels

**Step 2: Community Selection**
```python
selected_communities = await _get_community_report_by_ids(
    community_reports, 
    query,
    query_param.top_k,
    knowledge_graph_inst,
    entities_vdb,
    use_map_reduce
)
```
**What happens**: 
- Uses vector similarity search on community reports to find most relevant communities
- Each community report contains:
  - **Summary**: High-level description of the community's main themes
  - **Key Entities**: Important concepts within that community  
  - **Relationships**: How entities in this community connect
- "Retrieved 10 communities" means it found 10 community reports relevant to your question
- These are **pre-computed summaries**, not individual entities, making global search very fast
- Community reports act as "executive summaries" of different topic areas in your knowledge base

**Step 3: Grouping and Map-Reduce**
**What happens**: 
- "Grouping to 1 groups" means all selected communities fit within the context window
- If there were too many, it would group them and use map-reduce to process in batches
- Creates a comprehensive context from community reports

**Step 4: Response Generation**
```python
return await use_model_func(
    query, system_prompt=PROMPTS["global_query_system_prompt"] + schema_prompt, context_data=context
)
```
**What happens**: LLM generates response using the community reports as context

**Why Global Queries Are Powerful**: They provide high-level, synthesized information across the entire knowledge base by leveraging pre-computed community summaries.

---

## Local Query Mechanism

### Log Example:
```
⠼ Searching knowledge graph (local mode)...
INFO:nano-graphrag:Using 20 entities, 3 communities, 48 relations, 3 text units
```

### How It Works:

**Code Reference**: `nano-graphrag/nano_graphrag/_op.py` (lines 1343-1425)

**Step 1: Entity Retrieval**
```python
results = await entities_vdb.query(query, top_k=query_param.top_k)
```
**What happens**: 
- Finds the top entities most similar to your query using vector search
- "20 entities" means it found 20 relevant entities in the knowledge graph

**Step 2: Community and Relation Expansion**
```python
context = await _build_local_query_context(
    knowledge_graph_inst, community_reports, entities_vdb, text_chunks_db, query, selected_entities
)
```
**What happens**:
- Takes the 20 entities and finds their associated communities
- "3 communities" means these 20 entities belong to 3 different topic clusters  
- "48 relations" means there are 48 relationship edges connecting these entities
- Builds a subgraph of the most relevant part of the knowledge graph

**Step 3: Text Unit Retrieval**
**Code Reference**: `nano-graphrag/nano_graphrag/_op.py` (`_find_most_related_text_unit_from_entities` function, lines 1046-1200)

**What happens**:
```python
text_units = [
    split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
    for dp in node_datas
]
```
- Extracts source IDs from the selected entities  
- "3 text units" means it found 3 actual text chunks from your documents
- **CRITICAL**: This number was 0 before our fix - entities had document IDs instead of chunk IDs
- **The Fix Applied**: 
  - Creates document-to-chunk mapping: `doc_to_chunks_mapping[full_doc_id] = [chunk_ids...]`
  - Maps entity source_ids (document IDs like `doc-abc123`) to actual chunk IDs (`chunk-xyz789`)
  - Handles both data structures (with/without 'data' wrapper)
- **Result**: Now successfully retrieves the actual source text chunks for context

**Step 4: Context Construction**
**What happens**:
- Combines entities, their relationships, community context, and source text chunks
- Creates a rich, multi-layered context that includes both graph structure and original text

**Step 5: Response Generation**
**What happens**: LLM uses this detailed local context to provide precise, well-sourced answers

**Why Local Queries Are Detailed**: They drill down into specific parts of the knowledge graph and include actual source text, providing precise, evidence-backed responses.

---

## Naive Query Mechanism

### How It Works:

**Code Reference**: `nano-graphrag/nano_graphrag/_op.py` (lines 1514-1540)

**Simple Vector Search**:
```python
results = await chunks_vdb.query(query, top_k=query_param.top_k)
```

**What happens**:
- Bypasses the knowledge graph entirely
- Uses pure semantic similarity search on document chunks
- No entity extraction, no relationships, no communities
- Just finds text chunks most similar to your query
- Fastest but least sophisticated approach

**When to use**: Quick searches when you don't need the sophisticated reasoning of the knowledge graph.

---

## Community Detection and Hierarchical Levels

### Log Examples:
```
INFO:nano-graphrag:Each level has communities: {0: 10}
INFO:nano-graphrag:Total nodes with community assignments: 50
INFO:nano-graphrag:Generating by levels: [0]
o Processed 1 communities
O Processed 2 communities
```

### How Community Levels Work:

**Code Reference**: `nano-graphrag/nano_graphrag/_storage/gdb_networkx.py`

### Hierarchical Structure:
```python
# Example community assignments
{0: 10}  # Level 0 has 10 communities (fine-grained)
{1: 3}   # Level 1 has 3 communities (coarser groupings)  
{2: 1}   # Level 2 has 1 community (entire graph)
```

**What each level means**:
- **Level 0**: Fine-grained communities (specific topics, methods, results)
- **Level 1**: Broader themes (entire research areas, methodological approaches)  
- **Level 2**: Highest level (entire document themes, major research directions)

### Community Processing Logs Explained:

**"Each level has communities: {0: 10}"**:
- This shows the hierarchical structure: Level 0 has 10 distinct communities
- Level 0 = most granular (specific research topics, methods, findings)
- If there were multiple levels, you'd see `{0: 10, 1: 3, 2: 1}` meaning:
  - Level 0: 10 fine-grained communities
  - Level 1: 3 broader topic areas  
  - Level 2: 1 overarching theme

**"Total nodes with community assignments: 50"**:
- 50 entities have been successfully assigned to communities
- This confirms all extracted entities are participating in the knowledge graph structure

**"Generating by levels: [0]"**:
- The system is using only Level 0 communities (most detailed granularity)
- This provides the highest resolution of topical organization
- Higher levels would provide broader, more abstract groupings

**"o Processed 1 communities", "O Processed 2 communities"**:
- Shows real-time progress as community reports are being generated
- Each community gets analyzed by LLM to create descriptive summaries  
- The rotating animation (`o`, `O`, etc.) indicates active processing
- These summaries become the building blocks for global queries

**Community Report Generation Process**:
```python
# For each community at each level:
community_entities = get_entities_in_community(community_id)
community_relationships = get_relationships_between_entities(community_entities)
community_report = llm.summarize_community(entities, relationships, context)
```

### Why Hierarchical Communities Matter:
- **Fine-grained (Level 0)**: Specific technical details, precise methods
- **Coarse-grained (Level 1+)**: Broader themes, cross-cutting insights
- Allows queries to operate at different levels of abstraction
- Enables both detailed technical questions and high-level conceptual queries

---

## Code References

### Key Files:
- **`nano-graphrag/nano_graphrag/_op.py`**: Main query functions and entity extraction
- **`nano-graphrag/nano_graphrag/_storage/gdb_networkx.py`**: Community detection and graph operations
- **`nano-graphrag/genkg.py`**: Entity/relationship extraction and connectivity enhancement

### Critical Functions:
- **`local_query()` (line 1343)**: Detailed context with entities, relations, text chunks
- **`global_query()` (line 1426)**: High-level synthesis using community reports  
- **`naive_query()` (line 1514)**: Simple semantic search
- **`_build_local_query_context()`**: Constructs multi-layered context for local queries
- **`ensure_graph_connectivity()`**: Bridges disconnected document components

### Recent Fixes Applied:

#### 1. Community Detection Fix (`gdb_networkx.py`)
- **Problem**: Only largest connected component was processed for clustering
- **Impact**: Multi-document graphs had entities from smaller documents completely ignored
- **Solution**: Modified `_leiden_clustering` to process ALL connected components
- **Result**: All documents now contribute to community detection and query responses

#### 2. Text Unit Retrieval Fix (`_op.py`)
- **Problem**: Entities had document IDs (`doc-abc123`) instead of chunk IDs (`chunk-xyz789`)
- **Impact**: Local queries showed "0 text units" - no actual source text retrieved
- **Solution**: Added document-to-chunk mapping and dual data structure handling
- **Result**: Local queries now successfully retrieve actual text chunks for context

#### 3. Graph Connectivity Enhancement (`_op.py`)
- **Problem**: Documents created isolated graph components with no cross-document connections
- **Impact**: Queries couldn't bridge insights across different documents
- **Solution**: Added semantic similarity-based connectivity edges between components
- **Result**: Multi-document knowledge graphs are fully connected, enabling cross-document reasoning

#### 4. Data Structure Compatibility Fix (`_op.py`)
- **Problem**: Different sessions had different text chunk data structures (with/without 'data' wrapper)
- **Impact**: "Text unit missing 'data' field" warnings, failed text retrieval
- **Solution**: Handles both `{"data": {...}}` and `{"content": "...", "tokens": ...}` formats
- **Result**: Consistent behavior across all sessions regardless of data structure

---

## Complete Multi-Document Query Flow

### Before Fixes:
1. Insert Document A → Entities extracted, communities created
2. Insert Document B → Entities extracted, **but isolated from Document A**
3. Community detection → **Only processes largest component (Document A)**
4. Query execution → **Only retrieves content from Document A**
5. Result: Document B content completely ignored

### After Fixes:
1. Insert Document A → Entities extracted, communities created
2. Insert Document B → Entities extracted, **connectivity to Document A analyzed**
3. **Connectivity enhancement** → Semantic bridges added between documents
4. Community detection → **Processes ALL components from both documents**
5. Query execution → **Retrieves relevant content from both documents**
6. Result: Full multi-document reasoning and synthesis

This architecture now enables sophisticated question-answering that can:
- **Reason across multiple documents** using graph connectivity
- **Synthesize high-level insights** using hierarchical community reports
- **Provide detailed evidence-backed responses** using actual source text chunks
- **Handle diverse document structures** with robust data compatibility
- **Scale efficiently** through pre-computed community summaries and vector search