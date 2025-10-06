# ULTIMATE Querying Guide: Nano-GraphRAG Query Processing Mechanisms

This document provides a comprehensive, technical analysis of how nano-graphrag processes queries and generates responses. This guide covers all three query modes with detailed explanations of their differences, information selection strategies, and LLM integration patterns.

## Table of Contents

1. [Query Architecture Overview](#query-architecture-overview)
2. [Local Query Mode: Entity-Centric Retrieval](#local-query-mode-entity-centric-retrieval)
3. [Global Query Mode: Community-Based Analysis](#global-query-mode-community-based-analysis)
4. [Naive Query Mode: Direct Text Retrieval](#naive-query-mode-direct-text-retrieval)
5. [Information Selection and Ranking Algorithms](#information-selection-and-ranking-algorithms)
6. [Context Assembly and Token Management](#context-assembly-and-token-management)
7. [LLM Integration Patterns](#llm-integration-patterns)
8. [Configuration Parameters and Limits](#configuration-parameters-and-limits)
9. [Performance Analysis](#performance-analysis)
10. [Troubleshooting and Optimization](#troubleshooting-and-optimization)

---

## Query Architecture Overview

### Three-Mode Query System
```
Query Input
    ↓
Mode Selection [local | global | naive]
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│   Local Mode    │   Global Mode    │   Naive Mode    │
│                 │                  │                 │
│ Entity Search   │ Community        │ Chunk Search    │
│      ↓          │ Analysis         │      ↓          │
│ Context         │      ↓           │ Simple          │
│ Expansion       │ Map-Reduce       │ Concatenation   │
│      ↓          │      ↓           │      ↓          │
│ Multi-Table     │ Analyst          │ Direct          │
│ Context         │ Synthesis        │ Context         │
└─────────────────┴──────────────────┴─────────────────┘
    ↓                     ↓                    ↓
         LLM Response Generation
```

### Core Differences Between Modes

| Aspect | Local Mode | Global Mode | Naive Mode |
|--------|------------|-------------|------------|
| **Primary Data Source** | Entity embeddings | Community reports | Text chunk embeddings |
| **Context Complexity** | Multi-table structured | Hierarchical analysis | Simple concatenation |
| **Information Depth** | Entity-relationship focused | High-level insights | Raw text segments |
| **Processing Pattern** | Search → Expand → Structure | Select → Map → Reduce | Search → Truncate → Combine |
| **Best For** | Specific entity questions | Broad thematic analysis | Simple factual lookup |
| **Computational Cost** | Medium | High | Low |

---

## Local Query Mode: Entity-Centric Retrieval

**Location**: `_op.py:1351-1386`

### High-Level Process Flow
```
Query → Entity Vector Search → Context Expansion → Multi-Table Assembly → LLM Response
```

### Phase 1: Entity Discovery
**Location**: `_op.py:1227-1350` (`_build_local_query_context`)

#### Vector Similarity Search
```python
# Search for relevant entities
results = await entities_vdb.query(query, top_k=query_param.top_k)  # Default: 20
```

**Search Details**:
- **Embedding Target**: Entity name + description concatenated
- **Default Top-K**: 20 entities
- **Similarity Metric**: Cosine similarity (vector database dependent)
- **Fallback**: Returns `fail_response` if no entities found

#### Entity Data Retrieval
```python
# Get full entity metadata
node_datas = await asyncio.gather(*[
    knowledge_graph_inst.get_node(r["entity_name"]) for r in results
])

# Add ranking and similarity scores
for result, node_data in zip(results, node_datas):
    node_data.update({
        "entity_name": result["entity_name"],
        "rank": await knowledge_graph_inst.node_degree(result["entity_name"]),
        "similarity": result["distance"]
    })
```

**Entity Ranking Factors**:
1. **Vector Similarity**: Distance from query embedding
2. **Graph Centrality**: Node degree (number of connections)
3. **Community Membership**: Participation in important communities

### Phase 2: Context Expansion

#### Related Communities Discovery
**Location**: `_op.py:_find_most_related_community_from_entities`

```python
# Extract community memberships from entities
related_communities = []
for node_data in node_datas:
    if "clusters" in node_data:
        # Format: [{"cluster": 0, "level": 0}, {"cluster": 1, "level": 1}]
        community_memberships = json.loads(node_data["clusters"])
        related_communities.extend(community_memberships)

# Rank communities by frequency and importance
community_counts = Counter([
    str(membership["cluster"])
    for membership in related_communities
    if membership["level"] <= query_param.level  # Default: level 2
])

# Sort by: (frequency, community_rating)
sorted_communities = sorted(
    community_counts.keys(),
    key=lambda k: (
        community_counts[k],  # How many query entities are in this community
        community_data[k]["report_json"].get("rating", -1)  # LLM-assigned importance (0-10)
    ),
    reverse=True
)
```

**Community Selection Logic**:
- **Primary Factor**: Number of query entities in community (frequency)
- **Secondary Factor**: Community importance rating (LLM-generated 0-10 score)
- **Level Filtering**: Only includes communities at or below specified hierarchy level

#### Related Text Chunks Discovery
**Location**: `_op.py:_find_most_related_text_unit_from_entities`

```python
# Extract source chunk IDs from entities
text_units = []
for node_data in node_datas:
    # source_id format: "chunk-1<SEP>chunk-2<SEP>chunk-3"
    chunk_ids = node_data["source_id"].split(GRAPH_FIELD_SEP)
    text_units.extend(chunk_ids)

# Find one-hop neighbors for additional context
one_hop_edges = await asyncio.gather(*[
    knowledge_graph_inst.get_node_edges(node_data["entity_name"])
    for node_data in node_datas
])

# Calculate chunk relevance by relationship density
for chunk_id in all_chunk_ids:
    relation_count = 0
    for edge in entity_edges:
        neighbor_entity = edge[1]  # (source, target, edge_data)
        neighbor_chunks = neighbor_entity_source_chunks.get(neighbor_entity, set())
        if chunk_id in neighbor_chunks:
            relation_count += 1  # This chunk contains related entities

    chunk_scores[chunk_id] = {
        "data": await text_chunks_db.get_by_id(chunk_id),
        "relation_counts": relation_count,  # Higher = more interconnected
        "order": entity_ranking_index
    }
```

**Text Chunk Scoring Algorithm**:
1. **Direct Mention**: Chunks that mention query entities (primary)
2. **Relationship Density**: Chunks containing entities connected to query entities
3. **Entity Ranking**: Preserve order from initial entity ranking

#### Related Relationships Discovery
**Location**: `_op.py:_find_most_related_edges_from_entities`

```python
# Get all edges involving query entities
entity_edges = await asyncio.gather(*[
    knowledge_graph_inst.get_node_edges(node_data["entity_name"])
    for node_data in node_datas
])

# Calculate edge importance scores
for edge in all_edges:
    src_id, tgt_id = edge
    edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)

    # Rank by: weight, source entity importance, target entity importance
    edge_rank = (
        edge_data.get("weight", 0),
        src_entity_ranks.get(src_id, 0),
        tgt_entity_ranks.get(tgt_id, 0)
    )
```

### Phase 3: Multi-Table Context Assembly

#### CSV Format Construction
Local queries assemble context into structured CSV tables for LLM consumption:

##### 1. Community Reports Table
```csv
id,content
0,"Community: Machine Learning Frameworks - This community encompasses TensorFlow, PyTorch, and Keras, representing core deep learning technologies..."
1,"Community: Data Science Tools - Contains pandas, numpy, and scikit-learn, representing data manipulation and analysis capabilities..."
```

##### 2. Entities Table
```csv
id,entity,type,description,rank
0,TENSORFLOW,ORGANIZATION,"Open-source machine learning framework developed by Google",15
1,PYTORCH,ORGANIZATION,"Deep learning library developed by Facebook",12
2,SCIKIT_LEARN,ORGANIZATION,"Machine learning library for Python",10
```

##### 3. Relationships Table
```csv
id,source,target,description,weight,rank
0,TENSORFLOW,MACHINE_LEARNING,"TensorFlow is a framework for implementing ML algorithms",2.5,8
1,PYTORCH,DEEP_LEARNING,"PyTorch specializes in deep learning applications",3.0,7
```

##### 4. Sources Table (Optional)
```csv
id,content
0,"TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive ecosystem..."
1,"PyTorch is an optimized tensor library for deep learning using GPUs and CPUs. It provides maximum flexibility..."
```

**Configuration Control**:
- **include_text_chunks**: Default False (excludes Sources table to reduce token usage)
- **Token Limits**: Each table type has individual token limits for truncation

### Phase 4: LLM Response Generation

#### System Prompt Structure
```
---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format.

---Target response length and format---
{response_type}  # e.g., "Multiple paragraphs"

---Data tables---
{context_data}  # Assembled CSV tables
```

**Key Instructions to LLM**:
- Only use information from provided tables
- Don't make up information not in the data
- Incorporate relevant general knowledge when appropriate
- Follow specified response format and length

---

## Global Query Mode: Community-Based Analysis

**Location**: `_op.py:1434-1519`

### High-Level Process Flow
```
Query → Community Selection → Map Phase (Parallel Analysis) → Reduce Phase (Synthesis)
```

### Phase 1: Community Selection and Filtering

#### Community Retrieval Algorithm
```python
# Get all communities at or below specified level
community_schema = await knowledge_graph_inst.community_schema()
community_schema = {
    k: v for k, v in community_schema.items()
    if v["level"] <= query_param.level  # Default: level 2
}

# Sort by importance (entity count)
sorted_communities = sorted(
    community_schema.items(),
    key=lambda x: x[1]["occurrence"],  # Number of entities in community
    reverse=True
)

# Apply limits and filtering
communities = sorted_communities[:query_param.global_max_consider_community]  # Default: 512
community_datas = [c for c in community_datas
                  if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating]  # Default: 0
```

**Selection Criteria Priority**:
1. **Hierarchy Level**: Only communities at specified depth
2. **Entity Count** (occurrence): More entities = higher importance
3. **Quality Rating**: LLM-generated community importance score (0-10)
4. **Maximum Limit**: Caps total communities to prevent overwhelming

#### Log Message: "Retrieved 6 communities"
This indicates the number of communities that passed filtering criteria.

### Phase 2: Map Phase - Parallel Community Analysis

#### Community Grouping Strategy
```python
# Group communities to fit within token limits
community_groups = []
while len(communities_data):
    this_group = truncate_list_by_token_size(
        communities_data,
        key=lambda x: x["report_string"],  # Full community report text
        max_token_size=query_param.global_max_token_for_community_report  # Default: 16384
    )
    community_groups.append(this_group)
    communities_data = communities_data[len(this_group):]
```

**Log Message: "Grouping to 1 groups for global search"**
- Each group represents a batch that fits within token limits
- Groups are processed in parallel for efficiency

#### Parallel Analysis Process
```python
async def _process(community_group):
    # Format communities as structured data
    communities_csv = [["id", "content", "rating", "importance"]]
    for i, community in enumerate(community_group):
        communities_csv.append([
            i,
            community["report_string"],           # Full LLM-generated report
            community["report_json"].get("rating", 0),  # 0-10 importance score
            community["occurrence"]               # Number of entities
        ])

    # Send to LLM for point extraction
    sys_prompt = PROMPTS["global_map_rag_points"].format(context_data=community_context)
    response = await use_model_func(query, system_prompt=sys_prompt)

    # Expected JSON response format
    return {
        "points": [
            {"description": "Key insight about the query", "score": 85},
            {"description": "Another relevant finding", "score": 72}
        ]
    }
```

#### Map Phase System Prompt
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
{community_context_csv}
```

**Map Phase Output**: Each analyst (group) produces scored insights relevant to the query.

### Phase 3: Reduce Phase - Insight Synthesis

#### Point Aggregation and Ranking
```python
# Collect all points from all analysts
final_support_points = []
for analyst_index, analyst_results in enumerate(map_results):
    for point in analyst_results:
        if "description" not in point:
            continue
        final_support_points.append({
            "analyst": analyst_index,        # Which group provided this insight
            "answer": point["description"],  # The actual insight text
            "score": point.get("score", 1)   # Importance score from map phase
        })

# Filter and rank by importance
final_support_points = [p for p in final_support_points if p["score"] > 0]
final_support_points = sorted(final_support_points, key=lambda x: x["score"], reverse=True)

# Truncate to fit final context window
final_support_points = truncate_list_by_token_size(
    final_support_points,
    key=lambda x: x["answer"],
    max_token_size=query_param.global_max_token_for_community_report  # Default: 16384
)
```

#### Final Context Assembly
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

#### Reduce Phase System Prompt
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

**Reduce Phase Characteristics**:
- **Input**: Ranked insights from multiple analysts
- **Processing**: Synthesis and deduplication
- **Output**: Comprehensive answer combining multiple perspectives

---

## Naive Query Mode: Direct Text Retrieval

**Location**: `_op.py:1522-1553`

### High-Level Process Flow
```
Query → Chunk Vector Search → Truncation → Simple Concatenation → LLM Response
```

### Phase 1: Vector Search on Text Chunks
```python
# Direct similarity search on chunk embeddings
results = await chunks_vdb.query(query, top_k=query_param.top_k)  # Default: 20

# Retrieve full chunk content
chunks_ids = [r["id"] for r in results]
chunks = await text_chunks_db.get_by_ids(chunks_ids)
```

**Chunk Embedding Details**:
- **Embedding Content**: Full chunk text content
- **Similarity Metric**: Cosine similarity to query
- **No Re-ranking**: Direct vector similarity order preserved

### Phase 2: Token-Based Truncation
```python
# Simple truncation to fit context window
maybe_truncated_chunks = truncate_list_by_token_size(
    chunks,
    key=lambda x: x["content"],
    max_token_size=query_param.naive_max_token_for_text_unit  # Default: 12000
)

logger.info(f"Truncate {len(chunks)} to {len(maybe_truncated_chunks)} chunks")
```

**Log Message Example**: "Truncate 19 to 10 chunks"
- Indicates original retrieval vs. final context size
- Simple token-based cutoff, no sophisticated ranking

### Phase 3: Simple Concatenation
```python
# Basic concatenation with separators
section = "--New Chunk--\n".join([c["content"] for c in maybe_truncated_chunks])
```

**Context Format**:
```
--New Chunk--
TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources...

--New Chunk--
PyTorch is an optimized tensor library for deep learning using GPUs and CPUs. It provides maximum flexibility and speed for research and production...

--New Chunk--
Scikit-learn is a machine learning library for the Python programming language. It features various classification, regression and clustering algorithms...
```

### Phase 4: Simple LLM Interaction

#### Naive System Prompt
```
You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer, just say so. Do not make anything up.

---Target response length and format---
{response_type}
```

**Characteristics**:
- **Minimal Instructions**: Basic helper role
- **Simple Context**: Direct text chunks without structure
- **Conservative Guidance**: Explicit instruction not to hallucinate

---

## Information Selection and Ranking Algorithms

### Entity Ranking Algorithm (Local Mode)
```python
# Multi-factor ranking system
entity_score = (
    vector_similarity_score * 0.4 +     # Query relevance
    graph_centrality_score * 0.3 +      # Network importance
    community_importance_score * 0.3    # Community significance
)
```

**Factors Explained**:
1. **Vector Similarity**: Semantic relevance to query (0-1 normalized)
2. **Graph Centrality**: Node degree / max_degree (structural importance)
3. **Community Importance**: Average rating of communities entity belongs to

### Community Ranking Algorithm (Global Mode)
```python
# Hierarchical importance calculation
community_score = (
    entity_occurrence * 0.5 +           # Size and representativeness
    llm_quality_rating * 0.3 +          # LLM-assessed importance (0-10)
    hierarchical_level_bonus * 0.2      # Higher levels get bonus
)
```

**Ranking Factors**:
1. **Entity Occurrence**: Number of entities in community (primary factor)
2. **Quality Rating**: LLM-generated importance score (0-10 scale)
3. **Hierarchical Level**: Higher-level communities get preference

### Chunk Ranking Algorithm (Naive Mode)
```python
# Simple vector similarity ranking
chunk_score = cosine_similarity(query_embedding, chunk_embedding)
```

**No Re-ranking**: Naive mode preserves original vector database ordering.

### Relationship Ranking Algorithm (Local Mode)
```python
# Edge importance calculation
edge_score = (
    edge_weight * 0.4 +                 # Relationship strength
    src_entity_rank * 0.3 +             # Source entity importance
    tgt_entity_rank * 0.3               # Target entity importance
)
```

---

## Context Assembly and Token Management

### Token Management Strategy

#### Per-Component Token Limits (Local Mode)
```python
# From base.py QueryParam class
local_max_token_for_text_unit: int = 4000        # Sources table
local_max_token_for_local_context: int = 4800    # Entities + Relationships
local_max_token_for_community_report: int = 3200 # Community reports
```

**Total Local Context**: ~12,000 tokens (within most LLM context windows)

#### Global Mode Token Limits
```python
global_max_token_for_community_report: int = 16384  # Per analyst group
global_max_consider_community: float = 512          # Max communities to consider
```

#### Naive Mode Token Limits
```python
naive_max_token_for_text_unit = 12000  # Simple concatenation limit
```

### Truncation Algorithm
**Location**: `_utils.py:truncate_list_by_token_size`

```python
def truncate_list_by_token_size(data_list, key, max_token_size):
    total_tokens = 0
    truncated_list = []

    for item in data_list:
        content = key(item) if callable(key) else item[key]
        tokens = encode_string_by_tiktoken(content, model_name="gpt-4o")

        if total_tokens + len(tokens) <= max_token_size:
            truncated_list.append(item)
            total_tokens += len(tokens)
        else:
            break  # First-fit truncation strategy

    return truncated_list
```

**Truncation Strategy**: First-fit (preserves ranking order, stops at first item that would exceed limit)

### Context Structure Optimization

#### Local Mode: Multi-Table Structure
- **Advantages**: Structured data, clear relationships, supports complex reasoning
- **Disadvantages**: Higher token usage, complex assembly logic

#### Global Mode: Hierarchical Analysis
- **Advantages**: Comprehensive coverage, multiple perspectives, handles broad queries
- **Disadvantages**: Highest computational cost, complex map-reduce pipeline

#### Naive Mode: Simple Concatenation
- **Advantages**: Lowest latency, simple implementation, good for factual queries
- **Disadvantages**: No structure, limited reasoning capability, token inefficient

---

## LLM Integration Patterns

### Response Format Configuration
**Location**: `base.py:QueryParam`

```python
response_type: str = "Multiple Paragraphs"  # Default format
```

**Common Response Types**:
- "Multiple Paragraphs" (default)
- "Single Paragraph"
- "Bullet Points"
- "Detailed Analysis"
- Custom formats as needed

### Model Selection Strategy
```python
# Configuration in GraphRAG class
best_model_func: callable = gemini_2_5_flash_complete     # Primary model
cheap_model_func: callable = gemini_2_5_flash_complete    # Fallback/caching
```

**Usage Patterns**:
- **Best Model**: Query responses, community reports
- **Cheap Model**: Entity summarization, caching operations

### Special LLM Parameters

#### Global Mode JSON Response
```python
global_special_community_map_llm_kwargs: dict = {
    "response_format": {"type": "json_object"}
}
```

**Purpose**: Ensures structured JSON output from map phase analysts

#### Community Report Generation
```python
special_community_report_llm_kwargs: dict = {
    "response_format": {"type": "json_object"}
}
```

**Purpose**: Enforces structured community report format

### Error Handling and Fallbacks
```python
# Fallback responses for various failure modes
PROMPTS["fail_response"] = "I'm sorry, I couldn't find relevant information to answer your question."

# Specific failure cases
if not len(results):  # No entities found
    return PROMPTS["fail_response"]

if context is None:  # Context building failed
    logger.error("Local query failed: Could not build query context")
    return PROMPTS["fail_response"]
```

---

## Configuration Parameters and Limits

### Query-Level Configuration
**Location**: `base.py:QueryParam`

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|---------|
| `mode` | "global" | Query processing mode | Determines entire processing pipeline |
| `level` | 2 | Community hierarchy depth | Controls which communities are considered |
| `top_k` | 20 | Initial retrieval size | Affects recall vs precision trade-off |
| `response_type` | "Multiple Paragraphs" | Output format | Guides LLM response structure |

### Local Mode Configuration
| Parameter | Default | Description | Impact |
|-----------|---------|-------------|---------|
| `local_max_token_for_text_unit` | 4000 | Sources table token limit | Controls text chunk inclusion |
| `local_max_token_for_local_context` | 4800 | Entities/relationships limit | Affects structured context size |
| `local_max_token_for_community_report` | 3200 | Community reports limit | Controls background context |
| `local_community_single_one` | False | Use only top community | Reduces context complexity |
| `include_text_chunks` | False | Include sources table | Significant token usage impact |

### Global Mode Configuration
| Parameter | Default | Description | Impact |
|-----------|---------|-------------|---------|
| `global_min_community_rating` | 0 | Minimum quality threshold | Filters low-quality communities |
| `global_max_consider_community` | 512 | Maximum communities | Caps computational complexity |
| `global_max_token_for_community_report` | 16384 | Per-analyst token limit | Controls analysis depth |

### Naive Mode Configuration
| Parameter | Default | Description | Impact |
|-----------|---------|-------------|---------|
| `naive_max_token_for_text_unit` | 12000 | Total context limit | Simple truncation threshold |

### System-Level Configuration
**Location**: `graphrag.py`

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|---------|
| `enable_local` | True | Enable local mode | Feature availability |
| `enable_naive_rag` | True | Enable naive mode | Feature availability |
| `query_better_than_threshold` | 0.2 | Vector similarity threshold | Query quality filtering |

---

## Performance Analysis

### Computational Complexity by Mode

#### Local Mode Complexity
- **Entity Search**: O(d × log n) where d = embedding dimensions, n = entities
- **Context Expansion**: O(k × e) where k = top entities, e = average entity edges
- **Context Assembly**: O(c + r + t) where c = communities, r = relationships, t = text chunks
- **Total**: O(k × e) dominated by graph traversal

#### Global Mode Complexity
- **Community Selection**: O(c × log c) where c = total communities
- **Map Phase**: O(g × t) where g = groups, t = tokens per group (parallelizable)
- **Reduce Phase**: O(p × log p) where p = total points from analysts
- **Total**: O(g × t) dominated by LLM analysis time

#### Naive Mode Complexity
- **Chunk Search**: O(d × log m) where d = embedding dimensions, m = chunks
- **Truncation**: O(k) where k = retrieved chunks
- **Total**: O(d × log m) most efficient mode

### Memory Requirements

#### Local Mode Memory Usage
- **Entity Data**: ~5KB per entity × 20 entities = ~100KB
- **Context Tables**: ~50-100KB depending on text inclusion
- **Graph Traversal**: ~10KB per hop
- **Total**: ~200-500KB per query

#### Global Mode Memory Usage
- **Community Reports**: ~10-50KB per community
- **Analysis Groups**: ~500KB-2MB per group
- **Analyst Points**: ~1-5KB per point
- **Total**: ~1-10MB per query (highest memory usage)

#### Naive Mode Memory Usage
- **Chunk Content**: ~5KB per chunk × 10-20 chunks = ~50-100KB
- **Total**: ~50-200KB per query (most efficient)

### Latency Characteristics

#### Local Mode Latency
```
Vector Search:    100-500ms
Graph Traversal:  200-1000ms
Context Assembly: 50-200ms
LLM Response:     1000-5000ms
Total:           1350-6700ms
```

#### Global Mode Latency
```
Community Select: 100-500ms
Map Phase:        2000-10000ms (parallel)
Reduce Phase:     1000-5000ms
Total:           3100-15500ms (highest latency)
```

#### Naive Mode Latency
```
Vector Search:    100-500ms
Truncation:       10-50ms
LLM Response:     1000-5000ms
Total:           1110-5550ms (most efficient)
```

### Scaling Characteristics

#### Entity Count Scaling
- **Local Mode**: Linear degradation after 10K entities
- **Global Mode**: Minimal impact (uses community abstractions)
- **Naive Mode**: No impact (uses chunks, not entities)

#### Document Count Scaling
- **Local Mode**: Moderate impact (more entities/communities)
- **Global Mode**: Linear with community count
- **Naive Mode**: Linear with chunk count

#### Query Complexity Scaling
- **Simple Factual**: Naive > Local > Global
- **Multi-hop Reasoning**: Local > Global > Naive
- **Broad Analysis**: Global > Local > Naive

---

## Troubleshooting and Optimization

### Common Performance Issues

#### "No entities found for query"
**Cause**: Vector similarity below threshold or missing embeddings
**Solutions**:
- Check entity embedding completeness
- Lower `query_better_than_threshold`
- Verify query phrasing matches entity vocabulary
- Use naive mode as fallback

#### "Retrieved 0 communities"
**Cause**: Community rating filters or hierarchy level issues
**Solutions**:
- Lower `global_min_community_rating`
- Increase `level` parameter
- Check community report generation success
- Verify clustering completed successfully

#### "Truncate 50 to 5 chunks"
**Cause**: Token limits too restrictive for chunk size
**Solutions**:
- Increase `naive_max_token_for_text_unit`
- Reduce chunk size in document processing
- Consider local mode for better selectivity

### Optimization Strategies

#### For High Entity Counts (>10K)
1. **Increase Vector Search Precision**: Reduce `top_k` to 10-15
2. **Use Community Filtering**: Enable `local_community_single_one`
3. **Reduce Context Inclusion**: Set `include_text_chunks=False`

#### For Broad Queries
1. **Prefer Global Mode**: Better coverage of diverse topics
2. **Increase Community Limit**: Raise `global_max_consider_community`
3. **Use Higher Hierarchy Levels**: Set `level=3` or higher

#### For Simple Factual Queries
1. **Use Naive Mode**: Fastest response time
2. **Increase Chunk Retrieval**: Raise `top_k` to 30-50
3. **Optimize Chunk Size**: Smaller chunks for better precision

#### For Token-Constrained Environments
1. **Reduce Context Components**: Disable text chunks in local mode
2. **Use Smaller Community Limits**: Lower token thresholds
3. **Implement Custom Truncation**: Preserve highest-ranked content

### Monitoring and Debugging

#### Key Log Messages to Monitor
```
# Entity search success
INFO:nano-graphrag:Using 12 entities, 3 communities, 8 relations, 15 text units

# Community processing progress
INFO:nano-graphrag:Retrieved 6 communities
INFO:nano-graphrag:Grouping to 1 groups for global search

# Truncation indicators
INFO:nano-graphrag:Truncate 19 to 10 chunks
```

#### Performance Metrics to Track
- **Query Response Time**: End-to-end latency
- **Context Token Usage**: Efficiency of information selection
- **Entity/Community Hit Rate**: Relevance of retrieved information
- **Truncation Ratio**: Information loss indicators

#### Debug Configuration
```python
# Enable detailed logging
import logging
logging.getLogger("nano-graphrag").setLevel(logging.DEBUG)

# Query with context only (no LLM call)
query_param = QueryParam(only_need_context=True)
context = await graphrag.aquery(query, query_param)
print(context)  # Inspect assembled context
```

---

## Summary

The nano-graphrag query system provides three complementary approaches to information retrieval, each optimized for different use cases:

1. **Local Mode**: Entity-centric retrieval with structured multi-table context, ideal for specific questions about entities and their relationships
2. **Global Mode**: Community-based analysis with map-reduce processing, designed for broad thematic queries requiring comprehensive coverage
3. **Naive Mode**: Direct text retrieval with simple concatenation, optimized for factual queries requiring minimal processing overhead

The system's sophisticated information selection algorithms, hierarchical ranking systems, and configurable token management enable efficient scaling across different query complexities and knowledge graph sizes. Understanding these mechanisms allows for optimal configuration and troubleshooting of the query pipeline.