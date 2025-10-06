# ULTIMATE Document Insertion Guide: Nano-GraphRAG Knowledge Graph Construction

This document provides a comprehensive, technical deep-dive into how nano-graphrag transforms documents into a structured knowledge graph. This guide covers both high-level architecture and implementation details needed for creating diagram representations and understanding system behavior.

## Table of Contents

1. [High-Level Architecture Overview](#high-level-architecture-overview)
2. [Stage 1: Document Processing and Chunking](#stage-1-document-processing-and-chunking)
3. [Stage 2: Entity Extraction Methods](#stage-2-entity-extraction-methods)
4. [Stage 3: Graph Construction and Storage](#stage-3-graph-construction-and-storage)
5. [Stage 4: Community Detection](#stage-4-community-detection)
6. [Stage 5: Community Report Generation](#stage-5-community-report-generation)
7. [Stage 6: Vector Database Population](#stage-6-vector-database-population)
8. [Technical Configuration and Limits](#technical-configuration-and-limits)
9. [File Outputs and Storage Architecture](#file-outputs-and-storage-architecture)
10. [Performance Characteristics](#performance-characteristics)

---

## High-Level Architecture Overview

### System Flow Diagram
```
[PDF/Text Input]
    ↓
[Text Extraction & Chunking]
    ↓
[Entity Extraction] ←→ [Traditional LLM | GenKG Methods]
    ↓
[Graph Construction] → [Node/Edge Normalization & Merging]
    ↓
[Community Detection] → [Hierarchical Clustering (Leiden)]
    ↓
[Community Reports] → [LLM-Generated Structured Analysis]
    ↓
[Vector Database Population] → [Entity & Chunk Embeddings]
    ↓
[Storage Persistence] → [Multiple Storage Systems]
```

### Core Data Flow
1. **Input**: Raw documents (PDF, text)
2. **Processing**: Multi-stage transformation pipeline
3. **Output**: Structured knowledge graph with multiple access interfaces

---

## Stage 1: Document Processing and Chunking

### Document Ingestion
**Location**: `graphrag.py:280-295`

```python
# Document hash-based deduplication
new_docs = {
    compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
    for c in string_or_strings
}
```

**Key Features**:
- **Deduplication**: MD5 hash-based document identification prevents re-processing
- **Batch Processing**: Supports single documents or document lists
- **Content Normalization**: Strips whitespace and normalizes formatting

### Text Chunking Process
**Location**: `_op.py:32-62`

#### Chunking Algorithm: Token-Based Sliding Window
```python
def chunking_by_token_size(
    tokens_list, doc_keys, tiktoken_model,
    overlap_token_size=128,    # HARDCODED: Overlap between chunks
    max_token_size=1024        # HARDCODED: Max tokens per chunk
):
    for start in range(0, len(tokens), max_token_size - overlap_token_size):
        chunk_token.append(tokens[start : start + max_token_size])
```

**Technical Details**:
- **Tokenizer**: GPT-4o tiktoken encoding
- **Default Chunk Size**: 1200 tokens (configurable in GraphRAG class)
- **Default Overlap**: 100 tokens (configurable)
- **Sliding Window**: Ensures context preservation across chunk boundaries

#### Chunk Schema
```python
TextChunkSchema = {
    "tokens": int,              # Actual token count
    "content": str,             # Raw text content
    "full_doc_id": str,         # Source document hash ID
    "chunk_order_index": int    # Position within document
}
```

**Alternative Chunking**: Separator-based chunking available (`chunking_by_seperators`) using configurable text separators.

---

## Stage 2: Entity Extraction Methods

Nano-graphrag supports two distinct entity extraction approaches:

### Method A: Traditional LLM Prompt-Based Extraction
**Location**: `_op.py:304-399`

#### Multi-Round Gleaning Process
```python
# Initial extraction
final_result = await use_llm_func(entity_extract_prompt)

# Gleaning rounds (default: 1 additional round)
for gleaning_round in range(entity_extract_max_gleaning):
    glean_result = await use_llm_func(continue_prompt, history_messages=history)
    # Decision point: continue or stop
    if_continue = await use_llm_func(if_loop_prompt, history_messages=history)
    if if_continue.strip().lower() != "yes":
        break
```

#### LLM Prompt Structure
```
-Goal-
Given a text document, identify all entities and relationships.

-Steps-
1. Identify entities: ("entity"<|>ENTITY_NAME<|>ENTITY_TYPE<|>DESCRIPTION)
2. Identify relationships: ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>WEIGHT)
3. Return as delimited list using ## as delimiter
4. End with <|COMPLETE|>

-Entity Types-
organization, person, geo, event

Text: [CHUNK_CONTENT]
```

**Configuration**:
- **entity_extract_max_gleaning**: 1 (default additional rounds)
- **Tuple Delimiter**: `<|>`
- **Record Delimiter**: `##`
- **Completion Marker**: `<|COMPLETE|>`

### Method B: GenKG Advanced Extraction
**Location**: `_op.py:435-745`

#### Document-Level Processing
```python
# Group chunks back into full documents
papers_dict = {}
for chunk_key, chunk_data in chunks.items():
    doc_id = chunk_data.get("full_doc_id", chunk_key)
    papers_dict[doc_id] += chunk_data["content"] + "\n\n"

# Process each document as a whole
for doc_id, doc_content in papers_dict.items():
    # 1. Summarize document for better context
    summary = genkg.summarize_paper(doc_content, doc_id)

    # 2. Extract nodes with advanced prompting
    nodes_with_source = genkg.gemini_create_nodes(summary, node_limit, doc_id)

    # 3. Extract relationships with contextual analysis
    edges = genkg.create_edges_by_gemini(nodes_with_source, {doc_id: summary})
```

#### GenKG-Specific Configuration
- **genkg_node_limit**: 25 (hardcoded default per document)
- **genkg_llm_provider**: "gemini" (default)
- **genkg_model_name**: "gemini-2.5-flash" (default)

#### Node Normalization Process
```python
# Clean for nano-graphrag compatibility
clean_node_text = (node_text.strip()
                 .replace('(', ' ')
                 .replace(')', ' ')
                 .replace('-', ' ')
                 .replace('/', ' ')
                 .replace('&', 'AND'))

# Convert to uppercase for nano-graphrag compatibility
clean_node_text = ' '.join(clean_node_text.split()).upper()
```

#### Graph Connectivity Enhancement
GenKG includes automatic connectivity analysis:
```python
# Ensure graph connectivity across documents
enhanced_edges = genkg.ensure_graph_connectivity(nodes_with_source, edges_for_connectivity)

# Add connectivity edges with lower weights
edge_data = {
    "weight": 0.1,  # Lower weight for connectivity edges
    "description": "semantic_similarity",
    "source_id": "connectivity_enhancement"
}
```

---

## Stage 3: Graph Construction and Storage

### Entity and Relationship Merging
**Location**: `_op.py:206-300`

#### Node Merging Algorithm
```python
async def _merge_nodes_then_upsert(entity_name, nodes_data, knowledge_graph, global_config):
    # Merge with existing entities
    already_node = await knowledge_graph.get_node(entity_name)

    # Entity type determination (most common wins)
    entity_type = Counter([dp["entity_type"] for dp in nodes_data] + existing_types).most_common(1)[0][0]

    # Description merging (separated by GRAPH_FIELD_SEP)
    description = GRAPH_FIELD_SEP.join(sorted(set(all_descriptions)))

    # Source tracking
    source_id = GRAPH_FIELD_SEP.join(set(all_source_ids))

    # Auto-summarization if description too long
    description = await _handle_entity_relation_summary(entity_name, description, global_config)
```

#### Edge Merging Algorithm
```python
async def _merge_edges_then_upsert(src_id, tgt_id, edges_data, knowledge_graph, global_config):
    # Weight summation
    weight = sum([dp["weight"] for dp in edges_data] + existing_weights)

    # Description merging
    description = GRAPH_FIELD_SEP.join(sorted(set(all_descriptions)))

    # Order preservation (minimum order value)
    order = min([dp.get("order", 1) for dp in edges_data] + existing_orders)
```

### Auto-Summarization
**Location**: `_op.py:135-159`

When entity descriptions exceed token limits:
```python
# Trigger conditions
if len(tokens) >= entity_summary_to_max_tokens:  # Default: 500 tokens
    # Use cheap model for summarization
    summary = await cheap_model_func(summarize_prompt, max_tokens=summary_max_tokens)
```

**Configuration**:
- **entity_summary_to_max_tokens**: 500 (threshold for auto-summarization)
- **Uses**: cheap_model_func (typically faster/cheaper LLM)

### Graph Storage Schema
```python
# Node Data Structure
node_data = {
    "entity_type": str,         # Most common type across mentions
    "description": str,         # Merged and potentially summarized
    "source_id": str,          # GRAPH_FIELD_SEP separated chunk IDs
    "clusters": str            # JSON: [{"cluster": 0, "level": 0}, ...]
}

# Edge Data Structure
edge_data = {
    "weight": float,           # Summed weights from all mentions
    "description": str,        # Merged descriptions
    "source_id": str,         # GRAPH_FIELD_SEP separated chunk IDs
    "order": int              # Minimum order value
}
```

---

## Stage 4: Community Detection

### Hierarchical Clustering Process
**Location**: `_storage/gdb_networkx.py` (clustering method)

#### Leiden Algorithm Application
```python
async def clustering(self, algorithm: str = "leiden"):
    # Level 0: Base communities (fine-grained)
    communities = nx_leiden(self._graph, resolution=1.0, seed=0xDEADBEEF)

    # Level 1+: Meta-communities (hierarchical grouping)
    # Create meta-graph where each Level 0 community becomes a node
    meta_graph = build_meta_graph(level_0_communities)
    level_1_communities = nx_leiden(meta_graph)
```

#### Community Schema Structure
```python
community_schema = {
    "level": int,                    # Hierarchy level (0, 1, 2, ...)
    "title": str,                    # Auto-generated title
    "nodes": list[str],              # Entity names in community
    "edges": list[tuple[str, str]],  # Relationships within community
    "occurrence": int,               # Number of entities (importance metric)
    "sub_communities": list[str],    # Child communities (if level > 0)
    "parent_community": str          # Parent community (if level > 0)
}
```

**Log Message Explanation**:
```
INFO:nano-graphrag:Each level has communities: {0: 4, 1: 2}
```
- **Level 0**: 4 base communities (tightly connected clusters)
- **Level 1**: 2 super-communities (groups of Level 0 communities)

### Configuration Parameters
- **graph_cluster_algorithm**: "leiden" (default)
- **max_graph_cluster_size**: 10 (default)
- **graph_cluster_seed**: 0xDEADBEEF (reproducible results)

---

## Stage 5: Community Report Generation

### Top-Down Generation Strategy
**Location**: `_op.py:generate_community_report`

#### Processing Order
```python
# Generate from highest level to lowest
levels = sorted(set([c["level"] for c in communities]), reverse=True)
logger.info(f"Generating by levels: {levels}")  # e.g., [1, 0]

for level in levels:
    for community in communities_at_level:
        report = await generate_community_report_llm(community_data)
```

**Rationale**: Higher-level reports provide context for lower-level analysis.

#### Community Data Assembly
```python
async def _pack_single_community_describe(community):
    # Entities table (ranked by degree centrality)
    nodes_list_data = [
        [i, node_name, entity_type, description, degree]
        for i, (node_name, node_data) in enumerate(sorted_by_degree)
    ]

    # Relationships table
    edges_list_data = [
        [i, src, tgt, description, weight, rank]
        for i, edge_data in enumerate(community_edges)
    ]

    # Sub-community reports (if hierarchical)
    if level > 0:
        sub_reports = [already_reports[sub_id]["report_string"]
                      for sub_id in community["sub_communities"]]
```

#### LLM Report Generation Prompt
```
You are an AI assistant that helps a human analyst perform general information discovery.

# Goal
Write a comprehensive report of a community, given entities and relationships.

# Report Structure
- TITLE: Community name representing key entities
- SUMMARY: Executive summary of community structure
- IMPACT SEVERITY RATING: Float score 0-10 for importance
- RATING EXPLANATION: Single sentence explanation
- DETAILED FINDINGS: List of 5-10 key insights

Return output as JSON:
{
    "title": <report_title>,
    "summary": <executive_summary>,
    "rating": <impact_severity_rating>,
    "rating_explanation": <rating_explanation>,
    "findings": [
        {"summary": <insight_summary>, "explanation": <insight_explanation>}
    ]
}
```

#### Community Report Configuration
- **special_community_report_llm_kwargs**: `{"response_format": {"type": "json_object"}}`
- **Uses**: best_model_func (highest quality LLM)
- **Token Limit**: 12,000 per community context

---

## Stage 6: Vector Database Population

### Entity Embeddings
**Location**: `_op.py:423-431` and `_op.py:706-714`

```python
# Create embeddings for entity search
data_for_vdb = {
    compute_mdhash_id(entity["entity_name"], prefix="ent-"): {
        "content": entity["entity_name"] + entity["description"],
        "entity_name": entity["entity_name"]
    }
    for entity in all_entities_data
}
await entity_vdb.upsert(data_for_vdb)
```

**Key Details**:
- **Embedding Content**: Entity name + description concatenated
- **ID Format**: `ent-{md5_hash}`
- **Meta Fields**: entity_name for filtering

### Text Chunk Embeddings
**Location**: `graphrag.py:316-318`

```python
if self.enable_naive_rag:
    await self.chunks_vdb.upsert(inserting_chunks)
```

**Configuration**:
- **embedding_func**: Default gemini_embedding
- **embedding_batch_num**: 32 (batch processing)
- **embedding_func_max_async**: 16 (concurrent requests)

---

## Technical Configuration and Limits

### Hardcoded Limits and Defaults

#### Document Processing
| Parameter | Default Value | Description | Location |
|-----------|---------------|-------------|----------|
| `chunk_token_size` | 1200 | Tokens per chunk | `graphrag.py:74` |
| `chunk_overlap_token_size` | 100 | Overlap between chunks | `graphrag.py:75` |
| `tiktoken_model_name` | "gpt-4o" | Tokenizer model | `graphrag.py:76` |

#### Entity Extraction
| Parameter | Default Value | Description | Location |
|-----------|---------------|-------------|----------|
| `entity_extract_max_gleaning` | 1 | Additional extraction rounds | `graphrag.py:79` |
| `entity_summary_to_max_tokens` | 500 | Auto-summarization threshold | `graphrag.py:80` |
| `genkg_node_limit` | 25 | Entities per document (GenKG) | `graphrag.py:134` |

#### Graph Clustering
| Parameter | Default Value | Description | Location |
|-----------|---------------|-------------|----------|
| `graph_cluster_algorithm` | "leiden" | Community detection algorithm | `graphrag.py:83` |
| `max_graph_cluster_size` | 10 | Maximum cluster size | `graphrag.py:84` |
| `graph_cluster_seed` | 0xDEADBEEF | Reproducible clustering | `graphrag.py:85` |

#### LLM Configuration
| Parameter | Default Value | Description | Location |
|-----------|---------------|-------------|----------|
| `best_model_max_token_size` | 32768 | Context window for best model | `graphrag.py:116` |
| `cheap_model_max_token_size` | 32768 | Context window for cheap model | `graphrag.py:119` |
| `best_model_max_async` | 16 | Concurrent requests | `graphrag.py:117` |

#### Vector Database
| Parameter | Default Value | Description | Location |
|-----------|---------------|-------------|----------|
| `embedding_batch_num` | 32 | Embeddings per batch | `graphrag.py:108` |
| `embedding_func_max_async` | 16 | Concurrent embedding requests | `graphrag.py:109` |

---

## File Outputs and Storage Architecture

### Storage Systems Overview
Nano-graphrag uses multiple specialized storage systems:

#### 1. Graph Storage (NetworkX)
**File**: `graph_chunk_entity_relation.graphml`
```python
# Stores nodes and edges with all metadata
# Format: GraphML (XML-based graph format)
# Contains: normalized entity names, relationships, weights
```

#### 2. Key-Value Storage (JSON)
**Files**:
- `kv_store_full_docs.json` - Original documents
- `kv_store_text_chunks.json` - Text chunks with metadata
- `kv_store_community_reports.json` - Generated community analyses
- `kv_store_llm_response_cache.json` - LLM response caching

#### 3. Vector Storage (NanoVectorDB)
**Files**:
- `vdb_entities.json` - Entity embeddings for search
- `vdb_chunks.json` - Chunk embeddings for naive RAG

#### 4. GenKG Visualization Outputs (Optional)
**Files**:
- `output.html` - Interactive visualization
- `output.dashkg.json` - Graph data for dashboard
- `_genkg_viz_data.json` - Intermediate visualization data

### Storage Schema Details

#### Graph Storage Schema
```python
# Node attributes
{
    "entity_type": "ORGANIZATION",
    "description": "Description merged from all mentions",
    "source_id": "chunk-abc<SEP>chunk-def",  # Source chunks
    "clusters": '[{"cluster": 0, "level": 0}]'  # Community memberships
}

# Edge attributes
{
    "weight": 2.5,  # Summed weights
    "description": "Relationship description",
    "source_id": "chunk-abc<SEP>chunk-def",
    "order": 1
}
```

#### Community Reports Schema
```python
{
    "level": 0,
    "title": "Machine Learning Frameworks",
    "occurrence": 15,  # Number of entities
    "report_string": "Community report: This community focuses on...",
    "report_json": {
        "title": "Machine Learning Frameworks",
        "summary": "Executive summary...",
        "rating": 8.5,
        "rating_explanation": "High importance due to...",
        "findings": [
            {"summary": "Key insight", "explanation": "Detailed explanation"}
        ]
    }
}
```

### Directory Structure
```
working_dir/
├── graph_chunk_entity_relation.graphml          # Main knowledge graph
├── kv_store_full_docs.json                      # Original documents
├── kv_store_text_chunks.json                    # Text chunks
├── kv_store_community_reports.json              # Community analyses
├── kv_store_llm_response_cache.json            # LLM cache
├── vdb_entities.json                            # Entity embeddings
├── vdb_chunks.json                              # Chunk embeddings
├── output.html                                  # GenKG visualization (optional)
├── output.dashkg.json                           # GenKG dashboard data (optional)
└── _genkg_viz_data.json                        # GenKG intermediate data (optional)
```

---

## Performance Characteristics

### Computational Complexity

#### Time Complexity by Stage
1. **Chunking**: O(n) where n = document length
2. **Entity Extraction**: O(c × e) where c = chunks, e = LLM extraction time
3. **Graph Merging**: O(e²) where e = extracted entities
4. **Community Detection**: O(e³) for Leiden algorithm
5. **Report Generation**: O(r × t) where r = communities, t = LLM generation time

#### Bottlenecks and Optimization Points
1. **LLM API Calls**: Parallelized with configurable limits
2. **Entity Merging**: In-memory operations, scales with entity count
3. **Community Detection**: CPU-intensive, benefits from graph optimization
4. **Embedding Generation**: Batched and parallelized

### Memory Requirements
- **Base Memory**: ~100MB for framework
- **Per Document**: ~10-50MB depending on size and entity density
- **Graph Storage**: ~1-5MB per 1000 entities
- **Vector Embeddings**: ~6KB per entity (1536-dim embeddings)

### Scaling Characteristics
- **Documents**: Linear scaling up to 1000s of documents
- **Entities**: Quadratic scaling in worst case due to graph operations
- **Communities**: Logarithmic scaling with good clustering
- **Optimal Performance**: 10-100 documents, 1000-10000 entities per batch

### Performance Monitoring
Log messages indicate progress and bottlenecks:
```
INFO:nano-graphrag:[New Docs] inserting 1 docs
INFO:nano-graphrag:[New Chunks] inserting 19 chunks
INFO:nano-graphrag:[Entity Extraction]...
INFO:nano-graphrag:GenKG successfully extracted 25 entities
INFO:nano-graphrag:Each level has communities: {0: 4, 1: 2}
INFO:nano-graphrag:o Processed 4 communities
```

---

## Summary

The nano-graphrag document insertion process represents a sophisticated multi-stage pipeline that transforms raw documents into a structured, searchable knowledge graph. The system balances accuracy and performance through:

1. **Flexible Entity Extraction**: Supporting both traditional LLM prompts and advanced GenKG methods
2. **Hierarchical Knowledge Organization**: Multi-level community detection for different analysis granularities
3. **Multiple Access Patterns**: Vector search, graph traversal, and direct text retrieval
4. **Comprehensive Caching**: LLM response caching and incremental processing
5. **Rich Metadata**: Detailed provenance tracking and quality metrics

The resulting knowledge graph supports sophisticated querying capabilities while maintaining interpretability through generated community reports and visualization options.