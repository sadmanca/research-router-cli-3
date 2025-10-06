# Querying Process Flowchart Content

This file provides structured content for creating visual flowcharts of all three nano-graphrag query processing modes.

## Query Mode Selection

### Initial Decision Point
**Title**: Query Mode Selection
**Description**: System routes query to one of three processing modes based on configuration and query requirements.

**Routing Options**:
- **Local Mode**: Entity-focused questions requiring relationship context
- **Global Mode**: Broad thematic analysis requiring comprehensive coverage
- **Naive Mode**: Simple factual queries requiring fast responses

---

## Local Query Mode Flowchart

### Phase 1: Entity Discovery
**Title**: Entity Vector Search
**Description**: Search entity vector database using query embedding to find top 20 most semantically similar entities based on name+description content.

#### Step 1.1: Vector Similarity Search
**Title**: Vector Similarity Search (Top-20)
**Description**: Query embedding compared against entity embeddings using cosine similarity, returning 20 most relevant entities with distance scores.

#### Step 1.2: Entity Metadata Retrieval
**Title**: Entity Metadata Retrieval
**Description**: Fetch full entity data from knowledge graph including type, description, source chunks, community memberships, and calculate node degree (connection count).

### Phase 2: Context Expansion

#### Step 2.1: Community Discovery
**Title**: Related Communities Discovery
**Description**: Extract community memberships from query entities, rank communities by frequency (how many query entities belong) and LLM-assigned importance rating (0-10).

#### Step 2.2: Text Chunk Discovery
**Title**: Related Text Chunks Discovery
**Description**: Find source chunks that mention query entities plus one-hop neighbor chunks, scored by relationship density (interconnectedness of contained entities).

#### Step 2.3: Relationship Discovery
**Title**: Related Relationships Discovery
**Description**: Retrieve all edges involving query entities, ranked by edge weight, source entity importance, and target entity importance.

### Phase 3: Multi-Table Context Assembly

#### Step 3.1: Community Reports Table
**Title**: Community Reports Table Creation
**Description**: Format selected community reports as CSV table with ID, content, rating, and importance columns for LLM consumption.

#### Step 3.2: Entities Table
**Title**: Entities Table Creation
**Description**: Format entity data as CSV with ID, entity name, type, description, and rank columns, ordered by importance.

#### Step 3.3: Relationships Table
**Title**: Relationships Table Creation
**Description**: Format relationship data as CSV with ID, source, target, description, weight, and rank columns.

#### Step 3.4: Sources Table (Optional)
**Title**: Sources Table Creation
**Description**: Format text chunks as CSV with ID and content columns (only if include_text_chunks=True, default False).

### Phase 4: LLM Response Generation
**Title**: Structured LLM Response
**Description**: Send multi-table CSV context to LLM with instructions to synthesize information from tables only, following specified response format.

**Token Limits**:
- Community Reports: 3,200 tokens
- Entities + Relationships: 4,800 tokens
- Text Chunks: 4,000 tokens
- **Total**: ~12,000 tokens

---

## Global Query Mode Flowchart

### Phase 1: Community Selection

#### Step 1.1: Community Retrieval
**Title**: Community Retrieval & Filtering
**Description**: Get all communities at specified hierarchy level (default: ≤2), sort by entity count (occurrence), apply quality rating filter (≥0), limit to top 512 communities.

**Selection Criteria Priority**:
1. Hierarchy level filtering (≤ level parameter)
2. Entity count (occurrence) - higher is better
3. LLM quality rating (0-10 scale) - filter minimum threshold
4. Maximum limit cap (512 communities default)

### Phase 2: Map Phase (Parallel Analysis)

#### Step 2.1: Community Grouping
**Title**: Community Grouping by Token Limits
**Description**: Group communities into batches that fit within 16,384 token limit per group, ensuring efficient parallel processing without context overflow.

#### Step 2.2: Parallel Analyst Processing
**Title**: Parallel Analyst Processing
**Description**: Each group processed by separate "analyst" - LLM extracts key points relevant to query from community reports, assigning importance scores (0-100).

**Map Phase Output**: JSON format with points array containing description and score for each insight.

### Phase 3: Reduce Phase (Synthesis)

#### Step 3.1: Point Aggregation
**Title**: Point Aggregation & Ranking
**Description**: Collect insights from all analysts, filter out zero-scored points, rank by importance score, and truncate to fit final context window (16,384 tokens).

#### Step 3.2: Final Context Assembly
**Title**: Final Context Assembly
**Description**: Format analyst insights as ranked list with analyst ID, importance score, and insight text for final LLM synthesis.

#### Step 3.3: Comprehensive Response Generation
**Title**: Comprehensive Response Synthesis
**Description**: LLM synthesizes multiple analyst perspectives into comprehensive answer, removing irrelevant information and merging insights.

**Token Limits**:
- Per Analyst Group: 16,384 tokens
- Final Context: 16,384 tokens
- Max Communities: 512

---

## Naive Query Mode Flowchart

### Phase 1: Direct Chunk Search
**Title**: Direct Chunk Vector Search
**Description**: Search text chunk embeddings using query, retrieve top 20 most similar chunks based on full content similarity, no re-ranking applied.

### Phase 2: Token-Based Truncation
**Title**: Simple Token-Based Truncation
**Description**: Apply first-fit truncation to retrieved chunks, keeping chunks in similarity order until 12,000 token limit reached.

**Log Output**: "Truncate X to Y chunks" showing original vs final count

### Phase 3: Simple Concatenation
**Title**: Simple Text Concatenation
**Description**: Join truncated chunks with "--New Chunk--" separators, creating plain text context without structure or metadata.

### Phase 4: Direct LLM Response
**Title**: Direct LLM Response
**Description**: Send concatenated chunks to LLM with minimal instructions, conservative guidance to avoid hallucination.

**Token Limits**:
- Total Context: 12,000 tokens
- No sub-component limits (single concatenated text)

---

## Information Selection Algorithms

### Entity Ranking (Local Mode)
**Formula**: `entity_score = vector_similarity(0.4) + graph_centrality(0.3) + community_importance(0.3)`

**Factors**:
1. **Vector Similarity**: Semantic relevance to query (0-1)
2. **Graph Centrality**: Node degree / max_degree (structural importance)
3. **Community Importance**: Average rating of entity's communities

### Community Ranking (Global Mode)
**Formula**: `community_score = entity_occurrence(0.5) + llm_rating(0.3) + level_bonus(0.2)`

**Factors**:
1. **Entity Occurrence**: Number of entities in community (primary)
2. **LLM Quality Rating**: Generated importance score (0-10)
3. **Hierarchical Level**: Higher levels get preference bonus

### Chunk Ranking (Naive Mode)
**Formula**: `chunk_score = cosine_similarity(query_embedding, chunk_embedding)`

**No Re-ranking**: Preserves original vector database similarity ordering

### Relationship Ranking (Local Mode)
**Formula**: `edge_score = edge_weight(0.4) + src_entity_rank(0.3) + tgt_entity_rank(0.3)`

**Factors**:
1. **Edge Weight**: Relationship strength from graph
2. **Source Entity Rank**: Importance of source entity
3. **Target Entity Rank**: Importance of target entity

---

## Key Configuration Parameters

### Query-Level Parameters
- **mode**: "local" | "global" | "naive" (default: "global")
- **level**: Community hierarchy depth (default: 2)
- **top_k**: Initial retrieval size (default: 20)
- **response_type**: Output format (default: "Multiple Paragraphs")

### Mode-Specific Token Limits
**Local Mode**:
- Sources table: 4,000 tokens
- Entities + relationships: 4,800 tokens
- Community reports: 3,200 tokens

**Global Mode**:
- Per analyst group: 16,384 tokens
- Max communities: 512
- Final context: 16,384 tokens

**Naive Mode**:
- Total context: 12,000 tokens

### Performance Characteristics
**Local Mode**: 1,350-6,700ms (medium complexity)
**Global Mode**: 3,100-15,500ms (highest latency)
**Naive Mode**: 1,110-5,550ms (most efficient)

---

## Error Handling & Fallbacks

### Common Failure Points

#### "No entities found for query"
**Cause**: Vector similarity below threshold or missing embeddings
**Fallback**: Return fail_response message

#### "Retrieved 0 communities"
**Cause**: Community rating filters or hierarchy level issues
**Fallback**: Return fail_response message

#### "Truncate X to 0 chunks"
**Cause**: Token limits too restrictive for content size
**Fallback**: Return fail_response message

### Optimization Decision Points

#### High Entity Count (>10K)
**Action**: Reduce top_k to 10-15, enable single community mode, disable text chunks

#### Broad Query Type
**Action**: Prefer global mode, increase community limit, use higher hierarchy levels

#### Simple Factual Query
**Action**: Use naive mode, increase chunk retrieval, optimize for speed