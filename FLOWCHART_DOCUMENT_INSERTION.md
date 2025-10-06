# Document Insertion Process Flowchart Content

This file provides structured content for creating a visual flowchart of the nano-graphrag document insertion process, focusing on GenKG entity extraction.

## Main Flow Structure

### Step 1: Document Input
**Title**: Document Input & Deduplication
**Description**: Raw documents (PDF/text) are ingested and deduplicated using MD5 hash-based identification to prevent re-processing of existing content.

### Step 2: Text Chunking
**Title**: Text Chunking with Sliding Window
**Description**: Documents are tokenized using GPT-4o tiktoken and split into overlapping chunks of 1200 tokens with 100-token overlap to preserve context across boundaries.

### Step 3: GenKG Entity Extraction
**Title**: GenKG Entity Extraction (Document-Level)
**Description**: Chunks are regrouped by document, then GenKG processes each full document by summarizing it and extracting up to 25 entities with relationships using advanced Gemini prompting.

#### Sub-process 3a: Document Summarization
**Title**: Document Summarization
**Description**: GenKG creates a summary of the full document content to provide better context for entity extraction.

#### Sub-process 3b: Node Extraction
**Title**: Node Extraction (25 limit)
**Description**: GenKG extracts entities from the summary with a hardcoded limit of 25 entities per document, using advanced prompting techniques.

#### Sub-process 3c: Edge Extraction
**Title**: Relationship Extraction
**Description**: GenKG analyzes extracted nodes to identify relationships between entities, creating weighted connections with descriptive labels.

#### Sub-process 3d: Node Normalization
**Title**: Node Name Normalization
**Description**: Entity names are cleaned and normalized (uppercase, special character replacement) for nano-graphrag compatibility while preserving original descriptions.

### Step 4: Graph Construction
**Title**: Graph Construction & Merging
**Description**: Extracted entities and relationships are merged with existing graph data, combining duplicate entities and summing relationship weights across mentions.

#### Sub-process 4a: Entity Merging
**Title**: Entity Merging Algorithm
**Description**: Entities with identical names are merged using majority vote for type, concatenated descriptions (auto-summarized if >500 tokens), and combined source tracking.

#### Sub-process 4b: Relationship Merging
**Title**: Relationship Merging Algorithm
**Description**: Duplicate relationships are merged by summing weights, concatenating descriptions, and preserving minimum order values across all mentions.

### Step 5: Community Detection
**Title**: Hierarchical Community Detection
**Description**: Leiden algorithm creates hierarchical communities (Level 0: fine-grained clusters, Level 1+: meta-communities) with reproducible results using seed 0xDEADBEEF.

### Step 6: Community Report Generation
**Title**: Community Report Generation (Top-Down)
**Description**: LLM generates structured JSON reports for each community starting from highest level, including title, summary, importance rating (0-10), and detailed findings.

### Step 7: Vector Database Population
**Title**: Vector Database Population
**Description**: Entity embeddings (name + description) and text chunk embeddings are generated and stored for query-time retrieval using Gemini embedding model.

#### Sub-process 7a: Entity Embeddings
**Title**: Entity Embeddings Creation
**Description**: Each entity gets an embedding created from concatenated name and description, stored with "ent-{hash}" ID format for vector search.

#### Sub-process 7b: Chunk Embeddings
**Title**: Chunk Embeddings Creation
**Description**: Text chunks get embeddings for naive RAG mode, processed in batches of 32 with up to 16 concurrent requests.

### Step 8: Storage Persistence
**Title**: Multi-System Storage Persistence
**Description**: Data is persisted across multiple storage systems: GraphML for graph structure, JSON for key-value data, NanoVectorDB for embeddings, and optional GenKG visualizations.

## Output Files Generated

### Core Storage Files
- `graph_chunk_entity_relation.graphml` - Main knowledge graph structure
- `kv_store_full_docs.json` - Original documents
- `kv_store_text_chunks.json` - Text chunks with metadata
- `kv_store_community_reports.json` - Generated community analyses
- `vdb_entities.json` - Entity embeddings for search
- `vdb_chunks.json` - Chunk embeddings for naive RAG

### Optional GenKG Visualization Files
- `output.html` - Interactive graph visualization
- `output.dashkg.json` - Dashboard-compatible graph data
- `_genkg_viz_data.json` - Intermediate visualization data

## Key Technical Limits

- **Chunk Size**: 1200 tokens (default)
- **Chunk Overlap**: 100 tokens
- **GenKG Entity Limit**: 25 entities per document (hardcoded)
- **Auto-Summarization Threshold**: 500 tokens for entity descriptions
- **Community Detection Algorithm**: Leiden with seed 0xDEADBEEF
- **Embedding Batch Size**: 32 embeddings per batch
- **Concurrent Embedding Requests**: 16 maximum
- **LLM Context Window**: 32,768 tokens for best model

## Decision Points

### Entity Extraction Method Selection
**Condition**: `use_genkg_extraction` configuration flag
- **True**: Use GenKG document-level processing
- **False**: Use traditional chunk-level LLM prompting

### Visualization Generation
**Condition**: `genkg_create_visualization` configuration flag
- **True**: Generate HTML and JSON visualization files
- **False**: Skip visualization generation

### Auto-Summarization Trigger
**Condition**: Entity description length > 500 tokens
- **True**: Use cheap LLM model to summarize description
- **False**: Keep original description