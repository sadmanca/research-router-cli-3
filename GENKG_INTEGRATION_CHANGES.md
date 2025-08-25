# GenKG Integration Changes in Nano-GraphRAG

## Overview

This document explains how `genkg.py` was integrated into the nano-graphrag pipeline to enhance entity extraction and relationship creation using advanced LLM-based methods. The integration provides an alternative entity extraction mechanism while maintaining full compatibility with nano-graphrag's existing architecture.

## Key Files Modified

### 1. `nano-graphrag/nano_graphrag/graphrag.py`

**Location of Changes:** Lines 132-139 (Configuration), Lines 219-226 (Initialization), Lines 345-346 (Visualization)

#### Configuration Parameters Added:
```python
# GenKG integration
use_genkg_extraction: bool = False
genkg_node_limit: int = 25
genkg_llm_provider: str = "gemini"  
genkg_model_name: str = "gemini-2.5-flash"
genkg_create_visualization: bool = False
genkg_output_path: Optional[str] = None
```

#### Initialization Logic:
```python
# Configure GenKG if enabled
if self.use_genkg_extraction:
    logger.info("Using GenKG for entity extraction")
    self.entity_extraction_func = extract_entities_genkg
    
    # Set default output path if not provided and visualization is enabled
    if self.genkg_create_visualization and not self.genkg_output_path:
        self.genkg_output_path = os.path.join(self.working_dir, "output.html")
```

**Impact:** When `use_genkg_extraction=True`, the GraphRAG instance switches from the default `extract_entities` function to `extract_entities_genkg`, fundamentally changing how entities and relationships are extracted.

### 2. `nano-graphrag/nano_graphrag/_op.py`

**Location of Changes:** Lines 422-654 (New Function)

#### New Function Added:
```python
async def extract_entities_genkg(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
```

**Key Changes in Entity Extraction Pipeline:**

1. **Document Grouping**: Groups chunks by `full_doc_id` to reconstruct documents
2. **GenKG Processing**: Uses GenKG's `summarize_paper()` and `gemini_create_nodes()` methods
3. **Format Conversion**: Converts GenKG's format to nano-graphrag's expected format
4. **Node Normalization**: Applies Windows-compatible text cleaning and uppercase conversion
5. **Edge Processing**: Uses GenKG's `create_edges_by_gemini()` for relationship extraction

#### Processing Flow:
```python
# Group chunks by document
papers_dict = {}
for chunk_key, chunk_data in ordered_chunks:
    doc_id = chunk_data.get("full_doc_id", chunk_key)
    if doc_id not in papers_dict:
        papers_dict[doc_id] = ""
    papers_dict[doc_id] += chunk_data["content"] + "\n\n"

# Process each document with GenKG
async def _process_document(doc_id: str, doc_content: str):
    # 1. Summarize document
    summary = genkg.summarize_paper(doc_content, doc_id)
    
    # 2. Extract nodes using GenKG
    nodes_with_source = genkg.gemini_create_nodes(summary, node_limit, doc_id)
    
    # 3. Extract edges using GenKG
    edges = genkg.create_edges_by_gemini(nodes_with_source, {doc_id: summary})
    
    # 4. Convert to nano-graphrag format
    # ... format conversion code ...
```

### 3. New File: `nano-graphrag/genkg.py`

**Purpose:** Standalone GenKG implementation with LLM-based entity extraction

**Key Classes:**
- `LLMProvider`: Abstraction for different LLM providers (currently supports Gemini)
- `GenerateKG`: Main class for knowledge graph generation
- `KGNode` and `KGEdge`: Pydantic models for structured LLM responses

**Key Methods:**
- `summarize_paper()`: Creates focused summaries for entity extraction
- `gemini_create_nodes()`: Extracts entities using structured LLM prompts
- `create_edges_by_gemini()`: Creates relationships using LLM analysis
- `generate_knowledge_graph_from_chunks()`: Integration method for nano-graphrag

### 4. `research_router_cli/utils/config.py`

**Location of Changes:** Lines 108-112

```python
if self.has_gemini_config:
    config.update({
        'using_gemini': True,
        'use_genkg_extraction': True,  # ← NEW
        'genkg_node_limit': 25,        # ← NEW
        'genkg_create_visualization': True,  # ← NEW
        'genkg_llm_provider': 'gemini',     # ← NEW
        'genkg_model_name': 'gemini-2.5-flash'  # ← NEW
    })
```

**Impact:** Automatically enables GenKG extraction when Gemini configuration is detected.

## Insert Pipeline Changes

### Traditional Nano-GraphRAG Pipeline:
1. **Chunking** → Text split into chunks
2. **Entity Extraction** → LLM prompts extract entities using `extract_entities()`
3. **Graph Construction** → Entities and relationships added to graph
4. **Community Detection** → Graph clustering using Leiden algorithm
5. **Community Reports** → AI-generated summaries

### Modified Pipeline with GenKG:
1. **Chunking** → Text split into chunks *(unchanged)*
2. **Document Reconstruction** → Chunks grouped back into documents *(NEW)*
3. **Document Summarization** → GenKG creates focused summaries *(NEW)*
4. **Entity Extraction** → GenKG's LLM-based extraction using `extract_entities_genkg()` *(CHANGED)*
5. **Format Conversion** → GenKG format converted to nano-graphrag format *(NEW)*
6. **Graph Construction** → Entities and relationships added to graph *(unchanged)*
7. **Visualization Data Storage** → GenKG data saved for visualization *(NEW)*
8. **Community Detection** → Graph clustering using Leiden algorithm *(unchanged)*
9. **Community Reports** → AI-generated summaries *(unchanged)*
10. **GenKG Visualizations** → HTML/JSON outputs created *(NEW)*

## Specific Changes in Insert Flow

### Step 4: Entity Extraction (Line 325 in graphrag.py)
```python
# Before
maybe_new_kg = await extract_entities(inserting_chunks, ...)

# After (when use_genkg_extraction=True)
maybe_new_kg = await extract_entities_genkg(inserting_chunks, ...)
```

### Step 8: Visualization Generation (Lines 345-346 in graphrag.py)
```python
# NEW: Generate GenKG visualizations if enabled
if self.use_genkg_extraction and self.genkg_create_visualization:
    await self._generate_genkg_visualizations(inserting_chunks, new_docs)
```

## Data Flow Changes

### Traditional Flow:
```
Text Chunks → LLM Prompts → Entities/Relationships → Graph Storage
```

### GenKG Flow:
```
Text Chunks → Document Reconstruction → Summarization → GenKG LLM Analysis → 
Format Conversion → Graph Storage + Visualization Data Storage
```

## Configuration Impact

### Enabling GenKG:
```python
graphrag = GraphRAG(
    use_genkg_extraction=True,       # Switches to GenKG extraction
    genkg_node_limit=25,            # Entities per document
    genkg_create_visualization=True, # Enable HTML/JSON outputs
    genkg_llm_provider="gemini",    # LLM provider
    genkg_model_name="gemini-2.5-flash"  # Specific model
)
```

### Runtime Behavior Changes:
1. **Import Behavior**: GenKG is imported dynamically only when needed
2. **Error Handling**: Specific GenKG-related error messages and fallbacks
3. **Logging**: Additional log messages for GenKG processing steps
4. **File Outputs**: Creates `output.html`, `output.dashkg.json`, and `_genkg_viz_data.json`

## Storage Integration

### Visualization Data Storage:
The integration adds a new data storage mechanism for GenKG visualization data:

```python
# During extraction (in extract_entities_genkg)
genkg_data = {
    "nodes_with_source": [(dp["entity_name"], dp["source_id"]) for dp in all_entities_data],
    "edges": edges_for_viz,
    "papers_dict": papers_dict
}

# Stored in two places:
global_config["_genkg_viz_data"] = genkg_data  # In memory
# And as file: working_dir/_genkg_viz_data.json
```

### Later Retrieval (in _generate_genkg_visualizations):
```python
# Retrieved from file for visualization generation
viz_data_path = os.path.join(self.working_dir, "_genkg_viz_data.json")
with open(viz_data_path, 'r', encoding='utf-8') as f:
    genkg_viz_data = json.load(f)
```

## Backward Compatibility

The integration maintains full backward compatibility:
- Default behavior unchanged when `use_genkg_extraction=False`
- All existing nano-graphrag functionality preserved
- Same storage formats and query interfaces
- Identical graph structure and community detection

## Error Handling

### GenKG Import Failures:
```python
try:
    from genkg import GenerateKG
except ImportError as e:
    logger.error(f"Failed to import GenKG: {e}")
    raise ImportError(f"GenKG is required when use_genkg_extraction=True: {e}") from e
```

### Processing Failures:
- Document-level error handling with detailed error messages
- Graceful fallback behavior when GenKG processing fails
- Preservation of original error context for debugging

This integration successfully combines GenKG's advanced LLM-based entity extraction with nano-graphrag's efficient graph storage and querying capabilities, providing enhanced entity quality while maintaining the performance and functionality of the original system.