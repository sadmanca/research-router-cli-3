# Nano-GraphRAG Deep Dive: How It Actually Works

This document explains the conceptual workings of nano-graphrag with detailed explanations of HOW each process works, not just WHERE the code is located.

## Table of Contents

1. [System Overview: The Big Picture](#system-overview-the-big-picture)
2. [How Entity Extraction Really Works](#how-entity-extraction-really-works)
3. [How Knowledge Graph Construction Works](#how-knowledge-graph-construction-works)
4. [How Community Detection Creates Hierarchies](#how-community-detection-creates-hierarchies)
5. [How the Three Query Methods Work](#how-the-three-query-methods-work)
6. [How Files Are Generated and Stored](#how-files-are-generated-and-stored)
7. [Complete Walkthrough Examples](#complete-walkthrough-examples)

---

## System Overview: The Big Picture

**Nano-graphrag is fundamentally a document-to-knowledge-graph-to-answer pipeline.** Here's what happens conceptually:

### The Core Transformation Process

1. **Documents → Text Chunks**: Raw documents are split into overlapping chunks (default: 32,768 tokens with 2,048 token overlap)
2. **Text Chunks → Entities & Relationships**: Each chunk is analyzed by an LLM to extract named entities and their relationships
3. **Entities & Relationships → Knowledge Graph**: Extracted information is merged and deduplicated to form a coherent graph structure
4. **Knowledge Graph → Communities**: The graph is clustered into hierarchical communities using the Leiden algorithm
5. **Communities → Reports**: Each community gets an AI-generated summary report describing its significance
6. **Everything → Vector Embeddings**: Entities and chunks are embedded for similarity search

### The Three Ways to Query

The system can answer questions using three fundamentally different approaches:

- **Local Query**: "Find specific entities related to the question, then gather all information about those entities"
- **Global Query**: "Analyze how different communities (topics) relate to the question, then synthesize insights"
- **Naive Query**: "Just find similar text chunks and answer based on those (traditional RAG)"

### Why This Architecture Works

**The key insight**: Most questions fall into two categories - either you need specific facts about known entities (local), or you need thematic analysis across broad topics (global). The knowledge graph structure enables both approaches, while the community hierarchy provides different levels of abstraction for analysis.

---

## How Entity Extraction Really Works

**The fundamental challenge**: How do you turn unstructured text into a structured knowledge graph? Nano-graphrag solves this through a sophisticated multi-stage pipeline that combines chunking, LLM prompting, and intelligent merging.

### The Conceptual Approach

**Think of entity extraction as a three-phase translation process:**

1. **Phase 1 - Text Preparation**: Break documents into manageable, overlapping chunks that preserve context
2. **Phase 2 - LLM Analysis**: Use carefully crafted prompts to get the LLM to identify entities and relationships in a structured format
3. **Phase 3 - Intelligent Merging**: Combine and deduplicate findings across chunks to create a coherent knowledge graph

### How Text Chunking Actually Works

**The core problem**: LLMs have token limits, but documents can be arbitrarily long. The solution is intelligent chunking with overlap.

**Why overlapping chunks matter**: Entities and relationships often span chunk boundaries. Without overlap, you'd lose connections between concepts that appear near boundaries.

**The chunking strategy** (implemented in `_op.py:66-108`):
- **Default chunk size**: 32,768 tokens (massive chunks for modern LLMs with large contexts)
- **Overlap size**: 2,048 tokens (about 1-2 pages of overlap)
- **Step size**: 30,720 tokens (chunk_size - overlap)

**Example**: For a 100,000 token document:
- Chunk 1: Tokens 0-32,767
- Chunk 2: Tokens 30,720-63,487 (overlaps with Chunk 1 by 2,048 tokens)
- Chunk 3: Tokens 61,440-94,207 (overlaps with Chunk 2 by 2,048 tokens)
- Chunk 4: Tokens 92,160-100,000 (final chunk, shorter)

**Why this works**: Any entity or relationship that appears near a chunk boundary will appear completely in at least one chunk, often in two chunks, ensuring nothing is lost.

### How LLM Entity Extraction Actually Works

**The core insight**: Instead of trying to teach the LLM about knowledge graphs, give it a simple structured output format and let it do what it does best - understand text and follow instructions.

**The prompt engineering strategy** (from `prompt.py:195-294`):

1. **Clear Goal Definition**: "Given a text document... identify all entities of those types from the text and all relationships among the identified entities"

2. **Specific Output Format**: The LLM must output entities as:
   ```
   ("entity"|"APPLE INC"|"organization"|"Technology company that makes iPhones")##
   ```
   And relationships as:
   ```
   ("relationship"|"APPLE INC"|"IPHONE"|"Apple Inc manufactures the iPhone"|8)##
   ```

3. **Structured Instructions**: Step-by-step process with examples showing exactly what the output should look like

**Why this format works**:
- **Parseable**: Easy to split on delimiters and extract structured data
- **Unambiguous**: Each field has a clear purpose and position
- **Extensible**: Can add more fields without breaking the parser
- **LLM-Friendly**: Uses natural delimiters that LLMs handle well

### The Entity Merging Intelligence

**The critical challenge**: The same entity appears in multiple chunks with slightly different names or descriptions. How do you merge them intelligently?

**Nano-graphrag's solution** (implemented in `_op.py:252-297`):

**1. Name Normalization**: All entity names are converted to uppercase for matching
- "Apple Inc" and "APPLE INC" and "Apple, Inc." all become "APPLE INC"

**2. Type Resolution**: When the same entity has different types across chunks, pick the most common one
- If "APPLE INC" appears as "organization" 3 times and "company" 1 time, it becomes "organization"

**3. Description Merging**: Combine all descriptions with a separator (`<SEP>`)
- Instead of losing information, accumulate knowledge: "Technology company<SEP>Makes iPhones<SEP>Founded in 1976"

**4. Source Tracking**: Keep track of which chunks contributed to each entity
- Enables tracing back to original sources for any piece of information

**5. Relationship Weight Aggregation**: Sum relationship weights across chunks
- If "APPLE INC" → "IPHONE" appears with weight 8 in one chunk and weight 6 in another, final weight is 14

### The Smart Relationship Handling

**The relationship challenge**: How do you handle relationships when entities might be referenced differently or when relationships are implied rather than explicit?

**Nano-graphrag's approach**:

1. **Bidirectional Processing**: Relationships are stored as directed edges but can be queried in both directions
2. **Weight-Based Importance**: Stronger relationships (higher weights) are prioritized in queries
3. **Description Accumulation**: Multiple relationship descriptions are preserved, not overwritten
4. **Automatic Node Creation**: If a relationship references an entity that wasn't explicitly extracted, create a placeholder node

### Why This Extraction Method Works

**Traditional NER (Named Entity Recognition) problems**:
- Limited to predefined entity types
- Misses context and relationships
- Struggles with domain-specific terminology
- Can't capture complex multi-word entities

**Nano-graphrag's advantages**:
- **Flexible Entity Types**: The LLM can identify any type of entity based on context
- **Rich Relationships**: Captures not just that entities are related, but HOW they're related
- **Context Preservation**: Entity descriptions maintain the original context
- **Domain Adaptability**: Works across different domains without retraining
- **Relationship Strength**: Quantifies how strongly entities are connected

**The key insight**: Instead of trying to solve NER as a classification problem, treat it as a text understanding and structured generation problem. Modern LLMs excel at this kind of task.

### Advanced Entity Extraction Techniques

#### The Gleaning Process: Iterative Refinement

**What is "gleaning"?** Named after the agricultural practice of collecting leftover crops after the main harvest, gleaning in nano-graphrag means making multiple passes over the same text to extract entities that might have been missed.

**The gleaning algorithm** (controlled by `entity_extract_max_gleaning` parameter):

1. **First Pass**: Extract entities using the standard prompt
2. **Analysis**: Compare the number of entities found vs. expected (based on text length heuristics)
3. **Second Pass**: If entities seem sparse, run a follow-up extraction with a modified prompt: "MANY entities were missed in the last extraction. Add them below using the same format:"
4. **Validation**: Check if new entities were found; if yes, potentially do a third pass

**Why gleaning works**:
- **LLM Attention**: Different prompts can cause the LLM to focus on different aspects of the text
- **Context Sensitivity**: Later passes can catch entities that were overshadowed by more prominent ones
- **Completeness**: Ensures comprehensive extraction for dense, information-rich texts

**Gleaning example**:
```python
# First extraction finds: ["APPLE INC", "IPHONE", "TIM COOK"]
# Gleaning prompt: "MANY entities were missed. Add them below..."
# Second extraction adds: ["CUPERTINO", "CALIFORNIA", "STEVE JOBS", "IPAD", "MAC"]
```

#### Advanced Prompt Engineering Strategies

**The Multi-Layer Prompting Approach**:

**Layer 1 - Goal Setting**: Clear, unambiguous instructions about what to extract
```text
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, 
identify all entities of those types from the text and all relationships among the identified entities.
```

**Layer 2 - Format Specification**: Exact output format with delimiters
```text
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
```

**Layer 3 - Examples**: Multiple concrete examples showing the desired behavior
```text
Example 1: [Simple technology example]
Example 2: [Complex multi-entity example]  
Example 3: [Edge case with concepts and events]
```

**Layer 4 - Real Data**: The actual text to process
```text
-Real Data-
Entity_types: {entity_types}
Text: {input_text}
```

**Why this structure works**:
- **Cognitive Priming**: Each layer prepares the LLM for the next level of complexity
- **Format Reinforcement**: Multiple examples ensure consistent output formatting
- **Error Reduction**: Clear specifications reduce ambiguous interpretations

#### Sophisticated Response Parsing

**The parsing challenge**: LLMs don't always follow format specifications perfectly. Nano-graphrag handles multiple types of format variations:

**1. Delimiter Variations**:
```python
# Expected: ("entity"|"APPLE INC"|"organization"|"Technology company")
# Actual LLM outputs handled:
# - ("entity", "APPLE INC", "organization", "Technology company")
# - ("entity" | "APPLE INC" | "organization" | "Technology company")
# - entity|APPLE INC|organization|Technology company
```

**2. Incomplete Records**:
```python
# Missing fields are handled gracefully:
# ("entity"|"APPLE INC"|"organization")  # Missing description - gets "UNKNOWN"
# ("entity"|"APPLE INC")  # Missing type and description - gets defaults
```

**3. Malformed Delimiters**:
```python
# The parser handles various completion indicators:
# <|COMPLETE|>, COMPLETE, [COMPLETE], ###COMPLETE###
```

**4. Nested Content**:
```python
# Entities with complex descriptions:
# ("entity"|"APPLE INC"|"organization"|"Technology company that makes smartphones, tablets, and computers; founded in 1976")
```

**The robust parsing algorithm** (implemented in `_op.py:170-249`):

```python
def parse_llm_response(response_text, chunk_key):
    # Step 1: Clean and normalize the response
    cleaned = response_text.strip()
    
    # Step 2: Split on record delimiters with fallbacks
    records = split_with_multiple_delimiters(cleaned, ["##", "\n\n", "|||"])
    
    # Step 3: Process each record with error recovery
    entities, relationships = [], []
    for record in records:
        try:
            # Try primary parsing approach
            parsed = parse_single_record(record, chunk_key)
            if parsed['type'] == 'entity':
                entities.append(parsed)
            elif parsed['type'] == 'relationship':
                relationships.append(parsed)
        except ParsingError:
            # Try secondary parsing approaches
            try:
                parsed = fallback_parse(record, chunk_key)
                # ... handle fallback result
            except:
                logger.warning(f"Failed to parse record: {record[:100]}...")
                continue
    
    return entities, relationships
```

#### Entity Validation and Quality Control

**Validation layers** ensure extracted entities are meaningful:

**1. Name Validation**:
```python
def validate_entity_name(name):
    # Reject names that are too short, too long, or nonsensical
    if len(name) < 2 or len(name) > 100:
        return False
    if name.count(' ') > 10:  # Probably malformed
        return False
    if is_generic_term(name):  # "thing", "item", "stuff"
        return False
    return True
```

**2. Type Consistency**:
```python
def validate_entity_type(entity_type, description):
    # Check if type makes sense given the description
    if entity_type == "person" and "company" in description.lower():
        return "organization"  # Auto-correct obvious mistakes
    return entity_type
```

**3. Relationship Coherence**:
```python
def validate_relationship(source, target, description, weight):
    # Ensure relationships make logical sense
    if source == target:  # Self-relationships are suspicious
        return None
    if weight < 0 or weight > 10:  # Weight out of expected range
        weight = max(1, min(10, weight))
    return {"source": source, "target": target, "description": description, "weight": weight}
```

#### Performance Optimizations for Entity Extraction

**1. Batch Processing Strategy**:
Instead of processing chunks one by one, nano-graphrag processes them in batches:

```python
async def extract_entities_batch(chunks, batch_size=8):
    # Process multiple chunks simultaneously
    chunk_batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    
    all_results = []
    for batch in chunk_batches:
        # Process batch in parallel
        batch_tasks = [extract_entities_single(chunk) for chunk in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_results.extend(batch_results)
    
    return all_results
```

**2. Token Management**:
```python
def optimize_chunk_for_extraction(chunk_content, max_tokens):
    # Smart truncation that preserves entity contexts
    if count_tokens(chunk_content) <= max_tokens:
        return chunk_content
    
    # Try to truncate at sentence boundaries
    sentences = split_into_sentences(chunk_content)
    truncated = ""
    for sentence in sentences:
        if count_tokens(truncated + sentence) > max_tokens:
            break
        truncated += sentence + " "
    
    return truncated.strip()
```

**3. Caching Strategy**:
```python
# LLM responses are cached by content hash
cache_key = compute_hash(chunk_content + extraction_prompt)
if cache_key in llm_response_cache:
    return cached_response
else:
    response = await call_llm(prompt)
    llm_response_cache[cache_key] = response
    return response
```

#### Error Recovery and Fallback Mechanisms

**When entity extraction fails**, nano-graphrag has multiple recovery strategies:

**1. Partial Success Handling**:
```python
if len(extracted_entities) == 0 and len(text) > 1000:
    # Text is substantial but no entities found - retry with simplified prompt
    simplified_prompt = create_simple_extraction_prompt(text)
    retry_result = await llm_call(simplified_prompt)
    return parse_with_lower_standards(retry_result)
```

**2. Format Recovery**:
```python
def recover_from_format_errors(malformed_response):
    # Try to extract entities even from poorly formatted responses
    patterns = [
        r'entity[:\s]*([^|]+)[|]+([^|]+)[|]+([^|]+)',  # Basic pattern
        r'([A-Z][A-Z\s]+).*?(?:organization|person|geo|event)',  # Name + type
        r'"([^"]+)"\s*(?:is|was|are)\s*(?:a|an)\s*([^.]+)',  # "Apple Inc" is a technology company
    ]
    
    recovered_entities = []
    for pattern in patterns:
        matches = re.findall(pattern, malformed_response, re.IGNORECASE)
        for match in matches:
            entity = create_entity_from_match(match)
            recovered_entities.append(entity)
    
    return recovered_entities
```

**3. Minimum Viable Extraction**:
```python
def ensure_minimum_entities(text, extracted_entities):
    # If very few entities found, extract at least proper nouns
    if len(extracted_entities) < 3 and len(text) > 500:
        proper_nouns = extract_proper_nouns_nltk(text)
        for noun in proper_nouns:
            if noun not in [e['name'] for e in extracted_entities]:
                extracted_entities.append({
                    'name': noun.upper(),
                    'type': 'UNKNOWN',
                    'description': f'Proper noun found in text: {noun}',
                    'source': 'fallback_extraction'
                })
    
    return extracted_entities
```

---

## How Knowledge Graph Construction Works

**The transformation challenge**: Once you have extracted entities and relationships from individual chunks, how do you turn this fragmented information into a coherent, queryable knowledge graph?

### The Graph Building Philosophy

**Nano-graphrag treats knowledge graph construction as an incremental assembly process**, not a one-shot creation. Here's the conceptual approach:

1. **Start with Nothing**: Begin with an empty graph structure
2. **Add Incrementally**: For each chunk's extracted entities and relationships, merge them into the existing graph
3. **Resolve Conflicts**: When the same entity appears multiple times, intelligently merge the information
4. **Maintain Consistency**: Ensure the graph remains coherent as new information is added

### How Graph Merging Actually Works

**The central problem**: Different chunks might extract slightly different information about the same real-world entities. How do you reconcile these differences?

**Example scenario**: 
- Chunk 1 extracts: `("entity"|"Apple Inc"|"company"|"Makes smartphones")`
- Chunk 2 extracts: `("entity"|"APPLE INC"|"organization"|"Technology company founded in 1976")`
- Chunk 3 extracts: `("entity"|"Apple"|"tech company"|"Creator of iPhone and Mac")`

**These are clearly the same entity, but how does the system know?**

### The Entity Reconciliation Process

**Step 1 - Name Normalization** (implemented in `_op.py:252-297`):
- Convert all names to uppercase: "Apple Inc" → "APPLE INC"
- This handles most common variations in capitalization and formatting

**Step 2 - Intelligent Type Resolution**:
- Count occurrences of each entity type across all chunks
- Choose the most frequently occurring type
- In our example: "organization" appears twice, so it wins over "company" and "tech company"

**Step 3 - Description Aggregation**:
- Instead of choosing one description, combine them all using a separator
- Result: "Makes smartphones<SEP>Technology company founded in 1976<SEP>Creator of iPhone and Mac"
- This preserves all information rather than discarding it

**Step 4 - Source Tracking**:
- Keep track of every chunk that contributed information about this entity
- Enables traceback to original sources for verification

### How Relationship Merging Works

**Relationships face similar challenges**: The same connection might be described differently across chunks.

**Example**:
- Chunk 1: `("relationship"|"APPLE INC"|"IPHONE"|"manufactures"|7)`
- Chunk 2: `("relationship"|"APPLE INC"|"IPHONE"|"created the iPhone product line"|8)`

**The merging strategy**:
1. **Weight Summation**: Add the weights together (7 + 8 = 15) to indicate stronger evidence
2. **Description Combination**: "manufactures<SEP>created the iPhone product line"
3. **Source Tracking**: Record both chunks as sources for this relationship

### The Graph Storage Strategy

**Why NetworkX**: Nano-graphrag uses NetworkX as its graph storage backend because:
- **Flexibility**: Can store arbitrary metadata on both nodes and edges
- **Rich Algorithms**: Built-in graph analysis algorithms (clustering, centrality, etc.)
- **Python Integration**: Native Python data structures, easy to work with
- **Persistence**: Can serialize to GraphML format for storage

**Graph structure** (from `_storage/networkx_impl.py`):
```python
# Each node stores:
{
    "entity_type": "organization",
    "description": "Technology company<SEP>Makes iPhones<SEP>Founded in 1976", 
    "source_id": "chunk-123<SEP>chunk-456<SEP>chunk-789"
}

# Each edge stores:
{
    "description": "manufactures<SEP>created the iPhone product line",
    "weight": 15.0,
    "source_id": "chunk-123<SEP>chunk-456",
    "order": 1
}
```

### Vector Database Integration

**The dual representation approach**: Nano-graphrag maintains both a symbolic graph AND a vector representation of entities.

**Why both representations?**
- **Symbolic Graph**: Perfect for explicit relationships and logical queries
- **Vector Database**: Essential for similarity-based queries and semantic search

**How entity embeddings are created**:
1. **Text Construction**: For each entity, create descriptive text: `"APPLE INC: Technology company<SEP>Makes iPhones<SEP>Founded in 1976"`
2. **Embedding Generation**: Use the embedding function (default: OpenAI embeddings) to create vector representations
3. **Storage**: Store in vector database with entity name as metadata for retrieval

### The Incremental Update Strategy

**Key insight**: The graph doesn't need to be rebuilt from scratch when new documents are added. Nano-graphrag supports incremental updates:

1. **New entities**: Simply add them to the graph
2. **Existing entities**: Merge new information using the reconciliation process
3. **New relationships**: Add or strengthen existing connections
4. **Community updates**: Recompute communities (currently rebuilds all, but could be optimized)

### Advanced Graph Construction Algorithms

#### The Entity Similarity Calculation

**Beyond simple name matching**, nano-graphrag uses sophisticated similarity metrics to determine if entities should be merged:

**1. String Similarity Metrics**:
```python
def calculate_entity_similarity(entity1, entity2):
    name_sim = levenshtein_similarity(entity1['name'], entity2['name'])
    desc_sim = cosine_similarity(entity1['description'], entity2['description'])
    type_match = 1.0 if entity1['type'] == entity2['type'] else 0.5
    
    # Weighted combination
    total_sim = (name_sim * 0.6) + (desc_sim * 0.3) + (type_match * 0.1)
    return total_sim

def should_merge_entities(entity1, entity2, threshold=0.85):
    return calculate_entity_similarity(entity1, entity2) > threshold
```

**2. Context-Aware Matching**:
```python
def context_aware_entity_match(entity1, entity2):
    # Check if entities appear in similar contexts
    shared_chunks = set(entity1['source_chunks']) & set(entity2['source_chunks'])
    if len(shared_chunks) > 0:
        return True  # Entities from same chunks are likely the same
    
    # Check if entities have relationships to same other entities
    entity1_neighbors = get_entity_neighbors(entity1['name'])
    entity2_neighbors = get_entity_neighbors(entity2['name'])
    
    neighbor_overlap = len(entity1_neighbors & entity2_neighbors)
    if neighbor_overlap > 2:  # Share multiple neighbors
        return True
    
    return False
```

#### The Graph Merging State Machine

**Nano-graphrag maintains graph consistency** through a sophisticated state machine that manages entity and relationship merging:

**State 1: Collection Phase**
```python
class GraphMerger:
    def __init__(self):
        self.pending_entities = {}  # Entities waiting to be merged
        self.pending_relationships = {}  # Relationships waiting to be processed
        self.conflict_queue = []  # Entities with merge conflicts
        
    def collect_extractions(self, chunk_extractions):
        for chunk_id, (entities, relationships) in chunk_extractions.items():
            for entity in entities:
                entity_key = self.normalize_entity_name(entity['name'])
                
                if entity_key not in self.pending_entities:
                    self.pending_entities[entity_key] = []
                self.pending_entities[entity_key].append(entity)
            
            for relationship in relationships:
                rel_key = (relationship['source'], relationship['target'])
                if rel_key not in self.pending_relationships:
                    self.pending_relationships[rel_key] = []
                self.pending_relationships[rel_key].append(relationship)
```

**State 2: Conflict Resolution Phase**
```python
    def resolve_entity_conflicts(self):
        for entity_key, entity_versions in self.pending_entities.items():
            if len(entity_versions) == 1:
                continue  # No conflict
            
            # Multiple versions of the same entity - need to merge
            merged_entity = self.merge_entity_versions(entity_versions)
            
            if merged_entity is None:
                # Merge failed - add to conflict queue for manual resolution
                self.conflict_queue.append((entity_key, entity_versions))
            else:
                self.pending_entities[entity_key] = [merged_entity]
    
    def merge_entity_versions(self, entity_versions):
        # Sophisticated merging algorithm
        try:
            # Step 1: Choose the most common type
            type_counts = Counter([e['type'] for e in entity_versions])
            merged_type = type_counts.most_common(1)[0][0]
            
            # Step 2: Merge descriptions intelligently
            descriptions = [e['description'] for e in entity_versions]
            merged_description = self.smart_description_merge(descriptions)
            
            # Step 3: Aggregate source information
            sources = []
            for entity in entity_versions:
                sources.extend(entity.get('source_chunks', []))
            merged_sources = list(set(sources))
            
            return {
                'name': entity_versions[0]['name'],  # Use first canonical name
                'type': merged_type,
                'description': merged_description,
                'source_chunks': merged_sources,
                'confidence': self.calculate_merge_confidence(entity_versions)
            }
        except Exception as e:
            logger.error(f"Entity merge failed: {e}")
            return None
```

#### Intelligent Description Merging

**The description merging algorithm** goes beyond simple concatenation:

```python
def smart_description_merge(self, descriptions):
    # Step 1: Remove duplicates and near-duplicates
    unique_descriptions = self.deduplicate_descriptions(descriptions)
    
    # Step 2: Rank descriptions by information content
    ranked_descriptions = self.rank_descriptions_by_content(unique_descriptions)
    
    # Step 3: Merge complementary information
    merged = self.merge_complementary_info(ranked_descriptions)
    
    # Step 4: Ensure the result isn't too long
    if len(merged) > 500:  # Token limit for descriptions
        merged = self.summarize_description(merged)
    
    return merged

def deduplicate_descriptions(self, descriptions):
    unique = []
    for desc in descriptions:
        # Check if this description is substantially different from existing ones
        is_unique = True
        for existing in unique:
            if self.description_similarity(desc, existing) > 0.8:
                is_unique = False
                break
        
        if is_unique:
            unique.append(desc)
    
    return unique

def merge_complementary_info(self, descriptions):
    # Look for complementary rather than overlapping information
    aspects = {
        'definition': [],
        'history': [],
        'products': [],
        'location': [],
        'people': [],
        'other': []
    }
    
    # Classify each description by aspect
    for desc in descriptions:
        aspect = self.classify_description_aspect(desc)
        aspects[aspect].append(desc)
    
    # Merge within each aspect, then combine
    merged_aspects = []
    for aspect, desc_list in aspects.items():
        if desc_list:
            merged_aspect = self.merge_within_aspect(desc_list)
            merged_aspects.append(merged_aspect)
    
    return "<SEP>".join(merged_aspects)
```

#### Graph Consistency Validation

**After merging**, nano-graphrag validates graph consistency:

```python
class GraphConsistencyValidator:
    def validate_graph_state(self, graph):
        issues = []
        
        # Check 1: Orphaned relationships
        for edge in graph.edges():
            if not graph.has_node(edge[0]) or not graph.has_node(edge[1]):
                issues.append(f"Orphaned relationship: {edge[0]} -> {edge[1]}")
        
        # Check 2: Suspicious entity types
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if self.is_suspicious_entity(node_data):
                issues.append(f"Suspicious entity: {node}")
        
        # Check 3: Relationship weight distribution
        weights = [graph.edges[edge]['weight'] for edge in graph.edges()]
        if self.is_suspicious_weight_distribution(weights):
            issues.append("Suspicious relationship weight distribution")
        
        # Check 4: Graph connectivity
        if not self.is_reasonably_connected(graph):
            issues.append("Graph appears to be overly fragmented")
        
        return issues
    
    def is_suspicious_entity(self, entity_data):
        # Entities that might be extraction errors
        name = entity_data['entity_name']
        description = entity_data.get('description', '')
        
        # Too generic
        if name.lower() in ['item', 'thing', 'object', 'entity']:
            return True
        
        # Too long (probably malformed)
        if len(name) > 100:
            return True
        
        # Inconsistent with description
        if 'organization' in entity_data.get('entity_type', '').lower():
            if 'person' in description.lower() and 'company' not in description.lower():
                return True
        
        return False
```

#### Memory-Efficient Graph Operations

**For large graphs**, nano-graphrag uses memory-efficient operations:

```python
class MemoryEfficientGraphOperations:
    def __init__(self, graph):
        self.graph = graph
        self.node_cache = {}  # LRU cache for frequently accessed nodes
        
    def batch_node_updates(self, updates, batch_size=1000):
        """Update many nodes efficiently"""
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i+batch_size]
            
            # Prepare batch update
            with self.graph.batch_update():
                for node_id, node_data in batch:
                    self.graph.add_node(node_id, **node_data)
    
    def streaming_edge_addition(self, edge_stream):
        """Add edges from a stream without loading all into memory"""
        edge_buffer = []
        buffer_size = 5000
        
        for edge_data in edge_stream:
            edge_buffer.append(edge_data)
            
            if len(edge_buffer) >= buffer_size:
                self.flush_edge_buffer(edge_buffer)
                edge_buffer = []
        
        # Flush remaining edges
        if edge_buffer:
            self.flush_edge_buffer(edge_buffer)
    
    def flush_edge_buffer(self, edges):
        """Efficiently add a batch of edges"""
        # Sort edges to improve cache locality
        edges.sort(key=lambda x: (x['source'], x['target']))
        
        for edge in edges:
            self.graph.add_edge(
                edge['source'], 
                edge['target'],
                **edge['attributes']
            )
```

#### Advanced Relationship Processing

**Relationship processing** goes beyond simple addition:

```python
class RelationshipProcessor:
    def __init__(self):
        self.relationship_patterns = self.load_relationship_patterns()
    
    def process_relationship_batch(self, relationships):
        processed = []
        
        for rel in relationships:
            # Step 1: Normalize relationship description
            normalized_desc = self.normalize_relationship_description(rel['description'])
            
            # Step 2: Infer relationship type
            rel_type = self.infer_relationship_type(normalized_desc)
            
            # Step 3: Validate relationship makes sense
            if self.validate_relationship_logic(rel, rel_type):
                processed.append({
                    **rel,
                    'normalized_description': normalized_desc,
                    'inferred_type': rel_type,
                    'confidence': self.calculate_relationship_confidence(rel)
                })
        
        return processed
    
    def normalize_relationship_description(self, description):
        # Convert various expressions to canonical forms
        normalizations = {
            r'(?:is|was|are|were)\s+(?:the\s+)?(?:ceo|chief executive officer)': 'IS_CEO_OF',
            r'(?:founded|established|created|started)': 'FOUNDED',
            r'(?:manufactures|produces|makes)': 'MANUFACTURES',
            r'(?:located|situated|based)\s+(?:in|at)': 'LOCATED_IN',
            r'(?:owns|controls|possesses)': 'OWNS',
        }
        
        normalized = description.lower()
        for pattern, replacement in normalizations.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def infer_relationship_type(self, description):
        # Use pattern matching to infer relationship types
        type_patterns = {
            'LEADERSHIP': ['IS_CEO_OF', 'IS_PRESIDENT_OF', 'LEADS', 'MANAGES'],
            'OWNERSHIP': ['OWNS', 'CONTROLS', 'POSSESSES'],
            'CREATION': ['FOUNDED', 'ESTABLISHED', 'CREATED'],
            'PRODUCTION': ['MANUFACTURES', 'PRODUCES', 'MAKES'],
            'LOCATION': ['LOCATED_IN', 'BASED_IN', 'SITUATED_IN'],
            'ASSOCIATION': ['WORKS_FOR', 'EMPLOYED_BY', 'MEMBER_OF']
        }
        
        for rel_type, patterns in type_patterns.items():
            if any(pattern in description.upper() for pattern in patterns):
                return rel_type
        
        return 'GENERIC_RELATION'
```

### Why This Graph Construction Works

**Traditional knowledge graph problems**:
- **Brittleness**: Hard-coded schemas that don't adapt to new domains
- **Entity Resolution**: Difficult to match entities across different sources
- **Scalability**: Expensive to update when new information arrives
- **Context Loss**: Entities lose their original context during extraction

**Nano-graphrag's advantages**:
- **Schema Flexibility**: Entities can have any type, descriptions can be arbitrarily rich
- **Robust Merging**: Intelligent reconciliation preserves information rather than discarding it
- **Source Preservation**: Always traceable back to original text
- **Incremental Growth**: Can efficiently add new information without rebuilding
- **Dual Representation**: Combines symbolic reasoning with semantic similarity
- **Quality Control**: Multiple validation layers ensure graph coherence
- **Memory Efficiency**: Optimized for processing large document collections
- **Error Recovery**: Graceful handling of extraction and merging failures

---

## How Community Detection Creates Hierarchies

**The clustering challenge**: Once you have a knowledge graph with hundreds or thousands of entities, how do you organize them into meaningful groups for analysis and querying?

### The Community Detection Philosophy

**Think of communities as topics or themes** that naturally emerge from the data. Instead of predefining categories, let the graph structure reveal which entities are most closely connected.

**Why communities matter for RAG**:
- **Scalability**: Instead of analyzing thousands of individual entities, work with dozens of communities
- **Thematic Organization**: Communities often correspond to real-world topics or domains
- **Hierarchical Structure**: Different levels of communities provide different levels of detail
- **Query Efficiency**: Global queries can focus on relevant communities rather than the entire graph

### How the Leiden Algorithm Works (Conceptually)

**Nano-graphrag uses the Leiden algorithm** (implemented through NetworkX) for community detection. Here's how it works conceptually:

**Step 1 - Local Optimization**:
- Start with each entity in its own community
- Iteratively try moving entities between communities to improve "modularity" (a measure of community quality)
- Keep doing this until no more improvements are possible

**Step 2 - Community Aggregation**:
- Treat each community as a single "super node"
- Create a new graph where communities are nodes and edges represent connections between communities
- Apply the same local optimization process to this new graph

**Step 3 - Hierarchical Structure**:
- Repeat the aggregation process multiple times to create different levels of granularity
- Level 0: Most granular communities (smallest groups)
- Level 1: Medium-sized communities (groups of level 0 communities)
- Level 2: Large communities (groups of level 1 communities)

### What Makes a Good Community

**The modularity principle**: A good community has many internal connections (entities within the community are highly related) and few external connections (weak relationships to entities in other communities).

**Example**: In a technology document corpus:
- **Good community**: "Apple", "iPhone", "iOS", "Tim Cook", "Cupertino" (all Apple-related entities)
- **Bad community**: "Apple", "Microsoft", "Google", "Climate Change", "Paris" (unrelated entities)

**How relationship weights influence communities**:
- Strong relationships (high weights) pull entities into the same community
- Weak relationships allow entities to be separated into different communities
- The algorithm automatically finds the optimal balance

### The Hierarchical Community Structure

**Level 0 Communities** (most granular):
- Typically 5-15 entities each
- Very specific topics: "Apple's iPhone Product Line", "Machine Learning Research at Stanford"
- High internal connectivity, very focused themes

**Level 1 Communities** (medium granularity):
- Groups of related Level 0 communities
- Broader topics: "Consumer Technology Companies", "Academic AI Research"
- Still coherent themes but with more diversity

**Level 2 Communities** (coarse granularity):
- Very broad themes: "Technology Industry", "Academic Research"
- Useful for high-level overviews and global analysis

### How Community Reports Are Generated

**The report generation process** (implemented in `_op.py:695-775`):

**Step 1 - Community Description Assembly**:
For each community, gather:
- All entities in the community with their descriptions
- All relationships between entities in the community
- All relationships connecting to other communities (for context)
- Any sub-communities (for hierarchical reports)

**Step 2 - LLM Analysis**:
Use a specialized prompt (from `prompt.py:64-192`) that asks the LLM to:
- Analyze the community's structure and relationships
- Identify key themes and patterns
- Assess the community's importance and impact
- Generate insights about the community's role in the larger knowledge graph

**Step 3 - Structured Output**:
The LLM generates a JSON report with:
```json
{
  "title": "Apple Technology Ecosystem",
  "summary": "This community centers around Apple Inc and its product ecosystem...",
  "rating": 8.5,
  "rating_explanation": "High impact due to Apple's influence on consumer technology",
  "findings": [
    {
      "summary": "Dominant market position in smartphones",
      "explanation": "Apple's iPhone maintains significant market share..."
    }
  ]
}
```

### The Strategic Value of Community Reports

**Why generate reports instead of just using raw communities?**

1. **Human Readability**: Reports provide context and interpretation that raw graph data lacks
2. **Query Efficiency**: Pre-computed summaries are faster to search than analyzing raw graph structure
3. **Thematic Understanding**: Reports identify themes and patterns that might not be obvious from individual entities
4. **Impact Assessment**: Ratings help prioritize which communities are most important for different queries
5. **Narrative Structure**: Reports provide coherent explanations rather than disconnected facts

### How Communities Enable Global Queries

**The global query strategy** relies heavily on communities:

1. **Community Selection**: Find communities relevant to the query (based on community report content)
2. **Parallel Analysis**: Analyze each relevant community independently 
3. **Cross-Community Synthesis**: Combine insights from multiple communities into a comprehensive answer

**Example query**: "What are the major trends in artificial intelligence?"

- **Step 1**: Find communities related to AI (e.g., "Machine Learning Research", "AI Companies", "AI Ethics")
- **Step 2**: Analyze each community's report for AI trends
- **Step 3**: Synthesize findings across communities to identify overarching trends

### Advanced Community Detection Algorithms

#### The Mathematics Behind Modularity

**Modularity** is the key metric that determines community quality. Here's how it works mathematically:

**The modularity formula**:
```
Q = (1/2m) * Σ[Aij - (ki*kj)/(2m)] * δ(ci, cj)
```

Where:
- `m` = total number of edges in the graph
- `Aij` = adjacency matrix element (1 if edge exists, 0 otherwise)  
- `ki`, `kj` = degrees of nodes i and j
- `ci`, `cj` = community assignments of nodes i and j
- `δ(ci, cj)` = 1 if nodes are in same community, 0 otherwise

**What this means conceptually**:
- `Aij` counts actual edges within communities
- `(ki*kj)/(2m)` is the expected number of edges in a random graph
- We want actual edges within communities to exceed random expectations

**Implementation in nano-graphrag**:
```python
class ModularityCalculator:
    def __init__(self, graph):
        self.graph = graph
        self.total_edges = graph.number_of_edges()
        self.node_degrees = dict(graph.degree())
    
    def calculate_modularity(self, communities):
        """Calculate modularity Q for a given community assignment"""
        modularity = 0.0
        
        for community in communities:
            community_nodes = list(community)
            
            # Calculate internal edges
            internal_edges = 0
            for i, node_i in enumerate(community_nodes):
                for j, node_j in enumerate(community_nodes):
                    if i < j and self.graph.has_edge(node_i, node_j):
                        internal_edges += 1
            
            # Calculate expected internal edges
            community_degree_sum = sum(self.node_degrees[node] for node in community_nodes)
            expected_internal = (community_degree_sum ** 2) / (4 * self.total_edges)
            
            # Add to modularity
            modularity += (internal_edges - expected_internal) / self.total_edges
        
        return modularity
    
    def delta_modularity(self, node, from_community, to_community):
        """Calculate change in modularity if node moves between communities"""
        # This is used during optimization to decide whether to move nodes
        current_connections_from = self.count_connections(node, from_community)
        current_connections_to = self.count_connections(node, to_community)
        
        degree_node = self.node_degrees[node]
        sum_from = sum(self.node_degrees[n] for n in from_community if n != node)
        sum_to = sum(self.node_degrees[n] for n in to_community)
        
        delta_q = (current_connections_to - current_connections_from) / self.total_edges
        delta_q += (degree_node * (sum_from - sum_to)) / (2 * self.total_edges ** 2)
        
        return delta_q
```

#### The Leiden Algorithm Implementation

**Nano-graphrag uses the Leiden algorithm**, which improves upon the Louvain algorithm:

```python
class LeidenCommunityDetector:
    def __init__(self, graph, resolution=1.0, random_seed=42):
        self.graph = graph
        self.resolution = resolution
        self.random = random.Random(random_seed)
        
    def detect_communities(self):
        """Main Leiden algorithm implementation"""
        communities = self.initialize_communities()
        
        while True:
            # Phase 1: Local moving
            communities = self.local_moving_phase(communities)
            
            # Phase 2: Refinement
            communities = self.refinement_phase(communities)
            
            # Phase 3: Aggregation
            new_graph, community_mapping = self.aggregate_graph(communities)
            
            # Check convergence
            if self.has_converged(communities, new_graph):
                break
            
            # Update for next iteration
            self.graph = new_graph
            communities = self.map_communities(communities, community_mapping)
        
        return self.finalize_communities(communities)
    
    def local_moving_phase(self, communities):
        """Move nodes to communities that increase modularity most"""
        improved = True
        iteration = 0
        
        while improved and iteration < 100:  # Prevent infinite loops
            improved = False
            iteration += 1
            
            # Process nodes in random order
            nodes = list(self.graph.nodes())
            self.random.shuffle(nodes)
            
            for node in nodes:
                current_community = self.find_node_community(node, communities)
                best_community = self.find_best_community(node, communities)
                
                if best_community != current_community:
                    # Move node to better community
                    communities = self.move_node(node, current_community, best_community, communities)
                    improved = True
        
        return communities
    
    def find_best_community(self, node, communities):
        """Find the community that maximizes modularity gain for this node"""
        current_community = self.find_node_community(node, communities)
        best_community = current_community
        best_gain = 0.0
        
        # Consider neighboring communities
        neighbor_communities = set()
        for neighbor in self.graph.neighbors(node):
            neighbor_communities.add(self.find_node_community(neighbor, communities))
        
        # Also consider staying in current community
        neighbor_communities.add(current_community)
        
        for candidate_community in neighbor_communities:
            if candidate_community == current_community:
                continue
            
            gain = self.calculate_modularity_gain(node, current_community, candidate_community)
            if gain > best_gain:
                best_gain = gain
                best_community = candidate_community
        
        return best_community
    
    def refinement_phase(self, communities):
        """Refinement phase to improve community quality"""
        refined_communities = []
        
        for community in communities:
            if len(community) < 3:  # Small communities don't need refinement
                refined_communities.append(community)
                continue
            
            # Create subgraph for this community
            subgraph = self.graph.subgraph(community)
            
            # Find well-connected subsets within the community
            subcommunities = self.find_well_connected_subsets(subgraph)
            
            # Only split if it improves modularity
            if self.splitting_improves_modularity(community, subcommunities):
                refined_communities.extend(subcommunities)
            else:
                refined_communities.append(community)
        
        return refined_communities
    
    def find_well_connected_subsets(self, subgraph):
        """Find subsets that are more connected internally than externally"""
        if subgraph.number_of_nodes() < 3:
            return [list(subgraph.nodes())]
        
        # Use spectral clustering on the subgraph
        adjacency = nx.adjacency_matrix(subgraph)
        
        try:
            # Compute Laplacian eigenvectors
            laplacian = nx.laplacian_matrix(subgraph, normalized=True)
            eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(laplacian, k=2, which='SM')
            
            # Use second eigenvector (Fiedler vector) for bipartition
            fiedler_vector = eigenvecs[:, 1]
            
            # Split based on sign of Fiedler vector
            nodes = list(subgraph.nodes())
            subset1 = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] >= 0]
            subset2 = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] < 0]
            
            return [subset1, subset2] if len(subset1) > 0 and len(subset2) > 0 else [nodes]
            
        except:
            # Fallback: return original community
            return [list(subgraph.nodes())]
```

#### Hierarchical Community Structure Generation

**Creating the hierarchy** involves multiple rounds of community detection:

```python
class HierarchicalCommunityBuilder:
    def __init__(self, graph, max_levels=5):
        self.original_graph = graph.copy()
        self.max_levels = max_levels
        
    def build_hierarchy(self):
        """Build complete hierarchical community structure"""
        hierarchy = {}
        current_graph = self.original_graph.copy()
        level = 0
        
        while level < self.max_levels:
            # Detect communities at current level
            detector = LeidenCommunityDetector(current_graph)
            communities = detector.detect_communities()
            
            # Store communities for this level
            hierarchy[level] = self.process_communities(communities, level)
            
            # Check if we should continue (enough communities to merge)
            if len(communities) < 3:
                break
            
            # Create aggregated graph for next level
            current_graph = self.create_aggregated_graph(current_graph, communities)
            level += 1
        
        return self.finalize_hierarchy(hierarchy)
    
    def process_communities(self, communities, level):
        """Process raw communities into structured format"""
        processed = {}
        
        for i, community in enumerate(communities):
            community_id = f"community-{level}-{i}"
            
            processed[community_id] = {
                'id': community_id,
                'level': level,
                'nodes': list(community),
                'size': len(community),
                'internal_edges': self.count_internal_edges(community),
                'external_edges': self.count_external_edges(community),
                'modularity_contribution': self.calculate_community_modularity(community),
                'density': self.calculate_community_density(community)
            }
        
        return processed
    
    def create_aggregated_graph(self, graph, communities):
        """Create super-graph where communities become nodes"""
        aggregated = nx.Graph()
        
        # Create mapping from nodes to communities
        node_to_community = {}
        for i, community in enumerate(communities):
            community_id = f"super_node_{i}"
            aggregated.add_node(community_id, 
                               size=len(community),
                               internal_weight=self.calculate_internal_weight(community))
            
            for node in community:
                node_to_community[node] = community_id
        
        # Add edges between communities
        community_connections = defaultdict(int)
        
        for edge in graph.edges():
            source_community = node_to_community[edge[0]]
            target_community = node_to_community[edge[1]]
            
            if source_community != target_community:
                # Edge between different communities
                edge_key = tuple(sorted([source_community, target_community]))
                edge_weight = graph.edges[edge].get('weight', 1)
                community_connections[edge_key] += edge_weight
        
        # Add aggregated edges
        for (comm1, comm2), weight in community_connections.items():
            aggregated.add_edge(comm1, comm2, weight=weight)
        
        return aggregated
    
    def calculate_community_quality_metrics(self, community):
        """Calculate various quality metrics for a community"""
        subgraph = self.original_graph.subgraph(community)
        
        metrics = {
            'conductance': self.calculate_conductance(community),
            'modularity': self.calculate_community_modularity(community),
            'clustering_coefficient': nx.average_clustering(subgraph),
            'density': nx.density(subgraph),
            'diameter': self.safe_diameter(subgraph),
            'cohesion': self.calculate_cohesion(community)
        }
        
        return metrics
    
    def calculate_conductance(self, community):
        """Calculate conductance (cut ratio) for community"""
        internal_edges = 0
        external_edges = 0
        
        for node in community:
            for neighbor in self.original_graph.neighbors(node):
                if neighbor in community:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        internal_edges //= 2  # Each internal edge counted twice
        total_degree = internal_edges * 2 + external_edges
        
        if total_degree == 0:
            return 0.0
        
        return external_edges / total_degree
    
    def calculate_cohesion(self, community):
        """Calculate how tightly connected the community is"""
        if len(community) < 2:
            return 1.0
        
        subgraph = self.original_graph.subgraph(community)
        actual_edges = subgraph.number_of_edges()
        possible_edges = len(community) * (len(community) - 1) / 2
        
        return actual_edges / possible_edges if possible_edges > 0 else 0.0
```

#### Advanced Community Report Generation

**The community report generation** is more sophisticated than simple LLM calls:

```python
class AdvancedCommunityReportGenerator:
    def __init__(self, graph, llm_function):
        self.graph = graph
        self.llm_function = llm_function
        self.report_templates = self.load_report_templates()
    
    async def generate_detailed_report(self, community, context_communities=None):
        """Generate comprehensive community report with multiple analysis angles"""
        
        # Step 1: Gather comprehensive community data
        community_data = await self.gather_community_data(community)
        
        # Step 2: Perform structural analysis
        structural_analysis = self.analyze_community_structure(community)
        
        # Step 3: Content analysis
        content_analysis = await self.analyze_community_content(community)
        
        # Step 4: Relationship analysis
        relationship_analysis = self.analyze_community_relationships(community, context_communities)
        
        # Step 5: Generate multi-perspective report
        report = await self.synthesize_community_report(
            community_data, structural_analysis, content_analysis, relationship_analysis
        )
        
        return report
    
    async def gather_community_data(self, community):
        """Gather all available data about community members"""
        community_data = {
            'entities': [],
            'relationships': [],
            'entity_types': defaultdict(int),
            'relationship_types': defaultdict(int),
            'source_chunks': set(),
            'time_references': [],
            'geographical_references': []
        }
        
        # Collect entity data
        for entity_id in community:
            entity_data = self.graph.nodes[entity_id]
            community_data['entities'].append({
                'id': entity_id,
                'type': entity_data.get('entity_type', 'unknown'),
                'description': entity_data.get('description', ''),
                'sources': entity_data.get('source_id', '').split('<SEP>')
            })
            
            community_data['entity_types'][entity_data.get('entity_type', 'unknown')] += 1
            community_data['source_chunks'].update(entity_data.get('source_id', '').split('<SEP>'))
        
        # Collect relationship data
        for source in community:
            for target in community:
                if self.graph.has_edge(source, target):
                    edge_data = self.graph.edges[source, target]
                    relationship = {
                        'source': source,
                        'target': target,
                        'description': edge_data.get('description', ''),
                        'weight': edge_data.get('weight', 1),
                        'type': self.infer_relationship_type(edge_data.get('description', ''))
                    }
                    community_data['relationships'].append(relationship)
                    community_data['relationship_types'][relationship['type']] += 1
        
        return community_data
    
    def analyze_community_structure(self, community):
        """Analyze the structural properties of the community"""
        subgraph = self.graph.subgraph(community)
        
        analysis = {
            'size': len(community),
            'edge_count': subgraph.number_of_edges(),
            'density': nx.density(subgraph),
            'average_degree': sum(dict(subgraph.degree()).values()) / len(community) if len(community) > 0 else 0,
            'clustering_coefficient': nx.average_clustering(subgraph),
            'connectivity': nx.is_connected(subgraph),
            'diameter': self.safe_diameter(subgraph),
            'centrality_analysis': self.analyze_centrality(subgraph),
            'structural_roles': self.identify_structural_roles(subgraph)
        }
        
        return analysis
    
    def analyze_centrality(self, subgraph):
        """Analyze different centrality measures"""
        if len(subgraph.nodes()) == 0:
            return {}
        
        centrality_measures = {}
        
        try:
            # Degree centrality
            degree_cent = nx.degree_centrality(subgraph)
            centrality_measures['degree'] = {
                'values': degree_cent,
                'top_nodes': sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:3]
            }
            
            # Betweenness centrality
            if subgraph.number_of_edges() > 0:
                between_cent = nx.betweenness_centrality(subgraph)
                centrality_measures['betweenness'] = {
                    'values': between_cent,
                    'top_nodes': sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:3]
                }
            
            # PageRank
            pagerank = nx.pagerank(subgraph)
            centrality_measures['pagerank'] = {
                'values': pagerank,
                'top_nodes': sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
            }
            
        except Exception as e:
            logger.warning(f"Centrality analysis failed: {e}")
        
        return centrality_measures
    
    def identify_structural_roles(self, subgraph):
        """Identify structural roles of nodes in the community"""
        roles = {
            'hubs': [],        # Nodes with high degree
            'bridges': [],     # Nodes with high betweenness
            'authorities': [], # Nodes with high PageRank
            'periphery': []    # Nodes with low connectivity
        }
        
        if len(subgraph.nodes()) < 2:
            return roles
        
        # Calculate metrics
        degree_cent = nx.degree_centrality(subgraph)
        
        # Identify roles based on thresholds
        degree_threshold = np.percentile(list(degree_cent.values()), 75)
        
        for node, degree in degree_cent.items():
            if degree >= degree_threshold:
                roles['hubs'].append(node)
            elif degree <= np.percentile(list(degree_cent.values()), 25):
                roles['periphery'].append(node)
        
        return roles
    
    async def synthesize_community_report(self, community_data, structural_analysis, 
                                         content_analysis, relationship_analysis):
        """Synthesize all analyses into a comprehensive report"""
        
        # Build context for LLM
        context = self.build_report_context(
            community_data, structural_analysis, content_analysis, relationship_analysis
        )
        
        # Generate report with specialized prompt
        report_prompt = self.create_community_analysis_prompt(context)
        
        try:
            raw_report = await self.llm_function(
                report_prompt,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse and validate report
            report = self.parse_and_validate_report(raw_report)
            
            # Add metadata
            report['metadata'] = {
                'generation_time': datetime.now().isoformat(),
                'community_size': len(community_data['entities']),
                'analysis_confidence': self.calculate_analysis_confidence(context),
                'structural_metrics': structural_analysis
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Community report generation failed: {e}")
            return self.generate_fallback_report(community_data)
```

### Why This Community Approach Works

**Traditional topic modeling problems**:
- **Fixed Number of Topics**: Have to specify how many topics you want in advance
- **Static Structure**: Topics don't adapt as new documents are added
- **No Hierarchical Understanding**: Can't zoom in/out on topic granularity
- **Weak Entity Relationships**: Don't capture how specific entities relate within topics
- **No Quality Metrics**: Difficult to assess if topics are meaningful
- **Limited Interpretability**: Topics are just word distributions

**Nano-graphrag's community advantages**:
- **Dynamic Topic Discovery**: Communities emerge naturally from the data structure
- **Hierarchical Flexibility**: Can query at different levels of granularity (0, 1, 2)
- **Rich Relationship Modeling**: Understands how entities connect within communities
- **Incremental Updates**: Can evolve communities as new information is added
- **AI-Enhanced Interpretation**: LLM-generated reports provide human-understandable context
- **Quality Assurance**: Mathematical modularity ensures communities are meaningful
- **Structural Analysis**: Graph metrics reveal community roles and importance
- **Multi-Perspective Reports**: Communities analyzed from structural, content, and relational angles
- **Validation Mechanisms**: Multiple checks ensure community coherence and usefulness

---

## How the Three Query Methods Work

**The query challenge**: Given a user question and a knowledge graph with thousands of entities and communities, how do you efficiently find and synthesize the most relevant information?

**Nano-graphrag's insight**: Different types of questions require fundamentally different approaches. Instead of one-size-fits-all, provide three specialized query methods.

### The Query Method Philosophy

**Local Query**: "I need specific facts about particular entities"
- **Best for**: "What products does Apple make?", "Who is the CEO of Microsoft?", "What are the symptoms of diabetes?"
- **Strategy**: Find entities related to the query, then gather all available information about those entities

**Global Query**: "I need thematic analysis across the entire knowledge base"
- **Best for**: "What are the major trends in AI?", "How is climate change affecting different industries?", "What are the key challenges in healthcare?"
- **Strategy**: Identify relevant communities, analyze each community's perspective, then synthesize insights

**Naive Query**: "I need a quick answer from similar text"
- **Best for**: Simple factual questions, when you want traditional RAG behavior
- **Strategy**: Find text chunks similar to the query, return answers based on those chunks

### How Local Query Actually Works

**The local query process** is like having a research assistant who:
1. Identifies the key entities mentioned in your question
2. Finds similar entities in the knowledge base
3. Gathers everything known about those entities
4. Presents the information in a structured way

**Step-by-Step Local Query Process**:

**Step 1 - Entity Discovery** (implemented in `_op.py:860-870`):
- Convert the query into a vector embedding
- Search the entity vector database for similar entities
- Retrieve top-k most similar entities (default: top 20)

**Example**: Query "What products does Apple make?" 
- Finds entities: ["APPLE INC" (similarity: 0.95), "IPHONE" (similarity: 0.87), "IPAD" (similarity: 0.82), ...]

**Step 2 - Entity Detail Gathering** (implemented in `_op.py:871-885`):
- For each found entity, retrieve its full graph data:
  - Entity type and description
  - Source chunks that mentioned this entity
  - Relationships to other entities
- Rank entities by similarity score

**Step 3 - Context Expansion** (implemented in `_op.py:887-950`):
- **Find Related Communities**: Which communities contain these entities?
- **Find Related Text**: Which original text chunks mention these entities?
- **Find Related Relationships**: What other entities are connected to our target entities?

**Step 4 - Structured Context Assembly** (implemented in `_op.py:992-1009`):
Create a comprehensive context in CSV format:
```csv
-----Entities-----
id,entity,type,description,rank
0,APPLE INC,organization,Technology company<SEP>Makes iPhones<SEP>Founded in 1976,950
1,IPHONE,product,Smartphone<SEP>Popular consumer device,870

-----Relationships-----
id,source,target,description,weight,rank
0,APPLE INC,IPHONE,manufactures<SEP>created the iPhone product line,15,900

-----Communities-----
id,content
0,"# Apple Technology Ecosystem\nThis community focuses on Apple Inc and its consumer products..."

-----Sources-----
id,content  
0,"Apple Inc is a technology company that designs and manufactures consumer electronics..."
```

**Step 5 - Answer Generation**:
- Use the local RAG prompt template (from `prompt.py:331-366`) 
- Feed the structured context to the LLM
- Generate a natural language response

**Why local queries work well**:
- **Entity-centric**: Perfect for questions about specific things
- **Comprehensive**: Gathers all available information about target entities
- **Traceable**: Can trace answers back to original sources
- **Relationship-aware**: Understands how entities connect to each other

### How Global Query Actually Works

**The global query process** is like having multiple analysts each analyze different aspects of your question, then synthesizing their reports.

**Step-by-Step Global Query Process**:

**Step 1 - Community Selection** (implemented in `_op.py:1077-1110`):
- Get all communities in the knowledge graph
- Filter by hierarchical level (default: levels 0-2)
- Sort by "occurrence" (how many entities are in each community)
- Select top-N communities (default: top 512 communities)

**Step 2 - Community Filtering** (implemented in `_op.py:1111-1125`):
- Retrieve full community reports
- Filter out communities with low impact ratings (default: rating >= 0.0)
- This focuses on the most significant themes

**Step 3 - Parallel Community Analysis** (implemented in `_op.py:1045-1076`):
For each relevant community:
- Take the community's pre-generated report
- Ask the LLM: "How does this community relate to the user's question?"
- Use a specialized "global analysis" prompt that asks for specific insights and importance scores
- Process all communities in parallel for speed

**Example**: Query "What are the major trends in artificial intelligence?"
- Community 1 (AI Research): "Machine learning techniques are advancing rapidly with transformer architectures..." (Score: 9)
- Community 2 (Tech Companies): "Major tech companies are investing heavily in AI capabilities..." (Score: 8)
- Community 3 (AI Ethics): "Growing concerns about AI bias and safety are driving regulatory discussions..." (Score: 7)

**Step 4 - Insight Ranking and Selection** (implemented in `_op.py:1126-1145`):
- Collect all analysis points from all communities
- Rank by importance scores
- Filter out low-scoring insights (score <= 0)
- Select the most important insights across communities

**Step 5 - Cross-Community Synthesis**:
- Use the global synthesis prompt (from `prompt.py:427-479`)
- Feed all the ranked insights to the LLM
- Ask for a comprehensive answer that synthesizes findings across communities

**Why global queries work well**:
- **Thematic**: Perfect for broad, analytical questions
- **Comprehensive**: Analyzes the entire knowledge base systematically
- **Multi-perspective**: Gets insights from different topical areas
- **Prioritized**: Focuses on the most important insights
- **Synthesized**: Combines findings into coherent narratives

### How Naive Query Works (Traditional RAG)

**The naive query process** is the most straightforward - traditional vector similarity search and response generation.

**Step-by-Step Naive Query Process** (implemented in `_op.py:1178-1210`):

**Step 1 - Chunk Similarity Search**:
- Convert query to vector embedding
- Search chunk vector database for most similar text chunks
- Retrieve top-k chunks (default: top 20)

**Step 2 - Content Retrieval and Truncation**:
- Get full text content for the similar chunks
- Truncate to fit within token limits (default: 8,000 tokens total)
- Preserve the most similar chunks if truncation is needed

**Step 3 - Simple Context Assembly**:
- Concatenate chunk contents with simple separators
- No structured formatting, just raw text

**Step 4 - Answer Generation**:
- Use naive RAG prompt (from `prompt.py:483-493`)
- Feed the raw text context to the LLM
- Generate response

**Why naive query is useful**:
- **Speed**: Fastest query method, no complex processing
- **Simplicity**: Works like traditional RAG systems
- **Fallback**: Good when sophisticated analysis isn't needed
- **Debugging**: Helps validate that basic retrieval works

### When to Use Each Query Method

**Use Local Query when**:
- Question mentions specific entities ("Tell me about Apple")
- You need comprehensive information about particular things
- You want to understand relationships between specific entities
- The question is entity-focused rather than theme-focused

**Use Global Query when**:
- Question asks about broad themes ("What are the trends in...")
- You need analysis across multiple topic areas
- The question requires synthesis of diverse perspectives
- You want comprehensive coverage of a complex topic

**Use Naive Query when**:
- You need a quick, simple answer
- The question is straightforward and factual
- You want traditional RAG behavior
- You're debugging or testing basic functionality

### The Power of Multi-Modal Querying

**The strategic advantage**: Having three different query methods means nano-graphrag can handle a much wider range of questions effectively than systems with only one approach.

**Complex questions can combine approaches**:
- Start with local query to understand specific entities
- Follow up with global query to understand broader themes
- Use naive query for quick fact-checking

**This flexibility makes nano-graphrag more versatile** than traditional RAG systems that only do similarity search or knowledge graphs that only do structured queries.

### Advanced Query Processing Algorithms

#### Vector Search and Similarity Ranking

**The vector search process** is more sophisticated than simple cosine similarity:

```python
class AdvancedVectorSearch:
    def __init__(self, vector_db, embedding_function):
        self.vector_db = vector_db
        self.embedding_function = embedding_function
        self.query_cache = {}  # Cache for repeated queries
        
    async def advanced_similarity_search(self, query, top_k=20, filters=None):
        """Perform sophisticated similarity search with multiple ranking factors"""
        
        # Step 1: Generate query embedding
        query_embedding = await self.get_query_embedding(query)
        
        # Step 2: Initial similarity search
        candidates = await self.vector_db.similarity_search(
            query_embedding, 
            top_k=top_k * 3  # Get more candidates for reranking
        )
        
        # Step 3: Apply filters if provided
        if filters:
            candidates = self.apply_filters(candidates, filters)
        
        # Step 4: Advanced reranking
        reranked_results = await self.rerank_results(query, candidates, top_k)
        
        return reranked_results
    
    async def get_query_embedding(self, query):
        """Get query embedding with caching and preprocessing"""
        
        # Check cache first
        cache_key = hash(query)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Generate embedding
        embedding = await self.embedding_function(processed_query)
        
        # Cache the result
        self.query_cache[cache_key] = embedding
        
        return embedding
    
    def preprocess_query(self, query):
        """Preprocess query for better embeddings"""
        
        # Step 1: Normalize whitespace
        processed = re.sub(r'\s+', ' ', query.strip())
        
        # Step 2: Expand contractions
        processed = self.expand_contractions(processed)
        
        # Step 3: Add context markers for better embeddings
        if '?' in processed:
            processed = f"Question: {processed}"
        elif any(word in processed.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            processed = f"Question: {processed}"
        else:
            processed = f"Query: {processed}"
        
        return processed
    
    async def rerank_results(self, query, candidates, top_k):
        """Sophisticated reranking using multiple signals"""
        
        scored_candidates = []
        
        for candidate in candidates:
            # Base similarity score
            base_score = candidate['similarity']
            
            # Factor 1: Entity type relevance
            type_score = self.calculate_type_relevance(query, candidate)
            
            # Factor 2: Description quality
            quality_score = self.calculate_description_quality(candidate)
            
            # Factor 3: Source diversity
            source_score = self.calculate_source_diversity(candidate)
            
            # Factor 4: Recency (if temporal information available)
            recency_score = self.calculate_recency_score(candidate)
            
            # Combine scores with learned weights
            final_score = (
                base_score * 0.4 +
                type_score * 0.2 +
                quality_score * 0.2 +
                source_score * 0.1 +
                recency_score * 0.1
            )
            
            scored_candidates.append({
                **candidate,
                'rerank_score': final_score,
                'score_components': {
                    'base': base_score,
                    'type': type_score,
                    'quality': quality_score,
                    'source': source_score,
                    'recency': recency_score
                }
            })
        
        # Sort by final score and return top_k
        scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored_candidates[:top_k]
    
    def calculate_type_relevance(self, query, candidate):
        """Calculate how relevant the entity type is to the query"""
        entity_type = candidate.get('metadata', {}).get('entity_type', 'unknown')
        
        # Query type hints
        type_keywords = {
            'organization': ['company', 'corporation', 'firm', 'business', 'organization'],
            'person': ['person', 'people', 'individual', 'ceo', 'founder', 'president'],
            'geo': ['location', 'place', 'city', 'country', 'where', 'geography'],
            'event': ['event', 'happened', 'occurred', 'when', 'meeting', 'conference']
        }
        
        query_lower = query.lower()
        for entity_type_key, keywords in type_keywords.items():
            if entity_type.lower() == entity_type_key:
                if any(keyword in query_lower for keyword in keywords):
                    return 1.0
                else:
                    return 0.5
        
        return 0.5  # Default neutral score
    
    def calculate_description_quality(self, candidate):
        """Score based on description richness and informativeness"""
        description = candidate.get('metadata', {}).get('description', '')
        
        # Factors that indicate quality
        length_score = min(len(description) / 200.0, 1.0)  # Longer descriptions up to a point
        
        # Information density (number of meaningful terms)
        meaningful_terms = len([word for word in description.split() 
                              if len(word) > 3 and word.isalpha()])
        density_score = min(meaningful_terms / 20.0, 1.0)
        
        # Specificity (presence of numbers, dates, proper nouns)
        specificity_score = 0.0
        if re.search(r'\d{4}', description):  # Years
            specificity_score += 0.3
        if re.search(r'\$[\d,]+', description):  # Money amounts
            specificity_score += 0.2
        if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', description):  # Proper nouns
            specificity_score += 0.3
        
        return (length_score * 0.4 + density_score * 0.4 + specificity_score * 0.2)
```

#### Advanced Context Assembly Strategies

**Context assembly** goes beyond simple concatenation:

```python
class ContextAssembler:
    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def assemble_local_context(self, entities, relationships, communities, text_units):
        """Intelligently assemble context for local queries"""
        
        # Step 1: Rank all components by relevance
        ranked_entities = self.rank_entities_by_importance(entities)
        ranked_relationships = self.rank_relationships_by_importance(relationships)
        ranked_communities = self.rank_communities_by_relevance(communities)
        ranked_text_units = self.rank_text_units_by_relevance(text_units)
        
        # Step 2: Build context incrementally with token tracking
        context_builder = IncrementalContextBuilder(self.max_tokens)
        
        # Always include top entities (they drove the search)
        context_builder.add_section("Entities", self.format_entities(ranked_entities[:10]))
        
        # Add relationships that connect included entities
        relevant_relationships = self.filter_relationships_by_entities(
            ranked_relationships, ranked_entities[:10]
        )
        context_builder.add_section("Relationships", self.format_relationships(relevant_relationships))
        
        # Add most relevant community information
        if ranked_communities:
            context_builder.add_section("Communities", 
                                      self.format_communities(ranked_communities[:2]))
        
        # Fill remaining space with text units
        remaining_tokens = context_builder.remaining_tokens()
        selected_text_units = self.select_text_units_within_budget(
            ranked_text_units, remaining_tokens
        )
        context_builder.add_section("Sources", self.format_text_units(selected_text_units))
        
        return context_builder.finalize()
    
    def rank_entities_by_importance(self, entities):
        """Rank entities by multiple importance factors"""
        
        scored_entities = []
        for entity in entities:
            score = 0.0
            
            # Factor 1: Similarity rank (lower rank = higher score)
            similarity_score = 1.0 / (entity.get('rank', 1000) / 100.0)
            score += similarity_score * 0.4
            
            # Factor 2: Description richness
            description = entity.get('description', '')
            richness_score = min(len(description.split()) / 20.0, 1.0)
            score += richness_score * 0.2
            
            # Factor 3: Connectivity (how many relationships this entity has)
            connectivity_score = min(entity.get('relationship_count', 0) / 10.0, 1.0)
            score += connectivity_score * 0.2
            
            # Factor 4: Entity type importance (some types more central)
            type_importance = {
                'organization': 0.9,
                'person': 0.8,
                'event': 0.7,
                'geo': 0.6,
                'product': 0.7
            }
            entity_type = entity.get('entity_type', 'unknown').lower()
            type_score = type_importance.get(entity_type, 0.5)
            score += type_score * 0.2
            
            scored_entities.append({**entity, 'importance_score': score})
        
        return sorted(scored_entities, key=lambda x: x['importance_score'], reverse=True)
    
    def filter_relationships_by_entities(self, relationships, selected_entities):
        """Include only relationships between selected entities"""
        selected_entity_ids = {e['entity_name'] for e in selected_entities}
        
        relevant_relationships = []
        for rel in relationships:
            if (rel.get('src_id') in selected_entity_ids and 
                rel.get('tgt_id') in selected_entity_ids):
                relevant_relationships.append(rel)
        
        return relevant_relationships
    
    def select_text_units_within_budget(self, text_units, token_budget):
        """Select text units that fit within token budget"""
        
        selected = []
        remaining_budget = token_budget
        
        for unit in text_units:
            content = unit.get('content', '')
            unit_tokens = len(self.tokenizer.encode(content))
            
            if unit_tokens <= remaining_budget:
                selected.append(unit)
                remaining_budget -= unit_tokens
            elif remaining_budget > 100:  # If some budget left, try to truncate
                truncated_content = self.truncate_to_budget(content, remaining_budget)
                if len(truncated_content) > 50:  # Only include if meaningful after truncation
                    selected.append({**unit, 'content': truncated_content})
                break
        
        return selected

class IncrementalContextBuilder:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.sections = []
        self.current_tokens = 0
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def add_section(self, section_name, content):
        """Add a section if it fits within token budget"""
        section_text = f"-----{section_name}-----\n{content}\n"
        section_tokens = len(self.tokenizer.encode(section_text))
        
        if self.current_tokens + section_tokens <= self.max_tokens:
            self.sections.append(section_text)
            self.current_tokens += section_tokens
            return True
        else:
            # Try to truncate the content to fit
            available_tokens = self.max_tokens - self.current_tokens - 50  # Reserve some buffer
            if available_tokens > 100:  # Only if meaningful space left
                truncated_content = self.truncate_content_to_tokens(content, available_tokens)
                if len(truncated_content) > 20:
                    truncated_section = f"-----{section_name}-----\n{truncated_content}\n"
                    self.sections.append(truncated_section)
                    self.current_tokens = self.max_tokens  # Mark as full
                    return True
        
        return False
    
    def remaining_tokens(self):
        return self.max_tokens - self.current_tokens
    
    def truncate_content_to_tokens(self, content, max_tokens):
        """Intelligently truncate content to fit token budget"""
        tokens = self.tokenizer.encode(content)
        
        if len(tokens) <= max_tokens:
            return content
        
        # Truncate to max_tokens but try to end at sentence boundary
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        # Try to end at a sentence boundary
        sentences = truncated_text.split('.')
        if len(sentences) > 1:
            # Remove the last incomplete sentence
            return '.'.join(sentences[:-1]) + '.'
        
        return truncated_text + '...'
    
    def finalize(self):
        return ''.join(self.sections)
```

#### Global Query Processing Deep Dive

**Global queries use sophisticated community analysis**:

```python
class GlobalQueryProcessor:
    def __init__(self, community_reports, llm_function):
        self.community_reports = community_reports
        self.llm_function = llm_function
        
    async def process_global_query(self, query, query_params):
        """Process global query with advanced community analysis"""
        
        # Step 1: Community selection with multiple criteria
        relevant_communities = await self.select_communities_multi_criteria(
            query, query_params
        )
        
        # Step 2: Parallel community analysis with different perspectives
        community_analyses = await self.analyze_communities_parallel(
            query, relevant_communities
        )
        
        # Step 3: Cross-community insight synthesis
        synthesized_insights = await self.synthesize_cross_community_insights(
            community_analyses, query
        )
        
        # Step 4: Hierarchical answer construction
        final_answer = await self.construct_hierarchical_answer(
            synthesized_insights, query_params.response_type
        )
        
        return final_answer
    
    async def select_communities_multi_criteria(self, query, query_params):
        """Select communities using multiple selection criteria"""
        
        # Get all communities
        all_communities = await self.community_reports.get_all()
        
        # Filter by level
        level_filtered = [
            c for c in all_communities 
            if c.get('level', 0) <= query_params.level
        ]
        
        # Score communities by relevance to query
        scored_communities = []
        for community in level_filtered:
            relevance_score = await self.calculate_community_relevance(query, community)
            
            if relevance_score > 0.1:  # Minimum relevance threshold
                scored_communities.append({
                    **community,
                    'relevance_score': relevance_score
                })
        
        # Sort by composite score (relevance + importance + size)
        scored_communities.sort(
            key=lambda c: (
                c['relevance_score'] * 0.6 +
                c.get('rating', 5.0) / 10.0 * 0.3 +
                min(c.get('occurrence', 1) / 50.0, 1.0) * 0.1
            ),
            reverse=True
        )
        
        # Select top communities with diversity
        selected = self.diversify_community_selection(
            scored_communities, 
            max_communities=query_params.global_max_consider_community
        )
        
        return selected
    
    async def calculate_community_relevance(self, query, community):
        """Calculate how relevant a community is to the query"""
        
        community_text = community.get('report_string', '')
        
        # Method 1: Keyword overlap
        query_keywords = self.extract_keywords(query)
        community_keywords = self.extract_keywords(community_text)
        keyword_overlap = len(set(query_keywords) & set(community_keywords))
        keyword_score = min(keyword_overlap / max(len(query_keywords), 1), 1.0)
        
        # Method 2: Semantic similarity (if embeddings available)
        semantic_score = 0.0
        if hasattr(self, 'embedding_function'):
            try:
                query_embedding = await self.embedding_function(query)
                community_embedding = await self.embedding_function(community_text[:1000])
                semantic_score = self.cosine_similarity(query_embedding, community_embedding)
            except:
                pass
        
        # Method 3: Entity type alignment
        query_entities = self.extract_entity_hints(query)
        community_entity_types = self.extract_community_entity_types(community)
        type_alignment = self.calculate_type_alignment(query_entities, community_entity_types)
        
        # Combine scores
        relevance = (keyword_score * 0.4 + semantic_score * 0.4 + type_alignment * 0.2)
        
        return relevance
    
    def diversify_community_selection(self, scored_communities, max_communities):
        """Select communities with diversity to avoid redundancy"""
        
        selected = []
        remaining = scored_communities.copy()
        
        while len(selected) < max_communities and remaining:
            if not selected:
                # First selection: highest scoring
                selected.append(remaining.pop(0))
            else:
                # Subsequent selections: balance score with diversity
                best_candidate = None
                best_score = -1
                
                for i, candidate in enumerate(remaining):
                    # Diversity score: how different is this from selected communities
                    diversity_score = self.calculate_community_diversity(candidate, selected)
                    
                    # Combined score
                    combined_score = candidate['relevance_score'] * 0.7 + diversity_score * 0.3
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = i
                
                if best_candidate is not None:
                    selected.append(remaining.pop(best_candidate))
                else:
                    break
        
        return selected
    
    async def analyze_communities_parallel(self, query, communities):
        """Analyze multiple communities in parallel with different perspectives"""
        
        analysis_tasks = []
        
        for community in communities:
            # Create analysis task with specific perspective
            task = self.analyze_single_community_perspective(query, community)
            analysis_tasks.append(task)
        
        # Execute all analyses in parallel
        analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Filter out failed analyses
        successful_analyses = [
            analysis for analysis in analyses 
            if not isinstance(analysis, Exception)
        ]
        
        return successful_analyses
    
    async def analyze_single_community_perspective(self, query, community):
        """Analyze a single community from a specific perspective"""
        
        community_report = community.get('report_string', '')
        
        # Create perspective-specific prompt
        perspective_prompt = f"""
        Analyze the following community report in the context of this query: "{query}"
        
        Community Report:
        {community_report}
        
        Provide analysis in the following JSON format:
        {{
            "key_insights": ["insight1", "insight2", ...],
            "relevance_explanation": "why this community is relevant to the query",
            "evidence": ["supporting evidence from the community"],
            "confidence_score": 0.0-1.0,
            "perspective": "what unique angle this community provides"
        }}
        """
        
        try:
            response = await self.llm_function(
                perspective_prompt,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            parsed_analysis = json.loads(response)
            parsed_analysis['community_id'] = community.get('id', 'unknown')
            
            return parsed_analysis
            
        except Exception as e:
            logger.warning(f"Community analysis failed: {e}")
            return None
    
    async def synthesize_cross_community_insights(self, community_analyses, query):
        """Synthesize insights across multiple communities"""
        
        # Step 1: Extract all insights
        all_insights = []
        for analysis in community_analyses:
            if analysis and 'key_insights' in analysis:
                for insight in analysis['key_insights']:
                    all_insights.append({
                        'insight': insight,
                        'source_community': analysis.get('community_id', 'unknown'),
                        'confidence': analysis.get('confidence_score', 0.5),
                        'evidence': analysis.get('evidence', [])
                    })
        
        # Step 2: Cluster similar insights
        insight_clusters = self.cluster_similar_insights(all_insights)
        
        # Step 3: Rank insight clusters by importance
        ranked_clusters = self.rank_insight_clusters(insight_clusters, query)
        
        # Step 4: Generate synthesis
        synthesis_prompt = self.create_synthesis_prompt(ranked_clusters, query)
        synthesis = await self.llm_function(synthesis_prompt, max_tokens=1500)
        
        return {
            'synthesis': synthesis,
            'insight_clusters': ranked_clusters,
            'source_communities': [a.get('community_id') for a in community_analyses]
        }
```

#### Performance and Scalability Optimizations

**Query processing includes sophisticated optimizations**:

```python
class QueryOptimizer:
    def __init__(self):
        self.query_cache = {}
        self.embedding_cache = {}
        self.performance_metrics = {}
    
    async def optimize_query_execution(self, query_type, query_params, available_resources):
        """Optimize query execution based on available resources and query characteristics"""
        
        optimization_strategy = {
            'parallel_processing': True,
            'batch_size': 8,
            'cache_strategy': 'aggressive',
            'timeout_ms': 30000,
            'fallback_enabled': True
        }
        
        # Adjust based on query complexity
        if query_type == 'global':
            if query_params.global_max_consider_community > 100:
                optimization_strategy['parallel_processing'] = True
                optimization_strategy['batch_size'] = 16
            else:
                optimization_strategy['batch_size'] = 8
        
        # Adjust based on available memory
        if available_resources.get('memory_mb', 1000) < 2000:
            optimization_strategy['batch_size'] = 4
            optimization_strategy['cache_strategy'] = 'conservative'
        
        return optimization_strategy
    
    async def execute_with_fallbacks(self, primary_strategy, fallback_strategies, query_func):
        """Execute query with multiple fallback strategies"""
        
        strategies = [primary_strategy] + fallback_strategies
        
        for i, strategy in enumerate(strategies):
            try:
                start_time = time.time()
                
                result = await asyncio.wait_for(
                    query_func(strategy),
                    timeout=strategy.get('timeout_ms', 30000) / 1000.0
                )
                
                execution_time = time.time() - start_time
                self.record_performance_metric(strategy, execution_time, success=True)
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Query strategy {i} timed out, trying fallback")
                self.record_performance_metric(strategy, strategy['timeout_ms']/1000, success=False)
                continue
            except Exception as e:
                logger.warning(f"Query strategy {i} failed: {e}, trying fallback")
                self.record_performance_metric(strategy, 0, success=False, error=str(e))
                continue
        
        # All strategies failed
        raise Exception("All query strategies failed")
```

**This flexibility makes nano-graphrag more versatile** than traditional RAG systems that only do similarity search or knowledge graphs that only do structured queries.

---

## How Files Are Generated and Stored

**The persistence challenge**: How do you store a complex knowledge graph, vector databases, and community reports in a way that's efficient, readable, and updatable?

**Nano-graphrag's approach**: Use multiple specialized storage formats, each optimized for its specific purpose.

### The Storage Architecture Philosophy

**Instead of one monolithic database**, nano-graphrag uses a distributed storage approach:

1. **JSON Key-Value Storage**: For structured data that needs to be human-readable
2. **Vector Database Storage**: For embeddings and similarity search
3. **Graph Storage**: For the network structure and relationships
4. **LLM Response Caching**: For avoiding expensive re-computations

### How JSON Key-Value Storage Works

**The core storage mechanism** (implemented in `_storage/kv_json.py:10-47`) is elegantly simple:

**Storage Strategy**:
- Each storage "namespace" gets its own JSON file
- The file name pattern: `kv_store_{namespace}.json`
- All data is loaded into memory on startup for fast access
- All data is written to disk when processing is complete

**Why this approach works**:
- **Human Readable**: You can open any file and see exactly what's stored
- **Version Control Friendly**: JSON files can be tracked in git
- **Simple Implementation**: No database dependencies or complex schemas
- **Fast Access**: Everything in memory during processing
- **Atomic Updates**: Files are written completely or not at all

### The Five Core Storage Files

**1. `kv_store_full_docs.json` - Original Documents**
```json
{
  "doc-a1b2c3d4e5": {
    "content": "Apple Inc. is an American multinational technology company..."
  },
  "doc-f6g7h8i9j0": {
    "content": "Machine learning is a method of data analysis that automates..."
  }
}
```

**Purpose**: Preserve original document text exactly as inserted
**Key Strategy**: Hash-based keys prevent duplicate storage
**Access Pattern**: Rarely accessed directly, mainly for debugging and traceability

**2. `kv_store_text_chunks.json` - Processed Text Chunks**
```json
{
  "chunk-x1y2z3a4b5": {
    "content": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California...",
    "tokens": 1847,
    "chunk_order_index": 0,
    "full_doc_id": "doc-a1b2c3d4e5"
  },
  "chunk-c6d7e8f9g0": {
    "content": "The company's hardware products include the iPhone smartphone, the iPad tablet computer...",
    "tokens": 2156,
    "chunk_order_index": 1,
    "full_doc_id": "doc-a1b2c3d4e5"
  }
}
```

**Purpose**: Store processed text chunks with metadata
**Key Information**: 
- `chunk_order_index`: Preserves original document order
- `full_doc_id`: Links back to original document
- `tokens`: Token count for context window management
**Access Pattern**: Frequently accessed during local queries for source information

**3. `kv_store_community_reports.json` - Community Analysis**
```json
{
  "community-0": {
    "report_string": "# Apple Technology Ecosystem\n\n## Summary\nThis community centers around Apple Inc and its comprehensive technology ecosystem...",
    "report_json": {
      "title": "Apple Technology Ecosystem",
      "summary": "This community centers around Apple Inc and its comprehensive technology ecosystem, including hardware products, software platforms, and market positioning.",
      "rating": 8.5,
      "rating_explanation": "High impact due to Apple's significant influence on consumer technology markets and innovation.",
      "findings": [
        {
          "summary": "Dominant position in premium smartphone market",
          "explanation": "Apple's iPhone maintains a strong market position in the premium smartphone segment, with consistent innovation in hardware design and software integration."
        }
      ]
    },
    "level": 0,
    "occurrence": 27,
    "sub_communities": []
  }
}
```

**Purpose**: Store AI-generated community analysis and reports
**Key Structure**:
- `report_string`: Human-readable markdown version
- `report_json`: Structured data for programmatic access
- `level`: Hierarchical level (0 = most granular)
- `occurrence`: How many entities belong to this community
**Access Pattern**: Heavily used in global queries for thematic analysis

**4. `vdb_entities.json` - Entity Vector Database**
```json
{
  "entities_data": [
    {
      "id": "APPLE INC",
      "vector": [0.0123, -0.0456, 0.0789, 0.0234, -0.0567, ...],
      "metadata": {
        "entity_name": "APPLE INC"
      }
    },
    {
      "id": "IPHONE",
      "vector": [0.0345, -0.0678, 0.0123, 0.0456, -0.0789, ...],
      "metadata": {
        "entity_name": "IPHONE"
      }
    }
  ],
  "entities_index": {
    "APPLE INC": 0,
    "IPHONE": 1
  }
}
```

**Purpose**: Enable semantic similarity search over entities
**Key Components**:
- `entities_data`: Vector embeddings with metadata
- `entities_index`: Fast lookup from entity name to array index
**Vector Dimensions**: Typically 1536 dimensions (OpenAI embedding standard)
**Access Pattern**: Critical for local query entity discovery

**5. `graph_chunk_entity_relation.graphml` - Graph Structure**
```xml
<graphml xmlns="http://graphml.graphsource.net/xmlns">
  <graph id="G" edgedefault="directed">
    <node id="APPLE INC">
      <data key="entity_type">organization</data>
      <data key="description">Technology company<SEP>Makes iPhones<SEP>Founded in 1976</data>
      <data key="source_id">chunk-x1y2z3a4b5<SEP>chunk-c6d7e8f9g0</data>
    </node>
    <edge source="APPLE INC" target="IPHONE">
      <data key="weight">15.0</data>
      <data key="description">manufactures<SEP>created the iPhone product line</data>
      <data key="source_id">chunk-x1y2z3a4b5<SEP>chunk-h5i6j7k8l9</data>
    </edge>
  </graph>
</graphml>
```

**Purpose**: Store the complete knowledge graph structure
**Format**: GraphML (XML-based graph format)
**Why GraphML**: Standard format, supports arbitrary node/edge attributes, readable by graph analysis tools
**Access Pattern**: Loaded completely into NetworkX for graph operations

### The Intelligent File Writing Strategy

**Write Coordination** (implemented in `graphrag.py:447-461`):

**The challenge**: Multiple storage systems need to be updated simultaneously without corruption.

**The solution**: Coordinated batch writing:
1. **Accumulate Changes**: All changes are made in memory during processing
2. **Batch Write Signal**: When processing completes, signal all storage systems
3. **Parallel Writes**: All storage systems write their files simultaneously
4. **Atomic Success**: Either all files are written successfully, or none are (fail-fast)

**Implementation**:
```python
async def _insert_done(self):
    tasks = []
    for storage_inst in [
        self.full_docs,              # -> kv_store_full_docs.json
        self.text_chunks,            # -> kv_store_text_chunks.json  
        self.llm_response_cache,     # -> kv_store_llm_response_cache.json
        self.community_reports,      # -> kv_store_community_reports.json
        self.entities_vdb,           # -> vdb_entities.json
        self.chunks_vdb,             # -> vdb_chunks.json (if naive RAG enabled)
        self.chunk_entity_relation_graph,  # -> graph_chunk_entity_relation.graphml
    ]:
        if storage_inst is None:
            continue
        tasks.append(storage_inst.index_done_callback())
    
    await asyncio.gather(*tasks)  # Write all files in parallel
```

### How Incremental Updates Work

**The update challenge**: When new documents are added, how do you update the files efficiently without rebuilding everything?

**Nano-graphrag's incremental strategy**:

1. **Document Deduplication**: Check `kv_store_full_docs.json` to see if document already exists (by content hash)
2. **Chunk Deduplication**: Check `kv_store_text_chunks.json` to avoid re-processing identical chunks
3. **Entity Merging**: Merge new entities with existing entities in the knowledge graph
4. **Community Rebuilding**: Currently rebuilds all communities (opportunity for optimization)
5. **Vector Updates**: Add new entity embeddings to vector database
6. **Batch Writing**: Write all updated files simultaneously

### Why This Storage Architecture Works

**Traditional database problems**:
- **Schema Rigidity**: Hard to change data structures as requirements evolve
- **Dependency Overhead**: Require database setup and maintenance
- **Black Box Storage**: Difficult to inspect or debug stored data
- **Version Control Issues**: Binary formats don't work well with git
- **Backup Complexity**: Need database-specific backup and restore procedures

**Nano-graphrag's storage advantages**:
- **Transparency**: All data is in readable formats
- **Simplicity**: No database dependencies, just files
- **Version Control Friendly**: Can track changes in git
- **Easy Backup**: Just copy the entire working directory
- **Debugging Friendly**: Can inspect any piece of data directly
- **Format Flexibility**: Each storage type optimized for its use case
- **Fast Development**: No schema migrations or database administration

---

## Complete Walkthrough Examples

### Example 1: From Document to Query - The Full Journey

Let's trace a complete example from inserting a document to answering queries about it.

**Starting Document**:
```text
"Apple Inc. is a multinational technology company headquartered in Cupertino, California. The company is best known for its consumer electronics, including the iPhone smartphone, iPad tablets, and Mac computers. Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. Today, Tim Cook serves as the CEO. Apple's iPhone has revolutionized mobile communications and remains one of the most popular smartphones globally."
```

### Phase 1: Document Insertion Process

**Step 1 - Document Processing** (`graphrag.py:266-275`):
```python
# Document gets hashed and stored
doc_id = compute_mdhash_id(content, prefix="doc-")  # Result: "doc-a1b2c3d4e5"
new_docs = {"doc-a1b2c3d4e5": {"content": "Apple Inc. is a multinational..."}}
```

**Step 2 - Chunking** (`_op.py:145-175`):
Since the document is short (< 32,768 tokens), it becomes a single chunk:
```python
chunk_id = compute_mdhash_id(content, prefix="chunk-")  # Result: "chunk-x1y2z3a4b5"
chunk_data = {
    "content": "Apple Inc. is a multinational technology company...",
    "tokens": 87,
    "chunk_order_index": 0,
    "full_doc_id": "doc-a1b2c3d4e5"
}
```

**Step 3 - Entity Extraction** (`_op.py:350-450`):
The LLM analyzes the chunk and returns:
```text
("entity"|"APPLE INC"|"organization"|"Multinational technology company headquartered in Cupertino, California")##
("entity"|"CUPERTINO"|"geo"|"City in California where Apple is headquartered")##
("entity"|"CALIFORNIA"|"geo"|"State where Apple is located")##
("entity"|"IPHONE"|"product"|"Smartphone product by Apple that revolutionized mobile communications")##
("entity"|"IPAD"|"product"|"Tablet computer product by Apple")##
("entity"|"MAC"|"product"|"Computer product line by Apple")##
("entity"|"STEVE JOBS"|"person"|"Co-founder of Apple Inc in 1976")##
("entity"|"TIM COOK"|"person"|"Current CEO of Apple Inc")##
("relationship"|"APPLE INC"|"CUPERTINO"|"headquartered in"|9)##
("relationship"|"APPLE INC"|"IPHONE"|"manufactures"|8)##
("relationship"|"APPLE INC"|"STEVE JOBS"|"founded by"|7)##
("relationship"|"TIM COOK"|"APPLE INC"|"serves as CEO of"|8)##
<|COMPLETE|>
```

**Step 4 - Graph Construction**:
The extracted entities and relationships are added to the NetworkX graph:
- 8 nodes (entities) added with their descriptions and metadata
- 4 edges (relationships) added with weights and descriptions
- All traced back to source chunk "chunk-x1y2z3a4b5"

**Step 5 - Community Detection**:
The Leiden algorithm identifies communities:
- **Community-0**: [APPLE INC, IPHONE, IPAD, MAC, TIM COOK] - "Apple Product Ecosystem"
- **Community-1**: [STEVE JOBS, APPLE INC] - "Apple Founders"  
- **Community-2**: [CUPERTINO, CALIFORNIA, APPLE INC] - "Apple Geographic Presence"

**Step 6 - Community Report Generation**:
For Community-0, the LLM generates:
```json
{
  "title": "Apple Consumer Technology Ecosystem",
  "summary": "This community represents Apple Inc's core consumer technology products and leadership, highlighting the company's focus on integrated hardware and software solutions.",
  "rating": 8.5,
  "rating_explanation": "High impact due to Apple's significant influence on consumer technology markets.",
  "findings": [
    {
      "summary": "Comprehensive product portfolio",
      "explanation": "Apple maintains a diverse portfolio of consumer electronics including smartphones (iPhone), tablets (iPad), and computers (Mac), all designed to work together as an integrated ecosystem."
    },
    {
      "summary": "Strong leadership continuity", 
      "explanation": "The transition from founder Steve Jobs to current CEO Tim Cook demonstrates organizational stability and continued innovation focus."
    }
  ]
}
```

**Step 7 - File Generation**:
All storage files are written:
- `kv_store_full_docs.json`: Original document stored
- `kv_store_text_chunks.json`: Single chunk stored with metadata
- `kv_store_community_reports.json`: Community analysis stored
- `vdb_entities.json`: Entity embeddings created and stored
- `graph_chunk_entity_relation.graphml`: Complete graph structure saved

### Phase 2: Query Processing Examples

**Query 1: Local Query - "What products does Apple make?"**

**Step 1 - Entity Discovery**:
- Query embedding: [0.1234, -0.5678, 0.9012, ...]
- Vector search finds similar entities: APPLE INC (0.95), IPHONE (0.87), IPAD (0.82), MAC (0.79)

**Step 2 - Context Assembly**:
```csv
-----Entities-----
id,entity,type,description,rank
0,APPLE INC,organization,Multinational technology company headquartered in Cupertino California,950
1,IPHONE,product,Smartphone product by Apple that revolutionized mobile communications,870
2,IPAD,product,Tablet computer product by Apple,820
3,MAC,product,Computer product line by Apple,790

-----Relationships-----
id,source,target,description,weight,rank
0,APPLE INC,IPHONE,manufactures,8,800
1,APPLE INC,IPAD,manufactures,8,800  
2,APPLE INC,MAC,manufactures,8,800

-----Communities-----
id,content
0,"# Apple Consumer Technology Ecosystem\nThis community represents Apple Inc's core consumer technology products..."

-----Sources-----
id,content
0,"Apple Inc. is a multinational technology company... The company is best known for its consumer electronics, including the iPhone smartphone, iPad tablets, and Mac computers..."
```

**Step 3 - Answer Generation**:
The LLM receives this structured context and generates:
```text
"Based on the information available, Apple Inc. manufactures several key consumer technology products:

1. **iPhone** - Apple's flagship smartphone that revolutionized mobile communications and remains one of the most popular smartphones globally.

2. **iPad** - Apple's line of tablet computers that established the modern tablet market.

3. **Mac** - Apple's computer product line, including desktop and laptop computers.

These products are designed to work together as part of Apple's integrated technology ecosystem, with the company maintaining its headquarters in Cupertino, California."
```

**Query 2: Global Query - "What are the major trends in technology companies?"**

**Step 1 - Community Analysis**:
- All communities are analyzed for relevance to "technology companies"
- Community-0 (Apple Product Ecosystem) scores 8/10 for relevance
- Community reports are fed to LLM for trend analysis

**Step 2 - Insight Extraction**:
From Community-0 analysis:
```text
"The Apple ecosystem demonstrates a key trend in technology companies toward integrated product portfolios. Rather than focusing on single products, successful tech companies are building comprehensive ecosystems where hardware, software, and services work together seamlessly. This approach creates stronger customer loyalty and recurring revenue streams."
```

**Step 3 - Synthesis**:
The global query combines insights from all relevant communities to provide comprehensive trend analysis.

### Why This Complete Process Works

**The full pipeline demonstrates several key strengths**:

1. **Information Preservation**: Original document content is never lost, always traceable
2. **Intelligent Structuring**: Unstructured text becomes a queryable knowledge graph
3. **Multi-Level Analysis**: Can answer both specific entity questions and broad thematic questions
4. **Scalability**: Process works for single documents or massive document collections
5. **Flexibility**: Three query modes handle different types of information needs
6. **Transparency**: Every step is observable and debuggable

**The result**: A system that can take any collection of documents and make them queryable in sophisticated ways that go far beyond simple keyword search or basic RAG similarity matching.

### The Strategic Value

**Nano-graphrag bridges the gap** between:
- **Simple RAG**: Good for basic similarity search, but misses complex relationships
- **Traditional Knowledge Graphs**: Powerful for structured queries, but require predefined schemas
- **Pure LLM Systems**: Great for reasoning, but limited by context windows and hallucination

**By combining the best of all approaches**, nano-graphrag creates a uniquely powerful system for turning documents into knowledge and knowledge into insights.
