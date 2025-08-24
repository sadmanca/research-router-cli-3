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

### Why This Community Approach Works

**Traditional topic modeling problems**:
- **Fixed Number of Topics**: Have to specify how many topics you want in advance
- **Static Structure**: Topics don't adapt as new documents are added
- **No Hierarchical Understanding**: Can't zoom in/out on topic granularity
- **Weak Entity Relationships**: Don't capture how specific entities relate within topics

**Nano-graphrag's community advantages**:
- **Dynamic Topic Discovery**: Communities emerge naturally from the data
- **Hierarchical Flexibility**: Can query at different levels of granularity
- **Rich Relationship Modeling**: Understands how entities connect within communities
- **Incremental Updates**: Can evolve communities as new information is added
- **AI-Enhanced Interpretation**: LLM-generated reports provide human-understandable context

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
