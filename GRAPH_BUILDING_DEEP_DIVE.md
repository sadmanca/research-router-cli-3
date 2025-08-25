# Graph Building Deep Dive: Complete Technical Analysis

## Overview

This document provides an ultra-detailed technical analysis of how nano-graphrag builds knowledge graphs from documents and processes different types of queries. We'll explain every step, from text chunking to final LLM responses, including the mysterious log messages you see during processing.

## Part 1: Graph Building Process - "⠏ Building knowledge graph..."

### Step 1: Document Processing and Chunking

When you see initial processing, nano-graphrag is converting documents into manageable chunks:

```python
def chunking_by_token_size(tokens_list, doc_keys, tiktoken_model, overlap_token_size=128, max_token_size=1024):
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        # Create overlapping chunks with sliding window
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))
        
        # Decode token chunks back to text
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append({
                "tokens": lengths[i],
                "content": chunk.strip(),
                "chunk_order_index": i,
                "full_doc_id": doc_keys[index],
            })
    return results
```

**What's Happening:**
- Documents are tokenized using tiktoken (GPT-4o tokenizer)
- Text is split into overlapping chunks (default: 1024 tokens with 128 token overlap)
- Each chunk maintains reference to its source document (`full_doc_id`)
- Overlap ensures context isn't lost at chunk boundaries

### Step 2: Entity Extraction - The Core Graph Building

#### Traditional Method (`extract_entities`)

```python
async def extract_entities(chunks, knwoledge_graph_inst, entity_vdb, global_config):
    # Extract entities using LLM prompts for each chunk
    entity_extract_prompt = PROMPTS["entity_extraction"]
    
    async def _process_single_content(chunk_key_dp):
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        
        # Format prompt with chunk content
        hint_prompt = entity_extract_prompt.format(
            tuple_delimiter="<|>",
            record_delimiter="##", 
            completion_delimiter="<|COMPLETE|>",
            entity_types="organization,person,geo,event",
            input_text=content
        )
        
        # Get initial LLM response
        final_result = await use_llm_func(hint_prompt)
        
        # GLEANING PROCESS - Multiple rounds to catch missed entities
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):  # Default: 1 round
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            
            # Check if we should continue gleaning
            if_loop_result = await use_llm_func(if_loop_prompt, history_messages=history)
            if if_loop_result.strip().lower() != "yes":
                break
```

**The LLM Prompt Template:**
```
-Goal-
Given a text document, identify all entities of types [organization, person, geo, event] and relationships.

-Steps-
1. Identify entities: ("entity"<|>ENTITY_NAME<|>ENTITY_TYPE<|>DESCRIPTION)
2. Identify relationships: ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>WEIGHT)
3. Return as delimited list using ## as delimiter
4. End with <|COMPLETE|>

Text: [CHUNK_CONTENT]
```

#### GenKG Method (`extract_entities_genkg`)

```python
async def extract_entities_genkg(chunks, knwoledge_graph_inst, entity_vdb, global_config):
    # Group chunks back into documents
    papers_dict = {}
    for chunk_key, chunk_data in chunks.items():
        doc_id = chunk_data.get("full_doc_id", chunk_key)
        if doc_id not in papers_dict:
            papers_dict[doc_id] = ""
        papers_dict[doc_id] += chunk_data["content"] + "\n\n"
    
    async def _process_document(doc_id: str, doc_content: str):
        # 1. Summarize the document for better entity extraction
        summary = genkg.summarize_paper(doc_content, doc_id)
        
        # 2. Extract nodes using GenKG's advanced prompting
        nodes_with_source = genkg.gemini_create_nodes(summary, node_limit, doc_id)
        
        # 3. Extract edges using contextual relationship analysis
        edges = genkg.create_edges_by_gemini(nodes_with_source, {doc_id: summary})
```

### Step 3: Entity and Relationship Merging

After extraction, nano-graphrag merges duplicate entities and relationships:

```python
async def _merge_nodes_then_upsert(entity_name, nodes_data, knwoledge_graph_inst, global_config):
    # Check if entity already exists
    already_node = await knwoledge_graph_inst.get_node(entity_name)
    
    if already_node is not None:
        # Merge with existing data
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(already_node["source_id"].split(GRAPH_FIELD_SEP))
        already_description.append(already_node["description"])
    
    # Determine most common entity type
    entity_type = Counter([dp["entity_type"] for dp in nodes_data] + already_entity_types).most_common(1)[0][0]
    
    # Merge descriptions and sources
    description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in nodes_data] + already_description)))
    source_id = GRAPH_FIELD_SEP.join(set([dp["source_id"] for dp in nodes_data] + already_source_ids))
    
    # Summarize if description too long
    description = await _handle_entity_relation_summary(entity_name, description, global_config)
```

### Step 4: Graph Storage

The merged entities and relationships are stored in the graph storage system:

```python
await knwoledge_graph_inst.upsert_node(entity_name, node_data={
    "entity_type": entity_type,
    "description": description, 
    "source_id": source_id,
})

await knwoledge_graph_inst.upsert_edge(src_id, tgt_id, edge_data={
    "weight": weight,
    "description": description,
    "source_id": source_id,
    "order": order
})
```

## Part 2: Community Detection - "INFO:nano-graphrag:[Community Report]..."

### What "Each level has communities: {0: 4, 1: 2}" Means

This message indicates the hierarchical community structure detected by the Leiden algorithm:

```python
async def clustering(self, algorithm: str):
    # Apply Leiden algorithm to detect communities
    communities = nx_leiden(self._graph, resolution=1.0, seed=0xDEADBEEF)
    
    # Build hierarchical structure
    community_schema = {}
    for level in range(max_levels):
        level_communities = {}
        for community_id, nodes in communities_by_level[level].items():
            level_communities[f"{level}-{community_id}"] = {
                "level": level,
                "title": f"Community {community_id}",
                "nodes": list(nodes),
                "edges": [(u, v) for u, v in graph.edges() if u in nodes and v in nodes],
                "sub_communities": [] if level == 0 else [sub_comm_ids]
            }
```

**Hierarchical Structure:**
- **Level 0**: Base communities (4 communities) - tightly connected entity clusters
- **Level 1**: Super-communities (2 communities) - groups of related Level 0 communities
- Higher levels group lower levels into increasingly abstract clusters

### Community Report Generation Process

```python
async def generate_community_report(community_report_kv, knwoledge_graph_inst, global_config):
    communities_schema = await knwoledge_graph_inst.community_schema()
    
    # Process communities by level (highest first)
    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")  # This is the log message you see
    
    for level in levels:
        # Get communities at this level
        this_level_communities = [c for c in communities if c["level"] == level]
        
        # Generate reports for each community
        for community in this_level_communities:
            # Pack community data for LLM
            describe = await _pack_single_community_describe(
                knwoledge_graph_inst, community, max_token_size=12000
            )
            
            # Generate report using LLM
            prompt = community_report_prompt.format(input_text=describe)
            response = await use_llm_func(prompt, response_format={"type": "json_object"})
```

**Community Data Structure Sent to LLM:**
```csv
-----Reports-----
id,content
0,"Previous sub-community reports if available"

-----Entities-----
id,entity,type,description,degree
0,MACHINE_LEARNING,CONCEPT,"AI technique for pattern recognition",15
1,NEURAL_NETWORKS,CONCEPT,"Computing systems inspired by biological neural networks",12

-----Relationships-----
id,source,target,description,weight,rank
0,MACHINE_LEARNING,NEURAL_NETWORKS,"Neural networks are a subset of machine learning",3.5,8
```

The LLM then generates structured reports:
```json
{
    "title": "Machine Learning and AI Techniques",
    "summary": "This community focuses on artificial intelligence methodologies...",
    "rating": 8.5,
    "rating_explanation": "High importance due to central role in AI research",
    "findings": [
        {
            "summary": "Core AI Technologies", 
            "explanation": "Machine learning serves as the foundation..."
        }
    ]
}
```

## Part 3: Query Processing Deep Dive

### Local Query Processing - "⠇ Searching knowledge graph (local mode)..."

#### Step 1: Vector Search for Relevant Entities

```python
async def local_query(query, knowledge_graph_inst, entities_vdb, community_reports, text_chunks_db, query_param, global_config):
    # 1. Search for relevant entities using vector similarity
    results = await entities_vdb.query(query, top_k=query_param.top_k)  # Default: top_k=20
    
    if not len(results):
        logger.warning(f"No entities found for query: '{query}'")
        return PROMPTS["fail_response"]
    
    # Each result: {"entity_name": "ENTITY", "id": "ent-hash", "distance": 0.85}
```

#### Step 2: Build Local Context

```python
async def _build_local_query_context(query, knowledge_graph_inst, entities_vdb, community_reports, text_chunks_db, query_param):
    # Get detailed node data for found entities
    node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r["entity_name"]) for r in results])
    
    # Find related communities
    use_communities = await _find_most_related_community_from_entities(node_datas, query_param, community_reports)
    
    # Find related text chunks
    use_text_units = await _find_most_related_text_unit_from_entities(node_datas, query_param, text_chunks_db, knowledge_graph_inst)
    
    # Find related relationships/edges
    use_relations = await _find_most_related_edges_from_entities(node_datas, query_param, knowledge_graph_inst)
    
    logger.info(f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units")
```

#### How Related Communities Are Found

```python
async def _find_most_related_community_from_entities(node_datas, query_param, community_reports):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        # Each node stores which communities it belongs to
        related_communities.extend(json.loads(node_d["clusters"]))
    
    # Count community occurrences and rank by importance
    related_community_keys_counts = dict(Counter([str(dp["cluster"]) for dp in related_communities if dp["level"] <= query_param.level]))
    
    # Sort by: (occurrence_count, community_rating)
    sorted_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
```

#### Step 3: Context Assembly for LLM

The local query context is assembled in CSV format:

```python
return f"""
-----Reports-----
```csv
id,content
0,"Community report: This community focuses on machine learning algorithms and their applications in data science..."
1,"Community report: Neural network architectures including deep learning frameworks..."
```

-----Entities-----
```csv
id,entity,type,description,rank
0,MACHINE_LEARNING,CONCEPT,"Artificial intelligence technique for pattern recognition and prediction",15
1,TENSORFLOW,ORGANIZATION,"Open-source machine learning framework developed by Google",12
```

-----Relationships-----
```csv
id,source,target,description,weight,rank
0,TENSORFLOW,MACHINE_LEARNING,"TensorFlow is a framework for implementing machine learning algorithms",2.5,8
```

-----Sources-----
```csv
id,content
0,"Original text chunk: Machine learning has revolutionized the field of artificial intelligence..."
1,"Original text chunk: TensorFlow provides comprehensive tools for building ML models..."
```
"""
```

#### Step 4: LLM Response Generation

```python
sys_prompt_temp = PROMPTS["local_rag_response"]
sys_prompt = sys_prompt_temp.format(context_data=context, response_type=query_param.response_type)

response = await use_model_func(query, system_prompt=sys_prompt)
```

**The Local RAG System Prompt:**
```
---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---
{response_type}

---Data tables---
{context_data}
```

### Global Query Processing - "⠋ Searching knowledge graph (global mode)..."

Global queries work differently - they use community reports for high-level analysis:

#### Step 1: Community Selection

```python
async def global_query(query, knowledge_graph_inst, entities_vdb, community_reports, text_chunks_db, query_param, global_config):
    # Get all communities at or below specified level
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {k: v for k, v in community_schema.items() if v["level"] <= query_param.level}
    
    # Sort by importance (occurrence count)
    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],  # How many entities are in this community
        reverse=True,
    )
    
    # Take top N communities
    sorted_community_schemas = sorted_community_schemas[:query_param.global_max_consider_community]  # Default: 512
    
    # Filter by minimum rating
    community_datas = [c for c in community_datas if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating]
    
    logger.info(f"Retrieved {len(community_datas)} communities")  # This is the log message you see
```

#### Step 2: Map-Reduce Analysis

```python
async def _map_global_communities(query, communities_data, query_param, global_config):
    # Group communities into batches to fit token limits
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,  # Default: 12000
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group):]
    
    logger.info(f"Grouping to {len(community_groups)} groups for global search")  # This is the log you see
    
    async def _process(community_truncated_datas):
        # Format communities as CSV for LLM
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append([
                i,
                c["report_string"],  # Full community report
                c["report_json"].get("rating", 0),
                c["occurrence"],  # How many entities in community
            ])
        
        # Send to LLM for analysis
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        response = await use_model_func(query, system_prompt=sys_prompt)
        
        # Expected response format:
        return {"points": [
            {"description": "Key insight 1", "score": 85},
            {"description": "Key insight 2", "score": 72}
        ]}
```

**Global Map Prompt Template:**
```
---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

Each key point should have:
- Description: A comprehensive description of the point
- Importance Score: Integer 0-100 indicating relevance to the user's question

Response format:
{
    "points": [
        {"description": "Description of point 1", "score": score_value},
        {"description": "Description of point 2", "score": score_value}
    ]
}

---Data tables---
{community_context_csv}
```

#### Step 3: Reduce Phase

```python
# Combine all analyst points
final_support_points = []
for i, mc in enumerate(map_communities_points):
    for point in mc:
        final_support_points.append({
            "analyst": i,
            "answer": point["description"], 
            "score": point.get("score", 1),
        })

# Filter and rank by score
final_support_points = [p for p in final_support_points if p["score"] > 0]
final_support_points = sorted(final_support_points, key=lambda x: x["score"], reverse=True)

# Create final context for LLM
points_context = []
for dp in final_support_points:
    points_context.append(f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
""")
points_context = "\n".join(points_context)

# Generate final response
sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
response = await use_model_func(query, sys_prompt_temp.format(report_data=points_context, response_type=query_param.response_type))
```

### Naive Query Processing - "⠇ Searching knowledge graph (naive mode)..."

The simplest query mode - direct vector search on text chunks:

```python
async def naive_query(query, chunks_vdb, text_chunks_db, query_param, global_config):
    use_model_func = global_config["best_model_func"]
    
    # 1. Vector search on chunk embeddings
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    
    # 2. Retrieve full chunk content
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
    
    # 3. Truncate to fit token limits
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")  # This is the log you see
    
    # 4. Combine chunks for LLM
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    
    # 5. Generate response
    sys_prompt = PROMPTS["naive_rag_response"].format(content_data=section, response_type=query_param.response_type)
    response = await use_model_func(query, system_prompt=sys_prompt)
```

**Naive RAG System Prompt:**
```
You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response that responds to the user's question, summarizing all information appropriate for the response length and format.

---Target response length and format---
{response_type}
```

## Part 4: Token Management and Truncation

Throughout the process, nano-graphrag carefully manages token limits:

```python
def truncate_list_by_token_size(data_list, key, max_token_size):
    """Truncate list to fit within token limits"""
    total_tokens = 0
    truncated_list = []
    
    for item in data_list:
        content = key(item) if callable(key) else item[key]
        tokens = encode_string_by_tiktoken(content, model_name="gpt-4o")
        
        if total_tokens + len(tokens) <= max_token_size:
            truncated_list.append(item)
            total_tokens += len(tokens)
        else:
            break
    
    return truncated_list
```

**Token Limits by Query Type:**
- **Local**: 12,000 tokens for community reports, 8,000 for text units, 8,000 for relationships
- **Global**: 12,000 tokens per community group, 12,000 for final context
- **Naive**: 8,000 tokens for combined chunks

## Summary: The Complete Flow

1. **Graph Building**: Documents → Chunks → Entity Extraction → Graph Storage → Community Detection → Reports
2. **Local Query**: Query → Entity Vector Search → Related Communities/Relations/Text → Context Assembly → LLM Response
3. **Global Query**: Query → Community Selection → Map-Reduce Analysis → Final Response
4. **Naive Query**: Query → Chunk Vector Search → Direct Context → LLM Response

Each approach trades off between specificity and comprehensiveness, with local queries providing detailed entity-focused answers, global queries offering high-level insights across the entire knowledge base, and naive queries giving direct text-based responses.