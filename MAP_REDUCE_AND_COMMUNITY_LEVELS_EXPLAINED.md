# Map-Reduce and Community Levels: Complete Conceptual Guide

## What is Map-Reduce in the Context of Nano-GraphRAG?

Map-Reduce is a computational pattern borrowed from distributed computing that nano-graphrag uses to analyze large amounts of information (community reports) in parallel and then synthesize the results. Think of it like having multiple research assistants analyze different parts of a library, then combining their findings.

### Real-World Analogy

Imagine you're a research director trying to answer: "What are the main trends in AI research?"

**Without Map-Reduce (Traditional Approach):**
- You personally read through 100 research papers
- You try to remember and synthesize everything at once
- This is slow and you might miss connections or get overwhelmed

**With Map-Reduce (Nano-GraphRAG's Approach):**
- **Map Phase**: You hire 5 research assistants
  - Assistant 1 analyzes papers 1-20 and reports: "I see trends in deep learning and neural networks"
  - Assistant 2 analyzes papers 21-40 and reports: "I see trends in computer vision and NLP"
  - Assistant 3 analyzes papers 41-60 and reports: "I see trends in robotics and autonomous systems"
  - Assistant 4 analyzes papers 61-80 and reports: "I see trends in ethical AI and bias"
  - Assistant 5 analyzes papers 81-100 and reports: "I see trends in quantum computing and AI"

- **Reduce Phase**: You take all assistant reports and synthesize:
  - "The main trends are: 1) Deep learning dominance, 2) Computer vision breakthroughs, 3) Ethical AI concerns, 4) Emerging quantum-AI hybrid approaches"

## Part 1: How Community Levels Are Determined

### The Mathematical Foundation: Resolution Parameters

Community detection uses the Leiden algorithm with different "resolution" parameters to create hierarchical levels:

```python
def _detect_hierarchical_communities(self):
    """
    Create multiple levels of communities using different resolution parameters
    """
    import networkx.algorithms.community as nx_community
    
    # Level 0: Fine-grained communities (HIGH resolution = SMALL communities)
    level_0_communities = list(nx_community.greedy_modularity_communities(
        self._graph, 
        weight='weight',
        resolution=1.2  # Higher number = more sensitive = smaller communities
    ))
    
    # Level 1: Coarse-grained communities (LOW resolution = LARGE communities)  
    level_1_communities = list(nx_community.greedy_modularity_communities(
        self._graph,
        weight='weight', 
        resolution=0.8  # Lower number = less sensitive = larger communities
    ))
    
    # Level 2: Very coarse communities (VERY LOW resolution = VERY LARGE communities)
    level_2_communities = list(nx_community.greedy_modularity_communities(
        self._graph,
        weight='weight',
        resolution=0.4  # Even lower = even larger communities
    ))
```

### What Resolution Parameter Actually Means

Think of resolution like a microscope setting:

- **High Resolution (1.2)**: Like zooming in with a microscope
  - You see fine details and small groups
  - Creates many small, tightly-knit communities
  - Example: {TensorFlow, Keras} as one community, {PyTorch, Torchvision} as another

- **Low Resolution (0.8)**: Like zooming out  
  - You see broader patterns and larger groups
  - Creates fewer, larger communities
  - Example: {TensorFlow, Keras, PyTorch, Torchvision, Scikit-learn} all in one "Machine Learning Frameworks" community

### Concrete Example: Academic Paper Knowledge Graph

Let's say we have entities from AI research papers:

**Raw Entities:**
```
TENSORFLOW, KERAS, PYTORCH, TORCHVISION, SCIKIT_LEARN, PANDAS, 
NUMPY, MATPLOTLIB, JUPYTER, OPENCV, PIL, HUGGINGFACE, 
TRANSFORMERS, BERT, GPT, LSTM, CNN, RNN
```

**Level 0 Communities (Resolution = 1.2):** 6 small, specific communities
```
Community 0-0: [TENSORFLOW, KERAS]                    # TensorFlow ecosystem
Community 0-1: [PYTORCH, TORCHVISION]                 # PyTorch ecosystem  
Community 0-2: [SCIKIT_LEARN, PANDAS, NUMPY]         # Classical ML tools
Community 0-3: [MATPLOTLIB, JUPYTER]                 # Visualization tools
Community 0-4: [HUGGINGFACE, TRANSFORMERS, BERT, GPT] # NLP models
Community 0-5: [LSTM, CNN, RNN]                      # Neural architectures
```

**Level 1 Communities (Resolution = 0.8):** 3 broader communities
```
Community 1-0: [TENSORFLOW, KERAS, PYTORCH, TORCHVISION]           # Deep Learning Frameworks
Community 1-1: [SCIKIT_LEARN, PANDAS, NUMPY, MATPLOTLIB, JUPYTER]  # Data Science Stack
Community 1-2: [HUGGINGFACE, TRANSFORMERS, BERT, GPT, LSTM, CNN, RNN] # AI Models & Architectures
```

**Level 2 Communities (Resolution = 0.4):** 1 very broad community
```
Community 2-0: [All entities] # "Machine Learning & AI Ecosystem"
```

### The Assignment Algorithm

```python
def _assign_hierarchical_levels(self, all_communities_by_resolution):
    """
    Assign entities to hierarchical community levels based on containment relationships
    """
    hierarchical_structure = {}
    
    # Start with Level 0 (finest resolution)
    for level_0_idx, community_nodes in enumerate(all_communities_by_resolution[1.2]):
        community_id = f"0-{level_0_idx}"
        
        # Find parent community at Level 1
        parent_community = None
        for level_1_idx, level_1_nodes in enumerate(all_communities_by_resolution[0.8]):
            # Check if this Level 0 community is mostly contained in this Level 1 community
            overlap_ratio = len(community_nodes.intersection(level_1_nodes)) / len(community_nodes)
            if overlap_ratio >= 0.7:  # 70% overlap threshold
                parent_community = f"1-{level_1_idx}"
                break
        
        # Calculate community metrics
        hierarchical_structure[community_id] = {
            "level": 0,
            "nodes": list(community_nodes),
            "parent_community": parent_community,
            "sub_communities": [],  # Level 0 has no children
            "occurrence": len(community_nodes),  # Number of entities
            # ... other metrics
        }
    
    # Process Level 1 communities
    for level_1_idx, community_nodes in enumerate(all_communities_by_resolution[0.8]):
        community_id = f"1-{level_1_idx}"
        
        # Find child communities (Level 0 communities contained within)
        sub_communities = []
        for level_0_id, level_0_data in hierarchical_structure.items():
            if level_0_data.get("parent_community") == community_id:
                sub_communities.append(level_0_id)
        
        # Find parent community at Level 2
        parent_community = None
        for level_2_idx, level_2_nodes in enumerate(all_communities_by_resolution[0.4]):
            overlap_ratio = len(community_nodes.intersection(level_2_nodes)) / len(community_nodes)
            if overlap_ratio >= 0.7:
                parent_community = f"2-{level_2_idx}"
                break
        
        hierarchical_structure[community_id] = {
            "level": 1,
            "nodes": list(community_nodes),
            "parent_community": parent_community,
            "sub_communities": sub_communities,
            "occurrence": len(community_nodes),
        }
```

### Why This Hierarchy Matters

The hierarchical structure allows nano-graphrag to answer different types of questions:

- **Detailed Questions** → Use Level 0 communities (specific, focused)
  - "How does TensorFlow integrate with Keras?"
  - Answer uses Community 0-0: [TENSORFLOW, KERAS]

- **Medium-Scope Questions** → Use Level 1 communities (moderate scope)
  - "What are the main deep learning frameworks?"
  - Answer uses Community 1-0: [TENSORFLOW, KERAS, PYTORCH, TORCHVISION]

- **Broad Questions** → Use Level 2 communities (comprehensive)
  - "What is the overall AI/ML ecosystem?"
  - Answer uses Community 2-0: [All entities]

## Part 2: Community Grouping for Token Management

### The Problem: Token Limits

LLMs have context limits (e.g., GPT-4 has ~128k tokens). When doing global queries, we might have hundreds of community reports, each containing thousands of tokens. We can't send them all to the LLM at once.

**Example Problem:**
```
Community Report 1: 3,000 tokens
Community Report 2: 2,500 tokens  
Community Report 3: 4,000 tokens
... (50 more reports)
Total: 180,000 tokens → EXCEEDS LLM LIMIT!
```

### The Solution: Smart Grouping

```python
def _group_communities_by_tokens(self, communities, max_tokens_per_group=12000):
    """
    Group communities into batches that fit within token limits
    """
    groups = []
    current_group = []
    current_group_tokens = 0
    
    for community in communities:
        # Calculate tokens for this community report
        report_text = community["report_string"]
        community_tokens = len(encode_string_by_tiktoken(report_text))
        
        # If adding this community would exceed limit, start new group
        if current_group_tokens + community_tokens > max_tokens_per_group:
            if current_group:  # Save current group if not empty
                groups.append(current_group)
                current_group = []
                current_group_tokens = 0
        
        # Add community to current group
        current_group.append(community)
        current_group_tokens += community_tokens
    
    # Add final group if not empty
    if current_group:
        groups.append(current_group)
    
    logger.info(f"Grouped {len(communities)} communities into {len(groups)} groups")
    return groups
```

### Real Example of Grouping

Let's say we have 10 community reports:

```
Community A: "Machine Learning Frameworks..." (3,000 tokens)
Community B: "Data Science Tools..." (2,500 tokens)  
Community C: "Neural Network Architectures..." (4,000 tokens)
Community D: "Computer Vision Applications..." (3,500 tokens)
Community E: "Natural Language Processing..." (2,800 tokens)
Community F: "Robotics and AI..." (3,200 tokens)
Community G: "Ethical AI Considerations..." (2,100 tokens)
Community H: "Quantum Computing and AI..." (1,900 tokens)
Community I: "AI in Healthcare..." (3,800 tokens)
Community J: "Autonomous Systems..." (2,700 tokens)
```

**Grouping Process (max 12,000 tokens per group):**

**Group 1:** 
- Community A (3,000) + Community B (2,500) + Community C (4,000) = 9,500 tokens ✓
- Can we add Community D (3,500)? 9,500 + 3,500 = 13,000 > 12,000 ✗
- **Final Group 1: [A, B, C] = 9,500 tokens**

**Group 2:**
- Community D (3,500) + Community E (2,800) + Community F (3,200) = 9,500 tokens ✓
- Can we add Community G (2,100)? 9,500 + 2,100 = 11,600 ≤ 12,000 ✓
- **Final Group 2: [D, E, F, G] = 11,600 tokens**

**Group 3:**
- Community H (1,900) + Community I (3,800) + Community J (2,700) = 8,400 tokens ✓
- **Final Group 3: [H, I, J] = 8,400 tokens**

**Result:** `"Grouping to 3 groups for global search"` (this is the log message you see!)

## Part 3: The Map-Reduce Process in Detail

### Map Phase: Parallel Analysis

Each group gets sent to the LLM as a separate "analyst" with this prompt:

```python
async def _analyze_community_group(self, query, community_group):
    """
    Each group becomes an independent analyst
    """
    # Format the group's communities as CSV
    communities_csv = "-----Communities-----\nid,content,rating,importance\n"
    for i, community in enumerate(community_group):
        content = community["report_string"].replace('"', '""')  # Escape quotes
        rating = community["report_json"].get("rating", 0)       # LLM-assigned quality score
        importance = community["occurrence"]                     # Number of entities
        communities_csv += f'{i},"{content}",{rating},{importance}\n'
    
    # The analyst prompt
    map_prompt = f"""
    ---Role---
    You are Research Analyst #{group_index}. You have been given a subset of community reports to analyze.
    
    ---Goal---
    Analyze the community reports below and extract key insights that help answer the user's question.
    For each insight, assign an importance score (0-100).
    
    ---Question---
    {user_query}
    
    ---Your Assigned Communities---
    {communities_csv}
    
    ---Instructions---
    Return a JSON list of insights:
    {{
        "points": [
            {{"description": "insight 1", "score": 85}},
            {{"description": "insight 2", "score": 72}}
        ]
    }}
    """
    
    # Get response from LLM
    response = await self.llm_func(user_query, system_prompt=map_prompt)
    return json.loads(response)
```

### Real Map Phase Example

**User Question:** "What are the main challenges in modern AI development?"

**Analyst 1** (analyzing Group 1: [A, B, C]):
```json
{
  "points": [
    {"description": "Framework compatibility issues between TensorFlow and PyTorch create development friction", "score": 78},
    {"description": "Model deployment complexity increases with framework-specific requirements", "score": 72}
  ]
}
```

**Analyst 2** (analyzing Group 2: [D, E, F, G]):
```json
{
  "points": [
    {"description": "Ethical AI concerns around bias and fairness are becoming critical", "score": 92},
    {"description": "Computer vision models struggle with edge cases and adversarial inputs", "score": 68}
  ]
}
```

**Analyst 3** (analyzing Group 3: [H, I, J]):
```json
{
  "points": [
    {"description": "Healthcare AI faces regulatory compliance and data privacy challenges", "score": 88},
    {"description": "Quantum computing integration with AI is still experimental", "score": 45}
  ]
}
```

### Reduce Phase: Synthesis

Now we take all the analyst insights and synthesize them:

```python
async def _reduce_insights_synthesis(self, query, all_analyst_insights, query_params):
    """
    Combine insights from all analysts into final response
    """
    # Flatten all insights with analyst attribution
    all_insights = []
    for analyst_idx, analyst_results in enumerate(all_analyst_insights):
        for insight in analyst_results.get("points", []):
            all_insights.append({
                "analyst": analyst_idx,
                "description": insight["description"],
                "score": insight["score"]
            })
    
    # Sort by importance score and filter out low-scoring insights
    high_quality_insights = [i for i in all_insights if i["score"] >= 60]  # Threshold
    high_quality_insights.sort(key=lambda x: x["score"], reverse=True)
    
    # Format for final synthesis
    synthesis_context = ""
    for insight in high_quality_insights:
        synthesis_context += f"""----Analyst {insight['analyst']}----
Importance Score: {insight['score']}
{insight['description']}

"""
    
    # Final synthesis prompt
    reduce_prompt = f"""
    ---Role---
    You are the Research Director synthesizing insights from multiple analysts.
    
    ---Goal---  
    Create a comprehensive answer to the user's question by combining the best insights from your analysts.
    The analysts have already ranked their insights by importance (higher scores = more important).
    
    ---Question---
    {query}
    
    ---Analyst Reports (ranked by importance)---
    {synthesis_context}
    
    ---Instructions---
    Write a well-structured response that:
    1. Answers the user's question directly
    2. Incorporates the highest-scoring insights
    3. Identifies patterns across multiple analysts
    4. Provides a balanced, comprehensive view
    """
    
    final_response = await self.llm_func(query, system_prompt=reduce_prompt)
    return final_response
```

### The Final Synthesis

The Research Director LLM sees:

```
----Analyst 0----
Importance Score: 78
Framework compatibility issues between TensorFlow and PyTorch create development friction

----Analyst 0----  
Importance Score: 72
Model deployment complexity increases with framework-specific requirements

----Analyst 1----
Importance Score: 92
Ethical AI concerns around bias and fairness are becoming critical

----Analyst 2----
Importance Score: 88
Healthcare AI faces regulatory compliance and data privacy challenges

----Analyst 1----
Importance Score: 68
Computer vision models struggle with edge cases and adversarial inputs
```

And produces a final synthesized response:

```
The main challenges in modern AI development fall into three critical categories:

1. **Ethical and Regulatory Challenges (Highest Priority)**
   - Bias and fairness concerns are becoming critical as AI systems impact more lives
   - Healthcare and other regulated industries face complex compliance requirements
   - Data privacy regulations create additional development constraints

2. **Technical Integration Challenges** 
   - Framework compatibility issues between major platforms like TensorFlow and PyTorch
   - Model deployment complexity varies significantly across different frameworks
   - Computer vision systems still struggle with edge cases and adversarial inputs

3. **Infrastructure and Scalability Challenges**
   - Production deployment requires specialized infrastructure knowledge
   - Performance optimization across different hardware platforms remains complex

The analysis shows that ethical considerations have emerged as the top priority, with regulatory compliance being particularly challenging in sensitive domains like healthcare.
```

## Part 4: Why This Approach Works

### Advantages of Map-Reduce for Knowledge Synthesis

1. **Parallel Processing**: Multiple LLM calls happen simultaneously, much faster than sequential analysis

2. **Token Efficiency**: Each analyst only needs to process a manageable amount of information

3. **Specialized Focus**: Each analyst can deeply focus on their assigned communities without distraction

4. **Quality Control**: The reduce phase can filter out low-quality insights and identify patterns

5. **Scalability**: Can handle arbitrarily large knowledge bases by adding more analyst groups

### Real Performance Benefits

**Without Map-Reduce:**
- Single LLM call with 180,000 tokens → Exceeds context limit OR costs $50+ per query
- Response quality degrades with too much information

**With Map-Reduce:**
- 3 parallel LLM calls with 12,000 tokens each = 36,000 total tokens
- 1 final synthesis call with 5,000 tokens
- **Total: 41,000 tokens, ~$2 per query, much higher quality**

This is why nano-graphrag's global queries are both cost-effective and produce high-quality comprehensive responses even with massive knowledge bases.

The map-reduce approach transforms the impossible task of "analyze everything at once" into the manageable task of "have specialists analyze parts, then synthesize their expertise."