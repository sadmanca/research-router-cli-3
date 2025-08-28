# Context Limits and Response Formatting: Complete Technical Guide

## Overview

This document provides detailed information about how nano-graphrag manages LLM context limits, handles token budgets across the pipeline, and formats responses according to user preferences. Understanding these mechanisms is crucial for optimizing query performance and getting the desired output format.

## Part 1: LLM Context Limits and Token Management

### Understanding Context Windows

Different LLMs have different context limits that affect how much information can be processed:

```python
LLM_CONTEXT_LIMITS = {
    "gpt-4": 8192,           # 8K tokens (older models)
    "gpt-4-32k": 32768,      # 32K tokens  
    "gpt-4-turbo": 128000,   # 128K tokens
    "gpt-4o": 128000,        # 128K tokens (current default)
    "claude-3": 200000,      # 200K tokens
    "gemini-pro": 1048576,   # 1M tokens (but costly)
}
```

**What This Means in Practice:**
- **Input Context**: How much information (system prompt + user query + context data) can be sent to the LLM
- **Output Context**: How long the LLM's response can be (usually 4K-8K tokens reserved)
- **Total Budget**: Input + Output must not exceed the model's limit

### Nano-GraphRAG's Token Budget Allocation

```python
class TokenBudgetManager:
    def __init__(self, model_name="gpt-4o"):
        self.total_context_limit = 128000  # GPT-4o limit
        self.output_buffer = 4000         # Reserve for LLM response
        self.system_prompt_budget = 2000  # System prompt template
        self.user_query_budget = 500      # User question
        self.available_for_context = self.total_context_limit - self.output_buffer - self.system_prompt_budget - self.user_query_budget
        # Available: 128000 - 4000 - 2000 - 500 = 121,500 tokens for context data
        
    def allocate_by_query_type(self, query_type):
        if query_type == "local":
            return {
                "community_reports": int(self.available_for_context * 0.25),    # ~30,375 tokens
                "entities": int(self.available_for_context * 0.20),             # ~24,300 tokens  
                "relationships": int(self.available_for_context * 0.25),        # ~30,375 tokens
                "text_sources": int(self.available_for_context * 0.30),         # ~36,450 tokens
                "total_allocated": int(self.available_for_context)              # 121,500 tokens
            }
        elif query_type == "global":
            return {
                "community_analysis_per_group": 12000,    # Each analyst gets 12K tokens
                "synthesis_context": int(self.available_for_context * 0.15),  # ~18,225 tokens
                "meta_data": int(self.available_for_context * 0.05)           # ~6,075 tokens
            }
        else:  # naive
            return {
                "text_chunks": self.available_for_context  # All 121,500 tokens
            }
```

### Real Token Usage Examples

#### Local Query Token Breakdown

**Query:** "How does TensorFlow work with neural networks?"

```python
# System prompt tokens
system_prompt = """---Role---
You are a helpful assistant responding to questions about data in the tables provided.
..."""  # ~1,800 tokens

# User query tokens  
user_query = "How does TensorFlow work with neural networks?"  # ~12 tokens

# Context data tokens
community_reports = """-----Reports-----
```csv
id,content
0,"Community: Deep Learning Frameworks - TensorFlow is a comprehensive platform..."
```"""  # ~8,500 tokens

entities_data = """-----Entities-----
```csv  
id,entity,type,description,rank
0,TENSORFLOW,ORGANIZATION,"Open-source machine learning framework...",15
```"""  # ~3,200 tokens

relationships_data = """-----Relationships-----
```csv
id,source,target,description,weight,rank  
0,TENSORFLOW,NEURAL_NETWORKS,"TensorFlow provides tools for building neural networks",2.8,7
```"""  # ~5,100 tokens

text_sources = """-----Sources-----
```csv
id,content
0,"TensorFlow is an end-to-end open source platform for machine learning..."
```"""  # ~12,400 tokens

# Total input context: 1,800 + 12 + 8,500 + 3,200 + 5,100 + 12,400 = 31,012 tokens
# Remaining capacity: 128,000 - 31,012 = 96,988 tokens available for response
```

#### Global Query Token Management

**Query:** "What are the main trends in AI research?"

**Map Phase (Per Analyst):**
```python
# Each analyst gets their own LLM call
analyst_1_input = {
    "system_prompt": "You are Research Analyst #1...",  # ~1,500 tokens
    "user_query": "What are the main trends in AI research?",  # ~10 tokens  
    "community_data": """Communities 1-15 data...""",  # ~11,000 tokens
    "total": 12,510  # Well within limits
}

analyst_2_input = {
    "system_prompt": "You are Research Analyst #2...",  # ~1,500 tokens
    "user_query": "What are the main trends in AI research?",  # ~10 tokens
    "community_data": """Communities 16-30 data...""",  # ~10,800 tokens
    "total": 12,310  # Well within limits
}

# Each analyst produces ~500-1000 tokens of insights
```

**Reduce Phase:**
```python
synthesis_input = {
    "system_prompt": "You are the Research Director...",  # ~2,000 tokens
    "user_query": "What are the main trends in AI research?",  # ~10 tokens
    "analyst_insights": """
----Analyst 1----
Importance Score: 89
AI research is increasingly focused on large language models...
    
----Analyst 2----  
Importance Score: 82
Computer vision applications are expanding into healthcare...
""",  # ~15,000 tokens (all analyst insights combined)
    "total": 17,010  # Well within limits for final synthesis
}
```

### Token Overflow Handling

When context data exceeds limits, nano-graphrag uses intelligent truncation:

```python
def handle_token_overflow(context_components, token_limits):
    """
    Handle context overflow by prioritizing most important information
    """
    # Priority order: entities > relationships > community_reports > text_sources
    priority_order = ["entities", "relationships", "community_reports", "text_sources"]
    
    total_tokens = sum(len(encode_string_by_tiktoken(content)) for content in context_components.values())
    
    if total_tokens <= token_limits["total_available"]:
        return context_components  # No truncation needed
    
    logger.warning(f"Context overflow: {total_tokens} > {token_limits['total_available']} tokens")
    
    # Truncate in reverse priority order (least important first)
    truncated_components = {}
    remaining_budget = token_limits["total_available"]
    
    for component_name in priority_order:
        if component_name in context_components:
            component_content = context_components[component_name]
            component_tokens = len(encode_string_by_tiktoken(component_content))
            
            if component_tokens <= remaining_budget:
                # Component fits completely
                truncated_components[component_name] = component_content
                remaining_budget -= component_tokens
            else:
                # Truncate component to fit remaining budget
                truncated_content = truncate_string_by_token_size(
                    component_content, 
                    max_tokens=remaining_budget
                )
                truncated_components[component_name] = truncated_content
                remaining_budget = 0
                logger.info(f"Truncated {component_name} to fit token budget")
                break
    
    return truncated_components
```

## Part 2: Response Formatting Options

### Available Response Types

Nano-graphrag supports various response formats controlled by the `response_type` parameter:

```python
RESPONSE_TYPES = {
    "single_paragraph": "A single, comprehensive paragraph",
    "multiple_paragraphs": "Multiple detailed paragraphs with clear structure", 
    "bullet_points": "Organized bullet points with hierarchical structure",
    "numbered_list": "Numbered list format with detailed explanations",
    "executive_summary": "Executive summary with key findings and recommendations",
    "detailed_analysis": "Comprehensive analysis with multiple sections and subsections",
    "comparison": "Comparative analysis highlighting similarities and differences",
    "timeline": "Chronological or sequential presentation of information",
    "pros_cons": "Balanced presentation of advantages and disadvantages"
}
```

### How Response Formatting Works

#### System Prompt Integration

The response type is integrated into the system prompt template:

```python
def format_system_prompt(base_prompt, response_type, context_data):
    """
    Inject response formatting instructions into system prompt
    """
    formatting_instructions = {
        "single_paragraph": """
        ---Target response length and format---
        Provide a single, comprehensive paragraph that covers all key points. 
        Use clear, flowing sentences that connect ideas logically.
        Aim for 150-300 words.
        """,
        
        "multiple_paragraphs": """
        ---Target response length and format---
        Structure your response in 3-5 well-organized paragraphs.
        Each paragraph should focus on a distinct aspect or theme.
        Use topic sentences and smooth transitions between paragraphs.
        Aim for 300-600 words total.
        """,
        
        "bullet_points": """
        ---Target response length and format---
        Present information as organized bullet points with:
        • Main points clearly highlighted
        • Sub-points indented where appropriate  
        • Consistent formatting throughout
        • 5-10 main bullet points maximum
        """,
        
        "numbered_list": """
        ---Target response length and format---
        Structure as a numbered list with:
        1. Clear, descriptive headings for each item
        2. Detailed explanations under each number
        3. Logical ordering (importance, chronology, or thematic)
        4. 3-8 numbered items maximum
        """,
        
        "executive_summary": """
        ---Target response length and format---
        Format as an executive summary with:
        • Brief overview paragraph
        • Key findings (3-5 bullet points)
        • Recommendations or implications
        • Professional, concise tone
        • 200-400 words total
        """,
        
        "detailed_analysis": """
        ---Target response length and format---
        Provide a comprehensive analysis with:
        ## Main sections with clear headings
        - Detailed explanations under each section
        - Supporting evidence and examples
        - Logical flow from general to specific
        - 500-1000 words total
        """,
        
        "comparison": """
        ---Target response length and format---
        Structure as a comparison with:
        **Similarities:**
        • Shared characteristics or features
        • Common themes or patterns
        
        **Differences:**  
        • Contrasting aspects or approaches
        • Unique features or limitations
        
        **Conclusion:**
        • Overall assessment of relationships
        """,
        
        "pros_cons": """
        ---Target response length and format---
        Present as balanced analysis:
        **Advantages:**
        • Positive aspects with explanations
        • Benefits and opportunities
        
        **Disadvantages:**
        • Limitations and challenges  
        • Potential risks or drawbacks
        
        **Overall Assessment:**
        • Balanced conclusion weighing both sides
        """
    }
    
    formatting_instruction = formatting_instructions.get(
        response_type, 
        formatting_instructions["multiple_paragraphs"]  # Default
    )
    
    return f"{base_prompt}\n\n{formatting_instruction}\n\n---Data tables---\n{context_data}"
```

### Real Formatting Examples

#### Same Query, Different Formats

**Query:** "What are the main applications of machine learning?"
**Context:** ML frameworks, algorithms, applications from knowledge graph

**Single Paragraph Format:**
```
Machine learning applications span numerous domains, with the most prominent being computer vision for image recognition and autonomous vehicles, natural language processing for chatbots and translation services, recommendation systems for e-commerce and streaming platforms, predictive analytics in finance and healthcare for risk assessment and diagnosis, robotics for automation and control systems, and fraud detection in financial services. These applications leverage various algorithms including neural networks, decision trees, and ensemble methods, implemented through frameworks like TensorFlow, PyTorch, and scikit-learn, demonstrating ML's versatility across industries from technology and finance to healthcare and manufacturing.
```

**Bullet Points Format:**
```
• **Computer Vision Applications**
  - Image recognition and classification
  - Autonomous vehicle navigation
  - Medical imaging analysis
  - Quality control in manufacturing

• **Natural Language Processing**  
  - Chatbots and virtual assistants
  - Language translation services
  - Sentiment analysis for social media
  - Document summarization

• **Recommendation Systems**
  - E-commerce product suggestions
  - Streaming service content recommendations
  - Social media feed curation
  - Personalized advertising

• **Predictive Analytics**
  - Financial risk assessment
  - Healthcare diagnosis and treatment planning  
  - Weather forecasting
  - Supply chain optimization

• **Automation and Robotics**
  - Industrial process automation
  - Robotic control systems
  - Smart home devices
  - Agricultural automation
```

**Executive Summary Format:**
```
**Executive Summary: Machine Learning Applications**

Machine learning has become integral to modern technology infrastructure, driving innovation across multiple industry sectors through sophisticated data analysis and pattern recognition capabilities.

**Key Findings:**
• Computer vision applications dominate consumer technology, particularly in autonomous vehicles and medical imaging
• Natural language processing enables human-computer interaction through chatbots, translation, and content analysis
• Recommendation systems generate significant revenue for e-commerce and entertainment platforms
• Predictive analytics provides competitive advantages in finance, healthcare, and operations management
• Automation applications reduce costs and improve efficiency in manufacturing and service industries

**Strategic Implications:**
Organizations across all sectors should evaluate ML integration opportunities to maintain competitive positioning and operational efficiency.
```

**Detailed Analysis Format:**
```
## Machine Learning Applications: Comprehensive Analysis

### Computer Vision and Image Processing
Computer vision represents one of the most mature ML application areas, with widespread deployment in autonomous vehicles, medical imaging, and quality control systems. Deep learning architectures, particularly convolutional neural networks, have revolutionized image recognition accuracy, enabling applications like facial recognition, object detection, and medical diagnosis support systems.

### Natural Language Processing and Communication
NLP applications have transformed human-computer interaction through chatbots, virtual assistants, and language translation services. Recent advances in transformer architectures and large language models have significantly improved text understanding, generation, and multilingual capabilities, enabling more sophisticated conversational AI and content creation tools.

### Recommendation and Personalization Systems
Recommendation algorithms drive significant business value for e-commerce, streaming, and social media platforms. These systems analyze user behavior patterns, preferences, and contextual data to deliver personalized content and product suggestions, directly impacting user engagement and revenue generation.

### Predictive Analytics and Decision Support
Predictive modeling applications span finance, healthcare, and operations management, providing data-driven insights for risk assessment, diagnosis support, and resource optimization. These systems analyze historical data patterns to forecast future trends and identify potential issues before they occur.

### Automation and Control Systems
ML-powered automation systems optimize industrial processes, robotic control, and smart device functionality. These applications reduce operational costs, improve consistency, and enable adaptive responses to changing environmental conditions.
```

## Part 3: Context-Aware Response Optimization

### Dynamic Response Length Adjustment

Nano-graphrag adjusts response length based on available context and query complexity:

```python
def calculate_optimal_response_length(context_tokens, query_complexity, response_type):
    """
    Dynamically adjust response length based on context richness and query complexity
    """
    # Base response length by type
    base_lengths = {
        "single_paragraph": 200,
        "multiple_paragraphs": 400, 
        "bullet_points": 300,
        "numbered_list": 350,
        "executive_summary": 300,
        "detailed_analysis": 600,
        "comparison": 450,
        "pros_cons": 400
    }
    
    base_length = base_lengths.get(response_type, 400)
    
    # Adjust based on context richness
    context_multiplier = min(1.5, context_tokens / 20000)  # More context = longer responses
    
    # Adjust based on query complexity (number of entities/concepts)
    complexity_multiplier = min(1.3, query_complexity / 10)
    
    optimal_length = int(base_length * context_multiplier * complexity_multiplier)
    
    # Ensure within reasonable bounds
    return max(150, min(optimal_length, 1000))
```

### Context Quality Assessment

```python
def assess_context_quality(context_components):
    """
    Assess the quality and richness of context data to optimize response formatting
    """
    quality_metrics = {
        "entity_diversity": len(set(extract_entities_from_context(context_components))),
        "relationship_density": count_relationships(context_components) / max(1, count_entities(context_components)),
        "source_coverage": len(extract_sources(context_components)),
        "community_depth": max([c.get("level", 0) for c in extract_communities(context_components)] + [0])
    }
    
    # Calculate overall quality score
    quality_score = (
        min(quality_metrics["entity_diversity"] / 20, 1.0) * 0.3 +
        min(quality_metrics["relationship_density"] / 2, 1.0) * 0.3 +
        min(quality_metrics["source_coverage"] / 10, 1.0) * 0.2 +
        min(quality_metrics["community_depth"] / 2, 1.0) * 0.2
    )
    
    return quality_score, quality_metrics
```

### Response Quality Enhancement

```python
def enhance_response_quality(base_response, context_quality, response_type):
    """
    Post-process response to ensure quality and formatting consistency
    """
    enhancements = []
    
    # Add structure markers for bullet points
    if response_type == "bullet_points" and "•" not in base_response:
        enhancements.append("restructure_as_bullets")
    
    # Add section headers for detailed analysis
    if response_type == "detailed_analysis" and "#" not in base_response:
        enhancements.append("add_section_headers")
    
    # Add summary if context is rich but response is short
    if context_quality > 0.7 and len(base_response.split()) < 100:
        enhancements.append("expand_with_context")
    
    # Add source attribution if available
    if "sources" in context_quality and response_type in ["detailed_analysis", "executive_summary"]:
        enhancements.append("add_source_references")
    
    return apply_enhancements(base_response, enhancements)
```

## Part 4: Token Monitoring and Optimization

### Real-Time Token Monitoring

```python
class TokenMonitor:
    def __init__(self):
        self.usage_log = []
        self.total_tokens_used = 0
        self.cost_per_token = 0.00003  # GPT-4o pricing
        
    def log_llm_call(self, call_type, input_tokens, output_tokens, model_name):
        """Log each LLM interaction for monitoring and optimization"""
        call_data = {
            "timestamp": datetime.now(),
            "call_type": call_type,  # "local_query", "global_map", "global_reduce", etc.
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost": (input_tokens + output_tokens) * self.cost_per_token
        }
        
        self.usage_log.append(call_data)
        self.total_tokens_used += call_data["total_tokens"]
        
        logger.info(f"LLM Call [{call_type}]: {input_tokens} in + {output_tokens} out = {call_data['total_tokens']} tokens (~${call_data['estimated_cost']:.4f})")
        
    def get_usage_summary(self):
        """Generate usage summary for optimization"""
        if not self.usage_log:
            return "No LLM calls recorded"
            
        total_calls = len(self.usage_log)
        total_cost = sum(call["estimated_cost"] for call in self.usage_log)
        avg_tokens_per_call = self.total_tokens_used / total_calls
        
        call_type_breakdown = {}
        for call in self.usage_log:
            call_type = call["call_type"]
            if call_type not in call_type_breakdown:
                call_type_breakdown[call_type] = {"count": 0, "tokens": 0, "cost": 0}
            call_type_breakdown[call_type]["count"] += 1
            call_type_breakdown[call_type]["tokens"] += call["total_tokens"]
            call_type_breakdown[call_type]["cost"] += call["estimated_cost"]
            
        return {
            "total_calls": total_calls,
            "total_tokens": self.total_tokens_used,
            "total_cost": total_cost,
            "avg_tokens_per_call": avg_tokens_per_call,
            "breakdown_by_type": call_type_breakdown
        }
```

### Example Token Usage for Different Query Types

#### Local Query Token Usage
```python
# Example: "How does TensorFlow integrate with Keras?"
token_usage = {
    "entity_search": {
        "input_tokens": 0,    # Vector search, no LLM
        "output_tokens": 0
    },
    "context_assembly": {
        "input_tokens": 0,    # Data processing, no LLM  
        "output_tokens": 0
    },
    "response_generation": {
        "input_tokens": 15420,  # System prompt + query + context data
        "output_tokens": 387,   # Generated response
        "total": 15807,
        "cost": "$0.47"
    }
}
# Total Local Query Cost: ~$0.47
```

#### Global Query Token Usage
```python
# Example: "What are the main challenges in AI development?"
token_usage = {
    "community_selection": {
        "input_tokens": 0,    # Graph operations, no LLM
        "output_tokens": 0
    },
    "map_phase": {
        "analyst_1": {"input": 11250, "output": 543, "cost": "$0.35"},
        "analyst_2": {"input": 10890, "output": 612, "cost": "$0.34"},  
        "analyst_3": {"input": 9760, "output": 387, "cost": "$0.30"},
        "subtotal": {"input": 31900, "output": 1542, "cost": "$0.99"}
    },
    "reduce_phase": {
        "input_tokens": 8340,   # All analyst insights + synthesis prompt
        "output_tokens": 692,   # Final synthesized response
        "cost": "$0.27"
    }
}
# Total Global Query Cost: $0.99 + $0.27 = $1.26
```

#### Naive Query Token Usage
```python
# Example: "What is machine learning?"  
token_usage = {
    "chunk_search": {
        "input_tokens": 0,    # Vector search, no LLM
        "output_tokens": 0
    },
    "response_generation": {
        "input_tokens": 8934,   # System prompt + query + text chunks
        "output_tokens": 245,   # Generated response
        "total": 9179,
        "cost": "$0.28"
    }
}
# Total Naive Query Cost: ~$0.28
```

This comprehensive guide shows how nano-graphrag carefully manages context limits, optimizes token usage, and formats responses to deliver high-quality answers while maintaining cost efficiency across different query types and response formats.