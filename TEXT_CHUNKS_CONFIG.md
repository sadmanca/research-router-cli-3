# Text Chunks Configuration

This document explains how to configure whether text chunks are included in the query context when using the Research Router CLI.

## Overview

By default, **text chunks are DISABLED** (`include_text_chunks = False`) to test GraphRAG performance without relying on raw text retrieval. This allows you to evaluate how well the knowledge graph (entities, relationships, and community reports) performs on its own.

## Configuration Methods

### Method 1: Environment Variable (Recommended)

Set the `INCLUDE_TEXT_CHUNKS` environment variable in your `.env` file or shell:

```bash
# Enable text chunks
INCLUDE_TEXT_CHUNKS=true

# Disable text chunks (default)
INCLUDE_TEXT_CHUNKS=false
```

**In `.env` file:**
```env
# API Configuration
OPENAI_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here

# Text Chunks Configuration
INCLUDE_TEXT_CHUNKS=false
```

**In Windows CMD:**
```cmd
set INCLUDE_TEXT_CHUNKS=true
```

**In PowerShell:**
```powershell
$env:INCLUDE_TEXT_CHUNKS="true"
```

**In Linux/Mac:**
```bash
export INCLUDE_TEXT_CHUNKS=true
```

### Method 2: Command-Line Flags (Per-Query Override)

Override the default setting for individual queries:

```bash
# Force text chunks ON for this query
query "What is GraphRAG?" --text-chunks true
query "What is GraphRAG?" --chunks true

# Force text chunks OFF for this query
query "What is GraphRAG?" --text-chunks false
query "What is GraphRAG?" --chunks false

# Use default from config/env
query "What is GraphRAG?"
```

### Method 3: Check Current Configuration

View your current configuration:

```bash
config
```

This will show:
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Setting            ┃ Status   ┃ Value             ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Include Text Chunks│ ✗ Disabled│ False            │
└────────────────────┴──────────┴───────────────────┘
```

## Logging

When running queries, the system will clearly log whether text chunks are enabled:

```
Text chunks in context: DISABLED
Querying knowledge graph (local mode)...
Question: What is GraphRAG?

Text chunks: DISABLED - Excluding text chunks from query context
Using 15 entites, 3 communities, 42 relations, 0 text units
```

Or when enabled:

```
Text chunks in context: ENABLED
Querying knowledge graph (local mode)...
Question: What is GraphRAG?

Text chunks: ENABLED - Including text chunks in query context
Using 15 entites, 3 communities, 42 relations, 5 text units
```

## Query Modes

Text chunks setting applies to these query modes:

- **local**: Uses entities, relationships, communities, and (optionally) text chunks
- **global**: Uses community reports at different levels
- **naive**: Direct RAG retrieval (always uses text chunks regardless of this setting)

Note: The `naive` mode always uses text chunks as it's a traditional RAG approach, so this setting primarily affects `local` mode.

## Testing Scenarios

### Scenario 1: Pure Graph Performance
Test how well the knowledge graph alone can answer questions:
```bash
# Set in .env
INCLUDE_TEXT_CHUNKS=false

# Then query
query "Explain the main concepts" --mode=local
```

### Scenario 2: Graph + Text Chunks
Test performance with both graph structure and original text:
```bash
# Set in .env
INCLUDE_TEXT_CHUNKS=true

# Then query
query "Explain the main concepts" --mode=local
```

### Scenario 3: Compare Both
Run queries with both settings to compare:
```bash
# First without chunks
query "Explain the main concepts" --mode local --text-chunks false

# Then with chunks
query "Explain the main concepts" --mode local --text-chunks true
```

## Code References

### QueryParam in base.py
```python
@dataclass
class QueryParam:
    mode: Literal["local", "global", "naive"] = "global"
    include_text_chunks: bool = False  # Default is False
    # ... other parameters
```

### Context Building in _op.py
```python
# Conditionally retrieve text chunks based on query parameter
if query_param.include_text_chunks:
    logger.info("Text chunks: ENABLED - Including text chunks in query context")
    use_text_units = await _find_most_related_text_unit_from_entities(...)
else:
    logger.info("Text chunks: DISABLED - Excluding text chunks from query context")
    use_text_units = []
```

## Recommendations

1. **Default (False)**: Good for testing pure graph-based retrieval
2. **Enable (True)**: Use when you need fallback to original text for better recall
3. **Compare Both**: Run experiments with both settings to measure the value added by text chunks

## Troubleshooting

**Q: My queries aren't showing text chunks even with `INCLUDE_TEXT_CHUNKS=true`**  
A: Make sure to restart your CLI session after changing environment variables, or use the `--text-chunks true` flag.

**Q: How do I know if text chunks are actually being used?**  
A: Check the logs - they will clearly state "Text chunks: ENABLED" or "Text chunks: DISABLED".

**Q: Does this affect global mode?**  
A: No, global mode primarily uses community reports. Text chunks mainly affect local mode queries.

**Q: What's the difference between `--text-chunks` and `--chunks`?**  
A: They're aliases - both work the same way. Use whichever you prefer.
