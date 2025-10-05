# Query Command Examples - Text Chunks Configuration

## Quick Reference

The `--text-chunks` (or `--chunks`) flag works just like `--mode` - it accepts a value and doesn't become part of the query.

## Usage Examples

### Basic Query Syntax
```bash
query "<your question>" [--mode <mode>] [--text-chunks <true|false>]
```

### Using Default Configuration
```bash
# Uses INCLUDE_TEXT_CHUNKS from .env or config (default: false)
query "What is GraphRAG?"
query "Explain the main concepts" --mode local
```

### Enabling Text Chunks
```bash
# Long form
query "What is GraphRAG?" --text-chunks true

# Short form
query "What is GraphRAG?" --chunks true

# Combined with mode
query "What is GraphRAG?" --mode local --text-chunks true
query "What is GraphRAG?" --mode local --chunks true
```

### Disabling Text Chunks
```bash
# Long form
query "What is GraphRAG?" --text-chunks false

# Short form  
query "What is GraphRAG?" --chunks false

# Combined with mode
query "What is GraphRAG?" --mode local --text-chunks false
query "What is GraphRAG?" --mode local --chunks false
```

## Supported Values

Both `--text-chunks` and `--chunks` accept:
- **Enable**: `true`, `1`, `yes`, `on`
- **Disable**: `false`, `0`, `no`, `off`

Values are case-insensitive.

## Examples with Different Modes

### Local Mode (uses entities, relations, communities, and optionally text chunks)
```bash
# Without text chunks (pure graph)
query "What are the key concepts?" --mode local --text-chunks false

# With text chunks (graph + original text)
query "What are the key concepts?" --mode local --text-chunks true
```

### Global Mode (uses community reports)
```bash
# Text chunks don't significantly affect global mode
query "Summarize the main themes" --mode global
```

### Naive Mode (traditional RAG)
```bash
# Naive mode always uses text chunks regardless of this setting
query "Find specific details" --mode naive
```

## Comparison Testing

### A/B Test with Same Question
```bash
# Test 1: Pure graph
query "Explain the methodology" --mode local --text-chunks false

# Test 2: Graph + text chunks
query "Explain the methodology" --mode local --text-chunks true
```

### Testing Different Configurations
```bash
# Pure graph with local search
query "What is the main contribution?" --mode local --chunks false

# Community-based search (global)
query "What is the main contribution?" --mode global

# Traditional RAG
query "What is the main contribution?" --mode naive
```

## Common Patterns

### Default Settings (Recommended for Testing)
Set in `.env`:
```env
INCLUDE_TEXT_CHUNKS=false
```

Then query without flags:
```bash
query "Your question here" --mode local
# Uses default: text chunks disabled
```

### Override Default Per Query
```bash
# Your .env has INCLUDE_TEXT_CHUNKS=false, but you want chunks for this query
query "Your question here" --mode local --text-chunks true

# Your .env has INCLUDE_TEXT_CHUNKS=true, but you want to test without chunks
query "Your question here" --mode local --text-chunks false
```

### Interactive Query Mode
```bash
iquery
# Shows current text chunks setting
# Enter your questions
# Text chunks setting applies to all queries in the session
```

## Flag Aliases

All of these are equivalent:
```bash
--text-chunks true  ✓ (recommended)
--text_chunks true  ✓
--chunks true       ✓ (shorter)
```

## What Gets Logged

When you run a query, you'll see:
```
Text chunks in context: DISABLED
Querying knowledge graph (local mode)...
Question: What is GraphRAG?
Text chunks: DISABLED - Excluding text chunks from query context
Using 15 entites, 3 communities, 42 relations, 0 text units
```

Or with text chunks enabled:
```
Text chunks in context: ENABLED
Querying knowledge graph (local mode)...
Question: What is GraphRAG?
Text chunks: ENABLED - Including text chunks in query context
Using 15 entites, 3 communities, 42 relations, 5 text units
```

## Tips

1. **Default to False**: Start with `INCLUDE_TEXT_CHUNKS=false` in your `.env` to test pure graph performance
2. **Compare Results**: Run the same query with both settings to measure the impact
3. **Mode Matters**: Text chunks mainly affect `local` mode; `global` uses community reports
4. **Check Logs**: Always verify the "Text chunks: ENABLED/DISABLED" log message
5. **Use Aliases**: `--chunks` is shorter than `--text-chunks` for quick testing
