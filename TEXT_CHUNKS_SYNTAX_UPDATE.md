# Text Chunks Flag - Syntax Update

## Summary of Changes

The `--text-chunks` flag now works exactly like the `--mode` flag - it accepts a value directly and doesn't become part of the query text.

## Before vs After

### ❌ Old Syntax (Deprecated)
```bash
# These old flags are NO LONGER supported
query "What is GraphRAG?" --with-chunks
query "What is GraphRAG?" --no-chunks
```

**Problem**: These flags were boolean toggles, not as flexible as needed.

### ✅ New Syntax (Current)
```bash
# New flag accepts explicit values
query "What is GraphRAG?" --text-chunks true
query "What is GraphRAG?" --text-chunks false

# Short alias available
query "What is GraphRAG?" --chunks true
query "What is GraphRAG?" --chunks false
```

**Benefits**: 
- Consistent with `--mode` flag syntax
- Explicit true/false values are clearer
- Supports multiple value formats (true, 1, yes, on, etc.)
- Flag and value are not included in the query text sent to LLM

## Side-by-Side Comparison

| Old Syntax | New Syntax | Purpose |
|------------|-----------|---------|
| `--with-chunks` | `--text-chunks true` | Enable text chunks |
| `--no-chunks` | `--text-chunks false` | Disable text chunks |
| `--with_chunks` | `--chunks true` | Enable (alias) |
| `--no_chunks` | `--chunks false` | Disable (alias) |
| (none) | (none) | Use config default |

## Complete Query Examples

### Example 1: Pure Graph Query
```bash
# NEW (recommended)
query "Explain the methodology" --mode local --text-chunks false

# This is consistent with how mode works
query "Explain the methodology" --mode local
# (uses config default for text chunks)
```

### Example 2: Graph + Text Chunks
```bash
# NEW (recommended)
query "Explain the methodology" --mode local --text-chunks true

# Short form
query "Explain the methodology" --mode local --chunks true
```

### Example 3: Testing Different Configurations
```bash
# Test without text chunks
query "What are the key findings?" --mode local --text-chunks false

# Test with text chunks
query "What are the key findings?" --mode local --text-chunks true

# Use config default
query "What are the key findings?" --mode local
```

## How It Works Internally

When you run:
```bash
query "What is GraphRAG?" --mode local --text-chunks false
```

The parser:
1. Extracts `mode=local` from flags
2. Extracts `text-chunks=false` from flags
3. Passes `"What is GraphRAG?"` as the query text (flags NOT included)
4. LLM receives only the actual question, not the flags

This is exactly how `--mode` works, providing consistency.

## Accepted Values for --text-chunks

| Value | Result | Notes |
|-------|--------|-------|
| `true` | Enabled | Recommended |
| `false` | Disabled | Recommended |
| `1` | Enabled | Numeric |
| `0` | Disabled | Numeric |
| `yes` | Enabled | Alternative |
| `no` | Disabled | Alternative |
| `on` | Enabled | Alternative |
| `off` | Disabled | Alternative |

All values are case-insensitive: `TRUE`, `True`, `true` all work.

## Migration Guide

If you have scripts or documentation using the old syntax:

### Old
```bash
query "question" --with-chunks
query "question" --no-chunks
```

### New
```bash
query "question" --text-chunks true
query "question" --text-chunks false

# Or shorter
query "question" --chunks true
query "question" --chunks false
```

## Configuration Priority

The system follows this priority order:

1. **Command-line flag** (highest priority)
   ```bash
   query "..." --text-chunks true
   ```

2. **Environment variable**
   ```env
   INCLUDE_TEXT_CHUNKS=false
   ```

3. **Config default** (lowest priority)
   ```python
   Config._include_text_chunks = False
   ```

## Examples in Context

### Research Workflow
```bash
# Step 1: Insert documents
insert folder ./papers

# Step 2: Test pure graph performance
query "What are the main findings?" --mode local --text-chunks false

# Step 3: Compare with text chunks included
query "What are the main findings?" --mode local --text-chunks true

# Step 4: Use global mode for high-level overview
query "What are the main findings?" --mode global
```

### Quick Testing
```bash
# Quick test without text chunks
query "test question" --chunks false

# Quick test with text chunks  
query "test question" --chunks true
```

## Why This Change?

1. **Consistency**: Matches the `--mode` flag pattern
2. **Clarity**: Explicit `true`/`false` values are clearer than toggle flags
3. **Flexibility**: Easy to set programmatically or via scripts
4. **Correctness**: Flags don't pollute the query text sent to LLM
5. **Extensibility**: Easy to add more configuration flags in the future

## Summary

✅ Use: `--text-chunks true|false` or `--chunks true|false`  
❌ Don't use: `--with-chunks` or `--no-chunks` (deprecated)  
💡 Default: Uses `INCLUDE_TEXT_CHUNKS` from config (default: `false`)
