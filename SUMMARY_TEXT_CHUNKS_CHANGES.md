# Summary of Text Chunks Configuration Changes

## Changes Made

### 1. Added `include_text_chunks` flag to QueryParam (nano-graphrag/nano_graphrag/base.py)
- Added new parameter: `include_text_chunks: bool = False`
- Default is **False** to test performance without text chunks
- Placed right after basic query parameters for visibility

### 2. Updated context building logic (nano-graphrag/nano_graphrag/_op.py)
- Modified `_build_local_query_context()` function
- Added conditional text chunk retrieval based on `query_param.include_text_chunks`
- Added clear logging:
  - "Text chunks: ENABLED - Including text chunks in query context" when True
  - "Text chunks: DISABLED - Excluding text chunks from query context" when False
- When disabled, sets `use_text_units = []` instead of retrieving chunks

### 3. Added configuration support (research_router_cli/utils/config.py)
- Added `_include_text_chunks` property (default False)
- Reads from `INCLUDE_TEXT_CHUNKS` environment variable
- Added `include_text_chunks` property getter
- Added `set_include_text_chunks()` method for programmatic control
- Updated `show_config()` to display text chunks status

### 4. Updated query command (research_router_cli/commands/query.py)
- Modified `query()` method to accept optional `include_text_chunks` parameter
- Defaults to config value when not explicitly provided
- Added logging to show chunks status before query execution
- Modified `query_with_context()` similarly
- Updated `interactive_query()` to show current chunks status

### 5. Updated main CLI (main.py)
- Modified `_handle_query_command()` to parse chunk-related flags
- Supports `--with-chunks` flag to force enable text chunks
- Supports `--no-chunks` flag to force disable text chunks
- When no flag provided, uses config default

### 6. Added documentation (TEXT_CHUNKS_CONFIG.md)
- Comprehensive guide on how to configure text chunks
- Examples for all three configuration methods
- Explains logging and query modes
- Provides testing scenarios
- Includes troubleshooting section

## Usage Examples

### Using environment variable:
```bash
# In .env file
INCLUDE_TEXT_CHUNKS=false

# Then run query
python main.py
> query "What is GraphRAG?" --mode=local
```

### Using command-line flags:
```bash
# Force enable for one query
> query "What is GraphRAG?" --text-chunks true

# Force disable for one query  
> query "What is GraphRAG?" --text-chunks false

# Shorter alias
> query "What is GraphRAG?" --chunks true
> query "What is GraphRAG?" --chunks false
```

### Check current configuration:
```bash
> config
```

## Expected Log Output

When text chunks are disabled (default):
```
Text chunks in context: DISABLED
Querying knowledge graph (local mode)...
Question: What is GraphRAG?
Text chunks: DISABLED - Excluding text chunks from query context
Using 15 entites, 3 communities, 42 relations, 0 text units
```

When text chunks are enabled:
```
Text chunks in context: ENABLED
Querying knowledge graph (local mode)...
Question: What is GraphRAG?
Text chunks: ENABLED - Including text chunks in query context
Using 15 entites, 3 communities, 42 relations, 5 text units
```

## Testing

To test the implementation:

1. **Test default behavior (chunks disabled):**
   ```bash
   python main.py
   > query "test question" --mode=local
   # Should see "Text chunks: DISABLED" in logs
   ```

2. **Test with chunks enabled via env:**
   ```bash
   # Set INCLUDE_TEXT_CHUNKS=true in .env
   python main.py
   > query "test question" --mode=local
   # Should see "Text chunks: ENABLED" in logs
   ```

3. **Test command-line overrides:**
   ```bash
   python main.py
   > query "test question" --text-chunks true
   # Should see "ENABLED" even if env is false
   > query "test question" --text-chunks false
   # Should see "DISABLED" even if env is true
   > query "test question" --chunks true
   # Should also see "ENABLED"
   
   # Test with mode combined
   > query "test question" --mode local --text-chunks false
   # Should see "Text chunks: DISABLED"
   ```

4. **Check config display:**
   ```bash
   python main.py
   > config
   # Should show current text chunks setting
   ```

## Files Modified

1. `nano-graphrag/nano_graphrag/base.py` - Added flag to QueryParam
2. `nano-graphrag/nano_graphrag/_op.py` - Conditional text chunk retrieval + logging
3. `research_router_cli/utils/config.py` - Configuration support
4. `research_router_cli/commands/query.py` - Query command integration
5. `main.py` - CLI flag parsing

## Files Created

1. `TEXT_CHUNKS_CONFIG.md` - User documentation
2. `SUMMARY_TEXT_CHUNKS_CHANGES.md` - This file

## Benefits

1. **Clear control**: Easy to toggle text chunks on/off for testing
2. **Visible logging**: Always know whether chunks are being used
3. **Flexible configuration**: Three ways to control the setting
4. **Default for testing**: Set to False by default for pure graph evaluation
5. **Per-query overrides**: Can test different settings without restarting
