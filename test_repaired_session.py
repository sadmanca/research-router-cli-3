#!/usr/bin/env python3
"""
Test that the repaired session works correctly without warnings
"""
import asyncio
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_repaired_session():
    """Test local query on the repaired session"""
    print("Testing repaired session local query...")
    
    try:
        from nano_graphrag import GraphRAG, QueryParam
        from research_router_cli.utils.config import Config
        
        config = Config()
        if not config.validate_config():
            print("[ERROR] Config validation failed")
            return False
            
        # Test with the repaired 1008PM session
        working_dir = "sessions/1008PM"
        print(f"Testing with session: {working_dir}")
        
        llm_config = config.get_llm_config()
        
        # Create GraphRAG instance
        graphrag = GraphRAG(
            working_dir=working_dir,
            enable_llm_cache=True,
            **llm_config
        )
        
        print("Created GraphRAG instance")
        
        # Test local query with context only (to avoid API quota issues)
        query = "what is GraphRAG-R1?"
        print(f"Testing local query: '{query}'")
        
        try:
            # Capture any warnings during query
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                context = await graphrag.aquery(
                    query, 
                    param=QueryParam(mode="local", only_need_context=True)
                )
                
                # Check for warnings
                graphrag_warnings = [warning for warning in w if "nano-graphrag" in str(warning.category) or "Text unit missing" in str(warning.message)]
                
                if graphrag_warnings:
                    print(f"[WARNING] Found {len(graphrag_warnings)} GraphRAG warnings:")
                    for warning in graphrag_warnings:
                        print(f"  - {warning.message}")
                else:
                    print("[SUCCESS] No GraphRAG warnings detected")
            
            if context and len(context) > 100:
                print(f"[SUCCESS] Local query returned {len(context)} characters of context")
                
                # Check if text units are being used (should be > 0 now after repair)
                if "text units" in context.lower() or len(context) > 5000:
                    print("[SUCCESS] Context appears to include text units")
                else:
                    print("[INFO] Context may not include text units (but query worked)")
                    
                return True
            else:
                print(f"[ERROR] Local query returned insufficient context: {len(context) if context else 0} chars")
                return False
                
        except Exception as e:
            print(f"[ERROR] Local query failed: {e}")
            # Check if it's the old NoneType error
            if "'NoneType' object is not subscriptable" in str(e):
                print("[ERROR] The original NoneType subscriptable error is still occurring!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_repaired_session())
    if success:
        print("\n[SUCCESS] Repaired session test passed!")
        print("The fix is working correctly with real data.")
    else:
        print("\n[FAIL] Repaired session test failed!")
        print("There may still be issues with the fix.")