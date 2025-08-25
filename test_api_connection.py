#!/usr/bin/env python3
"""
Test API connection for Gemini to diagnose RetryError
"""
import asyncio
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_api_connection():
    """Test basic API connection to Gemini"""
    print("Testing API connection...")
    
    try:
        # Import the LLM function directly
        from nano_graphrag._llm import gemini_2_5_flash_complete
        from research_router_cli.utils.config import Config
        
        config = Config()
        if not config.validate_config():
            print("[ERROR] Config validation failed")
            return
            
        print("Config validated successfully")
        
        # Test a simple API call
        print("Testing simple API call...")
        
        simple_prompt = "What is 2+2?"
        
        try:
            response = await gemini_2_5_flash_complete(
                simple_prompt,
                system_prompt="You are a helpful assistant. Give a brief answer.",
                hashing_kv=None  # No caching for this test
            )
            
            print(f"[SUCCESS] API call completed!")
            print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            
        except Exception as api_error:
            print(f"[ERROR] API call failed: {api_error}")
            print(f"Error type: {type(api_error).__name__}")
            
            # Import to check specific error types
            try:
                import traceback
                print("Full traceback:")
                traceback.print_exc()
            except:
                pass
            
    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_api_connection())