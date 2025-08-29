#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to verify GenKG integration configuration.
This test checks if the integration is properly set up without making API calls.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "nano-graphrag"))

def test_genkg_import():
    """Test if GenKG can be imported from the integration"""
    try:
        from nano_graphrag._op import extract_entities_genkg
        print("[+] Successfully imported extract_entities_genkg")
        
        from nano_graphrag import GraphRAG
        print("[+] Successfully imported GraphRAG")
        
        # Test GraphRAG initialization with genkg options
        graphrag = GraphRAG(
            working_dir="./test_dir",
            use_genkg_extraction=True,
            genkg_node_limit=15,
            genkg_create_visualization=True,
            always_create_working_dir=False
        )
        
        print(f"[+] GraphRAG initialized with GenKG options:")
        print(f"  - use_genkg_extraction: {graphrag.use_genkg_extraction}")
        print(f"  - genkg_node_limit: {graphrag.genkg_node_limit}")
        print(f"  - genkg_create_visualization: {graphrag.genkg_create_visualization}")
        print(f"  - entity_extraction_func: {graphrag.entity_extraction_func.__name__}")
        
        # Check if genkg.py exists
        genkg_paths = [
            Path(__file__).parent / "nano-graphrag" / "genkg.py",
        ]
        
        genkg_found = False
        for path in genkg_paths:
            if path.exists():
                print(f"[+] Found genkg.py at: {path}")
                genkg_found = True
                break
                
        if not genkg_found:
            print("[-] genkg.py not found in expected locations")
            for path in genkg_paths:
                print(f"  Checked: {path}")
            return False
            
        # Test GenKG import directly
        try:
            sys.path.insert(0, str(path.parent))
            from nano_graphrag.genkg import GenerateKG
            print("[+] Successfully imported GenerateKG directly")
            
            genkg = GenerateKG(llm_provider="gemini", model_name="gemini-2.5-flash")
            print("[+] Successfully initialized GenerateKG instance")
            
        except Exception as e:
            print(f"[-] Failed to import/initialize GenKG: {e}")
            return False
        
        print("\n[SUCCESS] All basic integration tests passed!")
        print("The GenKG integration is properly configured.")
        print("\nTo test full functionality with API calls, ensure:")
        print("1. Activate virtual environment: .venv\\Scripts\\activate")
        print("2. Set GEMINI_API_KEY in .env file")
        print("3. Run: uv run python test_genkg_integration.py")
        
        return True
        
    except Exception as e:
        print(f"[-] Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_genkg_import()
    sys.exit(0 if success else 1)