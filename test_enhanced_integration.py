#!/usr/bin/env python3
"""Test script for enhanced KG integration"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_enhanced_kg_import():
    """Test that EnhancedKG can be imported"""
    try:
        from nano_graphrag.enhanced_kg import EnhancedKG
        print("[OK] EnhancedKG import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] EnhancedKG import failed: {e}")
        return False

def test_enhanced_insert_command_import():
    """Test that EnhancedInsertCommand can be imported"""
    try:
        from research_router_cli.commands.enhanced_insert import EnhancedInsertCommand
        print("[OK] EnhancedInsertCommand import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] EnhancedInsertCommand import failed: {e}")
        return False

def test_main_cli_import():
    """Test that main CLI can be imported"""
    try:
        from main import ResearchRouterCLI
        print("[OK] Main CLI import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Main CLI import failed: {e}")
        return False

def test_command_parser_integration():
    """Test that enhanced-insert command is in parser"""
    try:
        from research_router_cli.utils.command_parser import EnhancedCommandParser
        parser = EnhancedCommandParser()
        
        if "enhanced-insert" in parser.commands:
            print("[OK] enhanced-insert command registered in parser")
            return True
        else:
            print("[FAIL] enhanced-insert command not found in parser")
            return False
    except Exception as e:
        print(f"[FAIL] Command parser test failed: {e}")
        return False

def test_config_gemini_support():
    """Test that config supports Gemini API key"""
    try:
        from research_router_cli.utils.config import Config
        config = Config()
        
        # Test that has_gemini_config property exists
        has_gemini = hasattr(config, 'has_gemini_config')
        gemini_key = hasattr(config, 'gemini_api_key')
        
        if has_gemini and gemini_key:
            print("[OK] Config supports Gemini API configuration")
            return True
        else:
            print("[FAIL] Config missing Gemini API support")
            return False
    except Exception as e:
        print(f"[FAIL] Config test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Testing Enhanced Knowledge Graph Integration")
    print("=" * 50)
    
    tests = [
        test_enhanced_kg_import,
        test_enhanced_insert_command_import,
        test_command_parser_integration,
        test_config_gemini_support,
        test_main_cli_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All integration tests passed!")
        print("\nEnhanced Knowledge Graph integration is ready!")
        print("\nUsage:")
        print("  enhanced-insert paper.pdf                    # Single file with enhanced generation")
        print("  enhanced-insert papers/ --nodes 30          # Folder with custom node count")
        print("  enhanced-insert browse --formats html,json  # Interactive browser with export formats")
        print("  enhanced-insert files *.pdf                 # Multiple files pattern")
    else:
        print("Some integration tests failed")
        print("Please check the errors above and install missing dependencies")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())