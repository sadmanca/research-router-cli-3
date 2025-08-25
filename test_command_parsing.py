#!/usr/bin/env python3
"""
Test command parsing for query with --local flag
"""
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_router_cli.utils.command_parser import EnhancedCommandParser

def test_command_parsing():
    """Test how different query commands are parsed"""
    parser = EnhancedCommandParser()
    
    test_commands = [
        "query --local what is graphrag-r1?",
        "query mode --local what is graphrag-r1?",
        "query --mode local what is graphrag-r1?",
        "q --local what is graphrag-r1?"
    ]
    
    for cmd in test_commands:
        print(f"\n--- Testing: '{cmd}' ---")
        parsed = parser.parse_command(cmd, current_session="test")
        
        print(f"Command: {parsed.command}")
        print(f"Subcommand: {parsed.subcommand}")
        print(f"Args: {parsed.args}")
        print(f"Flags: {parsed.flags}")
        print(f"Suggestions: {parsed.suggestions}")
        
        # Determine mode as the main handler would
        mode = 'global'  # default
        
        if 'mode' in parsed.flags:
            mode = parsed.flags['mode']
        elif 'local' in parsed.flags:
            mode = 'local'
        elif 'global' in parsed.flags:
            mode = 'global'  
        elif 'naive' in parsed.flags:
            mode = 'naive'
            
        query_text = " ".join(parsed.args)
        
        print(f"Determined mode: {mode}")
        print(f"Query text: '{query_text}'")

if __name__ == "__main__":
    test_command_parsing()