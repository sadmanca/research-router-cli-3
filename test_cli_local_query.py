#!/usr/bin/env python3
"""
Test the CLI local query functionality
"""
import subprocess
import sys
import os

def test_cli_local_query():
    """Test local query via CLI interface"""
    print("Testing local query through CLI interface...")
    
    # Create a test input that simulates user commands
    test_commands = [
        "session switch 0923PM",
        "query --local what is graphrag-r1?",
        "exit"
    ]
    
    # Join commands with newlines
    input_text = "\n".join(test_commands) + "\n"
    
    try:
        # Run the CLI with the test commands
        result = subprocess.run(
            [sys.executable, "main.py"],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=60,  # 60 second timeout
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print("Exit code:", result.returncode)
        print("\n--- STDOUT ---")
        print(result.stdout)
        print("\n--- STDERR ---") 
        print(result.stderr)
        
        # Check if local query succeeded
        if "Local Search Result" in result.stdout and "'NoneType' object is not subscriptable" not in result.stderr:
            print("\n[SUCCESS] Local query works through CLI!")
        elif "'NoneType' object is not subscriptable" in result.stderr:
            print("\n[FAIL] NoneType subscriptable error still occurs")
        else:
            print("\n[UNCLEAR] Could not determine if local query worked")
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Test timed out")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_cli_local_query()