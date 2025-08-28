#!/usr/bin/env python3
"""Debug file support checking"""

from pathlib import Path
from research_router_cli.utils.file_processor import FileProcessor

# Test file processor
fp = FileProcessor()

test_files = [
    "machine_learning_basics.txt",
    "test.pdf", 
    "document.md",
    Path("machine_learning_basics.txt"),
    Path("test_docs/machine_learning_basics.txt")
]

print("Testing file processor:")
for test_file in test_files:
    print(f"  {test_file}: supported={fp.is_supported_file(test_file)}, exists={Path(test_file).exists()}")
    
print(f"\nSupported extensions: {fp.supported_extensions}")

# Test what happens with just filename
print(f"\nTesting filename-only check:")
filename = "machine_learning_basics.txt"
path = Path(filename)
print(f"  File: {filename}")
print(f"  Suffix: {path.suffix.lower()}")
print(f"  In supported: {path.suffix.lower() in fp.supported_extensions}")
print(f"  Exists: {path.exists()}")
print(f"  is_supported_file: {fp.is_supported_file(filename)}")