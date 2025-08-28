#!/usr/bin/env python3
"""
Manual test script to debug individual components
"""

import os
import sys
import requests
import time
from pathlib import Path

# Set up environment
os.environ["ADMIN_PASSWORD"] = "test123"
os.environ["GEMINI_API_KEY"] = "AIzaSyB5OuuNio6py4VB9XfNW518Wfkb4erTbUE"
os.environ["SECRET_KEY"] = "test-secret-key-for-sessions"

BASE_URL = "http://localhost:8000"

print("Manual Testing - Start server first with: python app.py")
print("Then run this script in another terminal")
print()

# Test session
session = requests.Session()

# Login
print("1. Testing login...")
response = session.post(f"{BASE_URL}/api/auth/login", json={"password": "test123"})
if response.status_code == 200:
    print("[OK] Logged in successfully")
else:
    print(f"[FAIL] Login failed: {response.status_code}")
    sys.exit(1)

# Create session
print("2. Creating test session...")
session_name = f"manual-test-{int(time.time())}"
response = session.post(f"{BASE_URL}/api/sessions/", json={"name": session_name})
if response.status_code == 200:
    print(f"[OK] Session '{session_name}' created")
else:
    print(f"[FAIL] Session creation failed: {response.status_code} - {response.text}")
    sys.exit(1)

# Test single file upload
print("3. Testing file upload...")
test_file_path = Path("test_docs/machine_learning_basics.txt")
if not test_file_path.exists():
    print(f"[FAIL] Test file not found: {test_file_path}")
    sys.exit(1)

with open(test_file_path, 'rb') as f:
    files = {'files': (test_file_path.name, f, 'text/plain')}
    response = session.post(f"{BASE_URL}/api/documents/upload", files=files)

if response.status_code == 200:
    data = response.json()
    upload_id = data.get("upload_id")
    print(f"[OK] Upload started, ID: {upload_id}")
    
    # Monitor progress
    print("4. Monitoring upload...")
    for i in range(30):  # Wait up to 30 seconds
        response = session.get(f"{BASE_URL}/api/documents/upload/{upload_id}/status")
        if response.status_code == 200:
            status_data = response.json()
            status = status_data.get("status")
            progress = status_data.get("progress", 0)
            current_file = status_data.get("current_file", "")
            
            print(f"   Status: {status}, Progress: {progress}%, File: {current_file}")
            
            if status == "completed":
                print("[OK] Upload completed successfully!")
                results = status_data.get("results", [])
                for result in results:
                    print(f"   File: {result.get('filename')} - {result.get('status')}: {result.get('message')}")
                break
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                print(f"[FAIL] Upload failed: {error}")
                break
                
            time.sleep(1)
        else:
            print(f"[FAIL] Status check failed: {response.status_code}")
            break
    
    # Test query if upload succeeded
    if status == "completed":
        print("5. Testing query...")
        response = session.post(f"{BASE_URL}/api/queries/", json={
            "question": "What is machine learning?",
            "mode": "global"
        })
        
        if response.status_code == 200:
            query_data = response.json()
            if query_data.get("success"):
                answer = query_data.get("answer", "")
                print(f"[OK] Query successful! Answer length: {len(answer)} characters")
                print(f"   Preview: {answer[:200]}...")
            else:
                print("[FAIL] Query indicated failure")
        else:
            print(f"[FAIL] Query failed: {response.status_code} - {response.text}")
    
else:
    print(f"[FAIL] Upload failed: {response.status_code} - {response.text}")

# Cleanup
print("6. Cleaning up...")
response = session.delete(f"{BASE_URL}/api/sessions/{session_name}")
if response.status_code == 200:
    print("[OK] Test session deleted")
else:
    print(f"[WARN] Failed to delete session: {response.status_code}")

print("Manual test completed!")