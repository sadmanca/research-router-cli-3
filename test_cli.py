#!/usr/bin/env python3
"""Simple test script to verify CLI functionality"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_basic_functionality():
    """Test basic CLI functionality without user interaction"""
    
    print("Testing imports...")
    try:
        from research_router_cli.commands.session import SessionManager
        from research_router_cli.commands.insert import InsertCommand
        from research_router_cli.commands.query import QueryCommand
        from research_router_cli.commands.arxiv import ArxivCommand
        from research_router_cli.utils.config import Config
        from research_router_cli.utils.colors import console
        from research_router_cli.utils.file_tracker import FileTracker
        from research_router_cli.utils.arxiv_client import ArxivClient
        from research_router_cli.utils.pdf_processor import PDFProcessor
        print("SUCCESS: All imports successful")
    except Exception as e:
        print(f"ERROR: Import error: {e}")
        return False
    
    print("\nTesting session manager...")
    try:
        session_manager = SessionManager()
        # Try to delete test session first, ignore if it doesn't exist
        try:
            session_manager.delete_session("test_session")
        except:
            pass
            
        session_manager.create_session("test_session")
        working_dir = session_manager.get_current_working_dir()
        if working_dir and working_dir.exists():
            print(f"SUCCESS: Session created at {working_dir}")
        else:
            print("ERROR: Session creation failed")
            return False
    except Exception as e:
        print(f"ERROR: Session manager error: {e}")
        return False
        
    print("\nTesting file tracker...")
    try:
        file_tracker = FileTracker(working_dir)
        await file_tracker.init_database()
        stats = await file_tracker.get_statistics()
        print(f"SUCCESS: File tracker initialized, stats: {stats}")
    except Exception as e:
        print(f"ERROR: File tracker error: {e}")
        return False
        
    print("\nTesting PDF processor...")
    try:
        pdf_processor = PDFProcessor()
        # Test with a non-existent file to verify error handling
        result = pdf_processor.extract_text_from_pdf("nonexistent.pdf")
        if result is None:  # Expected for non-existent file
            print("SUCCESS: PDF processor correctly handles non-existent files")
        else:
            print("ERROR: PDF processor should return None for non-existent files")
    except Exception as e:
        print(f"ERROR: PDF processor error: {e}")
        return False
        
    print("\nTesting ArXiv client...")
    try:
        downloads_dir = working_dir / "downloads"
        arxiv_client = ArxivClient(downloads_dir)
        stats = await arxiv_client.get_download_stats()
        print(f"SUCCESS: ArXiv client initialized, stats: {stats}")
    except Exception as e:
        print(f"ERROR: ArXiv client error: {e}")
        return False
        
    print("\nAll basic tests passed!")
    
    # Clean up test session
    try:
        import shutil
        test_session_dir = Path("./sessions/test_session")
        if test_session_dir.exists():
            shutil.rmtree(test_session_dir)
        sessions_file = Path("./sessions/sessions.json")
        if sessions_file.exists():
            sessions_file.unlink()
        print("SUCCESS: Test cleanup completed")
    except Exception as e:
        print(f"WARNING: Cleanup warning: {e}")
        
    return True

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    if success:
        print("\nSUCCESS: All tests passed! The enhanced CLI is ready to use.")
        sys.exit(0)
    else:
        print("\nERROR: Some tests failed. Please check the errors above.")
        sys.exit(1)