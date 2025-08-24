#!/usr/bin/env python3
"""
Setup script for the enhanced Research Router CLI
This script helps users migrate to the new enhanced version with improved UX
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    """Setup the enhanced CLI"""
    
    print("🚀 Research Router CLI - Enhanced Setup")
    print("="*50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: Please run this script from the research-router-cli directory")
        sys.exit(1)
    
    # Backup original main.py
    print("📦 Backing up original main.py...")
    if Path("main.py").exists():
        shutil.copy2("main.py", "main_original.py")
        print("✓ Original main.py backed up as main_original.py")
    
    # Replace main.py with enhanced version
    print("⚡ Installing enhanced main.py...")
    if Path("main_enhanced.py").exists():
        shutil.copy2("main_enhanced.py", "main.py")
        print("✓ Enhanced main.py installed")
    else:
        print("❌ Error: main_enhanced.py not found")
        sys.exit(1)
    
    # Check for .env file
    print("🔑 Checking configuration...")
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("⚠️  No .env file found. Creating from .env.example...")
            shutil.copy2(".env.example", ".env")
            print("✓ .env file created. Please edit it with your API key.")
        else:
            print("⚠️  No .env file found. Please create one with your OpenAI API key:")
            print("   OPENAI_API_KEY=your-key-here")
    else:
        print("✓ .env file exists")
    
    # Install enhanced dependencies
    print("📦 Installing enhanced dependencies...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Dependencies installed successfully")
        else:
            print("⚠️  Warning: Some dependencies may not have installed correctly")
            print("   You can manually run: pip install -r requirements.txt")
    except Exception as e:
        print(f"⚠️  Could not install dependencies automatically: {e}")
        print("   Please manually run: pip install -r requirements.txt")
    
    print("\n🎉 Enhanced CLI Setup Complete!")
    print("\n🚀 New Features:")
    print("  • Tab completion for commands and file paths")
    print("  • Command history with arrow key navigation")
    print("  • Fuzzy command matching (e.g., 'quer' → 'query')")
    print("  • Interactive file browser (insert browse)")
    print("  • Enhanced ArXiv search wizard (arxiv wizard)")
    print("  • Smart contextual suggestions")
    print("  • Improved error messages and help")
    print("  • Command aliases (q=query, s=session, i=insert, etc.)")
    
    print("\n📖 Getting Started:")
    print("  1. Run: python main.py")
    print("  2. Create your first session: session create my_research")
    print("  3. Add some PDFs: insert browse")
    print("  4. Query your knowledge: query 'your question'")
    
    print("\n💡 Pro Tips:")
    print("  • Use 'help <command>' for contextual help")
    print("  • Try 'arxiv wizard' for guided paper search")
    print("  • Use 'status' to see your current session state")
    print("  • All your existing sessions and data are preserved!")
    
    print("\n🔄 To revert to the original version:")
    print("  mv main_original.py main.py")

if __name__ == "__main__":
    main()