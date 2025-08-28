#!/usr/bin/env python3
"""
Simple launcher script for the Research Router Web App
"""

import os
import sys
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Import and run the web app
from web_app import app, config

def main():
    """Main function to run the web app"""
    
    print("🚀 Starting Research Router Web App")
    print("-" * 50)
    
    # Check configuration
    if not config.has_openai_config and not config.has_azure_openai_config:
        print("⚠️  WARNING: No OpenAI API configuration found!")
        print("   Please set OPENAI_API_KEY in your environment or .env file")
        print("   Example: OPENAI_API_KEY=your-api-key-here")
        print()
    else:
        print("✅ API configuration detected")
    
    # Print startup information
    print(f"📁 Working directory: {project_dir}")
    print(f"📂 Sessions directory: {project_dir / 'sessions'}")
    print()
    print("🌐 Web app will be available at:")
    print("   Local:   http://localhost:5000")
    print("   Network: http://0.0.0.0:5000")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the Flask app
        app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000,
            use_reloader=False  # Prevent double startup messages
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Research Router Web App")
    except Exception as e:
        print(f"\n❌ Error starting web app: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()