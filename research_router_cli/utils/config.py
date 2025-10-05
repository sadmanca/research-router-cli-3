"""Configuration management for the CLI"""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

console = Console()

class Config:
    def __init__(self):
        # Load environment variables from .env file if it exists
        env_file = Path('.env')
        if env_file.exists():
            load_dotenv(env_file)
            
        self._openai_api_key: Optional[str] = None
        self._azure_openai_endpoint: Optional[str] = None
        self._azure_openai_api_key: Optional[str] = None
        self._gemini_api_key: Optional[str] = None
        self._include_text_chunks: bool = False  # Default to False for testing without text chunks
        self._load_config()
        
    def _load_config(self):
        """Load configuration from environment variables"""
        self._openai_api_key = os.getenv('OPENAI_API_KEY')
        self._azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self._azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self._gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Load text chunks configuration (default False)
        include_chunks_env = os.getenv('INCLUDE_TEXT_CHUNKS', 'false').lower()
        self._include_text_chunks = include_chunks_env in ('true', '1', 'yes', 'on')
        
    @property
    def openai_api_key(self) -> Optional[str]:
        return self._openai_api_key
        
    @property
    def has_openai_config(self) -> bool:
        return self._openai_api_key is not None
        
    @property
    def has_azure_openai_config(self) -> bool:
        return (self._azure_openai_endpoint is not None and 
                self._azure_openai_api_key is not None)
    
    @property
    def gemini_api_key(self) -> Optional[str]:
        return self._gemini_api_key
        
    @property
    def has_gemini_config(self) -> bool:
        return self._gemini_api_key is not None
    
    @property
    def include_text_chunks(self) -> bool:
        """Whether to include text chunks in query context"""
        return self._include_text_chunks
    
    def set_include_text_chunks(self, value: bool):
        """Set whether to include text chunks in query context"""
        self._include_text_chunks = value
                
    def show_config(self):
        """Display current configuration"""
        table = Table(title="Configuration Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Value", style="magenta")
        
        # OpenAI API Key
        if self.has_openai_config:
            masked_key = f"{self._openai_api_key[:8]}...{self._openai_api_key[-4:]}"
            table.add_row("OpenAI API Key", "✓ Set", masked_key)
        else:
            table.add_row("OpenAI API Key", "✗ Not Set", "")
            
        # Azure OpenAI
        if self.has_azure_openai_config:
            table.add_row("Azure OpenAI", "✓ Set", self._azure_openai_endpoint)
        else:
            table.add_row("Azure OpenAI", "✗ Not Set", "")
        
        # Gemini API Key
        if self.has_gemini_config:
            masked_key = f"{self._gemini_api_key[:8]}...{self._gemini_api_key[-4:]}"
            table.add_row("Gemini API Key", "✓ Set", masked_key)
        else:
            table.add_row("Gemini API Key", "✗ Not Set", "")
        
        # Text chunks configuration
        chunks_status = "✓ Enabled" if self._include_text_chunks else "✗ Disabled"
        table.add_row("Include Text Chunks", chunks_status, str(self._include_text_chunks))
            
        console.print(table)
        
        if not self.has_openai_config and not self.has_azure_openai_config and not self.has_gemini_config:
            console.print("\n[red]⚠️  No API configuration found![/red]")
            console.print("Set your API key:")
            console.print("  export OPENAI_API_KEY='your-openai-api-key-here'")
            console.print("  export GEMINI_API_KEY='your-gemini-api-key-here'")
            console.print("\nOr create a .env file with:")
            console.print("  OPENAI_API_KEY=your-openai-api-key-here")
            console.print("  GEMINI_API_KEY=your-gemini-api-key-here")
            
    def validate_config(self) -> bool:
        """Validate that required configuration is present"""
        if not self.has_openai_config and not self.has_azure_openai_config and not self.has_gemini_config:
            console.print("[red]Error: No API configuration found[/red]")
            console.print("Please set OPENAI_API_KEY or GEMINI_API_KEY environment variable or configure Azure OpenAI")
            return False
        return True
        
    def get_llm_config(self) -> dict:
        """Get configuration for nano-graphrag LLM setup"""
        config = {}
        
        # Prioritize Gemini + GenKG if available
        if self.has_gemini_config:
            config.update({
                'using_gemini': True,
                'use_genkg_extraction': True,
                'genkg_node_limit': 25,
                'genkg_create_visualization': True,
                'genkg_llm_provider': 'gemini',
                'genkg_model_name': 'gemini-2.5-flash',
            })
        elif self.has_azure_openai_config:
            config['using_azure_openai'] = True
            # Azure specific config would go here
        elif self.has_openai_config:
            config['using_azure_openai'] = False
            # OpenAI config is handled via environment variable
            
        return config