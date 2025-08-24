#!/usr/bin/env python3
"""
Research Router CLI - Interactive knowledge graph CLI using nano-graphrag
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.insert import InsertCommand
from research_router_cli.commands.query import QueryCommand
from research_router_cli.commands.arxiv import ArxivCommand
from research_router_cli.utils.config import Config
from research_router_cli.utils.colors import console

app = typer.Typer(name="research-router", help="Interactive knowledge graph CLI")

class CLI:
    def __init__(self):
        self.session_manager = SessionManager()
        self.insert_command = InsertCommand(self.session_manager)
        self.query_command = QueryCommand(self.session_manager)
        self.arxiv_command = ArxivCommand(self.session_manager)
        self.config = Config()
        
    def show_welcome(self):
        welcome_text = """
Research Router CLI
Interactive knowledge graph creation and querying with nano-graphrag

Available commands:
  session create <name>     - Create a new research session
  session list             - List all sessions
  session switch <name>    - Switch to a different session
  session delete <name>    - Delete a session
  
  insert <pdf_path>        - Insert single PDF into knowledge graph
  insert files <f1> <f2>   - Insert multiple files at once
  insert folder <path>     - Insert all PDFs from folder
  insert folder <path> -r  - Insert PDFs recursively
  
  arxiv search <query>     - Search ArXiv papers
  arxiv download <id>      - Download specific ArXiv paper
  arxiv history           - Show ArXiv download history
  
  query <question>         - Query the knowledge graph (global mode)
  query --mode local <q>   - Local query mode
  query --mode global <q>  - Global query mode
  query --mode naive <q>   - Naive RAG mode
  iquery                   - Interactive query mode
  
  history                  - Show file insertion history
  duplicates               - Show duplicate files
  config                   - Show configuration
  status                   - Show current session status
  help                     - Show this help
  exit                     - Exit the CLI
        """
        console.print(Panel(welcome_text, title="Welcome", border_style="blue"))
        
    async def interactive_loop(self):
        self.show_welcome()
        
        while True:
            try:
                current_session = self.session_manager.current_session
                prompt_text = f"research-router ({current_session or 'no session'})> "
                command = Prompt.ask(prompt_text).strip()
                
                if not command:
                    continue
                    
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == "exit":
                    console.print("👋 Goodbye!")
                    break
                elif cmd == "help":
                    self.show_welcome()
                elif cmd == "config":
                    self.config.show_config()
                elif cmd == "status":
                    self._show_status()
                elif cmd == "iquery":
                    await self.query_command.interactive_query()
                elif cmd == "history":
                    await self._handle_history_command()
                elif cmd == "duplicates":
                    await self._handle_duplicates_command()
                elif cmd == "session":
                    await self._handle_session_command(parts[1:])
                elif cmd == "insert":
                    await self._handle_insert_command(parts[1:])
                elif cmd == "arxiv":
                    await self._handle_arxiv_command(parts[1:])
                elif cmd == "query":
                    await self._handle_query_command(parts[1:])
                else:
                    console.print(f"[red]Unknown command: {cmd}[/red]")
                    console.print("Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                
    async def _handle_session_command(self, args):
        if not args:
            console.print("[red]Session command requires an action (create, list, switch, delete)[/red]")
            return
            
        action = args[0].lower()
        if action == "create" and len(args) > 1:
            session_name = args[1]
            self.session_manager.create_session(session_name)
            # Reset instances when switching sessions
            self.insert_command.reset_instance()
            self.query_command.reset_instance()
            self.arxiv_command.reset_instance()
        elif action == "list":
            self.session_manager.list_sessions()
        elif action == "switch" and len(args) > 1:
            session_name = args[1]
            if self.session_manager.switch_session(session_name):
                # Reset instances when switching sessions
                self.insert_command.reset_instance()
                self.query_command.reset_instance()
                self.arxiv_command.reset_instance()
        elif action == "delete" and len(args) > 1:
            session_name = args[1]
            self.session_manager.delete_session(session_name)
            # Reset instances if we deleted the current session
            self.insert_command.reset_instance()
            self.query_command.reset_instance()
            self.arxiv_command.reset_instance()
        else:
            console.print("[red]Invalid session command. Use: create <name>, list, switch <name>, or delete <name>[/red]")
            
    async def _handle_insert_command(self, args):
        if not args:
            console.print("[red]Insert command requires arguments[/red]")
            console.print("Usage: insert <pdf_path> | insert files <f1> <f2> ... | insert folder <path> [-r]")
            return
            
        action = args[0].lower()
        
        if action == "files":
            # Insert multiple files: insert files file1.pdf file2.pdf ...
            if len(args) < 2:
                console.print("[red]insert files requires at least one file path[/red]")
                return
            file_paths = args[1:]
            await self.insert_command.insert_multiple_files(file_paths)
            
        elif action == "folder":
            # Insert folder: insert folder /path/to/folder [-r]
            if len(args) < 2:
                console.print("[red]insert folder requires a folder path[/red]")
                return
            folder_path = args[1]
            recursive = len(args) > 2 and args[2] == "-r"
            await self.insert_command.insert_folder(folder_path, recursive)
            
        else:
            # Single file: insert file.pdf
            pdf_path = args[0]
            await self.insert_command.insert_pdf(pdf_path)
        
    async def _handle_query_command(self, args):
        if not args:
            console.print("[red]Query command requires a question[/red]")
            return
            
        mode = "global"  # default
        query_text = ""
        
        # Parse --mode flag
        if len(args) >= 2 and args[0] == "--mode":
            mode = args[1]
            query_text = " ".join(args[2:])
        else:
            query_text = " ".join(args)
            
        await self.query_command.query(query_text, mode)
        
    async def _handle_arxiv_command(self, args):
        if not args:
            console.print("[red]ArXiv command requires an action[/red]")
            console.print("Usage: arxiv search <query> | arxiv download <id> | arxiv history")
            return
            
        action = args[0].lower()
        
        if action == "search":
            if len(args) < 2:
                console.print("[red]arxiv search requires a search query[/red]")
                return
            query = " ".join(args[1:])
            await self.arxiv_command.search_papers(query)
            
        elif action == "download":
            if len(args) < 2:
                console.print("[red]arxiv download requires an ArXiv ID[/red]")
                return
            arxiv_id = args[1]
            await self.arxiv_command.download_paper_by_id(arxiv_id)
            
        elif action == "history":
            limit = 20
            if len(args) > 1:
                try:
                    limit = int(args[1])
                except ValueError:
                    console.print("[yellow]Invalid limit, using default (20)[/yellow]")
            await self.arxiv_command.show_arxiv_history(limit)
            
        else:
            console.print("[red]Invalid arxiv command. Use: search <query>, download <id>, or history[/red]")
            
    async def _handle_history_command(self):
        """Show file insertion history"""
        if not self.session_manager.ensure_session():
            return
            
        try:
            from research_router_cli.utils.file_tracker import FileTracker
            working_dir = self.session_manager.get_current_working_dir()
            file_tracker = FileTracker(working_dir)
            await file_tracker.init_database()
            
            history = await file_tracker.get_insertion_history(50)
            if not history:
                console.print("[yellow]No file insertion history found[/yellow]")
                return
                
            from rich.table import Table
            table = Table(title=f"File Insertion History (last {len(history)} files)")
            table.add_column("Filename", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Pages", style="blue", width=8)
            table.add_column("Size (MB)", style="magenta", width=10)
            table.add_column("Date", style="dim")
            
            for item in history:
                status_color = "green" if item["extraction_status"] == "success" else "red"
                size_mb = round(item["size_bytes"] / 1024 / 1024, 2) if item["size_bytes"] else 0
                
                table.add_row(
                    item["filename"],
                    f"[{status_color}]{item['extraction_status']}[/{status_color}]",
                    str(item["pages"] or 0),
                    f"{size_mb:.2f}",
                    item["insertion_date"][:10]
                )
                
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error retrieving history: {e}[/red]")
            
    async def _handle_duplicates_command(self):
        """Show duplicate files"""
        if not self.session_manager.ensure_session():
            return
            
        try:
            from research_router_cli.utils.file_tracker import FileTracker
            working_dir = self.session_manager.get_current_working_dir()
            file_tracker = FileTracker(working_dir)
            await file_tracker.init_database()
            
            duplicates = await file_tracker.find_duplicates()
            if not duplicates:
                console.print("[green]No duplicate files found[/green]")
                return
                
            from rich.table import Table
            table = Table(title=f"Duplicate Files ({len(duplicates)} sets)")
            table.add_column("Files", style="cyan")
            table.add_column("Count", style="red", width=8)
            
            for dup in duplicates:
                files_list = "\n".join(dup["filenames"])
                table.add_row(files_list, str(dup["count"]))
                
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error retrieving duplicates: {e}[/red]")
        
    def _show_status(self):
        """Show current session and system status"""
        from rich.table import Table
        
        table = Table(title="Research Router Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Current session
        current = self.session_manager.current_session or "None"
        table.add_row("Current Session", current)
        
        # Working directory
        if self.session_manager.current_session:
            working_dir = self.session_manager.get_current_working_dir()
            table.add_row("Working Directory", str(working_dir))
            
            # Check for knowledge graph files
            if working_dir and self._has_knowledge_graph(working_dir):
                table.add_row("Knowledge Graph", "✓ Available")
            else:
                table.add_row("Knowledge Graph", "✗ Not found")
        else:
            table.add_row("Working Directory", "No session active")
            
        # API Configuration
        if self.config.has_openai_config:
            table.add_row("OpenAI API", "✓ Configured")
        elif self.config.has_azure_openai_config:
            table.add_row("Azure OpenAI API", "✓ Configured")
        else:
            table.add_row("API Configuration", "✗ Not configured")
            
        console.print(table)
        
    def _has_knowledge_graph(self, working_dir):
        """Check if a knowledge graph exists in the working directory"""
        from pathlib import Path
        
        graph_files = [
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json", 
            "vdb_entities.json"
        ]
        
        return any((working_dir / filename).exists() for filename in graph_files)

def main():
    """Main entry point for the CLI"""
    try:
        cli = CLI()
        asyncio.run(cli.interactive_loop())
    except KeyboardInterrupt:
        console.print("\nGoodbye!")
    except ImportError as e:
        console.print(f"[red]Error: Missing dependencies: {e}[/red]")
        console.print("Please run: uv pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()