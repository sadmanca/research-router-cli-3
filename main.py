#!/usr/bin/env python3
"""
Research Router CLI - Interactive knowledge graph CLI using nano-graphrag
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.insert import InsertCommand
from research_router_cli.commands.enhanced_insert import EnhancedInsertCommand
from research_router_cli.commands.query import QueryCommand
from research_router_cli.commands.arxiv import ArxivCommand
from research_router_cli.utils.config import Config
from research_router_cli.utils.colors import console, info_msg, warning_msg, error_msg, success_msg, highlight_msg
from research_router_cli.utils.command_parser import EnhancedCommandParser, ParsedCommand
from research_router_cli.utils.file_browser import FileBrowser
from research_router_cli.utils.arxiv_enhanced import EnhancedArxivClient, create_search_wizard

app = typer.Typer(name="research-router", help="Interactive knowledge graph CLI")


class ResearchRouterCLI:
    """Main CLI application"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.insert_command = InsertCommand(self.session_manager)
        self.enhanced_insert_command = EnhancedInsertCommand(self.session_manager)
        self.query_command = QueryCommand(self.session_manager)
        self.arxiv_command = ArxivCommand(self.session_manager)
        self.config = Config()
        
        # Command parsing
        self.command_parser = EnhancedCommandParser()
        
        # File browser
        self.file_browser = FileBrowser()
        
        # First-time setup flag
        self.first_run = not Path("./sessions").exists() or not any(Path("./sessions").iterdir())
    
    def show_welcome(self):
        """Show welcome screen"""
        console.print()
        console.print(highlight_msg("Research Router CLI"))
        console.print("[dim]Interactive knowledge graph CLI for research[/dim]")
        console.print()
        console.print(info_msg("Type 'help' for commands or start with 'session create <name>'"))
        console.print()
        
        # Add specific research router information
        if self.first_run:
            console.print(info_msg("First time setup detected. Create a session to get started!"))
    
    def get_input(self, current_session: Optional[str] = None) -> str:
        """Get basic input without autocomplete"""
        session_indicator = f"[{current_session}]" if current_session else "[no session]"
        console.print(f"[cyan]research-router[/cyan] [blue]{session_indicator}[/blue] [bold]>[/bold] ", end="")
        return input().strip()
    
    def run_interactive_mode(self):
        """Run the main interactive loop"""
        self.show_welcome()
        
        # Show first-time setup wizard if needed
        if self.first_run:
            self._show_setup_wizard()
        
        while True:
            try:
                current_session = self.session_manager.current_session
                has_knowledge_graph = False
                
                if current_session:
                    working_dir = self.session_manager.get_current_working_dir()
                    has_knowledge_graph = self._has_knowledge_graph(working_dir)
                
                # Get basic input
                command_input = self.get_input(current_session)
                
                if not command_input:
                    continue
                
                # Parse command
                parsed_cmd = self.command_parser.parse_command(command_input, current_session)
                
                # Handle suggestions and errors
                if parsed_cmd.suggestions:
                    for suggestion in parsed_cmd.suggestions:
                        if suggestion.startswith("Did you mean"):
                            console.print(warning_msg(suggestion))
                        elif "requires" in suggestion or "Usage:" in suggestion:
                            console.print(error_msg(suggestion))
                        else:
                            console.print(info_msg(suggestion))
                    
                    if not parsed_cmd.command or parsed_cmd.command not in self.command_parser.commands:
                        continue
                
                # Execute command
                should_continue = self._execute_command_sync(parsed_cmd)
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(error_msg(f"Unexpected error: {e}"))
                console.print(info_msg("Type 'help' for available commands."))
    
    def _execute_command_sync(self, parsed_cmd: ParsedCommand) -> bool:
        """Execute a parsed command synchronously. Returns False to exit loop."""
        cmd = parsed_cmd.command
        
        if cmd == "exit":
            console.print(success_msg("Goodbye!"))
            return False
        elif cmd == "help":
            self._handle_help_command(parsed_cmd.args)
        elif cmd == "config":
            self.config.show_config()
        elif cmd == "status":
            self._show_status()
        elif cmd in ["iquery", "history", "duplicates", "session", "insert", "enhanced-insert", "arxiv", "query"]:
            # Run async commands using asyncio.run
            return asyncio.run(self._execute_async_command(parsed_cmd))
        else:
            console.print(error_msg(f"Unknown command: {cmd}"))
            console.print(info_msg("Type 'help' for available commands."))
        
        return True
    
    async def _execute_async_command(self, parsed_cmd: ParsedCommand) -> bool:
        """Execute async commands"""
        cmd = parsed_cmd.command
        
        if cmd == "iquery":
            await self.query_command.interactive_query()
        elif cmd == "history":
            await self._handle_history_command(parsed_cmd.args)
        elif cmd == "duplicates":
            await self._handle_duplicates_command()
        elif cmd == "session":
            await self._handle_session_command(parsed_cmd)
        elif cmd == "insert":
            await self._handle_insert_command(parsed_cmd)
        elif cmd == "enhanced-insert":
            await self._handle_enhanced_insert_command(parsed_cmd)
        elif cmd == "arxiv":
            await self._handle_arxiv_command(parsed_cmd)
        elif cmd == "query":
            await self._handle_query_command(parsed_cmd)
        
        return True
    
    def _handle_help_command(self, args):
        """Handle help command"""
        if args and args[0] in self.command_parser.commands:
            # Show help for specific command
            command = args[0]
            help_text = self.command_parser.get_contextual_help(command)
            console.print(help_text)
        else:
            # Show general help
            self._show_basic_help()
    
    def _show_basic_help(self):
        """Show help screen"""
        console.print(highlight_msg("📋 Available Commands"))
        console.print()
        
        # Group commands by category
        categories = {
            "Session Management": ["session"],
            "Content Management": ["insert", "enhanced-insert", "query", "iquery"],
            "ArXiv Integration": ["arxiv"],
            "Information": ["history", "duplicates", "status", "config"],
            "Utility": ["help", "exit"]
        }
        
        for category, commands in categories.items():
            console.print(f"[bold cyan]{category}:[/bold cyan]")
            for cmd in commands:
                if cmd in self.command_parser.commands:
                    cmd_def = self.command_parser.commands[cmd]
                    aliases = f" ({', '.join(cmd_def.aliases)})" if cmd_def.aliases else ""
                    console.print(f"  [green]{cmd}[/green]{aliases} - {cmd_def.description}")
            console.print()
        
        console.print("[dim]Use 'help <command>' for detailed information about a specific command[/dim]")
    
    def _show_setup_wizard(self):
        """Show first-time setup wizard"""
        console.print(Panel(
            "[bold cyan]Welcome to Research Router CLI![/bold cyan]\n\n"
            "This appears to be your first time using the CLI.\n"
            "Let's get you started with a quick setup.",
            title="First Time Setup",
            border_style="green"
        ))
        
        # Check API configuration
        if not self.config.has_openai_config and not self.config.has_azure_openai_config:
            console.print(warning_msg("No API configuration found."))
            console.print(info_msg("Please set up your OpenAI API key in a .env file:"))
            console.print("[dim]OPENAI_API_KEY=your-key-here[/dim]")
            console.input("\nPress Enter when ready...")
        
        # Offer to create first session
        if console.input("Would you like to create your first research session? (y/n): ").lower().startswith('y'):
            session_name = console.input("Enter session name: ").strip()
            if session_name:
                self.session_manager.create_session(session_name)
                
        console.print(success_msg("Setup complete! Type 'help' to see available commands."))
    
    def _show_status(self):
        """Show status"""
        # Main status table
        table = Table(title="Research Router Status", show_header=False)
        table.add_column("Setting", style="cyan", width=20)
        table.add_column("Value", style="magenta")
        
        # Current session
        current = self.session_manager.current_session or "None"
        if current != "None":
            table.add_row("Current Session", f"[green]{current}[/green]")
        else:
            table.add_row("Current Session", "[red]None (create one with 'session create <name>')[/red]")
        
        # Working directory and knowledge graph
        if self.session_manager.current_session:
            working_dir = self.session_manager.get_current_working_dir()
            table.add_row("Working Directory", str(working_dir))
            
            if working_dir and self._has_knowledge_graph(working_dir):
                table.add_row("Knowledge Graph", "[green]✓ Available[/green]")
                # Add graph stats
                stats = self._get_knowledge_graph_stats(working_dir)
                if stats:
                    table.add_row("Graph Statistics", stats)
            else:
                table.add_row("Knowledge Graph", "[yellow]✗ Not found (use 'insert' to add documents)[/yellow]")
        
        # API Configuration
        if self.config.has_openai_config:
            table.add_row("OpenAI API", "[green]✓ Configured[/green]")
        elif self.config.has_azure_openai_config:
            table.add_row("Azure OpenAI API", "[green]✓ Configured[/green]")
        else:
            table.add_row("API Configuration", "[red]✗ Not configured (set OPENAI_API_KEY)[/red]")
        
        # Session count
        session_count = len(self.session_manager.sessions)
        table.add_row("Total Sessions", str(session_count))
        
        console.print(table)
        
        # Smart suggestions based on current state
        try:
            suggestions = self.command_parser.get_smart_suggestions(
                self.session_manager.current_session,
                self._has_knowledge_graph(self.session_manager.get_current_working_dir()) if self.session_manager.current_session else False
            )
            
            if suggestions:
                console.print("\n" + info_msg("Suggestions:"))
                for suggestion in suggestions[:3]:
                    console.print(f"  {suggestion}")
        except Exception as e:
            pass
    
    def _get_knowledge_graph_stats(self, working_dir) -> Optional[str]:
        """Get knowledge graph statistics"""
        try:
            stats = []
            
            # Count documents
            docs_file = working_dir / "kv_store_full_docs.json"
            if docs_file.exists():
                with open(docs_file, 'r', encoding='utf-8') as f:
                    docs = json.load(f)
                    stats.append(f"{len(docs)} documents")
            
            # Count text chunks  
            chunks_file = working_dir / "kv_store_text_chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    stats.append(f"{len(chunks)} chunks")
            
            return ", ".join(stats) if stats else None
            
        except Exception:
            return None
    
    async def _handle_session_command(self, parsed_cmd: ParsedCommand):
        """Handle session commands"""
        if not parsed_cmd.subcommand:
            console.print(error_msg("Session command requires an action (create, list, switch, delete)"))
            return
            
        action = parsed_cmd.subcommand
        
        if action == "create":
            if parsed_cmd.args:
                session_name = parsed_cmd.args[0]
            else:
                session_name = console.input("Enter session name: ").strip()
                if not session_name:
                    return
                    
            self.session_manager.create_session(session_name)
            # Reset instances when switching sessions
            self.insert_command.reset_instance()
            self.query_command.reset_instance()
            self.arxiv_command.reset_instance()
            
        elif action == "list":
            self.session_manager.list_sessions()
        elif action == "switch" and parsed_cmd.args:
            session_name = parsed_cmd.args[0]
            if self.session_manager.switch_session(session_name):
                # Reset instances when switching sessions
                self.insert_command.reset_instance()
                self.enhanced_insert_command.reset_instance()
                self.query_command.reset_instance()
                self.arxiv_command.reset_instance()
        elif action == "delete" and parsed_cmd.args:
            session_name = parsed_cmd.args[0]
            self.session_manager.delete_session(session_name)
            # Reset instances if we deleted the current session
            self.insert_command.reset_instance()
            self.query_command.reset_instance()
            self.arxiv_command.reset_instance()
        else:
            console.print(error_msg("Invalid session command. Use: create <name>, list, switch <name>, or delete <name>"))
    
    async def _handle_insert_command(self, parsed_cmd: ParsedCommand):
        """Handle insert commands with file browser support"""
        if not parsed_cmd.subcommand and not parsed_cmd.args:
            console.print(error_msg("Insert command requires arguments"))
            console.print(info_msg("Usage: insert <pdf_path> | insert files <f1> <f2> ... | insert folder <path> [-r] | insert browse"))
            return
            
        if parsed_cmd.subcommand == "browse" or (parsed_cmd.args and parsed_cmd.args[0] == "browse"):
            # Interactive file browser
            console.print(info_msg("🗂️ Opening interactive file browser..."))
            selected_files = self.file_browser.browse_for_files(
                file_filter="*.pdf", 
                multi_select=True
            )
            
            if selected_files:
                console.print(success_msg(f"Selected {len(selected_files)} files"))
                file_paths = [str(f) for f in selected_files]
                await self.insert_command.insert_multiple_files(file_paths)
            else:
                console.print(info_msg("No files selected"))
            return
            
        elif parsed_cmd.subcommand == "files":
            # Insert multiple files: insert files file1.pdf file2.pdf ...
            if not parsed_cmd.args:
                console.print(error_msg("insert files requires at least one file path"))
                return
            await self.insert_command.insert_multiple_files(parsed_cmd.args)
            
        elif parsed_cmd.subcommand == "folder":
            # Insert folder: insert folder /path/to/folder [-r]
            if not parsed_cmd.args:
                console.print(error_msg("insert folder requires a folder path"))
                return
            folder_path = parsed_cmd.args[0]
            recursive = parsed_cmd.flags.get('r', False) or parsed_cmd.flags.get('recursive', False)
            await self.insert_command.insert_folder(folder_path, recursive)
            
        else:
            # Single file: insert file.pdf
            if parsed_cmd.args:
                pdf_path = parsed_cmd.args[0]
                await self.insert_command.insert_pdf(pdf_path)
            else:
                console.print(error_msg("Please specify a file path or use 'insert browse'"))
    
    async def _handle_enhanced_insert_command(self, parsed_cmd: ParsedCommand):
        """Handle enhanced insert commands with advanced knowledge graph generation"""
        # If no subcommand and no args, show options
        if not parsed_cmd.subcommand and not parsed_cmd.args:
            await self.enhanced_insert_command.show_enhanced_options()
            return
            
        # Extract parameters
        nodes_per_paper = 25
        export_formats = ['html', 'json']
        
        # Parse flags for nodes per paper
        if parsed_cmd.flags.get('nodes'):
            try:
                nodes_per_paper = int(parsed_cmd.flags['nodes'])
            except ValueError:
                console.print(warning_msg("Invalid nodes value, using default 25"))
                
        # Parse export formats
        if parsed_cmd.flags.get('formats'):
            export_formats = parsed_cmd.flags['formats'].split(',')
            
        if parsed_cmd.subcommand == "browse" or (parsed_cmd.args and parsed_cmd.args[0] == "browse"):
            # Interactive file browser for enhanced insertion
            console.print(info_msg("🗂️ Opening interactive file browser for enhanced insertion..."))
            try:
                selected_files = self.file_browser.browse_for_files()
            except Exception as e:
                console.print(error_msg(f"File browser error: {e}"))
                return
                
            if selected_files:
                console.print(success_msg(f"Selected {len(selected_files)} files"))
                file_paths = [str(f) for f in selected_files]
                await self.enhanced_insert_command.enhanced_insert_multiple_files(file_paths, nodes_per_paper, export_formats)
            else:
                console.print(info_msg("No files selected"))
            return
            
        elif parsed_cmd.subcommand == "files":
            # Enhanced insert multiple files
            if not parsed_cmd.args:
                console.print(error_msg("enhanced-insert files requires at least one file path"))
                return
            await self.enhanced_insert_command.enhanced_insert_multiple_files(parsed_cmd.args, nodes_per_paper, export_formats)
            
        elif parsed_cmd.subcommand == "folder":
            # Enhanced insert folder (treat as multiple files)
            if not parsed_cmd.args:
                console.print(error_msg("enhanced-insert folder requires a folder path"))
                return
            folder_path = parsed_cmd.args[0]
            from pathlib import Path
            folder = Path(folder_path)
            if not folder.exists() or not folder.is_dir():
                console.print(error_msg(f"Folder not found: {folder_path}"))
                return
            
            # Get all PDF files in the folder
            pdf_files = list(folder.glob("*.pdf"))
            if not pdf_files:
                console.print(warning_msg(f"No PDF files found in {folder_path}"))
                return
                
            console.print(info_msg(f"Found {len(pdf_files)} PDF files in folder"))
            await self.enhanced_insert_command.enhanced_insert_multiple_files(pdf_files, nodes_per_paper, export_formats)
            
        else:
            # Handle direct file paths without subcommand
            if parsed_cmd.args:
                # Check if it looks like a file path (not a subcommand that wasn't recognized)
                first_arg = parsed_cmd.args[0]
                if any(first_arg.endswith(ext) for ext in ['.pdf']) or '/' in first_arg or '\\' in first_arg:
                    # Single file enhanced insertion
                    await self.enhanced_insert_command.enhanced_insert_pdf(first_arg, nodes_per_paper, export_formats)
                else:
                    # Multiple files or pattern
                    await self.enhanced_insert_command.enhanced_insert_multiple_files(parsed_cmd.args, nodes_per_paper, export_formats)
            else:
                console.print(error_msg("Please specify a file path or use 'enhanced-insert browse'"))
                console.print(info_msg("Usage: enhanced-insert <pdf_path> [--nodes 30] [--formats html,json]"))
                console.print(info_msg("       enhanced-insert browse"))
                console.print(info_msg("       enhanced-insert files <f1> <f2> ..."))
                console.print(info_msg("       enhanced-insert folder <path>"))
    
    async def _handle_query_command(self, parsed_cmd: ParsedCommand):
        """Handle query commands"""
        if not parsed_cmd.args:
            console.print(error_msg("Query command requires a question"))
            return
        
        # Determine mode from flags
        mode = 'global'  # default
        
        if 'mode' in parsed_cmd.flags:
            mode = parsed_cmd.flags['mode']
        elif 'local' in parsed_cmd.flags:
            mode = 'local'
        elif 'global' in parsed_cmd.flags:
            mode = 'global'  
        elif 'naive' in parsed_cmd.flags:
            mode = 'naive'
        
        # Determine text chunks setting from flags
        include_text_chunks = None  # None means use config default
        if 'text-chunks' in parsed_cmd.flags or 'text_chunks' in parsed_cmd.flags:
            # Handle --text-chunks true/false
            value = parsed_cmd.flags.get('text-chunks') or parsed_cmd.flags.get('text_chunks')
            if isinstance(value, str):
                include_text_chunks = value.lower() in ('true', '1', 'yes', 'on')
            else:
                include_text_chunks = bool(value)
        elif 'chunks' in parsed_cmd.flags:
            # Handle --chunks true/false
            value = parsed_cmd.flags['chunks']
            if isinstance(value, str):
                include_text_chunks = value.lower() in ('true', '1', 'yes', 'on')
            else:
                include_text_chunks = bool(value)
            
        query_text = " ".join(parsed_cmd.args)
            
        await self.query_command.query(query_text, mode, include_text_chunks)
    
    async def _handle_arxiv_command(self, parsed_cmd: ParsedCommand):
        """Handle ArXiv commands with wizard support"""
        if not parsed_cmd.subcommand and not parsed_cmd.args:
            console.print(error_msg("ArXiv command requires an action"))
            console.print(info_msg("Usage: arxiv search <query> | arxiv wizard | arxiv download <id> | arxiv history"))
            return
            
        action = parsed_cmd.subcommand or (parsed_cmd.args[0] if parsed_cmd.args else "")
        
        if action == "wizard":
            # Interactive search wizard
            console.print(info_msg("🧙 Starting ArXiv search wizard..."))
            search_config = create_search_wizard()
            
            if search_config and console.input("Execute this search? (y/n): ").lower().startswith('y'):
                # Create enhanced client and perform search
                working_dir = self.session_manager.get_current_working_dir()
                if working_dir:
                    downloads_dir = working_dir / "downloads"
                    enhanced_arxiv = EnhancedArxivClient(downloads_dir)
                    
                    papers = await enhanced_arxiv.smart_search(search_config['query'])
                    if papers:
                        enhanced_arxiv.display_search_results(papers, show_details=True)
                        
                        # Offer interactive selection
                        if console.input("Select papers for download? (y/n): ").lower().startswith('y'):
                            selected = await enhanced_arxiv.interactive_selection(papers)
                            if selected:
                                downloaded = await enhanced_arxiv.bulk_download_with_progress(selected)
                                enhanced_arxiv.show_download_summary(downloaded, len(selected))
                                
        elif action == "search":
            if not parsed_cmd.args:
                console.print(error_msg("arxiv search requires a search query"))
                return
            query = " ".join(parsed_cmd.args)
            
            # Use enhanced search - works even without a session
            working_dir = self.session_manager.get_current_working_dir()
            downloads_dir = working_dir / "downloads" if working_dir else Path("./temp_downloads")
            
            enhanced_arxiv = EnhancedArxivClient(downloads_dir)
            
            papers = await enhanced_arxiv.smart_search(query)
            if papers:
                enhanced_arxiv.display_search_results(papers)
            else:
                console.print(warning_msg("No papers found for your search query"))
                    
        elif action == "download":
            if not parsed_cmd.args:
                console.print(error_msg("arxiv download requires an ArXiv ID"))
                return
            arxiv_id = parsed_cmd.args[0]
            await self.arxiv_command.download_paper_by_id(arxiv_id)
            
        elif action == "history":
            limit = 20
            if parsed_cmd.args:
                try:
                    limit = int(parsed_cmd.args[0])
                except ValueError:
                    console.print(warning_msg("Invalid limit, using default (20)"))
            await self.arxiv_command.show_arxiv_history(limit)
            
        else:
            console.print(error_msg("Invalid arxiv command. Use: search <query>, wizard, download <id>, or history"))
    
    async def _handle_history_command(self, args):
        """Show file insertion history"""
        if not self.session_manager.ensure_session():
            return
            
        try:
            from research_router_cli.utils.file_tracker import FileTracker
            working_dir = self.session_manager.get_current_working_dir()
            file_tracker = FileTracker(working_dir)
            await file_tracker.init_database()
            
            limit = 50
            if args and args[0].isdigit():
                limit = int(args[0])
            
            history = await file_tracker.get_insertion_history(limit)
            if not history:
                console.print(warning_msg("No file insertion history found"))
                return
                
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
            console.print(error_msg(f"Error retrieving history: {e}"))
    
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
                console.print(success_msg("No duplicate files found"))
                return
                
            table = Table(title=f"Duplicate Files ({len(duplicates)} sets)")
            table.add_column("Files", style="cyan")
            table.add_column("Count", style="red", width=8)
            
            for dup in duplicates:
                files_list = "\n".join(dup["filenames"])
                table.add_row(files_list, str(dup["count"]))
                
            console.print(table)
            
        except Exception as e:
            console.print(error_msg(f"Error retrieving duplicates: {e}"))
    
    def _has_knowledge_graph(self, working_dir):
        """Check if a knowledge graph exists in the working directory"""
        if not working_dir:
            return False
            
        graph_files = [
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json", 
            "vdb_entities.json"
        ]
        
        return any((working_dir / filename).exists() for filename in graph_files)


def main():
    """Main entry point for the CLI"""
    try:
        cli = ResearchRouterCLI()
        cli.run_interactive_mode()
    except KeyboardInterrupt:
        console.print("\nGoodbye!")
    except ImportError as e:
        console.print(error_msg(f"Missing dependencies: {e}"))
        console.print(info_msg("Please run: uv pip install -r requirements.txt"))
        sys.exit(1)
    except Exception as e:
        console.print(error_msg(f"Unexpected error: {e}"))
        sys.exit(1)


if __name__ == "__main__":
    main()