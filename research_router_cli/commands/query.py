"""Query command for searching knowledge graphs"""

import asyncio
from typing import Optional

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from .session import SessionManager
from ..utils.colors import console, success_msg, error_msg, warning_msg, info_msg, progress_msg

class QueryCommand:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self._graphrag_instance = None
        
    async def query(self, question: str, mode: str = "global"):
        """Query the knowledge graph with local or global mode"""
        if not self.session_manager.ensure_session():
            return
            
        if not question.strip():
            console.print("[red]Error: Question cannot be empty[/red]")
            return
            
        # Validate mode
        valid_modes = ["local", "global", "naive"]
        if mode.lower() not in valid_modes:
            console.print(f"[red]Error: Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}[/red]")
            return
            
        console.print(f"[blue]Querying knowledge graph ({mode} mode)...[/blue]")
        console.print(f"[cyan]Question: {question}[/cyan]")
        
        # Get GraphRAG instance
        graphrag = await self._get_graphrag_instance()
        if not graphrag:
            return
            
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=f"Searching knowledge graph ({mode} mode)...", total=None)
                
                # Import QueryParam
                from nano_graphrag import QueryParam
                
                # Perform query
                result = await graphrag.aquery(
                    question, 
                    param=QueryParam(mode=mode.lower())
                )
                
            # Display result
            self._display_query_result(question, result, mode)
            
        except Exception as e:
            console.print(f"[red]Error during query: {e}[/red]")
            
    def _display_query_result(self, question: str, result: str, mode: str):
        """Display the query result in a formatted way"""
        # Create title based on mode
        mode_titles = {
            "local": "🎯 Local Search Result",
            "global": "🌍 Global Search Result", 
            "naive": "📄 Naive RAG Result"
        }
        
        title = mode_titles.get(mode, f"{mode.title()} Search Result")
        
        # Display question
        question_panel = Panel(
            question,
            title="[bold cyan]Question[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )
        console.print(question_panel)
        console.print()
        
        # Display result
        if result and result.strip():
            # Try to render as markdown if it contains markdown-like formatting
            if any(marker in result for marker in ['**', '*', '#', '-', '1.', '2.']):
                try:
                    markdown_result = Markdown(result)
                    result_panel = Panel(
                        markdown_result,
                        title=f"[bold green]{title}[/bold green]",
                        border_style="green",
                        padding=(1, 2)
                    )
                except:
                    # Fallback to plain text if markdown parsing fails
                    result_panel = Panel(
                        result,
                        title=f"[bold green]{title}[/bold green]",
                        border_style="green",
                        padding=(1, 2)
                    )
            else:
                result_panel = Panel(
                    result,
                    title=f"[bold green]{title}[/bold green]",
                    border_style="green",
                    padding=(1, 2)
                )
                
            console.print(result_panel)
        else:
            console.print("[yellow]No results found for your query.[/yellow]")
            
    async def query_with_context(self, question: str, mode: str = "global") -> Optional[str]:
        """Query and return only the context (for integration with other tools)"""
        if not self.session_manager.ensure_session():
            return None
            
        graphrag = await self._get_graphrag_instance()
        if not graphrag:
            return None
            
        try:
            from nano_graphrag import QueryParam
            
            # Use only_need_context=True to get raw context
            result = await graphrag.aquery(
                question, 
                param=QueryParam(mode=mode.lower(), only_need_context=True)
            )
            
            return result
            
        except Exception as e:
            console.print(f"[red]Error during context query: {e}[/red]")
            return None
            
    async def interactive_query(self):
        """Start an interactive query session"""
        if not self.session_manager.ensure_session():
            return
            
        console.print("[blue]Starting interactive query session. Type 'exit' to return to main CLI.[/blue]")
        console.print("[dim]Available modes: local, global, naive[/dim]")
        console.print()
        
        while True:
            try:
                # Get question
                question = console.input("[bold cyan]❓ Your question: [/bold cyan]").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['exit', 'quit', 'q']:
                    console.print("[blue]Exiting interactive query mode[/blue]")
                    break
                    
                # Get mode (with default)
                mode = console.input("[dim]Mode (local/global/naive) [global]: [/dim]").strip().lower()
                if not mode:
                    mode = "global"
                    
                if mode not in ["local", "global", "naive"]:
                    console.print(f"[yellow]Invalid mode '{mode}', using 'global'[/yellow]")
                    mode = "global"
                    
                console.print()
                await self.query(question, mode)
                console.print("\n" + "─" * 80 + "\n")
                
            except KeyboardInterrupt:
                console.print("\n[blue]Exiting interactive query mode[/blue]")
                break
            except EOFError:
                console.print("\n[blue]Exiting interactive query mode[/blue]")
                break
                
    async def _get_graphrag_instance(self):
        """Get or create a GraphRAG instance for the current session"""
        working_dir = self.session_manager.get_current_working_dir()
        if not working_dir:
            console.print("[red]Error: No current session working directory[/red]")
            return None
            
        # Check if knowledge graph exists
        if not self._has_knowledge_graph(working_dir):
            console.print("[yellow]No knowledge graph found in current session.[/yellow]")
            console.print("Use 'insert <pdf_path>' to add documents first.")
            return None
            
        try:
            # Import nano-graphrag
            from nano_graphrag import GraphRAG
            from ..utils.config import Config
            
            config = Config()
            if not config.validate_config():
                return None
                
            # Create or reuse GraphRAG instance
            if (self._graphrag_instance is None or 
                str(working_dir) != getattr(self._graphrag_instance, 'working_dir', None)):
                
                llm_config = config.get_llm_config()
                
                self._graphrag_instance = GraphRAG(
                    working_dir=str(working_dir),
                    enable_llm_cache=True,
                    **llm_config
                )
                
            return self._graphrag_instance
            
        except ImportError as e:
            console.print(f"[red]Error: nano-graphrag not available: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Error initializing GraphRAG: {e}[/red]")
            return None
            
    def _has_knowledge_graph(self, working_dir) -> bool:
        """Check if a knowledge graph exists in the working directory"""
        from pathlib import Path
        
        # Check for common nano-graphrag files
        graph_files = [
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json", 
            "vdb_entities.json"
        ]
        
        return any((working_dir / filename).exists() for filename in graph_files)
        
    def reset_instance(self):
        """Reset the GraphRAG instance (useful when switching sessions)"""
        self._graphrag_instance = None