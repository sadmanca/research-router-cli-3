"""Session history management for query/response persistence"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class SessionHistoryManager:
    """Manages query/response history for research sessions"""
    
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.history_file = self.session_dir / "session_history.json"
        self.session_dir.mkdir(exist_ok=True)
        
    def add_entry(self, query: str, response: str, mode: str = "global") -> bool:
        """Add a new query/response entry to the session history"""
        try:
            # Load existing history
            history = self._load_history()
            
            # Create new entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query.strip(),
                "response": response.strip(),
                "mode": mode,
                "response_length": len(response)
            }
            
            # Add to history
            history.append(entry)
            
            # Save updated history
            return self._save_history(history)
            
        except Exception as e:
            console.print(f"[red]Error saving query history: {e}[/red]")
            return False
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get session history, optionally limited to recent entries"""
        history = self._load_history()
        
        if limit and len(history) > limit:
            return history[-limit:]
        
        return history
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Get statistics about the session history"""
        history = self._load_history()
        
        if not history:
            return {
                "total_queries": 0,
                "modes_used": [],
                "avg_response_length": 0,
                "first_query": None,
                "last_query": None
            }
        
        modes = {}
        total_response_length = 0
        
        for entry in history:
            mode = entry.get("mode", "unknown")
            modes[mode] = modes.get(mode, 0) + 1
            total_response_length += entry.get("response_length", 0)
        
        return {
            "total_queries": len(history),
            "modes_used": list(modes.keys()),
            "mode_distribution": modes,
            "avg_response_length": total_response_length // len(history) if history else 0,
            "first_query": history[0]["timestamp"] if history else None,
            "last_query": history[-1]["timestamp"] if history else None
        }
    
    def clear_history(self) -> bool:
        """Clear all session history"""
        try:
            if self.history_file.exists():
                self.history_file.unlink()
            return True
        except Exception as e:
            console.print(f"[red]Error clearing history: {e}[/red]")
            return False
    
    def display_history(self, limit: Optional[int] = None, show_responses: bool = False):
        """Display session history in a formatted way"""
        history = self.get_history(limit)
        
        if not history:
            console.print("[yellow]No query history found for this session[/yellow]")
            return
        
        # Show stats first
        stats = self.get_history_stats()
        stats_panel = Panel(
            f"Total queries: {stats['total_queries']}\n"
            f"Modes used: {', '.join(stats['modes_used'])}\n"
            f"Average response length: {stats['avg_response_length']} chars\n"
            f"Last query: {stats['last_query'][:10] if stats['last_query'] else 'N/A'}",
            title="[bold cyan]Session Statistics[/bold cyan]",
            border_style="cyan"
        )
        console.print(stats_panel)
        console.print()
        
        # Display history table
        table = Table(title=f"Query History (showing {len(history)} entries)")
        table.add_column("#", style="dim", width=4)
        table.add_column("Date", style="cyan", width=10)
        table.add_column("Mode", style="green", width=8)
        table.add_column("Query", style="white")
        if show_responses:
            table.add_column("Response Preview", style="dim")
        
        for i, entry in enumerate(history, 1):
            date = entry["timestamp"][:10]  # Just the date part
            mode = entry.get("mode", "unknown")
            query = entry["query"]
            
            # Truncate long queries for table display
            if len(query) > 60:
                query = query[:57] + "..."
            
            row = [str(i), date, mode, query]
            
            if show_responses:
                response = entry["response"]
                # Truncate response preview
                if len(response) > 80:
                    response = response[:77] + "..."
                row.append(response)
            
            table.add_row(*row)
        
        console.print(table)
    
    def display_detailed_entry(self, entry_number: int):
        """Display a detailed view of a specific history entry"""
        history = self.get_history()
        
        if not history or entry_number < 1 or entry_number > len(history):
            console.print(f"[red]Invalid entry number. Available entries: 1-{len(history)}[/red]")
            return
        
        entry = history[entry_number - 1]
        
        # Display query
        query_panel = Panel(
            entry["query"],
            title=f"[bold cyan]Query #{entry_number} ({entry.get('mode', 'unknown')} mode)[/bold cyan]",
            subtitle=f"[dim]{entry['timestamp']}[/dim]",
            border_style="cyan"
        )
        console.print(query_panel)
        console.print()
        
        # Display response
        response = entry["response"]
        
        # Try to render as markdown if it contains formatting
        if any(marker in response for marker in ['**', '*', '#', '-', '1.', '2.']):
            try:
                markdown_response = Markdown(response)
                response_panel = Panel(
                    markdown_response,
                    title="[bold green]Response[/bold green]",
                    border_style="green"
                )
            except:
                response_panel = Panel(
                    response,
                    title="[bold green]Response[/bold green]",
                    border_style="green"
                )
        else:
            response_panel = Panel(
                response,
                title="[bold green]Response[/bold green]",
                border_style="green"
            )
        
        console.print(response_panel)
    
    def search_history(self, search_term: str, limit: Optional[int] = None) -> List[tuple]:
        """Search through history entries and return matching entries with their indices"""
        history = self.get_history()
        matches = []
        
        search_term_lower = search_term.lower()
        
        for i, entry in enumerate(history):
            query_match = search_term_lower in entry["query"].lower()
            response_match = search_term_lower in entry["response"].lower()
            
            if query_match or response_match:
                matches.append((i + 1, entry))
        
        if limit and len(matches) > limit:
            matches = matches[-limit:]
        
        return matches
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from JSON file"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_history(self, history: List[Dict[str, Any]]) -> bool:
        """Save history to JSON file"""
        try:
            self.session_dir.mkdir(exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            console.print(f"[red]Error saving history: {e}[/red]")
            return False
    
    def has_history(self) -> bool:
        """Check if session has any query history"""
        return self.history_file.exists() and len(self._load_history()) > 0