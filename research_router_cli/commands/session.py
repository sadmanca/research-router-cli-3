"""Session management for knowledge graph working directories"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

from ..utils.session_history import SessionHistoryManager

console = Console()

class SessionManager:
    def __init__(self, base_dir: str = "./sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.sessions_file = self.base_dir / "sessions.json"
        self.current_session: Optional[str] = None
        self.sessions: Dict[str, str] = self._load_sessions()
        
    def _load_sessions(self) -> Dict[str, str]:
        """Load existing sessions from file"""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                    return data.get('sessions', {})
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
        
    def _save_sessions(self):
        """Save sessions to file"""
        self.sessions_file.parent.mkdir(exist_ok=True)
        with open(self.sessions_file, 'w') as f:
            json.dump({
                'sessions': self.sessions,
                'current_session': self.current_session
            }, f, indent=2)
            
    def create_session(self, name: str) -> bool:
        """Create a new session with the given name"""
        if name in self.sessions:
            console.print(f"[yellow]Session '{name}' already exists[/yellow]")
            return False
            
        working_dir = self.base_dir / name
        working_dir.mkdir(exist_ok=True)
        
        self.sessions[name] = str(working_dir)
        self.current_session = name
        self._save_sessions()
        
        console.print(f"[green]Created session '{name}' at {working_dir}[/green]")
        console.print(f"[blue]Switched to session '{name}'[/blue]")
        return True
        
    def switch_session(self, name: str) -> bool:
        """Switch to an existing session"""
        if name not in self.sessions:
            console.print(f"[red]Session '{name}' does not exist[/red]")
            console.print("Use 'session list' to see available sessions")
            return False
            
        self.current_session = name
        self._save_sessions()
        console.print(f"[blue]Switched to session '{name}'[/blue]")
        
        # Show recent query history if available
        self._show_recent_history(name)
        return True
        
    def list_sessions(self):
        """List all available sessions"""
        if not self.sessions:
            console.print("[yellow]No sessions found. Create a session with 'session create <name>'[/yellow]")
            return
            
        table = Table(title="Research Sessions")
        table.add_column("Session Name", style="cyan")
        table.add_column("Working Directory", style="magenta")
        table.add_column("Status", style="green")
        
        for name, path in self.sessions.items():
            status = "active" if name == self.current_session else ""
            table.add_row(name, path, status)
            
        console.print(table)
        
    def get_current_working_dir(self) -> Optional[Path]:
        """Get the working directory for the current session"""
        if not self.current_session:
            return None
        return Path(self.sessions[self.current_session])
        
    def delete_session(self, name: str) -> bool:
        """Delete a session and optionally its files"""
        if name not in self.sessions:
            console.print(f"[red]Session '{name}' does not exist[/red]")
            return False
            
        working_dir = Path(self.sessions[name])
        del self.sessions[name]
        
        if self.current_session == name:
            self.current_session = None
            
        self._save_sessions()
        console.print(f"[green]Deleted session '{name}'[/green]")
        console.print(f"[yellow]Note: Files in {working_dir} were not deleted[/yellow]")
        return True
        
    def has_current_session(self) -> bool:
        """Check if there's a current active session"""
        return self.current_session is not None
        
    def ensure_session(self) -> bool:
        """Ensure there's an active session, return False if none"""
        if not self.has_current_session():
            console.print("[red]No active session. Create or switch to a session first.[/red]")
            console.print("Use: session create <name> or session switch <name>")
            return False
        return True
    
    def get_session_history_manager(self, session_name: Optional[str] = None) -> Optional[SessionHistoryManager]:
        """Get the session history manager for a session"""
        if not session_name:
            session_name = self.current_session
        
        if not session_name or session_name not in self.sessions:
            return None
            
        session_dir = Path(self.sessions[session_name])
        return SessionHistoryManager(session_dir)
    
    def _show_recent_history(self, session_name: str):
        """Show recent query history for a session with full responses"""
        try:
            history_manager = self.get_session_history_manager(session_name)
            if history_manager and history_manager.has_history():
                console.print()
                recent_history = history_manager.get_history(limit=5)  # Show more entries
                stats = history_manager.get_history_stats()
                
                console.print(f"[bold cyan]Session History ({stats['total_queries']} total queries)[/bold cyan]")
                console.print("-" * 80)
                console.print()
                
                # Display each query/response pair in full
                for i, entry in enumerate(recent_history, 1):
                    query = entry["query"]
                    response = entry["response"]
                    mode = entry.get("mode", "unknown")
                    timestamp = entry["timestamp"][:16].replace('T', ' ')  # Format: 2025-08-25 15:40
                    
                    # Display query
                    from rich.panel import Panel
                    query_panel = Panel(
                        query,
                        title=f"[bold cyan]Query #{i} ({mode} mode)[/bold cyan]",
                        subtitle=f"[dim]{timestamp}[/dim]",
                        border_style="cyan",
                        padding=(0, 1)
                    )
                    console.print(query_panel)
                    console.print()
                    
                    # Display response  
                    from rich.markdown import Markdown
                    # Try to render as markdown if it contains formatting
                    if any(marker in response for marker in ['**', '*', '#', '-', '1.', '2.']):
                        try:
                            markdown_response = Markdown(response)
                            response_panel = Panel(
                                markdown_response,
                                title="[bold green]Response[/bold green]",
                                border_style="green",
                                padding=(1, 2)
                            )
                        except:
                            response_panel = Panel(
                                response,
                                title="[bold green]Response[/bold green]",
                                border_style="green",
                                padding=(1, 2)
                            )
                    else:
                        response_panel = Panel(
                            response,
                            title="[bold green]Response[/bold green]",
                            border_style="green",
                            padding=(1, 2)
                        )
                    
                    console.print(response_panel)
                    
                    # Add separator between entries (except for the last one)
                    if i < len(recent_history):
                        console.print("\n" + "-" * 80 + "\n")
                
                console.print()
        except Exception as e:
            # Still show error for debugging but don't crash
            console.print(f"[yellow]Note: Could not display session history: {e}[/yellow]")