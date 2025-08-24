"""Session management for knowledge graph working directories"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

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