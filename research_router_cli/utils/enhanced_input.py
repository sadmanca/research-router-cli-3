"""Enhanced input handling with history, tab completion, and smart suggestions"""

import sys
from typing import List, Optional, Callable, Tuple
from pathlib import Path

# Try to import readline for better input handling (Unix/Linux/Mac)
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False
    # For Windows, we'll use basic input with manual history

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.columns import Columns

from .colors import console, info_msg, warning_msg, highlight_msg
from .command_parser import EnhancedCommandParser


class EnhancedInputHandler:
    """Enhanced input handler with history, tab completion, and suggestions"""
    
    def __init__(self, command_parser: EnhancedCommandParser):
        self.parser = command_parser
        self.history: List[str] = []
        self.history_file = Path.home() / ".research_router_history"
        self.max_history = 1000
        self._setup_readline()
        self._load_history()
        
    def _setup_readline(self):
        """Setup readline if available"""
        if not HAS_READLINE:
            return
            
        # Set up tab completion
        readline.set_completer(self._completer)
        readline.parse_and_bind('tab: complete')
        
        # Enable history navigation with up/down arrows
        readline.parse_and_bind('set editing-mode emacs')
        readline.parse_and_bind('"\e[A": previous-history')
        readline.parse_and_bind('"\e[B": next-history')
        
        # Set history length
        readline.set_history_length(self.max_history)
        
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion function"""
        if state == 0:
            # First call - generate completions
            line = readline.get_line_buffer()
            self._completion_matches = self._get_completions(line, text)
            
        # Return next completion
        if state < len(self._completion_matches):
            return self._completion_matches[state]
        return None
        
    def _get_completions(self, line: str, text: str) -> List[str]:
        """Get completions for current input"""
        completions = []
        
        # Command completions
        if ' ' not in line.strip():
            # Completing main command
            completions.extend(self.parser.get_command_completions(text))
        else:
            # Completing arguments/subcommands
            parts = line.split()
            if text:
                # Currently typing a word
                completions.extend(self.parser.get_command_completions(line))
            
            # File path completions for certain commands
            command = parts[0].lower() if parts else ""
            if command in ['insert', 'i', 'add']:
                completions.extend(self.parser.get_file_completions(text))
                
        return completions
        
    def _load_history(self):
        """Load command history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = [line.strip() for line in f.readlines()][-self.max_history:]
                    
                # Load into readline if available
                if HAS_READLINE:
                    for item in self.history:
                        readline.add_history(item)
        except Exception:
            pass  # Ignore history loading errors
            
    def _save_history(self):
        """Save command history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                # Save recent history
                recent_history = self.history[-self.max_history:]
                for item in recent_history:
                    f.write(f"{item}\n")
        except Exception:
            pass  # Ignore history saving errors
            
    def get_input(self, prompt_text: str, suggestions: Optional[List[str]] = None) -> str:
        """Get enhanced input with history and suggestions"""
        
        # Show suggestions if provided
        if suggestions:
            self._show_suggestions(suggestions)
            
        try:
            if HAS_READLINE:
                # Use readline for better input handling
                user_input = input(prompt_text).strip()
            else:
                # Fallback to basic input with Rich formatting
                user_input = console.input(f"[bold cyan]{prompt_text}[/bold cyan]").strip()
                
            # Add to history if not empty and different from last command
            if user_input and (not self.history or user_input != self.history[-1]):
                self.history.append(user_input)
                if HAS_READLINE:
                    readline.add_history(user_input)
                    
                # Save history periodically
                if len(self.history) % 10 == 0:
                    self._save_history()
                    
            return user_input
            
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C and Ctrl+D gracefully
            return ""
            
    def _show_suggestions(self, suggestions: List[str]):
        """Display contextual suggestions"""
        if not suggestions:
            return
            
        suggestion_panels = []
        for suggestion in suggestions[:3]:  # Limit to 3 suggestions
            if suggestion.startswith("💡"):
                # Smart suggestion
                suggestion_panels.append(Panel(
                    suggestion, 
                    border_style="blue",
                    padding=(0, 1),
                    title="[dim]Suggestion[/dim]"
                ))
            elif suggestion.startswith("Did you mean"):
                # Typo correction
                suggestion_panels.append(Panel(
                    suggestion,
                    border_style="yellow", 
                    padding=(0, 1),
                    title="[dim]Did you mean?[/dim]"
                ))
            else:
                # General suggestion
                suggestion_panels.append(Panel(
                    suggestion,
                    border_style="green",
                    padding=(0, 1),
                    title="[dim]Tip[/dim]"
                ))
                
        if suggestion_panels:
            console.print(Columns(suggestion_panels, equal=True, expand=True))
            console.print()
            
    def show_command_help(self, command: str, current_session: Optional[str] = None):
        """Show contextual help for a command"""
        help_text = self.parser.get_contextual_help(command, current_session)
        help_panel = Panel(
            help_text,
            title=f"[bold]Help: {command}[/bold]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(help_panel)
        
    def get_confirmation(self, message: str, default: bool = True) -> bool:
        """Get user confirmation with enhanced styling"""
        default_text = "Y/n" if default else "y/N"
        full_prompt = f"{message} [{default_text}]: "
        
        response = self.get_input(full_prompt)
        
        if not response:
            return default
        
        return response.lower().startswith('y')
    
    def show_command_history(self, limit: int = 10):
        """Show recent command history"""
        if not self.history:
            console.print(info_msg("No command history available"))
            return
            
        recent_history = self.history[-limit:]
        
        from rich.table import Table
        table = Table(title=f"Recent Commands (last {len(recent_history)})")
        table.add_column("#", style="dim", width=3)
        table.add_column("Command", style="cyan")
        table.add_column("When", style="dim")
        
        for i, cmd in enumerate(recent_history, 1):
            table.add_row(str(i), cmd, "recent")
            
        console.print(table)
        
    def cleanup(self):
        """Cleanup and save history on exit"""
        self._save_history()


class SmartPrompt:
    """Smart prompting system with context awareness"""
    
    def __init__(self, input_handler: EnhancedInputHandler):
        self.input_handler = input_handler
        
    def prompt_with_suggestions(self, 
                              base_prompt: str,
                              suggestions: Optional[List[str]] = None,
                              command_context: Optional[str] = None) -> str:
        """Prompt with contextual suggestions"""
        
        all_suggestions = suggestions or []
        
        # Add command-specific suggestions
        if command_context:
            all_suggestions.extend(
                self.input_handler.parser.get_smart_suggestions()
            )
            
        return self.input_handler.get_input(base_prompt, all_suggestions)
        
    def prompt_for_session_name(self, existing_sessions: List[str]) -> str:
        """Smart prompt for session names with validation"""
        suggestions = [
            "💡 Use descriptive names like 'ai_research' or 'literature_review'",
            "💡 Avoid spaces and special characters"
        ]
        
        while True:
            session_name = self.input_handler.get_input(
                "Session name: ", 
                suggestions
            ).strip()
            
            if not session_name:
                console.print(warning_msg("Session name cannot be empty"))
                continue
                
            if session_name in existing_sessions:
                console.print(warning_msg(f"Session '{session_name}' already exists"))
                continue
                
            # Basic validation
            if not session_name.replace('_', '').replace('-', '').isalnum():
                console.print(warning_msg("Session name should only contain letters, numbers, hyphens, and underscores"))
                continue
                
            return session_name
            
    def prompt_for_file_path(self, prompt_text: str = "File path: ") -> Optional[str]:
        """Smart prompt for file paths with validation and completion"""
        while True:
            file_path = self.input_handler.get_input(prompt_text)
            
            if not file_path:
                return None
                
            path = Path(file_path)
            
            if path.exists():
                return str(path)
            else:
                # Try to suggest similar paths
                suggestions = self.input_handler.parser.get_file_completions(file_path)
                if suggestions:
                    console.print(info_msg(f"File not found. Did you mean one of these?"))
                    for suggestion in suggestions[:5]:
                        console.print(f"  [dim]•[/dim] {suggestion}")
                else:
                    console.print(warning_msg(f"File not found: {file_path}"))
                    
                retry = self.input_handler.get_confirmation("Try again?", True)
                if not retry:
                    return None