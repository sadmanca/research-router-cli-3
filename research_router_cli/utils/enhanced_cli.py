"""Enhanced CLI interface with autocomplete and better UX - Windows CMD compatible"""

import os
import sys
import json
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import tempfile
import subprocess

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter, PathCompleter, Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.application import get_app
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from .colors import console, info_msg, success_msg, warning_msg, highlight_msg
from .terminal_utils import clear_screen_smart, get_terminal_manager
from .command_parser import EnhancedCommandParser


class ResearchRouterCompleter(Completer):
    """Custom completer for research router commands"""
    
    def __init__(self, parser: EnhancedCommandParser):
        self.parser = parser
        self.file_completer = PathCompleter()
        
    def get_completions(self, document, complete_event):
        """Get completions for the current input"""
        text = document.text_before_cursor
        
        # If text is empty, show main commands
        if not text.strip():
            for cmd in self.parser.commands.keys():
                yield Completion(cmd, start_position=0)
            return
            
        # Parse current input
        parts = text.split()
        
        if len(parts) == 1:
            # Complete main command
            partial_cmd = parts[0].lower()
            
            # Main commands
            for cmd in self.parser.commands.keys():
                if cmd.startswith(partial_cmd):
                    yield Completion(cmd, start_position=-len(partial_cmd))
                    
            # Aliases
            for cmd_name, cmd_def in self.parser.commands.items():
                for alias in cmd_def.aliases:
                    if alias.startswith(partial_cmd):
                        yield Completion(alias, start_position=-len(partial_cmd))
                        
        elif len(parts) == 2:
            # Complete subcommand
            cmd_name = self.parser._find_best_command_match(parts[0].lower())
            if cmd_name and cmd_name in self.parser.commands:
                cmd_def = self.parser.commands[cmd_name]
                partial_sub = parts[1].lower()
                
                for subcommand in cmd_def.subcommands:
                    if subcommand.startswith(partial_sub):
                        yield Completion(subcommand, start_position=-len(partial_sub))
                        
        # File path completion for relevant commands
        if len(parts) >= 2:
            cmd_name = self.parser._find_best_command_match(parts[0].lower())
            if cmd_name in ['insert', 'i', 'add']:
                # Use file completer for insert commands
                for completion in self.file_completer.get_completions(document, complete_event):
                    yield completion


class EnhancedCLI:
    """Enhanced CLI with better input handling, history, and autocomplete"""
    
    def __init__(self, parser: EnhancedCommandParser):
        self.parser = parser
        self.history_file = Path.home() / ".research_router_history"
        self.config_file = Path.home() / ".research_router_config.json"
        self.terminal_manager = get_terminal_manager(console)
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup prompt toolkit if available
        if PROMPT_TOOLKIT_AVAILABLE:
            self.history = FileHistory(str(self.history_file))
            self.completer = ResearchRouterCompleter(parser)
            self.auto_suggest = AutoSuggestFromHistory()
            self._setup_key_bindings()
        else:
            console.print(warning_msg(
                "Advanced features unavailable. Install prompt-toolkit for autocomplete: pip install prompt-toolkit"
            ))
            
    def _load_config(self) -> Dict[str, Any]:
        """Load CLI configuration"""
        default_config = {
            "autocomplete": True,
            "history": True,
            "suggestions": True,
            "clear_screen": True,
            "color_scheme": "default"
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception:
                pass
                
        return default_config
    
    def _save_config(self):
        """Save current configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass
    
    def _setup_key_bindings(self):
        """Setup custom key bindings"""
        self.kb = KeyBindings()
        
        @self.kb.add('c-l')  # Ctrl+L to clear screen
        def _(event):
            """Clear screen"""
            self.terminal_manager.clear_screen_complete()
            
        @self.kb.add('c-h')  # Ctrl+H for help
        def _(event):
            """Show help"""
            event.app.current_buffer.text = 'help'
            event.app.current_buffer.cursor_position = len('help')
            
    def get_input(self, 
                  prompt_text: str = "research-router", 
                  current_session: Optional[str] = None,
                  multiline: bool = False) -> str:
        """Enhanced input with autocomplete and history"""
        
        # Create prompt
        session_indicator = f"[{current_session}]" if current_session else "[no session]"
        
        if PROMPT_TOOLKIT_AVAILABLE and self.config.get("autocomplete", True):
            # Use prompt-toolkit for enhanced experience
            try:
                prompt_html = HTML(f'<ansicyan>{prompt_text}</ansicyan> <ansiblue>{session_indicator}</ansiblue> <ansibold>></ansibold> ')
                
                result = prompt(
                    prompt_html,
                    completer=self.completer if self.config.get("autocomplete") else None,
                    auto_suggest=self.auto_suggest if self.config.get("suggestions") else None,
                    history=self.history if self.config.get("history") else None,
                    complete_style=CompleteStyle.MULTI_COLUMN,
                    key_bindings=self.kb,
                    multiline=multiline,
                    wrap_lines=True
                )
                
                return result.strip()
                
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as e:
                console.print(warning_msg(f"Prompt error: {e}. Falling back to basic input."))
                return self._basic_input(prompt_text, session_indicator)
        else:
            # Fallback to basic input
            return self._basic_input(prompt_text, session_indicator)
    
    def _basic_input(self, prompt_text: str, session_indicator: str) -> str:
        """Basic input fallback"""
        try:
            console.print(f"[cyan]{prompt_text}[/cyan] [blue]{session_indicator}[/blue] [bold]>[/bold] ", end="")
            return input().strip()
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            return ""
    
    def show_welcome(self):
        """Show enhanced welcome screen"""
        if self.config.get("clear_screen", True):
            self.terminal_manager.clear_screen_complete()
            
        console.print()
        console.print(highlight_msg("🔬 Research Router CLI"))
        console.print("[dim]Enhanced terminal experience with autocomplete and smart suggestions[/dim]")
        
        if PROMPT_TOOLKIT_AVAILABLE:
            console.print("[green]✓[/green] Advanced features enabled")
            console.print("[dim]  • Tab completion for commands and paths[/dim]")
            console.print("[dim]  • Command history with arrow keys[/dim]") 
            console.print("[dim]  • Smart suggestions based on history[/dim]")
            console.print("[dim]  • Ctrl+L to clear screen, Ctrl+H for help[/dim]")
        else:
            console.print("[yellow]![/yellow] Basic mode - install prompt-toolkit for full features")
            
        console.print()
        console.print(info_msg("Type 'help' for commands or start with 'session create <name>'"))
        console.print()
    
    def show_command_help(self, command: Optional[str] = None):
        """Show enhanced command help"""
        self.terminal_manager.clear_screen_complete()
        
        if command:
            help_text = self.parser.get_contextual_help(command)
            console.print(help_text)
        else:
            console.print(highlight_msg("📋 Available Commands"))
            console.print()
            
            # Group commands by category
            categories = {
                "Session Management": ["session"],
                "Content Management": ["insert", "query", "iquery"],
                "ArXiv Integration": ["arxiv"],
                "Information": ["history", "duplicates", "status", "config"],
                "Utility": ["help", "exit"]
            }
            
            for category, commands in categories.items():
                console.print(f"[bold cyan]{category}:[/bold cyan]")
                for cmd in commands:
                    if cmd in self.parser.commands:
                        cmd_def = self.parser.commands[cmd]
                        aliases = f" ({', '.join(cmd_def.aliases)})" if cmd_def.aliases else ""
                        console.print(f"  [green]{cmd}[/green]{aliases} - {cmd_def.description}")
                console.print()
                
            console.print("[dim]Use 'help <command>' for detailed information about a specific command[/dim]")
            console.print("[dim]Tab completion is available for all commands and file paths[/dim]")
    
    def show_smart_suggestions(self, current_session: Optional[str] = None, **context):
        """Show contextual suggestions"""
        suggestions = self.parser.get_smart_suggestions(current_session, **context)
        
        if suggestions:
            console.print()
            console.print("[bold yellow]💡 Suggestions:[/bold yellow]")
            for suggestion in suggestions:
                console.print(f"  {suggestion}")
            console.print()
    
    def handle_command_error(self, parsed_command, error_msg: str):
        """Enhanced error handling with suggestions"""
        console.print(f"[red]❌ Error:[/red] {error_msg}")
        
        if parsed_command.suggestions:
            console.print()
            console.print("[yellow]💡 Did you mean:[/yellow]")
            for suggestion in parsed_command.suggestions[:3]:
                console.print(f"  • [cyan]{suggestion}[/cyan]")
            console.print()
    
    def configure(self):
        """Interactive configuration"""
        console.print(highlight_msg("⚙️ CLI Configuration"))
        console.print()
        
        # Configure autocomplete
        if PROMPT_TOOLKIT_AVAILABLE:
            current = "enabled" if self.config.get("autocomplete", True) else "disabled"
            console.print(f"Tab completion is currently [bold]{current}[/bold]")
            
            if console.input("Enable tab completion? (y/n): ").lower().startswith('n'):
                self.config["autocomplete"] = False
            else:
                self.config["autocomplete"] = True
        
        # Configure clear screen
        current = "enabled" if self.config.get("clear_screen", True) else "disabled"
        console.print(f"Automatic screen clearing is currently [bold]{current}[/bold]")
        
        if console.input("Enable automatic screen clearing? (y/n): ").lower().startswith('n'):
            self.config["clear_screen"] = False
        else:
            self.config["clear_screen"] = True
        
        # Save configuration
        self._save_config()
        console.print(success_msg("Configuration saved!"))
    
    def get_windows_cmd_info(self) -> Dict[str, str]:
        """Get Windows CMD specific information"""
        info = {
            "terminal": "Unknown",
            "version": "Unknown",
            "supports_ansi": "Unknown"
        }
        
        if sys.platform.startswith('win'):
            try:
                # Check if we're in Windows Terminal
                if 'WT_SESSION' in os.environ:
                    info["terminal"] = "Windows Terminal"
                elif 'ConEmuPID' in os.environ:
                    info["terminal"] = "ConEmu"
                else:
                    info["terminal"] = "Command Prompt"
                
                # Check Windows version
                try:
                    import platform
                    info["version"] = platform.platform()
                except:
                    pass
                    
                # Test ANSI support
                try:
                    sys.stdout.write('\033[0m')
                    sys.stdout.flush()
                    info["supports_ansi"] = "Yes"
                except:
                    info["supports_ansi"] = "No"
                    
            except Exception:
                pass
                
        return info
    
    def show_system_info(self):
        """Show system and terminal information"""
        console.print(highlight_msg("🖥️ System Information"))
        console.print()
        
        # Basic system info
        console.print(f"Platform: [cyan]{sys.platform}[/cyan]")
        console.print(f"Python: [cyan]{sys.version.split()[0]}[/cyan]")
        
        # Windows specific info
        if sys.platform.startswith('win'):
            win_info = self.get_windows_cmd_info()
            console.print(f"Terminal: [cyan]{win_info['terminal']}[/cyan]")
            console.print(f"ANSI Support: [cyan]{win_info['supports_ansi']}[/cyan]")
        
        # Feature availability
        console.print()
        console.print("[bold]Feature Support:[/bold]")
        console.print(f"Autocomplete: [{'green' if PROMPT_TOOLKIT_AVAILABLE else 'yellow'}]{'✓' if PROMPT_TOOLKIT_AVAILABLE else '⚠'}[/]")
        console.print(f"History: [{'green' if self.history_file.exists() else 'yellow'}]{'✓' if self.history_file.exists() else '⚠'}[/]")
        console.print(f"Config: [{'green' if self.config_file.exists() else 'yellow'}]{'✓' if self.config_file.exists() else '⚠'}[/]")


def create_enhanced_cli(parser: EnhancedCommandParser) -> EnhancedCLI:
    """Create enhanced CLI instance"""
    return EnhancedCLI(parser)


def install_prompt_toolkit_windows():
    """Helper to install prompt-toolkit on Windows"""
    try:
        console.print(info_msg("Installing prompt-toolkit for enhanced features..."))
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'prompt-toolkit'])
        console.print(success_msg("prompt-toolkit installed! Restart the CLI to use enhanced features."))
        return True
    except Exception as e:
        console.print(warning_msg(f"Failed to install prompt-toolkit: {e}"))
        return False