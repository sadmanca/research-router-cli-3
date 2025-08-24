"""Enhanced command parser with fuzzy matching, autocompletion, and validation"""

import shlex
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import glob

from .colors import console, warning_msg, error_msg, info_msg


@dataclass
class ParsedCommand:
    """Represents a parsed command with metadata"""
    command: str
    subcommand: Optional[str] = None
    args: List[str] = None
    flags: Dict[str, Any] = None
    original_input: str = ""
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.flags is None:
            self.flags = {}
        if self.suggestions is None:
            self.suggestions = []


@dataclass 
class CommandDef:
    """Definition of a command with metadata"""
    name: str
    aliases: List[str] = None
    subcommands: List[str] = None
    description: str = ""
    usage: str = ""
    requires_session: bool = False
    min_args: int = 0
    max_args: int = -1  # -1 means unlimited
    valid_flags: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.subcommands is None:
            self.subcommands = []
        if self.valid_flags is None:
            self.valid_flags = []


class EnhancedCommandParser:
    """Enhanced command parser with fuzzy matching and smart suggestions"""
    
    def __init__(self):
        self.commands: Dict[str, CommandDef] = {}
        self.command_history: List[str] = []
        self.max_history = 100
        self._setup_commands()
        
    def _setup_commands(self):
        """Setup command definitions"""
        self.commands = {
            "session": CommandDef(
                name="session",
                aliases=["s", "sess"],
                subcommands=["create", "list", "switch", "delete", "current"],
                description="Manage research sessions",
                usage="session <create|list|switch|delete> [name]",
                min_args=1,
                max_args=2
            ),
            "insert": CommandDef(
                name="insert",
                aliases=["i", "add"],
                subcommands=["files", "folder", "browse"],
                description="Insert PDFs into knowledge graph",
                usage="insert <path|files|folder|browse> [options]",
                requires_session=True,
                min_args=0,  # browse doesn't need args
                max_args=-1,
                valid_flags=["-r", "--recursive"]
            ),
            "query": CommandDef(
                name="query",
                aliases=["q", "search"],
                description="Query the knowledge graph",
                usage="query [--mode local|global|naive] <question>",
                requires_session=True,
                min_args=1,
                max_args=-1,
                valid_flags=["--mode"]
            ),
            "arxiv": CommandDef(
                name="arxiv",
                aliases=["arx", "paper"],
                subcommands=["search", "download", "history", "wizard"],
                description="Search and download ArXiv papers",
                usage="arxiv <search|download|history|wizard> [args]",
                min_args=0,  # wizard doesn't need args
                max_args=-1
            ),
            "iquery": CommandDef(
                name="iquery",
                aliases=["iq", "interactive"],
                description="Start interactive query mode",
                usage="iquery",
                requires_session=True,
                max_args=0
            ),
            "history": CommandDef(
                name="history",
                aliases=["hist", "h"],
                description="Show file insertion history",
                usage="history [limit]",
                requires_session=True,
                max_args=1
            ),
            "duplicates": CommandDef(
                name="duplicates",
                aliases=["dups", "dup"],
                description="Show duplicate files",
                usage="duplicates",
                requires_session=True,
                max_args=0
            ),
            "status": CommandDef(
                name="status",
                aliases=["stat", "info"],
                description="Show current session status",
                usage="status",
                max_args=0
            ),
            "config": CommandDef(
                name="config",
                aliases=["cfg", "conf"],
                description="Show configuration",
                usage="config",
                max_args=0
            ),
            "help": CommandDef(
                name="help",
                aliases=["?", "h"],
                description="Show help information",
                usage="help [command]",
                max_args=1
            ),
            "exit": CommandDef(
                name="exit",
                aliases=["quit", "q", "bye"],
                description="Exit the CLI",
                usage="exit",
                max_args=0
            )
        }
        
    def parse_command(self, input_text: str, current_session: Optional[str] = None) -> ParsedCommand:
        """Parse command with enhanced validation and suggestions"""
        input_text = input_text.strip()
        
        if not input_text:
            return ParsedCommand(command="", original_input=input_text)
            
        # Add to history
        if input_text not in self.command_history:
            self.command_history.append(input_text)
            if len(self.command_history) > self.max_history:
                self.command_history.pop(0)
        
        try:
            # Smart tokenization that handles quoted arguments
            parts = shlex.split(input_text)
        except ValueError:
            # Fallback to basic split if shlex fails
            parts = input_text.split()
            
        if not parts:
            return ParsedCommand(command="", original_input=input_text)
            
        # Extract command and find best match
        raw_command = parts[0].lower()
        command_name = self._find_best_command_match(raw_command)
        
        result = ParsedCommand(
            command=command_name or raw_command,
            original_input=input_text
        )
        
        # If no exact match found, provide suggestions
        if not command_name:
            result.suggestions = self._get_command_suggestions(raw_command)
            return result
            
        command_def = self.commands[command_name]
        remaining_parts = parts[1:]
        
        # Parse subcommand if applicable
        if remaining_parts and command_def.subcommands:
            potential_subcommand = remaining_parts[0].lower()
            best_subcommand = self._find_best_match(potential_subcommand, command_def.subcommands)
            
            if best_subcommand:
                result.subcommand = best_subcommand
                remaining_parts = remaining_parts[1:]
            elif potential_subcommand not in ["-h", "--help"]:
                # Suggest subcommands if not a help flag
                result.suggestions.extend([f"{command_name} {sc}" for sc in command_def.subcommands])
        
        # Parse flags and arguments
        result.args, result.flags = self._parse_args_and_flags(remaining_parts, command_def.valid_flags)
        
        # Validate command
        validation_errors = self._validate_command(result, command_def, current_session)
        if validation_errors:
            result.suggestions.extend(validation_errors)
            
        return result
    
    def _find_best_command_match(self, input_cmd: str) -> Optional[str]:
        """Find best matching command using exact match, aliases, or fuzzy matching"""
        # Exact command match
        if input_cmd in self.commands:
            return input_cmd
            
        # Check aliases
        for cmd_name, cmd_def in self.commands.items():
            if input_cmd in cmd_def.aliases:
                return cmd_name
                
        # Fuzzy matching for typos
        best_match = self._find_best_match(input_cmd, list(self.commands.keys()))
        if best_match and self._similarity(input_cmd, best_match) > 0.6:
            return best_match
            
        return None
    
    def _find_best_match(self, input_str: str, candidates: List[str]) -> Optional[str]:
        """Find best fuzzy match from candidates"""
        if not candidates:
            return None
            
        best_score = 0
        best_match = None
        
        for candidate in candidates:
            score = self._similarity(input_str.lower(), candidate.lower())
            if score > best_score:
                best_score = score
                best_match = candidate
                
        return best_match if best_score > 0.4 else None
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a, b).ratio()
    
    def _parse_args_and_flags(self, parts: List[str], valid_flags: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """Parse arguments and flags from command parts"""
        args = []
        flags = {}
        i = 0
        
        while i < len(parts):
            part = parts[i]
            
            if part.startswith('-') and valid_flags:
                if part in valid_flags:
                    # Check if next part is the flag value
                    if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                        flags[part.lstrip('-')] = parts[i + 1]
                        i += 2
                    else:
                        flags[part.lstrip('-')] = True
                        i += 1
                else:
                    # Unknown flag, treat as argument
                    args.append(part)
                    i += 1
            else:
                args.append(part)
                i += 1
                
        return args, flags
    
    def _validate_command(self, parsed: ParsedCommand, cmd_def: CommandDef, current_session: Optional[str]) -> List[str]:
        """Validate parsed command and return error messages"""
        errors = []
        
        # Check if session is required
        if cmd_def.requires_session and not current_session:
            errors.append(f"Command '{parsed.command}' requires an active session. Use 'session create <name>' first.")
            
        # Check argument count
        arg_count = len(parsed.args)
        if cmd_def.min_args > 0 and arg_count < cmd_def.min_args:
            errors.append(f"Command '{parsed.command}' requires at least {cmd_def.min_args} arguments. Usage: {cmd_def.usage}")
            
        if cmd_def.max_args >= 0 and arg_count > cmd_def.max_args:
            errors.append(f"Command '{parsed.command}' accepts at most {cmd_def.max_args} arguments. Usage: {cmd_def.usage}")
            
        # Validate subcommands
        if cmd_def.subcommands and not parsed.subcommand:
            errors.append(f"Command '{parsed.command}' requires a subcommand: {', '.join(cmd_def.subcommands)}")
            
        return errors
    
    def _get_command_suggestions(self, input_cmd: str) -> List[str]:
        """Get command suggestions for unknown input"""
        suggestions = []
        
        # Find similar commands
        for cmd_name in self.commands.keys():
            if self._similarity(input_cmd, cmd_name) > 0.3:
                suggestions.append(f"Did you mean '{cmd_name}'?")
                
        # Find similar aliases
        for cmd_name, cmd_def in self.commands.items():
            for alias in cmd_def.aliases:
                if self._similarity(input_cmd, alias) > 0.5:
                    suggestions.append(f"Did you mean '{alias}' ({cmd_name})?")
                    
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def get_command_completions(self, partial_input: str) -> List[str]:
        """Get command completions for partial input"""
        if not partial_input:
            return list(self.commands.keys())
            
        parts = partial_input.split()
        if len(parts) == 1:
            # Complete main command
            partial_cmd = parts[0].lower()
            completions = []
            
            # Add matching commands
            for cmd_name in self.commands.keys():
                if cmd_name.startswith(partial_cmd):
                    completions.append(cmd_name)
                    
            # Add matching aliases
            for cmd_name, cmd_def in self.commands.items():
                for alias in cmd_def.aliases:
                    if alias.startswith(partial_cmd):
                        completions.append(alias)
                        
            return sorted(set(completions))
            
        elif len(parts) == 2:
            # Complete subcommand
            cmd_name = self._find_best_command_match(parts[0].lower())
            if cmd_name and cmd_name in self.commands:
                cmd_def = self.commands[cmd_name]
                partial_sub = parts[1].lower()
                return [sc for sc in cmd_def.subcommands if sc.startswith(partial_sub)]
                
        return []
    
    def get_file_completions(self, partial_path: str) -> List[str]:
        """Get file path completions"""
        try:
            if not partial_path:
                partial_path = "."
                
            path = Path(partial_path)
            if path.is_dir():
                # List directory contents
                pattern = str(path / "*")
            else:
                # Complete filename
                pattern = str(path) + "*"
                
            matches = glob.glob(pattern)
            return sorted([str(Path(m)) for m in matches])
            
        except Exception:
            return []
    
    def get_contextual_help(self, command: str, current_session: Optional[str] = None) -> str:
        """Get contextual help for a command"""
        if command not in self.commands:
            return f"Unknown command: {command}. Type 'help' for available commands."
            
        cmd_def = self.commands[command]
        help_text = [
            f"[bold cyan]{cmd_def.name}[/bold cyan] - {cmd_def.description}",
            f"[dim]Usage: {cmd_def.usage}[/dim]"
        ]
        
        if cmd_def.aliases:
            help_text.append(f"[dim]Aliases: {', '.join(cmd_def.aliases)}[/dim]")
            
        if cmd_def.subcommands:
            help_text.append(f"[dim]Subcommands: {', '.join(cmd_def.subcommands)}[/dim]")
            
        if cmd_def.requires_session and not current_session:
            help_text.append("[yellow]⚠ Requires active session[/yellow]")
            
        return "\n".join(help_text)
    
    def get_smart_suggestions(self, current_session: Optional[str] = None, has_knowledge_graph: bool = False) -> List[str]:
        """Get smart suggestions based on current state"""
        suggestions = []
        
        if not current_session:
            suggestions.append("💡 Start by creating a session: [cyan]session create my_research[/cyan]")
        elif not has_knowledge_graph:
            suggestions.append("💡 Add some PDFs to get started: [cyan]insert document.pdf[/cyan]")
            suggestions.append("💡 Or search ArXiv: [cyan]arxiv search 'topic'[/cyan]")
        else:
            suggestions.append("💡 Query your knowledge graph: [cyan]query 'your question'[/cyan]")
            suggestions.append("💡 Try interactive mode: [cyan]iquery[/cyan]")
            
        return suggestions