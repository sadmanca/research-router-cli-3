"""Interactive file browser for easy file and folder selection"""

import os
from pathlib import Path
from typing import List, Optional, Union, Tuple
import mimetypes
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.prompt import IntPrompt, Confirm
from rich.text import Text
from rich.tree import Tree

from .colors import console, info_msg, warning_msg, error_msg, success_msg, highlight_msg
from .terminal_utils import clear_and_header, transition_to_view, clear_screen_smart


class FileBrowser:
    """Interactive file browser with keyboard navigation and file selection"""
    
    def __init__(self):
        self.current_path = Path.cwd()
        self.selected_files: List[Path] = []
        
    def browse_for_files(self, 
                        initial_path: Optional[Union[str, Path]] = None,
                        file_filter: str = "*.pdf",
                        multi_select: bool = True,
                        show_hidden: bool = False) -> List[Path]:
        """Interactive file browser that returns selected files"""
        
        if initial_path:
            self.current_path = Path(initial_path).resolve()
            if not self.current_path.exists():
                self.current_path = Path.cwd()
                
        # Clear screen and show clean header
        clear_and_header(
            console,
            "🗂️  Interactive File Browser",
            f"Navigate and select files for your knowledge graph • Filter: {file_filter}"
        )
        
        self.selected_files = []
        
        while True:
            try:
                choice = self._show_directory_listing(file_filter, show_hidden, multi_select)
                
                if choice == 'quit':
                    transition_to_view(console, "Exiting file browser...", 0.2)
                    break
                elif choice == 'up':
                    self._go_up()
                    # Clear and redraw after navigation
                    clear_and_header(
                        console,
                        "🗂️  Interactive File Browser", 
                        f"Current: {self.current_path} • Filter: {file_filter}"
                    )
                elif choice == 'select_current':
                    if multi_select and self.selected_files:
                        clear_and_header(console, "✅ File Selection Complete")
                        console.print(success_msg(f"Successfully selected {len(self.selected_files)} files:", False))
                        console.print()
                        for i, file_path in enumerate(self.selected_files, 1):
                            console.print(f"   [cyan]{i}.[/cyan] [green]{file_path.name}[/green]")
                            console.print(f"       [dim]{file_path.parent}[/dim]")
                        console.print(f"\n[bold green]🚀 These files will now be inserted into your knowledge graph![/bold green]")
                        console.input("\n[dim]Press Enter to continue...[/dim]")
                        break
                    else:
                        console.print(warning_msg("❌ No files selected. Select some files first!"))
                elif choice == 'toggle_hidden':
                    show_hidden = not show_hidden
                elif choice == 'clear_selection':
                    self.selected_files.clear()
                    console.print(info_msg("Selection cleared"))
                    # Redraw to show updated selection count
                    clear_and_header(
                        console,
                        "🗂️  Interactive File Browser",
                        f"Current: {self.current_path} • Selection cleared • Filter: {file_filter}"
                    )
                elif isinstance(choice, int):
                    result = self._handle_file_selection(choice, file_filter, multi_select)
                    # If we navigated to a new directory, clear and redraw
                    if result == 'navigated':
                        clear_and_header(
                            console,
                            "🗂️  Interactive File Browser",
                            f"Current: {self.current_path} • Filter: {file_filter}"
                        )
                    
            except KeyboardInterrupt:
                transition_to_view(console, "File browsing cancelled", 0.2)
                return []
            except Exception as e:
                console.print(error_msg(f"Error: {e}"))
                
        return self.selected_files
        
    def quick_path_selector(self, prompt: str = "Enter path", allow_autocomplete: bool = True) -> Optional[str]:
        """Quick path input with autocomplete suggestions"""
        console.print(info_msg(f"{prompt} (Tab for common paths, '?' for browser)"))
        
        # Show common paths
        common_paths = self._get_common_paths()
        if common_paths:
            console.print("[dim]Common paths:[/dim]")
            for i, (name, path) in enumerate(common_paths, 1):
                console.print(f"  [cyan]{i}.[/cyan] {name} [dim]({path})[/dim]")
            console.print()
            
        while True:
            try:
                user_input = console.input(f"[bold cyan]{prompt}: [/bold cyan]").strip()
                
                if not user_input:
                    return None
                    
                if user_input == '?':
                    # Launch file browser
                    files = self.browse_for_files(multi_select=False)
                    return str(files[0]) if files else None
                    
                if user_input.isdigit():
                    # Selected a common path
                    idx = int(user_input) - 1
                    if 0 <= idx < len(common_paths):
                        return str(common_paths[idx][1])
                        
                # Try to resolve the path
                path = Path(user_input).expanduser()
                if path.exists():
                    return str(path.resolve())
                else:
                    # Try autocomplete
                    suggestions = self._get_path_completions(user_input)
                    if suggestions:
                        console.print(info_msg("Did you mean:"))
                        for i, suggestion in enumerate(suggestions[:5], 1):
                            console.print(f"  [cyan]{i}.[/cyan] {suggestion}")
                        
                        choice = console.input("[dim]Select number or continue typing: [/dim]").strip()
                        if choice.isdigit():
                            idx = int(choice) - 1
                            if 0 <= idx < len(suggestions):
                                return suggestions[idx]
                    else:
                        console.print(warning_msg(f"Path not found: {user_input}"))
                        
            except KeyboardInterrupt:
                return None
    
    def _show_directory_listing(self, file_filter: str, show_hidden: bool, multi_select: bool) -> Union[str, int]:
        """Show directory contents and get user choice"""
        console.clear()
        
        # Header with current path
        header = Panel(
            f"[bold cyan]📁 Current Directory:[/bold cyan] {self.current_path}\n"
            f"[dim]Filter: {file_filter} | Selected: {len(self.selected_files)} files[/dim]\n"
            f"[yellow]💡 Instructions: Type number to navigate/select • 's' when done selecting • 'q' to quit[/yellow]",
            border_style="blue"
        )
        console.print(header)
        
        # Get directory contents
        items = self._get_directory_items(file_filter, show_hidden)
        
        if not items:
            console.print(warning_msg("No items found in this directory"))
            console.print(info_msg("📝 Commands: [u]p to go back, [q]uit to exit"))
            choice = console.input("\n[bold green]Your choice:[/bold green] ").strip().lower()
            return 'up' if choice == 'u' else 'quit'
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Name", style="cyan", min_width=30)
        table.add_column("Type", width=10)
        table.add_column("Size", width=10, justify="right")
        table.add_column("Modified", width=12, style="dim")
        
        for i, item in enumerate(items, 1):
            name = item['name']
            if item['path'] in self.selected_files:
                name = f"[green]✓[/green] {name}"
            elif item['is_dir']:
                name = f"📁 {name}"
            else:
                name = f"📄 {name}"
                
            table.add_row(
                str(i),
                name,
                item['type'],
                item['size'],
                item['modified']
            )
            
        console.print(table)
        
        # Show selected files if any
        if self.selected_files:
            console.print(f"\n[green]✅ Selected files ({len(self.selected_files)}):[/green]")
            for i, file_path in enumerate(self.selected_files[-3:], 1):  # Show last 3
                console.print(f"   {i}. {file_path.name}")
            if len(self.selected_files) > 3:
                console.print(f"   ... and {len(self.selected_files) - 3} more")
        
        # Show clear command instructions
        console.print(f"\n[bold yellow]📝 Commands:[/bold yellow]")
        console.print("  • [bold cyan]Number[/bold cyan] - Navigate to folder or select file")
        console.print("  • [bold green]s[/bold green] - Finish and use selected files")
        console.print("  • [bold red]u[/bold red] - Go up one directory")
        console.print("  • [bold red]q[/bold red] - Quit without selecting")
        if multi_select:
            console.print("  • [bold yellow]c[/bold yellow] - Clear all selections")
        console.print("  • [bold dim]h[/bold dim] - Toggle hidden files")
        
        # Get user input
        choice = console.input("\n[bold green]Your choice:[/bold green] ").strip().lower()
        
        if choice == 'u':
            return 'up'
        elif choice == 'q':
            return 'quit'
        elif choice == 's':
            return 'select_current'
        elif choice == 'c':
            return 'clear_selection'
        elif choice == 'h':
            return 'toggle_hidden'
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return idx
                
        console.print(warning_msg("Invalid choice"))
        console.input("Press Enter to continue...")
        return 'continue'
        
    def _get_directory_items(self, file_filter: str, show_hidden: bool) -> List[dict]:
        """Get directory items with metadata"""
        items = []
        
        try:
            for item in sorted(self.current_path.iterdir()):
                if not show_hidden and item.name.startswith('.'):
                    continue
                    
                # Get file info
                try:
                    stat = item.stat()
                    modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')
                    
                    if item.is_dir():
                        items.append({
                            'name': item.name + '/',
                            'path': item,
                            'is_dir': True,
                            'type': 'Directory',
                            'size': '',
                            'modified': modified
                        })
                    else:
                        # Check if file matches filter
                        if self._matches_filter(item.name, file_filter):
                            size_mb = round(stat.st_size / 1024 / 1024, 2)
                            size_str = f"{size_mb:.2f}MB" if size_mb > 0.01 else f"{stat.st_size}B"
                            
                            file_type = self._get_file_type(item)
                            
                            items.append({
                                'name': item.name,
                                'path': item,
                                'is_dir': False,
                                'type': file_type,
                                'size': size_str,
                                'modified': modified
                            })
                            
                except (OSError, PermissionError):
                    # Skip files we can't access
                    continue
                    
        except PermissionError:
            console.print(error_msg("Permission denied accessing this directory"))
            
        return items
        
    def _handle_file_selection(self, index: int, file_filter: str, multi_select: bool) -> str:
        """Handle file/directory selection"""
        items = self._get_directory_items(file_filter, False)
        
        if 0 <= index < len(items):
            item = items[index]
            
            if item['is_dir']:
                # Navigate into directory
                self.current_path = item['path']
                console.print(info_msg(f"📂 Navigated to: {item['name']}"))
                return 'navigated'
            else:
                # Select file
                if multi_select:
                    if item['path'] in self.selected_files:
                        self.selected_files.remove(item['path'])
                        console.print(info_msg(f"❌ Deselected: {item['name']}"))
                    else:
                        self.selected_files.append(item['path'])
                        console.print(success_msg(f"✅ Selected: {item['name']}"))
                    
                    # Show quick reminder
                    console.print(f"[dim]💡 Press 's' when done selecting (currently have {len(self.selected_files)} files)[/dim]")
                else:
                    self.selected_files = [item['path']]
                    console.print(success_msg(f"✅ Selected: {item['name']}"))
                return 'selected'
        else:
            console.print(error_msg("❌ Invalid choice. Please try again."))
            return 'error'
        
    def _go_up(self):
        """Navigate up one directory level"""
        parent = self.current_path.parent
        if parent != self.current_path:  # Not at root
            self.current_path = parent
        else:
            console.print(warning_msg("Already at root directory"))
            console.input("Press Enter to continue...")
            
    def _matches_filter(self, filename: str, filter_pattern: str) -> bool:
        """Check if filename matches filter pattern"""
        import fnmatch
        
        if not filter_pattern or filter_pattern == "*":
            return True
            
        return fnmatch.fnmatch(filename.lower(), filter_pattern.lower())
        
    def _get_file_type(self, path: Path) -> str:
        """Get file type description"""
        mime_type, _ = mimetypes.guess_type(str(path))
        
        if mime_type:
            if mime_type.startswith('application/pdf'):
                return 'PDF'
            elif mime_type.startswith('text/'):
                return 'Text'
            elif mime_type.startswith('image/'):
                return 'Image'
            else:
                return mime_type.split('/')[-1].upper()
        else:
            ext = path.suffix.lower()
            if ext == '.pdf':
                return 'PDF'
            elif ext in ['.txt', '.md', '.rst']:
                return 'Text'
            elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
                return 'Image'
            else:
                return ext.lstrip('.').upper() if ext else 'File'
                
    def _get_common_paths(self) -> List[Tuple[str, Path]]:
        """Get list of common/useful paths"""
        common_paths = []
        
        # Home directory
        common_paths.append(("Home", Path.home()))
        
        # Desktop
        desktop = Path.home() / "Desktop"
        if desktop.exists():
            common_paths.append(("Desktop", desktop))
            
        # Documents
        documents = Path.home() / "Documents"
        if documents.exists():
            common_paths.append(("Documents", documents))
            
        # Downloads
        downloads = Path.home() / "Downloads"
        if downloads.exists():
            common_paths.append(("Downloads", downloads))
            
        # Current directory
        common_paths.append(("Current", Path.cwd()))
        
        return common_paths
        
    def _get_path_completions(self, partial_path: str) -> List[str]:
        """Get path completions for partial input"""
        try:
            path = Path(partial_path).expanduser()
            
            if path.is_dir():
                # List directory contents
                items = [str(item) for item in path.iterdir() if not item.name.startswith('.')]
                return sorted(items)[:10]
            else:
                # Try to complete parent directory
                parent = path.parent
                if parent.exists():
                    pattern = path.name.lower()
                    items = []
                    for item in parent.iterdir():
                        if item.name.lower().startswith(pattern):
                            items.append(str(item))
                    return sorted(items)[:10]
                    
        except Exception:
            pass
            
        return []


def quick_file_select(prompt: str = "Select file", 
                     file_filter: str = "*.pdf",
                     multi_select: bool = False) -> List[str]:
    """Quick file selection utility"""
    browser = FileBrowser()
    
    console.print(Panel(
        f"[bold]{prompt}[/bold]\n"
        f"Quick options: [1] Browse files [2] Enter path manually",
        border_style="green"
    ))
    
    choice = console.input("Choose option (1/2): ").strip()
    
    if choice == "1":
        # Use file browser
        files = browser.browse_for_files(file_filter=file_filter, multi_select=multi_select)
        return [str(f) for f in files]
    else:
        # Manual path entry
        path = browser.quick_path_selector(prompt)
        return [path] if path else []