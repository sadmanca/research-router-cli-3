"""Cross-platform terminal utilities for better UX"""

import os
import sys
import subprocess
from typing import Optional
from rich.console import Console


class TerminalManager:
    """Manages terminal operations across different platforms and terminals"""
    
    def __init__(self, console: Console):
        self.console = console
        self.platform = sys.platform.lower()
        self.is_windows = self.platform.startswith('win')
        self.term = os.environ.get('TERM', '').lower()
        
    def clear_screen(self, force: bool = False) -> bool:
        """
        Clear the terminal screen using the most appropriate method
        
        Args:
            force: If True, use multiple methods to ensure clearing
            
        Returns:
            bool: True if clearing was successful
        """
        success = False
        
        try:
            # Method 1: Try Rich's built-in clear (most compatible)
            if hasattr(self.console, 'clear'):
                self.console.clear()
                success = True
            
            # Method 2: OS-specific clear command (fallback)
            if not success or force:
                if self.is_windows:
                    # Windows CMD/PowerShell
                    os.system('cls')
                    success = True
                else:
                    # Unix/Linux/macOS
                    os.system('clear')
                    success = True
                    
            # Method 3: ANSI escape sequences (universal fallback)
            if not success or force:
                if self._supports_ansi():
                    # Clear screen and move cursor to top-left
                    sys.stdout.write('\033[2J\033[H')
                    sys.stdout.flush()
                    success = True
                    
            # Method 4: Print newlines as last resort
            if not success:
                print('\n' * 100)  # Scroll old content out of view
                success = True
                
        except Exception:
            # Final fallback - just print newlines
            print('\n' * 50)
            success = True
            
        return success
    
    def clear_screen_complete(self) -> bool:
        """
        Complete screen clear - uses multiple methods to ensure clean state
        Especially important for Windows CMD which can be stubborn
        """
        success = True
        
        try:
            # Step 1: Clear with ANSI if supported
            if self._supports_ansi():
                sys.stdout.write('\033[2J\033[3J\033[H')  # Clear screen, scrollback, home cursor
                sys.stdout.flush()
            
            # Step 2: OS clear command
            if self.is_windows:
                # Try multiple Windows clearing methods
                try:
                    subprocess.run(['cls'], shell=True, check=True, capture_output=True)
                except:
                    os.system('cls')
            else:
                try:
                    subprocess.run(['clear'], check=True, capture_output=True)
                except:
                    os.system('clear')
                    
            # Step 3: Rich clear
            if hasattr(self.console, 'clear'):
                self.console.clear()
                
            # Step 4: Ensure cursor is at top
            if self._supports_ansi():
                sys.stdout.write('\033[H')  # Move cursor home
                sys.stdout.flush()
                
        except Exception as e:
            # Fallback to newlines
            print('\n' * 60)
            success = False
            
        return success
    
    def clear_and_show_header(self, title: str, subtitle: Optional[str] = None) -> None:
        """
        Clear screen and show a clean header
        
        Args:
            title: Main title to display
            subtitle: Optional subtitle
        """
        self.clear_screen_complete()
        
        # Add some space from top
        print()
        
        # Show header with Rich formatting
        if subtitle:
            self.console.print(f"[bold cyan]{title}[/bold cyan]")
            self.console.print(f"[dim]{subtitle}[/dim]")
        else:
            self.console.print(f"[bold cyan]{title}[/bold cyan]")
            
        print()  # Space after header
    
    def _supports_ansi(self) -> bool:
        """Check if terminal supports ANSI escape sequences"""
        # Windows 10+ CMD supports ANSI
        if self.is_windows:
            # Check Windows version
            try:
                import platform
                version = platform.version()
                # Windows 10 version 1511+ supports ANSI
                if version and len(version.split('.')) >= 3:
                    major_version = int(version.split('.')[0])
                    if major_version >= 10:
                        return True
            except:
                pass
            
            # Check for Windows Terminal, PowerShell, or other modern terminals
            if any(term in os.environ.get('TERM_PROGRAM', '').lower() for term in 
                   ['windows terminal', 'vscode']):
                return True
                
            # Check if ANSICON is present (enables ANSI in older Windows)
            if 'ANSICON' in os.environ:
                return True
                
            return False  # Conservative default for old Windows CMD
        else:
            # Most Unix terminals support ANSI
            return True
    
    def show_transition_message(self, message: str, delay: float = 0.5) -> None:
        """
        Show a brief transition message before clearing
        
        Args:
            message: Message to show
            delay: Delay in seconds before clearing
        """
        self.console.print(f"\n[dim]{message}[/dim]")
        
        if delay > 0:
            import time
            time.sleep(delay)
            
        self.clear_screen_complete()


# Global terminal manager instance
_terminal_manager: Optional[TerminalManager] = None


def get_terminal_manager(console: Console) -> TerminalManager:
    """Get or create terminal manager instance"""
    global _terminal_manager
    if _terminal_manager is None:
        _terminal_manager = TerminalManager(console)
    return _terminal_manager


def clear_screen_smart(console: Console, force: bool = False) -> bool:
    """Smart screen clearing function - main entry point"""
    manager = get_terminal_manager(console)
    return manager.clear_screen_complete() if force else manager.clear_screen()


def clear_and_header(console: Console, title: str, subtitle: Optional[str] = None) -> None:
    """Clear screen and show header - convenience function"""
    manager = get_terminal_manager(console)
    manager.clear_and_show_header(title, subtitle)


def transition_to_view(console: Console, message: str, delay: float = 0.3) -> None:
    """Transition to a new view with message"""
    manager = get_terminal_manager(console)
    manager.show_transition_message(message, delay)