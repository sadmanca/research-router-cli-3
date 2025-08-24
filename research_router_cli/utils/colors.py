"""Color scheme and styling utilities for the CLI"""

from rich.console import Console
from rich.theme import Theme
from rich.style import Style

# Define consistent color scheme
COLORS = {
    # Status colors
    'success': 'bright_green',
    'error': 'bright_red',
    'warning': 'bright_yellow',
    'info': 'bright_cyan',
    'progress': 'bright_blue',
    'highlight': 'bright_magenta',
    
    # UI colors  
    'primary': 'blue',
    'secondary': 'cyan',
    'accent': 'magenta',
    'muted': 'dim white',
    
    # Document status colors
    'new_doc': 'green',
    'duplicate': 'yellow',
    'error_doc': 'red',
    'processing': 'blue',
}

# Status symbols with colors (Windows-safe)
STATUS_SYMBOLS = {
    'success': '[+]',
    'error': '[-]', 
    'warning': '[!]',
    'info': '[i]',
    'processing': '[~]',
    'download': '[v]',
    'search': '[?]',
    'duplicate': '[=]',
    'new': '[*]',
}

# Create Rich theme
RESEARCH_THEME = Theme({
    'success': COLORS['success'],
    'error': COLORS['error'],
    'warning': COLORS['warning'],
    'info': COLORS['info'],
    'progress': COLORS['progress'],
    'highlight': COLORS['highlight'],
    'primary': COLORS['primary'],
    'secondary': COLORS['secondary'],
    'accent': COLORS['accent'],
    'muted': COLORS['muted'],
    'new_doc': COLORS['new_doc'],
    'duplicate': COLORS['duplicate'],
    'error_doc': COLORS['error_doc'],
    'processing': COLORS['processing'],
})

# Initialize console with theme
console = Console(theme=RESEARCH_THEME)

def success_msg(message: str, symbol: bool = True) -> str:
    """Format success message with optional symbol"""
    sym = f"[success]{STATUS_SYMBOLS['success']}[/success] " if symbol else ""
    return f"{sym}[success]{message}[/success]"

def error_msg(message: str, symbol: bool = True) -> str:
    """Format error message with optional symbol"""
    sym = f"[error]{STATUS_SYMBOLS['error']}[/error] " if symbol else ""
    return f"{sym}[error]{message}[/error]"

def warning_msg(message: str, symbol: bool = True) -> str:
    """Format warning message with optional symbol"""
    sym = f"[warning]{STATUS_SYMBOLS['warning']}[/warning] " if symbol else ""
    return f"{sym}[warning]{message}[/warning]"

def info_msg(message: str, symbol: bool = True) -> str:
    """Format info message with optional symbol"""
    sym = f"[info]{STATUS_SYMBOLS['info']}[/info] " if symbol else ""
    return f"{sym}[info]{message}[/info]"

def progress_msg(message: str, symbol: bool = True) -> str:
    """Format progress message with optional symbol"""
    sym = f"[progress]{STATUS_SYMBOLS['processing']}[/progress] " if symbol else ""
    return f"{sym}[progress]{message}[/progress]"

def highlight_msg(message: str) -> str:
    """Format highlighted message"""
    return f"[highlight]{message}[/highlight]"

def document_status_msg(filename: str, status: str, details: str = "") -> str:
    """Format document status message"""
    if status == 'success':
        color = 'new_doc'
        symbol = STATUS_SYMBOLS['success']
    elif status == 'duplicate':
        color = 'duplicate'
        symbol = STATUS_SYMBOLS['duplicate']
    elif status == 'error':
        color = 'error_doc'
        symbol = STATUS_SYMBOLS['error']
    elif status == 'processing':
        color = 'processing'
        symbol = STATUS_SYMBOLS['processing']
    else:
        color = 'info'
        symbol = STATUS_SYMBOLS['info']
    
    base_msg = f"[{color}]{symbol}[/{color}] [primary]{filename}[/primary]"
    if details:
        base_msg += f" [muted]({details})[/muted]"
    
    return base_msg

def arxiv_result_style(title: str, authors: str, published: str, is_selected: bool = False) -> str:
    """Format ArXiv search result"""
    selection_marker = "[highlight]>[/highlight] " if is_selected else "  "
    return f"{selection_marker}[primary]{title}[/primary]\n    [muted]by {authors} - {published}[/muted]"

def session_status_style(session_name: str, is_current: bool = False) -> str:
    """Format session status"""
    if is_current:
        return f"[highlight]>[/highlight] [success]{session_name}[/success] [muted](current)[/muted]"
    else:
        return f"  [primary]{session_name}[/primary]"