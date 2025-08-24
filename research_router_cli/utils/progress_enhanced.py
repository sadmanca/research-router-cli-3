"""Enhanced progress indicators with better time estimates and visual feedback"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeElapsedColumn, 
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpeedColumn
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from .colors import console, info_msg, success_msg, warning_msg


@dataclass
class TaskStats:
    """Statistics for tracking task progress"""
    start_time: datetime = field(default_factory=datetime.now)
    items_processed: int = 0
    total_items: int = 0
    bytes_processed: int = 0
    total_bytes: int = 0
    errors: int = 0
    
    @property
    def elapsed_time(self) -> timedelta:
        return datetime.now() - self.start_time
    
    @property
    def items_per_second(self) -> float:
        elapsed = self.elapsed_time.total_seconds()
        if elapsed > 0:
            return self.items_processed / elapsed
        return 0.0
    
    @property
    def bytes_per_second(self) -> float:
        elapsed = self.elapsed_time.total_seconds()
        if elapsed > 0:
            return self.bytes_processed / elapsed
        return 0.0
    
    @property
    def estimated_time_remaining(self) -> Optional[timedelta]:
        if self.total_items > 0 and self.items_processed > 0:
            rate = self.items_per_second
            if rate > 0:
                remaining_items = self.total_items - self.items_processed
                seconds_remaining = remaining_items / rate
                return timedelta(seconds=seconds_remaining)
        return None


class EnhancedProgress:
    """Enhanced progress tracking with better estimates and visual feedback"""
    
    def __init__(self):
        self.stats: Dict[str, TaskStats] = {}
        self.current_task_id: Optional[str] = None
        
    def create_progress_bar(self, 
                          show_speed: bool = True,
                          show_eta: bool = True,
                          show_percentage: bool = True) -> Progress:
        """Create a rich progress bar with enhanced columns"""
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
        ]
        
        if show_percentage:
            columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
            
        columns.append(MofNCompleteColumn())
        
        if show_speed:
            columns.append(SpeedColumn())
            
        columns.append(TimeElapsedColumn())
        
        if show_eta:
            columns.append(TimeRemainingColumn())
            
        return Progress(*columns)
    
    def track_file_processing(self, 
                            files: list,
                            task_name: str = "Processing files",
                            show_individual: bool = True) -> Progress:
        """Create progress tracker for file processing with size estimates"""
        
        # Calculate total size if possible
        total_size = 0
        for file_path in files:
            try:
                from pathlib import Path
                total_size += Path(file_path).stat().st_size
            except:
                pass
                
        progress = self.create_progress_bar(show_speed=True, show_eta=True)
        
        # Create main task
        main_task = progress.add_task(
            task_name, 
            total=len(files),
            start=True
        )
        
        # Create size task if we have size info
        size_task = None
        if total_size > 0:
            size_task = progress.add_task(
                "Data processed",
                total=total_size,
                start=True,
                visible=False  # Hidden by default
            )
            
        return progress, main_task, size_task
    
    def track_download_progress(self, 
                              papers: list,
                              task_name: str = "Downloading papers") -> Progress:
        """Track download progress with network-aware estimates"""
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[green]Network: [/green][blue]{task.fields[network_status]}[/blue]")
        )
        
        task = progress.add_task(
            task_name,
            total=len(papers),
            network_status="Connecting..."
        )
        
        return progress, task
    
    def show_processing_summary(self, 
                              stats: TaskStats,
                              task_name: str = "Processing",
                              show_details: bool = True):
        """Show a detailed processing summary"""
        
        # Create summary table
        table = Table(title=f"{task_name} Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Basic metrics
        table.add_row("Total Items", str(stats.total_items))
        table.add_row("Processed", str(stats.items_processed))
        table.add_row("Errors", str(stats.errors))
        table.add_row("Success Rate", f"{((stats.items_processed - stats.errors) / max(stats.items_processed, 1) * 100):.1f}%")
        
        # Time metrics
        table.add_row("Total Time", str(stats.elapsed_time).split('.')[0])
        table.add_row("Average Speed", f"{stats.items_per_second:.1f} items/sec")
        
        # Size metrics if available
        if stats.bytes_processed > 0:
            table.add_row("Data Processed", self._format_bytes(stats.bytes_processed))
            table.add_row("Data Speed", f"{self._format_bytes(stats.bytes_per_second)}/sec")
            
        console.print(table)
        
        # Show detailed breakdown if requested
        if show_details and stats.errors > 0:
            error_rate = (stats.errors / max(stats.items_processed, 1)) * 100
            if error_rate > 10:
                console.print(warning_msg(f"High error rate: {error_rate:.1f}%. Check your configuration."))
    
    def create_live_status(self, title: str = "Processing") -> Live:
        """Create a live-updating status display"""
        
        def make_status_table():
            table = Table.grid(padding=1)
            table.add_column(style="cyan", justify="right")
            table.add_column(style="magenta")
            
            if self.current_task_id and self.current_task_id in self.stats:
                stats = self.stats[self.current_task_id]
                table.add_row("Processed:", f"{stats.items_processed}/{stats.total_items}")
                table.add_row("Speed:", f"{stats.items_per_second:.1f} items/sec")
                table.add_row("Elapsed:", str(stats.elapsed_time).split('.')[0])
                
                if stats.estimated_time_remaining:
                    table.add_row("Remaining:", str(stats.estimated_time_remaining).split('.')[0])
                    
                if stats.errors > 0:
                    table.add_row("Errors:", f"[red]{stats.errors}[/red]")
            
            return Panel(table, title=title, border_style="blue")
            
        return Live(make_status_table(), refresh_per_second=2)
    
    def start_task_tracking(self, 
                          task_id: str,
                          total_items: int = 0,
                          total_bytes: int = 0):
        """Start tracking statistics for a task"""
        self.stats[task_id] = TaskStats(
            total_items=total_items,
            total_bytes=total_bytes
        )
        self.current_task_id = task_id
        
    def update_task_progress(self, 
                           task_id: str,
                           items_delta: int = 0,
                           bytes_delta: int = 0,
                           errors_delta: int = 0):
        """Update task progress statistics"""
        if task_id in self.stats:
            stats = self.stats[task_id]
            stats.items_processed += items_delta
            stats.bytes_processed += bytes_delta
            stats.errors += errors_delta
            
    def finish_task_tracking(self, task_id: str) -> TaskStats:
        """Finish tracking and return final statistics"""
        if task_id in self.stats:
            return self.stats[task_id]
        return TaskStats()
    
    def _format_bytes(self, bytes_count: float) -> str:
        """Format bytes for human-readable display"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} PB"


class ProgressManager:
    """Context manager for enhanced progress tracking"""
    
    def __init__(self, task_name: str, items: list, show_live_status: bool = True):
        self.task_name = task_name
        self.items = items
        self.show_live_status = show_live_status
        self.enhanced_progress = EnhancedProgress()
        self.task_id = f"{task_name}_{int(time.time())}"
        self.live_status = None
        
    def __enter__(self):
        self.enhanced_progress.start_task_tracking(
            self.task_id,
            total_items=len(self.items)
        )
        
        if self.show_live_status:
            self.live_status = self.enhanced_progress.create_live_status(self.task_name)
            self.live_status.start()
            
        return self.enhanced_progress
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live_status:
            self.live_status.stop()
            
        # Show final summary
        stats = self.enhanced_progress.finish_task_tracking(self.task_id)
        if stats.items_processed > 0:
            self.enhanced_progress.show_processing_summary(stats, self.task_name)


# Helper functions for common progress patterns

def track_file_insertion(files: list, session_name: str = ""):
    """Helper for tracking file insertion progress"""
    task_name = f"Inserting files{f' into {session_name}' if session_name else ''}"
    return ProgressManager(task_name, files, show_live_status=True)

def track_arxiv_downloads(papers: list):
    """Helper for tracking ArXiv download progress"""
    return ProgressManager("Downloading ArXiv papers", papers, show_live_status=True)

def track_knowledge_graph_building(chunks: list):
    """Helper for tracking knowledge graph building"""
    return ProgressManager("Building knowledge graph", chunks, show_live_status=True)