"""ArXiv command for searching and downloading papers"""

import asyncio
from pathlib import Path
from typing import List, Optional

from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.panel import Panel

from ..utils.arxiv_client import ArxivClient
from ..utils.file_tracker import FileTracker
from ..utils.colors import console, success_msg, error_msg, warning_msg, info_msg, progress_msg, arxiv_result_style
from .session import SessionManager

class ArxivCommand:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self._arxiv_client = None
        self._file_tracker = None
        
    async def search_papers(self, query: str, max_results: int = 20):
        """Search ArXiv for papers"""
        if not self.session_manager.ensure_session():
            return
            
        arxiv_client = await self._get_arxiv_client()
        if not arxiv_client:
            return
            
        # Search for papers
        papers = await arxiv_client.search_papers(query, max_results)
        if not papers:
            console.print(warning_msg("No papers found"))
            return
            
        # Display results
        console.print(Panel(f"Found {len(papers)} papers for: '{query}'", title="ArXiv Search Results", style="info"))
        
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="primary")
        table.add_column("Authors", style="muted", width=30)
        table.add_column("Published", style="secondary", width=12)
        table.add_column("ID", style="accent", width=15)
        
        for i, paper in enumerate(papers):
            # Truncate title if too long
            title = paper["title"][:60] + "..." if len(paper["title"]) > 60 else paper["title"]
            
            # Truncate authors if too long  
            authors = paper["authors"][:30] + "..." if len(paper["authors"]) > 30 else paper["authors"]
            
            table.add_row(
                str(i),
                title,
                authors,
                paper["published"],
                paper["id"]
            )
            
        console.print(table)
        
        # Interactive selection
        await self._interactive_download_selection(papers)
        
    async def download_paper_by_id(self, arxiv_id: str):
        """Download a specific paper by ArXiv ID"""
        if not self.session_manager.ensure_session():
            return
            
        arxiv_client = await self._get_arxiv_client()
        file_tracker = await self._get_file_tracker()
        if not arxiv_client or not file_tracker:
            return
            
        # Check if already downloaded
        if await file_tracker.is_arxiv_paper_downloaded(arxiv_id):
            console.print(warning_msg(f"Paper {arxiv_id} already downloaded"))
            return
            
        # Get paper info
        paper = arxiv_client.get_paper_by_id(arxiv_id)
        if not paper:
            console.print(error_msg(f"Paper not found: {arxiv_id}"))
            return
            
        # Download paper
        filepath = await arxiv_client.download_paper(paper)
        if filepath:
            # Record in database
            await file_tracker.record_arxiv_paper(
                arxiv_id,
                paper["title"],
                paper["authors"],
                paper["abstract"],
                paper["published"],
                str(filepath),
                ""  # We'll calculate hash later if needed
            )
            
            console.print(success_msg(f"Downloaded and recorded: {filepath.name}"))
            
            # Ask if user wants to insert into knowledge graph
            if Confirm.ask("Insert this paper into the knowledge graph?"):
                from .insert import InsertCommand
                insert_cmd = InsertCommand(self.session_manager)
                await insert_cmd.insert_pdf(filepath)
                
    async def show_arxiv_history(self, limit: int = 20):
        """Show ArXiv download history"""
        if not self.session_manager.ensure_session():
            return
            
        file_tracker = await self._get_file_tracker()
        if not file_tracker:
            return
            
        history = await file_tracker.get_arxiv_history(limit)
        if not history:
            console.print(info_msg("No ArXiv papers downloaded yet"))
            return
            
        table = Table(title=f"ArXiv Download History (last {len(history)} papers)")
        table.add_column("ArXiv ID", style="accent")
        table.add_column("Title", style="primary")
        table.add_column("Authors", style="muted", width=25)
        table.add_column("Downloaded", style="secondary")
        
        for paper in history:
            title = paper["title"][:50] + "..." if len(paper["title"]) > 50 else paper["title"]
            authors = paper["authors"][:25] + "..." if len(paper["authors"]) > 25 else paper["authors"]
            
            table.add_row(
                paper["arxiv_id"],
                title,
                authors,
                paper["download_date"][:10]  # Just the date part
            )
            
        console.print(table)
        
    async def _interactive_download_selection(self, papers: List[dict]):
        """Interactive selection and download of papers"""
        if not Confirm.ask("\nSelect papers to download?"):
            return
            
        console.print(info_msg("Enter paper numbers (comma-separated, e.g., '0,2,5') or 'all' for all papers:"))
        
        while True:
            try:
                selection = Prompt.ask("Selection").strip()
                
                if selection.lower() == 'all':
                    selected_indices = list(range(len(papers)))
                    break
                elif selection.lower() in ['none', 'exit', 'quit']:
                    console.print(info_msg("No papers selected"))
                    return
                else:
                    # Parse comma-separated indices
                    indices = []
                    for idx_str in selection.split(','):
                        idx = int(idx_str.strip())
                        if 0 <= idx < len(papers):
                            indices.append(idx)
                        else:
                            console.print(warning_msg(f"Invalid index: {idx}"))
                    
                    if indices:
                        selected_indices = indices
                        break
                    else:
                        console.print(warning_msg("No valid indices selected"))
                        
            except ValueError:
                console.print(warning_msg("Invalid input. Use numbers separated by commas."))
                
        if not selected_indices:
            console.print(info_msg("No papers selected"))
            return
            
        console.print(info_msg(f"Selected {len(selected_indices)} papers for download"))
        
        # Check for duplicates
        file_tracker = await self._get_file_tracker()
        if file_tracker:
            new_papers = []
            existing_papers = []
            
            for idx in selected_indices:
                paper = papers[idx]
                if await file_tracker.is_arxiv_paper_downloaded(paper["id"]):
                    existing_papers.append(paper)
                else:
                    new_papers.append((idx, paper))
                    
            if existing_papers:
                console.print(warning_msg(f"Skipping {len(existing_papers)} already downloaded papers"))
                
            if not new_papers:
                console.print(info_msg("All selected papers are already downloaded"))
                return
                
            selected_indices = [idx for idx, _ in new_papers]
            
        # Download papers
        arxiv_client = await self._get_arxiv_client()
        if not arxiv_client:
            return
            
        downloaded_paths = await arxiv_client.download_multiple_papers(papers, selected_indices)
        
        # Record downloads
        if file_tracker:
            for i, idx in enumerate(selected_indices):
                if i < len(downloaded_paths):
                    paper = papers[idx]
                    await file_tracker.record_arxiv_paper(
                        paper["id"],
                        paper["title"],
                        paper["authors"],
                        paper["abstract"],
                        paper["published"],
                        str(downloaded_paths[i]),
                        ""  # Hash will be calculated when needed
                    )
                    
        # Ask about inserting into knowledge graph
        if downloaded_paths and Confirm.ask("Insert downloaded papers into the knowledge graph?"):
            from .insert import InsertCommand
            insert_cmd = InsertCommand(self.session_manager)
            await insert_cmd.insert_multiple_files(downloaded_paths)
            
    async def _get_arxiv_client(self) -> Optional[ArxivClient]:
        """Get or create an ArXiv client"""
        if self._arxiv_client is not None:
            return self._arxiv_client
            
        working_dir = self.session_manager.get_current_working_dir()
        if not working_dir:
            console.print(error_msg("Error: No current session working directory"))
            return None
            
        # Create downloads subdirectory
        download_dir = working_dir / "downloads"
        download_dir.mkdir(exist_ok=True)
        
        self._arxiv_client = ArxivClient(download_dir)
        return self._arxiv_client
        
    async def _get_file_tracker(self) -> Optional[FileTracker]:
        """Get or create a FileTracker instance"""
        if self._file_tracker is not None:
            return self._file_tracker
            
        working_dir = self.session_manager.get_current_working_dir()
        if not working_dir:
            console.print(error_msg("Error: No current session working directory"))
            return None
            
        try:
            self._file_tracker = FileTracker(working_dir)
            await self._file_tracker.init_database()
            return self._file_tracker
        except Exception as e:
            console.print(error_msg(f"Error initializing file tracker: {e}"))
            return None
            
    def reset_instance(self):
        """Reset instances (useful when switching sessions)"""
        self._arxiv_client = None
        self._file_tracker = None