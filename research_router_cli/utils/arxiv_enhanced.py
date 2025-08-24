"""Enhanced ArXiv integration with better search and selection UX"""

import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import IntPrompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.columns import Columns
from rich.text import Text

from .colors import console, info_msg, warning_msg, error_msg, success_msg, highlight_msg
from .terminal_utils import clear_and_header, transition_to_view, clear_screen_smart
from .arxiv_client import ArxivClient


@dataclass
class ArxivPaper:
    """Enhanced ArXiv paper representation"""
    id: str
    title: str
    authors: List[str]
    summary: str
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: str
    relevance_score: float = 0.0
    is_recent: bool = False
    download_size: Optional[str] = None


class EnhancedArxivClient:
    """Enhanced ArXiv client with better UX and smart features"""
    
    def __init__(self, download_dir):
        self.arxiv_client = ArxivClient(download_dir)
        self.last_search_results: List[ArxivPaper] = []
        self.search_cache: Dict[str, List[ArxivPaper]] = {}
        self.max_cache_size = 10
        
    async def smart_search(self, query: str, max_results: int = 20) -> List[ArxivPaper]:
        """Enhanced search with better ranking and filtering"""
        
        # Check cache first
        cache_key = f"{query.lower()}:{max_results}"
        if cache_key in self.search_cache:
            console.print(info_msg("Using cached results"))
            return self.search_cache[cache_key]
        
        console.print(info_msg(f"Searching ArXiv for: '{query}'"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Searching ArXiv...", total=None)
            
            # Use the existing arxiv client
            results = await self.arxiv_client.search_papers(
                query=query,
                max_results=max_results * 2  # Get more results for better ranking
            )
            
        if not results:
            console.print(warning_msg("No papers found"))
            return []
            
        # Convert to enhanced format with scoring
        papers = []
        now = datetime.now()
        
        for result in results[:max_results]:  # Limit to requested amount
            # Handle date conversion - ArXiv client returns string dates
            published_str = result.get('published', '')
            updated_str = result.get('updated', '')
            
            try:
                published_date = datetime.strptime(published_str, '%Y-%m-%d') if published_str else now
            except ValueError:
                published_date = now
                
            try:
                updated_date = datetime.strptime(updated_str, '%Y-%m-%d') if updated_str else now
            except ValueError:
                updated_date = now
                
            # Convert authors from string to list if needed
            authors = result.get('authors', [])
            if isinstance(authors, str):
                authors = [name.strip() for name in authors.split(',')]
            
            paper = ArxivPaper(
                id=result.get('id', ''),
                title=result.get('title', ''),
                authors=authors,
                summary=result.get('abstract', result.get('summary', '')),  # Handle both field names
                published=published_date,
                updated=updated_date,
                categories=result.get('categories', []),
                pdf_url=result.get('pdf_url', ''),
                is_recent=self._is_recent(published_date)
            )
            
            # Calculate relevance score
            paper.relevance_score = self._calculate_relevance(paper, query)
            papers.append(paper)
            
        # Sort by relevance
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        # Cache results
        self.search_cache[cache_key] = papers
        if len(self.search_cache) > self.max_cache_size:
            # Remove oldest cache entry
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
            
        self.last_search_results = papers
        return papers
        
    def display_search_results(self, papers: List[ArxivPaper], show_details: bool = False):
        """Display search results in an enhanced table format"""
        if not papers:
            console.print(warning_msg("No papers to display"))
            return
            
        # Create table
        table = Table(title=f"ArXiv Search Results ({len(papers)} papers)")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan", min_width=40)
        table.add_column("Authors", style="green", width=25)
        table.add_column("Date", style="magenta", width=10)
        table.add_column("Score", style="yellow", width=6)
        
        if show_details:
            table.add_column("Categories", style="dim", width=15)
            
        for i, paper in enumerate(papers, 1):
            # Format title with recent indicator
            title = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
            if paper.is_recent:
                title = f"🔥 {title}"
                
            # Format authors
            authors_str = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += f" +{len(paper.authors)-2}"
                
            # Format date
            date_str = paper.published.strftime("%Y-%m-%d")
            
            # Format score
            score_str = f"{paper.relevance_score:.1f}"
            
            row = [str(i), title, authors_str, date_str, score_str]
            
            if show_details:
                cats = ", ".join(paper.categories[:2])
                if len(paper.categories) > 2:
                    cats += "..."
                row.append(cats)
                
            table.add_row(*row)
            
        console.print(table)
        
    async def interactive_selection(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """Interactive paper selection with preview"""
        if not papers:
            return []
            
        selected_papers = []
        
        while True:
            clear_and_header(console, "📋 Interactive Paper Selection", f"{len(papers)} papers found • Select papers to download")
            
            console.print("[bold yellow]📝 Commands:[/bold yellow]")
            console.print("  • [bold]Number[/bold] - Select/deselect paper")
            console.print("  • [bold]p <number>[/bold] - Preview paper details") 
            console.print("  • [bold]d[/bold] - Download selected papers")
            console.print("  • [bold]q[/bold] - Quit without downloading")
            console.print()
            
            self.display_search_results(papers)
            
            if selected_papers:
                console.print(success_msg(f"Selected {len(selected_papers)} papers:"))
                for paper in selected_papers:
                    console.print(f"  • {paper.title[:60]}...")
                    
            try:
                choice = console.input("\nYour choice: ").strip().lower()
                
                if choice == 'q':
                    return []
                elif choice == 'd':
                    return selected_papers
                elif choice.startswith('p '):
                    # Preview paper
                    try:
                        paper_num = int(choice[2:])
                        if 1 <= paper_num <= len(papers):
                            self._preview_paper(papers[paper_num - 1])
                        else:
                            console.print(error_msg("Invalid paper number"))
                    except ValueError:
                        console.print(error_msg("Invalid format. Use: p <number>"))
                elif choice.isdigit():
                    # Select paper
                    paper_num = int(choice)
                    if 1 <= paper_num <= len(papers):
                        paper = papers[paper_num - 1]
                        if paper not in selected_papers:
                            selected_papers.append(paper)
                            console.print(success_msg(f"Added: {paper.title[:50]}..."))
                        else:
                            selected_papers.remove(paper)
                            console.print(info_msg(f"Removed: {paper.title[:50]}..."))
                    else:
                        console.print(error_msg("Invalid paper number"))
                else:
                    console.print(warning_msg("Invalid command"))
                    
            except KeyboardInterrupt:
                return []
                
    def _preview_paper(self, paper: ArxivPaper):
        """Show detailed preview of a paper"""
        clear_and_header(console, "📄 Paper Preview", "Detailed view of selected paper")
        
        # Create rich preview
        preview_content = [
            f"[bold cyan]Title:[/bold cyan] {paper.title}",
            f"[bold green]Authors:[/bold green] {', '.join(paper.authors)}",
            f"[bold magenta]Published:[/bold magenta] {paper.published.strftime('%Y-%m-%d')}",
            f"[bold yellow]Categories:[/bold yellow] {', '.join(paper.categories)}",
            f"[bold blue]ArXiv ID:[/bold blue] {paper.id}",
            "",
            f"[bold]Abstract:[/bold]",
            self._format_abstract(paper.summary),
        ]
        
        preview_panel = Panel(
            "\n".join(preview_content),
            title="Paper Preview",
            border_style="cyan",
            padding=(1, 2)
        )
        
        console.print(preview_panel)
        console.input("\nPress Enter to return to selection...")
        
    def _format_abstract(self, abstract: str, max_length: int = 800) -> str:
        """Format abstract for display"""
        if len(abstract) <= max_length:
            return abstract
            
        # Find good break point
        truncated = abstract[:max_length]
        last_sentence = truncated.rfind('.')
        
        if last_sentence > max_length * 0.7:  # If we found a good break point
            return truncated[:last_sentence + 1] + "\n\n[dim]...(abstract truncated)[/dim]"
        else:
            return truncated + "...\n\n[dim](abstract truncated)[/dim]"
            
    async def bulk_download_with_progress(self, papers: List[ArxivPaper]) -> List[str]:
        """Download multiple papers with enhanced progress tracking"""
        if not papers:
            return []
            
        console.print(success_msg(f"Starting download of {len(papers)} papers..."))
        
        downloaded_files = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            
            main_task = progress.add_task("Overall Progress", total=len(papers))
            
            for i, paper in enumerate(papers, 1):
                # Update main progress
                progress.update(main_task, description=f"Downloading {i}/{len(papers)}: {paper.title[:30]}...")
                
                try:
                    # Get paper info and download
                    paper_dict = self.arxiv_client.get_paper_by_id(paper.id)
                    if paper_dict:
                        file_path = await self.arxiv_client.download_paper(paper_dict)
                        
                        if file_path:
                            downloaded_files.append(str(file_path))
                            console.print(success_msg(f"✓ Downloaded: {paper.title[:50]}..."))
                        else:
                            console.print(error_msg(f"✗ Failed: {paper.title[:50]}..."))
                    else:
                        console.print(error_msg(f"✗ Could not find paper: {paper.id}"))
                        
                except Exception as e:
                    console.print(error_msg(f"✗ Error downloading {paper.title[:30]}...: {e}"))
                    
                progress.advance(main_task)
                
                # Small delay to be nice to ArXiv servers
                await asyncio.sleep(0.5)
                
        return downloaded_files
        
    def show_download_summary(self, downloaded_files: List[str], requested_count: int):
        """Show download summary with statistics"""
        success_count = len(downloaded_files)
        failed_count = requested_count - success_count
        
        summary_data = [
            f"[bold green]Successfully downloaded:[/bold green] {success_count}",
            f"[bold red]Failed downloads:[/bold red] {failed_count}",
            f"[bold blue]Success rate:[/bold blue] {(success_count/requested_count*100):.1f}%"
        ]
        
        if downloaded_files:
            total_size = sum(self._get_file_size(f) for f in downloaded_files)
            summary_data.append(f"[bold magenta]Total size:[/bold magenta] {self._format_size(total_size)}")
            
        summary_panel = Panel(
            "\n".join(summary_data),
            title="Download Summary",
            border_style="green" if success_count == requested_count else "yellow"
        )
        
        console.print(summary_panel)
        
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get smart search suggestions based on query"""
        suggestions = []
        
        # Common research areas
        research_areas = [
            "machine learning", "deep learning", "neural networks",
            "natural language processing", "computer vision", "robotics",
            "artificial intelligence", "reinforcement learning", "transformers",
            "graph neural networks", "attention mechanism", "language models"
        ]
        
        # Find related areas
        query_lower = query.lower()
        for area in research_areas:
            if any(word in area for word in query_lower.split()):
                suggestions.append(f'"{area}"')
                
        # Add search refinements
        suggestions.extend([
            f'"{query}" AND (survey OR review)',  # Reviews
            f'"{query}" AND recent',  # Recent papers
            f'"{query}" AND (implementation OR code)',  # With code
        ])
        
        return suggestions[:5]
        
    def _calculate_relevance(self, paper: ArxivPaper, query: str) -> float:
        """Calculate relevance score for paper"""
        score = 0.0
        query_words = query.lower().split()
        
        # Title relevance (highest weight)
        title_words = paper.title.lower().split()
        title_matches = sum(1 for word in query_words if any(word in title_word for title_word in title_words))
        score += title_matches * 3.0
        
        # Abstract relevance
        abstract_words = paper.summary.lower().split()
        abstract_matches = sum(1 for word in query_words if any(word in abs_word for abs_word in abstract_words))
        score += abstract_matches * 1.0
        
        # Recency bonus
        if paper.is_recent:
            score += 0.5
            
        # Author count (slight preference for collaborative work)
        if 2 <= len(paper.authors) <= 5:
            score += 0.2
            
        return score
        
    def _is_recent(self, published: datetime, days: int = 365) -> bool:
        """Check if paper is recent"""
        try:
            if isinstance(published, str):
                published = datetime.strptime(published, '%Y-%m-%d')
            return (datetime.now() - published).days <= days
        except (ValueError, TypeError):
            return False  # If we can't parse the date, assume it's not recent
        
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            from pathlib import Path
            return Path(file_path).stat().st_size
        except:
            return 0
            
    def _format_size(self, bytes_size: int) -> str:
        """Format file size for display"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"


def create_search_wizard() -> Dict[str, Any]:
    """Interactive search wizard for building better queries"""
    console.print(Panel(
        "[bold cyan]ArXiv Search Wizard[/bold cyan]\n"
        "Let's build a powerful search query together!",
        border_style="blue"
    ))
    
    # Collect search parameters
    search_params = {}
    
    # Main topic
    search_params['topic'] = console.input("[bold]Main research topic: [/bold]").strip()
    
    # Additional keywords
    keywords = console.input("[dim]Additional keywords (optional): [/dim]").strip()
    if keywords:
        search_params['keywords'] = [k.strip() for k in keywords.split(',')]
    
    # Time filter
    console.print("\n[bold]Time filter:[/bold]")
    console.print("1. Any time")
    console.print("2. Last year")
    console.print("3. Last 2 years")
    console.print("4. Last 5 years")
    
    time_choice = console.input("Choose (1-4): ").strip()
    time_filters = {
        '2': 'recent',
        '3': '2years',
        '4': '5years'
    }
    if time_choice in time_filters:
        search_params['time_filter'] = time_filters[time_choice]
    
    # Paper type preference
    if Confirm.ask("Prefer review/survey papers?"):
        search_params['prefer_reviews'] = True
        
    # Build final query
    query_parts = [search_params['topic']]
    
    if search_params.get('keywords'):
        query_parts.extend(search_params['keywords'])
        
    if search_params.get('prefer_reviews'):
        query_parts.append('(survey OR review)')
        
    final_query = ' AND '.join(f'"{part}"' for part in query_parts)
    
    console.print(success_msg(f"Built query: {final_query}"))
    
    return {
        'query': final_query,
        'params': search_params
    }