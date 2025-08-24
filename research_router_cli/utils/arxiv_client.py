"""ArXiv API client for searching and downloading papers"""

import asyncio
import aiohttp
import aiofiles
import arxiv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union
from urllib.parse import urlparse

from .colors import console, success_msg, error_msg, warning_msg, info_msg, progress_msg
from rich.progress import Progress, DownloadColumn, TimeRemainingColumn, TransferSpeedColumn

class ArxivClient:
    def __init__(self, download_dir: Path):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
    async def search_papers(self, 
                           query: str, 
                           max_results: int = 20,
                           sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance) -> List[Dict]:
        """Search ArXiv for papers matching the query"""
        try:
            console.print(progress_msg(f"Searching ArXiv for: '{query}'"))
            
            # Create search client
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by
            )
            
            papers = []
            async_papers = []
            
            # Collect papers (arxiv library is synchronous)
            for paper in search.results():
                paper_info = {
                    "id": paper.entry_id.split('/')[-1],  # Get just the ID part
                    "title": paper.title,
                    "authors": ", ".join([author.name for author in paper.authors]),
                    "abstract": paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "updated": paper.updated.strftime("%Y-%m-%d"),
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "categories": paper.categories
                }
                papers.append(paper_info)
            
            console.print(success_msg(f"Found {len(papers)} papers"))
            return papers
            
        except Exception as e:
            console.print(error_msg(f"Error searching ArXiv: {e}"))
            return []
            
    async def download_paper(self, paper: Dict, custom_filename: Optional[str] = None) -> Optional[Path]:
        """Download a paper PDF from ArXiv"""
        try:
            arxiv_id = paper["id"]
            title = paper["title"]
            
            # Create filename
            if custom_filename:
                filename = custom_filename
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            else:
                # Clean title for filename
                clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_title = clean_title[:50] if len(clean_title) > 50 else clean_title
                filename = f"{arxiv_id}_{clean_title}.pdf"
            
            filepath = self.download_dir / filename
            
            # Check if file already exists
            if filepath.exists():
                console.print(warning_msg(f"File already exists: {filename}"))
                return filepath
                
            console.print(progress_msg(f"Downloading: {title}"))
            
            # Download the PDF
            async with aiohttp.ClientSession() as session:
                async with session.get(paper["pdf_url"]) as response:
                    if response.status == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        
                        with Progress(
                            "[progress.description]{task.description}",
                            "[progress.percentage]{task.percentage:>3.0f}%",
                            DownloadColumn(),
                            TransferSpeedColumn(),
                            TimeRemainingColumn(),
                        ) as progress:
                            task = progress.add_task(f"Downloading {filename}", total=total_size)
                            
                            async with aiofiles.open(filepath, 'wb') as f:
                                downloaded = 0
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                                    downloaded += len(chunk)
                                    progress.update(task, completed=downloaded)
                        
                        console.print(success_msg(f"Successfully downloaded: {filename}"))
                        return filepath
                    else:
                        console.print(error_msg(f"Failed to download paper (HTTP {response.status})"))
                        return None
                        
        except Exception as e:
            console.print(error_msg(f"Error downloading paper: {e}"))
            return None
            
    async def download_multiple_papers(self, papers: List[Dict], selected_indices: List[int]) -> List[Path]:
        """Download multiple selected papers"""
        downloaded_paths = []
        
        console.print(info_msg(f"Downloading {len(selected_indices)} papers..."))
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent downloads
        
        async def download_with_semaphore(paper):
            async with semaphore:
                return await self.download_paper(paper)
        
        # Download selected papers
        selected_papers = [papers[i] for i in selected_indices]
        tasks = [download_with_semaphore(paper) for paper in selected_papers]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                console.print(error_msg(f"Failed to download paper {i+1}: {result}"))
            elif result:
                downloaded_paths.append(result)
                
        console.print(success_msg(f"Successfully downloaded {len(downloaded_paths)} papers"))
        return downloaded_paths
        
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """Get paper information by ArXiv ID"""
        try:
            console.print(progress_msg(f"Fetching paper: {arxiv_id}"))
            
            search = arxiv.Search(id_list=[arxiv_id])
            papers = list(search.results())
            
            if papers:
                paper = papers[0]
                return {
                    "id": paper.entry_id.split('/')[-1],
                    "title": paper.title,
                    "authors": ", ".join([author.name for author in paper.authors]),
                    "abstract": paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "updated": paper.updated.strftime("%Y-%m-%d"),
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "categories": paper.categories
                }
            else:
                console.print(warning_msg(f"Paper not found: {arxiv_id}"))
                return None
                
        except Exception as e:
            console.print(error_msg(f"Error fetching paper: {e}"))
            return None
            
    def format_search_results(self, papers: List[Dict], selected_indices: List[int] = None) -> str:
        """Format search results for display"""
        if not papers:
            return "No papers found."
            
        selected_indices = selected_indices or []
        result_lines = []
        
        for i, paper in enumerate(papers):
            is_selected = i in selected_indices
            marker = "▶" if is_selected else " "
            
            # Truncate title if too long
            title = paper["title"]
            if len(title) > 60:
                title = title[:57] + "..."
                
            # Truncate authors if too long
            authors = paper["authors"]
            if len(authors) > 40:
                authors = authors[:37] + "..."
            
            result_lines.append(f"{marker} [{i:2d}] {title}")
            result_lines.append(f"      by {authors} • {paper['published']}")
            result_lines.append(f"      {paper['id']}")
            result_lines.append("")
            
        return "\n".join(result_lines)
        
    async def get_download_stats(self) -> Dict:
        """Get statistics about downloaded papers"""
        pdf_files = list(self.download_dir.glob("*.pdf"))
        
        total_size = sum(f.stat().st_size for f in pdf_files)
        
        return {
            "total_papers": len(pdf_files),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "download_dir": str(self.download_dir)
        }