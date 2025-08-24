"""Insert command for adding PDFs to knowledge graphs"""

import asyncio
from pathlib import Path
from typing import Union, List, Optional
import glob

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from ..utils.pdf_processor import PDFProcessor
from ..utils.file_tracker import FileTracker
from ..utils.colors import console, success_msg, error_msg, warning_msg, info_msg, progress_msg, document_status_msg
from .session import SessionManager

class InsertCommand:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.pdf_processor = PDFProcessor()
        self._graphrag_instance = None
        self._file_tracker = None
        
    async def insert_pdf(self, pdf_path: Union[str, Path]):
        """Insert a single PDF into the current session's knowledge graph"""
        if not self.session_manager.ensure_session():
            return
            
        path = Path(pdf_path)
        
        # Handle different input types
        if path.is_dir():
            await self._insert_directory(path)
        elif path.suffix.lower() == '.pdf':
            await self._insert_single_pdf(path)
        elif '*' in str(path) or '?' in str(path):
            await self._insert_glob_pattern(str(path))
        else:
            console.print(error_msg(f"Error: {path} is not a valid PDF file or directory"))
            
    async def insert_multiple_files(self, file_paths: List[Union[str, Path]]):
        """Insert multiple PDF files at once"""
        if not self.session_manager.ensure_session():
            return
            
        # Filter for valid PDF files
        pdf_paths = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.suffix.lower() == '.pdf' and path.exists():
                pdf_paths.append(path)
            else:
                console.print(warning_msg(f"Skipping invalid PDF: {path}"))
                
        if not pdf_paths:
            console.print(error_msg("No valid PDF files found"))
            return
            
        console.print(info_msg(f"Processing {len(pdf_paths)} PDF files..."))
        
        # Get file tracker
        file_tracker = await self._get_file_tracker()
        if not file_tracker:
            return
            
        # Check for duplicates
        new_files = []
        duplicates = []
        
        for pdf_path in pdf_paths:
            file_hash = self.pdf_processor.get_file_hash(pdf_path)
            if await file_tracker.is_file_inserted(file_hash):
                duplicate_info = await file_tracker.get_duplicate_info(file_hash)
                duplicates.append((pdf_path, duplicate_info))
            else:
                new_files.append(pdf_path)
                
        # Show duplicate summary
        if duplicates:
            console.print(warning_msg(f"Found {len(duplicates)} duplicate files:"))
            for pdf_path, dup_info in duplicates:
                duplicate_msg = f'already inserted as {dup_info["filename"]}'
                console.print(f"  {document_status_msg(pdf_path.name, 'duplicate', duplicate_msg)}")
                
        if not new_files:
            console.print(info_msg("All files are duplicates. No new files to insert."))
            return
            
        # Confirm insertion
        console.print(info_msg(f"Will insert {len(new_files)} new files"))
        if not Confirm.ask("Continue with insertion?"):
            console.print(info_msg("Operation cancelled"))
            return
            
        # Insert new files
        successful = 0
        failed = 0
        
        for pdf_path in new_files:
            try:
                success = await self._insert_single_pdf_with_tracking(pdf_path)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                console.print(error_msg(f"Error processing {pdf_path.name}: {e}"))
                failed += 1
                
        # Show summary
        console.print(success_msg(f"Successfully inserted {successful} files"))
        if failed > 0:
            console.print(error_msg(f"Failed to insert {failed} files"))
            
    async def insert_folder(self, folder_path: Union[str, Path], recursive: bool = False):
        """Insert all PDFs from a folder"""
        if not self.session_manager.ensure_session():
            return
            
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            console.print(error_msg(f"Invalid folder: {folder}"))
            return
            
        # Find PDF files
        if recursive:
            pdf_files = list(folder.rglob("*.pdf"))
            console.print(info_msg(f"Found {len(pdf_files)} PDF files in {folder} (recursive)"))
        else:
            pdf_files = list(folder.glob("*.pdf"))
            console.print(info_msg(f"Found {len(pdf_files)} PDF files in {folder}"))
            
        if not pdf_files:
            console.print(warning_msg("No PDF files found"))
            return
            
        await self.insert_multiple_files(pdf_files)
            
    async def _insert_single_pdf(self, pdf_path: Path):
        """Insert a single PDF file (legacy method, now uses tracking)"""
        return await self._insert_single_pdf_with_tracking(pdf_path)
        
    async def _insert_single_pdf_with_tracking(self, pdf_path: Path) -> bool:
        """Insert a single PDF file with duplicate tracking"""
        if not self.pdf_processor.is_supported_file(pdf_path):
            console.print(error_msg(f"Error: {pdf_path} is not a valid PDF file"))
            return False
            
        # Get file tracker
        file_tracker = await self._get_file_tracker()
        if not file_tracker:
            return False
            
        # Check for duplicates
        file_hash = self.pdf_processor.get_file_hash(pdf_path)
        if await file_tracker.is_file_inserted(file_hash):
            duplicate_info = await file_tracker.get_duplicate_info(file_hash)
            duplicate_msg = f'already inserted as {duplicate_info["filename"]}'
            console.print(document_status_msg(
                pdf_path.name, 
                'duplicate', 
                duplicate_msg
            ))
            return False
            
        console.print(progress_msg(f"Processing {pdf_path.name}..."))
        
        # Get PDF metadata
        pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
        if "error" in pdf_info:
            console.print(error_msg(f"Failed to read PDF metadata: {pdf_info['error']}"))
            return False
        
        # Extract text from PDF
        text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        if not text:
            console.print(error_msg(f"Failed to extract text from {pdf_path.name}"))
            # Record failed extraction
            await file_tracker.record_file_insertion(
                pdf_path.name,
                str(pdf_path),
                file_hash,
                pdf_info.get("size_bytes", 0),
                pdf_info.get("pages", 0),
                "extraction_failed"
            )
            return False
            
        # Get or create GraphRAG instance
        graphrag = await self._get_graphrag_instance()
        if not graphrag:
            return False
            
        # Insert into knowledge graph
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Building knowledge graph...", total=None)
                
                await graphrag.ainsert(text)
                
            # Record successful insertion
            await file_tracker.record_file_insertion(
                pdf_path.name,
                str(pdf_path),
                file_hash,
                pdf_info.get("size_bytes", 0),
                pdf_info.get("pages", 0),
                "success",
                pdf_info
            )
            
            console.print(document_status_msg(pdf_path.name, 'success', f'{pdf_info.get("pages", 0)} pages'))
            return True
            
        except Exception as e:
            console.print(error_msg(f"Error inserting {pdf_path.name}: {e}"))
            # Record failed insertion
            await file_tracker.record_file_insertion(
                pdf_path.name,
                str(pdf_path),
                file_hash,
                pdf_info.get("size_bytes", 0),
                pdf_info.get("pages", 0),
                "insertion_failed"
            )
            return False
            
    async def _insert_directory(self, directory: Path):
        """Insert all PDFs from a directory"""
        pdf_files = self.pdf_processor.find_pdfs_in_directory(directory)
        
        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {directory}[/yellow]")
            return
            
        console.print(f"[blue]Found {len(pdf_files)} PDF files to process[/blue]")
        
        # Extract text from all PDFs
        pdf_texts = self.pdf_processor.extract_text_from_multiple_pdfs(pdf_files)
        
        if not pdf_texts:
            console.print("[yellow]No text extracted from any PDF files[/yellow]")
            return
            
        # Combine all texts
        combined_text = self.pdf_processor.combine_pdf_texts(pdf_texts)
        
        # Get or create GraphRAG instance
        graphrag = await self._get_graphrag_instance()
        if not graphrag:
            return
            
        # Insert into knowledge graph
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Building knowledge graph from directory...", total=None)
                
                await graphrag.ainsert(combined_text)
                
            console.print(f"[green]✓ Successfully inserted {len(pdf_texts)} PDF files into knowledge graph[/green]")
            
        except Exception as e:
            console.print(f"[red]Error inserting directory contents: {e}[/red]")
            
    async def _insert_glob_pattern(self, pattern: str):
        """Insert PDFs matching a glob pattern"""
        from glob import glob
        
        pdf_files = [Path(f) for f in glob(pattern) if Path(f).suffix.lower() == '.pdf']
        
        if not pdf_files:
            console.print(f"[yellow]No PDF files found matching pattern: {pattern}[/yellow]")
            return
            
        console.print(f"[blue]Found {len(pdf_files)} PDF files matching pattern[/blue]")
        
        # Extract text from all matching PDFs
        pdf_texts = self.pdf_processor.extract_text_from_multiple_pdfs(pdf_files)
        
        if not pdf_texts:
            console.print("[yellow]No text extracted from any PDF files[/yellow]")
            return
            
        # Combine all texts
        combined_text = self.pdf_processor.combine_pdf_texts(pdf_texts)
        
        # Get or create GraphRAG instance
        graphrag = await self._get_graphrag_instance()
        if not graphrag:
            return
            
        # Insert into knowledge graph
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Building knowledge graph from pattern match...", total=None)
                
                await graphrag.ainsert(combined_text)
                
            console.print(f"[green]✓ Successfully inserted {len(pdf_texts)} PDF files into knowledge graph[/green]")
            
        except Exception as e:
            console.print(f"[red]Error inserting pattern matches: {e}[/red]")
            
    async def _get_graphrag_instance(self):
        """Get or create a GraphRAG instance for the current session"""
        if self._graphrag_instance is not None:
            return self._graphrag_instance
            
        working_dir = self.session_manager.get_current_working_dir()
        if not working_dir:
            console.print("[red]Error: No current session working directory[/red]")
            return None
            
        try:
            # Import nano-graphrag
            from nano_graphrag import GraphRAG
            from ..utils.config import Config
            
            config = Config()
            if not config.validate_config():
                return None
                
            # Create GraphRAG instance
            llm_config = config.get_llm_config()
            
            self._graphrag_instance = GraphRAG(
                working_dir=str(working_dir),
                enable_llm_cache=True,
                **llm_config
            )
            
            console.print(f"[blue]Initialized knowledge graph in {working_dir}[/blue]")
            return self._graphrag_instance
            
        except ImportError as e:
            console.print(f"[red]Error: nano-graphrag not available: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Error initializing GraphRAG: {e}[/red]")
            return None
            
    async def _get_file_tracker(self) -> Optional[FileTracker]:
        """Get or create a FileTracker instance for the current session"""
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
        """Reset the GraphRAG and FileTracker instances (useful when switching sessions)"""
        self._graphrag_instance = None
        self._file_tracker = None