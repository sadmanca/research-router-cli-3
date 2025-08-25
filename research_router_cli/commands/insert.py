"""Insert command for adding PDFs to knowledge graphs"""

import asyncio
from pathlib import Path
from typing import Union, List, Optional
import glob

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from ..utils.file_processor import FileProcessor
from ..utils.file_tracker import FileTracker
from ..utils.colors import console, success_msg, error_msg, warning_msg, info_msg, progress_msg, document_status_msg
from .session import SessionManager

class InsertCommand:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.file_processor = FileProcessor()
        self._graphrag_instance = None
        self._file_tracker = None
        
    async def insert_file(self, file_path: Union[str, Path]):
        """Insert a single file (PDF or text) into the current session's knowledge graph"""
        if not self.session_manager.ensure_session():
            return
            
        path = Path(file_path)
        
        # Handle different input types
        if path.is_dir():
            await self._insert_directory(path)
        elif self.file_processor.is_supported_file(path):
            await self._insert_single_file(path)
        elif '*' in str(path) or '?' in str(path):
            await self._insert_glob_pattern(str(path))
        else:
            console.print(error_msg(f"Error: {path} is not a supported file type or directory"))
            
    async def insert_multiple_files(self, file_paths: List[Union[str, Path]]):
        """Insert multiple PDF files at once"""
        if not self.session_manager.ensure_session():
            return
            
        # Filter for valid files
        valid_paths = []
        for file_path in file_paths:
            path = Path(file_path)
            if self.file_processor.is_supported_file(path):
                valid_paths.append(path)
            else:
                console.print(warning_msg(f"Skipping unsupported file: {path}"))
                
        if not valid_paths:
            console.print(error_msg("No valid files found"))
            return
            
        console.print(info_msg(f"Processing {len(valid_paths)} files..."))
        
        # Get file tracker
        file_tracker = await self._get_file_tracker()
        if not file_tracker:
            return
            
        # Check for duplicates
        new_files = []
        duplicates = []
        
        for file_path in valid_paths:
            file_hash = self.file_processor.get_file_hash(file_path)
            if await file_tracker.is_file_inserted(file_hash):
                duplicate_info = await file_tracker.get_duplicate_info(file_hash)
                duplicates.append((file_path, duplicate_info))
            else:
                new_files.append(file_path)
                
        # Show duplicate summary
        if duplicates:
            console.print(warning_msg(f"Found {len(duplicates)} duplicate files:"))
            for file_path, dup_info in duplicates:
                duplicate_msg = f'already inserted as {dup_info["filename"]}'
                console.print(f"  {document_status_msg(file_path.name, 'duplicate', duplicate_msg)}")
                
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
        
        for file_path in new_files:
            try:
                success = await self._insert_single_file_with_tracking(file_path)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                console.print(error_msg(f"Error processing {file_path.name}: {e}"))
                failed += 1
                
        # Show summary
        console.print(success_msg(f"Successfully inserted {successful} files"))
        if failed > 0:
            console.print(error_msg(f"Failed to insert {failed} files"))
            
    async def insert_folder(self, folder_path: Union[str, Path], recursive: bool = False):
        """Insert all supported files from a folder"""
        if not self.session_manager.ensure_session():
            return
            
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            console.print(error_msg(f"Invalid folder: {folder}"))
            return
            
        # Find supported files
        supported_files = self.file_processor.find_files_in_directory(folder, recursive)
        
        if not supported_files:
            console.print(warning_msg("No supported files found"))
            return
            
        await self.insert_multiple_files(supported_files)
            
    async def _insert_single_file(self, file_path: Path):
        """Insert a single file (legacy method, now uses tracking)"""
        return await self._insert_single_file_with_tracking(file_path)
        
    async def _insert_single_file_with_tracking(self, file_path: Path) -> bool:
        """Insert a single file with duplicate tracking"""
        if not self.file_processor.is_supported_file(file_path):
            console.print(error_msg(f"Error: {file_path} is not a supported file type"))
            return False
            
        # Get file tracker
        file_tracker = await self._get_file_tracker()
        if not file_tracker:
            return False
            
        # Check for duplicates
        file_hash = self.file_processor.get_file_hash(file_path)
        if await file_tracker.is_file_inserted(file_hash):
            duplicate_info = await file_tracker.get_duplicate_info(file_hash)
            duplicate_msg = f'already inserted as {duplicate_info["filename"]}'
            console.print(document_status_msg(
                file_path.name, 
                'duplicate', 
                duplicate_msg
            ))
            return False
            
        console.print(progress_msg(f"Processing {file_path.name}..."))
        
        # Get file metadata
        file_info = self.file_processor.get_file_info(file_path)
        if "error" in file_info:
            console.print(error_msg(f"Failed to read file metadata: {file_info['error']}"))
            return False
        
        # Extract text from file
        text = self.file_processor.extract_text_from_file(file_path)
        if not text:
            console.print(error_msg(f"Failed to extract text from {file_path.name}"))
            # Record failed extraction
            await file_tracker.record_file_insertion(
                file_path.name,
                str(file_path),
                file_hash,
                file_info.get("size_bytes", 0),
                file_info.get("pages", 0) or file_info.get("lines", 0),
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
                file_path.name,
                str(file_path),
                file_hash,
                file_info.get("size_bytes", 0),
                file_info.get("pages", 0) or file_info.get("lines", 0),
                "success",
                file_info
            )
            
            pages_or_lines = file_info.get("pages", 0) or file_info.get("lines", 0)
            unit = "pages" if file_info.get("pages") else "lines"
            console.print(document_status_msg(file_path.name, 'success', f'{pages_or_lines} {unit}'))
            return True
            
        except Exception as e:
            console.print(error_msg(f"Error inserting {file_path.name}: {e}"))
            # Record failed insertion
            await file_tracker.record_file_insertion(
                file_path.name,
                str(file_path),
                file_hash,
                file_info.get("size_bytes", 0),
                file_info.get("pages", 0) or file_info.get("lines", 0),
                "insertion_failed"
            )
            return False
            
    async def _insert_directory(self, directory: Path):
        """Insert all supported files from a directory"""
        supported_files = self.file_processor.find_files_in_directory(directory)
        
        if not supported_files:
            console.print(f"[yellow]No supported files found in {directory}[/yellow]")
            return
            
        console.print(f"[blue]Found {len(supported_files)} supported files to process[/blue]")
        
        # Extract text from all files
        file_texts = self.file_processor.extract_text_from_multiple_files(supported_files)
        
        if not file_texts:
            console.print("[yellow]No text extracted from any files[/yellow]")
            return
            
        # Combine all texts
        combined_text = self.file_processor.combine_file_texts(file_texts)
        
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
                
            console.print(f"[green]✓ Successfully inserted {len(file_texts)} files into knowledge graph[/green]")
            
        except Exception as e:
            console.print(f"[red]Error inserting directory contents: {e}[/red]")
            
    async def _insert_glob_pattern(self, pattern: str):
        """Insert files matching a glob pattern"""
        from glob import glob
        
        all_files = [Path(f) for f in glob(pattern)]
        supported_files = [f for f in all_files if self.file_processor.is_supported_file(f)]
        
        if not supported_files:
            console.print(f"[yellow]No supported files found matching pattern: {pattern}[/yellow]")
            return
            
        console.print(f"[blue]Found {len(supported_files)} supported files matching pattern[/blue]")
        
        # Extract text from all matching files
        file_texts = self.file_processor.extract_text_from_multiple_files(supported_files)
        
        if not file_texts:
            console.print("[yellow]No text extracted from any files[/yellow]")
            return
            
        # Combine all texts
        combined_text = self.file_processor.combine_file_texts(file_texts)
        
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
                
            console.print(f"[green]✓ Successfully inserted {len(file_texts)} files into knowledge graph[/green]")
            
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