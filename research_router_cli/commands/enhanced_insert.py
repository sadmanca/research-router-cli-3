"""Enhanced Insert command using new EnhancedKG generation"""

import asyncio
from pathlib import Path
from typing import Union, List, Optional, Dict
import glob

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from ..utils.pdf_processor import PDFProcessor
from ..utils.file_tracker import FileTracker
from ..utils.colors import console, success_msg, error_msg, warning_msg, info_msg, progress_msg, document_status_msg
from .session import SessionManager


class EnhancedInsertCommand:
    """Enhanced insert command using genkg.py methods with advanced visualization and nano-graphrag integration"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.pdf_processor = PDFProcessor()
        self._genkg_instance = None
        self._graphrag_instance = None
        self._file_tracker = None
        
    async def enhanced_insert_pdf(self, pdf_path: Union[str, Path], nodes_per_paper: int = 25, 
                                 export_formats: List[str] = None):
        """Insert a single PDF using enhanced knowledge graph generation"""
        if not self.session_manager.ensure_session():
            return
            
        if export_formats is None:
            export_formats = ['html', 'json']  # Default formats
            
        path = Path(pdf_path)
        
        # Handle different input types
        if path.is_dir():
            await self._enhanced_insert_directory(path, nodes_per_paper, export_formats)
        elif path.suffix.lower() == '.pdf':
            await self._enhanced_insert_single_pdf(path, nodes_per_paper, export_formats)
        elif '*' in str(path) or '?' in str(path):
            await self._enhanced_insert_glob_pattern(str(path), nodes_per_paper, export_formats)
        else:
            console.print(error_msg(f"Error: {path} is not a valid PDF file or directory"))
            
    async def enhanced_insert_multiple_files(self, file_paths: List[Union[str, Path]], 
                                           nodes_per_paper: int = 25, export_formats: List[str] = None):
        """Insert multiple PDF files using enhanced knowledge graph generation"""
        if not self.session_manager.ensure_session():
            return
            
        if export_formats is None:
            export_formats = ['html', 'json']
            
        # Filter for valid PDF files
        pdf_paths = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.suffix.lower() == '.pdf' and path.exists():
                pdf_paths.append(path)
            else:
                console.print(warning_msg(f"Skipping non-PDF or missing file: {path}"))
                
        if not pdf_paths:
            console.print(error_msg("No valid PDF files found"))
            return
            
        console.print(info_msg(f"Processing {len(pdf_paths)} PDF files with enhanced knowledge graph generation..."))
        
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
        console.print(info_msg(f"Will insert {len(new_files)} new files using enhanced knowledge graph generation"))
        if not Confirm.ask("Continue with enhanced insertion?"):
            console.print(info_msg("Operation cancelled"))
            return
            
        # Process files with enhanced KG
        try:
            await self._process_files_with_enhanced_kg(new_files, nodes_per_paper, export_formats)
            
            # Record successful insertions
            for pdf_path in new_files:
                try:
                    file_hash = self.pdf_processor.get_file_hash(pdf_path)
                    # Get file size
                    size_bytes = pdf_path.stat().st_size if pdf_path.exists() else 0
                    await file_tracker.record_file_insertion(
                        filename=pdf_path.name,
                        filepath=str(pdf_path),
                        file_hash=file_hash,
                        size_bytes=size_bytes
                    )
                except Exception as e:
                    console.print(warning_msg(f"Failed to record insertion for {pdf_path.name}: {e}"))
                    
            console.print(success_msg(f"Successfully processed {len(new_files)} files with enhanced knowledge graph"))
            
        except Exception as e:
            console.print(error_msg(f"Error during enhanced insertion: {e}"))

    async def _enhanced_insert_single_pdf(self, pdf_path: Path, nodes_per_paper: int, export_formats: List[str]):
        """Insert single PDF with enhanced KG generation"""
        console.print(info_msg(f"Processing {pdf_path.name} with enhanced knowledge graph generation..."))
        
        try:
            # Extract text from PDF
            pdf_texts = self.pdf_processor.extract_text_from_multiple_pdfs([pdf_path])
            if not pdf_texts:
                console.print(error_msg(f"Failed to extract text from {pdf_path.name}"))
                return False
                
            # Use enhanced KG generation
            await self._process_files_with_enhanced_kg([pdf_path], nodes_per_paper, export_formats, pdf_texts)
            return True
            
        except Exception as e:
            console.print(error_msg(f"Error processing {pdf_path.name}: {e}"))
            return False

    async def _enhanced_insert_directory(self, dir_path: Path, nodes_per_paper: int, export_formats: List[str]):
        """Insert all PDFs in directory with enhanced KG generation"""
        pdf_files = list(dir_path.glob("*.pdf"))
        if not pdf_files:
            console.print(warning_msg(f"No PDF files found in {dir_path}"))
            return
            
        console.print(info_msg(f"Found {len(pdf_files)} PDF files in {dir_path}"))
        await self.enhanced_insert_multiple_files(pdf_files, nodes_per_paper, export_formats)

    async def _enhanced_insert_glob_pattern(self, pattern: str, nodes_per_paper: int, export_formats: List[str]):
        """Insert files matching glob pattern with enhanced KG generation"""
        pdf_files = [Path(f) for f in glob.glob(pattern) if f.lower().endswith('.pdf')]
        if not pdf_files:
            console.print(warning_msg(f"No PDF files found matching pattern: {pattern}"))
            return
            
        console.print(info_msg(f"Found {len(pdf_files)} PDF files matching pattern"))
        await self.enhanced_insert_multiple_files(pdf_files, nodes_per_paper, export_formats)

    async def _process_files_with_enhanced_kg(self, pdf_paths: List[Path], nodes_per_paper: int, 
                                            export_formats: List[str], pdf_texts: Dict[Path, str] = None):
        """Process files using genkg.py methods and create nano-graphrag storage"""
        
        # Get or create genkg instance for enhanced visualization
        genkg = await self._get_genkg_instance()
        if not genkg:
            return
            
        # Get or create GraphRAG instance for nano-graphrag storage
        graphrag = await self._get_graphrag_instance()
        if not graphrag:
            return
            
        # Extract text if not provided
        if pdf_texts is None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Extracting text from PDFs...", total=None)
                pdf_texts = self.pdf_processor.extract_text_from_multiple_pdfs(pdf_paths)
                
        if not pdf_texts:
            console.print(error_msg("Failed to extract text from PDF files"))
            return
            
        # Convert paths to strings for compatibility
        paper_paths = [str(path) for path in pdf_paths]
        paper_texts = {str(path): text for path, text in pdf_texts.items()}
        
        # First, insert into nano-graphrag to create the core storage files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Creating nano-graphrag knowledge graph...", total=None)
            
            # Combine all texts for nano-graphrag insertion
            combined_text = '\n\n'.join(pdf_texts.values())
            await graphrag.ainsert(combined_text)
        
        console.print(success_msg("Nano-graphrag knowledge graph created successfully!"))
        
        # Then generate enhanced visualization using genkg.py
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Generating enhanced visualization...", total=None)
            
            # Set output paths
            working_dir = self.session_manager.get_current_working_dir()
            output_html = working_dir / "enhanced_knowledge_graph.html"
            output_json = working_dir / "enhanced_knowledge_graph.dashkg.json"
            
            # Generate the enhanced visualization using genkg
            knowledge_graph = genkg.generate_knowledge_graph(
                paper_paths=paper_paths,
                paper_texts=paper_texts,
                nodes_per_paper=nodes_per_paper,
                output_path=str(output_html) if 'html' in export_formats else None,
                display=False,
                advanced_visualization=True
            )
            
            if knowledge_graph is None:
                console.print(error_msg("Failed to generate enhanced visualization"))
                return
                
        # Show generation summary
        console.print(success_msg("Enhanced visualization generated successfully!"))
        console.print(info_msg(f"Visualization Nodes: {len(knowledge_graph.nodes)}"))
        console.print(info_msg(f"Visualization Edges: {len(knowledge_graph.edges)}"))
        
        if 'html' in export_formats and output_html.exists():
            console.print(info_msg(f"Interactive visualization: {output_html}"))
            
        if 'json' in export_formats and output_json.exists():
            console.print(info_msg(f"Graph data export: {output_json}"))

    async def _get_genkg_instance(self):
        """Get or create a genkg.py GenerateKG instance"""
        if self._genkg_instance is not None:
            return self._genkg_instance
            
        try:
            # Import genkg
            import sys
            from pathlib import Path
            nano_graphrag_path = Path(__file__).parent.parent.parent / "nano-graphrag"
            if str(nano_graphrag_path) not in sys.path:
                sys.path.insert(0, str(nano_graphrag_path))
            
            from genkg import GenerateKG
            from ..utils.config import Config
            
            config = Config()
            
            # Check for required API keys
            if not config.has_gemini_config:
                console.print(error_msg("Enhanced visualization requires Gemini API key"))
                console.print(info_msg("Set GEMINI_API_KEY in your environment or .env file"))
                return None
            
            # Create genkg instance
            self._genkg_instance = GenerateKG(
                llm_provider="gemini", 
                model_name="gemini-2.5-flash"
            )
            
            console.print(info_msg("Initialized enhanced visualization generator"))
            return self._genkg_instance
            
        except ImportError as e:
            console.print(error_msg(f"genkg.py not available: {e}"))
            console.print(info_msg("Install required dependencies: pip install sentence-transformers pyvis google-genai keybert spacy transformers"))
            return None
        except Exception as e:
            console.print(error_msg(f"Error initializing genkg: {e}"))
            return None
            
    async def _get_graphrag_instance(self):
        """Get or create a GraphRAG instance for nano-graphrag storage"""
        if self._graphrag_instance is not None:
            return self._graphrag_instance
            
        working_dir = self.session_manager.get_current_working_dir()
        if not working_dir:
            console.print(error_msg("No active session"))
            return None
            
        try:
            # Import nano-graphrag
            from nano_graphrag import GraphRAG
            from ..utils.config import Config
            
            config = Config()
            if not config.validate_config():
                return None
                
            # Create GraphRAG instance with genkg-based node/edge generation
            llm_config = config.get_llm_config()
            
            self._graphrag_instance = GraphRAG(
                working_dir=str(working_dir),
                enable_llm_cache=True,
                use_gemini_extraction=True,  # Enable genkg-based extraction
                gemini_node_limit=25,
                gemini_model_name="models/gemini-2.5-flash",
                gemini_combine_with_llm_extraction=False,  # Use only genkg, not default LLM extraction
                **llm_config
            )
            
            console.print(info_msg(f"Initialized nano-graphrag instance in {working_dir}"))
            return self._graphrag_instance
            
        except ImportError as e:
            console.print(error_msg(f"nano-graphrag not available: {e}"))
            return None
        except Exception as e:
            console.print(error_msg(f"Error initializing GraphRAG: {e}"))
            return None

    async def _get_file_tracker(self):
        """Get or create a FileTracker instance for the current session"""
        if self._file_tracker is not None:
            return self._file_tracker
            
        working_dir = self.session_manager.get_current_working_dir()
        if not working_dir:
            console.print(error_msg("No active session"))
            return None
            
        try:
            self._file_tracker = FileTracker(working_dir)
            await self._file_tracker.init_database()
            return self._file_tracker
        except Exception as e:
            console.print(error_msg(f"Error initializing file tracker: {e}"))
            return None
    
    def reset_instance(self):
        """Reset the genkg, GraphRAG and FileTracker instances (useful when switching sessions)"""
        self._genkg_instance = None
        self._graphrag_instance = None
        self._file_tracker = None

    async def show_enhanced_options(self):
        """Show available enhanced knowledge graph options"""
        console.print("\n[bold cyan]Enhanced Knowledge Graph Generation[/bold cyan]")
        console.print("Advanced features available:")
        console.print("  - [green]Smart paper summarization[/green] - Better concept extraction")
        console.print("  - [green]Semantic similarity analysis[/green] - Improved edge connections")
        console.print("  - [green]Interactive visualizations[/green] - Rich HTML graphs with tooltips")
        console.print("  - [green]Graph connectivity analysis[/green] - Ensures connected components")
        console.print("  - [green]Multiple export formats[/green] - HTML and .dashkg.json")
        console.print("  - [green]Advanced relationship detection[/green] - Context-aware edge creation")
        console.print("  - [green]Nano-graphrag integration[/green] - Full query support after insertion")
        
        console.print("\n[bold yellow]Usage Examples:[/bold yellow]")
        console.print("  enhanced-insert paper.pdf")
        console.print("  enhanced-insert *.pdf --nodes 30")
        console.print("  enhanced-insert /path/to/papers/")
        
        # Check configuration
        from ..utils.config import Config
        config = Config()
        
        if config.has_gemini_config:
            console.print("\n[green]Gemini API configured - Enhanced features available[/green]")
        else:
            console.print("\n[red]Gemini API not configured[/red]")
            console.print("Set GEMINI_API_KEY to use enhanced features")