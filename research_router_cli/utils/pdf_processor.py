"""PDF text extraction utilities"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Union

import fitz  # PyMuPDF
from rich.progress import Progress, TaskID
from tqdm import tqdm

from .colors import console, success_msg, error_msg, warning_msg, info_msg, progress_msg

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.supported_extensions = {'.pdf'}
        
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """Check if the file is a supported PDF"""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions and path.exists()
        
    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> Optional[str]:
        """Extract text from a single PDF file using PyMuPDF"""
        path = Path(pdf_path)
        
        if not self.is_supported_file(path):
            console.print(error_msg(f"Error: {path} is not a supported PDF file"))
            return None
            
        try:
            console.print(progress_msg(f"Extracting text from {path.name}..."))
            
            # Open PDF with PyMuPDF
            doc = fitz.open(str(path))
            text_parts = []
            
            # Use progress bar for large PDFs
            if len(doc) > 10:
                with Progress() as progress:
                    task = progress.add_task("Extracting pages...", total=len(doc))
                    
                    for page_num in range(len(doc)):
                        try:
                            page = doc[page_num]
                            page_text = page.get_text()
                            if page_text.strip():
                                text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                                text_parts.append(page_text)
                        except Exception as e:
                            logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                        
                        progress.update(task, advance=1)
            else:
                # Simple extraction for smaller PDFs
                for page_num in range(len(doc)):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        if page_text.strip():
                            text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
            
            doc.close()
            full_text = "\n".join(text_parts)
            
            if not full_text.strip():
                console.print(warning_msg(f"Warning: No text extracted from {path.name}"))
                return None
                
            console.print(success_msg(f"Successfully extracted {len(full_text)} characters from {path.name}"))
            return full_text
                
        except Exception as e:
            console.print(error_msg(f"Error processing {path.name}: {e}"))
            logger.error(f"Failed to process {path}: {e}")
            return None
            
    def extract_text_from_multiple_pdfs(self, pdf_paths: List[Union[str, Path]]) -> dict:
        """Extract text from multiple PDF files"""
        results = {}
        
        console.print(f"[blue]Processing {len(pdf_paths)} PDF files...[/blue]")
        
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            path = Path(pdf_path)
            if self.is_supported_file(path):
                text = self.extract_text_from_pdf(path)
                if text:
                    results[str(path)] = text
                else:
                    console.print(f"[yellow]Skipped {path.name} (no text extracted)[/yellow]")
            else:
                console.print(f"[red]Skipped {path.name} (not a valid PDF)[/red]")
                
        console.print(f"[green]Successfully processed {len(results)} PDF files[/green]")
        return results
        
    def find_pdfs_in_directory(self, directory: Union[str, Path], recursive: bool = False) -> List[Path]:
        """Find all PDF files in a directory"""
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            console.print(f"[red]Error: {directory} is not a valid directory[/red]")
            return []
            
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(dir_path.glob(pattern))
        
        console.print(f"[blue]Found {len(pdf_files)} PDF files in {directory}[/blue]")
        return pdf_files
        
    def get_pdf_info(self, pdf_path: Union[str, Path]) -> dict:
        """Get metadata from PDF file using PyMuPDF"""
        path = Path(pdf_path)
        
        if not self.is_supported_file(path):
            return {"error": "Invalid PDF file"}
            
        try:
            doc = fitz.open(str(path))
            metadata = doc.metadata
            page_count = len(doc)
            doc.close()
            
            return {
                "filename": path.name,
                "pages": page_count,
                "size_bytes": path.stat().st_size,
                "file_hash": self.get_file_hash(path),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", "")
            }
        except Exception as e:
            return {"error": str(e)}
            
    def combine_pdf_texts(self, pdf_texts: dict, separator: str = "\n\n" + "="*50 + "\n\n") -> str:
        """Combine text from multiple PDFs with separators"""
        if not pdf_texts:
            return ""
            
        combined_parts = []
        for pdf_path, text in pdf_texts.items():
            filename = Path(pdf_path).name
            combined_parts.append(f"SOURCE: {filename}")
            combined_parts.append(separator)
            combined_parts.append(text)
            combined_parts.append(separator)
            
        return "\n".join(combined_parts)
        
    def get_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of a file for duplicate detection"""
        path = Path(file_path)
        if not path.exists():
            return ""
            
        sha256_hash = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {path}: {e}")
            return ""