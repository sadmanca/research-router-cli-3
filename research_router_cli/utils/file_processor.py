"""General file processing utilities for PDFs and text files"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Union

import fitz  # PyMuPDF
from rich.progress import Progress, TaskID
from tqdm import tqdm

from .colors import console, success_msg, error_msg, warning_msg, info_msg, progress_msg

logger = logging.getLogger(__name__)

class FileProcessor:
    """General file processor that handles PDFs, text files, and markdown files"""
    
    def __init__(self):
        self.pdf_extensions = {'.pdf'}
        self.text_extensions = {'.txt', '.md', '.markdown', '.rst', '.text'}
        self.supported_extensions = self.pdf_extensions | self.text_extensions
        
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """Check if the file is supported (PDF or text file)"""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions and path.exists()
        
    def is_pdf_file(self, file_path: Union[str, Path]) -> bool:
        """Check if the file is a PDF"""
        path = Path(file_path)
        return path.suffix.lower() in self.pdf_extensions
        
    def is_text_file(self, file_path: Union[str, Path]) -> bool:
        """Check if the file is a text file"""
        path = Path(file_path)
        return path.suffix.lower() in self.text_extensions
        
    def extract_text_from_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Extract text from a file (PDF or text file)"""
        path = Path(file_path)
        
        if not self.is_supported_file(path):
            console.print(error_msg(f"Error: {path} is not a supported file type"))
            return None
            
        if self.is_pdf_file(path):
            return self._extract_text_from_pdf(path)
        elif self.is_text_file(path):
            return self._extract_text_from_text_file(path)
        else:
            console.print(error_msg(f"Error: Unsupported file type for {path}"))
            return None
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a PDF file using PyMuPDF"""
        try:
            console.print(progress_msg(f"Extracting text from PDF {pdf_path.name}..."))
            
            # Open PDF with PyMuPDF
            doc = fitz.open(str(pdf_path))
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
                console.print(warning_msg(f"Warning: No text extracted from {pdf_path.name}"))
                return None
                
            console.print(success_msg(f"Successfully extracted {len(full_text)} characters from PDF {pdf_path.name}"))
            return full_text
                
        except Exception as e:
            console.print(error_msg(f"Error processing PDF {pdf_path.name}: {e}"))
            logger.error(f"Failed to process {pdf_path}: {e}")
            return None
    
    def _extract_text_from_text_file(self, text_path: Path) -> Optional[str]:
        """Extract text from a text file"""
        try:
            console.print(progress_msg(f"Reading text file {text_path.name}..."))
            
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(text_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    
                    if text.strip():
                        console.print(success_msg(f"Successfully read {len(text)} characters from {text_path.name}"))
                        return text
                    else:
                        console.print(warning_msg(f"Warning: {text_path.name} appears to be empty"))
                        return None
                        
                except UnicodeDecodeError:
                    continue
                    
            # If all encodings failed
            console.print(error_msg(f"Could not decode {text_path.name} with any supported encoding"))
            return None
                
        except Exception as e:
            console.print(error_msg(f"Error reading text file {text_path.name}: {e}"))
            logger.error(f"Failed to read {text_path}: {e}")
            return None
            
    def extract_text_from_multiple_files(self, file_paths: List[Union[str, Path]]) -> dict:
        """Extract text from multiple files (PDFs and text files)"""
        results = {}
        
        console.print(f"[blue]Processing {len(file_paths)} files...[/blue]")
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            path = Path(file_path)
            if self.is_supported_file(path):
                text = self.extract_text_from_file(path)
                if text:
                    results[path] = text
                else:
                    console.print(f"[yellow]Skipped {path.name} (no text extracted)[/yellow]")
            else:
                console.print(f"[red]Skipped {path.name} (unsupported file type)[/red]")
                
        console.print(f"[green]Successfully processed {len(results)} files[/green]")
        return results
        
    def find_files_in_directory(self, directory: Union[str, Path], recursive: bool = False) -> List[Path]:
        """Find all supported files in a directory"""
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            console.print(f"[red]Error: {directory} is not a valid directory[/red]")
            return []
        
        found_files = []
        patterns = []
        
        # Add patterns for all supported extensions
        for ext in self.supported_extensions:
            if recursive:
                patterns.append(f"**/*{ext}")
            else:
                patterns.append(f"*{ext}")
        
        # Find files matching any pattern
        for pattern in patterns:
            found_files.extend(dir_path.glob(pattern))
        
        # Remove duplicates and sort
        found_files = sorted(set(found_files))
        
        console.print(f"[blue]Found {len(found_files)} supported files in {directory}[/blue]")
        return found_files
        
    def get_file_info(self, file_path: Union[str, Path]) -> dict:
        """Get metadata from a file (PDF or text file)"""
        path = Path(file_path)
        
        if not self.is_supported_file(path):
            return {"error": "Unsupported file type"}
            
        try:
            base_info = {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "file_hash": self.get_file_hash(path),
                "file_type": "PDF" if self.is_pdf_file(path) else "Text"
            }
            
            if self.is_pdf_file(path):
                # Get PDF-specific metadata
                try:
                    doc = fitz.open(str(path))
                    metadata = doc.metadata
                    page_count = len(doc)
                    doc.close()
                    
                    base_info.update({
                        "pages": page_count,
                        "title": metadata.get("title", ""),
                        "author": metadata.get("author", ""),
                        "subject": metadata.get("subject", ""),
                        "creator": metadata.get("creator", ""),
                        "producer": metadata.get("producer", ""),
                        "creation_date": metadata.get("creationDate", ""),
                        "modification_date": metadata.get("modDate", "")
                    })
                except Exception as e:
                    base_info["pdf_error"] = str(e)
            else:
                # Get text file info
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        line_count = content.count('\n') + 1
                        word_count = len(content.split())
                        
                    base_info.update({
                        "lines": line_count,
                        "words": word_count,
                        "characters": len(content)
                    })
                except Exception as e:
                    base_info["text_error"] = str(e)
                    
            return base_info
            
        except Exception as e:
            return {"error": str(e)}
            
    def combine_file_texts(self, file_texts: dict, separator: str = "\n\n" + "="*50 + "\n\n") -> str:
        """Combine text from multiple files with separators"""
        if not file_texts:
            return ""
            
        combined_parts = []
        for file_path, text in file_texts.items():
            filename = Path(file_path).name
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
    
    def get_supported_extensions_display(self) -> str:
        """Get a display string of supported extensions"""
        pdf_exts = ', '.join(sorted(self.pdf_extensions))
        text_exts = ', '.join(sorted(self.text_extensions))
        return f"PDFs: {pdf_exts} | Text files: {text_exts}"