"""File tracking system for duplicate prevention and history"""

import aiosqlite
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union

from .colors import console, success_msg, error_msg, warning_msg, info_msg

class FileTracker:
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.db_path = self.session_dir / "file_tracker.db"
        self.session_dir.mkdir(exist_ok=True)
        
    async def init_database(self):
        """Initialize the SQLite database with required tables"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS inserted_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    size_bytes INTEGER,
                    pages INTEGER,
                    insertion_date TEXT NOT NULL,
                    extraction_status TEXT DEFAULT 'success',
                    metadata_json TEXT,
                    UNIQUE(file_hash)
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS arxiv_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT NOT NULL,
                    title TEXT,
                    authors TEXT,
                    abstract TEXT,
                    published_date TEXT,
                    download_date TEXT,
                    filepath TEXT,
                    file_hash TEXT,
                    UNIQUE(arxiv_id)
                )
            ''')
            
            await db.commit()
            
    async def is_file_inserted(self, file_hash: str) -> bool:
        """Check if a file with this hash has already been inserted"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                'SELECT COUNT(*) FROM inserted_files WHERE file_hash = ?',
                (file_hash,)
            )
            count = await cursor.fetchone()
            return count[0] > 0
            
    async def get_duplicate_info(self, file_hash: str) -> Optional[Dict]:
        """Get information about a duplicate file"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                'SELECT filename, filepath, insertion_date FROM inserted_files WHERE file_hash = ?',
                (file_hash,)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "filename": row[0],
                    "filepath": row[1],
                    "insertion_date": row[2]
                }
            return None
            
    async def record_file_insertion(self, 
                                   filename: str,
                                   filepath: str,
                                   file_hash: str,
                                   size_bytes: int,
                                   pages: int = 0,
                                   extraction_status: str = 'success',
                                   metadata: Optional[Dict] = None) -> bool:
        """Record a successful file insertion"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO inserted_files
                    (filename, filepath, file_hash, size_bytes, pages, insertion_date, extraction_status, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    filename,
                    filepath,
                    file_hash,
                    size_bytes,
                    pages,
                    datetime.now().isoformat(),
                    extraction_status,
                    str(metadata) if metadata else None
                ))
                await db.commit()
                return True
        except Exception as e:
            console.print(error_msg(f"Failed to record file insertion: {e}"))
            return False
            
    async def record_arxiv_paper(self,
                                arxiv_id: str,
                                title: str,
                                authors: str,
                                abstract: str,
                                published_date: str,
                                filepath: str,
                                file_hash: str) -> bool:
        """Record an ArXiv paper download"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO arxiv_papers
                    (arxiv_id, title, authors, abstract, published_date, download_date, filepath, file_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    arxiv_id,
                    title,
                    authors,
                    abstract,
                    published_date,
                    datetime.now().isoformat(),
                    filepath,
                    file_hash
                ))
                await db.commit()
                return True
        except Exception as e:
            console.print(error_msg(f"Failed to record ArXiv paper: {e}"))
            return False
            
    async def get_insertion_history(self, limit: int = 50) -> List[Dict]:
        """Get recent file insertion history"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                SELECT filename, filepath, insertion_date, extraction_status, pages, size_bytes
                FROM inserted_files
                ORDER BY insertion_date DESC
                LIMIT ?
            ''', (limit,))
            rows = await cursor.fetchall()
            
            return [
                {
                    "filename": row[0],
                    "filepath": row[1],
                    "insertion_date": row[2],
                    "extraction_status": row[3],
                    "pages": row[4],
                    "size_bytes": row[5]
                }
                for row in rows
            ]
            
    async def get_arxiv_history(self, limit: int = 50) -> List[Dict]:
        """Get ArXiv download history"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                SELECT arxiv_id, title, authors, download_date, filepath
                FROM arxiv_papers
                ORDER BY download_date DESC
                LIMIT ?
            ''', (limit,))
            rows = await cursor.fetchall()
            
            return [
                {
                    "arxiv_id": row[0],
                    "title": row[1],
                    "authors": row[2],
                    "download_date": row[3],
                    "filepath": row[4]
                }
                for row in rows
            ]
            
    async def get_statistics(self) -> Dict:
        """Get session statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            # Count inserted files
            cursor = await db.execute('SELECT COUNT(*) FROM inserted_files')
            file_count = (await cursor.fetchone())[0]
            
            # Count ArXiv papers
            cursor = await db.execute('SELECT COUNT(*) FROM arxiv_papers')
            arxiv_count = (await cursor.fetchone())[0]
            
            # Get total pages
            cursor = await db.execute('SELECT SUM(pages) FROM inserted_files')
            total_pages = (await cursor.fetchone())[0] or 0
            
            # Get total size
            cursor = await db.execute('SELECT SUM(size_bytes) FROM inserted_files')
            total_size = (await cursor.fetchone())[0] or 0
            
            return {
                "total_files": file_count,
                "arxiv_papers": arxiv_count,
                "total_pages": total_pages,
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            }
            
    async def is_arxiv_paper_downloaded(self, arxiv_id: str) -> bool:
        """Check if an ArXiv paper has already been downloaded"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                'SELECT COUNT(*) FROM arxiv_papers WHERE arxiv_id = ?',
                (arxiv_id,)
            )
            count = await cursor.fetchone()
            return count[0] > 0
            
    async def find_duplicates(self) -> List[Dict]:
        """Find potential duplicate files based on content hash"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                SELECT file_hash, COUNT(*) as count, 
                       GROUP_CONCAT(filename) as filenames,
                       GROUP_CONCAT(filepath) as filepaths
                FROM inserted_files
                GROUP BY file_hash
                HAVING count > 1
            ''')
            rows = await cursor.fetchall()
            
            duplicates = []
            for row in rows:
                duplicates.append({
                    "file_hash": row[0],
                    "count": row[1],
                    "filenames": row[2].split(','),
                    "filepaths": row[3].split(',')
                })
            
            return duplicates